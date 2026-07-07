#!/usr/bin/env python3
"""Export a trained rsl_rl checkpoint to deployable .pt and .onnx artifacts.

Loads an ``rsl_rl`` PPO checkpoint from training, freezes the deterministic
mean head (so two consecutive same-obs calls produce byte-identical actions),
and writes one or both of:

- ``<output>.pt``  -- TorchScript, loadable by ``torch.jit.load``.
- ``<output>.onnx`` -- ONNX, loadable by ONNX Runtime (CUDA/TensorRT EPs on
  Jetson, CPU EP on the DGX for round-trip validation).

A sidecar ``<output>.json`` records the policy variant, observation/action
dimensions, source checkpoint, repo commit, export timestamp, and -- for
ONNX -- the opset version. The Jetson inference node reads the sidecar at
launch and refuses to start if the recorded variant disagrees with its
configured ``PolicyVariant`` -- that's the cross-host invariant this brief
anchors at export time.

CLI examples (DGX, under the Isaac Sim conda env)::

    # NoCam (MLP) -- TorchScript only.
    $ISAACLAB -p source/strafer_lab/scripts/export_policy.py \\
        --checkpoint logs/rsl_rl/strafer_navigation/<run>/model_<step>.pt \\
        --output models/strafer_nocam_v0 \\
        --variant NOCAM

    # Depth (RNN + depth encoder) -- both TorchScript and ONNX are
    # supported; ONNX uses the recurrent (obs, h_in) -> (actions, h_out)
    # signature, with hidden state owned by the inference loader.
    $ISAACLAB -p source/strafer_lab/scripts/export_policy.py \\
        --checkpoint logs/rsl_rl/strafer_navigation/<run>/model_<step>.pt \\
        --output models/strafer_depth_v0 \\
        --variant DEPTH \\
        --formats pt,onnx

    # NoCam, both formats.
    $ISAACLAB -p source/strafer_lab/scripts/export_policy.py \\
        --checkpoint logs/rsl_rl/strafer_navigation/<run>/model_<step>.pt \\
        --output models/strafer_nocam_v0 \\
        --variant NOCAM \\
        --formats pt,onnx

The pure-Python helpers (``export_torchscript``, ``export_onnx``,
``write_metadata_sidecar``, ``read_metadata_sidecar``) are importable from
this script without launching Isaac Sim, so unit tests in
``source/strafer_lab/tests/test_export_policy.py`` exercise the export
plumbing against a tiny dummy actor.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

# Adding the strafer_shared package import lazily so the helpers below stay
# importable when only torch is available (the test suite path).


# Default training environment IDs per variant. Used for both the CLI's
# ``--env`` default and as the ``env_id`` field in the metadata sidecar
# when the operator does not pass ``--env`` explicitly.
_DEFAULT_ENV_BY_VARIANT = {
    "NOCAM": "Isaac-Strafer-Nav-RLNoCam-Play-v0",
    "DEPTH": "Isaac-Strafer-Nav-RLDepth-Real-Play-v0",
    "NOCAM_SUBGOAL": "Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0",
    "DEPTH_SUBGOAL": "Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0",
}

# Variants whose envs build with no camera/depth prim in the scene. Cameras
# stay disabled for these unless more than one env forces a tiled render.
_CAMERA_LESS_VARIANTS = {"NOCAM", "NOCAM_SUBGOAL"}


def _should_enable_cameras(variant: str, num_envs: int) -> bool:
    """Whether ``--enable_cameras`` must be forced on for this export run.

    Camera-bearing variants don't load unless cameras are enabled; the
    camera-less variants build only when more than one env forces a tiled
    render. Keyed on the camera-less variant set, not a single literal name,
    so a new camera-less variant doesn't silently get cameras enabled.
    """
    return variant not in _CAMERA_LESS_VARIANTS or num_envs > 1


# ONNX opset 18 matches what rsl_rl's built-in exporter emits. Jetson's
# onnxruntime-gpu wheel supports it.
_DEFAULT_ONNX_OPSET = 18


# ---------------------------------------------------------------------------
# Pure-Python export helpers (no Isaac Sim dependency).
# ---------------------------------------------------------------------------


def export_torchscript(
    module: nn.Module,
    output_path: str | Path,
    *,
    obs_dim: int,
    is_recurrent: bool = False,
) -> Path:
    """TorchScript-script ``module`` and write it to ``output_path``.

    The deterministic-head freeze must happen before this call. Recurrent
    modules are expected to expose a ``reset()`` method so callers can
    initialise hidden state to a known value before measuring determinism.

    Round-trip verifies byte-identical action output across two calls with
    the same synthetic observation. For recurrent modules, ``reset()`` is
    called between calls so the comparison reflects the deterministic
    forward pass alone, not hidden-state evolution. Raises ``RuntimeError``
    if determinism is violated -- failing fast prevents shipping a model
    whose stochastic head was not frozen.

    Args:
        module: ``torch.nn.Module`` whose ``forward(x)`` takes a
            ``(1, obs_dim)`` tensor and returns ``(1, action_dim)``.
        output_path: Destination ``.pt`` path. Parent directory is created
            if missing.
        obs_dim: Observation dimensionality used for the round-trip probe.
        is_recurrent: Whether ``module`` carries hidden state across
            ``forward`` calls. Determines whether the determinism probe
            calls ``reset()`` between probes.

    Returns:
        The resolved output ``Path``.
    """
    output_path = Path(output_path)
    if output_path.suffix != ".pt":
        raise ValueError(
            f"TorchScript output path must end in '.pt', got '{output_path.suffix}'"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    module = module.eval().cpu()
    scripted = torch.jit.script(module)
    scripted.save(str(output_path))

    _verify_torchscript_determinism(output_path, obs_dim=obs_dim, is_recurrent=is_recurrent)
    return output_path


def export_onnx(
    module: nn.Module,
    output_path: str | Path,
    *,
    obs_dim: int,
    opset: int = _DEFAULT_ONNX_OPSET,
) -> Path:
    """Export ``module`` to ONNX and write it to ``output_path``.

    Handles both stateless single-input modules and the multi-input
    recurrent wrappers (``_OnnxRNNModel``, ``_OnnxDepthGRUModel``). When the
    module exposes ``get_dummy_inputs()`` / ``input_names`` / ``output_names``,
    those drive the export; otherwise the single ``obs`` / ``actions``
    signature is used.

    Round-trip verifies byte-identical output across two ONNX Runtime calls
    with the same synthetic observation. For recurrent modules the auxiliary
    inputs (``h_in``, ``c_in``) are zero-reset on each call so the
    determinism statement matches the recurrent contract: same observation
    + reset hidden state → byte-identical actions. Raises ``RuntimeError``
    if determinism is violated.

    Args:
        module: ``torch.nn.Module`` whose ``forward`` matches one of the two
            supported signatures. Stateless: ``forward(x: (1, obs_dim)) ->
            (1, action_dim)``. Recurrent: ``forward(obs, h_in[, c_in]) ->
            (actions, h_out[, c_out])`` with ``get_dummy_inputs()`` /
            ``input_names`` / ``output_names`` published as attributes.
        output_path: Destination ``.onnx`` path. Parent directory is
            created if missing.
        obs_dim: Observation dimensionality used for the round-trip probe's
            primary input. Ignored when the module supplies its own
            ``get_dummy_inputs()``.
        opset: ONNX opset version (default 18, matching rsl_rl).

    Returns:
        The resolved output ``Path``.
    """
    output_path = Path(output_path)
    if output_path.suffix != ".onnx":
        raise ValueError(
            f"ONNX output path must end in '.onnx', got '{output_path.suffix}'"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    module = module.eval().cpu()

    get_dummy_inputs = getattr(module, "get_dummy_inputs", None)
    if callable(get_dummy_inputs):
        dummy_inputs = tuple(get_dummy_inputs())
        input_names = list(getattr(module, "input_names", ["obs"]))
        output_names = list(getattr(module, "output_names", ["actions"]))
    else:
        dummy_inputs = (torch.zeros(1, obs_dim, dtype=torch.float32),)
        input_names = ["obs"]
        output_names = ["actions"]

    torch.onnx.export(
        module,
        dummy_inputs,
        str(output_path),
        export_params=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamo=False,
    )

    _verify_onnx_determinism(output_path, obs_dim=obs_dim)
    return output_path


def write_metadata_sidecar(
    output_stem: str | Path,
    *,
    policy_variant: str,
    obs_dim: int,
    action_dim: int,
    env_id: str,
    training_preset: str,
    source_checkpoint: str | Path,
    formats: Iterable[str],
    is_recurrent: bool,
    onnx_opset: int | None = None,
    tensorrt_engine_path: str | Path | None = None,
    tensorrt_version: str | None = None,
    git_commit: str | None = None,
    export_timestamp: str | None = None,
) -> Path:
    """Write the ``<output_stem>.json`` sidecar describing the export.

    The Jetson inference node reads this on launch and asserts the
    recorded variant + obs_dim match its configured ``PolicyVariant``.
    Mismatch is fatal at startup -- see the strafer-inference-package
    brief's cross-brief invariant.

    Args:
        output_stem: Path stem (no extension, or any extension -- only the
            stem is used). The sidecar is written to
            ``<stem-without-extension>.json``.
        policy_variant: ``PolicyVariant.name`` (e.g. ``"NOCAM"``).
        obs_dim: Observation dimensionality, validated against
            ``PolicyVariant.<variant>.obs_dim`` by the caller.
        action_dim: Action dimensionality (expected 3 -- vx, vy, omega).
        env_id: Registered Gym task ID the checkpoint trained against.
        training_preset: rsl_rl runner config name (e.g.
            ``"STRAFER_PPO_RUNNER_CFG"``).
        source_checkpoint: Path to the rsl_rl checkpoint that was exported.
        formats: Iterable of formats actually written (e.g.
            ``["pt", "onnx"]``).
        is_recurrent: Whether the exported policy carries hidden state
            across calls. Tells the inference node it must call
            ``policy.reset()`` at episode boundaries.
        onnx_opset: ONNX opset version (set when ``"onnx"`` is in
            ``formats``).
        tensorrt_engine_path: Optional path to a pre-built ``.engine``
            shipped alongside (avoids the ~30 s first-call build cost on
            Jetson).
        tensorrt_version: TensorRT runtime version pinned by the engine
            build, when ``tensorrt_engine_path`` is set.
        git_commit: Repo SHA at export time; auto-detected via
            ``git rev-parse HEAD`` when ``None``.
        export_timestamp: ISO 8601 timestamp; auto-set to ``datetime.now``
            when ``None``.

    Returns:
        The resolved sidecar ``Path``.
    """
    stem = Path(output_stem)
    sidecar_path = stem.with_suffix(".json")

    if git_commit is None:
        git_commit = _detect_git_commit()
    if export_timestamp is None:
        export_timestamp = datetime.now(timezone.utc).isoformat()

    formats_list = list(formats)
    payload: dict = {
        "policy_variant": policy_variant,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "env_id": env_id,
        "training_preset": training_preset,
        "source_checkpoint": str(source_checkpoint),
        "formats": formats_list,
        "is_recurrent": bool(is_recurrent),
        "git_commit": git_commit,
        "export_timestamp": export_timestamp,
    }
    if "onnx" in formats_list:
        payload["onnx_opset"] = onnx_opset if onnx_opset is not None else _DEFAULT_ONNX_OPSET
    if tensorrt_engine_path is not None:
        payload["tensorrt_engine_path"] = str(tensorrt_engine_path)
    if tensorrt_version is not None:
        payload["tensorrt_version"] = tensorrt_version

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sidecar_path


def read_metadata_sidecar(model_path: str | Path) -> dict:
    """Read the sidecar JSON next to a ``.pt`` / ``.onnx`` artifact.

    Looks up ``<model-stem>.json`` regardless of which format the caller
    has in hand -- both formats from a single export share one sidecar.
    """
    path = Path(model_path)
    return json.loads(path.with_suffix(".json").read_text())


# ---------------------------------------------------------------------------
# Internal verification helpers.
# ---------------------------------------------------------------------------


def _verify_torchscript_determinism(
    model_path: Path, *, obs_dim: int, is_recurrent: bool
) -> None:
    """Load and assert the exported ``.pt`` produces deterministic output."""
    rng = np.random.default_rng(seed=0)
    obs = rng.standard_normal(obs_dim).astype(np.float32)

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    with torch.no_grad():
        if is_recurrent:
            model.reset()
        out_a = model(torch.from_numpy(obs).reshape(1, obs_dim)).squeeze(0).numpy()
        if is_recurrent:
            model.reset()
        out_b = model(torch.from_numpy(obs).reshape(1, obs_dim)).squeeze(0).numpy()

    if not np.array_equal(out_a, out_b):
        max_abs = float(np.abs(out_a - out_b).max())
        raise RuntimeError(
            f"TorchScript export at {model_path} is non-deterministic: same "
            f"observation produced different actions across two calls "
            f"(max abs delta = {max_abs:.6e}). The stochastic head was not "
            f"frozen at export time -- the inference node's determinism "
            f"contract would fail on the robot."
        )

    if out_a.ndim != 1:
        raise RuntimeError(
            f"TorchScript export at {model_path} returned shape {out_a.shape}; "
            f"expected a 1-D action vector after squeeze."
        )


def _verify_onnx_determinism(model_path: Path, *, obs_dim: int) -> None:
    """Load and assert the exported ``.onnx`` produces deterministic output.

    For stateless ONNX (single input) this is the historical "same obs →
    same actions" probe. For recurrent ONNX (additional ``h_in`` / ``c_in``
    inputs) the auxiliary inputs are zero-reset on each call so the probe
    measures the deterministic forward pass alone, not hidden-state
    evolution — matching the contract the inference node asserts.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(seed=0)
    obs = rng.standard_normal(obs_dim).astype(np.float32).reshape(1, obs_dim)

    sess = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    inputs = sess.get_inputs()
    obs_name = inputs[0].name

    def _resolved_shape(shape) -> list[int]:
        # ORT reports symbolic dims as strings (or None) when the model
        # was exported with dynamic axes. The export path here does not
        # set dynamic axes, so dims should be concrete ints; fall back to
        # 1 for any non-integer so the zero-init feed still goes through.
        return [int(d) if isinstance(d, int) else 1 for d in shape]

    def _feed() -> dict[str, np.ndarray]:
        feed = {obs_name: obs}
        for inp in inputs[1:]:
            feed[inp.name] = np.zeros(_resolved_shape(inp.shape), dtype=np.float32)
        return feed

    outs_a = sess.run(None, _feed())
    outs_b = sess.run(None, _feed())

    output_names = [o.name for o in sess.get_outputs()]
    for name, a, b in zip(output_names, outs_a, outs_b):
        if not np.array_equal(a, b):
            max_abs = float(np.abs(a - b).max())
            raise RuntimeError(
                f"ONNX export at {model_path} is non-deterministic: output "
                f"'{name}' differs across two calls with reset auxiliary "
                f"inputs (max abs delta = {max_abs:.6e})."
            )

    actions = outs_a[0]
    squeezed = actions.squeeze(0)
    if squeezed.ndim != 1:
        raise RuntimeError(
            f"ONNX export at {model_path} returned actions shape {actions.shape}; "
            f"expected (1, action_dim) so squeeze yields a 1-D vector."
        )


def _detect_git_commit() -> str | None:
    """Return the current repo HEAD SHA, or ``None`` if git isn't available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    sha = result.stdout.strip()
    return sha or None


# ---------------------------------------------------------------------------
# CLI entry point (Isaac Sim required).
# ---------------------------------------------------------------------------


def _parse_formats(formats_arg: str) -> list[str]:
    """Parse ``--formats`` -- comma-separated list of ``pt`` / ``onnx``."""
    items = [f.strip().lower() for f in formats_arg.split(",") if f.strip()]
    valid = {"pt", "onnx"}
    bad = [f for f in items if f not in valid]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown format(s): {bad}; valid values are {sorted(valid)}"
        )
    if not items:
        raise argparse.ArgumentTypeError("--formats must list at least one of pt, onnx")
    seen: set[str] = set()
    deduped: list[str] = []
    for f in items:
        if f not in seen:
            deduped.append(f)
            seen.add(f)
    return deduped


def _split_output(output: str, formats: list[str]) -> tuple[Path, dict[str, Path]]:
    """Resolve the output stem + per-format paths from ``--output``."""
    output_path = Path(output)
    suffix = output_path.suffix.lower()
    if suffix in (".pt", ".onnx"):
        stem = output_path.with_suffix("")
        format_for_suffix = suffix.lstrip(".")
        if format_for_suffix not in formats:
            raise SystemExit(
                f"--output ends in '{suffix}' but '{format_for_suffix}' is not in --formats {formats}"
            )
    else:
        stem = output_path

    paths = {fmt: stem.with_suffix(f".{fmt}") for fmt in formats}
    return stem, paths


def main() -> None:
    """CLI entry: load checkpoint, export TorchScript and/or ONNX, write sidecar."""
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained rsl_rl PPO checkpoint to deployable .pt and .onnx "
            "artifacts loadable by strafer_shared.policy_interface.load_policy()."
        ),
    )
    # Intercept -h / --help before importing AppLauncher: AppLauncher's
    # add_app_launcher_args() calls parse_known_args() internally, which
    # fails on the parser's required args before the help action fires.
    # Operator wants `--help` to surface CLI documentation, not an Isaac
    # Sim startup or a confusing required-arg error.
    import sys as _sys

    _help_only = any(flag in _sys.argv[1:] for flag in ("-h", "--help"))
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model_<step>.pt produced by source/strafer_lab/scripts/train_strafer_navigation.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output path stem (no extension), or a path ending in .pt / .onnx. "
            "Both formats and the .json sidecar share the same stem."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=sorted(_DEFAULT_ENV_BY_VARIANT.keys()),
        help="PolicyVariant name; controls obs_dim validation and sidecar metadata.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help=(
            "Registered Gym task ID used to reconstruct the rsl_rl runner. "
            "Defaults to the deployment-target Play env for the chosen variant."
        ),
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=["pt"],
        help="Comma-separated list of formats to emit (default: 'pt'; e.g. 'pt,onnx').",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of envs spun up to satisfy the rsl_rl runner constructor (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed forwarded to the env config.",
    )
    parser.add_argument(
        "--onnx_opset",
        type=int,
        default=_DEFAULT_ONNX_OPSET,
        help=f"ONNX opset version (default: {_DEFAULT_ONNX_OPSET}).",
    )
    parser.add_argument(
        "--tensorrt_engine_path",
        type=str,
        default=None,
        help=(
            "Optional path to a pre-built TensorRT .engine to record in the "
            "sidecar (engine generation is operator-side; this script does not "
            "build engines)."
        ),
    )
    parser.add_argument(
        "--tensorrt_version",
        type=str,
        default=None,
        help="Pinned TensorRT runtime version when --tensorrt_engine_path is set.",
    )

    if _help_only:
        parser.print_help()
        return

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Default env ID from variant if the operator didn't override.
    if args.env is None:
        args.env = _DEFAULT_ENV_BY_VARIANT[args.variant]

    # Sanity-check the checkpoint exists before paying the Isaac Sim startup cost.
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    if _should_enable_cameras(args.variant, args.num_envs):
        args.enable_cameras = True

    output_stem, output_paths = _split_output(args.output, args.formats)

    # Launch Isaac Sim and reconstruct the rsl_rl runner exactly like
    # source/strafer_lab/scripts/play_strafer_navigation.py does. We do not step the env --
    # the runner is built only to deserialize the checkpoint into the
    # correctly-shaped actor model.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import traceback as _traceback
    try:
        import gymnasium as gym
        import importlib.metadata as _metadata

        from isaaclab_rl.rsl_rl import (
            RslRlVecEnvWrapper,
            handle_deprecated_rsl_rl_cfg,
        )
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        from strafer_shared.policy_interface import PolicyVariant

        import strafer_lab  # noqa: F401  (registers envs)

        variant = PolicyVariant[args.variant]

        env_cfg = parse_env_cfg(
            args.env,
            device=args.device,
            num_envs=args.num_envs,
        )
        env_cfg.seed = args.seed

        agent_cfg = load_cfg_from_registry(args.env, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))

        env = gym.make(args.env, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)

        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        runner.load(str(checkpoint_path))

        policy_model = runner.alg.get_policy()
        is_recurrent = bool(getattr(policy_model, "is_recurrent", False))
        action_dim = int(env.num_actions)

        if action_dim != 3:
            raise SystemExit(
                f"Expected env.num_actions == 3, got {action_dim}. The Strafer "
                f"contract is (vx, vy, omega); check the env registration."
            )

        # Validate variant.obs_dim matches the env's actor input width.
        # ``StraferDepthRNNModel`` reports ``obs_dim`` as the *compressed*
        # width (scalar + depth embedding), since that's what the
        # normalizer / RNN / MLP see. The export contract needs the raw
        # input width the policy callable accepts, which the depth model
        # exposes as ``_raw_obs_dim``. Fall back to ``obs_dim`` for plain
        # MLP/RNN policies that don't have the compression seam.
        obs_dim = int(
            getattr(policy_model, "_raw_obs_dim", None)
            or getattr(policy_model, "obs_dim", variant.obs_dim)
        )
        if obs_dim != variant.obs_dim:
            raise SystemExit(
                f"Variant '{args.variant}' has obs_dim={variant.obs_dim} but "
                f"the loaded policy reports obs_dim={obs_dim}. The checkpoint "
                f"was trained against a different env -- aborting export to "
                f"prevent shipping a mis-labeled artifact."
            )

        env.close()

        training_preset = type(agent_cfg).__name__

        formats_written: list[str] = []

        if "pt" in args.formats:
            print(f"[INFO] Exporting TorchScript -> {output_paths['pt']}")
            jit_module = policy_model.as_jit()
            export_torchscript(
                jit_module,
                output_paths["pt"],
                obs_dim=obs_dim,
                is_recurrent=is_recurrent,
            )
            formats_written.append("pt")

        if "onnx" in args.formats:
            print(f"[INFO] Exporting ONNX -> {output_paths['onnx']}")
            onnx_module = policy_model.as_onnx(verbose=False)
            export_onnx(
                onnx_module,
                output_paths["onnx"],
                obs_dim=obs_dim,
                opset=args.onnx_opset,
            )
            formats_written.append("onnx")

        sidecar_path = write_metadata_sidecar(
            output_stem,
            policy_variant=args.variant,
            obs_dim=obs_dim,
            action_dim=action_dim,
            env_id=args.env,
            training_preset=training_preset,
            source_checkpoint=checkpoint_path,
            formats=formats_written,
            is_recurrent=is_recurrent,
            onnx_opset=args.onnx_opset if "onnx" in formats_written else None,
            tensorrt_engine_path=args.tensorrt_engine_path,
            tensorrt_version=args.tensorrt_version,
        )
        print(f"[INFO] Wrote sidecar: {sidecar_path}")
        print("[INFO] Export complete.")

    except BaseException:
        # ``simulation_app.close()`` below calls ``os._exit(0)`` on some
        # Isaac Sim builds, which swallows the traceback and reports a
        # zero exit code. Print + flush before the finally clause runs so
        # the operator actually sees what went wrong.
        print("[ERROR] Export aborted with the following traceback:", flush=True)
        _traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
