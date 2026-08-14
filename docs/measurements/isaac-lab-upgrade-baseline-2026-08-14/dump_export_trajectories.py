#!/usr/bin/env python3
"""Freeze per-step actions and hidden state for every exported policy artifact.

Export-time determinism probes only prove an artifact reproduces itself. They
cannot see a *deterministic but wrong* artifact, which is the failure class a
runtime bump produces — the measured dynamo/onnxscript GRU corruption emitted
correct ports and wrong numbers. Comparing pre- and post-bump requires a frozen
trajectory: fixed observation sequences, and the actions and hidden states the
current runtime produces from them.

The observation sequences are written into the .npz alongside the outputs and
replayed with ``--replay`` rather than regenerated, so the comparison never
depends on an RNG stream staying stable across a numpy or torch version.

Two sequences per artifact:

  ``normal``  — standard normal over every dim.
  ``indist``  — in-distribution: the 19 scalar dims N(0,1), the 3600 depth dims
                U[0,1]. A depth backbone clamps and normalizes its input, so an
                all-normal sequence can drive the depth block into a saturated
                region where real drift is masked. For a NOCAM artifact there is
                no depth block, so this sequence differs from ``normal`` only by
                seed.

Recurrence is threaded per format, not reset per step: ONNX carries
``h_out -> h_in`` through raw ``sess.run``; TorchScript carries it in the
module's ``hidden_state`` buffer, which is zeroed once at sequence start.

Usage (no Kit boot, CPU only):

    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
    $STRAFER_ISAACLAB_PYTHON \
      docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/dump_export_trajectories.py \
      --models-dir models \
      --out docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/export-anchors

Post-bump, re-run with ``--replay <the baseline .npz>`` and diff the arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

STEPS = 128
N_SCALARS = 19  # the NOCAM prefix; depth rides after it (80*45 = 3600)

ARTIFACTS = (
    "strafer_depth_subgoal_v2_998",
    "strafer_nocam_subgoal_v0",
    "strafer_nocam_subgoal_gru_smoke",
)
SEEDS = {"normal": 1234, "indist": 5678}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_obs(kind: str, obs_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return rng.standard_normal((STEPS, obs_dim), dtype=np.float32)
    obs = np.empty((STEPS, obs_dim), dtype=np.float32)
    n = min(N_SCALARS, obs_dim)
    obs[:, :n] = rng.standard_normal((STEPS, n), dtype=np.float32)
    if obs_dim > n:
        obs[:, n:] = rng.random((STEPS, obs_dim - n), dtype=np.float32)
    return obs


def run_torchscript(path: pathlib.Path, obs: np.ndarray):
    """Per-step actions and hidden state from the TorchScript artifact."""
    import torch

    torch.set_num_threads(1)
    mod = torch.jit.load(str(path), map_location="cpu")
    mod.eval()
    bufs = dict(mod.named_buffers())
    hstate = bufs.get("hidden_state")
    if hstate is not None:
        hstate.zero_()

    actions, hiddens = [], []
    with torch.no_grad():
        for step in range(obs.shape[0]):
            out = mod(torch.from_numpy(obs[step : step + 1]))
            actions.append(out.detach().cpu().numpy().astype(np.float64))
            if hstate is not None:
                hiddens.append(hstate.detach().cpu().numpy().astype(np.float64).copy())
    return (
        np.concatenate(actions, axis=0),
        np.stack(hiddens, axis=0) if hiddens else None,
    )


def run_onnx(path: pathlib.Path, obs: np.ndarray):
    """Per-step actions and hidden state from the ONNX artifact."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    recurrent = "h_in" in in_names
    h = None
    if recurrent:
        shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[in_names.index("h_in")].shape]
        h = np.zeros(shape, dtype=np.float32)

    actions, hiddens = [], []
    for step in range(obs.shape[0]):
        feed = {"obs": obs[step : step + 1]}
        if recurrent:
            feed["h_in"] = h
        outs = dict(zip(out_names, sess.run(out_names, feed)))
        actions.append(np.asarray(outs["actions"], dtype=np.float64))
        if recurrent:
            h = np.asarray(outs["h_out"], dtype=np.float32)
            hiddens.append(h.astype(np.float64).copy())
    return (
        np.concatenate(actions, axis=0),
        np.stack(hiddens, axis=0) if hiddens else None,
    )


def versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in (
        "torch",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "numpy",
        "rsl-rl-lib",
        "isaacsim",
    ):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument(
        "--replay",
        type=pathlib.Path,
        default=None,
        help="Replay observation sequences from a prior dump instead of generating them.",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    replay = np.load(args.replay) if args.replay else None
    arrays: dict[str, np.ndarray] = {}
    manifest: dict = {
        "steps": STEPS,
        "n_scalars": N_SCALARS,
        "seeds": SEEDS,
        "obs_source": str(args.replay) if replay is not None else "generated",
        "versions": versions(),
        "artifacts": {},
    }

    for name in ARTIFACTS:
        sidecar_path = args.models_dir / f"{name}.json"
        sidecar = json.loads(sidecar_path.read_text())
        obs_dim = sidecar["obs_dim"]
        entry: dict = {
            "sidecar": sidecar,
            "sidecar_sha256": sha256(sidecar_path),
            "formats": {},
            "sequences": {},
        }

        for fmt, runner in (("pt", run_torchscript), ("onnx", run_onnx)):
            path = args.models_dir / f"{name}.{fmt}"
            if not path.exists():
                entry["formats"][fmt] = None
                continue
            entry["formats"][fmt] = {"sha256": sha256(path), "bytes": path.stat().st_size}

            for kind, seed in SEEDS.items():
                key_obs = f"{name}|{kind}|obs"
                if key_obs in arrays:
                    obs = arrays[key_obs]
                elif replay is not None and key_obs in replay:
                    obs = replay[key_obs]
                    arrays[key_obs] = obs
                else:
                    obs = make_obs(kind, obs_dim, seed)
                    arrays[key_obs] = obs

                act, hid = runner(path, obs)
                # Self-consistency: the same artifact on the same inputs twice.
                # A nonzero delta here would make every pre/post number noise.
                act2, hid2 = runner(path, obs)
                rerun_action_delta = float(np.max(np.abs(act - act2)))
                rerun_hidden_delta = (
                    float(np.max(np.abs(hid - hid2))) if hid is not None else None
                )

                arrays[f"{name}|{kind}|{fmt}|actions"] = act
                if hid is not None:
                    arrays[f"{name}|{kind}|{fmt}|hidden"] = hid
                entry["sequences"][f"{kind}|{fmt}"] = {
                    "action_shape": list(act.shape),
                    "hidden_shape": list(hid.shape) if hid is not None else None,
                    "action_abs_mean": float(np.mean(np.abs(act))),
                    "action_min": float(act.min()),
                    "action_max": float(act.max()),
                    "hidden_abs_mean": float(np.mean(np.abs(hid))) if hid is not None else None,
                    "rerun_action_delta": rerun_action_delta,
                    "rerun_hidden_delta": rerun_hidden_delta,
                }
                print(
                    f"{name:32s} {kind:7s} {fmt:4s} "
                    f"|a|={np.mean(np.abs(act)):.6f} rerun_da={rerun_action_delta:.3e}"
                )

        # Cross-format parity: the deploy contract assumes .pt and .onnx are the
        # same policy. Recorded per sequence so a post-bump re-export is compared
        # against a known pre-bump agreement, not against an assumed zero.
        for kind in SEEDS:
            a_pt = arrays.get(f"{name}|{kind}|pt|actions")
            a_ox = arrays.get(f"{name}|{kind}|onnx|actions")
            if a_pt is None or a_ox is None:
                continue
            h_pt = arrays.get(f"{name}|{kind}|pt|hidden")
            h_ox = arrays.get(f"{name}|{kind}|onnx|hidden")
            entry["sequences"][f"{kind}|cross_format"] = {
                "max_action_delta": float(np.max(np.abs(a_pt - a_ox))),
                "max_hidden_delta": (
                    float(np.max(np.abs(h_pt - h_ox)))
                    if h_pt is not None and h_ox is not None
                    else None
                ),
            }
            print(
                f"{name:32s} {kind:7s} pt-vs-onnx  "
                f"max_da={entry['sequences'][f'{kind}|cross_format']['max_action_delta']:.3e}"
            )

        manifest["artifacts"][name] = entry

    npz = args.out / "export-trajectories.npz"
    np.savez_compressed(npz, **arrays)
    (args.out / "export-trajectories-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest_hash = sha256(npz)
    (args.out / "export-trajectories.npz.sha256").write_text(
        f"{manifest_hash}  export-trajectories.npz\n"
    )
    print(f"\nwrote {npz} ({npz.stat().st_size/1e6:.2f} MB) sha256={manifest_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
