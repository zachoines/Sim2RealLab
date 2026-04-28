"""Generate Infinigen indoor scenes and stage them as USDC files.

Pipeline per scene:

1. ``python -m infinigen_examples.generate_indoors`` (coarse task) solves a
   floor plan and room population, saving ``scene.blend`` under a per-scene
   work directory. Invoked via the Python interpreter at
   ``STRAFER_INFINIGEN_PYTHON``, not via the Blender binary — Infinigen
   needs the source-built ``bpy`` wheel plus a stack of pure-Python deps
   (``gin-config``, numpy, etc.) that are only installed into that env.
   Blender's bundled Python does not have them.

2. ``python -m infinigen.tools.export --omniverse -f usdc`` converts the
   ``.blend`` to a USDC file plus a textures/ directory.

3. ``postprocess_scene_usd.py`` bakes colliders + ceiling-light emitters
   into the USDC. Runs via ``STRAFER_ISAACLAB_PYTHON`` (needs ``pxr``).

4. A top-level symlink ``Assets/generated/scenes/scene_<name>.usdc`` points
   at the USDC inside the per-scene subdir. ``strafer_env_cfg._get_scene_usd_paths``
   discovers files with the ``scene_*.usdc`` prefix at the top level of
   ``Assets/generated/scenes/``; the symlink satisfies that filter while
   keeping each scene's textures bundled with its USDC.

Presets wrap Infinigen's real knobs (``-g`` gin config files and ``-p``
gin-style parameter overrides).

Usage::

    python scripts/prep_room_usds.py generate \\
        --config high_quality_dgx \\
        --num-scenes 10 \\
        --output Assets/generated/scenes

    # Fast debug scene (single furnished room):
    python scripts/prep_room_usds.py generate \\
        --config fast_singleroom \\
        --num-scenes 1 \\
        --output Assets/generated/scenes

    # Import externally-generated scenes (e.g. from another host):
    python scripts/prep_room_usds.py ingest \\
        --source /mnt/transfer/infinigen_out \\
        --output Assets/generated/scenes

Prerequisites (all read from ``.env`` via ``env_setup.sh``):

- ``INFINIGEN_ROOT`` — Infinigen source checkout.
  ``generate_indoors`` imports ``infinigen`` relative to the source
  tree, so subprocesses run with that dir as their working directory.
- ``STRAFER_INFINIGEN_PYTHON`` — Python with ``bpy`` + ``gin-config`` +
  Infinigen deps installed.
- ``STRAFER_ISAACLAB_PYTHON`` — Python with ``pxr`` importable.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("prep_room_usds")


# ---------------------------------------------------------------------------
# Quality presets — DGX-specific and low-end fallbacks
# ---------------------------------------------------------------------------


@dataclass
class SceneGenConfig:
    """Scene generation preset.

    ``gin_configs`` and ``gin_overrides`` map directly onto Infinigen's
    ``-g``/``-p`` CLI flags. Everything else is provenance metadata
    recorded into ``scene_config.json`` next to each generated scene.
    """

    name: str
    # Gin config file stems (no ``.gin`` extension). Loaded from
    # ``infinigen_examples/configs_indoor/``. Order matters: later files
    # can override earlier ones.
    gin_configs: tuple[str, ...] = ("base_indoors",)
    # Gin parameter overrides, ``module.key=value`` strings. Applied after
    # the gin_configs so they always win.
    gin_overrides: tuple[str, ...] = ("compose_indoors.terrain_enabled=False",)
    # Cap on the number of rooms the constraint solver will try to fit.
    # Passed to Infinigen as ``restrict_solving.solve_max_rooms=N``. The
    # unconstrained default is known to wedge the solver on hard seeds;
    # capping keeps runtime bounded. ``5`` matches the DGX workhorse
    # preset's original intent; set to ``1`` for fast single-room scenes.
    max_rooms: int = 5
    # Texture image resolution for the USDC export pass. Higher values
    # give better perception training data but slow export considerably.
    texture_resolution: int = 1024
    # Seed base; scene i uses seed = random_seed_base + i.
    random_seed_base: int = 0
    # Human-readable preset description (recorded in scene_config.json).
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


# Full multi-room house with furniture — the workhorse preset.
# ``base_indoors`` is Infinigen's default full-quality recipe; terrain is
# disabled because we are generating interior scenes. ``max_rooms=5``
# caps the solver's search space; leaving it unbounded wedges on hard
# seeds for hours.
HIGH_QUALITY_DGX = SceneGenConfig(
    name="high_quality_dgx",
    gin_configs=("base_indoors",),
    gin_overrides=("compose_indoors.terrain_enabled=False",),
    max_rooms=5,
    texture_resolution=1024,
    description="Full multi-room furnished house on DGX Spark (<=5 rooms).",
)

# Fast single-room iteration — matches docs/HelloRoom.md's `-g fast_solve.gin
# singleroom.gin` recipe. Useful for debugging, CI smoke tests, and quick
# perception-pipeline validation.
FAST_SINGLEROOM = SceneGenConfig(
    name="fast_singleroom",
    gin_configs=("fast_solve", "singleroom"),
    gin_overrides=("compose_indoors.terrain_enabled=False",),
    max_rooms=1,
    texture_resolution=512,
    description="Single furnished room; fast solver for debug / CI.",
)

# Lower-memory baseline preserved for parity with earlier workstation
# workflows that transferred scenes onto the DGX via the ingest path.
WINDOWS_BASELINE = SceneGenConfig(
    name="windows_baseline",
    gin_configs=("fast_solve", "singleroom"),
    gin_overrides=("compose_indoors.terrain_enabled=False",),
    max_rooms=1,
    texture_resolution=512,
    description="Low-memory baseline for the Windows workstation.",
)

PRESETS: dict[str, SceneGenConfig] = {
    cfg.name: cfg for cfg in (HIGH_QUALITY_DGX, FAST_SINGLEROOM, WINDOWS_BASELINE)
}


# ---------------------------------------------------------------------------
# Invocation — wraps Infinigen's entry point via subprocess
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    scene_dir: Path
    blend_path: Path | None = None
    usd_path: Path | None = None
    symlink_path: Path | None = None
    returncode: int = 0
    stderr_tail: str = ""


# Return codes reserved for orchestrator-detected failures (subprocess itself
# reports its own non-zero rc). Picked above typical shell exit codes to
# stay out of the way.
_RC_BLEND_MISSING = 101
_RC_USDC_MISSING = 102


def generate_scenes(
    *,
    config: SceneGenConfig,
    num_scenes: int,
    output_dir: Path,
    infinigen_root: Path | None = None,
    infinigen_python: str | None = None,
) -> list[GenerationResult]:
    """Run Infinigen ``num_scenes`` times and stage USDCs under ``output_dir``.

    For each scene:

    - Coarse generate → ``output_dir/<scene_name>/coarse/scene.blend``
    - Export to USDC  → ``output_dir/<scene_name>/export/export_scene.blend/*.usdc``
      (plus a sibling ``textures/`` directory)
    - Symlink         → ``output_dir/<scene_name>.usdc`` points at the USDC
      inside the export tree so ``strafer_env_cfg._get_scene_usd_paths``
      discovers the scene without needing to recurse.

    A ``scene_config.json`` is written next to each scene for provenance.
    The orchestrator asserts that both the ``.blend`` (after coarse gen)
    and the ``.usdc`` (after export) actually exist; if either step
    silently produced no file, the result is flagged with a nonzero
    ``returncode`` instead of being reported as success.
    """
    infinigen_python = infinigen_python or _resolve_infinigen_python()
    # Resolve to absolute paths: the subprocess runs with cwd=infinigen_root,
    # so any relative output path would land inside Infinigen's source tree
    # rather than under the caller-intended directory.
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    infinigen_root = (
        Path(infinigen_root).resolve() if infinigen_root else _resolve_infinigen_root()
    )

    results: list[GenerationResult] = []
    for i in range(num_scenes):
        seed = config.random_seed_base + i
        scene_name = f"scene_{config.name}_{i:03d}_seed{seed}"
        scene_dir = output_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        coarse_dir = scene_dir / "coarse"
        export_dir = scene_dir / "export"

        logger.info("=== Scene %d/%d: %s (seed=%d) ===", i + 1, num_scenes, scene_name, seed)

        # --- Step 1: coarse generation ---
        gen_cmd = _build_generate_command(
            infinigen_python=infinigen_python,
            output_folder=coarse_dir,
            seed=seed,
            config=config,
        )
        gen_rc, gen_stderr = _run_subprocess(gen_cmd, cwd=infinigen_root)
        result = GenerationResult(scene_dir=scene_dir, returncode=gen_rc, stderr_tail=gen_stderr)

        if gen_rc != 0:
            logger.error("Scene %s: coarse gen failed (rc=%d)\n%s", scene_name, gen_rc, gen_stderr)
            results.append(result)
            continue

        blend_path = coarse_dir / "scene.blend"
        if not blend_path.exists():
            logger.error(
                "Scene %s: coarse gen returned rc=0 but %s is missing — silent failure in Infinigen\n"
                "Last stderr:\n%s",
                scene_name, blend_path, gen_stderr,
            )
            result.returncode = _RC_BLEND_MISSING
            results.append(result)
            continue
        result.blend_path = blend_path

        # --- Step 2: USDC export ---
        export_cmd = _build_export_command(
            infinigen_python=infinigen_python,
            input_folder=coarse_dir,
            output_folder=export_dir,
            texture_resolution=config.texture_resolution,
        )
        exp_rc, exp_stderr = _run_subprocess(export_cmd, cwd=infinigen_root)

        if exp_rc != 0:
            logger.error("Scene %s: export failed (rc=%d)\n%s", scene_name, exp_rc, exp_stderr)
            result.returncode = exp_rc
            result.stderr_tail = exp_stderr
            results.append(result)
            continue

        usd_candidates = sorted(export_dir.rglob("*.usdc"))
        if not usd_candidates:
            logger.error(
                "Scene %s: export returned rc=0 but no .usdc file was written under %s\n"
                "Last stderr:\n%s",
                scene_name, export_dir, exp_stderr,
            )
            result.returncode = _RC_USDC_MISSING
            results.append(result)
            continue
        result.usd_path = usd_candidates[0]

        # --- Step 3: post-process USDC (bake colliders + interior lights) ---
        post_rc, post_stderr = _run_postprocess(result.usd_path)
        if post_rc != 0:
            logger.error(
                "Scene %s: USDC post-process failed (rc=%d)\n%s",
                scene_name, post_rc, post_stderr,
            )
            result.returncode = post_rc
            result.stderr_tail = post_stderr
            results.append(result)
            continue

        # --- Step 4: top-level symlink for _get_scene_usd_paths() discovery ---
        link_path = output_dir / f"{scene_name}.usdc"
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(result.usd_path)
        result.symlink_path = link_path

        # --- Step 5: provenance ---
        (scene_dir / "scene_config.json").write_text(json.dumps(config.to_dict(), indent=2))

        logger.info("Scene %s OK → %s (USDC at %s)", scene_name, link_path, result.usd_path)
        results.append(result)

    return results


def _run_postprocess(usdc_path: Path) -> tuple[int, str]:
    """Run ``postprocess_scene_usd.py`` on the freshly-exported USDC.

    Runs under ``STRAFER_ISAACLAB_PYTHON`` (needs ``pxr``), not the
    Infinigen Python used for generation. Bakes colliders + ceiling-light
    emitters into the USDC so runtime sim launches don't pay a per-start
    traversal cost.
    """
    script = Path(__file__).resolve().parent / "postprocess_scene_usd.py"
    cmd = [
        _resolve_isaaclab_python(),
        str(script),
        "--usdc",
        str(usdc_path),
    ]
    return _run_subprocess(cmd, cwd=script.parent)


def _run_subprocess(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    """Run ``cmd`` in ``cwd``, stream its stderr+stdout live, and return
    ``(returncode, last 30 lines of merged output)``.

    Live streaming matters here: the coarse-gen task runs for tens of
    minutes and Infinigen's constraint solver prints step-by-step progress
    that lets you catch a wedged run early. A buffered ``capture_output``
    model (the old implementation) hides all of that until the subprocess
    exits, which turned a solver wedge into a silent 9-hour hang.
    """
    logger.info("  $ %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout for ordering
        text=True,
        bufsize=1,
    )
    tail: collections.deque[str] = collections.deque(maxlen=30)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line.rstrip())
    proc.wait()
    return proc.returncode, "\n".join(tail)


# Environment variables the resolvers read. ``env_setup.sh`` exports
# ``INFINIGEN_ROOT`` from ``.env``; the two Python env vars let callers
# point at alternate conda envs without touching ``.env``.
_INFINIGEN_ROOT_ENV_VAR = "INFINIGEN_ROOT"
_INFINIGEN_PYTHON_ENV_VAR = "STRAFER_INFINIGEN_PYTHON"
_ISAACLAB_PYTHON_ENV_VAR = "STRAFER_ISAACLAB_PYTHON"


def _resolve_env_python(env_var: str) -> str:
    """Read a Python-interpreter path from ``env_var`` and validate it."""
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"{env_var} is not set. Run `source env_setup.sh` from the repo "
            "root to load it from .env (copy .env.example to .env first if "
            "you have not yet)."
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise RuntimeError(f"{env_var}={value!r} points to a non-existent path.")
    return str(path)


def _resolve_infinigen_root() -> Path:
    """Resolve the Infinigen checkout via ``INFINIGEN_ROOT`` env var."""
    value = os.environ.get(_INFINIGEN_ROOT_ENV_VAR)
    if not value:
        raise RuntimeError(
            f"{_INFINIGEN_ROOT_ENV_VAR} is not set. Run `source env_setup.sh` "
            "from the repo root to load it from .env."
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(
            f"{_INFINIGEN_ROOT_ENV_VAR}={value!r} is not a directory."
        )
    return path


def _resolve_infinigen_python() -> str:
    """Return the Python interpreter with Infinigen + bpy + gin installed."""
    return _resolve_env_python(_INFINIGEN_PYTHON_ENV_VAR)


def _resolve_isaaclab_python() -> str:
    """Return the Python interpreter with ``pxr`` available (for USDC post-processing)."""
    return _resolve_env_python(_ISAACLAB_PYTHON_ENV_VAR)


def _build_generate_command(
    *,
    infinigen_python: str,
    output_folder: Path,
    seed: int,
    config: SceneGenConfig,
) -> list[str]:
    """Build the ``generate_indoors`` CLI invocation for one scene.

    Uses Infinigen's real CLI: ``-g`` for gin config files (without the
    ``.gin`` suffix) and ``-p`` for ``module.key=value`` overrides.
    """
    cmd: list[str] = [
        infinigen_python,
        "-m",
        "infinigen_examples.generate_indoors",
        "--output_folder",
        str(output_folder),
        "--seed",
        str(seed),
        "--task",
        "coarse",
    ]
    if config.gin_configs:
        cmd.append("-g")
        cmd.extend(config.gin_configs)

    overrides = list(config.gin_overrides)
    overrides.append(f"restrict_solving.solve_max_rooms={config.max_rooms}")
    cmd.append("-p")
    cmd.extend(overrides)
    return cmd


def _build_export_command(
    *,
    infinigen_python: str,
    input_folder: Path,
    output_folder: Path,
    texture_resolution: int,
) -> list[str]:
    """Build the ``infinigen.tools.export`` CLI invocation for one scene.

    ``--omniverse`` toggles schema adjustments (wattages, center-of-mass,
    zero-polygon-mesh removal) that the Omniverse / Isaac Sim USD runtime
    prefers. Dropping it produces a USDC that Isaac Sim can still load but
    with some material glitches.
    """
    return [
        infinigen_python,
        "-m",
        "infinigen.tools.export",
        "--input_folder",
        str(input_folder),
        "--output_folder",
        str(output_folder),
        "-f",
        "usdc",
        "-r",
        str(texture_resolution),
        "--omniverse",
    ]


# ---------------------------------------------------------------------------
# Ingest path — for scenes generated on Windows and transferred to DGX
# ---------------------------------------------------------------------------


def ingest_external_usd(
    *,
    source_dir: Path,
    output_dir: Path,
    preset_name: str = "windows_baseline",
) -> list[Path]:
    """Copy externally-generated scenes into the strafer_lab assets tree.

    The source directory is expected to contain one sub-directory per
    scene, each holding at least a ``*.usd`` file and optionally a
    ``*.blend`` file. Each scene is copied into ``output_dir`` and
    stamped with a ``scene_config.json`` noting which preset produced
    it (for provenance when the metadata extractor reads it).
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = PRESETS.get(preset_name, WINDOWS_BASELINE)
    imported: list[Path] = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue
        if not any(child.glob("*.usd")):
            logger.warning("Skipping %s: no .usd files", child)
            continue
        dest = output_dir / child.name
        if dest.exists():
            logger.warning("Skipping %s: destination already exists", dest)
            continue
        shutil.copytree(child, dest)
        (dest / "scene_config.json").write_text(json.dumps(preset.to_dict(), indent=2))
        imported.append(dest)
        logger.info("Imported %s → %s", child, dest)
    return imported


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Run Infinigen to produce new scenes")
    generate.add_argument(
        "--config", default="high_quality_dgx", choices=sorted(PRESETS.keys()),
    )
    generate.add_argument("--num-scenes", type=int, default=10)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--infinigen-root", type=Path, default=None)
    generate.add_argument(
        "--infinigen-python",
        default=None,
        help=f"Path to the Python interpreter that has Infinigen + bpy + gin "
             f"installed. Default: read from the {_INFINIGEN_PYTHON_ENV_VAR} "
             "env var (populated by env_setup.sh from .env).",
    )
    generate.add_argument("--seed-base", type=int, default=0)
    generate.add_argument(
        "--max-rooms",
        type=int,
        default=None,
        help="Override the preset's solver room cap "
        "(restrict_solving.solve_max_rooms). Presets default to 5 for "
        "high_quality_dgx and 1 for fast/windows; set higher for richer "
        "scenes, but solver runtime can blow up without a cap.",
    )

    ingest = sub.add_parser(
        "ingest",
        help="Import externally-generated scenes (e.g. from Windows) into the assets tree",
    )
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument(
        "--preset", default="windows_baseline", choices=sorted(PRESETS.keys()),
    )

    presets = sub.add_parser("presets", help="Print the available quality presets and exit")
    presets.add_argument("--json", action="store_true")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "presets":
        if args.json:
            print(json.dumps({k: v.to_dict() for k, v in PRESETS.items()}, indent=2))
        else:
            for name, preset in PRESETS.items():
                configs = ",".join(preset.gin_configs) or "(none)"
                print(
                    f"{name}: gin_configs=[{configs}] "
                    f"texture_res={preset.texture_resolution} — {preset.description}"
                )
        return 0

    if args.command == "generate":
        config = PRESETS[args.config]
        overrides: dict[str, Any] = {"random_seed_base": args.seed_base}
        if args.max_rooms is not None:
            overrides["max_rooms"] = args.max_rooms
        config = SceneGenConfig(**{**asdict(config), **overrides})
        results = generate_scenes(
            config=config,
            num_scenes=args.num_scenes,
            output_dir=args.output,
            infinigen_root=args.infinigen_root,
            infinigen_python=args.infinigen_python,
        )
        failures = [r for r in results if r.returncode != 0]
        if failures:
            logger.error("%d/%d scenes failed", len(failures), len(results))
            return 1
        logger.info("Successfully generated %d scenes", len(results))
        return 0

    if args.command == "ingest":
        imported = ingest_external_usd(
            source_dir=args.source,
            output_dir=args.output,
            preset_name=args.preset,
        )
        logger.info("Imported %d scenes", len(imported))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
