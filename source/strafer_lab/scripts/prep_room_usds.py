"""Generate high-quality Infinigen scenes and stage them as USDs.

The DGX Spark's 128 GB unified memory unlocks scene settings that are
infeasible on the 16 GB Windows workstation baseline: larger polygon
budgets, more diverse room types, richer per-room object counts, and
multi-room layouts. This script wraps Infinigen's generation entry
point with a curated config, then copies / post-processes the resulting
USDs into ``Assets/generated/scenes/`` where
``extract_scene_metadata.py`` (Task 8) can pick them up.

The Infinigen generator itself must run inside a Blender Python
environment. On DGX (Grace ARM64) this means a headless Blender build
with ``blender --background --python``. On hosts where Blender cannot
run headless, generate on a Windows workstation and transfer the
resulting USDs + ``.blend`` files to DGX — :func:`ingest_external_usd`
below handles that hand-off.

Usage examples:

    # Full run on DGX (requires Blender on PATH and Infinigen installed)
    python scripts/prep_room_usds.py generate \\
        --config high_quality_dgx \\
        --num-scenes 50 \\
        --output Assets/generated/scenes

    # Import externally-generated USDs (from the Windows workstation)
    python scripts/prep_room_usds.py ingest \\
        --source /mnt/transfer/infinigen_out \\
        --output Assets/generated/scenes
"""

from __future__ import annotations

import argparse
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
    """Tunable scene generation parameters.

    Defaults reflect the DGX Spark high-quality preset. See the
    :data:`PRESETS` dict for Windows / low-memory alternates.
    """

    name: str
    room_types: tuple[str, ...] = (
        "Kitchen",
        "Bedroom",
        "LivingRoom",
        "Hallway",
        "Bathroom",
        "Office",
        "DiningRoom",
    )
    min_rooms_per_scene: int = 2
    max_rooms_per_scene: int = 5
    min_objects_per_room: int = 8
    max_objects_per_room: int = 25
    target_poly_count_millions: float = 4.0
    max_poly_count_millions: float = 8.0
    enable_materials: bool = True
    enable_lighting_variation: bool = True
    enable_small_items: bool = True
    multi_story: bool = False
    scene_size_range_m: tuple[float, float] = (30.0, 120.0)
    random_seed_base: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


# High-quality DGX preset — exploits 128 GB unified memory for much
# larger polygon budgets than the 16 GB Windows baseline.
HIGH_QUALITY_DGX = SceneGenConfig(
    name="high_quality_dgx",
    target_poly_count_millions=4.0,
    max_poly_count_millions=8.0,
    max_objects_per_room=25,
    min_rooms_per_scene=3,
    max_rooms_per_scene=5,
)

# Matched Windows preset (for parity when offloading scene generation).
WINDOWS_BASELINE = SceneGenConfig(
    name="windows_baseline",
    target_poly_count_millions=1.5,
    max_poly_count_millions=3.0,
    max_objects_per_room=15,
    min_rooms_per_scene=1,
    max_rooms_per_scene=3,
)

# Low-memory fallback (for ARM64 Blender builds with constrained RAM).
LOW_MEMORY = SceneGenConfig(
    name="low_memory",
    target_poly_count_millions=0.8,
    max_poly_count_millions=1.5,
    max_objects_per_room=10,
    min_rooms_per_scene=1,
    max_rooms_per_scene=2,
    enable_small_items=False,
)

PRESETS: dict[str, SceneGenConfig] = {
    cfg.name: cfg for cfg in (HIGH_QUALITY_DGX, WINDOWS_BASELINE, LOW_MEMORY)
}


# ---------------------------------------------------------------------------
# Invocation — wraps Infinigen's entry point via subprocess
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    scene_dir: Path
    blend_path: Path | None = None
    usd_path: Path | None = None
    metadata_path: Path | None = None
    returncode: int = 0
    stderr_tail: str = ""


def generate_scenes(
    *,
    config: SceneGenConfig,
    num_scenes: int,
    output_dir: Path,
    infinigen_root: Path | None = None,
    blender_binary: str | None = None,
) -> list[GenerationResult]:
    """Run Infinigen ``num_scenes`` times and write USDs under ``output_dir``.

    This function shells out to Blender/Infinigen per scene so each run
    is isolated (Infinigen occasionally leaks state between runs). The
    wrapper writes a ``scene_config.json`` next to each generated scene
    so the downstream metadata extractor and dataset exporters can tie
    each scene back to the preset that produced it.

    If ``blender_binary`` is None, the resolver picks the best available
    Blender on this host via :func:`_resolve_blender_binary`. On DGX Spark
    that is the source-built 4.2.0 under ``~/Workspace/blender-build/``
    (apt ships only Blender 4.0.2 on aarch64 Ubuntu 24.04, and Infinigen
    pins 4.2.0 — see ``blender-build/README.md`` for the build recipe).
    """
    blender_binary = blender_binary or _resolve_blender_binary()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    infinigen_root = Path(infinigen_root) if infinigen_root else _guess_infinigen_root()

    results: list[GenerationResult] = []
    for i in range(num_scenes):
        seed = config.random_seed_base + i
        scene_name = f"{config.name}_{i:03d}_seed{seed}"
        scene_dir = output_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_infinigen_command(
            infinigen_root=infinigen_root,
            blender_binary=blender_binary,
            scene_dir=scene_dir,
            seed=seed,
            config=config,
        )
        logger.info("Running scene %d/%d: %s", i + 1, num_scenes, " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=str(infinigen_root) if infinigen_root.exists() else None,
            capture_output=True,
            text=True,
        )
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
        result = GenerationResult(
            scene_dir=scene_dir,
            returncode=proc.returncode,
            stderr_tail=stderr_tail,
        )
        if proc.returncode != 0:
            logger.error("Scene %s failed (rc=%d): %s", scene_name, proc.returncode, stderr_tail)
            results.append(result)
            continue

        blend_candidates = sorted(scene_dir.glob("*.blend"))
        usd_candidates = sorted(scene_dir.glob("*.usd"))
        result.blend_path = blend_candidates[0] if blend_candidates else None
        result.usd_path = usd_candidates[0] if usd_candidates else None

        config_path = scene_dir / "scene_config.json"
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

        results.append(result)
    return results


def _guess_infinigen_root() -> Path:
    """Best-effort guess for the Infinigen checkout location on DGX."""
    candidates = [
        Path.home() / "Workspace" / "infinigen",
        Path("/home/zachoines/Workspace/infinigen"),
        Path("/opt/infinigen"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


# -----------------------------------------------------------------------------
# Blender binary resolution — reads STRAFER_BLENDER_BIN from the process
# environment. The single source of truth is the shell-level variable that
# ``env_setup.sh`` loads from ``.env`` at session start. There is no Python-
# side ``.env`` fallback and no PATH fallback: Ubuntu 24.04 aarch64 ships
# Blender 4.0.2 via apt, which Infinigen (pinning ``bpy==4.2.0``) will refuse
# at generation time, and a dual-source lookup (env var + .env parsed by
# Python) would drift silently whenever someone edits ``.env`` without
# re-sourcing the shell wrapper. Failing loud is strictly better.
#
# If this resolver errors, the user forgot ``source env_setup.sh``; the
# error message tells them exactly what to do.
# -----------------------------------------------------------------------------

_BLENDER_ENV_VAR = "STRAFER_BLENDER_BIN"


def _resolve_blender_binary() -> str:
    """Return the absolute path of a Blender 4.2 binary from the environment.

    Reads :data:`_BLENDER_ENV_VAR` from :data:`os.environ`. Raises
    :class:`RuntimeError` if the variable is unset or points at a
    non-existent path.
    """
    value = os.environ.get(_BLENDER_ENV_VAR)
    if not value:
        raise RuntimeError(
            f"{_BLENDER_ENV_VAR} is not set. Run `source env_setup.sh` from "
            "the repo root to load it from .env (copy .env.example to .env "
            "first if you have not yet). Infinigen pins bpy==4.2.0 and the "
            "apt-installed /usr/bin/blender is 4.0.2 on Ubuntu 24.04 aarch64, "
            "so a silent PATH fallback would fail downstream anyway. See "
            "blender-build/README.md for the aarch64 source build recipe."
        )

    path = Path(value).expanduser()
    if not path.exists():
        raise RuntimeError(
            f"{_BLENDER_ENV_VAR}={value!r} points to a non-existent path. "
            "Check your .env or shell environment."
        )
    return str(path)


def _build_infinigen_command(
    *,
    infinigen_root: Path,
    blender_binary: str,
    scene_dir: Path,
    seed: int,
    config: SceneGenConfig,
) -> list[str]:
    """Build the CLI invocation for one Infinigen scene.

    Infinigen exposes a ``generate_indoors`` entry point through its
    own runner module. Room selection and object budgets are plumbed
    through overrides passed via ``--gin_param``-style overrides when
    the Infinigen version supports them. For older versions, these
    overrides are best-effort and the caller should verify the resulting
    scene_metadata matches the requested preset.
    """
    script = infinigen_root / "infinigen_examples" / "generate_indoors.py"
    cmd: list[str] = [
        blender_binary,
        "--background",
        "--python",
        str(script),
        "--",
        "--output_folder",
        str(scene_dir),
        "--seed",
        str(seed),
    ]

    cmd.extend(
        [
            "--override",
            f"configs.rooms.room_types={list(config.room_types)}",
            "--override",
            f"configs.rooms.min_rooms={config.min_rooms_per_scene}",
            "--override",
            f"configs.rooms.max_rooms={config.max_rooms_per_scene}",
            "--override",
            f"configs.objects.min_per_room={config.min_objects_per_room}",
            "--override",
            f"configs.objects.max_per_room={config.max_objects_per_room}",
            "--override",
            f"configs.geometry.target_million_polys={config.target_poly_count_millions}",
            "--override",
            f"configs.geometry.max_million_polys={config.max_poly_count_millions}",
            "--override",
            f"configs.materials.enable={str(config.enable_materials).lower()}",
            "--override",
            f"configs.lighting.variation={str(config.enable_lighting_variation).lower()}",
            "--override",
            f"configs.objects.include_small_items={str(config.enable_small_items).lower()}",
            "--override",
            f"configs.geometry.multi_story={str(config.multi_story).lower()}",
        ]
    )
    return cmd


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
        "--blender",
        default=None,
        help="Path to Blender 4.2 binary (default: auto-resolve via "
        "_resolve_blender_binary() — prefers the DGX source build at "
        "~/Workspace/blender-build/build_blender/bin/blender, falls back to "
        "/usr/bin/blender).",
    )
    generate.add_argument("--seed-base", type=int, default=0)

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
                print(f"{name}: max_objects_per_room={preset.max_objects_per_room} "
                      f"rooms=[{preset.min_rooms_per_scene},{preset.max_rooms_per_scene}] "
                      f"polys<={preset.max_poly_count_millions}M")
        return 0

    if args.command == "generate":
        config = PRESETS[args.config]
        config = SceneGenConfig(**{**asdict(config), "random_seed_base": args.seed_base})
        results = generate_scenes(
            config=config,
            num_scenes=args.num_scenes,
            output_dir=args.output,
            infinigen_root=args.infinigen_root,
            blender_binary=args.blender,
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
