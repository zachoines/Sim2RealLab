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

5. ``extract_scene_metadata.py --from-usd`` embeds the labeled
   ``objects[]`` into the USD's root-prim ``customData`` and applies the
   ``UsdSemantics`` detection labels. Runs under the Isaac Sim Kit
   launcher (``$ISAACLAB``) — the semantics schema is Kit-provided. This
   makes one ``generate`` yield a capture-ready *and* detections-ready
   scene (``--no-scene-metadata`` skips it for a geometry-only build).

6. ``validate_scene_connectivity.py`` generates the scene's occupancy grid
   (cached as ``<scene>/occupancy.npy``) via the occupancy-map extension,
   verifies room-to-room reachability with the shared grid planner, forces
   open any closed doors blocking a doorway, and merges the
   ``connectivity[]`` graph + ``multi_story`` flag into ``customData``. Also
   under the Kit launcher (``--no-connectivity`` skips it).

Presets wrap Infinigen's real knobs (``-g`` gin config files and ``-p``
gin-style parameter overrides).

Usage::

    python scripts/prep_room_usds.py generate \\
        --config high_quality_dgx \\
        --num-scenes 10 \\
        --output Assets/generated/scenes

    # Genuine single-room scene (exactly one furnished room):
    python scripts/prep_room_usds.py generate \\
        --config true_singleroom \\
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
- ``STRAFER_ISAACLAB_PYTHON`` — Python with ``pxr`` importable (postprocess).
- ``ISAACLAB`` — the Kit launcher (``isaaclab.sh -p``) used to embed scene
  metadata + UsdSemantics labels (skipped under ``--no-scene-metadata``).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import shlex
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
    #
    # NOTE: ``solve_max_rooms`` only limits how many rooms get *furnished*
    # (Infinigen's own docstring: "only place objects in at most this many
    # rooms"); it does NOT constrain the floorplan's room *count*. The
    # room-count floor is baked into ``home_room_constraints``'
    # ``node_constraint`` (Entrance/LivingRoom/Kitchen/Bedroom/Bathroom each
    # >= 1), so the constraint-solver path always yields ~5 rooms. To pin an
    # exact room count, use ``floor_plan_file`` below.
    max_rooms: int = 5
    # Predefined floor-plan contour to force an EXACT room count. When set,
    # names a JSON file shipped under this script's ``floor_plans/`` dir; the
    # generate command injects ``-p Solver.floor_plan=<abs path>`` so
    # Infinigen's ``PredefinedFloorPlanSolver`` builds precisely the rooms the
    # JSON defines (bypassing the constraint solver's room-count floor). The
    # rooms are still furnished against ``home_furniture_constraints`` per
    # their room type (e.g. a ``living-room_0/0`` room gets living-room
    # furniture), so a furnishable room still yields groundable objects.
    # Left ``None`` for the constraint-solved presets (the historical path).
    floor_plan_file: str | None = None
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

# Genuine single-room scene with EXACTLY one furnished room. The
# constraint-solver path can't do this — ``home_room_constraints`` forces
# Entrance/LivingRoom/Kitchen/Bedroom/Bathroom each >= 1 (a ~5-room floor),
# and ``solve_max_rooms`` only limits how many rooms get *furnished*. So the
# room count is pinned deterministically via a predefined floor-plan contour
# (``floor_plan_file`` -> ``-p Solver.floor_plan=<abs>`` ->
# ``PredefinedFloorPlanSolver``), whose single room is a furnishable
# LivingRoom that yields groundable objects. ``RoomConstants.n_stories=1``
# is pinned because the predefined-floor-plan path indexes one solidifier
# per story and crashes under a multistory overlay. Light (512-px, 1 room)
# so it loads without OOMing the GB10's unified memory.
TRUE_SINGLEROOM = SceneGenConfig(
    name="true_singleroom",
    gin_configs=("base_indoors",),
    gin_overrides=(
        "compose_indoors.terrain_enabled=False",
        "RoomConstants.n_stories=1",
    ),
    max_rooms=1,
    texture_resolution=512,
    floor_plan_file="true_singleroom.json",
    description="Exactly one furnished LivingRoom via a predefined floor plan.",
)

# Genuine two-room scene: exactly two connected rooms (a furnishable
# LivingRoom + an adjacent Kitchen) sharing a door, via a predefined
# floor-plan contour. The cross-room fixture for the eventual autonomy-stack
# cross-room lane; also proves the predefined-floor-plan room-count control
# does N=2. Same single-story pin + light textures as the single-room preset.
TRUE_TWOROOM = SceneGenConfig(
    name="true_tworoom",
    gin_configs=("base_indoors",),
    gin_overrides=(
        "compose_indoors.terrain_enabled=False",
        "RoomConstants.n_stories=1",
    ),
    max_rooms=2,
    texture_resolution=512,
    floor_plan_file="true_tworoom.json",
    description="Exactly two connected furnished rooms via a predefined floor plan.",
)

PRESETS: dict[str, SceneGenConfig] = {
    cfg.name: cfg for cfg in (HIGH_QUALITY_DGX, TRUE_SINGLEROOM, TRUE_TWOROOM)
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
    author_scene_metadata: bool = True,
    validate_connectivity: bool = True,
) -> list[GenerationResult]:
    """Run Infinigen ``num_scenes`` times and stage USDCs under ``output_dir``.

    For each scene:

    - Coarse generate → ``output_dir/<scene_name>/coarse/scene.blend``
    - Export to USDC  → ``output_dir/<scene_name>/export/export_scene.blend/*.usdc``
      (plus a sibling ``textures/`` directory)
    - Symlink         → ``output_dir/<scene_name>.usdc`` points at the USDC
      inside the export tree so ``strafer_env_cfg._get_scene_usd_paths``
      discovers the scene without needing to recurse.
    - Scene metadata  → ``extract_scene_metadata.py --from-usd`` embeds the
      labeled ``objects[]`` into the USD's ``customData`` and applies the
      ``UsdSemantics`` detection labels, so one ``generate`` yields a
      capture-ready *and* detections-ready scene. Runs under the Isaac Sim
      Kit launcher (``$ISAACLAB``) because the semantics schema needs it.
      Disable with ``author_scene_metadata=False``.
    - Connectivity    → ``validate_scene_connectivity.py`` generates the
      occupancy grid (cached as ``<scene>/occupancy.npy``), verifies
      room-to-room reachability, forces open doors blocking a doorway, and
      merges the ``connectivity[]`` graph + ``multi_story`` flag into
      ``customData``. Also under the Kit launcher (the occupancy-map
      extension is Kit-provided). Disable with ``validate_connectivity=False``.

    A ``scene_config.json`` is written next to each scene for provenance.
    The orchestrator asserts that both the ``.blend`` (after coarse gen)
    and the ``.usdc`` (after export) actually exist; if either step
    silently produced no file, the result is flagged with a nonzero
    ``returncode`` instead of being reported as success.
    """
    # Verify every downstream subprocess's env var is set before we
    # start the expensive coarse stage.
    validate_required_env_for_generate(
        author_scene_metadata=author_scene_metadata or validate_connectivity,
    )

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

        # --- Step 5: embed scene metadata + detection-label semantics ---
        if author_scene_metadata:
            meta_rc, meta_stderr = _run_extract_metadata(result.usd_path)
            if meta_rc != 0:
                logger.error(
                    "Scene %s: scene-metadata authoring failed (rc=%d)\n%s",
                    scene_name, meta_rc, meta_stderr,
                )
                result.returncode = meta_rc
                result.stderr_tail = meta_stderr
                results.append(result)
                continue

        # --- Step 6: connectivity graph + cached occupancy ---
        if validate_connectivity:
            conn_rc, conn_stderr = _run_validate_connectivity(result.usd_path)
            if conn_rc != 0:
                logger.error(
                    "Scene %s: connectivity validation failed (rc=%d)\n%s",
                    scene_name, conn_rc, conn_stderr,
                )
                result.returncode = conn_rc
                result.stderr_tail = conn_stderr
                results.append(result)
                continue

        # --- Step 7: provenance ---
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


def _run_extract_metadata(usdc_path: Path) -> tuple[int, str]:
    """Embed scene metadata + UsdSemantics labels into the freshly-staged USDC.

    Runs ``extract_scene_metadata.py --from-usd`` under the Isaac Sim Kit
    launcher (``$ISAACLAB`` / ``isaaclab.sh -p``), not the plain
    ``STRAFER_ISAACLAB_PYTHON``: applying the ``UsdSemantics.LabelsAPI`` the
    detections annotator reads needs the Kit runtime.
    """
    script = Path(__file__).resolve().parent / "extract_scene_metadata.py"
    cmd = [
        *_resolve_isaaclab_launcher(),
        str(script),
        "--from-usd",
        "--usd",
        str(usdc_path),
    ]
    return _run_subprocess(cmd, cwd=script.parents[3])


def _run_validate_connectivity(usdc_path: Path) -> tuple[int, str]:
    """Compute + author the room connectivity graph + cache the occupancy grid.

    Runs ``validate_scene_connectivity.py`` under the Isaac Sim Kit launcher
    (``$ISAACLAB``): the occupancy-map extension that reads the stage's
    physics colliders is Kit-provided. Must follow the metadata-authoring step
    — it reads ``rooms[]`` (back-filling from floor meshes when absent) and
    merges ``connectivity[]`` back into the same ``customData`` payload.
    """
    script = Path(__file__).resolve().parent / "validate_scene_connectivity.py"
    cmd = [
        *_resolve_isaaclab_launcher(),
        str(script),
        "--usd",
        str(usdc_path),
    ]
    return _run_subprocess(cmd, cwd=script.parents[3])


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
_ISAACLAB_LAUNCHER_ENV_VAR = "ISAACLAB"


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


def _resolve_isaaclab_launcher() -> list[str]:
    """Return the Isaac Sim Kit launcher argv (``isaaclab.sh -p``).

    Read from ``$ISAACLAB`` (a command string env_setup.sh loads from
    ``.env``). The scene-metadata authoring pass runs under it because the
    ``UsdSemantics`` schema is a Kit-provided plugin, unavailable to plain
    ``STRAFER_ISAACLAB_PYTHON``.
    """
    value = os.environ.get(_ISAACLAB_LAUNCHER_ENV_VAR)
    if not value:
        raise RuntimeError(
            f"{_ISAACLAB_LAUNCHER_ENV_VAR} is not set. Run `source env_setup.sh` "
            "from the repo root to load it from .env. It pins the Isaac Sim Kit "
            "launcher used to embed scene metadata + UsdSemantics labels."
        )
    parts = shlex.split(value)
    launcher = Path(parts[0]).expanduser() if parts else None
    if launcher is None or not launcher.exists():
        raise RuntimeError(
            f"{_ISAACLAB_LAUNCHER_ENV_VAR}={value!r} points to a non-existent path."
        )
    argv = [str(launcher), *parts[1:]]
    # The launcher must run the downstream script as a Python program
    # (``isaaclab.sh -p <script>``). $ISAACLAB is documented as the bare
    # launcher path — manual ``$ISAACLAB -p ...`` usage adds the ``-p`` itself —
    # but the auto-invocation here appends the script directly, so guarantee a
    # single ``-p`` whether or not the env var already carries one.
    if "-p" not in argv:
        argv.append("-p")
    return argv


def validate_required_env_for_generate(*, author_scene_metadata: bool = True) -> None:
    """Fail-fast check that every env var ``generate_scenes`` will need is set.

    The wrapper's pipeline takes multi-hour Infinigen runs before it
    needs ``STRAFER_ISAACLAB_PYTHON`` in :func:`_run_postprocess` and
    ``$ISAACLAB`` in :func:`_run_extract_metadata`. Without this upfront
    check, a missing var would crash after coarse + export completed and
    discard a day's worth of compute. Call this from the top of
    :func:`generate_scenes` and from any entry point that calls it.

    Raises the same :class:`RuntimeError` types the underlying resolvers
    raise, so existing error messages (with the ``source env_setup.sh``
    hint) are preserved.
    """
    _resolve_infinigen_root()
    _resolve_infinigen_python()
    _resolve_isaaclab_python()
    if author_scene_metadata:
        _resolve_isaaclab_launcher()


# Predefined floor-plan contours ship in the repo (not the vendored
# Infinigen tree) so they travel with this PR and stay reproducible. The
# ``-g`` gin-file mechanism only resolves stems inside Infinigen's own
# ``configs_indoor/`` tree, so a repo-local gin is unreachable; instead the
# room count is pinned via the ``-p Solver.floor_plan=<abs path>`` binding.
_FLOOR_PLANS_DIR = Path(__file__).resolve().parent / "floor_plans"


def _resolve_floor_plan_path(name: str) -> Path:
    """Resolve a predefined floor-plan JSON shipped under ``floor_plans/``.

    Presets reference the contour by bare filename; the generate command
    passes its *absolute* path to Infinigen via ``-p Solver.floor_plan=...``.
    The path must be absolute because the coarse-gen subprocess runs with
    ``cwd=INFINIGEN_ROOT`` — a relative path would resolve against the
    vendored Infinigen tree rather than this repo.
    """
    path = _FLOOR_PLANS_DIR / name
    if not path.is_file():
        raise RuntimeError(
            f"Floor-plan file {name!r} not found under {_FLOOR_PLANS_DIR}. "
            "A preset that sets floor_plan_file must reference a JSON shipped "
            "in the repo's scripts/floor_plans/ directory."
        )
    return path


def _build_generate_command(
    *,
    infinigen_python: str,
    output_folder: Path,
    seed: int,
    config: SceneGenConfig,
) -> list[str]:
    """Build the ``generate_indoors`` CLI invocation for one scene.

    Uses Infinigen's real CLI: ``-g`` for gin config files (without the
    ``.gin`` suffix) and ``-p`` for ``module.key=value`` overrides. Presets
    with a ``floor_plan_file`` additionally bind
    ``Solver.floor_plan=<abs path>`` so Infinigen's
    ``PredefinedFloorPlanSolver`` builds an exact room count.
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
    if config.floor_plan_file:
        floor_plan_path = _resolve_floor_plan_path(config.floor_plan_file)
        # Quote the path so gin parses the RHS as a string literal; the arg
        # travels through subprocess argv (no shell), so the quotes are part
        # of the binding and Infinigen's sanitize_override leaves it intact.
        overrides.append(f'Solver.floor_plan="{floor_plan_path}"')
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
# Ingest path — for scenes generated off-host and transferred to the DGX
# ---------------------------------------------------------------------------


def ingest_external_usd(
    *,
    source_dir: Path,
    output_dir: Path,
    preset_name: str = "high_quality_dgx",
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

    preset = PRESETS.get(preset_name, HIGH_QUALITY_DGX)
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
    generate.add_argument(
        "--no-scene-metadata",
        action="store_true",
        help="Skip embedding scene metadata + UsdSemantics labels into each "
        "USD (the geometry-only path). The scene loads but is neither "
        "capture-ready nor detections-ready until extract_scene_metadata runs.",
    )
    generate.add_argument(
        "--no-connectivity",
        action="store_true",
        help="Skip the room-connectivity step (occupancy grid + connectivity[] "
        "graph + door-open verification). The scene loads but carries no "
        "verified room graph until validate_scene_connectivity runs.",
    )

    ingest = sub.add_parser(
        "ingest",
        help="Import externally-generated scenes (e.g. from Windows) into the assets tree",
    )
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument(
        "--preset", default="high_quality_dgx", choices=sorted(PRESETS.keys()),
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
            author_scene_metadata=not args.no_scene_metadata,
            validate_connectivity=not args.no_connectivity,
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
