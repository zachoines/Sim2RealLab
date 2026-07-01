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

Scenes are described along two dimensions:

- **layout** — ``--rooms <type...>`` pins an EXACT tiled floor plan (the room
  count is the number of types given, duplicates allowed); omit it for an
  organic constraint-solved house (``--max-rooms N`` caps it).
- **quality/resources** — ``--quality {high,low}`` composes texture resolution
  + object density; ``--texture-res`` / ``--object-density`` /
  ``--geometry-detail`` override individual knobs (available memory is the
  driver — texture VRAM + geometry footprint at scene-load time).

Usage::

    # Organic multi-room house (the perception corpus):
    python scripts/prep_room_usds.py generate \\
        --quality high --num-scenes 10 \\
        --output Assets/generated/scenes

    # Exact single-room scene (one furnished living room), light:
    python scripts/prep_room_usds.py generate \\
        --rooms living-room --quality low --name singleroom \\
        --output Assets/generated/scenes

    # Exact two connected rooms; adversarial two-of-a-kind:
    python scripts/prep_room_usds.py generate \\
        --rooms living-room kitchen --quality low --output Assets/generated/scenes
    python scripts/prep_room_usds.py generate \\
        --rooms bedroom bedroom --quality low --output Assets/generated/scenes

    # List the available dimensions:
    python scripts/prep_room_usds.py info

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
# Generation dimensions — layout (rooms) and quality/resources
# ---------------------------------------------------------------------------


# Room types the constraint solver actually furnishes — ``home_furniture_``
# ``constraints`` only writes furniture rules for these, so a tiled scene's
# rooms must come from this set to contain groundable objects.
FURNISHABLE_ROOMS: tuple[str, ...] = (
    "living-room",
    "kitchen",
    "bedroom",
    "bathroom",
    "dining-room",
)

# Per-type room footprint (metres) for the tiler — a bathroom is small, a
# living room large. Height is uniform so a row of rooms shares full-height
# interior walls; only the width varies by type.
_ROOM_WIDTHS: dict[str, float] = {
    "living-room": 6.0,
    "dining-room": 5.0,
    "bedroom": 5.0,
    "kitchen": 4.0,
    "bathroom": 3.0,
}
_ROOM_HEIGHT = 5.0
_DEFAULT_ROOM_WIDTH = 5.0

# Object-count dial. All rooms are always furnished; this only scales the
# solver's annealing budget, and more steps place more objects. Maps to
# Infinigen's ``compose_indoors.solve_steps_{large,medium,small}`` — ``high``
# is ``base_indoors``' budget, ``low`` is the ``fast_solve`` budget (fewer
# props, faster).
_OBJECT_DENSITY_STEPS: dict[str, tuple[int, int, int]] = {
    "high": (300, 200, 50),
    "low": (100, 40, 5),
}

# Geometry/mesh complexity. ``standard`` is Infinigen's base indoor meshes
# (no displacement — the corpus baseline). ``displacement`` layers on
# ``real_geometry`` micro-geometry (much heavier; opt-in when memory allows).
_GEOMETRY_DETAILS: tuple[str, ...] = ("standard", "displacement")


@dataclass
class QualityTier:
    """A composed default for the memory-sensitive resource knobs."""

    texture_resolution: int
    object_density: str  # key into ``_OBJECT_DENSITY_STEPS``


# ``--quality`` picks a tier; each knob is individually overridable. Geometry
# stays ``standard`` in both tiers — displacement is a deliberate opt-in.
QUALITY_TIERS: dict[str, QualityTier] = {
    "high": QualityTier(texture_resolution=1024, object_density="high"),
    "low": QualityTier(texture_resolution=512, object_density="low"),
}


@dataclass
class SceneGenConfig:
    """Resolved parameters for one scene generation, along two dimensions.

    **Layout** — ``rooms`` is an explicit list of furnishable room types; the
    count is ``len(rooms)`` and duplicates are allowed (e.g. two bedrooms for
    adversarial same-class training). When set, the floorplan is pinned
    EXACTLY via a parametric predefined contour
    (``PredefinedFloorPlanSolver``). When ``None``, the constraint solver runs
    and the layout is organic — a natural multi-room house (the perception
    corpus), room count bounded by ``max_rooms``.

    **Quality / resources** — ``texture_resolution`` (VRAM),
    ``object_density`` (annealing budget → object count; all rooms are
    furnished either way), and ``geometry_detail`` (``standard`` base meshes
    vs ``displacement`` micro-geometry).

    Everything is recorded into ``scene_config.json`` next to each scene.
    """

    name: str
    # Explicit room-type layout (tiled mode). ``None`` -> organic solve.
    rooms: tuple[str, ...] | None = None
    # Solver room cap — organic mode only (how many rooms it fits + furnishes).
    # Ignored in tiled mode, where all listed rooms are furnished.
    max_rooms: int = 5
    texture_resolution: int = 1024
    object_density: str = "high"  # key into ``_OBJECT_DENSITY_STEPS``
    geometry_detail: str = "standard"  # one of ``_GEOMETRY_DETAILS``
    # Base gin config file stems (no ``.gin`` suffix) + always-on overrides.
    gin_configs: tuple[str, ...] = ("base_indoors",)
    gin_overrides: tuple[str, ...] = ("compose_indoors.terrain_enabled=False",)
    # Seed base; scene i uses seed = random_seed_base + i.
    random_seed_base: int = 0
    # Human-readable description (recorded in scene_config.json).
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def _build_floor_plan(rooms: Iterable[str]) -> dict[str, Any]:
    """Build a ``PredefinedFloorPlanSolver`` contour for an exact room list.

    Rooms are tiled left-to-right in a single row, each a rectangle of uniform
    height and per-type width. Adjacent rooms share a **solid** full-height
    wall (built by Infinigen's ``make_room`` from the two room boxes) carrying
    a single connecting ``door`` cutter — no ``interiors`` entries, because
    Infinigen routes those through ``make_window_cutter`` (glass panels), not a
    solid wall. The outer perimeter carries an entrance + windows. The returned
    dict matches Infinigen's floor-plan schema
    (``configs_indoor/floor_plans/predefined.json``): every ``shape`` is a
    ``shapely.*`` expression *string* (Infinigen ``eval``s it). Room keys use
    the ``<semantics>_<level>/<n>`` form so duplicate types get distinct
    instance indices (``bedroom_0/0`` + ``bedroom_0/1`` = two Bedrooms).
    """
    rooms = list(rooms)
    if not rooms:
        raise ValueError("a floor plan needs at least one room")
    door = 2.0
    win = 2.0
    h = _ROOM_HEIGHT
    y_lo = h / 2 - door / 2
    y_hi = h / 2 + door / 2
    fp: dict[str, dict[str, Any]] = {
        "rooms": {},
        "doors": {},
        "interiors": {},
        "windows": {},
        "entrance": {},
    }
    counts: dict[str, int] = {}
    x0 = 0.0
    for i, rtype in enumerate(rooms):
        w = _ROOM_WIDTHS.get(rtype, _DEFAULT_ROOM_WIDTH)
        n = counts.get(rtype, 0)
        counts[rtype] = n + 1
        fp["rooms"][f"{rtype}_0/{n}"] = {"shape": f"shapely.box({x0},0,{x0 + w},{h})"}
        cx = x0 + w / 2
        fp["windows"][f"window.{i}"] = {
            "shape": f"shapely.LineString([({cx - win / 2},{h}),({cx + win / 2},{h})])"
        }
        if i < len(rooms) - 1:
            x = x0 + w  # shared wall with the next room
            # Only a door cutter on the shared wall — the two room boxes already
            # build a solid full-height wall there via Infinigen's make_room. An
            # ``interiors`` entry would be run through make_window_cutter (tagged
            # Semantics.Window), punching an unwanted see-through interior panel.
            fp["doors"][f"door.{i}"] = {
                "shape": f"shapely.LineString([({x},{y_lo}),({x},{y_hi})])"
            }
        x0 += w
    # Entrance on the far-left exterior wall; a window on the far-right wall.
    fp["entrance"]["entrance"] = {
        "shape": f"shapely.LineString([(0,{y_lo}),(0,{y_hi})])"
    }
    fp["windows"]["window.right"] = {
        "shape": f"shapely.LineString([({x0},{h / 2 - win / 2}),({x0},{h / 2 + win / 2})])"
    }
    return fp


def resolve_rooms(
    *, rooms: list[str] | None, num_rooms: int | None, room_types: list[str] | None
) -> tuple[str, ...] | None:
    """Resolve the room-layout dimension from CLI-style inputs.

    ``rooms`` (explicit list) wins. Else ``num_rooms`` round-robins over
    ``room_types`` (default a single living-room). ``None`` for all -> organic
    (constraint-solved) mode. Validates every type is furnishable.
    """
    if rooms:
        resolved = tuple(rooms)
    elif num_rooms is not None:
        if num_rooms < 1:
            raise ValueError("--num-rooms must be >= 1")
        palette = room_types or ["living-room"]
        resolved = tuple(palette[i % len(palette)] for i in range(num_rooms))
    else:
        return None
    bad = sorted({r for r in resolved if r not in FURNISHABLE_ROOMS})
    if bad:
        raise ValueError(
            f"room types {bad} are not furnishable; choose from {list(FURNISHABLE_ROOMS)} "
            "(only these carry Infinigen furniture rules, so a scene of other "
            "types would have no groundable objects)."
        )
    return resolved


def _default_scene_name(rooms: tuple[str, ...] | None, quality: str) -> str:
    """Derive a scene name stem from the dimensions when ``--name`` is unset."""
    if rooms is None:
        return f"organic_{quality}"
    return f"{len(rooms)}room_{quality}"


def build_scene_config(
    *,
    name: str | None,
    rooms: tuple[str, ...] | None,
    quality: str,
    texture_resolution: int | None,
    object_density: str | None,
    geometry_detail: str | None,
    max_rooms: int,
    seed_base: int,
) -> SceneGenConfig:
    """Compose a :class:`SceneGenConfig` from the two dimensions.

    ``quality`` supplies the tier defaults; the explicit knob arguments (when
    not ``None``) override them.
    """
    tier = QUALITY_TIERS[quality]
    return SceneGenConfig(
        name=name or _default_scene_name(rooms, quality),
        rooms=rooms,
        max_rooms=max_rooms,
        texture_resolution=texture_resolution or tier.texture_resolution,
        object_density=object_density or tier.object_density,
        geometry_detail=geometry_detail or "standard",
        random_seed_base=seed_base,
        description=(
            f"{'organic ' + str(max_rooms) + '-room-max house' if rooms is None else 'tiled ' + '+'.join(rooms)}"
            f", quality={quality}"
        ),
    )


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

        # --- Step 0: tiled layout -> write this scene's predefined floor plan ---
        # The tiler emits the contour for the exact room list; it's written into
        # the scene dir (provenance + the path Solver.floor_plan reads). Organic
        # configs (rooms=None) skip this and let the constraint solver run.
        floor_plan_path: Path | None = None
        if config.rooms is not None:
            floor_plan_path = scene_dir / "floor_plan.json"
            floor_plan_path.write_text(json.dumps(_build_floor_plan(config.rooms), indent=2))

        # --- Step 1: coarse generation ---
        gen_cmd = _build_generate_command(
            infinigen_python=infinigen_python,
            output_folder=coarse_dir,
            seed=seed,
            config=config,
            floor_plan_path=floor_plan_path,
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


def _build_generate_command(
    *,
    infinigen_python: str,
    output_folder: Path,
    seed: int,
    config: SceneGenConfig,
    floor_plan_path: Path | None = None,
) -> list[str]:
    """Build the ``generate_indoors`` CLI invocation for one scene.

    Uses Infinigen's real CLI: ``-g`` for gin config files (without the
    ``.gin`` suffix) and ``-p`` for ``module.key=value`` overrides. The two
    scene dimensions map to overrides:

    * **layout** — a tiled config (``rooms`` set) binds
      ``Solver.floor_plan=<abs path>`` at ``floor_plan_path`` (a per-scene JSON
      the tiler wrote) so Infinigen's ``PredefinedFloorPlanSolver`` builds
      exactly those rooms, pins ``RoomConstants.n_stories=1`` (the predefined
      path indexes one solidifier per story), and furnishes all of them. An
      organic config caps the solver at ``max_rooms``.
    * **quality** — ``object_density`` sets the solver's step budget and
      ``geometry_detail=displacement`` layers on ``real_geometry``.
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

    # Object count / density — scale the annealing budget (all rooms furnished).
    large, medium, small = _OBJECT_DENSITY_STEPS[config.object_density]
    overrides += [
        f"compose_indoors.solve_steps_large={large}",
        f"compose_indoors.solve_steps_medium={medium}",
        f"compose_indoors.solve_steps_small={small}",
    ]

    # Geometry/mesh complexity — displacement is the heavy opt-in.
    if config.geometry_detail == "displacement":
        overrides += [
            'set_displacement_mode.displacement_mode="DISPLACEMENT"',
            "compose_indoors.enable_ocmesh_room=True",
        ]

    if config.rooms is not None:
        # Tiled: pin the exact contour, single story, furnish all listed rooms.
        # The path must be absolute — the subprocess runs cwd=INFINIGEN_ROOT, so
        # a relative path would resolve against the vendored Infinigen tree. The
        # quotes make gin parse the RHS as a string; the arg travels through
        # subprocess argv (no shell) so they survive to sanitize_override.
        if floor_plan_path is None:
            raise ValueError("a tiled config (rooms set) requires floor_plan_path")
        overrides += [
            f'Solver.floor_plan="{Path(floor_plan_path).resolve()}"',
            "RoomConstants.n_stories=1",
            f"restrict_solving.solve_max_rooms={len(config.rooms)}",
        ]
    else:
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
# Ingest path — for scenes generated off-host and transferred in
# ---------------------------------------------------------------------------


def ingest_external_usd(
    *,
    source_dir: Path,
    output_dir: Path,
    config: SceneGenConfig | None = None,
) -> list[Path]:
    """Copy externally-generated scenes into the strafer_lab assets tree.

    The source directory is expected to contain one sub-directory per
    scene, each holding at least a ``*.usd`` file and optionally a
    ``*.blend`` file. Each scene is copied into ``output_dir`` and stamped
    with a ``scene_config.json`` for provenance (a generic organic config by
    default, since the true off-host generation parameters are unknown).
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config or SceneGenConfig(name="ingested")
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
        (dest / "scene_config.json").write_text(json.dumps(config.to_dict(), indent=2))
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
    # --- Layout dimension: rooms (count + types) ---
    layout = generate.add_mutually_exclusive_group()
    layout.add_argument(
        "--rooms",
        nargs="+",
        metavar="TYPE",
        choices=FURNISHABLE_ROOMS,
        default=None,
        help="Explicit room-type list -> an EXACT tiled layout; duplicates OK "
        "(e.g. --rooms bedroom bedroom living-room). Omit for an organic "
        "constraint-solved house. Choices: " + ", ".join(FURNISHABLE_ROOMS),
    )
    layout.add_argument(
        "--num-rooms",
        type=int,
        default=None,
        help="Round-robin this many rooms over --room-types (alt to --rooms).",
    )
    generate.add_argument(
        "--room-types",
        nargs="+",
        metavar="TYPE",
        choices=FURNISHABLE_ROOMS,
        default=None,
        help="Palette for --num-rooms round-robin (default: living-room).",
    )
    # --- Quality / resource dimension ---
    generate.add_argument(
        "--quality",
        choices=sorted(QUALITY_TIERS),
        default="high",
        help="Resource tier composing texture-res + object-density defaults.",
    )
    generate.add_argument(
        "--texture-res", type=int, default=None,
        help="Override the tier's texture resolution (VRAM).",
    )
    generate.add_argument(
        "--object-density", choices=sorted(_OBJECT_DENSITY_STEPS), default=None,
        help="Override object count (all rooms furnished; scales solver budget).",
    )
    generate.add_argument(
        "--geometry-detail", choices=_GEOMETRY_DETAILS, default=None,
        help="standard = base meshes (default); displacement = real_geometry "
        "micro-geometry (heavier).",
    )
    generate.add_argument(
        "--name", default=None,
        help="Scene name stem (default derived from rooms + quality).",
    )
    generate.add_argument(
        "--max-rooms", type=int, default=5,
        help="Organic mode only: solver room cap (restrict_solving.solve_max_"
        "rooms). Ignored when --rooms/--num-rooms pin an exact layout.",
    )
    generate.add_argument("--num-scenes", type=int, default=1)
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
        help="Import externally-generated scenes into the assets tree",
    )
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument(
        "--quality", choices=sorted(QUALITY_TIERS), default="high",
        help="Provenance quality tier stamped into scene_config.json.",
    )

    info = sub.add_parser("info", help="Print the available generation dimensions and exit")
    info.add_argument("--json", action="store_true")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "info":
        payload = {
            "quality_tiers": {k: asdict(v) for k, v in QUALITY_TIERS.items()},
            "furnishable_rooms": list(FURNISHABLE_ROOMS),
            "object_density": sorted(_OBJECT_DENSITY_STEPS),
            "geometry_detail": list(_GEOMETRY_DETAILS),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("Quality tiers (--quality):")
            for name, tier in QUALITY_TIERS.items():
                print(
                    f"  {name}: texture_res={tier.texture_resolution} "
                    f"object_density={tier.object_density}"
                )
            print("Furnishable rooms (--rooms / --room-types): " + ", ".join(FURNISHABLE_ROOMS))
            print("Object density (--object-density): " + ", ".join(sorted(_OBJECT_DENSITY_STEPS)))
            print("Geometry detail (--geometry-detail): " + ", ".join(_GEOMETRY_DETAILS))
        return 0

    if args.command == "generate":
        try:
            rooms = resolve_rooms(
                rooms=args.rooms, num_rooms=args.num_rooms, room_types=args.room_types
            )
        except ValueError as exc:
            parser.error(str(exc))
        config = build_scene_config(
            name=args.name,
            rooms=rooms,
            quality=args.quality,
            texture_resolution=args.texture_res,
            object_density=args.object_density,
            geometry_detail=args.geometry_detail,
            max_rooms=args.max_rooms,
            seed_base=args.seed_base,
        )
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
        config = build_scene_config(
            name="ingested",
            rooms=None,
            quality=args.quality,
            texture_resolution=None,
            object_density=None,
            geometry_detail=None,
            max_rooms=5,
            seed_base=0,
        )
        imported = ingest_external_usd(
            source_dir=args.source,
            output_dir=args.output,
            config=config,
        )
        logger.info("Imported %d scenes", len(imported))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
