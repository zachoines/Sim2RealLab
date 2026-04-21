"""Author ``scenes_metadata.json`` with interior floor spawn points.

``strafer_env_cfg._get_scenes_metadata`` expects a combined
``Assets/generated/scenes/scenes_metadata.json`` of shape::

    {
        "scenes": {
            "<scene_stem>": {
                "spawn_points_xy": [[x, y], ...],
                "floor_bbox_xy": [[xmin, ymin], [xmax, ymax]],
                "source_usdc": "<absolute path>"
            },
            ...
        }
    }

Nothing in the scene pipeline currently writes this. ``prep_room_usds.py``
produces the USDC; ``extract_scene_metadata.py`` emits per-scene
object records but no spawn points. This script fills the gap by
walking every top-level ``scene_*.usdc`` symlink in the scenes
directory, opening the USD stage, finding the Infinigen-generated
floor meshes, and sampling interior points from them.

Floor detection: prims whose name matches ``<room>_<i>_<j>_floor``.
The match is strict enough to exclude ``FloorLampFactory_*`` and
similar furniture whose names contain ``floor`` as a substring.

Sampling: each floor prim contributes points proportional to its XY
bounding-box area, with a configurable wall-margin shrink applied so
sampled points stay clear of walls.

Requires ``pxr``. Runs under the interpreter at
``STRAFER_ISAACLAB_PYTHON`` (populated by ``env_setup.sh`` from ``.env``).

Usage::

    python source/strafer_lab/scripts/generate_scenes_metadata.py \\
        --scenes-dir Assets/generated/scenes

    # Larger pool / looser margin:
    python source/strafer_lab/scripts/generate_scenes_metadata.py \\
        --scenes-dir Assets/generated/scenes \\
        --points-per-scene 200 \\
        --wall-margin 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("generate_scenes_metadata")


_FLOOR_NAME_RE = re.compile(r"^[a-z]+(?:_[a-z]+)*(?:_\d+)+_floor$")
_ROOM_STRUCT_RE = re.compile(
    r"^[a-z]+(?:_[a-z]+)*(?:_\d+)+_(floor|ceiling|wall|exterior|staircase)$"
)


def _find_floor_bboxes(
    stage: Any,
) -> list[tuple[str, tuple[float, float, float, float, float, float]]]:
    """Return ``(prim_path, (xmin, ymin, zmin, xmax, ymax, zmax))`` per floor mesh.

    Z range is included so callers can place the robot on the actual
    floor surface instead of assuming the floor sits at world Z=0.
    """
    from pxr import Usd, UsdGeom  # type: ignore

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )

    found: list[tuple[str, tuple[float, float, float, float, float, float]]] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        name = prim.GetName()
        if not _FLOOR_NAME_RE.match(name):
            continue
        # Parent-with-same-name wrap (seen in Infinigen USD exports: each
        # Xform often has a Mesh child of the same name). Dedupe by keeping
        # only the outer Xform.
        parent = prim.GetParent()
        if parent is not None and parent.IsValid() and parent.GetName() == name:
            continue
        try:
            bbox = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bbox.IsEmpty():
                continue
            found.append(
                (
                    str(prim.GetPath()),
                    (
                        float(bbox.GetMin()[0]),
                        float(bbox.GetMin()[1]),
                        float(bbox.GetMin()[2]),
                        float(bbox.GetMax()[0]),
                        float(bbox.GetMax()[1]),
                        float(bbox.GetMax()[2]),
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping %s: bbox computation failed: %s", prim.GetPath(), exc)
    return found


def _find_obstacle_bboxes(
    stage: Any, *, min_height: float
) -> list[tuple[float, float, float, float]]:
    """Return XY bboxes for prims that would obstruct a robot spawn.

    Includes any ``Xform``/``Mesh`` prim directly under ``/World`` whose
    vertical extent is at least ``min_height`` meters and whose name
    does NOT match a room-structure prim (floor/ceiling/wall/...).
    Factory furniture, loose props, and doors all fall into this bucket.
    """
    from pxr import Usd, UsdGeom  # type: ignore

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )

    obstacles: list[tuple[float, float, float, float]] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        name = prim.GetName()
        if _ROOM_STRUCT_RE.match(name):
            continue
        parent = prim.GetParent()
        if parent is None or not parent.IsValid():
            continue
        # Only consider direct children of /World to avoid per-vertex
        # Mesh prims and double-counting. The factory/room scene graph
        # Infinigen emits has the canonical Xform at that depth.
        if str(parent.GetPath()) != "/World":
            continue
        type_name = str(prim.GetTypeName())
        if type_name not in {"Xform", "Mesh"}:
            continue
        try:
            bbox = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bbox.IsEmpty():
                continue
            z_extent = float(bbox.GetMax()[2]) - float(bbox.GetMin()[2])
            if z_extent < min_height:
                continue
            obstacles.append(
                (
                    float(bbox.GetMin()[0]),
                    float(bbox.GetMin()[1]),
                    float(bbox.GetMax()[0]),
                    float(bbox.GetMax()[1]),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping obstacle %s: bbox failed: %s", prim.GetPath(), exc)
    return obstacles


def _point_blocked(
    x: float,
    y: float,
    obstacles: list[tuple[float, float, float, float]],
    robot_radius: float,
) -> bool:
    """True if ``(x, y)`` falls inside any obstacle bbox expanded by the robot radius."""
    for xmin, ymin, xmax, ymax in obstacles:
        if (xmin - robot_radius) <= x <= (xmax + robot_radius) and (
            ymin - robot_radius
        ) <= y <= (ymax + robot_radius):
            return True
    return False


def _sample_floor_points(
    floors: list[tuple[str, tuple[float, float, float, float, float, float]]],
    obstacles: list[tuple[float, float, float, float]],
    *,
    total_points: int,
    wall_margin: float,
    robot_radius: float,
    rng: random.Random,
    oversample_factor: int = 20,
) -> list[list[float]]:
    """Draw area-weighted uniform samples from the union of floor bboxes.

    Rejects samples that fall inside any ``obstacles`` bbox (expanded by
    ``robot_radius``). Oversamples by ``oversample_factor`` to account
    for rejection so a cluttered room still yields a usable pool.
    """
    shrunken: list[tuple[float, float, float, float, float]] = []
    total_area = 0.0
    for _, (xmin, ymin, _zmin, xmax, ymax, _zmax) in floors:
        xmin += wall_margin
        ymin += wall_margin
        xmax -= wall_margin
        ymax -= wall_margin
        if xmax <= xmin or ymax <= ymin:
            continue
        area = (xmax - xmin) * (ymax - ymin)
        shrunken.append((xmin, ymin, xmax, ymax, area))
        total_area += area

    if not shrunken or total_area <= 0.0:
        return []

    points: list[list[float]] = []
    for xmin, ymin, xmax, ymax, area in shrunken:
        share = area / total_area
        target = max(1, int(round(total_points * share)))
        attempts = target * oversample_factor
        accepted = 0
        for _ in range(attempts):
            if accepted >= target:
                break
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if _point_blocked(x, y, obstacles, robot_radius):
                continue
            points.append([x, y])
            accepted += 1
        if accepted < target:
            logger.warning(
                "Floor bbox (%.2f, %.2f)..(%.2f, %.2f): only %d/%d clear points after %d attempts",
                xmin, ymin, xmax, ymax, accepted, target, attempts,
            )
    return points


def _process_scene(
    usdc_path: Path,
    *,
    points_per_scene: int,
    wall_margin: float,
    robot_radius: float,
    obstacle_min_height: float,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Open one scene's USDC and return its ``scenes_metadata`` entry."""
    from pxr import Usd  # type: ignore

    resolved = usdc_path.resolve()
    stage = Usd.Stage.Open(str(resolved))
    if stage is None:
        logger.error("Failed to open %s", resolved)
        return None

    floors = _find_floor_bboxes(stage)
    if not floors:
        logger.warning("No Infinigen-style floor prims found in %s", resolved)
        return None

    obstacles = _find_obstacle_bboxes(stage, min_height=obstacle_min_height)

    xs_min = min(b[1][0] for b in floors)
    ys_min = min(b[1][1] for b in floors)
    xs_max = max(b[1][3] for b in floors)
    ys_max = max(b[1][4] for b in floors)
    # Floor top surface height — assume a single-story scene where every
    # floor mesh's top face sits at (approximately) the same Z. Use the
    # max so multi-story scenes bias toward the upper floor rather than
    # sinking the robot into a lower floor.
    floor_top_z = max(b[1][5] for b in floors)

    spawn_points = _sample_floor_points(
        floors,
        obstacles,
        total_points=points_per_scene,
        wall_margin=wall_margin,
        robot_radius=robot_radius,
        rng=rng,
    )

    logger.info(
        "%s: %d floor(s), %d obstacle(s), union bbox (%.2f, %.2f)..(%.2f, %.2f), "
        "floor_top_z=%.3f, %d clear spawn points",
        usdc_path.name, len(floors), len(obstacles), xs_min, ys_min, xs_max, ys_max,
        floor_top_z, len(spawn_points),
    )

    return {
        "spawn_points_xy": spawn_points,
        "floor_bbox_xy": [[xs_min, ys_min], [xs_max, ys_max]],
        "floor_top_z": floor_top_z,
        "source_usdc": str(resolved),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path("Assets/generated/scenes"),
        help="Directory containing the top-level scene_*.usdc symlinks.",
    )
    parser.add_argument(
        "--points-per-scene",
        type=int,
        default=100,
        help="Total spawn points to sample per scene (split across floors).",
    )
    parser.add_argument(
        "--wall-margin",
        type=float,
        default=0.3,
        help="Shrink each floor bbox by this many meters on every side "
             "before sampling so spawns stay clear of walls.",
    )
    parser.add_argument(
        "--robot-radius",
        type=float,
        default=0.35,
        help="Minimum XY clearance between a spawn point and any obstacle "
             "bbox. The Strafer's base is ~0.3 m across; the default leaves "
             "a small cushion so the reset doesn't drop the robot into a "
             "wall or a piece of furniture.",
    )
    parser.add_argument(
        "--obstacle-min-height",
        type=float,
        default=0.3,
        help="Mesh vertical extent below which a prim is treated as "
             "floor-level decoration (rug, skirting board, door threshold) "
             "and skipped during obstacle detection. Default 0.3 m keeps "
             "tables/chairs/beds as obstacles but filters Infinigen's "
             "skirtingboard_support, whose XY bbox otherwise spans the "
             "entire house perimeter and blocks every sample.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    scenes_dir = args.scenes_dir.resolve()
    if not scenes_dir.is_dir():
        logger.error("Scenes directory not found: %s", scenes_dir)
        return 2

    rng = random.Random(args.seed)
    scene_entries: dict[str, dict[str, Any]] = {}
    for usdc in sorted(scenes_dir.glob("scene_*.usdc")):
        if not usdc.is_file() and not usdc.is_symlink():
            continue
        entry = _process_scene(
            usdc,
            points_per_scene=args.points_per_scene,
            wall_margin=args.wall_margin,
            robot_radius=args.robot_radius,
            obstacle_min_height=args.obstacle_min_height,
            rng=rng,
        )
        if entry is not None:
            scene_entries[usdc.stem] = entry

    if not scene_entries:
        logger.error("No scenes produced spawn points. Nothing written.")
        return 1

    output = scenes_dir / "scenes_metadata.json"
    output.write_text(json.dumps({"scenes": scene_entries}, indent=2))
    logger.info("Wrote %s with %d scene entries", output, len(scene_entries))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
