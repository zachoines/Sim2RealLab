"""Bake runtime scene modifications into an Infinigen-exported USDC.

Infinigen's USD export is visual-only and lacks the things Isaac Sim
needs for robot integration:

1. Collision shapes on most meshes. Without them the robot falls
   through walls and phases through furniture.
2. Light emitters at ``CeilingLightFactory_*`` fixtures. The fixture
   meshes are exported but no ``UsdLux`` emitters are authored, so the
   interiors render black.

Floor meshes are deliberately *excluded* from the collision pass — their
tessellated triangle edges catch the Strafer mecanum rollers and pull
the robot into the geometry. Robot collision is delegated to the clean
``/World/ground`` plane lifted to floor height by the env config.

Applying these at Kit startup works but traversing a ~7 GB USDC to
attach ``UsdPhysics.CollisionAPI`` on hundreds of meshes freezes the
env for ~60 s on every launch. This script bakes the same changes into
the USDC once at scene-generation time so subsequent launches are
fast.

Idempotent — re-running on an already-postprocessed USDC strips any
floor colliders the previous bake left behind and leaves all other
state alone (skips meshes that already carry ``CollisionAPI``, leaves
existing ``AutoSphereLight`` children alone).

Requires ``pxr``. Runs under the interpreter at
``STRAFER_ISAACLAB_PYTHON`` — the same one Isaac Sim ships with.
Invoked automatically by ``prep_room_usds.py`` after
``infinigen.tools.export``; can also be run manually on an existing
scene::

    python source/strafer_lab/scripts/postprocess_scene_usd.py \\
        --usdc Assets/generated/scenes/<scene>/export/export_scene.blend/export_scene.usdc
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("postprocess_scene_usd")


_CEILING_LIGHT_NAME_RE = re.compile(r"^CeilingLightFactory_\d+__spawn_asset_\d+_$")

# Floor meshes Infinigen exports are tessellated and have triangle edges /
# T-junctions that catch the Strafer mecanum rollers. Infinigen emits an
# Xform-with-same-name-Mesh-child pattern at file root depth 2/3, e.g.
# ``/World/bedroom_0_0_floor`` (Xform) and
# ``/World/bedroom_0_0_floor/bedroom_0_0_floor`` (Mesh). Note the path is
# rooted at ``/World/`` here, not ``/World/Room/`` — the env config later
# references this USDC under ``/World/Room`` at runtime, but during the
# offline postprocess pass we operate on the standalone file. Both the
# outer Xform and the inner Mesh leaf match so we can strip / skip either.
_DEFAULT_FLOOR_PRIM_PATTERN = r"^/World/[^/]+_floor(?:/[^/]+_floor)?$"


def _compile_floor_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def attach_mesh_colliders(
    stage: Any, floor_pattern: re.Pattern[str] | None = None
) -> int:
    """Attach ``CollisionAPI`` + ``MeshCollisionAPI`` to every scene ``Mesh``.

    Applies the ``none`` approximation (use the raw triangle mesh as the
    collider). This is the accurate choice for static world geometry;
    dynamic bodies would need ``convexHull`` / ``convexDecomposition``
    but nothing in an Infinigen indoor scene moves on its own.

    Skips:

    * Material subtrees.
    * Prims that already carry ``CollisionAPI`` so repeated invocations
      are a no-op.
    * Floor meshes matching ``floor_pattern`` — robot collision goes to
      the clean ``/World/ground`` plane the env config lifts to floor
      height. Pass ``None`` to disable the skip (debugging only).
    """
    from pxr import UsdPhysics  # type: ignore

    count = 0
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if prim.GetTypeName() != "Mesh":
            continue
        path = str(prim.GetPath())
        if "/_materials/" in path:
            continue
        if floor_pattern is not None and floor_pattern.match(path):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set("none")
        count += 1
    return count


def strip_floor_colliders(stage: Any, floor_pattern: re.Pattern[str]) -> int:
    """Remove ``CollisionAPI`` / ``MeshCollisionAPI`` from floor mesh prims.

    Walks every prim whose path matches ``floor_pattern``. If the prim
    carries collision authoring it is removed via ``RemoveAPI`` so
    PhysX no longer sees the floor as a collider. Idempotent — prims
    without authored collision are skipped.
    """
    from pxr import UsdPhysics  # type: ignore

    count = 0
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if not floor_pattern.match(str(prim.GetPath())):
            continue
        had_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        had_mesh = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        if not (had_collision or had_mesh):
            continue
        if had_mesh:
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if had_collision:
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
        count += 1
    return count


def inject_ceiling_light_emitters(stage: Any, intensity: float) -> int:
    """Create a ``UsdLux.SphereLight`` at every ``CeilingLightFactory_*`` prim.

    Infinigen exports the fixture mesh but not the emitter. Without
    this, dome / directional / area lights can't reach the interior and
    rooms render black.

    Skips ``*_SPLIT_GLA`` glass-shade variants (same location as the
    parent fixture, so a second light would just double up). Re-uses an
    existing ``AutoSphereLight`` child when present instead of creating
    a duplicate.
    """
    from pxr import UsdLux  # type: ignore

    count = 0
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if not _CEILING_LIGHT_NAME_RE.match(prim.GetName()):
            continue
        parent = prim.GetParent()
        if parent is None or not parent.IsValid():
            continue
        if parent.GetName() == prim.GetName():
            # inner Mesh twin — author on the outer Xform
            continue
        light_path = prim.GetPath().AppendChild("AutoSphereLight")
        light = UsdLux.SphereLight.Define(stage, light_path)
        light.CreateRadiusAttr(0.1)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr((1.0, 0.96, 0.88))  # warm white
        count += 1
    return count


def postprocess_usdc(
    usdc_path: Path,
    *,
    light_intensity: float,
    floor_pattern: re.Pattern[str],
    keep_floor_colliders: bool,
) -> None:
    from pxr import Usd  # type: ignore

    resolved = usdc_path.resolve()
    stage = Usd.Stage.Open(str(resolved))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {resolved}")

    floor_skip = None if keep_floor_colliders else floor_pattern
    stripped = (
        0 if keep_floor_colliders else strip_floor_colliders(stage, floor_pattern)
    )
    colliders = attach_mesh_colliders(stage, floor_pattern=floor_skip)
    lights = inject_ceiling_light_emitters(stage, light_intensity)
    stage.Save()
    logger.info(
        "%s: stripped %d floor collider(s), attached %d collider(s), injected %d light(s)",
        resolved, stripped, colliders, lights,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usdc",
        type=Path,
        required=True,
        help="USDC file to modify in place (typically "
             "Assets/generated/scenes/<scene>/export/export_scene.blend/export_scene.usdc).",
    )
    parser.add_argument(
        "--light-intensity",
        type=float,
        default=100000.0,
        help="Intensity for each injected SphereLight. Tune higher if "
             "rooms are still dim, lower if the scene blows out.",
    )
    parser.add_argument(
        "--floor-prim-pattern",
        type=str,
        default=_DEFAULT_FLOOR_PRIM_PATTERN,
        help="Regex matched against full USD prim paths to identify Infinigen "
             "floor meshes whose colliders must be stripped. Override only if "
             "you discover Infinigen export variants the default misses.",
    )
    parser.add_argument(
        "--keep-floor-colliders",
        action="store_true",
        help="Keep floor mesh colliders (debugging only). Reintroduces "
             "the wheel-catching behavior the floor strip was added to fix.",
    )
    args = parser.parse_args(argv)

    if not args.usdc.exists():
        logger.error("USDC not found: %s", args.usdc)
        return 2

    floor_pattern = _compile_floor_pattern(args.floor_prim_pattern)
    postprocess_usdc(
        args.usdc,
        light_intensity=args.light_intensity,
        floor_pattern=floor_pattern,
        keep_floor_colliders=args.keep_floor_colliders,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
