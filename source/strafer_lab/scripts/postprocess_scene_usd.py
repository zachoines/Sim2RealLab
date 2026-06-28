"""Bake runtime scene modifications into an Infinigen-exported USDC.

Infinigen's USD export is visual-only and lacks the things Isaac Sim
needs for robot integration:

1. Collision shapes on most meshes. Without them the robot falls
   through walls and phases through furniture.
2. Supplemental light emitters at ``CeilingLightFactory_*`` fixtures.
   Infinigen's export already carries its own physically-based emitters
   (``PointLampFactory_*`` ``UsdLux.SphereLight``s at ``normalize=True``,
   area ``RectLight``s, an environment ``DomeLight``) which dominate
   interior radiance; this pass adds a small low-power SphereLight at each
   ceiling fixture as a fill. It is NOT the interior-exposure control —
   over/under-exposure is corrected at the render layer (RTX auto-exposure
   via ``RenderCfg.carb_settings`` in the env config), not here.

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

It also strips Infinigen's stage-shot authoring cameras
(``camera_<room>_<index>`` under ``/World/Room``) as a cleanliness pass.
Those cameras are runtime-inert — Isaac Sim binds no render product to
them — so this is not a perf change; it just keeps ``usdview`` /
Omniverse Composer uncluttered when an operator opens the scene.

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


# Matched against the prim *name* (leaf), not the full path — unlike the
# floor / structural patterns below, which match the full prim path. A foreign
# source whose light fixtures use a different naming scheme overrides this via
# ``--ceiling-light-prim-pattern``.
_DEFAULT_CEILING_LIGHT_PRIM_PATTERN = r"^CeilingLightFactory_\d+__spawn_asset_\d+_$"

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

# Matches the room shell prims (walls, ceilings, exterior, attic), every
# door-factory prim, and perimeter trim moulding. These all need a collider that
# preserves the actual mesh: a convex approximation heals door / window cutouts
# shut and a simplified one pokes into the doorway, trapping the robot; for thin
# perimeter trim (skirting boards / baseboards), the convex hull of the trim ring
# is the *filled* room rectangle — a phantom floor-height slab the occupancy map
# then reads as wall-to-wall occupied. Each arm is a name allow-list, which has
# leaked before: the earlier ``(?:PanelDoor|LiteDoor|LouverDoor)`` door arm
# silently missed GlassPanel + SPLIT-glass variants, and trim named ``*_support``
# fell through where trim named ``*_ceiling`` matched the ceiling arm by
# coincidence — both then took the convex furniture default. The door arm matches
# ANY ``...DoorFactory`` and any per-asset suffix (frame ``_``, leaf ``__001``,
# ``__SPLIT_GLASS``). The trim arm is case-insensitive (scoped ``(?i:...)``) so
# ``*_SUPPORT`` / ``*_CEILING`` cased variants are caught; the wall / door arms
# stay case-sensitive, so a furniture prim carrying an uppercase ``_WALL`` /
# ``_CEILING`` substring is not pulled into the exact-mesh bucket.
_DEFAULT_STRUCTURAL_PRIM_PATTERN = (
    r"^/World/[^/]+_(?:wall|ceiling|roof|attic|exterior)"
    r"(?:_\d+)?(?:/.+)?$"
    r"|^/World/[A-Za-z]*Door[A-Za-z]*Factory_\d+__spawn_asset_\d+.*$"
    r"|^/World/.*(?i:skirting|baseboard|moulding|molding|cornice|casing|trim)[^/]*(?:/.+)?$"
)

# Infinigen's generate_indoors pipeline authors one stage-shot camera per
# room (named ``camera_<room>_<index>``) for its own offline Cycles render
# passes. Isaac Sim binds no render product to them, so they evaluate no
# per-tick RTX work — but they clutter ``usdview`` / Omniverse Composer when
# an operator opens the scene to debug, and add tiny per-prim USD-traversal
# overhead. They serve no role in the strafer runtime, so the scene-prep bake
# strips them. Matches the full prim path (like the floor / structural
# patterns above): the cameras sit directly under ``/World/Room``, and
# Infinigen's Xform-with-same-name-child export nests an inner twin
# (``/World/Room/camera_0_0/camera_0_0``), so the optional second group lets
# the matcher catch either the outer group or the inner twin.
_DEFAULT_STAGE_CAMERA_PRIM_PATTERN = (
    r"^/World/Room/camera_\d+_\d+(?:/camera_\d+_\d+)?$"
)

_VALID_APPROXIMATIONS: tuple[str, ...] = (
    "boundingCube",
    "boundingSphere",
    "convexHull",
    "convexDecomposition",
    "meshSimplification",
    "none",
)
_DEFAULT_APPROXIMATION = "convexHull"
# Structural prims (walls + doors) use the exact mesh as a triangle-mesh collider
# (``none`` = no approximation). They are static, so the per-triangle cost is
# acceptable, and only the exact mesh keeps door/window cutouts true to the
# geometry: ``meshSimplification`` (the previous default) leaves the collider
# poking into doorways, which traps the robot and seals doorways in the occupancy
# map. Furniture stays ``convexHull`` (cheap + conservatively over-approximated).
_DEFAULT_STRUCTURAL_APPROXIMATION = "none"


def _compile_floor_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _compile_ceiling_light_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def attach_mesh_colliders(
    stage: Any,
    floor_pattern: re.Pattern[str] | None = None,
    *,
    approximation: str = _DEFAULT_APPROXIMATION,
    structural_pattern: re.Pattern[str] | None = None,
    structural_approximation: str = _DEFAULT_STRUCTURAL_APPROXIMATION,
) -> tuple[int, int]:
    """Attach ``CollisionAPI`` + ``MeshCollisionAPI`` to every scene ``Mesh``.

    Prims matching ``structural_pattern`` get ``structural_approximation``;
    everything else gets ``approximation``. Structural prims default to
    ``none`` (the exact mesh as a triangle-mesh collider) because they are
    static and only the exact mesh keeps door / window cutouts true — convex
    shapes heal the cutout shut and ``meshSimplification`` pokes into the
    doorway.

    Skips material subtrees and floor meshes matching ``floor_pattern``
    (robot collision goes to ``/World/ground`` instead). Pass
    ``floor_pattern=None`` to disable the floor skip.

    Idempotent + approximation-correcting: re-running on an already-
    postprocessed USDC rewrites the approximation attribute to match
    the (per-prim) target.

    Returns ``(furniture_changed, structural_changed)`` — counts of
    prims whose authored approximation was added or migrated.
    """
    for name, approx in (("approximation", approximation),
                         ("structural_approximation", structural_approximation)):
        if approx not in _VALID_APPROXIMATIONS:
            raise ValueError(
                f"unknown {name}={approx!r}; valid: {_VALID_APPROXIMATIONS}",
            )
    from pxr import UsdPhysics  # type: ignore

    furniture_changed = 0
    structural_changed = 0
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
        is_structural = (
            structural_pattern is not None and structural_pattern.match(path)
        )
        target = structural_approximation if is_structural else approximation
        # Ensure the APIs are present; both Apply() calls are no-ops
        # if already authored.
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        # Migrate any prior approximation to the desired one. USD's
        # schema default for an unauthored attribute is ``"none"``,
        # so we compare only against the *authored* value — otherwise
        # a fresh prim with approximation=none would be skipped as a
        # "no-op" and never get the attribute written.
        approx_attr = mesh_api.GetApproximationAttr()
        previous = (
            approx_attr.Get()
            if approx_attr.IsValid() and approx_attr.HasAuthoredValue()
            else None
        )
        if previous != target:
            mesh_api.CreateApproximationAttr().Set(target)
            if is_structural:
                structural_changed += 1
            else:
                furniture_changed += 1
    return furniture_changed, structural_changed


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


def inject_ceiling_light_emitters(
    stage: Any,
    intensity: float,
    ceiling_light_pattern: re.Pattern[str] | None = None,
) -> int:
    """Create a ``UsdLux.SphereLight`` at every matching light-fixture prim.

    Matches each prim's *name* (leaf, not full path) against
    ``ceiling_light_pattern``; ``None`` falls back to
    :data:`_DEFAULT_CEILING_LIGHT_PRIM_PATTERN` (Infinigen's
    ``CeilingLightFactory_*`` naming).

    A small supplemental fill at the ceiling fixtures. Infinigen's export
    already carries the dominant interior emitters (``PointLampFactory_*``
    SphereLights at ``normalize=True``, area RectLights, an environment
    DomeLight), so this is not the interior-exposure control — see the
    module docstring.

    Skips ``*_SPLIT_GLA`` glass-shade variants (same location as the
    parent fixture, so a second light would just double up). Re-uses an
    existing ``AutoSphereLight`` child when present instead of creating
    a duplicate.

    A foreign-source adapter whose fixtures need different emitters
    (e.g. ``UsdLux.DiskLight`` at floor-up positions) can import this
    function and call it with a custom pattern, or bypass it entirely
    and author its own lights — the rest of :func:`postprocess_usdc`
    does not depend on it.
    """
    from pxr import UsdLux  # type: ignore

    pattern = ceiling_light_pattern or _compile_ceiling_light_pattern(
        _DEFAULT_CEILING_LIGHT_PRIM_PATTERN,
    )

    count = 0
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if not pattern.match(prim.GetName()):
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


def strip_stage_cameras(
    stage: Any,
    stage_camera_pattern: re.Pattern[str],
) -> int:
    """Remove Infinigen stage-shot authoring cameras from the stage.

    Infinigen authors one ``camera_<room>_<index>`` per room for its own
    offline Cycles render passes (see ``_DEFAULT_STAGE_CAMERA_PRIM_PATTERN``).
    Isaac Sim never binds a render product to them, so removing them is a
    *cleanliness* pass — it declutters ``usdview`` / Composer and trims a
    handful of prims from USD traversal, but it is **not** a perf lever and
    yields no measurable FPS change (the operator confirmed disabling these
    cameras in the editor produced no perf delta).

    Walks every prim whose full path matches ``stage_camera_pattern`` and
    removes it. Outermost-first removal means deleting an outer camera group
    also drops its inner same-named twin; the per-prim validity guard keeps
    the pass idempotent (a re-run finds nothing left to strip).

    Returns the number of matching prims found (all of which are removed).
    """
    matched = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsValid() and stage_camera_pattern.match(str(prim.GetPath()))
    ]
    # Shortest path first = outermost group first; removing it drops the inner
    # twin, which the validity guard then skips.
    for path in sorted(matched, key=len):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)
    return len(matched)


def postprocess_usdc(
    usdc_path: Path,
    *,
    light_intensity: float,
    floor_pattern: re.Pattern[str],
    keep_floor_colliders: bool,
    collider_approximation: str = _DEFAULT_APPROXIMATION,
    structural_pattern: re.Pattern[str] | None = None,
    structural_approximation: str = _DEFAULT_STRUCTURAL_APPROXIMATION,
    ceiling_light_pattern: re.Pattern[str] | None = None,
    stage_camera_pattern: re.Pattern[str] | None = None,
    keep_stage_cameras: bool = False,
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
    furniture, structural = attach_mesh_colliders(
        stage,
        floor_pattern=floor_skip,
        approximation=collider_approximation,
        structural_pattern=structural_pattern,
        structural_approximation=structural_approximation,
    )
    lights = inject_ceiling_light_emitters(
        stage, light_intensity, ceiling_light_pattern,
    )
    stage_cameras = (
        0
        if keep_stage_cameras or stage_camera_pattern is None
        else strip_stage_cameras(stage, stage_camera_pattern)
    )
    stage.Save()
    logger.info(
        "%s: stripped %d floor collider(s), attached %d %s + %d %s "
        "collider(s), injected %d light(s), stripped %d stage camera(s)",
        resolved, stripped,
        furniture, collider_approximation,
        structural, structural_approximation,
        lights, stage_cameras,
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
        help="Intensity for each injected supplemental SphereLight. This is "
             "fill at the ceiling fixtures, NOT the interior-exposure control: "
             "the baked Infinigen emitters dominate (these injected lights are a "
             "negligible fraction of total interior power), so changing it does "
             "not fix an over- or under-exposed scene. Exposure is corrected at "
             "the render layer (RTX auto-exposure in the env config).",
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
    parser.add_argument(
        "--collider-approximation",
        choices=_VALID_APPROXIMATIONS,
        default=_DEFAULT_APPROXIMATION,
        help="PhysX collider representation for furniture / freestanding "
             "meshes.",
    )
    parser.add_argument(
        "--structural-prim-pattern",
        type=str,
        default=_DEFAULT_STRUCTURAL_PRIM_PATTERN,
        help="Regex matched against full USD prim paths to identify "
             "structural prims (walls, ceilings, doors, etc.). "
             "Matching prims get --structural-approximation. Pass an "
             "empty string to disable the structural dispatch.",
    )
    parser.add_argument(
        "--structural-approximation",
        choices=_VALID_APPROXIMATIONS,
        default=_DEFAULT_STRUCTURAL_APPROXIMATION,
        help="PhysX collider representation for structural prims.",
    )
    parser.add_argument(
        "--ceiling-light-prim-pattern",
        type=str,
        default=_DEFAULT_CEILING_LIGHT_PRIM_PATTERN,
        help="Regex matched against each prim's NAME (the leaf, not the full "
             "path) to identify light-fixture prims that need a SphereLight "
             "emitter authored under them. Override if your source names "
             "light fixtures differently than Infinigen.",
    )
    parser.add_argument(
        "--stage-camera-prim-pattern",
        type=str,
        default=_DEFAULT_STAGE_CAMERA_PRIM_PATTERN,
        help="Regex matched against full USD prim paths to identify Infinigen "
             "stage-shot authoring cameras (one camera_<room>_<index> per "
             "room, authored for Infinigen's own offline Cycles passes) and "
             "remove them. CLEANLINESS ONLY, NOT A PERF LEVER: Isaac Sim binds "
             "no render product to these cameras, so stripping them declutters "
             "usdview / Composer but yields no measurable FPS change. Pass an "
             "empty string to disable, or override if your source names stage "
             "cameras differently than Infinigen.",
    )
    parser.add_argument(
        "--keep-stage-cameras",
        action="store_true",
        help="Keep the Infinigen stage-shot authoring cameras (skip the "
             "stage-camera strip). Inspection / debugging only; the cameras "
             "are runtime-inert either way.",
    )
    args = parser.parse_args(argv)

    if not args.usdc.exists():
        logger.error("USDC not found: %s", args.usdc)
        return 2

    floor_pattern = _compile_floor_pattern(args.floor_prim_pattern)
    structural_pattern = (
        re.compile(args.structural_prim_pattern)
        if args.structural_prim_pattern else None
    )
    ceiling_light_pattern = _compile_ceiling_light_pattern(
        args.ceiling_light_prim_pattern,
    )
    stage_camera_pattern = (
        re.compile(args.stage_camera_prim_pattern)
        if args.stage_camera_prim_pattern else None
    )
    postprocess_usdc(
        args.usdc,
        light_intensity=args.light_intensity,
        floor_pattern=floor_pattern,
        keep_floor_colliders=args.keep_floor_colliders,
        collider_approximation=args.collider_approximation,
        structural_pattern=structural_pattern,
        structural_approximation=args.structural_approximation,
        ceiling_light_pattern=ceiling_light_pattern,
        stage_camera_pattern=stage_camera_pattern,
        keep_stage_cameras=args.keep_stage_cameras,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
