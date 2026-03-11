#!/usr/bin/env python3
"""One-time preprocessing script to prepare Infinigen room USDs for Isaac Lab training.

This script processes each room_*/export_scene.usdc and exports to
Assets/generated/scenes/scene_NNNN.usdc (ready for ProcDepth env configs):
1. (Optional) Strip non-essential textures from source directories
2. Decimate high-poly meshes via Omniverse Scene Optimizer (95% reduction)
3. Apply tiered collision (structural→convexDecomposition, ground→convexHull, etc.)
4. Hide ceiling/top_wall prims via USD visibility
5. Center the floor bounding box at world origin (0, 0, 0)
6. Sample interior spawn points from floor meshes

USD hierarchy in exported files:
    /Environment (Xform) -- defaultPrim, empty; spawner sets translate here
        /Geometry (Xform) -- room reference + centering xform
            /...room prims...

The centering xform is on /Environment/Geometry so it survives when Isaac Lab's
MultiUsdFileCfg sets translate(0,0,0) on the defaultPrim (/Environment).

Must be run with Isaac Sim Python BEFORE training:
    cd C:\Worspace\IsaacLab && .\isaaclab.bat -p ..\Scripts\scenegen\prep_room_usds.py

After running, scenes are ready for _get_scene_usd_paths() / MultiUsdFileCfg.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

# Parse args before Isaac Sim launch
parser = argparse.ArgumentParser(description="Prepare Infinigen room USDs for training")
parser.add_argument(
    "--rooms_dir",
    type=str,
    default=str(Path(__file__).resolve().parents[2] / "Assets" / "generated" / "infinigen_rooms"),
    help="Directory containing room_*/export_scene.usdc",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=str(Path(__file__).resolve().parents[2] / "Assets" / "generated" / "scenes"),
    help="Output directory for processed scene_NNNN.usdc files",
)
parser.add_argument("--dry_run", action="store_true", help="Print what would be done without saving")
parser.add_argument("--no_decimate", action="store_true", help="Skip mesh decimation")
parser.add_argument(
    "--decimate_ratio", type=float, default=0.05,
    help="Fraction of faces to keep per mesh (default: 0.05 = keep 5%%)",
)
parser.add_argument(
    "--strip_textures", action="store_true",
    help="Delete non-DIFFUSE textures from source room dirs (destructive, opt-in)",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import USD/Omni modules (requires running SimulationApp)
import omni.kit.app
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics


# =============================================================================
# Constants
# =============================================================================

SPAWN_POINTS_PER_SCENE = 500

# Collision tier thresholds
ELEVATED_Z_OFFSET = 0.50  # objects with bottom Z above floor_z + this get no collision
SMALL_OBJECT_VOLUME = 0.01  # m^3; below this gets boundingSphere

# Prim name patterns for tier classification
STRUCTURAL_PATTERNS = ("floor", "wall")
NO_COLLISION_PATTERNS = (
    "ceiling", "top_wall", "light", "mirror", "picture", "art", "window", "place_cam",
)


# =============================================================================
# Mesh Decimation
# =============================================================================


def _count_mesh_faces(stage, root_path):
    """Count total faces across all meshes under root_path."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    total = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            fc = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
            if fc:
                total += len(fc)
    return total


def _enable_scene_optimizer():
    """Ensure the omni.scene.optimizer.core extension is loaded."""
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("omni.scene.optimizer.core"):
        manager.set_extension_enabled_immediate("omni.scene.optimizer.core", True)
        omni.kit.app.get_app().update()


def decimate_meshes(stage, root_path="/Environment", ratio=0.05,
                    skip_patterns=("floor", "wall")):
    """Decimate non-structural meshes using Omniverse Scene Optimizer.

    Uses omni.scene.optimizer.core's GPU-accelerated decimateMeshes operation
    which preserves mesh topology (no gaps/cracks). Structural meshes (floors,
    walls) are skipped to maintain environment integrity and spawn sampling.

    Args:
        stage: The USD stage.
        root_path: Root prim path to traverse.
        ratio: Fraction of faces to keep (e.g., 0.05 = keep 5%).
        skip_patterns: Prim name substrings to skip from decimation.

    Returns:
        (total_original_faces, total_decimated_faces, meshes_targeted)
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return 0, 0, 0

    # Count original faces and collect paths to decimate
    total_orig = 0
    decimate_paths = []

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        fc = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
        if not fc:
            continue
        total_orig += len(fc)

        name_lower = prim.GetName().lower()
        if any(pat in name_lower for pat in skip_patterns):
            continue
        decimate_paths.append(str(prim.GetPath()))

    if not decimate_paths:
        return total_orig, total_orig, 0

    # Scene Optimizer's reductionFactor is % of faces to REMOVE
    reduction_pct = (1.0 - ratio) * 100.0

    try:
        _enable_scene_optimizer()
        import omni.kit.commands

        omni.kit.commands.execute(
            "SceneOptimizerOperation",
            operation="decimateMeshes",
            args={
                "paths": decimate_paths,
                "reductionFactor": reduction_pct,
            },
        )
        omni.kit.app.get_app().update()
    except Exception as e:
        print(f"  [ERROR] Scene Optimizer decimation failed: {e}")
        return total_orig, total_orig, 0

    # Recount faces after decimation
    total_new = _count_mesh_faces(stage, root_path)

    return total_orig, total_new, len(decimate_paths)


# =============================================================================
# Tiered Collision Strategy
# =============================================================================


def classify_and_set_collision(stage, root_path="/Environment", floor_z=0.0):
    """Classify meshes into tiers and assign appropriate collision approximation.

    Tiers:
      - structural (floor/wall): triangle mesh (approximation="none") for exact geometry
      - ground_obstacle (large, near floor): convexHull
      - small_ground (tiny, near floor): boundingSphere
      - elevated (above robot reach / ceiling / decorative): no collision

    Args:
        stage: The USD stage.
        root_path: Root prim path to traverse.
        floor_z: Z coordinate of the floor (for height-relative classification).

    Returns:
        dict with counts per tier.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return {}

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    elevated_z = floor_z + ELEVATED_Z_OFFSET

    stats = {"structural": 0, "ground_obstacle": 0, "elevated": 0,
             "small_ground": 0, "skipped": 0}

    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            # Strip CollisionAPI from non-mesh prims (Infinigen artifacts like
            # place_cam_overhead that erroneously have physics applied).
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
            continue

        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points or len(points) == 0:
            # Remove any pre-existing collision on empty meshes to avoid
            # PhysX "does not have points" errors at runtime.
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
            stats["skipped"] += 1
            continue

        name_lower = prim.GetName().lower()

        # Compute world AABB
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        lo = rng.GetMin()
        hi = rng.GetMax()
        volume = max(0.0, (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]))
        bottom_z = lo[2]

        # Classification
        if any(p in name_lower for p in STRUCTURAL_PATTERNS) and "top_wall" not in name_lower:
            # Use raw triangle mesh for floors/walls — exact geometry,
            # no convex artifacts that snag the robot.
            tier = "structural"
            approximation = "none"
        elif any(p in name_lower for p in NO_COLLISION_PATTERNS) or bottom_z >= elevated_z:
            tier = "elevated"
            approximation = None
        elif volume <= SMALL_OBJECT_VOLUME:
            tier = "small_ground"
            approximation = "boundingSphere"
        else:
            tier = "ground_obstacle"
            approximation = "convexHull"

        stats[tier] += 1

        # Apply or remove collision
        if approximation is None:
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
        else:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
            else:
                mesh_col = UsdPhysics.MeshCollisionAPI(prim)
            mesh_col.GetApproximationAttr().Set(approximation)

    return stats


def _deactivate_degenerate_meshes(stage, root_path):
    """Deactivate mesh prims that are empty or degenerate (Infinigen temp artifacts).

    Uses SetActive(False) instead of RemovePrim since prims from USD references
    are read-only. Deactivated prims are skipped by PhysX, rendering, and traversals.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    to_deactivate = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points or len(points) == 0:
            to_deactivate.append(prim)
            continue
        # Catch degenerate meshes with zero-volume bounding box
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        lo = rng.GetMin()
        hi = rng.GetMax()
        volume = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
        if volume < 1e-8:
            to_deactivate.append(prim)
    for prim in to_deactivate:
        prim.SetActive(False)
    return len(to_deactivate)


def _hide_ceiling_prims(stage, root_path):
    """Hide ceiling/top_wall prims via USD visibility."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        name = prim.GetName().lower()
        if "ceiling" in name or "top_wall" in name:
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                count += 1
    return count


# =============================================================================
# Texture Stripping
# =============================================================================


def strip_texture_files(textures_dir):
    """Delete non-DIFFUSE texture files from the source textures directory.

    Omniverse gracefully falls back to flat shading for missing texture refs.

    Args:
        textures_dir: Path to the room_*/textures/ directory.

    Returns:
        Number of files deleted.
    """
    textures_dir = Path(textures_dir)
    if not textures_dir.is_dir():
        return 0
    deleted = 0
    for f in list(textures_dir.iterdir()):
        if not f.is_file():
            continue
        # Keep DIFFUSE textures, delete everything else (NORMAL, ROUGHNESS, METAL, EMIT, etc.)
        if "_DIFFUSE." in f.name:
            continue
        f.unlink()
        deleted += 1
    return deleted


# =============================================================================
# Floor Analysis (unchanged)
# =============================================================================


def find_floor_prims(stage, root_path="/Environment"):
    """Find ALL floor mesh prims by searching for '*floor*' under root.

    Multi-room Infinigen scenes have one floor mesh per room (e.g.
    bedroom_0_0_floor, bathroom_0_0_floor, hallway_0_0_floor).
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []
    floors = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) and "floor" in prim.GetName().lower():
            floors.append(prim)
    return floors


def compute_floor_bbox(stage, root_path="/Environment"):
    """Compute the world-aligned bounding box of all floor meshes (union).

    For multi-room scenes, unions bboxes of all *_floor meshes so centering
    places the navigable floor area at origin.
    Returns ((x_min, x_max), (y_min, y_max), floor_z).
    Falls back to the whole-room bbox if no floor meshes are found.
    """
    floor_prims = find_floor_prims(stage, root_path)
    if not floor_prims:
        target = stage.GetPrimAtPath(root_path)
        if not target or not target.IsValid():
            return ((-5, 5), (-5, 5), 0.0)
        floor_prims = [target]

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])

    # Union bbox across all floor prims
    x_lo, y_lo, z_lo = float("inf"), float("inf"), float("inf")
    x_hi, y_hi = float("-inf"), float("-inf")
    for prim in floor_prims:
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        lo = rng.GetMin()
        hi = rng.GetMax()
        x_lo = min(x_lo, lo[0])
        y_lo = min(y_lo, lo[1])
        z_lo = min(z_lo, lo[2])
        x_hi = max(x_hi, hi[0])
        y_hi = max(y_hi, hi[1])

    inset = 0.5
    return (
        (x_lo + inset, x_hi - inset),
        (y_lo + inset, y_hi - inset),
        z_lo,
    )


def center_at_origin(stage, root_path="/Environment"):
    """Translate the environment root so floor bbox center is at (0, 0, 0).

    Replicates the logic from infinigen_sdg_utils.center_scene_at_origin().
    Returns metadata dict with centered floor bbox.
    """
    x_range, y_range, floor_z = compute_floor_bbox(stage, root_path)
    cx = (x_range[0] + x_range[1]) / 2.0
    cy = (y_range[0] + y_range[1]) / 2.0

    root_prim = stage.GetPrimAtPath(root_path)
    xform = UsdGeom.Xformable(root_prim)

    # Check if centering op already exists (idempotent)
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate:centering":
            print("  Centering op already exists, updating...")
            op.Set(Gf.Vec3d(-cx, -cy, -floor_z))
            return _recompute_metadata(stage, root_path)

    xform.AddTranslateOp(opSuffix="centering").Set(Gf.Vec3d(-cx, -cy, -floor_z))
    return _recompute_metadata(stage, root_path)


def _extract_floor_triangles(stage, root_path):
    """Extract world-space triangles from all floor meshes.

    Returns list of ((ax,ay), (bx,by), (cx,cy), area) tuples — the XY
    projection of each triangle with its 2D area.
    """
    floor_prims = find_floor_prims(stage, root_path)
    triangles = []
    for prim in floor_prims:
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not face_counts or not face_indices:
            continue

        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        # Transform all points to world space once
        world_pts = []
        for pt in points:
            wp = xf.Transform(Gf.Vec3d(pt))
            world_pts.append((wp[0], wp[1]))

        # Walk face_indices using face_counts to extract triangles
        idx_offset = 0
        for nv in face_counts:
            if nv < 3:
                idx_offset += nv
                continue
            # Fan-triangulate: (v0, v1, v2), (v0, v2, v3), ...
            v0 = face_indices[idx_offset]
            for k in range(1, nv - 1):
                v1 = face_indices[idx_offset + k]
                v2 = face_indices[idx_offset + k + 1]
                a = world_pts[v0]
                b = world_pts[v1]
                c = world_pts[v2]
                # 2D cross product = 2 * signed area
                area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
                if area > 1e-8:
                    triangles.append((a, b, c, area))
            idx_offset += nv

    return triangles


def _sample_interior_points(triangles, num_points=500):
    """Sample uniformly distributed interior points from floor triangles.

    Uses area-weighted triangle selection + barycentric sampling to get
    a uniform distribution across the walkable floor surface.
    """
    if not triangles:
        return []

    areas = [t[3] for t in triangles]
    total_area = sum(areas)
    if total_area < 1e-8:
        return []

    # Cumulative distribution for area-weighted sampling
    cum_areas = []
    running = 0.0
    for a in areas:
        running += a
        cum_areas.append(running)

    points = []
    for _ in range(num_points):
        # Pick triangle weighted by area
        r = random.uniform(0, total_area)
        tri_idx = 0
        for i, ca in enumerate(cum_areas):
            if r <= ca:
                tri_idx = i
                break
        a, b, c, _ = triangles[tri_idx]

        # Barycentric sampling: uniform interior point
        r1 = random.random()
        r2 = random.random()
        sqrt_r1 = math.sqrt(r1)
        px = (1 - sqrt_r1) * a[0] + sqrt_r1 * (1 - r2) * b[0] + sqrt_r1 * r2 * c[0]
        py = (1 - sqrt_r1) * a[1] + sqrt_r1 * (1 - r2) * b[1] + sqrt_r1 * r2 * c[1]
        points.append([round(px, 3), round(py, 3)])

    return points


def _recompute_metadata(stage, root_path):
    """Recompute floor bbox and sample interior spawn points after centering."""
    omni.kit.app.get_app().update()
    x_range_c, y_range_c, floor_z_c = compute_floor_bbox(stage, root_path)

    # Extract triangles from all floor meshes and sample interior points
    triangles = _extract_floor_triangles(stage, root_path)
    spawn_points_xy = _sample_interior_points(triangles, SPAWN_POINTS_PER_SCENE)
    num_floors = len(find_floor_prims(stage, root_path))

    return {
        "x_min": round(x_range_c[0], 3),
        "x_max": round(x_range_c[1], 3),
        "y_min": round(y_range_c[0], 3),
        "y_max": round(y_range_c[1], 3),
        "floor_z": round(floor_z_c, 3),
        "num_floors": num_floors,
        "num_floor_triangles": len(triangles),
        "spawn_points_xy": spawn_points_xy,
    }


# =============================================================================
# Room Processing Pipeline
# =============================================================================


def process_room(room_usd_path: Path, export_path: Path, dry_run: bool = False):
    """Process a single room USD: decimate, set collision, hide ceilings, center.

    Args:
        room_usd_path: Path to the source room_*/export_scene.usdc.
        export_path: Path to write the processed scene_NNNN.usdc.
        dry_run: If True, print what would be done without saving.

    Returns:
        (metadata_dict, optimization_stats_dict)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {room_usd_path.parent.name}/{room_usd_path.name}")
    print(f"  -> {export_path.name}")
    print(f"{'='*60}")

    opt_stats = {}

    # Step 0: Strip non-essential textures (opt-in, before USD loads reference)
    if args.strip_textures:
        textures_dir = room_usd_path.parent / "textures"
        deleted_textures = strip_texture_files(textures_dir)
        print(f"  Stripped {deleted_textures} non-essential texture files")
        opt_stats["textures_stripped"] = deleted_textures
    else:
        opt_stats["textures_stripped"] = 0

    # Open a fresh stage with this room
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    omni.kit.app.get_app().update()

    # Create two-level hierarchy:
    #   /Environment (Xform) -- becomes defaultPrim; spawner writes translate(0,0,0) here
    #   /Environment/Geometry (Xform) -- room reference + centering xform (survives spawner)
    root_path = "/Environment"
    geom_path = "/Environment/Geometry"
    stage.DefinePrim(root_path, "Xform")
    geom_prim = stage.DefinePrim(geom_path, "Xform")
    abs_path = str(room_usd_path.resolve())
    geom_prim.GetReferences().AddReference(abs_path)
    omni.kit.app.get_app().update()

    # Step 0b: Deactivate empty/degenerate mesh prims (Infinigen temp artifacts)
    removed_empty = _deactivate_degenerate_meshes(stage, geom_path)
    if removed_empty:
        print(f"  Deactivated {removed_empty} empty/degenerate mesh prims")

    # Step 1: Decimate meshes (skip structural: floors + walls)
    if not args.no_decimate:
        orig_faces, new_faces, dec_count = decimate_meshes(
            stage, geom_path,
            ratio=args.decimate_ratio,
            skip_patterns=("floor", "wall"),
        )
        reduction = (1 - new_faces / max(orig_faces, 1)) * 100
        print(f"  Decimated {dec_count} meshes: {orig_faces:,} -> {new_faces:,} faces "
              f"({reduction:.1f}% reduction)")
        opt_stats["original_faces"] = orig_faces
        opt_stats["decimated_faces"] = new_faces
        opt_stats["reduction_pct"] = round(reduction, 1)
        opt_stats["meshes_decimated"] = dec_count
    else:
        print("  Skipping mesh decimation (--no_decimate)")

    # Step 2: Tiered collision strategy
    _, _, floor_z = compute_floor_bbox(stage, geom_path)
    collision_stats = classify_and_set_collision(stage, geom_path, floor_z=floor_z)
    print(f"  Collision tiers: {collision_stats}")
    opt_stats["collision_tiers"] = collision_stats

    # Step 3: Hide ceiling prims (visual only, collision handled by tier)
    hidden_count = _hide_ceiling_prims(stage, geom_path)
    print(f"  Hidden {hidden_count} ceiling prims")

    # Step 4: Compute pre-centering bbox
    x_range, y_range, floor_z = compute_floor_bbox(stage, geom_path)
    cx = (x_range[0] + x_range[1]) / 2.0
    cy = (y_range[0] + y_range[1]) / 2.0
    print(f"  Pre-center bbox: X=[{x_range[0]:.1f}, {x_range[1]:.1f}], "
          f"Y=[{y_range[0]:.1f}, {y_range[1]:.1f}], Z={floor_z:.3f}")
    print(f"  Center offset: ({-cx:.3f}, {-cy:.3f}, {-floor_z:.3f})")

    # Step 5: Center at origin (centering xform goes on /Environment/Geometry)
    metadata = center_at_origin(stage, geom_path)
    print(f"  Post-center bbox: X=[{metadata['x_min']:.1f}, {metadata['x_max']:.1f}], "
          f"Y=[{metadata['y_min']:.1f}, {metadata['y_max']:.1f}], "
          f"floor_z={metadata['floor_z']:.3f}")

    if dry_run:
        print("  [DRY RUN] Would save to:", str(export_path))
        return metadata, opt_stats

    # Step 6: Set defaultPrim so MultiUsdFileCfg can reference via <defaultPrim>
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))

    # Step 7: Flatten and export to the output scenes directory
    # Flatten resolves all references into a single layer so our modifications
    # (deactivation, collision, decimation, centering) are baked into the output
    # and Infinigen artifacts can't resurface through reference composition.
    export_path.parent.mkdir(parents=True, exist_ok=True)
    flattened = stage.Flatten()
    flattened.Export(str(export_path.resolve()))
    print(f"  Saved: {export_path}")

    return metadata, opt_stats


def main():
    rooms_dir = Path(args.rooms_dir)
    output_dir = Path(args.output_dir)

    if not rooms_dir.is_dir():
        print(f"[ERROR] Rooms directory not found: {rooms_dir}")
        sys.exit(1)

    room_usds = sorted(rooms_dir.glob("room_*/export_scene.usdc"))
    if not room_usds:
        print(f"[ERROR] No room_*/export_scene.usdc found in: {rooms_dir}")
        sys.exit(1)

    print(f"Found {len(room_usds)} rooms to process in: {rooms_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Options: decimate={'OFF' if args.no_decimate else f'ratio={args.decimate_ratio}'}, "
          f"strip_textures={'ON' if args.strip_textures else 'OFF'}")
    if args.dry_run:
        print("[DRY RUN MODE] No files will be modified.")

    results = {}
    all_opt_stats = {}
    for idx, room_usd in enumerate(room_usds):
        room_name = room_usd.parent.name
        export_path = output_dir / f"scene_{idx:04d}.usdc"
        metadata, opt_stats = process_room(room_usd, export_path, dry_run=args.dry_run)
        results[room_name] = metadata
        all_opt_stats[room_name] = opt_stats

    print(f"\n{'='*60}")
    print(f"Summary: processed {len(results)} rooms")
    print(f"{'='*60}")
    for name, meta in sorted(results.items()):
        npts = len(meta.get("spawn_points_xy", []))
        nfloors = meta.get("num_floors", 1)
        ntris = meta.get("num_floor_triangles", 0)
        opt = all_opt_stats.get(name, {})
        dec_info = ""
        if "original_faces" in opt:
            dec_info = f", {opt['original_faces']:,}->{opt['decimated_faces']:,} faces ({opt['reduction_pct']}%)"
        print(f"  {name}: floor=[{meta['x_min']:.1f},{meta['x_max']:.1f}]x"
              f"[{meta['y_min']:.1f},{meta['y_max']:.1f}], "
              f"floor_z={meta['floor_z']:.2f}, "
              f"{nfloors} floor(s), {ntris} tris, {npts} spawn pts{dec_info}")

    # Build scenes dict with optimization stats
    scenes = {}
    for idx, (name, meta) in enumerate(sorted(results.items())):
        scene_key = f"scene_{idx:04d}"
        meta_with_opt = dict(meta)
        opt = all_opt_stats.get(name, {})
        if opt:
            meta_with_opt["optimization"] = opt
        scenes[scene_key] = meta_with_opt

    # Compute conservative spawn bounds (intersection of all scene floor areas)
    # with additional robot-radius inset so the robot never spawns at room edges.
    ROBOT_INSET = 0.25  # safety margin for robot half-width (~0.2m)
    half_extents = []
    for m in scenes.values():
        hx = min(abs(m["x_min"]), abs(m["x_max"]))
        hy = min(abs(m["y_min"]), abs(m["y_max"]))
        half_extents.append(min(hx, hy) - ROBOT_INSET)
    safe_radius = max(0.5, min(half_extents))  # at least 0.5m

    scenes_metadata = {
        "scenes": scenes,
        "conservative_spawn": {
            "x": (round(-safe_radius, 2), round(safe_radius, 2)),
            "y": (round(-safe_radius, 2), round(safe_radius, 2)),
        },
    }

    if not args.dry_run:
        meta_path = output_dir / "scenes_metadata.json"
        meta_path.write_text(json.dumps(scenes_metadata, indent=2))
        print(f"\nMetadata saved: {meta_path}")
        print(f"  Scenes: {len(scenes)}")
        print(f"  Conservative spawn: x={scenes_metadata['conservative_spawn']['x']}, "
              f"y={scenes_metadata['conservative_spawn']['y']}")

    print("\nDone! Rooms are ready for Isaac Lab training.")


if __name__ == "__main__":
    main()
    simulation_app.close()
