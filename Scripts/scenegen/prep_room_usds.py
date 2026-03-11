#!/usr/bin/env python3
"""One-time preprocessing script to prepare Infinigen room USDs for Isaac Lab training.

This script processes each room_*/export_scene.usdc and exports to
Assets/generated/scenes/scene_NNNN.usdc (ready for ProcDepth env configs):
1. Center the floor bounding box at world origin (0, 0, 0)
2. Apply CollisionAPI + MeshCollisionAPI(none = triangle mesh) to all meshes
3. Hide ceiling/top_wall prims via USD visibility

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


def setup_collision_and_hide_ceiling(stage, root_path="/Environment"):
    """Apply CollisionAPI to all meshes and hide ceiling prims.

    Replicates the logic from infinigen_sdg_utils.setup_env().
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        print(f"  [WARN] root prim {root_path} not found")
        return 0, 0, 0

    mesh_count = 0
    hidden_count = 0

    skipped_count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            # Skip meshes with no points (placeholder prims like
            # place_cam_overhead, hoof_parent_temp, etc.)
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if not points or len(points) == 0:
                skipped_count += 1
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_col.GetApproximationAttr().Set("none")
            mesh_count += 1

        name = prim.GetName().lower()
        if "ceiling" in name or "top_wall" in name:
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                hidden_count += 1

    return mesh_count, hidden_count, skipped_count


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


SPAWN_POINTS_PER_SCENE = 500


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


def process_room(room_usd_path: Path, export_path: Path, dry_run: bool = False):
    """Process a single room USD: center, add collision, hide ceilings.

    Args:
        room_usd_path: Path to the source room_*/export_scene.usdc.
        export_path: Path to write the processed scene_NNNN.usdc.
        dry_run: If True, print what would be done without saving.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {room_usd_path.parent.name}/{room_usd_path.name}")
    print(f"  -> {export_path.name}")
    print(f"{'='*60}")

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
    omni.kit.app.get_app().update()

    # Step 1: Add collision and hide ceilings (traverse under Geometry)
    mesh_count, hidden_count, skipped_count = setup_collision_and_hide_ceiling(stage, geom_path)
    print(f"  Collision applied to {mesh_count} meshes, hidden {hidden_count} ceiling prims, skipped {skipped_count} empty meshes")

    # Step 2: Compute pre-centering bbox (from floor mesh under Geometry)
    x_range, y_range, floor_z = compute_floor_bbox(stage, geom_path)
    cx = (x_range[0] + x_range[1]) / 2.0
    cy = (y_range[0] + y_range[1]) / 2.0
    print(f"  Pre-center bbox: X=[{x_range[0]:.1f}, {x_range[1]:.1f}], Y=[{y_range[0]:.1f}, {y_range[1]:.1f}], Z={floor_z:.3f}")
    print(f"  Center offset: ({-cx:.3f}, {-cy:.3f}, {-floor_z:.3f})")

    # Step 3: Center at origin (centering xform goes on /Environment/Geometry)
    metadata = center_at_origin(stage, geom_path)
    print(f"  Post-center bbox: {metadata}")

    if dry_run:
        print("  [DRY RUN] Would save to:", str(export_path))
        return metadata

    # Step 4: Set defaultPrim so MultiUsdFileCfg can reference via <defaultPrim>
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))

    # Step 5: Export to the output scenes directory
    export_path.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(export_path.resolve()))
    print(f"  Saved: {export_path}")

    return metadata


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
    if args.dry_run:
        print("[DRY RUN MODE] No files will be modified.")

    results = {}
    for idx, room_usd in enumerate(room_usds):
        room_name = room_usd.parent.name
        export_path = output_dir / f"scene_{idx:04d}.usdc"
        metadata = process_room(room_usd, export_path, dry_run=args.dry_run)
        results[room_name] = metadata

    print(f"\n{'='*60}")
    print(f"Summary: processed {len(results)} rooms")
    print(f"{'='*60}")
    for name, meta in sorted(results.items()):
        npts = len(meta.get("spawn_points_xy", []))
        nfloors = meta.get("num_floors", 1)
        ntris = meta.get("num_floor_triangles", 0)
        print(f"  {name}: floor=[{meta['x_min']:.1f},{meta['x_max']:.1f}]x"
              f"[{meta['y_min']:.1f},{meta['y_max']:.1f}], "
              f"floor_z={meta['floor_z']:.2f}, "
              f"{nfloors} floor(s), {ntris} tris, {npts} spawn pts")

    # Filter out degenerate scenes (corrupted geometry with extreme bbox values)
    MAX_ROOM_EXTENT = 50.0  # no real room exceeds 50m in any dimension
    valid_scenes = {}
    skipped_scenes = []
    for idx, (name, meta) in enumerate(sorted(results.items())):
        scene_key = f"scene_{idx:04d}"
        extents_ok = (
            abs(meta["x_min"]) < MAX_ROOM_EXTENT
            and abs(meta["x_max"]) < MAX_ROOM_EXTENT
            and abs(meta["y_min"]) < MAX_ROOM_EXTENT
            and abs(meta["y_max"]) < MAX_ROOM_EXTENT
            and abs(meta["floor_z"]) < MAX_ROOM_EXTENT
        )
        if extents_ok:
            valid_scenes[scene_key] = meta
        else:
            skipped_scenes.append(scene_key)
            print(f"  [WARN] Skipping degenerate scene {scene_key} ({name}): bbox out of range")
            # Remove the exported file for degenerate scenes
            bad_path = output_dir / f"{scene_key}.usdc"
            if bad_path.exists():
                bad_path.unlink()
                print(f"    Removed: {bad_path}")

    if not valid_scenes:
        print("[ERROR] No valid scenes after filtering!")
        sys.exit(1)

    # Compute conservative spawn bounds (intersection of all valid scene floor areas)
    # with additional robot-radius inset so the robot never spawns at room edges.
    # For each scene, the spawn area is the smaller of its x/y half-extents.
    # The global conservative range uses the minimum half-extent across all scenes
    # applied symmetrically to both axes, so robots always fit in any room.
    ROBOT_INSET = 0.25  # safety margin for robot half-width (~0.2m)
    half_extents = []
    for m in valid_scenes.values():
        hx = min(abs(m["x_min"]), abs(m["x_max"]))
        hy = min(abs(m["y_min"]), abs(m["y_max"]))
        half_extents.append(min(hx, hy) - ROBOT_INSET)
    safe_radius = max(0.5, min(half_extents))  # at least 0.5m

    scenes_metadata = {
        "scenes": valid_scenes,
        "conservative_spawn": {
            "x": (round(-safe_radius, 2), round(safe_radius, 2)),
            "y": (round(-safe_radius, 2), round(safe_radius, 2)),
        },
    }

    if skipped_scenes:
        scenes_metadata["skipped_degenerate"] = skipped_scenes

    if not args.dry_run:
        meta_path = output_dir / "scenes_metadata.json"
        meta_path.write_text(json.dumps(scenes_metadata, indent=2))
        print(f"\nMetadata saved: {meta_path}")
        print(f"  Valid scenes: {len(valid_scenes)}, Skipped: {len(skipped_scenes)}")
        print(f"  Conservative spawn: x={scenes_metadata['conservative_spawn']['x']}, "
              f"y={scenes_metadata['conservative_spawn']['y']}")

    print("\nDone! Rooms are ready for Isaac Lab training.")


if __name__ == "__main__":
    main()
    simulation_app.close()
