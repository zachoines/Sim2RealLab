#!/usr/bin/env python3
"""One-time preprocessing script to prepare Infinigen room USDs for Isaac Lab training.

This script processes each room_*/export_scene.usdc and exports to
Assets/generated/scenes/scene_NNNN.usdc (ready for ProcDepth env configs):
1. Center the floor bounding box at world origin (0, 0, 0)
2. Apply CollisionAPI + MeshCollisionAPI(meshSimplification) to all meshes
3. Hide ceiling/top_wall prims via USD visibility

Must be run with Isaac Sim Python BEFORE training:
    cd IsaacLab
    .\\isaaclab.bat -p ..\\Scripts\\scenegen\\prep_room_usds.py

After running, scenes are ready for _get_scene_usd_paths() / MultiUsdFileCfg.
"""

import argparse
import json
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
                mesh_col.GetApproximationAttr().Set("meshSimplification")
            mesh_count += 1

        name = prim.GetName().lower()
        if "ceiling" in name or "top_wall" in name:
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                hidden_count += 1

    return mesh_count, hidden_count, skipped_count


def compute_floor_bbox(stage, root_path="/Environment"):
    """Compute the world-aligned bounding box of the environment.

    Returns ((x_min, x_max), (y_min, y_max), floor_z).
    Replicates the logic from infinigen_sdg_utils.get_env_floor_bbox().
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return ((-5, 5), (-5, 5), 0.0)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(root)
    rng = bbox.ComputeAlignedRange()
    lo = rng.GetMin()
    hi = rng.GetMax()

    inset = 0.5
    return (
        (lo[0] + inset, hi[0] - inset),
        (lo[1] + inset, hi[1] - inset),
        lo[2],
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


def _recompute_metadata(stage, root_path):
    """Recompute floor bbox after centering for metadata."""
    # Need to update the stage for transforms to take effect
    omni.kit.app.get_app().update()
    x_range_c, y_range_c, floor_z_c = compute_floor_bbox(stage, root_path)
    return {
        "x_min": round(x_range_c[0], 3),
        "x_max": round(x_range_c[1], 3),
        "y_min": round(y_range_c[0], 3),
        "y_max": round(y_range_c[1], 3),
        "floor_z": round(floor_z_c, 3),
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

    # Create root prim and add room as reference
    root_path = "/Environment"
    root_prim = stage.DefinePrim(root_path, "Xform")
    abs_path = str(room_usd_path.resolve())
    root_prim.GetReferences().AddReference(abs_path)
    omni.kit.app.get_app().update()
    omni.kit.app.get_app().update()

    # Step 1: Add collision and hide ceilings
    mesh_count, hidden_count, skipped_count = setup_collision_and_hide_ceiling(stage, root_path)
    print(f"  Collision applied to {mesh_count} meshes, hidden {hidden_count} ceiling prims, skipped {skipped_count} empty meshes")

    # Step 2: Compute pre-centering bbox
    x_range, y_range, floor_z = compute_floor_bbox(stage, root_path)
    cx = (x_range[0] + x_range[1]) / 2.0
    cy = (y_range[0] + y_range[1]) / 2.0
    print(f"  Pre-center bbox: X=[{x_range[0]:.1f}, {x_range[1]:.1f}], Y=[{y_range[0]:.1f}, {y_range[1]:.1f}], Z={floor_z:.3f}")
    print(f"  Center offset: ({-cx:.3f}, {-cy:.3f}, {-floor_z:.3f})")

    # Step 3: Center at origin
    metadata = center_at_origin(stage, root_path)
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
        print(f"  {name}: {meta}")

    # Compute conservative spawn bounds (intersection of all scene floor areas)
    # with additional robot-radius inset so the robot never spawns at room edges.
    ROBOT_INSET = 0.5  # extra safety margin beyond bbox inset
    all_x_min = max(m["x_min"] for m in results.values()) + ROBOT_INSET
    all_x_max = min(m["x_max"] for m in results.values()) - ROBOT_INSET
    all_y_min = max(m["y_min"] for m in results.values()) + ROBOT_INSET
    all_y_max = min(m["y_max"] for m in results.values()) - ROBOT_INSET

    scenes_metadata = {
        "scenes": {
            f"scene_{idx:04d}": meta
            for idx, (_, meta) in enumerate(sorted(results.items()))
        },
        "conservative_spawn": {
            "x": (round(all_x_min, 2), round(all_x_max, 2)),
            "y": (round(all_y_min, 2), round(all_y_max, 2)),
        },
    }

    if not args.dry_run:
        meta_path = output_dir / "scenes_metadata.json"
        meta_path.write_text(json.dumps(scenes_metadata, indent=2))
        print(f"\nMetadata saved: {meta_path}")
        print(f"  Conservative spawn: x={scenes_metadata['conservative_spawn']['x']}, "
              f"y={scenes_metadata['conservative_spawn']['y']}")

    print("\nDone! Rooms are ready for Isaac Lab training.")


if __name__ == "__main__":
    main()
    simulation_app.close()
