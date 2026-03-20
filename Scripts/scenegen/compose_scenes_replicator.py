#!/usr/bin/env python3
"""Compose procedural scenes using Infinigen environments + Replicator.

Architecture:
  1. Load Infinigen-generated room USDs (pre-exported from WSL/Linux).
  2. Setup environment (add colliders, optionally hide ceiling).
  3. Spawn obstacle assets from local packs (mesh + shape distractors).
  4. Randomize poses, run physics to settle objects.
  5. Randomize lighting and camera positions.
  6. Capture annotated frames via Replicator writers.
  7. Export composed scene USD + append to scene_manifest.jsonl.

Usage (must run with Isaac Sim Python):
    cd IsaacLab
    python ..\\Scripts\\scenegen\\compose_scenes_replicator.py \\
        --config ..\\Scripts\\scenegen\\config\\strafer_sdg.yaml --headless

    # Or with isaaclab.bat:
    .\\isaaclab.bat -p ..\\Scripts\\scenegen\\compose_scenes_replicator.py \\
        --config ..\\Scripts\\scenegen\\config\\strafer_sdg.yaml

Prerequisites:
    1. Infinigen rooms exported to Assets/generated/infinigen_rooms/
       (see Scripts/scenegen/generate_infinigen_rooms.sh)
    2. Asset manifest built via Scripts/scenegen/build_asset_manifest.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from itertools import cycle
from pathlib import Path

# ---------------------------------------------------------------------------
# Default configuration (overridden by --config YAML)
# Mirrors the NVIDIA tutorial YAML schema.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "environments": {
        "folders": [
            "../Assets/generated/infinigen_rooms/",
        ],
        "files": [],
    },
    "capture": {
        "total_captures": 6,
        "num_floating_captures_per_env": 1,
        "num_dropped_captures_per_env": 2,
        "num_cameras": 1,
        "resolution": [320, 240],
        "disable_render_products": True,
        "rt_subframes": 4,
        "path_tracing": False,
        "camera_look_at_target_offset": 0.1,
        "camera_distance_to_target_range": [1.5, 3.0],
        "num_scene_lights": 3,
    },
    "writers": [
        {
            "type": "BasicWriter",
            "kwargs": {
                "output_dir": "../Assets/generated/sdg_output/basicwriter",
                "rgb": True,
                "semantic_segmentation": True,
                "colorize_semantic_segmentation": True,
                "use_common_output_dir": False,
            },
        },
    ],
    # Mesh distractors from our local asset packs.
    # These provide ground-level clutter and obstacle density.
    # Infinigen rooms provide room shells only (walls, floor, basic furniture).
    "distractors": {
        "shape_distractors": {
            "num": 25,
            "gravity_disabled_chance": 0.25,
            "types": ["capsule", "cone", "cylinder", "sphere", "cube"],
        },
        "mesh_distractors": {
            "num": 15,
            "gravity_disabled_chance": 0.25,
            "folders": [
                "../Assets/SimReady_Furniture_Misc_01_NVD@10010/Assets/"
                "simready_content/common_assets/props/",
                "../Assets/Residential_NVD@10012/Assets/ArchVis/Residential/"
                "Decor/",
                "../Assets/Residential_NVD@10012/Assets/ArchVis/Residential/"
                "Kitchen/",
                "../Assets/Residential_NVD@10012/Assets/ArchVis/Residential/"
                "Appliances/",
            ],
            "files": [],
        },
    },
    "labeled_assets": {
        "auto_label": {
            "num": 5,
            "gravity_disabled_chance": 0.25,
            "wall_snap_chance": 0.6,
            "wall_snap_offset": 0.15,
            "wall_snap_min_spacing": 0.8,
            "folders": [
                "../Assets/Residential_NVD@10012/Assets/ArchVis/Residential/"
                "Decor/",
                "../Assets/Residential_NVD@10012/Assets/ArchVis/Residential/"
                "Plants/",
                "../Assets/Residential_NVD@10012/Assets/ArchVis/Residential/"
                "Electronics/",
                "../Assets/Residential_NVD@10012/Assets/ArchVis/Residential/"
                "Lighting/",
                "../Assets/Residential_NVD@10012/Assets/ArchVis/Residential/"
                "Entertainment/",
                "../Assets/Residential_NVD@10012/Assets/ArchVis/Residential/"
                "Food/",
            ],
            "files": [],
        },
    },
    "debug_mode": True,
    # Strafer-specific: export composed scene USDs for Isaac Lab training
    "scene_export": {
        "enabled": True,
        "out_dir": "../Assets/generated/scenes/",
        "manifest_path": "../Assets/generated/manifests/scene_manifest.jsonl",
    },
}


def load_config(config_path: str | None) -> dict:
    """Load config from YAML/JSON file, falling back to defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path:
        path = Path(config_path)
        if not path.exists():
            print(f"WARNING: Config file {path} not found, using defaults",
                  file=sys.stderr)
            return config
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    user_config = yaml.safe_load(f)
                except ImportError:
                    print("WARNING: PyYAML not available, trying JSON",
                          file=sys.stderr)
                    user_config = json.load(f)
            else:
                user_config = json.load(f)
        if user_config:
            _deep_update(config, user_config)
    return config


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def check_infinigen_rooms(config: dict) -> list[str]:
    """Check if Infinigen room USDs exist. Return list of paths found."""
    env_config = config.get("environments", {})
    folders = env_config.get("folders", [])
    files = env_config.get("files", [])

    found = list(files)
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"WARNING: Environment folder not found: {folder}",
                  file=sys.stderr)
            continue
        # Look for .usd/.usdc/.usda files recursively
        for ext in ("*.usd", "*.usdc", "*.usda"):
            found.extend(str(p) for p in sorted(folder_path.rglob(ext)))

    return found


def run_sdg_pipeline(config: dict) -> None:
    """Run the full Infinigen + Replicator SDG pipeline.

    This function requires Isaac Sim runtime (omni.usd, omni.replicator).
    """
    # These imports only work inside Isaac Sim
    try:
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Usd, UsdGeom, Sdf

        # Import our local utility module (implements the NVIDIA tutorial API)
        # Located at: Scripts/scenegen/infinigen_sdg_utils.py
        sys.path.insert(0, str(Path(__file__).parent))
        from infinigen_sdg_utils import (
            get_usd_paths,
            load_env,
            setup_env,
            get_floor_prims,
            get_wall_prims,
            get_env_floor_bbox,
            get_matching_prim_location,
            spawn_labeled_assets,
            spawn_shape_distractors,
            spawn_mesh_distractors,
            scatter_on_floor,
            snap_to_walls,
            create_scene_lights,
            randomize_lights,
            register_dome_light_randomizer,
            register_shape_distractors_color_randomizer,
            randomize_dome_lights,
            randomize_shape_distractor_colors,
            run_simulation,
            randomize_camera_poses,
            setup_writer,
            center_scene_at_origin,
            verify_collision_in_exported_scene,
        )
    except ImportError as e:
        print(f"ERROR: This script must run inside Isaac Sim.", file=sys.stderr)
        print(f"  Run with: isaaclab.bat -p {__file__}", file=sys.stderr)
        print(f"  Import error: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Load environment USD paths ---
    env_config = config.get("environments", {})
    env_urls = get_usd_paths(
        files=env_config.get("files", []),
        folders=env_config.get("folders", []),
    )
    if not env_urls:
        print("ERROR: No Infinigen environment USDs found.", file=sys.stderr)
        print("  Generate rooms first with generate_infinigen_rooms.sh",
              file=sys.stderr)
        sys.exit(1)

    env_cycle = cycle(env_urls)
    print(f"Found {len(env_urls)} environment USD(s)")

    # --- Capture settings ---
    capture_cfg = config.get("capture", {})
    total_captures = capture_cfg.get("total_captures", 15)
    num_floating = capture_cfg.get("num_floating_captures_per_env", 2)
    num_dropped = capture_cfg.get("num_dropped_captures_per_env", 3)
    num_cameras = capture_cfg.get("num_cameras", 2)
    resolution = tuple(capture_cfg.get("resolution", (640, 480)))
    rt_subframes = capture_cfg.get("rt_subframes", 8)
    debug_mode = config.get("debug_mode", False)
    cam_dist_range = capture_cfg.get("camera_distance_to_target_range", [1.5, 3.0])

    # --- Create cameras ---
    stage = omni.usd.get_context().get_stage()
    cameras = []
    for i in range(num_cameras):
        cam_prim = stage.DefinePrim(f"/Cameras/cam_{i}", "Camera")
        cam_prim.GetAttribute("clippingRange").Set((0.25, 1000))
        cameras.append(cam_prim)

    # --- Create render products ---
    render_products = []
    for cam in cameras:
        rp = rep.create.render_product(
            cam.GetPath(), resolution, name=f"rp_{cam.GetName()}"
        )
        render_products.append(rp)

    if capture_cfg.get("disable_render_products", False):
        for rp in render_products:
            try:
                rp.hydra_texture.set_updates_enabled(False)
            except AttributeError:
                pass  # API not available in this version

    # Track whether we should call orchestrator / writers
    render_enabled = not capture_cfg.get("disable_render_products", False)

    # --- Setup writers ---
    writers = []
    if render_enabled:
        for writer_config in config.get("writers", []):
            writer = setup_writer(writer_config)
            if writer:
                writer.attach(render_products)
                writers.append(writer)
                print(f"  Writer: {writer_config['type']}")

    # Helper for orchestrator step (API varies across Isaac Sim versions)
    def _orchestrator_step():
        try:
            rep.orchestrator.step(pause_timeline=False)
        except TypeError:
            try:
                rep.orchestrator.step()
            except Exception:
                omni.kit.app.get_app().update()

    # --- Scene manifest for export ---
    scene_export = config.get("scene_export", {})
    manifest_entries: list[dict] = []
    scene_counter = 0

    # --- Main capture loop ---
    capture_counter = 0
    print(f"\nStarting SDG loop: {total_captures} total captures")

    while capture_counter < total_captures:
        env_url = next(env_cycle)
        print(f"\n--- Loading environment: {env_url} ---")

        # 0. Clean up previous scene prims
        for cleanup_path in ["/LabeledAssets", "/ShapeDistractors",
                             "/MeshDistractors", "/SceneLights", "/DomeLight"]:
            old = stage.GetPrimAtPath(cleanup_path)
            if old and old.IsValid():
                stage.RemovePrim(cleanup_path)

        # 1. Load Infinigen room
        load_env(env_url, prim_path="/Environment")

        # 2. Setup (add colliders, hide ceiling)
        setup_env(root_path="/Environment")

        # 2b. Discover floor mesh prims for scatter_2d placement
        floor_prims = get_floor_prims(root_path="/Environment")

        # 2c. Discover wall mesh prims for wall-snap placement
        wall_prims = get_wall_prims(root_path="/Environment")

        # 3. Find working area (fallback to scene center)
        try:
            working_area_loc = get_matching_prim_location(
                "Table", root_path="/Environment"
            )
        except Exception:
            working_area_loc = (0.0, 0.0, 0.5)

        # 4. Spawn assets
        target_assets = spawn_labeled_assets(
            config=config.get("labeled_assets", {}),
            working_area_loc=working_area_loc,
        )

        shape_distractors = spawn_shape_distractors(
            config=config.get("distractors", {}).get("shape_distractors", {}),
            working_area_loc=working_area_loc,
        )

        mesh_distractors = spawn_mesh_distractors(
            config=config.get("distractors", {}).get("mesh_distractors", {}),
            working_area_loc=working_area_loc,
        )

        # 5. Place assets: wall-snap eligible labeled assets, then floor-scatter the rest
        auto_label_cfg = config.get("labeled_assets", {}).get("auto_label", {})
        wall_snap_chance = auto_label_cfg.get("wall_snap_chance", 0.0)
        wall_snap_offset = auto_label_cfg.get("wall_snap_offset", 0.15)
        wall_snap_spacing = auto_label_cfg.get("wall_snap_min_spacing", 0.8)

        # Partition labeled assets
        wall_snap_assets = []
        floor_scatter_labeled = []
        for prim in target_assets:
            if wall_prims and random.random() < wall_snap_chance:
                wall_snap_assets.append(prim)
            else:
                floor_scatter_labeled.append(prim)

        # 5a. Wall-snap eligible assets
        failed_wall_snap = []
        if wall_snap_assets:
            _, _, floor_z = get_env_floor_bbox()
            failed_wall_snap = snap_to_walls(
                assets=wall_snap_assets,
                wall_prims=wall_prims,
                floor_z=floor_z,
                offset=wall_snap_offset,
                min_spacing=wall_snap_spacing,
            )

        # 5b. Floor-scatter everything else (including wall-snap failures)
        assets_to_scatter = (floor_scatter_labeled + failed_wall_snap
                             + mesh_distractors + shape_distractors)
        scatter_on_floor(
            assets=assets_to_scatter,
            floor_prims=floor_prims,
        )

        # 6. Setup lights
        num_scene_lights = capture_cfg.get("num_scene_lights", 3)
        scene_lights = create_scene_lights(
            num_lights=num_scene_lights,
            working_area_loc=working_area_loc,
        )
        randomize_lights(scene_lights, working_area_loc)

        # 7. Setup dome light
        register_dome_light_randomizer()
        register_shape_distractors_color_randomizer(shape_distractors)

        # 8. Initial physics to resolve overlaps
        run_simulation(num_frames=4, render=True)

        # 9. Floating captures (before physics drop)
        for i in range(num_floating):
            # Randomize per-capture
            randomize_dome_lights()
            randomize_shape_distractor_colors()
            randomize_camera_poses(
                cameras,
                targets=target_assets,
                distance_range=cam_dist_range,
                polar_angle_range=(0, 75),
            )
            # Advance a few subframes then trigger writer capture
            for _ in range(rt_subframes):
                omni.kit.app.get_app().update()
            if render_enabled:
                _orchestrator_step()

        # 10. Physics drop simulation
        run_simulation(num_frames=200, render=False)

        # 11. Dropped captures (after physics settling)
        for i in range(num_dropped):
            randomize_dome_lights()
            randomize_shape_distractor_colors()
            randomize_camera_poses(
                cameras,
                targets=target_assets,
                distance_range=cam_dist_range,
                polar_angle_range=(0, 45),
            )
            for _ in range(rt_subframes):
                omni.kit.app.get_app().update()
            if render_enabled:
                _orchestrator_step()

        # 12. Center scene at origin for consistent training env alignment
        spawn_metadata = center_scene_at_origin(root_path="/Environment")

        # 13. Export composed scene as USDC (binary, much smaller than USDA)
        scene_id = f"scene_{scene_counter:04d}"
        if scene_export.get("enabled", False):
            out_dir = Path(scene_export.get("out_dir", "../Assets/generated/scenes/"))
            out_dir.mkdir(parents=True, exist_ok=True)
            scene_usd_path = str(out_dir / f"{scene_id}.usdc")
            stage.Export(scene_usd_path)
            print(f"[compose] Exported scene: {scene_usd_path}")

            # Verify collision APIs survived the export; fix if needed
            verify_collision_in_exported_scene(scene_usd_path)

            # Copy textures from the environment USD directory so that
            # relative texture paths (./textures/...) resolve correctly
            # next to the exported scene USDC.
            env_dir = Path(env_url).resolve().parent
            src_tex = env_dir / "textures"
            dst_tex = out_dir / "textures"
            if src_tex.is_dir():
                if dst_tex.exists():
                    shutil.rmtree(dst_tex)
                shutil.copytree(str(src_tex), str(dst_tex))
                print(f"[compose] Copied textures: {src_tex} -> {dst_tex}")
        else:
            scene_usd_path = ""

        # 13. Record scene for manifest
        entry = {
            "scene_id": scene_id,
            "environment_usd": env_url,
            "scene_usd": scene_usd_path,
            "num_targets": len(target_assets),
            "num_wall_snapped": len(wall_snap_assets) - len(failed_wall_snap),
            "num_shape_distractors": len(shape_distractors),
            "num_mesh_distractors": len(mesh_distractors),
        }
        if spawn_metadata:
            entry["spawn_box"] = spawn_metadata
        manifest_entries.append(entry)
        scene_counter += 1
        capture_counter += num_floating + num_dropped

    # Wait for all writes
    if render_enabled:
        try:
            rep.orchestrator.wait_until_complete()
        except (AttributeError, Exception) as e:
            print(f"[compose] Orchestrator wait skipped: {e}")
            # Flush with a few extra frames
            for _ in range(10):
                omni.kit.app.get_app().update()

    # Write scene manifest
    if scene_export.get("enabled", False) and manifest_entries:
        manifest_path = Path(scene_export["manifest_path"])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"\nWrote scene manifest: {manifest_path} ({len(manifest_entries)} scenes)")

    print(f"\nSDG pipeline complete: {capture_counter} captures from "
          f"{scene_counter} scenes")


def main():
    parser = argparse.ArgumentParser(
        description="Compose scenes using Infinigen + Replicator (Isaac Sim standalone)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML/JSON config file (overrides defaults)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check for Infinigen rooms, don't run pipeline",
    )
    parser.add_argument(
        "--headless", action="store_true", default=False,
        help="Run Isaac Sim in headless mode (no GUI)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Check for Infinigen rooms
    rooms = check_infinigen_rooms(config)
    print(f"Infinigen rooms found: {len(rooms)}")
    for r in rooms[:5]:
        print(f"  {r}")
    if len(rooms) > 5:
        print(f"  ... and {len(rooms) - 5} more")

    if args.check_only:
        if not rooms:
            print("\nNo rooms found. Generate them with:")
            print("  bash Scripts/scenegen/generate_infinigen_rooms.sh")
            sys.exit(1)
        sys.exit(0)

    if not rooms:
        print("\nNo Infinigen rooms found. The SDG pipeline requires room USDs.")
        print("Generate them first:")
        print("  bash Scripts/scenegen/generate_infinigen_rooms.sh")
        print("\nOr point to existing rooms in the config YAML under environments.folders")
        sys.exit(1)

    # Launch Isaac Sim (must happen before any omni imports)
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})

    try:
        run_sdg_pipeline(config)
    except Exception as e:
        import traceback
        print(f"\nERROR in SDG pipeline: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
