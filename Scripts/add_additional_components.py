"""
Import and integrate the top-rack-and-hardware assembly into the Strafer USD.

This script opens the manually-positioned top-rack-and-hardware.usd (containing
goRAIL supports, U-channel beams, etc.) and reparents each mesh component under
/World/strafer/frame/rails/ so they become part of the robot hierarchy.

Materials from the component file are NOT copied — the meshes will use whatever
material bindings exist in the target stage, or can be bound manually in Composer.

Pipeline order:
  1. add_additional_components.py   <- this script (on collapsed USD)
  2. setup_physics.py (within Isaac Sim)

Example:
  python Scripts/add_additional_components.py \
      --stage Assets/strafer/3209-0001-0006-collapsed.usd \
      --components Assets/misc/top-rack-and-hardware.usd \
      --output-usd Assets/strafer/3209-0001-0006-with-components.usd \
      --log ./add_components_log.txt
"""

import argparse
from pathlib import Path
from typing import List

from pxr import Sdf, Usd, UsdGeom, UsdShade

# ============================================================================
# Configuration
# ============================================================================

# Where to place the imported components in the target stage
RAILS_PATH = "/World/strafer/frame/rails"

# Source prim path inside top-rack-and-hardware.usd
SOURCE_ROOT = "/Root/top_rack_and_hardware"

# Component names to import (meshes directly under SOURCE_ROOT)
COMPONENT_NAMES = [
    "support_rail_rear_left",
    "support_rail_rear_right",
    "support_rail_front_left",
    "support_rail_front_right",
    "long_channel_left",
    "long_channel_right",
    "cross_channel_front",
    "cross_channel_center",
]


# ============================================================================
# Core Logic
# ============================================================================


def import_components(
    target_stage: Usd.Stage,
    component_stage: Usd.Stage,
    log: List[str],
    dry_run: bool = False,
) -> int:
    """Import component meshes from the component stage into the target stage.

    Each mesh under SOURCE_ROOT is copied into /World/strafer/frame/rails/{name}
    preserving its world transform from the component file. Material bindings
    are stripped since the source materials are not copied.
    """
    # Verify target parent exists
    rails_prim = target_stage.GetPrimAtPath(RAILS_PATH)
    if not rails_prim or not rails_prim.IsValid():
        log.append(f"[ERROR] Rails parent prim not found: {RAILS_PATH}")
        return 0

    source_root = component_stage.GetPrimAtPath(SOURCE_ROOT)
    if not source_root or not source_root.IsValid():
        log.append(f"[ERROR] Source root not found: {SOURCE_ROOT}")
        return 0

    added = 0
    for comp_name in COMPONENT_NAMES:
        source_path = f"{SOURCE_ROOT}/{comp_name}"
        source_prim = component_stage.GetPrimAtPath(source_path)
        if not source_prim or not source_prim.IsValid():
            log.append(f"[WARN] Source component not found: {source_path}")
            continue

        target_path = f"{RAILS_PATH}/{comp_name}"

        # Check if already exists
        existing = target_stage.GetPrimAtPath(target_path)
        if existing and existing.IsValid():
            log.append(f"[SKIP] Already exists: {target_path}")
            continue

        if dry_run:
            log.append(f"[DRY] Would copy {source_path} -> {target_path}")
            added += 1
            continue

        # Copy the prim spec from the component layer into the target layer
        source_layer = component_stage.GetRootLayer()
        target_layer = target_stage.GetEditTarget().GetLayer()

        if not Sdf.CopySpec(source_layer, source_path, target_layer, target_path):
            log.append(f"[ERROR] Failed to copy spec: {source_path} -> {target_path}")
            continue

        # Strip material binding so it doesn't reference a missing material
        copied_prim = target_stage.GetPrimAtPath(target_path)
        if copied_prim:
            binding = UsdShade.MaterialBindingAPI(copied_prim)
            if binding:
                binding.UnbindAllBindings()
                log.append(f"[INFO] Stripped material binding on {target_path}")

        log.append(f"[OK] Copied {source_path} -> {target_path}")
        added += 1

    return added


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Import top-rack-and-hardware components into Strafer USD."
    )
    parser.add_argument("--stage", required=True, help="Path to input USD stage (collapsed).")
    parser.add_argument(
        "--components",
        default="Assets/misc/top-rack-and-hardware.usd",
        help="Path to the component assembly USD.",
    )
    parser.add_argument(
        "--output-usd",
        help="Path to save modified stage (saves in-place if omitted).",
    )
    parser.add_argument("--log", help="Path to write a summary log.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan operations without writing USD changes.",
    )
    args = parser.parse_args()

    # Open both stages
    stage_path = str(Path(args.stage).resolve())
    stage = Usd.Stage.Open(stage_path)
    if stage is None:
        raise SystemExit(f"Failed to open stage: {args.stage}")
    stage.SetEditTarget(stage.GetRootLayer())

    comp_path = str(Path(args.components).resolve())
    comp_stage = Usd.Stage.Open(comp_path)
    if comp_stage is None:
        raise SystemExit(f"Failed to open component stage: {args.components}")

    log: List[str] = []
    log.append("=" * 60)
    log.append("Strafer Additional Component Import")
    log.append("=" * 60)
    log.append(f"Target stage: {stage_path}")
    log.append(f"Component file: {comp_path}")
    log.append(f"Dry run: {args.dry_run}")

    # Import component meshes
    log.append("\n--- Components ---")
    added = import_components(stage, comp_stage, log, args.dry_run)

    log.append(f"\n{'=' * 60}")
    log.append(f"Imported {added} component(s) (dry_run={args.dry_run})")
    log.append("=" * 60)

    if not args.dry_run:
        if args.output_usd:
            out_path = Path(args.output_usd)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stage.GetRootLayer().Export(str(out_path))
            log.append(f"Saved modified stage to: {out_path.resolve()}")
        else:
            stage.GetRootLayer().Save()
            log.append(f"Saved in-place: {stage_path}")

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log), encoding="utf-8")
        print(f"Wrote log to {log_path.resolve()}")
    else:
        print("\n".join(log))


if __name__ == "__main__":
    main()
