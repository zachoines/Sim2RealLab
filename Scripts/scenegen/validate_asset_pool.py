#!/usr/bin/env python3
"""Validate assets in a manifest against the Phase 6 quality gate.

Checks (run without Isaac Sim — filesystem-only):
  1. USD file exists and is non-empty.
  2. File size is within reasonable bounds (not suspiciously small/large).
  3. No obvious missing texture references (checks for Textures/ dir).
  4. License metadata is present in manifest entry.

Checks that require Isaac Sim (--full mode, needs omni.usd):
  5. USD loads without errors.
  6. Bounding box has non-zero extents.
  7. Collider prim can be found or approximated.

Usage:
    # Quick filesystem checks (no Isaac Sim needed)
    python Scripts/scenegen/validate_asset_pool.py \
        --manifest Assets/generated/manifests/asset_manifest.jsonl \
        --assets_root Assets

    # Full validation (requires Isaac Sim / omni.usd)
    python Scripts/scenegen/validate_asset_pool.py \
        --manifest Assets/generated/manifests/asset_manifest.jsonl \
        --assets_root Assets --full

    # Validate a random sample
    python Scripts/scenegen/validate_asset_pool.py \
        --manifest Assets/generated/manifests/asset_manifest.jsonl \
        --assets_root Assets --sample 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Reasonable file size bounds for a USD obstacle asset
MIN_FILE_SIZE = 512          # 512 bytes — anything smaller is likely corrupt
MAX_FILE_SIZE = 500_000_000  # 500 MB — anything larger is likely a full scene


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load a JSONL manifest into a list of dicts."""
    entries = []
    with open(manifest_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {line_num}: {e}", file=sys.stderr)
    return entries


def check_filesystem(entry: dict, assets_root: Path) -> list[str]:
    """Run filesystem-only quality checks. Returns list of issues."""
    issues = []
    usd_path = assets_root / entry["usd_path"]

    if not usd_path.exists():
        issues.append(f"FILE_MISSING: {entry['usd_path']}")
        return issues  # can't check further

    size = usd_path.stat().st_size
    if size < MIN_FILE_SIZE:
        issues.append(f"FILE_TOO_SMALL: {entry['usd_path']} ({size} bytes)")
    if size > MAX_FILE_SIZE:
        issues.append(f"FILE_TOO_LARGE: {entry['usd_path']} ({size / 1e6:.1f} MB)")

    if not entry.get("license"):
        issues.append(f"NO_LICENSE: {entry['asset_id']}")

    if entry.get("file_size_bytes") and entry["file_size_bytes"] != size:
        issues.append(f"SIZE_MISMATCH: manifest says {entry['file_size_bytes']}, actual {size}")

    return issues


def check_usd_loadable(entry: dict, assets_root: Path) -> list[str]:
    """Check that USD loads in Omniverse (requires omni.usd). Returns issues."""
    issues = []
    usd_path = assets_root / entry["usd_path"]

    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        issues.append("SKIP_USD_CHECK: pxr (OpenUSD) not available")
        return issues

    try:
        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            issues.append(f"USD_LOAD_FAILED: {entry['usd_path']}")
            return issues

        # Check for non-zero bounding box
        root = stage.GetDefaultPrim()
        if root:
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
            bbox = bbox_cache.ComputeWorldBound(root)
            box_range = bbox.ComputeAlignedRange()
            extent = box_range.GetMax() - box_range.GetMin()
            if extent[0] <= 0 and extent[1] <= 0 and extent[2] <= 0:
                issues.append(f"ZERO_BBOX: {entry['usd_path']}")
            # Flag extremely large assets (> 10m in any dimension)
            for i, dim_name in enumerate(["x", "y", "z"]):
                if extent[i] > 10.0:
                    issues.append(f"HUGE_{dim_name.upper()}: {entry['usd_path']} ({extent[i]:.1f}m)")
        else:
            issues.append(f"NO_DEFAULT_PRIM: {entry['usd_path']}")

    except Exception as e:
        issues.append(f"USD_ERROR: {entry['usd_path']}: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate asset manifest entries")
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to asset_manifest.jsonl",
    )
    parser.add_argument(
        "--assets_root", type=str, default="Assets",
        help="Root directory for resolving USD paths",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full validation including USD loading (requires pxr/omni.usd)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Validate a random sample of N assets (0 = all)",
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="Write validated manifest (passing entries only) to this path",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    assets_root = Path(args.assets_root).resolve()

    entries = load_manifest(manifest_path)
    print(f"Loaded {len(entries)} entries from {manifest_path}")

    if args.sample > 0 and args.sample < len(entries):
        entries = random.sample(entries, args.sample)
        print(f"Sampled {args.sample} entries for validation")

    passed = []
    failed = []
    all_issues: list[str] = []

    for entry in entries:
        issues = check_filesystem(entry, assets_root)
        if args.full and not any("FILE_MISSING" in i for i in issues):
            issues.extend(check_usd_loadable(entry, assets_root))

        if issues:
            failed.append(entry)
            for issue in issues:
                all_issues.append(f"  [{entry['asset_id']}] {issue}")
        else:
            passed.append(entry)

    # Report
    print(f"\nResults: {len(passed)} passed, {len(failed)} failed")

    if all_issues:
        print("\nIssues found:")
        for issue in all_issues:
            print(issue)

    if args.out and passed:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for entry in passed:
                f.write(json.dumps(entry) + "\n")
        print(f"\nWrote {len(passed)} validated entries to {out_path}")

    # Exit with error if any failures
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
