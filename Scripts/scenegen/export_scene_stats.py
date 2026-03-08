#!/usr/bin/env python3
"""Export statistics from scene and asset manifests.

Prints summary statistics for the asset pool and generated scenes,
useful for verifying Phase 6 pipeline outputs.

Usage:
    python Scripts/scenegen/export_scene_stats.py \
        --asset_manifest Assets/generated/manifests/asset_manifest.jsonl \
        --scene_manifest Assets/generated/manifests/scene_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_asset_stats(entries: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print(f"ASSET MANIFEST: {len(entries)} assets")
    print(f"{'=' * 60}")

    by_source: dict[str, int] = defaultdict(int)
    by_category: dict[str, int] = defaultdict(int)
    total_size = 0

    for e in entries:
        by_source[e["source"]] += 1
        by_category[e["category"]] += 1
        total_size += e.get("file_size_bytes", 0)

    print(f"\nTotal file size: {total_size / 1e6:.1f} MB")

    print(f"\nBy source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src:<25} {count:>5}")

    print(f"\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat:<25} {count:>5}")


def print_scene_stats(entries: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print(f"SCENE MANIFEST: {len(entries)} scenes")
    print(f"{'=' * 60}")

    by_layout: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    obstacle_counts = []

    for scene in entries:
        by_layout[scene["layout_class"]] += 1
        n_obs = scene["num_obstacles"]
        obstacle_counts.append(n_obs)
        for obs in scene.get("obstacles", []):
            by_source[obs["source"]] += 1

    total_obs = sum(obstacle_counts)
    print(f"\nTotal obstacles: {total_obs}")
    print(f"Avg per scene:   {total_obs / len(entries):.1f}")
    print(f"Min per scene:   {min(obstacle_counts)}")
    print(f"Max per scene:   {max(obstacle_counts)}")

    print(f"\nLayout distribution:")
    for layout, count in sorted(by_layout.items()):
        print(f"  {layout:<25} {count:>5} ({count / len(entries) * 100:.0f}%)")

    print(f"\nObstacle source distribution:")
    for source, count in sorted(by_source.items()):
        print(f"  {source:<25} {count:>5} ({count / total_obs * 100:.0f}%)")

    # Unique assets used
    unique_assets = set()
    for scene in entries:
        for obs in scene.get("obstacles", []):
            unique_assets.add(obs["asset_id"])
    print(f"\nUnique assets referenced: {len(unique_assets)}")


def main():
    parser = argparse.ArgumentParser(description="Export scene generation statistics")
    parser.add_argument("--asset_manifest", type=str, default="")
    parser.add_argument("--scene_manifest", type=str, default="")
    args = parser.parse_args()

    if not args.asset_manifest and not args.scene_manifest:
        print("ERROR: Provide at least one of --asset_manifest or --scene_manifest",
              file=sys.stderr)
        sys.exit(1)

    if args.asset_manifest:
        path = Path(args.asset_manifest)
        if path.exists():
            print_asset_stats(load_jsonl(path))
        else:
            print(f"WARNING: {path} not found", file=sys.stderr)

    if args.scene_manifest:
        path = Path(args.scene_manifest)
        if path.exists():
            print_scene_stats(load_jsonl(path))
        else:
            print(f"WARNING: {path} not found", file=sys.stderr)


if __name__ == "__main__":
    main()
