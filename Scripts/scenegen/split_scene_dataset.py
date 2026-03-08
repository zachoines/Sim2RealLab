#!/usr/bin/env python3
"""Split a scene manifest into train/val/test sets.

Reads scene_manifest.jsonl and produces split files listing scene IDs.
Stratifies by layout_class to maintain distribution across splits.

Usage:
    python Scripts/scenegen/split_scene_dataset.py \
        --scene_manifest Assets/generated/manifests/scene_manifest.jsonl \
        --out_dir Assets/generated/manifests/splits \
        --train_frac 0.6 --val_frac 0.2 --test_frac 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load_scene_manifest(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def stratified_split(
    scenes: list[dict],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split scene IDs with stratification by layout_class."""
    rng = random.Random(seed)

    # Group by layout class
    by_layout: dict[str, list[str]] = defaultdict(list)
    for scene in scenes:
        by_layout[scene["layout_class"]].append(scene["scene_id"])

    train_ids, val_ids, test_ids = [], [], []

    for layout, ids in sorted(by_layout.items()):
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))
        # test gets the remainder
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser(description="Split scene dataset into train/val/test")
    parser.add_argument("--scene_manifest", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="Assets/generated/manifests/splits")
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 0.01:
        print(f"ERROR: fractions must sum to 1.0, got {total}", file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(args.scene_manifest)
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    scenes = load_scene_manifest(manifest_path)
    print(f"Loaded {len(scenes)} scenes from {manifest_path}")

    train_ids, val_ids, test_ids = stratified_split(
        scenes, args.train_frac, args.val_frac, args.test_frac, args.seed
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_path = out_dir / f"{split_name}.txt"
        with open(split_path, "w") as f:
            for scene_id in sorted(ids):
                f.write(scene_id + "\n")
        print(f"  {split_name}: {len(ids)} scenes -> {split_path}")

    # Verify layout stratification
    by_layout: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    layout_map = {s["scene_id"]: s["layout_class"] for s in scenes}
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for sid in ids:
            by_layout[layout_map[sid]][split_name] += 1

    print("\nLayout stratification:")
    print(f"  {'Layout':<20} {'train':>6} {'val':>6} {'test':>6}")
    for layout in sorted(by_layout.keys()):
        counts = by_layout[layout]
        print(f"  {layout:<20} {counts['train']:>6} {counts['val']:>6} {counts['test']:>6}")


if __name__ == "__main__":
    main()
