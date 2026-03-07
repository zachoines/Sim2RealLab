#!/usr/bin/env python3
"""Split a scene manifest into train/val/test sets (Phase 6a.3 — STUB).

Reads scene_manifest.jsonl and produces:
  - Assets/generated/manifests/splits/train.txt
  - Assets/generated/manifests/splits/val.txt
  - Assets/generated/manifests/splits/test.txt

Each split file lists scene IDs, one per line.

Usage (once scene_manifest.jsonl exists):
    python Scripts/scenegen/split_scene_dataset.py \
        --scene_manifest Assets/generated/manifests/scene_manifest.jsonl \
        --train 120 --val 40 --test 40
"""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Split scene dataset (STUB)")
    parser.add_argument("--scene_manifest", type=str, required=True)
    parser.add_argument("--train", type=int, default=120)
    parser.add_argument("--val", type=int, default=40)
    parser.add_argument("--test", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("split_scene_dataset.py — STUB (Phase 6a.3)")
    print("=" * 60)
    print()
    print("This script is a placeholder. It will be implemented after")
    print("compose_scenes_replicator.py generates scene_manifest.jsonl.")
    print()
    print(f"  Scene manifest: {args.scene_manifest}")
    print(f"  Split: train={args.train}, val={args.val}, test={args.test}")
    sys.exit(0)


if __name__ == "__main__":
    main()
