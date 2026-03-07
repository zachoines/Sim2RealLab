#!/usr/bin/env python3
"""Build an asset manifest (JSONL) from local OpenUSD asset packs.

Scans the Residential and SimReady Furniture asset packs under --assets_root
and emits one JSON record per USD file with metadata useful for downstream
scene composition and quality gating.

Usage:
    python Scripts/scenegen/build_asset_manifest.py \
        --assets_root Assets \
        --out Assets/generated/manifests/asset_manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Asset pack definitions
# ---------------------------------------------------------------------------

ASSET_PACKS = {
    "simready_furniture": {
        "rel_root": "SimReady_Furniture_Misc_01_NVD@10010/Assets/simready_content/common_assets/props",
        "source": "simready_furniture",
        "license": "NVIDIA Omniverse License",
    },
    "residential": {
        "rel_root": "Residential_NVD@10012/Assets/ArchVis/Residential",
        "source": "residential",
        "license": "NVIDIA Omniverse License",
    },
}

# Category tagging for residential pack (parent dir -> tag)
# Used by scene composer to sample obstacle types by ratio.
RESIDENTIAL_CATEGORY_MAP = {
    "Appliances": "appliance",
    "Decor": "decor",
    "Electronics": "electronics",
    "Entertainment": "entertainment",
    "Fixtures": "fixture",
    "Food": "food",
    "Furniture": "furniture",
    "Kitchen": "kitchen",
    "Lighting": "lighting",
    "Misc": "misc",
    "Outdoors": "outdoor",
    "Plants": "plant",
}


def _classify_residential(rel_path: str) -> str:
    """Derive a category tag from a Residential pack relative path."""
    parts = Path(rel_path).parts
    # Expected: Residential / <Category> / <Subcategory> / file.usd
    for part in parts:
        if part in RESIDENTIAL_CATEGORY_MAP:
            return RESIDENTIAL_CATEGORY_MAP[part]
    return "unknown"


def _classify_simready(rel_path: str) -> str:
    """Derive a category tag from a SimReady Furniture pack relative path.

    SimReady props are flat: props/<asset_name>/<asset_name>.usd
    We use the asset directory name to infer rough category.
    """
    name = Path(rel_path).stem.lower()

    furniture_keywords = [
        "chair", "sofa", "table", "desk", "bed", "cabinet", "shelf",
        "stool", "bench", "loveseat", "ottoman", "bookcase", "dresser",
        "nightstand", "wardrobe", "seat", "couch", "file", "storage",
    ]
    decor_keywords = [
        "vase", "bowl", "lamp", "candle", "frame", "mirror", "rug",
        "cushion", "pillow", "basket", "pot", "planter", "holder",
        "sculpture", "statue", "globe", "clock",
    ]
    misc_keywords = [
        "cone", "rail", "board", "avocado", "boat", "pen", "tray",
    ]

    for kw in furniture_keywords:
        if kw in name:
            return "furniture"
    for kw in decor_keywords:
        if kw in name:
            return "decor"
    for kw in misc_keywords:
        if kw in name:
            return "misc"
    return "furniture"  # default for SimReady pack


def scan_simready(assets_root: Path) -> list[dict]:
    """Scan SimReady Furniture pack for primary USD assets.

    Each asset lives in props/<name>/<name>.usd.  We skip *_base.usd and
    *_inst*.usd variants (internal layering files).
    """
    pack = ASSET_PACKS["simready_furniture"]
    pack_root = assets_root / pack["rel_root"]
    if not pack_root.exists():
        print(f"WARNING: SimReady pack not found at {pack_root}", file=sys.stderr)
        return []

    entries = []
    for asset_dir in sorted(pack_root.iterdir()):
        if not asset_dir.is_dir():
            continue
        # Primary USD is <name>/<name>.usd
        primary_usd = asset_dir / f"{asset_dir.name}.usd"
        if not primary_usd.exists():
            continue
        rel = primary_usd.relative_to(assets_root)
        entries.append({
            "asset_id": f"simready_{asset_dir.name}",
            "usd_path": str(rel).replace("\\", "/"),
            "source": pack["source"],
            "category": _classify_simready(asset_dir.name),
            "name": asset_dir.name,
            "license": pack["license"],
            "file_size_bytes": primary_usd.stat().st_size,
        })
    return entries


def scan_residential(assets_root: Path) -> list[dict]:
    """Scan Residential pack for all .usd files recursively."""
    pack = ASSET_PACKS["residential"]
    pack_root = assets_root / pack["rel_root"]
    if not pack_root.exists():
        print(f"WARNING: Residential pack not found at {pack_root}", file=sys.stderr)
        return []

    entries = []
    for usd_path in sorted(pack_root.rglob("*.usd")):
        rel = usd_path.relative_to(assets_root)
        rel_from_residential = usd_path.relative_to(pack_root)
        category = _classify_residential(str(rel_from_residential))
        name = usd_path.stem
        entries.append({
            "asset_id": f"residential_{name}_{hash(str(rel)) % 10000:04d}",
            "usd_path": str(rel).replace("\\", "/"),
            "source": pack["source"],
            "category": category,
            "name": name,
            "license": pack["license"],
            "file_size_bytes": usd_path.stat().st_size,
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description="Build asset manifest from local USD packs")
    parser.add_argument(
        "--assets_root", type=str, default="Assets",
        help="Root directory containing asset packs (default: Assets)",
    )
    parser.add_argument(
        "--out", type=str,
        default="Assets/generated/manifests/asset_manifest.jsonl",
        help="Output JSONL manifest path",
    )
    args = parser.parse_args()

    assets_root = Path(args.assets_root).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning asset packs under: {assets_root}")

    entries = []
    entries.extend(scan_simready(assets_root))
    entries.extend(scan_residential(assets_root))

    if not entries:
        print("ERROR: No assets found. Check --assets_root path.", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Print summary
    by_source = {}
    by_category = {}
    for e in entries:
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
        by_category[e["category"]] = by_category.get(e["category"], 0) + 1

    print(f"\nWrote {len(entries)} assets to {out_path}")
    print("\nBy source:")
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")
    print("\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
