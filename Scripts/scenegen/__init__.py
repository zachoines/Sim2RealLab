"""Scene generation pipeline for Phase 6 synthetic obstacle diversity.

Follows the NVIDIA Infinigen + Replicator SDG architecture:
  1. generate_infinigen_rooms.sh  - Generate indoor rooms in WSL (Infinigen)
  2. build_asset_manifest.py      - Scan local asset packs -> asset_manifest.jsonl
  3. validate_asset_pool.py       - Quality-gate checks on manifest entries
  4. compose_scenes_replicator.py - Isaac Sim standalone: load rooms + spawn + Replicator
  5. split_scene_dataset.py       - Train/val/test split from scene_manifest.jsonl
  6. export_scene_stats.py        - Report statistics on manifests
"""
