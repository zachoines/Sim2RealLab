"""Round-trip + extension-tolerance tests for the scene-provider contract.

Locks in the artifact contract documented in
``docs/SCENE_PROVIDER_CONTRACT.md``: a conformant ``scene_metadata.json``
(plus a dummy USDC) flows through the mission picker and the LeRobot
writer with no Isaac Sim runtime. A future refactor that accidentally
tightens the field requirements breaks a test here rather than a
downstream second-source author.

The extension-tolerance case is the durable half: unknown fields
(reserved ``descriptors`` namespace + a genuinely-unknown field +
a top-level unknown field) must pass through the picker and writer
without error, and must NOT leak into the writer's sidecar parquet.

Also covers the ``--ceiling-light-prim-pattern`` knob on
``postprocess_scene_usd.py`` at the regex/CLI layer (no pxr needed —
the actual injection override is exercised in
``test_postprocess_collider_approx.py`` under ``pytest.importorskip``).

Runs in ``.venv_harness`` (needs lerobot, not Isaac Sim).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from strafer_lab.tools.lerobot_writer import (
    StraferLeRobotWriter,
    hash_scene_metadata,
    read_strafer_episodes,
)
from strafer_lab.tools.teleop_mission_picker import load_candidates


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def scene_tree():
    """Yield a temp ``Assets/generated/scenes`` style tree for one scene."""
    parent = Path(tempfile.mkdtemp(prefix="strafer_contract_test_"))
    scene = "scene_contract_alpha"
    scene_dir = parent / "scenes" / scene
    scene_dir.mkdir(parents=True)
    yield scene, parent / "scenes", scene_dir
    shutil.rmtree(parent, ignore_errors=True)


def _write_scene(scene_dir: Path, metadata: dict) -> tuple[Path, Path]:
    """Author scene_metadata.json + a dummy USDC + the root symlink."""
    meta_path = scene_dir / "scene_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # A dummy USDC stands in for the geometry — the contract round-trip
    # does not open it (no pxr); the writer hashes the sidecar, not the
    # USDC. We still create the bundle so the layout is realistic.
    usdc = scene_dir / "import" / "scene.usdc"
    usdc.parent.mkdir(parents=True, exist_ok=True)
    usdc.write_bytes(b"PXR-USDC-DUMMY")
    link = scene_dir.parent / f"{scene_dir.name}.usdc"
    link.symlink_to(usdc)
    return meta_path, link


def _minimal_metadata() -> dict:
    return {
        "objects": [
            {"label": "chair", "instance_id": 1, "position_3d": [2.5, 1.2, 0.5],
             "prim_path": "/World/Chair_1"},
            {"label": "table", "instance_id": 2, "position_3d": [3.1, 0.4, 0.7],
             "prim_path": "/World/Table_2"},
        ],
        "rooms": [],
        "room_adjacency": [],
    }


def _rgb(h: int = 360, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _drive_one_episode(writer: StraferLeRobotWriter, scene: str, cand, *, meta_hash: str):
    """Replay the capture driver's candidate → writer mapping for one episode."""
    writer.begin_episode(
        mission_text=cand.mission_text,
        scene_id=scene,
        target_label=cand.label,
        target_object_id=str(cand.instance_id),
        target_position_3d=list(cand.target_position_3d),
        start_pose=[0.0, 0.0, 0.0],
        source_driver="teleop",
        source_mission_source="scene-metadata",
    )
    for t in range(3):
        writer.add_frame(
            sim_time=float(t),
            pose=[0.0] * 7,
            achieved_vel=[0.0] * 3,
            action=[0.0] * 3,
            rgb_perception=_rgb(),
        )
    writer.end_episode(outcome="succeeded", outcome_category="on_course")


# ---------------------------------------------------------------------------
# Round-trip: minimal conformant scene → picker → writer
# ---------------------------------------------------------------------------


class TestContractRoundTrip:
    def test_minimal_scene_flows_picker_to_writer(self, scene_tree):
        scene, scenes_root, scene_dir = scene_tree
        meta_path, _link = _write_scene(scene_dir, _minimal_metadata())

        candidates = load_candidates(meta_path)
        assert [c.label for c in candidates] == ["chair", "table"]
        assert candidates[0].mission_text == "go to the chair"

        meta_hash = hash_scene_metadata(meta_path)
        writer_root = scene_dir / "dataset"
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id=f"strafer-test/{scene}",
            fps=8,
            capture_git_sha="contract",
            scene_metadata_hash=meta_hash,
            capture_policy_cam=False,
        ) as writer:
            _drive_one_episode(writer, scene, candidates[0], meta_hash=meta_hash)

        episodes = read_strafer_episodes(writer_root)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep["scene_id"] == scene
        assert ep["target_label"] == "chair"
        assert ep["target_object_id"] == "1"
        assert ep["scene_metadata_hash"] == meta_hash
        assert ep["target_position_3d"] == [2.5, 1.2, 0.5]

    def test_empty_rooms_does_not_crash_picker(self, scene_tree):
        """rooms == [] must fall back to single-room semantics (no suffix)."""
        scene, _scenes_root, scene_dir = scene_tree
        meta_path, _link = _write_scene(scene_dir, _minimal_metadata())
        candidates = load_candidates(meta_path)
        assert all(c.target_room_type is None for c in candidates)

    def test_origin_drop_is_a_producer_responsibility(self, scene_tree):
        """Dropping (0,0,0) sentinels is the producer's job, not the picker's.

        Consumers may assume surviving entries have non-origin positions
        precisely because the producer's ``_drop_origin_records`` removed
        the invalid ones. The picker does NOT re-filter — a producer that
        leaks an origin row leaks it all the way to the candidate list.
        """
        scene, _scenes_root, scene_dir = scene_tree
        meta = _minimal_metadata()
        meta["objects"].append(
            {"label": "lamp", "instance_id": 9, "position_3d": [0.0, 0.0, 0.0],
             "prim_path": "/World/Lamp_9"},
        )
        meta_path, _link = _write_scene(scene_dir, meta)
        candidates = load_candidates(meta_path)
        # The picker itself does not re-drop origin rows (the producer's
        # _drop_origin_records does), so the lamp is present but pinned at
        # the origin — proving why producers must drop it. Labels present:
        assert "lamp" in [c.label for c in candidates]
        lamp = next(c for c in candidates if c.label == "lamp")
        assert lamp.target_position_3d == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Extension tolerance — the durable half of the contract
# ---------------------------------------------------------------------------


class TestExtensionTolerance:
    def _metadata_with_unknowns(self) -> dict:
        return {
            "objects": [
                # Reserved namespace (mission-text-enrichment), not yet implemented.
                {"label": "bottle", "instance_id": 10, "position_3d": [1.0, 2.0, 0.8],
                 "prim_path": "/World/BottleFactory_10__spawn_asset_1_",
                 "descriptors": {"color_name": "red",
                                 "color_hsv": [0.0, 0.85, 0.55],
                                 "material": "glass",
                                 "material_subclass": "BlownGlass",
                                 "size_bucket": "small"}},
                # Genuinely-unknown nested field — no brief reserves it.
                {"label": "bottle", "instance_id": 11, "position_3d": [1.4, 2.1, 0.8],
                 "prim_path": "/World/BottleFactory_11__spawn_asset_2_",
                 "future_field": {"unexpected": "value"}},
            ],
            "rooms": [],
            "room_adjacency": [],
            # Top-level unknown field.
            "future_top_level_field": "anything",
        }

    def test_picker_consumes_unknown_fields(self, scene_tree):
        scene, _scenes_root, scene_dir = scene_tree
        meta_path, _link = _write_scene(scene_dir, self._metadata_with_unknowns())
        # No error, both bottles enumerated (distinct spawn tokens → no collapse).
        candidates = load_candidates(meta_path)
        assert sorted(c.instance_id for c in candidates) == [10, 11]
        assert all(c.label == "bottle" for c in candidates)

    def test_writer_does_not_leak_unknown_fields(self, scene_tree):
        scene, _scenes_root, scene_dir = scene_tree
        meta_path, _link = _write_scene(scene_dir, self._metadata_with_unknowns())
        candidates = load_candidates(meta_path)

        meta_hash = hash_scene_metadata(meta_path)
        writer_root = scene_dir / "dataset"
        with StraferLeRobotWriter(
            root=writer_root,
            repo_id=f"strafer-test/{scene}",
            fps=8,
            capture_git_sha="contract",
            scene_metadata_hash=meta_hash,
            capture_policy_cam=False,
        ) as writer:
            _drive_one_episode(writer, scene, candidates[0], meta_hash=meta_hash)

        episodes = read_strafer_episodes(writer_root)
        assert len(episodes) == 1
        ep = episodes[0]
        # Unknown field NAMES never become parquet columns.
        for unknown in ("descriptors", "future_field", "future_top_level_field"):
            assert unknown not in ep
        # The genuinely-unknown VALUES never appear anywhere in the record.
        blob = json.dumps(ep)
        for value in ("unexpected", "anything", "BlownGlass"):
            assert value not in blob
        # The known mapping still landed.
        assert ep["target_label"] == "bottle"
        assert ep["target_object_id"] == "10"


# ---------------------------------------------------------------------------
# Ceiling-light prim pattern (the parameterized Infinigen knob)
# ---------------------------------------------------------------------------


def _load_postprocess_module():
    """Load postprocess_scene_usd.py — top-level imports do not need pxr."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "postprocess_scene_usd.py"
    spec = importlib.util.spec_from_file_location("postprocess_scene_usd_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCeilingLightPattern:
    def test_default_keeps_infinigen_behaviour(self):
        pp = _load_postprocess_module()
        assert pp._DEFAULT_CEILING_LIGHT_PRIM_PATTERN == (
            r"^CeilingLightFactory_\d+__spawn_asset_\d+_$"
        )
        pat = pp._compile_ceiling_light_pattern(pp._DEFAULT_CEILING_LIGHT_PRIM_PATTERN)
        assert pat.match("CeilingLightFactory_123__spawn_asset_4_")

    def test_override_pattern_matches_foreign_names_default_does_not(self):
        pp = _load_postprocess_module()
        default = pp._compile_ceiling_light_pattern(
            pp._DEFAULT_CEILING_LIGHT_PRIM_PATTERN,
        )
        override = pp._compile_ceiling_light_pattern(r"^MyCeilingLight_\d+$")
        assert override.match("MyCeilingLight_7")
        # The override is necessary precisely because the default misses it.
        assert default.match("MyCeilingLight_7") is None
        # ...and the override does not accidentally match Infinigen names.
        assert override.match("CeilingLightFactory_123__spawn_asset_4_") is None

    def test_cli_help_lists_the_flag(self, capsys):
        pp = _load_postprocess_module()
        with pytest.raises(SystemExit):
            pp.main(["--usdc", "/nonexistent.usdc", "--help"])
        out = capsys.readouterr().out
        assert "--ceiling-light-prim-pattern" in out
