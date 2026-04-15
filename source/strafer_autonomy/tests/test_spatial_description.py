"""Tests for ``strafer_lab.tools.spatial_description`` and the
description pipeline Stage-2/Stage-3 helpers in
``source/strafer_lab/scripts/generate_descriptions.py``.

These tests run in the strafer_autonomy test suite so the conftest's
``strafer_lab`` namespace stub applies.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from strafer_lab.tools.spatial_description import (
    SpatialDescriptionBuilder,
    classify_bearing,
    classify_region,
    quat_to_yaw,
)


# Load generate_descriptions.py as a module for the unit tests. We do this
# via importlib.util so the script's top-level imports (shapely, etc.)
# still resolve, but we don't require it to be installed.
def _load_generate_descriptions():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "source" / "strafer_lab" / "scripts" / "generate_descriptions.py"
    spec = importlib.util.spec_from_file_location("generate_descriptions", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_descriptions"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def gen_mod():
    return _load_generate_descriptions()


# ---------------------------------------------------------------------------
# Quaternion / bearing / region helpers
# ---------------------------------------------------------------------------


class TestGeometryHelpers:
    def test_quat_to_yaw_identity(self):
        assert quat_to_yaw([0.0, 0.0, 0.0, 1.0]) == pytest.approx(0.0)

    def test_quat_to_yaw_90_ccw(self):
        # Rotation of +90° about Z.
        q = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        assert quat_to_yaw(q) == pytest.approx(math.pi / 2, abs=1e-6)

    def test_quat_to_yaw_180(self):
        q = [0.0, 0.0, 1.0, 0.0]
        # Either pi or -pi is acceptable.
        assert abs(abs(quat_to_yaw(q)) - math.pi) < 1e-6

    def test_classify_region(self):
        assert classify_region(0.5) == "near"
        assert classify_region(1.5) == "midway"
        assert classify_region(2.0) == "midway"
        assert classify_region(4.0) == "far"
        assert classify_region(10.0) == "far"

    def test_classify_bearing_ahead(self):
        assert classify_bearing(0.0) == "ahead"

    def test_classify_bearing_left(self):
        assert classify_bearing(math.pi / 2) == "left"

    def test_classify_bearing_right(self):
        assert classify_bearing(-math.pi / 2) == "right"

    def test_classify_bearing_behind(self):
        assert classify_bearing(math.pi) == "behind"


# ---------------------------------------------------------------------------
# SpatialDescriptionBuilder
# ---------------------------------------------------------------------------


@pytest.fixture()
def kitchen_metadata() -> dict:
    return {
        "rooms": [
            {
                "room_type": "Kitchen",
                "footprint_xy": [[0.0, 0.0], [6.0, 0.0], [6.0, 5.0], [0.0, 5.0]],
                "area_m2": 30.0,
                "story": 0,
            }
        ],
        "objects": [
            {
                "instance_id": 1,
                "label": "table",
                "semantic_tags": ["Furniture", "Table"],
                "position_3d": [3.0, 2.0, 0.0],
                "bbox_3d": {"min": [2.5, 1.5, 0.0], "max": [3.5, 2.5, 0.8]},
                "room_idx": 0,
                "relations": [{"type": "StableAgainst", "target": "wall_1"}],
                "materials": ["wood_oak"],
            },
            {
                "instance_id": 2,
                "label": "chair",
                "semantic_tags": ["Furniture", "Seating", "Chair"],
                "position_3d": [5.0, 2.0, 0.0],
                "bbox_3d": {"min": [4.8, 1.8, 0.0], "max": [5.2, 2.2, 0.9]},
                "room_idx": 0,
                "materials": ["wood_oak"],
            },
        ],
        "room_adjacency": [],
    }


class TestSpatialDescriptionBuilder:
    def test_robot_room(self, kitchen_metadata):
        builder = SpatialDescriptionBuilder(kitchen_metadata)
        facts = builder.build(
            {
                "robot_pos": [1.0, 1.0, 0.25],
                "robot_quat": [0.0, 0.0, 0.0, 1.0],
                "bboxes": [],
            }
        )
        assert facts["robot_room_type"] == "Kitchen"
        assert facts["visible_objects"] == []

    def test_distance_and_bearing_ahead(self, kitchen_metadata):
        builder = SpatialDescriptionBuilder(kitchen_metadata)
        # Robot at (1, 2), facing +x. Table at (3, 2) → 2 m straight ahead.
        facts = builder.build(
            {
                "robot_pos": [1.0, 2.0, 0.25],
                "robot_quat": [0.0, 0.0, 0.0, 1.0],
                "bboxes": [{"instance_id": 1}],
            }
        )
        assert len(facts["visible_objects"]) == 1
        obj = facts["visible_objects"][0]
        assert obj["label"] == "table"
        assert obj["distance_m"] == pytest.approx(2.0, abs=0.01)
        assert obj["bearing"] == "ahead"
        assert obj["region"] == "midway"
        assert obj["room_type"] == "Kitchen"
        assert "wood_oak" in obj["materials"]

    def test_sorted_by_distance(self, kitchen_metadata):
        builder = SpatialDescriptionBuilder(kitchen_metadata)
        facts = builder.build(
            {
                "robot_pos": [1.0, 2.0, 0.25],
                "robot_quat": [0.0, 0.0, 0.0, 1.0],
                "bboxes": [{"instance_id": 2}, {"instance_id": 1}],
            }
        )
        labels = [o["label"] for o in facts["visible_objects"]]
        assert labels == ["table", "chair"]  # table is closer

    def test_bbox_wired_through(self, kitchen_metadata):
        builder = SpatialDescriptionBuilder(kitchen_metadata)
        facts = builder.build(
            {
                "robot_pos": [1.0, 2.0, 0.25],
                "robot_quat": [0.0, 0.0, 0.0, 1.0],
                "bboxes": [{"instance_id": 1, "bbox_2d": [100, 200, 400, 500]}],
            }
        )
        assert facts["visible_objects"][0]["bbox_2d"] == [100, 200, 400, 500]

    def test_resolve_by_label(self, kitchen_metadata):
        builder = SpatialDescriptionBuilder(kitchen_metadata)
        facts = builder.build(
            {
                "robot_pos": [0.0, 0.0, 0.25],
                "robot_quat": [0.0, 0.0, 0.0, 1.0],
                "bboxes": [{"label": "table"}],
            }
        )
        assert len(facts["visible_objects"]) == 1
        assert facts["visible_objects"][0]["label"] == "table"

    def test_relations_serialized(self, kitchen_metadata):
        builder = SpatialDescriptionBuilder(kitchen_metadata)
        facts = builder.build(
            {
                "robot_pos": [1.0, 2.0, 0.25],
                "robot_quat": [0.0, 0.0, 0.0, 1.0],
                "bboxes": [{"instance_id": 1}],
            }
        )
        obj = facts["visible_objects"][0]
        assert "StableAgainst wall_1" in obj["relations"]


# ---------------------------------------------------------------------------
# generate_descriptions helper tests
# ---------------------------------------------------------------------------


class TestParseDescriptions:
    def test_parse_three_levels(self, gen_mod):
        raw = (
            "BRIEF: a kitchen with a table\n"
            "MEDIUM: wooden kitchen table with two chairs and a lamp\n"
            "DETAILED: a bright kitchen, wooden table in the center with a cutting board\n"
        )
        descs = gen_mod.parse_descriptions(raw)
        assert [d.level for d in descs] == ["brief", "medium", "detailed"]
        assert descs[0].text.startswith("a kitchen")

    def test_parse_tolerant_to_case(self, gen_mod):
        raw = "brief : short\nmedium: middle\ndetailed: long\n"
        descs = gen_mod.parse_descriptions(raw)
        assert len(descs) == 3

    def test_parse_falls_back_to_single_medium(self, gen_mod):
        raw = "Just a blob of prose without any prefix."
        descs = gen_mod.parse_descriptions(raw)
        assert len(descs) == 1
        assert descs[0].level == "medium"

    def test_parse_empty_returns_empty(self, gen_mod):
        assert gen_mod.parse_descriptions("") == []
        assert gen_mod.parse_descriptions("   \n\n") == []


class TestValidateDescription:
    def test_valid_when_labels_match(self, gen_mod):
        valid, offending = gen_mod.validate_description(
            "a wooden table with a chair", {"table", "chair"},
        )
        assert valid
        assert offending == []

    def test_rejects_missing_noun(self, gen_mod):
        # Scene has no couch, description mentions one → reject.
        valid, offending = gen_mod.validate_description(
            "a cozy couch next to the table", {"table", "chair"},
        )
        assert not valid
        assert "couch" in offending

    def test_tolerates_uncatalogued_words(self, gen_mod):
        # "wall" and "lighting" aren't in the common-noun catalog, so the
        # validator should not flag them even though they're not in
        # scene_labels.
        valid, offending = gen_mod.validate_description(
            "warm lighting against a cream wall", {"table"},
        )
        assert valid
        assert offending == []

    def test_plurals_tolerated(self, gen_mod):
        valid, _ = gen_mod.validate_description(
            "two chairs and a table", {"table", "chair"},
        )
        assert valid


class TestProcessFrame:
    def _write_metadata(self, root, scene_name, metadata):
        scene_dir = root / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "scene_metadata.json").write_text(json.dumps(metadata))
        return scene_dir

    def test_process_frame_validates(self, tmp_path, gen_mod, kitchen_metadata):
        self._write_metadata(tmp_path / "meta", "scene_01", kitchen_metadata)
        episode_dir = tmp_path / "perception" / "ep_0001"
        episode_dir.mkdir(parents=True)
        image_path = episode_dir / "frame_000.jpg"
        image_path.write_bytes(b"fake-jpeg")

        record = {
            "frame_id": "f1",
            "image_path": "frame_000.jpg",
            "scene_name": "scene_01",
            "scene_type": "infinigen",
            "robot_pos": [1.0, 2.0, 0.25],
            "robot_quat": [0.0, 0.0, 0.0, 1.0],
            "bboxes": [{"instance_id": 1, "bbox_2d": [10, 20, 100, 200]}],
        }

        def mock_vlm(prompt, image):
            return (
                "BRIEF: a wooden table ahead\n"
                "MEDIUM: a wooden table directly ahead with a chair to the right\n"
                "DETAILED: a bright kitchen with a wooden table ahead and a chair nearby\n"
            )

        builders: dict = {}
        labels: dict = {}
        stats = gen_mod.BatchStats()

        out = gen_mod.process_frame(
            record=record,
            episode_dir=episode_dir,
            scene_metadata_dir=tmp_path / "meta",
            builders=builders,
            label_sets=labels,
            vlm_runner=mock_vlm,
            stats=stats,
        )
        assert out is not None
        assert out["frame_id"] == "f1"
        assert len(out["descriptions"]) == 3
        assert stats.validated_descriptions == 3
        assert stats.rejected_descriptions == 0

    def test_process_frame_excludes_procroom(self, tmp_path, gen_mod, kitchen_metadata):
        self._write_metadata(tmp_path / "meta", "scene_01", kitchen_metadata)
        episode_dir = tmp_path / "perception" / "ep_0001"
        episode_dir.mkdir(parents=True)
        record = {
            "frame_id": "f1",
            "image_path": "frame_000.jpg",
            "scene_name": "scene_01",
            "scene_type": "procroom",
            "robot_pos": [1.0, 2.0, 0.25],
            "robot_quat": [0.0, 0.0, 0.0, 1.0],
            "bboxes": [],
        }
        out = gen_mod.process_frame(
            record=record,
            episode_dir=episode_dir,
            scene_metadata_dir=tmp_path / "meta",
            builders={},
            label_sets={},
            vlm_runner=lambda prompt, image: "",
            stats=gen_mod.BatchStats(),
        )
        assert out is None

    def test_process_frame_rejects_hallucinations(
        self, tmp_path, gen_mod, kitchen_metadata,
    ):
        self._write_metadata(tmp_path / "meta", "scene_01", kitchen_metadata)
        episode_dir = tmp_path / "perception" / "ep_0001"
        episode_dir.mkdir(parents=True)
        image_path = episode_dir / "frame_000.jpg"
        image_path.write_bytes(b"fake-jpeg")

        record = {
            "frame_id": "f1",
            "image_path": "frame_000.jpg",
            "scene_name": "scene_01",
            "scene_type": "infinigen",
            "robot_pos": [1.0, 2.0, 0.25],
            "robot_quat": [0.0, 0.0, 0.0, 1.0],
            "bboxes": [{"instance_id": 1}],
        }
        stats = gen_mod.BatchStats()
        out = gen_mod.process_frame(
            record=record,
            episode_dir=episode_dir,
            scene_metadata_dir=tmp_path / "meta",
            builders={},
            label_sets={},
            vlm_runner=lambda prompt, image: (
                "BRIEF: a chair and a sofa\n"  # sofa is not in scene → reject
                "MEDIUM: a wooden table ahead\n"
                "DETAILED: a warm kitchen with the table ahead and a chair nearby\n"
            ),
            stats=stats,
        )
        assert out is not None
        levels = {d["level"] for d in out["descriptions"]}
        assert "medium" in levels
        assert "detailed" in levels
        # The brief one mentioning 'sofa' should be rejected.
        assert "brief" not in levels
        assert stats.rejected_descriptions >= 1
