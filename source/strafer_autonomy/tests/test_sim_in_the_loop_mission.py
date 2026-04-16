"""Tests for strafer_lab.sim_in_the_loop.mission.

Pure Python — runs in .venv_vlm via the strafer_lab namespace stub.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strafer_lab.sim_in_the_loop.mission import MissionGenerator, MissionSpec


def _scene_metadata(objects: list[dict], rooms: list[dict] | None = None) -> dict:
    return {"rooms": rooms or [], "objects": objects, "room_adjacency": {}}


def _obj(
    instance_id: int,
    label: str,
    *,
    position=(1.0, 2.0, 0.0),
    room_idx: int | None = 0,
    semantic_tags: list[str] | None = None,
    prim_path: str | None = None,
) -> dict:
    return {
        "instance_id": instance_id,
        "label": label,
        "semantic_tags": semantic_tags or [],
        "prim_path": prim_path or f"/World/{label}_{instance_id}",
        "position_3d": list(position),
        "bbox_3d_min": [0.0, 0.0, 0.0],
        "bbox_3d_max": [1.0, 1.0, 1.0],
        "room_idx": room_idx,
        "relations": [],
        "materials": [],
    }


class TestMissionGeneratorEmissions:
    def test_yields_one_mission_per_object(self):
        gen = MissionGenerator(
            scene_name="kitchen_01",
            objects=[_obj(1, "Chair"), _obj(2, "Table")],
        )
        specs = list(gen)
        assert len(specs) == 2

    def test_mission_id_is_deterministic_and_unique(self):
        gen = MissionGenerator(
            scene_name="kitchen_01",
            objects=[_obj(1, "Chair"), _obj(2, "Chair"), _obj(3, "Table")],
        )
        ids = [spec.mission_id for spec in gen]
        assert ids == [
            "kitchen_01__chair__1",
            "kitchen_01__chair__2",
            "kitchen_01__table__3",
        ]

    def test_raw_command_is_natural_language(self):
        gen = MissionGenerator(scene_name="x", objects=[_obj(1, "Chair")])
        spec = next(iter(gen))
        assert spec.raw_command == "go to the chair"

    def test_target_position_propagates(self):
        gen = MissionGenerator(
            scene_name="x", objects=[_obj(1, "Chair", position=(3.0, -1.0, 0.5))],
        )
        spec = next(iter(gen))
        assert spec.target_position_3d == (3.0, -1.0, 0.5)

    def test_semantic_tags_and_prim_path_carry_over(self):
        gen = MissionGenerator(
            scene_name="x",
            objects=[_obj(1, "Chair", semantic_tags=["seating", "wood"], prim_path="/W/C")],
        )
        spec = next(iter(gen))
        assert spec.target_semantic_tags == ("seating", "wood")
        assert spec.target_prim_path == "/W/C"


class TestMissionGeneratorOrdering:
    def test_objects_are_sorted_by_label_then_instance(self):
        gen = MissionGenerator(
            scene_name="x",
            objects=[
                _obj(5, "Table"),
                _obj(1, "Chair"),
                _obj(3, "Chair"),
                _obj(2, "Chair"),
            ],
        )
        labels = [(s.target_label, s.target_instance_id) for s in gen]
        assert labels == [
            ("Chair", 1),
            ("Chair", 2),
            ("Chair", 3),
            ("Table", 5),
        ]


class TestMissionGeneratorFiltering:
    def test_max_missions_caps_emission(self):
        gen = MissionGenerator(
            scene_name="x",
            objects=[_obj(i, "Chair") for i in range(10)],
            max_missions=3,
        )
        assert len(list(gen)) == 3

    def test_blocked_labels_skipped(self):
        gen = MissionGenerator(
            scene_name="x",
            objects=[_obj(1, "Chair"), _obj(2, "Wall"), _obj(3, "Door")],
            blocked_labels=("wall", "door"),
        )
        assert [s.target_label for s in gen] == ["Chair"]

    def test_allowed_labels_filters(self):
        gen = MissionGenerator(
            scene_name="x",
            objects=[_obj(1, "Chair"), _obj(2, "Table"), _obj(3, "Lamp")],
            allowed_labels=("chair", "table"),
        )
        assert sorted(s.target_label for s in gen) == ["Chair", "Table"]

    def test_allowed_labels_is_case_insensitive(self):
        gen = MissionGenerator(
            scene_name="x",
            objects=[_obj(1, "Chair")],
            allowed_labels=("CHAIR",),
        )
        assert len(list(gen)) == 1

    def test_blocked_takes_precedence_over_allowed(self):
        gen = MissionGenerator(
            scene_name="x",
            objects=[_obj(1, "Chair")],
            allowed_labels=("chair",),
            blocked_labels=("chair",),
        )
        assert list(gen) == []

    def test_objects_with_empty_label_skipped(self):
        gen = MissionGenerator(
            scene_name="x",
            objects=[_obj(1, ""), _obj(2, "Chair")],
        )
        assert [s.target_label for s in gen] == ["Chair"]

    def test_objects_with_short_position_skipped(self):
        bad = _obj(1, "Chair")
        bad["position_3d"] = [1.0, 2.0]  # missing Z
        good = _obj(2, "Chair")
        gen = MissionGenerator(scene_name="x", objects=[bad, good])
        assert [s.target_instance_id for s in gen] == [2]


class TestMissionGeneratorFromMetadataPath:
    @pytest.fixture
    def metadata_file(self, tmp_path: Path) -> Path:
        scene_dir = tmp_path / "kitchen_01"
        scene_dir.mkdir()
        path = scene_dir / "scene_metadata.json"
        path.write_text(
            json.dumps(_scene_metadata(objects=[_obj(1, "Chair"), _obj(2, "Table")])),
            encoding="utf-8",
        )
        return path

    def test_loads_from_disk(self, metadata_file):
        gen = MissionGenerator.from_metadata_path(scene_metadata_path=metadata_file)
        specs = list(gen)
        assert len(specs) == 2

    def test_scene_name_defaults_to_parent_dir(self, metadata_file):
        gen = MissionGenerator.from_metadata_path(scene_metadata_path=metadata_file)
        spec = next(iter(gen))
        assert spec.scene_name == "kitchen_01"

    def test_scene_name_override_wins(self, metadata_file):
        gen = MissionGenerator.from_metadata_path(
            scene_metadata_path=metadata_file, scene_name="override",
        )
        spec = next(iter(gen))
        assert spec.scene_name == "override"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MissionGenerator.from_metadata_path(
                scene_metadata_path=tmp_path / "missing.json",
            )

    def test_kwargs_propagate_through(self, metadata_file):
        gen = MissionGenerator.from_metadata_path(
            scene_metadata_path=metadata_file,
            max_missions=1,
            blocked_labels=("chair",),
        )
        specs = list(gen)
        assert len(specs) == 1
        assert specs[0].target_label == "Table"


class TestMissionSpecImmutability:
    def test_frozen_dataclass(self):
        spec = MissionSpec(
            mission_id="a",
            scene_name="b",
            target_label="c",
            target_instance_id=1,
            target_position_3d=(0.0, 0.0, 0.0),
            target_room_idx=None,
            raw_command="go",
        )
        with pytest.raises((AttributeError, TypeError)):
            spec.mission_id = "mutated"  # type: ignore[misc]
