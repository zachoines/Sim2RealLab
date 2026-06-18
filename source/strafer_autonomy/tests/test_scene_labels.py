"""Tests for ``strafer_lab.tools.scene_labels`` pure-data accessors.

These tests live in ``strafer_autonomy/tests/`` because the sibling
``conftest.py`` installs a lightweight ``strafer_lab`` namespace stub
that lets us import ``strafer_lab.tools.scene_labels`` without pulling
in the Isaac-Lab-dependent ``strafer_lab/__init__.py``.

Since scene metadata moved into the USD ``customData``, loading it needs
``pxr`` (absent in this ``.venv_vlm`` suite). So this suite exercises the
``*_from_data`` accessors against in-memory dicts; the USD round-trip is
covered under ``make test-lab-pure`` in ``test_scene_provider_contract``
and ``test_scene_metadata_reader``.
"""

from __future__ import annotations

import pytest

from strafer_lab.tools.scene_labels import (
    get_objects_in_room,
    get_room_at_position,
    get_scene_label_set_from_data,
    iter_objects,
    iter_rooms,
)


@pytest.fixture()
def sample_metadata() -> dict:
    return {
        "rooms": [
            {
                "room_type": "Kitchen",
                "footprint_xy": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
                "area_m2": 12.0,
                "story": 0,
            },
            {
                "room_type": "Hallway",
                "footprint_xy": [[4.0, 0.0], [6.0, 0.0], [6.0, 3.0], [4.0, 3.0]],
                "area_m2": 6.0,
                "story": 0,
            },
        ],
        "objects": [
            {
                "instance_id": 1,
                "label": "table",
                "semantic_tags": ["Furniture", "Table"],
                "prim_path": "/World/Room/table_001",
                "position_3d": [2.0, 1.5, 0.0],
                "bbox_3d": {"min": [1.5, 1.0, 0.0], "max": [2.5, 2.0, 0.8]},
                "room_idx": 0,
                "relations": [{"type": "StableAgainst", "target": "wall_segment_3"}],
                "materials": ["wood_oak"],
            },
            {
                "instance_id": 2,
                "label": "chair",
                "semantic_tags": ["Furniture", "Seating", "Chair"],
                "position_3d": [2.5, 1.2, 0.0],
                "bbox_3d": {"min": [2.3, 1.0, 0.0], "max": [2.7, 1.4, 0.9]},
                "room_idx": 0,
                "materials": ["wood_oak"],
            },
            {
                "instance_id": 3,
                "label": "plant",
                "semantic_tags": ["Decoration"],
                "position_3d": [5.0, 1.5, 0.0],
                "bbox_3d": {"min": [4.8, 1.3, 0.0], "max": [5.2, 1.7, 0.6]},
                "room_idx": 1,
            },
        ],
        "room_adjacency": [[0, 1], [1, 0]],
    }


class TestIterRooms:
    def test_yields_typed_rooms(self, sample_metadata):
        rooms = list(iter_rooms(sample_metadata))
        assert len(rooms) == 2
        assert rooms[0].room_type == "Kitchen"
        assert rooms[0].area_m2 == pytest.approx(12.0)
        assert rooms[0].footprint_xy == (
            (0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0),
        )
        assert rooms[0].index == 0
        assert rooms[1].room_type == "Hallway"
        assert rooms[1].index == 1


class TestIterObjects:
    def test_yields_typed_objects(self, sample_metadata):
        objs = list(iter_objects(sample_metadata))
        assert [o.label for o in objs] == ["table", "chair", "plant"]
        assert objs[0].position_3d == (2.0, 1.5, 0.0)
        assert objs[0].bbox_3d_min == (1.5, 1.0, 0.0)
        assert objs[0].bbox_3d_max == (2.5, 2.0, 0.8)
        assert objs[0].room_idx == 0
        assert objs[0].relations[0]["target"] == "wall_segment_3"
        assert "wood_oak" in objs[0].materials


class TestGetSceneLabelSetFromData:
    def test_returns_unique_labels(self, sample_metadata):
        assert get_scene_label_set_from_data(sample_metadata) == {"table", "chair", "plant"}

    def test_skips_empty_labels(self):
        metadata = {
            "rooms": [],
            "objects": [
                {"instance_id": 1, "label": "", "position_3d": [0, 0, 0]},
                {"instance_id": 2, "label": "door", "position_3d": [0, 0, 0]},
            ],
        }
        assert get_scene_label_set_from_data(metadata) == {"door"}


class TestGetRoomAtPosition:
    def test_point_inside_kitchen(self, sample_metadata):
        room = get_room_at_position(sample_metadata, (2.0, 1.5))
        assert room is not None
        assert room.room_type == "Kitchen"

    def test_point_inside_hallway(self, sample_metadata):
        room = get_room_at_position(sample_metadata, (5.0, 1.5))
        assert room is not None
        assert room.room_type == "Hallway"

    def test_point_outside_returns_none(self, sample_metadata):
        assert get_room_at_position(sample_metadata, (100.0, 100.0)) is None


class TestGetObjectsInRoom:
    def test_kitchen_has_two_objects(self, sample_metadata):
        kitchen_objs = get_objects_in_room(sample_metadata, 0)
        assert {o.label for o in kitchen_objs} == {"table", "chair"}

    def test_hallway_has_one_object(self, sample_metadata):
        hallway_objs = get_objects_in_room(sample_metadata, 1)
        assert {o.label for o in hallway_objs} == {"plant"}
