"""Tests for strafer_lab.tools.mission_queue.

Parses a hand-authored fixture matching the documented
``mission_queue.yaml`` row shape (mission_text + paraphrases[] +
planned_path + per-row metadata). Pure Python — no Isaac Sim.
"""

from __future__ import annotations

import textwrap

import pytest

from strafer_lab.tools.mission_queue import (
    MissionQueueError,
    QueueMissionRow,
    load_mission_queue,
    parse_mission_row,
    queue_row_to_mission_spec,
)


FIXTURE_YAML = textwrap.dedent(
    """\
    - mission_id: "0001"
      scene_name: scene_test_seed0
      scene_seed: 0
      start_pose: {x: 0.5, y: 0.5, yaw: 0.0}
      target_label: chair
      target_position_3d: [4.2, 1.8, 0.0]
      target_room: living_room
      start_room: kitchen
      cross_room: true
      mission_text: "Go to the chair by hugging the south wall."
      paraphrases:
        - "Approach the chair while staying close to the south wall."
        - "Drive along the southern edge of the room to reach the chair."
      planned_path:
        - {x: 0.5, y: 0.4}
        - {x: 2.5, y: 0.3}
        - {x: 4.2, y: 1.8}
      generator_metadata:
        llm_model: "Qwen3-4B"
        llm_seed: 42

    - mission_id: 2
      target_label: table
      target_position_3d: [1.0, 1.0, 0.0]
      mission_text: "go to the table"
    """
)


@pytest.fixture()
def queue_path(tmp_path):
    path = tmp_path / "mission_queue.yaml"
    path.write_text(FIXTURE_YAML, encoding="utf-8")
    return path


class TestLoadMissionQueue:
    def test_parses_both_rows(self, queue_path):
        rows = load_mission_queue(queue_path)
        assert len(rows) == 2
        assert all(isinstance(r, QueueMissionRow) for r in rows)

    def test_full_row_fields(self, queue_path):
        row = load_mission_queue(queue_path)[0]
        assert row.mission_id == "0001"
        assert row.scene_name == "scene_test_seed0"
        assert row.scene_seed == 0
        assert row.start_pose == pytest.approx((0.5, 0.5, 0.0))
        assert row.target_label == "chair"
        assert row.target_position_3d == pytest.approx((4.2, 1.8, 0.0))
        assert row.target_room == "living_room"
        assert row.start_room == "kitchen"
        assert row.cross_room is True
        assert row.mission_text == "Go to the chair by hugging the south wall."
        assert len(row.paraphrases) == 2
        assert row.planned_path == ((0.5, 0.4), (2.5, 0.3), (4.2, 1.8))
        assert row.generator_metadata["llm_model"] == "Qwen3-4B"

    def test_minimal_row_defaults(self, queue_path):
        row = load_mission_queue(queue_path)[1]
        assert row.mission_id == "2"
        assert row.scene_name is None
        assert row.start_pose is None
        assert row.paraphrases == ()
        assert row.planned_path == ()
        assert row.generator_metadata == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(MissionQueueError, match="not found"):
            load_mission_queue(tmp_path / "nope.yaml")

    def test_non_list_document_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("mission_id: 1\n", encoding="utf-8")
        with pytest.raises(MissionQueueError, match="must be a list"):
            load_mission_queue(path)

    def test_empty_document_is_empty_queue(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_mission_queue(path) == []


class TestParseMissionRow:
    def test_missing_required_field_raises(self):
        with pytest.raises(MissionQueueError, match="missing required"):
            parse_mission_row({"mission_id": "x", "mission_text": "go"})

    def test_bad_target_position_raises(self):
        with pytest.raises(MissionQueueError, match="target_position_3d"):
            parse_mission_row({
                "mission_id": "x",
                "mission_text": "go",
                "target_label": "chair",
                "target_position_3d": [1.0],
            })

    def test_bad_planned_path_waypoint_raises(self):
        with pytest.raises(MissionQueueError, match="planned_path"):
            parse_mission_row({
                "mission_id": "x",
                "mission_text": "go",
                "target_label": "chair",
                "target_position_3d": [1.0, 2.0, 0.0],
                "planned_path": ["not-a-waypoint"],
            })

    def test_bad_start_pose_raises(self):
        with pytest.raises(MissionQueueError, match="start_pose"):
            parse_mission_row({
                "mission_id": "x",
                "mission_text": "go",
                "target_label": "chair",
                "target_position_3d": [1.0, 2.0, 0.0],
                "start_pose": [0.5, 0.5],
            })

    def test_planned_path_accepts_xy_lists(self):
        row = parse_mission_row({
            "mission_id": "x",
            "mission_text": "go",
            "target_label": "chair",
            "target_position_3d": [1.0, 2.0, 0.0],
            "planned_path": [[0.0, 0.1], [1.0, 1.1]],
        })
        assert row.planned_path == ((0.0, 0.1), (1.0, 1.1))


class TestQueueRowToMissionSpec:
    def test_mission_text_becomes_raw_command(self, queue_path):
        row = load_mission_queue(queue_path)[0]
        spec = queue_row_to_mission_spec(row, scene_name="scene_test_seed0")
        assert spec.raw_command == row.mission_text
        assert spec.mission_id == "scene_test_seed0__queue__0001"
        assert spec.scene_name == "scene_test_seed0"
        assert spec.target_label == "chair"
        assert spec.target_position_3d == pytest.approx((4.2, 1.8, 0.0))
        # Queue rows carry no instance id / room idx.
        assert spec.target_instance_id == -1
        assert spec.target_room_idx is None
