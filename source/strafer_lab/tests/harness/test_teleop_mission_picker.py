"""Tests for the scene-metadata mission-target picker.

The picker is the operator-facing surface for ``--mission-source
scene-metadata`` — it reads ``scene_metadata.json``, applies filters,
sorts the candidates, and presents them via a numeric console prompt.
The console-IO portion is exercised here through ``io.StringIO`` so we
never touch real stdin.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from strafer_lab.tools.teleop_mission_picker import (
    MissionCandidate,
    load_candidates,
    load_candidates_from_data,
    prompt_for_target,
    select_by_index,
)


def _synthetic_metadata() -> dict:
    return {
        "rooms": [
            {"room_type": "Kitchen", "footprint_xy": [[0, 0], [3, 0], [3, 3], [0, 3]]},
            {"room_type": "Bedroom", "footprint_xy": [[3, 0], [6, 0], [6, 3], [3, 3]]},
        ],
        "objects": [
            {"instance_id": 1, "label": "Chair", "position_3d": [1.0, 1.0, 0.0], "room_idx": 0,
             "prim_path": "/World/Chair_1"},
            {"instance_id": 2, "label": "Chair", "position_3d": [4.5, 2.0, 0.0], "room_idx": 1},
            {"instance_id": 3, "label": "wall", "position_3d": [0, 0, 0], "room_idx": 0},
            {"instance_id": 4, "label": "Bed",   "position_3d": [5.0, 1.0, 0.0], "room_idx": 1},
            {"instance_id": 5, "label": "floor", "position_3d": [0, 0, 0]},
            {"instance_id": 6, "label": "Table", "position_3d": [2.0, 2.0, 0.0], "room_idx": 0},
        ],
    }


class TestLoadCandidatesFromData:
    def test_drops_default_blocked_labels(self):
        cands = load_candidates_from_data(_synthetic_metadata())
        labels = [c.label.lower() for c in cands]
        assert "wall" not in labels
        assert "floor" not in labels
        assert "ceiling" not in labels

    def test_sorts_by_label_then_instance(self):
        cands = load_candidates_from_data(_synthetic_metadata())
        # Bed (1) → Chair (2) → Table (1) in alphabetical order
        labels_in_order = [c.label for c in cands]
        assert labels_in_order == ["Bed", "Chair", "Chair", "Table"]
        chairs = [c for c in cands if c.label == "Chair"]
        assert chairs[0].instance_id < chairs[1].instance_id

    def test_indices_are_zero_based_and_dense(self):
        cands = load_candidates_from_data(_synthetic_metadata())
        assert [c.index for c in cands] == list(range(len(cands)))

    def test_mission_text_is_canonical(self):
        cands = load_candidates_from_data(_synthetic_metadata())
        chair = next(c for c in cands if c.label == "Chair")
        assert chair.mission_text == "go to the chair"

    def test_room_type_resolution(self):
        cands = load_candidates_from_data(_synthetic_metadata())
        bed = next(c for c in cands if c.label == "Bed")
        assert bed.target_room_type == "Bedroom"
        # Table is in room 0 → Kitchen
        table = next(c for c in cands if c.label == "Table")
        assert table.target_room_type == "Kitchen"

    def test_allowed_labels_filter(self):
        cands = load_candidates_from_data(
            _synthetic_metadata(),
            allowed_labels=["Chair"],
        )
        assert all(c.label.lower() == "chair" for c in cands)
        assert len(cands) == 2

    def test_blocked_labels_override_default(self):
        cands = load_candidates_from_data(
            _synthetic_metadata(),
            blocked_labels=["Chair"],
        )
        # Without the default block list, wall/floor reappear — that's
        # the intended escape hatch for niche debugging captures.
        labels = [c.label.lower() for c in cands]
        assert "chair" not in labels
        assert "wall" in labels

    def test_object_with_short_position_is_skipped(self):
        data = {
            "objects": [
                {"instance_id": 1, "label": "Chair", "position_3d": [1.0, 2.0]},
                {"instance_id": 2, "label": "Bed",   "position_3d": [1.0, 2.0, 3.0]},
            ],
        }
        cands = load_candidates_from_data(data)
        labels = [c.label for c in cands]
        assert labels == ["Bed"]

    def test_object_with_empty_label_is_skipped(self):
        data = {
            "objects": [
                {"instance_id": 1, "label": "",     "position_3d": [0, 0, 0]},
                {"instance_id": 2, "label": "Lamp", "position_3d": [0, 0, 0]},
            ],
        }
        cands = load_candidates_from_data(data)
        assert [c.label for c in cands] == ["Lamp"]


class TestDedupByInstance:
    def test_default_collapses_subprims_to_one(self):
        # Infinigen authors multi-part objects (e.g. a cabinet has top,
        # body, and door sub-prims) as multiple rows sharing
        # (label, instance_id). Picker should show one entry per
        # logical object by default.
        data = {
            "objects": [
                {"instance_id": 70173, "label": "cabinet", "position_3d": [9.10, 5.49, 0.88]},
                {"instance_id": 70173, "label": "cabinet", "position_3d": [8.92, 5.49, 0.92]},
                {"instance_id": 70173, "label": "cabinet", "position_3d": [9.10, 5.49, 0.92]},
                {"instance_id": 999,   "label": "chair",   "position_3d": [1.0, 1.0, 0.5]},
            ],
        }
        cands = load_candidates_from_data(data)
        cabinets = [c for c in cands if c.label == "cabinet"]
        assert len(cabinets) == 1

    def test_dedup_keeps_median_z_subprim(self):
        # Three rows with Z = 0.88, 0.92, 0.92 → median pick is the
        # middle of the sorted list (0.92, second occurrence).
        data = {
            "objects": [
                {"instance_id": 70173, "label": "cabinet", "position_3d": [9.10, 5.49, 0.88]},
                {"instance_id": 70173, "label": "cabinet", "position_3d": [8.92, 5.49, 0.92]},
                {"instance_id": 70173, "label": "cabinet", "position_3d": [9.10, 5.49, 0.92]},
            ],
        }
        cands = load_candidates_from_data(data)
        assert len(cands) == 1
        assert cands[0].target_position_3d[2] == pytest.approx(0.92)

    def test_dedup_off_preserves_all_subprims(self):
        data = {
            "objects": [
                {"instance_id": 70173, "label": "cabinet", "position_3d": [9.10, 5.49, 0.88]},
                {"instance_id": 70173, "label": "cabinet", "position_3d": [8.92, 5.49, 0.92]},
            ],
        }
        cands = load_candidates_from_data(data, dedup_by_instance=False)
        assert len(cands) == 2

    def test_dedup_does_not_collapse_distinct_instances(self):
        # Two cabinets with DIFFERENT instance_ids must stay as two rows
        # even though they share a label.
        data = {
            "objects": [
                {"instance_id": 1, "label": "cabinet", "position_3d": [1.0, 1.0, 1.0]},
                {"instance_id": 2, "label": "cabinet", "position_3d": [5.0, 5.0, 1.0]},
            ],
        }
        cands = load_candidates_from_data(data)
        assert len(cands) == 2
        assert {c.instance_id for c in cands} == {1, 2}


class TestSelectByIndex:
    @pytest.fixture
    def cands(self):
        return load_candidates_from_data(_synthetic_metadata())

    def test_picks_in_range(self, cands):
        result = select_by_index(cands, "1")
        assert result is not None
        assert result.index == 1

    def test_handles_whitespace_and_newline(self, cands):
        result = select_by_index(cands, "  2 \n")
        assert result is not None
        assert result.index == 2

    @pytest.mark.parametrize("bad", ["", " ", "abc", "10", "-1", "1.5"])
    def test_invalid_returns_none(self, cands, bad):
        assert select_by_index(cands, bad) is None


class TestPromptForTarget:
    def test_returns_choice_on_valid_input(self):
        cands = load_candidates_from_data(_synthetic_metadata())
        stdin = io.StringIO("0\n")
        stdout = io.StringIO()
        choice = prompt_for_target(cands, stream_in=stdin, stream_out=stdout)
        assert choice is not None
        assert choice.index == 0

    def test_reprompts_on_invalid_then_accepts(self):
        cands = load_candidates_from_data(_synthetic_metadata())
        stdin = io.StringIO("abc\n99\n2\n")
        stdout = io.StringIO()
        choice = prompt_for_target(cands, stream_in=stdin, stream_out=stdout)
        assert choice is not None and choice.index == 2
        # Re-prompt banner should fire twice (after abc and after 99).
        assert stdout.getvalue().count("invalid index") == 2

    def test_eof_returns_none(self):
        cands = load_candidates_from_data(_synthetic_metadata())
        stdin = io.StringIO("")  # immediate EOF
        stdout = io.StringIO()
        choice = prompt_for_target(cands, stream_in=stdin, stream_out=stdout)
        assert choice is None

    def test_empty_candidates_returns_none(self):
        stdin = io.StringIO("0\n")
        stdout = io.StringIO()
        choice = prompt_for_target([], stream_in=stdin, stream_out=stdout)
        assert choice is None
        assert "no targets" in stdout.getvalue()


class TestLoadCandidatesFromFile:
    def test_reads_and_filters_from_disk(self, tmp_path: Path):
        scene_meta = tmp_path / "scene_metadata.json"
        scene_meta.write_text(json.dumps(_synthetic_metadata()))
        cands = load_candidates(scene_meta)
        assert [c.label for c in cands] == ["Bed", "Chair", "Chair", "Table"]

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_candidates(tmp_path / "missing.json")
