"""Tests for strafer_lab.tools.perception_writer.

Runs in ``.venv_vlm`` via the ``strafer_lab`` namespace stub in
:mod:`conftest`. Exercises the per-episode directory lifecycle, JSONL
record shape, optional depth serialization, incremental episode
numbering, discard semantics, and JSON coercion for numpy scalars —
all without touching Isaac Sim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from strafer_lab.tools.perception_writer import (
    PerceptionFrameWriter,
    PerceptionWriterStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_rgb(width: int = 64, height: int = 48, fill: int = 128) -> np.ndarray:
    """A small uint8 RGB image for writer tests."""
    return np.full((height, width, 3), fill, dtype=np.uint8)


def _make_depth(width: int = 64, height: int = 48, value: float = 1.5) -> np.ndarray:
    return np.full((height, width), value, dtype=np.float32)


@pytest.fixture()
def writer(tmp_path) -> PerceptionFrameWriter:
    return PerceptionFrameWriter(output_root=tmp_path)


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


class TestStats:
    def test_default_values(self):
        s = PerceptionWriterStats()
        assert s.episodes_started == 0
        assert s.episodes_kept == 0
        assert s.episodes_discarded == 0
        assert s.frames_written == 0
        assert s.frames_dropped == 0

    def test_to_dict(self):
        s = PerceptionWriterStats(
            episodes_started=3,
            episodes_kept=2,
            episodes_discarded=1,
            frames_written=150,
            frames_dropped=5,
        )
        assert s.to_dict() == {
            "episodes_started": 3,
            "episodes_kept": 2,
            "episodes_discarded": 1,
            "frames_written": 150,
            "frames_dropped": 5,
        }


# ---------------------------------------------------------------------------
# Writer construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_output_root(self, tmp_path):
        root = tmp_path / "nested" / "perception"
        assert not root.exists()
        PerceptionFrameWriter(output_root=root)
        assert root.is_dir()

    def test_next_episode_index_fresh_dir(self, writer):
        assert writer.next_episode_index == 0

    def test_next_episode_index_resumes_from_existing(self, tmp_path):
        """Incremental collection: if episode_0000 and episode_0002 exist,
        the next index is 3 (gaps are not reused)."""
        (tmp_path / "episode_0000").mkdir()
        (tmp_path / "episode_0002").mkdir()
        w = PerceptionFrameWriter(output_root=tmp_path)
        assert w.next_episode_index == 3

    def test_next_episode_index_ignores_unrelated_dirs(self, tmp_path):
        (tmp_path / "episode_0000").mkdir()
        (tmp_path / "notes.md").touch()
        (tmp_path / "random_dir").mkdir()
        w = PerceptionFrameWriter(output_root=tmp_path)
        assert w.next_episode_index == 1


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------


class TestEpisodeLifecycle:
    def test_begin_creates_episode_dir(self, writer):
        ep_dir = writer.begin_episode()
        assert ep_dir.is_dir()
        assert ep_dir.name == "episode_0000"
        assert writer.current_episode_dir == ep_dir
        assert writer.stats.episodes_started == 1

    def test_cannot_begin_twice(self, writer):
        writer.begin_episode()
        with pytest.raises(RuntimeError, match="already open"):
            writer.begin_episode()

    def test_end_kept_advances_index(self, writer):
        writer.begin_episode()
        writer.end_episode(keep=True)
        assert writer.stats.episodes_kept == 1
        assert writer.next_episode_index == 1
        assert writer.current_episode_dir is None

    def test_end_discarded_wipes_dir_and_reuses_index(self, writer):
        ep_dir = writer.begin_episode()
        # Write a frame so the dir has content to wipe.
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=None,
            scene_name="scene_01",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
        )
        assert ep_dir.exists()
        writer.end_episode(keep=False)
        assert not ep_dir.exists()
        assert writer.stats.episodes_discarded == 1
        assert writer.next_episode_index == 0  # slot reused

    def test_end_without_begin_is_noop(self, writer):
        # Should not raise when called before any begin.
        writer.end_episode(keep=True)
        assert writer.stats.episodes_kept == 0

    def test_close_finalises_in_progress_episode(self, writer):
        writer.begin_episode()
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=None,
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
        )
        writer.close()
        assert writer.stats.episodes_kept == 1


# ---------------------------------------------------------------------------
# Frame IO
# ---------------------------------------------------------------------------


class TestSaveFrame:
    def test_raises_outside_episode(self, writer):
        with pytest.raises(RuntimeError, match="outside of an open episode"):
            writer.save_frame(
                frame_id=0,
                rgb=_make_rgb(),
                depth=None,
                scene_name="s",
                scene_type="infinigen",
                robot_pos=[0, 0, 0],
                robot_quat=[0, 0, 0, 1],
            )

    def test_rgb_jpeg_written_with_frame_index(self, writer):
        ep_dir = writer.begin_episode()
        writer.save_frame(
            frame_id="f0",
            rgb=_make_rgb(),
            depth=None,
            scene_name="scene_01",
            scene_type="infinigen",
            robot_pos=[1.0, 2.0, 0.25],
            robot_quat=[0.0, 0.0, 0.0, 1.0],
        )
        jpeg = ep_dir / "frame_0000.jpg"
        assert jpeg.is_file()
        img = Image.open(jpeg)
        assert img.size == (64, 48)  # (width, height) in PIL

    def test_frames_jsonl_record_shape(self, writer):
        ep_dir = writer.begin_episode()
        writer.save_frame(
            frame_id="f0",
            rgb=_make_rgb(),
            depth=None,
            scene_name="scene_01",
            scene_type="infinigen",
            robot_pos=[1.0, 2.0, 0.25],
            robot_quat=[0.0, 0.0, 0.0, 1.0],
            cam_pos=[1.2, 2.0, 0.5],
            cam_quat=[0.5, -0.5, 0.5, -0.5],
            bboxes=[
                {"label": "table", "bbox_2d": [10, 20, 100, 200], "semantic_id": 1, "occlusion_ratio": 0.0},
            ],
            image_width=640,
            image_height=360,
        )
        jsonl = (ep_dir / "frames.jsonl").read_text().strip().splitlines()
        assert len(jsonl) == 1
        record = json.loads(jsonl[0])
        assert record["frame_id"] == "f0"
        assert record["image_path"] == "frame_0000.jpg"
        assert record["image_width"] == 640
        assert record["image_height"] == 360
        assert record["scene_name"] == "scene_01"
        assert record["scene_type"] == "infinigen"
        assert record["robot_pos"] == [1.0, 2.0, 0.25]
        assert record["robot_quat"] == [0.0, 0.0, 0.0, 1.0]
        assert record["cam_pos"] == [1.2, 2.0, 0.5]
        assert record["cam_quat"] == [0.5, -0.5, 0.5, -0.5]
        assert record["bboxes"] == [
            {"label": "table", "bbox_2d": [10, 20, 100, 200], "semantic_id": 1, "occlusion_ratio": 0.0},
        ]
        assert "depth_path" not in record

    def test_image_dims_default_to_rgb_shape(self, writer):
        writer.begin_episode()
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(width=640, height=360),
            depth=None,
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
        )
        record = json.loads(
            (writer.current_episode_dir / "frames.jsonl").read_text().strip()
        )
        assert record["image_width"] == 640
        assert record["image_height"] == 360

    def test_rgba_input_sliced_to_rgb(self, writer):
        writer.begin_episode()
        rgba = np.full((48, 64, 4), 200, dtype=np.uint8)
        writer.save_frame(
            frame_id=0,
            rgb=rgba,
            depth=None,
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
        )
        jpeg = writer.current_episode_dir / "frame_0000.jpg"
        img = np.array(Image.open(jpeg))
        assert img.shape == (48, 64, 3)

    def test_float_rgb_clamped_and_cast(self, writer):
        """Isaac Sim sometimes returns float RGB; writer should coerce."""
        writer.begin_episode()
        rgb_float = np.full((48, 64, 3), 300.0, dtype=np.float32)  # > 255
        writer.save_frame(
            frame_id=0,
            rgb=rgb_float,
            depth=None,
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
        )
        jpeg = writer.current_episode_dir / "frame_0000.jpg"
        img = np.array(Image.open(jpeg))
        assert img.dtype == np.uint8
        assert (img == 255).all()  # clamped

    def test_rejects_bad_rgb_shape(self, writer):
        writer.begin_episode()
        with pytest.raises(ValueError, match="rgb must be"):
            writer.save_frame(
                frame_id=0,
                rgb=np.zeros((48, 64), dtype=np.uint8),  # missing channel dim
                depth=None,
                scene_name="s",
                scene_type="infinigen",
                robot_pos=[0, 0, 0],
                robot_quat=[0, 0, 0, 1],
            )

    def test_depth_saved_as_npy(self, writer):
        ep_dir = writer.begin_episode()
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=_make_depth(value=2.5),
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
        )
        depth_file = ep_dir / "frame_0000.depth.npy"
        assert depth_file.is_file()
        loaded = np.load(depth_file)
        assert loaded.dtype == np.float32
        assert loaded.shape == (48, 64)
        assert (loaded == 2.5).all()

    def test_depth_recorded_in_jsonl(self, writer):
        writer.begin_episode()
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=_make_depth(),
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
        )
        record = json.loads(
            (writer.current_episode_dir / "frames.jsonl").read_text().strip()
        )
        assert record["depth_path"] == "frame_0000.depth.npy"

    def test_depth_disabled_skips_save(self, tmp_path):
        w = PerceptionFrameWriter(output_root=tmp_path, depth_enabled=False)
        ep_dir = w.begin_episode()
        w.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=_make_depth(),  # supplied but should be ignored
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
        )
        assert not (ep_dir / "frame_0000.depth.npy").exists()
        record = json.loads((ep_dir / "frames.jsonl").read_text().strip())
        assert "depth_path" not in record

    def test_depth_channel_last_axis_squeezed(self, writer):
        """Isaac Sim depth often comes back as (H, W, 1); writer should
        squeeze the trailing singleton before saving."""
        writer.begin_episode()
        depth_3d = np.full((48, 64, 1), 3.14, dtype=np.float32)
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=depth_3d,
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
        )
        loaded = np.load(writer.current_episode_dir / "frame_0000.depth.npy")
        assert loaded.shape == (48, 64)

    def test_rejects_bad_depth_shape(self, writer):
        writer.begin_episode()
        with pytest.raises(ValueError, match="depth must be"):
            writer.save_frame(
                frame_id=0,
                rgb=_make_rgb(),
                depth=np.zeros((10, 20, 3), dtype=np.float32),
                scene_name="s",
                scene_type="infinigen",
                robot_pos=[0, 0, 0],
                robot_quat=[0, 0, 0, 1],
            )

    def test_frame_counter_increments_within_episode(self, writer):
        writer.begin_episode()
        for i in range(3):
            writer.save_frame(
                frame_id=i,
                rgb=_make_rgb(),
                depth=None,
                scene_name="s",
                scene_type="infinigen",
                robot_pos=[0, 0, 0],
                robot_quat=[0, 0, 0, 1],
            )
        assert writer.current_frame_count == 3
        assert writer.stats.frames_written == 3
        jsonl = (writer.current_episode_dir / "frames.jsonl").read_text().splitlines()
        assert len(jsonl) == 3
        paths = [json.loads(line)["image_path"] for line in jsonl]
        assert paths == ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]

    def test_frame_counter_resets_between_episodes(self, writer):
        writer.begin_episode()
        writer.save_frame(
            frame_id=0, rgb=_make_rgb(), depth=None,
            scene_name="s", scene_type="infinigen",
            robot_pos=[0, 0, 0], robot_quat=[0, 0, 0, 1],
        )
        writer.end_episode(keep=True)
        writer.begin_episode()
        writer.save_frame(
            frame_id=0, rgb=_make_rgb(), depth=None,
            scene_name="s", scene_type="infinigen",
            robot_pos=[0, 0, 0], robot_quat=[0, 0, 0, 1],
        )
        assert writer.current_frame_count == 1
        # Second episode directory exists with its own frame_0000.
        assert (writer.output_root / "episode_0001" / "frame_0000.jpg").exists()


# ---------------------------------------------------------------------------
# Numpy / tensor coercion
# ---------------------------------------------------------------------------


class TestCoercion:
    def test_numpy_arrays_for_pose_coerced_to_lists(self, writer):
        writer.begin_episode()
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=None,
            scene_name="s",
            scene_type="infinigen",
            robot_pos=np.array([1.0, 2.0, 0.25], dtype=np.float32),
            robot_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        record = json.loads(
            (writer.current_episode_dir / "frames.jsonl").read_text().strip()
        )
        assert record["robot_pos"] == [1.0, 2.0, pytest.approx(0.25, abs=1e-4)]

    def test_bbox_dicts_with_numpy_scalars_are_jsonable(self, writer):
        """Replicator output contains numpy scalars; the writer must
        coerce them via the _jsonify default so dumps does not crash."""
        writer.begin_episode()
        bbox_with_np = {
            "label": "table",
            "bbox_2d": [np.int32(10), np.int32(20), np.int32(100), np.int32(200)],
            "semantic_id": np.uint32(5),
            "occlusion_ratio": np.float32(0.1),
        }
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=None,
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
            bboxes=[bbox_with_np],
        )
        record = json.loads(
            (writer.current_episode_dir / "frames.jsonl").read_text().strip()
        )
        assert record["bboxes"][0]["label"] == "table"
        assert record["bboxes"][0]["bbox_2d"] == [10, 20, 100, 200]
        assert record["bboxes"][0]["semantic_id"] == 5
        assert record["bboxes"][0]["occlusion_ratio"] == pytest.approx(0.1, abs=1e-5)


# ---------------------------------------------------------------------------
# Extras, drop counter, stats IO
# ---------------------------------------------------------------------------


class TestExtrasAndStats:
    def test_extras_merged_into_record(self, writer):
        writer.begin_episode()
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=None,
            scene_name="s",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
            extras={"gamepad": {"lx": 0.5, "ly": -0.3}, "difficulty": 3},
        )
        record = json.loads(
            (writer.current_episode_dir / "frames.jsonl").read_text().strip()
        )
        assert record["gamepad"] == {"lx": 0.5, "ly": -0.3}
        assert record["difficulty"] == 3

    def test_extras_do_not_override_core_fields(self, writer):
        writer.begin_episode()
        writer.save_frame(
            frame_id=0,
            rgb=_make_rgb(),
            depth=None,
            scene_name="real_scene",
            scene_type="infinigen",
            robot_pos=[0, 0, 0],
            robot_quat=[0, 0, 0, 1],
            extras={"scene_name": "malicious_override"},  # must be ignored
        )
        record = json.loads(
            (writer.current_episode_dir / "frames.jsonl").read_text().strip()
        )
        assert record["scene_name"] == "real_scene"

    def test_drop_frame_increments_counter(self, writer):
        writer.begin_episode()
        writer.drop_frame()
        writer.drop_frame()
        assert writer.stats.frames_dropped == 2
        assert writer.stats.frames_written == 0

    def test_write_stats_creates_json_file(self, writer):
        writer.begin_episode()
        writer.save_frame(
            frame_id=0, rgb=_make_rgb(), depth=None,
            scene_name="s", scene_type="infinigen",
            robot_pos=[0, 0, 0], robot_quat=[0, 0, 0, 1],
        )
        writer.end_episode(keep=True)
        stats_path = writer.write_stats()
        assert stats_path.name == "writer_stats.json"
        loaded = json.loads(stats_path.read_text())
        assert loaded["episodes_kept"] == 1
        assert loaded["frames_written"] == 1
