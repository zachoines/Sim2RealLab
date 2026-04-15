"""Tests for ``strafer_lab.tools.dataset_export``."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from strafer_lab.tools.dataset_export import (
    ExportOptions,
    ExportStats,
    export_clip_csv,
    export_vlm_grounding_jsonl,
    format_qwen_grounding_answer,
    pixel_bbox_to_qwen,
    run_export,
)


class TestPixelBboxToQwen:
    def test_full_image(self):
        assert pixel_bbox_to_qwen((0, 0, 640, 360), image_width=640, image_height=360) == (0, 0, 1000, 1000)

    def test_quarter_image(self):
        out = pixel_bbox_to_qwen((0, 0, 320, 180), image_width=640, image_height=360)
        assert out == (0, 0, 500, 500)

    def test_clamped_to_bounds(self):
        out = pixel_bbox_to_qwen((0, 0, 1000, 1000), image_width=640, image_height=360)
        assert out == (0, 0, 1000, 1000)

    def test_degenerate_widened(self):
        out = pixel_bbox_to_qwen((100, 100, 100, 100), image_width=640, image_height=360)
        assert out[0] < out[2]
        assert out[1] < out[3]

    def test_invalid_dims_raise(self):
        with pytest.raises(ValueError):
            pixel_bbox_to_qwen((0, 0, 10, 10), image_width=0, image_height=10)


class TestFormatQwenAnswer:
    def test_format(self):
        assert (
            format_qwen_grounding_answer("table", (100, 200, 400, 700))
            == "<ref>table</ref><box>(100,200),(400,700)</box>"
        )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _make_descriptions_tree(root: Path) -> None:
    _write_jsonl(
        root / "ep_0001" / "descriptions.jsonl",
        [
            {
                "frame_id": "f1",
                "image_path": "frame_000.jpg",
                "scene_name": "scene_01",
                "scene_type": "infinigen",
                "descriptions": [
                    {"level": "brief", "text": "a kitchen"},
                    {"level": "medium", "text": "a wooden kitchen table"},
                ],
            },
            {
                "frame_id": "f2",
                "image_path": "frame_001.jpg",
                "scene_name": "procroom_01",
                "scene_type": "procroom",
                "descriptions": [
                    {"level": "brief", "text": "a cyan box"},
                ],
            },
            {
                "frame_id": "f3",
                "image_path": "frame_002.jpg",
                "scene_name": "scene_01",
                "scene_type": "infinigen",
                "descriptions": [],
            },
        ],
    )
    _write_jsonl(
        root / "ep_0002" / "descriptions.jsonl",
        [
            {
                "frame_id": "f4",
                "image_path": "frame_000.jpg",
                "scene_name": "scene_02",
                "scene_type": "infinigen",
                "descriptions": [
                    {"level": "brief", "text": "a hallway"},
                ],
            },
        ],
    )


def _make_perception_tree(root: Path) -> None:
    _write_jsonl(
        root / "ep_0001" / "frames.jsonl",
        [
            {
                "frame_id": "f1",
                "image_path": "frame_000.jpg",
                "scene_name": "scene_01",
                "scene_type": "infinigen",
                "image_width": 640,
                "image_height": 360,
                "robot_pos": [0, 0, 0],
                "robot_quat": [0, 0, 0, 1],
                "bboxes": [
                    {"instance_id": 1, "label": "table", "bbox_2d": [100, 100, 300, 300]},
                    {"instance_id": 2, "label": "chair", "bbox_2d": [400, 100, 500, 300]},
                ],
            },
            {
                "frame_id": "f2",
                "image_path": "frame_001.jpg",
                "scene_name": "procroom_01",
                "scene_type": "procroom",
                "image_width": 640,
                "image_height": 360,
                "bboxes": [{"label": "cyan_box", "bbox_2d": [10, 10, 20, 20]}],
            },
        ],
    )


def _make_scene_metadata(root: Path) -> None:
    scene_dir = root / "scene_01"
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "scene_metadata.json").write_text(
        json.dumps(
            {
                "rooms": [],
                "objects": [
                    {"label": "table", "instance_id": 1, "position_3d": [0, 0, 0]},
                    {"label": "chair", "instance_id": 2, "position_3d": [0, 0, 0]},
                    {"label": "lamp", "instance_id": 3, "position_3d": [0, 0, 0]},
                    {"label": "plant", "instance_id": 4, "position_3d": [0, 0, 0]},
                ],
            }
        )
    )


class TestExportClipCsv:
    def test_emits_rows_excluding_procroom(self, tmp_path):
        descriptions_root = tmp_path / "descriptions"
        _make_descriptions_tree(descriptions_root)

        output = tmp_path / "clip.csv"
        stats = ExportStats()
        export_clip_csv(
            descriptions_root=descriptions_root,
            output_path=output,
            stats=stats,
        )
        with output.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert stats.clip_rows == 3
        assert stats.procroom_skipped == 1
        assert stats.frames_without_descriptions == 1
        assert all(row["image_path"].startswith(("ep_0001", "ep_0002")) for row in rows)
        assert set(row["image_path"] for row in rows) == {
            "ep_0001/frame_000.jpg",
            "ep_0002/frame_000.jpg",
        }


class TestExportVlmJsonl:
    def test_emits_positives_and_negatives(self, tmp_path):
        perception_root = tmp_path / "perception"
        _make_perception_tree(perception_root)
        metadata_root = tmp_path / "metadata"
        _make_scene_metadata(metadata_root)

        output = tmp_path / "vlm.jsonl"
        stats = ExportStats()
        export_vlm_grounding_jsonl(
            perception_root=perception_root,
            descriptions_root=None,
            output_path=output,
            scene_metadata_dir=metadata_root,
            negative_ratio=3,
            stats=stats,
            seed=0,
        )
        with output.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        positives = [r for r in records if "not visible" not in r["conversations"][1]["content"]]
        negatives = [r for r in records if "not visible" in r["conversations"][1]["content"]]
        assert len(positives) == 2
        assert stats.vlm_positive_rows == 2
        # scene_01 has 4 labels (table, chair, lamp, plant); frame has 2
        # positives (table, chair). candidates = {lamp, plant}, negative_ratio*2 = 6
        # but only 2 candidates → expect 2 negatives.
        assert len(negatives) == 2
        assert stats.vlm_negative_rows == 2
        assert stats.procroom_skipped == 1

        # Positive assistant content uses 0..1000 coordinates.
        pos_answer = positives[0]["conversations"][1]["content"]
        assert "<ref>" in pos_answer
        assert "<box>" in pos_answer


class TestRunExport:
    def test_writes_all_three_files(self, tmp_path):
        perception_root = tmp_path / "perception"
        descriptions_root = tmp_path / "descriptions"
        metadata_root = tmp_path / "metadata"
        output_root = tmp_path / "out"
        _make_perception_tree(perception_root)
        _make_descriptions_tree(descriptions_root)
        _make_scene_metadata(metadata_root)

        options = ExportOptions(
            perception_root=perception_root,
            descriptions_root=descriptions_root,
            output_root=output_root,
            scene_metadata_dir=metadata_root,
            vlm_negative_ratio=3,
        )
        stats = run_export(options)
        assert (output_root / "clip_descriptions.csv").exists()
        assert (output_root / "vlm_grounding.jsonl").exists()
        assert (output_root / "dataset_export_stats.json").exists()
        assert stats.clip_rows == 3
        assert stats.vlm_positive_rows == 2
        assert stats.vlm_negative_rows == 2
