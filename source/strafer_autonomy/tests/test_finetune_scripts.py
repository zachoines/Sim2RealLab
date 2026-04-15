"""Tests for the CLIP fine-tune CSV loader and the VLM data-prep script.

These tests exercise the data-plumbing helpers — they do NOT launch
training or load real models. Heavy imports (torch, open_clip) inside
``finetune_clip.train`` are deliberately avoided here; the unit tests
load the script via ``importlib`` and only touch the pure-Python
helpers.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script(module_name: str, rel_path: str):
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "source" / "strafer_lab" / "scripts" / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def finetune_clip_mod():
    return _load_script("finetune_clip", "finetune_clip.py")


@pytest.fixture(scope="module")
def prepare_vlm_mod():
    return _load_script("prepare_vlm_finetune_data", "prepare_vlm_finetune_data.py")


# ---------------------------------------------------------------------------
# finetune_clip helpers
# ---------------------------------------------------------------------------


class TestCLIPSampleLoader:
    def _write_csv(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "description"])
            for row in rows:
                writer.writerow(row)

    def test_skips_missing_images(self, finetune_clip_mod, tmp_path):
        csv_path = tmp_path / "clip.csv"
        image_root = tmp_path / "images"
        image_root.mkdir()
        (image_root / "ep_0001").mkdir()
        present_img = image_root / "ep_0001" / "frame_000.jpg"
        present_img.write_bytes(b"fake")
        self._write_csv(
            csv_path,
            [
                ["ep_0001/frame_000.jpg", "a kitchen"],
                ["ep_0001/frame_missing.jpg", "missing file"],
                ["ep_0001/frame_000.jpg", "a wooden table"],
            ],
        )
        samples = finetune_clip_mod.load_samples(csv_path, image_root=image_root)
        assert len(samples) == 2
        assert all(s.image_path == present_img for s in samples)

    def test_parse_args_round_trip(self, finetune_clip_mod, tmp_path):
        cfg = finetune_clip_mod.parse_args(
            [
                "--data", str(tmp_path / "clip.csv"),
                "--image-root", str(tmp_path),
                "--output", str(tmp_path / "out"),
                "--epochs", "3",
                "--batch-size", "8",
                "--lr", "2e-5",
            ]
        )
        assert cfg.epochs == 3
        assert cfg.batch_size == 8
        assert cfg.lr == pytest.approx(2e-5)
        assert cfg.export_onnx is True

    def test_parse_args_no_export_onnx(self, finetune_clip_mod, tmp_path):
        cfg = finetune_clip_mod.parse_args(
            [
                "--data", str(tmp_path / "clip.csv"),
                "--image-root", str(tmp_path),
                "--output", str(tmp_path / "out"),
                "--no-export-onnx",
            ]
        )
        assert cfg.export_onnx is False


# ---------------------------------------------------------------------------
# prepare_vlm_finetune_data
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _populate_trees(tmp_path: Path) -> tuple[Path, Path, Path]:
    perception_root = tmp_path / "perception"
    descriptions_root = tmp_path / "descriptions"
    metadata_root = tmp_path / "metadata"

    _write_jsonl(
        perception_root / "ep_0001" / "frames.jsonl",
        [
            {
                "frame_id": "f1",
                "image_path": "frame_000.jpg",
                "scene_name": "scene_01",
                "scene_type": "infinigen",
                "image_width": 640,
                "image_height": 360,
                "bboxes": [
                    {"instance_id": 1, "label": "table", "bbox_2d": [100, 100, 300, 300]},
                    {"instance_id": 2, "label": "chair", "bbox_2d": [350, 150, 450, 320]},
                    {"instance_id": 3, "label": "lamp", "bbox_2d": [500, 50, 580, 220]},
                ],
            },
            {
                "frame_id": "f2",
                "image_path": "frame_001.jpg",
                "scene_name": "procroom_01",
                "scene_type": "procroom",
                "image_width": 640,
                "image_height": 360,
                "bboxes": [{"label": "cyan_box", "bbox_2d": [0, 0, 10, 10]}],
            },
        ],
    )

    _write_jsonl(
        descriptions_root / "ep_0001" / "descriptions.jsonl",
        [
            {
                "frame_id": "f1",
                "image_path": "frame_000.jpg",
                "scene_name": "scene_01",
                "scene_type": "infinigen",
                "descriptions": [
                    {"level": "brief", "text": "kitchen with table"},
                    {"level": "medium", "text": "a wooden table and chair"},
                ],
            }
        ],
    )

    scene_dir = metadata_root / "scene_01"
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
                    {"label": "door", "instance_id": 5, "position_3d": [0, 0, 0]},
                    {"label": "window", "instance_id": 6, "position_3d": [0, 0, 0]},
                ],
            }
        )
    )
    return perception_root, descriptions_root, metadata_root


class TestPrepareVLMFineTuneData:
    def test_generate_examples_mix(self, prepare_vlm_mod, tmp_path):
        perception_root, descriptions_root, metadata_root = _populate_trees(tmp_path)
        config = prepare_vlm_mod.VLMDataPrepConfig(
            perception_root=perception_root,
            descriptions_root=descriptions_root,
            scene_metadata_dir=metadata_root,
            output_path=tmp_path / "vlm.jsonl",
            negative_ratio=3,
            multi_object_fraction=0.20,
            description_fraction=0.10,
            seed=0,
        )
        examples, stats = prepare_vlm_mod.generate_examples(config)
        assert stats.single_object_examples == 3  # table, chair, lamp
        assert stats.negative_examples == 3  # 3 positives × ratio 1
        assert stats.multi_object_examples == 1  # 3 visible, in [2..10]
        assert stats.procroom_skipped == 1
        assert stats.description_examples >= 0

        # Confirm single-object examples use Qwen coordinates
        single = next(
            e for e in examples
            if "Locate the table" in e["conversations"][0]["content"]
        )
        answer = single["conversations"][1]["content"]
        assert answer.startswith("<ref>table</ref>")
        assert "<box>" in answer

        # Negative examples only talk about labels not in the frame.
        frame_labels = {"table", "chair", "lamp"}
        for ex in examples:
            user = ex["conversations"][0]["content"]
            assistant = ex["conversations"][1]["content"]
            if "not visible" in assistant:
                # Extract the label after "Locate the ".
                label = user.split("Locate the ", 1)[1].split(" in this image", 1)[0]
                assert label not in frame_labels

    def test_multi_object_answer_concatenates(self, prepare_vlm_mod, tmp_path):
        perception_root, descriptions_root, metadata_root = _populate_trees(tmp_path)
        config = prepare_vlm_mod.VLMDataPrepConfig(
            perception_root=perception_root,
            descriptions_root=descriptions_root,
            scene_metadata_dir=metadata_root,
            output_path=tmp_path / "vlm.jsonl",
            seed=0,
        )
        examples, _ = prepare_vlm_mod.generate_examples(config)
        multi_examples = [
            e for e in examples
            if "List all visible objects" in e["conversations"][0]["content"]
        ]
        assert len(multi_examples) == 1
        answer = multi_examples[0]["conversations"][1]["content"]
        assert answer.count("<ref>") == 3
        assert "<ref>table</ref>" in answer
        assert "<ref>chair</ref>" in answer
        assert "<ref>lamp</ref>" in answer

    def test_write_examples_to_disk(self, prepare_vlm_mod, tmp_path):
        perception_root, descriptions_root, metadata_root = _populate_trees(tmp_path)
        config = prepare_vlm_mod.VLMDataPrepConfig(
            perception_root=perception_root,
            descriptions_root=descriptions_root,
            scene_metadata_dir=metadata_root,
            output_path=tmp_path / "vlm.jsonl",
            seed=0,
        )
        prepare_vlm_mod.write_examples(config)
        assert config.output_path.exists()
        stats_path = config.output_path.with_suffix(".stats.json")
        assert stats_path.exists()
        payload = json.loads(stats_path.read_text())
        assert payload["single_object_examples"] == 3
        assert payload["procroom_skipped"] == 1
