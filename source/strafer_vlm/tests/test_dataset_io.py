"""Unit tests for dataset loading (flat + chat format)."""

from __future__ import annotations

import json

import pytest

from strafer_vlm.training.dataset_io import load_grounding_dataset


# ---------------------------------------------------------------------------
# Flat format
# ---------------------------------------------------------------------------

class TestFlatFormat:
    def test_flat_jsonl(self, tiny_jpeg_path, tmp_path):
        record = {
            "image": tiny_jpeg_path.name,
            "prompt": "chair",
            "found": True,
            "bbox_2d": [100, 200, 500, 800],
            "label": "chair",
        }
        ds_path = tmp_path / "flat.jsonl"
        ds_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        examples = load_grounding_dataset(str(ds_path))
        assert len(examples) == 1
        assert examples[0].target.found is True
        assert examples[0].target.bbox_2d == (100, 200, 500, 800)
        assert examples[0].prompt == "Locate: chair"

    def test_flat_json_list(self, tiny_jpeg_path, tmp_path):
        records = [
            {
                "image": tiny_jpeg_path.name,
                "prompt": "door",
                "found": True,
                "bbox_2d": [10, 20, 90, 80],
            },
            {
                "image": tiny_jpeg_path.name,
                "prompt": "window",
                "found": False,
            },
        ]
        ds_path = tmp_path / "flat.json"
        ds_path.write_text(json.dumps(records), encoding="utf-8")
        examples = load_grounding_dataset(str(ds_path))
        assert len(examples) == 2
        assert examples[0].target.found is True
        assert examples[1].target.found is False

    def test_flat_json_examples_key(self, tiny_jpeg_path, tmp_path):
        data = {
            "examples": [
                {
                    "image": tiny_jpeg_path.name,
                    "prompt": "table",
                    "found": True,
                    "bbox_2d": [50, 50, 200, 200],
                }
            ]
        }
        ds_path = tmp_path / "wrapped.json"
        ds_path.write_text(json.dumps(data), encoding="utf-8")
        examples = load_grounding_dataset(str(ds_path))
        assert len(examples) == 1

    def test_flat_alternate_field_names(self, tiny_jpeg_path, tmp_path):
        record = {
            "image_path": tiny_jpeg_path.name,
            "query": "exit sign",
            "target": {"found": True, "bbox_2d": [10, 10, 100, 100]},
        }
        ds_path = tmp_path / "alt.jsonl"
        ds_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        examples = load_grounding_dataset(str(ds_path))
        assert len(examples) == 1
        assert examples[0].target.found is True

    def test_flat_img_field(self, tiny_jpeg_path, tmp_path):
        record = {
            "img": tiny_jpeg_path.name,
            "text": "fire extinguisher",
            "response": '{"found": true, "bbox_2d": [10, 20, 30, 40]}',
        }
        ds_path = tmp_path / "img_field.jsonl"
        ds_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        examples = load_grounding_dataset(str(ds_path))
        assert len(examples) == 1


# ---------------------------------------------------------------------------
# Chat format
# ---------------------------------------------------------------------------

class TestChatFormat:
    def test_chat_messages(self, tiny_jpeg_path, tmp_path):
        record = {
            "image": tiny_jpeg_path.name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(tiny_jpeg_path)},
                        {"type": "text", "text": "chair"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": '{"found": true, "bbox_2d": [100, 200, 500, 800]}',
                },
            ],
        }
        ds_path = tmp_path / "chat.jsonl"
        ds_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        examples = load_grounding_dataset(str(ds_path))
        assert len(examples) == 1
        assert examples[0].target.found is True

    def test_chat_string_content(self, tiny_jpeg_path, tmp_path):
        record = {
            "image": tiny_jpeg_path.name,
            "messages": [
                {"role": "user", "content": "door"},
                {
                    "role": "assistant",
                    "content": '{"found": false}',
                },
            ],
        }
        ds_path = tmp_path / "chat_str.jsonl"
        ds_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        examples = load_grounding_dataset(str(ds_path))
        assert len(examples) == 1
        assert examples[0].target.found is False


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestDatasetErrors:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_grounding_dataset("does_not_exist.jsonl")

    def test_empty_dataset(self, tmp_path):
        ds_path = tmp_path / "empty.jsonl"
        ds_path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No valid examples"):
            load_grounding_dataset(str(ds_path))

    def test_invalid_json_line(self, tmp_path):
        ds_path = tmp_path / "bad.jsonl"
        ds_path.write_text("{not valid json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_grounding_dataset(str(ds_path))

    def test_missing_image_file(self, tmp_path):
        record = {
            "image": "nonexistent.jpg",
            "prompt": "chair",
            "found": True,
            "bbox_2d": [10, 20, 30, 40],
        }
        ds_path = tmp_path / "bad_img.jsonl"
        ds_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="Image path does not exist"):
            load_grounding_dataset(str(ds_path))

    def test_missing_prompt(self, tiny_jpeg_path, tmp_path):
        record = {
            "image": tiny_jpeg_path.name,
            "found": True,
            "bbox_2d": [10, 20, 30, 40],
        }
        ds_path = tmp_path / "no_prompt.jsonl"
        ds_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Missing prompt"):
            load_grounding_dataset(str(ds_path))

    def test_unsupported_json_root(self, tmp_path):
        ds_path = tmp_path / "bad_root.json"
        ds_path.write_text('"just a string"', encoding="utf-8")
        with pytest.raises(ValueError, match="must be a list"):
            load_grounding_dataset(str(ds_path))
