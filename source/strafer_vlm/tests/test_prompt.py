"""Unit tests for normalize_prompt, serialize_target, and bbox_iou."""

from __future__ import annotations

import json

import pytest

from strafer_vlm.inference.parsing import (
    GroundingTarget,
    bbox_iou,
    normalize_prompt,
    serialize_target,
)


# -----------------------------------------------------------------------
# normalize_prompt
# -----------------------------------------------------------------------


class TestNormalizePrompt:
    def test_plain_text(self):
        assert normalize_prompt("fire extinguisher") == "Locate: fire extinguisher"

    def test_strips_whitespace(self):
        assert normalize_prompt("  door  ") == "Locate: door"

    def test_already_prefixed(self):
        assert normalize_prompt("Locate: door") == "Locate: door"

    def test_case_insensitive_prefix(self):
        assert normalize_prompt("locate: exit sign") == "locate: exit sign"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_prompt("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_prompt("   ")


# -----------------------------------------------------------------------
# serialize_target
# -----------------------------------------------------------------------


class TestSerializeTarget:
    def test_found_with_bbox(self):
        target = GroundingTarget(found=True, bbox_2d=(100, 200, 300, 400), label="chair")
        result = json.loads(serialize_target(target))
        assert result["found"] is True
        assert result["bbox_2d"] == [100, 200, 300, 400]
        assert result["label"] == "chair"

    def test_not_found(self):
        target = GroundingTarget(found=False)
        result = json.loads(serialize_target(target))
        assert result["found"] is False
        assert "bbox_2d" not in result

    def test_confidence_included(self):
        target = GroundingTarget(found=True, bbox_2d=(10, 20, 30, 40), confidence=0.92345)
        result = json.loads(serialize_target(target))
        assert result["confidence"] == 0.9234  # round(0.92345,4) — banker's rounding

    def test_confidence_none_excluded(self):
        target = GroundingTarget(found=True, bbox_2d=(1, 2, 3, 4))
        result = json.loads(serialize_target(target))
        assert "confidence" not in result

    def test_deterministic_output(self):
        target = GroundingTarget(found=True, bbox_2d=(10, 20, 30, 40))
        a = serialize_target(target)
        b = serialize_target(target)
        assert a == b

    def test_round_trip(self):
        target = GroundingTarget(
            found=True, bbox_2d=(100, 200, 300, 400), label="table", confidence=0.88
        )
        serialized = serialize_target(target)
        loaded = json.loads(serialized)
        assert loaded["found"] is True
        assert loaded["bbox_2d"] == [100, 200, 300, 400]
        assert loaded["label"] == "table"
        assert loaded["confidence"] == 0.88


# -----------------------------------------------------------------------
# bbox_iou
# -----------------------------------------------------------------------


class TestBboxIoU:
    def test_perfect_overlap(self):
        iou = bbox_iou((0, 0, 100, 100), (0, 0, 100, 100))
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self):
        iou = bbox_iou((0, 0, 50, 50), (60, 60, 100, 100))
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self):
        iou = bbox_iou((0, 0, 100, 100), (50, 50, 150, 150))
        # Intersection: 50×50 = 2500. Union: 10000+10000-2500 = 17500
        assert iou == pytest.approx(2500 / 17500)

    def test_one_none(self):
        assert bbox_iou(None, (0, 0, 10, 10)) is None
        assert bbox_iou((0, 0, 10, 10), None) is None

    def test_both_none(self):
        assert bbox_iou(None, None) is None

    def test_contained_bbox(self):
        iou = bbox_iou((0, 0, 100, 100), (25, 25, 75, 75))
        inner_area = 50 * 50
        outer_area = 100 * 100
        assert iou == pytest.approx(inner_area / outer_area)

    def test_touching_but_no_overlap(self):
        iou = bbox_iou((0, 0, 50, 50), (50, 0, 100, 50))
        assert iou == pytest.approx(0.0)
