"""Tests for strafer_lab.tools.bbox_extractor.

These tests run in ``.venv_vlm`` via the ``strafer_lab`` namespace stub
installed by :mod:`conftest`. They exercise the pure-Python parser and
the ``ReplicatorBboxExtractor`` class via a mock annotator, so no
Isaac Sim / Omniverse dependencies are required.

The parser is tested against synthetic dicts that mirror the exact
``bounding_box_2d_tight`` output shape documented in Isaac Sim 5.1's
``data_visualization_writer.py`` schema comment:

    ('semanticId', '<u4'), ('x_min', '<i4'), ('y_min', '<i4'),
    ('x_max', '<i4'), ('y_max', '<i4'), ('occlusionRatio', '<f4')

Rows are realised as list-of-dict for clarity in test fixtures; the
parser uses the same ``row["field"]`` access pattern numpy structured
arrays expose, so the contract carries over.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from strafer_lab.tools.bbox_extractor import (
    UNKNOWN_LABEL,
    DetectedBbox,
    ReplicatorBboxExtractor,
    parse_bbox_data,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic Replicator outputs that match the real schema
# ---------------------------------------------------------------------------


def _bbox_row(
    *,
    semantic_id: int,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    occlusion: float = 0.0,
) -> dict[str, object]:
    """One row in the shape a numpy structured array would produce."""
    return {
        "semanticId": semantic_id,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "occlusionRatio": occlusion,
    }


def _raw_output(
    rows: list[dict[str, object]],
    id_to_labels: dict[str, object] | None,
) -> dict[str, object]:
    """Wrap rows in the ``{"data": ..., "info": {...}}`` shape."""
    info: dict[str, object] = {}
    if id_to_labels is not None:
        info["idToLabels"] = id_to_labels
    return {"data": rows, "info": info}


# ---------------------------------------------------------------------------
# DetectedBbox dataclass
# ---------------------------------------------------------------------------


class TestDetectedBbox:
    def test_to_dict_round_trip(self):
        bbox = DetectedBbox(
            semantic_id=5,
            label="table",
            labels=("table",),
            bbox_2d=(120, 340, 450, 720),
            occlusion_ratio=0.12,
        )
        d = bbox.to_dict()
        assert d == {
            "label": "table",
            "labels": ["table"],
            "bbox_2d": [120, 340, 450, 720],
            "semantic_id": 5,
            "occlusion_ratio": pytest.approx(0.12),
        }

    def test_to_dict_multilabel(self):
        bbox = DetectedBbox(
            semantic_id=7,
            label="chair",
            labels=("chair", "seat", "Furniture"),
            bbox_2d=(0, 0, 10, 10),
            occlusion_ratio=0.0,
        )
        d = bbox.to_dict()
        assert d["label"] == "chair"
        assert d["labels"] == ["chair", "seat", "Furniture"]

    def test_is_degenerate_zero_area(self):
        bbox = DetectedBbox(
            semantic_id=1, label="x", labels=("x",),
            bbox_2d=(10, 10, 10, 10), occlusion_ratio=0.0,
        )
        assert bbox.is_degenerate

    def test_is_degenerate_inverted(self):
        bbox = DetectedBbox(
            semantic_id=1, label="x", labels=("x",),
            bbox_2d=(20, 10, 10, 20), occlusion_ratio=0.0,
        )
        assert bbox.is_degenerate

    def test_not_degenerate(self):
        bbox = DetectedBbox(
            semantic_id=1, label="x", labels=("x",),
            bbox_2d=(10, 10, 20, 20), occlusion_ratio=0.0,
        )
        assert not bbox.is_degenerate


# ---------------------------------------------------------------------------
# parse_bbox_data
# ---------------------------------------------------------------------------


class TestParseBboxDataBasics:
    def test_none_returns_empty(self):
        assert parse_bbox_data(None) == []

    def test_missing_data_key(self):
        assert parse_bbox_data({"info": {}}) == []

    def test_empty_rows(self):
        raw = _raw_output([], {"0": {"class": "table"}})
        assert parse_bbox_data(raw) == []

    def test_single_bbox(self):
        raw = _raw_output(
            [_bbox_row(semantic_id=1, x_min=100, y_min=200, x_max=400, y_max=700)],
            {"1": {"class": "table"}},
        )
        result = parse_bbox_data(raw)
        assert len(result) == 1
        assert result[0] == DetectedBbox(
            semantic_id=1,
            label="table",
            labels=("table",),
            bbox_2d=(100, 200, 400, 700),
            occlusion_ratio=0.0,
        )

    def test_multiple_bboxes_different_ids(self):
        raw = _raw_output(
            [
                _bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=50, y_max=50),
                _bbox_row(semantic_id=2, x_min=60, y_min=60, x_max=120, y_max=120),
                _bbox_row(semantic_id=3, x_min=200, y_min=100, x_max=400, y_max=500),
            ],
            {
                "1": {"class": "table"},
                "2": {"class": "chair"},
                "3": {"class": "door"},
            },
        )
        result = parse_bbox_data(raw)
        assert [b.label for b in result] == ["table", "chair", "door"]

    def test_multiple_bboxes_same_id(self):
        """Two table instances share a semantic id; both are returned."""
        raw = _raw_output(
            [
                _bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=50, y_max=50),
                _bbox_row(semantic_id=1, x_min=100, y_min=100, x_max=200, y_max=200),
            ],
            {"1": {"class": "table"}},
        )
        result = parse_bbox_data(raw)
        assert len(result) == 2
        assert all(b.label == "table" for b in result)
        assert all(b.semantic_id == 1 for b in result)


class TestParseBboxDataLabelResolution:
    def test_missing_id_to_labels_falls_back_to_unknown(self):
        raw = {
            "data": [_bbox_row(semantic_id=42, x_min=0, y_min=0, x_max=10, y_max=10)],
            "info": {},
        }
        result = parse_bbox_data(raw)
        assert len(result) == 1
        assert result[0].label == UNKNOWN_LABEL
        assert result[0].labels == (UNKNOWN_LABEL,)

    def test_unknown_semantic_id_falls_back_to_unknown(self):
        """Row references a semantic id not present in idToLabels."""
        raw = _raw_output(
            [_bbox_row(semantic_id=99, x_min=0, y_min=0, x_max=10, y_max=10)],
            {"1": {"class": "table"}},  # 99 not in map
        )
        result = parse_bbox_data(raw)
        assert result[0].label == UNKNOWN_LABEL

    def test_multilabel_comma_separated(self):
        raw = _raw_output(
            [_bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=10, y_max=10)],
            {"1": {"class": "chair,seat,Furniture"}},
        )
        result = parse_bbox_data(raw)
        assert result[0].label == "chair"
        assert result[0].labels == ("chair", "seat", "Furniture")

    def test_multilabel_with_whitespace(self):
        raw = _raw_output(
            [_bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=10, y_max=10)],
            {"1": {"class": "chair , seat ,  Furniture "}},
        )
        result = parse_bbox_data(raw)
        assert result[0].labels == ("chair", "seat", "Furniture")

    def test_empty_class_string_falls_back(self):
        raw = _raw_output(
            [_bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=10, y_max=10)],
            {"1": {"class": ""}},
        )
        result = parse_bbox_data(raw)
        assert result[0].label == UNKNOWN_LABEL

    def test_info_value_without_class_key(self):
        """Some writer variants put the string directly under the id."""
        raw = _raw_output(
            [_bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=10, y_max=10)],
            {"1": "table"},
        )
        result = parse_bbox_data(raw)
        assert result[0].label == "table"

    def test_numeric_semantic_id_key_is_stringified(self):
        """idToLabels keys are strings in Replicator's output — confirm the
        parser looks them up as strings even when the row's semanticId is a
        Python int or numpy integer."""
        raw = _raw_output(
            [_bbox_row(semantic_id=42, x_min=0, y_min=0, x_max=10, y_max=10)],
            {"42": {"class": "lamp"}},
        )
        result = parse_bbox_data(raw)
        assert result[0].label == "lamp"


class TestParseBboxDataFiltering:
    def test_drop_degenerate_default(self):
        raw = _raw_output(
            [
                _bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=0, y_max=0),    # degenerate
                _bbox_row(semantic_id=2, x_min=10, y_min=10, x_max=20, y_max=20),  # valid
                _bbox_row(semantic_id=3, x_min=50, y_min=50, x_max=40, y_max=60),  # inverted
            ],
            {"1": {"class": "a"}, "2": {"class": "b"}, "3": {"class": "c"}},
        )
        result = parse_bbox_data(raw)
        assert len(result) == 1
        assert result[0].label == "b"

    def test_keep_degenerate_when_disabled(self):
        raw = _raw_output(
            [_bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=0, y_max=0)],
            {"1": {"class": "a"}},
        )
        result = parse_bbox_data(raw, drop_degenerate=False)
        assert len(result) == 1
        assert result[0].is_degenerate

    def test_min_occlusion_visible_filter(self):
        """Filter out bboxes whose visible fraction is below the threshold."""
        raw = _raw_output(
            [
                _bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=10, y_max=10, occlusion=0.0),  # fully visible
                _bbox_row(semantic_id=2, x_min=0, y_min=0, x_max=10, y_max=10, occlusion=0.5),  # half visible
                _bbox_row(semantic_id=3, x_min=0, y_min=0, x_max=10, y_max=10, occlusion=0.9),  # mostly hidden
            ],
            {"1": {"class": "a"}, "2": {"class": "b"}, "3": {"class": "c"}},
        )
        result = parse_bbox_data(raw, min_occlusion_visible=0.3)
        assert [b.label for b in result] == ["a", "b"]

    def test_occlusion_ratio_passthrough(self):
        raw = _raw_output(
            [_bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=10, y_max=10, occlusion=0.42)],
            {"1": {"class": "x"}},
        )
        result = parse_bbox_data(raw)
        assert result[0].occlusion_ratio == pytest.approx(0.42)


class TestParseBboxDataRowShapes:
    """The parser must handle numpy structured rows, Python tuples, and
    dict-like rows (used by tests) transparently."""

    def test_tuple_rows_positional_access(self):
        """Simulate a numpy structured array that only supports indexed
        access — our _row_field helper must fall through to positional."""
        # Plain tuple, 6 fields in the documented order.
        rows = [(1, 100, 200, 400, 700, 0.1)]
        raw = {"data": rows, "info": {"idToLabels": {"1": {"class": "table"}}}}
        result = parse_bbox_data(raw)
        assert len(result) == 1
        assert result[0].bbox_2d == (100, 200, 400, 700)
        assert result[0].occlusion_ratio == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# ReplicatorBboxExtractor
# ---------------------------------------------------------------------------


class TestReplicatorBboxExtractor:
    def test_constructs_with_injected_annotator(self):
        mock = MagicMock()
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path="/rp/test",
            annotator=mock,
        )
        assert extractor.annotator is mock
        assert extractor._attached

    def test_extract_delegates_to_parser(self):
        mock = MagicMock()
        mock.get_data.return_value = _raw_output(
            [_bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=10, y_max=10)],
            {"1": {"class": "door"}},
        )
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path="/rp/test",
            annotator=mock,
        )
        result = extractor.extract()
        assert len(result) == 1
        assert result[0].label == "door"
        mock.get_data.assert_called_once()

    def test_extract_empty_frame(self):
        mock = MagicMock()
        mock.get_data.return_value = {"data": [], "info": {"idToLabels": {}}}
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path="/rp/test",
            annotator=mock,
        )
        assert extractor.extract() == []

    def test_extract_as_dicts(self):
        mock = MagicMock()
        mock.get_data.return_value = _raw_output(
            [
                _bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=10, y_max=10),
                _bbox_row(semantic_id=2, x_min=20, y_min=20, x_max=30, y_max=30),
            ],
            {"1": {"class": "a"}, "2": {"class": "b"}},
        )
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path="/rp/test",
            annotator=mock,
        )
        dicts = extractor.extract_as_dicts()
        assert all(isinstance(d, dict) for d in dicts)
        assert [d["label"] for d in dicts] == ["a", "b"]
        assert [d["bbox_2d"] for d in dicts] == [[0, 0, 10, 10], [20, 20, 30, 30]]

    def test_extract_drop_degenerate_applied(self):
        mock = MagicMock()
        mock.get_data.return_value = _raw_output(
            [
                _bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=0, y_max=0),    # degenerate
                _bbox_row(semantic_id=2, x_min=10, y_min=10, x_max=20, y_max=20),
            ],
            {"1": {"class": "a"}, "2": {"class": "b"}},
        )
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path="/rp/test",
            annotator=mock,
        )
        result = extractor.extract()
        assert len(result) == 1
        assert result[0].label == "b"

    def test_extract_with_occlusion_filter(self):
        mock = MagicMock()
        mock.get_data.return_value = _raw_output(
            [
                _bbox_row(semantic_id=1, x_min=0, y_min=0, x_max=10, y_max=10, occlusion=0.0),
                _bbox_row(semantic_id=2, x_min=0, y_min=0, x_max=10, y_max=10, occlusion=0.8),
            ],
            {"1": {"class": "a"}, "2": {"class": "b"}},
        )
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path="/rp/test",
            annotator=mock,
            min_occlusion_visible=0.5,
        )
        result = extractor.extract()
        assert [b.label for b in result] == ["a"]

    def test_raises_when_annotator_is_none_after_init(self):
        """Defensive check for the internal _attached sentinel."""
        mock = MagicMock()
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path="/rp/test",
            annotator=mock,
        )
        # Simulate an unexpected detach — extract() should raise rather than
        # swallow the state corruption.
        extractor.annotator = None
        with pytest.raises(RuntimeError, match="no annotator"):
            extractor.extract()

    def test_semantic_types_kwarg_accepted_without_touching_omni(self):
        """semantic_types is a constructor kwarg; confirm the class accepts
        it when an annotator is injected (so the omni branch never runs)."""
        mock = MagicMock()
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path="/rp/test",
            semantic_types=("class",),
            annotator=mock,
        )
        assert extractor.semantic_types == ("class",)
