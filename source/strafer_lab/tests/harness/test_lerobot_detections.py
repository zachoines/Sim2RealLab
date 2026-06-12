"""Unit tests for the first-class detections columns (pure-Python half).

Exercises :mod:`strafer_lab.tools.lerobot_detections` — feature
declaration, padding/truncation packing, vocab accumulation and JSON
round-trip — without lerobot. The writer-integrated round-trip lives in
``test_lerobot_writer.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from strafer_lab.tools.bbox_extractor import DetectedBbox
from strafer_lab.tools.lerobot_detections import (
    DETECTIONS_BBOX,
    DETECTIONS_LABEL_ID,
    DETECTIONS_OCCLUSION,
    DETECTIONS_VALID,
    PAD_LABEL_ID,
    DetectionLabelVocab,
    detections_features,
    pack_detections,
    read_detection_labels,
    vocab_path,
)


def _det(
    label: str,
    bbox: tuple[int, int, int, int],
    occlusion: float = 0.0,
    semantic_id: int = 1,
) -> DetectedBbox:
    return DetectedBbox(
        semantic_id=semantic_id,
        label=label,
        labels=(label,),
        bbox_2d=bbox,
        occlusion_ratio=occlusion,
    )


class TestDetectionsFeatures:
    def test_declares_four_columns_with_exact_shapes(self):
        feats = detections_features(8)
        assert feats[DETECTIONS_BBOX]["dtype"] == "float32"
        assert feats[DETECTIONS_BBOX]["shape"] == (8, 4)
        assert feats[DETECTIONS_LABEL_ID]["dtype"] == "int64"
        assert feats[DETECTIONS_LABEL_ID]["shape"] == (8,)
        assert feats[DETECTIONS_OCCLUSION]["dtype"] == "float32"
        assert feats[DETECTIONS_OCCLUSION]["shape"] == (8,)
        assert feats[DETECTIONS_VALID]["dtype"] == "bool"
        assert feats[DETECTIONS_VALID]["shape"] == (8,)

    def test_rejects_nonpositive_max(self):
        with pytest.raises(ValueError, match="detections_max"):
            detections_features(0)


class TestVocab:
    def test_first_seen_order_and_stability(self):
        vocab = DetectionLabelVocab()
        assert vocab.id_for("chair") == 0
        assert vocab.id_for("sofa") == 1
        assert vocab.id_for("chair") == 0
        assert vocab.labels == ("chair", "sofa")

    def test_write_read_round_trip(self, tmp_path):
        root = tmp_path / "dataset"
        vocab = DetectionLabelVocab(["chair", "sofa", "lamp"])
        path = vocab.write(root)
        assert path == vocab_path(root)
        assert read_detection_labels(root) == ("chair", "sofa", "lamp")

    def test_read_missing_returns_empty(self, tmp_path):
        assert read_detection_labels(tmp_path / "nope") == ()


class TestPackDetections:
    def test_pads_to_max_with_sentinels(self):
        vocab = DetectionLabelVocab()
        packed = pack_detections(
            [_det("chair", (10, 20, 110, 220), occlusion=0.25)], 4, vocab,
        )
        assert packed[DETECTIONS_BBOX].shape == (4, 4)
        assert packed[DETECTIONS_BBOX].dtype == np.float32
        np.testing.assert_array_equal(
            packed[DETECTIONS_BBOX][0], [10.0, 20.0, 110.0, 220.0],
        )
        np.testing.assert_array_equal(packed[DETECTIONS_BBOX][1:], 0.0)
        assert packed[DETECTIONS_LABEL_ID].dtype == np.int64
        assert packed[DETECTIONS_LABEL_ID][0] == 0
        assert (packed[DETECTIONS_LABEL_ID][1:] == PAD_LABEL_ID).all()
        assert packed[DETECTIONS_OCCLUSION].dtype == np.float32
        assert packed[DETECTIONS_OCCLUSION][0] == pytest.approx(0.25)
        assert packed[DETECTIONS_VALID].dtype == np.bool_
        np.testing.assert_array_equal(
            packed[DETECTIONS_VALID], [True, False, False, False],
        )

    def test_empty_frame_packs_all_padding(self):
        packed = pack_detections([], 3, DetectionLabelVocab())
        assert not packed[DETECTIONS_VALID].any()
        assert (packed[DETECTIONS_LABEL_ID] == PAD_LABEL_ID).all()
        np.testing.assert_array_equal(packed[DETECTIONS_BBOX], 0.0)

    def test_drops_degenerate_boxes(self):
        packed = pack_detections(
            [
                _det("chair", (50, 50, 50, 80)),  # zero width
                _det("sofa", (10, 10, 40, 40)),
            ],
            4,
            DetectionLabelVocab(),
        )
        assert packed[DETECTIONS_VALID].sum() == 1
        np.testing.assert_array_equal(
            packed[DETECTIONS_BBOX][0], [10.0, 10.0, 40.0, 40.0],
        )

    def test_truncation_keeps_largest_area_deterministically(self):
        small = _det("lamp", (0, 0, 10, 10))
        mid = _det("chair", (0, 0, 50, 50))
        big = _det("sofa", (0, 0, 100, 100))
        packed_a = pack_detections([small, mid, big], 2, DetectionLabelVocab())
        packed_b = pack_detections([big, small, mid], 2, DetectionLabelVocab())
        # Largest two survive regardless of input order; rows identical.
        np.testing.assert_array_equal(
            packed_a[DETECTIONS_BBOX], packed_b[DETECTIONS_BBOX],
        )
        areas = (
            packed_a[DETECTIONS_BBOX][:, 2] - packed_a[DETECTIONS_BBOX][:, 0]
        ) * (
            packed_a[DETECTIONS_BBOX][:, 3] - packed_a[DETECTIONS_BBOX][:, 1]
        )
        np.testing.assert_array_equal(areas, [10000.0, 2500.0])
        assert packed_a[DETECTIONS_VALID].all()

    def test_no_truncation_preserves_input_order(self):
        first = _det("lamp", (0, 0, 10, 10))
        second = _det("sofa", (0, 0, 100, 100))
        packed = pack_detections([first, second], 4, DetectionLabelVocab())
        np.testing.assert_array_equal(
            packed[DETECTIONS_BBOX][0], [0.0, 0.0, 10.0, 10.0],
        )
        np.testing.assert_array_equal(
            packed[DETECTIONS_BBOX][1], [0.0, 0.0, 100.0, 100.0],
        )

    def test_vocab_accumulates_across_frames(self):
        vocab = DetectionLabelVocab()
        pack_detections([_det("chair", (0, 0, 10, 10))], 2, vocab)
        packed = pack_detections(
            [_det("sofa", (0, 0, 10, 10)), _det("chair", (5, 5, 15, 15))],
            2,
            vocab,
        )
        assert vocab.labels == ("chair", "sofa")
        np.testing.assert_array_equal(packed[DETECTIONS_LABEL_ID], [1, 0])
