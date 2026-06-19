"""Tests for the detections-overlay drawing helper.

The dataset-driving wrapper needs a real capture (video + parquet); the
pure drawing helper is unit-testable on a numpy frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from strafer_lab.tools.detections_overlay import _color_for, draw_detection_boxes


def _blank(h=60, w=80):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDrawDetectionBoxes:
    def test_draws_valid_box_skips_padding(self):
        pytest.importorskip("cv2")
        frame = _blank()
        bboxes = [[10, 10, 40, 40], [50, 5, 70, 25]]
        out = draw_detection_boxes(
            frame, bboxes, label_ids=[0, 1], valid=[True, False], labels=["chair", "door"],
        )
        # The valid box left a coloured rectangle; the padding row did not.
        assert out[10, 10].any(), "valid box should mark its corner"
        assert not out[5, 50].any(), "invalid/padding row must not be drawn"

    def test_occlusion_filter_drops_heavily_occluded(self):
        pytest.importorskip("cv2")
        frame = _blank()
        out = draw_detection_boxes(
            frame, [[10, 10, 40, 40]], label_ids=[0], valid=[True], labels=["chair"],
            occlusions=[0.9], occlusion_max=0.5,
        )
        assert not out.any(), "a box above occlusion_max must be skipped"

    def test_out_of_range_label_id_does_not_crash(self):
        pytest.importorskip("cv2")
        frame = _blank()
        out = draw_detection_boxes(
            frame, [[10, 10, 40, 40]], label_ids=[7], valid=[True], labels=["chair"],
        )
        assert out[10, 10].any()


class TestColorPalette:
    def test_deterministic_and_cycles(self):
        assert _color_for(0) == _color_for(10)  # palette length 10
        assert _color_for(0) != _color_for(1)
