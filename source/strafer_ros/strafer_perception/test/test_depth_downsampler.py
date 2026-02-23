"""Tests for the strafer_perception depth downsampler node."""

import numpy as np
import pytest

from strafer_shared.constants import (
    DEPTH_WIDTH,
    DEPTH_HEIGHT,
    DEPTH_CLIP_NEAR,
    DEPTH_CLIP_FAR,
)
from strafer_perception.depth_downsampler import process_depth


class TestDepthDownsamplerLogic:
    """Unit tests for the depth processing pipeline.

    These test the real process_depth() function without requiring a
    running ROS2 node or RealSense camera.
    """

    def test_output_shape(self):
        """Downsampled output must match policy input resolution."""
        raw = np.full((360, 640), 2000, dtype=np.uint16)  # 2.0m everywhere
        result = process_depth(raw)
        assert result.shape == (DEPTH_HEIGHT, DEPTH_WIDTH)

    def test_output_dtype(self):
        """Output must be float32 (meters)."""
        raw = np.full((360, 640), 1500, dtype=np.uint16)
        result = process_depth(raw)
        assert result.dtype == np.float32

    def test_millimeters_to_meters(self):
        """Uniform depth should convert mm -> m correctly."""
        depth_mm = 3000  # 3.0 meters
        raw = np.full((360, 640), depth_mm, dtype=np.uint16)
        result = process_depth(raw)
        np.testing.assert_allclose(result, 3.0, atol=0.01)

    def test_near_clip(self):
        """Depths below DEPTH_CLIP_NEAR must be zeroed."""
        # 100mm = 0.1m, well below DEPTH_CLIP_NEAR (0.4m)
        raw = np.full((360, 640), 100, dtype=np.uint16)
        result = process_depth(raw)
        assert np.all(result == 0.0)

    def test_far_clip(self):
        """Depths beyond DEPTH_CLIP_FAR must be zeroed."""
        # 10000mm = 10.0m, beyond DEPTH_CLIP_FAR (6.0m)
        raw = np.full((360, 640), 10000, dtype=np.uint16)
        result = process_depth(raw)
        assert np.all(result == 0.0)

    def test_zero_depth_clipped(self):
        """Zero depth (no return / invalid) must be zeroed."""
        raw = np.zeros((360, 640), dtype=np.uint16)
        result = process_depth(raw)
        assert np.all(result == 0.0)

    def test_valid_range_passes(self):
        """Depths within [DEPTH_CLIP_NEAR, DEPTH_CLIP_FAR] must be preserved."""
        depth_mm = 2000  # 2.0m -- well within valid range
        raw = np.full((360, 640), depth_mm, dtype=np.uint16)
        result = process_depth(raw)
        assert np.all(result > 0.0)
        np.testing.assert_allclose(result, 2.0, atol=0.01)

    def test_mixed_valid_invalid(self):
        """Frame with a mix of valid and invalid depths."""
        raw = np.zeros((360, 640), dtype=np.uint16)
        # Top half: valid (2.0m)
        raw[:180, :] = 2000
        # Bottom half: too far (10.0m)
        raw[180:, :] = 10000

        result = process_depth(raw)
        # Top region should have valid depth, bottom should be zero
        assert result.shape == (DEPTH_HEIGHT, DEPTH_WIDTH)
        top_half = result[: DEPTH_HEIGHT // 2, :]
        bot_half = result[DEPTH_HEIGHT // 2 :, :]
        assert np.mean(top_half) > 1.0
        assert np.all(bot_half == 0.0)
