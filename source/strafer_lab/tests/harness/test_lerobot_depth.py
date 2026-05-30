"""Unit tests for the 16UC1 PNG depth sidecar."""

from __future__ import annotations

import numpy as np
import pytest

from strafer_lab.tools.lerobot_depth import (
    depth_root,
    episode_dir,
    frame_path,
    read_depth_png,
    write_depth_png,
)


class TestPaths:
    def test_depth_root(self, tmp_path):
        assert depth_root(tmp_path) == tmp_path / "videos" / "observation.depth.perception"

    def test_episode_dir_is_zero_padded(self, tmp_path):
        path = episode_dir(tmp_path, 17)
        assert path.name == "episode-000017"

    def test_frame_path_is_zero_padded(self, tmp_path):
        path = frame_path(tmp_path, 3, 42)
        assert path.name == "000042.png"
        assert path.parent.name == "episode-000003"


class TestWriteReadRoundTrip:
    def test_constant_depth_recovers(self, tmp_path):
        depth = np.full((48, 64), 1.234, dtype=np.float32)  # 1.234m everywhere
        path = tmp_path / "test.png"
        write_depth_png(path, depth)
        assert path.is_file()

        recovered = read_depth_png(path)
        assert recovered.shape == (48, 64)
        assert recovered.dtype == np.float32
        # 1mm precision → max abs error ≤ 1e-3 m
        assert np.allclose(recovered, depth, atol=1e-3)

    def test_zero_depth_round_trips(self, tmp_path):
        depth = np.zeros((32, 32), dtype=np.float32)
        path = tmp_path / "zero.png"
        write_depth_png(path, depth)
        assert np.allclose(read_depth_png(path), depth)

    def test_nan_and_inf_become_zero(self, tmp_path):
        depth = np.array([
            [0.5, np.nan, 1.0],
            [np.inf, 2.0, -np.inf],
        ], dtype=np.float32)
        path = tmp_path / "edge.png"
        write_depth_png(path, depth)
        recovered = read_depth_png(path)
        assert recovered[0, 0] == pytest.approx(0.5, abs=1e-3)
        assert recovered[0, 1] == 0.0
        assert recovered[0, 2] == pytest.approx(1.0, abs=1e-3)
        assert recovered[1, 0] == 0.0
        assert recovered[1, 1] == pytest.approx(2.0, abs=1e-3)
        assert recovered[1, 2] == 0.0

    def test_clip_above_max(self, tmp_path):
        # 70m > 16-bit limit of 65.535m → clipped to max
        depth = np.array([[70.0]], dtype=np.float32)
        path = tmp_path / "clip.png"
        write_depth_png(path, depth)
        recovered = read_depth_png(path)
        # Max representable is 65535 * 0.001 = 65.535
        assert recovered[0, 0] == pytest.approx(65.535, abs=1e-3)

    def test_varying_depth_preserves_structure(self, tmp_path):
        # Realistic-shaped depth array — gradient that exercises full range
        h, w = 60, 80
        x = np.linspace(0.1, 6.0, w, dtype=np.float32)
        depth = np.broadcast_to(x, (h, w)).copy()
        path = tmp_path / "grad.png"
        write_depth_png(path, depth)
        recovered = read_depth_png(path)
        # Per-pixel error within 1mm
        assert np.all(np.abs(recovered - depth) <= 1e-3 + 1e-6)
        # Monotonicity preserved
        assert np.all(np.diff(recovered, axis=1) >= 0)


class TestRejectMalformed:
    def test_rejects_3d_input(self, tmp_path):
        depth = np.zeros((10, 10, 1), dtype=np.float32)
        with pytest.raises(ValueError, match=r"must be \(H, W\)"):
            write_depth_png(tmp_path / "x.png", depth)

    def test_rejects_1d_input(self, tmp_path):
        depth = np.zeros((10,), dtype=np.float32)
        with pytest.raises(ValueError, match=r"must be \(H, W\)"):
            write_depth_png(tmp_path / "x.png", depth)
