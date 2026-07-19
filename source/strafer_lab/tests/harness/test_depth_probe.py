"""Tests for the RTX Real-Time 2.0 depth-integrity probe's analysis gate.

Pure numpy — exercises the cadence / frame-diff / RT1-vs-RT2 parity checks the
probe uses to decide whether the depth stream is intact under the new renderer.
The Kit-side capture is operator-run; this locks the gate logic.
"""
from __future__ import annotations

import numpy as np

from strafer_lab.tools.depth_probe import (
    FAR_CLIP_M,
    check_cadence,
    check_frame_diff_under_motion,
    depth_parity,
)

_DT = 1.0 / 120.0


# --- cadence ---------------------------------------------------------------


def test_cadence_uniform_passes():
    times = np.arange(1, 33) * _DT
    ok, msg = check_cadence(times, _DT)
    assert ok, msg


def test_cadence_duplicate_frame_fails():
    # A duplicated frame shows as a zero-length sim-time step.
    times = np.array([1, 2, 2, 3, 4], dtype=float) * _DT
    ok, msg = check_cadence(times, _DT)
    assert not ok and "increasing" in msg


def test_cadence_extra_or_dropped_frame_fails():
    # A dropped frame doubles one inter-frame delta.
    times = np.array([1, 2, 4, 5], dtype=float) * _DT
    ok, msg = check_cadence(times, _DT)
    assert not ok and "outside" in msg


def test_cadence_needs_two_frames():
    ok, _ = check_cadence([_DT], _DT)
    assert not ok


# --- frame-diff under motion -----------------------------------------------


def _gradient(h=8, w=8):
    return np.tile(np.linspace(1.0, 2.0, w), (h, 1))


def test_frame_diff_moving_passes():
    base = _gradient()
    depths = np.stack([base + i * 0.05 for i in range(6)], axis=0)
    ok, msg = check_frame_diff_under_motion(depths)
    assert ok, msg


def test_frame_diff_frozen_frames_fail():
    base = _gradient()
    depths = np.stack([base, base, base], axis=0)  # frozen
    ok, msg = check_frame_diff_under_motion(depths)
    assert not ok and ("repeated" in msg or "frozen" in msg)


def test_frame_diff_needs_two_frames():
    ok, _ = check_frame_diff_under_motion(_gradient()[None])
    assert not ok


def test_frame_diff_all_inf_pair_fails():
    depths = np.full((2, 4, 4), np.inf)
    ok, _ = check_frame_diff_under_motion(depths)
    assert not ok


# --- RT1 vs RT2 parity -----------------------------------------------------


def test_parity_identical_passes():
    a = _gradient()
    ok, max_diff, _ = depth_parity(a, a.copy())
    assert ok and max_diff == 0.0


def test_parity_within_budget_passes():
    a = _gradient()
    b = a + 5e-4  # under the 1e-3 m default budget
    ok, max_diff, msg = depth_parity(a, b)
    assert ok, msg
    assert max_diff <= 1e-3


def test_parity_over_budget_fails():
    a = _gradient()
    b = a + 2e-3
    ok, max_diff, _ = depth_parity(a, b)
    assert not ok and max_diff > 1e-3


def test_parity_shape_mismatch_fails():
    ok, _, msg = depth_parity(np.zeros((4, 4)), np.zeros((4, 5)))
    assert not ok and "shape" in msg


def test_parity_excludes_far_and_nonfinite_pixels():
    # Near geometry agrees to sub-mm; a beyond-far-clip pixel and an inf pixel
    # differ wildly but must be excluded from the statistic.
    a = _gradient()
    b = a + 5e-4
    a[0, 0] = FAR_CLIP_M + 100.0
    b[0, 0] = 1.0            # beyond-far-clip in `a` -> excluded
    a[1, 1] = np.inf         # non-finite -> excluded
    b[1, 1] = 3.0
    ok, max_diff, _ = depth_parity(a, b)
    assert ok
    assert max_diff <= 1e-3


def test_parity_no_valid_pixels_fails():
    a = np.full((4, 4), np.inf)
    ok, _, msg = depth_parity(a, a.copy())
    assert not ok and "no co-finite" in msg
