# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free oracle for the depth-noise variance tests' analytic yardstick.

Pins the two facts that keep those Kit tests (test_holes / test_frame_drops /
test_gaussian) stable across a camera-resolution change, by arithmetic:

1. The expected variance is resolution-invariant. The scene reads
   ``distance_to_image_plane`` off a fronto-parallel wall at a fixed 2.0 m, so
   every wall pixel reports the same depth regardless of vertical FOV; the
   z-scaled noise formulas carry no height/width term.

2. The chi-squared CI uses the wall-pixel count as its df, not the pooled
   pixel x timestep count. The pooled count collapses the CI onto a systematic
   residual (temporal diffs are autocorrelated; the residual is per-pixel); the
   wall-pixel df recalibrates from the pixel count while the pooled df does not.

Imports only installed constants + numpy/scipy, mirroring
``variance_ratio_test_spatial`` as an independent oracle rather than importing
the Kit-test tree.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from strafer_shared.constants import (
    DEPTH_CLIP_FAR,               # 6.0 m — depth normalization / sensor max range
    DEPTH_HEIGHT,                 # 45 — the policy-camera height
    DEPTH_WIDTH,                  # 80
    D555_FOCAL_LENGTH_MM,         # 1.93 mm  (pinhole focal length)
    D555_HORIZONTAL_APERTURE_MM,  # 3.68 mm  (horizontal aperture)
)

# ---------------------------------------------------------------------------
# Derivation inputs (documented sources; not fitted).
# ---------------------------------------------------------------------------
# Two policy grids spanned to demonstrate resolution-invariance. The 16:9 grid
# is the live camera resolution; the 4:3 grid is a differing-aspect comparison.
GRID_16_9 = (DEPTH_HEIGHT, DEPTH_WIDTH)   # (45, 80)
GRID_4_3 = (60, 80)

MAX_RANGE_M = DEPTH_CLIP_FAR              # 6.0
WALL_DEPTH_M = 2.0                        # test scene wall (perpendicular depth)

# Test-scene noise parameters (mirror the Kit tests' TEST_* / D555 defaults).
HOLE_PROBABILITY = 0.05                   # TEST_HOLE_PROBABILITY (test_holes)
FRAME_DROP_PROB = 0.10                    # TEST_FRAME_DROP_PROB (test_frame_drops)
DISPARITY_NOISE_PX = 0.08                 # D555_DISPARITY_NOISE_PX
FOCAL_LENGTH_PX = 673.0                   # D555_FOCAL_LENGTH_PX (native res)
BASELINE_M = 0.095                        # D555_BASELINE_M

# Statistical-collection shape (test_sim.common.constants).
NUM_ENVS = 64
N_SAMPLES_STEPS = 400
TIMESTEPS = N_SAMPLES_STEPS - 1           # first differences per pixel = 399
CONFIDENCE_LEVEL = 0.95

# Pooled independent-pixel counts each Kit test accumulates (a count of
# independent spatial units, not a fitted tolerance). test_holes caps at 200
# wall pixels/env (MAX_WALL_PIXELS_PER_ENV) -> 64*200; the uncapped tests select
# ~2480 wall pixels/env -> ~158720.
HOLES_WALL_PIXELS = NUM_ENVS * 200        # = 12_800
FRESH_WALL_PIXELS = 158_720

# Representative systematic residuals (first-difference variance ratios) the
# corrected CI must admit: ~0.15% (holes) and ~0.04% (fresh frame) off 1.0.
HOLES_RATIO = 0.9985
FRESH_RATIO = 1.000413


# ---------------------------------------------------------------------------
# Derivation helpers (pure arithmetic — the analytic yardstick).
# ---------------------------------------------------------------------------
def _vertical_fov_deg(height: int, width: int) -> float:
    """Isaac Sim derives VFOV from the resolution aspect ratio (square pixels)."""
    h_fov = 2.0 * np.arctan(D555_HORIZONTAL_APERTURE_MM / (2.0 * D555_FOCAL_LENGTH_MM))
    v_fov = 2.0 * np.arctan(np.tan(h_fov / 2.0) / (width / height))
    return float(np.degrees(v_fov))


def _expected_hole_diff_variance() -> float:
    """Var(y[t]-y[t-1]) with only holes: 2*p*(1-p)*(1-d)^2, d = depth/max_range."""
    d = WALL_DEPTH_M / MAX_RANGE_M
    return 2.0 * HOLE_PROBABILITY * (1.0 - HOLE_PROBABILITY) * (1.0 - d) ** 2


def _expected_fresh_diff_variance() -> float:
    """Var(y[t]-y[t-1]) with stereo gaussian + frame drops: (1-p_drop)*2*sigma^2.

    Stereo error propagation sigma_z = z^2 * disp / (f * B), normalized by
    max_range. z = WALL_DEPTH_M is the perpendicular wall depth at every pixel.
    """
    sigma_z = WALL_DEPTH_M ** 2 * DISPARITY_NOISE_PX / (FOCAL_LENGTH_PX * BASELINE_M)
    sigma_norm = sigma_z / MAX_RANGE_M
    return (1.0 - FRAME_DROP_PROB) * 2.0 * sigma_norm ** 2


def _chi2_ratio_ci(df: int) -> tuple[float, float]:
    """95% chi-squared CI for a variance ratio at ``df`` degrees of freedom.

    Same computation as variance_ratio_test / variance_ratio_test_spatial:
    (chi2.ppf(a/2, df)/df, chi2.ppf(1-a/2, df)/df).
    """
    alpha = 1.0 - CONFIDENCE_LEVEL
    return chi2.ppf(alpha / 2.0, df) / df, chi2.ppf(1.0 - alpha / 2.0, df) / df


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_expected_variance_is_resolution_invariant():
    """The analytic yardstick is identical across policy grids.

    The vertical FOV genuinely changes with resolution (asserted below), but the
    expected variances do not: perpendicular wall depth is 2.0 m at every wall
    pixel regardless of VFOV, so both formulas carry no height/width term. This
    is the crux — the model expectation does NOT move with resolution; only the
    pixel count (df) and the fixed-seed realization do.
    """
    # Geometry genuinely differs between the 16:9 and 4:3 grids...
    assert _vertical_fov_deg(*GRID_4_3) - _vertical_fov_deg(*GRID_16_9) > 10.0

    # ...yet the expected variances are the same resolution-invariant closed
    # forms (holes: 2*p*(1-p)*(1-1/3)^2; fresh: (1-p_drop)*2*(z^2*disp/(f*B)/max)^2).
    assert _expected_hole_diff_variance() == pytest.approx(0.0422222222, rel=1e-9)
    assert _expected_fresh_diff_variance() == pytest.approx(1.2525429e-06, rel=1e-6)


def test_vertical_fov_follows_resolution_aspect():
    """VFOV (the geometry axis that moves with resolution) differs by aspect.

    56.4 deg at 80x45 (16:9), 71.1 deg at 80x60 (4:3) — the resolution-derived
    vertical FOV that shifts the wall-pixel count and the fixed-seed realization
    while leaving the expected variance untouched.
    """
    assert _vertical_fov_deg(*GRID_16_9) == pytest.approx(56.41, abs=0.05)
    assert _vertical_fov_deg(*GRID_4_3) == pytest.approx(71.13, abs=0.05)
    assert _vertical_fov_deg(*GRID_4_3) > _vertical_fov_deg(*GRID_16_9)


@pytest.mark.parametrize(
    "name, wall_pixels, ratio",
    [
        ("holes", HOLES_WALL_PIXELS, HOLES_RATIO),
        ("fresh_frame", FRESH_WALL_PIXELS, FRESH_RATIO),
    ],
)
def test_pixel_df_admits_residual_that_temporal_df_rejects(name, wall_pixels, ratio):
    """The wall-pixel df admits the systematic residual; the pooled
    pixel x timestep df rejects it.

    This is the whole fix, by arithmetic: same measured ratio, two different
    effective sample sizes. The wall-pixel count is the conservative
    independent-unit ceiling.
    """
    spatial_low, spatial_high = _chi2_ratio_ci(wall_pixels - 1)
    temporal_low, temporal_high = _chi2_ratio_ci(wall_pixels * TIMESTEPS - 1)

    # Corrected yardstick: residual is comfortably inside.
    assert spatial_low <= ratio <= spatial_high, (
        f"{name}: ratio {ratio} outside spatial-df CI "
        f"[{spatial_low:.5f}, {spatial_high:.5f}] (df={wall_pixels - 1})"
    )
    # Pooled pixel*timestep df collapses the CI below the residual — the exact
    # false failure this recalibration removes.
    assert not (temporal_low <= ratio <= temporal_high), (
        f"{name}: ratio {ratio} unexpectedly inside temporal-df CI "
        f"[{temporal_low:.6f}, {temporal_high:.6f}] — the over-precision is gone?"
    )


def test_ci_halfwidth_recalibrates_by_arithmetic():
    """CI half-width follows 1.96*sqrt(2/df) from the live pixel count.

    Confirms the interval is a closed-form function of the independent-unit
    count, so a resolution change recalibrates it without any seed or tolerance
    edit.
    """
    for wall_pixels in (HOLES_WALL_PIXELS, FRESH_WALL_PIXELS):
        df = wall_pixels - 1
        low, high = _chi2_ratio_ci(df)
        measured_halfwidth = (high - low) / 2.0
        normal_approx = 1.96 * np.sqrt(2.0 / df)
        assert measured_halfwidth == pytest.approx(normal_approx, rel=0.02)
