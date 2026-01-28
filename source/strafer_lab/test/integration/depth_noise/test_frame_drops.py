# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for depth camera frame drop noise through the full environment pipeline.

Tests the frame drop noise component by enabling it along with small Gaussian noise.
Gaussian noise is needed to make frame drops detectable (without noise variance,
dropped vs fresh frames are identical).

Uses the dedicated test scene with a wall at known distance (2.0m) to provide
stable pixels at a known depth.

FRAME DROP NOISE MODEL:
With probability p_drop, the entire depth image is replaced with the previous frame.
This models real depth camera behavior where frames may be delayed or repeated.

When a frame is dropped, diff[t] = y[t] - y[t-1] = 0 exactly.
When a fresh frame arrives, diff[t] depends on when the previous fresh frame was.

The key testable property is that frame drops produce exactly zero differences
for all pixels simultaneously. We can detect frame drops by counting timesteps
where all pixel differences are exactly zero.

See also:
    test_gaussian.py - Gaussian noise component
    test_holes.py - Hole noise component

Usage:
    cd source/strafer_lab
    pytest test/integration/depth_noise/test_frame_drops.py -v -s
"""

# Isaac Sim must be launched BEFORE importing Isaac Lab modules
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

# --- Imports that require Isaac Sim runtime ---

import sys
from pathlib import Path

# Add this directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import pytest
from scipy import stats

# Import shared utilities (must come after Isaac Sim launch)
from utils import (
    NUM_ENVS,
    N_SAMPLES_STEPS,
    CONFIDENCE_LEVEL,
    DEPTH_START_IDX,
    DEPTH_MAX_RANGE,
    TEST_WALL_DISTANCE,
    TEST_WALL_DEPTH_NORMALIZED,
    DEBUG_OUTPUT_DIR,
    identify_wall_pixels,
    collect_stationary_observations,
    create_depth_test_env,
    create_frame_drops_with_gaussian_cfg,
    set_simulation_app,
    debug_camera_orientation,
)


# =============================================================================
# Test Configuration
# =============================================================================

# Frame drop parameters for testing
TEST_FRAME_DROP_PROB = 0.10   # 10% frame drop rate
TEST_BASE_NOISE_STD = 0.005   # 5mm Gaussian noise to make drops detectable


# =============================================================================
# Module-scoped Environment
# =============================================================================

_module_env = None

# Store simulation app reference for cleanup
set_simulation_app(simulation_app)


def _get_or_create_test_env():
    """Get or create the shared test environment with frame drops and Gaussian noise.

    Uses the dedicated test scene with a wall at known distance.
    """
    global _module_env

    if _module_env is not None:
        return _module_env

    noise_cfg = create_frame_drops_with_gaussian_cfg(
        frame_drop_prob=TEST_FRAME_DROP_PROB,
        base_noise_std=TEST_BASE_NOISE_STD,
    )
    # Use test scene with wall at known distance
    _module_env = create_depth_test_env(noise_cfg, num_envs=NUM_ENVS, use_test_scene=True)

    return _module_env


@pytest.fixture(scope="module")
def depth_env():
    """Provide shared depth camera test environment with frame drops."""
    env = _get_or_create_test_env()
    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up environment after all tests complete."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None
    simulation_app.close()


# =============================================================================
# Tests: Frame Drop Detection
# =============================================================================

class TestFrameDropVariance:
    """Test frame drop detection through zero-difference analysis.

    Uses the dedicated test scene with a wall at TEST_WALL_DISTANCE (2.0m).

    KEY INSIGHT:
    Frame drops produce EXACTLY zero differences for ALL pixels simultaneously.
    This is the distinguishing feature - Gaussian noise alone would never produce
    exactly zero differences across all pixels at the same timestep.

    We detect frame drops by finding timesteps where the maximum absolute
    difference across all pixels is exactly zero (or within floating point tolerance).
    """

    def test_frame_drop_detection(self, depth_env):
        """Verify frame drops can be detected as zero-difference timesteps.

        When a frame is dropped, y[t] = y[t-1] exactly, so diff = 0 for ALL pixels.
        We count timesteps where this occurs and verify the rate matches p_drop.
        """
        obs = collect_stationary_observations(depth_env, N_SAMPLES_STEPS)

        print(f"\n  Frame drop detection test:")
        print(f"    Expected frame drop rate: {TEST_FRAME_DROP_PROB}")

        # Use ALL depth pixels, not just wall pixels
        # Frame drops affect the entire image uniformly
        depth_obs = obs[:, 0, DEPTH_START_IDX:]  # Shape: (n_steps, n_pixels)
        first_diffs = depth_obs[1:] - depth_obs[:-1]  # Shape: (n_steps-1, n_pixels)

        n_steps = first_diffs.shape[0]
        n_pixels = first_diffs.shape[1]

        print(f"    Total timesteps: {n_steps}")
        print(f"    Total pixels per frame: {n_pixels}")

        # For each timestep, compute max absolute difference across all pixels
        # Frame drops will have max_abs_diff = 0 exactly
        max_abs_diff_per_step = first_diffs.abs().max(dim=1).values

        # Count timesteps with exactly zero max diff (within floating point tolerance)
        # Use a very small threshold to catch floating point zeros
        threshold = 1e-9
        zero_diff_mask = max_abs_diff_per_step < threshold
        n_zero_steps = zero_diff_mask.sum().item()

        observed_rate = n_zero_steps / n_steps

        print(f"    Zero-diff timesteps: {n_zero_steps}")
        print(f"    Observed drop rate: {observed_rate:.4f}")

        # Binomial test: is observed rate consistent with expected rate?
        result = stats.binomtest(n_zero_steps, n_steps, TEST_FRAME_DROP_PROB)
        alpha = 1 - CONFIDENCE_LEVEL

        print(f"    Binomial test p-value: {result.pvalue:.4f}")
        print(f"    Significance level (alpha): {alpha}")

        # Pass if we fail to reject H0 (rates are consistent)
        assert result.pvalue > alpha, (
            f"Frame drop rate mismatch: observed={observed_rate:.4f}, "
            f"expected={TEST_FRAME_DROP_PROB}, p-value={result.pvalue:.4f} <= alpha={alpha}"
        )

    def test_non_drop_frames_have_variance(self, depth_env):
        """Verify non-dropped frames have expected Gaussian variance.

        After excluding dropped frames (zero-diff timesteps), the remaining
        frames should show variance consistent with Gaussian noise.

        This test requires wall pixels at known depth to verify variance,
        so robot must face the wall.
        """
        print(f"\n  Non-drop frame variance test:")

        # Face the wall; orientation is expected to be correct via reset_robot_pose + test scene.
        obs = collect_stationary_observations(depth_env, N_SAMPLES_STEPS, face_wall=True)
        depth_obs = obs[:, 0, DEPTH_START_IDX:]

        camera = depth_env.scene["d555_camera"]
        height, width = camera.cfg.height, camera.cfg.width

        is_wall_like = identify_wall_pixels(
            depth_obs,
            tolerance=0.15,
            max_std=0.15,
            height=height,
            width=width,
            expected_depth=TEST_WALL_DEPTH_NORMALIZED,
        )
        n_wall_pixels = is_wall_like.sum().item()

        if n_wall_pixels < 50:
            print("    WARNING: Expected wall pixels not detected. Capturing debug orientation views...")
            debug_camera_orientation(depth_env)
            pytest.fail(
                f"Insufficient wall pixels when facing wall: {n_wall_pixels}. "
                f"Debug images saved to {DEBUG_OUTPUT_DIR}."
            )

        first_diffs = depth_obs[1:] - depth_obs[:-1]

        # Identify dropped frames (zero max diff)
        max_abs_diff_per_step = first_diffs.abs().max(dim=1).values
        is_dropped = max_abs_diff_per_step < 1e-9
        is_fresh = ~is_dropped

        n_dropped = is_dropped.sum().item()
        n_fresh = is_fresh.sum().item()

        print(f"    Dropped frames: {n_dropped}")
        print(f"    Fresh frames: {n_fresh}")

        assert n_fresh > 100, f"Insufficient fresh frames for analysis: {n_fresh}"

        # Get differences for fresh frames at wall pixels
        fresh_diffs = first_diffs[is_fresh][:, is_wall_like]
        measured_var = fresh_diffs.var().item()
        measured_std_norm = fresh_diffs.std().item()
        measured_std_m = measured_std_norm * DEPTH_MAX_RANGE

        # Expected variance for fresh frames: 2 * sigma^2
        # But consecutive fresh frames may have a dropped frame between them,
        # which doesn't change the variance formula for independent noise
        noise_std_normalized = TEST_BASE_NOISE_STD / DEPTH_MAX_RANGE
        expected_var = 2 * (noise_std_normalized ** 2)
        expected_std_norm = noise_std_normalized
        expected_std_m = expected_std_norm * DEPTH_MAX_RANGE

        ratio = measured_var / expected_var

        print(f"    Measured variance (fresh frames): {measured_var:.2e}")
        print(f"    Expected variance (2*sigma^2): {expected_var:.2e}")
        print(f"    Ratio: {ratio:.4f}")
        print(f"    Measured std: {measured_std_norm:.3e} (~{measured_std_m*1000:.2f} mm)")
        print(f"    Expected std: {expected_std_norm:.3e} (~{expected_std_m*1000:.2f} mm)")

        # Use chi-squared test
        n_samples = fresh_diffs.numel()
        df = n_samples - 1
        alpha = 1 - CONFIDENCE_LEVEL
        chi2_low = stats.chi2.ppf(alpha / 2, df)
        chi2_high = stats.chi2.ppf(1 - alpha / 2, df)
        ci_low = chi2_low / df
        ci_high = chi2_high / df

        print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI for ratio: [{ci_low:.4f}, {ci_high:.4f}]")

        # On failure, dump debug views to confirm camera orientation and wall visibility
        if not (ci_low <= ratio <= ci_high):
            print("    Variance outside CI; capturing debug camera orientation images...")
            debug_camera_orientation(depth_env)

        assert ci_low <= ratio <= ci_high, (
            f"Fresh frame variance mismatch: ratio={ratio:.4f} "
            f"outside {CONFIDENCE_LEVEL*100:.0f}% CI [{ci_low:.4f}, {ci_high:.4f}]"
        )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
