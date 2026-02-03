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

VARIANCE TESTING:
For fresh frames, we expect first-difference variance of 2σ² where σ is the
configured noise std (in normalized units). We use a two-sided chi-squared test
to verify this precisely.

See also:
    test_gaussian.py - Gaussian noise component
    test_holes.py - Hole noise component

Usage:
    cd source/strafer_lab
    pytest test/integration/depth_noise/test_frame_drops.py -v -s
"""

# Isaac Sim must be launched BEFORE importing Isaac Lab modules
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
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
    D555_BASELINE_M,
    D555_FOCAL_LENGTH_PX,
    D555_DISPARITY_NOISE_PX,
    stereo_depth_noise_std,
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
TEST_DISPARITY_NOISE_PX = D555_DISPARITY_NOISE_PX  # Use D555 stereo noise


# =============================================================================
# Utility: baseline variance correction
# =============================================================================


# =============================================================================
# Module-scoped Environment
# =============================================================================

_module_env = None

# Store simulation app reference for cleanup
set_simulation_app(simulation_app)


def _get_or_create_test_env():
    """Get or create the shared test environment with frame drops and stereo Gaussian noise.

    Uses the dedicated test scene with a wall at known distance.
    Stereo noise is needed to make frame drops detectable.
    """
    global _module_env

    if _module_env is not None:
        return _module_env

    noise_cfg = create_frame_drops_with_gaussian_cfg(
        frame_drop_prob=TEST_FRAME_DROP_PROB,
        disparity_noise_px=TEST_DISPARITY_NOISE_PX,
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

        Pools data across all parallel environments for robust statistics.
        """
        # Seed RNGs to make results independent of earlier tests (order changes)
        torch.manual_seed(42)
        np.random.seed(42)

        obs = collect_stationary_observations(depth_env, N_SAMPLES_STEPS, n_settle_steps=10)

        print(f"\n  Frame drop detection:")
        print(f"    Parallel environments: {obs.shape[1]}")

        # Pool across all environments
        total_steps = 0
        total_zero_steps = 0

        for env_idx in range(obs.shape[1]):
            depth_obs = obs[:, env_idx, DEPTH_START_IDX:]  # Shape: (n_steps, n_pixels)
            first_diffs = depth_obs[1:] - depth_obs[:-1]  # Shape: (n_steps-1, n_pixels)

            n_steps = first_diffs.shape[0]

            # For each timestep, compute max absolute difference across all pixels
            max_abs_diff_per_step = first_diffs.abs().max(dim=1).values

            # Count timesteps with exactly zero max diff (frame drops)
            threshold = 1e-9
            zero_diff_mask = max_abs_diff_per_step < threshold
            n_zero_steps = zero_diff_mask.sum().item()

            total_steps += n_steps
            total_zero_steps += n_zero_steps

        observed_rate = total_zero_steps / total_steps

        print(f"    Total timesteps: {total_steps}")
        print(f"    Expected drop prob: {TEST_FRAME_DROP_PROB}")
        print(f"    Measured drop prob: {observed_rate:.6f}")

        # Binomial test: is observed rate consistent with expected rate?
        result = stats.binomtest(total_zero_steps, total_steps, TEST_FRAME_DROP_PROB)
        alpha = 1 - CONFIDENCE_LEVEL

        print(f"    Binomial test p-value: {result.pvalue:.4f} (alpha={alpha})")

        # Pass if we fail to reject H0 (rates are consistent)
        assert result.pvalue > alpha, (
            f"Frame drop rate mismatch: observed={observed_rate:.4f}, "
            f"expected={TEST_FRAME_DROP_PROB}, p-value={result.pvalue:.4f} <= alpha={alpha}"
        )

    def test_fresh_frame_variance(self, depth_env):
        """Verify first-difference variance matches expected with frame drops.

        Uses the same methodology as the diagnostic test to ensure consistency.
        Frame drops are included in the analysis and accounted for analytically.

        ANALYTICAL MODEL WITH FRAME DROPS:
        When frame drop probability is p:
        - Fresh→Fresh transition (prob 1-p): diff has variance 2σ²
        - Any transition involving drop: diff = 0 (variance 0)

        Overall: E[Var(diff)] ≈ (1-p) * 2σ²

        We measure variance on ALL first differences (including zeros from drops)
        and compare against (1-p) * 2σ² where σ is the noise std.
        """
        print(f"\n  Fresh frame variance (with frame drops):")

        # Seed RNGs to make results independent of earlier tests (order changes)
        torch.manual_seed(42)
        np.random.seed(42)

        # Collect observations with extra settle time to ensure stable state
        obs = collect_stationary_observations(
            depth_env, N_SAMPLES_STEPS, face_wall=True,
            n_settle_steps=100,  # Extra settling for noise model state
        )

        camera = depth_env.scene["d555_camera"]
        height, width = camera.cfg.height, camera.cfg.width

        # Pool wall pixel differences across ALL environments
        all_diffs = []
        total_wall_pixels = 0
        envs_with_wall = 0
        wall_depth_means_m = []
        # Track empirical drop rate (using full frame diffs)
        total_steps = 0
        total_zero_steps = 0
        # Track per-sample noise variance (in normalized units^2) to avoid approximation
        noise_std_sq_sum = 0.0
        noise_std_sq_count = 0

        for env_idx in range(obs.shape[1]):
            depth_obs_env = obs[:, env_idx, DEPTH_START_IDX:]

            # Empirical drop detection on FULL image to get observed p_drop for expectation
            full_first_diffs = depth_obs_env[1:] - depth_obs_env[:-1]
            max_abs_full = full_first_diffs.abs().max(dim=1).values
            zero_mask = max_abs_full < 1e-9
            total_zero_steps += zero_mask.sum().item()
            total_steps += max_abs_full.numel()

            # Identify wall pixels for this environment (use looser tolerance like diagnostic)
            is_wall_pixel = identify_wall_pixels(
                depth_obs_env,
                tolerance=0.05,
                max_std=0.10,  # Looser tolerance to match diagnostic
                height=height,
                width=width,
                expected_depth=TEST_WALL_DEPTH_NORMALIZED,
            )
            n_wall = is_wall_pixel.sum().item()

            if n_wall < 50:
                if env_idx < 3:
                    print(f"    Env {env_idx}: Only {n_wall} wall pixels, skipping")
                continue

            envs_with_wall += 1
            total_wall_pixels += n_wall

            # Get wall pixel observations and compute first differences
            wall_depths = depth_obs_env[:, is_wall_pixel]  # (n_steps, n_wall_pixels)
            # Track measured wall depth (normalized -> meters)
            wall_depth_means_m.append((wall_depths.mean() * DEPTH_MAX_RANGE).item())

            # Accumulate expected per-sample noise variance in normalized units
            wall_depths_m = wall_depths * DEPTH_MAX_RANGE
            disparity_coeff = D555_DISPARITY_NOISE_PX / (D555_FOCAL_LENGTH_PX * D555_BASELINE_M)
            noise_std_normalized_sq = ((wall_depths_m.square() * disparity_coeff) / DEPTH_MAX_RANGE).square()
            noise_std_sq_sum += noise_std_normalized_sq.sum().item()
            noise_std_sq_count += noise_std_normalized_sq.numel()

            first_diffs = wall_depths[1:] - wall_depths[:-1]
            all_diffs.append(first_diffs.flatten())

        assert len(all_diffs) > 0, (
            f"No environments with sufficient wall pixels. "
            f"Check test scene geometry and camera orientation."
        )
        assert len(wall_depth_means_m) > 0, "Wall depth measurement failed (no wall pixels found)."
        assert noise_std_sq_count > 0, "Failed to accumulate noise variance estimates."

        # Combine all differences
        all_diffs_tensor = torch.cat(all_diffs)
        n_samples = all_diffs_tensor.numel()
        measured_var = all_diffs_tensor.var().item()

        # Expected variance accounting for frame drops:
        # For each first difference d[t] = y[t] - y[t-1]:
        #   - If y[t] is dropped (prob p): d[t] = 0 exactly (variance contribution = 0)
        #   - If y[t] is fresh (prob 1-p): d[t] = noise[t] - noise[τ(t-1)]
        #     where τ(t-1) is the time of the last fresh frame.
        #     Since noise values at different times are IID, Var(d[t]) = 2σ²
        #
        # Therefore: E[Var(diff)] = p·0 + (1-p)·2σ² = (1-p)·2σ²
        # This is the EXACT formula, not an approximation.
        #
        # Using stereo depth error propagation: σ_z = (z² / (f · B)) · σ_d
        mean_wall_depth_m = float(torch.tensor(wall_depth_means_m).mean().item())
        # Empirical drop rate from observed zero-diff steps
        observed_drop_rate = total_zero_steps / total_steps if total_steps > 0 else 0.0
        p_fresh = 1 - observed_drop_rate

        # Use per-sample averaged noise variance (normalized units) for the expectation
        mean_noise_std_sq_normalized = noise_std_sq_sum / noise_std_sq_count
        expected_var = (
            p_fresh * 2 * mean_noise_std_sq_normalized
        )
        expected_noise_std_m = (mean_noise_std_sq_normalized ** 0.5) * DEPTH_MAX_RANGE

        # Convert to physical units for reporting
        # Implied noise std from measured variance: σ = sqrt(var / (2 * (1-p)))
        measured_std_m = (measured_var / (2 * p_fresh)) ** 0.5 * DEPTH_MAX_RANGE
        expected_std_m = expected_noise_std_m

        ratio = measured_var / expected_var

        print(f"    Summary:")
        print(f"      Parallel environments: {obs.shape[1]}")
        print(f"      Wall pixels (total): {total_wall_pixels}")
        print(f"      Samples (diffs): {n_samples}")
        print(f"      Wall depth (mean): {mean_wall_depth_m:.4f} m")
        print(f"      Drop prob (config): {TEST_FRAME_DROP_PROB}")
        print(f"      Drop prob (measured): {observed_drop_rate:.6f}")
        print(f"      Variance (expected): {expected_var:.2e}")
        print(f"      Variance (measured): {measured_var:.2e}")
        print(f"      Noise std (expected): ~{expected_std_m*1000:.2f} mm")
        print(f"      Noise std (measured): ~{measured_std_m*1000:.2f} mm")
        print(f"      Variance ratio: {ratio:.4f}")

        df = n_samples - 1
        alpha = 1 - CONFIDENCE_LEVEL
        chi2_low = stats.chi2.ppf(alpha / 2, df)
        chi2_high = stats.chi2.ppf(1 - alpha / 2, df)

        ci_low = chi2_low / df
        ci_high = chi2_high / df

        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for ratio: [{ci_low:.4f}, {ci_high:.4f}]")

        in_ci = ci_low <= ratio <= ci_high

        print(f"      In statistical CI: {in_ci}")

        # debug_camera_orientation(depth_env)

        assert in_ci, (
            f"Fresh frame variance outside 95% CI: ratio={ratio:.6f}, "
            f"CI=[{ci_low:.6f}, {ci_high:.6f}], "
            f"observed p_drop={observed_drop_rate:.6f}, samples={n_samples}"
        )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
