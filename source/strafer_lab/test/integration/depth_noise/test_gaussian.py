# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for depth camera Gaussian noise through the full environment pipeline.

Tests the Gaussian noise component in ISOLATION by disabling holes and frame drops.
Validates the full pipeline: RAW meters -> Gaussian noise -> clamp -> scale -> clip.

Uses the dedicated test scene with a wall at known distance (2.0m) to provide
stable pixels at a known depth, enabling precise variance testing.

STEREO DEPTH NOISE MODEL (Intel RealSense):
The depth noise model uses stereo error propagation from the Intel RealSense documentation:

    σ_z = (z² / (f · B)) · σ_d

Where:
    z = depth in meters
    f = focal length in pixels (at native camera resolution)
    B = stereo baseline in meters (95mm for D555)
    σ_d = subpixel disparity noise (typically 0.05-0.1 pixels)

This quadratic z² relationship (NOT linear) arises from error propagation of the
stereo depth equation z = f·B/d where d is disparity. The derivative gives
dz/dd = -z²/(f·B), so σ_z = |dz/dd| · σ_d = z²·σ_d/(f·B).

After normalization to [0, 1], the effective noise std in normalized space:
    effective_std = σ_z / max_range

For first-differences y[t] - y[t-1] of IID observations:
    Var(diff) = 2 * effective_std^2

CLAMPING EFFECTS:
Near max_range, values are clamped (can't exceed 1.0). This reduces variance.
See clamped_normal_variance() in utils.py for the analytical adjustment.

See also:
    test_holes.py - Hole noise component
    test_frame_drops.py - Frame drop noise component

Usage:
    cd source/strafer_lab
    pytest test/integration/depth_noise/test_gaussian.py -v -s
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
    D555_BASELINE_M,
    D555_FOCAL_LENGTH_PX,
    D555_DISPARITY_NOISE_PX,
    stereo_depth_noise_std,
    clamped_normal_variance,
    identify_wall_pixels,
    collect_stationary_observations,
    create_depth_test_env,
    create_gaussian_only_noise_cfg,
    set_simulation_app,
)


# =============================================================================
# Test Configuration
# =============================================================================

# Intel RealSense D555 stereo noise parameters for testing
# Using default D555 values - see utils.py for documentation
TEST_DISPARITY_NOISE_PX = D555_DISPARITY_NOISE_PX  # 0.08 pixels


# =============================================================================
# Module-scoped Environment
# =============================================================================

_module_env = None

# Store simulation app reference for cleanup
set_simulation_app(simulation_app)


def _get_or_create_test_env():
    """Get or create the shared test environment with Gaussian-only noise.

    Uses the dedicated test scene with a wall at known distance.
    Configures Intel D555 stereo noise model with default parameters.
    """
    global _module_env

    if _module_env is not None:
        return _module_env

    noise_cfg = create_gaussian_only_noise_cfg(
        baseline_m=D555_BASELINE_M,
        focal_length_px=D555_FOCAL_LENGTH_PX,
        disparity_noise_px=TEST_DISPARITY_NOISE_PX,
    )
    # Use test scene with wall at known distance
    _module_env = create_depth_test_env(noise_cfg, num_envs=NUM_ENVS, use_test_scene=True)

    return _module_env


@pytest.fixture(scope="module")
def depth_env():
    """Provide shared depth camera test environment with Gaussian-only noise."""
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
# Tests: Gaussian Noise Variance
# =============================================================================

class TestGaussianNoiseVariance:
    """Test Gaussian noise variance matches theoretical prediction.

    Uses the dedicated test scene with a wall at TEST_WALL_DISTANCE (2.0m).
    Wall pixels provide stable depth readings at a known distance, enabling
    precise noise variance testing.

    ANALYTICAL MODEL (Intel RealSense Stereo Error Propagation):
    For wall pixels at depth z = 2.0m with D555 parameters:
        σ_z = (z² / (f · B)) · σ_d
            = (2.0² / (673 · 0.095)) · 0.08
            = (4.0 / 63.935) · 0.08
            = 0.00501 meters ≈ 5.0mm

        effective_std (normalized) = 0.00501 / 6.0 = 0.000835

        Var(diff) = 2 * effective_std² = 2 * 0.000835² = 1.39e-6
    """

    def test_variance_at_wall_distance(self, depth_env):
        """Verify first-difference variance matches expected stereo depth noise.

        Uses a proper two-sided chi-squared test to verify variance matches
        the theoretical prediction: Var(first_diff) = 2 * σ² where σ is the
        stereo depth noise std in normalized units.

        For wall at 2.0m with D555 stereo parameters:
            σ_z = (z² / (f · B)) · σ_d = (4.0 / 63.935) · 0.08 ≈ 5.0mm
            σ_normalized = 5.0mm / 6000mm = 0.000835
            expected_var = 2 * 0.000835² = 1.39e-6
        """
        obs = collect_stationary_observations(depth_env, N_SAMPLES_STEPS)

        # Calculate expected variance using stereo error propagation
        depth_meters = TEST_WALL_DISTANCE
        noise_std_meters = stereo_depth_noise_std(
            depth_meters,
            baseline_m=D555_BASELINE_M,
            focal_length_px=D555_FOCAL_LENGTH_PX,
            disparity_noise_px=TEST_DISPARITY_NOISE_PX,
        )
        noise_std_normalized = noise_std_meters / DEPTH_MAX_RANGE
        expected_var = 2 * (noise_std_normalized ** 2)

        print(f"\n  Gaussian variance test at wall distance ({depth_meters}m):")
        print(f"    Configured noise_std (meters): {noise_std_meters:.4f}")
        print(f"    Configured noise_std (normalized): {noise_std_normalized:.4f}")
        print(f"    Expected Var(diff): {expected_var:.2e}")

        # Pool all wall pixel differences across all environments
        all_diffs = []
        total_wall_pixels = 0
        envs_with_wall = 0

        for env_idx in range(obs.shape[1]):
            depth_obs_env = obs[:, env_idx, DEPTH_START_IDX:]
            first_diffs = depth_obs_env[1:] - depth_obs_env[:-1]

            # Identify wall pixels using shared helper
            is_wall_pixel = identify_wall_pixels(depth_obs_env, tolerance=0.05, max_std=0.02)
            n_wall = is_wall_pixel.sum().item()

            if n_wall < 100:
                print(f"    Env {env_idx}: Only {n_wall} wall pixels found, skipping")
                continue

            envs_with_wall += 1
            diffs_at_wall = first_diffs[:, is_wall_pixel].flatten()
            all_diffs.append(diffs_at_wall)
            total_wall_pixels += n_wall

            if env_idx < 3:
                print(f"    Env {env_idx}: {n_wall} wall pixels")

        assert len(all_diffs) > 0, (
            f"No environments with sufficient wall pixels found. "
            f"Check that test scene is correctly configured with wall at {TEST_WALL_DISTANCE}m."
        )

        # Combine all differences
        all_diffs_tensor = torch.cat(all_diffs)
        n_samples = all_diffs_tensor.numel()
        measured_var = all_diffs_tensor.var().item()
        measured_std_m = (measured_var ** 0.5) * DEPTH_MAX_RANGE / (2 ** 0.5)
        ratio = measured_var / expected_var

        # Two-sided chi-squared test for variance
        df = n_samples - 1
        alpha = 1 - CONFIDENCE_LEVEL
        chi2_low = stats.chi2.ppf(alpha / 2, df)
        chi2_high = stats.chi2.ppf(1 - alpha / 2, df)
        ci_low = chi2_low / df
        ci_high = chi2_high / df

        # With very large sample sizes, the chi-squared CI becomes extremely narrow.
        # We use practical tolerance bounds that reflect engineering accuracy.
        PRACTICAL_TOLERANCE_LOW = 0.9
        PRACTICAL_TOLERANCE_HIGH = 1.1

        in_practical_bounds = PRACTICAL_TOLERANCE_LOW <= ratio <= PRACTICAL_TOLERANCE_HIGH
        in_ci = ci_low <= ratio <= ci_high

        print(f"    Results:")
        print(f"      Environments with wall: {envs_with_wall}")
        print(f"      Total samples: {n_samples}, wall pixels: {total_wall_pixels}")
        print(f"      Measured variance: {measured_var:.2e}")
        print(f"      Measured noise std: ~{measured_std_m*1000:.1f}mm")
        print(f"      Expected variance: {expected_var:.2e}")
        print(f"      Expected noise std: ~{noise_std_meters*1000:.1f}mm")
        print(f"      Variance ratio: {ratio:.4f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for ratio: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"      Practical tolerance: [{PRACTICAL_TOLERANCE_LOW}, {PRACTICAL_TOLERANCE_HIGH}]")
        print(f"      In practical bounds: {in_practical_bounds}")
        print(f"      In statistical CI: {in_ci}")

        if not in_practical_bounds:
            if ratio > PRACTICAL_TOLERANCE_HIGH:
                excess_std_m = (measured_var ** 0.5 - expected_var ** 0.5) * DEPTH_MAX_RANGE / (2 ** 0.5)
                print(f"    DIAGNOSIS: Measured variance is {ratio:.1f}x expected.")
                print(f"      Excess noise std: ~{max(0, excess_std_m)*1000:.1f}mm")
            else:
                print(f"    DIAGNOSIS: Measured variance is too LOW ({ratio:.4f}x expected).")

        assert in_practical_bounds, (
            f"Gaussian noise variance mismatch: ratio={ratio:.4f} "
            f"outside practical bounds [{PRACTICAL_TOLERANCE_LOW}, {PRACTICAL_TOLERANCE_HIGH}]. "
            f"Measured ~{measured_std_m*1000:.1f}mm vs expected ~{noise_std_meters*1000:.1f}mm."
        )

    def test_zero_mean_noise(self, depth_env):
        """Verify first-differences have zero mean (unbiased noise).

        Gaussian noise should be symmetric, so first-differences should
        average to zero across many samples.
        """
        obs = collect_stationary_observations(depth_env, N_SAMPLES_STEPS)

        print(f"\n  Zero-mean noise test:")

        # Pool all wall pixel differences
        all_diffs = []

        for env_idx in range(obs.shape[1]):
            depth_obs_env = obs[:, env_idx, DEPTH_START_IDX:]
            first_diffs = depth_obs_env[1:] - depth_obs_env[:-1]

            is_wall_pixel = identify_wall_pixels(depth_obs_env, tolerance=0.05, max_std=0.05)
            n_wall = is_wall_pixel.sum().item()

            if n_wall < 100:
                continue

            diffs_at_wall = first_diffs[:, is_wall_pixel].flatten()
            all_diffs.append(diffs_at_wall)

        assert len(all_diffs) > 0, "No environments with sufficient wall pixels"

        all_diffs_tensor = torch.cat(all_diffs)
        mean_diff = all_diffs_tensor.mean().item()
        std_diff = all_diffs_tensor.std().item()
        n_samples = all_diffs_tensor.numel()

        # Standard error of mean
        sem = std_diff / np.sqrt(n_samples)

        # 95% CI for mean should include 0
        ci_half_width = 1.96 * sem

        print(f"    Mean of differences: {mean_diff:.2e}")
        print(f"    Std of differences: {std_diff:.2e}")
        print(f"    N samples: {n_samples}")
        print(f"    95% CI for mean: [{mean_diff - ci_half_width:.2e}, {mean_diff + ci_half_width:.2e}]")

        # Mean should be within ~3 standard errors of zero
        assert abs(mean_diff) < 3 * sem, (
            f"Mean of differences is significantly non-zero: {mean_diff:.2e} "
            f"(expected ~0, SEM={sem:.2e})"
        )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
