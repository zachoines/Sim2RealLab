# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for depth camera hole noise through the full environment pipeline.

Tests the hole noise component in ISOLATION by disabling Gaussian noise and frame drops.
Validates the full pipeline: RAW meters -> holes -> clamp -> scale -> clip.

Uses the dedicated test scene with a wall at known distance (2.0m) to provide
stable pixels at a known depth, enabling precise hole detection testing.

HOLE NOISE MODEL:
Each pixel independently becomes a "hole" (set to max_range) with probability p
at each timestep. This models real depth camera behavior where stereo matching
or ToF measurement fails for certain pixels.

ANALYTICAL DERIVATION:
With only holes enabled, at each timestep a pixel is either:
    y[t] = d   (true depth, normalized) with probability (1 - p)
    y[t] = 1.0 (max_range, normalized)  with probability p

First differences y[t] - y[t-1] have four cases:
    - Both normal:    diff = 0,     P = (1-p)^2
    - Both holes:     diff = 0,     P = p^2
    - Normal -> Hole: diff = 1 - d, P = (1-p) * p
    - Hole -> Normal: diff = d - 1, P = p * (1-p)

Expected diff = 0 (symmetric transitions cancel)

Var(diff) = E[diff^2] = 2 * p * (1-p) * (1 - d)^2

For wall pixels at d = 2.0m (normalized 0.333):
    jump_size = 1 - 0.333 = 0.667
    Var(diff) = 2 * p * (1-p) * 0.667^2

For pixels already at max_range (d = 1.0): Var(diff) = 0 (holes have no effect).

See also:
    test_gaussian.py - Gaussian noise component
    test_frame_drops.py - Frame drop noise component

Usage:
    cd source/strafer_lab
    pytest test/integration/depth_noise/test_holes.py -v -s
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
    TEST_WALL_DISTANCE,
    TEST_WALL_DEPTH_NORMALIZED,
    hole_diff_variance,
    identify_wall_pixels_with_holes,
    collect_stationary_observations,
    collect_stationary_observations_facing_away,
    create_depth_test_env,
    create_holes_only_noise_cfg,
    set_simulation_app,
)


# =============================================================================
# Test Configuration
# =============================================================================

# Hole noise parameters for testing
TEST_HOLE_PROBABILITY = 0.05  # 5% - high enough to get reliable statistics


# =============================================================================
# Module-scoped Environment
# =============================================================================

_module_env = None

# Store simulation app reference for cleanup
set_simulation_app(simulation_app)


def _get_or_create_test_env():
    """Get or create the shared test environment with holes-only noise.

    Uses the dedicated test scene with a wall at known distance.
    """
    global _module_env

    if _module_env is not None:
        return _module_env

    noise_cfg = create_holes_only_noise_cfg(hole_probability=TEST_HOLE_PROBABILITY)
    # Use test scene with wall at known distance
    _module_env = create_depth_test_env(noise_cfg, num_envs=NUM_ENVS, use_test_scene=True)

    return _module_env


@pytest.fixture(scope="module")
def depth_env():
    """Provide shared depth camera test environment with holes-only noise."""
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
# Tests: Hole Noise Variance
# =============================================================================

class TestHoleNoiseVariance:
    """Test hole noise variance matches theoretical prediction.

    Uses the dedicated test scene with a wall at TEST_WALL_DISTANCE (2.0m).
    Wall pixels provide stable depth readings at a known distance, enabling
    precise hole detection testing.

    ANALYTICAL MODEL:
    For wall pixels at depth d = 2.0m (normalized 0.333):
        jump_size = 1.0 - 0.333 = 0.667
        Var(diff) = 2 * p * (1-p) * jump_size^2

    Wall pixels are ideal for hole testing because:
    1. Known, stable true depth (not at max_range)
    2. Large jump size when holes occur (0.667 vs smaller for near-max pixels)
    3. No confounding from scene dynamics or edge effects
    """

    def test_hole_rate_on_wall_pixels(self, depth_env):
        """Verify the observed hole rate matches the configured probability on wall pixels.

        Wall pixels have known true depth. When a hole occurs, the pixel jumps
        to max_range (1.0 normalized). We count these jumps to measure the hole rate.
        """
        obs = collect_stationary_observations(depth_env, N_SAMPLES_STEPS)

        print(f"\n  Hole rate test on wall pixels (d={TEST_WALL_DISTANCE}m):")
        print(f"    Expected wall depth (normalized): {TEST_WALL_DEPTH_NORMALIZED:.3f}")

        # Pool across all environments for better statistics
        total_at_wall = 0
        total_holes = 0

        for env_idx in range(obs.shape[1]):
            depth_obs_env = obs[:, env_idx, DEPTH_START_IDX:]

            # Identify wall pixels using shared helper
            is_wall_pixel = identify_wall_pixels_with_holes(depth_obs_env, tolerance=0.05)
            n_wall = is_wall_pixel.sum().item()

            if n_wall < 10:
                continue

            # For wall pixels, count observations at max_range (holes)
            wall_obs = depth_obs_env[:, is_wall_pixel]
            n_at_max = (wall_obs > 0.99).sum().item()
            n_total = wall_obs.numel()

            total_at_wall += n_total
            total_holes += n_at_max

            print(f"    Env {env_idx}: {n_wall} wall pixels, hole rate={n_at_max/n_total:.4f}")

        assert total_at_wall > 100, (
            f"Insufficient wall pixel observations: {total_at_wall}. "
            f"Check that test scene is correctly configured with wall at {TEST_WALL_DISTANCE}m."
        )

        observed_rate = total_holes / total_at_wall

        # Two-sided binomial test: is observed rate consistent with expected rate?
        # H0: true rate = TEST_HOLE_PROBABILITY
        # We fail to reject H0 if p-value > alpha (1 - CONFIDENCE_LEVEL)
        result = stats.binomtest(total_holes, total_at_wall, TEST_HOLE_PROBABILITY)
        alpha = 1 - CONFIDENCE_LEVEL

        print(f"    Results:")
        print(f"      Total wall pixel observations: {total_at_wall}")
        print(f"      Observed hole rate: {observed_rate:.4f}")
        print(f"      Expected rate: {TEST_HOLE_PROBABILITY}")
        print(f"      Binomial test p-value: {result.pvalue:.4f}")
        print(f"      Significance level (alpha): {alpha}")

        # Pass if we fail to reject H0 (p-value > alpha means rates are consistent)
        assert result.pvalue > alpha, (
            f"Hole rate mismatch: observed={observed_rate:.4f}, "
            f"expected={TEST_HOLE_PROBABILITY}, "
            f"p-value={result.pvalue:.4f} <= alpha={alpha}"
        )

    def test_hole_variance_on_wall_pixels(self, depth_env):
        """Verify first-difference variance from holes matches theory on wall pixels.

        For wall pixels at depth d = TEST_WALL_DEPTH_NORMALIZED (~0.333):
            jump_size = 1.0 - d = 0.667
            Var(diff) = 2 * p * (1-p) * jump_size^2

        STATISTICAL METHOD:
        With a single environment, we pool all wall pixel observations and use
        a chi-squared test for variance.
        """
        obs = collect_stationary_observations(depth_env, N_SAMPLES_STEPS)

        # Expected variance for wall pixels
        expected_var = hole_diff_variance(TEST_WALL_DEPTH_NORMALIZED, TEST_HOLE_PROBABILITY)

        print(f"\n  Hole variance test on wall pixels (d={TEST_WALL_DISTANCE}m):")
        print(f"    Analytical derivation:")
        print(f"      wall depth (normalized): {TEST_WALL_DEPTH_NORMALIZED:.3f}")
        print(f"      jump_size: {1.0 - TEST_WALL_DEPTH_NORMALIZED:.3f}")
        print(f"      hole_probability: {TEST_HOLE_PROBABILITY}")
        print(f"      expected_var: {expected_var:.6e}")

        # Pool all wall pixel differences across all environments
        all_diffs = []
        total_wall_pixels = 0

        for env_idx in range(obs.shape[1]):
            depth_obs_env = obs[:, env_idx, DEPTH_START_IDX:]
            first_diffs = depth_obs_env[1:] - depth_obs_env[:-1]

            # Identify wall pixels using shared helper
            is_wall_pixel = identify_wall_pixels_with_holes(depth_obs_env, tolerance=0.05)
            n_wall = is_wall_pixel.sum().item()

            if n_wall < 10:
                print(f"    Env {env_idx}: Only {n_wall} wall pixels found, skipping")
                continue

            diffs_at_wall = first_diffs[:, is_wall_pixel].flatten()
            all_diffs.append(diffs_at_wall)
            total_wall_pixels += n_wall
            print(f"    Env {env_idx}: {n_wall} wall pixels")

        assert len(all_diffs) > 0, (
            f"No environments with wall pixels found. "
            f"Check that test scene is correctly configured with wall at {TEST_WALL_DISTANCE}m."
        )

        # Combine all differences
        all_diffs_tensor = torch.cat(all_diffs)
        n_samples = all_diffs_tensor.numel()
        measured_var = all_diffs_tensor.var().item()
        ratio = measured_var / expected_var

        # Chi-squared test for variance
        df = n_samples - 1
        alpha = 1 - CONFIDENCE_LEVEL
        chi2_low = stats.chi2.ppf(alpha / 2, df)
        chi2_high = stats.chi2.ppf(1 - alpha / 2, df)
        ci_low = chi2_low / df
        ci_high = chi2_high / df

        print(f"    Results:")
        print(f"      Total samples: {n_samples}, wall pixels: {total_wall_pixels}")
        print(f"      Measured variance: {measured_var:.2e}")
        print(f"      Expected variance: {expected_var:.2e}")
        print(f"      Variance ratio: {ratio:.4f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for ratio: [{ci_low:.4f}, {ci_high:.4f}]")

        # Pass if ratio is within expected range
        assert ci_low <= ratio <= ci_high, (
            f"Hole noise variance mismatch: ratio={ratio:.4f} "
            f"outside {CONFIDENCE_LEVEL*100:.0f}% CI [{ci_low:.4f}, {ci_high:.4f}]"
        )

    def test_no_variance_at_max_range(self, depth_env):
        """Verify holes produce NO additional variance on pixels already at max_range.

        For pixels at true depth = max_range, holes set them to max_range which is
        their existing value. So first-differences should be zero (no noise effect).

        The robot is oriented to face AWAY from the wall, looking into open space
        where all pixels are at max_range.
        """
        # Face away from wall to see max-range pixels (open space)
        obs = collect_stationary_observations_facing_away(depth_env, N_SAMPLES_STEPS)

        print(f"\n  Holes at max_range test:")

        # Pool across environments
        total_max_pixels = 0
        total_variance = 0.0

        for env_idx in range(obs.shape[1]):
            depth_obs_env = obs[:, env_idx, DEPTH_START_IDX:]
            first_diffs = depth_obs_env[1:] - depth_obs_env[:-1]

            # Pixels always at max_range
            is_always_max = (depth_obs_env > 0.999).all(dim=0)
            n_always_max = is_always_max.sum().item()

            if n_always_max < 10:
                continue

            diffs_at_max = first_diffs[:, is_always_max]
            measured_var = diffs_at_max.var().item()
            total_max_pixels += n_always_max
            total_variance += measured_var * n_always_max

            print(f"    Env {env_idx}: {n_always_max} max-range pixels, var={measured_var:.2e}")

        if total_max_pixels < 10:
            pytest.skip(f"Insufficient max-range pixels across all envs: {total_max_pixels}")

        avg_variance = total_variance / total_max_pixels

        print(f"    Results:")
        print(f"      Total max-range pixels: {total_max_pixels}")
        print(f"      Average variance: {avg_variance:.2e}")
        print(f"      Expected: ~0 (holes have no effect on max-range pixels)")

        # With no Gaussian noise and no hole effect, variance should be essentially zero
        # Allow small numerical tolerance
        assert avg_variance < 1e-8, (
            f"Unexpected variance at max_range pixels: {avg_variance:.2e}. "
            f"Holes should have no effect on pixels already at max_range."
        )


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
