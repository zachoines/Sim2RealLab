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

# Import shared utilities (must come after Isaac Sim launch)
from utils import (
    NUM_ENVS,
    N_SAMPLES_STEPS,
    N_SETTLE_STEPS,
    CONFIDENCE_LEVEL,
    DEPTH_START_IDX,
    TEST_WALL_DISTANCE,
    TEST_WALL_DEPTH_NORMALIZED,
    hole_diff_variance,
    identify_wall_pixels,
    get_geometric_wall_mask,
    collect_stationary_observations,
    create_depth_test_env,
    create_holes_only_noise_cfg,
    set_simulation_app,
    debug_camera_orientation,
    # Statistical utilities from common module
    binomial_test,
    variance_ratio_test,
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
        torch.manual_seed(42)
        np.random.seed(42)

        obs = collect_stationary_observations(depth_env, N_SAMPLES_STEPS)

        print(f"\n  Hole rate on wall pixels:")

        total_at_wall = 0
        total_holes = 0

        for env_idx in range(obs.shape[1]):
            depth_obs_env = obs[:, env_idx, DEPTH_START_IDX:]

            mask = identify_wall_pixels(depth_obs_env, tolerance=0.05, max_std=0.02)
            if mask.sum().item() < 50:
                # Fallback to geometric mask if statistical selection is too small
                mask = get_geometric_wall_mask(depth_env).to(depth_env.device)

            wall_obs = depth_obs_env[:, mask]
            n_at_max = (wall_obs > 0.99).sum().item()
            n_total = wall_obs.numel()

            total_at_wall += n_total
            total_holes += n_at_max

        assert total_at_wall > 100, (
            f"Insufficient wall pixel observations: {total_at_wall}. "
            f"Check that test scene is correctly configured with wall at {TEST_WALL_DISTANCE}m."
        )

        # Two-sided binomial test: is observed rate consistent with expected rate?
        # H0: true rate = TEST_HOLE_PROBABILITY
        # We fail to reject H0 if p-value > alpha (1 - CONFIDENCE_LEVEL)
        result = binomial_test(total_holes, total_at_wall, TEST_HOLE_PROBABILITY)

        print(f"    Summary:")
        print(f"      Parallel environments: {obs.shape[1]}")
        print(f"      Wall pixels (observations): {total_at_wall}")
        print(f"      Hole prob (expected): {TEST_HOLE_PROBABILITY}")
        print(f"      Hole prob (measured): {result.observed_rate:.6f}")
        print(f"      Binomial p-value: {result.p_value:.4f} (alpha={1 - CONFIDENCE_LEVEL})")

        if not result.reject_null:
            debug_camera_orientation(depth_env)

        # Pass if we fail to reject H0 (rates are consistent)
        assert not result.reject_null, (
            f"Hole rate mismatch: observed={result.observed_rate:.4f}, "
            f"expected={TEST_HOLE_PROBABILITY}, "
            f"p-value={result.p_value:.4f}"
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
        torch.manual_seed(42)
        np.random.seed(42)

        obs = collect_stationary_observations(
            depth_env, 
            N_SAMPLES_STEPS * 2,  # Extra steps for better stats 
            face_wall=True,
            n_settle_steps=N_SETTLE_STEPS,  
        )

        print(f"\n  Hole variance on wall pixels:")

        # Pool all wall pixel differences across all environments
        all_diffs = []
        total_wall_pixels = 0
        wall_depth_means = []

        for env_idx in range(obs.shape[1]):
            depth_obs_env = obs[:, env_idx, DEPTH_START_IDX:]
            first_diffs = depth_obs_env[1:] - depth_obs_env[:-1]

            mask = identify_wall_pixels(depth_obs_env, tolerance=0.05, max_std=0.02)
            n_wall = mask.sum().item()
            if n_wall < 50:
                mask = get_geometric_wall_mask(depth_env).to(depth_env.device)
                n_wall = mask.sum().item()

            if n_wall < 10:
                continue

            diffs_at_wall = first_diffs[:, mask].flatten()
            all_diffs.append(diffs_at_wall)
            total_wall_pixels += n_wall
            wall_depth_means.append((depth_obs_env[:, mask].mean()).item())

        assert len(all_diffs) > 0, (
            f"No environments with wall pixels found. "
            f"Check that test scene is correctly configured with wall at {TEST_WALL_DISTANCE}m."
        )

        # Combine all differences
        all_diffs_tensor = torch.cat(all_diffs)
        n_samples = all_diffs_tensor.numel()
        measured_var = all_diffs_tensor.var().item()

        # The observed MEAN depth is biased by holes:
        #   observed_mean = (1-p) * true_depth + p * 1.0
        # We need the TRUE depth (when no hole) for the variance formula.
        # Rearranging: true_depth = (observed_mean - p) / (1 - p)
        observed_mean_depth = float(torch.tensor(wall_depth_means).mean().item())
        p = TEST_HOLE_PROBABILITY
        true_wall_depth_norm = (observed_mean_depth - p) / (1 - p)

        # Use the true (unbiased) wall depth for expected variance
        expected_var = hole_diff_variance(true_wall_depth_norm, TEST_HOLE_PROBABILITY)

        # Chi-squared test for variance
        result = variance_ratio_test(measured_var, expected_var, n_samples)

        print(f"    Summary:")
        print(f"      Parallel environments: {obs.shape[1]}")
        print(f"      Wall pixels (total): {total_wall_pixels}")
        print(f"      Samples (diffs): {n_samples}")
        print(f"      Wall depth (observed mean): {observed_mean_depth:.4f} (biased by holes)")
        print(f"      Wall depth (true, unbiased): {true_wall_depth_norm:.4f}")
        print(f"      Variance (expected): {result.expected_var:.2e}")
        print(f"      Variance (measured): {result.measured_var:.2e}")
        print(f"      Variance ratio: {result.ratio:.4f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for ratio: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
        print(f"      In statistical CI: {result.in_ci}")

        if not result.in_ci:
            debug_camera_orientation(depth_env)

        # Pass if ratio is within expected range
        assert result.in_ci, (
            f"Hole noise variance mismatch: ratio={result.ratio:.4f} "
            f"outside {CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
        )
        
   


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
