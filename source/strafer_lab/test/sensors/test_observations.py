# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for observation pipeline structure and physics correctness.

These tests validate that:
- Observation structure matches expected dimensions
- Physics signals (like gravity) are correctly measured
- Noise is unbiased (zero mean)

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/sensors/test_observations.py -v
"""

import numpy as np
import torch
import pytest

from test.common import (
    CONFIDENCE_LEVEL,
    N_SAMPLES_STEPS,
    IMU_ACCEL_MAX,
    one_sample_t_test,
)

from test.sensors.conftest import (
    collect_stationary_observations,
    extract_noise_samples,
    IMU_ACCEL_SLICE,
)


# =============================================================================
# Constants
# =============================================================================

# Physical constants
GRAVITY = 9.81  # m/s²

# Expected observation structure for NoCam config
# imu_accel(3) + imu_gyro(3) + encoders(4) + goal(2) + action(3) = 15
EXPECTED_OBS_DIMS = [(3,), (3,), (4,), (2,), (3,)]
EXPECTED_TOTAL_DIM = 15


# =============================================================================
# Tests: Observation Pipeline Structure
# =============================================================================


def test_observation_structure(noisy_env):
    """Verify observation term structure matches expected dimensions.

    Observation structure for NoCam config (15 dims total):
    - imu_linear_acceleration: (3,) - normalized by max_accel
    - imu_angular_velocity: (3,) - normalized by max_angular_vel
    - wheel_encoder_velocities: (4,) - normalized by max_ticks_per_sec
    - goal_position: (2,) - relative [x, y] to goal (meters)
    - last_action: (3,) - previous [vx, vy, omega] command
    """
    obs_manager = noisy_env.observation_manager

    if hasattr(obs_manager, "_group_obs_term_dim"):
        group_cfg = obs_manager._group_obs_term_dim

        assert "policy" in group_cfg, "Missing 'policy' observation group"

        term_dims = group_cfg["policy"]

        print(f"\n  Observation structure validation:")
        print(f"    Number of terms: {len(term_dims)} (expected {len(EXPECTED_OBS_DIMS)})")

        assert len(term_dims) == len(EXPECTED_OBS_DIMS), (
            f"Expected {len(EXPECTED_OBS_DIMS)} terms, got {len(term_dims)}"
        )

        term_names = ["imu_accel", "imu_gyro", "encoders", "goal", "action"]
        for i, (actual, expected) in enumerate(zip(term_dims, EXPECTED_OBS_DIMS)):
            print(f"    Term {i} ({term_names[i]}): {actual} (expected {expected})")
            assert actual == expected, f"Term {i}: expected {expected}, got {actual}"

        total = sum(d[0] for d in term_dims)
        print(f"    Total dimensions: {total} (expected {EXPECTED_TOTAL_DIM})")
        assert total == EXPECTED_TOTAL_DIM, f"Expected {EXPECTED_TOTAL_DIM} total dims, got {total}"
    else:
        # Fallback: verify total dimension from actual observation
        obs_dict, _ = noisy_env.reset()
        total_dim = obs_dict["policy"].shape[1]
        print(f"\n  Observation total dimension: {total_dim} (expected {EXPECTED_TOTAL_DIM})")
        assert total_dim == EXPECTED_TOTAL_DIM, f"Expected {EXPECTED_TOTAL_DIM} dims, got {total_dim}"


def test_imu_gravity(noisy_env):
    """Verify IMU measures gravity correctly when robot is stationary.

    When the robot is stationary and upright, the accelerometer should measure
    approximately 9.81 m/s² total acceleration (from gravity). The Z-axis
    should dominate since the robot is upright.

    Uses one-sample t-test to verify the measured gravity magnitude is
    consistent with the expected value.
    """
    obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)

    # Extract and de-normalize IMU readings
    imu_accel_normalized = obs[:, :, IMU_ACCEL_SLICE]  # (n_steps, num_envs, 3)
    imu_accel_raw = imu_accel_normalized * IMU_ACCEL_MAX  # Convert to m/s²

    # Compute magnitude for each sample
    magnitudes = torch.norm(imu_accel_raw, dim=2)  # (n_steps, num_envs)
    gravity_samples = magnitudes.cpu().numpy().flatten()

    # One-sample t-test: is measured gravity consistent with expected?
    result = one_sample_t_test(gravity_samples, null_value=GRAVITY)

    print(f"\n  IMU gravity measurement (one-sample t-test):")
    print(f"    N samples: {result.n_samples:,}")
    print(f"    Expected gravity: {GRAVITY:.3f} m/s²")
    print(f"    Measured mean: {result.mean:.3f} m/s²")
    print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.3f}, {result.ci_high:.3f}] m/s²")
    print(f"    p-value: {result.p_value:.4f}")
    print(f"    Gravity in CI: {not result.reject_null}")

    # Test 1: Gravity should be within CI or close to expected
    gravity_error = abs(result.mean - GRAVITY)
    max_acceptable_error = 0.5  # Allow 0.5 m/s² error (5% of g)

    assert gravity_error < max_acceptable_error or not result.reject_null, (
        f"Gravity measurement inconsistent: mean={result.mean:.3f} m/s², "
        f"expected={GRAVITY:.3f} m/s², error={gravity_error:.3f} m/s²"
    )

    # Test 2: Z-axis should dominate (robot is upright)
    mean_components = imu_accel_raw.mean(dim=(0, 1)).cpu().numpy()
    xy_magnitude = np.sqrt(mean_components[0] ** 2 + mean_components[1] ** 2)
    z_magnitude = abs(mean_components[2])

    print(f"    Component analysis:")
    print(f"      Mean X: {mean_components[0]:.3f} m/s²")
    print(f"      Mean Y: {mean_components[1]:.3f} m/s²")
    print(f"      Mean Z: {mean_components[2]:.3f} m/s²")
    print(f"      XY magnitude: {xy_magnitude:.3f} m/s²")
    print(f"      Z magnitude: {z_magnitude:.3f} m/s²")

    # Z should be much larger than XY for an upright robot
    assert z_magnitude > xy_magnitude, (
        f"Robot appears tilted: Z={z_magnitude:.2f} m/s², XY={xy_magnitude:.2f} m/s²"
    )


# =============================================================================
# Tests: Statistical Properties
# =============================================================================


def test_noise_has_zero_mean(noisy_env):
    """Verify noise has approximately zero mean (unbiased).

    Uses one-sample t-test to verify the mean is not significantly
    different from zero at the configured confidence level.
    """
    obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)

    # Extract noise for accelerometer
    noise_samples = extract_noise_samples(obs, IMU_ACCEL_SLICE)

    # One-sample t-test: is mean significantly different from 0?
    result = one_sample_t_test(noise_samples, null_value=0.0)

    std = np.std(noise_samples)

    print(f"\n  Accelerometer noise mean test (one-sample t-test):")
    print(f"    N samples: {result.n_samples:,}")
    print(f"    Mean: {result.mean:.6f}")
    print(f"    Std: {std:.6f}")
    print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI for mean: [{result.ci_low:.2e}, {result.ci_high:.2e}]")
    print(f"    p-value: {result.p_value:.4f}")
    print(f"    Reject null (mean ≠ 0): {result.reject_null}")

    # Test passes if we fail to reject the null hypothesis (mean = 0)
    assert not result.reject_null, (
        f"Noise mean {result.mean:.6f} is significantly different from zero "
        f"(p={result.p_value:.4f} < α={1 - CONFIDENCE_LEVEL:.2f})"
    )
