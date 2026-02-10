# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for IMU sensor physics correctness.

These tests validate that IMU sensors measure physical quantities correctly,
such as gravity when the robot is stationary.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/sensors/test_imu.py -v
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
    IMU_ACCEL_SLICE,
)


# =============================================================================
# Constants
# =============================================================================

# Physical constants
GRAVITY = 9.81  # m/s²


# =============================================================================
# Tests: IMU Physics Validation
# =============================================================================


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
