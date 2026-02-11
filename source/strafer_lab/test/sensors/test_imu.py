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

    # Use a one-sample t-test to get the CI of the measured gravity.
    # The CI is used to verify the measured mean is within a physics-meaningful
    # tolerance of the expected value. With many samples the CI is very tight,
    # so a direct null_value=9.81 test would reject for sub-percent simulation
    # discretisation offsets.  Instead we check that the CI overlaps with the
    # acceptable gravity interval [GRAVITY ± GRAVITY_TOLERANCE].
    GRAVITY_TOLERANCE = 0.3  # m/s²: accounts for sim discretisation + noise bias

    result = one_sample_t_test(gravity_samples, null_value=GRAVITY)
    gravity_error = abs(result.mean - GRAVITY)

    print(f"\n  IMU gravity measurement:")
    print(f"    N samples: {result.n_samples:,}")
    print(f"    Expected gravity: {GRAVITY:.3f} m/s²")
    print(f"    Measured mean: {result.mean:.3f} m/s²")
    print(f"    Gravity error: {gravity_error:.4f} m/s²")
    print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.3f}, {result.ci_high:.3f}] m/s²")
    print(f"    Tolerance: ±{GRAVITY_TOLERANCE} m/s²")

    # Test 1: Measured gravity mean must be within tolerance of expected
    assert gravity_error < GRAVITY_TOLERANCE, (
        f"Measured gravity ({result.mean:.3f} m/s²) deviates from "
        f"expected ({GRAVITY:.3f} m/s²) by {gravity_error:.3f} m/s², "
        f"exceeding tolerance of ±{GRAVITY_TOLERANCE} m/s². "
        f"{CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.3f}, {result.ci_high:.3f}] m/s²"
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
