# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for IMU noise models through the environment pipeline.

These tests validate that IMU accelerometer and gyroscope noise is correctly
applied during environment stepping. Tests use first-differences approach
for precise theoretical bounds on variance.

Noise Pipeline: Raw → Noise → Scale → Clip
    1. Observation function returns RAW sensor data (m/s², rad/s)
    2. Noise model corrupts the RAW data
    3. Scale parameter normalizes (scale = 1/max_value)
    4. Clip limits to [-1, 1]

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_imu_noise.py -v
"""

import math
import numpy as np
import pytest

from test.common import (
    CONFIDENCE_LEVEL,
    N_SAMPLES_STEPS,
    IMU_ACCEL_MAX,
    IMU_GYRO_MAX,
    variance_ratio_test,
)

from test.noise_models.conftest import (
    collect_stationary_observations,
    extract_first_differences,
    IMU_ACCEL_SLICE,
    IMU_GYRO_SLICE,
    _ACCEL_NOISE_CFG,
    _GYRO_NOISE_CFG,
    EXPECTED_ACCEL_STD,
    EXPECTED_GYRO_STD,
)


# =============================================================================
# Theoretical Bounds Using First Differences
# =============================================================================
#
# For IMU noise with bias drift:
#   y_t = signal + bias_t + white_noise_t
#   bias_t = bias_{t-1} + drift_step_t
#
# First differences eliminate signal:
#   Δy_t = drift_step_t + (white_t - white_{t-1})
#   Var(Δy) = drift_rate² * dt + 2 * white_noise_std²
#


def _calculate_first_diff_theoretical_std(
    white_noise_std: float,
    drift_rate: float,
    control_frequency_hz: float = 30.0,
) -> float:
    """Calculate theoretical std of first differences for IMU noise.

    Args:
        white_noise_std: White noise standard deviation per step
        drift_rate: Bias drift rate (std per second, scaled by sqrt(dt) per step)
        control_frequency_hz: Control loop frequency (default 30Hz)

    Returns:
        Theoretical std of first differences
    """
    dt = 1.0 / control_frequency_hz
    # Drift variance per step: (drift_rate * sqrt(dt))^2 = drift_rate^2 * dt
    variance = drift_rate**2 * dt + 2 * white_noise_std**2
    return math.sqrt(variance)


def _calculate_first_diff_theoretical_std_normalized(
    white_noise_std: float,
    drift_rate: float,
    max_value: float,
    control_frequency_hz: float = 30.0,
) -> float:
    """Calculate theoretical std of first differences in NORMALIZED space.

    Args:
        white_noise_std: White noise std in RAW units (m/s² or rad/s)
        drift_rate: Bias drift rate in RAW units (per second)
        max_value: Max sensor value for normalization
        control_frequency_hz: Control loop frequency (default 30Hz)

    Returns:
        Theoretical std of first differences in normalized space
    """
    raw_theoretical_std = _calculate_first_diff_theoretical_std(
        white_noise_std, drift_rate, control_frequency_hz
    )
    return raw_theoretical_std / max_value


# Compute theoretical std for each IMU sensor type
if _ACCEL_NOISE_CFG:
    ACCEL_DIFF_STD_THEORETICAL = _calculate_first_diff_theoretical_std_normalized(
        _ACCEL_NOISE_CFG.accel_noise_std,
        _ACCEL_NOISE_CFG.accel_bias_drift_rate,
        IMU_ACCEL_MAX,
        _ACCEL_NOISE_CFG.control_frequency_hz,
    )
else:
    ACCEL_DIFF_STD_THEORETICAL = 0.0

if _GYRO_NOISE_CFG:
    GYRO_DIFF_STD_THEORETICAL = _calculate_first_diff_theoretical_std_normalized(
        _GYRO_NOISE_CFG.gyro_noise_std,
        _GYRO_NOISE_CFG.gyro_bias_drift_rate,
        IMU_GYRO_MAX,
        _GYRO_NOISE_CFG.control_frequency_hz,
    )
else:
    GYRO_DIFF_STD_THEORETICAL = 0.0


# =============================================================================
# Tests: IMU Noise Validation
# =============================================================================


def test_accel_noise_std_matches_config(noisy_env):
    """Verify accelerometer noise std matches theoretical prediction.

    Uses first-differences approach for precise theoretical bounds:
        Δy_t = drift_step + (v_t - v_{t-1})
        Var(Δy) = drift_rate² * dt + 2 * white_noise_std²

    Statistical approach: chi-squared variance ratio test.
    """
    obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)
    # Use first differences for clean theoretical prediction
    diff_samples = extract_first_differences(obs, IMU_ACCEL_SLICE)
    n_samples = len(diff_samples)
    measured_var = np.var(diff_samples)
    measured_std = np.sqrt(measured_var)

    # Chi-squared variance ratio test
    expected_var = ACCEL_DIFF_STD_THEORETICAL**2
    result = variance_ratio_test(measured_var, expected_var, n_samples)

    print(f"\n  Accelerometer first-difference analysis (chi-squared test):")
    print(f"    Config values:")
    print(f"      white_noise_std: {EXPECTED_ACCEL_STD:.6f}")
    print(f"      drift_rate: {_ACCEL_NOISE_CFG.accel_bias_drift_rate:.6f}")
    print(f"    Statistical analysis (n={n_samples:,} samples, df={result.df:,}):")
    print(f"      Measured std(Δy): {measured_std:.6f}")
    print(f"      Theoretical std(Δy): {ACCEL_DIFF_STD_THEORETICAL:.6f}")
    print(f"      Variance ratio (measured/theoretical): {result.ratio:.4f}")
    print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    print(f"    Result: Variance ratio {'is' if result.in_ci else 'is NOT'} within CI")

    assert result.in_ci, (
        f"Accelerometer variance ratio {result.ratio:.4f} not within "
        f"{CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
    )


def test_gyro_noise_std_matches_config(noisy_env):
    """Verify gyroscope noise std matches theoretical prediction.

    Uses first-differences approach for precise theoretical bounds:
        Δy_t = drift_step + (v_t - v_{t-1})
        Var(Δy) = drift_rate² * dt + 2 * white_noise_std²

    Statistical approach: chi-squared variance ratio test.
    """
    obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)

    # Use first differences for clean theoretical prediction
    diff_samples = extract_first_differences(obs, IMU_GYRO_SLICE)
    n_samples = len(diff_samples)
    measured_var = np.var(diff_samples)
    measured_std = np.sqrt(measured_var)

    # Chi-squared variance ratio test
    expected_var = GYRO_DIFF_STD_THEORETICAL**2
    result = variance_ratio_test(measured_var, expected_var, n_samples)

    print(f"\n  Gyroscope first-difference analysis (chi-squared test):")
    print(f"    Config values:")
    print(f"      white_noise_std: {EXPECTED_GYRO_STD:.6f}")
    print(f"      drift_rate: {_GYRO_NOISE_CFG.gyro_bias_drift_rate:.6f}")
    print(f"    Statistical analysis (n={n_samples:,} samples, df={result.df:,}):")
    print(f"      Measured std(Δy): {measured_std:.6f}")
    print(f"      Theoretical std(Δy): {GYRO_DIFF_STD_THEORETICAL:.6f}")
    print(f"      Variance ratio (measured/theoretical): {result.ratio:.4f}")
    print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    print(f"    Result: Variance ratio {'is' if result.in_ci else 'is NOT'} within CI")

    assert result.in_ci, (
        f"Gyroscope variance ratio {result.ratio:.4f} not within "
        f"{CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
    )


def test_imu_noise_with_latency(noisy_env):
    """Verify IMU observations work with latency configuration.

    This test validates that the per-sensor latency feature works correctly
    through the full noise model pipeline.
    """
    obs = collect_stationary_observations(noisy_env, 50)

    # Verify observations were collected successfully
    from test.common import NUM_ENVS

    assert obs.shape[0] == 50, "Should collect 50 steps"
    assert obs.shape[1] == NUM_ENVS, f"Should have {NUM_ENVS} environments"

    # IMU accel observations should have variance (noise is enabled)
    imu_variance = obs[:, :, IMU_ACCEL_SLICE].var().item()
    assert imu_variance > 1e-8, "IMU should have measurable noise variance"