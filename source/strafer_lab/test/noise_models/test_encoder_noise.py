# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for encoder noise models through the environment pipeline.

These tests validate that wheel encoder noise is correctly applied during
environment stepping, and that noise configuration settings are respected.

Encoder Noise Components:
    1. Gaussian noise: std = velocity_noise_std * max_velocity
    2. Missed ticks: discrete ±1 errors with probability p_miss
    3. Extra ticks: discrete ±1 errors with probability p_extra
    4. Quantization: rounds to nearest integer (adds uniform noise variance 1/12)

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_encoder_noise.py -v
"""

import math
import numpy as np
import pytest

from test.common import (
    CONFIDENCE_LEVEL,
    N_SAMPLES_STEPS,
    variance_ratio_test,
)

from test.noise_models.conftest import (
    collect_stationary_observations,
    extract_noise_samples,
    ENCODER_SLICE,
    _ENCODER_NOISE_CFG,
)


# =============================================================================
# Encoder Noise Theoretical Bounds
# =============================================================================


def _calculate_encoder_theoretical_std(
    velocity_noise_std: float,
    max_velocity: float,
    missed_tick_prob: float,
    extra_tick_prob: float,
    enable_quantization: bool,
) -> float:
    """Calculate theoretical std for encoder noise in NORMALIZED space.

    Args:
        velocity_noise_std: Fractional noise (e.g., 0.02 = 2%)
        max_velocity: Max velocity for scaling (e.g., 3000 ticks/sec)
        missed_tick_prob: Probability of missing a tick
        extra_tick_prob: Probability of extra tick
        enable_quantization: Whether quantization is enabled

    Returns:
        Theoretical std of encoder noise in normalized space
    """
    # RAW space variances (in ticks² or ticks²/s²)
    gaussian_variance_raw = (velocity_noise_std * max_velocity) ** 2
    tick_error_variance_raw = (missed_tick_prob + extra_tick_prob) * 1.0
    quantization_variance_raw = (1.0 / 12.0) if enable_quantization else 0.0

    total_variance_raw = gaussian_variance_raw + tick_error_variance_raw + quantization_variance_raw
    std_raw = math.sqrt(total_variance_raw)

    # Convert to normalized space: std_normalized = std_raw / max_velocity
    std_normalized = std_raw / max_velocity
    return std_normalized


# Compute encoder theoretical std
if _ENCODER_NOISE_CFG:
    ENCODER_STD_THEORETICAL = _calculate_encoder_theoretical_std(
        _ENCODER_NOISE_CFG.velocity_noise_std,
        _ENCODER_NOISE_CFG.max_velocity,
        _ENCODER_NOISE_CFG.missed_tick_prob,
        _ENCODER_NOISE_CFG.extra_tick_prob,
        _ENCODER_NOISE_CFG.enable_quantization,
    )
else:
    ENCODER_STD_THEORETICAL = 0.0


# =============================================================================
# Tests: Encoder Noise Validation
# =============================================================================


def test_encoder_noise_std_matches_config(noisy_env):
    """Verify encoder velocity noise std matches theoretical prediction.

    EncoderNoiseModel applies:
    1. Gaussian noise: std = velocity_noise_std * max_velocity
    2. Missed ticks: discrete ±1 errors with probability p_miss
    3. Extra ticks: discrete ±1 errors with probability p_extra
    4. Quantization: rounds to nearest integer (adds uniform noise variance 1/12)

    Note: Unlike IMU noise, encoder noise has NO drift component,
    so we can directly measure std without using first-differences.
    """
    obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)

    noise_samples = extract_noise_samples(obs, ENCODER_SLICE)
    n_samples = len(noise_samples)
    measured_var = np.var(noise_samples)
    measured_std = np.sqrt(measured_var)

    # Chi-squared variance ratio test
    expected_var = ENCODER_STD_THEORETICAL**2
    result = variance_ratio_test(measured_var, expected_var, n_samples)

    # Get config values for display
    vel_noise_std = _ENCODER_NOISE_CFG.velocity_noise_std
    max_vel = _ENCODER_NOISE_CFG.max_velocity
    p_miss = _ENCODER_NOISE_CFG.missed_tick_prob
    p_extra = _ENCODER_NOISE_CFG.extra_tick_prob
    quant_enabled = _ENCODER_NOISE_CFG.enable_quantization

    # Compute variance components for display
    gaussian_var = (vel_noise_std * max_vel) ** 2
    tick_var = p_miss + p_extra
    quant_var = (1.0 / 12.0) if quant_enabled else 0.0
    total_var_raw = gaussian_var + tick_var + quant_var

    print(f"\n  Encoder noise analysis (chi-squared test):")
    print(f"    Config values:")
    print(f"      velocity_noise_std: {vel_noise_std:.6f}")
    print(f"      max_velocity: {max_vel:.1f}")
    print(f"      missed_tick_prob: {p_miss:.6f}")
    print(f"      extra_tick_prob: {p_extra:.6f}")
    print(f"      enable_quantization: {quant_enabled}")
    print(f"    Theoretical variance breakdown (raw space):")
    print(f"      Gaussian: ({vel_noise_std:.4f} × {max_vel:.0f})² = {gaussian_var:.6f}")
    print(f"      Tick errors: {p_miss:.6f} + {p_extra:.6f} = {tick_var:.6f}")
    print(f"      Quantization: 1/12 = {quant_var:.6f}")
    print(f"      Total variance (raw): {total_var_raw:.6f}")
    print(f"    Statistical analysis (n={n_samples:,} samples, df={result.df:,}):")
    print(f"      Measured std: {measured_std:.6f}")
    print(f"      Theoretical std: {ENCODER_STD_THEORETICAL:.6f}")
    print(f"      Variance ratio (measured/theoretical): {result.ratio:.4f}")
    print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    print(f"    Result: Variance ratio {'is' if result.in_ci else 'is NOT'} within CI")

    assert result.in_ci, (
        f"Encoder variance ratio {result.ratio:.4f} not within "
        f"{CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
    )


def test_encoder_noise_with_latency(noisy_env):
    """Verify encoder observations work with latency configuration.

    Since REAL_ROBOT_CONTRACT has encoder_latency_steps=0 by default,
    this test verifies the noise model works correctly without latency.
    """
    obs = collect_stationary_observations(noisy_env, 50)

    # Encoder observations should have variance (noise is enabled)
    encoder_variance = obs[:, :, ENCODER_SLICE].var().item()
    assert encoder_variance > 1e-8, "Encoder should have measurable noise variance"
