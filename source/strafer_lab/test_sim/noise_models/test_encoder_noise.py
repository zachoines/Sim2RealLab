# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for encoder noise models through the environment pipeline.

These tests validate that wheel encoder noise is correctly applied during
environment stepping, and that noise configuration settings are respected.

Encoder Noise Components:
    1. Gaussian noise: std = velocity_noise_std * max_velocity
    2. Quantization: rounds to nearest integer (adds uniform noise variance 1/12)

The policy's encoder term reads the shared wheel reading, which carries this noise once per control step.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test_sim/noise_models/test_encoder_noise.py -v
"""

import math
import numpy as np
import pytest

from test_sim.common import (
    N_SAMPLES_STEPS,
    variance_ratio_test,
)
from test_sim.noise_models.seeded_stream import CASE_ALPHA, CASE_CONFIDENCE, seeded_stream

from test_sim.noise_models.conftest import (
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
    enable_quantization: bool,
) -> float:
    """Calculate theoretical std for encoder noise in NORMALIZED space.

    Args:
        velocity_noise_std: Fractional noise (e.g., 0.02 = 2%)
        max_velocity: Max velocity for scaling (e.g., 3000 ticks/sec)
        enable_quantization: Whether quantization is enabled

    Returns:
        Theoretical std of encoder noise in normalized space
    """
    # RAW space variances (in ticks² or ticks²/s²)
    gaussian_variance_raw = (velocity_noise_std * max_velocity) ** 2
    quantization_variance_raw = (1.0 / 12.0) if enable_quantization else 0.0

    total_variance_raw = gaussian_variance_raw + quantization_variance_raw
    std_raw = math.sqrt(total_variance_raw)

    # Convert to normalized space: std_normalized = std_raw / max_velocity
    std_normalized = std_raw / max_velocity
    return std_normalized


# Compute encoder theoretical std
if _ENCODER_NOISE_CFG:
    ENCODER_STD_THEORETICAL = _calculate_encoder_theoretical_std(
        _ENCODER_NOISE_CFG.velocity_noise_std,
        _ENCODER_NOISE_CFG.max_velocity,
        _ENCODER_NOISE_CFG.enable_quantization,
    )
else:
    ENCODER_STD_THEORETICAL = 0.0


# =============================================================================
# Tests: Encoder Noise Validation
# =============================================================================


@seeded_stream(201)
def test_encoder_noise_std_matches_config(noisy_env):
    """Verify encoder velocity noise std matches theoretical prediction.

    EncoderNoiseModel applies:
    1. Gaussian noise: std = velocity_noise_std * max_velocity
    2. Quantization: rounds to nearest integer (adds uniform noise variance 1/12)

    Note: Unlike IMU noise, encoder noise has NO drift component,
    so we can directly measure std without using first-differences.
    Tested on its own stream at a nominal alpha 1e-6: the samples are
    deviations from 256 per-(env, wheel) means, so its true size is 7.7e-6.
    At 102 400 samples a sigma 1.4 % low fails with power 0.98, and one
    1.4 % high with power 0.80.
    """
    obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)

    noise_samples = extract_noise_samples(obs, ENCODER_SLICE)
    n_samples = len(noise_samples)
    measured_var = np.var(noise_samples)
    measured_std = np.sqrt(measured_var)

    # Chi-squared variance ratio test
    expected_var = ENCODER_STD_THEORETICAL**2
    result = variance_ratio_test(measured_var, expected_var, n_samples, confidence_level=CASE_CONFIDENCE)

    # Get config values for display
    vel_noise_std = _ENCODER_NOISE_CFG.velocity_noise_std
    max_vel = _ENCODER_NOISE_CFG.max_velocity
    quant_enabled = _ENCODER_NOISE_CFG.enable_quantization

    # Compute variance components for display
    gaussian_var = (vel_noise_std * max_vel) ** 2
    quant_var = (1.0 / 12.0) if quant_enabled else 0.0
    total_var_raw = gaussian_var + quant_var

    print(f"\n  Encoder noise analysis (chi-squared test):")
    print(f"    Config values:")
    print(f"      velocity_noise_std: {vel_noise_std:.6f}")
    print(f"      max_velocity: {max_vel:.1f}")
    print(f"      enable_quantization: {quant_enabled}")
    print(f"    Theoretical variance breakdown (raw space):")
    print(f"      Gaussian: ({vel_noise_std:.4f} × {max_vel:.0f})² = {gaussian_var:.6f}")
    print(f"      Quantization: 1/12 = {quant_var:.6f}")
    print(f"      Total variance (raw): {total_var_raw:.6f}")
    print(f"    Statistical analysis (n={n_samples:,} samples, df={result.df:,}):")
    print(f"      Measured std: {measured_std:.6f}")
    print(f"      Theoretical std: {ENCODER_STD_THEORETICAL:.6f}")
    print(f"      Variance ratio (measured/theoretical): {result.ratio:.4f}")
    print(f"      CI at alpha {CASE_ALPHA:.0e}: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    print(f"    Result: Variance ratio {'is' if result.in_ci else 'is NOT'} within CI")

    assert result.in_ci, (
        f"Encoder variance ratio {result.ratio:.4f} not within "
        f"the CI at alpha {CASE_ALPHA:.0e} [{result.ci_low:.4f}, {result.ci_high:.4f}]"
    )



