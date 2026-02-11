# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for encoder noise model in isolation.

These tests validate the EncoderNoiseModel class directly without a
simulation environment. They verify:
- Velocity noise variance matches config (fraction × max_velocity)
- Quantization produces integer tick values
- Missed tick probability matches configured rate

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_encoder_noise_unit.py -v
"""

import torch
import numpy as np

from test.common import (
    CONFIDENCE_LEVEL,
    chi_squared_variance_test,
    binomial_test,
    DEVICE,
)

# -- imports resolved after AppLauncher (root conftest) --
from strafer_lab.tasks.navigation.mdp.noise_models import (
    EncoderNoiseModel,
    EncoderNoiseModelCfg,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 50_000
N_ENVS = 32

TEST_VEL_NOISE_FRACTION = 0.02
TEST_MAX_VELOCITY = 3000.0
TEST_TICKS_PER_RADIAN = 85.57
TEST_MISSED_TICK_PROB = 0.02  # Elevated for testability


# =============================================================================
# Tests
# =============================================================================


def test_velocity_noise_std():
    """Verify encoder noise variance matches configured std.

    The actual noise std = velocity_noise_std × max_velocity.
    With quantization and tick errors disabled, model(zeros) gives
    pure Gaussian noise. Single wheel tested to avoid inflated N.
    """
    expected_std = TEST_VEL_NOISE_FRACTION * TEST_MAX_VELOCITY
    cfg = EncoderNoiseModelCfg(
        velocity_noise_std=TEST_VEL_NOISE_FRACTION,
        max_velocity=TEST_MAX_VELOCITY,
        enable_quantization=False,
        ticks_per_radian=TEST_TICKS_PER_RADIAN,
        missed_tick_prob=0.0,
        extra_tick_prob=0.0,
        output_size=4,
        latency_steps=0,
    )
    model = EncoderNoiseModel(cfg, N_ENVS, DEVICE)

    # Use fewer samples for chi-squared to accommodate float32 GPU precision.
    n_chi2 = 5_000
    clean = torch.zeros(N_ENVS, 4, device=DEVICE)
    samples = []
    for _ in range(n_chi2):
        noisy = model(clean.clone())
        # Single env, single wheel to keep chi-squared df reasonable
        samples.append(noisy[0, 0].cpu().item())

    samples = np.array(samples)
    result = chi_squared_variance_test(samples, expected_std**2)

    print(f"\n  Encoder velocity noise test:")
    print(f"    Expected std: {expected_std:.2f} ticks/s")
    print(f"    Measured std: {np.std(samples):.2f} ticks/s")
    print(f"    Variance ratio: {result.ratio:.4f}")
    print(f"    In CI: {result.in_ci} [{result.ci_low:.4f}, {result.ci_high:.4f}]")

    assert result.in_ci, (
        f"Encoder noise variance doesn't match config. "
        f"Expected σ={expected_std:.2f}, got {np.std(samples):.2f} "
        f"(ratio={result.ratio:.4f}, CI=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
    )


def test_quantization_produces_integers():
    """Verify quantization rounds output to integer ticks.

    With quantization enabled, output values should be exact integers
    (within floating-point tolerance).
    """
    cfg = EncoderNoiseModelCfg(
        velocity_noise_std=TEST_VEL_NOISE_FRACTION,
        max_velocity=TEST_MAX_VELOCITY,
        enable_quantization=True,
        ticks_per_radian=TEST_TICKS_PER_RADIAN,
        missed_tick_prob=0.0,
        extra_tick_prob=0.0,
        output_size=4,
        latency_steps=0,
    )
    model = EncoderNoiseModel(cfg, N_ENVS, DEVICE)

    # Non-integer input to ensure quantization is actually acting
    clean = torch.full((N_ENVS, 4), 100.7, device=DEVICE)
    noisy = model(clean.clone())

    # After quantization, all values should be close to integers
    residuals = torch.abs(noisy - torch.round(noisy))
    max_residual = residuals.max().item()

    print(f"\n  Quantization test:")
    print(f"    Max residual from integer: {max_residual:.8f}")

    assert max_residual < 1e-5, (
        f"Quantization did not produce integers: max residual = {max_residual:.8f}"
    )


def test_missed_tick_probability():
    """Verify missed tick rate matches configured probability.

    Uses binomial test to verify the proportion of samples where the
    output differs from expected (input + noise, without quantization)
    by exactly ±1 tick (missed or extra tick).
    """
    cfg = EncoderNoiseModelCfg(
        velocity_noise_std=0.0,  # No Gaussian noise
        max_velocity=TEST_MAX_VELOCITY,
        enable_quantization=False,
        ticks_per_radian=TEST_TICKS_PER_RADIAN,
        missed_tick_prob=TEST_MISSED_TICK_PROB,
        extra_tick_prob=0.0,  # Only test missed ticks
        output_size=4,
        latency_steps=0,
    )
    model = EncoderNoiseModel(cfg, N_ENVS, DEVICE)

    # Constant input — any deviation is a missed tick
    clean = torch.full((N_ENVS, 4), 500.0, device=DEVICE)

    n_total = 0
    n_missed = 0
    for _ in range(N_SAMPLES):
        noisy = model(clean.clone())
        diff = (noisy - clean).abs()
        # Missed tick → output changed by any amount
        n_missed += int((diff > 0.5).sum().item())
        n_total += N_ENVS * 4

    result = binomial_test(n_missed, n_total, TEST_MISSED_TICK_PROB)

    print(f"\n  Missed tick probability test:")
    print(f"    Expected rate: {TEST_MISSED_TICK_PROB:.4f}")
    print(f"    Observed rate: {n_missed / n_total:.4f}")
    print(f"    n_total: {n_total}")
    print(f"    Binomial p-value: {result.p_value:.4f}")

    assert not result.reject_null, (
        f"Missed tick rate doesn't match config. "
        f"Expected {TEST_MISSED_TICK_PROB:.4f}, got {n_missed / n_total:.4f} "
        f"(p={result.p_value:.4f})"
    )
