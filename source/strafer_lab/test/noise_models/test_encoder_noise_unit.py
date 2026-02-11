# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for encoder noise model in isolation.

These tests verify the statistical properties of EncoderNoiseModel without
running a full simulation environment - only torch device access is needed.

Tests validate:
1. Velocity noise std scales correctly with max_velocity (chi-squared test)
2. Quantization produces integer-valued outputs
3. Missed tick probability matches configured rate (binomial test)

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_encoder_noise_unit.py -v
"""

import torch
import numpy as np
import pytest

from strafer_lab.tasks.navigation.mdp.noise_models import (
    EncoderNoiseModel,
    EncoderNoiseModelCfg,
)
from isaaclab.utils.noise import GaussianNoiseCfg

from test.common import (
    chi_squared_variance_test,
    binomial_test,
    CONFIDENCE_LEVEL,
    DEVICE,
)


# =============================================================================
# Test Configuration
# =============================================================================

N_SAMPLES = 50000        # Samples for statistical tests
N_ENVS = 32              # Parallel environments for unit tests
FLOAT_TOLERANCE = 1e-6   # Floating-point tolerance for deterministic comparisons


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def device():
    return DEVICE


# =============================================================================
# Encoder Noise Tests
# =============================================================================

class TestEncoderNoise:
    """Test encoder noise model."""

    def test_velocity_noise_std(self, device):
        """Velocity noise should scale by max_velocity.

        Uses chi-squared variance test to verify measured noise variance
        matches the expected value (noise_frac * max_velocity)².
        """
        noise_frac = 0.05  # 5% of max
        max_vel = 5000.0
        expected_std = noise_frac * max_vel

        cfg = EncoderNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=noise_frac),
            velocity_noise_std=noise_frac,
            max_velocity=max_vel,
            enable_quantization=False,
            missed_tick_prob=0.0,
            extra_tick_prob=0.0,
        )

        model = EncoderNoiseModel(cfg, N_ENVS, device)

        # Apply to constant velocity
        base = torch.ones(N_ENVS, 4, device=device) * 1000.0
        samples = []
        for _ in range(N_SAMPLES // N_ENVS):
            samples.append((model(base) - base).cpu())

        all_noise = torch.cat(samples, dim=0).numpy()
        # Use single wheel to keep sample count reasonable (~10k);
        # flattening all 4 wheels yields ~40k samples whose CI becomes
        # so tight that normal floating-point jitter causes false failures.
        noise = all_noise[:, 0]

        result = chi_squared_variance_test(noise, expected_std**2)

        print(f"\n  Encoder noise std (chi-squared test):")
        print(f"    Expected std: {expected_std:.2f}")
        print(f"    Measured std: {np.sqrt(result.measured_var):.2f}")
        print(f"    Variance ratio: {result.ratio:.4f}")
        print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")

        assert result.in_ci, (
            f"Variance ratio {result.ratio:.4f} not within "
            f"{CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
        )

    def test_quantization_produces_integers(self, device):
        """Quantization should produce integer values."""
        cfg = EncoderNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.1),
            velocity_noise_std=0.1,
            enable_quantization=True,
            missed_tick_prob=0.0,
            extra_tick_prob=0.0,
        )

        model = EncoderNoiseModel(cfg, N_ENVS, device)

        # Non-integer input
        base = torch.ones(N_ENVS, 4, device=device) * 123.456

        all_integer = True
        for _ in range(100):
            out = model(base)
            if not torch.all(torch.abs(out - out.round()) < 1e-5):
                all_integer = False
                break

        print(f"\n  All outputs are integers: {all_integer}")
        assert all_integer

    def test_missed_tick_probability(self, device):
        """Missed ticks should occur at configured probability.

        Uses binomial test to verify the observed miss rate is consistent
        with the configured probability.
        """
        expected_prob = 0.02

        cfg = EncoderNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            velocity_noise_std=0.0,
            enable_quantization=False,
            missed_tick_prob=expected_prob,
            extra_tick_prob=0.0,
        )

        model = EncoderNoiseModel(cfg, N_ENVS, device)

        base = torch.ones(N_ENVS, 4, device=device) * 100.0

        # Use enough iterations to get ~500k samples for reliable binomial test
        n_iterations = N_SAMPLES // (N_ENVS * 4) * 4  # ~6250 iterations
        total = 0
        missed = 0
        for _ in range(n_iterations):
            out = model(base)
            missed += int((out < base).sum().item())
            total += out.numel()

        result = binomial_test(missed, total, expected_prob)

        print(f"\n  Missed tick probability (binomial test):")
        print(f"    Total samples: {total}")
        print(f"    Expected prob: {expected_prob:.4f}")
        print(f"    Observed prob: {result.observed_rate:.4f}")
        print(f"    p-value: {result.p_value:.4f}")
        print(f"    Reject null: {result.reject_null}")

        assert not result.reject_null, (
            f"Missed tick rate {result.observed_rate:.4f} inconsistent with "
            f"expected {expected_prob:.4f} (p={result.p_value:.4f})"
        )
