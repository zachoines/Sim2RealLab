# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RGB camera noise model in isolation.

These tests verify the statistical properties of RGBNoiseModel without
running a full simulation environment - only torch device access is needed.

Tests validate:
1. Pixel noise std matches configured value (chi-squared test)
2. Brightness variation stays within configured range (deterministic bounds)
3. Output is clamped to valid [0, 1] range

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_rgb_noise.py -v
"""

import torch
import numpy as np
import pytest

from strafer_lab.tasks.navigation.mdp.noise_models import (
    RGBNoiseModel,
    RGBNoiseModelCfg,
)
from isaaclab.utils.noise import GaussianNoiseCfg

from test.common import (
    chi_squared_variance_test,
    CONFIDENCE_LEVEL,
    DEVICE,
)


# =============================================================================
# Test Configuration
# =============================================================================

N_SAMPLES = 50000        # Samples for statistical tests
N_ENVS = 32              # Parallel environments for unit tests
FLOAT_TOLERANCE = 1e-6   # Floating-point tolerance for deterministic comparisons

# RGB-specific test parameters
TEST_PIXEL_NOISE_STD = 0.04


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def device():
    return DEVICE


# =============================================================================
# RGB Camera Noise Tests
# =============================================================================

class TestRGBNoise:
    """Test RGB camera noise model.

    Note: RGBNoiseModel works on [0,1] RGB data (not mean-centered).
    The observation function returns [0,1] directly from uint8/255.0 conversion.
    """

    def test_pixel_noise_std(self, device):
        """Pixel noise should match configured std.

        Uses chi-squared variance test to verify measured noise variance
        matches the configured pixel noise standard deviation.
        """
        cfg = RGBNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=TEST_PIXEL_NOISE_STD),
            pixel_noise_std=TEST_PIXEL_NOISE_STD,
            brightness_range=(1.0, 1.0),  # No brightness variation
            frame_drop_prob=0.0,
            latency_steps=0,  # Disable latency to avoid delay buffer warmup
        )

        model = RGBNoiseModel(cfg, N_ENVS, device)

        # Input is [0,1] RGB data
        base = torch.ones(N_ENVS, 100, device=device) * 0.5

        n_iterations = N_SAMPLES // N_ENVS
        samples = []
        for _ in range(n_iterations):
            samples.append((model(base) - 0.5).cpu())

        all_noise = torch.cat(samples, dim=0).numpy()
        # Use single pixel column to keep sample count at ~N_SAMPLES
        noise = all_noise[:, 0]

        result = chi_squared_variance_test(noise, TEST_PIXEL_NOISE_STD**2)

        print(f"\n  Pixel noise std (chi-squared test):")
        print(f"    Expected std: {TEST_PIXEL_NOISE_STD:.4f}")
        print(f"    Measured std: {np.sqrt(result.measured_var):.4f}")
        print(f"    Variance ratio: {result.ratio:.4f}")
        print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")

        assert result.in_ci, (
            f"Variance ratio {result.ratio:.4f} not within "
            f"{CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
        )

    def test_brightness_variation(self, device):
        """Brightness should vary within configured range.

        This is a deterministic bound check: brightness is sampled from
        uniform[b_min, b_max], so all factors should be within that range.
        """
        b_min, b_max = 0.8, 1.2

        cfg = RGBNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            pixel_noise_std=0.0,
            brightness_range=(b_min, b_max),
            frame_drop_prob=0.0,
            latency_steps=0,  # Disable latency to avoid delay buffer warmup
        )

        model = RGBNoiseModel(cfg, N_ENVS, device)

        # Input is [0,1] RGB data
        base = torch.ones(N_ENVS, 100, device=device) * 0.5

        brightness_factors = []
        for _ in range(500):
            out = model(base)
            # brightness = output / input (clamped to [0,1], so may be truncated)
            factor = out.mean(dim=1) / 0.5
            brightness_factors.append(factor.cpu())

        factors = torch.cat(brightness_factors, dim=0).numpy()

        print(f"\n  Expected range: [{b_min:.2f}, {b_max:.2f}]")
        print(f"  Observed range: [{factors.min():.3f}, {factors.max():.3f}]")
        print(f"  Mean: {factors.mean():.3f}")

        assert factors.min() >= b_min - FLOAT_TOLERANCE, \
            f"Brightness factor {factors.min():.4f} below minimum {b_min}"
        assert factors.max() <= b_max + FLOAT_TOLERANCE, \
            f"Brightness factor {factors.max():.4f} above maximum {b_max}"

    def test_output_clamped_to_valid_range(self, device):
        """Output should be clamped to [0, 1]."""
        cfg = RGBNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.2),  # High noise to test clamping
            pixel_noise_std=0.2,
            brightness_range=(0.5, 1.5),  # Extreme brightness
            frame_drop_prob=0.0,
            latency_steps=0,  # Disable latency to avoid delay buffer warmup
        )

        model = RGBNoiseModel(cfg, N_ENVS, device)

        # Test near boundaries
        base_low = torch.ones(N_ENVS, 1000, device=device) * 0.1
        base_high = torch.ones(N_ENVS, 1000, device=device) * 0.9

        for _ in range(100):
            out_low = model(base_low)
            out_high = model(base_high)

            assert out_low.min() >= 0.0, f"Output below 0: {out_low.min()}"
            assert out_low.max() <= 1.0, f"Output above 1: {out_low.max()}"
            assert out_high.min() >= 0.0, f"Output below 0: {out_high.min()}"
            assert out_high.max() <= 1.0, f"Output above 1: {out_high.max()}"

        print(f"\n  All outputs clamped to [0, 1]: PASSED")
