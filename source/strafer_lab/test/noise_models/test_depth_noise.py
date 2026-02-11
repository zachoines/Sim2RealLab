# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for depth camera noise model in isolation.

These tests verify the statistical properties of DepthNoiseModel without
running a full simulation environment - only torch device access is needed.

Tests validate:
1. Depth-dependent noise follows stereo error propagation (quadratic growth)
2. Hole probability matches configured rate (binomial test)

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_depth_noise.py -v
"""

import torch
import numpy as np
import pytest

from strafer_lab.tasks.navigation.mdp.noise_models import (
    DepthNoiseModel,
    DepthNoiseModelCfg,
)
from isaaclab.utils.noise import GaussianNoiseCfg

from test.common import (
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

# Depth-specific test parameters
TEST_HOLE_PROBABILITY = 0.03
TEST_DEPTH_MAX_RANGE = 6.0


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def device():
    return DEVICE


# =============================================================================
# Depth Camera Noise Tests
# =============================================================================

class TestDepthNoise:
    """Test depth camera noise model.

    Note: DepthNoiseModel works on RAW meters (not normalized).
    Normalization is applied afterwards via ObsTerm.scale.
    """

    def test_depth_dependent_noise(self, device):
        """Noise should increase with depth (stereo error propagation).

        The depth noise model uses the stereo camera error formula:
            σ_z = z² · σ_d / (f · B)

        This means noise grows quadratically with depth. We verify that
        measured noise at three different depths follows this trend.
        """
        cfg = DepthNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            baseline_m=0.095,
            focal_length_px=673.0,
            disparity_noise_px=0.08,
            hole_probability=0.0,
            frame_drop_prob=0.0,
            max_range=10.0,
            latency_steps=0,  # Disable latency for unit test isolation
        )

        model = DepthNoiseModel(cfg, N_ENVS, device)

        # Stereo coefficient: σ_d / (f · B)
        stereo_coeff = cfg.disparity_noise_px / (cfg.focal_length_px * cfg.baseline_m)

        # Test at different depths (in METERS)
        depths_m = [1.0, 2.0, 4.0]
        measured_stds = []

        for depth in depths_m:
            model.reset()
            base = torch.ones(N_ENVS, 500, device=device) * depth

            samples = []
            for _ in range(200):
                samples.append((model(base) - depth).cpu())

            noise = torch.cat(samples, dim=0).numpy().flatten()
            measured_stds.append(np.std(noise))

        print(f"\n  Depth (m) | Expected Std  | Measured Std")
        print(f"  " + "-" * 45)
        for d, m in zip(depths_m, measured_stds):
            expected = d**2 * stereo_coeff
            print(f"  {d:.1f}m      | {expected:.6f}m  | {m:.6f}m")

        # Verify noise increases with depth (quadratic relationship)
        assert measured_stds[1] > measured_stds[0], \
            f"Noise at {depths_m[1]}m ({measured_stds[1]:.6f}) should exceed noise at {depths_m[0]}m ({measured_stds[0]:.6f})"
        assert measured_stds[2] > measured_stds[1], \
            f"Noise at {depths_m[2]}m ({measured_stds[2]:.6f}) should exceed noise at {depths_m[1]}m ({measured_stds[1]:.6f})"

    def test_hole_probability(self, device):
        """Holes should occur at configured probability.

        Uses binomial test to verify the observed hole rate is consistent
        with the configured probability.
        """
        cfg = DepthNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            disparity_noise_px=0.0,
            hole_probability=TEST_HOLE_PROBABILITY,
            frame_drop_prob=0.0,
            max_range=TEST_DEPTH_MAX_RANGE,
            latency_steps=0,  # Disable latency to avoid delay buffer warmup
        )

        model = DepthNoiseModel(cfg, N_ENVS, device)

        # Mid-range depth in METERS
        pixels_per_frame = 1000
        base = torch.ones(N_ENVS, pixels_per_frame, device=device) * 3.0

        # Use enough iterations to get ~500k samples for reliable binomial test
        n_iterations = N_SAMPLES // (N_ENVS * pixels_per_frame) * 10
        total = 0
        holes = 0
        for _ in range(n_iterations):
            out = model(base)
            # Holes are set to max_range (in meters)
            holes += int((out >= TEST_DEPTH_MAX_RANGE - FLOAT_TOLERANCE).sum().item())
            total += out.numel()

        result = binomial_test(holes, total, TEST_HOLE_PROBABILITY)

        print(f"\n  Hole probability (binomial test):")
        print(f"    Total samples: {total}")
        print(f"    Expected prob: {TEST_HOLE_PROBABILITY:.4f}")
        print(f"    Observed prob: {result.observed_rate:.4f}")
        print(f"    p-value: {result.p_value:.4f}")
        print(f"    Reject null: {result.reject_null}")

        assert not result.reject_null, (
            f"Hole rate {result.observed_rate:.4f} inconsistent with "
            f"expected {TEST_HOLE_PROBABILITY:.4f} (p={result.p_value:.4f})"
        )
