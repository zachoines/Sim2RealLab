# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for depth camera noise model in isolation.

These tests validate the DepthNoiseModel class directly without a
simulation environment. They verify:
- Depth-dependent stereo noise (σ ∝ z²) with chi-squared test
- Hole probability matches configured rate

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_depth_noise.py -v
"""

import torch
import numpy as np

from test.common import (
    chi_squared_variance_test,
    binomial_test,
    DEVICE,
)

# -- imports resolved after AppLauncher (root conftest) --
from strafer_lab.tasks.navigation.mdp.noise_models import (
    DepthNoiseModel,
    DepthNoiseModelCfg,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 50_000
N_ENVS = 32
N_PIXELS = 256  # 16×16 depth image, flattened

TEST_BASELINE_M = 0.095
TEST_FOCAL_PX = 673.0
TEST_DISPARITY_NOISE_PX = 0.08
TEST_HOLE_PROBABILITY = 0.03
TEST_MAX_RANGE = 6.0
TEST_MIN_RANGE = 0.2


def _make_depth_model(hole_probability: float = 0.0) -> DepthNoiseModel:
    """Create a DepthNoiseModel for testing."""
    cfg = DepthNoiseModelCfg(
        baseline_m=TEST_BASELINE_M,
        focal_length_px=TEST_FOCAL_PX,
        disparity_noise_px=TEST_DISPARITY_NOISE_PX,
        hole_probability=hole_probability,
        min_range=TEST_MIN_RANGE,
        max_range=TEST_MAX_RANGE,
        latency_steps=0,
    )
    return DepthNoiseModel(cfg, N_ENVS, DEVICE)


# =============================================================================
# Tests
# =============================================================================


def test_depth_dependent_noise():
    """Verify depth noise increases with distance (stereo noise model).

    The stereo noise formula is: σ = z² × σ_d / (f × B)
    at distances [1m, 2m, 4m], noise std should scale as z².

    Tests both:
    1. Monotone ordering: std(1m) < std(2m) < std(4m)
    2. Chi-squared variance test at a reference depth against theoretical σ
    """
    model = _make_depth_model(hole_probability=0.0)
    stereo_coeff = TEST_DISPARITY_NOISE_PX / (TEST_FOCAL_PX * TEST_BASELINE_M)

    # Use fewer samples for chi-squared to accommodate float32 GPU precision.
    n_chi2 = 5_000
    test_depths = [1.0, 2.0, 4.0]
    measured_stds = {}

    for z in test_depths:
        # Flattened depth image: (N_ENVS, N_PIXELS)
        clean = torch.full((N_ENVS, N_PIXELS), z, device=DEVICE)
        samples = []
        for _ in range(n_chi2):
            noisy = model(clean.clone())
            # Single env, single pixel to keep chi-squared df reasonable
            samples.append(noisy[0, 0].cpu().item() - z)

        flat = np.array(samples)
        measured_stds[z] = np.std(flat)

    # 1. Monotone ordering
    for i in range(len(test_depths) - 1):
        z_near, z_far = test_depths[i], test_depths[i + 1]
        assert measured_stds[z_near] < measured_stds[z_far], (
            f"Noise should increase with depth: "
            f"std({z_near}m)={measured_stds[z_near]:.4f} >= "
            f"std({z_far}m)={measured_stds[z_far]:.4f}"
        )

    # 2. Chi-squared test at reference depth (2m)
    ref_depth = 2.0
    expected_std = ref_depth**2 * stereo_coeff
    clean_ref = torch.full((N_ENVS, N_PIXELS), ref_depth, device=DEVICE)
    ref_samples = []
    for _ in range(n_chi2):
        noisy = model(clean_ref.clone())
        # Single env, single pixel
        ref_samples.append(noisy[0, 0].cpu().item() - ref_depth)
    ref_flat = np.array(ref_samples)

    result = chi_squared_variance_test(ref_flat, expected_std**2)

    print(f"\n  Depth-dependent noise test:")
    for z in test_depths:
        expected = z**2 * stereo_coeff
        print(f"    z={z}m: measured σ={measured_stds[z]:.6f}, expected σ={expected:.6f}")
    print(f"    Chi-squared at 2m: ratio={result.ratio:.4f}, in_ci={result.in_ci}")
    print(f"    CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")

    assert result.in_ci, (
        f"Noise at reference depth doesn't match stereo formula. "
        f"Expected σ²={expected_std**2:.6f}, got {result.measured_var:.6f} "
        f"(ratio={result.ratio:.4f}, CI=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
    )


def test_hole_probability():
    """Verify hole insertion rate matches configured probability.

    Holes replace depth values with max_range. Uses binomial test to
    verify the proportion of max_range pixels matches hole_probability.
    """
    model = _make_depth_model(hole_probability=TEST_HOLE_PROBABILITY)

    # Mid-range depth unlikely to trigger max_range by noise alone
    clean = torch.full((N_ENVS, N_PIXELS), 3.0, device=DEVICE)

    n_total = 0
    n_holes = 0
    for _ in range(N_SAMPLES):
        noisy = model(clean.clone())
        # Holes are replaced with max_range
        n_holes += int((noisy >= TEST_MAX_RANGE - 1e-3).sum().item())
        n_total += N_ENVS * N_PIXELS

    result = binomial_test(n_holes, n_total, TEST_HOLE_PROBABILITY)

    print(f"\n  Hole probability test:")
    print(f"    Expected rate: {TEST_HOLE_PROBABILITY}")
    print(f"    Observed rate: {n_holes / n_total:.4f}")
    print(f"    Binomial p-value: {result.p_value:.4f}")

    assert not result.reject_null, (
        f"Hole rate doesn't match config. "
        f"Expected {TEST_HOLE_PROBABILITY}, got {n_holes / n_total:.4f} "
        f"(p={result.p_value:.4f})"
    )
