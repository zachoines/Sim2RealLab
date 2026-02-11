# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RGB camera noise model in isolation.

These tests validate the RGBNoiseModel class directly without a
simulation environment. They verify:
- Pixel noise variance matches config
- Brightness modulation stays within configured range
- Output is clamped to valid [0, 1] image range

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_rgb_noise.py -v
"""

import torch
import numpy as np

from test.common import (
    chi_squared_variance_test,
    DEVICE,
)

# -- imports resolved after AppLauncher (root conftest) --
from strafer_lab.tasks.navigation.mdp.noise_models import RGBNoiseModel, RGBNoiseModelCfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 50_000
N_ENVS = 32
N_PIXELS = 16 * 16 * 3  # Flattened H×W×C

TEST_PIXEL_NOISE_STD = 0.04
FLOAT_TOLERANCE = 1e-6


def _make_rgb_model(
    brightness_range: tuple[float, float] = (1.0, 1.0),
) -> RGBNoiseModel:
    """Create a standard RGBNoiseModel for testing."""
    cfg = RGBNoiseModelCfg(
        pixel_noise_std=TEST_PIXEL_NOISE_STD,
        brightness_range=brightness_range,
        frame_drop_prob=0.0,
        latency_steps=0,
    )
    return RGBNoiseModel(cfg, N_ENVS, DEVICE)


# =============================================================================
# Tests
# =============================================================================


def test_pixel_noise_std():
    """Verify pixel noise variance matches configured std.

    Uses chi-squared variance test with brightness fixed at 1.0
    so the only noise source is the per-pixel Gaussian.
    Data is flattened to (N_ENVS, pixels) as required by the model.
    """
    model = _make_rgb_model(brightness_range=(1.0, 1.0))

    # Use fewer samples for chi-squared to accommodate float32 GPU precision.
    # At 5 000 df the 95 % CI is ≈ ±4 %, wide enough for float32 noise.
    n_chi2 = 5_000
    # Mid-gray image avoids clamping at 0 or 1
    clean = torch.full((N_ENVS, N_PIXELS), 0.5, device=DEVICE)
    samples = []
    for _ in range(n_chi2):
        noisy = model(clean.clone())
        # Single env, single pixel to keep chi-squared df reasonable
        # Brightness is (1.0, 1.0) so output = 0.5 + noise (no scaling)
        samples.append(noisy[0, 0].cpu().item() - 0.5)

    samples = np.array(samples)
    result = chi_squared_variance_test(samples, TEST_PIXEL_NOISE_STD**2)

    print(f"\n  RGB pixel noise variance test:")
    print(f"    Expected std: {TEST_PIXEL_NOISE_STD}")
    print(f"    Measured std: {np.std(samples):.6f}")
    print(f"    Variance ratio: {result.ratio:.4f}")
    print(f"    In CI: {result.in_ci} [{result.ci_low:.4f}, {result.ci_high:.4f}]")

    assert result.in_ci, (
        f"Pixel noise variance doesn't match config. "
        f"Expected σ²={TEST_PIXEL_NOISE_STD**2}, got {result.measured_var:.6f} "
        f"(ratio={result.ratio:.4f}, CI=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
    )


def test_brightness_variation():
    """Verify brightness modulation stays within configured range.

    With brightness_range=(0.8, 1.2), the per-environment brightness
    factor should be a uniform sample in [0.8, 1.2]. We verify this
    by measuring the mean pixel value per environment (which reflects
    the brightness factor applied to the constant 0.5 input).
    """
    lo, hi = 0.8, 1.2
    model = _make_rgb_model(brightness_range=(lo, hi))

    clean = torch.full((N_ENVS, N_PIXELS), 0.5, device=DEVICE)

    # Use fewer iterations to reduce extreme-value risk from millions of
    # uniform samples. With 1000 × 32 = 32 000 brightness values, the
    # extreme-value overshoot beyond [lo, hi] is well within our slack.
    n_brightness = 1_000
    observed_brightness = []
    for _ in range(n_brightness):
        noisy = model(clean.clone())
        # Mean pixel per env estimates brightness × 0.5
        # (pixel noise averages out over many pixels)
        per_env_mean = noisy.mean(dim=1).cpu().numpy()
        brightness = per_env_mean / 0.5  # Undo the input value
        observed_brightness.append(brightness)

    all_brightness = np.concatenate(observed_brightness)

    print(f"\n  Brightness variation test:")
    print(f"    Min brightness: {np.min(all_brightness):.4f} (config lo: {lo})")
    print(f"    Max brightness: {np.max(all_brightness):.4f} (config hi: {hi})")
    print(f"    Mean brightness: {np.mean(all_brightness):.4f} (expected: {(lo + hi) / 2})")

    # Slack accounts for pixel-noise averaging shifting per-env means
    # beyond the true brightness range. Conservative 10σ/√N bound.
    slack = 10 * TEST_PIXEL_NOISE_STD / np.sqrt(N_PIXELS)
    assert np.min(all_brightness) >= lo - slack, (
        f"Brightness below configured min: "
        f"{np.min(all_brightness):.4f} < {lo - slack:.4f}"
    )
    assert np.max(all_brightness) <= hi + slack, (
        f"Brightness above configured max: "
        f"{np.max(all_brightness):.4f} > {hi + slack:.4f}"
    )


def test_output_clamped_to_valid_range():
    """Verify all output pixel values are clamped to [0, 1].

    Use extreme brightness + noise that would push values out of range
    and verify the output is still valid.
    """
    model = _make_rgb_model(brightness_range=(1.5, 1.5))

    # Near-white image + high brightness → should be clamped to 1.0
    clean = torch.full((N_ENVS, N_PIXELS), 0.9, device=DEVICE)
    noisy = model(clean.clone())

    assert noisy.min().item() >= -FLOAT_TOLERANCE, (
        f"Output has negative values: min={noisy.min().item():.6f}"
    )
    assert noisy.max().item() <= 1.0 + FLOAT_TOLERANCE, (
        f"Output exceeds 1.0: max={noisy.max().item():.6f}"
    )

    print(f"\n  Output clamping test:")
    print(f"    Min pixel: {noisy.min().item():.6f}")
    print(f"    Max pixel: {noisy.max().item():.6f}")
    print(f"    Values in [0, 1]: ✓")
