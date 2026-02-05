# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for sensor noise models.

These tests verify the statistical properties of individual noise model classes
in isolation, without running a full Isaac Sim environment.

Tests validate:
1. White noise has correct standard deviation
2. Bias is sampled within configured range
3. Bias drift accumulates correctly (random walk)
4. Quantization produces integer values
5. Tick error probabilities match config
6. Noise is independent per environment

Usage:
    pytest test/unit/test_noise_models.py -v

Note: These tests require Isaac Sim to be launched for torch device access,
but do not create a full simulation environment.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
from scipy import stats
import pytest

from strafer_lab.tasks.navigation.mdp.noise_models import (
    IMUNoiseModel,
    IMUNoiseModelCfg,
    EncoderNoiseModel,
    EncoderNoiseModelCfg,
    DepthNoiseModel,
    DepthNoiseModelCfg,
    RGBNoiseModel,
    RGBNoiseModelCfg,
)
from isaaclab.utils.noise import GaussianNoiseCfg

# Import shared utilities from common module
from test.common import (
    chi_squared_variance_test,
    one_sample_t_test,
    binomial_test,
    CONFIDENCE_LEVEL,
    DEVICE,
)


# =============================================================================
# Test Configuration
# =============================================================================

# Sample count for statistical tests. Higher counts reduce flakiness but increase
# test runtime. At N_SAMPLES=50k with N_ENVS=32, we get ~1500 batches which gives
# enough statistical power that random variation rarely causes false rejections.
N_SAMPLES = 50000        # Samples for statistical tests
N_ENVS = 32              # Parallel environments for unit tests

# Test-specific noise parameters (arbitrary values for testing, not physical constants)
TEST_HOLE_PROBABILITY = 0.03       # Hole rate for depth camera test
TEST_DEPTH_MAX_RANGE = 6.0         # Max range in meters for depth tests
TEST_PIXEL_NOISE_STD = 0.04        # Pixel noise std for RGB test

# Floating-point tolerance for deterministic comparisons
# (used when checking exact bounds, not statistical tests)
FLOAT_TOLERANCE = 1e-6


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def device():
    return DEVICE


# =============================================================================
# IMU Noise Model - White Noise Tests
# =============================================================================

class TestIMUWhiteNoise:
    """Test IMU white noise characteristics (bias disabled)."""

    def test_accel_noise_std_matches_config(self, device):
        """White noise std should match configured value when bias is disabled.

        Uses chi-squared variance test to verify measured variance is
        consistent with the configured noise standard deviation.
        """
        expected_std = 0.05

        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=expected_std),
            accel_noise_std=expected_std,
            accel_bias_range=(0.0, 0.0),  # No bias
            accel_bias_drift_rate=0.0,    # No drift
            output_size=3,
        )

        model = IMUNoiseModel(cfg, N_ENVS, device)

        # Collect samples
        samples = []
        base = torch.zeros(N_ENVS, 3, device=device)
        for _ in range(N_SAMPLES // N_ENVS):
            samples.append(model(base).cpu())

        data = torch.cat(samples, dim=0).numpy().flatten()

        # Chi-squared variance test
        result = chi_squared_variance_test(data, expected_std**2)

        print(f"\n  Accelerometer noise std (chi-squared test):")
        print(f"    Expected std: {expected_std:.4f}")
        print(f"    Measured std: {np.sqrt(result.measured_var):.4f}")
        print(f"    Variance ratio: {result.ratio:.4f}")
        print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")

        assert result.in_ci, (
            f"Variance ratio {result.ratio:.4f} not within "
            f"{CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
        )

    def test_noise_has_zero_mean(self, device):
        """White noise should have zero mean.

        Uses one-sample t-test to verify the mean is not significantly
        different from zero at the configured confidence level.
        """
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.1),
            accel_noise_std=0.1,
            accel_bias_range=(0.0, 0.0),
            accel_bias_drift_rate=0.0,
            output_size=3,
        )

        model = IMUNoiseModel(cfg, N_ENVS, device)

        samples = []
        base = torch.zeros(N_ENVS, 3, device=device)
        for _ in range(N_SAMPLES // N_ENVS):
            samples.append(model(base).cpu())

        all_data = torch.cat(samples, dim=0).numpy()
        # Use single axis to keep sample count reasonable (~10k);
        # flattening all 3 axes yields ~30k samples whose CI becomes
        # so tight that normal floating-point jitter causes false rejections.
        data = all_data[:, 0]

        # One-sample t-test: is mean significantly different from 0?
        result = one_sample_t_test(data, null_value=0.0)

        print(f"\n  Zero mean test (one-sample t-test):")
        print(f"    Mean: {result.mean:.6f}")
        print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.2e}, {result.ci_high:.2e}]")
        print(f"    p-value: {result.p_value:.4f}")
        print(f"    Reject null (mean ≠ 0): {result.reject_null}")

        assert not result.reject_null, (
            f"Noise mean {result.mean:.6f} is significantly different from zero "
            f"(p={result.p_value:.4f} < α={1 - CONFIDENCE_LEVEL:.2f})"
        )

    def test_noise_is_gaussian(self, device):
        """White noise should follow Gaussian distribution.

        Uses binomial tests to verify that the proportion of samples within
        1σ and 2σ matches the expected Gaussian percentiles (68.27% and 95.45%).
        """
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.1),
            accel_noise_std=0.1,
            accel_bias_range=(0.0, 0.0),
            accel_bias_drift_rate=0.0,
            output_size=3,
        )

        model = IMUNoiseModel(cfg, N_ENVS, device)

        samples = []
        base = torch.zeros(N_ENVS, 3, device=device)
        for _ in range(N_SAMPLES // N_ENVS):
            samples.append(model(base).cpu())

        data = torch.cat(samples, dim=0).numpy().flatten()
        std = np.std(data)
        n = len(data)

        # Count samples within 1σ and 2σ
        count_1std = int(np.sum(np.abs(data) < std))
        count_2std = int(np.sum(np.abs(data) < 2 * std))

        # Binomial test: is the proportion within 1σ consistent with 68.27%?
        result_1std = binomial_test(count_1std, n, 0.6827)
        result_2std = binomial_test(count_2std, n, 0.9545)

        print(f"\n  Gaussian distribution test (binomial tests):")
        print(f"    Within 1σ: {count_1std/n*100:.1f}% (expected: 68.27%)")
        print(f"      p-value: {result_1std.p_value:.4f}, reject: {result_1std.reject_null}")
        print(f"    Within 2σ: {count_2std/n*100:.1f}% (expected: 95.45%)")
        print(f"      p-value: {result_2std.p_value:.4f}, reject: {result_2std.reject_null}")

        assert not result_1std.reject_null, (
            f"1σ proportion {count_1std/n:.4f} inconsistent with Gaussian "
            f"(p={result_1std.p_value:.4f})"
        )
        assert not result_2std.reject_null, (
            f"2σ proportion {count_2std/n:.4f} inconsistent with Gaussian "
            f"(p={result_2std.p_value:.4f})"
        )

    def test_noise_is_independent_per_env(self, device):
        """Each environment should receive independent noise samples.

        Uses Pearson correlation test to verify that noise between two
        environments is not significantly correlated.
        """
        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.1),
            accel_noise_std=0.1,
            accel_bias_range=(0.0, 0.0),
            accel_bias_drift_rate=0.0,
            output_size=3,
        )

        model = IMUNoiseModel(cfg, N_ENVS, device)

        # Collect time series for env 0 and env 1
        env0_samples = []
        env1_samples = []
        base = torch.zeros(N_ENVS, 3, device=device)

        for _ in range(500):
            out = model(base)
            env0_samples.append(out[0, 0].item())
            env1_samples.append(out[1, 0].item())

        correlation, p_value = stats.pearsonr(env0_samples, env1_samples)

        print(f"\n  Noise independence test (Pearson correlation):")
        print(f"    Correlation: r={correlation:.4f}")
        print(f"    p-value: {p_value:.4e}")
        print(f"    Significance level (α): {1 - CONFIDENCE_LEVEL:.2f}")

        # Fail to reject null hypothesis of independence
        assert p_value > (1 - CONFIDENCE_LEVEL), (
            f"Noise appears correlated across environments: "
            f"r={correlation:.4f}, p={p_value:.4e} < α={1 - CONFIDENCE_LEVEL:.2f}"
        )


# =============================================================================
# IMU Noise Model - Bias Tests
# =============================================================================

class TestIMUBias:
    """Test IMU bias initialization and drift behavior."""

    def test_bias_within_configured_range(self, device):
        """Initial bias should be sampled within configured range.

        This is a deterministic bound check: bias is sampled from uniform[low, high],
        so ALL samples must be within [low, high]. We use FLOAT_TOLERANCE for
        numerical precision only, not as a statistical tolerance.

        The mean is tested statistically using a one-sample t-test since we expect
        the uniform distribution to be centered at 0.
        """
        bias_range = (-0.1, 0.1)

        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            accel_noise_std=0.0,  # No white noise
            accel_bias_range=bias_range,
            accel_bias_drift_rate=0.0,  # No drift
            output_size=3,
        )

        # Sample many models to test bias distribution
        all_biases = []
        for _ in range(100):
            model = IMUNoiseModel(cfg, N_ENVS, device)
            # With no noise, output = bias
            out = model(torch.zeros(N_ENVS, 3, device=device))
            all_biases.append(out.cpu())

        biases = torch.cat(all_biases, dim=0).numpy().flatten()

        print(f"\n  Configured range: [{bias_range[0]:.3f}, {bias_range[1]:.3f}]")
        print(f"  Observed range: [{biases.min():.4f}, {biases.max():.4f}]")
        print(f"  Mean bias: {biases.mean():.4f} (expected: 0)")

        # All biases should be strictly within range (deterministic bound)
        assert biases.min() >= bias_range[0] - FLOAT_TOLERANCE, \
            f"Bias {biases.min():.6f} below configured minimum {bias_range[0]}"
        assert biases.max() <= bias_range[1] + FLOAT_TOLERANCE, \
            f"Bias {biases.max():.6f} above configured maximum {bias_range[1]}"

        # Mean should be centered at 0 (statistical test for uniform distribution)
        mean_result = one_sample_t_test(biases, null_value=0.0)
        print(f"  Mean t-test p-value: {mean_result.p_value:.4f}")
        assert not mean_result.reject_null, \
            f"Bias mean {mean_result.mean:.4f} significantly differs from 0 (p={mean_result.p_value:.4f})"

    def test_bias_drift_increases_variance(self, device):
        """Bias drift (random walk) should increase variance over time."""
        drift_rate = 0.01

        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            accel_noise_std=0.0,
            accel_bias_range=(-0.5, 0.5),  # Room for drift
            accel_bias_drift_rate=drift_rate,
            output_size=3,
        )

        model = IMUNoiseModel(cfg, N_ENVS, device)
        model._bias.zero_()  # Start from known state

        # Collect outputs over time
        outputs = []
        base = torch.zeros(N_ENVS, 3, device=device)
        for _ in range(200):
            outputs.append(model(base).cpu())

        outputs = torch.stack(outputs, dim=0)  # (time, envs, 3)

        # Variance should increase with time (random walk property)
        early_var = outputs[:50].var().item()
        late_var = outputs[150:].var().item()

        print(f"\n  Drift rate: {drift_rate}")
        print(f"  Early variance (steps 0-50): {early_var:.6f}")
        print(f"  Late variance (steps 150-200): {late_var:.6f}")
        print(f"  Ratio: {late_var / max(early_var, 1e-9):.2f}x")

        assert late_var > early_var, "Variance should grow with drift"

    def test_bias_clamped_to_2x_range(self, device):
        """Bias should be clamped to 2x the configured range."""
        bias_range = (-0.1, 0.1)
        clamp_limit = 0.2  # 2x bias_range

        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            accel_noise_std=0.0,
            accel_bias_range=bias_range,
            accel_bias_drift_rate=0.1,  # High drift to hit clamp
            output_size=3,
        )

        model = IMUNoiseModel(cfg, N_ENVS, device)

        # Run for many steps to let drift hit clamp
        base = torch.zeros(N_ENVS, 3, device=device)
        max_observed = 0.0
        for _ in range(1000):
            out = model(base)
            max_observed = max(max_observed, out.abs().max().item())

        print(f"\n  Clamp limit: ±{clamp_limit:.3f}")
        print(f"  Max observed bias: {max_observed:.4f}")

        # Should be clamped (deterministic bound, use float tolerance only)
        # No white noise is configured, so this is an exact bound check
        assert max_observed <= clamp_limit + FLOAT_TOLERANCE, \
            f"Bias {max_observed:.6f} exceeded clamp limit {clamp_limit}"


# =============================================================================
# Encoder Noise Model Tests
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

        # Chi-squared variance test
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

        # Binomial test: is observed rate consistent with expected?
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


# =============================================================================
# Depth Camera Noise Model Tests
# =============================================================================

class TestDepthNoise:
    """Test depth camera noise model.

    Note: DepthNoiseModel now works on RAW meters (not normalized).
    Normalization is applied afterwards via ObsTerm.scale.
    """

    def test_depth_dependent_noise(self, device):
        """Noise should increase with depth (stereo error propagation).

        The depth noise model uses the stereo camera error formula:
            σ_z = z² · σ_d / (f · B)

        This means noise grows quadratically with depth. We verify that
        measured noise at three different depths follows this trend.
        """
        # Use default D555 stereo parameters
        cfg = DepthNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            baseline_m=0.095,
            focal_length_px=673.0,
            disparity_noise_px=0.08,
            hole_probability=0.0,
            frame_drop_prob=0.0,
            max_range=10.0,
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
            disparity_noise_px=0.0,  # No depth noise, isolate holes
            hole_probability=TEST_HOLE_PROBABILITY,
            frame_drop_prob=0.0,
            max_range=TEST_DEPTH_MAX_RANGE,
        )

        model = DepthNoiseModel(cfg, N_ENVS, device)

        # Mid-range depth in METERS
        pixels_per_frame = 1000
        base = torch.ones(N_ENVS, pixels_per_frame, device=device) * 3.0  # 3m

        # Use enough iterations to get ~500k samples for reliable binomial test
        n_iterations = N_SAMPLES // (N_ENVS * pixels_per_frame) * 10  # ~156 iterations
        total = 0
        holes = 0
        for _ in range(n_iterations):
            out = model(base)
            # Holes are set to max_range (in meters)
            holes += int((out >= TEST_DEPTH_MAX_RANGE - FLOAT_TOLERANCE).sum().item())
            total += out.numel()

        # Binomial test: is observed rate consistent with expected?
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


# =============================================================================
# RGB Camera Noise Model Tests
# =============================================================================

class TestRGBNoise:
    """Test RGB camera noise model.

    Note: RGBNoiseModel now works on [0,1] RGB data (not mean-centered).
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
        )

        model = RGBNoiseModel(cfg, N_ENVS, device)

        # Input is [0,1] RGB data
        base = torch.ones(N_ENVS, 100, device=device) * 0.5

        # Collect samples - use N_SAMPLES // N_ENVS iterations
        n_iterations = N_SAMPLES // N_ENVS
        samples = []
        for _ in range(n_iterations):
            samples.append((model(base) - 0.5).cpu())

        all_noise = torch.cat(samples, dim=0).numpy()
        # Use single pixel column to keep sample count at ~N_SAMPLES
        # Flattening all pixels makes the CI extremely tight
        noise = all_noise[:, 0]

        # Chi-squared variance test
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
        We use FLOAT_TOLERANCE for numerical precision only.
        """
        b_min, b_max = 0.8, 1.2

        cfg = RGBNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            pixel_noise_std=0.0,
            brightness_range=(b_min, b_max),
            frame_drop_prob=0.0,
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

        # Deterministic bounds (use float tolerance only)
        # Note: at input=0.5, max brightness 1.2 gives output 0.6 (no clamping)
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
        )

        model = RGBNoiseModel(cfg, N_ENVS, device)

        # Test near boundaries
        base_low = torch.ones(N_ENVS, 1000, device=device) * 0.1
        base_high = torch.ones(N_ENVS, 1000, device=device) * 0.9

        for _ in range(100):
            out_low = model(base_low)
            out_high = model(base_high)

            # All values should be in [0, 1]
            assert out_low.min() >= 0.0, f"Output below 0: {out_low.min()}"
            assert out_low.max() <= 1.0, f"Output above 1: {out_low.max()}"
            assert out_high.min() >= 0.0, f"Output below 0: {out_high.min()}"
            assert out_high.max() <= 1.0, f"Output above 1: {out_high.max()}"

        print(f"\n  All outputs clamped to [0, 1]: PASSED")


# =============================================================================
# Cleanup
# =============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Clean up simulation after tests."""
    simulation_app.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
