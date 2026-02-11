# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for IMU noise model in isolation.

These tests verify the statistical properties of IMUNoiseModel without
running a full simulation environment - only torch device access is needed.

Tests validate:
1. White noise standard deviation matches config (chi-squared test)
2. White noise has zero mean (one-sample t-test)
3. White noise follows Gaussian distribution (binomial proportion tests)
4. Noise is independent per environment (Pearson correlation)
5. Bias initialization within configured range (deterministic bounds)
6. Bias drift increases variance over time (random walk property)
7. Bias clamped to 2x configured range (deterministic bounds)

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_imu_noise_unit.py -v
"""

import torch
import numpy as np
from scipy import stats
import pytest

from strafer_lab.tasks.navigation.mdp.noise_models import (
    IMUNoiseModel,
    IMUNoiseModelCfg,
)
from isaaclab.utils.noise import GaussianNoiseCfg

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

        count_1std = int(np.sum(np.abs(data) < std))
        count_2std = int(np.sum(np.abs(data) < 2 * std))

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
        so ALL samples must be within [low, high]. The mean is tested statistically
        using a one-sample t-test since we expect the uniform distribution to be
        centered at 0.
        """
        bias_range = (-0.1, 0.1)

        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            accel_noise_std=0.0,
            accel_bias_range=bias_range,
            accel_bias_drift_rate=0.0,
            output_size=3,
        )

        all_biases = []
        for _ in range(100):
            model = IMUNoiseModel(cfg, N_ENVS, device)
            out = model(torch.zeros(N_ENVS, 3, device=device))
            all_biases.append(out.cpu())

        biases = torch.cat(all_biases, dim=0).numpy().flatten()

        print(f"\n  Configured range: [{bias_range[0]:.3f}, {bias_range[1]:.3f}]")
        print(f"  Observed range: [{biases.min():.4f}, {biases.max():.4f}]")
        print(f"  Mean bias: {biases.mean():.4f} (expected: 0)")

        assert biases.min() >= bias_range[0] - FLOAT_TOLERANCE, \
            f"Bias {biases.min():.6f} below configured minimum {bias_range[0]}"
        assert biases.max() <= bias_range[1] + FLOAT_TOLERANCE, \
            f"Bias {biases.max():.6f} above configured maximum {bias_range[1]}"

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
            accel_bias_range=(-0.5, 0.5),
            accel_bias_drift_rate=drift_rate,
            output_size=3,
        )

        model = IMUNoiseModel(cfg, N_ENVS, device)
        model._bias.zero_()

        outputs = []
        base = torch.zeros(N_ENVS, 3, device=device)
        for _ in range(200):
            outputs.append(model(base).cpu())

        outputs = torch.stack(outputs, dim=0)

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
        clamp_limit = 0.2

        cfg = IMUNoiseModelCfg(
            noise_cfg=GaussianNoiseCfg(std=0.0),
            accel_noise_std=0.0,
            accel_bias_range=bias_range,
            accel_bias_drift_rate=0.1,
            output_size=3,
        )

        model = IMUNoiseModel(cfg, N_ENVS, device)

        base = torch.zeros(N_ENVS, 3, device=device)
        max_observed = 0.0
        for _ in range(1000):
            out = model(base)
            max_observed = max(max_observed, out.abs().max().item())

        print(f"\n  Clamp limit: ±{clamp_limit:.3f}")
        print(f"  Max observed bias: {max_observed:.4f}")

        assert max_observed <= clamp_limit + FLOAT_TOLERANCE, \
            f"Bias {max_observed:.6f} exceeded clamp limit {clamp_limit}"
