# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for IMU noise model (white noise and bias drift) in isolation.

These tests validate the IMUNoiseModel class directly without a simulation
environment. They verify statistical properties of the noise generation:
- Gaussian distribution of accelerometer noise
- Zero-mean noise
- Independence across environments
- Bias drift range compliance and clamp behavior

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_imu_noise_unit.py -v
"""

import torch
import numpy as np
from scipy import stats

from test.common import (
    CONFIDENCE_LEVEL,
    chi_squared_variance_test,
    one_sample_t_test,
    DEVICE,
)

# -- imports resolved after AppLauncher (root conftest) --
from strafer_lab.tasks.navigation.mdp.noise_models import IMUNoiseModel, IMUNoiseModelCfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 50_000
N_ENVS = 32

ACCEL_NOISE_STD = 0.02
GYRO_NOISE_STD = 0.005
BIAS_RANGE = (-0.1, 0.1)
BIAS_DRIFT_RATE = 0.001


def _make_white_noise_only_model(sensor_type: str = "accel") -> IMUNoiseModel:
    """Create an IMUNoiseModel with bias/drift disabled for pure white noise tests."""
    cfg = IMUNoiseModelCfg(
        sensor_type=sensor_type,
        accel_noise_std=ACCEL_NOISE_STD,
        gyro_noise_std=GYRO_NOISE_STD,
        accel_bias_range=(0.0, 0.0),
        gyro_bias_range=(0.0, 0.0),
        accel_bias_drift_rate=0.0,
        gyro_bias_drift_rate=0.0,
        output_size=3,
        latency_steps=0,
    )
    return IMUNoiseModel(cfg, N_ENVS, DEVICE)


def _make_full_imu_model(sensor_type: str = "accel") -> IMUNoiseModel:
    """Create a full IMUNoiseModel with bias and drift enabled."""
    cfg = IMUNoiseModelCfg(
        sensor_type=sensor_type,
        accel_noise_std=ACCEL_NOISE_STD,
        gyro_noise_std=GYRO_NOISE_STD,
        accel_bias_range=BIAS_RANGE,
        gyro_bias_range=BIAS_RANGE,
        accel_bias_drift_rate=BIAS_DRIFT_RATE,
        gyro_bias_drift_rate=BIAS_DRIFT_RATE,
        output_size=3,
        latency_steps=0,
    )
    return IMUNoiseModel(cfg, N_ENVS, DEVICE)


# =============================================================================
# Tests: IMU White Noise Properties
# =============================================================================


def test_accel_white_noise_variance():
    """Verify accelerometer noise variance matches configured std.

    With bias and drift disabled, model(zeros) returns pure white noise.
    Uses chi-squared variance test on a single axis.
    """
    model = _make_white_noise_only_model("accel")

    # Use fewer samples for chi-squared to accommodate float32 GPU precision.
    # At 5 000 df the 95 % CI is ≈ ±4 %, wide enough for float32 noise.
    n_chi2 = 5_000
    clean = torch.zeros(N_ENVS, 3, device=DEVICE)
    samples = []
    for _ in range(n_chi2):
        noisy = model(clean.clone())
        # Single env, single axis to keep chi-squared df reasonable
        samples.append(noisy[0, 0].cpu().item())

    samples = np.array(samples)
    result = chi_squared_variance_test(samples, ACCEL_NOISE_STD**2)

    print(f"\n  Accel noise variance test:")
    print(f"    Expected std: {ACCEL_NOISE_STD}")
    print(f"    Measured std: {np.std(samples):.6f}")
    print(f"    n_samples: {len(samples)}")
    print(f"    Variance ratio: {result.ratio:.4f}")
    print(f"    In CI: {result.in_ci} [{result.ci_low:.4f}, {result.ci_high:.4f}]")

    assert result.in_ci, (
        f"Accel noise variance doesn't match config. "
        f"Expected σ²={ACCEL_NOISE_STD**2}, got {result.measured_var:.6f} "
        f"(ratio={result.ratio:.4f}, CI=[{result.ci_low:.4f}, {result.ci_high:.4f}])"
    )


def test_accel_noise_has_zero_mean():
    """Verify accelerometer noise is zero-mean using one-sample t-test.

    With bias disabled, the noise mean should not significantly differ
    from zero at the configured confidence level.
    """
    model = _make_white_noise_only_model("accel")

    clean = torch.zeros(N_ENVS, 3, device=DEVICE)
    samples = []
    for _ in range(N_SAMPLES):
        noisy = model(clean.clone())
        samples.append(noisy[:, 0].cpu().numpy())

    samples = np.concatenate(samples)
    result = one_sample_t_test(samples, null_value=0.0)

    print(f"\n  Accel zero-mean test:")
    print(f"    Sample mean: {result.mean:.6f}")
    print(f"    t-statistic: {result.t_statistic:.4f}")
    print(f"    p-value: {result.p_value:.4f}")

    assert not result.reject_null, (
        f"Accel noise mean significantly differs from zero "
        f"(mean={result.mean:.6f}, p={result.p_value:.4f})"
    )


def test_accel_noise_is_gaussian():
    """Verify accelerometer noise follows Gaussian distribution.

    Uses Kolmogorov-Smirnov test to check normality after standardizing.
    """
    model = _make_white_noise_only_model("accel")

    clean = torch.zeros(N_ENVS, 3, device=DEVICE)
    samples = []
    for _ in range(N_SAMPLES):
        noisy = model(clean.clone())
        samples.append(noisy[:, 0].cpu().numpy())

    samples = np.concatenate(samples)
    ks_stat, ks_p = stats.kstest(
        (samples - np.mean(samples)) / np.std(samples),
        "norm",
    )

    alpha = 1 - CONFIDENCE_LEVEL

    print(f"\n  Accel Gaussian distribution test (KS):")
    print(f"    KS statistic: {ks_stat:.6f}")
    print(f"    p-value: {ks_p:.4f}")

    assert ks_p > alpha, (
        f"Accel noise does not follow Gaussian distribution "
        f"(KS stat={ks_stat:.6f}, p={ks_p:.4f})"
    )


def test_accel_noise_independent_per_env():
    """Verify noise is independently generated per environment.

    Pearson correlation between two env noise streams should not be
    significant at the configured confidence level.
    """
    model = _make_white_noise_only_model("accel")

    clean = torch.zeros(N_ENVS, 3, device=DEVICE)
    env0_samples = []
    env1_samples = []
    for _ in range(N_SAMPLES):
        noisy = model(clean.clone())
        env0_samples.append(noisy[0, 0].cpu().item())
        env1_samples.append(noisy[1, 0].cpu().item())

    r, p = stats.pearsonr(env0_samples, env1_samples)
    alpha = 1 - CONFIDENCE_LEVEL

    print(f"\n  Inter-env independence test:")
    print(f"    Pearson r: {r:.6f}")
    print(f"    p-value: {p:.4f}")

    assert p > alpha, (
        f"Noise appears correlated across envs (r={r:.4f}, p={p:.4f})"
    )


# =============================================================================
# Tests: IMU Bias Drift Properties
# =============================================================================


def test_bias_within_configured_range():
    """Verify initial bias is sampled within the configured range.

    After reset(), the internal _bias tensor should contain values
    uniformly distributed within accel_bias_range.
    """
    model = _make_full_imu_model("accel")

    n_trials = 1000
    biases = []
    for _ in range(n_trials):
        model.reset()
        biases.append(model._bias[0, 0].cpu().item())

    biases = np.array(biases)
    bias_lo, bias_hi = BIAS_RANGE

    print(f"\n  Bias range test:")
    print(f"    Min sampled bias: {np.min(biases):.6f}")
    print(f"    Max sampled bias: {np.max(biases):.6f}")
    print(f"    Configured range: [{bias_lo}, {bias_hi}]")

    assert np.all(biases >= bias_lo), f"Bias below configured minimum: {np.min(biases)}"
    assert np.all(biases <= bias_hi), f"Bias above configured maximum: {np.max(biases)}"


def test_bias_drift_increases_variance():
    """Verify bias drift increases spread over time.

    Uses Welch's t-test to compare the absolute deviation of bias
    values after 10 steps vs after 500 steps. The later sample should
    have significantly greater spread.
    """
    model = _make_full_imu_model("accel")

    # Track how much bias deviates from its initial value over time.
    # After more __call__ steps, the deviation should be larger.
    n_trials = 500
    early_deviations = []
    late_deviations = []
    clean = torch.zeros(N_ENVS, 3, device=DEVICE)

    for _ in range(n_trials):
        model.reset()
        initial_bias = model._bias[0, 0].cpu().item()

        # After a few steps of drift
        for _ in range(10):
            model(clean.clone())
        early_deviations.append(abs(model._bias[0, 0].cpu().item() - initial_bias))

        # After many more steps of drift
        for _ in range(990):
            model(clean.clone())
        late_deviations.append(abs(model._bias[0, 0].cpu().item() - initial_bias))

    early = np.array(early_deviations)
    late = np.array(late_deviations)

    # Late deviations should be significantly larger (one-sided Mann-Whitney U)
    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(late, early, alternative="greater")
    alpha = 1 - CONFIDENCE_LEVEL

    print(f"\n  Bias drift variance increase test:")
    print(f"    Early deviation mean: {np.mean(early):.6f}")
    print(f"    Late deviation mean:  {np.mean(late):.6f}")
    print(f"    Mann-Whitney U: {stat:.1f}")
    print(f"    p-value: {p_value:.4f}")

    assert p_value < alpha, (
        f"Bias drift did not significantly increase deviation "
        f"(p={p_value:.4f})"
    )


def test_bias_clamped_to_2x_range():
    """Verify bias values are clamped to 2x configured range.

    The IMUNoiseModel clamps bias to prevent unbounded drift.
    Verify no sample exceeds 2 * max(|bias_range|) after heavy drift.
    """
    model = _make_full_imu_model("accel")

    # Derive clamp limit from config — not hardcoded
    clamp_limit = 2 * max(abs(BIAS_RANGE[0]), abs(BIAS_RANGE[1]))

    n_trials = 200
    max_bias_seen = 0.0
    clean = torch.zeros(N_ENVS, 3, device=DEVICE)

    for _ in range(n_trials):
        model.reset()
        for _ in range(5000):
            model(clean.clone())
        max_val = torch.abs(model._bias).max().cpu().item()
        max_bias_seen = max(max_bias_seen, max_val)

    print(f"\n  Bias clamp test:")
    print(f"    Configured range: {BIAS_RANGE}")
    print(f"    Derived clamp limit: ±{clamp_limit:.4f}")
    print(f"    Max |bias| observed: {max_bias_seen:.6f}")

    assert max_bias_seen <= clamp_limit + 1e-6, (
        f"Bias exceeded clamp limit: {max_bias_seen:.6f} > {clamp_limit:.4f}"
    )
