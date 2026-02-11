# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for noise configuration behavior.

These tests validate general noise model behavior:
- Noise is enabled/disabled correctly via enable_corruption
- Noise is independent across parallel environments
- Noise changes each timestep (not frozen)
- Noise is unbiased (zero mean)

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_noise_config.py -v
"""

import numpy as np
from scipy import stats
import pytest

from test.common import (
    CONFIDENCE_LEVEL,
    N_SAMPLES_STEPS,
    one_sample_t_test,
)

from test.noise_models.conftest import (
    collect_stationary_observations,
    extract_noise_samples,
    IMU_ACCEL_SLICE,
)


# =============================================================================
# Tests: Noise Configuration Behavior
# =============================================================================


def test_noise_affects_observations(noisy_env):
    """Verify observations have variance when noise is enabled.

    Uses one-sample t-test on per-step variances to verify that the
    mean variance across steps is significantly greater than zero.
    This replaces an arbitrary threshold with a proper statistical test.
    """
    obs = collect_stationary_observations(noisy_env, 50)

    # Compute variance per step across envs for IMU accel
    per_step_var = obs[:, :, IMU_ACCEL_SLICE].var(dim=1).mean(dim=1)  # (n_steps,)
    var_samples = per_step_var.cpu().numpy()

    result = one_sample_t_test(var_samples, null_value=0.0)

    print(f"\n  Noise-enabled observation variance test (one-sample t-test):")
    print(f"    Mean per-step variance: {result.mean:.6e}")
    print(f"    p-value (H0: variance = 0): {result.p_value:.4f}")
    print(f"    Reject null: {result.reject_null}")

    # Variance should be SIGNIFICANTLY greater than zero
    assert result.reject_null and result.mean > 0, (
        f"Observation variance not significantly above zero "
        f"(mean={result.mean:.2e}, p={result.p_value:.4f}). Is noise enabled?"
    )


def test_noise_is_independent_across_envs(noisy_env):
    """Verify that custom NoiseModel generates independent noise per environment.

    This is critical for effective multi-environment RL training.
    With independent noise, each environment experiences different sensor
    corruptions, providing diverse training samples.

    Statistical approach:
    - Compute Pearson correlation between noise from env 0 and env 1
    - Under null hypothesis of independence, test r ≈ 0
    """
    obs = collect_stationary_observations(noisy_env, 200)

    # Extract IMU accel observations: (n_steps, num_envs, 3)
    imu_accel = obs[:, :, IMU_ACCEL_SLICE]

    # Subtract per-env mean to get noise component only
    env_mean = imu_accel.mean(dim=0, keepdim=True)
    noise_only = imu_accel - env_mean

    # Compare noise between env 0 and env 1
    env0_noise = noise_only[:, 0, :].cpu().numpy().flatten()
    env1_noise = noise_only[:, 1, :].cpu().numpy().flatten()

    # Compute Pearson correlation and p-value
    correlation, p_value = stats.pearsonr(env0_noise, env1_noise)
    n = len(env0_noise)

    # Standard error of correlation under null hypothesis
    se_r = 1.0 / np.sqrt(n - 3)

    print(f"\n  Noise independence test (correlation analysis):")
    print(f"    Sample size: n={n:,}")
    print(f"    Correlation between env 0 and env 1: r={correlation:.4f}")
    print(f"    Standard error under H0 (independence): {se_r:.6f}")
    print(f"    Two-tailed p-value: {p_value:.4e}")
    print(f"    Significance level (α): {1 - CONFIDENCE_LEVEL:.2f}")

    # Test: correlation should not be significantly different from zero
    assert p_value > (1 - CONFIDENCE_LEVEL), (
        f"Noise appears correlated across environments: "
        f"r={correlation:.4f}, p={p_value:.4e} < α={1 - CONFIDENCE_LEVEL:.2f}"
    )


def test_noise_changes_each_step(noisy_env):
    """Verify noise is different at each timestep (not frozen).

    Uses one-sample t-test on mean absolute step-to-step differences
    to verify the changes are significantly greater than zero.
    """
    obs = collect_stationary_observations(noisy_env, 50)

    # Mean absolute difference between consecutive steps
    step_diffs = []
    for i in range(len(obs) - 1):
        diff = (obs[i + 1, :, IMU_ACCEL_SLICE] - obs[i, :, IMU_ACCEL_SLICE]).abs().mean()
        step_diffs.append(diff.item())

    step_diffs = np.array(step_diffs)
    result = one_sample_t_test(step_diffs, null_value=0.0)

    print(f"\n  Step-to-step noise change test (one-sample t-test):")
    print(f"    Mean |Δobs| per step: {result.mean:.6f}")
    print(f"    p-value (H0: Δ = 0): {result.p_value:.4f}")
    print(f"    Reject null: {result.reject_null}")

    # Step differences should be significantly greater than zero
    assert result.reject_null and result.mean > 0, (
        f"Step-to-step differences not significantly above zero "
        f"(mean={result.mean:.2e}, p={result.p_value:.4f}). Noise may be frozen."
    )


# =============================================================================
# Tests: Statistical Properties
# =============================================================================


def test_noise_has_zero_mean(noisy_env):
    """Verify noise has approximately zero mean (unbiased).

    Uses one-sample t-test to verify the mean is not significantly
    different from zero at the configured confidence level.
    """
    obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)

    # Extract noise for accelerometer
    noise_samples = extract_noise_samples(obs, IMU_ACCEL_SLICE)

    # One-sample t-test: is mean significantly different from 0?
    result = one_sample_t_test(noise_samples, null_value=0.0)

    std = np.std(noise_samples)

    print(f"\n  Accelerometer noise mean test (one-sample t-test):")
    print(f"    N samples: {result.n_samples:,}")
    print(f"    Mean: {result.mean:.6f}")
    print(f"    Std: {std:.6f}")
    print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI for mean: [{result.ci_low:.2e}, {result.ci_high:.2e}]")
    print(f"    p-value: {result.p_value:.4f}")
    print(f"    Reject null (mean ≠ 0): {result.reject_null}")

    # Test passes if we fail to reject the null hypothesis (mean = 0)
    assert not result.reject_null, (
        f"Noise mean {result.mean:.6f} is significantly different from zero "
        f"(p={result.p_value:.4f} < α={1 - CONFIDENCE_LEVEL:.2f})"
    )
