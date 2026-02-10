# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for noise configuration behavior.

These tests validate general noise model behavior:
- Noise is enabled/disabled correctly via enable_corruption
- Noise is independent across parallel environments
- Noise changes each timestep (not frozen)

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_noise_config.py -v
"""

import numpy as np
from scipy import stats
import pytest

from test.common import CONFIDENCE_LEVEL

from test.noise_models.conftest import (
    collect_stationary_observations,
    IMU_ACCEL_SLICE,
)


# =============================================================================
# Tests: Noise Configuration Behavior
# =============================================================================


def test_noise_affects_observations(noisy_env):
    """Verify observations have variance when noise is enabled.

    This is a sanity check that the enable_corruption flag works.
    """
    obs = collect_stationary_observations(noisy_env, 50)

    # Compute total variance across all observations
    total_variance = obs.var().item()

    print(f"\n  Total observation variance (with noise): {total_variance:.6f}")

    # With noise enabled, there should be measurable variance
    # even for a stationary robot
    assert total_variance > 1e-6, (
        f"Observation variance too low ({total_variance:.2e}). Is noise enabled?"
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
    """Verify noise is different at each timestep (not frozen)."""
    obs = collect_stationary_observations(noisy_env, 50)

    # Compare consecutive steps
    step_diffs = []
    for i in range(len(obs) - 1):
        diff = (obs[i + 1, :, IMU_ACCEL_SLICE] - obs[i, :, IMU_ACCEL_SLICE]).abs().mean()
        step_diffs.append(diff.item())

    mean_diff = np.mean(step_diffs)

    print(f"\n  Mean absolute change between steps: {mean_diff:.6f}")

    # If noise is working, observations should change between steps
    assert mean_diff > 1e-6, (
        f"Observations not changing between steps ({mean_diff:.2e}). Noise may be frozen."
    )
