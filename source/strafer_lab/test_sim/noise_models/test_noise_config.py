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
    isaaclab -p -m pytest test_sim/noise_models/test_noise_config.py -v
"""

import numpy as np
from scipy import stats
import pytest
import torch

from test_sim.common import (
    N_SAMPLES_STEPS,
    one_sample_t_test,
)
from test_sim.noise_models.seeded_stream import CASE_ALPHA, CASE_CONFIDENCE, seeded_stream

from test_sim.noise_models.conftest import (
    collect_stationary_observations,
    IMU_ACCEL_SLICE,
)


# =============================================================================
# Tests: Noise Configuration Behavior
# =============================================================================


@seeded_stream(501)
def test_noise_affects_observations(noisy_env):
    """Verify observations have variance when noise is enabled.

    Uses one-sample t-test on per-step variances to verify that the
    mean variance across steps is significantly greater than zero.
    This replaces an arbitrary threshold with a proper statistical test. At
    alpha 1e-6 on its own stream it is scale-free: any nonzero noise gives
    t near √50 over the per-step variances' relative spread (at most about
    0.10 across 64 envs and 3 axes, so t of about 70 or more) against the
    5.6 needed at 49 df.
    """
    obs = collect_stationary_observations(noisy_env, 50)

    # Compute variance per step across envs for IMU accel
    per_step_var = obs[:, :, IMU_ACCEL_SLICE].var(dim=1).mean(dim=1)  # (n_steps,)
    var_samples = per_step_var.cpu().numpy()

    result = one_sample_t_test(var_samples, null_value=0.0, confidence_level=CASE_CONFIDENCE)

    print(f"\n  Noise-enabled observation variance test (one-sample t-test):")
    print(f"    Mean per-step variance: {result.mean:.6e}")
    print(f"    p-value (H0: variance = 0): {result.p_value:.4f}")
    print(f"    Reject null: {result.reject_null}")

    # Variance should be SIGNIFICANTLY greater than zero
    assert result.reject_null and result.mean > 0, (
        f"Observation variance not significantly above zero "
        f"(mean={result.mean:.2e}, p={result.p_value:.4f}). Is noise enabled?"
    )


@seeded_stream(502)
def test_noise_is_independent_across_envs(noisy_env):
    """Verify that custom NoiseModel generates independent noise per environment.

    This is critical for effective multi-environment RL training.
    With independent noise, each environment experiences different sensor
    corruptions, providing diverse training samples.

    Statistical approach:
    - Compute Pearson correlation between noise from env 0 and env 1
    - Under null hypothesis of independence, test r ≈ 0
    Tested at alpha 1e-6 on its own stream: at 600 pairs a correlation of
    0.25 still fails (power 0.9), and a stream shared between envs (r 1)
    fails at any n.
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
    print(f"    Significance level (α): {CASE_ALPHA:.0e}")

    # Test: correlation should not be significantly different from zero
    assert p_value > CASE_ALPHA, (
        f"Noise appears correlated across environments: "
        f"r={correlation:.4f}, p={p_value:.4e} < α={CASE_ALPHA:.0e}"
    )


@seeded_stream(503)
def test_noise_changes_each_step(noisy_env):
    """Verify noise is different at each timestep (not frozen).

    Uses one-sample t-test on mean absolute step-to-step differences
    to verify the changes are significantly greater than zero. At alpha
    1e-6 on its own stream it is scale-free: any step-to-step change gives
    t near √49 over the per-step means' relative spread (about 0.055 across
    64 envs and 3 axes, so t of about 130) against the 5.6 needed at 48 df.
    """
    obs = collect_stationary_observations(noisy_env, 50)

    # Mean absolute difference between consecutive steps
    step_diffs = []
    for i in range(len(obs) - 1):
        diff = (obs[i + 1, :, IMU_ACCEL_SLICE] - obs[i, :, IMU_ACCEL_SLICE]).abs().mean()
        step_diffs.append(diff.item())

    step_diffs = np.array(step_diffs)
    result = one_sample_t_test(step_diffs, null_value=0.0, confidence_level=CASE_CONFIDENCE)

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


@seeded_stream(504)
def test_noise_has_zero_mean(noisy_env):
    """Verify the accelerometer's noise, its observation minus its clean value, has zero mean.

    Each step the clean value is the term evaluated on the state the step's
    observation was computed on, scaled as the manager scales it (the term has
    no clip, and its noise is added before the scale). Each env carries its
    own bias for the whole read (drawn per axis at the reset, then drifting),
    so the unit of the test is the per-(env, axis) mean noise: 192 values,
    tested against zero with a one-sample t-test at alpha 1e-6 on its own
    stream. At 192 means a mean noise of 0.011 m/s² (46 % of the bias draws'
    spread) still fails (power 0.9).
    """
    names = noisy_env.observation_manager._group_obs_term_names["policy"]
    term = noisy_env.observation_manager._group_obs_term_cfgs["policy"][names.index("imu_linear_acceleration")]
    assert names[0] == "imu_linear_acceleration" and IMU_ACCEL_SLICE == slice(0, 3), "the accel term is not first"
    assert term.clip is None and term.modifiers is None, "a clip or modifier would bend the noise"

    clean = []
    obs = collect_stationary_observations(
        noisy_env, N_SAMPLES_STEPS,
        on_step=lambda env: clean.append(term.func(env, **term.params) * term.scale),
    )
    noise = obs[:, :, IMU_ACCEL_SLICE] - torch.stack(clean, dim=0)  # (steps, envs, 3), observation units
    assert noise.std() > 0, "the accelerometer term carries no noise"
    per_env_axis = (noise.mean(dim=0) / term.scale).flatten().cpu().numpy()  # m/s²
    result = one_sample_t_test(per_env_axis, null_value=0.0, confidence_level=CASE_CONFIDENCE)

    print(f"\n  Accelerometer noise mean test (observation minus clean value, one-sample t-test):")
    print(f"    per-(env, axis) means: {result.n_samples}")
    print(f"    mean of means: {result.mean:.6f} m/s², sd {np.std(per_env_axis, ddof=1):.6f} m/s²")
    print(f"    noise sd over all samples: {(noise / term.scale).std().item():.6f} m/s²")
    print(f"    t: {result.t_statistic:.4f}, p-value: {result.p_value:.4g}")

    assert not result.reject_null, (
        f"Noise mean {result.mean:.6f} m/s² is significantly different from zero "
        f"(p={result.p_value:.4g} < α={CASE_ALPHA:.0e})"
    )
