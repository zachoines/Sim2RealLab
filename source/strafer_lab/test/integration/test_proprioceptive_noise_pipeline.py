# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for proprioceptive sensor noise through the full environment pipeline.

These tests validate that IMU and encoder noise is correctly applied during environment
stepping, not just when calling noise model classes in isolation.

Approach:
1. Create an environment with IDEAL config (no noise)
2. Create an environment with REALISTIC config (noise enabled)
3. Step both environments with ZERO actions (stationary robot)
4. For stationary robot, any observation variance comes from noise
5. Statistically validate noise characteristics match config

Note: Due to Isaac Sim's single-context limitation, we can only run one env
config per test session. We use a workaround: compare repeated observations
of a stationary robot to measure noise-induced variance.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/integration/test_proprioceptive_noise_pipeline.py -v
"""

# Isaac Sim must be launched before importing Isaac Lab modules
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# --- Imports that require Isaac Sim runtime ---

import torch
import numpy as np
from scipy import stats
import pytest

from isaaclab.envs import ManagerBasedRLEnv

# Import shared utilities from common module
from test.common import (
    variance_ratio_test,
    one_sample_t_test,
    CONFIDENCE_LEVEL,
    NUM_ENVS,
    N_SETTLE_STEPS,
    N_SAMPLES_STEPS,
    DEVICE,
    IMU_ACCEL_MAX,
    IMU_GYRO_MAX,
    ENCODER_VEL_MAX,
)

# Import robot utilities from common module
from test.common.robot import (
    get_env_origins,
    reset_robot_pose,
    freeze_robot_in_place,
    clear_frozen_state,
)

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Ideal,
    ObsCfg_NoCam_Ideal,
    ObsCfg_NoCam_Realistic,
)
from strafer_lab.tasks.navigation.sim_real_cfg import (
    REAL_ROBOT_CONTRACT,
    get_imu_accel_noise,
    get_imu_gyro_noise,
    get_encoder_noise,
)
from strafer_lab.tasks.navigation.mdp.noise_models import DelayBuffer


# =============================================================================
# Test Configuration
# =============================================================================
# Note: CONFIDENCE_LEVEL, DEVICE, NUM_ENVS, N_SETTLE_STEPS, N_SAMPLES_STEPS,
# IMU_ACCEL_MAX, IMU_GYRO_MAX, ENCODER_VEL_MAX are imported from test.common

# Observation term indices (for NoCam config: 15 total dims)
# imu_accel(3) + imu_gyro(3) + encoders(4) + goal(2) + action(3)
IMU_ACCEL_SLICE = slice(0, 3)
IMU_GYRO_SLICE = slice(3, 6)
ENCODER_SLICE = slice(6, 10)
GOAL_SLICE = slice(10, 12)
ACTION_SLICE = slice(12, 15)

# =============================================================================
# Noise Pipeline: Raw → Noise → Scale → Clip
# =============================================================================
#
# The Isaac Lab observation pipeline processes in order:
#   1. Observation function returns RAW sensor data (e.g., m/s², rad/s, ticks/s)
#   2. Noise model corrupts the RAW data (physically correct!)
#   3. Scale parameter normalizes (scale = 1/max_value)
#   4. Clip limits to [-1, 1]
#
# Therefore, measured noise in normalized output space:
#   measured_std = raw_noise_std * scale = raw_noise_std / max_value
#

# Expected noise std values (derived from actual config via helper functions)
# These are RAW noise std values (before scale is applied)
_ACCEL_NOISE_CFG = get_imu_accel_noise(REAL_ROBOT_CONTRACT)
_GYRO_NOISE_CFG = get_imu_gyro_noise(REAL_ROBOT_CONTRACT)
_ENCODER_NOISE_CFG = get_encoder_noise(REAL_ROBOT_CONTRACT)

# RAW noise std (in physical units)
RAW_ACCEL_STD = _ACCEL_NOISE_CFG.accel_noise_std if _ACCEL_NOISE_CFG else 0.0  # m/s²
RAW_GYRO_STD = _GYRO_NOISE_CFG.gyro_noise_std if _GYRO_NOISE_CFG else 0.0      # rad/s
RAW_ENCODER_STD = (_ENCODER_NOISE_CFG.velocity_noise_std * _ENCODER_NOISE_CFG.max_velocity
                   if _ENCODER_NOISE_CFG else 0.0)  # ticks/s

# NORMALIZED noise std (what we measure in observation output)
# normalized_std = raw_std / max_value = raw_std * scale
EXPECTED_ACCEL_STD = RAW_ACCEL_STD / IMU_ACCEL_MAX if RAW_ACCEL_STD else 0.0
EXPECTED_GYRO_STD = RAW_GYRO_STD / IMU_GYRO_MAX if RAW_GYRO_STD else 0.0
EXPECTED_ENCODER_STD = RAW_ENCODER_STD / ENCODER_VEL_MAX if RAW_ENCODER_STD else 0.0

# =============================================================================
# Theoretical Bounds Using First Differences
# =============================================================================
#
# PROBLEM: Computing variance of raw IMU observations is complex because:
#   - Bias drift accumulates as a random walk (correlated over time)
#   - Mean subtraction removes part of drift variance non-trivially
#
# SOLUTION: Use first differences (Allan variance-like approach)
#
# The IMU noise model produces (in RAW space):
#   y_t = signal + bias_t + white_noise_t
#   
# Where bias evolves as (with drift scaled by sqrt(dt) for proper time integration):
#   bias_t = bias_{t-1} + drift_step_t,  drift_step ~ N(0, (drift_rate * sqrt(dt))²)
#
# For a STATIONARY robot (signal constant), first differences give:
#   Δy_t = y_t - y_{t-1}
#        = (bias_t - bias_{t-1}) + (white_noise_t - white_noise_{t-1})
#        = drift_step_t + (v_t - v_{t-1})
#
# The variance (in RAW space) is:
#   Var(Δy_raw) = drift_rate² * dt + 2 * white_noise_std²
#
# Where dt = 1/control_frequency_hz (e.g., 1/30 for 30Hz control).
#
# After scaling to normalized space:
#   Var(Δy_normalized) = Var(Δy_raw) / max_value²
#   Std(Δy_normalized) = Std(Δy_raw) / max_value
#
# This gives us a PRECISE THEORETICAL PREDICTION we can test against!
#

def _calculate_first_diff_theoretical_std(
    white_noise_std: float,
    drift_rate: float,
    control_frequency_hz: float = 30.0,
) -> float:
    """Calculate theoretical std of first differences for IMU noise.
    
    For the noise model:
        y_t = bias_t + white_noise_t
        bias_t = bias_{t-1} + N(0, (drift_rate * sqrt(dt))²)
    
    The first difference is:
        Δy_t = y_t - y_{t-1} = drift_step + (v_t - v_{t-1})
    
    With variance (drift scaled by sqrt(dt) for random walk):
        Var(Δy) = drift_rate² * dt + 2 * white_noise_std²
        
    Where dt = 1/control_frequency_hz.
        
    Args:
        white_noise_std: White noise standard deviation per step
        drift_rate: Bias drift rate (std per second, scaled by sqrt(dt) per step)
        control_frequency_hz: Control loop frequency (default 30Hz)
        
    Returns:
        Theoretical std of first differences
    """
    import math
    dt = 1.0 / control_frequency_hz
    # Drift variance per step: (drift_rate * sqrt(dt))^2 = drift_rate^2 * dt
    variance = drift_rate**2 * dt + 2 * white_noise_std**2
    return math.sqrt(variance)


def _calculate_first_diff_theoretical_std_normalized(
    white_noise_std: float,
    drift_rate: float,
    max_value: float,
    control_frequency_hz: float = 30.0,
) -> float:
    """Calculate theoretical std of first differences in NORMALIZED space.
    
    Args:
        white_noise_std: White noise std in RAW units (m/s² or rad/s)
        drift_rate: Bias drift rate in RAW units (per second)
        max_value: Max sensor value for normalization
        control_frequency_hz: Control loop frequency (default 30Hz)
        
    Returns:
        Theoretical std of first differences in normalized space
    """
    # Theoretical std in RAW space
    raw_theoretical_std = _calculate_first_diff_theoretical_std(
        white_noise_std, drift_rate, control_frequency_hz
    )
    # Convert to normalized space
    return raw_theoretical_std / max_value

# Theoretical std for each sensor type using first-differences approach
# Note: Values are in NORMALIZED space (what we measure from observations)
if _ACCEL_NOISE_CFG:
    ACCEL_DIFF_STD_THEORETICAL = _calculate_first_diff_theoretical_std_normalized(
        _ACCEL_NOISE_CFG.accel_noise_std,
        _ACCEL_NOISE_CFG.accel_bias_drift_rate,
        IMU_ACCEL_MAX,
        _ACCEL_NOISE_CFG.control_frequency_hz,
    )
else:
    ACCEL_DIFF_STD_THEORETICAL = 0.0

if _GYRO_NOISE_CFG:
    GYRO_DIFF_STD_THEORETICAL = _calculate_first_diff_theoretical_std_normalized(
        _GYRO_NOISE_CFG.gyro_noise_std,
        _GYRO_NOISE_CFG.gyro_bias_drift_rate,
        IMU_GYRO_MAX,
        _GYRO_NOISE_CFG.control_frequency_hz,
    )
else:
    GYRO_DIFF_STD_THEORETICAL = 0.0


# =============================================================================
# Encoder Noise Theoretical Bounds
# =============================================================================
#
# Pipeline: raw obs (ticks/s) → noise → scale (1/5000) → clip
#
# EncoderNoiseModel applies to RAW data:
#   1. Gaussian noise: std = velocity_noise_std * max_velocity (in ticks/s)
#   2. Missed ticks (prob=p_miss): discrete ±1 tick errors
#   3. Extra ticks (prob=p_extra): discrete ±1 tick errors
#   4. Quantization: rounds to nearest tick
#
# Total RAW variance:
#   Var_raw = (velocity_noise_std * max_velocity)²
#           + (p_miss + p_extra) * 1²    (discrete tick errors)
#           + 1/12                       (quantization noise)
#
# After scaling to normalized space (scale = 1/max_velocity):
#   Var_normalized = Var_raw / max_velocity²
#   Std_normalized = Std_raw / max_velocity
#

def _calculate_encoder_theoretical_std(
    velocity_noise_std: float,
    max_velocity: float,
    missed_tick_prob: float,
    extra_tick_prob: float,
    enable_quantization: bool,
) -> float:
    """Calculate theoretical std for encoder noise in NORMALIZED space.
    
    Pipeline: raw obs (ticks/s) → noise (in raw space) → scale (1/max) → output
    
    Noise is applied to RAW data, then scaled:
    1. Gaussian noise in raw space: std_raw = velocity_noise_std * max_velocity
    2. Tick errors: ±1 tick in raw space
    3. Quantization: round to nearest tick
    4. Scale to normalized: output = raw / max_velocity
    
    Therefore:
        std_normalized = std_raw / max_velocity
    
    Args:
        velocity_noise_std: Fractional noise (e.g., 0.02 = 2%)
        max_velocity: Max velocity for scaling (e.g., 5000 ticks/sec)
        missed_tick_prob: Probability of missing a tick
        extra_tick_prob: Probability of extra tick
        enable_quantization: Whether quantization is enabled
        
    Returns:
        Theoretical std of encoder noise in normalized space
    """
    import math
    
    # RAW space variances (in ticks² or ticks²/s²)
    # Primary Gaussian noise in raw space
    gaussian_variance_raw = (velocity_noise_std * max_velocity) ** 2
    
    # Discrete tick errors: each contributes variance = prob * (1 tick)²
    tick_error_variance_raw = (missed_tick_prob + extra_tick_prob) * 1.0
    
    # Quantization noise: uniform on [-0.5, 0.5] has variance 1/12
    quantization_variance_raw = (1.0 / 12.0) if enable_quantization else 0.0
    
    total_variance_raw = gaussian_variance_raw + tick_error_variance_raw + quantization_variance_raw
    std_raw = math.sqrt(total_variance_raw)
    
    # Convert to normalized space: std_normalized = std_raw / max_velocity
    std_normalized = std_raw / max_velocity
    return std_normalized


# Compute encoder theoretical std
if _ENCODER_NOISE_CFG:
    ENCODER_STD_THEORETICAL = _calculate_encoder_theoretical_std(
        _ENCODER_NOISE_CFG.velocity_noise_std,
        _ENCODER_NOISE_CFG.max_velocity,
        _ENCODER_NOISE_CFG.missed_tick_prob,
        _ENCODER_NOISE_CFG.extra_tick_prob,
        _ENCODER_NOISE_CFG.enable_quantization,
    )
else:
    ENCODER_STD_THEORETICAL = 0.0


# =============================================================================
# Module-scoped Fixtures
# =============================================================================

_module_env = None
_env_config_type = None  # Track which config is loaded


def _get_or_create_env(use_noise: bool = True):
    """Get or create the shared test environment.
    
    Args:
        use_noise: If True, use Realistic config with noise enabled.
                   If False, use Ideal config without noise.
    """
    global _module_env, _env_config_type
    
    config_type = "realistic" if use_noise else "ideal"
    
    # If env exists with different config, we can't change it (single context)
    if _module_env is not None:
        if _env_config_type != config_type:
            raise RuntimeError(
                f"Cannot switch env config from '{_env_config_type}' to '{config_type}'. "
                "Isaac Sim only allows one SimulationContext per process."
            )
        return _module_env
    
    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Ideal()  # Always ideal actions for predictability
    
    if use_noise:
        cfg.observations = ObsCfg_NoCam_Realistic()
    else:
        cfg.observations = ObsCfg_NoCam_Ideal()
    
    _module_env = ManagerBasedRLEnv(cfg)
    _env_config_type = config_type
    _module_env.reset()
    
    return _module_env


@pytest.fixture(scope="module")
def noisy_env():
    """Provide Strafer environment with realistic noise enabled (no camera)."""
    env = _get_or_create_env(use_noise=True)
    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up the environment after all tests complete."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None
    simulation_app.close()


# =============================================================================
# Helper Functions
# =============================================================================
# Robot control functions (get_env_origins, reset_robot_pose, freeze_robot_in_place,
# clear_frozen_state) are imported from test.common.robot module.


def collect_stationary_observations(env, n_steps: int, freeze_robot: bool = True) -> torch.Tensor:
    """Collect observations from a stationary robot over multiple steps.

    With zero actions and settled physics, any observation variance
    comes from sensor noise.

    Args:
        env: The Isaac Lab environment
        n_steps: Number of observation steps to collect
        freeze_robot: If True (default), continuously zero out robot velocities
                      to eliminate physics settling noise. This ensures measured
                      variance comes purely from sensor noise.

    Returns:
        Tensor of shape (n_steps, num_envs, obs_dim)
    """
    # Full environment reset to ensure clean state
    env.reset()

    # Reset robots to fixed positions at grid origins (avoids randomization overlap)
    reset_robot_pose(env)

    # Clear frozen state so it captures fresh pose on next freeze call
    clear_frozen_state()

    # Zero action (stationary)
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)

    # Let physics settle
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)
        if freeze_robot:
            freeze_robot_in_place(env)

    # Collect observations
    observations = []
    for _ in range(n_steps):
        obs_dict, _, _, _, _ = env.step(zero_action)
        observations.append(obs_dict["policy"].clone())
        if freeze_robot:
            freeze_robot_in_place(env)

    return torch.stack(observations, dim=0)


def extract_noise_samples(observations: torch.Tensor, term_slice: slice) -> np.ndarray:
    """Extract samples for a specific observation term and flatten for analysis.
    
    For a stationary robot, the "signal" should be constant and any
    deviation is noise. We compute deviation from per-environment mean.
    
    Args:
        observations: (n_steps, num_envs, obs_dim)
        term_slice: Slice for the observation term
        
    Returns:
        Flattened noise samples as numpy array
    """
    term_obs = observations[:, :, term_slice]  # (n_steps, num_envs, term_dim)
    
    # Compute mean per env (the "true" signal)
    mean_per_env = term_obs.mean(dim=0, keepdim=True)  # (1, num_envs, term_dim)
    
    # Deviation from mean is the noise
    noise = term_obs - mean_per_env
    
    return noise.cpu().numpy().flatten()


def extract_first_differences(observations: torch.Tensor, term_slice: slice) -> np.ndarray:
    """Extract first differences for a specific observation term.
    
    First differences eliminate the need to estimate the "true" signal
    and give us a clean theoretical prediction for variance:
    
        Δy_t = y_t - y_{t-1}
        Var(Δy) = drift_rate² + 2 * white_noise_std²
    
    Args:
        observations: (n_steps, num_envs, obs_dim)
        term_slice: Slice for the observation term
        
    Returns:
        Flattened first-difference samples as numpy array
        Shape: ((n_steps-1) * num_envs * term_dim,)
    """
    term_obs = observations[:, :, term_slice]  # (n_steps, num_envs, term_dim)
    
    # Compute first differences: y[t] - y[t-1]
    first_diffs = term_obs[1:] - term_obs[:-1]  # (n_steps-1, num_envs, term_dim)
    
    return first_diffs.cpu().numpy().flatten()


# =============================================================================
# Tests: IMU Noise
# =============================================================================

class TestIMUNoiseFromEnv:
    """Test IMU noise characteristics from environment observations."""

    def test_accel_noise_std_matches_config(self, noisy_env):
        """Verify accelerometer noise std matches theoretical prediction using chi-squared test.

        Uses first-differences approach for precise theoretical bounds:

            Δy_t = y_t - y_{t-1} = drift_step + (v_t - v_{t-1})
            Var(Δy) = drift_rate² + 2 * white_noise_std²

        Statistical approach:
        - For Gaussian noise, (n-1) * S² / σ² ~ χ²(n-1)
        - We compute a confidence interval for our measured std
        - Test passes if theoretical std falls within this CI
        """
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)

        # Use first differences for clean theoretical prediction
        diff_samples = extract_first_differences(obs, IMU_ACCEL_SLICE)
        n_samples = len(diff_samples)
        measured_var = np.var(diff_samples)
        measured_std = np.sqrt(measured_var)

        # Chi-squared variance ratio test
        expected_var = ACCEL_DIFF_STD_THEORETICAL ** 2
        result = variance_ratio_test(measured_var, expected_var, n_samples)

        print(f"\n  Accelerometer first-difference analysis (chi-squared test):")
        print(f"    Config values:")
        print(f"      white_noise_std: {EXPECTED_ACCEL_STD:.6f}")
        print(f"      drift_rate: {_ACCEL_NOISE_CFG.accel_bias_drift_rate:.6f}")
        print(f"    Theoretical Var(Dy) = drift^2 * dt + 2*white^2")
        print(f"      = {_ACCEL_NOISE_CFG.accel_bias_drift_rate**2:.9f} + 2*{EXPECTED_ACCEL_STD**2:.9f}")
        print(f"    Statistical analysis (n={n_samples:,} samples, df={result.df:,}):")
        print(f"      Measured std(Dy): {measured_std:.6f}")
        print(f"      Theoretical std(Dy): {ACCEL_DIFF_STD_THEORETICAL:.6f}")
        print(f"      Variance ratio (measured/theoretical): {result.ratio:.4f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for ratio: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
        print(f"    Result: Variance ratio {'is' if result.in_ci else 'is NOT'} within CI")

        assert result.in_ci, (
            f"Accelerometer variance ratio {result.ratio:.4f} not within "
            f"{CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
        )

    def test_gyro_noise_std_matches_config(self, noisy_env):
        """Verify gyroscope noise std matches theoretical prediction using chi-squared test.

        Uses first-differences approach for precise theoretical bounds:

            Δy_t = y_t - y_{t-1} = drift_step + (v_t - v_{t-1})
            Var(Δy) = drift_rate² + 2 * white_noise_std²

        Statistical approach:
        - For Gaussian noise, (n-1) * S² / σ² ~ χ²(n-1)
        - We compute a confidence interval for our measured std
        - Test passes if theoretical std falls within this CI
        """
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)

        # Use first differences for clean theoretical prediction
        diff_samples = extract_first_differences(obs, IMU_GYRO_SLICE)
        n_samples = len(diff_samples)
        measured_var = np.var(diff_samples)
        measured_std = np.sqrt(measured_var)

        # Chi-squared variance ratio test
        expected_var = GYRO_DIFF_STD_THEORETICAL ** 2
        result = variance_ratio_test(measured_var, expected_var, n_samples)

        print(f"\n  Gyroscope first-difference analysis (chi-squared test):")
        print(f"    Config values:")
        print(f"      white_noise_std: {EXPECTED_GYRO_STD:.6f}")
        print(f"      drift_rate: {_GYRO_NOISE_CFG.gyro_bias_drift_rate:.6f}")
        print(f"    Theoretical Var(Dy) = drift^2 * dt + 2*white^2")
        print(f"      = {_GYRO_NOISE_CFG.gyro_bias_drift_rate**2:.9f} + 2*{EXPECTED_GYRO_STD**2:.9f}")
        print(f"    Statistical analysis (n={n_samples:,} samples, df={result.df:,}):")
        print(f"      Measured std(Dy): {measured_std:.6f}")
        print(f"      Theoretical std(Dy): {GYRO_DIFF_STD_THEORETICAL:.6f}")
        print(f"      Variance ratio (measured/theoretical): {result.ratio:.4f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for ratio: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
        print(f"    Result: Variance ratio {'is' if result.in_ci else 'is NOT'} within CI")

        assert result.in_ci, (
            f"Gyroscope variance ratio {result.ratio:.4f} not within "
            f"{CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
        )


# =============================================================================
# Tests: Encoder Noise
# =============================================================================

class TestEncoderNoiseFromEnv:
    """Test encoder noise characteristics from environment observations."""

    def test_encoder_noise_std_matches_config(self, noisy_env):
        """Verify encoder velocity noise std matches theoretical prediction using chi-squared test.
        
        EncoderNoiseModel applies:
        1. Gaussian noise: std = velocity_noise_std * max_velocity
        2. Missed ticks: discrete ±1 errors with probability p_miss
        3. Extra ticks: discrete ±1 errors with probability p_extra
        4. Quantization: rounds to nearest integer (adds uniform noise variance 1/12)
        
        Total variance = (velocity_noise_std * max_velocity)² + p_miss + p_extra + 1/12
        
        Statistical approach:
        - Compute sample variance from noise samples
        - Use chi-squared variance ratio test
        - Test passes if variance ratio falls within CI
        
        Note: Unlike IMU noise, encoder noise has NO drift component,
        so we can directly measure std without using first-differences.
        """
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)
        
        noise_samples = extract_noise_samples(obs, ENCODER_SLICE)
        n_samples = len(noise_samples)
        measured_var = np.var(noise_samples)
        measured_std = np.sqrt(measured_var)
        
        # Chi-squared variance ratio test
        expected_var = ENCODER_STD_THEORETICAL ** 2
        result = variance_ratio_test(measured_var, expected_var, n_samples)
        
        # Get config values for display
        vel_noise_std = _ENCODER_NOISE_CFG.velocity_noise_std
        max_vel = _ENCODER_NOISE_CFG.max_velocity
        p_miss = _ENCODER_NOISE_CFG.missed_tick_prob
        p_extra = _ENCODER_NOISE_CFG.extra_tick_prob
        quant_enabled = _ENCODER_NOISE_CFG.enable_quantization
        
        # Compute variance components for display
        gaussian_var = (vel_noise_std * max_vel) ** 2
        tick_var = p_miss + p_extra
        quant_var = (1.0 / 12.0) if quant_enabled else 0.0
        total_var_raw = gaussian_var + tick_var + quant_var
        
        print(f"\n  Encoder noise analysis (chi-squared test):")
        print(f"    Config values:")
        print(f"      velocity_noise_std: {vel_noise_std:.6f}")
        print(f"      max_velocity: {max_vel:.1f}")
        print(f"      missed_tick_prob: {p_miss:.6f}")
        print(f"      extra_tick_prob: {p_extra:.6f}")
        print(f"      enable_quantization: {quant_enabled}")
        print(f"    Theoretical variance breakdown (raw space):")
        print(f"      Gaussian: ({vel_noise_std:.4f} × {max_vel:.0f})² = {gaussian_var:.6f}")
        print(f"      Tick errors: {p_miss:.6f} + {p_extra:.6f} = {tick_var:.6f}")
        print(f"      Quantization: 1/12 = {quant_var:.6f}")
        print(f"      Total variance (raw): {total_var_raw:.6f}")
        print(f"    Statistical analysis (n={n_samples:,} samples, df={result.df:,}):")
        print(f"      Measured std: {measured_std:.6f}")
        print(f"      Theoretical std: {ENCODER_STD_THEORETICAL:.6f}")
        print(f"      Variance ratio (measured/theoretical): {result.ratio:.4f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for ratio: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
        print(f"    Result: Variance ratio {'is' if result.in_ci else 'is NOT'} within CI")
        
        assert result.in_ci, (
            f"Encoder variance ratio {result.ratio:.4f} not within "
            f"{CONFIDENCE_LEVEL*100:.0f}% CI [{result.ci_low:.4f}, {result.ci_high:.4f}]"
        )


# =============================================================================
# Tests: Noise Enables/Disables Correctly
# =============================================================================

class TestNoiseConfiguration:
    """Test that noise configuration is respected."""
    
    def test_noise_affects_observations(self, noisy_env):
        """Verify observations have variance when noise is enabled.
        
        This is a sanity check that the enable_corruption flag works.
        """
        obs = collect_stationary_observations(noisy_env, 50)
        
        # Compute total variance across all observations
        total_variance = obs.var().item()
        
        print(f"\n  Total observation variance (with noise): {total_variance:.6f}")
        
        # With noise enabled, there should be measurable variance
        # even for a stationary robot
        assert total_variance > 1e-6, \
            f"Observation variance too low ({total_variance:.2e}). Is noise enabled?"
    
    def test_noise_is_independent_across_envs(self, noisy_env):
        """Verify that custom NoiseModel generates independent noise per environment.
        
        This is critical for effective multi-environment RL training.
        With independent noise, each environment experiences different sensor
        corruptions, providing diverse training samples.
        
        Statistical approach:
        - Compute Pearson correlation between noise from env 0 and env 1
        - Under null hypothesis of independence, test r ≈ 0
        - Use p-value from pearsonr for hypothesis test
        
        Note: The custom IMUNoiseModel, EncoderNoiseModel, etc. use torch.randn_like()
        which generates independent samples per element, achieving true per-env independence.
        """
        obs = collect_stationary_observations(noisy_env, 200)  # Fewer steps OK for correlation
        
        # Extract IMU accel observations: (n_steps, num_envs, 3)
        imu_accel = obs[:, :, IMU_ACCEL_SLICE]
        
        # Subtract per-env mean to get noise component only
        # The mean represents the physics signal (gravity + robot state)
        env_mean = imu_accel.mean(dim=0, keepdim=True)  # (1, num_envs, 3)
        noise_only = imu_accel - env_mean  # (n_steps, num_envs, 3)
        
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
        print(f"    Note: p > α indicates independent noise (fail to reject H0)")

        # Test: correlation should not be significantly different from zero
        # If p-value > α, we fail to reject the null hypothesis of independence
        assert p_value > (1 - CONFIDENCE_LEVEL), (
            f"Noise appears correlated across environments: "
            f"r={correlation:.4f}, p={p_value:.4e} < α={1 - CONFIDENCE_LEVEL:.2f}"
        )
    
    def test_noise_changes_each_step(self, noisy_env):
        """Verify noise is different at each timestep (not frozen)."""
        obs = collect_stationary_observations(noisy_env, 50)
        
        # Compare consecutive steps
        step_diffs = []
        for i in range(len(obs) - 1):
            diff = (obs[i+1, :, IMU_ACCEL_SLICE] - obs[i, :, IMU_ACCEL_SLICE]).abs().mean()
            step_diffs.append(diff.item())
        
        mean_diff = np.mean(step_diffs)
        
        print(f"\n  Mean absolute change between steps: {mean_diff:.6f}")
        
        # If noise is working, observations should change between steps
        assert mean_diff > 1e-6, \
            f"Observations not changing between steps ({mean_diff:.2e}). Noise may be frozen."


# =============================================================================
# Tests: Statistical Properties
# =============================================================================

class TestNoiseStatisticalProperties:
    """Test statistical properties of observation noise."""
    
    def test_noise_has_zero_mean(self, noisy_env):
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


# =============================================================================
# Tests: Observation Pipeline Structure and Physics
# =============================================================================

# Physical constants for gravity test
GRAVITY = 9.81  # m/s²

# Expected observation structure for NoCam config
EXPECTED_OBS_DIMS = [(3,), (3,), (4,), (2,), (3,)]  # imu_accel, imu_gyro, encoders, goal, action
EXPECTED_TOTAL_DIM = 15


class TestObservationPipeline:
    """Test observation pipeline structure and physics correctness.

    These tests validate that:
    - Observation structure matches expected dimensions
    - Physics signals (like gravity) are correctly measured
    """

    def test_observation_structure(self, noisy_env):
        """Verify observation term structure matches expected dimensions.

        Observation structure for NoCam config (15 dims total):
        - imu_linear_acceleration: (3,) - normalized by max_accel
        - imu_angular_velocity: (3,) - normalized by max_angular_vel
        - wheel_encoder_velocities: (4,) - normalized by max_ticks_per_sec
        - goal_position: (2,) - relative [x, y] to goal (meters)
        - last_action: (3,) - previous [vx, vy, omega] command
        """
        obs_manager = noisy_env.observation_manager

        if hasattr(obs_manager, '_group_obs_term_dim'):
            group_cfg = obs_manager._group_obs_term_dim

            assert "policy" in group_cfg, "Missing 'policy' observation group"

            term_dims = group_cfg["policy"]

            print(f"\n  Observation structure validation:")
            print(f"    Number of terms: {len(term_dims)} (expected {len(EXPECTED_OBS_DIMS)})")

            assert len(term_dims) == len(EXPECTED_OBS_DIMS), (
                f"Expected {len(EXPECTED_OBS_DIMS)} terms, got {len(term_dims)}"
            )

            for i, (actual, expected) in enumerate(zip(term_dims, EXPECTED_OBS_DIMS)):
                term_names = ["imu_accel", "imu_gyro", "encoders", "goal", "action"]
                print(f"    Term {i} ({term_names[i]}): {actual} (expected {expected})")
                assert actual == expected, f"Term {i}: expected {expected}, got {actual}"

            total = sum(d[0] for d in term_dims)
            print(f"    Total dimensions: {total} (expected {EXPECTED_TOTAL_DIM})")
            assert total == EXPECTED_TOTAL_DIM, f"Expected {EXPECTED_TOTAL_DIM} total dims, got {total}"
        else:
            # Fallback: verify total dimension from actual observation
            obs_dict, _ = noisy_env.reset()
            total_dim = obs_dict["policy"].shape[1]
            print(f"\n  Observation total dimension: {total_dim} (expected {EXPECTED_TOTAL_DIM})")
            assert total_dim == EXPECTED_TOTAL_DIM, f"Expected {EXPECTED_TOTAL_DIM} dims, got {total_dim}"

    def test_imu_gravity(self, noisy_env):
        """Verify IMU measures gravity correctly when robot is stationary.

        When the robot is stationary and upright, the accelerometer should measure
        approximately 9.81 m/s² total acceleration (from gravity). The Z-axis
        should dominate since the robot is upright.

        Uses one-sample t-test to verify the measured gravity magnitude is
        consistent with the expected value.
        """
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)

        # Extract and de-normalize IMU readings
        imu_accel_normalized = obs[:, :, IMU_ACCEL_SLICE]  # (n_steps, num_envs, 3)
        imu_accel_raw = imu_accel_normalized * IMU_ACCEL_MAX  # Convert to m/s²

        # Compute magnitude for each sample
        magnitudes = torch.norm(imu_accel_raw, dim=2)  # (n_steps, num_envs)
        gravity_samples = magnitudes.cpu().numpy().flatten()

        # One-sample t-test: is measured gravity consistent with expected?
        result = one_sample_t_test(gravity_samples, null_value=GRAVITY)

        print(f"\n  IMU gravity measurement (one-sample t-test):")
        print(f"    N samples: {result.n_samples:,}")
        print(f"    Expected gravity: {GRAVITY:.3f} m/s²")
        print(f"    Measured mean: {result.mean:.3f} m/s²")
        print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{result.ci_low:.3f}, {result.ci_high:.3f}] m/s²")
        print(f"    p-value: {result.p_value:.4f}")
        print(f"    Gravity in CI: {not result.reject_null}")

        # Test 1: Gravity should be within CI (or p-value indicates consistency)
        # Note: With noise enabled, we may have small systematic bias, so we use
        # a slightly relaxed check - gravity should be close to expected
        gravity_error = abs(result.mean - GRAVITY)
        max_acceptable_error = 0.5  # Allow 0.5 m/s² error (5% of g)

        assert gravity_error < max_acceptable_error or not result.reject_null, (
            f"Gravity measurement inconsistent: mean={result.mean:.3f} m/s², "
            f"expected={GRAVITY:.3f} m/s², error={gravity_error:.3f} m/s²"
        )

        # Test 2: Z-axis should dominate (robot is upright)
        mean_components = imu_accel_raw.mean(dim=(0, 1)).cpu().numpy()  # Mean across steps and envs
        xy_magnitude = np.sqrt(mean_components[0]**2 + mean_components[1]**2)
        z_magnitude = abs(mean_components[2])

        print(f"    Component analysis:")
        print(f"      Mean X: {mean_components[0]:.3f} m/s²")
        print(f"      Mean Y: {mean_components[1]:.3f} m/s²")
        print(f"      Mean Z: {mean_components[2]:.3f} m/s²")
        print(f"      XY magnitude: {xy_magnitude:.3f} m/s²")
        print(f"      Z magnitude: {z_magnitude:.3f} m/s²")

        # Z should be much larger than XY for an upright robot
        assert z_magnitude > xy_magnitude, (
            f"Robot appears tilted: Z={z_magnitude:.2f} m/s², XY={xy_magnitude:.2f} m/s²"
        )


# =============================================================================
# Tests: Observation Latency (DelayBuffer)
# =============================================================================

class TestNoiseModelLatency:
    """Test observation latency through noise models.

    The per-sensor latency feature (POINT_OF_IMPROVEMENT #7) adds DelayBuffer
    to each noise model. These tests verify the latency implementation.
    """

    def test_delay_buffer_zero_passthrough(self):
        """Verify delay_steps=0 returns input unchanged."""
        buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=3, delay_steps=0, device=DEVICE)

        # Generate random input
        data = torch.randn(NUM_ENVS, 3, device=DEVICE)
        output = buffer(data)

        # Should be identical (no delay)
        assert torch.allclose(output, data), "Zero delay should pass through unchanged"

    def test_delay_buffer_exact_delay(self):
        """Verify output is delayed by exactly delay_steps."""
        delay_steps = 3
        buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=4, delay_steps=delay_steps, device=DEVICE)

        # Store inputs for verification
        inputs = []
        outputs = []

        # Push data through buffer
        for i in range(delay_steps + 5):
            data = torch.full((NUM_ENVS, 4), float(i + 1), device=DEVICE)
            inputs.append(data.clone())
            outputs.append(buffer(data).clone())

        # First delay_steps outputs should be zeros (buffer was empty)
        for i in range(delay_steps):
            assert torch.allclose(outputs[i], torch.zeros_like(outputs[i])), (
                f"Output {i} should be zeros (buffer warming up)"
            )

        # After warming up, output should be exactly delay_steps behind input
        for i in range(delay_steps, delay_steps + 5):
            expected = inputs[i - delay_steps]
            assert torch.allclose(outputs[i], expected), (
                f"Output {i} should equal input {i - delay_steps}"
            )

    def test_delay_buffer_reset_clears_history(self):
        """Verify reset() clears buffer history."""
        delay_steps = 2
        buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=3, delay_steps=delay_steps, device=DEVICE)

        # Fill buffer with non-zero data
        for _ in range(delay_steps + 1):
            data = torch.randn(NUM_ENVS, 3, device=DEVICE)
            buffer(data)

        # Reset buffer
        buffer.reset()

        # After reset, outputs should be zeros again
        test_data = torch.ones(NUM_ENVS, 3, device=DEVICE)
        output = buffer(test_data)
        assert torch.allclose(output, torch.zeros_like(output)), (
            "After reset, delayed output should be zeros"
        )

    def test_delay_buffer_per_env_reset(self):
        """Verify reset(env_ids) only clears specified environments."""
        delay_steps = 1
        buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=2, delay_steps=delay_steps, device=DEVICE)

        # Fill buffer with identifiable data
        fill_data = torch.arange(NUM_ENVS, device=DEVICE).unsqueeze(1).expand(-1, 2).float()
        buffer(fill_data)

        # Now buffer has fill_data stored; next call should return it
        # First, push again to have fill_data in delayed position
        second_data = fill_data + 100
        output1 = buffer(second_data)

        # output1 should be fill_data
        assert torch.allclose(output1, fill_data), "First delayed output should be fill_data"

        # Reset only first 10 environments
        reset_ids = list(range(10))
        buffer.reset(reset_ids)

        # Push new data
        third_data = fill_data + 200
        output2 = buffer(third_data)

        # Reset envs should get zeros, others should get second_data
        assert torch.allclose(output2[:10], torch.zeros(10, 2, device=DEVICE)), (
            "Reset env outputs should be zeros"
        )
        assert torch.allclose(output2[10:], second_data[10:]), (
            "Non-reset env outputs should be previous data"
        )

    def test_imu_noise_with_latency(self, noisy_env):
        """Verify IMU observations are delayed when latency_steps > 0.

        This test validates that the per-sensor latency feature works correctly
        through the full noise model pipeline. Since REAL_ROBOT_CONTRACT has
        imu_latency_steps=0 by default, this test mainly verifies the wiring
        works without causing errors.
        """
        obs = collect_stationary_observations(noisy_env, 50)

        # Verify observations were collected successfully
        assert obs.shape[0] == 50, "Should collect 50 steps"
        assert obs.shape[1] == NUM_ENVS, f"Should have {NUM_ENVS} environments"

        # IMU accel observations should have variance (noise is enabled)
        imu_variance = obs[:, :, IMU_ACCEL_SLICE].var().item()
        assert imu_variance > 1e-8, "IMU should have measurable noise variance"

    def test_encoder_noise_with_latency(self, noisy_env):
        """Verify encoder observations work with latency configuration.

        Since REAL_ROBOT_CONTRACT has encoder_latency_steps=0 by default,
        this test verifies the noise model works correctly without latency.
        """
        obs = collect_stationary_observations(noisy_env, 50)

        # Encoder observations should have variance (noise is enabled)
        encoder_variance = obs[:, :, ENCODER_SLICE].var().item()
        assert encoder_variance > 1e-8, "Encoder should have measurable noise variance"


# =============================================================================
# Run module cleanup on import (for running outside pytest)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

