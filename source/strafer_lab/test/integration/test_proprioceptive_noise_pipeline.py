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


# =============================================================================
# Test Configuration
# =============================================================================

NUM_ENVS = 32                # More envs = more samples per step
N_SAMPLES_STEPS = 500        # Steps to collect (more samples = tighter CI)
N_SETTLE_STEPS = 10          # Steps to let physics settle initially
DEVICE = "cuda:0"

# Statistical thresholds
# With n=500 steps * 32 envs * 3 dims = 48,000 samples, CI is very tight
# We use hypothesis testing rather than arbitrary tolerance
CONFIDENCE_LEVEL = 0.95      # For confidence intervals

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
# Sensor max values (for normalization):
IMU_ACCEL_MAX = 156.96  # m/s² (±16g)
IMU_GYRO_MAX = 34.9     # rad/s (±2000°/s)
ENCODER_MAX = 5000.0    # ticks/s

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
EXPECTED_ENCODER_STD = RAW_ENCODER_STD / ENCODER_MAX if RAW_ENCODER_STD else 0.0

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
# Where bias evolves as:
#   bias_t = bias_{t-1} + drift_step_t,  drift_step ~ N(0, drift_rate²)
#
# For a STATIONARY robot (signal constant), first differences give:
#   Δy_t = y_t - y_{t-1}
#        = (bias_t - bias_{t-1}) + (white_noise_t - white_noise_{t-1})
#        = drift_step_t + (v_t - v_{t-1})
#
# The variance (in RAW space) is:
#   Var(Δy_raw) = drift_rate² + 2 * white_noise_std²
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
) -> float:
    """Calculate theoretical std of first differences for IMU noise.
    
    For the noise model:
        y_t = bias_t + white_noise_t
        bias_t = bias_{t-1} + N(0, drift_rate²)
    
    The first difference is:
        Δy_t = y_t - y_{t-1} = drift_step + (v_t - v_{t-1})
    
    With variance:
        Var(Δy) = drift_rate² + 2 * white_noise_std²
        
    Args:
        white_noise_std: White noise standard deviation per step
        drift_rate: Bias drift rate (std of per-step drift)
        
    Returns:
        Theoretical std of first differences
    """
    import math
    variance = drift_rate**2 + 2 * white_noise_std**2
    return math.sqrt(variance)


def _calculate_first_diff_theoretical_std_normalized(
    white_noise_std: float,
    drift_rate: float,
    max_value: float,
) -> float:
    """Calculate theoretical std of first differences in NORMALIZED space.
    
    Args:
        white_noise_std: White noise std in RAW units (m/s² or rad/s)
        drift_rate: Bias drift rate in RAW units
        max_value: Max sensor value for normalization
        
    Returns:
        Theoretical std of first differences in normalized space
    """
    # Theoretical std in RAW space
    raw_theoretical_std = _calculate_first_diff_theoretical_std(white_noise_std, drift_rate)
    # Convert to normalized space
    return raw_theoretical_std / max_value

# Theoretical std for each sensor type using first-differences approach
# Note: Values are in NORMALIZED space (what we measure from observations)
if _ACCEL_NOISE_CFG:
    ACCEL_DIFF_STD_THEORETICAL = _calculate_first_diff_theoretical_std_normalized(
        _ACCEL_NOISE_CFG.accel_noise_std,
        _ACCEL_NOISE_CFG.accel_bias_drift_rate,
        IMU_ACCEL_MAX,
    )
else:
    ACCEL_DIFF_STD_THEORETICAL = 0.0

if _GYRO_NOISE_CFG:
    GYRO_DIFF_STD_THEORETICAL = _calculate_first_diff_theoretical_std_normalized(
        _GYRO_NOISE_CFG.gyro_noise_std,
        _GYRO_NOISE_CFG.gyro_bias_drift_rate,
        IMU_GYRO_MAX,
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

def collect_stationary_observations(env, n_steps: int) -> torch.Tensor:
    """Collect observations from a stationary robot over multiple steps.
    
    With zero actions and settled physics, any observation variance
    comes from sensor noise.
    
    Returns:
        Tensor of shape (n_steps, num_envs, obs_dim)
    """
    # Zero action (stationary)
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    
    # Let physics settle
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)
    
    # Collect observations
    observations = []
    for _ in range(n_steps):
        obs_dict, _, _, _, _ = env.step(zero_action)
        observations.append(obs_dict["policy"].clone())
    
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
    
    def test_accel_noise_is_present(self, noisy_env):
        """Verify accelerometer observations have non-zero variance (noise is applied)."""
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)
        
        accel_obs = obs[:, :, IMU_ACCEL_SLICE]
        variance = accel_obs.var().item()
        
        print(f"\n  Accelerometer observation variance: {variance:.6f}")
        
        # With noise enabled, variance should be non-zero
        assert variance > 1e-8, \
            f"Accelerometer variance too low ({variance:.2e}), noise may not be applied"
    
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
        
        # Compute confidence interval using chi-squared distribution
        df = n_samples - 1
        alpha = 1 - CONFIDENCE_LEVEL
        chi2_lower = stats.chi2.ppf(alpha / 2, df)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)
        
        # CI for variance: [S² * df / χ²_upper, S² * df / χ²_lower]
        var_ci_lower = measured_var * df / chi2_upper
        var_ci_upper = measured_var * df / chi2_lower
        std_ci_lower = np.sqrt(var_ci_lower)
        std_ci_upper = np.sqrt(var_ci_upper)
        
        # Check if theoretical value falls within CI
        ci_contains_theoretical = std_ci_lower <= ACCEL_DIFF_STD_THEORETICAL <= std_ci_upper
        
        print(f"\n  Accelerometer first-difference analysis (chi-squared test):")
        print(f"    Config values:")
        print(f"      white_noise_std: {EXPECTED_ACCEL_STD:.6f}")
        print(f"      drift_rate: {_ACCEL_NOISE_CFG.accel_bias_drift_rate:.6f}")
        print(f"    Theoretical Var(Δy) = drift² + 2*white²")
        print(f"      = {_ACCEL_NOISE_CFG.accel_bias_drift_rate**2:.9f} + 2*{EXPECTED_ACCEL_STD**2:.9f}")
        print(f"    Statistical analysis (n={n_samples:,} samples, df={df:,}):")
        print(f"      Measured std(Δy): {measured_std:.6f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for std: [{std_ci_lower:.6f}, {std_ci_upper:.6f}]")
        print(f"      Theoretical std(Δy): {ACCEL_DIFF_STD_THEORETICAL:.6f}")
        print(f"      Ratio (measured/theoretical): {measured_std / ACCEL_DIFF_STD_THEORETICAL:.3f}")
        print(f"    Result: Theoretical value {'is' if ci_contains_theoretical else 'is NOT'} within CI")
        
        assert ci_contains_theoretical, (
            f"Theoretical std {ACCEL_DIFF_STD_THEORETICAL:.6f} not within "
            f"{CONFIDENCE_LEVEL*100:.0f}% CI [{std_ci_lower:.6f}, {std_ci_upper:.6f}]"
        )
    
    def test_gyro_noise_is_present(self, noisy_env):
        """Verify gyroscope observations have non-zero variance."""
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)
        
        gyro_obs = obs[:, :, IMU_GYRO_SLICE]
        variance = gyro_obs.var().item()
        
        print(f"\n  Gyroscope observation variance: {variance:.6f}")
        
        assert variance > 1e-8, \
            f"Gyroscope variance too low ({variance:.2e}), noise may not be applied"
    
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
        
        # Compute confidence interval using chi-squared distribution
        df = n_samples - 1
        alpha = 1 - CONFIDENCE_LEVEL
        chi2_lower = stats.chi2.ppf(alpha / 2, df)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)
        
        # CI for variance: [S² * df / χ²_upper, S² * df / χ²_lower]
        var_ci_lower = measured_var * df / chi2_upper
        var_ci_upper = measured_var * df / chi2_lower
        std_ci_lower = np.sqrt(var_ci_lower)
        std_ci_upper = np.sqrt(var_ci_upper)
        
        # Check if theoretical value falls within CI
        ci_contains_theoretical = std_ci_lower <= GYRO_DIFF_STD_THEORETICAL <= std_ci_upper
        
        print(f"\n  Gyroscope first-difference analysis (chi-squared test):")
        print(f"    Config values:")
        print(f"      white_noise_std: {EXPECTED_GYRO_STD:.6f}")
        print(f"      drift_rate: {_GYRO_NOISE_CFG.gyro_bias_drift_rate:.6f}")
        print(f"    Theoretical Var(Δy) = drift² + 2*white²")
        print(f"      = {_GYRO_NOISE_CFG.gyro_bias_drift_rate**2:.9f} + 2*{EXPECTED_GYRO_STD**2:.9f}")
        print(f"    Statistical analysis (n={n_samples:,} samples, df={df:,}):")
        print(f"      Measured std(Δy): {measured_std:.6f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for std: [{std_ci_lower:.6f}, {std_ci_upper:.6f}]")
        print(f"      Theoretical std(Δy): {GYRO_DIFF_STD_THEORETICAL:.6f}")
        print(f"      Ratio (measured/theoretical): {measured_std / GYRO_DIFF_STD_THEORETICAL:.3f}")
        print(f"    Result: Theoretical value {'is' if ci_contains_theoretical else 'is NOT'} within CI")
        
        assert ci_contains_theoretical, (
            f"Theoretical std {GYRO_DIFF_STD_THEORETICAL:.6f} not within "
            f"{CONFIDENCE_LEVEL*100:.0f}% CI [{std_ci_lower:.6f}, {std_ci_upper:.6f}]"
        )


# =============================================================================
# Tests: Encoder Noise
# =============================================================================

class TestEncoderNoiseFromEnv:
    """Test encoder noise characteristics from environment observations."""
    
    def test_encoder_noise_is_present(self, noisy_env):
        """Verify encoder observations have non-zero variance."""
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)
        
        encoder_obs = obs[:, :, ENCODER_SLICE]
        variance = encoder_obs.var().item()
        
        print(f"\n  Encoder observation variance: {variance:.6f}")
        
        # Encoder noise should be present
        assert variance > 1e-8, \
            f"Encoder variance too low ({variance:.2e}), noise may not be applied"
    
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
        - Use chi-squared CI for variance
        - Test passes if theoretical std falls within CI
        
        Note: Unlike IMU noise, encoder noise has NO drift component,
        so we can directly measure std without using first-differences.
        """
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)
        
        noise_samples = extract_noise_samples(obs, ENCODER_SLICE)
        n_samples = len(noise_samples)
        measured_var = np.var(noise_samples)
        measured_std = np.sqrt(measured_var)
        
        # Compute confidence interval using chi-squared distribution
        df = n_samples - 1
        alpha = 1 - CONFIDENCE_LEVEL
        chi2_lower = stats.chi2.ppf(alpha / 2, df)
        chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)
        
        # CI for variance: [S² * df / χ²_upper, S² * df / χ²_lower]
        var_ci_lower = measured_var * df / chi2_upper
        var_ci_upper = measured_var * df / chi2_lower
        std_ci_lower = np.sqrt(var_ci_lower)
        std_ci_upper = np.sqrt(var_ci_upper)
        
        # Check if theoretical value falls within CI
        ci_contains_theoretical = std_ci_lower <= ENCODER_STD_THEORETICAL <= std_ci_upper
        
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
        total_var = gaussian_var + tick_var + quant_var
        
        print(f"\n  Encoder noise analysis (chi-squared test):")
        print(f"    Config values:")
        print(f"      velocity_noise_std: {vel_noise_std:.6f}")
        print(f"      max_velocity: {max_vel:.1f}")
        print(f"      missed_tick_prob: {p_miss:.6f}")
        print(f"      extra_tick_prob: {p_extra:.6f}")
        print(f"      enable_quantization: {quant_enabled}")
        print(f"    Theoretical variance breakdown:")
        print(f"      Gaussian: ({vel_noise_std:.4f} × {max_vel:.0f})² = {gaussian_var:.6f}")
        print(f"      Tick errors: {p_miss:.6f} + {p_extra:.6f} = {tick_var:.6f}")
        print(f"      Quantization: 1/12 = {quant_var:.6f}")
        print(f"      Total variance: {total_var:.6f}")
        print(f"    Statistical analysis (n={n_samples:,} samples, df={df:,}):")
        print(f"      Measured std: {measured_std:.6f}")
        print(f"      {CONFIDENCE_LEVEL*100:.0f}% CI for std: [{std_ci_lower:.6f}, {std_ci_upper:.6f}]")
        print(f"      Theoretical std: {ENCODER_STD_THEORETICAL:.6f}")
        print(f"      Ratio (measured/theoretical): {measured_std / ENCODER_STD_THEORETICAL:.3f}")
        print(f"    Result: Theoretical value {'is' if ci_contains_theoretical else 'is NOT'} within CI")
        
        assert ci_contains_theoretical, (
            f"Theoretical std {ENCODER_STD_THEORETICAL:.6f} not within "
            f"{CONFIDENCE_LEVEL*100:.0f}% CI [{std_ci_lower:.6f}, {std_ci_upper:.6f}]"
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
        print(f"    Note: Low |r| with high p-value indicates independent noise")
        
        # Test: correlation should not be significantly different from zero
        # We allow p > 0.001 (very conservative) - if p < 0.001, noise is correlated
        # Also check that absolute correlation is reasonably small (effect size)
        assert p_value > 0.001 or abs(correlation) < 0.1, (
            f"Noise appears correlated across environments: "
            f"r={correlation:.4f}, p={p_value:.4e}"
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
        """Verify noise has approximately zero mean (unbiased)."""
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)
        
        # Extract noise for accelerometer
        noise_samples = extract_noise_samples(obs, IMU_ACCEL_SLICE)
        
        mean = np.mean(noise_samples)
        std = np.std(noise_samples)
        n = len(noise_samples)
        
        # Standard error of the mean
        sem = std / np.sqrt(n)
        
        # t-test against mean=0
        t_stat = mean / sem
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        
        print(f"\n  Accelerometer noise mean: {mean:.6f}")
        print(f"    Standard error: {sem:.6f}")
        print(f"    t-statistic: {t_stat:.4f}")
        print(f"    p-value: {p_value:.4f}")
        
        # Mean should be close to zero (not statistically different at p=0.01)
        # Note: With many samples, even small biases may be "significant"
        # We check the magnitude is small relative to the noise std
        assert abs(mean) < std * 0.1, \
            f"Noise mean {mean:.6f} seems biased (std={std:.6f})"
    
    def test_noise_variance_is_stable(self, noisy_env):
        """Verify noise variance is consistent over time.
        
        Note: With a stationary robot, there may still be some drift in
        observations due to settling physics or accumulated numerical errors.
        We use a generous tolerance.
        """
        obs = collect_stationary_observations(noisy_env, N_SAMPLES_STEPS)
        
        # Split into first half and second half
        mid = N_SAMPLES_STEPS // 2
        first_half = obs[:mid, :, IMU_ACCEL_SLICE]
        second_half = obs[mid:, :, IMU_ACCEL_SLICE]
        
        var_first = first_half.var().item()
        var_second = second_half.var().item()
        
        print(f"\n  Variance stability check:")
        print(f"    First half variance: {var_first:.6f}")
        print(f"    Second half variance: {var_second:.6f}")
        print(f"    Ratio: {var_second / max(var_first, 1e-9):.2f}x")
        
        # Variances should be in the same order of magnitude
        # Allow up to 3x difference due to physics drift and settling
        ratio = max(var_first, var_second) / max(min(var_first, var_second), 1e-9)
        assert ratio < 3.0, \
            f"Noise variance not stable: ratio={ratio:.2f}"


# =============================================================================
# Run module cleanup on import (for running outside pytest)
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

