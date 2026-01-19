# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for observation pipeline.

These tests verify that observations are:
- Correctly shaped and structured
- Within expected normalized ranges (clamped to [-1, 1])
- Physically accurate (gravity, velocity, encoders match expected physics)

Observation structure for NoCam config (15 dims total):
- imu_linear_acceleration: (3,) - normalized by max_accel (156.96 m/s²)
- imu_angular_velocity: (3,) - normalized by max_angular_vel (34.9 rad/s)
- wheel_encoder_velocities: (4,) - normalized by max_ticks_per_sec (5000)
- goal_position: (2,) - relative [x, y] to goal (meters, NOT normalized)
- last_action: (3,) - previous [vx, vy, omega] command ([-1, 1])

Usage:
    cd IsaacLab
    pytest ../source/strafer_lab/test/integration/test_observation_pipeline.py -v

    Or via isaaclab.bat:
    ./isaaclab.bat -p -m pytest ../source/strafer_lab/test/integration/test_observation_pipeline.py -v
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
)

# =============================================================================
# Physical Constants
# =============================================================================

# Normalization constants (from strafer_env_cfg.py)
MAX_ACCEL = 156.96       # m/s² (BNO055 ±16g range)
MAX_ANGULAR_VEL = 34.9   # rad/s (BNO055 ±2000 dps)
MAX_TICKS_PER_SEC = 5000.0  # encoder ticks/sec

# Robot physical parameters (from actions.py)
WHEEL_RADIUS = 0.048     # 48mm wheel radius
MAX_WHEEL_RPM = 312.0    # goBILDA 5203 max RPM
MAX_WHEEL_ANGULAR_VEL = MAX_WHEEL_RPM * 2 * np.pi / 60  # ~32.67 rad/s

# Physics constants
GRAVITY = 9.81  # m/s²

# =============================================================================
# Statistical Test Configuration
# =============================================================================

# Sample sizes for statistical tests
N_SAMPLES_STAT = 50          # Samples per env for statistical tests
N_STEPS_STABILITY = 50       # Steps for NaN/Inf stability check
N_STEPS_CLAMP = 100          # Steps for clamping validation
N_SETTLE_STEPS = 30          # Steps to let physics settle before measurement
N_STEADY_STATE_STEPS = 30    # Steps to reach steady-state after command
N_RESETS = 5                 # Number of reset validations

# Statistical thresholds
CONFIDENCE_LEVEL = 0.95      # 95% confidence interval
P_VALUE_THRESHOLD = 0.01     # Significance level for t-tests (1%)
                             # p > 0.01 means fail to reject null hypothesis
SIGMA_THRESHOLD = 3.0        # Number of std devs for significance (3σ)

# Numerical tolerances
NORM_BOUND = 1.001           # Max normalized value (1.0 + float epsilon)
MIN_SIGNAL_THRESHOLD = 0.01  # Minimum expected signal (avoid false zeros)
MAX_CV_THRESHOLD = 0.10      # Max coefficient of variation for uniform motion
MIN_WHEEL_VEL = 0.05         # Min normalized wheel velocity to be "spinning"
MIN_GYRO_DELTA = 0.02        # Min gyro change to detect rotation
MAX_GOAL_DISTANCE = 20.0     # Max reasonable goal distance (meters)

# =============================================================================
# Test Environment Configuration
# =============================================================================

NUM_ENVS = 4                 # Number of parallel environments for tests
DEVICE = "cuda:0"            # Device to run tests on


# =============================================================================
# Module-scoped Fixture (one env for all tests)
# =============================================================================

# Module-level environment - created once, shared across all tests
# This is necessary because ManagerBasedRLEnv creates its own SimulationContext
# and Isaac Sim only allows one context per process.
_module_env = None


def _get_or_create_env():
    """Get or create the shared test environment."""
    global _module_env
    if _module_env is None:
        cfg = StraferNavEnvCfg_NoCam()
        cfg.scene.num_envs = NUM_ENVS
        cfg.actions = ActionsCfg_Ideal()
        cfg.observations = ObsCfg_NoCam_Ideal()
        _module_env = ManagerBasedRLEnv(cfg)
        _module_env.reset()
    return _module_env


@pytest.fixture(scope="module")
def strafer_env():
    """Provide the shared Strafer environment for tests.
    
    Uses module scope because ManagerBasedRLEnv creates its own SimulationContext
    and Isaac Sim only allows one context per process. All tests share this env.
    """
    env = _get_or_create_env()
    yield env
    # Note: cleanup happens at module teardown via atexit or pytest hooks


def pytest_sessionfinish(session, exitstatus):
    """Clean up the environment after all tests complete."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None
    simulation_app.close()


# =============================================================================
# Tests
# =============================================================================

def test_no_nan_inf(strafer_env):
    """Test 1: No NaN or Inf values in observations during operation."""
    env = strafer_env
    nan_count = 0
    inf_count = 0
    
    for step in range(N_STEPS_STABILITY):
        # Random actions to exercise all code paths
        actions = torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
        obs_dict, _, _, _, _ = env.step(actions)
        
        policy_obs = obs_dict["policy"]
        
        if torch.isnan(policy_obs).any():
            nan_count += 1
        if torch.isinf(policy_obs).any():
            inf_count += 1
    
    assert nan_count == 0, f"Found NaN values in {nan_count}/{N_STEPS_STABILITY} steps"
    assert inf_count == 0, f"Found Inf values in {inf_count}/{N_STEPS_STABILITY} steps"


def test_observation_structure(strafer_env):
    """Test 2: Observation term structure matches expected dimensions."""
    env = strafer_env
    obs_manager = env.observation_manager
    
    if hasattr(obs_manager, '_group_obs_term_dim'):
        group_cfg = obs_manager._group_obs_term_dim
        
        assert "policy" in group_cfg, "Missing 'policy' observation group"
        
        term_dims = group_cfg["policy"]
        # Expected: imu_accel(3) + imu_gyro(3) + encoders(4) + goal(2) + action(3) = 15
        expected_dims = [(3,), (3,), (4,), (2,), (3,)]
        
        assert len(term_dims) == len(expected_dims), \
            f"Expected {len(expected_dims)} terms, got {len(term_dims)}"
        
        for i, (actual, expected) in enumerate(zip(term_dims, expected_dims)):
            assert actual == expected, f"Term {i}: expected {expected}, got {actual}"
        
        total = sum(d[0] for d in term_dims)
        assert total == 15, f"Expected 15 total dims, got {total}"
    else:
        # Fallback: verify total dimension
        obs_dict, _ = env.reset()
        total_dim = obs_dict["policy"].shape[1]
        assert total_dim == 15, f"Expected 15 dims, got {total_dim}"


def test_normalized_clamping(strafer_env):
    """Test 3: All normalized observations are clamped to [-1, 1]."""
    env = strafer_env
    env.reset()
    
    all_obs = []
    
    for step in range(N_STEPS_CLAMP):
        # Aggressive oscillating actions to push limits
        t = step / N_STEPS_CLAMP
        actions = torch.tensor([[
            np.sin(2 * np.pi * t),       # vx oscillates full range
            np.cos(2 * np.pi * t),       # vy oscillates full range
            np.sin(4 * np.pi * t)        # omega oscillates
        ]], device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
        
        obs_dict, _, _, _, _ = env.step(actions)
        all_obs.append(obs_dict["policy"].cpu())
    
    all_obs = torch.cat(all_obs, dim=0)
    
    # Extract components
    imu_accel = all_obs[:, 0:3]
    imu_gyro = all_obs[:, 3:6]
    encoders = all_obs[:, 6:10]
    goal_pos = all_obs[:, 10:12]
    last_action = all_obs[:, 12:15]
    
    imu_accel_max = imu_accel.abs().max().item()
    imu_gyro_max = imu_gyro.abs().max().item()
    encoder_max = encoders.abs().max().item()
    last_action_max = last_action.abs().max().item()
    
    # All normalized sensors are clamped to [-1, 1]
    assert imu_accel_max <= NORM_BOUND, f"IMU accel not clamped: {imu_accel_max:.4f}"
    assert imu_gyro_max <= NORM_BOUND, f"IMU gyro not clamped: {imu_gyro_max:.4f}"
    assert encoder_max <= NORM_BOUND, f"Encoder not clamped: {encoder_max:.4f}"
    assert last_action_max <= NORM_BOUND, f"Last action not clamped: {last_action_max:.4f}"
    
    # Verify normalization is happening (values shouldn't all be tiny)
    assert imu_accel_max > MIN_SIGNAL_THRESHOLD, \
        f"IMU accel suspiciously small ({imu_accel_max:.4f})"
    
    # Goal position is NOT normalized, but should be reasonable
    assert goal_pos.abs().max().item() < MAX_GOAL_DISTANCE, \
        f"Goal position unreasonably large: {goal_pos.abs().max().item():.2f}m"


def test_imu_gravity(strafer_env):
    """Test 4: IMU measures gravity correctly when robot is stationary."""
    env = strafer_env
    env.reset()
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    
    # Let robot settle
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)
    
    # Collect IMU readings while stationary
    gravity_samples = []
    
    for _ in range(N_SAMPLES_STAT):
        obs_dict, _, _, _, _ = env.step(zero_action)
        imu_accel = obs_dict["policy"][:, 0:3].cpu()
        imu_raw = imu_accel * MAX_ACCEL  # De-normalize
        magnitudes = torch.norm(imu_raw, dim=1)
        gravity_samples.extend(magnitudes.numpy().tolist())
    
    gravity_samples = np.array(gravity_samples)
    n = len(gravity_samples)
    
    # Compute statistics
    sample_mean = gravity_samples.mean()
    sample_std = gravity_samples.std(ddof=1)
    standard_error = sample_std / np.sqrt(n)
    
    # Confidence Interval
    alpha = 1 - CONFIDENCE_LEVEL
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    ci_lower = sample_mean - t_critical * standard_error
    ci_upper = sample_mean + t_critical * standard_error
    
    # One-sample t-test
    t_statistic, p_value = stats.ttest_1samp(gravity_samples, GRAVITY)
    
    # Test: gravity in CI or p-value indicates consistency
    gravity_in_ci = ci_lower <= GRAVITY <= ci_upper
    consistent_with_gravity = p_value > P_VALUE_THRESHOLD
    
    assert gravity_in_ci or consistent_with_gravity, \
        f"Gravity measurement inconsistent: mean={sample_mean:.3f}, " \
        f"CI=[{ci_lower:.3f}, {ci_upper:.3f}], p={p_value:.4f}"
    
    # Test: Z-axis dominates (robot is upright)
    imu_readings = []
    for _ in range(N_SAMPLES_STAT // 2):
        obs_dict, _, _, _, _ = env.step(zero_action)
        imu_accel = obs_dict["policy"][:, 0:3].cpu() * MAX_ACCEL
        imu_readings.append(imu_accel)
    imu_readings = torch.stack(imu_readings, dim=0)
    mean_components = imu_readings.mean(dim=(0, 1)).numpy()
    
    xy_magnitude = np.sqrt(mean_components[0]**2 + mean_components[1]**2)
    z_magnitude = abs(mean_components[2])
    xy_std = imu_readings[:, :, :2].std().item()
    
    z_dominates = z_magnitude > xy_magnitude + SIGMA_THRESHOLD * xy_std
    assert z_dominates, f"Robot appears tilted: Z={z_magnitude:.2f}, XY={xy_magnitude:.2f}"


def test_encoder_forward_motion(strafer_env):
    """Test 5: Encoder velocity matches commanded forward motion."""
    env = strafer_env
    env.reset()
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    
    # Let settle
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)
    
    # Command half-speed forward
    forward_action = torch.tensor([[0.5, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
    
    # Reach steady state
    for _ in range(N_STEADY_STATE_STEPS):
        env.step(forward_action)
    
    # Collect encoder readings
    encoder_samples = []
    for _ in range(N_SAMPLES_STAT):
        obs_dict, _, _, _, _ = env.step(forward_action)
        encoders = obs_dict["policy"][:, 6:10].cpu()
        encoder_samples.append(encoders)
    
    encoder_samples = torch.stack(encoder_samples, dim=0)
    flat_encoders = encoder_samples.reshape(-1, 4)
    mean_encoders = flat_encoders.mean(dim=0)
    n = flat_encoders.shape[0]
    
    # Verify all wheels spinning (t-test vs 0)
    for i in range(4):
        wheel_samples = flat_encoders[:, i].numpy()
        t_stat, p_val = stats.ttest_1samp(wheel_samples, 0.0)
        spinning = p_val < P_VALUE_THRESHOLD and abs(mean_encoders[i]) > MIN_WHEEL_VEL
        assert spinning, f"Wheel {i} not spinning: mean={mean_encoders[i]:.4f}, p={p_val:.4f}"
    
    # Check mecanum pattern: diagonal pairs same sign, pairs opposite
    signs = torch.sign(mean_encoders)
    diag1_same = signs[0] == signs[3]
    diag2_same = signs[1] == signs[2]
    pairs_opposite = signs[0] != signs[1]
    
    assert diag1_same and diag2_same and pairs_opposite, \
        f"Encoder pattern doesn't match mecanum kinematics: signs={signs.numpy()}"
    
    # All wheels similar magnitude (CV test)
    cv = mean_encoders.abs().std() / mean_encoders.abs().mean()
    assert cv < MAX_CV_THRESHOLD, f"Wheel speeds vary too much: CV={cv.item():.4f}"


def test_gyro_rotation(strafer_env):
    """Test 6: Gyro detects angular velocity during rotation."""
    env = strafer_env
    env.reset()
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    
    # Let settle
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)
    
    # Collect baseline gyro (stationary)
    baseline_samples = []
    for _ in range(N_SAMPLES_STAT):
        obs_dict, _, _, _, _ = env.step(zero_action)
        gyro = obs_dict["policy"][:, 3:6].cpu()
        baseline_samples.append(gyro)
    baseline_samples = torch.stack(baseline_samples, dim=0)
    baseline_flat = baseline_samples.reshape(-1, 3)
    
    # Command rotation
    rotate_action = torch.tensor([[0.0, 0.0, 0.5]], device=env.device).repeat(env.num_envs, 1)
    
    # Reach rotation speed
    for _ in range(N_STEADY_STATE_STEPS):
        env.step(rotate_action)
    
    # Collect rotating gyro
    rotating_samples = []
    for _ in range(N_SAMPLES_STAT):
        obs_dict, _, _, _, _ = env.step(rotate_action)
        gyro = obs_dict["policy"][:, 3:6].cpu()
        rotating_samples.append(gyro)
    rotating_samples = torch.stack(rotating_samples, dim=0)
    rotating_flat = rotating_samples.reshape(-1, 3)
    
    rotating_mean = rotating_flat.mean(dim=0)
    baseline_mean = baseline_flat.mean(dim=0)
    
    # Two-sample t-test: rotating Z vs baseline Z
    t_stat, p_value = stats.ttest_ind(
        rotating_flat[:, 2].numpy(),
        baseline_flat[:, 2].numpy()
    )
    
    assert p_value < P_VALUE_THRESHOLD, \
        f"Gyro Z change not statistically significant: p={p_value:.4f}"
    
    effect_size = abs(rotating_mean[2] - baseline_mean[2])
    assert effect_size > MIN_GYRO_DELTA, \
        f"Gyro Z effect size too small: {effect_size:.4f}"
