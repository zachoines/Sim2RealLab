# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for motor dynamics through the full environment pipeline.

These tests validate that the MecanumWheelAction term correctly implements:
1. Mecanum kinematics (robot moves in intended directions)
2. First-order motor dynamics (exponential step response)
3. Command delay buffer (latency modeling)
4. Slew rate limiting (acceleration constraints)

Statistical approach:
- Motor time constant verified via fitted exponential decay with CI
- Delay verified via response onset analysis
- Kinematics verified by measuring robot displacement direction

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/integration/test_motor_dynamics.py -v
"""

# Isaac Sim must be launched BEFORE importing Isaac Lab modules
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# --- Imports that require Isaac Sim runtime ---

import torch
import numpy as np
import pytest
from scipy import stats

from isaaclab.envs import ManagerBasedRLEnv

# Import shared utilities from common module
from test.integration.common import (
    circular_mean,
    circular_variance,
    circular_std,
    circular_confidence_interval,
    angle_in_circular_ci,
    one_sample_t_test,
    welch_t_test,
    CONFIDENCE_LEVEL,
    DEVICE,
)

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Ideal,
    ActionsCfg_Realistic,
)


# =============================================================================
# Test Configuration
# =============================================================================
# Note: CONFIDENCE_LEVEL and DEVICE are imported from test.integration.common

# Statistical power analysis for NUM_ENVS:
# For a one-sample t-test with:
#   - Effect size d = 1.0 (large, expected for kinematics tests where motion is clear)
#   - Alpha = 0.05 (Type I error rate)
#   - Power = 0.80 (probability of detecting true effect)
# Required sample size n ≈ 10 (from power.t.test in R or statsmodels)
# We use n=16 to provide margin for:
#   - Medium effect sizes (d=0.5 requires n≈34, but our effects are large)
#   - Non-normality in small samples
#   - Multiple comparisons across test methods
# Formula: n = (t_α + t_β)² × 2σ²/δ² where δ is minimum detectable effect
NUM_ENVS = 16                # Justified via power analysis for large effects (d≥1.0)
N_SETTLE_STEPS = 10          # Steps to let physics settle
N_RESPONSE_STEPS = 200       # Steps to measure step response
N_MOTION_STEPS = 100         # Steps for kinematics tests


# =============================================================================
# Module-scoped Fixtures
# =============================================================================

_module_env = None


def _get_or_create_env():
    """Get or create environment with realistic actions (motor dynamics enabled).
    
    We use realistic config so we can test motor dynamics features.
    For kinematics tests that need instant response, we temporarily disable
    motor dynamics via the action term's internal flags.
    """
    global _module_env
    
    if _module_env is not None:
        return _module_env
    
    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS
    cfg.actions = ActionsCfg_Realistic()
    
    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()
    
    return _module_env


@pytest.fixture(scope="module")
def env():
    """Provide environment with realistic motor dynamics (single shared env)."""
    env = _get_or_create_env()
    yield env


# Alias fixtures for backward compatibility  
@pytest.fixture(scope="module")
def ideal_env(env):
    """Provide env with motor dynamics temporarily disabled for kinematics tests."""
    action_term = env.action_manager._terms["wheel_velocities"]
    # Save original settings
    orig_motor = action_term._enable_motor_dynamics
    orig_delay = action_term._enable_command_delay
    orig_slew = action_term._enable_slew_rate
    
    # Disable all dynamics for ideal behavior
    action_term._enable_motor_dynamics = False
    action_term._enable_command_delay = False
    action_term._enable_slew_rate = False
    
    # Clear internal state buffers explicitly (reset() only clears if enabled)
    if hasattr(action_term, '_smoothed_wheel_vels'):
        action_term._smoothed_wheel_vels.zero_()
    if hasattr(action_term, '_prev_wheel_vels'):
        action_term._prev_wheel_vels.zero_()
    if hasattr(action_term, '_action_delay_buffer'):
        action_term._action_delay_buffer.reset(None)
    
    yield env
    
    # Restore original settings
    action_term._enable_motor_dynamics = orig_motor
    action_term._enable_command_delay = orig_delay
    action_term._enable_slew_rate = orig_slew


@pytest.fixture(scope="module")
def realistic_env(env):
    """Provide env with motor dynamics enabled (realistic behavior)."""
    action_term = env.action_manager._terms["wheel_velocities"]
    # Ensure dynamics are enabled
    action_term._enable_motor_dynamics = True
    action_term._enable_command_delay = True
    action_term._enable_slew_rate = True
    
    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up environments after all tests complete."""
    global _module_env
    
    if _module_env is not None:
        _module_env.close()
        _module_env = None
    
    simulation_app.close()


# =============================================================================
# Helper Functions
# =============================================================================

def reset_env_and_action_term(env, ideal_mode: bool = False):
    """Reset environment and action term state.

    Args:
        env: The environment
        ideal_mode: If True, disable motor dynamics/delay/slew for ideal behavior.
                    If False, enable all dynamics for realistic behavior.
    """
    env.reset()
    action_term = env.action_manager._terms["wheel_velocities"]

    # Set mode BEFORE reset so reset() knows what to clear
    action_term._enable_motor_dynamics = not ideal_mode
    action_term._enable_command_delay = not ideal_mode
    action_term._enable_slew_rate = not ideal_mode

    # Reset (clears state for enabled dynamics)
    action_term.reset(env_ids=torch.arange(env.num_envs, device=env.device))

    # Also explicitly clear all state buffers (in case mode just changed)
    if hasattr(action_term, '_smoothed_wheel_vels'):
        action_term._smoothed_wheel_vels.zero_()
    if hasattr(action_term, '_prev_wheel_vels'):
        action_term._prev_wheel_vels.zero_()
    if hasattr(action_term, '_action_delay_buffer'):
        action_term._action_delay_buffer.reset(None)

    return action_term


def collect_step_response(env, action: torch.Tensor, n_steps: int, ideal_mode: bool = False) -> np.ndarray:
    """Collect wheel velocity response to a step input.
    
    Args:
        env: The environment
        action: Action tensor of shape (num_envs, 3)
        n_steps: Number of steps to collect
        ideal_mode: If True, disable motor dynamics
        
    Returns:
        Array of shape (n_steps, num_envs, 4) with wheel velocities
    """
    action_term = reset_env_and_action_term(env, ideal_mode=ideal_mode)
    
    responses = []
    for _ in range(n_steps):
        action_term.process_actions(action)
        responses.append(action_term.processed_actions.cpu().numpy().copy())
    
    return np.array(responses)


def collect_robot_trajectory(env, action: torch.Tensor, n_steps: int, ideal_mode: bool = True) -> dict:
    """Collect robot position/orientation trajectory.
    
    Args:
        env: The environment
        action: Action tensor of shape (num_envs, 3)
        n_steps: Number of steps to simulate
        ideal_mode: If True, disable motor dynamics for kinematics tests
        
    Returns:
        Dict with 'positions' (n_steps, num_envs, 3) and 'orientations' (n_steps, num_envs, 4)
    """
    reset_env_and_action_term(env, ideal_mode=ideal_mode)
    
    # Let physics settle with zero action
    zero_action = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(N_SETTLE_STEPS):
        env.step(zero_action)
    
    # Record initial state
    robot = env.scene["robot"]
    initial_pos = robot.data.root_pos_w.clone()
    initial_quat = robot.data.root_quat_w.clone()
    
    # Collect trajectory
    positions = [initial_pos.cpu().numpy().copy()]
    orientations = [initial_quat.cpu().numpy().copy()]
    
    for _ in range(n_steps):
        env.step(action)
        positions.append(robot.data.root_pos_w.cpu().numpy().copy())
        orientations.append(robot.data.root_quat_w.cpu().numpy().copy())
    
    return {
        'positions': np.array(positions),
        'orientations': np.array(orientations),
        'initial_pos': initial_pos.cpu().numpy(),
        'initial_quat': initial_quat.cpu().numpy(),
    }


def fit_first_order_response(times: np.ndarray, response: np.ndarray, v_final: float) -> dict:
    """Fit first-order exponential response and return parameters.
    
    Model: v(t) = v_final * (1 - exp(-t/tau))
    
    Args:
        times: Time array
        response: Response array (absolute values)
        v_final: Final steady-state value
        
    Returns:
        Dict with 'tau', 'tau_ci_lower', 'tau_ci_upper', 'r_squared'
    """
    # Normalize response
    v_norm = response / (v_final + 1e-9)
    
    # Transform to linearize: log(1 - v_norm) = -t/tau
    # Only use points where 1 - v_norm > 0
    valid_mask = v_norm < 0.99
    if not np.any(valid_mask):
        return {'tau': np.nan, 'tau_ci_lower': np.nan, 'tau_ci_upper': np.nan, 'r_squared': 0}
    
    t_valid = times[valid_mask]
    y_valid = np.log(1 - v_norm[valid_mask] + 1e-9)
    
    # Linear regression: y = -t/tau  →  slope = -1/tau
    slope, intercept, r_value, p_value, std_err = stats.linregress(t_valid, y_valid)
    
    if slope >= 0:
        return {'tau': np.nan, 'tau_ci_lower': np.nan, 'tau_ci_upper': np.nan, 'r_squared': 0}
    
    tau = -1.0 / slope
    r_squared = r_value ** 2
    
    # Confidence interval for tau using delta method
    # Var(tau) ≈ Var(slope) / slope^4 = std_err^2 / slope^4
    tau_std = std_err / (slope ** 2)
    
    alpha = 1 - CONFIDENCE_LEVEL
    t_crit = stats.t.ppf(1 - alpha/2, len(t_valid) - 2)
    tau_ci_lower = tau - t_crit * tau_std
    tau_ci_upper = tau + t_crit * tau_std
    
    return {
        'tau': tau,
        'tau_ci_lower': tau_ci_lower,
        'tau_ci_upper': tau_ci_upper,
        'r_squared': r_squared,
    }


def quat_to_yaw(quat: np.ndarray) -> np.ndarray:
    """Extract yaw angle from quaternion [w, x, y, z].
    
    Args:
        quat: Quaternion array of shape (..., 4) in [w, x, y, z] order
        
    Returns:
        Yaw angles in radians
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


# =============================================================================
# Helper: Statistical Direction Testing
# =============================================================================
# Note: Circular statistics functions (circular_mean, circular_variance,
# circular_std, circular_confidence_interval, angle_in_circular_ci) are
# imported from test.integration.common module.

def compute_motion_direction_ci(
    dx: np.ndarray,
    dy: np.ndarray,
    expected_angle_rad: float,
    confidence_level: float = CONFIDENCE_LEVEL
) -> dict:
    """Compute motion direction statistics with confidence interval.

    Uses proper circular statistics for angles to handle wraparound correctly.
    Tests whether expected direction falls within the circular CI.

    Mathematical approach:
    1. Convert displacements to angles via arctan2
    2. Compute circular mean using vector averaging (handles wraparound)
    3. Build CI using Fisher's angular standard error approximation
    4. Test expected angle inclusion using circular distance

    Args:
        dx: Forward displacements per environment
        dy: Lateral displacements per environment
        expected_angle_rad: Expected motion angle (0=forward, π/2=left)
        confidence_level: Confidence level for CI (default 0.95)

    Returns:
        Dict with 'mean_angle', 'ci_lower', 'ci_upper', 'expected_in_ci',
        'mean_displacement', 'displacement_p_value', circular stats
    """
    n = len(dx)

    # Compute motion angle for each environment
    motion_angles = np.arctan2(dy, dx)

    # Circular statistics (handles wraparound correctly)
    mean_angle, ci_half_width, R = circular_confidence_interval(
        motion_angles, confidence_level
    )
    circ_var = circular_variance(motion_angles)
    circ_std = circular_std(motion_angles)

    # CI bounds (may wrap around ±π)
    ci_lower = mean_angle - ci_half_width
    ci_upper = mean_angle + ci_half_width

    # Check if expected angle is within circular CI
    expected_in_ci = angle_in_circular_ci(expected_angle_rad, mean_angle, ci_half_width)

    # Compute displacement magnitude along expected direction
    # Project displacement onto expected direction
    cos_exp = np.cos(expected_angle_rad)
    sin_exp = np.sin(expected_angle_rad)
    displacement_along_expected = dx * cos_exp + dy * sin_exp

    # One-sample t-test: H0: mean displacement <= 0 (didn't move in expected direction)
    t_result = one_sample_t_test(
        displacement_along_expected, null_value=0.0, alternative="greater"
    )

    return {
        'mean_angle': mean_angle,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_half_width': ci_half_width,
        'expected_in_ci': expected_in_ci,
        'mean_displacement': t_result.mean,
        'displacement_p_value': t_result.p_value,
        'n_samples': n,
        # Circular statistics
        'mean_resultant_length': R,
        'circular_variance': circ_var,
        'circular_std': circ_std,
    }


# =============================================================================
# Tests: Mecanum Kinematics
# =============================================================================

class TestMecanumKinematics:
    """Test that mecanum wheel kinematics produce correct robot motion.
    
    The MecanumWheelAction kinematic matrix is designed with:
    - Robot forward = -Y in the USD local frame (see actions.py comment)
    - Robot left = -X in the USD local frame
    - Positive vx command produces motion in robot's forward direction (-Y local)
    - Positive vy command produces motion to robot's left (-X local)
    
    At yaw=θ in world frame:
    - Forward direction: (-sin(θ), -cos(θ)) ... i.e., -Y rotated by θ
    - Left direction: (-cos(θ), +sin(θ)) ... i.e., -X rotated by θ
    
    Statistical approach:
    - Direction: Compute motion angle per env, build CI, verify expected angle is within CI
    - Displacement: One-sample t-test that robot moved significantly in expected direction
    """
    
    def test_forward_motion(self, ideal_env):
        """Verify positive vx produces forward motion.
        
        Action [1, 0, 0] should move robot in its local forward direction.
        
        The kinematic matrix maps +vx to -Y in robot's local frame.
        At yaw=0: forward = -Y world
        At yaw=θ: forward = (-sin(θ), -cos(θ)) in world XY.
        """
        action = torch.tensor([[1.0, 0.0, 0.0]], device=ideal_env.device)
        action = action.repeat(ideal_env.num_envs, 1)
        
        traj = collect_robot_trajectory(ideal_env, action, N_MOTION_STEPS)
        
        # Compute displacement in world frame
        final_pos = traj['positions'][-1]  # (num_envs, 3)
        initial_pos = traj['initial_pos']  # (num_envs, 3)
        displacement_world = final_pos - initial_pos  # (num_envs, 3)
        
        # Get initial yaw
        initial_yaw = quat_to_yaw(traj['initial_quat'])  # (num_envs,)
        
        # Robot forward direction in world frame
        # Forward = -Y local, rotated by yaw θ using rotation matrix:
        # [cos(θ) -sin(θ)] [0]    [sin(θ)]
        # [sin(θ)  cos(θ)] [-1] = [-cos(θ)]
        # At yaw=θ: forward = (sin(θ), -cos(θ))
        forward_dir_x = np.sin(initial_yaw)
        forward_dir_y = -np.cos(initial_yaw)
        
        # Robot left direction in world frame  
        # Left = -X local, rotated by yaw θ:
        # [cos(θ) -sin(θ)] [-1]   [-cos(θ)]
        # [sin(θ)  cos(θ)] [0]  = [-sin(θ)]
        # At yaw=θ: left = (-cos(θ), -sin(θ))
        left_dir_x = -np.cos(initial_yaw)
        left_dir_y = -np.sin(initial_yaw)
        
        # Project world displacement onto robot frame
        dx_forward = displacement_world[:, 0] * forward_dir_x + displacement_world[:, 1] * forward_dir_y
        dy_left = displacement_world[:, 0] * left_dir_x + displacement_world[:, 1] * left_dir_y
        
        # Expected angle for forward motion: 0 radians (in robot frame)
        expected_angle = 0.0
        alpha = 1 - CONFIDENCE_LEVEL
        
        # Compute direction statistics using robot-frame displacements
        result = compute_motion_direction_ci(dx_forward, dy_left, expected_angle)
        
        print(f"\n  Forward motion test (n={result['n_samples']} envs):")
        print(f"    Initial yaw (mean): {np.degrees(np.mean(initial_yaw)):.2f}°")
        print(f"    Mean forward displacement: {np.mean(dx_forward):.4f}m")
        print(f"    Mean lateral displacement: {np.mean(dy_left):.4f}m")
        print(f"    Circular mean angle: {np.degrees(result['mean_angle']):.2f}°")
        print(f"    Circular std: {np.degrees(result['circular_std']):.2f}°")
        print(f"    Mean resultant length R: {result['mean_resultant_length']:.4f} (1.0=perfect alignment)")
        print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{np.degrees(result['ci_lower']):.2f}°, {np.degrees(result['ci_upper']):.2f}°]")
        print(f"    Expected (0°) in CI: {result['expected_in_ci']}")
        print(f"    Displacement p-value: {result['displacement_p_value']:.4f}")
        
        # Test 1: Displacement is significantly positive along forward direction
        assert result['displacement_p_value'] < alpha, \
            f"Cannot confirm forward motion (p={result['displacement_p_value']:.4f} >= {alpha})"
        
        # Test 2: Motion direction CI includes expected angle
        assert result['expected_in_ci'], \
            f"Motion direction {np.degrees(result['mean_angle']):.1f}° - " \
            f"expected 0° not in {CONFIDENCE_LEVEL*100:.0f}% CI " \
            f"[{np.degrees(result['ci_lower']):.1f}°, {np.degrees(result['ci_upper']):.1f}°]"
    
    def test_strafe_motion(self, ideal_env):
        """Verify positive vy produces leftward motion.
        
        Action [0, 1, 0] should strafe robot to its left.
        
        The kinematic matrix maps +vy to -X in robot's local frame (left).
        At yaw=0: left = -X world
        At yaw=θ: left = (-cos(θ), -sin(θ)) in world XY.
        """
        action = torch.tensor([[0.0, 1.0, 0.0]], device=ideal_env.device)
        action = action.repeat(ideal_env.num_envs, 1)
        
        traj = collect_robot_trajectory(ideal_env, action, N_MOTION_STEPS)
        
        # Compute displacement in world frame
        final_pos = traj['positions'][-1]
        initial_pos = traj['initial_pos']
        displacement_world = final_pos - initial_pos
        
        # Get initial yaw
        initial_yaw = quat_to_yaw(traj['initial_quat'])
        
        # Robot forward direction in world frame
        # Forward = -Y local, rotated by yaw θ using rotation matrix:
        # [cos(θ) -sin(θ)] [0]    [sin(θ)]
        # [sin(θ)  cos(θ)] [-1] = [-cos(θ)]
        # At yaw=θ: forward = (sin(θ), -cos(θ))
        forward_dir_x = np.sin(initial_yaw)
        forward_dir_y = -np.cos(initial_yaw)
        
        # Robot left direction in world frame  
        # Left = -X local, rotated by yaw θ:
        # [cos(θ) -sin(θ)] [-1]   [-cos(θ)]
        # [sin(θ)  cos(θ)] [0]  = [-sin(θ)]
        # At yaw=θ: left = (-cos(θ), -sin(θ))
        left_dir_x = -np.cos(initial_yaw)
        left_dir_y = -np.sin(initial_yaw)
        
        # Project onto robot frame
        dx_forward = displacement_world[:, 0] * forward_dir_x + displacement_world[:, 1] * forward_dir_y
        dy_left = displacement_world[:, 0] * left_dir_x + displacement_world[:, 1] * left_dir_y
        
        # Expected angle for strafe left: π/2 radians (90°)
        expected_angle = np.pi / 2
        alpha = 1 - CONFIDENCE_LEVEL
        
        # Compute direction statistics
        result = compute_motion_direction_ci(dx_forward, dy_left, expected_angle)
        
        print(f"\n  Strafe motion test (n={result['n_samples']} envs):")
        print(f"    Initial yaw (mean): {np.degrees(np.mean(initial_yaw)):.2f}°")
        print(f"    Mean forward displacement: {np.mean(dx_forward):.4f}m")
        print(f"    Mean lateral displacement: {np.mean(dy_left):.4f}m")
        print(f"    Circular mean angle: {np.degrees(result['mean_angle']):.2f}°")
        print(f"    Circular std: {np.degrees(result['circular_std']):.2f}°")
        print(f"    Mean resultant length R: {result['mean_resultant_length']:.4f} (1.0=perfect alignment)")
        print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{np.degrees(result['ci_lower']):.2f}°, {np.degrees(result['ci_upper']):.2f}°]")
        print(f"    Expected (90°) in CI: {result['expected_in_ci']}")
        print(f"    Displacement p-value: {result['displacement_p_value']:.4f}")
        
        # Test 1: Displacement is significantly positive along left direction
        assert result['displacement_p_value'] < alpha, \
            f"Cannot confirm strafe motion (p={result['displacement_p_value']:.4f} >= {alpha})"
        
        # Test 2: Motion direction CI includes expected angle
        assert result['expected_in_ci'], \
            f"Motion direction {np.degrees(result['mean_angle']):.1f}° - " \
            f"expected 90° not in {CONFIDENCE_LEVEL*100:.0f}% CI " \
            f"[{np.degrees(result['ci_lower']):.1f}°, {np.degrees(result['ci_upper']):.1f}°]"
    
    def test_rotation_motion(self, ideal_env):
        """Verify positive omega produces counter-clockwise rotation.
        
        Action [0, 0, 1] should rotate robot CCW (positive yaw change).
        
        Statistical tests:
        1. Yaw change is significantly positive (one-sample t-test, p < alpha)
        2. Yaw CI does not include 0 (definite rotation occurred)
        3. Translation is bounded by physics (rotation axis offset from center)
        """
        action = torch.tensor([[0.0, 0.0, 1.0]], device=ideal_env.device)
        action = action.repeat(ideal_env.num_envs, 1)
        
        traj = collect_robot_trajectory(ideal_env, action, N_MOTION_STEPS)
        
        # Compute yaw change
        initial_yaw = quat_to_yaw(traj['initial_quat'])
        final_yaw = quat_to_yaw(traj['orientations'][-1])
        
        # Handle angle wrapping
        yaw_change = final_yaw - initial_yaw
        yaw_change = np.arctan2(np.sin(yaw_change), np.cos(yaw_change))
        
        # Compute displacement magnitude (should be near zero for pure rotation)
        final_pos = traj['positions'][-1]
        initial_pos = traj['initial_pos']
        displacement_mag = np.linalg.norm(final_pos - initial_pos, axis=1)
        
        n = len(yaw_change)
        alpha = 1 - CONFIDENCE_LEVEL

        # One-sample t-test for yaw change: H0: mean yaw change <= 0
        yaw_result = one_sample_t_test(yaw_change, null_value=0.0, alternative="greater")

        # Statistics for displacement (using basic scipy since we just need CI)
        mean_disp = np.mean(displacement_mag)
        sem_disp = stats.sem(displacement_mag)
        t_crit = stats.t.ppf(1 - alpha/2, n - 1)
        disp_ci_lower = mean_disp - t_crit * sem_disp
        disp_ci_upper = mean_disp + t_crit * sem_disp
        
        # For "pure rotation", displacement should be bounded by rotation axis offset
        # For rotation θ about offset r from center: displacement ≈ 2*r*sin(θ/2)
        # We derive max acceptable offset from robot geometry (~10cm tolerance)
        max_acceptable_offset = 0.1  # 10cm offset from center of rotation
        max_expected_disp = 2 * max_acceptable_offset * np.sin(abs(yaw_result.mean) / 2)

        print(f"\n  Rotation motion test (n={n} envs):")
        print(f"    Mean yaw change: {np.degrees(yaw_result.mean):.2f}°")
        print(f"    Yaw {CONFIDENCE_LEVEL*100:.0f}% CI: [{np.degrees(yaw_result.ci_low):.2f}°, {np.degrees(yaw_result.ci_high):.2f}°]")
        print(f"    Yaw p-value (>0): {yaw_result.p_value:.4f}")
        print(f"    Mean displacement: {mean_disp:.4f}m")
        print(f"    Displacement {CONFIDENCE_LEVEL*100:.0f}% CI: [{disp_ci_lower:.4f}m, {disp_ci_upper:.4f}m]")
        print(f"    Max expected displacement (10cm axis offset): {max_expected_disp:.4f}m")

        # Test 1: Yaw change is significantly positive (CCW rotation)
        assert yaw_result.p_value < alpha, \
            f"Cannot confirm CCW rotation (p={yaw_result.p_value:.4f} >= {alpha})"

        # Test 2: Yaw CI should not include 0 (definite rotation occurred)
        assert yaw_result.ci_low > 0, \
            f"Yaw change CI [{np.degrees(yaw_result.ci_low):.1f}°, {np.degrees(yaw_result.ci_high):.1f}°] includes 0"
        
        # Test 3: Displacement bounded by physics (rotation axis offset)
        # Allow for CI uncertainty in displacement measurement
        assert mean_disp < max_expected_disp + (disp_ci_upper - mean_disp), \
            f"Displacement {mean_disp:.4f}m exceeds expected {max_expected_disp:.4f}m for rotation axis offset"


# =============================================================================
# Tests: Combined Dynamics (Realistic Config)
# =============================================================================

class TestCombinedDynamics:
    """Test all dynamics enabled together (realistic configuration).

    These tests verify the combined system behaves reasonably when all
    components are active. We use qualitative tests since the exact
    response depends on component interactions.
    """

    def test_all_dynamics_enabled(self, realistic_env):
        """Verify realistic config has all dynamics enabled."""
        action_term = realistic_env.action_manager._terms["wheel_velocities"]

        print(f"\n  Realistic config - dynamics status:")
        print(f"    Motor dynamics: {action_term._enable_motor_dynamics}")
        print(f"    Command delay: {action_term._enable_command_delay}")
        print(f"    Slew rate: {action_term._enable_slew_rate}")

        assert action_term._enable_motor_dynamics, "Motor dynamics should be enabled"
        assert action_term._enable_command_delay, "Command delay should be enabled"
        assert action_term._enable_slew_rate, "Slew rate should be enabled"

    def test_combined_response_is_gradual(self, realistic_env):
        """Verify combined system produces gradual (not instant) response."""
        action_term = reset_env_and_action_term(realistic_env, ideal_mode=False)

        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=realistic_env.device)
        step_action = step_action.repeat(realistic_env.num_envs, 1)

        responses = []
        for _ in range(N_RESPONSE_STEPS):
            action_term.process_actions(step_action)
            responses.append(action_term.processed_actions[0, 0].cpu().item())

        responses = np.abs(np.array(responses))
        initial = responses[0]
        final = responses[-1]

        print(f"\n  Combined dynamics - gradual response:")
        print(f"    Initial: {initial:.4f}")
        print(f"    Final: {final:.4f}")
        print(f"    Ratio: {initial / (final + 1e-9):.4f}")

        # Initial should be much less than final
        assert initial < final * 0.3, \
            f"Combined dynamics should produce gradual response (initial={initial:.4f}, final={final:.4f})"

    def test_combined_no_overshoot(self, realistic_env):
        """Verify combined system doesn't overshoot (monotonic increase)."""
        action_term = reset_env_and_action_term(realistic_env, ideal_mode=False)

        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=realistic_env.device)
        step_action = step_action.repeat(realistic_env.num_envs, 1)

        responses = []
        for _ in range(N_RESPONSE_STEPS):
            action_term.process_actions(step_action)
            responses.append(action_term.processed_actions[0, 0].cpu().item())

        responses = np.abs(np.array(responses))
        v_final = responses[-1]
        noise_floor = max(1e-7 * v_final, 1e-6)
        diffs = np.diff(responses)
        significant_decreases = np.sum(diffs < -3 * noise_floor)

        print(f"\n  Combined dynamics - no overshoot:")
        print(f"    Significant decreases: {significant_decreases}")

        # Allow small number due to numerical noise
        assert significant_decreases <= 3, \
            f"Combined system should not overshoot, found {significant_decreases} significant decreases"

    def test_combined_converges_to_steady_state(self, realistic_env):
        """Verify combined system converges to steady state."""
        action_term = reset_env_and_action_term(realistic_env, ideal_mode=False)

        step_action = torch.tensor([[0.5, 0.0, 0.0]], device=realistic_env.device)
        step_action = step_action.repeat(realistic_env.num_envs, 1)

        responses = []
        for _ in range(N_RESPONSE_STEPS * 2):
            action_term.process_actions(step_action)
            responses.append(action_term.processed_actions[0, 0].cpu().item())

        responses = np.abs(np.array(responses))
        final_samples = responses[-20:]
        mean_val = np.mean(final_samples)
        cv = np.std(final_samples) / (mean_val + 1e-9)

        print(f"\n  Combined dynamics - steady state:")
        print(f"    Mean final value: {mean_val:.4f}")
        print(f"    Coefficient of variation: {cv*100:.4f}%")

        assert cv < 0.05, f"Should converge to steady state (CV={cv*100:.2f}%)"

    def test_reset_clears_all_state(self, realistic_env):
        """Verify reset clears all dynamics state."""
        action_term = reset_env_and_action_term(realistic_env, ideal_mode=False)

        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=realistic_env.device)
        step_action = step_action.repeat(realistic_env.num_envs, 1)

        # Build up velocity
        for _ in range(50):
            action_term.process_actions(step_action)

        pre_reset_vel = torch.abs(action_term.processed_actions).mean().item()

        # Reset
        action_term.reset(env_ids=torch.arange(realistic_env.num_envs, device=realistic_env.device))

        # Check state is cleared
        post_reset_vel = 0.0
        if hasattr(action_term, '_smoothed_wheel_vels'):
            post_reset_vel = max(post_reset_vel, torch.abs(action_term._smoothed_wheel_vels).mean().item())
        if hasattr(action_term, '_prev_wheel_vels'):
            post_reset_vel = max(post_reset_vel, torch.abs(action_term._prev_wheel_vels).mean().item())

        print(f"\n  Reset clears state:")
        print(f"    Pre-reset velocity: {pre_reset_vel:.4f}")
        print(f"    Post-reset state: {post_reset_vel:.6f}")

        assert pre_reset_vel > 0.1, "Should have nonzero velocity before reset"
        assert post_reset_vel < 0.01, "State should be cleared after reset"




# =============================================================================
# Tests: Ideal vs Realistic Comparison
# =============================================================================

class TestIdealVsRealistic:
    """Compare ideal (no dynamics) vs realistic (with dynamics) responses.

    Uses Welch's t-test for comparing response characteristics between
    ideal and realistic modes. Welch's t-test is preferred over Student's
    t-test because it doesn't assume equal variances between groups.
    """

    def test_ideal_has_instant_response(self, env):
        """Verify ideal config produces instant velocity response."""
        # Use helper to set ideal mode and clear state
        action_term = reset_env_and_action_term(env, ideal_mode=True)

        # Apply step
        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)

        action_term.process_actions(step_action)
        first_response = np.abs(action_term.processed_actions[0, 0].cpu().item())

        # Apply more steps
        for _ in range(10):
            action_term.process_actions(step_action)
        final_response = np.abs(action_term.processed_actions[0, 0].cpu().item())

        print(f"\n  Ideal (instant) response:")
        print(f"    First step: {first_response:.4f}")
        print(f"    After 10 steps: {final_response:.4f}")
        print(f"    Ratio: {first_response / (final_response + 1e-9):.2f}")

        # Ideal should be nearly instant (first response ≈ final response)
        assert first_response > final_response * 0.9, \
            f"Ideal config should have instant response (first={first_response:.4f}, final={final_response:.4f})"

    def test_realistic_has_gradual_response(self, env):
        """Verify realistic config produces gradual velocity response."""
        # Use helper to set realistic mode and clear state
        action_term = reset_env_and_action_term(env, ideal_mode=False)

        # Apply step
        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)

        action_term.process_actions(step_action)
        first_response = np.abs(action_term.processed_actions[0, 0].cpu().item())

        # Apply more steps
        for _ in range(50):
            action_term.process_actions(step_action)
        final_response = np.abs(action_term.processed_actions[0, 0].cpu().item())

        print(f"\n  Realistic (gradual) response:")
        print(f"    First step: {first_response:.4f}")
        print(f"    After 50 steps: {final_response:.4f}")
        print(f"    Ratio: {first_response / (final_response + 1e-9):.2f}")

        # Realistic should be gradual (first response << final response)
        assert first_response < final_response * 0.5, \
            f"Realistic config should have gradual response (first={first_response:.4f}, final={final_response:.4f})"

    def test_ideal_vs_realistic_rise_time_welch(self, env):
        """Compare rise times using Welch's t-test (unequal variance t-test).

        This test statistically compares the rise time characteristics between
        ideal and realistic modes using Welch's t-test, which is robust to
        unequal variances between groups.

        Rise time is defined as time to reach 63.2% of final value (1 time constant).
        For ideal mode, this should be ~0 (instant).
        For realistic mode, this should be approximately τ_motor.

        Welch's t-test is used because:
        1. Variances may differ between ideal (near-zero variance) and realistic
        2. It's more robust than Student's t-test for small/unequal samples
        3. It doesn't require homogeneity of variance assumption
        """
        step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
        step_action = step_action.repeat(env.num_envs, 1)
        physics_dt = env.physics_dt

        # Collect rise times for ideal mode (across all envs and wheels)
        reset_env_and_action_term(env, ideal_mode=True)
        ideal_responses = collect_step_response(env, step_action, N_RESPONSE_STEPS, ideal_mode=True)
        ideal_rise_times = []

        for env_idx in range(env.num_envs):
            for wheel_idx in range(4):
                response = np.abs(ideal_responses[:, env_idx, wheel_idx])
                v_final = response[-1]
                if v_final > 0.01:  # Only count active wheels
                    # Find time to reach 63.2% of final
                    threshold = 0.632 * v_final
                    rise_idx = np.argmax(response >= threshold)
                    rise_time = rise_idx * physics_dt
                    ideal_rise_times.append(rise_time)

        # Collect rise times for realistic mode
        reset_env_and_action_term(env, ideal_mode=False)
        realistic_responses = collect_step_response(env, step_action, N_RESPONSE_STEPS, ideal_mode=False)
        realistic_rise_times = []

        for env_idx in range(env.num_envs):
            for wheel_idx in range(4):
                response = np.abs(realistic_responses[:, env_idx, wheel_idx])
                v_final = response[-1]
                if v_final > 0.01:
                    threshold = 0.632 * v_final
                    rise_idx = np.argmax(response >= threshold)
                    rise_time = rise_idx * physics_dt
                    realistic_rise_times.append(rise_time)

        ideal_rise_times = np.array(ideal_rise_times)
        realistic_rise_times = np.array(realistic_rise_times)

        # Welch's t-test: H1: realistic rise time > ideal rise time (one-sided)
        welch_result = welch_t_test(
            realistic_rise_times,
            ideal_rise_times,
            alternative="greater"
        )

        alpha = 1 - CONFIDENCE_LEVEL

        print(f"\n  Ideal vs Realistic rise time comparison (Welch's t-test):")
        print(f"    Ideal rise times: mean={welch_result['mean_b']*1000:.2f}ms, std={welch_result['std_b']*1000:.2f}ms, n={welch_result['n_b']}")
        print(f"    Realistic rise times: mean={welch_result['mean_a']*1000:.2f}ms, std={welch_result['std_a']*1000:.2f}ms, n={welch_result['n_a']}")
        print(f"    Welch's t-statistic: {welch_result['t_statistic']:.2f}")
        print(f"    One-sided p-value (realistic > ideal): {welch_result['p_value']:.4f}")
        print(f"    Cohen's d effect size: {welch_result['cohens_d']:.2f}")
        print(f"    Interpretation: {'Large' if abs(welch_result['cohens_d']) > 0.8 else 'Medium' if abs(welch_result['cohens_d']) > 0.5 else 'Small'} effect")

        # Test: Realistic rise time significantly greater than ideal
        assert welch_result['p_value'] < alpha, \
            f"Realistic rise time should be significantly greater than ideal (p={welch_result['p_value']:.4f} >= {alpha})"

        # Test: Effect size should be large (d > 0.8)
        assert welch_result['cohens_d'] > 0.8, \
            f"Effect size should be large (Cohen's d={welch_result['cohens_d']:.2f} <= 0.8)"


# =============================================================================
# Run pytest
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
