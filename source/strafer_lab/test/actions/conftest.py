# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures and utilities for action dynamics tests.

This conftest.py provides:
- Environment fixtures for action term testing (unit + integration)
- Helper functions for configuring action dynamics components
- Response analysis utilities (exponential fit, linear fit)
- Trajectory collection and kinematics helpers for integration tests

NOTE: AppLauncher is initialized in test/conftest.py (root).
This file must NOT launch Isaac Sim - only provide fixtures and helpers.
"""

import torch
import numpy as np
from scipy import stats
import pytest

from isaaclab.envs import ManagerBasedRLEnv

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Realistic,
)

from test.common import (
    CONFIDENCE_LEVEL,
    NUM_ENVS as COMMON_NUM_ENVS,
    N_SETTLE_STEPS,
    DEVICE,
    circular_mean,
    circular_variance,
    circular_std,
    circular_confidence_interval,
    angle_in_circular_ci,
    one_sample_t_test,
    welch_t_test,
)
from test.common.robot import reset_robot_pose


# =============================================================================
# Test Configuration Constants
# =============================================================================

# Unit tests use minimal envs for speed
NUM_ENVS_UNIT = 4

# Integration tests use more envs for statistical power
NUM_ENVS_INTEGRATION = COMMON_NUM_ENVS  # 64, from test.common

N_RESPONSE_STEPS = 200       # Steps to measure step response
N_MOTION_STEPS = 10          # Steps for kinematics tests

# Thresholds for deterministic analytical tests (not statistical tolerances)
R_SQUARED_EXPONENTIAL_FIT = 0.99    # First-order systems fit exponential near-perfectly
R_SQUARED_LINEAR_FIT = 0.999        # Constant acceleration gives perfect linear ramp
CORRELATION_SIGNAL_PRESERVE = 0.999 # Delay buffer preserves signal exactly
FLOAT_PRECISION_FACTOR = 1.001      # Allow 0.1% for floating-point accumulation

# Max acceptable angular error for kinematics direction tests.
# Mecanum wheel physics creates coupling between strafe and forward motion
# (roller friction asymmetry), so the measured direction will never be
# exactly 0° or 90°. This tolerance catches kinematic matrix errors while
# allowing for normal physics coupling effects.
MAX_DIRECTION_ERROR_DEG = 15.0


# =============================================================================
# Module-scoped Environment
# =============================================================================

_module_env = None


def _get_or_create_env():
    """Get or create environment for testing action term.

    Uses COMMON_NUM_ENVS (64) so both unit and integration tests can share
    the same environment. Unit tests that only need 4 envs simply index
    into the first few.
    """
    global _module_env

    if _module_env is not None:
        return _module_env

    cfg = StraferNavEnvCfg_NoCam()
    cfg.scene.num_envs = NUM_ENVS_INTEGRATION
    cfg.actions = ActionsCfg_Realistic()

    _module_env = ManagerBasedRLEnv(cfg)
    _module_env.reset()

    return _module_env


@pytest.fixture(scope="module")
def action_env():
    """Provide environment for action term unit testing."""
    env = _get_or_create_env()
    yield env


@pytest.fixture(scope="module")
def ideal_env():
    """Provide env with motor dynamics disabled for kinematics tests.

    Temporarily disables all dynamics (motor, delay, slew) so that
    commanded velocities are applied instantly. This isolates the
    kinematic matrix for directional testing.
    """
    env = _get_or_create_env()
    action_term = env.action_manager._terms["wheel_velocities"]

    # Save original settings
    orig_motor = action_term._enable_motor_dynamics
    orig_delay = action_term._enable_command_delay
    orig_slew = action_term._enable_slew_rate

    # Disable all dynamics for ideal behavior
    action_term._enable_motor_dynamics = False
    action_term._enable_command_delay = False
    action_term._enable_slew_rate = False

    # Clear internal state buffers
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
def realistic_env():
    """Provide env with all motor dynamics enabled (realistic behavior)."""
    env = _get_or_create_env()
    action_term = env.action_manager._terms["wheel_velocities"]

    # Ensure dynamics are enabled
    action_term._enable_motor_dynamics = True
    action_term._enable_command_delay = True
    action_term._enable_slew_rate = True

    yield env


def pytest_sessionfinish(session, exitstatus):
    """Clean up the environment after all tests complete."""
    global _module_env
    if _module_env is not None:
        _module_env.close()
        _module_env = None
    # Note: simulation_app.close() is handled by root test/conftest.py


# =============================================================================
# Helper Functions - Action Term Configuration
# =============================================================================


def configure_action_term_dynamics(
    env,
    enable_motor: bool = False,
    enable_delay: bool = False,
    enable_slew: bool = False
):
    """Configure action term with specific dynamics enabled/disabled.

    This allows testing each dynamic component in isolation.

    Args:
        env: The environment
        enable_motor: Enable first-order motor dynamics filter
        enable_delay: Enable command delay buffer
        enable_slew: Enable slew rate (acceleration) limiting

    Returns:
        The configured action term
    """
    env.reset()
    action_term = env.action_manager._terms["wheel_velocities"]

    # Configure specific dynamics
    action_term._enable_motor_dynamics = enable_motor
    action_term._enable_command_delay = enable_delay
    action_term._enable_slew_rate = enable_slew

    # Reset and clear all state buffers
    action_term.reset(env_ids=torch.arange(env.num_envs, device=env.device))

    if hasattr(action_term, '_smoothed_wheel_vels'):
        action_term._smoothed_wheel_vels.zero_()
    if hasattr(action_term, '_prev_wheel_vels'):
        action_term._prev_wheel_vels.zero_()
    if hasattr(action_term, '_action_delay_buffer'):
        action_term._action_delay_buffer.reset(None)

    return action_term


def reset_env_and_action_term(env, ideal_mode: bool = False):
    """Reset environment and action term state.

    Args:
        env: The environment
        ideal_mode: If True, disable motor dynamics/delay/slew for ideal behavior.
                    If False, enable all dynamics for realistic behavior.

    Returns:
        The configured action term
    """
    env.reset()

    # Position robots at grid origins (avoids overlap/interference between envs)
    reset_robot_pose(env)

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


# =============================================================================
# Helper Functions - Response Analysis
# =============================================================================


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
    tau_std = std_err / (slope ** 2)
    t_crit = stats.t.ppf(0.975, len(t_valid) - 2)
    tau_ci_lower = tau - t_crit * tau_std
    tau_ci_upper = tau + t_crit * tau_std

    return {
        'tau': tau,
        'tau_ci_lower': tau_ci_lower,
        'tau_ci_upper': tau_ci_upper,
        'r_squared': r_squared,
    }


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


# =============================================================================
# Helper Functions - Trajectory Collection (Integration Tests)
# =============================================================================


def collect_robot_trajectory(env, action: torch.Tensor, n_steps: int, ideal_mode: bool = True) -> dict:
    """Collect robot position/orientation trajectory.

    Args:
        env: The environment
        action: Action tensor of shape (num_envs, 3)
        n_steps: Number of steps to simulate
        ideal_mode: If True, disable motor dynamics for kinematics tests

    Returns:
        Dict with 'positions', 'orientations', 'initial_pos', 'initial_quat'
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


# =============================================================================
# Helper Functions - Kinematics Analysis
# =============================================================================


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


def project_trajectory_to_robot_frame(traj: dict) -> tuple:
    """Project world-frame trajectory into kinematic body frame using per-step heading.

    Uses the ROS/kinematic body frame convention (+X forward, +Y left),
    which matches the MecanumWheelAction kinematic matrix convention.
    Note: This differs from the USD mesh forward direction (-Y).

    Instead of projecting total displacement onto the initial heading (which
    becomes inaccurate when the robot rotates during motion), this projects
    each simulation step's displacement onto the robot's heading at that step
    and accumulates the result.

    Args:
        traj: Trajectory dict from collect_robot_trajectory.

    Returns:
        Tuple of (dx_forward, dy_left) arrays, each of shape (num_envs,).
        dx_forward: Accumulated displacement along body +X (kinematic forward).
        dy_left: Accumulated displacement along body +Y (kinematic left).
    """
    positions = traj['positions']        # (n_steps+1, num_envs, 3)
    orientations = traj['orientations']  # (n_steps+1, num_envs, 4)

    # Per-step world-frame displacements
    deltas = np.diff(positions, axis=0)  # (n_steps, num_envs, 3)

    # Get robot heading at the start of each step
    yaws = quat_to_yaw(orientations[:-1])  # (n_steps, num_envs)

    # Robot frame directions in world coordinates
    # Kinematic (ROS) body frame: +X = forward, +Y = left
    # At yaw θ: body +X in world = (cos θ, sin θ)
    #           body +Y in world = (-sin θ, cos θ)
    forward_x = np.cos(yaws)
    forward_y = np.sin(yaws)
    left_x = -np.sin(yaws)
    left_y = np.cos(yaws)

    # Project each step's displacement onto robot frame and sum
    dx_forward = np.sum(
        deltas[:, :, 0] * forward_x + deltas[:, :, 1] * forward_y, axis=0
    )
    dy_left = np.sum(
        deltas[:, :, 0] * left_x + deltas[:, :, 1] * left_y, axis=0
    )

    return dx_forward, dy_left


def compute_motion_direction_ci(
    dx: np.ndarray,
    dy: np.ndarray,
    expected_angle_rad: float,
    confidence_level: float = CONFIDENCE_LEVEL
) -> dict:
    """Compute motion direction statistics with confidence interval.

    Uses proper circular statistics for angles to handle wraparound correctly.

    Args:
        dx: Forward displacements per environment
        dy: Lateral displacements per environment
        expected_angle_rad: Expected motion angle (0=forward, π/2=left)
        confidence_level: Confidence level for CI

    Returns:
        Dict with direction statistics including CI and displacement p-value.
    """
    n = len(dx)

    # Compute motion angle for each environment
    motion_angles = np.arctan2(dy, dx)

    # Circular statistics (handles wraparound correctly)
    mean_angle, ci_half_width, R = circular_confidence_interval(
        motion_angles, confidence_level
    )
    circ_var = circular_variance(motion_angles)
    circ_std_val = circular_std(motion_angles)

    # CI bounds
    ci_lower = mean_angle - ci_half_width
    ci_upper = mean_angle + ci_half_width

    # Check if expected angle is within circular CI
    expected_in_ci = angle_in_circular_ci(expected_angle_rad, mean_angle, ci_half_width)

    # Project displacement onto expected direction
    cos_exp = np.cos(expected_angle_rad)
    sin_exp = np.sin(expected_angle_rad)
    displacement_along_expected = dx * cos_exp + dy * sin_exp

    # One-sample t-test: H0: mean displacement <= 0
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
        'mean_resultant_length': R,
        'circular_variance': circ_var,
        'circular_std': circ_std_val,
    }
