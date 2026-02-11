# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for mecanum wheel kinematics.

These tests verify that the MecanumWheelAction kinematic matrix produces
correct robot motion directions. Motor dynamics are disabled (ideal mode)
to isolate the kinematic behavior.

Statistical approach:
- Direction: Compute motion angle per env, build circular CI, verify expected
- Displacement: One-sample t-test that robot moved significantly in expected direction

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/actions/test_kinematics.py -v
"""

import pytest
import torch
import numpy as np
from flaky import flaky

from test.actions.conftest import (
    collect_robot_trajectory,
    quat_to_yaw,
    project_trajectory_to_robot_frame,
    compute_motion_direction_ci,
    N_MOTION_STEPS,
    MAX_DIRECTION_ERROR_DEG,
)
from test.common import (
    CONFIDENCE_LEVEL,
    one_sample_t_test,
)

# Strafe needs more sim steps because roller-to-ground contact
# generates higher variance than forward driving.
N_STRAFE_STEPS = N_MOTION_STEPS * 2


def test_forward_motion(ideal_env):
    """Verify positive vx produces forward motion.

    Action [1, 0, 0] should move robot in its local forward direction.

    Uses per-step heading projection: each simulation step's world-frame
    displacement is projected onto the robot's heading at that step, then
    accumulated. This correctly handles any heading drift during motion.
    """
    action = torch.tensor([[1.0, 0.0, 0.0]], device=ideal_env.device)
    action = action.repeat(ideal_env.num_envs, 1)

    traj = collect_robot_trajectory(ideal_env, action, N_MOTION_STEPS)

    # Project trajectory onto robot frame using per-step heading
    dx_forward, dy_left = project_trajectory_to_robot_frame(traj)

    initial_yaw = quat_to_yaw(traj['initial_quat'])

    # Expected angle for forward motion: 0 radians (in robot frame)
    expected_angle = 0.0
    alpha = 1 - CONFIDENCE_LEVEL

    result = compute_motion_direction_ci(dx_forward, dy_left, expected_angle)

    direction_error_deg = np.degrees(abs(np.arctan2(
        np.sin(result['mean_angle'] - expected_angle),
        np.cos(result['mean_angle'] - expected_angle))))

    print(f"\n  Forward motion test (n={result['n_samples']} envs):")
    print(f"    Initial yaw (mean): {np.degrees(np.mean(initial_yaw)):.2f}°")
    print(f"    Mean forward displacement: {np.mean(dx_forward):.4f}m")
    print(f"    Mean lateral displacement: {np.mean(dy_left):.4f}m")
    print(f"    Circular mean angle: {np.degrees(result['mean_angle']):.2f}°")
    print(f"    Circular std: {np.degrees(result['circular_std']):.2f}°")
    print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{np.degrees(result['ci_lower']):.2f}°, {np.degrees(result['ci_upper']):.2f}°]")
    print(f"    Direction error: {direction_error_deg:.2f}°")
    print(f"    Displacement p-value: {result['displacement_p_value']:.4f}")

    # Test 1: Displacement significantly positive along forward direction
    assert result['displacement_p_value'] < alpha, \
        f"Cannot confirm forward motion (p={result['displacement_p_value']:.4f} >= {alpha})"

    # Test 2: Mean direction close to expected
    assert direction_error_deg < MAX_DIRECTION_ERROR_DEG, \
        f"Motion direction {np.degrees(result['mean_angle']):.1f}° deviates " \
        f"{direction_error_deg:.1f}° from expected 0° (max: {MAX_DIRECTION_ERROR_DEG}°)"


@flaky(max_runs=2, min_passes=1)
def test_strafe_motion(ideal_env):
    """Verify positive vy produces leftward motion.

    Action [0, 0.5, 0] should strafe robot to its left.

    Uses per-step heading projection to handle heading drift that mecanum
    wheel physics induces during sustained strafing.

    Marked flaky because mecanum strafing has higher variance from
    roller-to-ground contact mechanics.
    """
    action = torch.tensor([[0.0, 1.0, 0.0]], device=ideal_env.device)
    action = action.repeat(ideal_env.num_envs, 1)

    traj = collect_robot_trajectory(ideal_env, action, N_STRAFE_STEPS)

    dx_forward, dy_left = project_trajectory_to_robot_frame(traj)
    initial_yaw = quat_to_yaw(traj['initial_quat'])

    # Expected angle for strafe left: π/2 radians (90°)
    expected_angle = np.pi / 2
    alpha = 1 - CONFIDENCE_LEVEL

    result = compute_motion_direction_ci(dx_forward, dy_left, expected_angle)

    direction_error_deg = np.degrees(abs(np.arctan2(
        np.sin(result['mean_angle'] - expected_angle),
        np.cos(result['mean_angle'] - expected_angle))))

    print(f"\n  Strafe motion test (n={result['n_samples']} envs):")
    print(f"    Initial yaw (mean): {np.degrees(np.mean(initial_yaw)):.2f}°")
    print(f"    Mean forward displacement: {np.mean(dx_forward):.4f}m")
    print(f"    Mean lateral displacement: {np.mean(dy_left):.4f}m")
    print(f"    Circular mean angle: {np.degrees(result['mean_angle']):.2f}°")
    print(f"    Circular std: {np.degrees(result['circular_std']):.2f}°")
    print(f"    {CONFIDENCE_LEVEL*100:.0f}% CI: [{np.degrees(result['ci_lower']):.2f}°, {np.degrees(result['ci_upper']):.2f}°]")
    print(f"    Direction error: {direction_error_deg:.2f}°")
    print(f"    Displacement p-value: {result['displacement_p_value']:.4f}")

    # Test 1: Displacement significantly positive along left direction
    assert result['displacement_p_value'] < alpha, \
        f"Cannot confirm strafe motion (p={result['displacement_p_value']:.4f} >= {alpha})"

    # Test 2: Mean direction close to expected
    assert direction_error_deg < MAX_DIRECTION_ERROR_DEG, \
        f"Motion direction {np.degrees(result['mean_angle']):.1f}° deviates " \
        f"{direction_error_deg:.1f}° from expected 90° (max: {MAX_DIRECTION_ERROR_DEG}°)"


def test_rotation_motion(ideal_env):
    """Verify positive omega produces counter-clockwise rotation.

    Action [0, 0, 1] should rotate robot CCW (positive yaw change).

    Statistical tests:
    1. Yaw change is significantly positive (one-sample t-test, p < alpha)
    2. Yaw CI does not include 0 (definite rotation occurred)
    3. Translation is bounded by physics (rotation axis offset from center)
    """
    from scipy import stats as sp_stats

    action = torch.tensor([[0.0, 0.0, 1.0]], device=ideal_env.device)
    action = action.repeat(ideal_env.num_envs, 1)

    traj = collect_robot_trajectory(ideal_env, action, N_MOTION_STEPS)

    # Compute yaw change
    initial_yaw = quat_to_yaw(traj['initial_quat'])
    final_yaw = quat_to_yaw(traj['orientations'][-1])

    # Handle angle wrapping
    yaw_change = final_yaw - initial_yaw
    yaw_change = np.arctan2(np.sin(yaw_change), np.cos(yaw_change))

    # Displacement magnitude (should be near zero for pure rotation)
    final_pos = traj['positions'][-1]
    initial_pos = traj['initial_pos']
    displacement_mag = np.linalg.norm(final_pos - initial_pos, axis=1)

    n = len(yaw_change)
    alpha = 1 - CONFIDENCE_LEVEL

    # One-sample t-test for yaw change: H0: mean yaw change <= 0
    yaw_result = one_sample_t_test(yaw_change, null_value=0.0, alternative="greater")

    # Displacement statistics
    mean_disp = np.mean(displacement_mag)
    sem_disp = sp_stats.sem(displacement_mag)
    t_crit = sp_stats.t.ppf(1 - alpha / 2, n - 1)
    disp_ci_upper = mean_disp + t_crit * sem_disp

    # Max acceptable displacement from rotation axis offset (~10cm)
    max_acceptable_offset = 0.1
    max_expected_disp = 2 * max_acceptable_offset * np.sin(abs(yaw_result.mean) / 2)

    print(f"\n  Rotation motion test (n={n} envs):")
    print(f"    Mean yaw change: {np.degrees(yaw_result.mean):.2f}°")
    print(f"    Yaw {CONFIDENCE_LEVEL*100:.0f}% CI: [{np.degrees(yaw_result.ci_low):.2f}°, {np.degrees(yaw_result.ci_high):.2f}°]")
    print(f"    Yaw p-value (>0): {yaw_result.p_value:.4f}")
    print(f"    Mean displacement: {mean_disp:.4f}m")
    print(f"    Max expected displacement: {max_expected_disp:.4f}m")

    # Test 1: Yaw change is significantly positive (CCW rotation)
    assert yaw_result.p_value < alpha, \
        f"Cannot confirm CCW rotation (p={yaw_result.p_value:.4f} >= {alpha})"

    # Test 2: Yaw CI should not include 0
    assert yaw_result.ci_low > 0, \
        f"Yaw change CI [{np.degrees(yaw_result.ci_low):.1f}°, {np.degrees(yaw_result.ci_high):.1f}°] includes 0"

    # Test 3: Displacement bounded by rotation axis offset
    assert mean_disp < max_expected_disp + (disp_ci_upper - mean_disp), \
        f"Displacement {mean_disp:.4f}m exceeds expected {max_expected_disp:.4f}m"
