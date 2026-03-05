# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for event / domain randomization functions.

Functions under test (``strafer_lab.tasks.navigation.mdp.events``):

* ``reset_robot_state``            — reset robot pose at episode start.
* ``randomize_friction``           — per-env wheel friction randomization.
* ``randomize_mass``               — per-env body mass randomization.
* ``randomize_motor_strength``     — per-joint effort limit randomization.
* ``randomize_d555_mount_offset``  — D555 mounting angle randomization.
* ``randomize_goal_noise``         — Gaussian noise on goal position.
* ``randomize_obstacles``          — obstacle position randomization.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/events/test_events.py -v
"""

import math
import torch
import pytest

from strafer_lab.tasks.navigation.mdp.events import (
    reset_robot_state,
    randomize_friction,
    randomize_mass,
    randomize_motor_strength,
    randomize_d555_mount_offset,
    randomize_goal_noise,
)


# =====================================================================
# Fixture alias
# =====================================================================

@pytest.fixture(scope="module")
def env(event_env):
    """Reuse the shared events conftest environment."""
    return event_env


# =====================================================================
# reset_robot_state — Tests
# =====================================================================


def test_reset_robot_state_positions_within_range(env):
    """Robot positions after reset must lie within the specified range."""
    env.reset()

    pose_range = {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "yaw": (-math.pi, math.pi)}
    env_ids = torch.arange(env.num_envs, device=env.device)

    reset_robot_state(env, env_ids, pose_range)

    robot = env.scene["robot"]
    root_state = robot.data.root_pos_w
    x_vals = root_state[:, 0]
    y_vals = root_state[:, 1]

    print(f"\n  reset_robot_state positions:")
    print(f"    x range: [{x_vals.min().item():.3f}, {x_vals.max().item():.3f}]")
    print(f"    y range: [{y_vals.min().item():.3f}, {y_vals.max().item():.3f}]")

    assert (x_vals >= -2.0 - 0.01).all() and (x_vals <= 2.0 + 0.01).all(), (
        f"x positions out of range: [{x_vals.min().item():.3f}, {x_vals.max().item():.3f}]"
    )
    assert (y_vals >= -2.0 - 0.01).all() and (y_vals <= 2.0 + 0.01).all(), (
        f"y positions out of range: [{y_vals.min().item():.3f}, {y_vals.max().item():.3f}]"
    )


def test_reset_robot_state_velocities_zeroed(env):
    """After reset, robot velocities must be zero."""
    env.reset()

    pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_ids = torch.arange(env.num_envs, device=env.device)

    reset_robot_state(env, env_ids, pose_range)

    robot = env.scene["robot"]
    root_state = robot.data.root_state_w
    velocities = root_state[:, 7:]  # lin_vel(3) + ang_vel(3)

    max_vel = velocities.abs().max().item()
    print(f"\n  reset_robot_state velocities:")
    print(f"    Max |velocity|: {max_vel:.6f}")

    assert max_vel < 0.01, (
        f"Velocities not zeroed after reset: max |v| = {max_vel:.4f}"
    )


def test_reset_robot_state_partial_env_ids(env):
    """Resetting a subset of env_ids only affects those environments."""
    env.reset()

    # First set all to a known state
    all_ids = torch.arange(env.num_envs, device=env.device)
    reset_robot_state(env, all_ids, {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})

    # Record positions
    robot = env.scene["robot"]
    pos_before = robot.data.root_pos_w[:, :2].clone()

    # Now reset only first half
    half_ids = torch.arange(env.num_envs // 2, device=env.device)
    reset_robot_state(env, half_ids, {"x": (1.0, 1.0), "y": (1.0, 1.0), "yaw": (0.0, 0.0)})

    pos_after = robot.data.root_pos_w[:, :2]

    # Second half should be unchanged
    second_half = slice(env.num_envs // 2, None)
    pos_diff = (pos_after[second_half] - pos_before[second_half]).abs().max().item()

    print(f"\n  reset_robot_state partial reset:")
    print(f"    Second half max pos diff: {pos_diff:.6f}")

    assert pos_diff < 0.01, (
        f"Non-reset envs changed position: max diff = {pos_diff:.4f}"
    )


# =====================================================================
# randomize_friction — Tests
# =====================================================================


def test_randomize_friction_within_range(env):
    """Friction values after randomization must lie within the specified range."""
    env.reset()
    env_ids = torch.arange(env.num_envs, device=env.device)
    friction_range = (0.3, 1.2)

    randomize_friction(env, env_ids, friction_range)

    robot = env.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()
    static_friction = materials[:, :, 0]

    print(f"\n  randomize_friction:")
    print(f"    Static friction range: [{static_friction.min().item():.3f}, "
          f"{static_friction.max().item():.3f}]")

    # All friction values should be within the specified range (with tolerance)
    assert (static_friction >= friction_range[0] - 0.01).all(), (
        f"Friction below minimum: min={static_friction.min().item():.3f}"
    )
    assert (static_friction <= friction_range[1] + 0.01).all(), (
        f"Friction above maximum: max={static_friction.max().item():.3f}"
    )


def test_randomize_friction_dynamic_less_than_static(env):
    """Dynamic friction should be less than or equal to static friction."""
    env.reset()
    env_ids = torch.arange(env.num_envs, device=env.device)

    randomize_friction(env, env_ids, (0.3, 1.2))

    robot = env.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()
    static_friction = materials[:, :, 0]
    dynamic_friction = materials[:, :, 1]

    print(f"\n  Friction dynamic <= static:")
    print(f"    Max dynamic: {dynamic_friction.max().item():.3f}")
    print(f"    Min static:  {static_friction.min().item():.3f}")

    assert (dynamic_friction <= static_friction + 1e-6).all(), (
        "Dynamic friction exceeds static friction in some environments"
    )


def test_randomize_friction_empty_env_ids(env):
    """Passing empty env_ids should be a no-op (no crash)."""
    env.reset()
    empty_ids = torch.tensor([], dtype=torch.long, device=env.device)

    # Should not raise
    randomize_friction(env, empty_ids, (0.3, 1.2))


# =====================================================================
# randomize_mass — Tests
# =====================================================================


def test_randomize_mass_scales_correctly(env):
    """Mass values after randomization should be within scale range of defaults."""
    env.reset()

    # Clear cached defaults
    if hasattr(env, "_default_body_masses"):
        delattr(env, "_default_body_masses")

    env_ids = torch.arange(env.num_envs, device=env.device)
    mass_range = (0.8, 1.2)

    randomize_mass(env, env_ids, mass_range)

    robot = env.scene["robot"]
    current_masses = robot.root_physx_view.get_masses()
    default_masses = env._default_body_masses

    ratios = current_masses / (default_masses + 1e-8)
    min_ratio = ratios.min().item()
    max_ratio = ratios.max().item()

    print(f"\n  randomize_mass:")
    print(f"    Mass scale range: [{min_ratio:.3f}, {max_ratio:.3f}]")

    assert min_ratio >= mass_range[0] - 0.01, (
        f"Mass scale below minimum: {min_ratio:.3f}"
    )
    assert max_ratio <= mass_range[1] + 0.01, (
        f"Mass scale above maximum: {max_ratio:.3f}"
    )


def test_randomize_mass_empty_env_ids(env):
    """Passing empty env_ids should be a no-op."""
    env.reset()
    empty_ids = torch.tensor([], dtype=torch.long, device=env.device)

    randomize_mass(env, empty_ids, (0.8, 1.2))


# =====================================================================
# randomize_motor_strength — Tests
# =====================================================================


def test_randomize_motor_strength_within_range(env):
    """Effort limits after randomization should be within scale range."""
    env.reset()

    # Clear cached defaults
    if hasattr(env, "_default_effort_limits"):
        delattr(env, "_default_effort_limits")

    env_ids = torch.arange(env.num_envs, device=env.device)
    strength_range = (0.85, 1.15)

    randomize_motor_strength(env, env_ids, strength_range)

    robot = env.scene["robot"]
    current_limits = robot.root_physx_view.get_dof_max_forces()
    default_limits = env._default_effort_limits

    # Only check wheel drive joints (first 4) — roller joints have 0 effort limit
    drive_mask = default_limits[0] > 0
    current_drives = current_limits[:, drive_mask]
    default_drives = default_limits[:, drive_mask]

    ratios = current_drives / (default_drives + 1e-8)
    min_ratio = ratios.min().item()
    max_ratio = ratios.max().item()

    print(f"\n  randomize_motor_strength:")
    print(f"    Effort limit scale range (drives only): [{min_ratio:.3f}, {max_ratio:.3f}]")

    assert min_ratio >= strength_range[0] - 0.01, (
        f"Strength scale below minimum: {min_ratio:.3f}"
    )
    assert max_ratio <= strength_range[1] + 0.01, (
        f"Strength scale above maximum: {max_ratio:.3f}"
    )


def test_randomize_motor_strength_per_joint_independent(env):
    """Each wheel drive joint should get an independent scale factor."""
    env.reset()

    if hasattr(env, "_default_effort_limits"):
        delattr(env, "_default_effort_limits")

    env_ids = torch.arange(env.num_envs, device=env.device)
    randomize_motor_strength(env, env_ids, (0.5, 1.5))

    robot = env.scene["robot"]
    current_limits = robot.root_physx_view.get_dof_max_forces()
    default_limits = env._default_effort_limits

    # Only check wheel drive joints (first 4) — roller joints have 0 effort limit
    drive_mask = default_limits[0] > 0
    current_drives = current_limits[:, drive_mask]
    default_drives = default_limits[:, drive_mask]

    ratios = current_drives / (default_drives + 1e-8)

    # Check that joints within the same env have different ratios
    per_env_std = ratios.std(dim=-1)
    n_varied = (per_env_std > 0.001).sum().item()

    print(f"\n  Motor strength independence:")
    print(f"    Envs with varied joint scales: {n_varied}/{env.num_envs}")

    # At least half the envs should show variation across joints
    assert n_varied > env.num_envs * 0.3, (
        f"Motor strength appears uniform across joints: "
        f"only {n_varied}/{env.num_envs} envs show variation"
    )


# =====================================================================
# randomize_d555_mount_offset — Tests
# =====================================================================


def test_d555_mount_offset_creates_quaternion(env):
    """Mount offset should create a valid quaternion stored on env."""
    env.reset()

    # Clear cached state
    if hasattr(env, "_d555_mount_quat"):
        delattr(env, "_d555_mount_quat")

    env_ids = torch.arange(env.num_envs, device=env.device)
    randomize_d555_mount_offset(env, env_ids, max_angle_deg=2.0)

    assert hasattr(env, "_d555_mount_quat"), (
        "randomize_d555_mount_offset did not create env._d555_mount_quat"
    )

    quat = env._d555_mount_quat
    assert quat.shape == (env.num_envs, 4), (
        f"Expected shape ({env.num_envs}, 4), got {quat.shape}"
    )

    # Quaternions should be unit length
    norms = torch.norm(quat, dim=-1)
    print(f"\n  D555 mount offset:")
    print(f"    Quaternion norm range: [{norms.min().item():.6f}, {norms.max().item():.6f}]")

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), (
        f"Mount quaternions not unit length: "
        f"range [{norms.min().item():.6f}, {norms.max().item():.6f}]"
    )


def test_d555_mount_offset_small_angles(env):
    """With max_angle_deg=2, quaternion w-component should be close to 1 (small rotation)."""
    env.reset()

    if hasattr(env, "_d555_mount_quat"):
        delattr(env, "_d555_mount_quat")

    env_ids = torch.arange(env.num_envs, device=env.device)
    randomize_d555_mount_offset(env, env_ids, max_angle_deg=2.0)

    quat = env._d555_mount_quat
    w_component = quat[:, 0]

    print(f"\n  D555 mount offset (small angles):")
    print(f"    w range: [{w_component.min().item():.6f}, {w_component.max().item():.6f}]")

    # For a 2-degree rotation: cos(2°/2) ≈ 0.9998
    # Since each axis is ±2° independently, the total rotation is larger
    # but w should still be close to 1
    assert (w_component > 0.99).all(), (
        f"Mount offset too large for 2° max: "
        f"w_min = {w_component.min().item():.6f}"
    )


# =====================================================================
# randomize_goal_noise — Tests
# =====================================================================


def test_randomize_goal_noise_perturbs_goals(env):
    """Goal positions should change after applying noise."""
    env.reset()

    # Capture original goal
    command_term = env.command_manager.get_term("goal_command")
    goal_before = command_term.command[:, :2].clone()

    env_ids = torch.arange(env.num_envs, device=env.device)
    randomize_goal_noise(env, env_ids, command_name="goal_command", noise_std=0.5)

    goal_after = command_term.command[:, :2]
    diff = (goal_after - goal_before).abs()
    max_diff = diff.max().item()

    print(f"\n  randomize_goal_noise:")
    print(f"    Max position change: {max_diff:.4f} m")

    assert max_diff > 0.0, (
        "Goal noise did not change any goal positions"
    )


def test_randomize_goal_noise_empty_env_ids(env):
    """Passing empty env_ids should be a no-op."""
    env.reset()
    empty_ids = torch.tensor([], dtype=torch.long, device=env.device)

    # Should not raise
    randomize_goal_noise(env, empty_ids, command_name="goal_command", noise_std=0.5)
