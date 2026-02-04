# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Robot control utilities for integration tests.

This module provides common functions for positioning and controlling robots
during integration tests. These utilities ensure robots are properly spaced
on the grid and can be frozen in place for stationary measurements.
"""

import torch


def get_env_origins(env) -> torch.Tensor:
    """Get the world-frame origin positions for each environment.

    Isaac Lab's GridCloner places environments on a grid. Each environment
    has its own origin point in world coordinates.

    Args:
        env: The Isaac Lab environment

    Returns:
        Tensor of shape (num_envs, 3) with XYZ origin for each environment
    """
    return env.scene.env_origins


def reset_robot_pose(env, face_wall: bool = True):
    """Reset each robot to its environment's origin with specified orientation.

    IMPORTANT: Each robot must be positioned relative to its environment's
    grid origin, not at absolute (0,0,0). Isaac Lab's GridCloner places
    environments on a grid, and each robot needs to stay in its designated
    grid cell to avoid collisions with other robots.

    Args:
        env: The Isaac Lab environment
        face_wall: If True (default), robot faces -Y direction (identity quaternion).
                   If False, robot faces +Y direction (180° rotation around Z).
                   This parameter is primarily used by depth tests to orient
                   the camera toward or away from a wall.
    """
    robot = env.scene["robot"]
    num_envs = env.num_envs
    device = env.device

    # Get each environment's origin position (set by GridCloner)
    env_origins = get_env_origins(env)  # Shape: (num_envs, 3)

    # Create desired root state
    # Root state format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z,
    #                     vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
    root_state = torch.zeros(num_envs, 13, device=device)

    # Position: environment origin + slight elevation
    root_state[:, 0] = env_origins[:, 0]        # x = grid origin x
    root_state[:, 1] = env_origins[:, 1]        # y = grid origin y
    root_state[:, 2] = env_origins[:, 2] + 0.1  # z = grid origin z + slight elevation

    if face_wall:
        # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
        # Robot forward is -Y in USD local frame, which aligns with world -Y
        root_state[:, 3] = 1.0   # quat_w
        root_state[:, 4] = 0.0   # quat_x
        root_state[:, 5] = 0.0   # quat_y
        root_state[:, 6] = 0.0   # quat_z
    else:
        # 180° rotation around Z axis: (w, x, y, z) = (0, 0, 0, 1)
        # Robot forward (-Y local) now points to world +Y (away from wall)
        root_state[:, 3] = 0.0   # quat_w
        root_state[:, 4] = 0.0   # quat_x
        root_state[:, 5] = 0.0   # quat_y
        root_state[:, 6] = 1.0   # quat_z

    # Velocities: all zero (stationary)
    # root_state[:, 7:13] already zero

    # Write root state to simulation
    robot.write_root_state_to_sim(root_state)

    # Also reset joint positions and velocities to zero
    joint_pos = torch.zeros(num_envs, robot.num_joints, device=device)
    joint_vel = torch.zeros(num_envs, robot.num_joints, device=device)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


# Module-level state for freeze_robot_in_place
# Using a dict to avoid issues with function attributes in different contexts
_frozen_state = {
    "root_state": None,
    "joint_pos": None,
}


def clear_frozen_state():
    """Clear the frozen robot state.

    Call this after resetting robot poses to ensure fresh state capture
    on the next freeze_robot_in_place call.
    """
    _frozen_state["root_state"] = None
    _frozen_state["joint_pos"] = None


def freeze_robot_in_place(env):
    """Zero out all robot velocities to freeze it in place.

    This is called after each simulation step to prevent any micro-settling
    from affecting sensor measurements. The robot's position is preserved from
    the first call after a reset, making the robot behave like a kinematic
    object during stationary sampling.

    Note: Call clear_frozen_state() after reset_robot_pose() to capture
    fresh state on the next freeze call.

    Args:
        env: The Isaac Lab environment
    """
    robot = env.scene["robot"]
    num_envs = env.num_envs
    device = env.device

    # On the first call after a reset, capture the desired frozen pose
    if _frozen_state["root_state"] is None:
        _frozen_state["root_state"] = robot.data.root_state_w.clone()
        _frozen_state["joint_pos"] = robot.data.joint_pos.clone()

    frozen_root = _frozen_state["root_state"].clone()
    # Ensure velocities are zero
    frozen_root[:, 7:13] = 0.0
    robot.write_root_state_to_sim(frozen_root)

    frozen_joint_pos = _frozen_state["joint_pos"]
    joint_vel = torch.zeros(num_envs, robot.num_joints, device=device)
    robot.write_joint_state_to_sim(frozen_joint_pos, joint_vel)
