# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Robot control utilities for integration tests.

This module provides common functions for positioning and controlling robots
during integration tests. These utilities ensure robots are properly spaced
on the grid and can be frozen in place for stationary measurements.
"""

import torch
import warp as wp


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
        face_wall: If True (default), robot keeps the identity quaternion so
                   body +X and the front-mounted camera face the wall. If
                   False, rotate 180 degrees around Z so robot/camera face
                   away from the wall along world -X.
    """
    robot = env.scene["robot"]
    num_envs = env.num_envs
    device = env.device

    # Get each environment's origin position (set by GridCloner)
    env_origins = get_env_origins(env)  # Shape: (num_envs, 3)

    # Create desired root state
    # Root state format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z,
    #                     vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
    root_pose = torch.zeros(num_envs, 7, device=device)

    # Position: environment origin + slight elevation
    root_pose[:, 0] = env_origins[:, 0]        # x = grid origin x
    root_pose[:, 1] = env_origins[:, 1]        # y = grid origin y
    root_pose[:, 2] = env_origins[:, 2] + 0.1  # z = grid origin z + slight elevation

    if face_wall:
        # Identity quaternion (x, y, z, w) = (0, 0, 0, 1)
        # Robot body +X remains aligned with world +X.
        root_pose[:, 3] = 0.0   # quat_x
        root_pose[:, 4] = 0.0   # quat_y
        root_pose[:, 5] = 0.0   # quat_z
        root_pose[:, 6] = 1.0   # quat_w
    else:
        # 180-degree rotation around Z axis: (x, y, z, w) = (0, 0, 1, 0)
        # Robot body +X now points to world -X (away from wall).
        root_pose[:, 3] = 0.0   # quat_x
        root_pose[:, 4] = 0.0   # quat_y
        root_pose[:, 5] = 1.0   # quat_z
        root_pose[:, 6] = 0.0   # quat_w

    # Write root pose and zero velocity to simulation
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(num_envs, 6, device=device)
    )

    # Also reset joint positions and velocities to zero
    joint_pos = torch.zeros(num_envs, robot.num_joints, device=device)
    joint_vel = torch.zeros(num_envs, robot.num_joints, device=device)
    robot.write_joint_position_to_sim_index(position=joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel)


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
        _frozen_state["root_state"] = torch.cat([
            wp.to_torch(robot.data.root_link_pose_w),
            wp.to_torch(robot.data.root_link_vel_w),
        ], dim=-1).clone()
        _frozen_state["joint_pos"] = wp.to_torch(robot.data.joint_pos).clone()

    frozen_pose = _frozen_state["root_state"][:, :7].clone()
    robot.write_root_pose_to_sim_index(root_pose=frozen_pose)
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(num_envs, 6, device=device)
    )

    frozen_joint_pos = _frozen_state["joint_pos"]
    joint_vel = torch.zeros(num_envs, robot.num_joints, device=device)
    robot.write_joint_position_to_sim_index(position=frozen_joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
