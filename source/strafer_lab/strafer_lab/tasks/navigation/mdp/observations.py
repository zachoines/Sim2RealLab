"""Custom observation functions for Strafer navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def base_lin_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Get robot base linear velocity in local frame.
    
    Returns:
        Linear velocity (x, y, z) in robot's local frame. Shape: (num_envs, 3)
    """
    robot = env.scene["robot"]
    return robot.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Get robot base angular velocity in local frame.
    
    Returns:
        Angular velocity (roll, pitch, yaw) in robot's local frame. Shape: (num_envs, 3)
    """
    robot = env.scene["robot"]
    return robot.data.root_ang_vel_b


def projected_gravity(env: ManagerBasedEnv) -> torch.Tensor:
    """Get gravity vector projected into robot's local frame.
    
    Useful for detecting if robot is tilted or flipped.
    
    Returns:
        Projected gravity vector. Shape: (num_envs, 3)
    """
    robot = env.scene["robot"]
    return robot.data.projected_gravity_b


def goal_position_relative(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Get goal position relative to robot in local frame.
    
    Args:
        env: The environment instance.
        command_name: Name of the command manager providing goal positions.
    
    Returns:
        Relative goal position (x, y) in robot's local frame. Shape: (num_envs, 2)
    """
    # Get goal position from command manager
    command = env.command_manager.get_command(command_name)
    goal_pos_w = command[:, :2]  # (x, y) in world frame
    
    # Get robot position and orientation
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[:, :2]
    robot_quat_w = robot.data.root_quat_w
    
    # Compute relative position
    rel_pos_w = goal_pos_w - robot_pos_w
    
    # Rotate to local frame using yaw
    # Extract yaw from quaternion (simplified for 2D)
    yaw = 2.0 * torch.atan2(robot_quat_w[:, 3], robot_quat_w[:, 0])
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    
    rel_pos_local = torch.stack([
        cos_yaw * rel_pos_w[:, 0] + sin_yaw * rel_pos_w[:, 1],
        -sin_yaw * rel_pos_w[:, 0] + cos_yaw * rel_pos_w[:, 1],
    ], dim=-1)
    
    return rel_pos_local


def last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the previous action taken.
    
    Useful for learning smooth policies.
    
    Returns:
        Previous action tensor. Shape: (num_envs, action_dim)
    """
    return env.action_manager.action
