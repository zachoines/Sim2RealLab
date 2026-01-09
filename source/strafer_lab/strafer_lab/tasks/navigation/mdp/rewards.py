"""Custom reward functions for Strafer navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def goal_reached_reward(
    env: ManagerBasedEnv,
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Reward for reaching the goal position.
    
    Args:
        env: The environment instance.
        threshold: Distance threshold for considering goal reached.
        command_name: Name of the command manager providing goal positions.
    
    Returns:
        Binary reward: 1.0 if goal reached, 0.0 otherwise.
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    
    distance = torch.norm(goal_pos - robot_pos, dim=-1)
    return (distance < threshold).float()


def goal_progress_reward(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward for making progress toward the goal.
    
    This is a dense reward based on the change in distance to goal.
    
    Args:
        env: The environment instance.
        command_name: Name of the command manager providing goal positions.
    
    Returns:
        Progress reward (positive if moving toward goal).
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    
    # Current distance
    current_distance = torch.norm(goal_pos - robot_pos, dim=-1)
    
    # Get previous distance from environment state
    if not hasattr(env, "_prev_goal_distance"):
        env._prev_goal_distance = current_distance.clone()
    
    # Progress = reduction in distance
    progress = env._prev_goal_distance - current_distance
    
    # Update previous distance
    env._prev_goal_distance = current_distance.clone()
    
    return progress


def heading_to_goal_reward(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward for facing toward the goal.
    
    Args:
        env: The environment instance.
        command_name: Name of the command manager providing goal positions.
    
    Returns:
        Heading alignment reward (1.0 when facing goal, -1.0 when facing away).
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_quat = robot.data.root_quat_w
    
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    
    # Direction to goal
    to_goal = goal_pos - robot_pos
    goal_angle = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    
    # Robot heading (yaw from quaternion)
    robot_yaw = 2.0 * torch.atan2(robot_quat[:, 3], robot_quat[:, 0])
    
    # Angular difference
    angle_diff = goal_angle - robot_yaw
    # Normalize to [-pi, pi]
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    
    # Reward based on cosine of angle difference
    return torch.cos(angle_diff)


def energy_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalty for energy consumption (motor effort).
    
    Encourages efficient motion by penalizing high motor torques.
    
    Returns:
        Negative reward proportional to total motor effort.
    """
    robot = env.scene["robot"]
    
    # Sum of squared joint efforts
    efforts = robot.data.applied_torque
    energy = torch.sum(efforts ** 2, dim=-1)
    
    return energy


def action_smoothness_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalty for jerky/non-smooth actions.
    
    Encourages smooth motion by penalizing large action changes.
    
    Returns:
        Negative reward proportional to action rate of change.
    """
    current_action = env.action_manager.action
    
    if not hasattr(env, "_prev_action"):
        env._prev_action = current_action.clone()
    
    # Rate of change
    action_diff = current_action - env._prev_action
    smoothness_cost = torch.sum(action_diff ** 2, dim=-1)
    
    env._prev_action = current_action.clone()
    
    return smoothness_cost
