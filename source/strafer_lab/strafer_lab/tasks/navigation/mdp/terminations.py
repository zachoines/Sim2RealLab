"""Custom termination functions for Strafer navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_flipped(env: ManagerBasedEnv, threshold: float = 0.5) -> torch.Tensor:
    """Terminate episode if robot is flipped over.
    
    Checks if the robot's up vector (z-axis) is pointing downward,
    indicating the robot has tipped over.
    
    Args:
        env: The environment instance.
        threshold: Cosine threshold for "flipped" detection.
                  0.5 = 60 degrees from vertical.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot = env.scene["robot"]
    
    # Projected gravity in robot frame
    # If robot is upright, gravity projects to (0, 0, -1)
    # If flipped, gravity projects to (0, 0, +1)
    projected_gravity = robot.data.projected_gravity_b
    
    # Check if z-component of projected gravity is positive (flipped)
    # projected_gravity_b is a unit vector: upright ≈ (0,0,-1), flipped ≈ (0,0,+1)
    # threshold=0.5 triggers at ~60° from vertical
    is_flipped = projected_gravity[:, 2] > threshold
    
    return is_flipped


def goal_reached(
    env: ManagerBasedEnv,
    command_name: str,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Terminate episode when robot reaches the goal.

    Args:
        env: The environment instance.
        command_name: Name of the command manager providing goal positions.
        threshold: Distance threshold (meters) for goal reached.

    Returns:
        Boolean tensor indicating which environments reached the goal.
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]

    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]

    distance = torch.norm(goal_pos - robot_pos, dim=-1)
    return distance < threshold
