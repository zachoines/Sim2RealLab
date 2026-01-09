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
    # Normal: gravity_z ≈ -9.8 (negative)
    # Flipped: gravity_z ≈ +9.8 (positive)
    is_flipped = projected_gravity[:, 2] > threshold * 9.81
    
    return is_flipped
