"""Custom event functions for Strafer navigation task.

Events are used for:
- Resetting robot state at episode start
- Domain randomization during training
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_robot_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict,
) -> None:
    """Reset robot to random initial pose.
    
    Args:
        env: The environment instance.
        env_ids: Indices of environments to reset.
        pose_range: Dictionary with keys 'x', 'y', 'yaw' specifying
                   (min, max) ranges for each component.
    """
    robot = env.scene["robot"]
    num_resets = len(env_ids)
    device = env.device
    
    # Sample random positions
    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    yaw_range = pose_range.get("yaw", (0.0, 0.0))
    
    x = torch.rand(num_resets, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y = torch.rand(num_resets, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    z = torch.full((num_resets,), 0.1, device=device)  # Fixed height above ground
    
    yaw = torch.rand(num_resets, device=device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    
    # Convert yaw to quaternion (w, x, y, z)
    quat = torch.zeros(num_resets, 4, device=device)
    quat[:, 0] = torch.cos(yaw / 2)  # w
    quat[:, 3] = torch.sin(yaw / 2)  # z
    
    # Set robot state
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0] = x  # pos_x
    root_state[:, 1] = y  # pos_y
    root_state[:, 2] = z  # pos_z
    root_state[:, 3:7] = quat  # orientation
    root_state[:, 7:] = 0.0  # zero velocity
    
    robot.write_root_state_to_sim(root_state, env_ids)


def randomize_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
) -> None:
    """Randomize ground friction coefficient.
    
    Domain randomization to improve sim-to-real transfer.
    
    Args:
        env: The environment instance.
        env_ids: Indices of environments to randomize.
        friction_range: (min, max) friction coefficient range.
    """
    # Note: This is a placeholder - actual implementation depends on
    # how physics materials are configured in the scene
    # 
    # For full implementation, you would:
    # 1. Access the ground plane's physics material
    # 2. Modify the static/dynamic friction coefficients
    #
    # Example (if ground plane has material API):
    # ground = env.scene["ground"]
    # friction = torch.rand(len(env_ids)) * (friction_range[1] - friction_range[0]) + friction_range[0]
    # ground.set_friction_coefficients(friction, env_ids)
    pass
