"""Custom termination functions for Strafer navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


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


def sustained_collision(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    max_steps: int = 10,
) -> torch.Tensor:
    """Terminate if robot has been in sustained collision for max_steps.

    Uses a leaky counter: +1 on collision frames, -0.5 on non-collision
    frames (clamped to 0).  This prevents physics-bounce oscillations from
    resetting the counter — a robot wedged against a wall with alternating
    contact/no-contact frames still accumulates toward termination, while
    a genuine recovery (multiple consecutive non-contact frames) drains
    the counter back to zero.

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity config for the contact sensor.
        threshold: Force magnitude threshold (N) for counting a collision.
        max_steps: Accumulated collision steps before termination.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w
    force_mag = torch.norm(net_forces, dim=-1)
    has_collision = (force_mag > threshold).any(dim=-1)

    if not hasattr(env, "_collision_step_count"):
        env._collision_step_count = torch.zeros(env.num_envs, device=env.device)

    # Leaky counter: +1 on contact, -0.5 on no-contact (clamp ≥ 0)
    env._collision_step_count = torch.where(
        has_collision,
        env._collision_step_count + 1.0,
        (env._collision_step_count - 0.5).clamp(min=0.0),
    )

    # Reset counter on episode reset
    reset_mask = env.episode_length_buf == 0
    env._collision_step_count[reset_mask] = 0

    return env._collision_step_count >= max_steps
