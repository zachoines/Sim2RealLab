"""Custom termination functions for Strafer navigation task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg
import warp as wp


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
    projected_gravity = wp.to_torch(robot.data.projected_gravity_b)
    
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
    robot_pos = wp.to_torch(robot.data.root_pos_w)[:, :2]

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
    net_forces = wp.to_torch(contact_sensor.data.net_forces_w)
    force_mag = torch.norm(net_forces, dim=-1)
    has_collision = (force_mag > threshold).any(dim=-1)

    if not hasattr(env, "_collision_step_count"):
        env._collision_step_count = torch.zeros(env.num_envs, device=env.device)

    # Isaac Lab increments episode_length_buf before computing terminations
    # inside env.step(), so the first post-reset step sees episode_length_buf==1.
    # Clear the stale counter before the leaky update so the new episode starts
    # from zero whether the term is called directly after env.reset() or through
    # the first env.step() after a reset.
    reset_mask = env.episode_length_buf <= 1
    env._collision_step_count[reset_mask] = 0.0

    # Leaky counter: +1 on contact, -0.5 on no-contact (clamp ≥ 0)
    env._collision_step_count = torch.where(
        has_collision,
        env._collision_step_count + 1.0,
        (env._collision_step_count - 0.5).clamp(min=0.0),
    )

    return env._collision_step_count >= max_steps


def path_complete(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the robot parks at the end of its planned path.

    Reads the dwell-gated completion flag the subgoal command term maintains:
    the robot has held within ``dwell_radius_m`` of the path's final point at
    low speed (``dwell_speed_max_m_s``) for ``dwell_steps`` consecutive control
    steps. Both this termination and ``path_complete_reward`` read this one
    flag, so success is parking, not touching at speed.

    Args:
        env: The environment instance.
        command_name: Name of the subgoal command term.

    Returns:
        Boolean tensor indicating which environments completed their path.
    """
    return env.command_manager.get_term(command_name).path_complete


def off_path_divergence(
    env: ManagerBasedEnv,
    command_name: str,
    max_off_path_m: float = 0.5,
) -> torch.Tensor:
    """Terminate when the robot strays too far from its planned path.

    The path is planned through known-free space, so a large cross-track
    error means the policy abandoned the corridor the planner cleared —
    the episode carries no further useful path-tracking signal.

    Args:
        env: The environment instance.
        command_name: Name of the subgoal command term.
        max_off_path_m: Cross-track distance threshold (meters).

    Returns:
        Boolean tensor indicating which environments diverged off-path.
    """
    term = env.command_manager.get_term(command_name)
    return term.cross_track_error > max_off_path_m


def path_time_out(
    env: ManagerBasedEnv,
    command_name: str,
    action_name: str,
    intercept_s: float,
    time_per_m_s: float,
    reference_speed_scale: float,
    ceiling_s: float,
) -> torch.Tensor:
    """Time out each env's episode at a limit scaled to its path and its speed cap.

    The limit is ``intercept_s + time_per_m_s * L`` seconds, L the installed path's arc in metres, multiplied by
    ``reference_speed_scale / s`` where the episode's command-envelope scale s is below the reference (s is 1 where the
    action cfg has no envelope), and held within [the env's episode length, ``ceiling_s``]. Stateless: at ``compute()``
    the arc and s are the ending episode's, since the action term redraws s and the command installs the next path at
    the reset that follows.

    Args:
        env: The environment instance.
        command_name: Name of the subgoal command term (its ``path_cursor.total_arc`` is L).
        action_name: Name of the action term (its ``speed_scale`` is s when its cfg sets ``speed_scale_range``).
        intercept_s: Seconds the limit allows at zero arc.
        time_per_m_s: Seconds the limit adds per metre of arc.
        reference_speed_scale: The envelope scale the intercept and slope hold at; a smaller s stretches them.
        ceiling_s: The longest limit, in seconds.

    Returns:
        Boolean tensor indicating which environments have reached their limit.
    """
    base = env.max_episode_length
    arc = env.command_manager.get_term(command_name).path_cursor.total_arc
    action_term = env.action_manager.get_term(action_name)
    if getattr(action_term.cfg, "speed_scale_range", None) is None:
        stretch = torch.ones_like(arc)
    else:
        stretch = (reference_speed_scale / action_term.speed_scale).clamp(min=1.0)
    seconds = stretch * (intercept_s + time_per_m_s * arc)
    top = max(base, math.ceil(round(ceiling_s / env.step_dt, 9)))
    limit = torch.ceil(seconds / env.step_dt).long().clamp(base, top)
    return env.episode_length_buf >= limit
