"""Custom reward functions for Strafer navigation task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


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

    Handles two discontinuities that would produce false spikes:
    - Episode reset: ``episode_length_buf == 0``
    - Mid-episode goal resample: ``GoalCommand.goal_resampled`` flag

    In both cases the previous distance is re-seeded to the current distance
    so that the delta is zero on that step.

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

    # Initialize previous distance if not present
    if not hasattr(env, "_prev_goal_distance"):
        env._prev_goal_distance = current_distance.clone()

    # Reset previous distance for environments that just reset
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._prev_goal_distance[reset_mask] = current_distance[reset_mask]

    # Reset previous distance for mid-episode goal resamples
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "goal_resampled") and command_term.goal_resampled.any():
        resample_mask = command_term.goal_resampled
        env._prev_goal_distance[resample_mask] = current_distance[resample_mask]

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
    
    # Robot heading (yaw from quaternion, full formula handles pitch/roll)
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
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

    # Initialize previous action if not present
    if not hasattr(env, "_prev_action"):
        env._prev_action = current_action.clone()

    # Reset previous action for environments that just reset
    # This prevents incorrect penalty spikes after episode reset
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._prev_action[reset_mask] = current_action[reset_mask]

    # Rate of change (L1 norm — penalizes large and small changes
    # proportionally, unlike L2 which quadratically amplifies large jumps
    # and creates a "don't move at all" gradient)
    action_diff = current_action - env._prev_action
    smoothness_cost = torch.sum(torch.abs(action_diff), dim=-1)

    env._prev_action = current_action.clone()

    return smoothness_cost


def goal_proximity_potential(
    env: ManagerBasedEnv,
    command_name: str,
    sigma: float = 0.3,
) -> torch.Tensor:
    """Potential-based proximity shaping — rewards change in closeness, not closeness itself.

    Uses an exponential potential ``phi(s) = exp(-distance / sigma)`` and returns
    the one-step difference ``phi(s_t) - phi(s_{t-1})``.  This gives strong
    gradient near the goal (where the exponential curve is steep) without the
    loitering incentive of a per-step proximity reward — once the robot stops
    moving, the potential delta is zero and no reward accrues.

    Handles two discontinuities that would produce false spikes:
    - Episode reset: ``episode_length_buf == 0``
    - Mid-episode goal resample: ``GoalCommand.goal_resampled`` flag

    In both cases the previous potential is re-seeded to the current potential
    so that the delta is zero on that step (same pattern as ``goal_progress_reward``).

    Complements ``goal_progress_reward`` (linear potential ``phi = -distance``):
    linear shaping gives uniform gradient at all distances, while exponential
    shaping amplifies gradient in the final approach where linear flattens out.

    Args:
        env: The environment instance.
        command_name: Name of the command manager providing goal positions.
        sigma: Temperature parameter. Smaller = sharper gradient near goal.

    Returns:
        Potential difference in [-1, 1]. Shape: (num_envs,)
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]

    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]

    distance = torch.norm(goal_pos - robot_pos, dim=-1)
    phi = torch.exp(-distance / sigma)

    # Initialize previous potential if not present
    if not hasattr(env, "_prev_proximity_phi"):
        env._prev_proximity_phi = phi.clone()

    # Reset previous potential for environments that just reset
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._prev_proximity_phi[reset_mask] = phi[reset_mask]

    # Reset previous potential for mid-episode goal resamples
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "goal_resampled") and command_term.goal_resampled.any():
        resample_mask = command_term.goal_resampled
        env._prev_proximity_phi[resample_mask] = phi[resample_mask]

    # Potential difference: positive when getting closer
    reward = phi - env._prev_proximity_phi

    # Update previous potential
    env._prev_proximity_phi = phi.clone()

    return reward


def collision_penalty(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty for colliding with obstacles.

    Uses the contact sensor's ``force_matrix_w`` to detect obstacle-specific
    contact forces. Requires ``filter_prim_paths_expr`` on the sensor config
    so that ``force_matrix_w`` is populated with per-obstacle forces.

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity config for the contact sensor.
        threshold: Force magnitude threshold for counting a collision.

    Returns:
        Binary collision indicator: 1.0 if any obstacle contact force > threshold.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # force_matrix_w shape: (num_envs, num_bodies, num_filter_prims, 3)
    # Only populated when filter_prim_paths_expr is set on the sensor.
    force_matrix = contact_sensor.data.force_matrix_w
    # Magnitude per (body, obstacle) pair → (num_envs, num_bodies, num_obstacles)
    force_mag = torch.norm(force_matrix, dim=-1)
    # Collapse body and obstacle dims: any obstacle contact on any body counts
    has_collision = (force_mag > threshold).any(dim=-1).any(dim=-1).float()
    return has_collision


def speed_near_goal_penalty(
    env: ManagerBasedEnv,
    command_name: str,
    distance_threshold: float = 1.0,
    min_speed: float = 0.15,
) -> torch.Tensor:
    """Graduated penalty for excessive speed near the goal.

    Encourages smooth deceleration on approach rather than crashing into the
    goal at full speed, while always allowing a low creep speed for the final
    approach.

    The penalty has two factors that multiply together:
    - **Excess speed**: ``max(0, speed - min_speed)``.  Speed at or below
      ``min_speed`` is never penalized, so the robot can always creep in.
    - **Proximity ramp**: ``max(0, 1 - distance / distance_threshold)``.
      Linearly increases from 0 at ``distance_threshold`` to 1 at distance 0.
      At the boundary there is no penalty; it strengthens smoothly as the
      robot gets closer.

    Args:
        env: The environment instance.
        command_name: Name of the command manager providing goal positions.
        distance_threshold: Distance at which the penalty begins to ramp up.
        min_speed: Speed (m/s) below which no penalty is applied.  Should be
            fast enough for the robot to comfortably reach the goal.

    Returns:
        Speed penalty when near goal. Shape: (num_envs,)
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    robot_vel = robot.data.root_lin_vel_b[:, :2]

    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]

    distance = torch.norm(goal_pos - robot_pos, dim=-1)
    speed = torch.norm(robot_vel, dim=-1)

    # Only penalize speed above the minimum creep threshold
    excess_speed = (speed - min_speed).clamp(min=0.0)

    # Linear ramp: 0 at distance_threshold, 1 at distance 0
    proximity_ramp = (1.0 - distance / distance_threshold).clamp(min=0.0)

    return excess_speed * proximity_ramp


def alive_bonus(env: ManagerBasedEnv) -> torch.Tensor:
    """Small positive reward per step for staying alive (not flipped/terminated).

    Note: With continuous multi-goal episodes where only timeout/flip terminate,
    this is a constant that provides no gradient. Consider removing from RewardsCfg.

    Returns:
        Constant 1.0 for all environments. Shape: (num_envs,)
    """
    return torch.ones(env.num_envs, device=env.device)


def collision_sustained_penalty(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Escalating penalty for sustained obstacle contact.

    Returns the total obstacle contact force magnitude, clipped to [0, 1].
    Unlike the binary ``collision_penalty``, this grows with contact intensity,
    penalizing the robot more for pushing hard into an obstacle than for
    a brief bump.

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity config for the contact sensor.
        threshold: Force below which contact is ignored.

    Returns:
        Normalized contact intensity in [0, 1]. Shape: (num_envs,)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = contact_sensor.data.force_matrix_w
    # Sum force magnitudes across all bodies and obstacles
    force_mag = torch.norm(force_matrix, dim=-1)  # (N, B, M)
    total_force = force_mag.sum(dim=-1).sum(dim=-1)  # (N,)
    # Zero out below threshold, normalize to [0, 1] (cap at 50N)
    total_force = torch.clamp(total_force - threshold, min=0.0) / 50.0
    return torch.clamp(total_force, max=1.0)


# ---------------------------------------------------------------------------
# Scene-geometry collision rewards
# ---------------------------------------------------------------------------
# Intended for Infinigen configs where scene geometry has variable/unknown mesh prims.


def collision_penalty_net(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary collision penalty using net contact forces.

    Uses ``net_forces_w`` which reports total contact force on the sensor
    body from ALL contacts.  Since body_link is wheel-suspended (~10 cm
    above ground), any significant force on body_link indicates collision
    with scene geometry (walls, furniture, clutter).

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity config for the contact sensor.
        threshold: Force magnitude threshold (N) for counting a collision.

    Returns:
        Binary collision indicator: 1.0 if net contact force > threshold.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w shape: (num_envs, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w
    # Magnitude per body -> (num_envs, num_bodies)
    force_mag = torch.norm(net_forces, dim=-1)
    # Any body with force above threshold -> (num_envs,)
    has_collision = (force_mag > threshold).any(dim=-1).float()
    return has_collision


def collision_sustained_penalty_net(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Escalating collision penalty using net contact forces.

    Returns normalized total contact force magnitude, clipped to [0, 1].

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity config for the contact sensor.
        threshold: Force below which contact is ignored (N).

    Returns:
        Normalized contact intensity in [0, 1]. Shape: (num_envs,)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w shape: (num_envs, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w
    # Magnitude per body -> (num_envs, num_bodies)
    force_mag = torch.norm(net_forces, dim=-1)
    # Sum across bodies -> (num_envs,)
    total_force = force_mag.sum(dim=-1)
    # Zero out below threshold, normalize to [0, 1] (cap at 50N)
    total_force = torch.clamp(total_force - threshold, min=0.0) / 50.0
    return torch.clamp(total_force, max=1.0)
