"""Custom event functions for Strafer navigation task.

Events are used for:
- Resetting robot state at episode start
- Domain randomization during training
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg


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
    
    # Set robot state (positions are relative to each env's origin)
    root_state = robot.data.default_root_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids]
    root_state[:, 0] = env_origins[:, 0] + x  # pos_x
    root_state[:, 1] = env_origins[:, 1] + y  # pos_y
    root_state[:, 2] = env_origins[:, 2] + z  # pos_z
    root_state[:, 3:7] = quat  # orientation
    root_state[:, 7:] = 0.0  # zero velocity

    robot.write_root_state_to_sim(root_state, env_ids)


# Cache for spawn points tensor (loaded once, reused across resets)
_spawn_points_cache: dict[str, torch.Tensor] = {}


def reset_robot_state_on_floor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    spawn_points_xy: list[list[float]],
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
) -> None:
    """Reset robot to a random interior floor position.

    Samples from precomputed interior spawn points (from scenes_metadata.json)
    generated via area-weighted barycentric sampling of floor mesh triangles.
    Points are uniformly distributed across all walkable floor surfaces,
    including multi-room scenes with multiple floor meshes.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to reset.
        spawn_points_xy: List of [x, y] interior floor positions (env-local frame).
        yaw_range: (min, max) yaw range in radians.
    """
    robot = env.scene["robot"]
    num_resets = len(env_ids)
    device = env.device

    # Lazy-build and cache the points tensor
    cache_key = str(id(spawn_points_xy))
    if cache_key not in _spawn_points_cache or _spawn_points_cache[cache_key].device != device:
        _spawn_points_cache[cache_key] = torch.tensor(spawn_points_xy, device=device, dtype=torch.float32)
    pts = _spawn_points_cache[cache_key]

    # Sample random point indices
    indices = torch.randint(0, len(pts), (num_resets,), device=device)
    xy = pts[indices]  # (num_resets, 2)

    z = torch.full((num_resets,), 0.1, device=device)
    yaw = torch.rand(num_resets, device=device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]

    # Convert yaw to quaternion (w, x, y, z)
    quat = torch.zeros(num_resets, 4, device=device)
    quat[:, 0] = torch.cos(yaw / 2)
    quat[:, 3] = torch.sin(yaw / 2)

    root_state = robot.data.default_root_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids]
    root_state[:, 0] = env_origins[:, 0] + xy[:, 0]
    root_state[:, 1] = env_origins[:, 1] + xy[:, 1]
    root_state[:, 2] = env_origins[:, 2] + z
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0

    robot.write_root_state_to_sim(root_state, env_ids)


def randomize_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
) -> None:
    """Randomize wheel-ground friction coefficient for domain randomization.

    Modifies the physics material properties of the robot's contact surfaces
    (wheels) to simulate varying floor conditions. This helps with sim-to-real
    transfer by training policies robust to different friction levels.

    The friction value is randomized per-environment at reset time, simulating
    conditions like carpet vs tile vs polished concrete.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to randomize.
        friction_range: (min, max) friction coefficient range.
            Typical values: (0.3, 1.2) for indoor surfaces.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene["robot"]
    device = env.device

    # Get current material properties from PhysX view
    # Shape: (num_envs, max_num_shapes, 3) where 3 = [static, dynamic, restitution]
    materials = robot.root_physx_view.get_material_properties()

    # Ensure env_ids is on the same device as materials for indexing
    # PhysX view may return materials on a different device than env.device
    materials_device = materials.device
    env_ids_device = env_ids.to(materials_device)

    # Sample random friction values for each environment being reset
    num_resets = len(env_ids)
    friction_values = (
        torch.rand(num_resets, device=materials_device)
        * (friction_range[1] - friction_range[0])
        + friction_range[0]
    )

    # Apply friction to all shapes in the reset environments
    # Expand friction to match shape dimension: (num_resets,) -> (num_resets, num_shapes)
    num_shapes = materials.shape[1]
    friction_expanded = friction_values.unsqueeze(1).expand(-1, num_shapes)

    # Update static friction (index 0) and dynamic friction (index 1)
    # Dynamic friction should be <= static friction for physical consistency
    materials[env_ids_device, :, 0] = friction_expanded  # Static friction
    materials[env_ids_device, :, 1] = friction_expanded * 0.9  # Dynamic friction (slightly lower)
    # Keep restitution (index 2) unchanged

    # Apply the modified materials back to simulation
    # Note: set_material_properties expects env_ids on CPU
    env_ids_cpu = env_ids.cpu() if env_ids.device.type != "cpu" else env_ids
    robot.root_physx_view.set_material_properties(materials, env_ids_cpu)


def randomize_obstacles(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    obstacle_names: list[str],
    position_range: dict,
    min_robot_dist: float = 0.6,
) -> None:
    """Randomize obstacle positions at episode reset.

    Places obstacles at random positions within the specified range, ensuring
    they don't overlap the robot spawn area.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to reset.
        obstacle_names: Scene entity names for each obstacle.
        position_range: Dict with 'x' and 'y' as (min, max) tuples.
        min_robot_dist: Minimum distance from origin (robot spawn area).
    """
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    device = env.device
    x_range = position_range.get("x", (-4.0, 4.0))
    y_range = position_range.get("y", (-4.0, 4.0))
    z_height = 0.15  # half-height of 0.3m box

    for obs_name in obstacle_names:
        try:
            obstacle = env.scene[obs_name]
        except KeyError:
            continue

        # Sample positions with rejection for robot proximity
        accepted = torch.zeros(num_resets, dtype=torch.bool, device=device)
        ox = torch.zeros(num_resets, device=device)
        oy = torch.zeros(num_resets, device=device)

        for _ in range(20):
            remaining = ~accepted
            n = remaining.sum().item()
            if n == 0:
                break
            cx = torch.rand(n, device=device) * (x_range[1] - x_range[0]) + x_range[0]
            cy = torch.rand(n, device=device) * (y_range[1] - y_range[0]) + y_range[0]
            dist_from_origin = torch.sqrt(cx ** 2 + cy ** 2)
            far_enough = dist_from_origin >= min_robot_dist

            idx = torch.where(remaining)[0]
            newly_accepted = idx[far_enough]
            ox[newly_accepted] = cx[far_enough]
            oy[newly_accepted] = cy[far_enough]
            accepted[newly_accepted] = True

        # Fallback: place remaining at min_robot_dist in random direction
        remaining = ~accepted
        if remaining.any():
            n = remaining.sum().item()
            angle = torch.rand(n, device=device) * (2.0 * math.pi)
            ox[remaining] = min_robot_dist * torch.cos(angle)
            oy[remaining] = min_robot_dist * torch.sin(angle)

        # Build root state (positions are relative to each env's origin)
        root_state = obstacle.data.default_root_state[env_ids].clone()
        env_origins = env.scene.env_origins[env_ids]
        root_state[:, 0] = env_origins[:, 0] + ox
        root_state[:, 1] = env_origins[:, 1] + oy
        root_state[:, 2] = env_origins[:, 2] + z_height
        root_state[:, 3] = 1.0  # w (identity quaternion)
        root_state[:, 4:7] = 0.0
        root_state[:, 7:] = 0.0  # zero velocity

        obstacle.write_root_state_to_sim(root_state, env_ids)


def randomize_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mass_range: tuple[float, float],
) -> None:
    """Randomize robot base mass for domain randomization.

    Simulates mass variation from battery charge, added payloads, etc.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to randomize.
        mass_range: (min_scale, max_scale) multiplier for default mass.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene["robot"]
    device = env.device
    num_resets = len(env_ids)

    # Get current body masses from simulation
    body_masses = robot.root_physx_view.get_masses()
    body_device = body_masses.device
    env_ids_dev = env_ids.to(body_device)

    # Initialize default masses on first call
    if not hasattr(env, "_default_body_masses"):
        env._default_body_masses = body_masses.clone()

    # Sample random scale per environment
    scales = (
        torch.rand(num_resets, device=body_device)
        * (mass_range[1] - mass_range[0])
        + mass_range[0]
    )

    # Apply scale to all bodies in the reset environments
    num_bodies = body_masses.shape[1]
    scales_expanded = scales.unsqueeze(1).expand(-1, num_bodies)
    body_masses[env_ids_dev] = env._default_body_masses[env_ids_dev] * scales_expanded

    robot.root_physx_view.set_masses(body_masses, env_ids_dev)


def randomize_goal_noise(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    noise_std: float = 0.25,
) -> None:
    """Add Gaussian noise to goal positions (simulates VLM localization error).

    Args:
        env: The environment instance.
        env_ids: Indices of environments to apply noise to.
        command_name: Name of the command term to perturb.
        noise_std: Standard deviation of position noise in meters.
    """
    if len(env_ids) == 0:
        return

    command_term = env.command_manager.get_term(command_name)
    device = env.device

    noise = torch.randn(len(env_ids), 2, device=device) * noise_std
    command_term._goal[env_ids, :2] += noise


def randomize_motor_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    strength_range: tuple[float, float],
) -> None:
    """Randomize per-wheel motor strength for domain randomization.

    Scales the joint effort limit for each wheel independently, simulating
    motor-to-motor variation, degradation, or voltage sag. The scale is
    sampled independently per wheel so one motor can be weaker than others.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to randomize.
        strength_range: (min_scale, max_scale) multiplier for default effort limits.
            Example: (0.85, 1.15) = ±15% variation.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene["robot"]

    # Get current effort limits from PhysX view
    # Shape: (num_envs, num_joints)
    effort_limits = robot.root_physx_view.get_dof_max_forces()
    effort_device = effort_limits.device
    env_ids_dev = env_ids.to(effort_device)

    # Store defaults on first call
    if not hasattr(env, "_default_effort_limits"):
        env._default_effort_limits = effort_limits.clone()

    num_resets = len(env_ids)
    num_joints = effort_limits.shape[1]

    # Sample independent scale per joint per environment
    scales = (
        torch.rand(num_resets, num_joints, device=effort_device)
        * (strength_range[1] - strength_range[0])
        + strength_range[0]
    )

    effort_limits[env_ids_dev] = env._default_effort_limits[env_ids_dev] * scales

    env_ids_cpu = env_ids.cpu() if env_ids.device.type != "cpu" else env_ids
    robot.root_physx_view.set_dof_max_forces(effort_limits, env_ids_cpu)


def randomize_d555_mount_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_angle_deg: float = 2.0,
) -> None:
    """Randomize D555 camera/IMU mounting orientation offset.

    The Intel RealSense D555 is bolted to a bracket on the chassis. Small
    mounting imprecision means the sensor frame differs slightly from the
    nominal body frame. Since the IMU, depth camera, and RGB camera all
    share the same physical housing, they share a single random rotation.

    This event samples a small random rotation (roll, pitch, yaw each
    within ±max_angle_deg) per environment and stores it as a quaternion
    on ``env._d555_mount_quat``. IMU observation functions apply this
    rotation to sensor readings. For the depth/RGB camera the effect is
    negligible (1-2° ≈ 1-2 pixel shift) and the CNN learns invariance.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to randomize.
        max_angle_deg: Maximum offset per axis in degrees.
    """
    if len(env_ids) == 0:
        return

    device = env.device
    num_resets = len(env_ids)
    max_rad = math.radians(max_angle_deg)

    # Initialize storage on first call (identity = no offset)
    if not hasattr(env, "_d555_mount_quat"):
        env._d555_mount_quat = torch.zeros(env.num_envs, 4, device=device)
        env._d555_mount_quat[:, 0] = 1.0  # w = 1 (identity)

    # Sample small random Euler angles in [-max_rad, max_rad]
    roll = (torch.rand(num_resets, device=device) * 2.0 - 1.0) * max_rad
    pitch = (torch.rand(num_resets, device=device) * 2.0 - 1.0) * max_rad
    yaw = (torch.rand(num_resets, device=device) * 2.0 - 1.0) * max_rad

    mount_quat = quat_from_euler_xyz(roll, pitch, yaw)
    env._d555_mount_quat[env_ids] = mount_quat


def reset_robot_proc_room(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
) -> None:
    """Reset robot to a random BFS-reachable position in a procedural room.

    Reads from dynamic per-env spawn points computed by ``generate_proc_room``
    and stored on ``env._proc_room_spawn_pts``.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to reset.
        yaw_range: (min, max) yaw range in radians.
    """
    robot = env.scene["robot"]
    num_resets = len(env_ids)
    device = env.device

    # Read per-env spawn points (populated by generate_proc_room EventTerm)
    spawn_pts = env._proc_room_spawn_pts[env_ids]       # (B, K, 2)
    spawn_count = env._proc_room_spawn_count[env_ids]    # (B,)

    # Sample random index per env (bounded by actual count)
    max_idx = spawn_count.clamp(min=1)  # avoid div-by-zero
    indices = (torch.rand(num_resets, device=device) * max_idx.float()).long()
    indices = indices.clamp(max=spawn_pts.shape[1] - 1)

    # Gather selected XY positions
    xy = spawn_pts[torch.arange(num_resets, device=device), indices]  # (B, 2)

    z = torch.full((num_resets,), 0.1, device=device)
    yaw = torch.rand(num_resets, device=device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]

    # Convert yaw to quaternion (w, x, y, z)
    quat = torch.zeros(num_resets, 4, device=device)
    quat[:, 0] = torch.cos(yaw / 2)
    quat[:, 3] = torch.sin(yaw / 2)

    root_state = robot.data.default_root_state[env_ids].clone()
    env_origins = env.scene.env_origins[env_ids]
    root_state[:, 0] = env_origins[:, 0] + xy[:, 0]
    root_state[:, 1] = env_origins[:, 1] + xy[:, 1]
    root_state[:, 2] = env_origins[:, 2] + z
    root_state[:, 3:7] = quat
    root_state[:, 7:] = 0.0

    robot.write_root_state_to_sim(root_state, env_ids)


def randomize_proc_room_difficulty(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    max_level: int = 7,
) -> None:
    """Sample uniform random room difficulty per env on episode reset.

    Sets ``env._proc_room_difficulty`` which is read by ``generate_proc_room``
    to control the number of walls, furniture, and clutter placed.

    Args:
        env: The environment instance.
        env_ids: Indices of environments being reset.
        max_level: Maximum difficulty level (inclusive).
    """
    if not hasattr(env, "_proc_room_difficulty"):
        env._proc_room_difficulty = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
    env._proc_room_difficulty[env_ids] = torch.randint(
        0, max_level + 1, (len(env_ids),), device=env.device
    )
