"""Custom observation functions for Strafer navigation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import TiledCamera, Camera


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


def depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    max_depth: float = 6.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Get flattened depth image from D555 camera.
    
    Preprocesses the depth image by:
    1. Replacing infinity values with max_depth
    2. Normalizing to [0, 1] range (optional)
    3. Flattening to 1D for MLP policies
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the camera sensor.
        max_depth: Maximum depth value for normalization and inf replacement.
        normalize: Whether to normalize depth to [0, 1]. Defaults to True.
    
    Returns:
        Flattened depth image. Shape: (num_envs, height * width)
    """
    # Get the camera sensor
    sensor: TiledCamera | Camera = env.scene.sensors[sensor_cfg.name]
    
    # Get depth image: shape (num_envs, height, width, 1)
    depth = sensor.data.output["distance_to_image_plane"].clone()
    
    # Replace inf/nan values with max_depth
    depth = torch.where(
        torch.isinf(depth) | torch.isnan(depth),
        torch.full_like(depth, max_depth),
        depth
    )
    
    # Clamp to valid range
    depth = torch.clamp(depth, 0.0, max_depth)
    
    # Normalize to [0, 1] if requested
    if normalize:
        depth = depth / max_depth
    
    # Flatten: (num_envs, height, width, 1) -> (num_envs, height * width)
    num_envs = depth.shape[0]
    depth_flat = depth.view(num_envs, -1)
    
    return depth_flat


def rgb_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    normalize: bool = True,
) -> torch.Tensor:
    """Get flattened RGB image from D555 camera.
    
    WARNING: RGB observations are very high dimensional (80*60*3 = 14400 dims).
    Consider using a CNN encoder or feature extractor for RGB-based policies.
    
    Preprocesses the RGB image by:
    1. Converting to float [0, 1]
    2. Mean-centering (optional)
    3. Flattening to 1D
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the camera sensor.
        normalize: Whether to normalize and mean-center. Defaults to True.
    
    Returns:
        Flattened RGB image. Shape: (num_envs, height * width * 3)
    """
    # Get the camera sensor
    sensor: TiledCamera | Camera = env.scene.sensors[sensor_cfg.name]
    
    # Get RGB image: shape (num_envs, height, width, 3)
    rgb = sensor.data.output["rgb"].clone()
    
    # Convert to float [0, 1]
    rgb = rgb.float() / 255.0
    
    # Mean-center if normalizing
    if normalize:
        # Compute mean across spatial dimensions (H, W)
        mean = rgb.mean(dim=(1, 2), keepdim=True)
        rgb = rgb - mean
    
    # Flatten: (num_envs, height, width, 3) -> (num_envs, height * width * 3)
    num_envs = rgb.shape[0]
    rgb_flat = rgb.view(num_envs, -1)
    
    return rgb_flat
