"""Custom observation functions for Strafer navigation task.

Includes realistic sensor models for:
- Intel RealSense D555 IMU (BMI055)
- GoBilda 5203 motor encoders (537.7 PPR)
- D555 RGB/Depth camera
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import TiledCamera, Camera, Imu


# =============================================================================
# GoBilda 5203 Motor Encoder Constants
# =============================================================================

# Encoder resolution formula: ((((1+(46/17))) * (1+(46/11))) * 28)
# = (63/17) * (57/11) * 28 = 537.6897... ≈ 537.7 PPR
ENCODER_PPR_OUTPUT_SHAFT = 537.7  # Pulses per revolution at output shaft
ENCODER_PPR_ENCODER_SHAFT = 28    # Pulses per revolution at encoder shaft

# Conversion factors
RADIANS_TO_ENCODER_TICKS = ENCODER_PPR_OUTPUT_SHAFT / (2.0 * math.pi)  # ~85.57 ticks/rad
ENCODER_TICKS_TO_RADIANS = (2.0 * math.pi) / ENCODER_PPR_OUTPUT_SHAFT  # ~0.01169 rad/tick


# =============================================================================
# Motor Encoder Observations
# =============================================================================

# Wheel drive joint names (must match USD joint names)
# Order: [wheel_1, wheel_2, wheel_3, wheel_4] = [FL, FR, RL, RR]
WHEEL_JOINT_NAMES = ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]


def _get_wheel_joint_indices(robot) -> list[int]:
    """Get indices of wheel drive joints from robot articulation.
    
    Caches the indices after first lookup for efficiency.
    """
    if not hasattr(robot, "_wheel_joint_indices"):
        indices = []
        for name in WHEEL_JOINT_NAMES:
            try:
                idx = robot.joint_names.index(name)
                indices.append(idx)
            except ValueError:
                # Joint not found - this is an error
                raise ValueError(f"Wheel joint '{name}' not found in robot. Available: {robot.joint_names}")
        robot._wheel_joint_indices = indices
    return robot._wheel_joint_indices


def wheel_encoder_positions(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg | None = None,
    normalize: bool = False,
    max_ticks: float = 10000.0,
) -> torch.Tensor:
    """Get wheel encoder positions in ticks (537.7 PPR at output shaft).
    
    Models the GoBilda 5203 series motor encoders which provide:
    - Quadrature encoding (A/B channels)
    - Hall effect (magnetic) sensors
    - 28 PPR at encoder shaft, 537.7 PPR at gearbox output
    
    The encoder counts are cumulative and can be positive or negative
    depending on rotation direction. In practice, these would wrap at
    16-bit or 32-bit limits, but we keep them as continuous floats.
    
    Args:
        env: The environment instance.
        asset_cfg: Asset configuration for the robot. Defaults to "robot".
        normalize: Whether to normalize ticks to [-1, 1] range.
        max_ticks: Max ticks for normalization (default 10000 = ~18.6 revs).
    
    Returns:
        Encoder positions in ticks. Shape: (num_envs, 4)
        Order: [wheel_1, wheel_2, wheel_3, wheel_4] = [FL, FR, RL, RR]
    """
    asset_name = asset_cfg.name if asset_cfg is not None else "robot"
    robot = env.scene[asset_name]
    
    # Get indices for wheel joints only
    wheel_indices = _get_wheel_joint_indices(robot)
    
    # Get joint positions for wheel joints only
    joint_pos = robot.data.joint_pos[:, wheel_indices]  # Shape: (num_envs, 4)
    
    # Convert radians to encoder ticks
    encoder_ticks = joint_pos * RADIANS_TO_ENCODER_TICKS
    
    if normalize:
        encoder_ticks = torch.clamp(encoder_ticks / max_ticks, -1.0, 1.0)
    
    return encoder_ticks


def wheel_encoder_velocities(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Get wheel encoder velocities in ticks per second (RAW).
    
    This represents what you'd compute from encoder tick deltas:
    velocity = (current_ticks - previous_ticks) / dt
    
    At 312 RPM (max motor speed): 312/60 * 537.7 = 2796 ticks/sec
    
    Note: Returns RAW values. Normalization should be handled via the
    ObservationTermCfg.scale parameter so that noise is applied to
    raw sensor values (physically correct).
    
    Args:
        env: The environment instance.
        asset_cfg: Asset configuration for the robot. Defaults to "robot".
    
    Returns:
        Encoder velocities in ticks/sec. Shape: (num_envs, 4)
    """
    asset_name = asset_cfg.name if asset_cfg is not None else "robot"
    robot = env.scene[asset_name]
    
    # Get indices for wheel joints only
    wheel_indices = _get_wheel_joint_indices(robot)
    
    # Get joint velocities for wheel joints only
    joint_vel = robot.data.joint_vel[:, wheel_indices]  # Shape: (num_envs, 4)
    
    # Convert rad/s to ticks/s (RAW - no normalization)
    encoder_vel = joint_vel * RADIANS_TO_ENCODER_TICKS
    
    return encoder_vel


def wheel_encoder_deltas(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg | None = None,
    normalize: bool = True,
    max_delta: float = 200.0,
) -> torch.Tensor:
    """Get wheel encoder tick deltas (change since last step).
    
    This is what you'd directly measure from a quadrature encoder
    between consecutive readings. At 30 Hz control and 312 RPM:
    max_delta ≈ 2796 / 30 ≈ 93 ticks/step
    
    Args:
        env: The environment instance.
        asset_cfg: Asset configuration for the robot. Defaults to "robot".
        normalize: Whether to normalize to [-1, 1] range.
        max_delta: Max delta for normalization (default 200 ticks).
    
    Returns:
        Encoder tick deltas. Shape: (num_envs, 4)
    """
    asset_name = asset_cfg.name if asset_cfg is not None else "robot"
    robot = env.scene[asset_name]
    
    # Get indices for wheel joints only
    wheel_indices = _get_wheel_joint_indices(robot)
    
    # Get joint velocities for wheel joints and multiply by dt
    joint_vel = robot.data.joint_vel[:, wheel_indices]
    dt = env.step_dt  # Environment step dt
    
    # Delta ticks = velocity * dt * conversion
    delta_ticks = joint_vel * dt * RADIANS_TO_ENCODER_TICKS
    
    if normalize:
        delta_ticks = torch.clamp(delta_ticks / max_delta, -1.0, 1.0)
    
    return delta_ticks


# =============================================================================
# IMU Observations (D555 BMI055)
# =============================================================================


def imu_angular_velocity(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get angular velocity from D555 IMU (gyroscope) in rad/s (RAW).
    
    Models the BMI055 gyroscope in the Intel RealSense D555:
    - Range: ±125/±250/±500/±1000/±2000 °/s (we use ±2000 °/s = 34.9 rad/s)
    - Output: 3-axis angular velocity in sensor frame
    
    Note: Returns RAW values in rad/s. Normalization should be handled via
    the ObservationTermCfg.scale parameter so that noise is applied to
    raw sensor values (physically correct).
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the IMU sensor.
    
    Returns:
        Angular velocity (roll, pitch, yaw rate) in rad/s. Shape: (num_envs, 3)
    """
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    ang_vel = sensor.data.ang_vel_b.clone()
    
    return ang_vel


def imu_linear_acceleration(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get linear acceleration from D555 IMU (accelerometer) in m/s² (RAW).
    
    Models the BMI055 accelerometer in the Intel RealSense D555:
    - Range: ±2g/±4g/±8g/±16g (we use ±16g = 156.96 m/s²)
    - Output: 3-axis linear acceleration including gravity
    - When stationary and level: reads (0, 0, +9.81) m/s²
    
    This replaces the separate base_lin_vel and projected_gravity
    observations with a single realistic IMU measurement.
    
    Note: Returns RAW values in m/s². Normalization should be handled via
    the ObservationTermCfg.scale parameter so that noise is applied to
    raw sensor values (physically correct).
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the IMU sensor.
    
    Returns:
        Linear acceleration (ax, ay, az) in m/s². Shape: (num_envs, 3)
    """
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    lin_acc = sensor.data.lin_acc_b.clone()
    
    return lin_acc


def imu_orientation(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get orientation quaternion from IMU.
    
    Note: Raw IMUs don't directly output orientation - this would typically
    come from sensor fusion (complementary filter, Kalman filter, etc.).
    The D555 can provide this through its built-in IMU fusion.
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the IMU sensor.
    
    Returns:
        Orientation quaternion (w, x, y, z). Shape: (num_envs, 4)
    """
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    return sensor.data.quat_w.clone()


def imu_projected_gravity(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get gravity vector projected into IMU frame.
    
    Derived from accelerometer when stationary or from sensor fusion.
    Useful for tilt detection.
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the IMU sensor.
    
    Returns:
        Projected gravity unit vector. Shape: (num_envs, 3)
    """
    sensor: Imu = env.scene.sensors[sensor_cfg.name]
    return sensor.data.projected_gravity_b.clone()





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
) -> torch.Tensor:
    """Get flattened depth image from D555 camera in RAW meters.
    
    Returns RAW depth in meters so that:
    1. Noise is applied to physical depth values (enables depth-dependent noise)
    2. Normalization happens via ObsTermCfg.scale parameter (Isaac Lab standard)
    
    Pipeline: RAW meters → noise → scale (1/max_depth) → [0, 1] output
    
    Preprocesses the depth image by:
    1. Replacing infinity/NaN values with max_depth
    2. Clamping to [0, max_depth] range
    3. Flattening to 1D for MLP policies
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the camera sensor.
        max_depth: Maximum depth value for inf replacement and clamping.
    
    Returns:
        Flattened depth image in meters. Shape: (num_envs, height * width)
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
    
    # Clamp to valid range (in meters)
    depth = torch.clamp(depth, 0.0, max_depth)
    
    # Flatten: (num_envs, height, width, 1) -> (num_envs, height * width)
    num_envs = depth.shape[0]
    depth_flat = depth.view(num_envs, -1)
    
    return depth_flat


def rgb_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get flattened RGB image from D555 camera in [0, 1] range.
    
    Returns RGB as float [0, 1] so that:
    1. Noise is applied to normalized pixel values
    2. Output is already bounded, no additional scale needed
    
    WARNING: RGB observations are very high dimensional (80*60*3 = 14400 dims).
    Consider using a CNN encoder or feature extractor for RGB-based policies.
    
    Preprocesses the RGB image by:
    1. Converting uint8 [0, 255] to float [0, 1]
    2. Flattening to 1D
    
    Note: Mean-centering is NOT done here. If needed for your network,
    apply it as a separate preprocessing step or modifier.
    
    Args:
        env: The environment instance.
        sensor_cfg: Scene entity configuration for the camera sensor.
    
    Returns:
        Flattened RGB image in [0, 1]. Shape: (num_envs, height * width * 3)
    """
    # Get the camera sensor
    sensor: TiledCamera | Camera = env.scene.sensors[sensor_cfg.name]
    
    # Get RGB image: shape (num_envs, height, width, 3)
    rgb = sensor.data.output["rgb"].clone()
    
    # Convert uint8 [0, 255] to float [0, 1]
    rgb = rgb.float() / 255.0
    
    # Flatten: (num_envs, height, width, 3) -> (num_envs, height * width * 3)
    num_envs = rgb.shape[0]
    rgb_flat = rgb.view(num_envs, -1)
    
    return rgb_flat
