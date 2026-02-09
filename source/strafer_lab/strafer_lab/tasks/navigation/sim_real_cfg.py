"""Sim-to-Real Configuration Contract for Strafer Robot.

This module defines configurable abstraction layers that model real-world
imperfections for robust sim-to-real transfer:

1. TIMING & LATENCY
   - Command rate / control frequency
   - Sensor observation delays (buffering)
   - Action command delays (network/driver lag)

2. ACTUATION MODEL
   - Motor response dynamics (first-order lag)
   - Command delay buffers
   - Velocity/torque limits and slew rates

3. SENSOR NOISE & FAILURES
   - IMU: bias drift, gaussian noise, temperature effects
   - Encoders: quantization, missed ticks, electrical noise
   - Depth camera: holes, noise, range limits, dropped frames
   - RGB camera: motion blur, exposure variation

Usage:
    from strafer_lab.tasks.navigation.sim_real_cfg import (
        SimRealContractCfg,
        REAL_ROBOT_CONTRACT,
        IDEAL_SIM_CONTRACT,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    UniformNoiseCfg,
    NoiseModelCfg,
    NoiseModelWithAdditiveBiasCfg,
)

# Import custom noise models that generate independent per-environment noise
from strafer_lab.tasks.navigation.mdp.noise_models import (
    IMUNoiseModelCfg,
    EncoderNoiseModelCfg,
    DepthNoiseModelCfg,
    RGBNoiseModelCfg,
)


# =============================================================================
# Timing & Latency Configuration
# =============================================================================

@configclass
class TimingCfg:
    """Timing and latency configuration for sim-to-real transfer.
    
    Models real-world timing imperfections:
    - Control loop frequency and jitter
    - Sensor data latency (observation delay)
    - Command latency (action delay)
    """
    
    # Control frequency
    control_frequency_hz: float = 30.0
    """Target control loop frequency in Hz. Default: 30 Hz (matching ROS2 node rate)."""
    
    control_frequency_jitter_pct: float = 0.0
    """Random jitter in control frequency as percentage. 0.1 = ±10% variation."""
    
    # Observation latency (sensor → policy)
    obs_latency_steps: int = 0
    """Fixed observation latency in control steps. 1 step @ 30Hz = 33ms."""
    
    obs_latency_steps_range: tuple[int, int] = (0, 0)
    """Random observation latency range [min, max] steps. Sampled per reset."""
    
    # Action latency (policy → actuator)
    action_latency_steps: int = 0
    """Fixed action command latency in physics steps."""
    
    action_latency_steps_range: tuple[int, int] = (0, 0)
    """Random action latency range [min, max] steps. Sampled per reset."""


# =============================================================================
# Actuator Model Configuration
# =============================================================================

@configclass
class ActuatorModelCfg:
    """Actuator dynamics configuration for GoBilda 5203 motors.
    
    Models real motor behavior:
    - First-order response dynamics (motor inertia + driver response)
    - Velocity and acceleration limits
    """
    
    # Motor response dynamics (first-order system)
    enable_motor_dynamics: bool = False
    """Enable first-order motor response model (exponential smoothing)."""
    
    motor_time_constant_s: float = 0.05
    """Motor time constant in seconds. Controls response speed. 
    Real GoBilda 5203 ~50ms under load."""
    
    motor_time_constant_range: tuple[float, float] = (0.03, 0.08)
    """Range for randomized motor time constant [min, max] seconds."""
    
    # Velocity limits
    max_velocity_rad_s: float = 32.67
    """Maximum motor velocity in rad/s. GoBilda 5203 = 312 RPM = 32.67 rad/s."""
    
    # Acceleration limits (slew rate)
    max_acceleration_rad_s2: float = 100.0
    """Maximum acceleration for velocity commands. Prevents instant velocity changes."""


# =============================================================================
# Sensor Noise Configuration
# =============================================================================

@configclass
class IMUNoiseCfg:
    """IMU sensor noise configuration for D555 (BMI055).
    
    Based on BMI055 datasheet specifications:
    - Accelerometer: ±16g range, 0.98 mg/√Hz noise density
    - Gyroscope: ±2000°/s range, 0.014 °/s/√Hz noise density
    """
    
    enable_noise: bool = True
    """Enable IMU noise injection."""
    
    # Accelerometer noise
    accel_noise_density: float = 0.0098  # m/s² per √Hz (0.98 mg/√Hz)
    """Accelerometer white noise density. BMI055 typical: 0.98 mg/√Hz."""
    
    accel_bias_stability: float = 0.04  # m/s² (40 μg)
    """Accelerometer bias instability (drift over time)."""
    
    accel_random_walk: float = 0.001
    """Accelerometer random walk (bias drift rate)."""
    
    # Gyroscope noise
    gyro_noise_density: float = 0.00024  # rad/s per √Hz (0.014 °/s/√Hz)
    """Gyroscope white noise density. BMI055 typical: 0.014 °/s/√Hz."""
    
    gyro_bias_stability: float = 0.0017  # rad/s (0.1 °/s)
    """Gyroscope bias instability."""
    
    gyro_random_walk: float = 0.0001
    """Gyroscope random walk (bias drift rate)."""
    
    # Temperature effects (optional)
    enable_temp_effects: bool = False
    """Enable temperature-dependent bias drift."""
    
    temp_coefficient: float = 0.015
    """Temperature coefficient for bias drift (per °C from 25°C)."""


@configclass
class EncoderNoiseCfg:
    """Encoder noise configuration for GoBilda 5203 (Hall effect).
    
    Models real encoder imperfections:
    - Quantization (discrete tick counts)
    - Electrical noise / missed ticks
    - Velocity estimation noise
    """
    
    enable_noise: bool = True
    """Enable encoder noise injection."""
    
    # Quantization (inherent in tick counting)
    enable_quantization: bool = True
    """Quantize positions to discrete encoder ticks."""
    
    # Tick counting errors
    missed_tick_probability: float = 0.001
    """Probability of missing a tick (electrical noise). Per tick per step."""
    
    extra_tick_probability: float = 0.0005
    """Probability of counting an extra tick (electrical noise)."""
    
    # Velocity estimation noise
    velocity_noise_std: float = 0.02
    """Gaussian noise on velocity as fraction of max velocity."""
    
    # Velocity quantization (from discrete position differencing)
    velocity_quantization_ticks_per_step: float = 1.0
    """Minimum detectable velocity change in ticks per control step."""


@configclass
class DepthCameraNoiseCfg:
    """Depth camera noise configuration for Intel RealSense D555.

    Uses the Intel RealSense stereo depth error propagation model:
        σ_z = (z² / (f · B)) · σ_d

    Where:
        z = depth in meters
        f = focal length in pixels (at native resolution)
        B = stereo baseline in meters
        σ_d = subpixel disparity noise in pixels

    This quadratic z² relationship matches real RealSense behavior.

    Reference: Intel RealSense documentation on depth quality and error propagation
    https://openaccess.thecvf.com/content_cvpr_2017_workshops/w15/papers/Keselman_Intel_RealSense_Stereoscopic_CVPR_2017_paper.pdf
    """

    enable_noise: bool = True
    """Enable depth camera noise injection."""

    # Intel RealSense D555 stereo parameters
    baseline_m: float = 0.095
    """Stereo baseline in meters (95mm for D555)."""

    focal_length_px: float = 673.0
    """Focal length in pixels at native 1280x720 resolution."""

    disparity_noise_px: float = 0.08
    """Subpixel disparity noise (typical: 0.05-0.1 pixels)."""

    # Invalid pixels (holes from stereo matching failures)
    hole_probability: float = 0.01
    """Probability of invalid pixel (set to max_depth)."""

    hole_cluster_size: int = 3
    """Average size of hole clusters in pixels."""

    # Range limits
    min_range_m: float = 0.2
    """Minimum valid depth range in meters. Closer = invalid."""

    max_range_m: float = 6.0
    """Maximum valid depth range in meters. Further = invalid."""

    # Dropped frames
    frame_drop_probability: float = 0.001
    """Probability of dropping a frame (return previous frame)."""

    # Temporal noise (flickering)
    enable_temporal_noise: bool = False
    """Enable frame-to-frame temporal noise (flickering)."""
    
    temporal_noise_std: float = 0.005
    """Temporal noise standard deviation in meters."""


@configclass
class RGBCameraNoiseCfg:
    """RGB camera noise configuration for D555.
    
    Models real RGB camera imperfections:
    - Sensor noise
    - Motion blur
    - Exposure variations
    - Dropped frames
    """
    
    enable_noise: bool = True
    """Enable RGB camera noise injection."""
    
    # Sensor noise
    pixel_noise_std: float = 0.02
    """Per-pixel Gaussian noise as fraction of [0,1] range."""
    
    # Motion blur (simplified as Gaussian blur based on velocity)
    enable_motion_blur: bool = False
    """Enable velocity-dependent motion blur."""
    
    motion_blur_strength: float = 0.1
    """Motion blur kernel size factor."""
    
    # Exposure variations
    enable_exposure_variation: bool = False
    """Enable random exposure/brightness variations."""
    
    exposure_variation_range: tuple[float, float] = (0.9, 1.1)
    """Brightness multiplier range [min, max]."""
    
    # Dropped frames
    frame_drop_probability: float = 0.001
    """Probability of dropping a frame."""


@configclass 
class SensorFailureCfg:
    """Sensor failure modes for robustness training.
    
    Simulates catastrophic sensor failures to train robust policies.
    """
    
    enable_failures: bool = False
    """Enable random sensor failures."""
    
    # IMU failures
    imu_failure_probability: float = 0.0001
    """Probability of IMU failure per step (returns zeros)."""
    
    imu_stuck_probability: float = 0.0001
    """Probability of IMU getting stuck (returns last value)."""
    
    # Encoder failures
    encoder_failure_probability: float = 0.0001
    """Probability of encoder failure per step (returns zeros)."""
    
    # Camera failures
    camera_failure_probability: float = 0.001
    """Probability of camera failure (returns black/max depth)."""


# =============================================================================
# Combined Sensor Noise Configuration
# =============================================================================

@configclass
class SensorNoiseCfg:
    """Combined sensor noise configuration."""
    
    imu: IMUNoiseCfg = IMUNoiseCfg()
    """IMU noise configuration."""
    
    encoders: EncoderNoiseCfg = EncoderNoiseCfg()
    """Encoder noise configuration."""
    
    depth_camera: DepthCameraNoiseCfg = DepthCameraNoiseCfg()
    """Depth camera noise configuration."""
    
    rgb_camera: RGBCameraNoiseCfg = RGBCameraNoiseCfg()
    """RGB camera noise configuration."""
    
    failures: SensorFailureCfg = SensorFailureCfg()
    """Sensor failure modes configuration."""


# =============================================================================
# Complete Sim-Real Contract
# =============================================================================

@configclass
class SimRealContractCfg:
    """Complete sim-to-real contract configuration.
    
    Bundles all abstraction layers into a single configuration
    that defines the "reality gap" to be bridged.
    
    Example:
        # Use realistic settings for training
        contract = REAL_ROBOT_CONTRACT
        
        # Or customize specific aspects
        contract = SimRealContractCfg(
            timing=TimingCfg(obs_latency_steps=1),
            sensors=SensorNoiseCfg(imu=IMUNoiseCfg(enable_noise=True)),
        )
    """
    
    timing: TimingCfg = TimingCfg()
    """Timing and latency configuration."""
    
    actuator: ActuatorModelCfg = ActuatorModelCfg()
    """Actuator dynamics configuration."""
    
    sensors: SensorNoiseCfg = SensorNoiseCfg()
    """Sensor noise configuration."""
    
    # Domain randomization scale
    domain_randomization_scale: float = 1.0
    """Scale factor for all domain randomization. 0.0 = none, 1.0 = full."""


# =============================================================================
# Preset Configurations
# =============================================================================

def create_ideal_contract() -> SimRealContractCfg:
    """Create ideal simulation contract with no noise or delays.
    
    Use for debugging, visualization, and baseline comparisons.
    """
    return SimRealContractCfg(
        timing=TimingCfg(
            obs_latency_steps=0,
            action_latency_steps=0,
        ),
        actuator=ActuatorModelCfg(
            enable_motor_dynamics=False,
            max_acceleration_rad_s2=float("inf"),
        ),
        sensors=SensorNoiseCfg(
            imu=IMUNoiseCfg(enable_noise=False),
            encoders=EncoderNoiseCfg(enable_noise=False),
            depth_camera=DepthCameraNoiseCfg(enable_noise=False),
            rgb_camera=RGBCameraNoiseCfg(enable_noise=False),
            failures=SensorFailureCfg(enable_failures=False),
        ),
        domain_randomization_scale=0.0,
    )


def create_real_robot_contract() -> SimRealContractCfg:
    """Create realistic contract matching real Strafer robot.
    
    Use for training policies intended for real-world deployment.
    Includes realistic noise, delays, and failure modes.
    """
    return SimRealContractCfg(
        timing=TimingCfg(
            control_frequency_hz=30.0,
            control_frequency_jitter_pct=0.05,  # ±5% jitter
            obs_latency_steps=1,  # 33ms sensor delay
            obs_latency_steps_range=(0, 2),  # 0-66ms random
            action_latency_steps=1,  # 33ms command delay
            action_latency_steps_range=(0, 2),  # 0-66ms random
        ),
        actuator=ActuatorModelCfg(
            enable_motor_dynamics=True,
            motor_time_constant_s=0.05,  # 50ms response
            motor_time_constant_range=(0.03, 0.08),
            max_velocity_rad_s=32.67,  # 312 RPM
            max_acceleration_rad_s2=100.0,
        ),
        sensors=SensorNoiseCfg(
            imu=IMUNoiseCfg(
                enable_noise=True,
                accel_noise_density=0.0098,
                accel_bias_stability=0.04,
                gyro_noise_density=0.00024,
                gyro_bias_stability=0.0017,
            ),
            encoders=EncoderNoiseCfg(
                enable_noise=True,
                enable_quantization=True,
                missed_tick_probability=0.001,
                velocity_noise_std=0.02,
            ),
            depth_camera=DepthCameraNoiseCfg(
                enable_noise=True,
                # Intel D555 stereo parameters (default values)
                baseline_m=0.095,
                focal_length_px=673.0,
                disparity_noise_px=0.08,
                hole_probability=0.01,
                frame_drop_probability=0.001,
            ),
            rgb_camera=RGBCameraNoiseCfg(
                enable_noise=True,
                pixel_noise_std=0.02,
                frame_drop_probability=0.001,
            ),
            failures=SensorFailureCfg(enable_failures=False),
        ),
        domain_randomization_scale=1.0,
    )


def create_robust_training_contract() -> SimRealContractCfg:
    """Create aggressive contract for training robust policies.
    
    Includes higher noise levels and occasional sensor failures
    to train policies that handle worst-case scenarios.
    """
    return SimRealContractCfg(
        timing=TimingCfg(
            control_frequency_hz=30.0,
            control_frequency_jitter_pct=0.10,  # ±10% jitter
            obs_latency_steps=1,
            obs_latency_steps_range=(0, 3),  # Up to 100ms random
            action_latency_steps=1,
            action_latency_steps_range=(0, 4),  # Up to 133ms random
        ),
        actuator=ActuatorModelCfg(
            enable_motor_dynamics=True,
            motor_time_constant_s=0.06,  # Slightly slower
            motor_time_constant_range=(0.02, 0.10),  # Wide range
            max_acceleration_rad_s2=80.0,
        ),
        sensors=SensorNoiseCfg(
            imu=IMUNoiseCfg(
                enable_noise=True,
                accel_noise_density=0.015,  # 1.5x typical
                accel_bias_stability=0.06,
                gyro_noise_density=0.00036,  # 1.5x typical
                gyro_bias_stability=0.0025,
            ),
            encoders=EncoderNoiseCfg(
                enable_noise=True,
                enable_quantization=True,
                missed_tick_probability=0.005,  # 5x typical
                velocity_noise_std=0.05,  # 2.5x typical
            ),
            depth_camera=DepthCameraNoiseCfg(
                enable_noise=True,
                # Intel D555 stereo parameters with increased disparity noise
                baseline_m=0.095,
                focal_length_px=673.0,
                disparity_noise_px=0.16,  # 2x typical for robust training
                hole_probability=0.03,  # 3x typical
                frame_drop_probability=0.01,  # 10x typical
            ),
            rgb_camera=RGBCameraNoiseCfg(
                enable_noise=True,
                pixel_noise_std=0.05,  # 2.5x typical
                frame_drop_probability=0.01,
            ),
            failures=SensorFailureCfg(
                enable_failures=True,
                imu_failure_probability=0.0001,
                encoder_failure_probability=0.0001,
                camera_failure_probability=0.001,
            ),
        ),
        domain_randomization_scale=1.5,  # Extra randomization
    )


# Convenient preset instances
IDEAL_SIM_CONTRACT = create_ideal_contract()
REAL_ROBOT_CONTRACT = create_real_robot_contract()
ROBUST_TRAINING_CONTRACT = create_robust_training_contract()


# =============================================================================
# Contract-to-Config Helpers
# =============================================================================

def get_imu_accel_noise(contract: SimRealContractCfg) -> IMUNoiseModelCfg | None:
    """Get accelerometer noise config from contract.
    
    Returns IMUNoiseModelCfg which generates independent noise per environment.
    Noise is in RAW units (m/s²) - normalization happens via ObsTerm.scale.
    
    Noise density conversion:
        noise_density [unit/√Hz] → std [unit/sample] = density * √(sample_rate_hz)
    """
    if not contract.sensors.imu.enable_noise:
        return None
    # Convert noise density to std using actual control frequency
    # noise_density [m/s²/√Hz] → std [m/s²/sample] = density * √(sample_rate_hz)
    import math
    sample_rate = contract.timing.control_frequency_hz
    std = contract.sensors.imu.accel_noise_density * math.sqrt(sample_rate)
    return IMUNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=std),
        sensor_type='accel',
        control_frequency_hz=sample_rate,
        accel_noise_std=std,
        accel_bias_range=(-contract.sensors.imu.accel_bias_stability, 
                          contract.sensors.imu.accel_bias_stability),
        accel_bias_drift_rate=contract.sensors.imu.accel_random_walk,
        output_size=3,
    )


def get_imu_gyro_noise(contract: SimRealContractCfg) -> IMUNoiseModelCfg | None:
    """Get gyroscope noise config from contract.
    
    Returns IMUNoiseModelCfg which generates independent noise per environment.
    Noise is in RAW units (rad/s) - normalization happens via ObsTerm.scale.
    
    Noise density conversion:
        noise_density [unit/√Hz] → std [unit/sample] = density * √(sample_rate_hz)
    """
    if not contract.sensors.imu.enable_noise:
        return None
    # Convert noise density to std using actual control frequency
    # noise_density [rad/s/√Hz] → std [rad/s/sample] = density * √(sample_rate_hz)
    import math
    sample_rate = contract.timing.control_frequency_hz
    std = contract.sensors.imu.gyro_noise_density * math.sqrt(sample_rate)
    return IMUNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=std),
        sensor_type='gyro',
        control_frequency_hz=sample_rate,
        gyro_noise_std=std,
        gyro_bias_range=(-contract.sensors.imu.gyro_bias_stability,
                         contract.sensors.imu.gyro_bias_stability),
        gyro_bias_drift_rate=contract.sensors.imu.gyro_random_walk,
        output_size=3,
    )


def get_encoder_noise(contract: SimRealContractCfg) -> EncoderNoiseModelCfg | None:
    """Get encoder velocity noise config from contract.
    
    Returns EncoderNoiseModelCfg which generates independent noise per environment,
    and includes quantization and tick error simulation.
    """
    if not contract.sensors.encoders.enable_noise:
        return None
    return EncoderNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=contract.sensors.encoders.velocity_noise_std),
        enable_quantization=contract.sensors.encoders.enable_quantization,
        velocity_noise_std=contract.sensors.encoders.velocity_noise_std,
        missed_tick_prob=contract.sensors.encoders.missed_tick_probability,
        extra_tick_prob=contract.sensors.encoders.extra_tick_probability,
    )


def get_depth_noise(contract: SimRealContractCfg) -> DepthNoiseModelCfg | None:
    """Get depth camera noise config from contract.

    Returns DepthNoiseModelCfg using Intel RealSense stereo error propagation:
        σ_z = (z² / (f · B)) · σ_d

    Includes depth-dependent noise, holes, and frame drops.
    """
    if not contract.sensors.depth_camera.enable_noise:
        return None

    # Compute noise at 1m for informational GaussianNoiseCfg
    cfg = contract.sensors.depth_camera
    noise_at_1m = cfg.disparity_noise_px / (cfg.focal_length_px * cfg.baseline_m)

    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=noise_at_1m),
        baseline_m=cfg.baseline_m,
        focal_length_px=cfg.focal_length_px,
        disparity_noise_px=cfg.disparity_noise_px,
        hole_probability=cfg.hole_probability,
        min_range=cfg.min_range_m,
        max_range=cfg.max_range_m,
        frame_drop_prob=cfg.frame_drop_probability,
    )


def get_rgb_noise(contract: SimRealContractCfg) -> RGBNoiseModelCfg | None:
    """Get RGB camera noise config from contract.
    
    Returns RGBNoiseModelCfg which generates independent noise per environment.
    """
    if not contract.sensors.rgb_camera.enable_noise:
        return None
    return RGBNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=contract.sensors.rgb_camera.pixel_noise_std),
        pixel_noise_std=contract.sensors.rgb_camera.pixel_noise_std,
        brightness_range=contract.sensors.rgb_camera.exposure_variation_range,
        frame_drop_prob=contract.sensors.rgb_camera.frame_drop_probability,
    )


def get_action_config_params(contract: SimRealContractCfg) -> dict:
    """Get action config parameters from contract.
    
    Returns dict to spread into MecanumWheelActionCfg.
    """
    import math
    base_delay = max(contract.timing.action_latency_steps, 0)
    range_min, range_max = contract.timing.action_latency_steps_range
    min_delay = max(0, base_delay + min(range_min, range_max))
    max_delay = max(0, base_delay + max(range_min, range_max))
    motor_rpm = contract.actuator.max_velocity_rad_s * 60.0 / (2.0 * math.pi)
    return {
        "motor_rpm": motor_rpm,
        "max_wheel_angular_vel": contract.actuator.max_velocity_rad_s,
        "enable_motor_dynamics": contract.actuator.enable_motor_dynamics,
        "motor_time_constant": contract.actuator.motor_time_constant_s,
        "min_delay_steps": min_delay,
        "max_delay_steps": max_delay,
        "max_acceleration_rad_s2": contract.actuator.max_acceleration_rad_s2,
    }
