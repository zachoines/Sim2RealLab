"""Custom noise models for sim-to-real transfer.

Implements realistic sensor noise models that go beyond simple Gaussian noise:
- IMU: bias drift, temperature effects
- Encoders: quantization, missed ticks
- Cameras: depth-dependent noise, holes, dropped frames

These integrate with Isaac Lab's NoiseModel system and can be used
directly in ObsTerm configurations.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.noise import NoiseModel
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelCfg, GaussianNoiseCfg

if TYPE_CHECKING:
    from .sim_real_cfg import (
        IMUNoiseCfg,
        EncoderNoiseCfg,
        DepthCameraNoiseCfg,
        RGBCameraNoiseCfg,
    )


# =============================================================================
# IMU Noise Model
# =============================================================================

@configclass
class IMUNoiseModelCfg(NoiseModelCfg):
    """Configuration for IMU noise model with bias drift."""
    
    class_type: type = None  # Set below after class definition
    
    # Accelerometer
    accel_noise_std: float = 0.01  # m/s² white noise
    accel_bias_range: tuple[float, float] = (-0.05, 0.05)  # m/s² bias range
    accel_bias_drift_rate: float = 0.001  # m/s² per second drift
    
    # Gyroscope  
    gyro_noise_std: float = 0.001  # rad/s white noise
    gyro_bias_range: tuple[float, float] = (-0.002, 0.002)  # rad/s bias range
    gyro_bias_drift_rate: float = 0.0001  # rad/s per second drift
    
    # Combined output size (3 for accel OR gyro, 6 for both)
    output_size: int = 3


class IMUNoiseModel(NoiseModel):
    """IMU noise model with bias drift.
    
    Models realistic IMU behavior:
    - Additive white Gaussian noise
    - Slowly drifting bias (random walk)
    - Optional temperature-dependent effects
    """
    
    def __init__(self, noise_model_cfg: IMUNoiseModelCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg: IMUNoiseModelCfg = noise_model_cfg
        
        # Initialize bias terms (sampled at reset)
        self._bias = torch.zeros(num_envs, self.cfg.output_size, device=device)
        
        # Initialize with random bias
        self.reset()
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset bias to random values within range."""
        if env_ids is None:
            env_ids = slice(None)
        
        # Sample new bias values
        if self.cfg.output_size == 3:
            # Single sensor (accel or gyro)
            bias_range = self.cfg.accel_bias_range
        else:
            # Combined (not implemented yet)
            bias_range = self.cfg.accel_bias_range
        
        bias_min, bias_max = bias_range
        self._bias[env_ids] = (
            torch.rand(self._get_num_envs(env_ids), self.cfg.output_size, device=self._device) 
            * (bias_max - bias_min) + bias_min
        )
    
    def _get_num_envs(self, env_ids) -> int:
        """Get number of environments from env_ids."""
        if isinstance(env_ids, slice) and env_ids == slice(None):
            return self._num_envs
        return len(env_ids)
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply IMU noise: bias + white noise."""
        # Determine which noise params to use based on data shape
        noise_std = self.cfg.accel_noise_std if data.shape[-1] == 3 else self.cfg.accel_noise_std
        
        # Add bias (slowly drift it)
        drift = torch.randn_like(self._bias) * self.cfg.accel_bias_drift_rate
        self._bias = self._bias + drift
        
        # Clamp bias to reasonable range
        bias_min, bias_max = self.cfg.accel_bias_range
        self._bias = torch.clamp(self._bias, bias_min * 2, bias_max * 2)
        
        # Add white noise + bias
        noise = torch.randn_like(data) * noise_std
        return data + self._bias + noise


# Set class_type after class definition
IMUNoiseModelCfg.class_type = IMUNoiseModel


# =============================================================================
# Encoder Noise Model
# =============================================================================

@configclass
class EncoderNoiseModelCfg(NoiseModelCfg):
    """Configuration for encoder noise model with quantization."""
    
    class_type: type = None  # Set below
    
    # Quantization
    enable_quantization: bool = True
    ticks_per_radian: float = 85.57  # 537.7 PPR / 2π
    
    # Noise
    velocity_noise_std: float = 0.02  # Fraction of max velocity
    max_velocity: float = 5000.0  # Max ticks/sec for normalization
    
    # Errors
    missed_tick_prob: float = 0.001
    extra_tick_prob: float = 0.0005


class EncoderNoiseModel(NoiseModel):
    """Encoder noise model with quantization and tick errors.
    
    Models realistic encoder behavior:
    - Quantization to discrete ticks
    - Random missed/extra ticks
    - Velocity estimation noise
    """
    
    def __init__(self, noise_model_cfg: EncoderNoiseModelCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg: EncoderNoiseModelCfg = noise_model_cfg
        
        # Track accumulated tick errors for position observations
        self._tick_error_accumulator = None
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset tick error accumulator."""
        if self._tick_error_accumulator is not None:
            if env_ids is None:
                self._tick_error_accumulator.zero_()
            else:
                self._tick_error_accumulator[env_ids] = 0.0
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply encoder noise: quantization + tick errors + noise."""
        noisy_data = data.clone()
        
        # Initialize accumulator if needed
        if self._tick_error_accumulator is None:
            self._tick_error_accumulator = torch.zeros_like(data)
        
        # Add velocity noise (Gaussian)
        noise_std = self.cfg.velocity_noise_std * self.cfg.max_velocity
        noisy_data = noisy_data + torch.randn_like(data) * noise_std
        
        # Simulate missed/extra ticks (random discrete errors)
        if self.cfg.missed_tick_prob > 0:
            missed = torch.rand_like(data) < self.cfg.missed_tick_prob
            tick_sign = torch.sign(noisy_data)
            noisy_data = torch.where(missed, noisy_data - tick_sign, noisy_data)
        
        if self.cfg.extra_tick_prob > 0:
            extra = torch.rand_like(data) < self.cfg.extra_tick_prob
            tick_sign = torch.sign(noisy_data)
            noisy_data = torch.where(extra, noisy_data + tick_sign, noisy_data)
        
        # Quantization (round to nearest tick)
        if self.cfg.enable_quantization:
            noisy_data = torch.round(noisy_data)
        
        return noisy_data


EncoderNoiseModelCfg.class_type = EncoderNoiseModel


# =============================================================================
# Depth Camera Noise Model
# =============================================================================

@configclass
class DepthNoiseModelCfg(NoiseModelCfg):
    """Configuration for depth camera noise model."""
    
    class_type: type = None  # Set below
    
    # Depth-dependent noise
    base_noise_std: float = 0.001  # meters at 1m
    depth_noise_coeff: float = 0.002  # additional noise per meter
    
    # Holes (invalid pixels)
    hole_probability: float = 0.01
    
    # Range limits
    min_range: float = 0.2  # meters
    max_range: float = 6.0  # meters
    
    # Frame drops
    frame_drop_prob: float = 0.001
    
    # Image dimensions (for unflattening if needed)
    height: int = 60
    width: int = 80


class DepthNoiseModel(NoiseModel):
    """Depth camera noise model with realistic imperfections.
    
    Models:
    - Depth-dependent Gaussian noise
    - Random invalid pixels (holes)
    - Range limiting
    - Dropped frames
    """
    
    def __init__(self, noise_model_cfg: DepthNoiseModelCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg: DepthNoiseModelCfg = noise_model_cfg
        
        # Store previous frame for drops
        self._prev_frame = None
        self._frame_dropped = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset frame drop state."""
        if env_ids is None:
            self._prev_frame = None
            self._frame_dropped.zero_()
        else:
            self._frame_dropped[env_ids] = False
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply depth camera noise."""
        # Data shape: (num_envs, height * width) - flattened depth
        noisy_data = data.clone()
        
        # Denormalize if normalized (assume [0, 1] range)
        max_range = self.cfg.max_range
        depth_meters = noisy_data * max_range
        
        # Depth-dependent noise: std = base + coeff * depth
        noise_std = self.cfg.base_noise_std + self.cfg.depth_noise_coeff * depth_meters
        depth_noise = torch.randn_like(depth_meters) * noise_std
        depth_meters = depth_meters + depth_noise
        
        # Add holes (random invalid pixels)
        if self.cfg.hole_probability > 0:
            holes = torch.rand_like(depth_meters) < self.cfg.hole_probability
            depth_meters = torch.where(holes, torch.full_like(depth_meters, max_range), depth_meters)
        
        # Apply range limits
        depth_meters = torch.clamp(depth_meters, 0.0, max_range)
        too_close = depth_meters < self.cfg.min_range
        depth_meters = torch.where(too_close, torch.full_like(depth_meters, max_range), depth_meters)
        
        # Renormalize
        noisy_data = depth_meters / max_range
        
        # Frame drops (return previous frame)
        if self.cfg.frame_drop_prob > 0 and self._prev_frame is not None:
            drop = torch.rand(self._num_envs, device=self._device) < self.cfg.frame_drop_prob
            noisy_data[drop] = self._prev_frame[drop]
        
        # Store for next frame
        self._prev_frame = noisy_data.clone()
        
        return noisy_data


DepthNoiseModelCfg.class_type = DepthNoiseModel


# =============================================================================
# RGB Camera Noise Model  
# =============================================================================

@configclass
class RGBNoiseModelCfg(NoiseModelCfg):
    """Configuration for RGB camera noise model."""
    
    class_type: type = None  # Set below
    
    # Pixel noise
    pixel_noise_std: float = 0.02  # Fraction of [0, 1] range
    
    # Brightness variation
    brightness_range: tuple[float, float] = (0.9, 1.1)
    
    # Frame drops
    frame_drop_prob: float = 0.001


class RGBNoiseModel(NoiseModel):
    """RGB camera noise model.
    
    Models:
    - Per-pixel Gaussian noise
    - Random brightness variation
    - Dropped frames
    """
    
    def __init__(self, noise_model_cfg: RGBNoiseModelCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg: RGBNoiseModelCfg = noise_model_cfg
        
        self._prev_frame = None
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset state."""
        if env_ids is None:
            self._prev_frame = None
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply RGB noise."""
        noisy_data = data.clone()
        
        # Add pixel noise
        noise = torch.randn_like(noisy_data) * self.cfg.pixel_noise_std
        noisy_data = noisy_data + noise
        
        # Random brightness variation (per env)
        b_min, b_max = self.cfg.brightness_range
        brightness = torch.rand(self._num_envs, 1, device=self._device) * (b_max - b_min) + b_min
        noisy_data = noisy_data * brightness
        
        # Clamp to valid range
        noisy_data = torch.clamp(noisy_data, -1.0, 1.0)  # Assuming mean-centered
        
        # Frame drops
        if self.cfg.frame_drop_prob > 0 and self._prev_frame is not None:
            drop = torch.rand(self._num_envs, device=self._device) < self.cfg.frame_drop_prob
            noisy_data[drop] = self._prev_frame[drop]
        
        self._prev_frame = noisy_data.clone()
        
        return noisy_data


RGBNoiseModelCfg.class_type = RGBNoiseModel


# =============================================================================
# Factory Functions
# =============================================================================

def create_imu_accel_noise_cfg(
    noise_std: float = 0.01,
    bias_range: tuple[float, float] = (-0.05, 0.05),
    bias_drift_rate: float = 0.001,
) -> IMUNoiseModelCfg:
    """Create IMU accelerometer noise config."""
    return IMUNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=noise_std),
        accel_noise_std=noise_std,
        accel_bias_range=bias_range,
        accel_bias_drift_rate=bias_drift_rate,
        output_size=3,
    )


def create_imu_gyro_noise_cfg(
    noise_std: float = 0.001,
    bias_range: tuple[float, float] = (-0.002, 0.002),
    bias_drift_rate: float = 0.0001,
) -> IMUNoiseModelCfg:
    """Create IMU gyroscope noise config."""
    return IMUNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=noise_std),
        gyro_noise_std=noise_std,
        gyro_bias_range=bias_range,
        gyro_bias_drift_rate=bias_drift_rate,
        output_size=3,
    )


def create_encoder_noise_cfg(
    velocity_noise_std: float = 0.02,
    enable_quantization: bool = True,
    missed_tick_prob: float = 0.001,
) -> EncoderNoiseModelCfg:
    """Create encoder noise config."""
    return EncoderNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=velocity_noise_std),
        enable_quantization=enable_quantization,
        velocity_noise_std=velocity_noise_std,
        missed_tick_prob=missed_tick_prob,
    )


def create_depth_noise_cfg(
    base_noise_std: float = 0.001,
    hole_probability: float = 0.01,
    frame_drop_prob: float = 0.001,
) -> DepthNoiseModelCfg:
    """Create depth camera noise config."""
    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=base_noise_std),
        base_noise_std=base_noise_std,
        hole_probability=hole_probability,
        frame_drop_prob=frame_drop_prob,
    )


def create_rgb_noise_cfg(
    pixel_noise_std: float = 0.02,
    frame_drop_prob: float = 0.001,
) -> RGBNoiseModelCfg:
    """Create RGB camera noise config."""
    return RGBNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=pixel_noise_std),
        pixel_noise_std=pixel_noise_std,
        frame_drop_prob=frame_drop_prob,
    )
