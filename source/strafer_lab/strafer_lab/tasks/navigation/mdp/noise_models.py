"""Custom noise models for sim-to-real transfer.

Implements realistic sensor noise models that go beyond simple Gaussian noise:
- IMU: bias drift, temperature effects
- Encoders: quantization, missed ticks
- Cameras: depth-dependent noise, holes, dropped frames

These integrate with Isaac Lab's NoiseModel system and can be used
directly in ObsTerm configurations.

IMPORTANT: Each NoiseModel class is defined BEFORE its corresponding Cfg class
so that class_type can reference it directly. This ensures the class_type
attribute is properly set when the Cfg is instantiated.
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

class IMUNoiseModel(NoiseModel):
    """IMU noise model with bias drift.
    
    Models realistic IMU behavior:
    - Additive white Gaussian noise
    - Slowly drifting bias (random walk)
    - Optional temperature-dependent effects
    
    The model uses sensor_type to determine which parameters to use:
    - "accel": uses accel_noise_std, accel_bias_range, accel_bias_drift_rate
    - "gyro": uses gyro_noise_std, gyro_bias_range, gyro_bias_drift_rate
    """
    
    def __init__(self, noise_model_cfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg = noise_model_cfg
        
        # Time step for drift scaling (drift rates are specified per-second)
        # For random walk: per-step std = per-second std * sqrt(dt)
        self._dt = 1.0 / self.cfg.control_frequency_hz
        self._sqrt_dt = self._dt ** 0.5
        
        # Determine which sensor parameters to use
        self._sensor_type = self.cfg.sensor_type
        if self._sensor_type == 'accel':
            self._noise_std = self.cfg.accel_noise_std
            self._bias_range = self.cfg.accel_bias_range
            self._drift_rate = self.cfg.accel_bias_drift_rate
        elif self._sensor_type == 'gyro':
            self._noise_std = self.cfg.gyro_noise_std
            self._bias_range = self.cfg.gyro_bias_range
            self._drift_rate = self.cfg.gyro_bias_drift_rate
        else:
            raise ValueError(f"sensor_type must be 'accel' or 'gyro', got '{self._sensor_type}'")
        
        # Initialize bias terms (sampled at reset)
        self._bias = torch.zeros(num_envs, self.cfg.output_size, device=device)
        
        # Initialize with random bias
        self.reset()
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset bias to random values within range."""
        if env_ids is None:
            env_ids = slice(None)
        
        # Sample new bias values from configured range
        bias_min, bias_max = self._bias_range
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
        """Apply IMU noise: bias + white noise.
        
        The drift rate is scaled by sqrt(dt) to properly model a random walk
        where drift_rate is specified per-second.
        """
        # Add bias drift (random walk)
        # Per-step drift std = per-second drift rate * sqrt(dt)
        drift = torch.randn_like(self._bias) * self._drift_rate * self._sqrt_dt
        self._bias = self._bias + drift
        
        # Clamp bias to 2x the configured range
        bias_min, bias_max = self._bias_range
        self._bias = torch.clamp(self._bias, bias_min * 2, bias_max * 2)
        
        # Add white noise + bias
        noise = torch.randn_like(data) * self._noise_std
        return data + self._bias + noise


@configclass
class IMUNoiseModelCfg(NoiseModelCfg):
    """Configuration for IMU noise model with bias drift.
    
    Set sensor_type='accel' or 'gyro' to control which parameters are used.
    
    The drift rate parameters (accel_bias_drift_rate, gyro_bias_drift_rate) are
    specified in units per second. The model internally converts to per-step
    values using control_frequency_hz.
    """
    
    # Set class_type directly to the NoiseModel class (defined above)
    class_type: type = IMUNoiseModel
    
    # Sensor type: 'accel' or 'gyro' (determines which parameters to use)
    sensor_type: str = 'accel'
    
    # Control frequency for time-scaling drift (drift rates are per-second)
    control_frequency_hz: float = 30.0
    
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


# =============================================================================
# Encoder Noise Model
# =============================================================================

class EncoderNoiseModel(NoiseModel):
    """Encoder noise model with quantization and tick errors.
    
    Models realistic encoder behavior:
    - Quantization to discrete ticks
    - Random missed/extra ticks
    - Velocity estimation noise
    """
    
    def __init__(self, noise_model_cfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg = noise_model_cfg
        
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
        """Apply encoder noise: quantization + tick errors + noise.
        
        Expects RAW input data in ticks/sec (not normalized).
        Noise is applied in physical units for realism.
        
        Pipeline: raw obs → noise → scale (normalize) → clip
        """
        noisy_data = data.clone()
        
        # Initialize accumulator if needed
        if self._tick_error_accumulator is None:
            self._tick_error_accumulator = torch.zeros_like(data)
        
        # Add velocity noise (Gaussian) in raw ticks/sec
        # velocity_noise_std is a fraction, so actual std = fraction * max_velocity
        noise_std = self.cfg.velocity_noise_std * self.cfg.max_velocity
        noisy_data = noisy_data + torch.randn_like(data) * noise_std
        
        # Simulate missed/extra ticks (random discrete errors)
        # Each tick error is ±1 tick/timestep in raw units
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


@configclass
class EncoderNoiseModelCfg(NoiseModelCfg):
    """Configuration for encoder noise model with quantization."""
    
    class_type: type = EncoderNoiseModel
    
    # Quantization
    enable_quantization: bool = True
    ticks_per_radian: float = 85.57  # 537.7 PPR / 2π
    
    # Noise
    velocity_noise_std: float = 0.02  # Fraction of max velocity
    max_velocity: float = 5000.0  # Max ticks/sec for normalization
    
    # Errors
    missed_tick_prob: float = 0.001
    extra_tick_prob: float = 0.0005


# =============================================================================
# Depth Camera Noise Model
# =============================================================================

class DepthNoiseModel(NoiseModel):
    """Depth camera noise model with realistic stereo depth error propagation.

    Models:
    - Stereo depth noise using Intel RealSense error propagation formula
    - Random invalid pixels (holes)
    - Range limiting
    - Dropped frames

    STEREO DEPTH ERROR MODEL (Intel RealSense):
    From stereo geometry, depth z = f·B/d where:
      - f = focal length (pixels)
      - B = stereo baseline (meters)
      - d = disparity (pixels)

    Error propagation gives: σ_z = (z² / (f·B)) · σ_d

    Where σ_d is the subpixel disparity noise (typically 0.05-0.1 pixels for
    good stereo matching algorithms). This quadratic depth dependence matches
    real RealSense behavior much better than linear models.

    Reference: Intel RealSense documentation on depth quality and error propagation
    https://openaccess.thecvf.com/content_cvpr_2017_workshops/w15/papers/Keselman_Intel_RealSense_Stereoscopic_CVPR_2017_paper.pdf
    """

    def __init__(self, noise_model_cfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg = noise_model_cfg

        # Precompute stereo depth noise coefficient: σ_d / (f · B)
        # σ_z = z² · (σ_d / (f · B)) = z² · stereo_coeff
        self._stereo_coeff = self.cfg.disparity_noise_px / (
            self.cfg.focal_length_px * self.cfg.baseline_m
        )

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
        """Apply depth camera noise to RAW depth in meters.

        Expects RAW input data in meters (not normalized).
        Noise is applied in physical units for realism.

        Pipeline: RAW meters → noise (this) → scale (normalize via ObsTerm.scale)
        """
        # Data shape: (num_envs, height * width) - flattened depth in METERS
        noisy_data = data.clone()
        max_range = self.cfg.max_range

        # Stereo depth noise: σ_z = z² · σ_d / (f · B)
        # This is the physically correct error propagation for stereo depth cameras.
        # Noise increases with z² (quadratic), not linearly.
        noise_std = noisy_data.square() * self._stereo_coeff
        depth_noise = torch.randn_like(noisy_data) * noise_std
        noisy_data = noisy_data + depth_noise

        # Add holes (random invalid pixels) - set to max_range
        if self.cfg.hole_probability > 0:
            holes = torch.rand_like(noisy_data) < self.cfg.hole_probability
            noisy_data = torch.where(holes, torch.full_like(noisy_data, max_range), noisy_data)

        # Apply range limits (in meters)
        noisy_data = torch.clamp(noisy_data, 0.0, max_range)
        too_close = noisy_data < self.cfg.min_range
        noisy_data = torch.where(too_close, torch.full_like(noisy_data, max_range), noisy_data)

        # Frame drops (return previous frame)
        if self.cfg.frame_drop_prob > 0 and self._prev_frame is not None:
            drop = torch.rand(self._num_envs, device=self._device) < self.cfg.frame_drop_prob
            noisy_data[drop] = self._prev_frame[drop]

        # Store for next frame
        self._prev_frame = noisy_data.clone()

        return noisy_data


@configclass
class DepthNoiseModelCfg(NoiseModelCfg):
    """Configuration for depth camera noise model with stereo error propagation.

    Uses the Intel RealSense stereo depth error model:
        σ_z = (z² / (f · B)) · σ_d

    Where:
        z = depth in meters
        f = focal_length_px (at native camera resolution)
        B = baseline_m (stereo camera baseline)
        σ_d = disparity_noise_px (subpixel matching accuracy)

    Intel RealSense D555 reference values:
        - Baseline: 95mm (0.095m)
        - Native resolution: 1280x720
        - Horizontal FOV: ~87°
        - Focal length: ~673 pixels (at 1280 width)
        - Typical disparity noise: 0.05-0.1 pixels

    At 2.0m depth with default D555 values:
        σ_z = (2.0² / (673 · 0.095)) · 0.08 ≈ 5.0mm
    """

    class_type: type = DepthNoiseModel

    # Intel RealSense D555 stereo parameters
    # These define the depth-dependent noise via error propagation
    baseline_m: float = 0.095  # Stereo baseline in meters (95mm for D555)
    focal_length_px: float = 673.0  # Focal length in pixels at native resolution
    disparity_noise_px: float = 0.08  # Subpixel disparity noise (typical: 0.05-0.1)

    # Holes (invalid pixels from stereo matching failures)
    hole_probability: float = 0.01

    # Range limits
    min_range: float = 0.2  # meters (D555 min range: 0.4m, we allow some margin)
    max_range: float = 6.0  # meters (D555 max range at optimal accuracy)

    # Frame drops
    frame_drop_prob: float = 0.001

    # Image dimensions (for unflattening if needed)
    height: int = 60
    width: int = 80


# =============================================================================
# RGB Camera Noise Model  
# =============================================================================

class RGBNoiseModel(NoiseModel):
    """RGB camera noise model.
    
    Models:
    - Per-pixel Gaussian noise
    - Random brightness variation
    - Dropped frames
    """
    
    def __init__(self, noise_model_cfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg = noise_model_cfg
        
        self._prev_frame = None
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset state."""
        if env_ids is None:
            self._prev_frame = None
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply RGB noise to [0, 1] normalized pixel values.
        
        Expects input data in [0, 1] range (standard RGB normalization).
        
        Pipeline: [0,1] RGB → noise (this) → output still [0,1]
        """
        noisy_data = data.clone()
        
        # Add pixel noise (Gaussian, independent per pixel)
        noise = torch.randn_like(noisy_data) * self.cfg.pixel_noise_std
        noisy_data = noisy_data + noise
        
        # Random brightness variation (per env, multiplicative)
        b_min, b_max = self.cfg.brightness_range
        brightness = torch.rand(self._num_envs, 1, device=self._device) * (b_max - b_min) + b_min
        noisy_data = noisy_data * brightness
        
        # Clamp to valid [0, 1] range
        noisy_data = torch.clamp(noisy_data, 0.0, 1.0)
        
        # Frame drops (return previous frame)
        if self.cfg.frame_drop_prob > 0 and self._prev_frame is not None:
            drop = torch.rand(self._num_envs, device=self._device) < self.cfg.frame_drop_prob
            noisy_data[drop] = self._prev_frame[drop]
        
        self._prev_frame = noisy_data.clone()
        
        return noisy_data


@configclass
class RGBNoiseModelCfg(NoiseModelCfg):
    """Configuration for RGB camera noise model."""
    
    class_type: type = RGBNoiseModel
    
    # Pixel noise
    pixel_noise_std: float = 0.02  # Fraction of [0, 1] range
    
    # Brightness variation
    brightness_range: tuple[float, float] = (0.9, 1.1)
    
    # Frame drops
    frame_drop_prob: float = 0.001


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
    baseline_m: float = 0.095,
    focal_length_px: float = 673.0,
    disparity_noise_px: float = 0.08,
    hole_probability: float = 0.01,
    frame_drop_prob: float = 0.001,
) -> DepthNoiseModelCfg:
    """Create depth camera noise config with Intel RealSense stereo error model.

    Args:
        baseline_m: Stereo baseline in meters (default: 0.095m for D555)
        focal_length_px: Focal length in pixels at native resolution (default: 673 for D555)
        disparity_noise_px: Subpixel disparity noise (default: 0.08 pixels)
        hole_probability: Probability of invalid pixel (default: 0.01)
        frame_drop_prob: Probability of frame drop (default: 0.001)

    Returns:
        DepthNoiseModelCfg configured for stereo depth noise
    """
    # Compute expected noise at 1m for the GaussianNoiseCfg (informational only)
    noise_at_1m = (1.0 ** 2) * disparity_noise_px / (focal_length_px * baseline_m)
    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=noise_at_1m),
        baseline_m=baseline_m,
        focal_length_px=focal_length_px,
        disparity_noise_px=disparity_noise_px,
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
