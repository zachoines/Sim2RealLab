"""Custom noise models for sim-to-real transfer.

Implements realistic sensor noise models that go beyond simple Gaussian noise:
- IMU: bias drift, temperature effects
- Encoders: quantization, missed ticks
- Cameras: depth-dependent noise, holes, dropped frames
- Observation latency: per-sensor delay buffers

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
# Delay Buffer for Observation Latency
# =============================================================================

class DelayBuffer:
    """Ring buffer for simulating observation latency.

    Stores past observations and returns delayed observations based on
    the configured latency (in control steps).

    Args:
        num_envs: Number of parallel environments.
        obs_size: Size of observation vector per environment.
        delay_steps: Number of steps to delay (0 = no delay).
        device: Torch device for tensors.
    """

    def __init__(self, num_envs: int, obs_size: int, delay_steps: int, device: str):
        self._num_envs = num_envs
        self._obs_size = obs_size
        self._delay_steps = delay_steps
        self._device = device

        # Ring buffer: (delay_steps + 1, num_envs, obs_size)
        # +1 because we need to store current + past observations
        if delay_steps > 0:
            self._buffer = torch.zeros(
                delay_steps + 1, num_envs, obs_size, device=device
            )
            self._write_idx = 0
        else:
            self._buffer = None

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset buffer for specified environments."""
        if self._buffer is None:
            return

        if env_ids is None:
            self._buffer.zero_()
        else:
            self._buffer[:, env_ids] = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Store current observation and return delayed observation.

        Args:
            data: Current observation tensor of shape (num_envs, obs_size).

        Returns:
            Delayed observation tensor of same shape.
        """
        if self._buffer is None or self._delay_steps == 0:
            return data

        # Store current observation at write index
        self._buffer[self._write_idx] = data

        # Compute read index (delay_steps behind write)
        read_idx = (self._write_idx - self._delay_steps) % (self._delay_steps + 1)

        # Get delayed observation
        delayed = self._buffer[read_idx].clone()

        # Advance write index
        self._write_idx = (self._write_idx + 1) % (self._delay_steps + 1)

        return delayed


# =============================================================================
# IMU Noise Model
# =============================================================================

class IMUNoiseModel(NoiseModel):
    """IMU noise model with bias drift and failure modes.

    Models realistic IMU behavior:
    - Additive white Gaussian noise
    - Slowly drifting bias (random walk)
    - Optional temperature-dependent effects
    - Sensor failure modes (zeros or stuck values)

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

        # Track previous output for "stuck" failure mode
        self._prev_output = None

        # Delay buffer for observation latency
        self._delay_buffer = DelayBuffer(
            num_envs, self.cfg.output_size, self.cfg.latency_steps, device
        ) if self.cfg.latency_steps > 0 else None

        # Initialize with random bias
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset bias to random values within range."""
        if env_ids is None:
            env_ids = slice(None)
            self._prev_output = None
        elif self._prev_output is not None:
            # Reset stuck state for specific environments
            self._prev_output[env_ids] = 0.0

        # Reset delay buffer
        if self._delay_buffer is not None:
            self._delay_buffer.reset(env_ids if env_ids != slice(None) else None)

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
        """Apply IMU noise: bias + white noise + failure modes.

        The drift rate is scaled by sqrt(dt) to properly model a random walk
        where drift_rate is specified per-second.

        Failure modes:
        - failure_probability: sensor returns zeros
        - stuck_probability: sensor returns previous value
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
        noisy_data = data + self._bias + noise

        # Apply failure modes
        if self.cfg.failure_probability > 0 or self.cfg.stuck_probability > 0:
            # Initialize prev_output if needed
            if self._prev_output is None:
                self._prev_output = noisy_data.clone()

            # Failure mode: return zeros
            if self.cfg.failure_probability > 0:
                failed = torch.rand(self._num_envs, device=self._device) < self.cfg.failure_probability
                noisy_data[failed] = 0.0

            # Stuck mode: return previous value
            if self.cfg.stuck_probability > 0:
                stuck = torch.rand(self._num_envs, device=self._device) < self.cfg.stuck_probability
                noisy_data[stuck] = self._prev_output[stuck]

            # Store for next step
            self._prev_output = noisy_data.clone()

        # Apply observation latency
        if self._delay_buffer is not None:
            noisy_data = self._delay_buffer(noisy_data)

        return noisy_data


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

    # Failure modes (for robustness training)
    failure_probability: float = 0.0
    """Probability of sensor failure per step (returns zeros)."""

    stuck_probability: float = 0.0
    """Probability of sensor getting stuck per step (returns previous value)."""

    # Observation latency
    latency_steps: int = 0
    """Observation latency in control steps. 0 = no delay."""


# =============================================================================
# Encoder Noise Model
# =============================================================================

class EncoderNoiseModel(NoiseModel):
    """Encoder noise model with quantization, tick errors, and failure modes.

    Models realistic encoder behavior:
    - Quantization to discrete ticks
    - Random missed/extra ticks
    - Velocity estimation noise
    - Sensor failure mode (returns zeros)
    """

    def __init__(self, noise_model_cfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg = noise_model_cfg

        # Track accumulated tick errors for position observations
        self._tick_error_accumulator = None

        # Delay buffer for observation latency
        self._delay_buffer = DelayBuffer(
            num_envs, self.cfg.output_size, self.cfg.latency_steps, device
        ) if self.cfg.latency_steps > 0 else None

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset tick error accumulator and delay buffer."""
        if self._tick_error_accumulator is not None:
            if env_ids is None:
                self._tick_error_accumulator.zero_()
            else:
                self._tick_error_accumulator[env_ids] = 0.0

        # Reset delay buffer
        if self._delay_buffer is not None:
            self._delay_buffer.reset(env_ids)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply encoder noise: quantization + tick errors + noise + failures.

        Expects RAW input data in ticks/sec (not normalized).
        Noise is applied in physical units for realism.

        Pipeline: raw obs → noise → scale (normalize) → clip

        Failure mode:
        - failure_probability: encoder returns zeros (electrical failure)
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

        # Apply failure mode: return zeros for failed encoders
        if self.cfg.failure_probability > 0:
            failed = torch.rand(self._num_envs, device=self._device) < self.cfg.failure_probability
            noisy_data[failed] = 0.0

        # Apply observation latency
        if self._delay_buffer is not None:
            noisy_data = self._delay_buffer(noisy_data)

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
    max_velocity: float = 3000.0  # Max ticks/sec (312 RPM ≈ 2796 ticks/s + margin)

    # Errors
    missed_tick_prob: float = 0.001
    extra_tick_prob: float = 0.0005

    # Failure modes (for robustness training)
    failure_probability: float = 0.0
    """Probability of encoder failure per step (returns zeros)."""

    # Observation latency
    latency_steps: int = 0
    """Observation latency in control steps. 0 = no delay."""

    # Output size (for delay buffer initialization)
    output_size: int = 4
    """Number of encoder values (4 for mecanum wheels)."""


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

        # Delay buffer for observation latency (initialized lazily due to variable obs size)
        self._delay_buffer = None
        self._latency_steps = self.cfg.latency_steps

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset frame drop state and delay buffer."""
        if env_ids is None:
            self._prev_frame = None
            self._frame_dropped.zero_()
            if self._delay_buffer is not None:
                self._delay_buffer.reset(None)
        else:
            self._frame_dropped[env_ids] = False
            if self._delay_buffer is not None:
                self._delay_buffer.reset(env_ids)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply depth camera noise to RAW depth in meters.

        Expects RAW input data in meters (not normalized).
        Noise is applied in physical units for realism.

        Pipeline: RAW meters → noise (this) → scale (normalize via ObsTerm.scale)

        Failure mode:
        - failure_probability: camera returns max_range for all pixels
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

        # Camera failure: return max_range for all pixels (camera returns "far" everywhere)
        if self.cfg.failure_probability > 0:
            failed = torch.rand(self._num_envs, device=self._device) < self.cfg.failure_probability
            noisy_data[failed] = max_range

        # Store for next frame
        self._prev_frame = noisy_data.clone()

        # Apply observation latency (lazily initialize buffer based on actual obs size)
        if self._latency_steps > 0:
            if self._delay_buffer is None:
                obs_size = noisy_data.shape[1] if noisy_data.dim() > 1 else 1
                self._delay_buffer = DelayBuffer(
                    self._num_envs, obs_size, self._latency_steps, self._device
                )
            noisy_data = self._delay_buffer(noisy_data)

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

    # Failure modes (for robustness training)
    failure_probability: float = 0.0
    """Probability of camera failure per step (returns max_range for all pixels)."""

    # Observation latency
    latency_steps: int = 1
    """Observation latency in control steps. 1 = one step delay (default for cameras)."""


# =============================================================================
# RGB Camera Noise Model  
# =============================================================================

class RGBNoiseModel(NoiseModel):
    """RGB camera noise model with failure modes.

    Models:
    - Per-pixel Gaussian noise
    - Random brightness variation
    - Dropped frames
    - Camera failure mode (black image)
    """

    def __init__(self, noise_model_cfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg = noise_model_cfg

        self._prev_frame = None

        # Delay buffer for observation latency (initialized lazily due to variable obs size)
        self._delay_buffer = None
        self._latency_steps = self.cfg.latency_steps

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset state and delay buffer."""
        if env_ids is None:
            self._prev_frame = None
            if self._delay_buffer is not None:
                self._delay_buffer.reset(None)
        elif self._delay_buffer is not None:
            self._delay_buffer.reset(env_ids)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply RGB noise to [0, 1] normalized pixel values.

        Expects input data in [0, 1] range (standard RGB normalization).

        Pipeline: [0,1] RGB → noise (this) → output still [0,1]

        Failure mode:
        - failure_probability: camera returns black image (zeros)
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

        # Camera failure: return black image (zeros)
        if self.cfg.failure_probability > 0:
            failed = torch.rand(self._num_envs, device=self._device) < self.cfg.failure_probability
            noisy_data[failed] = 0.0

        self._prev_frame = noisy_data.clone()

        # Apply observation latency (lazily initialize buffer based on actual obs size)
        if self._latency_steps > 0:
            if self._delay_buffer is None:
                obs_size = noisy_data.shape[1] if noisy_data.dim() > 1 else 1
                self._delay_buffer = DelayBuffer(
                    self._num_envs, obs_size, self._latency_steps, self._device
                )
            noisy_data = self._delay_buffer(noisy_data)

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

    # Failure modes (for robustness training)
    failure_probability: float = 0.0
    """Probability of camera failure per step (returns black image - zeros)."""

    # Observation latency
    latency_steps: int = 1
    """Observation latency in control steps. 1 = one step delay (default for cameras)."""
