"""Custom noise models for sim-to-real transfer.

Implements realistic sensor noise models that go beyond simple Gaussian noise:
- IMU: bias drift
- Encoders: the reported speed's lag, quantization
- Cameras: depth-dependent noise, holes, dropped frames
- Observation latency: per-sensor delay buffers

These integrate with Isaac Lab's NoiseModel system and can be used
directly in ObsTerm configurations.

IMPORTANT: Each NoiseModel class is defined BEFORE its corresponding Cfg class
so that class_type can reference it directly. This ensures the class_type
attribute is properly set when the Cfg is instantiated.
"""

from __future__ import annotations

import math

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.noise import NoiseModel
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import NoiseModelCfg, GaussianNoiseCfg

from strafer_shared.constants import DEPTH_HEIGHT, DEPTH_NEARFIELD_FILL, DEPTH_WIDTH, ENCODER_TICKS_TO_RADIANS
from strafer_shared.mecanum_kinematics import INVERSE_KINEMATIC_MATRIX

from .hold_process import HoldProcess

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
    the configured latency (in control steps). Until an env's ring has been
    written it holds that env's first frame since its own reset.

    With ``delay_steps_range`` the latency is drawn per environment at every
    reset instead of being one number shared by the whole batch, so the policy
    sees an age *distribution* on the modality rather than a constant offset a
    recurrent net can calibrate away. The fixed-latency path is unchanged when
    no range is given.

    Args:
        num_envs: Number of parallel environments.
        obs_size: Size of observation vector per environment.
        delay_steps: Number of steps to delay (0 = no delay).
        device: Torch device for tensors.
        delay_steps_range: Absolute per-env latency band [min, max] in control
            steps, sampled at reset. None keeps the fixed ``delay_steps``.
    """

    def __init__(
        self,
        num_envs: int,
        obs_size: int,
        delay_steps: int,
        device: str,
        delay_steps_range: tuple[int, int] | None = None,
    ):
        self._num_envs = num_envs
        self._obs_size = obs_size
        self._delay_steps = delay_steps
        self._device = device

        if delay_steps_range is not None:
            lo, hi = int(min(delay_steps_range)), int(max(delay_steps_range))
            self._delay_range = (max(0, lo), max(0, hi))
        else:
            self._delay_range = None

        # Ring depth must cover the largest latency any env can draw.
        max_delay = delay_steps if self._delay_range is None else self._delay_range[1]
        self._max_delay = max_delay

        # Ring buffer: (max_delay + 1, num_envs, obs_size)
        # +1 because we need to store current + past observations
        if max_delay > 0:
            self._buffer = torch.zeros(max_delay + 1, num_envs, obs_size, device=device)
            self._write_idx = 0
            self._delays = torch.full(
                (num_envs,), delay_steps, dtype=torch.long, device=device
            )
            self._env_index = torch.arange(num_envs, device=device)
            self._primed = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._needs_prime = True
            if self._delay_range is not None:
                self._sample_delays(None)
        else:
            self._buffer = None

    def _sample_delays(self, env_ids: Sequence[int] | None):
        """Draw the per-env latency for the given envs from the configured band."""
        lo, hi = self._delay_range
        count = self._num_envs if env_ids is None else len(env_ids)
        if count == 0:
            return
        drawn = torch.randint(lo, hi + 1, (count,), device=self._device)
        if env_ids is None:
            self._delays.copy_(drawn)
        else:
            self._delays[env_ids] = drawn

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset buffer for specified environments."""
        if self._buffer is None:
            return

        if env_ids is None:
            self._buffer.zero_()
            self._primed.zero_()
        else:
            self._buffer[:, env_ids] = 0.0
            self._primed[env_ids] = False
        self._needs_prime = True

        if self._delay_range is not None:
            self._sample_delays(env_ids)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Store current observation and return delayed observation.

        Args:
            data: Current observation tensor of shape (num_envs, obs_size).

        Returns:
            Delayed observation tensor of same shape.
        """
        if self._buffer is None or self._max_delay == 0:
            return data

        ring = self._max_delay + 1

        # Fill an unwritten ring with the live frame. Only the
        # envs a reset actually cleared are primed, so a partial reset cannot
        # overwrite the in-flight history of the envs that kept running. The
        # host-side flag keeps this off the steady-state path rather than
        # syncing the mask back every step.
        if self._needs_prime:
            fresh = ~self._primed
            self._buffer[:, fresh] = data[fresh]
            self._primed.fill_(True)
            self._needs_prime = False

        # Store current observation at write index
        self._buffer[self._write_idx] = data

        # Compute read index (delay behind write), per env when randomized
        if self._delay_range is None:
            # A scalar index is a view into the ring; the write that comes
            # round to that slot would otherwise change it under the caller.
            read_idx = (self._write_idx - self._delay_steps) % ring
            delayed = self._buffer[read_idx].clone()
        else:
            # Advanced indexing already materializes a copy.
            read_idx = torch.remainder(self._write_idx - self._delays, ring)
            delayed = self._buffer[read_idx, self._env_index]

        # Advance write index
        self._write_idx = (self._write_idx + 1) % ring

        return delayed


# =============================================================================
# IMU Noise Model
# =============================================================================

class IMUNoiseModel(NoiseModel):
    """IMU noise model with bias drift and failure modes.

    Models realistic IMU behavior:
    - Additive white Gaussian noise
    - Slowly drifting bias (random walk)
    - A stuck sensor, which repeats its previous reading

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

        # The previous reading a stuck sensor repeats, and whether each env has one since its last reset
        self._prev_output = None
        self._has_prev = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Initialize with random bias
        self._delay_buffer = None
        self.reset()

        # Delay buffer for observation latency, built after the first bias draw so a latency band draws after it
        latency_range = self.cfg.latency_steps_range
        self._delay_buffer = DelayBuffer(
            num_envs, self.cfg.output_size, self.cfg.latency_steps, device, delay_steps_range=latency_range
        ) if self.cfg.latency_steps > 0 or latency_range is not None else None

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset bias to random values within range, forget the previous reading, and reset the delay buffer (whose
        latency band, if set, draws after the bias)."""
        if env_ids is None:
            env_ids = slice(None)
        self._has_prev[env_ids] = False

        # Sample new bias values from configured range
        bias_min, bias_max = self._bias_range
        self._bias[env_ids] = (
            torch.rand(self._get_num_envs(env_ids), self.cfg.output_size, device=self._device)
            * (bias_max - bias_min) + bias_min
        )

        # Reset delay buffer
        if self._delay_buffer is not None:
            self._delay_buffer.reset(env_ids if env_ids != slice(None) else None)

    def _get_num_envs(self, env_ids) -> int:
        """Get number of environments from env_ids."""
        if isinstance(env_ids, slice) and env_ids == slice(None):
            return self._num_envs
        return len(env_ids)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply IMU noise: bias + white noise + failure modes.

        The drift rate is scaled by sqrt(dt) to properly model a random walk
        where drift_rate is specified per-second.

        Failure mode:
        - stuck_probability: the sensor repeats its previous reading (on an env's first step after a reset there is
          none, and the current reading passes)
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

        # A stuck sensor repeats its previous reading
        if self.cfg.stuck_probability > 0:
            if self._prev_output is None:
                self._prev_output = noisy_data.clone()
            stuck = torch.rand(self._num_envs, device=self._device) < self.cfg.stuck_probability
            stuck &= self._has_prev
            noisy_data[stuck] = self._prev_output[stuck]
            self._prev_output = noisy_data.clone()
            self._has_prev.fill_(True)

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

    # Failure mode (for robustness training)
    stuck_probability: float = 0.0
    """Probability per step that the sensor repeats its previous reading."""

    # Observation latency
    latency_steps: int = 0
    """Observation latency in control steps. 0 = no delay."""

    latency_steps_range: tuple[int, int] | None = None
    """Absolute per-env latency band [min, max] in control steps, drawn at every reset
    after the bias. None keeps the fixed ``latency_steps`` for every env."""


# =============================================================================
# Encoder Noise Model
# =============================================================================

class EncoderNoiseModel(NoiseModel):
    """Encoder noise model: the reported speed's lag, noise, quantization and failure modes.

    Models the wheel speed the robot reports:
    - The reported speed's lag behind the wheel (``ReportedSpeedLag``), off at a lag of 0
    - Velocity estimation noise
    - Quantization to discrete ticks
    - A stuck encoder, which repeats its previous reading

    Its output is the one wheel-speed reading of the control step: it keeps it as ``last_output`` and offers itself to
    the ``BodyVelocityNoiseModel`` built next in the same observation group, whose body velocity is the kinematics of
    this reading, as the robot's odom is of the speeds its ``joint_states`` carry.
    """

    def __init__(self, noise_model_cfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg = noise_model_cfg
        self._lag = ReportedSpeedLag(noise_model_cfg.reported_speed_form, noise_model_cfg.reported_speed_lag_s, num_envs,
                                     noise_model_cfg.output_size, 1.0 / noise_model_cfg.control_frequency_hz, device)
        self.last_output: torch.Tensor | None = None
        _offer_wheel_reading(self)

        # The previous reading a stuck encoder repeats, and whether each env has one since its last reset
        self._prev_output = None
        self._has_prev = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Delay buffer for observation latency
        self._delay_buffer = DelayBuffer(
            num_envs, self.cfg.output_size, self.cfg.latency_steps, device
        ) if self.cfg.latency_steps > 0 else None

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the lag and the delay buffer, and forget the previous reading."""
        self._has_prev[slice(None) if env_ids is None else env_ids] = False
        self._lag.reset(None if env_ids is None else torch.as_tensor(env_ids, device=self._device))

        # Reset delay buffer
        if self._delay_buffer is not None:
            self._delay_buffer.reset(env_ids)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the reported speed's lag, then encoder noise: velocity noise + quantization + failures.

        Expects RAW input data in ticks/sec (not normalized).
        Noise is applied in physical units for realism.

        Pipeline: raw obs → noise → scale (normalize) → clip

        Failure mode:
        - stuck_probability: the encoders repeat their previous reading (on an env's first step after a reset there
          is none, and the current reading passes)
        """
        noisy_data = self._lag(data).clone()

        # Add velocity noise (Gaussian) in raw ticks/sec
        # velocity_noise_std is a fraction, so actual std = fraction * max_velocity
        noise_std = self.cfg.velocity_noise_std * self.cfg.max_velocity
        noisy_data = noisy_data + torch.randn_like(data) * noise_std

        # Quantization (round to nearest tick)
        if self.cfg.enable_quantization:
            noisy_data = torch.round(noisy_data)

        # A stuck encoder repeats its previous reading
        if self.cfg.stuck_probability > 0:
            if self._prev_output is None:
                self._prev_output = noisy_data.clone()
            stuck = torch.rand(self._num_envs, device=self._device) < self.cfg.stuck_probability
            stuck &= self._has_prev
            noisy_data[stuck] = self._prev_output[stuck]
            self._prev_output = noisy_data.clone()
            self._has_prev.fill_(True)

        # Apply observation latency
        if self._delay_buffer is not None:
            noisy_data = self._delay_buffer(noisy_data)

        self.last_output = noisy_data
        return noisy_data


@configclass
class EncoderNoiseModelCfg(NoiseModelCfg):
    """Configuration for encoder noise model with quantization."""

    class_type: type = EncoderNoiseModel

    # Quantization
    enable_quantization: bool = True

    # Noise
    velocity_noise_std: float = 0.02  # Fraction of max velocity
    max_velocity: float = 3000.0  # Max ticks/sec (312 RPM ≈ 2796 ticks/s + margin)

    # The reported speed's lag, applied before the noise by the shared wheel reading
    reported_speed_form: str = "first_order"
    """The lag's form: "first_order" (a filter whose 63 % rise time is the parameter), "delay" (a pure delay) or
    "window" (a moving average over the parameter's span)."""

    reported_speed_lag_s: float = 0.0
    """The lag's parameter in seconds (the time constant, the delay, or the window's span); 0 = no lag."""

    control_frequency_hz: float = 30.0
    """The rate the observation is computed at, to convert the lag's seconds to steps."""

    # Failure mode (for robustness training)
    stuck_probability: float = 0.0
    """Probability per step that the encoders repeat their previous reading."""

    # Observation latency
    latency_steps: int = 0
    """Observation latency in control steps. 0 = no delay."""

    # Output size (for delay buffer initialization)
    output_size: int = 4
    """Number of encoder values (4 for mecanum wheels)."""


# =============================================================================
# Depth Camera Noise Model
# =============================================================================

TOO_CLOSE_FILLS = ("near", "max")
HOLE_FILLS = ("median", "near", "max")


class DepthNoiseModel(NoiseModel):
    """Depth camera noise model with realistic stereo depth error propagation.

    Models:
    - Stereo depth noise using Intel RealSense error propagation formula
    - Random invalid pixels (holes), outvoted by their neighbours
    - Range limiting, saturating the near field
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

    RESOLUTION AXIS. f is the native-resolution focal length (673 px at 1280
    wide), so σ_z is a native-disparity quantity, but it is drawn i.i.d. per
    80x45 policy pixel, after the reduction rather than before it. Training and
    deployment share the reduction -- a median of 64 native pixels per policy
    pixel -- so the clean fields agree; what this model adds is injected
    downstream of it. A median attenuates an i.i.d. native field 6.46x and a
    perfectly correlated one not at all, so whether σ_d is to be read as native
    or as already post-reduction-equivalent still turns on the within-block
    correlation of real sensor depth, which no capture has measured; both
    readings are consistent with the shipped value.

    Reference: Intel RealSense documentation on depth quality and error propagation
    https://openaccess.thecvf.com/content_cvpr_2017_workshops/w15/papers/Keselman_Intel_RealSense_Stereoscopic_CVPR_2017_paper.pdf
    """

    def __init__(self, noise_model_cfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self.cfg = noise_model_cfg

        # An unrecognised convention would otherwise fall through to a default
        # and silently reinstate the behaviour these fields exist to select.
        if self.cfg.too_close_fill not in TOO_CLOSE_FILLS:
            raise ValueError(
                f"too_close_fill must be one of {TOO_CLOSE_FILLS}, got "
                f"'{self.cfg.too_close_fill}'"
            )
        if self.cfg.hole_fill not in HOLE_FILLS:
            raise ValueError(
                f"hole_fill must be one of {HOLE_FILLS}, got '{self.cfg.hole_fill}'"
            )

        # Precompute stereo depth noise coefficient: σ_d / (f · B)
        # σ_z = z² · (σ_d / (f · B)) = z² · stereo_coeff
        self._focal_baseline = self.cfg.focal_length_px * self.cfg.baseline_m
        self._stereo_coeff = self.cfg.disparity_noise_px / self._focal_baseline

        # With disparity_noise_px_range the subpixel noise is drawn per
        # environment at every reset instead of being one number shared by the
        # whole batch, so the policy sees a texture *distribution* on the
        # modality rather than one amplitude it can key on. The fixed path is
        # unchanged when no range is given, down to the bit.
        self._disparity_range = self.cfg.disparity_noise_px_range
        self._env_coeff = None
        if self._disparity_range is not None:
            low, high = sorted(self._disparity_range)
            if low <= 0.0:
                raise ValueError(
                    "disparity_noise_px_range is drawn log-uniformly and both "
                    f"ends must be positive, got {self.cfg.disparity_noise_px_range}. "
                    "A low end of 0.002 already reaches a field far smoother "
                    "than the deploy path delivers."
                )
            self._disparity_range = (low, high)
            # Held in double and cast at use: a draw landing on a band end then
            # divides exactly as the scalar path's Python float does, so the
            # band is a reparameterisation rather than a second model.
            self._env_coeff = torch.zeros(num_envs, 1, dtype=torch.float64, device=device)

        # Store previous frame for drops
        self._prev_frame = None
        self._frame_dropped = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Cleared per env on reset, so a teleported env cannot re-emit the
        # frame it saw before the teleport.
        self._has_prev_frame = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Bursty stream holds, layered on the memoryless per-frame drop above.
        # The two model different causes of "this frame did not advance" — the
        # sensor's own dropout and a stalled transport/consumption stream — and
        # compose as a union, so a repeat is either.
        self._hold = HoldProcess(
            num_envs,
            device,
            hold_fraction_range=self.cfg.hold_fraction_range,
            hold_run_range=self.cfg.hold_run_range,
            burst_weight=self.cfg.hold_burst_weight,
            burst_run=self.cfg.hold_burst_run_steps,
        )

        # Delay buffer for observation latency (initialized lazily due to variable obs size)
        self._delay_buffer = None
        self._latency_steps = self.cfg.latency_steps
        self._latency_steps_range = self.cfg.latency_steps_range

        # Drawn last, after every other per-env parameter above has taken its
        # own draw, so adding the band does not move theirs. A band pinned to
        # one value then reproduces the fixed path from the same seed.
        if self._disparity_range is not None:
            self._sample_disparity(None)

    def _sample_disparity(self, env_ids: Sequence[int] | None):
        """Draw the per-env subpixel disparity noise from the configured band.

        Log-uniform: σ_d is a scale parameter, and a uniform draw over a band
        spanning two decades puts almost all of its mass in the loud decade.
        The closed form is exact at a pinned band, so the band stays a
        reparameterisation of the fixed path.
        """
        low, high = self._disparity_range
        count = self._num_envs if env_ids is None else len(env_ids)
        if count == 0:
            return
        drawn = torch.rand(count, 1, dtype=torch.float64, device=self._device)
        drawn = low * (high / low) ** drawn / self._focal_baseline
        if env_ids is None:
            self._env_coeff.copy_(drawn)
        else:
            self._env_coeff[env_ids] = drawn

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset frame drop state, hold state, and delay buffer."""
        self._hold.reset(env_ids)
        if env_ids is None:
            self._prev_frame = None
            self._frame_dropped.zero_()
            self._has_prev_frame.zero_()
            if self._delay_buffer is not None:
                self._delay_buffer.reset(None)
        else:
            self._frame_dropped[env_ids] = False
            self._has_prev_frame[env_ids] = False
            if self._delay_buffer is not None:
                self._delay_buffer.reset(env_ids)

        # Last, for the reason ``__init__`` draws it last.
        if self._disparity_range is not None:
            self._sample_disparity(env_ids)

    def _neighbourhood_median(
        self, data: torch.Tensor, invalid: torch.Tensor
    ) -> torch.Tensor:
        """Median of each invalid pixel's valid 3x3 neighbours.

        Written only at the invalid pixels, which are a few per cent of a
        frame, so the cost tracks the hole rate rather than the frame size.
        A pixel on the frame's edge reads the neighbours it actually has.
        Where a neighbourhood is invalid throughout there is nothing to read
        and the near fill stands in.

        The median is the lower of the two middle neighbours on an even count,
        which biases an ambiguous pixel towards the nearer surface.
        """
        height, width = self.cfg.height, self.cfg.width
        flat = data.reshape(-1, height * width)
        rows = invalid.reshape(flat.shape).nonzero(as_tuple=False)
        if rows.numel() == 0:
            return data

        env, pixel = rows[:, 0], rows[:, 1]
        y, x = pixel // width, pixel % width
        # The border is padded so every window is the same shape, but the pad
        # counts as invalid: a replicated cell would let one real neighbour vote
        # twice and pull an edge pixel's median off its surface.
        padded = torch.nn.functional.pad(
            flat.reshape(-1, 1, height, width), (1, 1, 1, 1), mode="replicate"
        )[:, 0]
        valid = torch.nn.functional.pad(
            (~invalid).reshape(-1, 1, height, width).to(data.dtype),
            (1, 1, 1, 1),
        )[:, 0]

        step = torch.arange(3, device=data.device)
        ys = (y[:, None, None] + step[None, :, None]).expand(-1, 3, 3)
        xs = (x[:, None, None] + step[None, None, :]).expand(-1, 3, 3)
        envs = env[:, None, None].expand(-1, 3, 3)
        windows = padded[envs, ys, xs].reshape(-1, 9)
        weights = valid[envs, ys, xs].reshape(-1, 9)

        # Invalid neighbours sort last, so the index counts only valid ones.
        candidates, _ = torch.where(
            weights > 0, windows, torch.full_like(windows, float("inf"))
        ).sort(dim=-1)
        count = weights.sum(dim=-1)
        index = ((count.clamp(min=1).long() - 1) // 2).unsqueeze(-1)
        median = candidates.gather(-1, index).squeeze(-1)
        median = torch.where(
            count > 0, median, torch.full_like(median, self.cfg.min_range)
        )

        filled = flat.clone()
        filled[env, pixel] = median
        return filled.reshape(data.shape)

    def _invalid_fill(self, data: torch.Tensor, invalid: torch.Tensor) -> torch.Tensor:
        """Depth to write at invalid pixels, per the configured convention.

        ``"median"`` lets a pixel read its neighbourhood, so a stereo failure is
        outvoted by the surface around it instead of reading as open space. That
        is the property the deploy reduction has and a per-pixel constant does
        not; the two are not the same operation, and they part company where a
        whole neighbourhood is invalid — deployment reads the far clamp there,
        this takes the near fill. The constants stay selectable so either
        convention's statistics can be measured.
        """
        if self.cfg.hole_fill == "near":
            return torch.full_like(data, self.cfg.min_range)
        if self.cfg.hole_fill == "max":
            return torch.full_like(data, self.cfg.max_range)
        return self._neighbourhood_median(data, invalid)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply depth camera noise to RAW depth in meters.

        Expects RAW input data in meters (not normalized).
        Noise is applied in physical units for realism.

        Pipeline: RAW meters → noise (this) → scale (normalize via ObsTerm.scale)

        A failed frame is a dropped one: it repeats the previous frame (``frame_drop_prob``).
        """
        # Data shape: (num_envs, height * width) - flattened depth in METERS
        if self.cfg.height * self.cfg.width != data.shape[-1]:
            raise ValueError(
                f"depth noise cfg declares {self.cfg.height}x{self.cfg.width} "
                f"but received {data.shape[-1]} pixels; the neighbourhood "
                "median reads a pixel's neighbours, so the shape has to be the "
                "frame's own"
            )
        noisy_data = data.clone()
        max_range = self.cfg.max_range

        # Stereo depth noise: σ_z = z² · σ_d / (f · B)
        # This is the physically correct error propagation for stereo depth cameras.
        # Noise increases with z² (quadratic), not linearly.
        coeff = (self._stereo_coeff if self._env_coeff is None
                 else self._env_coeff.to(noisy_data.dtype))
        noise_std = noisy_data.square() * coeff
        depth_noise = torch.randn_like(noisy_data) * noise_std
        noisy_data = noisy_data + depth_noise

        # An invalid pixel reads its neighbourhood rather than a constant, so
        # an isolated stereo failure cannot present as open space.
        if self.cfg.hole_probability > 0:
            holes = torch.rand_like(noisy_data) < self.cfg.hole_probability
            noisy_data = torch.where(holes, self._invalid_fill(noisy_data, holes), noisy_data)

        # Apply range limits (in meters)
        noisy_data = torch.clamp(noisy_data, 0.0, max_range)
        too_close = noisy_data < self.cfg.min_range
        near = self.cfg.min_range if self.cfg.too_close_fill == "near" else max_range
        noisy_data = torch.where(too_close, torch.full_like(noisy_data, near), noisy_data)

        # Frame drops (return previous frame)
        if self.cfg.frame_drop_prob > 0 and self._prev_frame is not None:
            drop = torch.rand(self._num_envs, device=self._device) < self.cfg.frame_drop_prob
            drop &= self._has_prev_frame
            noisy_data[drop] = self._prev_frame[drop]

        # Stream holds (return previous frame for the length of the run). The
        # process advances every step so its run statistics stay well-defined;
        # only the emission is conditional on this episode having produced a
        # frame to hold.
        if self._hold.enabled:
            held = self._hold.step()
            if self._prev_frame is not None:
                # Out-of-place: step() hands back the process's own state
                # tensor, so masking in place would cancel a run the process
                # believes is still going.
                held = held & self._has_prev_frame
                noisy_data[held] = self._prev_frame[held]

        # Store for next frame
        self._prev_frame = noisy_data.clone()
        self._has_prev_frame.fill_(True)

        # Apply observation latency (lazily initialize buffer based on actual obs size)
        if self._latency_steps > 0 or self._latency_steps_range is not None:
            if self._delay_buffer is None:
                obs_size = noisy_data.shape[1] if noisy_data.dim() > 1 else 1
                self._delay_buffer = DelayBuffer(
                    self._num_envs,
                    obs_size,
                    self._latency_steps,
                    self._device,
                    delay_steps_range=self._latency_steps_range,
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

    disparity_noise_px_range: tuple[float, float] | None = None
    """Per-env subpixel disparity noise band [min, max], drawn log-uniformly at
    reset. Both ends must be positive. None keeps the fixed
    ``disparity_noise_px`` for every env."""

    # Holes (invalid pixels from stereo matching failures)
    hole_probability: float = 0.01

    # Range limits
    # Below the D555's own 0.4 m floor on purpose: this is the near fill, and
    # too_close_fill decides whether a reading below it takes that value.
    min_range: float = DEPTH_NEARFIELD_FILL
    max_range: float = 6.0  # meters (D555 max range at optimal accuracy)

    # Which value an unresolvable reading takes. Both are selectable so either
    # convention's statistics can be measured.
    too_close_fill: str = "near"
    """Value written below ``min_range``: ``"near"`` (``min_range``, the value
    the observation term already wrote there) or ``"max"``."""

    hole_fill: str = "median"
    """Value written at invalid pixels: ``"median"`` of the valid 3x3
    neighbours, ``"near"``, or ``"max"``."""

    # Frame drops
    frame_drop_prob: float = 0.001

    # Stream holds (bursty repeats, layered on the memoryless drop above)
    hold_fraction_range: tuple[float, float] = (0.0, 0.0)
    """Per-env stationary share of steps on which the frame does not advance,
    sampled at reset. A max of 0 leaves the hold process inert."""

    hold_run_range: tuple[float, float] = (1.0, 1.0)
    """Per-env mean base hold-run length [min, max] in control steps, sampled
    at reset."""

    hold_burst_weight: float = 0.0
    """Share of hold runs drawn from the long burst component."""

    hold_burst_run_steps: float = 1.0
    """Mean length of a burst hold run, in control steps."""

    # Image dimensions, used to unflatten for the neighbourhood median.
    height: int = DEPTH_HEIGHT
    width: int = DEPTH_WIDTH

    # Observation latency
    latency_steps: int = 1
    """Observation latency in control steps. 1 = one step delay (default for cameras)."""

    latency_steps_range: tuple[int, int] | None = None
    """Absolute per-env latency band [min, max] in control steps, sampled at
    reset. None keeps the fixed ``latency_steps`` for every env."""


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


# =============================================================================
# The shared wheel reading
# =============================================================================

REPORTED_SPEED_FORMS = ("first_order", "delay", "window")


class ReportedSpeedLag:
    """The lag of the wheel speed the robot reports behind the wheel's own speed, per env, in one of three forms.

    The parameter is in seconds and is converted at the control step, so it means the same at any decimation: a
    first-order filter with alpha = 1 - exp(-dt / lag), a delay of round(lag / dt) steps, or a moving average over
    max(1, round(lag / dt)) steps. A lag of 0 is the identity. An env's first reading after a reset seeds the state,
    so the lag starts from the wheel's speed, not from zero.
    """

    def __init__(self, form: str, lag_s: float, num_envs: int, width: int, step_dt: float, device):
        if form not in REPORTED_SPEED_FORMS:
            raise ValueError(f"reported_speed_form {form!r} is not one of {REPORTED_SPEED_FORMS}")
        if lag_s < 0:
            raise ValueError(f"reported_speed_lag_s {lag_s} is negative")
        self.form = form
        self.enabled = lag_s > 0
        self._fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        if not self.enabled:
            return
        if form == "first_order":
            self._alpha = -math.expm1(-step_dt / lag_s)
            self._state = torch.zeros(num_envs, width, device=device)
        else:
            steps = round(lag_s / step_dt)
            self._length = max(1, steps) if form == "window" else steps + 1
            self._history = torch.zeros(num_envs, self._length, width, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        fresh = self._fresh
        if self.form == "first_order":
            self._state = torch.where(fresh.unsqueeze(-1), x, self._state + self._alpha * (x - self._state))
            out = self._state
        else:
            self._history = torch.roll(self._history, shifts=1, dims=1)
            self._history[:, 0] = x
            self._history[fresh] = x[fresh].unsqueeze(1).expand(-1, self._length, -1)
            out = self._history[:, -1] if self.form == "delay" else self._history.mean(dim=1)
        self._fresh = torch.zeros_like(fresh)
        return out.clone()


_OFFERED_WHEEL_READING: list = []


def _offer_wheel_reading(model: "EncoderNoiseModel") -> None:
    """Offer the encoder model just built to the body-velocity model built next in the same observation group."""
    _OFFERED_WHEEL_READING[:] = [model]


class BodyVelocityNoiseModel(NoiseModel):
    """The body velocity of the control step's one wheel-speed reading: the mecanum kinematics of the output of the
    ``EncoderNoiseModel`` built just before it in the same observation group, so the encoder and body-velocity terms
    carry one noise chain, as on the robot. It takes the encoder model on construction and releases the offer, so a
    group without an encoder model before it fails to build."""

    def __init__(self, noise_model_cfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        if not _OFFERED_WHEEL_READING:
            raise RuntimeError("BodyVelocityNoiseModel needs the group's wheel_encoder_velocities term, with an "
                               "EncoderNoiseModel, before it")
        source = _OFFERED_WHEEL_READING.pop()
        if source._num_envs != num_envs or str(source._device) != str(device):
            raise RuntimeError("BodyVelocityNoiseModel's encoder model belongs to another env")
        self._source = source
        self._inverse_k = torch.as_tensor(INVERSE_KINEMATIC_MATRIX, dtype=torch.float32, device=device)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        ticks = self._source.last_output
        if ticks is None:
            return data
        wheel_rad_s = ticks * ENCODER_TICKS_TO_RADIANS
        return (wheel_rad_s @ self._inverse_k.T.to(wheel_rad_s.dtype))[:, :2]


@configclass
class BodyVelocityNoiseModelCfg(NoiseModelCfg):
    """The body velocity of the group's one wheel-speed reading (``BodyVelocityNoiseModel``); it adds no noise of its
    own."""

    class_type: type = BodyVelocityNoiseModel
    noise_cfg: GaussianNoiseCfg = GaussianNoiseCfg(std=0.0)
