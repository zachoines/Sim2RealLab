"""Sim-to-Real Configuration Contract for Strafer Robot.

This module defines configurable abstraction layers that model real-world
imperfections for robust sim-to-real transfer:

1. TIMING & LATENCY
   - Sensor observation delays (buffering), fixed and per-env sampled
   - Action command delays (network/driver lag)
   - Stream holds: a frame or command that does not advance, in runs

2. ACTUATION MODEL
   - Motor response dynamics (first-order lag)
   - Command delay buffers
   - Velocity/torque limits and slew rates

3. SENSOR NOISE & FAILURES
   - IMU: bias drift, gaussian noise, temperature effects
   - Encoders: quantization, missed ticks, electrical noise
   - Depth camera: holes, noise, range limits, dropped frames, bursty holds
   - RGB camera: motion blur, exposure variation

Usage:
    from strafer_lab.tasks.navigation.sim_real_cfg import (
        SimRealContractCfg,
        REAL_ROBOT_CONTRACT,
        IDEAL_SIM_CONTRACT,
    )
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

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
    - Sensor data latency (observation delay) - per-sensor
    - Command latency (action delay) and command holds

    Latency and holds are separate axes: a latency shifts a modality in time,
    a hold repeats it. A fixed-per-episode latency is a constant offset a
    recurrent policy recalibrates away, which is why the depth latency is also
    sampled per environment.
    """

    # Control frequency
    control_frequency_hz: float = 30.0
    """Nominal control loop frequency in Hz, used to scale IMU noise density
    and bias random-walk steps. The env's tick period is structural
    (``sim.dt`` x ``decimation``); this field does not set it."""

    # Per-sensor observation latency (steps)
    # Different sensors have different processing pipelines and latencies
    imu_latency_steps: int = 0
    """IMU observation latency in control steps. IMU is typically fastest (~1-2ms)."""

    encoder_latency_steps: int = 0
    """Encoder observation latency in control steps. Encoders are fast (~5ms)."""

    depth_latency_steps: int = 1
    """Depth camera latency in control steps. Stereo matching adds delay (~33-66ms)."""

    depth_latency_steps_range: tuple[int, int] | None = None
    """Absolute per-env depth latency band [min, max] in control steps, drawn
    at every reset. None keeps ``depth_latency_steps`` for every env."""

    rgb_latency_steps: int = 1
    """RGB camera latency in control steps. Image processing adds delay (~33-66ms)."""

    # Action latency (policy → actuator)
    action_latency_steps: int = 0
    """Fixed action command latency in physics steps."""

    action_latency_steps_range: tuple[int, int] = (0, 0)
    """Random action latency range [min, max] steps. Sampled per reset."""

    # Command holds (policy → actuator): no new command this control step
    action_hold_fraction_range: tuple[float, float] = (0.0, 0.0)
    """Per-env stationary share of control steps that carry no new command, so
    the chassis re-executes the previous one. Sampled at reset."""

    action_hold_run_range: tuple[float, float] = (1.0, 1.0)
    """Per-env mean command-hold run length [min, max] in control steps."""


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
    max_acceleration_rad_s2: float = 900.0
    """Maximum acceleration for velocity commands in rad/s².
    GoBilda 5203 on 4S LiPo: ~900 rad/s² under light load (frame weight).
    Near-instant response at 32.67 rad/s max → 0-to-full in ~36ms."""

    max_acceleration_range: tuple[float, float] = (900.0, 900.0)
    """Range for randomized max acceleration [min, max] rad/s².
    Models battery state (fresh 16.8V vs depleted 14.0V) and load variation."""


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

    # Stream holds — bursty repeats layered on the memoryless drop above.
    # The drop models the sensor's own dropout; the hold models a transport or
    # consumption stall, whose repeats arrive in runs rather than singly.
    hold_fraction_range: tuple[float, float] = (0.0, 0.0)
    """Per-env stationary share of steps on which the depth frame has not
    advanced. Sampled at reset; a max of 0 leaves the hold inert."""

    hold_run_range: tuple[float, float] = (1.0, 1.0)
    """Per-env mean base hold-run length [min, max] in control steps."""

    hold_burst_weight: float = 0.0
    """Share of hold runs drawn from the long burst component."""

    hold_burst_run_steps: float = 1.0
    """Mean length of a burst hold run, in control steps."""

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
# Localization Configuration
# =============================================================================

# The measured 1x drift class: the map->odom movement recorded on the rig,
# stated as the RMS displacement of the 2-D offset and the heading sigma. The
# tier bands below are multiples of this pair, which keeps them anchored to a
# measurement rather than to a shape someone liked.
MEASURED_DRIFT_POSITION_RMS_M = 0.166
MEASURED_DRIFT_HEADING_DEG = 6.7


@configclass
class LocalizationDriftCfg:
    """Drift of the SLAM frame the policy reads its referent through.

    Two error classes on one axis. The wander is a correlated random walk on
    the SE(2) offset — localization error is integrated, not resampled. The
    jump is a loop closure landing: ``map->odom`` moves in a single step, which
    no random walk produces.

    Both magnitudes scale with one per-environment gain because they were
    measured together and their effects do not separate: the offset component
    perpendicular to the bearing is what produces bearing error, and at the
    nominal subgoal lookahead its contribution is the same size as the heading
    sigma. Sweeping them apart would measure nearly the same thing twice.
    """

    enable_drift: bool = False
    """Enable referent-frame drift on the goal-shaped observations."""

    position_rms_m: float = MEASURED_DRIFT_POSITION_RMS_M
    """RMS displacement of the 2-D offset at gain 1, in metres."""

    heading_sigma_deg: float = MEASURED_DRIFT_HEADING_DEG
    """Stationary heading sigma at gain 1, in degrees."""

    gain_range: tuple[float, float] = (0.0, 0.0)
    """Per-env multiplier band [min, max] on both magnitudes, drawn at reset.
    A max of 0 leaves the wander inert."""

    tau_s: float = 2.0
    """Correlation time of the wander, in seconds. An assumption, not a
    measurement: justified by RTAB-Map's 1-10 Hz map->odom refresh, and the
    bench item that would pin it is still open."""

    jump_rate_hz: float = 0.0
    """Poisson rate of loop-closure snaps. Zero until the rig ride-along
    measures the closure rate."""

    jump_position_range_m: tuple[float, float] = (0.0, 0.0)
    """Snap displacement band [min, max] in metres, in a random direction.
    Ships at zero: the mechanism is in the tree, the distribution is not
    measured yet."""

    jump_heading_range_deg: tuple[float, float] = (0.0, 0.0)
    """Snap heading band [min, max] in degrees, with a random sign. Ships at
    zero for the same reason as the displacement band."""


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
            timing=TimingCfg(depth_latency_steps=1),
            sensors=SensorNoiseCfg(imu=IMUNoiseCfg(enable_noise=True)),
        )
    """
    
    timing: TimingCfg = TimingCfg()
    """Timing and latency configuration."""
    
    actuator: ActuatorModelCfg = ActuatorModelCfg()
    """Actuator dynamics configuration."""
    
    sensors: SensorNoiseCfg = SensorNoiseCfg()
    """Sensor noise configuration."""

    localization: LocalizationDriftCfg = LocalizationDriftCfg()
    """Referent-frame drift configuration. Not a sensor: the depth image and
    the encoders are unaffected by where the map frame thinks the robot is."""

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
            action_latency_steps=0,
            # No per-sensor latency in ideal mode
            imu_latency_steps=0,
            encoder_latency_steps=0,
            depth_latency_steps=0,
            rgb_latency_steps=0,
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
        localization=LocalizationDriftCfg(enable_drift=False),
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
            action_latency_steps=1,  # 33ms command delay
            action_latency_steps_range=(0, 2),  # 0-66ms random
            # Per-sensor latency (realistic values)
            imu_latency_steps=0,  # IMU is very fast (~1-2ms)
            encoder_latency_steps=0,  # Encoders are fast (~5ms)
            depth_latency_steps=1,  # Stereo matching adds delay (~33ms)
            depth_latency_steps_range=(0, 2),  # mean-preserving spread of the above
            rgb_latency_steps=1,  # Image processing adds delay (~33ms)
            # Command holds stay near zero: the residual after the deployed
            # node's inference cadence, not the cadence itself.
            action_hold_fraction_range=(0.0, 0.05),
            action_hold_run_range=(1.0, 1.2),
        ),
        actuator=ActuatorModelCfg(
            enable_motor_dynamics=True,
            motor_time_constant_s=0.05,  # 50ms response
            motor_time_constant_range=(0.03, 0.08),
            max_velocity_rad_s=32.67,  # 312 RPM
            max_acceleration_rad_s2=900.0,
            max_acceleration_range=(700.0, 1100.0),  # ±~20% for battery state
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
                # The fraction is measured: a 23 Hz effective arrival rate
                # against a 30 Hz tick is a 0.22 hold fraction, and the band
                # spans clean to somewhat worse. The run length is not —
                # arrival rate constrains the fraction, not the law behind it.
                hold_fraction_range=(0.0, 0.35),
                hold_run_range=(1.0, 1.6),
            ),
            rgb_camera=RGBCameraNoiseCfg(
                enable_noise=True,
                pixel_noise_std=0.02,
                frame_drop_probability=0.001,
            ),
            failures=SensorFailureCfg(enable_failures=False),
        ),
        localization=LocalizationDriftCfg(
            enable_drift=True,
            # Up to half the measured class: the realistic tier spans clean
            # localization to a fraction of what the rig recorded.
            gain_range=(0.0, 0.5),
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
            action_latency_steps=1,
            action_latency_steps_range=(0, 4),  # Up to 133ms random
            # Per-sensor latency (aggressive values for robustness)
            imu_latency_steps=1,  # Add slight delay to IMU
            encoder_latency_steps=1,  # Add slight delay to encoders
            depth_latency_steps=2,  # Higher camera delay (~66ms)
            depth_latency_steps_range=(1, 3),  # mean-preserving spread of the above
            rgb_latency_steps=2,  # Higher camera delay (~66ms)
            # Wider than realistic on both knobs, still a residual band.
            action_hold_fraction_range=(0.0, 0.25),
            action_hold_run_range=(1.0, 1.5),
        ),
        actuator=ActuatorModelCfg(
            enable_motor_dynamics=True,
            motor_time_constant_s=0.06,  # Slightly slower
            motor_time_constant_range=(0.02, 0.10),  # Wide range
            max_acceleration_rad_s2=900.0,
            max_acceleration_range=(500.0, 1200.0),  # Wide: depleted+loaded → fresh+free
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
                # Same split: the 0.60 fraction is the worst measured arrival
                # rate (12 Hz against a 30 Hz tick); the burst weight and length
                # are an assumed shape, carried from the evaluation profile that
                # hit that fraction. Deploy records arrival counts, not the
                # run-length distribution behind them.
                hold_fraction_range=(0.0, 0.60),
                hold_run_range=(1.0, 2.0),
                hold_burst_weight=0.25,
                hold_burst_run_steps=6.0,
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
        localization=LocalizationDriftCfg(
            enable_drift=True,
            # Past the measured class, the way the temporal bands reach past
            # the measured arrival profiles: the sensitivity arm cost 24-28%
            # of completion at 1x, so the band has to span it rather than sit
            # on its edge.
            gain_range=(0.0, 1.25),
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

    # Get failure probabilities from SensorFailureCfg
    failures = contract.sensors.failures
    failure_prob = failures.imu_failure_probability if failures.enable_failures else 0.0
    stuck_prob = failures.imu_stuck_probability if failures.enable_failures else 0.0

    return IMUNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=std),
        sensor_type='accel',
        control_frequency_hz=sample_rate,
        accel_noise_std=std,
        accel_bias_range=(-contract.sensors.imu.accel_bias_stability,
                          contract.sensors.imu.accel_bias_stability),
        accel_bias_drift_rate=contract.sensors.imu.accel_random_walk,
        output_size=3,
        failure_probability=failure_prob,
        stuck_probability=stuck_prob,
        latency_steps=contract.timing.imu_latency_steps,
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

    # Get failure probabilities from SensorFailureCfg
    failures = contract.sensors.failures
    failure_prob = failures.imu_failure_probability if failures.enable_failures else 0.0
    stuck_prob = failures.imu_stuck_probability if failures.enable_failures else 0.0

    return IMUNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=std),
        sensor_type='gyro',
        control_frequency_hz=sample_rate,
        gyro_noise_std=std,
        gyro_bias_range=(-contract.sensors.imu.gyro_bias_stability,
                         contract.sensors.imu.gyro_bias_stability),
        gyro_bias_drift_rate=contract.sensors.imu.gyro_random_walk,
        output_size=3,
        failure_probability=failure_prob,
        stuck_probability=stuck_prob,
        latency_steps=contract.timing.imu_latency_steps,
    )


def get_encoder_noise(contract: SimRealContractCfg) -> EncoderNoiseModelCfg | None:
    """Get encoder velocity noise config from contract.

    Returns EncoderNoiseModelCfg which generates independent noise per environment,
    and includes quantization and tick error simulation.
    """
    if not contract.sensors.encoders.enable_noise:
        return None

    # Get failure probability from SensorFailureCfg
    failures = contract.sensors.failures
    failure_prob = failures.encoder_failure_probability if failures.enable_failures else 0.0

    return EncoderNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=contract.sensors.encoders.velocity_noise_std),
        enable_quantization=contract.sensors.encoders.enable_quantization,
        velocity_noise_std=contract.sensors.encoders.velocity_noise_std,
        missed_tick_prob=contract.sensors.encoders.missed_tick_probability,
        extra_tick_prob=contract.sensors.encoders.extra_tick_probability,
        failure_probability=failure_prob,
        latency_steps=contract.timing.encoder_latency_steps,
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

    # Get failure probability from SensorFailureCfg
    failures = contract.sensors.failures
    failure_prob = failures.camera_failure_probability if failures.enable_failures else 0.0

    return DepthNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=noise_at_1m),
        baseline_m=cfg.baseline_m,
        focal_length_px=cfg.focal_length_px,
        disparity_noise_px=cfg.disparity_noise_px,
        hole_probability=cfg.hole_probability,
        min_range=cfg.min_range_m,
        max_range=cfg.max_range_m,
        frame_drop_prob=cfg.frame_drop_probability,
        hold_fraction_range=cfg.hold_fraction_range,
        hold_run_range=cfg.hold_run_range,
        hold_burst_weight=cfg.hold_burst_weight,
        hold_burst_run_steps=cfg.hold_burst_run_steps,
        failure_probability=failure_prob,
        latency_steps=contract.timing.depth_latency_steps,
        latency_steps_range=contract.timing.depth_latency_steps_range,
    )


def get_rgb_noise(contract: SimRealContractCfg) -> RGBNoiseModelCfg | None:
    """Get RGB camera noise config from contract.

    Returns RGBNoiseModelCfg which generates independent noise per environment.
    """
    if not contract.sensors.rgb_camera.enable_noise:
        return None

    # Get failure probability from SensorFailureCfg
    failures = contract.sensors.failures
    failure_prob = failures.camera_failure_probability if failures.enable_failures else 0.0

    return RGBNoiseModelCfg(
        noise_cfg=GaussianNoiseCfg(std=contract.sensors.rgb_camera.pixel_noise_std),
        pixel_noise_std=contract.sensors.rgb_camera.pixel_noise_std,
        brightness_range=contract.sensors.rgb_camera.exposure_variation_range,
        frame_drop_prob=contract.sensors.rgb_camera.frame_drop_probability,
        failure_probability=failure_prob,
        latency_steps=contract.timing.rgb_latency_steps,
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
        "hold_fraction_range": contract.timing.action_hold_fraction_range,
        "hold_run_range": contract.timing.action_hold_run_range,
        "max_acceleration_rad_s2": contract.actuator.max_acceleration_rad_s2,
        "max_acceleration_range": contract.actuator.max_acceleration_range,
    }


def get_subgoal_drift_params(contract: SimRealContractCfg) -> dict | None:
    """Get referent-frame drift parameters from contract.

    Returns the params dict for the ``randomize_subgoal_drift`` event term, or
    ``None`` on a tier that carries no drift — the caller omits the term
    entirely rather than installing an inert one, so an ideal-tier env is
    structurally identical to a tree without this mechanism.
    """
    drift = contract.localization
    if not drift.enable_drift:
        return None
    return {
        "position_rms_m": drift.position_rms_m,
        "heading_sigma_deg": drift.heading_sigma_deg,
        "gain_range": drift.gain_range,
        "tau_s": drift.tau_s,
        "jump_rate_hz": drift.jump_rate_hz,
        "jump_position_range_m": drift.jump_position_range_m,
        "jump_heading_range_deg": drift.jump_heading_range_deg,
    }
