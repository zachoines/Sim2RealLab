"""Strafer robot constants -- single source of truth for sim and real.

These values are duplicated in the Isaac Lab simulation configs. Any change here
must be reflected in the corresponding simulation files:
  - source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py
  - source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py
  - source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py
"""

import math

# =============================================================================
# GoBilda Strafer Chassis Geometry
# =============================================================================

# Axles protrude 15.2mm from frame; wheels are 38mm wide (96mm diameter)
WHEEL_RADIUS = 0.048  # meters (96mm diameter mecanum wheel)
WHEEL_WIDTH = 0.038  # meters (38mm mecanum wheel width)
WHEEL_BASE = 0.336  # front-to-rear axle distance (meters, center-to-center)
TRACK_WIDTH = 0.4284  # left-to-right axle distance (meters, center-to-center)

# Chassis frame dimensions
CHASSIS_LENGTH = 0.432  # meters (432mm frame length)
CHASSIS_WIDTH = 0.360  # meters (360mm frame width)
CHASSIS_HEIGHT = 0.040  # meters (~40mm tall aluminum channel)
CHASSIS_GROUND_CLEARANCE = 0.024  # meters (24mm, bottom of frame to floor)

# Camera housing dimensions (Intel RealSense D555)
CAMERA_LENGTH = 0.025  # meters (depth)
CAMERA_WIDTH = 0.090  # meters (width)
CAMERA_HEIGHT = 0.025  # meters (height)

# =============================================================================
# Mass Budget (kilograms)
# =============================================================================

# Measured / spec-sheet values
MASS_KIT = 4.149  # GoBilda Strafer kit (frame + 4 motors + 4 wheels)
MASS_CAMERA = 0.337  # Intel RealSense D555
MASS_ROBOCLAW = 0.061  # RoboClaw ST 2x45A (per unit, ×2 on robot)
MASS_MISC = 0.500  # Estimate: LiPo + wires + buck converter + mounting hw

MASS_TOTAL = MASS_KIT + MASS_CAMERA + 2 * MASS_ROBOCLAW + MASS_MISC  # ~5.108 kg

# Per-component estimates for URDF mass distribution
# GoBilda doesn't break down kit mass; these are approximate
MASS_WHEEL_ASSEMBLY = 0.320  # Per corner: ~200g motor + ~120g mecanum wheel
MASS_FRAME = MASS_KIT - 4 * MASS_WHEEL_ASSEMBLY  # ~2.869 kg (bare frame + hardware)

# Chassis link gets frame + electronics (everything bolted to frame)
MASS_CHASSIS = MASS_FRAME + 2 * MASS_ROBOCLAW + MASS_MISC  # ~3.491 kg

# =============================================================================
# GoBilda 5203 Yellow Jacket Motor (19.2:1 ratio)
# =============================================================================

MOTOR_MAX_RPM = 312.0  # RPM at output shaft
MAX_WHEEL_ANGULAR_VEL = MOTOR_MAX_RPM * 2.0 * math.pi / 60.0  # ~32.67 rad/s

# Encoder: hall-effect quadrature
ENCODER_PPR_OUTPUT_SHAFT = 537.7  # Pulses per revolution at output shaft
ENCODER_PPR_ENCODER_SHAFT = 28  # Pulses per revolution at encoder shaft

# Conversion factors
RADIANS_TO_ENCODER_TICKS = ENCODER_PPR_OUTPUT_SHAFT / (
    2.0 * math.pi
)  # ~85.57 ticks/rad
ENCODER_TICKS_TO_RADIANS = (
    2.0 * math.pi
) / ENCODER_PPR_OUTPUT_SHAFT  # ~0.01169 rad/tick

# =============================================================================
# Derived Velocity Limits
# =============================================================================

MAX_LINEAR_VEL = WHEEL_RADIUS * MAX_WHEEL_ANGULAR_VEL  # ~1.568 m/s
K = (WHEEL_BASE / 2.0) + (TRACK_WIDTH / 2.0)  # ~0.382 m (lever arm)
MAX_ANGULAR_VEL = MAX_LINEAR_VEL / K  # ~4.10 rad/s

# =============================================================================
# Per-Wheel Sign Correction
# Matches strafer_env_cfg.py wheel_axis_signs for [FL, FR, RL, RR]
# =============================================================================

WHEEL_AXIS_SIGNS = (-1.0, 1.0, -1.0, 1.0)

# Wheel ordering: index 0=FL, 1=FR, 2=RL, 3=RR
WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
WHEEL_JOINT_NAMES = ("wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive")

# =============================================================================
# Observation Normalization Scales
# Must match strafer_env_cfg.py lines 225-234
# =============================================================================

IMU_ACCEL_MAX = 156.96  # ±16g in m/s²
IMU_GYRO_MAX = 34.9  # ±2000°/s in rad/s
ENCODER_VEL_MAX = 3000.0  # Max ticks/sec (312 RPM ≈ 2796 ticks/s + margin)
DEPTH_MAX = 6.0  # Max depth in meters

IMU_ACCEL_SCALE = 1.0 / IMU_ACCEL_MAX
IMU_GYRO_SCALE = 1.0 / IMU_GYRO_MAX
ENCODER_VEL_SCALE = 1.0 / ENCODER_VEL_MAX
DEPTH_SCALE = 1.0 / DEPTH_MAX

# =============================================================================
# Camera (Intel RealSense D555)
# =============================================================================

CAMERA_OFFSET_X = 0.20  # meters forward from body_link
CAMERA_OFFSET_Y = 0.0  # meters left from body_link
CAMERA_OFFSET_Z = 0.25  # meters up from body_link

DEPTH_WIDTH = 80  # Policy input resolution (downsampled)
DEPTH_HEIGHT = 60
DEPTH_CLIP_NEAR = 0.4  # meters
DEPTH_CLIP_FAR = 6.0  # meters

# =============================================================================
# RoboClaw Motor Controllers
# =============================================================================

ROBOCLAW_FRONT_ADDRESS = 0x80  # 128 decimal (factory default)
ROBOCLAW_REAR_ADDRESS = 0x81  # 129 decimal (DIP switch 1)
ROBOCLAW_BAUD_RATE = 115200
ROBOCLAW_FRONT_PORT = "/dev/roboclaw0"  # fallback; auto-detect overrides at runtime
ROBOCLAW_REAR_PORT = "/dev/roboclaw1"  # fallback; auto-detect overrides at runtime

# Velocity PID gains (written to RAM on every startup by driver and test scripts).
ROBOCLAW_PID_P = 15000
ROBOCLAW_PID_I = 750
ROBOCLAW_PID_D = 0
ROBOCLAW_QPPS = 2796  # max ticks/sec at 312 RPM (537.7 PPR)
