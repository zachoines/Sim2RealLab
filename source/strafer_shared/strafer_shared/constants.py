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

WHEEL_RADIUS = 0.048        # meters (96mm diameter mecanum wheel)
WHEEL_BASE = 0.304          # front-to-rear axle distance (meters, center-to-center)
TRACK_WIDTH = 0.304         # left-to-right axle distance (meters, center-to-center)

# =============================================================================
# GoBilda 5203 Yellow Jacket Motor (19.2:1 ratio)
# =============================================================================

MOTOR_MAX_RPM = 312.0                                          # RPM at output shaft
MAX_WHEEL_ANGULAR_VEL = MOTOR_MAX_RPM * 2.0 * math.pi / 60.0  # ~32.67 rad/s

# Encoder: hall-effect quadrature
ENCODER_PPR_OUTPUT_SHAFT = 537.7    # Pulses per revolution at output shaft
ENCODER_PPR_ENCODER_SHAFT = 28      # Pulses per revolution at encoder shaft

# Conversion factors
RADIANS_TO_ENCODER_TICKS = ENCODER_PPR_OUTPUT_SHAFT / (2.0 * math.pi)  # ~85.57 ticks/rad
ENCODER_TICKS_TO_RADIANS = (2.0 * math.pi) / ENCODER_PPR_OUTPUT_SHAFT  # ~0.01169 rad/tick

# =============================================================================
# Derived Velocity Limits
# =============================================================================

MAX_LINEAR_VEL = WHEEL_RADIUS * MAX_WHEEL_ANGULAR_VEL          # ~1.568 m/s
K = (WHEEL_BASE / 2.0) + (TRACK_WIDTH / 2.0)                  # ~0.304 m (lever arm)
MAX_ANGULAR_VEL = MAX_LINEAR_VEL / K                           # ~5.16 rad/s

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

IMU_ACCEL_MAX = 156.96      # ±16g in m/s²
IMU_GYRO_MAX = 34.9         # ±2000°/s in rad/s
ENCODER_VEL_MAX = 3000.0    # Max ticks/sec (312 RPM ≈ 2796 ticks/s + margin)
DEPTH_MAX = 6.0             # Max depth in meters

IMU_ACCEL_SCALE = 1.0 / IMU_ACCEL_MAX
IMU_GYRO_SCALE = 1.0 / IMU_GYRO_MAX
ENCODER_VEL_SCALE = 1.0 / ENCODER_VEL_MAX
DEPTH_SCALE = 1.0 / DEPTH_MAX

# =============================================================================
# Camera (Intel RealSense D555)
# =============================================================================

CAMERA_OFFSET_X = 0.20      # meters forward from body_link
CAMERA_OFFSET_Y = 0.0       # meters left from body_link
CAMERA_OFFSET_Z = 0.25      # meters up from body_link

DEPTH_WIDTH = 80             # Policy input resolution (downsampled)
DEPTH_HEIGHT = 60
DEPTH_CLIP_NEAR = 0.4       # meters
DEPTH_CLIP_FAR = 6.0        # meters

# =============================================================================
# RoboClaw Motor Controllers
# =============================================================================

ROBOCLAW_FRONT_ADDRESS = 0x80   # 128 decimal (factory default)
ROBOCLAW_REAR_ADDRESS = 0x81    # 129 decimal (DIP switch 1)
ROBOCLAW_BAUD_RATE = 115200
