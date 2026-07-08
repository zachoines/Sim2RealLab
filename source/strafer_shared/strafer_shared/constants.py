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
TRACK_WIDTH = 0.4132  # left-to-right axle distance (meters, center-to-center)

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
K = (WHEEL_BASE / 2.0) + (TRACK_WIDTH / 2.0)  # ~0.375 m (lever arm)
MAX_ANGULAR_VEL = MAX_LINEAR_VEL / K  # ~4.19 rad/s

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
DEPTH_MIN = 0.4  # Min usable depth in meters (D555 stereo baseline limit)
DEPTH_MAX = 6.0  # Max depth in meters

BODY_VEL_MAX = 2.0  # Max body velocity in m/s (robot tops ~1.57 m/s, with margin)
GOAL_DIST_MAX = 10.0  # Max goal distance in meters (goals sampled ±5m)

IMU_ACCEL_SCALE = 1.0 / IMU_ACCEL_MAX
IMU_GYRO_SCALE = 1.0 / IMU_GYRO_MAX
ENCODER_VEL_SCALE = 1.0 / ENCODER_VEL_MAX
DEPTH_SCALE = 1.0 / DEPTH_MAX
BODY_VEL_SCALE = 1.0 / BODY_VEL_MAX
GOAL_DIST_SCALE = 1.0 / GOAL_DIST_MAX
HEADING_SCALE = 1.0 / 3.141592653589793  # Heading error [-pi, pi] -> [-1, 1]

# =============================================================================
# Camera (Intel RealSense D555)
# =============================================================================

CAMERA_OFFSET_X = 0.20  # meters forward from body_link
CAMERA_OFFSET_Y = 0.0  # meters left from body_link
CAMERA_OFFSET_Z = 0.25  # meters up from body_link

# Policy depth input resolution (downsampled from the 640x360 perception
# stream). 80x45 is 16:9 — deliberately matching the real D555 / perception
# camera aspect. Isaac Sim / RTX derives a camera's vertical FOV from its
# render resolution aspect ratio (square pixels) and IGNORES the authored
# vertical_aperture — a Kit probe confirmed this directly (verticalAperture on
# the prim changed but the rendered depth did not). So the ONLY lever on the
# sim policy camera's vertical FOV is its resolution: at 16:9 the derived VFOV
# is 2*atan((3.68*45/80)/(2*1.93)) = 56.4 deg, matching the real sensor and the
# perception camera. A 4:3 80x60 render spans ~71 deg, so training depth would
# be ~1.26x vertically magnified vs the 16:9 stream deployment block-averages
# and feeds the policy. Deployment must downsample 640x360 -> 80x45 to match.
DEPTH_WIDTH = 80
DEPTH_HEIGHT = 45
DEPTH_CLIP_NEAR = 0.4  # meters — D555 stereo min range (real hardware limit)
DEPTH_CLIP_FAR = 6.0  # meters
DEPTH_SIM_CLIP_NEAR = 0.01  # meters — sim renders below D555 min range
DEPTH_NEARFIELD_FILL = 0.2  # meters — replacement value for sub-0.4m pixels

# Sim-only renderer frustum far clip. Distinct from DEPTH_CLIP_FAR — the
# depth-sensor 6 m limit is enforced in software in observations.depth_image
# (and is irrelevant for RGB), whereas the frustum clip is a renderer-level
# cull that affects EVERY channel the camera emits (RGB, depth, semantic,
# bbox). Using DEPTH_CLIP_FAR for the frustum cuts RGB at 6 m too, leaving
# the policy / VLM blind to anything in the room beyond that. 50 m is well
# beyond any indoor scene we run; pinhole-camera frustum culling is cheap
# so making this generous costs nothing.
D555_RENDER_FAR_CLIP_M = 50.0

# D555 native capture resolution — used by the Jetson perception pipeline
# (/d555/color/image_sync, /d555/aligned_depth_to_color/image_sync, VLM
# grounding client) and mirrored by the Isaac Sim perception camera
# (strafer_lab.tasks.navigation.d555_cfg). Distinct from DEPTH_WIDTH /
# DEPTH_HEIGHT above, which is the 80×45 DOWNSAMPLED policy input only.
# Both are 16:9, so the downsample is a clean 8× block-average (640/80,
# 360/45) that preserves the sensor's vertical FOV into the policy grid.
PERCEPTION_WIDTH = 640
PERCEPTION_HEIGHT = 360

# D555 lens / sensor specs, from Intel's datasheet. Values are in
# millimeters, which matches both the real-world spec and the unit
# convention Isaac Sim's PinholeCameraCfg expects for focal_length /
# horizontal_aperture.
D555_FOCAL_LENGTH_MM = 1.93
D555_HORIZONTAL_APERTURE_MM = 3.68

# D555 native frame rate. Both the color and the aligned depth streams
# run at this rate on real hardware; the sim TiledCameraCfg uses the
# derived period.
CAMERA_HZ = 30
CAMERA_UPDATE_PERIOD_S = 1.0 / CAMERA_HZ

# Bosch BMI055 IMU (integrated into the D555) native rate.
IMU_HZ = 200
IMU_UPDATE_PERIOD_S = 1.0 / IMU_HZ

# =============================================================================
# Mapping / Navigation
# =============================================================================

MAP_RESOLUTION = 0.05  # meters per cell — shared by RTAB-Map grid and Nav2 costmap

# Rolling-subgoal lookahead distance (meters). The single source of truth for
# the pure-pursuit lookahead used on BOTH sides of the planner-follows seam:
# the subgoal-tracking training env advances its subgoal this far along the
# planned path's arc length, and the hybrid backend that follows Nav2's path
# at deployment must select its subgoal at the same distance. The policy
# observes only the resulting subgoal pose (relative position / distance /
# bearing), so this distance — not the path's waypoint spacing — is what must
# match across train and deploy. Pin both references here so the value cannot
# drift between the two lanes.
SUBGOAL_LOOKAHEAD_M = 1.0

# Autonomous navigation velocity scaling (fraction of hardware max).
# Hardware max is ~1.57 m/s / ~4.10 rad/s — too fast for indoor autonomy.
NAV_VEL_SCALE = 0.5  # forward/strafe: 50% of MAX_LINEAR_VEL (~0.78 m/s)
NAV_REVERSE_SCALE = 0.7  # reverse: 70% of nav forward (D555 is front-facing, no rear sensing)

NAV_LINEAR_VEL = round(MAX_LINEAR_VEL * NAV_VEL_SCALE, 4)
NAV_ANGULAR_VEL = round(MAX_ANGULAR_VEL * NAV_VEL_SCALE, 4)
NAV_REVERSE_VEL = round(NAV_LINEAR_VEL * NAV_REVERSE_SCALE, 4)

# =============================================================================
# Policy step rate
# =============================================================================
# Single source of truth for the trained-policy step period. The Isaac Lab
# task config (strafer_env_cfg.py: cfg.sim.dt, cfg.decimation) and the
# Jetson inference node both consume these so sim and real cannot drift.

POLICY_SIM_DT = 1.0 / 120.0
POLICY_DECIMATION = 4
POLICY_PERIOD_S = POLICY_SIM_DT * POLICY_DECIMATION  # 1/30 s = 30 Hz

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

# =============================================================================
# ROS2 Topic + Frame ID Conventions
# =============================================================================
# Canonical topic and frame names for the Strafer ROS2 graph. Both the real
# Jetson stack (strafer_ros/*) and the Isaac Sim bridge (strafer_lab.bridge)
# must agree on these strings so sim-in-the-loop consumers (Nav2, RTAB-Map,
# VLM grounding clients) see the same wire format regardless of source.

TOPIC_COLOR_IMAGE = "/d555/color/image_raw"
TOPIC_COLOR_CAMERA_INFO = "/d555/color/camera_info"
TOPIC_DEPTH_IMAGE = "/d555/depth/image_rect_raw"
TOPIC_DEPTH_CAMERA_INFO = "/d555/depth/camera_info"
TOPIC_ODOM = "/strafer/odom"
TOPIC_CMD_VEL = "/cmd_vel"
# Wheel encoder velocities/positions. The real robot's driver publishes this
# from RoboClaw encoder reads; the sim bridge publishes the sim articulation's
# wheel-joint state on the same topic so the inference obs pipeline reconstructs
# the same wheel-FK body velocity from either source.
TOPIC_JOINT_STATES = "/strafer/joint_states"
# Fused IMU. On the real robot this is imu_filter_madgwick's output; the sim
# bridge publishes the D555 IMU sensor's body-frame accel/gyro on the same
# topic so the policy sees IMU on the same contract in sim and on hardware.
TOPIC_IMU_FILTERED = "/d555/imu/filtered"
# Sim-time clock published by the Isaac Sim bridge. The Jetson side sets
# `use_sim_time:=True` on every ROS 2 node launched by
# `bringup_sim_in_the_loop.launch.py` so the whole cross-host stack shares
# the bridge's monotonic sim clock and TF chains compose without stamp
# conflicts between wall-time and sim-time publishers.
TOPIC_CLOCK = "/clock"

FRAME_ODOM = "odom"
FRAME_BASE_LINK = "base_link"
FRAME_D555_LINK = "d555_link"
FRAME_D555_COLOR_OPTICAL = "d555_color_optical_frame"
