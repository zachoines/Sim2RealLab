"""MDP components for Strafer navigation task.

This module provides custom observation, action, reward, and event functions
for the Strafer mecanum wheel robot navigation environment.

Sensor Models:
- GoBilda 5203 motor encoders (537.7 PPR at output shaft)
- Intel RealSense D555 IMU (BMI055 accelerometer/gyroscope)
- Intel RealSense D555 RGB-D camera

Sim-to-Real Features:
- Realistic sensor noise models (IMU bias drift, encoder quantization, depth holes)
- Motor dynamics (first-order response, command delay)
- Configurable latency and failure modes
"""

# Import standard MDP functions from Isaac Lab
from isaaclab.envs.mdp import *  # noqa: F401, F403

# Custom observations - Encoders
from .observations import (
    wheel_encoder_positions,
    wheel_encoder_velocities,
    wheel_encoder_deltas,
    ENCODER_PPR_OUTPUT_SHAFT,
    RADIANS_TO_ENCODER_TICKS,
)

# Custom observations - IMU (D555 BMI055)
from .observations import (
    imu_angular_velocity,
    imu_linear_acceleration,
    imu_orientation,
    imu_projected_gravity,
)

# Custom observations - Camera (D555)
from .observations import (
    depth_image,
    rgb_image,
)

# Custom observations - Other
from .observations import (
    goal_position_relative,
    goal_distance,
    goal_heading_relative,
    goal_heading_to_goal,
    body_velocity_xy,
    privileged_ground_truth,
    last_action,
)

# Custom noise models for sim-to-real
from .noise_models import (
    # Delay buffer for observation latency
    DelayBuffer,
    # Noise model classes
    IMUNoiseModel,
    IMUNoiseModelCfg,
    EncoderNoiseModel,
    EncoderNoiseModelCfg,
    DepthNoiseModel,
    DepthNoiseModelCfg,
    RGBNoiseModel,
    RGBNoiseModelCfg,
)

# Custom actions
from .actions import MecanumWheelActionCfg

# Custom rewards
from .rewards import (
    goal_reached_reward,
    goal_progress_reward,
    goal_proximity_potential,
    heading_to_goal_reward,
    energy_penalty,
    action_smoothness_penalty,
    collision_penalty,
    collision_sustained_penalty,
    collision_penalty_net,
    collision_sustained_penalty_net,
    speed_near_goal_penalty,
    alive_bonus,
)

# Custom terminations
from .terminations import robot_flipped, goal_reached, sustained_collision

# Custom events
from .events import (
    reset_robot_state,
    reset_robot_state_on_floor,
    reset_robot_proc_room,
    randomize_friction,
    randomize_obstacles,
    randomize_mass,
    randomize_goal_noise,
    randomize_motor_strength,
    randomize_d555_mount_offset,
    randomize_proc_room_difficulty,
)

# Custom commands
from .commands import GoalCommandCfg, GoalCommandProcRoomCfg

# Curriculum learning
from .curriculums import GoalDistanceCurriculum, ObstacleCurriculum, RoomComplexityCurriculum

# Procedural room generation
from .proc_room import generate_proc_room, build_proc_room_collection_cfg
