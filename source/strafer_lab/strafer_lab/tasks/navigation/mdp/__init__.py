"""MDP components for Strafer navigation task.

This module provides custom observation, action, reward, and event functions
for the Strafer mecanum wheel robot navigation environment.

Sensor Models:
- GoBilda 5203 motor encoders (537.7 PPR at output shaft)
- Intel RealSense D555 IMU (BMI055 accelerometer/gyroscope)
- Intel RealSense D555 RGB-D camera
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

# Custom observations - Legacy (backward compatibility)
from .observations import (
    base_lin_vel,
    base_ang_vel,
    projected_gravity,
    goal_position_relative,
    last_action,
)

# Custom actions
from .actions import MecanumWheelActionCfg

# Custom rewards
from .rewards import (
    goal_reached_reward,
    goal_progress_reward,
    heading_to_goal_reward,
    energy_penalty,
    action_smoothness_penalty,
)

# Custom terminations
from .terminations import robot_flipped

# Custom events
from .events import reset_robot_state, randomize_friction

# Custom commands
from .commands import GoalCommandCfg
