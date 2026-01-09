"""MDP components for Strafer navigation task.

This module provides custom observation, action, reward, and event functions
for the Strafer mecanum wheel robot navigation environment.
"""

# Import standard MDP functions from Isaac Lab
from isaaclab.envs.mdp import *  # noqa: F401, F403

# Custom observations
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
