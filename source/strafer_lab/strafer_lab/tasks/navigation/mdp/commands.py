"""Custom command generators for Strafer navigation task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GoalCommand(CommandTerm):
    """Command term for goal-based navigation.
    
    Generates random goal positions within a specified range.
    The robot should navigate to reach these goals.
    """

    cfg: GoalCommandCfg

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRLEnv):
        # Initialize goal positions before calling super().__init__ since reset() is called there
        self._goal_pos = torch.zeros(env.num_envs, 2, device=env.device)
        
        # Initialize metrics
        self._distance_to_goal = torch.zeros(env.num_envs, device=env.device)
        self._goal_reached_count = torch.zeros(env.num_envs, device=env.device)
        
        # Call parent constructor (this calls reset which uses _goal_pos)
        super().__init__(cfg, env)
        
        # Get robot for position reference
        self._robot = env.scene[cfg.asset_name]
        
        # Store metrics in the metrics dict for logging
        self.metrics["distance_to_goal"] = self._distance_to_goal
        self.metrics["goal_reached_count"] = self._goal_reached_count

    """
    Properties
    """
    
    @property
    def command(self) -> torch.Tensor:
        """The goal position command. Shape is (num_envs, 2) for (x, y) positions."""
        return self._goal_pos

    """
    Implementation of abstract methods
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample goal positions for specified environments.
        
        Args:
            env_ids: Environment indices to resample commands for.
        """
        num_resets = len(env_ids)
        if num_resets == 0:
            return
            
        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        
        # Sample random goals
        x = torch.rand(num_resets, device=self.device)
        y = torch.rand(num_resets, device=self.device)
        
        # Scale to range
        x = x * (self.cfg.goal_range.pos_x[1] - self.cfg.goal_range.pos_x[0]) + self.cfg.goal_range.pos_x[0]
        y = y * (self.cfg.goal_range.pos_y[1] - self.cfg.goal_range.pos_y[0]) + self.cfg.goal_range.pos_y[0]
        
        self._goal_pos[env_ids, 0] = x
        self._goal_pos[env_ids, 1] = y

    def _update_command(self):
        """Update the command based on current state.
        
        For goal commands, the goal remains static until resampled.
        """
        # Goal positions don't change until resampled
        pass

    def _update_metrics(self):
        """Update metrics based on current state."""
        # Get robot root position
        root_pos = self._robot.data.root_pos_w[:, :2]  # (num_envs, 2) for x, y
        
        # Compute distance to goal
        self._distance_to_goal[:] = torch.norm(root_pos - self._goal_pos, dim=1)
        
        # Count goals reached (within threshold)
        goal_threshold = 0.5  # meters
        self._goal_reached_count += (self._distance_to_goal < goal_threshold).float()


@configclass
class GoalCommandCfg(CommandTermCfg):
    """Configuration for goal command term."""

    class_type: type = GoalCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    resampling_time_range: tuple[float, float] = (10.0, 15.0)
    """Time range for resampling goals (min, max) in seconds."""

    debug_vis: bool = False
    """Whether to visualize goal positions."""

    @configclass
    class Ranges:
        """Goal position ranges."""
        pos_x: tuple[float, float] = (-5.0, 5.0)
        pos_y: tuple[float, float] = (-5.0, 5.0)

    goal_range: Ranges = Ranges()
    """Goal position sampling range."""
