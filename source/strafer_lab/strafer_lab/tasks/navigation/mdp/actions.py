"""Custom action terms for Strafer mecanum wheel robot."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MecanumWheelAction(ActionTerm):
    """Action term for controlling mecanum wheel velocities.
    
    This action term converts velocity commands to individual wheel velocities
    for a 4-wheel mecanum drive robot.
    
    The action space is [vx, vy, omega] representing:
    - vx: Forward/backward velocity
    - vy: Left/right (strafe) velocity
    - omega: Rotational velocity
    
    These are converted to individual wheel velocities using mecanum kinematics.
    """

    cfg: MecanumWheelActionCfg
    _asset: object  # Articulation

    def __init__(self, cfg: MecanumWheelActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get the robot articulation
        self._asset = env.scene[cfg.asset_name]
        
        # Find wheel joint indices
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        
        # Mecanum kinematics parameters
        # Wheel radius and robot dimensions (adjust based on actual robot)
        self._wheel_radius = 0.05  # meters
        self._wheel_base = 0.3    # distance between front and rear wheels
        self._track_width = 0.25  # distance between left and right wheels
        
        # Kinematic matrix for mecanum wheels
        # Maps [vx, vy, omega] to [fl, fr, rl, rr] wheel velocities
        L = self._wheel_base / 2
        W = self._track_width / 2
        self._kinematic_matrix = torch.tensor([
            [1, -1, -(L + W)],  # Front left
            [1,  1,  (L + W)],  # Front right
            [1,  1, -(L + W)],  # Rear left
            [1, -1,  (L + W)],  # Rear right
        ], dtype=torch.float32, device=env.device) / self._wheel_radius

    @property
    def action_dim(self) -> int:
        """Dimension of the action space: [vx, vy, omega]."""
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions before processing."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed wheel velocity commands."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Convert [vx, vy, omega] to wheel velocities."""
        self._raw_actions = actions
        
        # Scale actions
        scaled_actions = actions * self.cfg.scale
        
        # Convert to wheel velocities using mecanum kinematics
        # wheel_vels shape: (num_envs, 4)
        self._processed_actions = torch.matmul(scaled_actions, self._kinematic_matrix.T)

    def apply_actions(self):
        """Apply wheel velocity targets to the robot."""
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class MecanumWheelActionCfg(ActionTermCfg):
    """Configuration for mecanum wheel action term."""

    class_type: type = MecanumWheelAction

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    joint_names: list[str] = MISSING
    """Regex pattern(s) for wheel joint names."""

    scale: float = 1.0
    """Scale factor for velocity commands."""
