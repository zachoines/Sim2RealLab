"""Custom command generators for Strafer navigation task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GoalCommand(CommandTerm):
    """Command term for goal-based navigation.

    Generates random goal positions and desired arrival headings.
    The command tensor has shape ``(num_envs, 3)``::

        command[:, 0:2] = (x, y) goal position in world frame
        command[:, 2]   = desired heading angle in radians [-pi, pi]
    """

    cfg: GoalCommandCfg

    def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRLEnv):
        # Initialize goal tensor (x, y, heading) before super().__init__ since reset() is called there
        self._goal = torch.zeros(env.num_envs, 3, device=env.device)

        # Initialize metrics
        self._distance_to_goal = torch.zeros(env.num_envs, device=env.device)
        self._goal_reached_count = torch.zeros(env.num_envs, device=env.device)

        # Call parent constructor (this calls reset which uses _goal)
        super().__init__(cfg, env)

        # Get robot for position reference
        self._robot = env.scene[cfg.asset_name]

        # Store metrics in the metrics dict for logging
        self.metrics["distance_to_goal"] = self._distance_to_goal
        self.metrics["goal_reached_count"] = self._goal_reached_count

        # Track per-env cooldown to avoid resampling on consecutive steps
        self._resample_cooldown = torch.zeros(env.num_envs, device=env.device)

        # Flag set on the step a mid-episode resample occurs.
        # Reward functions should check this to avoid discontinuity penalties.
        self.goal_resampled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The goal command. Shape is (num_envs, 3) for (x, y, heading)."""
        return self._goal

    """
    Implementation of abstract methods
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample goal positions for specified environments.

        Ensures goals are at least ``cfg.min_goal_distance`` from the robot.
        Uses rejection sampling with a fixed number of attempts, then falls
        back to placing the goal at min distance along a random direction.

        Args:
            env_ids: Environment indices to resample commands for.
        """
        num_resets = len(env_ids)
        if num_resets == 0:
            return

        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        # Robot positions for distance check (world frame)
        robot_pos = self._robot.data.root_pos_w[env_ids, :2]
        # Environment origins to convert local goal range to world frame
        env_origins = self._env.scene.env_origins[env_ids, :2]
        min_dist = self.cfg.min_goal_distance

        # Rejection sampling (up to 10 attempts)
        # Goals are sampled in env-local frame then converted to world frame
        accepted = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
        goal_x = torch.zeros(num_resets, device=self.device)
        goal_y = torch.zeros(num_resets, device=self.device)

        for _ in range(10):
            remaining = ~accepted
            n_remaining = remaining.sum().item()
            if n_remaining == 0:
                break

            x = torch.rand(n_remaining, device=self.device)
            y = torch.rand(n_remaining, device=self.device)
            x = x * (self.cfg.goal_range.pos_x[1] - self.cfg.goal_range.pos_x[0]) + self.cfg.goal_range.pos_x[0]
            y = y * (self.cfg.goal_range.pos_y[1] - self.cfg.goal_range.pos_y[0]) + self.cfg.goal_range.pos_y[0]

            # Convert to world frame for distance check against robot
            x_world = x + env_origins[remaining, 0]
            y_world = y + env_origins[remaining, 1]
            candidates = torch.stack([x_world, y_world], dim=-1)
            dist = torch.norm(candidates - robot_pos[remaining], dim=-1)
            far_enough = dist >= min_dist

            # Place accepted candidates (stored in world frame)
            remaining_indices = torch.where(remaining)[0]
            newly_accepted = remaining_indices[far_enough]
            goal_x[newly_accepted] = x_world[far_enough]
            goal_y[newly_accepted] = y_world[far_enough]
            accepted[newly_accepted] = True

        # Fallback: place remaining goals at min_dist in a random direction from robot
        remaining = ~accepted
        if remaining.any():
            n = remaining.sum().item()
            angle = torch.rand(n, device=self.device) * (2.0 * math.pi)
            goal_x[remaining] = robot_pos[remaining, 0] + min_dist * torch.cos(angle)
            goal_y[remaining] = robot_pos[remaining, 1] + min_dist * torch.sin(angle)

        self._goal[env_ids, 0] = goal_x
        self._goal[env_ids, 1] = goal_y

        # Sample random desired heading in [-pi, pi]
        heading_range = self.cfg.goal_range.heading
        self._goal[env_ids, 2] = (
            torch.rand(num_resets, device=self.device)
            * (heading_range[1] - heading_range[0])
            + heading_range[0]
        )

    def _update_command(self):
        """Check for goal reach and resample a new goal mid-episode.

        When multi-goal is enabled, reaching a goal triggers an immediate
        resample instead of terminating the episode. A short cooldown
        (``goal_reach_cooldown_s``) prevents double-counting on consecutive steps.

        Sets ``self.goal_resampled`` flag so reward functions can skip the
        discontinuous step (e.g. goal_progress would see a false distance spike).
        """
        # Clear flag from previous step
        self.goal_resampled[:] = False

        if not self.cfg.multi_goal:
            return

        # Decrement cooldown
        dt = self._env.step_dt
        self._resample_cooldown -= dt
        self._resample_cooldown.clamp_(min=0.0)

        # Check which envs reached the goal and are not on cooldown
        reached = (self._distance_to_goal < self.cfg.goal_reach_threshold) & (
            self._resample_cooldown <= 0.0
        )
        reached_ids = reached.nonzero().flatten()

        if len(reached_ids) > 0:
            self._resample(reached_ids)
            self._resample_cooldown[reached_ids] = self.cfg.goal_reach_cooldown_s
            self.goal_resampled[reached_ids] = True

    def _update_metrics(self):
        """Update metrics based on current state."""
        root_pos = self._robot.data.root_pos_w[:, :2]

        self._distance_to_goal[:] = torch.norm(root_pos - self._goal[:, :2], dim=1)

        reached = self._distance_to_goal < self.cfg.goal_reach_threshold
        self._goal_reached_count += reached.float()

    """
    Debug visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle visibility of goal markers."""
        if debug_vis:
            if not hasattr(self, "_goal_sphere_vis"):
                self._goal_sphere_vis = VisualizationMarkers(self.cfg.goal_sphere_visualizer_cfg)
                self._goal_heading_vis = VisualizationMarkers(self.cfg.goal_heading_visualizer_cfg)
            self._goal_sphere_vis.set_visibility(True)
            self._goal_heading_vis.set_visibility(True)
        else:
            if hasattr(self, "_goal_sphere_vis"):
                self._goal_sphere_vis.set_visibility(False)
                self._goal_heading_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update goal markers each frame.

        Sphere color shifts green → yellow → red based on distance.
        Arrow shows desired arrival heading.
        """
        if not self._robot.is_initialized:
            return

        num_envs = self._goal.shape[0]

        # -- Goal sphere position (slightly above ground so it's visible)
        goal_pos = torch.zeros(num_envs, 3, device=self.device)
        goal_pos[:, 0] = self._goal[:, 0]
        goal_pos[:, 1] = self._goal[:, 1]
        goal_pos[:, 2] = 0.15  # hover above ground plane

        # -- Color selection based on distance: 0=green (close), 1=yellow (mid), 2=red (far)
        # Thresholds: <1m = green, 1-3m = yellow, >3m = red
        marker_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        dist = self._distance_to_goal
        marker_indices[dist > 1.0] = 1  # yellow
        marker_indices[dist > 3.0] = 2  # red

        self._goal_sphere_vis.visualize(
            translations=goal_pos,
            marker_indices=marker_indices,
        )

        # -- Heading cone at goal position, offset above the sphere
        arrow_pos = goal_pos.clone()
        arrow_pos[:, 2] += 0.2  # sit on top of sphere

        zeros = torch.zeros(num_envs, device=self.device)
        # Tip cone from +Z (vertical) to +X (horizontal) with -90° pitch,
        # then apply the heading yaw so the cone points in the goal direction.
        pitch_neg90 = torch.full((num_envs,), -math.pi / 2, device=self.device)
        tip_quat = quat_from_euler_xyz(zeros, pitch_neg90, zeros)
        heading_quat = quat_from_euler_xyz(zeros, zeros, self._goal[:, 2])
        arrow_quat = quat_mul(heading_quat, tip_quat)

        self._goal_heading_vis.visualize(
            translations=arrow_pos,
            orientations=arrow_quat,
        )


# ---------------------------------------------------------------------------
# Marker configurations
# ---------------------------------------------------------------------------

_GOAL_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_sphere",
    markers={
        "goal_close": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "goal_mid": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
        "goal_far": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)

_GOAL_HEADING_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_heading",
    markers={
        "arrow": sim_utils.ConeCfg(
            radius=0.08,
            height=0.3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),
        ),
    },
)


@configclass
class GoalCommandCfg(CommandTermCfg):
    """Configuration for goal command term."""

    class_type: type = GoalCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    resampling_time_range: tuple[float, float] = (10.0, 15.0)
    """Time range for resampling goals (min, max) in seconds."""

    min_goal_distance: float = 1.0
    """Minimum distance (meters) between sampled goal and robot position."""

    multi_goal: bool = True
    """If True, resample a new goal mid-episode when the robot reaches the current one
    instead of relying on episode termination."""

    goal_reach_threshold: float = 0.3
    """Distance (meters) at which the goal is considered reached."""

    goal_reach_cooldown_s: float = 0.5
    """Seconds to wait after resampling before checking goal reach again.
    Prevents double-counting on consecutive steps."""

    debug_vis: bool = False
    """Whether to visualize goal positions."""

    goal_sphere_visualizer_cfg: VisualizationMarkersCfg = _GOAL_SPHERE_CFG
    """Sphere marker config. Three prototypes for distance-based color (green/yellow/red)."""

    goal_heading_visualizer_cfg: VisualizationMarkersCfg = _GOAL_HEADING_CFG
    """Arrow marker config for desired arrival heading."""

    @configclass
    class Ranges:
        """Goal position and heading ranges."""
        pos_x: tuple[float, float] = (-3.0, 3.0)
        pos_y: tuple[float, float] = (-3.0, 3.0)
        heading: tuple[float, float] = (-3.141592653589793, 3.141592653589793)

    goal_range: Ranges = Ranges()
    """Goal position sampling range."""
