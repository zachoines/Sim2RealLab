"""Custom command generators for Strafer navigation task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from strafer_shared.constants import MAP_RESOLUTION, SUBGOAL_LOOKAHEAD_M

from ..path_planner import PathCursor, PathPlanningError, perturb_waypoints, plan_path
from .proc_room import GRID_RES, GRID_SIZE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import warp as wp


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

        # Build spawn points tensor if provided (for Infinigen goal placement)
        self._spawn_points: torch.Tensor | None = None
        if cfg.spawn_points_xy:
            self._spawn_points = torch.tensor(cfg.spawn_points_xy, device=env.device, dtype=torch.float32)

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
        When ``spawn_points_xy`` is configured, samples from precomputed floor
        positions (Infinigen). Otherwise uses uniform box sampling with
        rejection for min distance, falling back to min distance along a
        random direction.

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
        robot_pos = wp.to_torch(self._robot.data.root_pos_w)[env_ids, :2]
        # Environment origins to convert local goal range to world frame
        env_origins = self._env.scene.env_origins[env_ids, :2]
        min_dist = self.cfg.min_goal_distance

        if self._spawn_points is not None:
            # Sample from precomputed spawn points (env-local frame)
            goal_x, goal_y = self._resample_from_spawn_points(
                num_resets, robot_pos, env_origins, min_dist
            )
        else:
            # Uniform box sampling with rejection
            goal_x, goal_y = self._resample_from_range(
                num_resets, robot_pos, env_origins, min_dist
            )

        self._goal[env_ids, 0] = goal_x
        self._goal[env_ids, 1] = goal_y

        # NOTE: Do NOT reset _goal_reached_count here — _resample_command is
        # called both on episode reset AND mid-episode goal resampling.  The
        # base CommandTerm.reset() already zeros all metrics (including this
        # counter) on episode reset.  Zeroing here would wipe the counter every
        # time a goal is reached mid-episode, preventing curriculum advancement.

        # Sample random desired heading in [-pi, pi]
        heading_range = self.cfg.goal_range.heading
        self._goal[env_ids, 2] = (
            torch.rand(num_resets, device=self.device)
            * (heading_range[1] - heading_range[0])
            + heading_range[0]
        )

    def _resample_from_spawn_points(
        self,
        num_resets: int,
        robot_pos: torch.Tensor,
        env_origins: torch.Tensor,
        min_dist: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample goals from precomputed spawn points with min-distance rejection."""
        pts = self._spawn_points
        accepted = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
        goal_x = torch.zeros(num_resets, device=self.device)
        goal_y = torch.zeros(num_resets, device=self.device)

        for _ in range(10):
            remaining = ~accepted
            n_remaining = remaining.sum().item()
            if n_remaining == 0:
                break

            indices = torch.randint(0, len(pts), (n_remaining,), device=self.device)
            xy_local = pts[indices]  # (n_remaining, 2)

            x_world = xy_local[:, 0] + env_origins[remaining, 0]
            y_world = xy_local[:, 1] + env_origins[remaining, 1]
            candidates = torch.stack([x_world, y_world], dim=-1)
            dist = torch.norm(candidates - robot_pos[remaining], dim=-1)
            far_enough = dist >= min_dist

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

        return goal_x, goal_y

    def _resample_from_range(
        self,
        num_resets: int,
        robot_pos: torch.Tensor,
        env_origins: torch.Tensor,
        min_dist: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample goals uniformly within goal_range with min-distance rejection."""
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

        return goal_x, goal_y

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
        root_pos = wp.to_torch(self._robot.data.root_pos_w)[:, :2]

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

    spawn_points_xy: list[list[float]] | None = None
    """Precomputed valid floor positions for goal sampling (env-local frame).
    When provided (Infinigen), goals are sampled from these points instead of
    the uniform goal_range box. Set from scenes_metadata.json at config time."""

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


class GoalCommandProcRoom(GoalCommand):
    """Goal command that samples from dynamic per-env BFS reachable points.

    Reads spawn points from ``env._proc_room_spawn_pts`` populated by
    ``generate_proc_room``. Falls back to min-distance placement when
    rejection sampling fails.
    """

    cfg: GoalCommandProcRoomCfg

    def __init__(self, cfg: GoalCommandProcRoomCfg, env: ManagerBasedRLEnv):
        # Skip building static spawn points — we use dynamic ones
        cfg.spawn_points_xy = None
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        num_resets = len(env_ids)
        if num_resets == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        robot_pos = wp.to_torch(self._robot.data.root_pos_w)[env_ids, :2]
        env_origins = self._env.scene.env_origins[env_ids, :2]
        min_dist = self.cfg.min_goal_distance

        # Read dynamic per-env spawn points
        spawn_pts = self._env._proc_room_spawn_pts[env_ids]       # (B, K, 2)
        spawn_count = self._env._proc_room_spawn_count[env_ids]    # (B,)

        accepted = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
        goal_x = torch.zeros(num_resets, device=self.device)
        goal_y = torch.zeros(num_resets, device=self.device)

        max_idx = spawn_count.clamp(min=1)

        for _ in range(10):
            remaining = ~accepted
            n_remaining = remaining.sum().item()
            if n_remaining == 0:
                break

            rem_idx = torch.where(remaining)[0]
            # Sample random point index per env, bounded by that env's count
            rand_frac = torch.rand(n_remaining, device=self.device)
            pt_idx = (rand_frac * max_idx[remaining].float()).long()
            pt_idx = pt_idx.clamp(max=spawn_pts.shape[1] - 1)

            # Gather selected XY (env-local frame)
            xy_local = spawn_pts[remaining][
                torch.arange(n_remaining, device=self.device), pt_idx
            ]  # (n_remaining, 2)

            x_world = xy_local[:, 0] + env_origins[remaining, 0]
            y_world = xy_local[:, 1] + env_origins[remaining, 1]
            candidates = torch.stack([x_world, y_world], dim=-1)
            dist = torch.norm(candidates - robot_pos[remaining], dim=-1)
            far_enough = dist >= min_dist

            newly_accepted = rem_idx[far_enough]
            goal_x[newly_accepted] = x_world[far_enough]
            goal_y[newly_accepted] = y_world[far_enough]
            accepted[newly_accepted] = True

        # Fallback: place at min_dist in random direction
        remaining = ~accepted
        if remaining.any():
            n = remaining.sum().item()
            angle = torch.rand(n, device=self.device) * (2.0 * math.pi)
            goal_x[remaining] = robot_pos[remaining, 0] + min_dist * torch.cos(angle)
            goal_y[remaining] = robot_pos[remaining, 1] + min_dist * torch.sin(angle)

        self._goal[env_ids, 0] = goal_x
        self._goal[env_ids, 1] = goal_y

        # NOTE: Do NOT reset _goal_reached_count here — see GoalCommand._resample_command.

        # Random heading
        heading_range = self.cfg.goal_range.heading
        self._goal[env_ids, 2] = (
            torch.rand(num_resets, device=self.device)
            * (heading_range[1] - heading_range[0])
            + heading_range[0]
        )


@configclass
class GoalCommandProcRoomCfg(GoalCommandCfg):
    """Configuration for goal command in procedural rooms."""

    class_type: type = GoalCommandProcRoom


# ---------------------------------------------------------------------------
# Subgoal command (path-following)
# ---------------------------------------------------------------------------

# Env-local XY of the occupancy grid's (0, 0) cell corner (grid centered on
# the env origin, matching the procedural-room generator's rasterization).
_PROC_ROOM_GRID_ORIGIN = (
    -GRID_SIZE * GRID_RES / 2.0,
    -GRID_SIZE * GRID_RES / 2.0,
)


def dwell_step(
    counter: torch.Tensor,
    within_radius: torch.Tensor,
    below_speed: torch.Tensor,
    dwell_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the per-env parking-dwell counter by one control step.

    Success is *parking*, not touching: an env's path counts complete only after
    the robot has held inside the completion radius at low speed for
    ``dwell_steps`` consecutive control steps. Each env satisfying both
    conditions this step (``within_radius`` and ``below_speed``) increments its
    counter; any env that breaks either condition resets to zero, so the dwell
    must be unbroken.

    Pure function of its arguments — no env or module state — so its truth table
    is unit-testable on the Kit-free path.

    Args:
        counter: (num_envs,) int, consecutive holding-step count from last step.
        within_radius: (num_envs,) bool, robot inside the completion radius.
        below_speed: (num_envs,) bool, robot body speed at/below the dwell cap.
        dwell_steps: consecutive holding steps required to fire completion.

    Returns:
        ``(new_counter, parked)``: the advanced counter and a bool mask that is
        True for envs whose counter has reached ``dwell_steps``.
    """
    hold = within_radius & below_speed
    new_counter = torch.where(hold, counter + 1, torch.zeros_like(counter))
    parked = new_counter >= dwell_steps
    return new_counter, parked


class SubgoalCommand(CommandTerm):
    """Command term emitting a rolling subgoal along a planned path.

    At episode reset, plans a collision-free path from the robot's spawn
    position to a sampled reachable goal on the env's inflated occupancy
    grid (built by the procedural-room generator), then per tick projects
    the robot onto the path and emits the pose ``lookahead_m`` ahead of the
    projection along arc length. The command tensor has shape
    ``(num_envs, 3)``::

        command[:, 0:2] = (x, y) subgoal position in world frame
        command[:, 2]   = path tangent heading (rad) at the subgoal

    Deployment contract: the policy observes only the resulting subgoal pose
    (relative position / distance / bearing), never the waypoints or the
    cursor. So transfer to a real planner reduces to a single requirement —
    a planner-follows backend tracking an externally published path (Nav2's
    ``/plan``) must select its subgoal with the same rule used here: closest
    point on the path, advanced ``lookahead_m`` along arc length. The shared
    lookahead distance is ``SUBGOAL_LOOKAHEAD_M``; the path's waypoint spacing
    is *not* part of the contract, because the policy never sees the
    waypoints. ``lookahead_randomization_m`` widens the requirement from a
    single distance the deployed selector must hit exactly into a band the
    policy is trained to track across, and ``waypoint_noise_std_m`` covers the
    residual train/deploy planner-shape disagreement.

    Public per-tick state read by path-tracking rewards and terminations:

    - ``cross_track_error`` (num_envs,): distance to the closest path point.
    - ``along_track_progress`` (num_envs,): monotonic arc-length advance
      since the previous step (zero on the step a new path is installed).
    - ``path_complete`` (num_envs,) bool: robot has *parked* — held within
      ``dwell_radius_m`` of the path's final point at/under
      ``dwell_speed_max_m_s`` for ``dwell_steps`` consecutive control steps.
    """

    cfg: SubgoalCommandCfg

    def __init__(self, cfg: SubgoalCommandCfg, env: ManagerBasedRLEnv):
        device = env.device
        # Buffers used by reset() must exist before super().__init__.
        self._subgoal = torch.zeros(env.num_envs, 3, device=device)
        self._path_cursor = PathCursor(env.num_envs, cfg.max_path_points, device)

        # Public per-tick path-tracking state.
        self.cross_track_error = torch.zeros(env.num_envs, device=device)
        self.along_track_progress = torch.zeros(env.num_envs, device=device)
        self.path_complete = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        # Consecutive control steps the robot has held parked at the path end
        # (see ``dwell_step``); drives the dwell-gated ``path_complete`` flag.
        self.dwell_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=device)

        self._distance_to_subgoal = torch.zeros(env.num_envs, device=device)
        # 1.0 on envs whose last plan fell back to a straight segment.
        self._path_fallback = torch.zeros(env.num_envs, device=device)

        # Per-env lookahead distance. Constant at ``lookahead_m`` unless
        # ``lookahead_randomization_m`` is set, in which case it is resampled
        # per env at each path reset. Held as a plain buffer (NOT a metric):
        # the base reset() zeros every metric tensor, which would wipe the
        # lookahead the same step it is sampled.
        self._lookahead = torch.full((env.num_envs,), cfg.lookahead_m, device=device)

        # Waypoint-noise / lookahead RNG, seeded from the torch seed for
        # reproducibility.
        self._np_rng = np.random.default_rng(torch.initial_seed() % (2**32))

        super().__init__(cfg, env)

        self._robot = env.scene[cfg.asset_name]

        self.metrics["cross_track_error"] = self.cross_track_error
        self.metrics["along_track_progress"] = self.along_track_progress
        self.metrics["distance_to_subgoal"] = self._distance_to_subgoal
        self.metrics["path_fallback"] = self._path_fallback

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The subgoal command. Shape is (num_envs, 3) for (x, y, heading)."""
        return self._subgoal

    @property
    def path_cursor(self) -> PathCursor:
        """The underlying per-env path buffers (read-only consumers only)."""
        return self._path_cursor

    """
    Implementation of abstract methods
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Plan a fresh path per env and emit its initial subgoal.

        Reads the per-env free-space grids and reachable spawn points
        populated by ``generate_proc_room`` (which runs as a reset event
        before command resampling). Goal candidates are drawn from the
        BFS-reachable spawn set at least ``min_goal_distance`` from the
        robot; if every candidate fails to plan, falls back to a straight
        segment to the last candidate (flagged in the ``path_fallback``
        metric — the BFS solvability guarantee makes this rare).
        """
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        env_ids = env_ids.to(dtype=torch.long)

        robot_xy_w = wp.to_torch(self._robot.data.root_pos_w)[:, :2]
        env_origins = self._env.scene.env_origins[:, :2]
        start_local = (robot_xy_w[env_ids] - env_origins[env_ids]).cpu().numpy()
        origins_np = env_origins[env_ids].cpu().numpy()

        free_grids = self._env._proc_room_free_space[env_ids].cpu().numpy()
        spawn_pts = self._env._proc_room_spawn_pts[env_ids].cpu().numpy()
        spawn_count = self._env._proc_room_spawn_count[env_ids].cpu().numpy()

        spacing = self.cfg.path_spacing_m
        paths_world: list[torch.Tensor] = []
        for k in range(len(env_ids)):
            env_idx = int(env_ids[k])
            count = max(int(spawn_count[k]), 1)
            candidates = spawn_pts[k, :count]
            dists = np.linalg.norm(candidates - start_local[k], axis=-1)
            far_idx = np.flatnonzero(dists >= self.cfg.min_goal_distance)
            if len(far_idx) == 0:
                far_idx = np.array([int(dists.argmax())])
            order = self._np_rng.permutation(far_idx)[: self.cfg.max_goal_attempts]

            path = None
            goal = candidates[order[-1]]
            for cand_idx in order:
                goal = candidates[cand_idx]
                try:
                    path = plan_path(
                        start_local[k],
                        goal,
                        free_grids[k],
                        grid_res=GRID_RES,
                        grid_origin_xy=_PROC_ROOM_GRID_ORIGIN,
                        discretization_m=spacing,
                    )
                    break
                except PathPlanningError:
                    continue

            if path is None:
                # Straight-segment fallback; flagged for monitoring.
                path = np.stack([start_local[k], goal]).astype(np.float32)
                self._path_fallback[env_idx] = 1.0
            else:
                self._path_fallback[env_idx] = 0.0
                path = perturb_waypoints(
                    path,
                    self.cfg.waypoint_noise_std_m,
                    free_grids[k],
                    grid_res=GRID_RES,
                    grid_origin_xy=_PROC_ROOM_GRID_ORIGIN,
                    rng=self._np_rng,
                )

            paths_world.append(torch.as_tensor(path + origins_np[k], dtype=torch.float32))

        self._path_cursor.set_paths(env_ids, paths_world)

        # Resample the per-env lookahead distance when randomization is on, so
        # the policy learns to track a subgoal at any distance in the band
        # rather than only the deployment default. The deployed selector then
        # needs only to land inside the band, not hit one exact distance.
        band = self.cfg.lookahead_randomization_m
        if band is not None:
            lo, hi = band
            self._lookahead[env_ids] = (
                torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
            )

        # Clear tracking state and emit the initial subgoal so observations
        # computed right after reset see a valid command. The cursor was just
        # rewound, so this restricted update cannot leak progress into the
        # next step's along-track reward.
        self.path_complete[env_ids] = False
        self.dwell_counter[env_ids] = 0
        self.along_track_progress[env_ids] = 0.0
        state = self._path_cursor.update(
            robot_xy_w, self._lookahead, env_ids=env_ids
        )
        self._subgoal[env_ids, :2] = state.subgoal_xy
        self._subgoal[env_ids, 2] = state.subgoal_heading
        self.cross_track_error[env_ids] = state.cross_track

    def _update_command(self):
        """Advance the cursor to the robot's projection and refresh the subgoal,
        cross-track / along-track state, and the dwell-gated completion flag."""
        robot_xy_w = wp.to_torch(self._robot.data.root_pos_w)[:, :2]
        state = self._path_cursor.update(robot_xy_w, self._lookahead)
        self._subgoal[:, :2] = state.subgoal_xy
        self._subgoal[:, 2] = state.subgoal_heading
        self.cross_track_error[:] = state.cross_track
        self.along_track_progress[:] = state.along_track_progress

        # Dwell-gated completion: success is parking at the path end, not
        # touching it. The flag fires only after the robot holds inside the
        # radius at low speed for cfg.dwell_steps consecutive control steps —
        # this method runs once per control step, so the count is decimation-
        # aware. Body-frame planar speed mirrors the deployment odometry a
        # runtime arrival governor was rejected in favor of.
        within_radius = state.end_distance <= self.cfg.dwell_radius_m
        speed = torch.norm(
            wp.to_torch(self._robot.data.root_lin_vel_b)[:, :2], dim=-1
        )
        below_speed = speed <= self.cfg.dwell_speed_max_m_s
        new_counter, parked = dwell_step(
            self.dwell_counter, within_radius, below_speed, self.cfg.dwell_steps
        )
        self.dwell_counter[:] = new_counter
        self.path_complete[:] = parked

    def _update_metrics(self):
        robot_xy_w = wp.to_torch(self._robot.data.root_pos_w)[:, :2]
        self._distance_to_subgoal[:] = torch.norm(
            robot_xy_w - self._subgoal[:, :2], dim=-1
        )

    """
    Debug visualization
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle subgoal + path waypoint markers."""
        if debug_vis:
            if not hasattr(self, "_subgoal_sphere_vis"):
                self._subgoal_sphere_vis = VisualizationMarkers(
                    self.cfg.subgoal_sphere_visualizer_cfg
                )
                self._subgoal_heading_vis = VisualizationMarkers(
                    self.cfg.subgoal_heading_visualizer_cfg
                )
                self._path_points_vis = VisualizationMarkers(
                    self.cfg.path_points_visualizer_cfg
                )
            self._subgoal_sphere_vis.set_visibility(True)
            self._subgoal_heading_vis.set_visibility(True)
            self._path_points_vis.set_visibility(True)
        else:
            if hasattr(self, "_subgoal_sphere_vis"):
                self._subgoal_sphere_vis.set_visibility(False)
                self._subgoal_heading_vis.set_visibility(False)
                self._path_points_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Draw the rolling subgoal (sphere + tangent arrow) and the path."""
        if not self._robot.is_initialized:
            return

        num_envs = self._subgoal.shape[0]

        subgoal_pos = torch.zeros(num_envs, 3, device=self.device)
        subgoal_pos[:, :2] = self._subgoal[:, :2]
        subgoal_pos[:, 2] = 0.15
        self._subgoal_sphere_vis.visualize(translations=subgoal_pos)

        arrow_pos = subgoal_pos.clone()
        arrow_pos[:, 2] += 0.2
        zeros = torch.zeros(num_envs, device=self.device)
        pitch_neg90 = torch.full((num_envs,), -math.pi / 2, device=self.device)
        tip_quat = quat_from_euler_xyz(zeros, pitch_neg90, zeros)
        heading_quat = quat_from_euler_xyz(zeros, zeros, self._subgoal[:, 2])
        self._subgoal_heading_vis.visualize(
            translations=arrow_pos,
            orientations=quat_mul(heading_quat, tip_quat),
        )

        # Path waypoints, subsampled to ~one marker per 0.2 m of arc.
        stride = max(1, int(round(0.2 / max(self.cfg.path_spacing_m, 1e-3))))
        paths = self._path_cursor.paths[:, ::stride]            # (B, P', 2)
        n_pts = paths.shape[1]
        point_idx = torch.arange(0, n_pts * stride, stride, device=self.device)
        valid = point_idx.unsqueeze(0) < self._path_cursor.path_len.unsqueeze(1)
        pts = paths[valid]                                       # (N, 2)
        translations = torch.zeros(pts.shape[0], 3, device=self.device)
        translations[:, :2] = pts
        translations[:, 2] = 0.05
        self._path_points_vis.visualize(translations=translations)


_SUBGOAL_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/subgoal_sphere",
    markers={
        "subgoal": sim_utils.SphereCfg(
            radius=0.12,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 1.0)),
        ),
    },
)

_SUBGOAL_HEADING_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/subgoal_heading",
    markers={
        "arrow": sim_utils.ConeCfg(
            radius=0.06,
            height=0.25,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 1.0)),
        ),
    },
)

_PATH_POINTS_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/path_points",
    markers={
        "waypoint": sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
    },
)


@configclass
class SubgoalCommandCfg(CommandTermCfg):
    """Configuration for the rolling-subgoal command term."""

    class_type: type = SubgoalCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    """Time range for resampling a new path (min, max) in seconds. Defaults
    to effectively never: one path per episode, episodes end via the
    path-complete / off-path / timeout terminations."""

    lookahead_m: float = SUBGOAL_LOOKAHEAD_M
    """Arc-length distance (meters) the subgoal leads the robot's projection
    onto the path. The single quantity the policy actually observes about the
    path, and the value the deployed planner-follows backend must reproduce —
    pinned to the shared ``SUBGOAL_LOOKAHEAD_M`` so the two lanes cannot
    drift."""

    lookahead_randomization_m: tuple[float, float] | None = None
    """When set, the per-env lookahead is resampled uniformly from this
    ``(min, max)`` band at each path reset instead of held at ``lookahead_m``.
    Trains the policy to track a subgoal across a range of distances so the
    deployed selector only has to land inside the band, not hit one exact
    distance. ``None`` keeps the fixed ``lookahead_m`` (baseline)."""

    min_goal_distance: float = 1.0
    """Minimum straight-line distance (meters) between the robot and the
    sampled path endpoint."""

    max_goal_attempts: int = 10
    """Goal candidates tried per env before falling back to a straight
    segment."""

    path_spacing_m: float = MAP_RESOLUTION
    """Arc-length spacing (meters) of path waypoints. Defaults to the shared
    deployment map resolution so training paths match the discretization of
    the paths the deployed planner publishes."""

    waypoint_noise_std_m: float = MAP_RESOLUTION / 2.0
    """Std-dev (meters) of the truncated Gaussian perturbation applied to
    interior waypoints at plan time. Bounds the train/deploy planner
    disagreement the policy is robust to; see ``perturb_waypoints``."""

    dwell_radius_m: float = 0.3
    """Radius (meters) around the path's final point the robot must hold inside
    for the path to count as complete. Kept at the former instant-touch
    completion threshold so only the *dwell* requirement is new, not the
    radius."""

    dwell_speed_max_m_s: float = 0.1
    """Body-frame planar speed (m/s) at or below which the robot counts as
    parked while inside ``dwell_radius_m``. Completion accrues only while both
    hold. ``float('inf')`` disables the speed gate (instant-touch on radius)."""

    dwell_steps: int = 10
    """Consecutive control steps the robot must hold inside ``dwell_radius_m``
    at/under ``dwell_speed_max_m_s`` before the path completes — parking, not
    touching, collects the sparse completion bonus. ~0.33 s at 30 Hz control."""

    max_path_points: int = 512
    """Per-env waypoint buffer size; longer paths are truncated head-first."""

    debug_vis: bool = False
    """Whether to visualize the subgoal and path."""

    subgoal_sphere_visualizer_cfg: VisualizationMarkersCfg = _SUBGOAL_SPHERE_CFG
    """Marker config for the rolling subgoal."""

    subgoal_heading_visualizer_cfg: VisualizationMarkersCfg = _SUBGOAL_HEADING_CFG
    """Arrow marker config for the subgoal's path-tangent heading."""

    path_points_visualizer_cfg: VisualizationMarkersCfg = _PATH_POINTS_CFG
    """Marker config for the planned path's waypoints."""
