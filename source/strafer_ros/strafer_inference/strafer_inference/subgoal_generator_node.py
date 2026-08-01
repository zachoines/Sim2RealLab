"""Rolling-subgoal generator node: goal telemetry -> plan -> subgoal pose.

Owns hybrid replanning: while the inference node's active-goal telemetry
is fresh, calls Nav2's ``ComputePathToPose`` on its own cadence and
anchors the path from the action result (``/plan`` is a fallback input).
Each tick it looks up the ``map -> base_link`` TF, runs the pure
:class:`RollingSubgoalGenerator`, and publishes the rolling subgoal.

Four non-obvious contracts:

- **Anchoring.** Training installs ONE path per goal, in the world frame,
  and only ever advances a monotonic cursor along it
  (``SubgoalCommand._resample_command`` -> ``PathCursor.set_paths`` at
  goal resample; ``_update_command`` never re-installs). Deploy must
  reproduce that. Every ``ComputePathToPose`` result is planned from the
  robot's *current* pose (``use_start=False``) and Nav2's
  ``planner_server`` additionally republishes ``/plan`` at ~12 Hz, so
  installing on arrival re-roots the path under the robot continuously:
  measured cross-track then stays at ~0.03 m (one costmap cell) over
  metres of travel and **no corrective lateral signal can ever
  accumulate**, which is the signal training taught the policy to
  consume. Under ``subgoal_anchoring: mission`` (the shipped default) a
  fresh plan REPLACES the anchored path only under the admission rules
  in :func:`~strafer_inference.generator.evaluate_admission`; otherwise
  it counts only as planner liveness and is discarded.
  ``subgoal_anchoring: rolling`` is the named legacy fallback that
  re-roots on every plan, kept so the behavioural re-validation can A/B
  the two semantics on the rig.
- The active-goal topic is status telemetry (inference node -> here),
  NOT a command channel: ``navigate_to_pose`` remains the only way to
  command a mission. Its staleness stops replanning, the plan ages past
  ``path_timeout_s``, and the subgoal is suppressed.
- The subgoal has its own topic, separate from the goal: the goal is
  latched per mission and resets the recurrent policy, while the subgoal
  is a per-tick setpoint — sharing a channel would reset it continuously.
- Suppressing the subgoal on a stale plan is right for a dead planner but
  self-locking when the planner is refusing the robot's *own pose* as a
  start: the policy zero-twists, so the pose never changes and nothing
  clears the refusal. ``fallback_planner_id`` and ``starvation_hold_s``
  break that cycle; both are degraded modes and both log. Anchoring does
  not weaken them: plan LIVENESS (``_last_plan_rx_t``, the input to
  ``_plan_fresh``) is refreshed by any valid plan for the active goal
  whether or not it is admitted, so a refusing planner still starves the
  guard exactly as before.

ROS glue only -- selection math is in :mod:`strafer_inference.generator`.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from strafer_shared.constants import SUBGOAL_LOOKAHEAD_M

from .generator import (
    ADMIT_ROLLING,
    RollingSubgoalGenerator,
    arc_length_projection,
    evaluate_admission,
)

# An in-flight replan older than this is treated as lost: rclpy leaves
# the future pending forever if the planner dies mid-request.
_REPLAN_ABANDON_S = 2.0

# A /plan whose terminal pose is farther than this from the active goal
# was computed for a different goal; don't install it.
_PLAN_GOAL_MATCH_M = 0.5

# Anchoring modes for the ``subgoal_anchoring`` parameter.
_ANCHOR_MISSION = "mission"
_ANCHOR_ROLLING = "rolling"
_ANCHOR_MODES = (_ANCHOR_MISSION, _ANCHOR_ROLLING)

# Waypoint quantum for the plan-identity key. Nav2 emits float64 poses on
# a 0.05 m costmap grid, so 1 mm is far finer than any real difference
# between two distinct plans and coarse enough that a bit-identical
# republish keys identically.
_PLAN_KEY_QUANT_M = 1e-3

# nav2_costmap_2d publishes its costmap as an OccupancyGrid through a
# translation table: 254 (LETHAL) -> 100, 253 (INSCRIBED_INFLATED) -> 99,
# 255 (NO_INFORMATION) -> -1, everything else linearly into 1..97. 99 is
# therefore "the robot's inscribed radius is in contact or worse", which
# is the threshold a global plan is invalid at.
_COSTMAP_INSCRIBED = 99


def _default_update_period() -> float:
    """Read the policy step period from strafer_shared at call time.

    Indirected through a function (not a module-level import) so a test
    patching ``POLICY_SIM_DT`` / ``POLICY_DECIMATION`` changes the value
    the node picks up -- mirrors the inference node.
    """
    from strafer_shared.constants import POLICY_DECIMATION, POLICY_SIM_DT

    return POLICY_SIM_DT * POLICY_DECIMATION


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion for a planar yaw rotation."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class SubgoalGeneratorNode(Node):
    """Follows Nav2's ``/plan`` and emits the rolling subgoal pose."""

    def __init__(self, *, parameter_overrides: Optional[list] = None) -> None:
        super().__init__(
            "strafer_subgoal_generator",
            parameter_overrides=parameter_overrides or [],
        )

        self.declare_parameter("plan_topic", "/plan")
        self.declare_parameter("subgoal_topic", "/strafer/subgoal")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("update_period_s", _default_update_period())
        # Sourced from the shared constant so train and deploy cannot drift.
        self.declare_parameter("lookahead_m", SUBGOAL_LOOKAHEAD_M)
        # 0 (or negative) means "use the path as published" (no truncation).
        self.declare_parameter("max_path_points", 0)
        # Generator half of the split stale-plan budget (wall-clock): stop
        # publishing the subgoal once the plan ages past this, so a dead
        # planner reaches the inference watchdog rather than the policy
        # chasing it.
        self.declare_parameter("path_timeout_s", 1.0)
        # Replan ownership: while the inference node's active-goal telemetry
        # is fresh, request ComputePathToPose on this cadence. Must stay
        # below path_timeout_s or the plan goes stale between replans.
        self.declare_parameter("replan_period_s", 0.5)
        self.declare_parameter(
            "active_goal_topic", "/strafer_inference/active_goal"
        )
        # Telemetry keep-alive is ~1 Hz; 2.5 s tolerates one missed message
        # plus publish jitter before replanning stops.
        self.declare_parameter("goal_telemetry_timeout_s", 2.5)
        self.declare_parameter("planner_action", "/compute_path_to_pose")
        self.declare_parameter("planner_id", "GridBased")
        # Escape-hatch planner able to start from a cell inside the inflation
        # halo. Empty disables it; the shipped YAML names one.
        self.declare_parameter("fallback_planner_id", "")
        self.declare_parameter("planner_refusals_before_fallback", 2)
        # Above the halo width the robot wedges in, so releasing the fallback
        # cannot drop it straight back into the same refusal.
        self.declare_parameter("fallback_release_m", 0.25)
        # Republish budget for the last valid subgoal while the planner refuses
        # a stationary robot. 0 disables the hold.
        self.declare_parameter("starvation_hold_s", 5.0)
        self.declare_parameter("starvation_stationary_m", 0.05)
        self.declare_parameter("starvation_stationary_s", 1.0)

        # --- anchoring semantics -------------------------------------
        # "mission": one anchored path per goal, replaced only under the
        # admission rules (the training contract). "rolling": the legacy
        # re-root-on-every-plan behaviour, kept as a named fallback so the
        # rig re-validation can A/B the two.
        self.declare_parameter("subgoal_anchoring", _ANCHOR_MISSION)
        # Admission class (ii): above this the corridor is genuinely lost.
        # 0.5 m is just under the global costmap's 0.55 m inflation radius
        # (strafer_navigation/config/nav2_params.yaml) and ~15x the 0.03 m
        # cross-track measured under the old re-rooting behaviour, so it
        # cannot fire on ordinary tracking error.
        self.declare_parameter("admission_cross_track_m", 0.5)
        # Admission class (i): the remaining anchored path now crosses
        # lethal/inscribed costmap cells. Needs the global costmap; the
        # check abstains (never admits) while none has been received.
        self.declare_parameter("admission_collision_check", True)
        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("costmap_lethal_cost", _COSTMAP_INSCRIBED)
        # A costmap older than this is not trusted to veto an anchor.
        self.declare_parameter("costmap_timeout_s", 5.0)
        # Periodic one-line anchoring/admission summary. 0 disables.
        self.declare_parameter("anchor_log_period_s", 10.0)

        self._map_frame: str = self.get_parameter("map_frame").value
        self._base_frame: str = self.get_parameter("base_frame").value
        self._path_timeout_s = float(self.get_parameter("path_timeout_s").value)
        self._replan_period_s = float(
            self.get_parameter("replan_period_s").value
        )
        self._goal_telemetry_timeout_s = float(
            self.get_parameter("goal_telemetry_timeout_s").value
        )
        self._planner_id: str = self.get_parameter("planner_id").value
        if self._replan_period_s >= self._path_timeout_s:
            # The single-node budget rule that replaced the autonomy
            # client's replan-vs-suppression warning.
            self.get_logger().warning(
                f"replan_period_s={self._replan_period_s:.2f} s is not below "
                f"path_timeout_s={self._path_timeout_s:.2f} s; the plan can "
                "go stale between replans and flap the subgoal suppression."
            )
        lookahead_m = float(self.get_parameter("lookahead_m").value)
        max_points_param = int(self.get_parameter("max_path_points").value)
        max_points = max_points_param if max_points_param >= 2 else None

        self._generator = RollingSubgoalGenerator(
            lookahead_m=lookahead_m, max_points=max_points
        )

        anchoring = str(self.get_parameter("subgoal_anchoring").value).strip().lower()
        if anchoring not in _ANCHOR_MODES:
            raise ValueError(
                f"subgoal_anchoring={anchoring!r} is not one of {_ANCHOR_MODES}"
            )
        self._rolling_anchoring = anchoring == _ANCHOR_ROLLING
        self._admission_cross_track_m = float(
            self.get_parameter("admission_cross_track_m").value
        )
        self._admission_collision_check = bool(
            self.get_parameter("admission_collision_check").value
        )
        self._costmap_lethal_cost = int(
            self.get_parameter("costmap_lethal_cost").value
        )
        self._costmap_timeout_s = float(
            self.get_parameter("costmap_timeout_s").value
        )

        # Monotonic receipt time of the latest valid plan. Drives the
        # plan-staleness guard: subgoal publishing stops once the plan ages
        # past path_timeout_s. This is plan LIVENESS, deliberately NOT
        # "when the anchored path was installed" -- under mission anchoring
        # most valid plans are rejected, and treating a rejection as
        # silence would starve the RC-6 guards into firing against a
        # perfectly healthy planner.
        self._last_plan_rx_t: Optional[float] = None
        self._stale_plan_logged = False

        # Identity of the last plan CONSUMED for an admission decision.
        # planner_server republishes /plan at ~12 Hz, so arrival is
        # meaningless as a "new plan" signal; the key is content + stamp.
        self._last_plan_key: Optional[tuple] = None
        # Monotonic time the anchored path was installed, and the robot's
        # cross-track against it as of the last tick (cached so the plan
        # callback needs no TF lookup of its own).
        self._anchor_installed_t: Optional[float] = None
        self._last_cross_track_m: Optional[float] = None
        # Set on a mission boundary (first goal, or a preempting goal at a
        # new pose); forces the next valid plan to be admitted even if the
        # new goal is close enough to the old one to pass the provenance
        # check. Cleared when that plan anchors.
        self._new_mission_pending = False
        # Admission bookkeeping, surfaced by the periodic anchor log.
        self._plans_seen = 0
        self._plans_republished = 0
        self._anchors_admitted = 0
        self._anchors_rejected = 0
        self._admit_reasons: dict[str, int] = {}
        self._anchor_log_period_s = float(
            self.get_parameter("anchor_log_period_s").value
        )
        self._last_anchor_log_t = time.monotonic()

        # Latest global costmap for admission class (i), as (grid, info,
        # rx_time). Written by its own callback in the default mutex group,
        # so it is serialized against the tick and needs no lock.
        self._costmap_grid: Optional[np.ndarray] = None
        self._costmap_info = None
        self._costmap_rx_t: Optional[float] = None
        self._costmap_absent_logged = False
        self._costmap_frame_warned = False

        # Active-goal telemetry from the inference node: gates replanning
        # and provides the ComputePathToPose target.
        self._active_goal: Optional[PoseStamped] = None
        self._last_goal_telemetry_rx_t: Optional[float] = None
        self._replan_inflight = False
        # Wall-clock send time of the in-flight request. rclpy never
        # resolves a pending action future if the server dies mid-request,
        # so an in-flight older than _REPLAN_ABANDON_S is treated as lost
        # rather than wedging replanning forever.
        self._replan_sent_t: Optional[float] = None
        # Goal xy the in-flight request was computed for; a result for a
        # since-changed goal is discarded.
        self._replan_goal_xy: Optional[tuple[float, float]] = None
        self._planner_unready_logged = False

        self._fallback_planner_id: str = str(
            self.get_parameter("fallback_planner_id").value
        ).strip()
        self._refusals_before_fallback = int(
            self.get_parameter("planner_refusals_before_fallback").value
        )
        self._fallback_release_m = float(
            self.get_parameter("fallback_release_m").value
        )
        self._starvation_hold_s = float(
            self.get_parameter("starvation_hold_s").value
        )
        self._stationary_m = float(
            self.get_parameter("starvation_stationary_m").value
        )
        # Both windows convert to TICKS: the tick timer runs on the node clock,
        # so a tick count is a sim-time budget at any RTF, and these budget the
        # robot's motion. (The plan-freshness window above stays wall-clock —
        # it guards a dead planner, which is a wall-clock event.)
        tick_period = float(self.get_parameter("update_period_s").value)
        self._hold_ticks_budget = (
            max(0, round(self._starvation_hold_s / tick_period))
            if tick_period > 0.0 else 0
        )
        self._stationary_ticks_required = (
            max(1, round(
                float(self.get_parameter("starvation_stationary_s").value)
                / tick_period
            ))
            if tick_period > 0.0 else 1
        )
        self._planner_refusals = 0
        self._fallback_engaged_at_xy: Optional[tuple[float, float]] = None
        # Goal the installed path was computed for; the hold may only republish
        # a subgoal that still belongs to the active goal.
        self._path_goal_xy: Optional[tuple[float, float]] = None
        self._last_robot_xy: Optional[np.ndarray] = None
        self._stationary_ticks = 0
        self._hold_ticks_remaining = 0
        # One window per starvation episode, re-armed only by a fresh plan, so
        # an unrecoverable wedge fails loud instead of holding forever.
        self._hold_armed = True

        plan_topic = self.get_parameter("plan_topic").value
        subgoal_topic = self.get_parameter("subgoal_topic").value
        active_goal_topic = self.get_parameter("active_goal_topic").value
        planner_action = self.get_parameter("planner_action").value

        self._subgoal_pub = self.create_publisher(PoseStamped, subgoal_topic, 10)
        self._plan_sub = self.create_subscription(
            Path, plan_topic, self._on_plan, 10
        )
        self._active_goal_sub = self.create_subscription(
            PoseStamped, active_goal_topic, self._on_active_goal, 10
        )
        # Global costmap for admission class (i). nav2_costmap_2d latches it
        # TRANSIENT_LOCAL, and nav2_params sets always_send_full_costmap:
        # true, so a plain full-grid subscription is enough (no
        # costmap_updates stitching).
        self._costmap_topic = str(self.get_parameter("costmap_topic").value)
        self._costmap_sub = None
        if self._admission_collision_check:
            self._costmap_sub = self.create_subscription(
                OccupancyGrid,
                self._costmap_topic,
                self._on_costmap,
                QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                ),
            )
        self._planner_client = ActionClient(
            self, ComputePathToPose, planner_action
        )
        # Replan cadence on the steady (wall) clock, matching the
        # wall-clock plan-freshness window (time.monotonic): a node-clock
        # timer would run at sim rate and, at low RTF, fire far slower
        # than the plan goes stale, starving the subgoal.
        self._replan_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._replan_timer = self.create_timer(
            self._replan_period_s, self._on_replan_tick,
            clock=self._replan_clock,
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        update_period = float(self.get_parameter("update_period_s").value)
        self._timer = self.create_timer(update_period, self._on_tick)

        self.get_logger().info(
            f"strafer_subgoal_generator up: plan_topic={plan_topic} "
            f"subgoal_topic={subgoal_topic} lookahead={lookahead_m:.4f} m "
            f"update_period={update_period:.4f} s "
            f"max_path_points={max_points if max_points is not None else 'unbounded'} "
            f"replan_period={self._replan_period_s:.2f} s "
            f"active_goal_topic={active_goal_topic} "
            f"planner={self._planner_id} "
            f"fallback_planner={self._fallback_planner_id or '<disabled>'} "
            f"starvation_hold={self._starvation_hold_s:.1f}s"
            f"({self._hold_ticks_budget} ticks)"
        )
        self.get_logger().info(
            f"anchoring={anchoring} "
            f"(admission: cross_track>{self._admission_cross_track_m:.2f} m, "
            f"collision_check="
            f"{self._costmap_topic if self._admission_collision_check else '<disabled>'}"
            f"@cost>={self._costmap_lethal_cost}, goal_changed). "
            + (
                "ROLLING: every accepted plan re-roots the path under the "
                "robot — the legacy behaviour, cross-track cannot develop."
                if self._rolling_anchoring
                else "MISSION: one anchored path per goal; a fresh plan is "
                "planner liveness unless it is admitted."
            )
        )

    def _note_plan_alive(self) -> None:
        """Bookkeeping every VALID plan earns, admitted or not.

        Plan liveness and starvation-guard state key off "a planner
        answered with a usable path", never off "we installed it". Under
        mission anchoring most valid plans are rejected by admission, and
        letting a rejection read as planner silence would age the plan out
        and trip the RC-6 guards against a healthy planner.
        """
        self._last_plan_rx_t = time.monotonic()
        self._planner_refusals = 0
        self._end_starvation_hold(rearm=True)

    def _install_path(self, msg: Path, source: str) -> None:
        """Anchor a fresh global plan as THE path for this mission.

        Seeds the cursor by projecting the robot onto the new path rather
        than rewinding to zero, so an admitted replacement preserves
        progress-toward-goal. For the usual Nav2 plan — computed with
        ``use_start=False``, hence starting under the robot — the
        projection is ~0 and this is identical to a rewind; it matters for
        a plan that starts elsewhere, where a rewind would put the subgoal
        one lookahead from the path's start instead of from the robot.
        """
        if not msg.poses:
            self.get_logger().warning(
                f"Empty path from {source} (planner produced no path); "
                "keeping the previous plan.",
                throttle_duration_sec=5.0,
            )
            return
        path_xy = np.array(
            [(p.pose.position.x, p.pose.position.y) for p in msg.poses],
            dtype=np.float64,
        )

        initial_cursor: Optional[float] = None
        robot_xy = self._last_robot_xy
        if robot_xy is not None and len(path_xy) > 1:
            initial_cursor, _ = arc_length_projection(path_xy, robot_xy)

        self._generator.set_path(path_xy, initial_cursor=initial_cursor)
        self._anchor_installed_t = time.monotonic()
        self._last_cross_track_m = None
        # A fresh plan ends any starvation episode: clear the refusal streak,
        # record which goal this path serves, close and re-arm the hold window.
        self._note_plan_alive()
        self._path_goal_xy = (
            (
                self._active_goal.pose.position.x,
                self._active_goal.pose.position.y,
            )
            if self._active_goal is not None else None
        )
        self.get_logger().debug(
            f"Anchored plan from {source}: {len(msg.poses)} poses, total arc "
            f"{self._generator.total_arc:.3f} m; cursor seeded at "
            f"{self._generator.cursor_arc:.3f} m."
        )

    @staticmethod
    def _plan_key(msg: Path) -> tuple:
        """Identity of a plan by CONTENT + STAMP, not by arrival.

        ``planner_server`` republishes the same path on ``/plan`` at ~12 Hz
        independently of our own request cadence, so "a Path message
        arrived" says nothing about whether the plan is new. Quantising to
        ``_PLAN_KEY_QUANT_M`` keeps the key exact for a bit-identical
        republish while staying far below any real difference between two
        distinct plans.
        """
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        quant = _PLAN_KEY_QUANT_M
        pts = tuple(
            (
                round(p.pose.position.x / quant),
                round(p.pose.position.y / quant),
            )
            for p in msg.poses
        )
        return (stamp, len(msg.poses), pts)

    def _anchor_in_collision(self) -> bool:
        """True if the REMAINING anchored path crosses blocked costmap cells.

        Only the portion ahead of the cursor is checked: the robot has
        already driven the rest, and an obstacle appearing behind it is not
        a reason to re-plan. Abstains (returns False) with no costmap, a
        stale costmap, or a frame mismatch — admission class (i) must never
        fire on absent evidence, and the other classes still apply.
        """
        if not self._admission_collision_check:
            return False
        grid = self._costmap_grid
        if grid is None or self._costmap_info is None:
            if not self._costmap_absent_logged:
                self.get_logger().warning(
                    f"No costmap received on {self._costmap_topic!r}; the "
                    "collision admission rule is inactive (cross-track and "
                    "goal-change rules still apply)."
                )
                self._costmap_absent_logged = True
            return False
        if (
            self._costmap_rx_t is None
            or time.monotonic() - self._costmap_rx_t > self._costmap_timeout_s
        ):
            self.get_logger().warning(
                "Costmap is older than "
                f"{self._costmap_timeout_s:.1f} s; skipping the collision "
                "admission rule.",
                throttle_duration_sec=10.0,
            )
            return False

        path = self._generator.path
        arc = self._generator.arc
        if path is None or arc is None or len(path) == 0:
            return False

        ahead = path[arc >= self._generator.cursor_arc]
        if len(ahead) == 0:
            ahead = path[-1:]

        info = self._costmap_info
        res = info.resolution
        if res <= 0.0:
            return False
        cols = np.floor(
            (ahead[:, 0] - info.origin.position.x) / res
        ).astype(np.int64)
        rows = np.floor(
            (ahead[:, 1] - info.origin.position.y) / res
        ).astype(np.int64)
        # Waypoints outside the costmap are unknown, not blocked.
        inside = (
            (cols >= 0) & (cols < info.width)
            & (rows >= 0) & (rows < info.height)
        )
        if not np.any(inside):
            return False
        costs = grid[rows[inside], cols[inside]]
        return bool(np.any(costs >= self._costmap_lethal_cost))

    def _consider_plan(self, msg: Path, source: str) -> None:
        """Route a valid plan: liveness always, anchoring only on admission."""
        if not msg.poses:
            self.get_logger().warning(
                f"Empty path from {source} (planner produced no path); "
                "keeping the previous plan.",
                throttle_duration_sec=5.0,
            )
            return

        key = self._plan_key(msg)
        if key == self._last_plan_key:
            # A republish of the plan we already decided on. It still
            # proves the planner is alive; it is not a new decision.
            self._plans_republished += 1
            self._last_plan_rx_t = time.monotonic()
            return
        self._last_plan_key = key
        self._plans_seen += 1

        goal_changed = self._new_mission_pending or (
            self._active_goal is not None
            and not self._path_serves_active_goal()
        )
        decision = evaluate_admission(
            has_anchor=self._generator.has_path,
            rolling_mode=self._rolling_anchoring,
            goal_changed=goal_changed,
            anchor_in_collision=(
                self._anchor_in_collision()
                if self._generator.has_path and not self._rolling_anchoring
                else False
            ),
            cross_track_m=self._last_cross_track_m,
            cross_track_bound_m=self._admission_cross_track_m,
        )
        self._admit_reasons[decision.reason] = (
            self._admit_reasons.get(decision.reason, 0) + 1
        )

        if not decision.admit:
            # The anchor stands. The plan is still evidence the planner is
            # healthy, so it clears refusals and refreshes the freshness
            # window exactly as an installed plan would.
            self._anchors_rejected += 1
            self._note_plan_alive()
            self.get_logger().debug(
                f"Holding the anchored path; discarded a fresh plan from "
                f"{source} ({len(msg.poses)} poses). cross_track="
                + (
                    f"{self._last_cross_track_m:.3f} m"
                    if self._last_cross_track_m is not None
                    else "unknown"
                )
            )
            return

        self._anchors_admitted += 1
        self._new_mission_pending = False
        if decision.reason != ADMIT_ROLLING:
            self.get_logger().info(
                f"Anchoring a new path from {source}: {decision.reason}"
                + (f" ({decision.detail})" if decision.detail else "")
            )
        self._install_path(msg, source=source)

    def _maybe_log_anchor_status(self) -> None:
        """Periodic one-liner so anchoring is legible without a parity run."""
        if self._anchor_log_period_s <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_anchor_log_t < self._anchor_log_period_s:
            return
        self._last_anchor_log_t = now
        reasons = " ".join(
            f"{k}={v}" for k, v in sorted(self._admit_reasons.items())
        ) or "none"
        anchor_age = (
            f"{now - self._anchor_installed_t:.1f}s"
            if self._anchor_installed_t is not None else "n/a"
        )
        self.get_logger().info(
            f"anchor status: mode={'rolling' if self._rolling_anchoring else 'mission'} "
            f"plans_new={self._plans_seen} republished={self._plans_republished} "
            f"admitted={self._anchors_admitted} held={self._anchors_rejected} "
            f"[{reasons}] anchor_age={anchor_age} "
            f"cursor={self._generator.cursor_arc:.2f}/"
            f"{self._generator.total_arc:.2f} m cross_track="
            + (
                f"{self._last_cross_track_m:.3f} m"
                if self._last_cross_track_m is not None else "n/a"
            )
        )

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        if msg.header.frame_id and msg.header.frame_id != self._map_frame:
            if not self._costmap_frame_warned:
                self.get_logger().warning(
                    f"Costmap frame {msg.header.frame_id!r} != map_frame "
                    f"{self._map_frame!r}; the collision admission rule is "
                    "inactive (the anchored path is in map_frame)."
                )
                self._costmap_frame_warned = True
            return
        self._costmap_grid = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width
        )
        self._costmap_info = msg.info
        self._costmap_rx_t = time.monotonic()

    def _on_plan(self, msg: Path) -> None:
        # Fallback input only — the primary path comes from the
        # ComputePathToPose result. The planner also mirrors our own
        # requests onto /plan, so with a mission active accept only paths
        # ending at the active goal; else a superseded request's echo
        # reinstalls the discarded path.
        if self._active_goal is not None and msg.poses:
            terminal = msg.poses[-1]
            dx = terminal.pose.position.x - self._active_goal.pose.position.x
            dy = terminal.pose.position.y - self._active_goal.pose.position.y
            if math.hypot(dx, dy) > _PLAN_GOAL_MATCH_M:
                self.get_logger().debug(
                    "Ignoring /plan that does not end at the active goal."
                )
                return
        self._consider_plan(msg, source="/plan")

    # ------------------------------------------------------------------
    # Replan ownership (active-goal telemetry -> ComputePathToPose)
    # ------------------------------------------------------------------

    def _on_active_goal(self, msg: PoseStamped) -> None:
        previous = self._active_goal
        self._active_goal = msg
        self._last_goal_telemetry_rx_t = time.monotonic()
        # A new mission or a preempting goal retargets immediately rather
        # than waiting out the cadence timer.
        if previous is None or self._goal_xy_changed(previous, msg):
            # Mission boundary: the anchored path belongs to the previous
            # mission and must be replaced even when the new goal happens
            # to sit within _PLAN_GOAL_MATCH_M of the old one, which the
            # goal-provenance check alone would read as "same goal".
            self._new_mission_pending = True
            self._request_replan()

    @staticmethod
    def _goal_xy_changed(
        previous: PoseStamped, current: PoseStamped,
    ) -> bool:
        dx = current.pose.position.x - previous.pose.position.x
        dy = current.pose.position.y - previous.pose.position.y
        return math.hypot(dx, dy) > 1e-3

    def _goal_telemetry_fresh(self, now_monotonic_s: float) -> bool:
        return (
            self._last_goal_telemetry_rx_t is not None
            and now_monotonic_s - self._last_goal_telemetry_rx_t
            <= self._goal_telemetry_timeout_s
        )

    def _on_replan_tick(self) -> None:
        if self._active_goal is None:
            return
        if not self._goal_telemetry_fresh(time.monotonic()):
            # Mission ended (telemetry stopped): drop the goal so
            # replanning stops and the /plan fallback is no longer gated
            # on a dead goal. The plan then ages out and the subgoal is
            # suppressed.
            self._active_goal = None
            return
        self._request_replan()

    def _request_replan(self) -> None:
        if self._active_goal is None:
            return
        if self._replan_inflight:
            if (
                self._replan_sent_t is not None
                and time.monotonic() - self._replan_sent_t
                > _REPLAN_ABANDON_S
            ):
                self.get_logger().warning(
                    "In-flight ComputePathToPose request got no response "
                    f"in {_REPLAN_ABANDON_S:.1f} s (planner died "
                    "mid-request?); abandoning it and resuming replanning."
                )
                self._replan_inflight = False
            else:
                return
        if not self._planner_client.server_is_ready():
            if not self._planner_unready_logged:
                self.get_logger().warning(
                    "ComputePathToPose action server is not available; "
                    "cannot replan (will keep retrying on the cadence)."
                )
                self._planner_unready_logged = True
            return
        self._planner_unready_logged = False

        goal = ComputePathToPose.Goal()
        goal.goal = self._active_goal
        goal.planner_id = self._select_planner_id()
        goal.use_start = False  # plan from the robot's current pose
        self._replan_inflight = True
        self._replan_sent_t = time.monotonic()
        self._replan_goal_xy = (
            self._active_goal.pose.position.x,
            self._active_goal.pose.position.y,
        )
        send_future = self._planner_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_replan_goal_response)

    def _on_replan_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # rcl teardown / transport errors
            self._replan_inflight = False
            # Not counted as a refusal: the request never reached a planner.
            self.get_logger().warning(
                f"ComputePathToPose request failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return
        if goal_handle is None or not goal_handle.accepted:
            self._replan_inflight = False
            self._note_planner_refusal("request rejected")
            self.get_logger().warning(
                "ComputePathToPose rejected the replan request.",
                throttle_duration_sec=5.0,
            )
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_replan_result)

    def _on_replan_result(self, future) -> None:
        self._replan_inflight = False
        try:
            result = future.result()
        except Exception as exc:  # rcl teardown / transport errors
            # Transport failure, not a refusal (see _note_planner_refusal).
            self.get_logger().warning(
                f"ComputePathToPose result failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return
        if result is None:
            self.get_logger().warning(
                "ComputePathToPose returned no result.",
                throttle_duration_sec=5.0,
            )
            return
        # Discard a path computed for a goal that was preempted while the
        # request was in flight, and immediately re-request for the new
        # goal — the retarget in _on_active_goal was swallowed by the
        # in-flight guard.
        if self._active_goal is not None and self._replan_goal_xy is not None:
            dx = self._active_goal.pose.position.x - self._replan_goal_xy[0]
            dy = self._active_goal.pose.position.y - self._replan_goal_xy[1]
            if math.hypot(dx, dy) > 1e-3:
                self.get_logger().debug(
                    "Discarding path for a superseded goal."
                )
                self._request_replan()
                return
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self._note_planner_refusal(f"status {result.status}")
            self.get_logger().warning(
                f"ComputePathToPose did not succeed (status {result.status}); "
                "keeping the previous plan.",
                throttle_duration_sec=5.0,
            )
            return
        self._consider_plan(result.result.path, source="ComputePathToPose")

    # ------------------------------------------------------------------
    # Planner escape hatch + bounded starvation hold
    # ------------------------------------------------------------------

    def _note_planner_refusal(self, reason: str) -> None:
        """Count one refusal by a planner that answered, and engage the
        fallback at the threshold.

        Only an aborted result or a rejected request counts. An unavailable
        server, an abandoned in-flight request, and transport exceptions never
        reached a working planner, so they must keep starving the watchdog
        instead of arming a guard against a fault it cannot fix.
        """
        self._planner_refusals += 1
        if (
            self._fallback_planner_id
            and self._fallback_engaged_at_xy is None
            and self._planner_refusals >= self._refusals_before_fallback
            and self._last_robot_xy is not None
        ):
            self._fallback_engaged_at_xy = (
                float(self._last_robot_xy[0]), float(self._last_robot_xy[1])
            )
            self.get_logger().warning(
                f"{self._planner_id} refused {self._planner_refusals} "
                f"consecutive replans ({reason}); the robot's own pose is "
                f"likely inside the costmap inflation halo. Switching to "
                f"{self._fallback_planner_id!r}, which plans from an inflated "
                f"start, until the robot has moved "
                f"{self._fallback_release_m:.2f} m."
            )

    def _select_planner_id(self) -> str:
        """Planner for the next request: primary, or the escape hatch."""
        return (
            self._fallback_planner_id
            if self._fallback_engaged_at_xy is not None
            else self._planner_id
        )

    def _release_fallback_if_clear(self, robot_xy: np.ndarray) -> None:
        """Hand planning back to the primary once the robot has moved away.

        Keyed on distance rather than on a fallback success: releasing as soon
        as the fallback produces a path returns the robot to the pose the
        primary refuses, and the plan goes stale in the gap.
        """
        if self._fallback_engaged_at_xy is None:
            return
        moved = math.hypot(
            robot_xy[0] - self._fallback_engaged_at_xy[0],
            robot_xy[1] - self._fallback_engaged_at_xy[1],
        )
        if moved >= self._fallback_release_m:
            self._fallback_engaged_at_xy = None
            self.get_logger().warning(
                f"Robot moved {moved:.2f} m since the planner started "
                f"refusing; returning to planner {self._planner_id!r}."
            )

    def _track_stationary(self, robot_xy: np.ndarray) -> None:
        """Count consecutive ticks with no meaningful robot motion.

        Drift is measured from the anchor the streak started at, not from the
        previous tick, so slow creep accumulates and breaks the streak instead
        of reading as stationary indefinitely.
        """
        if self._last_robot_xy is None:
            self._last_robot_xy = robot_xy
            self._stationary_ticks = 0
            return
        drift = math.hypot(
            robot_xy[0] - self._last_robot_xy[0],
            robot_xy[1] - self._last_robot_xy[1],
        )
        if drift <= self._stationary_m:
            self._stationary_ticks += 1
        else:
            self._stationary_ticks = 0
            self._last_robot_xy = robot_xy

    def _path_serves_active_goal(self) -> bool:
        """True while the held plan was computed for the current goal."""
        if self._active_goal is None or self._path_goal_xy is None:
            return False
        return math.hypot(
            self._active_goal.pose.position.x - self._path_goal_xy[0],
            self._active_goal.pose.position.y - self._path_goal_xy[1],
        ) <= _PLAN_GOAL_MATCH_M

    def _end_starvation_hold(self, *, rearm: bool) -> None:
        self._hold_ticks_remaining = 0
        if rearm:
            self._hold_armed = True

    def _starvation_hold_publishes(self) -> bool:
        """Whether to republish the last subgoal despite a stale plan."""
        if self._hold_ticks_budget <= 0:
            return False
        if self._planner_refusals < self._refusals_before_fallback:
            return False    # no answering planner refused; ordinary suppression
        if not self._path_serves_active_goal():
            # The held path leads to a superseded goal, so republishing it would
            # aim the policy at a stale target.
            return False

        if self._hold_ticks_remaining > 0:
            self._hold_ticks_remaining -= 1
            if self._hold_ticks_remaining == 0:
                self.get_logger().error(
                    f"Starvation hold exhausted after "
                    f"{self._starvation_hold_s:.1f} s of policy time and the "
                    "planner is still refusing; suppressing the rolling "
                    "subgoal, so the inference watchdog will zero-twist. The "
                    "robot needs a manual holonomic strafe on /cmd_vel."
                )
                return False
            return True

        if not self._hold_armed:
            return False    # window already spent this episode
        if self._stationary_ticks < self._stationary_ticks_required:
            return False    # robot is moving; ordinary stale plan

        self._hold_ticks_remaining = self._hold_ticks_budget
        self._hold_armed = False
        self.get_logger().warning(
            f"Planner starvation: plan stale, {self._planner_refusals} "
            f"consecutive planner refusals, robot stationary for "
            f"{self._stationary_ticks_required} ticks. Republishing the last "
            f"valid subgoal for up to {self._starvation_hold_s:.1f} s of policy "
            "time so the policy can drive clear. DEGRADED: the subgoal is no "
            "longer being refreshed from a plan."
        )
        return True

    def _plan_fresh(self, now_monotonic_s: float) -> bool:
        """True if a plan (action result or /plan) arrived within
        ``path_timeout_s``."""
        return (
            self._last_plan_rx_t is not None
            and now_monotonic_s - self._last_plan_rx_t <= self._path_timeout_s
        )

    def _lookup_robot_xy(self) -> Optional[np.ndarray]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().debug(f"TF lookup failed: {exc}")
            return None
        xy = np.array(
            [tf.transform.translation.x, tf.transform.translation.y],
            dtype=np.float64,
        )
        if xy.shape != (2,):
            # A malformed transform is "no pose", not a crash: the tick now
            # projects this position onto the anchored path, so a
            # non-scalar translation would raise inside the control path
            # instead of degrading to the same no-pose branch a failed
            # lookup already takes.
            self.get_logger().debug(
                f"TF transform did not yield a planar pose (shape {xy.shape})"
            )
            return None
        return xy

    def _on_tick(self) -> None:
        # Pose lookup precedes both guards below: they key off robot motion, and
        # the wedge they exist for can hold with a stale plan or with no plan at
        # all, so gating the lookup behind either would blind them.
        robot_xy = self._lookup_robot_xy()
        if robot_xy is not None:
            self._track_stationary(robot_xy)
            self._release_fallback_if_clear(robot_xy)
            # Non-mutating projection: refreshes the cross-track the
            # admission rule reads without disturbing the cursor the policy
            # is driven from. Kept ahead of every guard below so a plan
            # arriving during a stale-plan window is still judged against a
            # current measurement.
            projection = self._generator.project(robot_xy)
            if projection is not None:
                self._last_cross_track_m = projection[1]

        self._maybe_log_anchor_status()

        if not self._generator.has_path:
            return  # No plan yet -- do not publish a subgoal.

        if not self._plan_fresh(time.monotonic()):
            # The plan went stale (planner died / replanning stopped).
            # Suppress the subgoal so the inference node's subgoal watchdog
            # zero-twists, rather than rolling the cursor along a stale
            # path forever -- unless suppressing is itself what holds the
            # robot in the pose the planner refuses.
            if not self._starvation_hold_publishes():
                if not self._stale_plan_logged:
                    self.get_logger().warning(
                        f"plan is stale (older than {self._path_timeout_s:.1f} s); "
                        "suppressing rolling-subgoal output until a fresh plan arrives."
                    )
                    self._stale_plan_logged = True
                return
            self.get_logger().warning(
                "Starvation hold: republishing the last valid subgoal "
                f"({self._hold_ticks_remaining} ticks left); the plan is stale "
                "and the planner is refusing the robot's pose.",
                throttle_duration_sec=1.0,
            )
        else:
            self._stale_plan_logged = False

        if robot_xy is None:
            return

        state = self._generator.update(robot_xy)
        if state is None:
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.pose.position.x = float(state.subgoal_xy[0])
        msg.pose.position.y = float(state.subgoal_xy[1])
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quaternion(state.subgoal_heading)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self._subgoal_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SubgoalGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
