"""Rolling-subgoal generator node: goal telemetry -> plan -> subgoal pose.

Owns hybrid replanning: while the inference node's active-goal telemetry
is fresh, calls Nav2's ``ComputePathToPose`` action on its own cadence
and installs the path from the **action result**. The ``/plan`` topic
subscription remains as a fallback input only. Each tick it looks up the
robot pose via the same ``map -> base_link`` TF the inference node uses,
runs the pure :class:`RollingSubgoalGenerator`, and publishes the rolling
subgoal as a ``geometry_msgs/PoseStamped`` on a dedicated topic.

The active-goal topic is **status telemetry**, direction inference node ->
here: it gates replanning and provides the plan target. It is NOT a goal
command channel — ``navigate_to_pose`` on the inference node remains the
sole way to command a mission. Telemetry staleness (mission ended, node
died) stops replanning; the plan then ages past ``path_timeout_s``, the
subgoal is suppressed, and the inference watchdog zero-twists.

A dedicated topic is deliberate: the mission goal reaches the inference
node through its ``navigate_to_pose`` action (latched per mission; a new
or preempting goal resets the policy's hidden state), while the rolling
subgoal is a streamed setpoint that advances every tick -- the two must
not share a channel, or a recurrent policy would reset continuously. The
published pose is in the ``map`` frame; the body-frame observation
transform the policy consumes lives in the inference node.

ROS glue only -- all selection math is in :mod:`strafer_inference.generator`.
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
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node

from strafer_shared.constants import SUBGOAL_LOOKAHEAD_M

from .generator import RollingSubgoalGenerator

# In-flight replan requests older than this are treated as lost (planner
# died mid-request; rclpy leaves the future pending forever). Comfortably
# above worst-case planner latency, well below a mission timeout.
_REPLAN_ABANDON_S = 2.0

# A /plan whose terminal pose is farther than this from the active goal
# was computed for a different (e.g. just-preempted) goal — do not
# install it over the action-result path. Matches Nav2's default goal
# tolerance scale.
_PLAN_GOAL_MATCH_M = 0.5


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

        # Monotonic receipt time of the latest valid plan. Drives the
        # plan-staleness guard: subgoal publishing stops once the plan ages
        # past path_timeout_s.
        self._last_plan_rx_t: Optional[float] = None
        self._stale_plan_logged = False

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
        self._planner_client = ActionClient(
            self, ComputePathToPose, planner_action
        )
        self._replan_timer = self.create_timer(
            self._replan_period_s, self._on_replan_tick
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
            f"active_goal_topic={active_goal_topic}"
        )

    def _install_path(self, msg: Path, source: str) -> None:
        """Install a fresh global plan and rewind the cursor."""
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
        self._generator.set_path(path_xy)
        self._last_plan_rx_t = time.monotonic()
        self.get_logger().debug(
            f"New plan from {source}: {len(msg.poses)} poses, total arc "
            f"{self._generator.total_arc:.3f} m; cursor rewound."
        )

    def _on_plan(self, msg: Path) -> None:
        # Fallback input only — the primary path arrives through the
        # ComputePathToPose action result in _on_replan_result. The
        # planner server also mirrors OUR requests onto /plan, so during
        # a mission only accept paths that actually end at the active
        # goal — otherwise the side-effect echo of a just-superseded
        # request would reinstall the discarded path.
        if self._active_goal is not None and msg.poses:
            terminal = msg.poses[-1]
            dx = terminal.pose.position.x - self._active_goal.pose.position.x
            dy = terminal.pose.position.y - self._active_goal.pose.position.y
            if math.hypot(dx, dy) > _PLAN_GOAL_MATCH_M:
                self.get_logger().debug(
                    "Ignoring /plan that does not end at the active goal."
                )
                return
        self._install_path(msg, source="/plan")

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
        # Stale telemetry means "no mission" (ended, or the inference node
        # died): stop fueling new plans; the plan then ages out and the
        # subgoal is suppressed.
        if self._active_goal is None or not self._goal_telemetry_fresh(
            time.monotonic()
        ):
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
        goal.planner_id = self._planner_id
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
            self.get_logger().warning(
                f"ComputePathToPose request failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return
        if goal_handle is None or not goal_handle.accepted:
            self._replan_inflight = False
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
            self.get_logger().warning(
                f"ComputePathToPose did not succeed (status {result.status}); "
                "keeping the previous plan.",
                throttle_duration_sec=5.0,
            )
            return
        self._install_path(result.result.path, source="ComputePathToPose")

    def _plan_fresh(self, now_monotonic_s: float) -> bool:
        """True if a plan (action result or /plan) arrived within
        ``path_timeout_s``."""
        return (
            self._last_plan_rx_t is not None
            and now_monotonic_s - self._last_plan_rx_t <= self._path_timeout_s
        )

    def _on_tick(self) -> None:
        if not self._generator.has_path:
            return  # No plan yet -- do not publish a subgoal.

        if not self._plan_fresh(time.monotonic()):
            # The plan went stale (planner died / replanning stopped).
            # Suppress the subgoal so the inference node's subgoal watchdog
            # zero-twists, rather than rolling the cursor along a stale
            # path forever.
            if not self._stale_plan_logged:
                self.get_logger().warning(
                    f"plan is stale (older than {self._path_timeout_s:.1f} s); "
                    "suppressing rolling-subgoal output until a fresh plan arrives."
                )
                self._stale_plan_logged = True
            return
        self._stale_plan_logged = False

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
            return

        robot_xy = np.array(
            [tf.transform.translation.x, tf.transform.translation.y],
            dtype=np.float64,
        )
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
