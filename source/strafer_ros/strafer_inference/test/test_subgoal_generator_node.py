"""Node-level tests for the subgoal-generator's plan-staleness guard and
its replan ownership (active-goal telemetry -> ComputePathToPose).

Spins up rclpy long enough to construct the node; the rolling-subgoal
selection math itself is covered rclpy-free in test_generator.py.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.clock import ClockType
from rclpy.parameter import Parameter

from strafer_inference.generator import (
    ADMIT_COLLISION,
    ADMIT_CROSS_TRACK,
    ADMIT_GOAL_CHANGED,
    ADMIT_NO_ANCHOR,
    ADMIT_ROLLING,
    REJECT_ANCHOR_HELD,
)
from strafer_inference.subgoal_generator_node import SubgoalGeneratorNode


@pytest.fixture(scope="module", autouse=True)
def _rclpy_session():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_overrides(**values) -> list[Parameter]:
    type_map = {
        str: Parameter.Type.STRING,
        bool: Parameter.Type.BOOL,
        int: Parameter.Type.INTEGER,
        float: Parameter.Type.DOUBLE,
    }
    return [Parameter(k, type_map[type(v)], v) for k, v in values.items()]


def _node(**overrides) -> SubgoalGeneratorNode:
    return SubgoalGeneratorNode(
        parameter_overrides=_make_overrides(path_timeout_s=1.0, **overrides)
    )


def _pose(x: float, y: float) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = "map"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
    return msg


def _path(*xy: tuple[float, float], stamp_ns: int = 0) -> Path:
    msg = Path()
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    for x, y in xy:
        ps = PoseStamped()
        ps.pose.position.x = x
        ps.pose.position.y = y
        msg.poses.append(ps)
    return msg


def _straight_path(n: int = 11, *, x0: float = 0.0, stamp_ns: int = 0) -> Path:
    """An n-metre path along +x starting at ``x0`` — the shape a Nav2 plan
    computed with ``use_start=False`` takes when the robot sits at ``x0``."""
    return _path(*[(x0 + float(i), 0.0) for i in range(n)], stamp_ns=stamp_ns)


def _plan_to_goal(robot_xy, goal_xy=(10.0, 0.0), *, spacing: float = 0.5,
                  stamp_ns: int = 0) -> Path:
    """A path from the robot's current pose to the goal, as Nav2 returns.

    Rooted under the robot in x AND y; rooting in x only would show zero
    cross-track by construction and hide the effect under test.
    """
    start = np.asarray(robot_xy, dtype=float)
    goal = np.asarray(goal_xy, dtype=float)
    total = float(np.linalg.norm(goal - start))
    n = max(int(total / spacing), 1)
    pts = [tuple(start + (goal - start) * (i / n)) for i in range(n + 1)]
    return _path(*pts, stamp_ns=stamp_ns)


def _costmap(
    *,
    blocked_cells: tuple[tuple[int, int], ...] = (),
    resolution: float = 0.05,
    width: int = 400,
    height: int = 400,
    origin_x: float = -10.0,
    origin_y: float = -10.0,
    frame_id: str = "map",
    cost: int = 100,
) -> OccupancyGrid:
    """A global costmap with the named ``(col, row)`` cells at ``cost``."""
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.info.resolution = resolution
    msg.info.width = width
    msg.info.height = height
    msg.info.origin.position.x = origin_x
    msg.info.origin.position.y = origin_y
    data = [0] * (width * height)
    for col, row in blocked_cells:
        data[row * width + col] = cost
    msg.data = data
    return msg


def _cells_for(node: SubgoalGeneratorNode, *xy: tuple[float, float]):
    """(col, row) costmap indices, divide-then-floor as the node does.

    ``//`` takes a different float path and would place the obstacle one
    cell off the waypoint.
    """
    info = _costmap().info
    return tuple(
        (
            int(np.floor((x - info.origin.position.x) / info.resolution)),
            int(np.floor((y - info.origin.position.y) / info.resolution)),
        )
        for x, y in xy
    )


class TestPlanFresh(unittest.TestCase):
    def test_none_is_stale(self) -> None:
        node = _node()
        try:
            self.assertFalse(node._plan_fresh(100.0))  # never received a plan
        finally:
            node.destroy_node()

    def test_within_timeout_is_fresh(self) -> None:
        node = _node()
        try:
            node._last_plan_rx_t = 100.0
            self.assertTrue(node._plan_fresh(100.5))     # 0.5 s <= 1.0 s budget
            self.assertTrue(node._plan_fresh(101.0))     # 1.0 s == budget
            self.assertFalse(node._plan_fresh(101.001))  # just past budget
        finally:
            node.destroy_node()


class TestStalePlanSuppressesSubgoal(unittest.TestCase):
    def test_tick_suppresses_publish_when_plan_stale(self) -> None:
        node = _node()
        try:
            # has_path True (a plan was installed) but its receipt is ancient.
            node._generator.set_path(np.array([(0.0, 0.0), (1.0, 0.0)]))
            node._last_plan_rx_t = 0.0  # far older than path_timeout_s
            pub = MagicMock()
            node._subgoal_pub = pub
            node._on_tick()
            pub.publish.assert_not_called()
            self.assertTrue(node._stale_plan_logged)
        finally:
            node.destroy_node()

    def test_no_path_returns_before_staleness_check(self) -> None:
        node = _node()
        try:
            pub = MagicMock()
            node._subgoal_pub = pub
            node._on_tick()  # no plan yet -> early return, no publish, no crash
            pub.publish.assert_not_called()
        finally:
            node.destroy_node()

    def test_fresh_tick_resets_stale_log_flag(self) -> None:
        import time

        node = _node()
        try:
            node._generator.set_path(np.array([(0.0, 0.0), (1.0, 0.0)]))
            node._subgoal_pub = MagicMock()
            node._last_plan_rx_t = 0.0  # stale
            node._on_tick()
            self.assertTrue(node._stale_plan_logged)
            # A fresh plan resets the flag (so the warning can re-fire on a
            # future stale transition); the reset happens before the TF lookup.
            node._last_plan_rx_t = time.monotonic()
            node._on_tick()  # TF lookup fails (no transform) -> returns
            self.assertFalse(node._stale_plan_logged)
        finally:
            node.destroy_node()

    def test_tf_consulted_while_plan_stale_for_wedge_guards(self) -> None:
        """The pose lookup precedes the staleness check and the has_path
        guard.

        The parked-in-inflation guards key off whether the robot is moving, and
        the wedge they exist for holds with a stale plan or with no plan at all,
        so gating the lookup behind either would blind them. Suppression is
        unaffected: still no publish.
        """
        node = _node()
        try:
            node._generator.set_path(np.array([(0.0, 0.0), (1.0, 0.0)]))
            pub = MagicMock()
            node._subgoal_pub = pub
            node._tf_buffer = MagicMock()
            node._last_plan_rx_t = 0.0  # stale
            node._on_tick()
            node._tf_buffer.lookup_transform.assert_called_once()
            pub.publish.assert_not_called()
        finally:
            node.destroy_node()

    def test_tf_consulted_before_has_path_guard(self) -> None:
        node = _node()
        try:
            node._subgoal_pub = MagicMock()
            node._tf_buffer = MagicMock()
            node._on_tick()  # no plan at all
            node._tf_buffer.lookup_transform.assert_called_once()
        finally:
            node.destroy_node()


class TestReplanCadenceClock(unittest.TestCase):
    """The replan cadence must run on the wall (steady) clock, not the
    node clock. Under use_sim_time the node clock is sim time, and at a
    sub-unity RTF a sim-clock cadence fires far slower in wall time than
    the wall-clock plan-freshness window (time.monotonic), starving the
    plan and suppressing the subgoal — the failure this guards.
    """

    def test_replan_timer_is_wall_clock_under_sim_time(self) -> None:
        node = _node(use_sim_time=True)
        try:
            # Node clock is sim (ROS_TIME); the replan cadence stays wall.
            self.assertEqual(node.get_clock().clock_type, ClockType.ROS_TIME)
            self.assertEqual(
                node._replan_timer.clock.clock_type, ClockType.STEADY_TIME
            )
        finally:
            node.destroy_node()

    def test_replan_fires_on_wall_time_with_frozen_sim_clock(self) -> None:
        # use_sim_time=True + no /clock => sim clock frozen at 0, so a
        # ROS-time timer would never fire; the steady-clock cadence must.
        node = _node(use_sim_time=True)
        try:
            calls = {"n": 0}
            node._on_replan_tick = lambda: calls.__setitem__(  # type: ignore
                "n", calls["n"] + 1
            )
            node._replan_timer.callback = node._on_replan_tick
            deadline = time.monotonic() + 1.3
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            self.assertGreaterEqual(calls["n"], 2)  # ~2-3 at 0.5 s period
        finally:
            node.destroy_node()


class TestReplanOwnership(unittest.TestCase):
    """The generator owns hybrid replanning: fresh active-goal telemetry
    fuels ComputePathToPose requests on the cadence; stale telemetry (no
    mission) stops them; a moved goal (preemption) retargets immediately;
    a result for a superseded goal is discarded.
    """

    def _armed_node(self) -> SubgoalGeneratorNode:
        """Node with fresh telemetry and a ready mocked planner client."""
        node = _node()
        node._active_goal = _pose(2.0, 1.0)
        node._last_goal_telemetry_rx_t = time.monotonic()
        client = MagicMock()
        client.server_is_ready.return_value = True
        node._planner_client = client
        return node

    def test_first_telemetry_triggers_immediate_replan(self) -> None:
        node = _node()
        try:
            node._request_replan = MagicMock()  # type: ignore
            node._on_active_goal(_pose(2.0, 0.0))
            node._request_replan.assert_called_once()
        finally:
            node.destroy_node()

    def test_keepalive_same_pose_does_not_retrigger(self) -> None:
        node = _node()
        try:
            node._request_replan = MagicMock()  # type: ignore
            node._on_active_goal(_pose(2.0, 0.0))
            node._on_active_goal(_pose(2.0, 0.0))  # ~1 Hz keep-alive
            node._request_replan.assert_called_once()
        finally:
            node.destroy_node()

    def test_moved_goal_retriggers_immediately(self) -> None:
        # A preempting goal is a fresh accept -> fresh telemetry with a
        # new pose; the generator retargets without waiting the cadence.
        node = _node()
        try:
            node._request_replan = MagicMock()  # type: ignore
            node._on_active_goal(_pose(2.0, 0.0))
            node._on_active_goal(_pose(-3.0, 0.0))
            self.assertEqual(node._request_replan.call_count, 2)
        finally:
            node.destroy_node()

    def test_tick_noops_when_idle(self) -> None:
        node = _node()
        try:
            client = MagicMock()
            client.server_is_ready.return_value = True
            node._planner_client = client
            node._on_replan_tick()  # no active goal ever
            client.send_goal_async.assert_not_called()
        finally:
            node.destroy_node()

    def test_tick_noops_and_clears_goal_when_telemetry_stale(self) -> None:
        node = self._armed_node()
        try:
            node._last_goal_telemetry_rx_t = time.monotonic() - 10.0
            node._on_replan_tick()
            node._planner_client.send_goal_async.assert_not_called()
            # Goal dropped so the /plan fallback is not gated on a dead
            # mission's goal.
            self.assertIsNone(node._active_goal)
        finally:
            node.destroy_node()

    def test_tick_sends_compute_path_request(self) -> None:
        node = self._armed_node()
        try:
            node._on_replan_tick()
            node._planner_client.send_goal_async.assert_called_once()
            goal = node._planner_client.send_goal_async.call_args[0][0]
            self.assertEqual(goal.planner_id, "GridBased")
            self.assertFalse(goal.use_start)
            self.assertAlmostEqual(goal.goal.pose.position.x, 2.0)
            self.assertTrue(node._replan_inflight)

            node._on_replan_tick()  # in-flight guard: no stacking
            node._planner_client.send_goal_async.assert_called_once()
        finally:
            node.destroy_node()

    def test_planner_unready_skips_and_logs_once(self) -> None:
        node = self._armed_node()
        try:
            node._planner_client.server_is_ready.return_value = False
            node._on_replan_tick()
            node._on_replan_tick()
            node._planner_client.send_goal_async.assert_not_called()
            self.assertFalse(node._replan_inflight)
        finally:
            node.destroy_node()

    @staticmethod
    def _result_future(path: Path, status: int = GoalStatus.STATUS_SUCCEEDED):
        wrapper = MagicMock()
        wrapper.status = status
        wrapper.result.path = path
        future = MagicMock()
        future.result.return_value = wrapper
        return future

    def test_result_installs_path_and_freshens_plan(self) -> None:
        node = self._armed_node()
        try:
            node._replan_inflight = True
            node._replan_goal_xy = (2.0, 1.0)

            node._on_replan_result(
                self._result_future(_path((0.0, 0.0), (2.0, 1.0)))
            )

            self.assertFalse(node._replan_inflight)
            self.assertTrue(node._generator.has_path)
            self.assertTrue(node._plan_fresh(time.monotonic()))
        finally:
            node.destroy_node()

    def test_non_succeeded_result_keeps_previous_plan(self) -> None:
        node = self._armed_node()
        try:
            node._replan_inflight = True
            node._replan_goal_xy = (2.0, 1.0)

            node._on_replan_result(
                self._result_future(
                    _path((0.0, 0.0), (2.0, 1.0)),
                    status=GoalStatus.STATUS_ABORTED,
                )
            )

            self.assertFalse(node._replan_inflight)
            self.assertFalse(node._generator.has_path)
            self.assertIsNone(node._last_plan_rx_t)
        finally:
            node.destroy_node()

    def test_result_for_superseded_goal_discarded_and_rerequested(self) -> None:
        node = self._armed_node()
        try:
            node._replan_inflight = True
            node._replan_goal_xy = (2.0, 1.0)
            node._active_goal = _pose(-3.0, 0.0)  # preempted meanwhile

            node._on_replan_result(
                self._result_future(_path((0.0, 0.0), (2.0, 1.0)))
            )

            self.assertFalse(node._generator.has_path)
            self.assertIsNone(node._last_plan_rx_t)
            # The retarget swallowed by the in-flight guard is re-fired
            # here, for the NEW goal.
            node._planner_client.send_goal_async.assert_called_once()
            goal = node._planner_client.send_goal_async.call_args[0][0]
            self.assertAlmostEqual(goal.goal.pose.position.x, -3.0)
            self.assertTrue(node._replan_inflight)
        finally:
            node.destroy_node()

    def test_stale_inflight_is_abandoned_and_resends(self) -> None:
        """A planner death mid-request leaves the future pending forever;
        the abandonment window must un-wedge replanning."""
        node = self._armed_node()
        try:
            node._replan_inflight = True
            node._replan_sent_t = time.monotonic() - 10.0  # long lost

            node._on_replan_tick()

            node._planner_client.send_goal_async.assert_called_once()
        finally:
            node.destroy_node()

    def test_goal_response_exception_clears_inflight(self) -> None:
        node = self._armed_node()
        try:
            node._replan_inflight = True
            future = MagicMock()
            future.result.side_effect = RuntimeError("client torn down")

            node._on_replan_goal_response(future)

            self.assertFalse(node._replan_inflight)
        finally:
            node.destroy_node()

    def test_plan_topic_ignores_path_for_a_different_goal(self) -> None:
        """The planner mirrors our own requests onto /plan; a mission must
        not reinstall a superseded request's path via the fallback."""
        node = self._armed_node()  # active goal at (2.0, 1.0)
        try:
            node._on_plan(_path((0.0, 0.0), (-3.0, 0.0)))  # old goal's path
            self.assertFalse(node._generator.has_path)

            node._on_plan(_path((0.0, 0.0), (2.0, 1.0)))  # matches goal
            self.assertTrue(node._generator.has_path)
        finally:
            node.destroy_node()

    def test_plan_topic_installs_unconditionally_when_idle(self) -> None:
        node = _node()  # no active goal: legacy topic-driven fallback
        try:
            node._on_plan(_path((0.0, 0.0), (9.0, 9.0)))
            self.assertTrue(node._generator.has_path)
        finally:
            node.destroy_node()

    def test_rejected_request_clears_inflight(self) -> None:
        node = self._armed_node()
        try:
            node._replan_inflight = True
            goal_handle = MagicMock()
            goal_handle.accepted = False
            future = MagicMock()
            future.result.return_value = goal_handle

            node._on_replan_goal_response(future)

            self.assertFalse(node._replan_inflight)
        finally:
            node.destroy_node()

    def test_accepted_request_chains_result_callback(self) -> None:
        node = self._armed_node()
        try:
            node._replan_inflight = True
            goal_handle = MagicMock()
            goal_handle.accepted = True
            result_future = MagicMock()
            goal_handle.get_result_async.return_value = result_future
            future = MagicMock()
            future.result.return_value = goal_handle

            node._on_replan_goal_response(future)

            result_future.add_done_callback.assert_called_once()
            self.assertTrue(node._replan_inflight)  # still awaiting result
        finally:
            node.destroy_node()


class TestParkedInInflationDeadlock(unittest.TestCase):
    """A robot parked in the costmap inflation halo is refused by the planner
    as a start pose; every replan aborts, the plan ages out, the subgoal is
    suppressed and the policy zero-twists — so the pose never changes and
    nothing clears the refusal.

    Two guards break that cycle. Both must stay loud: starvation must stop being
    permanent without stopping being visible.
    """

    GUARDS = dict(
        fallback_planner_id="GridBasedRelaxed",
        planner_refusals_before_fallback=2,
        fallback_release_m=0.25,
        starvation_hold_s=1.0,
        starvation_stationary_m=0.05,
        starvation_stationary_s=0.1,
        update_period_s=0.1,          # -> 10 hold ticks, 1 stationary tick
    )

    def _node_guarded(self, **extra) -> SubgoalGeneratorNode:
        node = _node(**{**self.GUARDS, **extra})
        node._subgoal_pub = MagicMock()
        return node

    @staticmethod
    def _park(node: SubgoalGeneratorNode, x: float = 0.0, y: float = 0.0):
        """Pin the robot's pose so the tick sees a deterministic position."""
        node._lookup_robot_xy = lambda: np.array(  # type: ignore[method-assign]
            [x, y], dtype=np.float64
        )

    def _wedge(self, node: SubgoalGeneratorNode) -> None:
        """Mission in progress, one good plan installed, then the planner
        starts refusing and the plan ages out."""
        node._active_goal = _pose(2.0, 0.0)
        node._last_goal_telemetry_rx_t = time.monotonic()
        node._install_path(_path((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
                           source="test")
        node._note_planner_refusal("status 6")
        node._note_planner_refusal("status 6")
        node._last_plan_rx_t = 0.0     # aged far past path_timeout_s

    # -- guard 1: the escape-hatch planner --------------------------------

    def test_primary_planner_used_until_refusals_reach_threshold(self) -> None:
        node = self._node_guarded()
        try:
            self._park(node)
            node._on_tick()                       # seed a robot pose
            self.assertEqual(node._select_planner_id(), "GridBased")
            node._note_planner_refusal("status 6")
            self.assertEqual(node._select_planner_id(), "GridBased")
            node._note_planner_refusal("status 6")
            self.assertEqual(node._select_planner_id(), "GridBasedRelaxed")
        finally:
            node.destroy_node()

    def test_fallback_disabled_by_default_in_code(self) -> None:
        # Code defaults leave both guards off, so a bare `ros2 run` never
        # engages them; the shipped subgoal_generator.yaml arms them.
        node = _node()
        try:
            self.assertEqual(node._fallback_planner_id, "")
            for _ in range(10):
                node._note_planner_refusal("status 6")
            self.assertEqual(node._select_planner_id(), "GridBased")
        finally:
            node.destroy_node()

    def test_shipped_yaml_arms_the_fallback(self) -> None:
        import os
        import yaml
        from ament_index_python.packages import get_package_share_directory

        path = os.path.join(
            get_package_share_directory("strafer_inference"),
            "config", "subgoal_generator.yaml",
        )
        with open(path) as f:
            params = yaml.safe_load(f)
        p = params["strafer_subgoal_generator"]["ros__parameters"]
        # Must match the plugin registered in strafer_navigation's
        # nav2_params.yaml, or the fallback request is rejected by name.
        self.assertEqual(p["fallback_planner_id"], "GridBasedRelaxed")
        self.assertGreater(p["starvation_hold_s"], 0.0)

    def test_fallback_planner_id_is_registered_in_nav2_params(self) -> None:
        """Cross-package pin: the id the generator asks for must be a plugin
        the planner server actually loads, and it must NOT be the primary —
        the whole point is a planner that clears the robot's own cell."""
        import os
        import yaml
        from ament_index_python.packages import get_package_share_directory

        gen = yaml.safe_load(open(os.path.join(
            get_package_share_directory("strafer_inference"),
            "config", "subgoal_generator.yaml",
        )))["strafer_subgoal_generator"]["ros__parameters"]
        nav = yaml.safe_load(open(os.path.join(
            get_package_share_directory("strafer_navigation"),
            "config", "nav2_params.yaml",
        )))["planner_server"]["ros__parameters"]

        fallback = gen["fallback_planner_id"]
        self.assertIn(fallback, nav["planner_plugins"])
        self.assertIn(gen["planner_id"], nav["planner_plugins"])
        self.assertNotEqual(fallback, gen["planner_id"])
        self.assertEqual(
            nav[fallback]["plugin"], "nav2_navfn_planner/NavfnPlanner",
            "the escape hatch must be a planner that clears the robot's own "
            "cell before propagating; SmacPlanner2D refuses an inflated start",
        )

    def test_fallback_released_only_after_the_robot_clears_the_halo(self) -> None:
        node = self._node_guarded()
        try:
            self._park(node, 0.0, 0.0)
            node._on_tick()
            node._note_planner_refusal("status 6")
            node._note_planner_refusal("status 6")
            self.assertEqual(node._select_planner_id(), "GridBasedRelaxed")

            # Nudged, but still inside the halo it wedged in.
            self._park(node, 0.10, 0.0)
            node._on_tick()
            self.assertEqual(node._select_planner_id(), "GridBasedRelaxed")

            # Clear of it -> the primary is trusted again.
            self._park(node, 0.30, 0.0)
            node._on_tick()
            self.assertEqual(node._select_planner_id(), "GridBased")
        finally:
            node.destroy_node()

    def test_unavailable_planner_is_not_a_refusal(self) -> None:
        """A planner that is DOWN is a different fault: it must keep starving
        the watchdog rather than tripping these guards."""
        node = self._node_guarded()
        try:
            node._active_goal = _pose(2.0, 0.0)
            node._last_goal_telemetry_rx_t = time.monotonic()
            client = MagicMock()
            client.server_is_ready.return_value = False
            node._planner_client = client
            for _ in range(5):
                node._on_replan_tick()
            self.assertEqual(node._planner_refusals, 0)
            self.assertEqual(node._select_planner_id(), "GridBased")
        finally:
            node.destroy_node()

    def test_transport_failure_is_not_a_refusal(self) -> None:
        """Same boundary on the other two failure paths: an rcl teardown or
        transport error means the request never reached a planner, so it must
        not open a hold window on what is really a dead planner."""
        node = self._node_guarded()
        try:
            for _ in range(5):
                broken = MagicMock()
                broken.result.side_effect = RuntimeError("client torn down")
                node._on_replan_goal_response(broken)
                node._on_replan_result(broken)
            self.assertEqual(node._planner_refusals, 0)
            self.assertEqual(node._select_planner_id(), "GridBased")
        finally:
            node.destroy_node()

    def test_aborted_result_is_a_refusal(self) -> None:
        """The signature these guards exist for: the planner answers, and
        says no."""
        node = self._node_guarded()
        try:
            node._active_goal = _pose(2.0, 0.0)
            node._replan_goal_xy = (2.0, 0.0)
            wrapper = MagicMock()
            wrapper.status = GoalStatus.STATUS_ABORTED
            future = MagicMock()
            future.result.return_value = wrapper
            node._on_replan_result(future)
            self.assertEqual(node._planner_refusals, 1)
        finally:
            node.destroy_node()

    # -- guard 2: the bounded starvation hold ------------------------------

    def test_stale_plan_is_republished_while_wedged_then_stops(self) -> None:
        node = self._node_guarded()
        try:
            self._park(node)
            node._on_tick()          # seed pose (plan still fresh)
            self._wedge(node)

            # Stationary for one tick, then the window opens.
            node._on_tick()
            published = 0
            for _ in range(40):
                node._subgoal_pub.publish.reset_mock()
                node._on_tick()
                published += node._subgoal_pub.publish.call_count

            # Bounded: ~10 ticks of hold (1.0 s / 0.1 s), never unbounded.
            self.assertGreater(published, 0, "the deadlock stayed permanent")
            self.assertLessEqual(published, 12)
            # And it ends fail-loud: suppression resumes.
            node._subgoal_pub.publish.reset_mock()
            node._on_tick()
            node._subgoal_pub.publish.assert_not_called()
        finally:
            node.destroy_node()

    def test_hold_does_not_fire_while_the_robot_is_moving(self) -> None:
        """A stale plan on a MOVING robot is the ordinary case the original
        suppression is for — the policy must not chase a stale path."""
        node = self._node_guarded()
        try:
            self._park(node, 0.0, 0.0)
            node._on_tick()
            self._wedge(node)
            for i in range(1, 20):
                self._park(node, 0.5 * i, 0.0)   # clearly moving
                node._subgoal_pub.publish.reset_mock()
                node._on_tick()
                node._subgoal_pub.publish.assert_not_called()
        finally:
            node.destroy_node()

    def test_hold_never_republishes_a_path_for_a_previous_goal(self) -> None:
        """The cross-mission wedge: a NEW goal the planner has never produced
        a path for must NOT drive the policy at the old mission's subgoal.
        That case is the escape-hatch planner's job, not the hold's."""
        node = self._node_guarded()
        try:
            self._park(node)
            node._on_tick()
            self._wedge(node)
            node._active_goal = _pose(-5.0, 4.0)   # new mission, no new plan

            for _ in range(30):
                node._on_tick()
            node._subgoal_pub.publish.assert_not_called()
        finally:
            node.destroy_node()

    def test_hold_is_one_window_per_episode(self) -> None:
        """After the window is spent, a still-wedged robot fails loud rather
        than republishing a stale subgoal forever."""
        node = self._node_guarded()
        try:
            self._park(node)
            node._on_tick()
            self._wedge(node)
            for _ in range(60):                    # exhaust the window
                node._on_tick()
            self.assertFalse(node._hold_armed)
            node._subgoal_pub.publish.reset_mock()
            for _ in range(30):
                node._on_tick()
            node._subgoal_pub.publish.assert_not_called()
        finally:
            node.destroy_node()

    def test_fresh_plan_rearms_the_hold_and_clears_refusals(self) -> None:
        node = self._node_guarded()
        try:
            self._park(node)
            node._on_tick()
            self._wedge(node)
            for _ in range(60):
                node._on_tick()
            self.assertFalse(node._hold_armed)

            node._install_path(_path((0.0, 0.0), (2.0, 0.0)), source="test")
            self.assertTrue(node._hold_armed)
            self.assertEqual(node._planner_refusals, 0)
            self.assertEqual(node._hold_ticks_remaining, 0)
        finally:
            node.destroy_node()

    def test_hold_budget_is_counted_in_policy_ticks_not_wall_seconds(self) -> None:
        """At the rig's measured ~0.13 RTF a wall-clock budget would buy well
        under a second of real robot motion. The tick timer runs on the node
        clock, so a tick count is a sim-time budget at any RTF."""
        node = _node(starvation_hold_s=5.0, update_period_s=1.0 / 30.0)
        try:
            self.assertEqual(node._hold_ticks_budget, 150)   # 5 s @ 30 Hz
        finally:
            node.destroy_node()

    def test_zero_hold_disables_the_republish(self) -> None:
        node = self._node_guarded(starvation_hold_s=0.0)
        try:
            self._park(node)
            node._on_tick()
            self._wedge(node)
            for _ in range(30):
                node._on_tick()
            node._subgoal_pub.publish.assert_not_called()
        finally:
            node.destroy_node()


class TestComposedStaleBound(unittest.TestCase):
    """Regression anchor for the stale-plan -> STOP bound. The two serial
    stages each carry ~half of the trust budget so they compose to ~2.0 s:
    the generator stops publishing the subgoal one stage after /plan dies,
    then the inference watchdog trips one stage later. If either default
    drifts back to the full budget the composed bound doubles, so pin the
    sum here.
    """

    def test_generator_and_inference_budgets_compose_to_two_seconds(self) -> None:
        from strafer_inference.watchdog import WatchdogTimeouts

        # Build WITHOUT a path_timeout_s override so _path_timeout_s reflects
        # the DECLARED default; this fails if the generator's param default
        # drifts back to the full budget (which would double the composed
        # bound). The _node() helper injects 1.0, which would mask that.
        node = SubgoalGeneratorNode(parameter_overrides=[])
        try:
            generator_budget = node._path_timeout_s
        finally:
            node.destroy_node()
        # Inference half: the WatchdogTimeouts.path default the inference node
        # feeds from its own path_timeout_s param.
        inference_budget = WatchdogTimeouts(
            imu=0.2, joint_states=0.2, odom=0.2, depth=0.5, tf=0.5,
        ).path
        self.assertEqual(generator_budget, pytest.approx(1.0))
        self.assertEqual(inference_budget, pytest.approx(1.0))
        self.assertEqual(
            generator_budget + inference_budget, pytest.approx(2.0)
        )


# =============================================================================
# Mission anchoring: one path per goal, replaced only on admission
# =============================================================================


def _anchored_node(**overrides) -> SubgoalGeneratorNode:
    """A node with a 0..10 m anchor along +x and the robot on it at 0."""
    node = _node(**overrides)
    node._active_goal = _pose(10.0, 0.0)
    node._last_goal_telemetry_rx_t = time.monotonic()
    node._last_robot_xy = np.array([0.0, 0.0])
    node._new_mission_pending = True
    node._consider_plan(_straight_path(stamp_ns=1), source="test")
    node._new_mission_pending = False
    return node


class TestAnchoringMode(unittest.TestCase):
    def test_mission_is_the_shipped_default_in_code(self) -> None:
        node = _node()
        try:
            self.assertFalse(node._rolling_anchoring)
            self.assertEqual(
                node.get_parameter("subgoal_anchoring").value, "mission"
            )
        finally:
            node.destroy_node()

    def test_shipped_yaml_selects_mission_anchoring(self) -> None:
        import os
        import yaml
        from ament_index_python.packages import get_package_share_directory

        path = os.path.join(
            get_package_share_directory("strafer_inference"),
            "config", "subgoal_generator.yaml",
        )
        with open(path) as fh:
            params = yaml.safe_load(fh)
        p = params["strafer_subgoal_generator"]["ros__parameters"]
        self.assertEqual(p["subgoal_anchoring"], "mission")
        self.assertGreater(p["admission_cross_track_m"], 0.0)
        self.assertTrue(p["admission_collision_check"])

    def test_unknown_mode_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="subgoal_anchoring"):
            _node(subgoal_anchoring="sometimes")

    def test_rolling_mode_is_selectable_as_the_named_fallback(self) -> None:
        node = _node(subgoal_anchoring="rolling")
        try:
            self.assertTrue(node._rolling_anchoring)
        finally:
            node.destroy_node()


class TestAnchorHeldAgainstRepeatPlans(unittest.TestCase):
    """A fresh plan is planner liveness, not a new path."""

    def test_second_distinct_plan_does_not_re_root_the_anchor(self) -> None:
        node = _anchored_node()
        try:
            node._last_robot_xy = np.array([4.0, 0.05])
            node._last_cross_track_m = 0.05
            anchor_before = node._generator.path.copy()
            # A path rooted under the robot, as Nav2 sends.
            node._consider_plan(
                _straight_path(x0=4.0, stamp_ns=2), source="test"
            )
            np.testing.assert_allclose(node._generator.path, anchor_before)
            self.assertEqual(node._anchors_rejected, 1)
            self.assertEqual(node._admit_reasons.get(REJECT_ANCHOR_HELD), 1)
        finally:
            node.destroy_node()

    def test_rejected_plan_still_counts_as_planner_liveness(self) -> None:
        """A held anchor must not read as a dead planner."""
        node = _anchored_node()
        try:
            node._last_cross_track_m = 0.05
            node._last_plan_rx_t = 0.0            # aged far past path_timeout_s
            node._planner_refusals = 3
            node._hold_armed = False
            node._consider_plan(
                _straight_path(x0=4.0, stamp_ns=3), source="test"
            )
            self.assertTrue(node._plan_fresh(time.monotonic()))
            self.assertEqual(node._planner_refusals, 0)
            self.assertTrue(node._hold_armed)
        finally:
            node.destroy_node()

    def test_republished_identical_plan_is_deduped(self) -> None:
        """planner_server mirrors /plan, so arrival is meaningless."""
        node = _anchored_node()
        try:
            seen_before = node._plans_seen
            for _ in range(5):
                node._consider_plan(_straight_path(stamp_ns=1), source="test")
            self.assertEqual(node._plans_seen, seen_before)
            self.assertEqual(node._plans_republished, 5)
            self.assertTrue(node._plan_fresh(time.monotonic()))
        finally:
            node.destroy_node()

    def test_identical_geometry_with_a_new_stamp_is_a_new_plan(self) -> None:
        node = _anchored_node()
        try:
            node._last_cross_track_m = 0.05
            node._consider_plan(_straight_path(stamp_ns=99), source="test")
            self.assertEqual(node._plans_seen, 2)
            self.assertEqual(node._plans_republished, 0)
        finally:
            node.destroy_node()

    def test_rolling_mode_re_roots_on_every_plan(self) -> None:
        node = _node(subgoal_anchoring="rolling")
        try:
            node._active_goal = _pose(10.0, 0.0)
            node._last_robot_xy = np.array([0.0, 0.0])
            node._consider_plan(_straight_path(stamp_ns=1), source="test")
            node._last_cross_track_m = 0.0
            node._consider_plan(
                _straight_path(x0=4.0, stamp_ns=2), source="test"
            )
            self.assertEqual(node._generator.path[0][0], pytest.approx(4.0))
            self.assertEqual(node._admit_reasons.get(ADMIT_ROLLING), 2)
        finally:
            node.destroy_node()


class TestAdmissionRulesAtTheNode(unittest.TestCase):
    def test_first_plan_of_a_mission_anchors(self) -> None:
        node = _node()
        try:
            node._active_goal = _pose(10.0, 0.0)
            node._consider_plan(_straight_path(stamp_ns=1), source="test")
            self.assertTrue(node._generator.has_path)
            self.assertEqual(node._admit_reasons.get(ADMIT_NO_ANCHOR), 1)
        finally:
            node.destroy_node()

    def test_moved_goal_admits_a_replacement(self) -> None:
        node = _anchored_node()
        try:
            node._last_cross_track_m = 0.01
            node._active_goal = _pose(-5.0, 3.0)   # far from the anchor's goal
            node._consider_plan(
                _path((0.0, 0.0), (-5.0, 3.0), stamp_ns=4), source="test"
            )
            self.assertEqual(node._admit_reasons.get(ADMIT_GOAL_CHANGED), 1)
            self.assertEqual(len(node._generator.path), 2)
        finally:
            node.destroy_node()

    def test_new_mission_admits_even_when_the_goal_barely_moved(self) -> None:
        """A new mission must not inherit the previous anchor just because
        its goal sits within the provenance tolerance."""
        node = _anchored_node()
        try:
            node._last_cross_track_m = 0.01
            # Inside _PLAN_GOAL_MATCH_M, so provenance alone reads "same goal".
            node._on_active_goal(_pose(10.1, 0.0))
            self.assertTrue(node._new_mission_pending)
            node._consider_plan(
                _straight_path(x0=4.0, stamp_ns=5), source="test"
            )
            self.assertEqual(node._admit_reasons.get(ADMIT_GOAL_CHANGED), 1)
            self.assertFalse(node._new_mission_pending)
            self.assertEqual(node._generator.path[0][0], pytest.approx(4.0))
        finally:
            node.destroy_node()

    def test_cross_track_below_the_bound_holds_the_anchor(self) -> None:
        node = _anchored_node(admission_cross_track_m=0.5)
        try:
            node._last_cross_track_m = 0.49
            node._consider_plan(
                _straight_path(x0=4.0, stamp_ns=6), source="test"
            )
            self.assertEqual(node._admit_reasons.get(REJECT_ANCHOR_HELD), 1)
        finally:
            node.destroy_node()

    def test_cross_track_above_the_bound_admits(self) -> None:
        node = _anchored_node(admission_cross_track_m=0.5)
        try:
            node._last_cross_track_m = 0.51
            node._consider_plan(
                _straight_path(x0=4.0, stamp_ns=7), source="test"
            )
            self.assertEqual(node._admit_reasons.get(ADMIT_CROSS_TRACK), 1)
            self.assertEqual(node._generator.path[0][0], pytest.approx(4.0))
        finally:
            node.destroy_node()

    def test_admitted_replacement_seeds_the_cursor_by_projection(self) -> None:
        node = _anchored_node(admission_cross_track_m=0.5)
        try:
            # Robot is 4 m along, badly off-corridor; the replacement path
            # still runs 0..10 m (it did not root at the robot).
            node._last_robot_xy = np.array([4.0, 0.9])
            node._last_cross_track_m = 0.9
            node._consider_plan(_straight_path(stamp_ns=8), source="test")
            self.assertEqual(node._admit_reasons.get(ADMIT_CROSS_TRACK), 1)
            # Progress preserved: the cursor lands at the robot's projection,
            # not rewound to the path's start.
            self.assertEqual(node._generator.cursor_arc, pytest.approx(4.0))
        finally:
            node.destroy_node()


class TestCollisionAdmissionRule(unittest.TestCase):
    def test_blocked_cell_ahead_of_the_cursor_admits(self) -> None:
        node = _anchored_node()
        try:
            node._last_cross_track_m = 0.01
            node._on_costmap(
                _costmap(blocked_cells=_cells_for(node, (6.0, 0.0)))
            )
            self.assertTrue(node._anchor_in_collision())
            node._consider_plan(
                _straight_path(x0=4.0, stamp_ns=9), source="test"
            )
            self.assertEqual(node._admit_reasons.get(ADMIT_COLLISION), 1)
        finally:
            node.destroy_node()

    def test_blocked_cell_behind_the_cursor_is_ignored(self) -> None:
        node = _anchored_node()
        try:
            node._generator.update(np.array([6.0, 0.0]))   # cursor -> 6.0 m
            node._last_cross_track_m = 0.01
            node._on_costmap(
                _costmap(blocked_cells=_cells_for(node, (2.0, 0.0)))
            )
            self.assertFalse(node._anchor_in_collision())
            node._consider_plan(
                _straight_path(x0=6.0, stamp_ns=10), source="test"
            )
            self.assertEqual(node._admit_reasons.get(REJECT_ANCHOR_HELD), 1)
        finally:
            node.destroy_node()

    def test_inscribed_cost_counts_as_blocked(self) -> None:
        node = _anchored_node()
        try:
            node._on_costmap(
                _costmap(blocked_cells=_cells_for(node, (6.0, 0.0)), cost=99)
            )
            self.assertTrue(node._anchor_in_collision())
        finally:
            node.destroy_node()

    def test_sub_inscribed_inflation_cost_is_not_blocked(self) -> None:
        node = _anchored_node()
        try:
            node._on_costmap(
                _costmap(blocked_cells=_cells_for(node, (6.0, 0.0)), cost=98)
            )
            self.assertFalse(node._anchor_in_collision())
        finally:
            node.destroy_node()

    def test_no_costmap_abstains_and_warns_once(self) -> None:
        node = _anchored_node()
        try:
            warn = MagicMock()
            node.get_logger().warning = warn
            self.assertFalse(node._anchor_in_collision())
            self.assertFalse(node._anchor_in_collision())
            self.assertEqual(warn.call_count, 1)
        finally:
            node.destroy_node()

    def test_stale_costmap_abstains(self) -> None:
        node = _anchored_node(costmap_timeout_s=1.0)
        try:
            node._on_costmap(
                _costmap(blocked_cells=_cells_for(node, (6.0, 0.0)))
            )
            node._costmap_rx_t = time.monotonic() - 5.0
            self.assertFalse(node._anchor_in_collision())
        finally:
            node.destroy_node()

    def test_costmap_in_a_foreign_frame_is_rejected(self) -> None:
        node = _anchored_node()
        try:
            node._on_costmap(
                _costmap(
                    blocked_cells=_cells_for(node, (6.0, 0.0)),
                    frame_id="odom",
                )
            )
            self.assertIsNone(node._costmap_grid)
            self.assertFalse(node._anchor_in_collision())
        finally:
            node.destroy_node()

    def test_waypoints_outside_the_costmap_are_unknown_not_blocked(self) -> None:
        node = _anchored_node()
        try:
            # A tiny costmap that does not cover the path at all.
            node._on_costmap(_costmap(width=4, height=4, origin_x=50.0,
                                      origin_y=50.0))
            self.assertFalse(node._anchor_in_collision())
        finally:
            node.destroy_node()

    def test_check_disabled_never_subscribes_or_fires(self) -> None:
        node = _anchored_node(admission_collision_check=False)
        try:
            self.assertIsNone(node._costmap_sub)
            self.assertFalse(node._anchor_in_collision())
        finally:
            node.destroy_node()


class TestAnchoredCursorMonotonicityAtTheNode(unittest.TestCase):
    """Driving a mission at Nav2's replan cadence leaves the cursor
    monotonic and lets cross-track grow."""

    def _drive(self, node, drift_per_step: float, steps: int = 12):
        """One mission at Nav2's plan cadence, in the node's own order:
        robot moves, plan arrives (plus a republish), tick advances the
        cursor. The admission rule therefore sees the previous tick's
        cross-track, as it does in the node.
        """
        cursors, cross, remaining = [], [], []
        for k in range(1, steps + 1):
            robot = np.array([0.5 * k, drift_per_step * k])
            node._last_robot_xy = robot
            node._consider_plan(
                _plan_to_goal(robot, stamp_ns=100 + k),
                source="ComputePathToPose",
            )
            node._consider_plan(
                _plan_to_goal(robot, stamp_ns=100 + k), source="/plan"
            )
            state = node._generator.update(robot)
            node._last_cross_track_m = state.cross_track
            cursors.append(state.cursor_arc)
            cross.append(state.cross_track)
            remaining.append(state.total_arc - state.cursor_arc)
        return cursors, cross, remaining

    def test_cursor_is_monotonic_under_continuous_replanning(self) -> None:
        # Drift stays under admission_cross_track_m, so monotonicity is
        # asserted over one continuous anchored path.
        node = _anchored_node()
        try:
            cursors, _, _ = self._drive(node, 0.03)
            for prev, nxt in zip(cursors, cursors[1:]):
                self.assertGreaterEqual(nxt, prev - 1e-12)
            self.assertGreater(cursors[-1], 3.0)
            self.assertEqual(node._anchors_admitted, 1)   # the anchor itself
        finally:
            node.destroy_node()

    def test_cross_track_develops_instead_of_pinning_at_one_cell(self) -> None:
        node = _anchored_node()
        try:
            _, cross, _ = self._drive(node, 0.03)

            self.assertGreater(max(cross), 0.3)
            self.assertGreater(cross[-1], cross[0])
        finally:
            node.destroy_node()

    def test_losing_the_corridor_admits_and_preserves_progress(self) -> None:
        """Drift past the bound replaces the anchor; the remaining arc must
        not jump backwards across the replacement."""
        node = _anchored_node(admission_cross_track_m=0.5)
        try:
            _, cross, remaining = self._drive(node, 0.08)
            self.assertGreater(max(cross), 0.5)
            self.assertGreaterEqual(node._anchors_admitted, 2)
            self.assertGreater(node._admit_reasons.get(ADMIT_CROSS_TRACK, 0), 0)
            for prev, nxt in zip(remaining, remaining[1:]):
                self.assertLessEqual(nxt, prev + 1e-9)
        finally:
            node.destroy_node()

    def test_rolling_mode_reproduces_the_measured_failure(self) -> None:
        """Control arm — the legacy semantics, so the A/B is testable."""
        node = _node(subgoal_anchoring="rolling")
        try:
            node._active_goal = _pose(10.0, 0.0)
            node._last_robot_xy = np.array([0.0, 0.0])
            node._consider_plan(_straight_path(stamp_ns=1), source="test")
            _, cross, _ = self._drive(node, 0.05)
            # Every plan re-roots under the robot, so no cross-track can
            # accumulate.
            self.assertLess(max(cross), 1e-6)
        finally:
            node.destroy_node()

    def test_repeat_plans_do_not_rewind_the_cursor(self) -> None:
        node = _anchored_node()
        try:
            node._generator.update(np.array([5.0, 0.0]))
            node._last_cross_track_m = 0.0
            self.assertEqual(node._generator.cursor_arc, pytest.approx(5.0))
            for k in range(6):
                node._consider_plan(
                    _straight_path(x0=5.0, stamp_ns=200 + k), source="test"
                )
            self.assertEqual(node._generator.cursor_arc, pytest.approx(5.0))
            self.assertEqual(node._anchors_admitted, 1)   # the anchor itself
        finally:
            node.destroy_node()


class TestAnchorStatusLogging(unittest.TestCase):
    def test_periodic_log_reports_the_admission_histogram(self) -> None:
        node = _anchored_node(anchor_log_period_s=0.0001)
        try:
            info = MagicMock()
            node.get_logger().info = info
            node._last_cross_track_m = 0.01
            node._consider_plan(_straight_path(x0=4.0, stamp_ns=11),
                                source="test")
            node._last_anchor_log_t = 0.0
            node._maybe_log_anchor_status()
            line = info.call_args[0][0]
            self.assertIn("anchor status:", line)
            self.assertIn("mode=mission", line)
            self.assertIn("held=1", line)
            self.assertIn(REJECT_ANCHOR_HELD, line)
        finally:
            node.destroy_node()

    def test_zero_period_disables_the_log(self) -> None:
        node = _anchored_node(anchor_log_period_s=0.0)
        try:
            info = MagicMock()
            node.get_logger().info = info
            node._last_anchor_log_t = 0.0
            node._maybe_log_anchor_status()
            info.assert_not_called()
        finally:
            node.destroy_node()
