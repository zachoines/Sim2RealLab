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
from nav_msgs.msg import Path
from rclpy.clock import ClockType
from rclpy.parameter import Parameter

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


def _path(*xy: tuple[float, float]) -> Path:
    msg = Path()
    for x, y in xy:
        ps = PoseStamped()
        ps.pose.position.x = x
        ps.pose.position.y = y
        msg.poses.append(ps)
    return msg


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
