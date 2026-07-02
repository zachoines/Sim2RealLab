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

    def test_tf_not_consulted_while_plan_stale(self) -> None:
        node = _node()
        try:
            node._generator.set_path(np.array([(0.0, 0.0), (1.0, 0.0)]))
            node._subgoal_pub = MagicMock()
            node._tf_buffer = MagicMock()
            node._last_plan_rx_t = 0.0  # stale -> suppress before TF
            node._on_tick()
            node._tf_buffer.lookup_transform.assert_not_called()
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

    def test_tick_noops_when_telemetry_stale(self) -> None:
        node = self._armed_node()
        try:
            node._last_goal_telemetry_rx_t = time.monotonic() - 10.0
            node._on_replan_tick()
            node._planner_client.send_goal_async.assert_not_called()
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
