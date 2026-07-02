"""Phase 3 node-integration tests for the runtime: model-load failure
modes, reset-trigger wiring, and the recurrent two-pronged determinism
assertion at the policy-stub seam.

These tests spin up rclpy long enough to construct the node; they do
NOT subscribe / publish across the wire. The action server's blocking
``execute_callback`` is driven through a mocked goal handle.
"""

from __future__ import annotations

import math
import threading
import time
import unittest
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.parameter import Parameter

from strafer_inference.inference_node import InferenceNode


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
    overrides = []
    for name, value in values.items():
        if isinstance(value, list):
            ptype = Parameter.Type.STRING_ARRAY
        else:
            ptype = type_map[type(value)]
        overrides.append(Parameter(name, ptype, value))
    return overrides


class _FakeRecurrentPolicy:
    """Stand-in recurrent policy whose hidden state evolves on call and
    zeros on reset. Mirrors the contract that ``test_recurrent_contract_e2e``
    pins at the cross-format level; the inference node's job is to call
    reset() at the right boundaries, and this stub makes those boundaries
    observable through ``reset_calls`` and through the action stream.
    """

    is_recurrent = True

    def __init__(self) -> None:
        self.call_count = 0
        self.reset_calls = 0
        self._h = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.call_count += 1
        self._h += float(np.asarray(obs).sum()) * 0.001
        return np.array(
            [
                math.tanh(float(obs[0]) + self._h),
                math.tanh(float(obs[1]) - self._h),
                math.tanh(self._h),
            ],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self.reset_calls += 1
        self._h = 0.0


def _make_pose(x: float, y: float) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = "map"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
    return msg


# =============================================================================
# Model-load failure → action server unadvertised
# =============================================================================


class TestModelLoadFailure(unittest.TestCase):
    """Brief: model-load failure is fatal — the action server must NOT
    advertise, so JetsonRosClient's wait_for_server times out and the
    backend dispatcher falls back to nav2 per Phase 4.
    """

    def test_empty_model_path_skips_action_server(self) -> None:
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            self.assertIsNone(node._action_server)
            self.assertEqual(node._policy_load_error, "model_path is empty")
        finally:
            node.destroy_node()

    def test_nonexistent_model_path_skips_action_server(self) -> None:
        node = InferenceNode(
            parameter_overrides=_make_overrides(
                model_path="/nonexistent/path/to/policy.onnx",
            )
        )
        try:
            self.assertIsNone(node._action_server)
            self.assertIn(
                "not found", node._policy_load_error,
                f"Expected 'not found' in error, got: {node._policy_load_error}",
            )
        finally:
            node.destroy_node()

    def test_unloaded_node_still_runs_tick_path(self) -> None:
        """Tick should noop gracefully when no policy is loaded — the
        watchdog stale-source path still runs and a None policy short-
        circuits before any inference call."""
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            # Watchdog will report all sources stale; the early-return
            # publishes zero twist. Just verify no exception.
            node._on_tick()
        finally:
            node.destroy_node()

    def test_idle_watchdog_zero_twists_without_warning(self) -> None:
        """Idle (no active goal): the watchdog trips on goal/tf/subgoal
        every tick, but that is the resting state between missions — it
        must still zero-twist while staying quiet (no WARN spam at the
        tick rate, the operator complaint)."""
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            node._cmd_vel_pub = MagicMock()
            logger = MagicMock()
            node.get_logger = lambda: logger  # type: ignore

            node._on_tick()  # no active goal

            logger.warning.assert_not_called()
            node._cmd_vel_pub.publish.assert_called_once()  # safe zero twist
        finally:
            node.destroy_node()

    def test_mission_active_watchdog_warns_on_stale_source(self) -> None:
        """A stale source WHILE a goal executes is a real fault (e.g. the
        subgoal stream stalled) — that still warns."""
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            node._cmd_vel_pub = MagicMock()
            logger = MagicMock()
            node.get_logger = lambda: logger  # type: ignore
            node._active_goal_count = 1  # a mission is executing

            node._on_tick()  # streams stale -> mid-mission fault

            logger.warning.assert_called()
            node._cmd_vel_pub.publish.assert_called_once()
        finally:
            node.destroy_node()


# =============================================================================
# Reset triggers (Recurrent contract point 4)
# =============================================================================


class TestResetTriggers(unittest.TestCase):

    def _make_node_with_policy(self, **param_overrides) -> tuple[InferenceNode, _FakeRecurrentPolicy]:
        node = InferenceNode(
            parameter_overrides=_make_overrides(model_path="", **param_overrides)
        )
        fake = _FakeRecurrentPolicy()
        node._policy = fake
        return node, fake

    def test_handle_accepted_takes_ownership_and_executes(self) -> None:
        node, _fake = self._make_node_with_policy()
        try:
            goal_handle = MagicMock()
            node._handle_accepted(goal_handle)
            self.assertIs(node._current_goal_handle, goal_handle)
            goal_handle.execute.assert_called_once()
        finally:
            node.destroy_node()

    def test_superseded_at_entry_aborts_without_stomping(self) -> None:
        """An execute task that starts after a newer goal was accepted
        must abort before touching the live mission's goal pose or
        hidden state."""
        node, fake = self._make_node_with_policy()
        try:
            goal_handle = MagicMock()
            goal_handle.request.pose = _make_pose(2.0, 0.0)
            goal_handle.is_cancel_requested = False
            node._current_goal_handle = MagicMock()  # a newer goal owns

            node._execute_callback(goal_handle)

            goal_handle.abort.assert_called_once()
            goal_handle.succeed.assert_not_called()
            self.assertEqual(fake.reset_calls, 0)
            self.assertIsNone(node._last_goal_map)
        finally:
            node.destroy_node()

    def test_execute_aborts_when_preempted_mid_loop(self) -> None:
        """Ownership moving to a newer goal mid-mission makes the running
        loop abort. The goal's own mission start already reset hidden
        state — a preempting goal update IS a new mission boundary
        (this replaces the retired /strafer/goal topic path).
        """
        node, fake = self._make_node_with_policy()
        try:
            goal_handle = MagicMock()
            goal_handle.request.pose = _make_pose(2.0, 0.0)
            goal_handle.is_cancel_requested = False
            node._current_goal_handle = goal_handle  # self-owned at entry

            def _distance_then_preempt(goal_pose):
                node._current_goal_handle = MagicMock()  # newer goal lands
                return 1.0

            node._current_goal_distance = _distance_then_preempt  # type: ignore

            node._execute_callback(goal_handle)

            goal_handle.abort.assert_called_once()
            goal_handle.succeed.assert_not_called()
            self.assertEqual(fake.reset_calls, 1)
        finally:
            node.destroy_node()

    def test_finished_successor_still_supersedes(self) -> None:
        """Ownership is never cleared, only replaced: a preempting goal
        that already finished must still kill its predecessor — no
        zombie mission resumes when the successor exits within the old
        loop's sleep window."""
        node, _fake = self._make_node_with_policy()
        try:
            node._current_goal_distance = lambda goal_pose: None  # type: ignore
            handle_a = MagicMock()
            handle_a.request.pose = _make_pose(2.0, 0.0)
            handle_a.is_cancel_requested = False
            handle_b = MagicMock()
            handle_b.request.pose = _make_pose(-3.0, 0.0)
            handle_b.is_cancel_requested = True  # finishes on first tick

            node._handle_accepted(handle_a)
            thread_a = threading.Thread(
                target=node._execute_callback, args=(handle_a,))
            thread_a.start()
            _wait_until(lambda: node._active_goal_count == 1)

            node._handle_accepted(handle_b)
            node._execute_callback(handle_b)  # runs and exits immediately
            handle_b.canceled.assert_called_once()

            thread_a.join(timeout=2.0)  # A must still die
            self.assertFalse(thread_a.is_alive())
            handle_a.abort.assert_called_once()
            self.assertIs(node._current_goal_handle, handle_b)
            self.assertFalse(node._goal_active)
        finally:
            node.destroy_node()

    def test_distance_uses_passed_goal_not_shared_map(self) -> None:
        """The succeed check evaluates the caller's own captured goal, not
        the shared _last_goal_map a successor may have overwritten during
        the <=50 ms drain window."""
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            node._last_goal_map = _make_pose(100.0, 100.0)  # a successor's pose
            tf = MagicMock()
            tf.transform.translation.x = 0.0
            tf.transform.translation.y = 0.0
            node._tf_buffer.lookup_transform = MagicMock(  # type: ignore
                return_value=tf
            )

            distance = node._current_goal_distance(_make_pose(3.0, 4.0))
            self.assertAlmostEqual(distance, 5.0)
        finally:
            node.destroy_node()

    def test_action_server_execute_resets_then_polls_to_succeed(self) -> None:
        """The execute_callback must reset hidden state on goal accept
        (contract trigger 4.1) and then poll the goal-distance until
        the goal is reached."""
        node, fake = self._make_node_with_policy(
            goal_reached_distance_m=0.25,
            mission_timeout_s=5.0,
        )
        try:
            goal_handle = MagicMock()
            goal_handle.request.pose = _make_pose(2.0, 0.0)
            goal_handle.is_cancel_requested = False

            distances = iter([1.0, 0.5, 0.2])  # third tick crosses threshold

            def _fake_distance(goal_pose):
                try:
                    return next(distances)
                except StopIteration:
                    return 0.0

            node._current_goal_distance = _fake_distance  # type: ignore

            node._execute_callback(goal_handle)

            self.assertEqual(fake.reset_calls, 1)
            goal_handle.succeed.assert_called_once()
            goal_handle.abort.assert_not_called()
        finally:
            node.destroy_node()

    def test_action_server_aborts_with_no_policy(self) -> None:
        """If somehow execute_callback is reached without a loaded
        policy, abort fast rather than NPE."""
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            node._policy = None
            goal_handle = MagicMock()
            goal_handle.request.pose = _make_pose(1.0, 0.0)
            node._execute_callback(goal_handle)
            goal_handle.abort.assert_called_once()
        finally:
            node.destroy_node()

    def test_action_server_honors_cancel(self) -> None:
        node, _fake = self._make_node_with_policy()
        try:
            goal_handle = MagicMock()
            goal_handle.request.pose = _make_pose(2.0, 0.0)
            goal_handle.is_cancel_requested = True
            node._current_goal_distance = lambda goal_pose: 1.0  # type: ignore

            node._execute_callback(goal_handle)
            goal_handle.canceled.assert_called_once()
            goal_handle.succeed.assert_not_called()
        finally:
            node.destroy_node()


# =============================================================================
# Goal-active flag — watchdog freshness for the latched action goal
# =============================================================================


def _wait_until(cond, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


class TestGoalActiveFlag(unittest.TestCase):
    """``_goal_active`` models active-goal presence for the watchdog:
    ``True`` for the whole blocking mission loop, ``False`` on every exit
    path (succeed / cancel / timeout-abort / exception), and robust to
    briefly overlapping goals via the underlying counter.
    """

    def _make_node_with_policy(self, **overrides) -> InferenceNode:
        params = dict(
            model_path="",
            goal_reached_distance_m=0.25,
            mission_timeout_s=5.0,
        )
        params.update(overrides)
        node = InferenceNode(parameter_overrides=_make_overrides(**params))
        node._policy = _FakeRecurrentPolicy()
        return node

    def _make_goal_handle(self) -> MagicMock:
        goal_handle = MagicMock()
        goal_handle.request.pose = _make_pose(2.0, 0.0)
        goal_handle.is_cancel_requested = False
        return goal_handle

    def test_flag_true_across_mission_then_cleared_on_success(self) -> None:
        node = self._make_node_with_policy()
        try:
            self.assertFalse(node._goal_active)
            goal_handle = self._make_goal_handle()

            distances = iter([1.0, 0.5, 0.2])
            observed: list[bool] = []

            def _fake_distance(goal_pose):
                # Runs inside the mission loop: the flag must already be up.
                observed.append(node._goal_active)
                try:
                    return next(distances)
                except StopIteration:
                    return 0.0

            node._current_goal_distance = _fake_distance  # type: ignore

            node._execute_callback(goal_handle)

            goal_handle.succeed.assert_called_once()
            self.assertEqual(len(observed), 3)
            self.assertTrue(all(observed))
            self.assertFalse(node._goal_active)
        finally:
            node.destroy_node()

    def test_flag_cleared_on_cancel(self) -> None:
        node = self._make_node_with_policy()
        try:
            goal_handle = self._make_goal_handle()
            goal_handle.is_cancel_requested = True
            node._current_goal_distance = lambda goal_pose: 1.0  # type: ignore

            node._execute_callback(goal_handle)

            goal_handle.canceled.assert_called_once()
            self.assertFalse(node._goal_active)
        finally:
            node.destroy_node()

    def test_flag_cleared_when_mission_loop_raises(self) -> None:
        node = self._make_node_with_policy()
        try:
            goal_handle = self._make_goal_handle()
            node._current_goal_distance = MagicMock(  # type: ignore
                side_effect=RuntimeError("tf buffer died")
            )

            with self.assertRaises(RuntimeError):
                node._execute_callback(goal_handle)

            self.assertFalse(node._goal_active)
        finally:
            node.destroy_node()

    def test_no_policy_abort_leaves_flag_false(self) -> None:
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            node._policy = None
            goal_handle = self._make_goal_handle()

            node._execute_callback(goal_handle)

            goal_handle.abort.assert_called_once()
            self.assertFalse(node._goal_active)
        finally:
            node.destroy_node()

    def test_flag_cleared_on_mission_timeout_abort(self) -> None:
        node = self._make_node_with_policy(mission_timeout_s=0.01)
        try:
            goal_handle = self._make_goal_handle()
            node._current_goal_distance = lambda goal_pose: None  # type: ignore

            node._execute_callback(goal_handle)

            goal_handle.abort.assert_called_once()
            self.assertFalse(node._goal_active)
        finally:
            node.destroy_node()

    def test_overlapping_goal_exit_does_not_clear_survivor(self) -> None:
        """A goal exiting while another still executes (a client cancel
        is fire-and-forget, so the next goal's accept can overlap the
        old loop's drain) must not clear goal freshness under the
        survivor.
        """
        node = self._make_node_with_policy()
        try:
            node._current_goal_distance = lambda goal_pose: None  # type: ignore
            handle_a = self._make_goal_handle()
            handle_b = self._make_goal_handle()

            thread_a = threading.Thread(
                target=node._execute_callback, args=(handle_a,))
            thread_b = threading.Thread(
                target=node._execute_callback, args=(handle_b,))
            thread_a.start()
            thread_b.start()
            _wait_until(lambda: node._active_goal_count == 2)

            handle_b.is_cancel_requested = True
            thread_b.join(timeout=2.0)
            self.assertFalse(thread_b.is_alive())
            handle_b.canceled.assert_called_once()
            self.assertTrue(node._goal_active)

            handle_a.is_cancel_requested = True
            thread_a.join(timeout=2.0)
            self.assertFalse(thread_a.is_alive())
            self.assertFalse(node._goal_active)
        finally:
            node.destroy_node()

    def test_preemption_keeps_goal_fresh_across_boundary(self) -> None:
        """Newest-goal-wins: while goal B replaces goal A, the watchdog
        goal source never goes absent — the preempted goal's exit must
        not clear freshness under its successor.
        """
        node = self._make_node_with_policy()
        try:
            node._current_goal_distance = lambda goal_pose: None  # type: ignore
            handle_a = self._make_goal_handle()
            handle_b = self._make_goal_handle()

            node._handle_accepted(handle_a)
            thread_a = threading.Thread(
                target=node._execute_callback, args=(handle_a,))
            thread_a.start()
            _wait_until(lambda: node._active_goal_count == 1)

            node._handle_accepted(handle_b)
            thread_b = threading.Thread(
                target=node._execute_callback, args=(handle_b,))
            thread_b.start()

            thread_a.join(timeout=2.0)  # A exits via the preempt check
            self.assertFalse(thread_a.is_alive())
            handle_a.abort.assert_called_once()

            _wait_until(lambda: node._active_goal_count == 1)  # B is up
            self.assertTrue(node._goal_active)

            handle_b.is_cancel_requested = True
            thread_b.join(timeout=2.0)
            self.assertFalse(thread_b.is_alive())
            handle_b.canceled.assert_called_once()
            self.assertFalse(node._goal_active)
            self.assertIs(node._current_goal_handle, handle_b)
        finally:
            node.destroy_node()

    def test_tick_treats_active_goal_as_fresh(self) -> None:
        """Pins the _on_tick → stale_sources(goal_active=...) wiring:
        with every streamed source fresh and no topic goal ever
        received, the tick zero-twists when idle but publishes nothing
        while a goal is active (clean watchdog + no policy holds the
        channel).
        """
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            now = time.monotonic()
            node._last_imu_rx_t = now
            node._last_joint_states_rx_t = now
            node._last_odom_rx_t = now
            node._last_depth_rx_t = now
            node._tf_age_s = lambda: 0.0  # type: ignore
            node._cmd_vel_pub = MagicMock()

            node._active_goal_count = 1
            node._on_tick()
            node._cmd_vel_pub.publish.assert_not_called()

            node._active_goal_count = 0
            node._on_tick()
            node._cmd_vel_pub.publish.assert_called_once()
        finally:
            node.destroy_node()


# =============================================================================
# Active-goal telemetry — status feed for the subgoal generator
# =============================================================================


class TestActiveGoalTelemetry(unittest.TestCase):
    """The mission loop publishes its accepted goal at accept and as a
    keep-alive while executing — status telemetry for the subgoal
    generator's replan ownership, never a command channel.
    """

    def _make_node(self, **overrides) -> InferenceNode:
        params = dict(
            model_path="",
            mission_timeout_s=5.0,
            active_goal_keepalive_period_s=0.05,
        )
        params.update(overrides)
        node = InferenceNode(parameter_overrides=_make_overrides(**params))
        node._policy = _FakeRecurrentPolicy()
        node._active_goal_pub = MagicMock()
        return node

    def test_publishes_at_accept_and_keeps_alive(self) -> None:
        node = self._make_node()
        try:
            goal_handle = MagicMock()
            goal_handle.request.pose = _make_pose(2.0, 0.0)
            goal_handle.is_cancel_requested = False
            node._current_goal_distance = lambda goal_pose: None  # type: ignore

            thread = threading.Thread(
                target=node._execute_callback, args=(goal_handle,))
            thread.start()
            # 1 at accept + keep-alives every 0.05 s.
            _wait_until(
                lambda: node._active_goal_pub.publish.call_count >= 3
            )
            goal_handle.is_cancel_requested = True
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())

            for call in node._active_goal_pub.publish.call_args_list:
                self.assertIs(call[0][0], goal_handle.request.pose)
        finally:
            node.destroy_node()

    def test_no_telemetry_when_superseded_at_entry(self) -> None:
        node = self._make_node()
        try:
            goal_handle = MagicMock()
            goal_handle.request.pose = _make_pose(2.0, 0.0)
            goal_handle.is_cancel_requested = False
            node._current_goal_handle = MagicMock()  # a newer goal owns

            node._execute_callback(goal_handle)

            node._active_goal_pub.publish.assert_not_called()
        finally:
            node.destroy_node()

    def test_no_telemetry_without_policy(self) -> None:
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            node._policy = None
            node._active_goal_pub = MagicMock()
            goal_handle = MagicMock()
            goal_handle.request.pose = _make_pose(2.0, 0.0)

            node._execute_callback(goal_handle)

            node._active_goal_pub.publish.assert_not_called()
        finally:
            node.destroy_node()


# =============================================================================
# Determinism contract — the two-pronged recurrent assertion
# =============================================================================


class TestDeterminismTwoProng(unittest.TestCase):
    """Recurrent-contract point 5: two same-obs calls produce byte-
    identical actions IFF reset() is called between them. Without an
    intervening reset, the actions differ by construction. Asserted
    here at the policy stub the inference node holds — the cross-
    format version is in test_recurrent_contract_e2e.py.
    """

    def test_reset_between_yields_byte_identical(self) -> None:
        policy = _FakeRecurrentPolicy()
        obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        policy.reset()
        a = policy(obs)
        policy.reset()
        b = policy(obs)
        np.testing.assert_array_equal(a, b)

    def test_no_reset_between_evolves_action(self) -> None:
        policy = _FakeRecurrentPolicy()
        obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        policy.reset()
        a = policy(obs)
        b = policy(obs)
        self.assertFalse(
            np.array_equal(a, b),
            "recurrent policy stub must thread hidden state across "
            "calls; got byte-identical actions, suggesting the stub "
            "is broken and the inference-node tests built on it are vacuous",
        )

    def test_initial_state_is_zero_on_construction(self) -> None:
        """Contract point 2: hidden state is zero on construction.
        Verified by checking the first call after construction matches
        the first call after explicit reset."""
        obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        a_fresh = _FakeRecurrentPolicy()(obs)
        p = _FakeRecurrentPolicy()
        p.reset()
        a_reset = p(obs)
        np.testing.assert_array_equal(a_fresh, a_reset)


# =============================================================================
# Subscriber rx-time bookkeeping (watchdog input)
# =============================================================================


class TestRxTimeBookkeeping(unittest.TestCase):
    """The watchdog reads these monotonic rx-time fields; verify each
    subscriber callback populates the right one.
    """

    def test_imu_callback_records_rx_time(self) -> None:
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            self.assertIsNone(node._last_imu_rx_t)
            from sensor_msgs.msg import Imu
            t0 = time.monotonic()
            node._on_imu(Imu())
            self.assertIsNotNone(node._last_imu_rx_t)
            self.assertGreaterEqual(node._last_imu_rx_t, t0)
        finally:
            node.destroy_node()

    def test_odom_callback_records_rx_time(self) -> None:
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            from nav_msgs.msg import Odometry
            node._on_odom(Odometry())
            self.assertIsNotNone(node._last_odom_rx_t)
        finally:
            node.destroy_node()

    def test_joint_states_callback_records_rx_time(self) -> None:
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            from sensor_msgs.msg import JointState
            node._on_joint_states(JointState())
            self.assertIsNotNone(node._last_joint_states_rx_t)
        finally:
            node.destroy_node()


# =============================================================================
# Variant-aware wiring (Option A: depth + subgoal subscribers, watchdog, obs
# are gated on what the loaded variant's fields contain)
# =============================================================================


class TestVariantAwareWiring(unittest.TestCase):

    def _node(self, variant: str) -> InferenceNode:
        return InferenceNode(
            parameter_overrides=_make_overrides(
                model_path="", policy_variant=variant
            )
        )

    def test_depth_variant_subscribes_depth_not_subgoal(self) -> None:
        node = self._node("DEPTH")
        try:
            self.assertTrue(node._has_depth)
            self.assertFalse(node._uses_subgoal)
            self.assertIsNotNone(node._depth_sub)
            self.assertIsNone(node._subgoal_sub)
        finally:
            node.destroy_node()

    def test_nocam_subgoal_subscribes_subgoal_not_depth(self) -> None:
        node = self._node("NOCAM_SUBGOAL")
        try:
            self.assertFalse(node._has_depth)
            self.assertTrue(node._uses_subgoal)
            self.assertIsNone(node._depth_sub)
            self.assertIsNotNone(node._subgoal_sub)
        finally:
            node.destroy_node()

    def test_nocam_subscribes_neither_depth_nor_subgoal(self) -> None:
        """Regression anchor: the no-camera, goal-referent variant wires
        neither the depth nor the subgoal subscriber.
        """
        node = self._node("NOCAM")
        try:
            self.assertFalse(node._has_depth)
            self.assertFalse(node._uses_subgoal)
            self.assertIsNone(node._depth_sub)
            self.assertIsNone(node._subgoal_sub)
        finally:
            node.destroy_node()

    def test_subgoal_callback_caches_without_resetting(self) -> None:
        """The rolling subgoal advances every tick; caching it must record
        the rx time but must NOT fire the mid-mission hidden-state reset.
        """
        node = self._node("NOCAM_SUBGOAL")
        fake = _FakeRecurrentPolicy()
        node._policy = fake
        try:
            node._on_subgoal(_make_pose(1.0, 0.0))
            self.assertIsNotNone(node._last_subgoal_rx_t)
            self.assertEqual(node._last_subgoal_map.pose.position.x, 1.0)
            node._on_subgoal(_make_pose(2.0, 0.0))  # subgoal advances
            self.assertEqual(fake.reset_calls, 0)
        finally:
            node.destroy_node()

    def test_nocam_subgoal_tick_zero_twists_when_watchdog_stale(self) -> None:
        """A NOCAM_SUBGOAL node never receives depth; with no sensors fed the
        watchdog is stale and the tick must publish a single zero Twist (the
        safety stop) -- not merely avoid crashing on the disabled depth source.
        """
        from geometry_msgs.msg import Twist

        node = self._node("NOCAM_SUBGOAL")
        node._cmd_vel_pub = MagicMock()
        try:
            node._on_tick()
            node._cmd_vel_pub.publish.assert_called_once()
            published = node._cmd_vel_pub.publish.call_args[0][0]
            self.assertIsInstance(published, Twist)
            self.assertEqual(published.linear.x, 0.0)
            self.assertEqual(published.linear.y, 0.0)
            self.assertEqual(published.angular.z, 0.0)
        finally:
            node.destroy_node()

    def test_nocam_subgoal_watchdog_swaps_depth_for_subgoal(self) -> None:
        """The hybrid freshness guard: a NOCAM_SUBGOAL node drops the depth
        source and adds a subgoal source (path_timeout_s budget), wired from
        the node's own gating flags and timeouts.
        """
        from strafer_inference.watchdog import stale_sources

        node = self._node("NOCAM_SUBGOAL")
        try:
            self.assertEqual(node._timeouts.path, 1.0)
            now = 1000.0
            stale = stale_sources(
                now_monotonic_s=now,
                last_imu_rx_t=now,
                last_joint_states_rx_t=now, last_odom_rx_t=now,
                last_depth_rx_t=None, tf_age_s=0.0,
                timeouts=node._timeouts,
                depth_enabled=node._has_depth,
                last_subgoal_rx_t=None,
                subgoal_enabled=node._uses_subgoal,
            )
            self.assertNotIn("depth", stale)   # no camera on this variant
            self.assertIn("subgoal", stale)    # but the rolling subgoal is gated
        finally:
            node.destroy_node()

    def test_depth_direct_watchdog_has_no_subgoal_source(self) -> None:
        """Regression: the DEPTH strafer_direct node keeps the depth source
        and adds no subgoal source.
        """
        from strafer_inference.watchdog import stale_sources

        node = self._node("DEPTH")
        try:
            now = 1000.0
            stale = stale_sources(
                now_monotonic_s=now,
                last_imu_rx_t=now,
                last_joint_states_rx_t=now, last_odom_rx_t=now,
                last_depth_rx_t=None, tf_age_s=0.0,
                timeouts=node._timeouts,
                depth_enabled=node._has_depth,
                last_subgoal_rx_t=None,
                subgoal_enabled=node._uses_subgoal,
            )
            self.assertIn("depth", stale)        # camera variant
            self.assertNotIn("subgoal", stale)   # not a subgoal variant
        finally:
            node.destroy_node()
