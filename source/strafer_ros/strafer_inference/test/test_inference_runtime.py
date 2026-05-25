"""Phase 3 node-integration tests for the runtime: model-load failure
modes, reset-trigger wiring, and the recurrent two-pronged determinism
assertion at the policy-stub seam.

These tests spin up rclpy long enough to construct the node; they do
NOT subscribe / publish across the wire. The action server's blocking
``execute_callback`` is driven through a mocked goal handle.
"""

from __future__ import annotations

import math
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

    def test_subscriber_goal_update_triggers_reset_when_flag_on(self) -> None:
        """is_mid_mission_reset=True + goal pose moves above the
        threshold → reset() fires once per detected jump."""
        node, fake = self._make_node_with_policy(
            is_mid_mission_reset=True,
            mid_mission_reset_distance_m=0.05,
        )
        try:
            node._on_goal(_make_pose(1.0, 0.0))
            # First message: no previous → no reset.
            self.assertEqual(fake.reset_calls, 0)

            node._on_goal(_make_pose(1.5, 0.0))  # jumped 0.5 m
            self.assertEqual(fake.reset_calls, 1)

            node._on_goal(_make_pose(1.51, 0.0))  # jumped 0.01 m (below threshold)
            self.assertEqual(fake.reset_calls, 1)
        finally:
            node.destroy_node()

    def test_subscriber_goal_update_no_reset_when_flag_off(self) -> None:
        """is_mid_mission_reset=False disables the mid-mission trigger
        — the policy keeps its hidden state across a goal change.
        Operator opt-out for envs where the regrounded goal is
        continuous with the previous (e.g. tracking a moving target)."""
        node, fake = self._make_node_with_policy(
            is_mid_mission_reset=False,
            mid_mission_reset_distance_m=0.05,
        )
        try:
            node._on_goal(_make_pose(1.0, 0.0))
            node._on_goal(_make_pose(5.0, 5.0))
            self.assertEqual(fake.reset_calls, 0)
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

            def _fake_distance():
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
            node._current_goal_distance = lambda: 1.0  # type: ignore

            node._execute_callback(goal_handle)
            goal_handle.canceled.assert_called_once()
            goal_handle.succeed.assert_not_called()
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

    def test_goal_callback_records_rx_time(self) -> None:
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            node._on_goal(_make_pose(1.0, 0.0))
            self.assertIsNotNone(node._last_goal_rx_t)
        finally:
            node.destroy_node()
