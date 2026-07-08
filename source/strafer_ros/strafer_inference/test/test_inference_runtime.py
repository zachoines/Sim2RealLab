"""Phase 3 node-integration tests for the runtime: model-load failure
modes, reset-trigger wiring, and the recurrent two-pronged determinism
assertion at the policy-stub seam.

These tests spin up rclpy long enough to construct the node; they do
NOT subscribe / publish across the wire. The action server's blocking
``execute_callback`` is driven through a mocked goal handle.
"""

from __future__ import annotations

import math
import os
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
from sensor_msgs.msg import Image

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


class _WrongShapePolicy:
    """Recurrent stub whose output is NOT (3,) — drives the node's action-
    shape error branch. Tracks call_count to pin frame-consume ordering."""

    is_recurrent = True

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        pass


def _make_pose(x: float, y: float) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = "map"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
    return msg


def _depth_msg(h: int = 4, w: int = 4) -> Image:
    """A minimal valid 32FC1 depth frame (content irrelevant to the gate)."""
    msg = Image()
    msg.encoding = "32FC1"
    msg.height = h
    msg.width = w
    msg.data = np.zeros((h, w), dtype=np.float32).tobytes()
    return msg


# =============================================================================
# Sim-only obs-timeout env override
# =============================================================================


class TestObsTimeoutEnvOverride(unittest.TestCase):
    """STRAFER_OBS_TIMEOUT_S loosens the imu/joint_states/odom freshness
    budget for the render-bound sim bridge; unset keeps the real-robot
    yaml default. Same pattern as the executor's OBSERVATION_MAX_AGE_S."""

    def test_env_overrides_obs_sources_only(self) -> None:
        os.environ["STRAFER_OBS_TIMEOUT_S"] = "1.5"
        try:
            node = InferenceNode(
                parameter_overrides=_make_overrides(model_path="")
            )
        finally:
            del os.environ["STRAFER_OBS_TIMEOUT_S"]
        try:
            self.assertEqual(node._timeouts.imu, 1.5)
            self.assertEqual(node._timeouts.joint_states, 1.5)
            self.assertEqual(node._timeouts.odom, 1.5)
            self.assertEqual(node._timeouts.depth, 0.5)  # untouched
            self.assertEqual(node._timeouts.tf, 0.5)     # untouched
        finally:
            node.destroy_node()

    def test_unset_keeps_param_default(self) -> None:
        os.environ.pop("STRAFER_OBS_TIMEOUT_S", None)
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            self.assertEqual(node._timeouts.imu, 0.2)
        finally:
            node.destroy_node()


class TestDepthTimeoutEnvOverride(unittest.TestCase):
    """STRAFER_DEPTH_TIMEOUT_S loosens ONLY the depth freshness budget for the
    sim bridge's slow (~3 Hz) depth feed; unset keeps the real-robot yaml
    default. Independent of STRAFER_OBS_TIMEOUT_S (imu/joint_states/odom)."""

    def test_env_overrides_depth_source_only(self) -> None:
        os.environ["STRAFER_DEPTH_TIMEOUT_S"] = "2.0"
        try:
            node = InferenceNode(
                parameter_overrides=_make_overrides(model_path="")
            )
        finally:
            del os.environ["STRAFER_DEPTH_TIMEOUT_S"]
        try:
            self.assertEqual(node._timeouts.depth, 2.0)
            self.assertEqual(node._timeouts.imu, 0.2)          # untouched
            self.assertEqual(node._timeouts.joint_states, 0.2)  # untouched
            self.assertEqual(node._timeouts.odom, 0.2)          # untouched
            self.assertEqual(node._timeouts.tf, 0.5)            # untouched
        finally:
            node.destroy_node()

    def test_unset_keeps_param_default(self) -> None:
        os.environ.pop("STRAFER_DEPTH_TIMEOUT_S", None)
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            self.assertEqual(node._timeouts.depth, 0.5)
        finally:
            node.destroy_node()

    def test_obs_and_depth_overrides_are_independent(self) -> None:
        os.environ["STRAFER_OBS_TIMEOUT_S"] = "1.0"
        os.environ["STRAFER_DEPTH_TIMEOUT_S"] = "2.0"
        try:
            node = InferenceNode(
                parameter_overrides=_make_overrides(model_path="")
            )
        finally:
            del os.environ["STRAFER_OBS_TIMEOUT_S"]
            del os.environ["STRAFER_DEPTH_TIMEOUT_S"]
        try:
            self.assertEqual(node._timeouts.imu, 1.0)
            self.assertEqual(node._timeouts.depth, 2.0)
        finally:
            node.destroy_node()


# =============================================================================
# Provider resolution — TRT engine-cache options wired onto the TRT entry
# =============================================================================


class TestResolveOnnxProviders(unittest.TestCase):
    """`_resolve_onnx_providers` keeps the plain string list working and, when
    the TRT engine cache is configured, upgrades only the
    TensorrtExecutionProvider entry to a (name, options) tuple. load_policy
    forwards the mixed list to ORT verbatim."""

    _PREF = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    def _node(self, **overrides):
        node = InferenceNode(
            parameter_overrides=_make_overrides(model_path="", **overrides)
        )
        self.addCleanup(node.destroy_node)
        return node

    def test_cache_disabled_returns_plain_string_list(self) -> None:
        node = self._node(onnx_providers=self._PREF, trt_engine_cache_enable=False)
        self.assertEqual(node._resolve_onnx_providers(), self._PREF)

    def test_cache_enabled_no_path_returns_plain_list(self) -> None:
        node = self._node(
            onnx_providers=self._PREF,
            trt_engine_cache_enable=True,
            trt_engine_cache_path="",
        )
        self.assertEqual(node._resolve_onnx_providers(), self._PREF)

    def test_cache_enabled_upgrades_only_trt_entry(self) -> None:
        import shutil
        import tempfile

        cache = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, cache, ignore_errors=True)
        node = self._node(
            onnx_providers=self._PREF,
            trt_engine_cache_enable=True,
            trt_engine_cache_path=cache,
        )
        resolved = node._resolve_onnx_providers()
        trt, cuda, cpu = resolved
        self.assertEqual(cuda, "CUDAExecutionProvider")
        self.assertEqual(cpu, "CPUExecutionProvider")
        self.assertEqual(trt[0], "TensorrtExecutionProvider")
        self.assertTrue(trt[1]["trt_engine_cache_enable"])
        self.assertEqual(trt[1]["trt_engine_cache_path"], cache)
        self.assertTrue(os.path.isdir(cache))

    def test_cpu_only_list_with_cache_enabled_stays_plain(self) -> None:
        # No TRT entry to augment -> plain list even with the cache configured.
        node = self._node(
            onnx_providers=["CPUExecutionProvider"],
            trt_engine_cache_enable=True,
            trt_engine_cache_path="/tmp/does-not-matter",
        )
        self.assertEqual(
            node._resolve_onnx_providers(), ["CPUExecutionProvider"]
        )


# =============================================================================
# ONNX thread pinning — keep a tiny MLP from spinning every core
# =============================================================================


class TestOnnxThreadPinning(unittest.TestCase):
    """load_policy must pin the ONNX intra-op thread count (default 1) so
    a ~50 us MLP does not spin all 6 Jetson cores under the ORT default
    (0 = one spin-waiting thread per core), starving RTAB-Map / Nav2.
    Exercises the real deployed model; skips where it is not present.
    """

    def _model_path(self) -> str:
        repo = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        return os.path.join(repo, "models", "strafer_nocam_subgoal_v0.onnx")

    def test_default_pins_to_one_thread(self) -> None:
        from strafer_shared.policy_interface import load_policy, PolicyVariant

        model = self._model_path()
        if not os.path.isfile(model):
            self.skipTest("deployed NOCAM_SUBGOAL model not present")
        policy = load_policy(
            model, PolicyVariant.NOCAM_SUBGOAL,
            onnx_providers=["CPUExecutionProvider"],
        )
        self.assertEqual(
            policy._sess.get_session_options().intra_op_num_threads, 1
        )

    def test_explicit_thread_count_is_applied(self) -> None:
        from strafer_shared.policy_interface import load_policy, PolicyVariant

        model = self._model_path()
        if not os.path.isfile(model):
            self.skipTest("deployed NOCAM_SUBGOAL model not present")
        policy = load_policy(
            model, PolicyVariant.NOCAM_SUBGOAL,
            onnx_providers=["CPUExecutionProvider"], onnx_intra_op_threads=2,
        )
        self.assertEqual(
            policy._sess.get_session_options().intra_op_num_threads, 2
        )


class TestDepthSubgoalArtifactLoads(unittest.TestCase):
    """The deployed DEPTH_SUBGOAL artifact loads as a recurrent policy and
    runs a 4819-dim observation to a 3-vector action. Exercises the real
    artifact; skips where it is not present.
    """

    def _model_path(self) -> str:
        repo = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        return os.path.join(repo, "models", "strafer_depth_subgoal_v0.onnx")

    def test_loads_recurrent_and_infers(self) -> None:
        from strafer_shared.policy_interface import load_policy, PolicyVariant

        model = self._model_path()
        if not os.path.isfile(model):
            self.skipTest("deployed DEPTH_SUBGOAL model not present")
        policy = load_policy(
            model, PolicyVariant.DEPTH_SUBGOAL,
            onnx_providers=["CPUExecutionProvider"],
        )
        self.assertTrue(policy.is_recurrent)

        obs = np.zeros(PolicyVariant.DEPTH_SUBGOAL.obs_dim, dtype=np.float32)
        action = np.asarray(policy(obs)).reshape(-1)
        self.assertEqual(action.shape, (3,))
        # reset() zeros hidden state, so two same-obs calls each preceded by a
        # reset match: the recurrence contract the node relies on at goal accept.
        probe = np.ones_like(obs)
        policy.reset()
        first = np.asarray(policy(probe)).reshape(-1)
        policy.reset()
        again = np.asarray(policy(probe)).reshape(-1)
        np.testing.assert_allclose(first, again, atol=1e-6)


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

    def test_idle_watchdog_publishes_nothing_and_stays_quiet(self) -> None:
        """Idle (no active goal): the watchdog trips on goal/tf/subgoal
        every tick, but that is the resting state between missions. The
        node must publish NOTHING — cmd_vel lands on the shared /cmd_vel
        (launch remap), owned by Nav2 / rotate / teleop between missions
        — and stay quiet (no WARN spam at the tick rate)."""
        node = InferenceNode(parameter_overrides=_make_overrides(model_path=""))
        try:
            node._cmd_vel_pub = MagicMock()
            logger = MagicMock()
            node.get_logger = lambda: logger  # type: ignore

            node._on_tick()  # no active goal

            logger.warning.assert_not_called()
            node._cmd_vel_pub.publish.assert_not_called()
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

    def test_mission_end_publishes_final_stop(self) -> None:
        """On mission end (succeed / cancel / timeout) the loop publishes a
        final zero Twist — consumers hold the last /cmd_vel, and the sim
        bridge has no stop-on-silence watchdog, so without this the robot
        coasts at the last policy velocity past the goal."""
        node = self._make_node_with_policy(mission_timeout_s=5.0)
        try:
            node._cmd_vel_pub = MagicMock()
            goal_handle = self._make_goal_handle()
            node._current_goal_distance = lambda goal_pose: 0.1  # type: ignore

            node._execute_callback(goal_handle)  # succeeds immediately

            goal_handle.succeed.assert_called_once()
            last = node._cmd_vel_pub.publish.call_args[0][0]
            self.assertEqual((last.linear.x, last.linear.y, last.angular.z),
                             (0.0, 0.0, 0.0))
        finally:
            node.destroy_node()

    def test_preempt_exit_does_not_stop_the_successor(self) -> None:
        """A preempted goal must NOT publish a stop in its finally — the
        successor owns /cmd_vel and a zero here would fight its commands."""
        node = self._make_node_with_policy()
        try:
            node._cmd_vel_pub = MagicMock()
            goal_handle = self._make_goal_handle()
            node._current_goal_handle = goal_handle

            def _preempt():
                node._current_goal_handle = MagicMock()  # newer goal owns
                return None
            node._current_goal_distance = lambda goal_pose: _preempt()  # type: ignore

            node._execute_callback(goal_handle)  # exits via preempt

            goal_handle.abort.assert_called_once()
            node._cmd_vel_pub.publish.assert_not_called()
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
        with every streamed source fresh, an active goal gives a clean
        watchdog (no policy → holds the channel, no publish); dropping
        the goal_active kwarg would make the goal source stale and
        zero-twist mid-mission instead.
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
            node._on_tick()  # clean watchdog: holds (no zero-twist)
            node._cmd_vel_pub.publish.assert_not_called()

            node._last_imu_rx_t = now - 60.0  # mid-mission stale source
            node._on_tick()
            node._cmd_vel_pub.publish.assert_called_once()  # zero-twist
        finally:
            node.destroy_node()


# =============================================================================
# Mission deadline clock domain
# =============================================================================


class TestMissionDeadlineClockDomain(unittest.TestCase):
    """The mission deadline ticks on the NODE clock (sim time under
    use_sim_time, where mission progress also runs) — a wall deadline
    shrinks the budget by the sim RTF (60 s at RTF 0.1 is 6 sim s).
    Pinned with a frozen fake clock: wall time passes far beyond the
    budget with no timeout; advancing the fake clock times out."""

    def test_deadline_follows_node_clock_not_wall(self) -> None:
        from types import SimpleNamespace
        from rclpy.time import Time as RclpyTime

        node = InferenceNode(parameter_overrides=_make_overrides(
            model_path="", mission_timeout_s=0.05,
        ))
        node._policy = _FakeRecurrentPolicy()
        node._active_goal_pub = MagicMock()
        real_get_clock = node.get_clock
        try:
            fake = {"now": RclpyTime(seconds=100)}
            node.get_clock = (  # type: ignore
                lambda: SimpleNamespace(now=lambda: fake["now"])
            )
            goal_handle = MagicMock()
            goal_handle.request.pose = _make_pose(2.0, 0.0)
            goal_handle.is_cancel_requested = False
            node._current_goal_distance = lambda goal_pose: None  # type: ignore

            thread = threading.Thread(
                target=node._execute_callback, args=(goal_handle,))
            thread.start()
            time.sleep(0.4)  # wall time far past the 0.05 s budget
            self.assertTrue(thread.is_alive())  # frozen clock: no timeout
            goal_handle.abort.assert_not_called()

            fake["now"] = RclpyTime(seconds=100.2)  # past the budget
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            goal_handle.abort.assert_called_once()
        finally:
            node.get_clock = real_get_clock  # type: ignore
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
        """A NOCAM_SUBGOAL node never receives depth; with no sensors fed
        and a mission executing, the watchdog is stale and the tick must
        publish a single zero Twist (the safety stop) -- not merely avoid
        crashing on the disabled depth source. (Idle publishes nothing:
        the shared /cmd_vel belongs to Nav2/teleop between missions.)
        """
        from geometry_msgs.msg import Twist

        node = self._node("NOCAM_SUBGOAL")
        node._cmd_vel_pub = MagicMock()
        node._active_goal_count = 1  # mission executing
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

    # DEPTH_SUBGOAL is the only variant that enables both the depth and the
    # subgoal path; these pin that the field-driven wiring composes them
    # without a per-variant branch.

    def test_depth_subgoal_subscribes_both_depth_and_subgoal(self) -> None:
        node = self._node("DEPTH_SUBGOAL")
        try:
            self.assertTrue(node._has_depth)
            self.assertTrue(node._uses_subgoal)
            self.assertIsNotNone(node._depth_sub)
            self.assertIsNotNone(node._subgoal_sub)
        finally:
            node.destroy_node()

    def test_depth_subgoal_watchdog_has_both_depth_and_subgoal(self) -> None:
        """DEPTH_SUBGOAL keeps the depth source (camera restored vs
        NOCAM_SUBGOAL) AND adds the rolling-subgoal source: both trip
        independently, and both-fresh with an active goal is clean.
        """
        from strafer_inference.watchdog import stale_sources

        node = self._node("DEPTH_SUBGOAL")
        try:
            now = 1000.0

            def _stale(*, depth_rx, subgoal_rx):
                return stale_sources(
                    now_monotonic_s=now,
                    last_imu_rx_t=now,
                    last_joint_states_rx_t=now, last_odom_rx_t=now,
                    last_depth_rx_t=depth_rx, tf_age_s=0.0,
                    timeouts=node._timeouts,
                    depth_enabled=node._has_depth,
                    last_subgoal_rx_t=subgoal_rx,
                    subgoal_enabled=node._uses_subgoal,
                    goal_active=True,
                )

            # Both fresh + goal active -> clean.
            self.assertEqual(_stale(depth_rx=now, subgoal_rx=now), [])
            # Stale depth trips (subgoal not implicated).
            stale = _stale(depth_rx=None, subgoal_rx=now)
            self.assertIn("depth", stale)
            self.assertNotIn("subgoal", stale)
            # Stale subgoal trips (depth not implicated).
            stale = _stale(depth_rx=now, subgoal_rx=None)
            self.assertIn("subgoal", stale)
            self.assertNotIn("depth", stale)
        finally:
            node.destroy_node()


# =============================================================================
# Depth-freshness gate — one inference per fresh depth frame
# =============================================================================


class TestDepthFreshnessGate(unittest.TestCase):
    """A depth variant runs at most one inference per fresh depth frame:
    catch-up ticks under the slow sim depth feed must not replay a stale
    frame into the recurrent GRU. The gate sits after the watchdog (a
    stale-source zero-twist still preempts) and holds the channel on a
    skip (idle semantics — no publish). No-camera variants never gate.
    """

    def _depth_node(self, variant: str = "DEPTH") -> InferenceNode:
        node = InferenceNode(
            parameter_overrides=_make_overrides(
                model_path="", policy_variant=variant,
            )
        )
        node._policy = _FakeRecurrentPolicy()
        node._cmd_vel_pub = MagicMock()
        node._active_goal_count = 1  # mission executing: watchdog goal-fresh
        node._tf_age_s = lambda: 0.0  # type: ignore
        # Isolate the gate from obs assembly: a fixed non-None obs so every
        # ungated tick reaches (and counts) an inference.
        node._assemble_observation_or_none = (  # type: ignore
            lambda: np.zeros(4, dtype=np.float32)
        )
        return node

    def _stamp_streams_fresh(self, node: InferenceNode) -> None:
        now = time.monotonic()
        node._last_imu_rx_t = now
        node._last_joint_states_rx_t = now
        node._last_odom_rx_t = now
        node._last_subgoal_rx_t = now
        node._last_depth_rx_t = now  # within the depth timeout window

    def _tick(self, node: InferenceNode) -> None:
        # Re-stamp before each tick so a slow test host can't false-trip the
        # watchdog. The gate keys on the depth *frame counter*, not the rx
        # time, so re-stamping alone never advances a frame.
        self._stamp_streams_fresh(node)
        node._on_tick()

    def test_on_depth_increments_frame_counter(self) -> None:
        node = self._depth_node("DEPTH")
        try:
            self.assertEqual(node._depth_seq, 0)
            node._on_depth(_depth_msg())
            self.assertEqual(node._depth_seq, 1)
            node._on_depth(_depth_msg())
            self.assertEqual(node._depth_seq, 2)
        finally:
            node.destroy_node()

    def test_dropped_depth_frame_does_not_advance_counter(self) -> None:
        """A frame the callback rejects (wrong encoding) must not bump the
        counter — else the gate would treat a non-frame as fresh."""
        node = self._depth_node("DEPTH")
        try:
            bad = _depth_msg()
            bad.encoding = "mono8"
            node._on_depth(bad)
            self.assertEqual(node._depth_seq, 0)
        finally:
            node.destroy_node()

    def test_burst_one_frame_yields_one_inference(self) -> None:
        """Brief scenario: 3 catch-up ticks over 1 depth frame -> exactly 1
        inference/publish; the next frame -> the next inference."""
        node = self._depth_node("DEPTH")
        try:
            node._on_depth(_depth_msg())   # one fresh frame
            for _ in range(3):             # three catch-up ticks
                self._tick(node)
            self.assertEqual(node._policy.call_count, 1)
            self.assertEqual(node._cmd_vel_pub.publish.call_count, 1)

            node._on_depth(_depth_msg())   # next frame
            self._tick(node)
            self.assertEqual(node._policy.call_count, 2)
            self.assertEqual(node._cmd_vel_pub.publish.call_count, 2)
        finally:
            node.destroy_node()

    def test_gate_skip_holds_channel_without_publishing(self) -> None:
        """A skip is a hold, not a stop: no publish on a skipped tick (the
        downstream cmd watchdogs are the stop floors)."""
        node = self._depth_node("DEPTH")
        try:
            node._on_depth(_depth_msg())
            self._tick(node)  # inference + publish
            baseline = node._cmd_vel_pub.publish.call_count
            self._tick(node)  # no new frame -> skip
            self._tick(node)
            self.assertEqual(node._policy.call_count, 1)
            self.assertEqual(node._cmd_vel_pub.publish.call_count, baseline)
        finally:
            node.destroy_node()

    def test_fresh_frame_every_tick_never_skips(self) -> None:
        """Real-robot cadence: a new frame before each tick advances the
        counter every time, so the gate never skips."""
        node = self._depth_node("DEPTH")
        try:
            for _ in range(5):
                node._on_depth(_depth_msg())  # fresh frame each tick
                self._tick(node)
            self.assertEqual(node._policy.call_count, 5)
            self.assertEqual(node._cmd_vel_pub.publish.call_count, 5)
        finally:
            node.destroy_node()

    def test_watchdog_zero_twists_even_when_gate_would_skip(self) -> None:
        """Ordering pin: with the frame already consumed (gate would skip),
        a stale source mid-mission still zero-twists FIRST — safety before
        the freshness gate."""
        node = self._depth_node("DEPTH")
        try:
            node._on_depth(_depth_msg())
            self._tick(node)  # consumes the frame
            self.assertEqual(node._policy.call_count, 1)
            self.assertEqual(node._cmd_vel_pub.publish.call_count, 1)

            # No new frame (gate would skip) AND imu goes stale mid-mission.
            self._stamp_streams_fresh(node)
            node._last_imu_rx_t = time.monotonic() - 60.0
            node._on_tick()

            self.assertEqual(node._policy.call_count, 1)  # no inference ran
            self.assertEqual(node._cmd_vel_pub.publish.call_count, 2)
            published = node._cmd_vel_pub.publish.call_args[0][0]
            self.assertEqual(
                (published.linear.x, published.linear.y, published.angular.z),
                (0.0, 0.0, 0.0),
            )
        finally:
            node.destroy_node()

    def test_depth_subgoal_burst_gates_on_depth_frame(self) -> None:
        """The deployed hybrid variant (depth + subgoal) gates on the depth
        frame exactly as DEPTH; the subgoal source stays fresh throughout."""
        node = self._depth_node("DEPTH_SUBGOAL")
        try:
            self.assertTrue(node._has_depth)
            self.assertTrue(node._uses_subgoal)
            node._on_depth(_depth_msg())
            for _ in range(3):
                self._tick(node)
            self.assertEqual(node._policy.call_count, 1)
            node._on_depth(_depth_msg())
            self._tick(node)
            self.assertEqual(node._policy.call_count, 2)
        finally:
            node.destroy_node()

    def test_shape_error_tick_still_consumes_frame(self) -> None:
        """The consume sits BEFORE the action-shape check: a malformed policy
        output has already advanced the recurrent state, so the frame is
        consumed. The next frameless tick must gate-skip (not re-run the
        policy on the same frame and double-advance the GRU)."""
        node = self._depth_node("DEPTH")
        node._policy = _WrongShapePolicy()
        try:
            node._on_depth(_depth_msg())
            self._tick(node)  # shape-error path: zero-twist, frame consumed
            self.assertEqual(node._policy.call_count, 1)
            published = node._cmd_vel_pub.publish.call_args[0][0]
            self.assertEqual(
                (published.linear.x, published.linear.y, published.angular.z),
                (0.0, 0.0, 0.0),
            )
            self._tick(node)  # no new frame -> skip despite the earlier error
            self.assertEqual(node._policy.call_count, 1)  # NOT re-run
        finally:
            node.destroy_node()

    def test_transient_obs_none_does_not_consume_pending_frame(self) -> None:
        """The consume sits AFTER the obs-None early-return: a fresh frame
        that fails obs assembly (e.g. a malformed JointState) is NOT consumed,
        so it is still inferred once obs recovers — no permanently lost frame."""
        node = self._depth_node("DEPTH")
        try:
            obs_stream = iter([None, np.zeros(4, dtype=np.float32)])
            node._assemble_observation_or_none = (  # type: ignore
                lambda: next(obs_stream)
            )
            node._on_depth(_depth_msg())  # one fresh frame pending
            self._tick(node)  # obs None -> zero-twist, frame NOT consumed
            self.assertEqual(node._policy.call_count, 0)
            published = node._cmd_vel_pub.publish.call_args[0][0]
            self.assertEqual(
                (published.linear.x, published.linear.y, published.angular.z),
                (0.0, 0.0, 0.0),
            )
            self._tick(node)  # obs recovers, SAME frame -> inference runs
            self.assertEqual(node._policy.call_count, 1)
        finally:
            node.destroy_node()

    def test_nocam_variant_never_gates(self) -> None:
        """No-camera variant has no depth field: the gate is unreachable, so
        every clean tick infers though no frame counter ever advances, and
        the consume path (guarded on _has_depth) leaves the sentinel alone."""
        node = self._depth_node("NOCAM")
        try:
            self.assertFalse(node._has_depth)
            for _ in range(3):
                self._tick(node)
            self.assertEqual(node._policy.call_count, 3)
            self.assertEqual(node._cmd_vel_pub.publish.call_count, 3)
            self.assertEqual(node._last_inferred_depth_seq, -1)
        finally:
            node.destroy_node()

    def test_nocam_subgoal_variant_never_gates(self) -> None:
        node = self._depth_node("NOCAM_SUBGOAL")
        try:
            self.assertFalse(node._has_depth)
            self.assertTrue(node._uses_subgoal)
            for _ in range(3):
                self._tick(node)
            self.assertEqual(node._policy.call_count, 3)
            self.assertEqual(node._last_inferred_depth_seq, -1)
        finally:
            node.destroy_node()
