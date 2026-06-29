"""Unit tests for JetsonRosClient — no ROS runtime needed.

Tests bypass __init__ (which starts rclpy) by constructing the object
with ``object.__new__`` and manually wiring the internal state.  ROS
message objects are replaced with lightweight MagicMocks.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strafer_autonomy.clients.ros_client import (
    JetsonRosClient,
    RosClientConfig,
    _ClockStallDetector,
)
from strafer_autonomy.schemas import GoalPoseCandidate, Pose3D, SceneObservation, SkillResult

pytestmark = pytest.mark.requires_ros


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(config: RosClientConfig | None = None) -> JetsonRosClient:
    """Create a JetsonRosClient without starting rclpy."""
    from rclpy.time import Time as RclpyTime

    client = object.__new__(JetsonRosClient)
    client._config = config or RosClientConfig()
    client._cache_lock = threading.Lock()
    client._latest_color = None
    client._latest_color_rx_t = time.monotonic()
    client._latest_depth = None
    client._latest_cam_info = None
    client._latest_odom = None
    client._nav_lock = threading.Lock()
    client._active_goal_handle = None
    client._detections_pub = None  # Tests opt in via TestPublishDetections.
    client._grounding_frame_pub = None  # Tests opt in via TestPublishGroundingFrame.
    client._detections_fg_pub = None  # Tests opt in via TestPublishDetectionsFG.
    # Mock the ROS node — use a real rclpy.time.Time advancing with the
    # monotonic wall clock so Time + Duration arithmetic and Time
    # comparisons in _wait_for_future work, and ``.to_msg()`` returns a
    # real ``builtin_interfaces.msg.Time`` for goal-stamp assignments.
    client._node = MagicMock()
    client._node.get_clock.return_value.now.side_effect = (
        lambda: RclpyTime(seconds=time.monotonic())
    )
    return client


def _make_stamp(sec: int = 100, nanosec: int = 300_000_000) -> SimpleNamespace:
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _make_future(*, done: bool = True, result=None) -> MagicMock:
    """Create a mock ROS future compatible with the event-driven _wait_for_future.

    When *done* is True, ``add_done_callback`` immediately invokes the
    callback (simulating an already-resolved future).  When False, the
    callback is stored but never fired, so ``_wait_for_future`` will
    time out.
    """
    future = MagicMock()
    future.done.return_value = done
    if result is not None:
        future.result.return_value = result

    def _add_done_cb(cb):
        if done:
            cb(future)

    future.add_done_callback.side_effect = _add_done_cb
    return future


def _make_color_msg(stamp: SimpleNamespace | None = None) -> MagicMock:
    msg = MagicMock()
    msg.header.stamp = stamp or _make_stamp()
    msg.header.frame_id = "d555_color_optical_frame"
    return msg


def _make_depth_msg(stamp: SimpleNamespace | None = None) -> MagicMock:
    msg = MagicMock()
    msg.header.stamp = stamp or _make_stamp()
    return msg


def _make_cam_info_msg() -> SimpleNamespace:
    return SimpleNamespace(
        height=360,
        width=640,
        k=[615.0, 0.0, 320.0, 0.0, 615.0, 180.0, 0.0, 0.0, 1.0],
        d=[0.0, 0.0, 0.0, 0.0, 0.0],
        p=[615.0, 0.0, 320.0, 0.0, 0.0, 615.0, 180.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        distortion_model="plumb_bob",
    )


def _make_odom_msg() -> MagicMock:
    msg = MagicMock()
    msg.header.frame_id = "odom"
    msg.pose.pose.position.x = 1.0
    msg.pose.pose.position.y = 2.0
    msg.pose.pose.position.z = 0.0
    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = 0.0
    msg.pose.pose.orientation.w = 1.0
    msg.twist.twist.linear.x = 0.1
    msg.twist.twist.linear.y = 0.0
    msg.twist.twist.angular.z = 0.0
    return msg


# ------------------------------------------------------------------
# Tests — capture_scene_observation
# ------------------------------------------------------------------


class TestCaptureSceneObservation(unittest.TestCase):
    """Tests for JetsonRosClient.capture_scene_observation."""

    @patch("cv_bridge.CvBridge")
    def test_basic_capture_returns_scene_observation(self, MockBridge: MagicMock) -> None:
        bridge = MockBridge.return_value
        color_arr = np.zeros((360, 640, 3), dtype=np.uint8)
        depth_arr = np.ones((360, 640), dtype=np.uint16) * 2000  # 2000 mm = 2.0 m
        bridge.imgmsg_to_cv2.side_effect = [color_arr, depth_arr]

        client = _make_client()
        client._latest_color = _make_color_msg()
        client._latest_depth = _make_depth_msg()
        client._latest_cam_info = _make_cam_info_msg()
        client._latest_odom = _make_odom_msg()

        obs = client.capture_scene_observation()

        self.assertIsInstance(obs, SceneObservation)
        self.assertEqual(obs.camera_frame, "d555_color_optical_frame")
        self.assertAlmostEqual(obs.stamp_sec, 100.3)
        self.assertEqual(obs.color_image_bgr.shape, (360, 640, 3))
        self.assertEqual(obs.aligned_depth_m.shape, (360, 640))
        np.testing.assert_allclose(obs.aligned_depth_m, 2.0, atol=1e-4)
        self.assertTrue(obs.tf_snapshot_ready)
        self.assertEqual(len(obs.observation_id), 12)

    @patch("cv_bridge.CvBridge")
    def test_capture_accepts_float32_depth(self, MockBridge: MagicMock) -> None:
        """Isaac Sim bridge publishes 32FC1 (m); keep as-is without rescaling."""
        bridge = MockBridge.return_value
        color_arr = np.zeros((360, 640, 3), dtype=np.uint8)
        depth_arr = np.full((360, 640), 2.0, dtype=np.float32)
        bridge.imgmsg_to_cv2.side_effect = [color_arr, depth_arr]

        client = _make_client()
        client._latest_color = _make_color_msg()
        client._latest_depth = _make_depth_msg()

        obs = client.capture_scene_observation()

        np.testing.assert_allclose(obs.aligned_depth_m, 2.0, atol=1e-6)

    @patch("cv_bridge.CvBridge")
    def test_capture_without_odom_still_succeeds(self, MockBridge: MagicMock) -> None:
        bridge = MockBridge.return_value
        bridge.imgmsg_to_cv2.side_effect = [
            np.zeros((360, 640, 3), dtype=np.uint8),
            np.zeros((360, 640), dtype=np.uint16),
        ]

        client = _make_client()
        client._latest_color = _make_color_msg()
        client._latest_depth = _make_depth_msg()

        obs = client.capture_scene_observation()

        self.assertIsNone(obs.robot_pose_map)
        self.assertFalse(obs.tf_snapshot_ready)
        self.assertEqual(obs.camera_info, {})

    def test_capture_raises_when_no_frames(self) -> None:
        client = _make_client()
        with self.assertRaises(RuntimeError) as ctx:
            client.capture_scene_observation()
        self.assertIn("No color or depth frames", str(ctx.exception))

    def test_capture_raises_when_only_color(self) -> None:
        client = _make_client()
        client._latest_color = _make_color_msg()
        with self.assertRaises(RuntimeError):
            client.capture_scene_observation()

    def test_capture_raises_when_only_depth(self) -> None:
        client = _make_client()
        client._latest_depth = _make_depth_msg()
        with self.assertRaises(RuntimeError):
            client.capture_scene_observation()

    def test_capture_raises_on_stale_frame(self) -> None:
        """Frame received 50s ago on the wall clock → age >> max_age."""
        client = _make_client()
        client._latest_color = _make_color_msg()
        client._latest_depth = _make_depth_msg()
        client._latest_color_rx_t = time.monotonic() - 50.0
        with self.assertRaises(RuntimeError) as ctx:
            client.capture_scene_observation()
        self.assertIn("old", str(ctx.exception))

    @patch("cv_bridge.CvBridge")
    def test_camera_info_fields(self, MockBridge: MagicMock) -> None:
        bridge = MockBridge.return_value
        bridge.imgmsg_to_cv2.side_effect = [
            np.zeros((360, 640, 3), dtype=np.uint8),
            np.zeros((360, 640), dtype=np.uint16),
        ]

        client = _make_client()
        client._latest_color = _make_color_msg()
        client._latest_depth = _make_depth_msg()
        client._latest_cam_info = _make_cam_info_msg()

        obs = client.capture_scene_observation()

        self.assertEqual(obs.camera_info["height"], 360)
        self.assertEqual(obs.camera_info["width"], 640)
        self.assertEqual(obs.camera_info["distortion_model"], "plumb_bob")
        self.assertEqual(len(obs.camera_info["k"]), 9)

    @patch("cv_bridge.CvBridge")
    def test_robot_pose_map_fields(self, MockBridge: MagicMock) -> None:
        bridge = MockBridge.return_value
        bridge.imgmsg_to_cv2.side_effect = [
            np.zeros((360, 640, 3), dtype=np.uint8),
            np.zeros((360, 640), dtype=np.uint16),
        ]

        client = _make_client()
        client._latest_color = _make_color_msg()
        client._latest_depth = _make_depth_msg()
        client._latest_odom = _make_odom_msg()

        obs = client.capture_scene_observation()

        self.assertIsNotNone(obs.robot_pose_map)
        self.assertAlmostEqual(obs.robot_pose_map["x"], 1.0)
        self.assertAlmostEqual(obs.robot_pose_map["y"], 2.0)
        self.assertAlmostEqual(obs.robot_pose_map["qw"], 1.0)
        self.assertEqual(obs.robot_pose_map["frame_id"], "odom")

    @patch("cv_bridge.CvBridge")
    def test_depth_conversion_mm_to_meters(self, MockBridge: MagicMock) -> None:
        bridge = MockBridge.return_value
        bridge.imgmsg_to_cv2.side_effect = [
            np.zeros((60, 80, 3), dtype=np.uint8),
            np.array([[500, 1000], [2500, 0]], dtype=np.uint16),
        ]

        client = _make_client()
        client._latest_color = _make_color_msg()
        client._latest_depth = _make_depth_msg()

        obs = client.capture_scene_observation()

        np.testing.assert_allclose(obs.aligned_depth_m[0, 0], 0.5, atol=1e-4)
        np.testing.assert_allclose(obs.aligned_depth_m[0, 1], 1.0, atol=1e-4)
        np.testing.assert_allclose(obs.aligned_depth_m[1, 0], 2.5, atol=1e-4)
        np.testing.assert_allclose(obs.aligned_depth_m[1, 1], 0.0, atol=1e-4)


# ------------------------------------------------------------------
# Tests — subscription callbacks
# ------------------------------------------------------------------


class TestSubscriptionCallbacks(unittest.TestCase):
    """Verify that callbacks store messages in the cache."""

    def test_color_callback(self) -> None:
        client = _make_client()
        msg = MagicMock()
        client._on_color(msg)
        self.assertIs(client._latest_color, msg)

    def test_depth_callback(self) -> None:
        client = _make_client()
        msg = MagicMock()
        client._on_depth(msg)
        self.assertIs(client._latest_depth, msg)

    def test_cam_info_callback(self) -> None:
        client = _make_client()
        msg = MagicMock()
        client._on_cam_info(msg)
        self.assertIs(client._latest_cam_info, msg)

    def test_odom_callback(self) -> None:
        client = _make_client()
        msg = MagicMock()
        client._on_odom(msg)
        self.assertIs(client._latest_odom, msg)


# ------------------------------------------------------------------
# Tests — helpers
# ------------------------------------------------------------------


class TestHelpers(unittest.TestCase):

    def test_stamp_to_sec(self) -> None:
        stamp = SimpleNamespace(sec=10, nanosec=500_000_000)
        self.assertAlmostEqual(JetsonRosClient._stamp_to_sec(stamp), 10.5)

    def test_yaw_from_identity_quaternion(self) -> None:
        q = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        self.assertAlmostEqual(JetsonRosClient._yaw_from_quaternion(q), 0.0)

    def test_yaw_from_90_degree_rotation(self) -> None:
        import math
        # 90° about Z: qz=sin(45°), qw=cos(45°)
        q = SimpleNamespace(x=0.0, y=0.0, z=math.sin(math.pi / 4), w=math.cos(math.pi / 4))
        self.assertAlmostEqual(JetsonRosClient._yaw_from_quaternion(q), math.pi / 2, places=5)

    def test_normalize_angle(self) -> None:
        import math
        self.assertAlmostEqual(JetsonRosClient._normalize_angle(3 * math.pi), math.pi, places=5)
        self.assertAlmostEqual(JetsonRosClient._normalize_angle(-3 * math.pi), -math.pi, places=5)
        self.assertAlmostEqual(JetsonRosClient._normalize_angle(0.5), 0.5, places=5)


# ------------------------------------------------------------------
# Tests — config
# ------------------------------------------------------------------


class TestRosClientConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = RosClientConfig()
        self.assertEqual(cfg.default_goal_frame, "map")
        self.assertEqual(cfg.default_nav_timeout_s, 90.0)
        self.assertEqual(cfg.observation_max_age_s, 0.5)

    def test_custom_values(self) -> None:
        cfg = RosClientConfig(observation_max_age_s=2.0, default_goal_frame="odom")
        self.assertEqual(cfg.observation_max_age_s, 2.0)
        self.assertEqual(cfg.default_goal_frame, "odom")


# ------------------------------------------------------------------
# Tests — get_robot_state (Task 6)
# ------------------------------------------------------------------


class TestGetRobotState(unittest.TestCase):

    def test_state_with_odom(self) -> None:
        client = _make_client()
        client._latest_odom = _make_odom_msg()

        state = client.get_robot_state()

        self.assertIn("timestamp", state)
        self.assertIsNotNone(state["pose"])
        self.assertAlmostEqual(state["pose"]["x"], 1.0)
        self.assertAlmostEqual(state["pose"]["y"], 2.0)
        self.assertAlmostEqual(state["pose"]["qw"], 1.0)
        self.assertAlmostEqual(state["velocity"]["linear_x"], 0.1)
        self.assertEqual(state["odom_frame"], "odom")
        self.assertFalse(state["navigation_active"])

    def test_state_without_odom(self) -> None:
        client = _make_client()

        state = client.get_robot_state()

        self.assertIsNone(state["pose"])
        self.assertIsNone(state["velocity"])

    def test_navigation_active_flag(self) -> None:
        client = _make_client()
        client._active_goal_handle = MagicMock()  # simulate active nav

        state = client.get_robot_state()

        self.assertTrue(state["navigation_active"])


# ------------------------------------------------------------------
# Tests — cancel_active_navigation (Task 5)
# ------------------------------------------------------------------


class TestCancelActiveNavigation(unittest.TestCase):

    def test_cancel_when_no_goal(self) -> None:
        client = _make_client()
        self.assertFalse(client.cancel_active_navigation())

    def test_cancel_active_goal(self) -> None:
        client = _make_client()
        goal_handle = MagicMock()
        cancel_future = _make_future(done=True)
        goal_handle.cancel_goal_async.return_value = cancel_future
        client._active_goal_handle = goal_handle

        result = client.cancel_active_navigation()

        self.assertTrue(result)
        goal_handle.cancel_goal_async.assert_called_once()
        self.assertIsNone(client._active_goal_handle)

    def test_cancel_timeout_returns_false(self) -> None:
        client = _make_client()
        goal_handle = MagicMock()
        cancel_future = _make_future(done=False)
        goal_handle.cancel_goal_async.return_value = cancel_future
        client._active_goal_handle = goal_handle

        # Use a very short timeout config to avoid slow test
        result = client.cancel_active_navigation()
        # Will timeout after 5s default, but done() always returns False
        # The test monkey-patches _wait_for_future instead:
        self.assertFalse(result)


# ------------------------------------------------------------------
# Tests — project_detection_to_goal_pose (Task 4)
# ------------------------------------------------------------------


class TestProjectDetectionToGoalPose(unittest.TestCase):

    def _make_response(self, found: bool = True) -> MagicMock:
        resp = MagicMock()
        resp.found = found
        resp.depth_valid = True
        resp.quality_flags = ["near_min_range"]
        resp.message = "Projected target"
        resp.goal_pose.header.frame_id = "map"
        resp.goal_pose.pose.position.x = 3.0
        resp.goal_pose.pose.position.y = 4.0
        resp.goal_pose.pose.position.z = 0.0
        resp.goal_pose.pose.orientation.x = 0.0
        resp.goal_pose.pose.orientation.y = 0.0
        resp.goal_pose.pose.orientation.z = 0.0
        resp.goal_pose.pose.orientation.w = 1.0
        resp.target_pose.pose.position.x = 4.0
        resp.target_pose.pose.position.y = 4.0
        resp.target_pose.pose.position.z = 0.0
        resp.target_pose.pose.orientation.x = 0.0
        resp.target_pose.pose.orientation.y = 0.0
        resp.target_pose.pose.orientation.z = 0.0
        resp.target_pose.pose.orientation.w = 1.0
        return resp

    @patch("strafer_msgs.srv.ProjectDetectionToGoalPose")
    def test_successful_projection(self, MockSrv: MagicMock) -> None:
        client = _make_client()

        # Mock the service client
        srv_client = MagicMock()
        srv_client.wait_for_service.return_value = True
        future = _make_future(done=True, result=self._make_response(found=True))
        srv_client.call_async.return_value = future
        client._projection_client = srv_client

        result = client.project_detection_to_goal_pose(
            request_id="test-123",
            image_stamp_sec=100.0,
            bbox_2d=(100, 200, 300, 400),
            standoff_m=0.7,
            target_label="chair",
        )

        self.assertIsInstance(result, GoalPoseCandidate)
        self.assertTrue(result.found)
        self.assertIsNotNone(result.goal_pose)
        self.assertAlmostEqual(result.goal_pose.x, 3.0)
        self.assertTrue(result.depth_valid)
        self.assertIn("near_min_range", result.quality_flags)

    @patch("strafer_msgs.srv.ProjectDetectionToGoalPose")
    def test_projection_not_found(self, MockSrv: MagicMock) -> None:
        client = _make_client()
        srv_client = MagicMock()
        srv_client.wait_for_service.return_value = True
        resp = self._make_response(found=False)
        resp.found = False
        future = _make_future(done=True, result=resp)
        srv_client.call_async.return_value = future
        client._projection_client = srv_client

        result = client.project_detection_to_goal_pose(
            request_id="test-456",
            image_stamp_sec=100.0,
            bbox_2d=(100, 200, 300, 400),
            standoff_m=0.7,
        )

        self.assertFalse(result.found)
        self.assertIsNone(result.goal_pose)

    def test_projection_service_unavailable_raises(self) -> None:
        client = _make_client()
        srv_client = MagicMock()
        srv_client.wait_for_service.return_value = False
        client._projection_client = srv_client

        with self.assertRaises(RuntimeError) as ctx:
            client.project_detection_to_goal_pose(
                request_id="test-789",
                image_stamp_sec=100.0,
                bbox_2d=(100, 200, 300, 400),
                standoff_m=0.7,
            )
        self.assertIn("not available", str(ctx.exception))


# ------------------------------------------------------------------
# Tests — navigate_to_pose (Task 3)
# ------------------------------------------------------------------


class TestNavigateToPose(unittest.TestCase):

    def _make_nav_goal_handle(self, accepted: bool = True, status: int = 4) -> MagicMock:
        """Create a mock goal handle. Status 4 = STATUS_SUCCEEDED."""
        goal_handle = MagicMock()
        goal_handle.accepted = accepted
        result_response = MagicMock()
        result_response.status = status
        result_future = _make_future(done=True, result=result_response)
        goal_handle.get_result_async.return_value = result_future
        return goal_handle

    @patch("nav2_msgs.action.NavigateToPose")
    @patch("rclpy.action.ActionClient")
    def test_nav_server_unavailable(self, MockActionClient: MagicMock, _: MagicMock) -> None:
        client = _make_client()
        action_client = MagicMock()
        action_client.wait_for_server.return_value = False
        client._nav2_client = action_client

        result = client.navigate_to_pose(
            step_id="nav-1", goal_pose=Pose3D(x=1.0, y=2.0),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "nav2_unavailable")

    @patch("action_msgs.msg.GoalStatus")
    @patch("nav2_msgs.action.NavigateToPose")
    @patch("rclpy.action.ActionClient")
    def test_goal_rejected(self, MockAC: MagicMock, MockNav: MagicMock, _: MagicMock) -> None:
        client = _make_client()
        action_client = MagicMock()
        action_client.wait_for_server.return_value = True

        send_future = _make_future(done=True, result=self._make_nav_goal_handle(accepted=False))
        action_client.send_goal_async.return_value = send_future
        client._nav2_client = action_client

        result = client.navigate_to_pose(
            step_id="nav-2", goal_pose=Pose3D(x=1.0, y=2.0),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "goal_rejected")

    @patch("action_msgs.msg.GoalStatus")
    @patch("nav2_msgs.action.NavigateToPose")
    @patch("rclpy.action.ActionClient")
    def test_successful_navigation(self, MockAC: MagicMock, MockNav: MagicMock, MockStatus: MagicMock) -> None:
        MockStatus.STATUS_SUCCEEDED = 4

        client = _make_client()
        action_client = MagicMock()
        action_client.wait_for_server.return_value = True

        goal_handle = self._make_nav_goal_handle(accepted=True, status=4)
        send_future = _make_future(done=True, result=goal_handle)
        action_client.send_goal_async.return_value = send_future
        client._nav2_client = action_client

        result = client.navigate_to_pose(
            step_id="nav-3", goal_pose=Pose3D(x=1.0, y=2.0),
        )

        self.assertEqual(result.status, "succeeded")
        self.assertIsNotNone(result.started_at)

    @patch("action_msgs.msg.GoalStatus")
    @patch("nav2_msgs.action.NavigateToPose")
    @patch("rclpy.action.ActionClient")
    def test_navigation_canceled(self, MockAC: MagicMock, MockNav: MagicMock, MockStatus: MagicMock) -> None:
        MockStatus.STATUS_SUCCEEDED = 4
        MockStatus.STATUS_CANCELED = 5

        client = _make_client()
        action_client = MagicMock()
        action_client.wait_for_server.return_value = True

        goal_handle = self._make_nav_goal_handle(accepted=True, status=5)
        send_future = _make_future(done=True, result=goal_handle)
        action_client.send_goal_async.return_value = send_future
        client._nav2_client = action_client

        result = client.navigate_to_pose(
            step_id="nav-4", goal_pose=Pose3D(x=1.0, y=2.0),
        )

        self.assertEqual(result.status, "canceled")


# ------------------------------------------------------------------
# Tests — rotate_in_place
# ------------------------------------------------------------------


class TestRotateInPlace(unittest.TestCase):

    def test_no_odom_fails(self) -> None:
        client = _make_client()

        result = client.rotate_in_place(step_id="rot-1", yaw_delta_rad=0.5)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "no_odom")

    def test_already_at_target_succeeds(self) -> None:
        """When yaw_delta is 0, we're already at the target yaw."""
        client = _make_client()
        client._latest_odom = _make_odom_msg()  # yaw=0

        # Create a mock publisher
        pub = MagicMock()
        client._cmd_vel_pub = pub

        result = client.rotate_in_place(
            step_id="rot-2", yaw_delta_rad=0.0, tolerance_rad=0.2,
        )

        self.assertEqual(result.status, "succeeded")


# ------------------------------------------------------------------
# Tests — rotate_in_place deadline source (sim-clock vs wall-clock)
# ------------------------------------------------------------------


class TestRotateInPlaceDeadlineSource(unittest.TestCase):
    """The deadline must come from ``node.get_clock().now()`` (sim-aware),
    not ``time.monotonic()`` (wall). At sub-unity RTF a wall deadline trips
    in fractional sim-seconds and rotates always fail.
    """

    def _unconverged_client(self) -> JetsonRosClient:
        """Client with odom yaw=0 but a non-zero rotation request, so the
        loop never converges and is forced onto the deadline path."""
        client = _make_client()
        client._latest_odom = _make_odom_msg()  # yaw=0
        client._cmd_vel_pub = MagicMock()
        return client

    def test_use_sim_time_true_frozen_clock_trips_stall_detector(self) -> None:
        """``use_sim_time:=true`` analog: ``/clock`` frozen (crashed bridge).

        The sim-clock deadline never trips (sim time never advances), so
        the sim-time stall detector ends the loop after
        ``clock_stall_bail_wall_s`` of wall time with no /clock progress.
        Crucially the bail window is independent of ``timeout`` (here 5 s),
        so a frozen clock is caught fast even with a large sim budget —
        whereas the old ``2 * timeout`` wall cap would have waited 10 s.
        """
        from rclpy.time import Time

        bail = 0.15
        client = _make_client(RosClientConfig(clock_stall_bail_wall_s=bail))
        client._latest_odom = _make_odom_msg()  # yaw=0
        client._cmd_vel_pub = MagicMock()
        # Sim clock pinned at t=0 → sim deadline never trips, never advances.
        client._node.get_clock.return_value.now.side_effect = (
            lambda: Time(seconds=0)
        )

        timeout = 5.0  # large sim budget; the stall detector should win
        start = time.monotonic()
        result = client.rotate_in_place(
            step_id="rot-frozen", yaw_delta_rad=1.0,
            tolerance_rad=0.001, timeout_s=timeout,
        )
        elapsed = time.monotonic() - start

        self.assertEqual(result.status, "timeout")
        # Fires ~bail wall-seconds in, independent of (and far below) timeout.
        self.assertGreaterEqual(elapsed, bail * 0.8)
        self.assertLess(elapsed, bail + 1.0)
        # Clock was polled by the loop, not just consulted once.
        self.assertGreater(
            client._node.get_clock.return_value.now.call_count, 2
        )

    def test_use_sim_time_true_sim_deadline_trips_before_wall(self) -> None:
        """``use_sim_time:=true`` analog: ``/clock`` jumps ahead of wall.

        Stubs ``clock.now`` so the second read crosses the deadline while
        wall time has barely moved. With a wall-clock deadline this would
        wait the full ``timeout`` wall-seconds; with the sim-clock
        deadline the loop exits as soon as sim crosses.
        """
        from rclpy.time import Time

        client = self._unconverged_client()
        # First read sets deadline at 0 + timeout. Subsequent reads return
        # values past the deadline so the loop trips on the first iteration.
        sim_seconds = iter([0.0, 100.0, 200.0, 300.0])
        client._node.get_clock.return_value.now.side_effect = (
            lambda: Time(seconds=next(sim_seconds))
        )

        timeout = 5.0  # large wall budget; sim trip should fire well before
        start = time.monotonic()
        result = client.rotate_in_place(
            step_id="rot-sim-jump", yaw_delta_rad=1.0,
            tolerance_rad=0.001, timeout_s=timeout,
        )
        elapsed = time.monotonic() - start

        self.assertEqual(result.status, "timeout")
        # Sim deadline trips on the first loop iteration → well under wall.
        self.assertLess(elapsed, 1.0)

    def test_use_sim_time_false_uses_wall_clock_natively(self) -> None:
        """Real-robot bringup: ``use_sim_time=False``.

        ``node.get_clock().now()`` returns wall-rate time, so the sim-clock
        deadline behaves identically to the old wall-clock deadline.
        ``_make_client``'s default already wires the mocked clock to
        ``time.monotonic``, which is the wall-clock case.
        """
        client = self._unconverged_client()  # uses _make_client default

        timeout = 0.05
        start = time.monotonic()
        result = client.rotate_in_place(
            step_id="rot-real", yaw_delta_rad=1.0,
            tolerance_rad=0.001, timeout_s=timeout,
        )
        elapsed = time.monotonic() - start

        self.assertEqual(result.status, "timeout")
        # Wall-rate clock → deadline trips at ~timeout, not 2 * timeout.
        self.assertGreaterEqual(elapsed, timeout * 0.8)
        self.assertLess(elapsed, 2.0 * timeout)


# ------------------------------------------------------------------
# Tests — rotate_in_place cancel_event
# ------------------------------------------------------------------


class TestRotateInPlaceCancelEvent(unittest.TestCase):
    """``rotate_in_place`` honors a ``cancel_event`` mid-rotation.

    Without this, a cancel arriving via the executor's cancel path is
    invisible to the rotate loop and the chassis keeps rotating until
    tolerance or timeout — a real-robot safety bug.
    """

    def test_cancel_mid_rotation_publishes_zero_and_returns_canceled(self) -> None:
        from geometry_msgs.msg import Twist

        client = _make_client()
        client._latest_odom = _make_odom_msg()  # yaw=0
        pub = MagicMock()
        client._cmd_vel_pub = pub

        cancel = threading.Event()

        def _trip_cancel() -> None:
            time.sleep(0.05)
            cancel.set()

        tripper = threading.Thread(target=_trip_cancel, daemon=True)
        tripper.start()

        start = time.monotonic()
        # Request a 1-rad rotation we'll never converge on (odom is pinned
        # at yaw=0) with a 5 s timeout — the cancel must trip first.
        result = client.rotate_in_place(
            step_id="rot-cancel",
            yaw_delta_rad=1.0,
            tolerance_rad=0.001,
            timeout_s=5.0,
            cancel_event=cancel,
        )
        elapsed = time.monotonic() - start
        tripper.join(timeout=1.0)

        self.assertEqual(result.status, "canceled")
        self.assertEqual(result.error_code, "rotation_canceled")
        # Returns within one publish period of the check (loop sleeps
        # 20 ms between iterations; allow generous wall budget).
        self.assertLess(elapsed, 0.5)

        # The final publish must be a zero Twist (angular.z == 0).
        self.assertGreater(pub.publish.call_count, 0)
        last_twist = pub.publish.call_args_list[-1].args[0]
        self.assertIsInstance(last_twist, Twist)
        self.assertEqual(last_twist.angular.z, 0.0)
        self.assertEqual(last_twist.linear.x, 0.0)
        self.assertEqual(last_twist.linear.y, 0.0)

    def test_cancel_already_set_returns_immediately(self) -> None:
        """A cancel that arrives before the loop's first iteration still
        zeroes the publisher and returns canceled — the rotate-loop
        publish path is the same regardless of when the event trips."""
        from geometry_msgs.msg import Twist

        client = _make_client()
        client._latest_odom = _make_odom_msg()
        pub = MagicMock()
        client._cmd_vel_pub = pub

        cancel = threading.Event()
        cancel.set()

        start = time.monotonic()
        result = client.rotate_in_place(
            step_id="rot-pre-cancel",
            yaw_delta_rad=1.0,
            tolerance_rad=0.001,
            timeout_s=5.0,
            cancel_event=cancel,
        )
        elapsed = time.monotonic() - start

        self.assertEqual(result.status, "canceled")
        self.assertEqual(result.error_code, "rotation_canceled")
        self.assertLess(elapsed, 0.1)
        last_twist = pub.publish.call_args_list[-1].args[0]
        self.assertIsInstance(last_twist, Twist)
        self.assertEqual(last_twist.angular.z, 0.0)

    def test_no_cancel_event_kwarg_keeps_legacy_behavior(self) -> None:
        """Backward compat: callers that don't pass ``cancel_event`` still
        get the converged-on-target success path."""
        client = _make_client()
        client._latest_odom = _make_odom_msg()
        client._cmd_vel_pub = MagicMock()

        result = client.rotate_in_place(
            step_id="rot-no-cancel", yaw_delta_rad=0.0, tolerance_rad=0.2,
        )
        self.assertEqual(result.status, "succeeded")


# ------------------------------------------------------------------
# Tests — publish_zero_cmd_vel (cancel-path fail-safe)
# ------------------------------------------------------------------


class TestPublishZeroCmdVel(unittest.TestCase):
    """The executor's cancel path publishes a zero ``Twist`` directly,
    independent of any active ``rotate_in_place`` loop, to handle the
    race where cancel arrives before a rotate has entered its loop."""

    def test_publish_zero_creates_publisher_and_publishes_zero(self) -> None:
        from geometry_msgs.msg import Twist

        client = _make_client()
        pub = MagicMock()
        client._node.create_publisher.return_value = pub

        client.publish_zero_cmd_vel()

        # Publisher created with the right topic + type.
        args, _ = client._node.create_publisher.call_args
        self.assertEqual(args[0], Twist)
        self.assertEqual(args[1], "/cmd_vel")

        # A single zero Twist published.
        self.assertEqual(pub.publish.call_count, 1)
        twist = pub.publish.call_args.args[0]
        self.assertIsInstance(twist, Twist)
        self.assertEqual(twist.angular.z, 0.0)
        self.assertEqual(twist.linear.x, 0.0)
        self.assertEqual(twist.linear.y, 0.0)

    def test_publish_zero_reuses_existing_publisher(self) -> None:
        client = _make_client()
        pub = MagicMock()
        client._cmd_vel_pub = pub

        client.publish_zero_cmd_vel()
        client.publish_zero_cmd_vel()

        # Already-existing publisher reused; create_publisher never invoked.
        self.assertFalse(client._node.create_publisher.called)
        self.assertEqual(pub.publish.call_count, 2)


# ------------------------------------------------------------------
# Tests — _wait_for_future
# ------------------------------------------------------------------


class TestWaitForFuture(unittest.TestCase):

    def test_immediate_done(self) -> None:
        from rclpy.time import Time

        client = _make_client()
        # Pin sim time at 0 so the deadline never trips first.
        client._node.get_clock.return_value.now.side_effect = (
            lambda: Time(seconds=0)
        )
        future = _make_future(done=True)
        self.assertTrue(client._wait_for_future(future, 1.0))

    def test_timeout_on_sim_clock(self) -> None:
        """Sim-time deadline trips before the wall-clock safety cap."""
        from rclpy.time import Time

        client = _make_client()
        # Step the mocked sim clock forward 1 s per call after the
        # initial deadline read at t=0. timeout_s=1.0 -> deadline=1.0s;
        # the next clock.now() returns 2.0s, crossing the deadline. Wall
        # time barely advances, so the wall-cap (2.0s) does not fire.
        sim_seconds = iter([0.0, 2.0, 3.0, 4.0])
        client._node.get_clock.return_value.now.side_effect = (
            lambda: Time(seconds=next(sim_seconds))
        )
        future = _make_future(done=False)
        start = time.monotonic()
        self.assertFalse(client._wait_for_future(future, 1.0))
        elapsed = time.monotonic() - start
        # Sim-time deadline trips before the 2.0s wall cap.
        self.assertLess(elapsed, 1.5)

    def test_frozen_clock_trips_stall_detector(self) -> None:
        """A frozen sim clock must not wedge the executor — the stall
        detector fires after ``clock_stall_bail_wall_s`` of wall time with
        no /clock progress, independent of the (large) sim ``timeout_s``.
        """
        from rclpy.time import Time

        bail = 0.15
        client = _make_client(RosClientConfig(clock_stall_bail_wall_s=bail))
        # Sim clock never advances past 0.
        client._node.get_clock.return_value.now.side_effect = (
            lambda: Time(seconds=0)
        )
        future = _make_future(done=False)
        start = time.monotonic()
        # Large sim budget; with the old 2*timeout cap this would wait 10 s.
        self.assertFalse(client._wait_for_future(future, 5.0))
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, bail * 0.8)
        self.assertLess(elapsed, bail + 1.0)

    def test_slow_but_live_clock_does_not_trip_detector(self) -> None:
        """Sub-unity RTF: /clock advances slowly but never freezes. With a
        tiny bail window that would trip a *frozen* clock immediately, the
        future still completes because every poll shows sim-time progress.
        This is the regression the audit fixes — the old wall cap aborted
        here. (Multi-poll slow-clock behavior is proved deterministically
        in ``TestClockStallDetector``.)
        """
        from rclpy.time import Time

        # Sim advancing, all reads well under the 100 s deadline.
        sim_seconds = iter([0.0, 0.5, 1.0, 1.5, 2.0])
        client = _make_client(RosClientConfig(clock_stall_bail_wall_s=0.01))
        client._node.get_clock.return_value.now.side_effect = (
            lambda: Time(seconds=next(sim_seconds))
        )
        future = _make_future(done=True)
        self.assertTrue(client._wait_for_future(future, 100.0))


# ------------------------------------------------------------------
# Tests — _ClockStallDetector (sim-time-progress stall detector)
# ------------------------------------------------------------------


class TestClockStallDetector(unittest.TestCase):
    """The detector replaces the absolute ``2 * timeout`` wall cap. It must
    tolerate a slow-but-live ``/clock`` (sub-unity RTF) and fire only when
    sim time stops advancing entirely for ``bail_wall_s`` of wall time.
    """

    @staticmethod
    def _ns(seconds: float) -> int:
        return int(seconds * 1e9)

    def test_frozen_clock_stalls_after_bail_window(self) -> None:
        """Sim time pinned: stalls exactly once wall passes the bail window."""
        det = _ClockStallDetector(
            bail_wall_s=15.0, sim_now_ns=self._ns(0.0), wall_now_s=100.0
        )
        # 14.9 s of wall with no sim progress — not yet stalled.
        det.update(sim_now_ns=self._ns(0.0), wall_now_s=114.9)
        self.assertFalse(det.is_stalled(wall_now_s=114.9))
        # 15.0 s of wall with no sim progress — stalled.
        det.update(sim_now_ns=self._ns(0.0), wall_now_s=115.0)
        self.assertTrue(det.is_stalled(wall_now_s=115.0))

    def test_slow_but_live_clock_never_stalls(self) -> None:
        """Sub-unity RTF: 1 ms of sim per 1 s of wall (RTF≈0.001) for a
        wall span that dwarfs the bail window. Each advance resets the
        timer, so the detector never fires — the core regression fix.
        """
        det = _ClockStallDetector(
            bail_wall_s=15.0, sim_now_ns=self._ns(0.0), wall_now_s=0.0
        )
        # 60 polls × 1 s wall = 60 s wall (4× the 15 s bail), sim creeps
        # forward 2 ms each poll (above the 1 ms progress epsilon).
        for i in range(1, 61):
            sim_ns = self._ns(0.002 * i)
            wall = float(i)
            det.update(sim_now_ns=sim_ns, wall_now_s=wall)
            self.assertFalse(
                det.is_stalled(wall_now_s=wall),
                f"spurious stall at poll {i}",
            )

    def test_rtf_one_parity_never_stalls(self) -> None:
        """Real-robot analog: sim advances at wall rate. Never stalls, so
        on real hardware the primary deadline governs — indistinguishable
        from the old wall cap (which also never fired before the deadline).
        """
        det = _ClockStallDetector(
            bail_wall_s=15.0, sim_now_ns=self._ns(1000.0), wall_now_s=1000.0
        )
        for t in range(1001, 1100):
            det.update(sim_now_ns=self._ns(t), wall_now_s=float(t))
            self.assertFalse(det.is_stalled(wall_now_s=float(t)))

    def test_sub_epsilon_jitter_is_not_progress(self) -> None:
        """A frozen clock that only jitters sub-millisecond must still
        stall — republish noise below the 1 ms epsilon isn't progress.
        """
        det = _ClockStallDetector(
            bail_wall_s=10.0, sim_now_ns=self._ns(5.0), wall_now_s=0.0
        )
        # +0.5 ms jitter each poll, never crossing the 1 ms epsilon.
        det.update(sim_now_ns=self._ns(5.0) + 500_000, wall_now_s=9.9)
        self.assertFalse(det.is_stalled(wall_now_s=9.9))
        det.update(sim_now_ns=self._ns(5.0) + 500_000, wall_now_s=10.0)
        self.assertTrue(det.is_stalled(wall_now_s=10.0))

    def test_nonpositive_bail_disables_detector(self) -> None:
        """``clock_stall_bail_wall_s <= 0`` opts out: never reports stalled
        even with a frozen clock, leaving the sim-clock deadline as the
        sole bound (documented escape hatch).
        """
        det = _ClockStallDetector(
            bail_wall_s=0.0, sim_now_ns=self._ns(0.0), wall_now_s=0.0
        )
        det.update(sim_now_ns=self._ns(0.0), wall_now_s=10_000.0)
        self.assertFalse(det.is_stalled(wall_now_s=10_000.0))


# ------------------------------------------------------------------
# Tests — publish_detections (vision_msgs/Detection2DArray overlay)
# ------------------------------------------------------------------


class _StubMsg:
    """Minimal stand-in for a generated ROS message — accepts arbitrary attrs.

    ``center`` mirrors the vision_msgs 4.x (Humble) ``Pose2D`` shape:
    ``position.x/.y`` + ``theta``. The fallback path for older builds
    that expose ``.x/.y`` directly is exercised by a dedicated test that
    swaps ``BoundingBox2D`` for ``_StubMsgLegacyBbox``.
    """

    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None, frame_id="")
        self.detections: list[Any] = []
        self.results: list[Any] = []
        self.bbox: Any = None
        # BoundingBox2D members — Humble schema (Pose2D w/ nested position)
        self.center = SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0),
            theta=0.0,
        )
        self.size_x = 0.0
        self.size_y = 0.0
        # ObjectHypothesisWithPose members
        self.hypothesis = SimpleNamespace(class_id="", score=0.0)
        self.pose = SimpleNamespace()


class _StubMsgLegacyBbox(_StubMsg):
    """BoundingBox2D stub with legacy geometry_msgs/Pose2D shape (x/y direct)."""

    def __init__(self) -> None:
        super().__init__()
        self.center = SimpleNamespace(x=0.0, y=0.0, theta=0.0)


@contextmanager
def _stub_vision_msgs():
    """Inject vision_msgs / builtin_interfaces stubs for the duration of a test.

    The publisher does ``from vision_msgs.msg import (...)`` at call time, so
    we register lightweight modules in ``sys.modules`` that expose the same
    class names. Parametrized stamp construction is delegated to a dataclass-
    style ``Time`` stub mirroring ``builtin_interfaces.msg.Time``.
    """
    saved = {k: sys.modules.get(k) for k in (
        "vision_msgs", "vision_msgs.msg", "builtin_interfaces", "builtin_interfaces.msg",
    )}
    try:
        vm_pkg = types.ModuleType("vision_msgs")
        vm_msg = types.ModuleType("vision_msgs.msg")
        vm_msg.Detection2D = _StubMsg
        vm_msg.Detection2DArray = _StubMsg
        vm_msg.BoundingBox2D = _StubMsg
        vm_msg.ObjectHypothesisWithPose = _StubMsg
        vm_pkg.msg = vm_msg  # type: ignore[attr-defined]
        sys.modules["vision_msgs"] = vm_pkg
        sys.modules["vision_msgs.msg"] = vm_msg

        bi_pkg = types.ModuleType("builtin_interfaces")
        bi_msg = types.ModuleType("builtin_interfaces.msg")

        class _StubTime:
            def __init__(self, sec: int = 0, nanosec: int = 0) -> None:
                self.sec = int(sec)
                self.nanosec = int(nanosec)

        bi_msg.Time = _StubTime
        bi_pkg.msg = bi_msg  # type: ignore[attr-defined]
        sys.modules["builtin_interfaces"] = bi_pkg
        sys.modules["builtin_interfaces.msg"] = bi_msg

        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class TestPublishDetections(unittest.TestCase):
    """publish_detections converts Qwen [0,1000] bboxes → pixels and stamps."""

    def _captured(self, client: JetsonRosClient) -> Any:
        pub = client._detections_pub  # type: ignore[attr-defined]
        self.assertEqual(pub.publish.call_count, 1)
        return pub.publish.call_args[0][0]

    def test_publishes_single_detection_with_pixel_bbox(self) -> None:
        with _stub_vision_msgs():
            client = _make_client()
            pub = MagicMock()
            client._detections_pub = pub

            client.publish_detections(
                image_stamp_sec=100.5,
                image_frame_id="d555_color_optical_frame",
                image_width=640,
                image_height=360,
                detections=[((100, 200, 500, 600), "door", 0.91)],
            )

            msg = self._captured(client)
            self.assertEqual(msg.header.frame_id, "d555_color_optical_frame")
            # 100.5s → sec=100, nanosec=500_000_000.
            self.assertEqual(msg.header.stamp.sec, 100)
            self.assertEqual(msg.header.stamp.nanosec, 500_000_000)
            self.assertEqual(len(msg.detections), 1)

            det = msg.detections[0]
            # Center: ((100+500)/2, (200+600)/2) = (300, 400) in [0,1000].
            #   pixel_x = 300 * 640 / 1000 = 192.0
            #   pixel_y = 400 * 360 / 1000 = 144.0
            # Size: (400 * 640/1000, 400 * 360/1000) = (256, 144).
            # Humble's vision_msgs/Pose2D nests x/y under .position.
            self.assertAlmostEqual(det.bbox.center.position.x, 192.0)
            self.assertAlmostEqual(det.bbox.center.position.y, 144.0)
            self.assertAlmostEqual(det.bbox.size_x, 256.0)
            self.assertAlmostEqual(det.bbox.size_y, 144.0)
            self.assertEqual(det.results[0].hypothesis.class_id, "door")
            self.assertAlmostEqual(det.results[0].hypothesis.score, 0.91)

    def test_falls_back_to_legacy_pose2d_shape(self) -> None:
        """Pre-Humble vision_msgs (geometry_msgs/Pose2D) exposes x/y directly."""
        with _stub_vision_msgs():
            # Swap in the legacy bbox stub for this one test.
            sys.modules["vision_msgs.msg"].BoundingBox2D = _StubMsgLegacyBbox  # type: ignore[attr-defined]

            client = _make_client()
            pub = MagicMock()
            client._detections_pub = pub

            client.publish_detections(
                image_stamp_sec=1.0,
                image_frame_id="frame",
                image_width=640,
                image_height=360,
                detections=[((0, 0, 1000, 1000), "door", 0.5)],
            )

            msg = self._captured(client)
            det = msg.detections[0]
            self.assertAlmostEqual(det.bbox.center.x, 320.0)
            self.assertAlmostEqual(det.bbox.center.y, 180.0)
            # Sanity: legacy stub doesn't grow a .position attr behind our back.
            self.assertFalse(hasattr(det.bbox.center, "position"))

    def test_empty_detections_publishes_empty_array(self) -> None:
        """An empty detection list must still publish — clears stale overlay."""
        with _stub_vision_msgs():
            client = _make_client()
            pub = MagicMock()
            client._detections_pub = pub

            client.publish_detections(
                image_stamp_sec=0.0,
                image_frame_id="d555_color_optical_frame",
                image_width=640,
                image_height=360,
                detections=[],
            )

            msg = self._captured(client)
            self.assertEqual(msg.detections, [])
            self.assertEqual(msg.header.stamp.sec, 0)
            self.assertEqual(msg.header.stamp.nanosec, 0)

    def test_publisher_is_reused_across_calls(self) -> None:
        """The eagerly created publisher is reused across publish calls."""
        with _stub_vision_msgs():
            client = _make_client()
            pub = MagicMock()
            client._detections_pub = pub

            for _ in range(3):
                client.publish_detections(
                    image_stamp_sec=1.0,
                    image_frame_id="frame",
                    image_width=100,
                    image_height=100,
                    detections=[],
                )

            self.assertEqual(pub.publish.call_count, 3)

    def test_no_publish_when_vision_msgs_missing(self) -> None:
        """Missing vision_msgs at startup → publisher None → no-op + no crash."""
        # Note: no _stub_vision_msgs() here so the import path stays "real".
        client = _make_client()
        # _make_client already sets _detections_pub = None.

        # Should not raise even though vision_msgs may not be installed —
        # the early-return guards the subsequent imports.
        client.publish_detections(
            image_stamp_sec=1.0,
            image_frame_id="frame",
            image_width=100,
            image_height=100,
            detections=[((0, 0, 1000, 1000), "door", 0.9)],
        )

    def test_handles_label_and_confidence_none(self) -> None:
        """Missing label / confidence default to '' / 0.0 (not crash)."""
        with _stub_vision_msgs():
            client = _make_client()
            pub = MagicMock()
            client._detections_pub = pub

            client.publish_detections(
                image_stamp_sec=1.0,
                image_frame_id="frame",
                image_width=640,
                image_height=360,
                detections=[((0, 0, 1000, 1000), None, None)],
            )

            msg = self._captured(client)
            det = msg.detections[0]
            self.assertEqual(det.results[0].hypothesis.class_id, "")
            self.assertAlmostEqual(det.results[0].hypothesis.score, 0.0)


# ------------------------------------------------------------------
# Tests — publish_grounding_frame (latched source-image companion)
# ------------------------------------------------------------------


@contextmanager
def _stub_grounding_frame_deps():
    """Inject cv_bridge / builtin_interfaces stubs for grounding-frame tests."""
    saved = {k: sys.modules.get(k) for k in (
        "cv_bridge", "builtin_interfaces", "builtin_interfaces.msg",
    )}
    try:
        # cv_bridge stub: CvBridge().cv2_to_imgmsg(arr, encoding=...) → stub Image.
        class _StubImage:
            def __init__(self) -> None:
                self.header = SimpleNamespace(stamp=None, frame_id="")
                self.encoding = ""
                self.data = b""
                self.height = 0
                self.width = 0

        class _StubBridge:
            def cv2_to_imgmsg(self, arr, encoding="bgr8"):  # noqa: D401
                msg = _StubImage()
                msg.encoding = encoding
                shape = getattr(arr, "shape", (0, 0))
                msg.height = int(shape[0]) if len(shape) >= 1 else 0
                msg.width = int(shape[1]) if len(shape) >= 2 else 0
                return msg

        cb_mod = types.ModuleType("cv_bridge")
        cb_mod.CvBridge = _StubBridge  # type: ignore[attr-defined]
        sys.modules["cv_bridge"] = cb_mod

        bi_pkg = types.ModuleType("builtin_interfaces")
        bi_msg = types.ModuleType("builtin_interfaces.msg")

        class _StubTime:
            def __init__(self, sec: int = 0, nanosec: int = 0) -> None:
                self.sec = int(sec)
                self.nanosec = int(nanosec)

        bi_msg.Time = _StubTime
        bi_pkg.msg = bi_msg  # type: ignore[attr-defined]
        sys.modules["builtin_interfaces"] = bi_pkg
        sys.modules["builtin_interfaces.msg"] = bi_msg

        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class TestPublishGroundingFrame(unittest.TestCase):
    """publish_grounding_frame encodes the source image and stamps it."""

    def test_publishes_image_with_source_stamp_and_frame(self) -> None:
        with _stub_grounding_frame_deps():
            client = _make_client()
            pub = MagicMock()
            client._grounding_frame_pub = pub

            image = np.zeros((360, 640, 3), dtype=np.uint8)
            client.publish_grounding_frame(
                image_bgr=image,
                image_stamp_sec=100.5,
                image_frame_id="d555_color_optical_frame",
            )

            self.assertEqual(pub.publish.call_count, 1)
            img_msg = pub.publish.call_args[0][0]
            self.assertEqual(img_msg.encoding, "bgr8")
            self.assertEqual(img_msg.height, 360)
            self.assertEqual(img_msg.width, 640)
            self.assertEqual(img_msg.header.frame_id, "d555_color_optical_frame")
            # 100.5s → sec=100, nanosec=500_000_000.
            self.assertEqual(img_msg.header.stamp.sec, 100)
            self.assertEqual(img_msg.header.stamp.nanosec, 500_000_000)

    def test_no_publish_when_frame_pub_is_none(self) -> None:
        """Missing publisher → no-op, no crash (mirrors detections path)."""
        # No stub context here — proves the guard short-circuits before any
        # cv_bridge / builtin_interfaces import is attempted.
        client = _make_client()
        # _make_client doesn't initialize _grounding_frame_pub; do it ourselves.
        client._grounding_frame_pub = None

        client.publish_grounding_frame(
            image_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
            image_stamp_sec=1.0,
            image_frame_id="frame",
        )


# ------------------------------------------------------------------
# Tests — foxglove_msgs/ImageAnnotations companion publisher
# ------------------------------------------------------------------


def _add_foxglove_msgs_stub(stack: dict[str, Any]) -> None:
    """Inject a foxglove_msgs.msg stub into sys.modules.

    Saves prior bindings into ``stack`` so the caller's finally-block can
    restore them. Mirrors the pattern used by ``_stub_vision_msgs``.
    """
    stack["foxglove_msgs"] = sys.modules.get("foxglove_msgs")
    stack["foxglove_msgs.msg"] = sys.modules.get("foxglove_msgs.msg")

    fg_pkg = types.ModuleType("foxglove_msgs")
    fg_msg = types.ModuleType("foxglove_msgs.msg")

    class _StubPointsAnnotation:
        UNKNOWN = 0
        POINTS = 1
        LINE_LOOP = 2
        LINE_STRIP = 3
        LINE_LIST = 4

        def __init__(self) -> None:
            self.timestamp = None
            self.type = 0
            self.points: list[Any] = []
            self.outline_color = None
            self.outline_colors: list[Any] = []
            self.fill_color = None
            self.thickness = 0.0

    class _StubTextAnnotation:
        def __init__(self) -> None:
            self.timestamp = None
            self.position = None
            self.text = ""
            self.font_size = 0.0
            self.text_color = None
            self.background_color = None

    class _StubImageAnnotations:
        def __init__(self) -> None:
            self.circles: list[Any] = []
            self.points: list[Any] = []
            self.texts: list[Any] = []

    class _StubColor:
        def __init__(self, r=0.0, g=0.0, b=0.0, a=0.0) -> None:
            self.r = r
            self.g = g
            self.b = b
            self.a = a

    class _StubPoint2:
        def __init__(self, x=0.0, y=0.0) -> None:
            self.x = x
            self.y = y

    fg_msg.ImageAnnotations = _StubImageAnnotations
    fg_msg.PointsAnnotation = _StubPointsAnnotation
    fg_msg.TextAnnotation = _StubTextAnnotation
    fg_msg.Color = _StubColor
    fg_msg.Point2 = _StubPoint2
    fg_pkg.msg = fg_msg  # type: ignore[attr-defined]
    sys.modules["foxglove_msgs"] = fg_pkg
    sys.modules["foxglove_msgs.msg"] = fg_msg


class TestPublishDetectionsFG(unittest.TestCase):
    """Verifies publish_detections also emits the Foxglove-native overlay."""

    def test_publishes_line_loop_and_text_annotation(self) -> None:
        stack: dict[str, Any] = {}
        with _stub_vision_msgs():
            _add_foxglove_msgs_stub(stack)
            try:
                client = _make_client()
                vm_pub = MagicMock()
                fg_pub = MagicMock()
                client._detections_pub = vm_pub
                client._detections_fg_pub = fg_pub

                client.publish_detections(
                    image_stamp_sec=100.5,
                    image_frame_id="d555_color_optical_frame",
                    image_width=640,
                    image_height=360,
                    detections=[((100, 200, 500, 600), "door", 0.91)],
                )

                # Both pubs fire once for one detection.
                self.assertEqual(vm_pub.publish.call_count, 1)
                self.assertEqual(fg_pub.publish.call_count, 1)

                anno = fg_pub.publish.call_args[0][0]
                # One LINE_LOOP for the bbox.
                self.assertEqual(len(anno.points), 1)
                pa = anno.points[0]
                self.assertEqual(pa.type, 2)  # LINE_LOOP
                # 4 corners, all at pixel scale (1000-unit → 640x360).
                # x: [100, 500] * 640/1000 = [64, 320]
                # y: [200, 600] * 360/1000 = [72, 216]
                xs = [p.x for p in pa.points]
                ys = [p.y for p in pa.points]
                self.assertEqual(sorted(set(xs)), [64.0, 320.0])
                self.assertEqual(sorted(set(ys)), [72.0, 216.0])
                self.assertEqual(pa.timestamp.sec, 100)
                self.assertEqual(pa.timestamp.nanosec, 500_000_000)

                # One TextAnnotation for the label.
                self.assertEqual(len(anno.texts), 1)
                ta = anno.texts[0]
                self.assertIn("door", ta.text)
                self.assertIn("0.91", ta.text)
            finally:
                for k, v in stack.items():
                    if v is None:
                        sys.modules.pop(k, None)
                    else:
                        sys.modules[k] = v

    def test_no_publish_when_fg_pub_is_none(self) -> None:
        """foxglove_msgs missing at startup → Detection2DArray still goes out."""
        with _stub_vision_msgs():
            client = _make_client()
            vm_pub = MagicMock()
            client._detections_pub = vm_pub
            client._detections_fg_pub = None  # foxglove_msgs absent

            # Should not raise even though no foxglove_msgs stub is registered.
            client.publish_detections(
                image_stamp_sec=1.0,
                image_frame_id="frame",
                image_width=640,
                image_height=360,
                detections=[((0, 0, 1000, 1000), "x", 0.5)],
            )
            self.assertEqual(vm_pub.publish.call_count, 1)

    def test_empty_detections_publishes_empty_annotations(self) -> None:
        """Empty detection list still publishes an empty ImageAnnotations.

        That's the clear-overlay path — without an empty publish here,
        Foxglove keeps drawing the last bbox after a mission ends or a
        scan heading rejects all candidates.
        """
        stack: dict[str, Any] = {}
        with _stub_vision_msgs():
            _add_foxglove_msgs_stub(stack)
            try:
                client = _make_client()
                client._detections_pub = MagicMock()
                fg_pub = MagicMock()
                client._detections_fg_pub = fg_pub

                client.publish_detections(
                    image_stamp_sec=0.0,
                    image_frame_id="frame",
                    image_width=640,
                    image_height=360,
                    detections=[],
                )
                self.assertEqual(fg_pub.publish.call_count, 1)
                anno = fg_pub.publish.call_args[0][0]
                self.assertEqual(anno.points, [])
                self.assertEqual(anno.texts, [])
            finally:
                for k, v in stack.items():
                    if v is None:
                        sys.modules.pop(k, None)
                    else:
                        sys.modules[k] = v


# ------------------------------------------------------------------
# Tests — Phase 4: navigate_to_pose backend dispatch
# ------------------------------------------------------------------


class TestResolveExecutionBackend(unittest.TestCase):
    """`_resolve_execution_backend` resolves per-step > env var > default
    and falls back to nav2 on unknown values rather than failing the
    mission silently.
    """

    def test_per_step_takes_precedence_over_env(self) -> None:
        from strafer_autonomy.clients.ros_client import _resolve_execution_backend

        with patch.dict(
            "os.environ",
            {"STRAFER_NAV_BACKEND": "strafer_direct"},
            clear=False,
        ):
            self.assertEqual(
                _resolve_execution_backend("nav2"), "nav2",
            )

    def test_env_var_picked_up_when_per_step_none(self) -> None:
        from strafer_autonomy.clients.ros_client import _resolve_execution_backend

        with patch.dict(
            "os.environ",
            {"STRAFER_NAV_BACKEND": "strafer_direct"},
            clear=False,
        ):
            self.assertEqual(
                _resolve_execution_backend(None), "strafer_direct",
            )

    def test_unset_env_var_defaults_to_nav2(self) -> None:
        """Real-robot bringup leaves the env var unset and must keep
        nav2 unchanged — the brief's compatibility anchor."""
        from strafer_autonomy.clients.ros_client import _resolve_execution_backend

        env = {
            k: v for k, v in os.environ.items()
            if k != "STRAFER_NAV_BACKEND"
        }
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(_resolve_execution_backend(None), "nav2")

    def test_unknown_value_falls_back_to_nav2(self) -> None:
        """A typo or unsupported backend name must NOT silently fail; it
        falls back to nav2 with a logged error."""
        from strafer_autonomy.clients.ros_client import _resolve_execution_backend

        with patch.dict(
            "os.environ",
            {"STRAFER_NAV_BACKEND": "mppi_local_only"},
            clear=False,
        ):
            self.assertEqual(_resolve_execution_backend(None), "nav2")

    def test_hybrid_backend_is_recognized(self) -> None:
        """hybrid_nav2_strafer is a supported backend as of the hybrid
        runtime; it resolves to itself rather than falling back to nav2."""
        from strafer_autonomy.clients.ros_client import _resolve_execution_backend

        with patch.dict(
            "os.environ",
            {"STRAFER_NAV_BACKEND": "hybrid_nav2_strafer"},
            clear=False,
        ):
            self.assertEqual(
                _resolve_execution_backend(None), "hybrid_nav2_strafer",
            )

    def test_unknown_per_step_falls_back(self) -> None:
        from strafer_autonomy.clients.ros_client import _resolve_execution_backend

        self.assertEqual(
            _resolve_execution_backend("strafer-direct"), "nav2",
        )


class TestNavigateToPoseDispatch(unittest.TestCase):
    """The public navigate_to_pose dispatches between Nav2 and
    strafer_direct backends. With strafer_direct selected and the
    inference action server unavailable, the call falls back to nav2
    *for this mission* per the brief's per-mission fallback rule.
    """

    def _patch_clean_env(self):
        env = {
            k: v for k, v in os.environ.items()
            if k != "STRAFER_NAV_BACKEND"
        }
        return patch.dict("os.environ", env, clear=True)

    def test_unset_env_routes_to_nav2(self) -> None:
        with self._patch_clean_env():
            client = _make_client()
            nav2_called: dict[str, bool] = {"called": False}
            direct_called: dict[str, bool] = {"called": False}

            def fake_nav2(**kw):
                nav2_called["called"] = True
                return SkillResult(
                    step_id=kw["step_id"], skill="navigate_to_pose",
                    status="succeeded",
                    started_at=0.0, finished_at=0.0,
                )

            def fake_direct(**kw):
                direct_called["called"] = True
                return None

            client._navigate_via_nav2 = fake_nav2  # type: ignore
            client._navigate_via_strafer_direct = fake_direct  # type: ignore

            result = client.navigate_to_pose(
                step_id="s1",
                goal_pose=Pose3D(x=1.0, y=0.0, z=0.0, qx=0, qy=0, qz=0, qw=1),
            )
            self.assertTrue(nav2_called["called"])
            self.assertFalse(direct_called["called"])
            self.assertEqual(result.status, "succeeded")

    def test_strafer_direct_routes_to_inference_then_returns(self) -> None:
        with patch.dict(
            "os.environ",
            {"STRAFER_NAV_BACKEND": "strafer_direct"},
            clear=False,
        ):
            client = _make_client()

            def fake_direct(**kw):
                return SkillResult(
                    step_id=kw["step_id"], skill="navigate_to_pose",
                    status="succeeded",
                    message="strafer_direct navigation completed successfully.",
                    started_at=0.0, finished_at=0.0,
                )

            def fake_nav2(**kw):
                raise AssertionError(
                    "nav2 path must not be called when strafer_direct succeeds"
                )

            client._navigate_via_strafer_direct = fake_direct  # type: ignore
            client._navigate_via_nav2 = fake_nav2  # type: ignore

            result = client.navigate_to_pose(
                step_id="s2",
                goal_pose=Pose3D(x=1.0, y=0.0, z=0.0, qx=0, qy=0, qz=0, qw=1),
            )
            self.assertEqual(result.status, "succeeded")
            self.assertIn("strafer_direct", result.message)

    def test_strafer_direct_unavailable_falls_back_to_nav2(self) -> None:
        """Inference action server unavailable (None return) →
        per-mission fallback to nav2 per the brief."""
        with patch.dict(
            "os.environ",
            {"STRAFER_NAV_BACKEND": "strafer_direct"},
            clear=False,
        ):
            client = _make_client()

            def fake_direct(**kw):
                return None  # action server unavailable

            def fake_nav2(**kw):
                return SkillResult(
                    step_id=kw["step_id"], skill="navigate_to_pose",
                    status="succeeded",
                    message="Navigation completed successfully.",
                    started_at=0.0, finished_at=0.0,
                )

            client._navigate_via_strafer_direct = fake_direct  # type: ignore
            client._navigate_via_nav2 = fake_nav2  # type: ignore

            result = client.navigate_to_pose(
                step_id="s3",
                goal_pose=Pose3D(x=1.0, y=0.0, z=0.0, qx=0, qy=0, qz=0, qw=1),
            )
            self.assertEqual(result.status, "succeeded")
            self.assertNotIn("strafer_direct", result.message)

    def test_per_step_argument_overrides_env_var(self) -> None:
        with patch.dict(
            "os.environ",
            {"STRAFER_NAV_BACKEND": "strafer_direct"},
            clear=False,
        ):
            client = _make_client()
            nav2_called: dict[str, bool] = {"called": False}
            direct_called: dict[str, bool] = {"called": False}

            def fake_nav2(**kw):
                nav2_called["called"] = True
                return SkillResult(
                    step_id=kw["step_id"], skill="navigate_to_pose",
                    status="succeeded",
                    started_at=0.0, finished_at=0.0,
                )

            def fake_direct(**kw):
                direct_called["called"] = True
                return SkillResult(
                    step_id=kw["step_id"], skill="navigate_to_pose",
                    status="succeeded",
                    started_at=0.0, finished_at=0.0,
                )

            client._navigate_via_nav2 = fake_nav2  # type: ignore
            client._navigate_via_strafer_direct = fake_direct  # type: ignore

            client.navigate_to_pose(
                step_id="s4",
                goal_pose=Pose3D(x=1.0, y=0.0, z=0.0, qx=0, qy=0, qz=0, qw=1),
                execution_backend="nav2",
            )
            self.assertTrue(nav2_called["called"])
            self.assertFalse(direct_called["called"])

    def test_unknown_backend_falls_back_to_nav2(self) -> None:
        with self._patch_clean_env():
            client = _make_client()
            nav2_called: dict[str, bool] = {"called": False}

            def fake_nav2(**kw):
                nav2_called["called"] = True
                return SkillResult(
                    step_id=kw["step_id"], skill="navigate_to_pose",
                    status="succeeded",
                    started_at=0.0, finished_at=0.0,
                )

            client._navigate_via_nav2 = fake_nav2  # type: ignore
            client._navigate_via_strafer_direct = (  # type: ignore
                lambda **kw: (_ for _ in ()).throw(
                    AssertionError("strafer_direct must not run for unknown backends")
                )
            )
            client._navigate_via_hybrid = (  # type: ignore
                lambda **kw: (_ for _ in ()).throw(
                    AssertionError("hybrid must not run for unknown backends")
                )
            )

            client.navigate_to_pose(
                step_id="s5",
                goal_pose=Pose3D(x=1.0, y=0.0, z=0.0, qx=0, qy=0, qz=0, qw=1),
                execution_backend="teleop_only",
            )
            self.assertTrue(nav2_called["called"])

    def test_hybrid_routes_to_hybrid_then_returns(self) -> None:
        with patch.dict(
            "os.environ",
            {"STRAFER_NAV_BACKEND": "hybrid_nav2_strafer"},
            clear=False,
        ):
            client = _make_client()

            def fake_hybrid(**kw):
                return SkillResult(
                    step_id=kw["step_id"], skill="navigate_to_pose",
                    status="succeeded",
                    message="hybrid navigation completed successfully.",
                    started_at=0.0, finished_at=0.0,
                )

            def fake_nav2(**kw):
                raise AssertionError(
                    "nav2 path must not be called when hybrid succeeds"
                )

            client._navigate_via_hybrid = fake_hybrid  # type: ignore
            client._navigate_via_nav2 = fake_nav2  # type: ignore

            result = client.navigate_to_pose(
                step_id="h1",
                goal_pose=Pose3D(x=1.0, y=0.0, z=0.0, qx=0, qy=0, qz=0, qw=1),
            )
            self.assertEqual(result.status, "succeeded")
            self.assertIn("hybrid", result.message)

    def test_hybrid_unavailable_falls_back_to_nav2(self) -> None:
        """Inference server (or Nav2 planner) unavailable (None return) →
        per-mission fallback to nav2, same rule as strafer_direct."""
        with patch.dict(
            "os.environ",
            {"STRAFER_NAV_BACKEND": "hybrid_nav2_strafer"},
            clear=False,
        ):
            client = _make_client()

            def fake_hybrid(**kw):
                return None  # action server / planner unavailable

            def fake_nav2(**kw):
                return SkillResult(
                    step_id=kw["step_id"], skill="navigate_to_pose",
                    status="succeeded",
                    message="Navigation completed successfully.",
                    started_at=0.0, finished_at=0.0,
                )

            client._navigate_via_hybrid = fake_hybrid  # type: ignore
            client._navigate_via_nav2 = fake_nav2  # type: ignore

            result = client.navigate_to_pose(
                step_id="h2",
                goal_pose=Pose3D(x=1.0, y=0.0, z=0.0, qx=0, qy=0, qz=0, qw=1),
            )
            self.assertEqual(result.status, "succeeded")
            self.assertNotIn("hybrid", result.message)


class TestNavigateViaHybridInternals(unittest.TestCase):
    """Direct coverage of _navigate_via_hybrid: the two-stage server
    fallback and the planner-only trigger contract (dispatch-level routing
    is covered in TestNavigateToPoseDispatch).
    """

    def _goal(self) -> Pose3D:
        return Pose3D(x=1.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)

    def test_planner_unavailable_returns_none_and_sends_no_inference_goal(self):
        client = _make_client()
        direct = MagicMock()
        direct.wait_for_server.return_value = True   # inference up
        planner = MagicMock()
        planner.wait_for_server.return_value = False  # planner action down
        client._strafer_direct_client = direct
        client._planner_client = planner

        result = client._navigate_via_hybrid(
            step_id="hi", goal_pose=self._goal(),
            behavior_tree=None, timeout_s=5.0,
        )
        # None => caller falls back to nav2; and no orphaned inference goal.
        self.assertIsNone(result)
        direct.send_goal_async.assert_not_called()

    def test_initial_replan_uses_gridbased_planner_only(self):
        from action_msgs.msg import GoalStatus
        from builtin_interfaces.msg import Time as TimeMsg

        client = _make_client()
        # Real Time for the PoseStamped stamp (rosidl validates the field).
        client._node.get_clock.return_value.now.return_value.to_msg.return_value = (
            TimeMsg()
        )

        direct = MagicMock()
        direct.wait_for_server.return_value = True
        planner = MagicMock()
        planner.wait_for_server.return_value = True
        client._strafer_direct_client = direct
        client._planner_client = planner

        send_future = MagicMock()
        goal_handle = MagicMock()
        goal_handle.accepted = True
        send_future.result.return_value = goal_handle
        direct.send_goal_async.return_value = send_future
        result_future = MagicMock()
        result_future.result.return_value = MagicMock(
            status=GoalStatus.STATUS_SUCCEEDED
        )
        goal_handle.get_result_async.return_value = result_future

        # Skip the real clock-driven wait loop.
        client._wait_for_future = lambda *a, **k: True  # type: ignore
        client._wait_for_nav_result = lambda *a, **k: (True, False)  # type: ignore

        result = client._navigate_via_hybrid(
            step_id="hi", goal_pose=self._goal(),
            behavior_tree=None, timeout_s=5.0,
        )
        self.assertEqual(result.status, "succeeded")
        # The planner was triggered planner-only with the GridBased planner and
        # the robot's current pose (use_start False); the controller server is
        # never engaged (no controller client is constructed or called).
        self.assertTrue(planner.send_goal_async.called)
        plan_goal = planner.send_goal_async.call_args[0][0]
        self.assertEqual(plan_goal.planner_id, "GridBased")
        self.assertFalse(plan_goal.use_start)
