"""Unit tests for JetsonRosClient — no ROS runtime needed.

Tests bypass __init__ (which starts rclpy) by constructing the object
with ``object.__new__`` and manually wiring the internal state.  ROS
message objects are replaced with lightweight MagicMocks.
"""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strafer_autonomy.clients.ros_client import JetsonRosClient, RosClientConfig
from strafer_autonomy.schemas import GoalPoseCandidate, Pose3D, SceneObservation, SkillResult

pytestmark = pytest.mark.requires_ros


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_client(config: RosClientConfig | None = None) -> JetsonRosClient:
    """Create a JetsonRosClient without starting rclpy."""
    from builtin_interfaces.msg import Time

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
    # Mock the ROS node — use a real Time msg so PoseStamped assignment works
    real_time = Time(sec=100, nanosec=400_000_000)
    clock_now = MagicMock()
    clock_now.to_msg.return_value = real_time
    client._node = MagicMock()
    client._node.get_clock.return_value.now.return_value = clock_now
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
# Tests — _wait_for_future
# ------------------------------------------------------------------


class TestWaitForFuture(unittest.TestCase):

    def test_immediate_done(self) -> None:
        future = _make_future(done=True)
        self.assertTrue(JetsonRosClient._wait_for_future(future, 1.0))

    def test_timeout(self) -> None:
        future = _make_future(done=False)
        # Use a very short timeout
        self.assertFalse(JetsonRosClient._wait_for_future(future, 0.05))
