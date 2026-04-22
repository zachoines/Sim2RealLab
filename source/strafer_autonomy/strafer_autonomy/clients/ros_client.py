"""Jetson-local ROS client abstractions and implementations."""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from strafer_autonomy.schemas import GoalPoseCandidate, Pose3D, SceneObservation, SkillResult

logger = logging.getLogger(__name__)


@runtime_checkable
class RosClient(Protocol):
    """Executor-facing interface for robot-local observation and skill execution."""

    def capture_scene_observation(self) -> SceneObservation:
        """Return the latest synchronized robot observation bundle."""

    def get_robot_state(self) -> dict[str, Any]:
        """Return the latest robot state snapshot for planning and status."""

    def get_map_pose(self) -> dict[str, float] | None:
        """Return the robot's current pose in the map frame, or ``None``
        if the ``map -> base_link`` transform is not yet available."""

    def project_detection_to_goal_pose(
        self,
        *,
        request_id: str,
        image_stamp_sec: float,
        bbox_2d: tuple[int, int, int, int],
        standoff_m: float,
        target_label: str | None = None,
    ) -> GoalPoseCandidate:
        """Project a 2D detection into a reachable robot goal pose."""

    def navigate_to_pose(
        self,
        *,
        step_id: str,
        goal_pose: Pose3D,
        execution_backend: str = "nav2",
        behavior_tree: str | None = None,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Execute goal-directed motion through the selected local backend."""

    def cancel_active_navigation(self) -> bool:
        """Cancel the currently active motion backend if one exists."""

    def rotate_in_place(
        self,
        *,
        step_id: str,
        yaw_delta_rad: float,
        tolerance_rad: float = 0.1,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Rotate the robot in place by the given yaw delta."""

    def orient_relative_to_target(
        self,
        *,
        step_id: str,
        target_pose: Pose3D,
        mode: str,
        yaw_offset_rad: float = 0.0,
        tolerance_rad: float = 0.1,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Rotate the robot relative to a projected target pose."""


@dataclass(frozen=True)
class RosClientConfig:
    """Jetson-local runtime settings for the ROS adapter."""

    observation_max_age_s: float = 0.5
    default_goal_frame: str = "map"
    default_nav_timeout_s: float = 90.0
    default_navigation_backend: str = "nav2"
    default_orientation_tolerance_rad: float = 0.1
    default_rotate_speed_rad_s: float = 0.5
    default_rotate_timeout_s: float = 15.0


class JetsonRosClient:
    """ROS2 adapter for the Jetson-resident mission executor.

    Creates an internal ROS2 node that subscribes to camera, depth,
    odometry, and TF topics.  A background ``SingleThreadedExecutor``
    processes callbacks so cached sensor data stays fresh.

    All ROS imports are deferred to ``__init__`` and method bodies so
    the module can be imported safely in non-ROS environments (unit tests
    for planner / VLM code that mock the entire client).
    """

    # D555 perception topics (timestamp-fixed by strafer_perception)
    TOPIC_COLOR = "/d555/color/image_sync"
    TOPIC_DEPTH = "/d555/aligned_depth_to_color/image_sync"
    TOPIC_CAM_INFO = "/d555/color/camera_info_sync"
    TOPIC_ODOM = "/strafer/odom"

    def __init__(self, config: RosClientConfig | None = None) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        self._config = config or RosClientConfig()

        if not rclpy.ok():
            rclpy.init()

        self._node: Node = Node("jetson_ros_client")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        # Thread-safe sensor cache. The `*_rx_t` fields record the wall-
        # clock time (monotonic) when each message was received, so
        # freshness checks do not rely on header stamps — those may be
        # in sim-time when the bridge is upstream, and comparing them to
        # the node's system clock yields nonsense ages.
        self._cache_lock = threading.Lock()
        self._latest_color = None
        self._latest_color_rx_t: float | None = None
        self._latest_depth = None
        self._latest_cam_info = None
        self._latest_odom = None
        self._latest_costmap = None
        self._tf_buffer: Any = None
        self._tf_listener: Any = None

        # Nav2 goal tracking
        self._nav_lock = threading.Lock()
        self._active_goal_handle = None

        self._setup_subscriptions()

        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name="ros-client-spin",
        )
        self._spin_thread.start()
        self._node.get_logger().info("JetsonRosClient ready.")

    @property
    def config(self) -> RosClientConfig:
        """Return the immutable ROS client configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _setup_subscriptions(self) -> None:
        from nav_msgs.msg import OccupancyGrid, Odometry
        from sensor_msgs.msg import CameraInfo, Image

        self._node.create_subscription(Image, self.TOPIC_COLOR, self._on_color, 10)
        self._node.create_subscription(Image, self.TOPIC_DEPTH, self._on_depth, 10)
        self._node.create_subscription(CameraInfo, self.TOPIC_CAM_INFO, self._on_cam_info, 10)
        self._node.create_subscription(Odometry, self.TOPIC_ODOM, self._on_odom, 10)
        self._node.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self._on_costmap, 1,
        )

        # TF buffer for SLAM tracking freshness checks
        try:
            import tf2_ros
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        except Exception:
            logger.debug("tf2_ros not available; SLAM tracking checks disabled")

    def _on_color(self, msg: Any) -> None:
        with self._cache_lock:
            self._latest_color = msg
            self._latest_color_rx_t = time.monotonic()

    def _on_depth(self, msg: Any) -> None:
        with self._cache_lock:
            self._latest_depth = msg

    def _on_cam_info(self, msg: Any) -> None:
        with self._cache_lock:
            self._latest_cam_info = msg

    def _on_odom(self, msg: Any) -> None:
        with self._cache_lock:
            self._latest_odom = msg

    def _on_costmap(self, msg: Any) -> None:
        with self._cache_lock:
            self._latest_costmap = msg

    # ------------------------------------------------------------------
    # Safety pre-checks
    # ------------------------------------------------------------------

    def check_costmap_at_pose(self, x: float, y: float) -> str:
        """Check the Nav2 global costmap cell at the given world (x, y).

        Returns one of ``"free"``, ``"occupied"``, ``"unknown"``, or
        ``"no_costmap"`` if no costmap has been received yet.
        """
        with self._cache_lock:
            costmap = self._latest_costmap
        if costmap is None:
            return "no_costmap"

        info = costmap.info
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)
        if col < 0 or col >= info.width or row < 0 or row >= info.height:
            return "unknown"
        cell = costmap.data[row * info.width + col]
        if cell == -1:
            return "unknown"
        if cell >= 65:
            return "occupied"
        return "free"

    def get_map_pose(self) -> dict[str, float] | None:
        """Look up ``map -> base_link`` and return the pose dict.

        Returns ``None`` if the transform is not yet available (e.g.
        SLAM has not produced the map frame yet).
        """
        if self._tf_buffer is None:
            return None
        try:
            from rclpy.time import Time
            tf = self._tf_buffer.lookup_transform("map", "base_link", Time())
        except Exception:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        return {
            "x": t.x, "y": t.y, "z": t.z,
            "qx": r.x, "qy": r.y, "qz": r.z, "qw": r.w,
        }

    def check_slam_tracking(self, threshold_s: float = 5.0) -> tuple[bool, float]:
        """Check ``map -> odom`` TF freshness.

        Returns ``(is_fresh, age_s)``. If TF is unavailable, returns
        ``(False, float('inf'))``.
        """
        if self._tf_buffer is None:
            return False, float("inf")
        try:
            from rclpy.time import Time
            transform = self._tf_buffer.lookup_transform(
                "map", "odom", Time(),
            )
            tf_stamp_sec = self._stamp_to_sec(transform.header.stamp)
            now = self._stamp_to_sec(self._node.get_clock().now().to_msg())
            age = now - tf_stamp_sec
            return age < threshold_s, age
        except Exception:
            return False, float("inf")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stamp_to_sec(stamp: Any) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def _yaw_from_quaternion(q: Any) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ------------------------------------------------------------------
    # Task 1: capture_scene_observation
    # ------------------------------------------------------------------

    def capture_scene_observation(self) -> SceneObservation:
        """Read the latest cached RGB, depth, camera info, and robot pose data.

        Subscribes to the timestamp-fixed D555 topics and ``/strafer/odom``.
        Raises ``RuntimeError`` if no frames are cached or if the cached
        frames are older than ``observation_max_age_s``.
        """
        from cv_bridge import CvBridge
        import numpy as np

        with self._cache_lock:
            color_msg = self._latest_color
            color_rx_t = self._latest_color_rx_t
            depth_msg = self._latest_depth
            cam_info_msg = self._latest_cam_info
            odom_msg = self._latest_odom

        if color_msg is None or depth_msg is None:
            raise RuntimeError(
                "No color or depth frames cached. Is the perception stack running?"
            )

        # Measure receive-to-now on the wall clock, not on header stamps:
        # header stamps can be sim-time (upstream bridge) while the node
        # clock is wall time, which previously produced absurd ages.
        age = time.monotonic() - (color_rx_t if color_rx_t is not None else 0.0)
        if color_rx_t is None or age > self._config.observation_max_age_s:
            raise RuntimeError(
                f"Cached color frame is {age:.1f}s old "
                f"(max {self._config.observation_max_age_s}s). "
                "The perception stack may have stopped."
            )

        color_stamp = self._stamp_to_sec(color_msg.header.stamp)

        bridge = CvBridge()
        color_bgr = bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")

        # Aligned depth arrives as 16UC1 in millimeters; convert to float32 meters
        depth_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        aligned_depth_m = depth_raw.astype(np.float32) * 0.001

        camera_info: dict[str, Any] = {}
        if cam_info_msg is not None:
            camera_info = {
                "height": cam_info_msg.height,
                "width": cam_info_msg.width,
                "k": list(cam_info_msg.k),
                "d": list(cam_info_msg.d),
                "p": list(cam_info_msg.p),
                "distortion_model": cam_info_msg.distortion_model,
            }

        robot_pose_map: dict[str, Any] | None = None
        if odom_msg is not None:
            p = odom_msg.pose.pose
            robot_pose_map = {
                "x": p.position.x,
                "y": p.position.y,
                "z": p.position.z,
                "qx": p.orientation.x,
                "qy": p.orientation.y,
                "qz": p.orientation.z,
                "qw": p.orientation.w,
                "frame_id": odom_msg.header.frame_id,
            }

        return SceneObservation(
            observation_id=uuid.uuid4().hex[:12],
            stamp_sec=color_stamp,
            color_image_bgr=color_bgr,
            aligned_depth_m=aligned_depth_m,
            camera_frame=color_msg.header.frame_id or "d555_color_optical_frame",
            camera_info=camera_info,
            robot_pose_map=robot_pose_map,
            tf_snapshot_ready=odom_msg is not None,
        )

    # ------------------------------------------------------------------
    # Task 3: navigate_to_pose
    # ------------------------------------------------------------------

    def navigate_to_pose(
        self,
        *,
        step_id: str,
        goal_pose: Pose3D,
        execution_backend: str = "nav2",
        behavior_tree: str | None = None,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Send a goal to Nav2 and block until completion, timeout, or cancel.

        Creates a ``NavigateToPose`` action client on first call and
        reuses it for subsequent goals.  The active goal handle is tracked
        so ``cancel_active_navigation`` can abort it.
        """
        from nav2_msgs.action import NavigateToPose as Nav2NavigateToPose
        from geometry_msgs.msg import PoseStamped
        from rclpy.action import ActionClient
        from action_msgs.msg import GoalStatus

        started_at = time.time()
        timeout = timeout_s or self._config.default_nav_timeout_s

        if not hasattr(self, "_nav2_client"):
            self._nav2_client = ActionClient(
                self._node, Nav2NavigateToPose, "navigate_to_pose"
            )

        if not self._nav2_client.wait_for_server(timeout_sec=10.0):
            return SkillResult(
                step_id=step_id, skill="navigate_to_pose", status="failed",
                error_code="nav2_unavailable",
                message="Nav2 action server not available within 10 s.",
                started_at=started_at, finished_at=time.time(),
            )

        goal_msg = Nav2NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self._config.default_goal_frame
        goal_msg.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_pose.x
        goal_msg.pose.pose.position.y = goal_pose.y
        goal_msg.pose.pose.position.z = goal_pose.z
        goal_msg.pose.pose.orientation.x = goal_pose.qx
        goal_msg.pose.pose.orientation.y = goal_pose.qy
        goal_msg.pose.pose.orientation.z = goal_pose.qz
        goal_msg.pose.pose.orientation.w = goal_pose.qw
        if behavior_tree:
            goal_msg.behavior_tree = behavior_tree

        logger.info(
            "Sending Nav2 goal: frame=%s pos=(%.3f, %.3f, %.3f) "
            "orient=(%.3f, %.3f, %.3f, %.3f)",
            goal_msg.pose.header.frame_id,
            goal_pose.x, goal_pose.y, goal_pose.z,
            goal_pose.qx, goal_pose.qy, goal_pose.qz, goal_pose.qw,
        )

        send_future = self._nav2_client.send_goal_async(goal_msg)
        if not self._wait_for_future(send_future, 10.0):
            return SkillResult(
                step_id=step_id, skill="navigate_to_pose", status="failed",
                error_code="goal_send_timeout",
                message="Timed out sending goal to Nav2.",
                started_at=started_at, finished_at=time.time(),
            )

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            logger.error(
                "Nav2 REJECTED goal: pos=(%.3f, %.3f, %.3f)",
                goal_pose.x, goal_pose.y, goal_pose.z,
            )
            return SkillResult(
                step_id=step_id, skill="navigate_to_pose", status="failed",
                error_code="goal_rejected",
                message="Nav2 rejected the navigation goal.",
                started_at=started_at, finished_at=time.time(),
            )

        with self._nav_lock:
            self._active_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        timed_out = not self._wait_for_future(result_future, timeout)

        with self._nav_lock:
            self._active_goal_handle = None

        if timed_out:
            goal_handle.cancel_goal_async()
            return SkillResult(
                step_id=step_id, skill="navigate_to_pose", status="timeout",
                error_code="navigation_timeout",
                message=f"Navigation timed out after {timeout:.0f}s.",
                started_at=started_at, finished_at=time.time(),
            )

        status = result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            return SkillResult(
                step_id=step_id, skill="navigate_to_pose", status="succeeded",
                message="Navigation completed successfully.",
                started_at=started_at, finished_at=time.time(),
            )
        if status == GoalStatus.STATUS_CANCELED:
            return SkillResult(
                step_id=step_id, skill="navigate_to_pose", status="canceled",
                error_code="navigation_canceled",
                message="Navigation was canceled.",
                started_at=started_at, finished_at=time.time(),
            )
        return SkillResult(
            step_id=step_id, skill="navigate_to_pose", status="failed",
            error_code="navigation_failed",
            message=f"Navigation failed (GoalStatus {status}).",
            started_at=started_at, finished_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Task 4: project_detection_to_goal_pose (service client)
    # ------------------------------------------------------------------

    def project_detection_to_goal_pose(
        self,
        *,
        request_id: str,
        image_stamp_sec: float,
        bbox_2d: tuple[int, int, int, int],
        standoff_m: float,
        target_label: str | None = None,
    ) -> GoalPoseCandidate:
        """Call the robot-local ``ProjectDetectionToGoalPose`` service.

        Translates the ROS service response into a ``GoalPoseCandidate``
        for the mission executor.
        """
        from strafer_msgs.srv import ProjectDetectionToGoalPose

        if not hasattr(self, "_projection_client"):
            self._projection_client = self._node.create_client(
                ProjectDetectionToGoalPose,
                "/strafer/project_detection_to_goal_pose",
            )

        if not self._projection_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("ProjectDetectionToGoalPose service not available.")

        req = ProjectDetectionToGoalPose.Request()
        req.bbox_normalized_1000 = [float(v) for v in bbox_2d]
        req.image_stamp_sec = image_stamp_sec
        req.standoff_m = standoff_m
        req.target_label = target_label or ""

        future = self._projection_client.call_async(req)
        if not self._wait_for_future(future, 10.0):
            raise RuntimeError("ProjectDetectionToGoalPose service call timed out.")

        resp = future.result()

        goal_pose: Pose3D | None = None
        target_pose: Pose3D | None = None
        if resp.found:
            gp = resp.goal_pose.pose
            goal_pose = Pose3D(
                x=gp.position.x, y=gp.position.y, z=gp.position.z,
                qx=gp.orientation.x, qy=gp.orientation.y,
                qz=gp.orientation.z, qw=gp.orientation.w,
            )
            tp = resp.target_pose.pose
            target_pose = Pose3D(
                x=tp.position.x, y=tp.position.y, z=tp.position.z,
                qx=tp.orientation.x, qy=tp.orientation.y,
                qz=tp.orientation.z, qw=tp.orientation.w,
            )

        return GoalPoseCandidate(
            request_id=request_id,
            found=resp.found,
            goal_frame=resp.goal_pose.header.frame_id or self._config.default_goal_frame,
            goal_pose=goal_pose,
            target_pose=target_pose,
            standoff_m=standoff_m,
            depth_valid=resp.depth_valid,
            quality_flags=tuple(resp.quality_flags),
            message=resp.message,
        )

    # ------------------------------------------------------------------
    # Task 5: cancel_active_navigation
    # ------------------------------------------------------------------

    def cancel_active_navigation(self) -> bool:
        """Cancel the active Nav2 goal, if one is in progress.

        Returns ``True`` if a goal was canceled, ``False`` if nothing
        was active.
        """
        with self._nav_lock:
            goal_handle = self._active_goal_handle

        if goal_handle is None:
            return False

        cancel_future = goal_handle.cancel_goal_async()
        if not self._wait_for_future(cancel_future, 5.0):
            logger.warning("Timed out waiting for Nav2 cancel confirmation.")
            return False

        with self._nav_lock:
            self._active_goal_handle = None

        return True

    # ------------------------------------------------------------------
    # Task 6: get_robot_state
    # ------------------------------------------------------------------

    def get_robot_state(self) -> dict[str, Any]:
        """Compose the latest robot state from odom and navigation status.

        Returns a dict suitable for the planner's ``robot_state`` field
        and the ``report_status`` skill output.
        """
        with self._cache_lock:
            odom_msg = self._latest_odom

        state: dict[str, Any] = {"timestamp": time.time()}

        if odom_msg is not None:
            p = odom_msg.pose.pose
            t = odom_msg.twist.twist
            state["pose"] = {
                "x": p.position.x,
                "y": p.position.y,
                "z": p.position.z,
                "qx": p.orientation.x,
                "qy": p.orientation.y,
                "qz": p.orientation.z,
                "qw": p.orientation.w,
            }
            state["velocity"] = {
                "linear_x": t.linear.x,
                "linear_y": t.linear.y,
                "angular_z": t.angular.z,
            }
            state["odom_frame"] = odom_msg.header.frame_id
        else:
            state["pose"] = None
            state["velocity"] = None

        with self._nav_lock:
            state["navigation_active"] = self._active_goal_handle is not None

        return state

    # ------------------------------------------------------------------
    # rotate_in_place (needed by scan_for_target)
    # ------------------------------------------------------------------

    def rotate_in_place(
        self,
        *,
        step_id: str,
        yaw_delta_rad: float,
        tolerance_rad: float = 0.1,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Rotate in place by publishing ``cmd_vel`` angular-Z.

        Reads current yaw from odometry, computes a target yaw, and
        publishes angular velocity until the error is within tolerance
        or the timeout expires.
        """
        from geometry_msgs.msg import Twist

        started_at = time.time()
        timeout = timeout_s or self._config.default_rotate_timeout_s

        if not hasattr(self, "_cmd_vel_pub"):
            self._cmd_vel_pub = self._node.create_publisher(
                Twist, "/cmd_vel", 10
            )

        with self._cache_lock:
            odom_msg = self._latest_odom

        if odom_msg is None:
            return SkillResult(
                step_id=step_id, skill="rotate_in_place", status="failed",
                error_code="no_odom",
                message="No odometry available for rotation.",
                started_at=started_at, finished_at=time.time(),
            )

        initial_yaw = self._yaw_from_quaternion(odom_msg.pose.pose.orientation)
        target_yaw = self._normalize_angle(initial_yaw + yaw_delta_rad)
        speed = self._config.default_rotate_speed_rad_s
        direction = 1.0 if yaw_delta_rad >= 0 else -1.0

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cache_lock:
                odom_msg = self._latest_odom
            if odom_msg is None:
                time.sleep(0.02)
                continue

            current_yaw = self._yaw_from_quaternion(odom_msg.pose.pose.orientation)
            error = self._normalize_angle(target_yaw - current_yaw)

            if abs(error) < tolerance_rad:
                self._cmd_vel_pub.publish(Twist())  # stop
                return SkillResult(
                    step_id=step_id, skill="rotate_in_place", status="succeeded",
                    outputs={"final_yaw_error_rad": abs(error)},
                    message="Rotation completed.",
                    started_at=started_at, finished_at=time.time(),
                )

            twist = Twist()
            twist.angular.z = direction * speed
            self._cmd_vel_pub.publish(twist)
            time.sleep(0.02)

        self._cmd_vel_pub.publish(Twist())  # stop
        return SkillResult(
            step_id=step_id, skill="rotate_in_place", status="timeout",
            error_code="rotation_timeout",
            message=f"Rotation timed out after {timeout:.0f}s.",
            started_at=started_at, finished_at=time.time(),
        )

    # ------------------------------------------------------------------
    # orient_relative_to_target (deferred from MVP)
    # ------------------------------------------------------------------

    def orient_relative_to_target(
        self,
        *,
        step_id: str,
        target_pose: Pose3D,
        mode: str,
        yaw_offset_rad: float = 0.0,
        tolerance_rad: float = 0.1,
        timeout_s: float | None = None,
    ) -> SkillResult:
        """Execute the future robot-specific orientation behavior."""

        raise NotImplementedError(
            "Target-relative orientation is not implemented yet. "
            "Add a Strafer-specific action once the navigation and projection path is working."
        )

    # ------------------------------------------------------------------
    # Internal: future waiting
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_for_future(future: Any, timeout_s: float) -> bool:
        """Block until *future* completes or *timeout_s* elapses.

        Returns ``True`` if the future completed, ``False`` on timeout.
        Uses a :class:`threading.Event` signalled by the future's
        done-callback so the calling thread sleeps efficiently instead
        of polling.
        """
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        return event.wait(timeout=timeout_s)
