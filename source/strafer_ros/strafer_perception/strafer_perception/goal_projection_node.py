"""ROS2 service node: project a 2D VLM detection into a 3D goal pose.

Pipeline:
    bbox (Qwen normalized 0-1000) → pixel coords (camera intrinsics) →
    depth lookup at bbox center → 3D point in camera frame →
    TF2 transform to map frame → standoff offset → GoalPoseCandidate

Subscribes to:
    /d555/aligned_depth_to_color/image_sync  — 16UC1 depth in mm
    /d555/color/camera_info_sync             — camera intrinsics

Provides service:
    /strafer/project_detection_to_goal_pose  — ProjectDetectionToGoalPose.srv
"""

from __future__ import annotations

import math

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
from strafer_msgs.srv import ProjectDetectionToGoalPose

# Depth median kernel half-size (pixels).  A 5x5 window around bbox
# center is sampled and the median taken to reject outliers.
_DEPTH_KERNEL_HALF = 2

# Maximum allowed depth (meters).  Beyond this the stereo estimate is noisy.
_DEPTH_MAX_M = 6.0

# Minimum allowed depth (meters).  Below the D555 stereo baseline limit.
_DEPTH_MIN_M = 0.3


class GoalProjectionNode(Node):
    """Project 2D VLM bounding boxes into map-frame goal poses."""

    def __init__(self) -> None:
        super().__init__("goal_projection")

        self._bridge = CvBridge()

        # Latest depth frame + camera info
        self._latest_depth: Image | None = None
        self._latest_cam_info: CameraInfo | None = None

        # TF2 for camera → map transform
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(
            Image,
            "/d555/aligned_depth_to_color/image_sync",
            self._depth_cb,
            10,
        )
        self.create_subscription(
            CameraInfo,
            "/d555/color/camera_info_sync",
            self._cam_info_cb,
            10,
        )

        self.create_service(
            ProjectDetectionToGoalPose,
            "/strafer/project_detection_to_goal_pose",
            self._handle_projection,
        )

        self.get_logger().info("GoalProjectionNode ready.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _depth_cb(self, msg: Image) -> None:
        self._latest_depth = msg

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        self._latest_cam_info = msg

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------

    def _handle_projection(
        self,
        request: ProjectDetectionToGoalPose.Request,
        response: ProjectDetectionToGoalPose.Response,
    ) -> ProjectDetectionToGoalPose.Response:
        depth_msg = self._latest_depth
        cam_info = self._latest_cam_info

        if depth_msg is None or cam_info is None:
            response.found = False
            response.depth_valid = False
            response.message = "No depth or camera_info received yet."
            return response

        # 1. Bbox normalized [0, 1000] → pixel coordinates
        x1_n, y1_n, x2_n, y2_n = request.bbox_normalized_1000
        img_w = cam_info.width
        img_h = cam_info.height
        cx_px = ((x1_n + x2_n) / 2.0) * img_w / 1000.0
        cy_px = ((y1_n + y2_n) / 2.0) * img_h / 1000.0
        u = int(round(cx_px))
        v = int(round(cy_px))

        # 2. Depth lookup with median kernel
        depth_cv = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_m = self._median_depth(depth_cv, u, v)

        if depth_m is None:
            response.found = False
            response.depth_valid = False
            response.message = (
                f"Invalid depth at pixel ({u}, {v}): "
                "zero, NaN, or out of range."
            )
            return response

        response.depth_valid = True

        # 3. Pixel + depth → 3D point in camera frame (pinhole model)
        fx = cam_info.k[0]
        fy = cam_info.k[4]
        ppx = cam_info.k[2]
        ppy = cam_info.k[5]

        if fx == 0.0 or fy == 0.0:
            response.found = False
            response.message = "Camera intrinsics have zero focal length."
            return response

        x_cam = (u - ppx) * depth_m / fx
        y_cam = (v - ppy) * depth_m / fy
        z_cam = depth_m

        # 4. Transform camera → map via TF2.
        # Use the node's current clock (wall- or sim-time, depending on
        # use_sim_time) as the lookup time with a short wait. The
        # `rclpy.time.Time()` zero-is-latest idiom has been unreliable
        # under sim-time + cross-host TF — tf2 has been seen rejecting
        # it with "extrapolation into the past" when the buffer's
        # earliest entry is newer than epoch-zero.
        camera_frame = depth_msg.header.frame_id or "d555_color_optical_frame"
        target_frame = "map"

        lookup_time = self.get_clock().now()
        lookup_timeout = Duration(seconds=0.5)
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame, camera_frame, lookup_time, timeout=lookup_timeout,
            )
        except TransformException as exc:
            response.found = False
            response.message = (
                f"TF lookup {camera_frame} → {target_frame} failed at "
                f"t={lookup_time.nanoseconds * 1e-9:.3f}s: {exc}"
            )
            return response

        target_in_map = self._transform_point(tf, x_cam, y_cam, z_cam)

        # Fill target_pose (raw projected point, identity orientation)
        response.target_pose = PoseStamped()
        response.target_pose.header.frame_id = target_frame
        response.target_pose.header.stamp = self.get_clock().now().to_msg()
        response.target_pose.pose.position.x = target_in_map[0]
        response.target_pose.pose.position.y = target_in_map[1]
        response.target_pose.pose.position.z = target_in_map[2]
        response.target_pose.pose.orientation.w = 1.0

        # 5. Compute goal pose with standoff
        # Get robot position from TF (base_link → map) at the same
        # clock the camera lookup used, so both project against a
        # coherent snapshot.
        try:
            robot_tf = self._tf_buffer.lookup_transform(
                target_frame, "base_link", lookup_time, timeout=lookup_timeout,
            )
        except TransformException:
            # Fallback: place goal at standoff directly in front of target
            # along the camera axis projected into the map XY plane
            robot_tf = None

        goal_x, goal_y, goal_yaw = self._compute_standoff_pose(
            target_in_map, request.standoff_m, robot_tf
        )

        response.goal_pose = PoseStamped()
        response.goal_pose.header.frame_id = target_frame
        response.goal_pose.header.stamp = self.get_clock().now().to_msg()
        response.goal_pose.pose.position.x = goal_x
        response.goal_pose.pose.position.y = goal_y
        response.goal_pose.pose.position.z = 0.0
        # Yaw → quaternion (rotation about Z)
        response.goal_pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
        response.goal_pose.pose.orientation.w = math.cos(goal_yaw / 2.0)

        # Quality flags
        flags: list[str] = []
        if depth_m < _DEPTH_MIN_M + 0.1:
            flags.append("near_min_range")
        if depth_m > _DEPTH_MAX_M - 0.5:
            flags.append("near_max_range")
        if request.standoff_m > depth_m:
            flags.append("standoff_clipped")
        response.quality_flags = flags

        response.found = True
        response.message = (
            f"Projected '{request.target_label or 'target'}' at "
            f"depth {depth_m:.2f}m → goal ({goal_x:.2f}, {goal_y:.2f}) "
            f"yaw {math.degrees(goal_yaw):.0f}°"
        )

        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _median_depth(
        depth: np.ndarray, u: int, v: int
    ) -> float | None:
        """Return median depth in meters from a 5x5 window, or None if invalid.

        Accepts both depth encodings the stack sees:
          - 16UC1 (uint16, millimetres) — real D555 driver output.
          - 32FC1 (float32, metres)     — Isaac Sim ROS 2 bridge output.
        """
        h, w = depth.shape[:2]
        u = max(_DEPTH_KERNEL_HALF, min(u, w - _DEPTH_KERNEL_HALF - 1))
        v = max(_DEPTH_KERNEL_HALF, min(v, h - _DEPTH_KERNEL_HALF - 1))

        patch = depth[
            v - _DEPTH_KERNEL_HALF : v + _DEPTH_KERNEL_HALF + 1,
            u - _DEPTH_KERNEL_HALF : u + _DEPTH_KERNEL_HALF + 1,
        ]
        if depth.dtype == np.uint16:
            valid = patch[patch > 0].astype(np.float64) * 0.001  # mm → m
        else:
            finite = np.isfinite(patch) & (patch > 0)
            valid = patch[finite].astype(np.float64)
        valid = valid[(valid >= _DEPTH_MIN_M) & (valid <= _DEPTH_MAX_M)]

        if len(valid) == 0:
            return None
        return float(np.median(valid))

    @staticmethod
    def _transform_point(
        tf_stamped: Any, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        """Apply a geometry_msgs/TransformStamped to a 3D point."""
        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation

        # Quaternion rotation: p' = q * p * q_conj
        # Expand for a single point:
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # Rotation matrix from quaternion
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qz * qw)
        r02 = 2.0 * (qx * qz + qy * qw)
        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)
        r20 = 2.0 * (qx * qz - qy * qw)
        r21 = 2.0 * (qy * qz + qx * qw)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        px = r00 * x + r01 * y + r02 * z + t.x
        py = r10 * x + r11 * y + r12 * z + t.y
        pz = r20 * x + r21 * y + r22 * z + t.z
        return (px, py, pz)

    @staticmethod
    def _compute_standoff_pose(
        target_map: tuple[float, float, float],
        standoff_m: float,
        robot_tf: Any | None,
    ) -> tuple[float, float, float]:
        """Compute a goal pose at ``standoff_m`` from the target.

        The robot should face the target at the goal pose. If ``robot_tf``
        is available, the standoff is along the robot→target vector.
        Otherwise, a simple X-axis offset is applied.

        Returns (x, y, yaw) in the map frame.
        """
        tx, ty, _ = target_map

        if robot_tf is not None:
            rx = robot_tf.transform.translation.x
            ry = robot_tf.transform.translation.y
            dx = tx - rx
            dy = ty - ry
            dist = math.hypot(dx, dy)
            if dist < 0.01:
                # Robot is already at the target; face forward (yaw=0)
                return (tx - standoff_m, ty, 0.0)
            # Unit vector robot → target
            ux = dx / dist
            uy = dy / dist
        else:
            # No robot TF; assume target is in front along +X
            ux, uy = 1.0, 0.0

        # Goal is standoff_m back from target along the approach vector
        goal_x = tx - ux * standoff_m
        goal_y = ty - uy * standoff_m
        # Yaw: face the target
        goal_yaw = math.atan2(uy, ux)
        return (goal_x, goal_y, goal_yaw)


# Needed for typing in _transform_point
from typing import Any  # noqa: E402


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GoalProjectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
