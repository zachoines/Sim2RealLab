"""Jetson-local ROS client abstractions and implementations."""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, runtime_checkable

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

    def get_global_costmap_bounds(self) -> "CostmapBounds | None":
        """Return the map-frame extent of the global costmap, or ``None``
        if no costmap message has been received yet."""

    def get_global_costmap_snapshot(self) -> "CostmapSnapshot | None":
        """Return the latest global costmap cells alongside their map-frame
        bounds, or ``None`` if no costmap message has been received yet."""

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
    # Wall-clock window after which a non-advancing ``/clock`` is treated
    # as a stalled / crashed bridge and the motion wait bails out. This
    # replaces the old absolute ``2 * timeout`` wall cap, which fired
    # mid-motion at sub-unity RTF (a 90 s sim deadline maps to ~1800 s of
    # wall time at RTF=0.05, but a ``2 * 90 = 180`` s wall cap tripped long
    # before the rotation finished). The stall detector instead bails only
    # when sim time stops advancing AT ALL for this many wall seconds, so a
    # slow ``/clock`` is tolerated while a frozen one is still caught.
    # Mirrors ``donut_warmup``'s ``clock_stall_bail_wall_s`` (default 15 s).
    clock_stall_bail_wall_s: float = 15.0


class _ProgressTracker:
    """Tracks Nav2 ``distance_remaining`` and reports stalls on lack of improvement.

    Semantics: a sample "improves" if it beats the previous best by at
    least ``stall_progress_m``. The tracker remembers ``best_distance``
    and the timestamp ``best_t`` at which it was achieved. The mission is
    stalled if no improvement has happened in the last ``stall_window_s``
    of clock-time.

    Why best-ever instead of first-sample-in-window: Nav2's
    ``distance_remaining`` is the length of the *current* planned path,
    not Euclidean distance to goal. The planner re-plans on costmap
    updates and can produce a longer path than the previous tick (e.g.
    when a new obstacle appears in a sparse / partially-mapped scene).
    A first-vs-latest comparison would treat the post-re-plan jump as
    "negative progress" and fire spuriously. Best-ever ignores upward
    jumps and only resets the stall timer on genuine improvements.

    Limitation: a re-plan that lengthens the path by more than
    ``stall_window_s * NAV_LINEAR_VEL`` worth of meters can still
    spuriously stall, because the robot has to drive that far before
    beating its previous best. Single-room scenes don't hit this; large
    multi-room re-plans might. The principled fix is a multi-layer
    watchdog (chassis-motion + plan-health + goal-progress signals),
    tracked in ``docs/tasks/active/nav-stall-multilayer-watchdog.md``.

    Pure helper: no ROS / clock dependencies, so unit tests can drive it
    by injecting samples directly.
    """

    def __init__(self, *, stall_progress_m: float, stall_window_s: float) -> None:
        if stall_progress_m <= 0.0 or stall_window_s <= 0.0:
            raise ValueError(
                "stall_progress_m and stall_window_s must both be positive."
            )
        self.stall_progress_m = float(stall_progress_m)
        self.stall_window_s = float(stall_window_s)
        self._best_distance: float | None = None
        self._best_t: float | None = None
        self._latest_t: float | None = None

    def record(self, *, t_s: float, distance_remaining_m: float) -> None:
        """Ingest one Nav2 feedback sample.

        ``t_s`` is the executor node's clock — sim-time under
        ``use_sim_time:=true`` (Isaac Sim bridge), wall-time on real
        hardware. Out-of-order samples are accepted but only forward
        time progression resets the stall timer correctly; in practice
        Nav2 feedback arrives monotonically so this is a non-issue.
        """
        t = float(t_s)
        d = float(distance_remaining_m)
        self._latest_t = t
        if self._best_distance is None or d < self._best_distance - self.stall_progress_m:
            self._best_distance = d
            self._best_t = t

    def is_stalled(self) -> bool:
        if self._best_t is None or self._latest_t is None:
            return False
        return (self._latest_t - self._best_t) >= self.stall_window_s


class _ClockStallDetector:
    """Detects a stalled ``/clock`` by watching sim-time for progress.

    Replaces the old absolute ``2 * timeout`` wall-clock safety cap on the
    executor's motion waits. That cap existed to stop a *crashed* bridge
    from wedging the executor, but it was an absolute wall bound: at
    sub-unity RTF it fired mid-motion while ``/clock`` was still advancing
    legitimately (e.g. RTF=0.05 turns a 90 s sim deadline into ~1800 s of
    wall time, but the ``2 * 90 = 180`` s cap tripped first). This detector
    instead distinguishes "no progress" from "slow progress": it bails only
    when sim time fails to advance AT ALL for ``bail_wall_s`` of wall time,
    which catches a genuinely paused / disconnected bridge without
    penalizing a slow-but-live ``/clock``.

    Caller protocol (mirrors ``donut_warmup.DonutWarmup.run``): on each
    poll, ``update`` with the current ``(sim_now_ns, wall_now_s)`` then
    check ``is_stalled``. A real sim clock advancing at any rate keeps the
    high-water mark fresh and never trips. On real hardware
    (``use_sim_time=False``) ``/clock`` tracks wall time, so sim time always
    advances and the detector never fires — the primary sim==wall deadline
    governs, identical to today's behavior.

    Pure helper: callers feed it ``(sim_ns, wall_s)`` pairs so unit tests
    drive it without a real clock. ``bail_wall_s <= 0`` disables the
    detector (it never reports stalled).
    """

    # Sub-millisecond ``/clock`` republish jitter shouldn't count as
    # progress; require at least 1 ms of advance to reset the stall timer.
    _SIM_PROGRESS_EPSILON_NS = 1_000_000  # 1 ms

    def __init__(
        self, *, bail_wall_s: float, sim_now_ns: int, wall_now_s: float
    ) -> None:
        self.bail_wall_s = float(bail_wall_s)
        self._last_sim_progress_ns = int(sim_now_ns)
        self._last_sim_progress_wall_s = float(wall_now_s)

    def update(self, *, sim_now_ns: int, wall_now_s: float) -> None:
        """Record a poll. Resets the stall timer if sim time advanced."""
        if sim_now_ns > self._last_sim_progress_ns + self._SIM_PROGRESS_EPSILON_NS:
            self._last_sim_progress_ns = int(sim_now_ns)
            self._last_sim_progress_wall_s = float(wall_now_s)

    def is_stalled(self, *, wall_now_s: float) -> bool:
        """True if sim time hasn't advanced for ``bail_wall_s`` wall sec."""
        if self.bail_wall_s <= 0.0:
            return False
        return (wall_now_s - self._last_sim_progress_wall_s) >= self.bail_wall_s


@dataclass(frozen=True)
class CostmapBounds:
    """Axis-aligned rectangle covered by the Nav2 global costmap, in the map frame."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    resolution: float


@dataclass(frozen=True)
class CostmapSnapshot:
    """Map-frame snapshot of the Nav2 global costmap.

    ``data`` is an int8 numpy array shape ``(height, width)`` with cell
    semantics: ``-1`` is unknown, ``0..occupied_threshold - 1`` is free,
    values ``>= occupied_threshold`` are occupied. The Nav2 default
    ``occupied_threshold`` is 65. Row-major indexing: ``data[row, col]``
    maps to map-frame ``(bounds.min_x + col*res, bounds.min_y + row*res)``.
    """

    bounds: CostmapBounds
    width: int
    height: int
    data: Any  # numpy int8 ndarray shape (height, width)


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

    # VLM detection overlay topic — published by `publish_detections()` when a
    # grounding skill resolves. Foxglove's Image panel attaches this as an
    # annotation source on the RGB feed.
    TOPIC_DETECTIONS = "/d555/color/detections"

    # Frozen source frame for the most recent successful grounding. Lets
    # Foxglove draw a stable image+bbox overlay (the bbox is in pixel coords
    # for *this* frame, not the live camera). Latched; refreshed only on
    # accepted detections so the last-grounded view persists between calls.
    TOPIC_GROUNDING_FRAME = "/d555/color/grounding_frame"

    # Foxglove-native image-overlay companion. Foxglove Studio 2.x does not
    # auto-decode `vision_msgs/Detection2DArray` for image annotations even
    # though the topic exists on the graph — so we *also* publish
    # `foxglove_msgs/ImageAnnotations`, which the Image panel renders
    # natively. The Detection2DArray topic stays as the canonical semantic
    # output (RViz, bag replay, downstream ROS consumers); this companion
    # exists only for Foxglove rendering.
    TOPIC_DETECTIONS_FG = "/d555/color/detections_fg"

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

        # Eagerly created so the topic appears on the graph from boot —
        # otherwise lazy creation hides "vision_msgs not installed" behind
        # the same symptom as "no grounding has fired yet."
        self._detections_pub: Any = None
        self._grounding_frame_pub: Any = None
        self._detections_fg_pub: Any = None

        self._setup_subscriptions()
        self._setup_publishers()

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

    def _setup_publishers(self) -> None:
        """Create publishers up front so topics are discoverable from boot.

        ``vision_msgs`` is an optional runtime dep (installed via
        ``ros-humble-vision-msgs``). When it is missing we log a loud
        warning and leave the publisher unset — ``publish_detections`` then
        no-ops, but every other skill keeps working. This is preferred to
        crashing the executor or hiding the missing-package failure behind
        the lazy-create path.

        The detections topic uses ``TRANSIENT_LOCAL`` durability so DDS
        replays the most recent message to late subscribers — Foxglove
        attaching mid-mission, or ``ros2 topic echo`` opened after a scan
        completed, still see the current overlay state instead of an
        empty wire. Grounding fires for milliseconds at a time and then
        the topic goes idle for the duration of navigation; without a
        latched QoS, anyone who subscribes during that idle window sees
        nothing.
        """
        try:
            from vision_msgs.msg import Detection2DArray
        except ImportError:
            self._node.get_logger().warning(
                "vision_msgs not installed; %s overlay disabled. "
                "Install with: sudo apt install ros-humble-vision-msgs",
                self.TOPIC_DETECTIONS,
            )
            return
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        latched_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._detections_pub = self._node.create_publisher(
            Detection2DArray, self.TOPIC_DETECTIONS, latched_qos,
        )

        # Companion frozen-source-frame publisher. The bbox in a
        # Detection2DArray is in pixel-space against the exact image that
        # was sent to the VLM; publishing that image alongside lets a
        # Foxglove panel paired to this topic render a stable
        # image+overlay even as the robot keeps moving past the moment
        # of capture. Latched with the same QoS so late subscribers get
        # the most recent grounded view.
        from sensor_msgs.msg import Image
        self._grounding_frame_pub = self._node.create_publisher(
            Image, self.TOPIC_GROUNDING_FRAME, latched_qos,
        )

        # Foxglove-native ImageAnnotations companion. Foxglove Studio 2.x
        # exposes Detection2DArray as a topic but does not render it as
        # an image overlay — the Image panel's Annotations dropdown
        # silently drops it. Publishing the same bbox data as
        # ``foxglove_msgs/ImageAnnotations`` is the supported path. If
        # ``foxglove_msgs`` is missing we degrade the same way as for
        # ``vision_msgs``: warn loudly at startup, then keep the rest
        # of the executor running.
        try:
            from foxglove_msgs.msg import ImageAnnotations
        except ImportError:
            self._node.get_logger().warning(
                "foxglove_msgs not installed; %s Foxglove-native overlay disabled. "
                "Install with: sudo apt install ros-humble-foxglove-msgs",
                self.TOPIC_DETECTIONS_FG,
            )
        else:
            self._detections_fg_pub = self._node.create_publisher(
                ImageAnnotations, self.TOPIC_DETECTIONS_FG, latched_qos,
            )

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

    def get_global_costmap_bounds(self) -> CostmapBounds | None:
        """Return the map-frame extent of the latest global costmap.

        Returns ``None`` when the costmap subscription has not yet seen a
        message (the structured "not yet available" sentinel callers use
        to fall back to non-staged navigation).
        """
        with self._cache_lock:
            costmap = self._latest_costmap
        if costmap is None:
            return None
        info = costmap.info
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        return CostmapBounds(
            min_x=origin_x,
            min_y=origin_y,
            max_x=origin_x + info.width * info.resolution,
            max_y=origin_y + info.height * info.resolution,
            resolution=info.resolution,
        )

    def get_global_costmap_snapshot(self) -> CostmapSnapshot | None:
        """Return the latest global costmap cells plus their bounds.

        The returned ``CostmapSnapshot.data`` is a numpy int8 array shape
        ``(height, width)`` with Nav2 cell semantics (``-1`` unknown,
        ``< 65`` free, ``>= 65`` occupied). Returns ``None`` if the
        costmap subscription has not seen a message yet — frontier
        callers should treat this as "no exploration possible yet."
        """
        import numpy as np

        with self._cache_lock:
            costmap = self._latest_costmap
        if costmap is None:
            return None
        info = costmap.info
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        bounds = CostmapBounds(
            min_x=origin_x,
            min_y=origin_y,
            max_x=origin_x + info.width * info.resolution,
            max_y=origin_y + info.height * info.resolution,
            resolution=info.resolution,
        )
        # ``data`` arrives as a flat list/array of int8 cells in row-major
        # order from Nav2; reshape into (H, W) for spatial indexing.
        flat = np.asarray(costmap.data, dtype=np.int8)
        grid = flat.reshape(info.height, info.width)
        return CostmapSnapshot(
            bounds=bounds,
            width=int(info.width),
            height=int(info.height),
            data=grid,
        )

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

        # Real D555 driver publishes aligned depth as 16UC1 (mm); the Isaac
        # Sim ROS 2 bridge publishes it as 32FC1 (m). Normalise to float32 m.
        depth_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_raw.dtype == np.uint16:
            aligned_depth_m = depth_raw.astype(np.float32) * 0.001
        else:
            aligned_depth_m = depth_raw.astype(np.float32)

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
        stall_progress_m: float | None = None,
        stall_window_s: float | None = None,
    ) -> SkillResult:
        """Send a goal to Nav2 and block until completion, timeout, or cancel.

        Creates a ``NavigateToPose`` action client on first call and
        reuses it for subsequent goals.  The active goal handle is tracked
        so ``cancel_active_navigation`` can abort it.

        When both ``stall_progress_m`` and ``stall_window_s`` are provided,
        a watchdog runs alongside the deadline: if Nav2's
        ``distance_remaining`` feedback does not decrease by at least
        ``stall_progress_m`` over the most recent ``stall_window_s`` of
        sim-time, the goal is canceled with
        ``error_code=navigation_stalled``. The watchdog only fires once
        the window is fully populated, so the first ``stall_window_s`` of
        any navigation cannot trip it.
        """
        from nav2_msgs.action import NavigateToPose as Nav2NavigateToPose
        from geometry_msgs.msg import PoseStamped
        from rclpy.action import ActionClient
        from action_msgs.msg import GoalStatus

        started_at = time.time()
        timeout = timeout_s or self._config.default_nav_timeout_s

        tracker: _ProgressTracker | None = None
        if stall_progress_m is not None and stall_window_s is not None:
            tracker = _ProgressTracker(
                stall_progress_m=stall_progress_m,
                stall_window_s=stall_window_s,
            )

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

        def _on_feedback(feedback_msg: Any) -> None:
            if tracker is None:
                return
            try:
                distance = float(feedback_msg.feedback.distance_remaining)
                # Executor node's clock: sim-time under use_sim_time:=true,
                # wall-time on real hardware. Both flow through the same path.
                t_s = self._node.get_clock().now().nanoseconds * 1e-9
                tracker.record(t_s=t_s, distance_remaining_m=distance)
            except Exception:
                logger.debug("Nav2 feedback handling failed", exc_info=True)

        send_future = self._nav2_client.send_goal_async(
            goal_msg, feedback_callback=_on_feedback,
        )
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
        completed, stalled = self._wait_for_nav_result(
            result_future, timeout_s=timeout, tracker=tracker,
        )

        with self._nav_lock:
            self._active_goal_handle = None

        if stalled:
            goal_handle.cancel_goal_async()
            return SkillResult(
                step_id=step_id, skill="navigate_to_pose", status="failed",
                error_code="navigation_stalled",
                message=(
                    f"Navigation made no progress (>= {stall_progress_m:.2f} m) "
                    f"in the last {stall_window_s:.0f} s; canceled."
                ),
                started_at=started_at, finished_at=time.time(),
            )
        if not completed:
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
    # Detection overlay publisher (vision_msgs/Detection2DArray)
    # ------------------------------------------------------------------

    def publish_detections(
        self,
        *,
        image_stamp_sec: float,
        image_frame_id: str,
        image_width: int,
        image_height: int,
        detections: "Iterable[tuple[tuple[int, int, int, int], str | None, float | None]]" = (),
    ) -> None:
        """Publish a ``vision_msgs/Detection2DArray`` for Foxglove overlay.

        ``detections`` is an iterable of ``(bbox_qwen_1000, label, confidence)``
        tuples where ``bbox_qwen_1000`` is ``(x1, y1, x2, y2)`` in the VLM's
        normalized [0, 1000] coordinate space (see
        ``strafer_msgs/srv/ProjectDetectionToGoalPose`` and
        ``goal_projection_node`` for the same convention). Each bbox is
        rescaled to pixel coordinates against the source image's dimensions
        before publishing — Foxglove's Image annotation overlay expects
        pixel coords.

        ``header.stamp`` is set from ``image_stamp_sec`` (the source image's
        stamp) so Foxglove can pair the overlay with the right RGB frame.
        Empty ``detections`` publishes an empty array, which clears any
        stale overlay from a previous grounding call.
        """
        if self._detections_pub is None:
            # vision_msgs missing at startup — already logged once; stay
            # silent here so we don't spam the log on every grounding call.
            return

        from builtin_interfaces.msg import Time as TimeMsg
        from vision_msgs.msg import (
            BoundingBox2D,
            Detection2D,
            Detection2DArray,
            ObjectHypothesisWithPose,
        )

        sec = int(image_stamp_sec)
        nanosec = int(round((image_stamp_sec - sec) * 1e9))
        # Round-up edge case: e.g. 1.9999999999 → sec=1, nanosec=1e9.
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000

        msg = Detection2DArray()
        msg.header.stamp = TimeMsg(sec=sec, nanosec=nanosec)
        msg.header.frame_id = image_frame_id

        scale_x = float(image_width) / 1000.0
        scale_y = float(image_height) / 1000.0

        for bbox_qwen, label, confidence in detections:
            x1, y1, x2, y2 = (float(v) for v in bbox_qwen)
            cx_px = (x1 + x2) * 0.5 * scale_x
            cy_px = (y1 + y2) * 0.5 * scale_y
            size_x_px = abs(x2 - x1) * scale_x
            size_y_px = abs(y2 - y1) * scale_y

            det = Detection2D()
            det.header = msg.header

            bbox = BoundingBox2D()
            # vision_msgs 4.x (Humble) replaced geometry_msgs/Pose2D with
            # vision_msgs/Pose2D, which nests x/y under .position. Older
            # builds expose .x/.y directly. Branch so we work on both.
            if hasattr(bbox.center, "position"):
                bbox.center.position.x = cx_px
                bbox.center.position.y = cy_px
            else:
                bbox.center.x = cx_px
                bbox.center.y = cy_px
            bbox.center.theta = 0.0
            bbox.size_x = size_x_px
            bbox.size_y = size_y_px
            det.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label or ""
            hyp.hypothesis.score = float(confidence) if confidence is not None else 0.0
            det.results.append(hyp)

            msg.detections.append(det)

        self._detections_pub.publish(msg)

        # Companion Foxglove-native overlay. Foxglove Studio 2.x doesn't
        # render Detection2DArray as an image annotation; foxglove_msgs/
        # ImageAnnotations is the supported schema. Same bbox data, just
        # framed for the Foxglove Image panel's renderer.
        self._publish_foxglove_annotations(
            sec=sec,
            nanosec=nanosec,
            scale_x=scale_x,
            scale_y=scale_y,
            detections=list(msg.detections),  # already-built Detection2D objects
            source=detections,                # original (bbox_qwen, label, conf) tuples
        )

    def _publish_foxglove_annotations(
        self,
        *,
        sec: int,
        nanosec: int,
        scale_x: float,
        scale_y: float,
        detections: "list[Any]",  # noqa: ARG002 — kept for symmetry / future debug
        source: "Iterable[tuple[tuple[int, int, int, int], str | None, float | None]]",
    ) -> None:
        """Emit the same bbox data as ``foxglove_msgs/ImageAnnotations``.

        One ``PointsAnnotation`` (LINE_LOOP of four corners) per bbox plus
        a label ``TextAnnotation`` above each box. The annotation
        timestamp matches the Detection2DArray header so a Foxglove panel
        configured with ``synchronize: true`` against the grounding-frame
        topic pairs them exactly.

        No-ops cleanly if ``foxglove_msgs`` was missing at startup.
        """
        if self._detections_fg_pub is None:
            return
        from builtin_interfaces.msg import Time as TimeMsg
        from foxglove_msgs.msg import (
            Color,
            ImageAnnotations,
            Point2,
            PointsAnnotation,
            TextAnnotation,
        )

        stamp = TimeMsg(sec=sec, nanosec=nanosec)
        green = Color(r=0.2, g=1.0, b=0.4, a=1.0)
        black = Color(r=0.0, g=0.0, b=0.0, a=0.0)
        white = Color(r=1.0, g=1.0, b=1.0, a=1.0)
        text_bg = Color(r=0.0, g=0.0, b=0.0, a=0.6)

        anno = ImageAnnotations()
        for bbox_qwen, label, confidence in source:
            x1, y1, x2, y2 = (float(v) for v in bbox_qwen)
            x1_px = x1 * scale_x
            y1_px = y1 * scale_y
            x2_px = x2 * scale_x
            y2_px = y2 * scale_y

            pa = PointsAnnotation()
            pa.timestamp = stamp
            pa.type = PointsAnnotation.LINE_LOOP
            pa.points = [
                Point2(x=x1_px, y=y1_px),
                Point2(x=x2_px, y=y1_px),
                Point2(x=x2_px, y=y2_px),
                Point2(x=x1_px, y=y2_px),
            ]
            pa.outline_color = green
            pa.fill_color = black  # transparent fill; outline only
            pa.thickness = 2.0
            anno.points.append(pa)

            if label:
                text = label
                if confidence is not None:
                    text = f"{label} {confidence:.2f}"
                ta = TextAnnotation()
                ta.timestamp = stamp
                ta.position = Point2(x=x1_px, y=max(0.0, y1_px - 4.0))
                ta.text = text
                ta.font_size = 14.0
                ta.text_color = white
                ta.background_color = text_bg
                anno.texts.append(ta)

        self._detections_fg_pub.publish(anno)

    def publish_grounding_frame(
        self,
        *,
        image_bgr: Any,
        image_stamp_sec: float,
        image_frame_id: str,
    ) -> None:
        """Publish the exact source image used for the most recent grounding.

        Pairs with ``publish_detections``: the bbox lives in pixel coords
        against this specific image, so Foxglove can pair the frozen frame
        + Detection2DArray (`synchronize: true`) and render a stable
        image+overlay that doesn't drift as the robot moves. Latched
        ``TRANSIENT_LOCAL`` so the most recent grounded view is always
        available to late subscribers.

        Caller is expected to invoke this only for accepted detections —
        the frozen frame should *not* refresh on empty publishes
        (rejection / not-found), so the operator keeps seeing the
        last-grounded view until the next real detection.
        """
        if self._grounding_frame_pub is None:
            return

        from builtin_interfaces.msg import Time as TimeMsg
        from cv_bridge import CvBridge

        sec = int(image_stamp_sec)
        nanosec = int(round((image_stamp_sec - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000

        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
        img_msg.header.stamp = TimeMsg(sec=sec, nanosec=nanosec)
        img_msg.header.frame_id = image_frame_id
        self._grounding_frame_pub.publish(img_msg)

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

        # Sim-clock deadline (mirrors `_wait_for_future` / `navigate_to_pose`).
        # Honors `use_sim_time`: at sub-unity RTF, `timeout` sim-seconds is
        # `timeout / RTF` wall-seconds. A sim-time stall detector (not an
        # absolute wall cap) guards against a frozen `/clock` wedging the
        # executor — it bails only if sim time stops advancing entirely.
        from rclpy.duration import Duration

        clock = self._node.get_clock()
        deadline = clock.now() + Duration(seconds=timeout)
        stall = _ClockStallDetector(
            bail_wall_s=self._config.clock_stall_bail_wall_s,
            sim_now_ns=clock.now().nanoseconds,
            wall_now_s=time.monotonic(),
        )
        while True:
            now = clock.now()
            if now >= deadline:
                break
            wall_now = time.monotonic()
            stall.update(sim_now_ns=now.nanoseconds, wall_now_s=wall_now)
            if stall.is_stalled(wall_now_s=wall_now):
                logger.warning(
                    "rotate_in_place: /clock stalled — no sim-time progress "
                    "in %.1fs of wall clock; stopping so the executor isn't "
                    "wedged.",
                    self._config.clock_stall_bail_wall_s,
                )
                break
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

    def _wait_for_future(self, future: Any, timeout_s: float) -> bool:
        """Block on *future* using the executor node's clock.

        Honors ``use_sim_time``: sim launches with ``use_sim_time:=true``
        tick on ``/clock``, real launches tick on the system clock. This
        keeps the executor's timeout enforcement aligned with Nav2,
        RTAB-Map, and the BT navigator, all of which already tick on
        the same clock.

        A sim-time stall detector bounds the wait if ``/clock`` stops
        advancing (e.g. sim bridge crash mid-mission) so the executor
        cannot wedge waiting on a frozen sim clock. Unlike the old
        ``2 * timeout_s`` absolute wall cap, it tolerates a slow-but-live
        ``/clock`` at sub-unity RTF and fires only on genuine stalls. On
        real hardware the system clock advances at wall rate, so the
        detector never trips before the primary timeout fires.

        Returns ``True`` if the future completed, ``False`` on timeout.
        """
        from rclpy.duration import Duration

        done = threading.Event()
        future.add_done_callback(lambda _: done.set())

        clock = self._node.get_clock()
        deadline = clock.now() + Duration(seconds=timeout_s)
        stall = _ClockStallDetector(
            bail_wall_s=self._config.clock_stall_bail_wall_s,
            sim_now_ns=clock.now().nanoseconds,
            wall_now_s=time.monotonic(),
        )

        poll_dt = max(0.01, min(0.1, timeout_s / 10.0))
        while not done.is_set():
            now = clock.now()
            if now >= deadline:
                return False
            wall_now = time.monotonic()
            stall.update(sim_now_ns=now.nanoseconds, wall_now_s=wall_now)
            if stall.is_stalled(wall_now_s=wall_now):
                logger.warning(
                    "Future wait: /clock stalled — no sim-time progress in "
                    "%.1fs of wall clock; is the bridge alive?",
                    self._config.clock_stall_bail_wall_s,
                )
                return False
            if done.wait(timeout=poll_dt):
                return True
        return True

    def _wait_for_nav_result(
        self,
        future: Any,
        *,
        timeout_s: float,
        tracker: "_ProgressTracker | None",
    ) -> tuple[bool, bool]:
        """``_wait_for_future`` plus an optional progress watchdog.

        Returns ``(completed, stalled)``:

        - ``(True,  False)`` — the Nav2 result future resolved before the
          deadline or stall watchdog fired.
        - ``(False, False)`` — the sim-clock deadline (or the ``/clock``
          stall detector) tripped first.
        - ``(False, True)``  — the progress watchdog tripped first.

        ``stalled`` and the deadline cannot both be true; the loop
        returns on the first triggering condition.

        Two distinct stall signals coexist here: ``tracker`` watches Nav2
        ``distance_remaining`` for *goal* progress (no closer in
        ``nav_stall_window_s`` → ``navigation_stalled``), while the
        ``_ClockStallDetector`` watches ``/clock`` itself for *sim-time*
        progress (frozen bridge → deadline-class failure). A slow RTF
        trips neither.
        """
        from rclpy.duration import Duration

        done = threading.Event()
        future.add_done_callback(lambda _: done.set())

        clock = self._node.get_clock()
        deadline = clock.now() + Duration(seconds=timeout_s)
        stall = _ClockStallDetector(
            bail_wall_s=self._config.clock_stall_bail_wall_s,
            sim_now_ns=clock.now().nanoseconds,
            wall_now_s=time.monotonic(),
        )

        poll_dt = max(0.01, min(0.1, timeout_s / 10.0))
        while not done.is_set():
            now = clock.now()
            if now >= deadline:
                return (False, False)
            wall_now = time.monotonic()
            stall.update(sim_now_ns=now.nanoseconds, wall_now_s=wall_now)
            if stall.is_stalled(wall_now_s=wall_now):
                logger.warning(
                    "Nav2 wait: /clock stalled — no sim-time progress in "
                    "%.1fs of wall clock; is the bridge alive?",
                    self._config.clock_stall_bail_wall_s,
                )
                return (False, False)
            if tracker is not None and tracker.is_stalled():
                return (False, True)
            if done.wait(timeout=poll_dt):
                return (True, False)
        return (True, False)
