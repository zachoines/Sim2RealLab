#!/usr/bin/env python3
"""Drive-pattern validation for the Strafer robot.

Publishes open-loop cmd_vel commands to drive a predefined shape while
collecting odometry, SLAM, encoder, and map data.  After the pattern
completes, prints a pass/fail validation report and generates a video
showing the map being built with the robot's path overlaid.

Requires at minimum: strafer_driver + strafer_description (base.launch.py).
SLAM topics (/rtabmap/map, map→odom TF) are optional — metrics that need
them are skipped gracefully.

Usage
-----
    # Basic square test (driver + description running):
    ros2 run strafer_bringup validate_drive --ros-args -p pattern:=square

    # Mecanum strafing test:
    ros2 run strafer_bringup validate_drive --ros-args -p pattern:=strafe_square

    # Full stack with SLAM + custom output:
    ros2 run strafer_bringup validate_drive --ros-args \\
        -p pattern:=circle -p output_dir:=/tmp/test1
"""

import csv
import math
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_ros import (
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ═════════════════════════════════════════════════════════════════════════
# Data types
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    """Open-loop velocity command for a fixed duration."""
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    duration: float = 0.0

    @property
    def is_motion(self) -> bool:
        return abs(self.vx) > 1e-3 or abs(self.vy) > 1e-3 or abs(self.omega) > 1e-3


@dataclass
class Pose2D:
    stamp: float
    x: float
    y: float
    yaw: float


@dataclass
class JointSample:
    stamp: float
    velocities: Tuple[float, ...]


@dataclass
class MapFrame:
    stamp: float
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    data: np.ndarray  # int8


@dataclass
class Metric:
    name: str
    passed: bool
    value: str
    threshold: str
    detail: str = ""


# ═════════════════════════════════════════════════════════════════════════
# Pattern generators
# ═════════════════════════════════════════════════════════════════════════

_PAUSE = 0.5  # seconds between motion segments


def _stop(duration: float = _PAUSE) -> Segment:
    return Segment(0.0, 0.0, 0.0, duration)


def build_square(side=1.0, vel=0.2, omega=0.5, **_kw) -> List[Segment]:
    """Forward + 90° CCW turn, repeated 4x.  Tests rotational accuracy."""
    fwd_t = side / vel
    turn_t = (math.pi / 2) / omega
    segs: List[Segment] = []
    for _ in range(4):
        segs.append(Segment(vel, 0.0, 0.0, fwd_t))
        segs.append(_stop())
        segs.append(Segment(0.0, 0.0, omega, turn_t))
        segs.append(_stop())
    return segs


def build_strafe_square(side=1.0, vel=0.2, **_kw) -> List[Segment]:
    """Fwd → left → back → right.  No rotation — tests mecanum strafing."""
    t = side / vel
    return [
        Segment(vel, 0.0, 0.0, t), _stop(),
        Segment(0.0, vel, 0.0, t), _stop(),
        Segment(-vel, 0.0, 0.0, t), _stop(),
        Segment(0.0, -vel, 0.0, t), _stop(),
    ]


def build_circle(radius=0.5, vel=0.2, **_kw) -> List[Segment]:
    """Constant arc — one full CCW loop."""
    omega = vel / radius
    t = (2 * math.pi) / omega
    return [Segment(vel, 0.0, omega, t)]


def build_figure8(radius=0.5, vel=0.2, **_kw) -> List[Segment]:
    """Two opposing full loops (CCW then CW)."""
    omega = vel / radius
    loop_t = (2 * math.pi) / omega
    return [
        Segment(vel, 0.0, omega, loop_t),
        Segment(vel, 0.0, -omega, loop_t),
    ]


PATTERNS = {
    "square": build_square,
    "strafe_square": build_strafe_square,
    "circle": build_circle,
    "figure8": build_figure8,
}


def total_duration(segs: List[Segment]) -> float:
    return sum(s.duration for s in segs)


# ═════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════

def compute_odom_drift(poses: List[Pose2D]) -> Metric:
    """Distance between first and last odom pose (closure error)."""
    if len(poses) < 2:
        return Metric("Odom closure", False, "N/A", "< 0.30 m",
                       "Insufficient data")
    s, e = poses[0], poses[-1]
    d = math.hypot(e.x - s.x, e.y - s.y)
    return Metric(
        "Odom closure", d < 0.30, f"{d:.3f} m", "< 0.30 m",
        f"({s.x:.2f},{s.y:.2f}) \u2192 ({e.x:.2f},{e.y:.2f})",
    )


def compute_slam_deviation(
    odom: List[Pose2D], slam: List[Pose2D],
) -> Metric:
    """Mean odom-vs-SLAM trajectory deviation."""
    if not slam:
        return Metric("SLAM vs Odom", True, "N/A", "< 0.15 m",
                       "No SLAM data \u2014 skipped")
    devs: List[float] = []
    for sp in slam:
        nearest = min(odom, key=lambda op: abs(op.stamp - sp.stamp))
        if abs(nearest.stamp - sp.stamp) < 0.5:
            devs.append(math.hypot(sp.x - nearest.x, sp.y - nearest.y))
    if not devs:
        return Metric("SLAM vs Odom", True, "N/A", "< 0.15 m",
                       "No matching timestamps")
    mean_d = sum(devs) / len(devs)
    return Metric(
        "SLAM vs Odom", mean_d < 0.15, f"{mean_d:.3f} m", "< 0.15 m",
        f"{len(devs)} pairs, max {max(devs):.3f} m",
    )


def compute_encoder_health(
    joints: List[JointSample],
    segments: List[Segment],
    pattern_start: float,
) -> Metric:
    """Check all 4 encoders report non-zero velocity during motion."""
    intervals: List[Tuple[float, float]] = []
    t = 0.0
    for seg in segments:
        if seg.is_motion:
            intervals.append((pattern_start + t,
                               pattern_start + t + seg.duration))
        t += seg.duration

    moving = [
        j for j in joints
        if any(a <= j.stamp <= b for a, b in intervals)
    ]
    if not moving:
        return Metric("Encoder health", False, "N/A", "> 90 %",
                       "No joint data during motion")

    names = ["FL", "FR", "RL", "RR"]
    parts: List[str] = []
    healthy = True
    for i, name in enumerate(names):
        active = sum(1 for j in moving if abs(j.velocities[i]) > 0.01)
        pct = active / len(moving) * 100
        parts.append(f"{name} {pct:.0f}%")
        if pct < 90:
            healthy = False
    return Metric("Encoder health", healthy, ", ".join(parts), "> 90 %",
                   f"{len(moving)} samples in motion windows")


def compute_map_quality(maps: List[MapFrame]) -> Metric:
    """Check the final map has meaningful content."""
    if not maps:
        return Metric("Map quality", True, "N/A", "> 10 % mapped",
                       "No map data \u2014 skipped")
    m = maps[-1]
    total = m.data.size
    known = int(np.sum(m.data >= 0))
    pct = known / total * 100 if total else 0
    return Metric(
        "Map quality", pct > 10, f"{pct:.1f} %", "> 10 % mapped",
        f"{known}/{total} cells known",
    )


# ═════════════════════════════════════════════════════════════════════════
# Output: CSV
# ═════════════════════════════════════════════════════════════════════════

def save_poses_csv(poses: List[Pose2D], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp", "x", "y", "yaw"])
        for p in poses:
            w.writerow([f"{p.stamp:.4f}", f"{p.x:.4f}",
                        f"{p.y:.4f}", f"{p.yaw:.4f}"])


# ═════════════════════════════════════════════════════════════════════════
# Output: report
# ═════════════════════════════════════════════════════════════════════════

def format_report(
    metrics: List[Metric], pattern: str, duration: float,
) -> str:
    W = 72
    lines = [
        "\u2550" * W,
        "  STRAFER VALIDATION REPORT".center(W),
        "\u2550" * W,
        f"  Pattern:   {pattern}",
        f"  Duration:  {duration:.1f} s",
        "\u2500" * W,
    ]
    all_pass = True
    for m in metrics:
        tag = "PASS" if m.passed else "FAIL"
        if not m.passed:
            all_pass = False
        lines.append(
            f"  [{tag}] {m.name:<20s}  {m.value:<18s}  "
            f"(threshold: {m.threshold})"
        )
        if m.detail:
            lines.append(f"         {m.detail}")
    lines.append("\u2500" * W)
    verdict = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    lines.append(f"  Result: {verdict}")
    lines.append("\u2550" * W)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
# Output: video
# ═════════════════════════════════════════════════════════════════════════

def _world_bounds(odom, slam, maps, margin=1.0):
    """Axis-aligned bounding box of all data in world coordinates."""
    xs = [p.x for p in odom] + [p.x for p in slam]
    ys = [p.y for p in odom] + [p.y for p in slam]
    for m in maps:
        xs += [m.origin_x, m.origin_x + m.width * m.resolution]
        ys += [m.origin_y, m.origin_y + m.height * m.resolution]
    if not xs:
        return -2.0, -2.0, 2.0, 2.0
    return (min(xs) - margin, min(ys) - margin,
            max(xs) + margin, max(ys) + margin)


def _blit_map(img, mf, world_x_min, world_y_max, ppm):
    """Render an OccupancyGrid MapFrame onto the canvas (vectorized)."""
    h, w = mf.height, mf.width
    data = mf.data.reshape(h, w)

    map_img = np.full((h, w, 3), 50, dtype=np.uint8)
    free = (data >= 0) & (data <= 50)
    occ = data > 50
    map_img[free] = [220, 220, 220]
    map_img[occ] = [30, 20, 60]
    map_img = np.flipud(map_img)

    scale = mf.resolution * ppm
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    map_img = cv2.resize(map_img, (new_w, new_h),
                         interpolation=cv2.INTER_NEAREST)

    px = int((mf.origin_x - world_x_min) * ppm)
    py = int((world_y_max - (mf.origin_y + h * mf.resolution)) * ppm)

    sy, sx = max(0, -py), max(0, -px)
    dy, dx = max(0, py), max(0, px)
    ch, cw = img.shape[:2]
    ey = min(new_h - sy, ch - dy)
    ex = min(new_w - sx, cw - dx)
    if ey > 0 and ex > 0:
        img[dy:dy + ey, dx:dx + ex] = map_img[sy:sy + ey, sx:sx + ex]


def _draw_grid(img, x_min, y_min, x_max, y_max, ppm, spacing=0.5):
    """Subtle reference grid when no map is available."""
    h, w = img.shape[:2]
    color = (70, 70, 70)
    x = math.ceil(x_min / spacing) * spacing
    while x <= x_max:
        px = int((x - x_min) * ppm)
        if 0 <= px < w:
            cv2.line(img, (px, 0), (px, h), color, 1)
        x += spacing
    y = math.ceil(y_min / spacing) * spacing
    while y <= y_max:
        py = int((y_max - y) * ppm)
        if 0 <= py < h:
            cv2.line(img, (0, py), (w, py), color, 1)
        y += spacing


def generate_video(
    odom: List[Pose2D],
    slam: List[Pose2D],
    maps: List[MapFrame],
    output_path: str,
    pattern_name: str = "",
    fps: int = 10,
    canvas_px: int = 800,
) -> bool:
    """Render validation video.  Returns True on success."""
    if not _HAS_CV2 or not odom:
        return False

    x0, y0, x1, y1 = _world_bounds(odom, slam, maps)
    span = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = cx - span / 2, cx + span / 2
    y0, y1 = cy - span / 2, cy + span / 2
    ppm = canvas_px / span

    def w2p(x, y):
        return int((x - x0) * ppm), int((y1 - y) * ppm)

    t_start = odom[0].stamp
    t_end = odom[-1].stamp
    n_frames = max(1, int((t_end - t_start) * fps)) + 1
    freeze_frames = fps * 3  # 3-second freeze at end

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps),
                             (canvas_px, canvas_px))
    if not writer.isOpened():
        return False

    map_sorted = sorted(maps, key=lambda m: m.stamp)

    for fi in range(n_frames + freeze_frames):
        t = min(t_start + fi / fps, t_end)
        img = np.full((canvas_px, canvas_px, 3), 50, dtype=np.uint8)

        # ── Map background ──────────────────────────────────────────
        cur_map = None
        for mf in reversed(map_sorted):
            if mf.stamp <= t:
                cur_map = mf
                break
        if cur_map:
            _blit_map(img, cur_map, x0, y1, ppm)
        else:
            _draw_grid(img, x0, y0, x1, y1, ppm)

        # ── Odom trail (blue) ───────────────────────────────────────
        odom_pts = [w2p(p.x, p.y) for p in odom if p.stamp <= t]
        for i in range(1, len(odom_pts)):
            cv2.line(img, odom_pts[i - 1], odom_pts[i], (255, 150, 0), 2)

        # ── SLAM trail (green) ──────────────────────────────────────
        slam_pts = [w2p(p.x, p.y) for p in slam if p.stamp <= t]
        for i in range(1, len(slam_pts)):
            cv2.line(img, slam_pts[i - 1], slam_pts[i], (0, 220, 0), 2)

        # ── Start marker (green diamond) ────────────────────────────
        sp = w2p(odom[0].x, odom[0].y)
        cv2.drawMarker(img, sp, (0, 200, 0), cv2.MARKER_DIAMOND, 14, 2)

        # ── Current position (red circle + heading arrow) ──────────
        cur_poses = [p for p in odom if p.stamp <= t]
        if cur_poses:
            cur = cur_poses[-1]
            cp = w2p(cur.x, cur.y)
            cv2.circle(img, cp, 7, (0, 0, 255), -1)
            al = 18
            dx = int(al * math.cos(cur.yaw))
            dy = int(-al * math.sin(cur.yaw))
            cv2.arrowedLine(img, cp, (cp[0] + dx, cp[1] + dy),
                            (0, 0, 255), 2, tipLength=0.35)

        # ── Text overlay ────────────────────────────────────────────
        elapsed = t - t_start
        cv2.putText(img, f"{pattern_name}  t={elapsed:.1f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        # ── Legend ──────────────────────────────────────────────────
        cv2.rectangle(img, (8, canvas_px - 60), (114, canvas_px - 6),
                      (30, 30, 30), -1)
        cv2.putText(img, "- Odom", (14, canvas_px - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1)
        cv2.putText(img, "- SLAM", (14, canvas_px - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        writer.write(img)

    writer.release()
    return True


# ═════════════════════════════════════════════════════════════════════════
# ROS2 Node
# ═════════════════════════════════════════════════════════════════════════

class _State(Enum):
    WARMUP = auto()
    RUNNING = auto()
    SETTLING = auto()
    REPORTING = auto()


class ValidateDriveNode(Node):

    def __init__(self):
        super().__init__("validate_drive")

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter("pattern", "square")
        self.declare_parameter("output_dir", "/tmp/strafer_validation")
        self.declare_parameter("linear_vel", 0.2)
        self.declare_parameter("angular_vel", 0.5)
        self.declare_parameter("side_length", 1.0)
        self.declare_parameter("radius", 0.5)
        self.declare_parameter("warmup_sec", 5.0)
        self.declare_parameter("video_fps", 10)

        gp = self.get_parameter
        self._pattern_name: str = gp("pattern").value
        self._output_dir: str = gp("output_dir").value
        vel: float = gp("linear_vel").value
        omega: float = gp("angular_vel").value
        side: float = gp("side_length").value
        radius: float = gp("radius").value
        self._warmup_sec: float = gp("warmup_sec").value
        self._video_fps: int = gp("video_fps").value

        if self._pattern_name not in PATTERNS:
            valid = ", ".join(PATTERNS)
            self.get_logger().fatal(
                f"Unknown pattern '{self._pattern_name}'. Choose: {valid}")
            raise SystemExit(1)
        builder = PATTERNS[self._pattern_name]
        self._segments = builder(
            side=side, vel=vel, omega=omega, radius=radius)

        dur = total_duration(self._segments)
        self.get_logger().info(
            f"Pattern '{self._pattern_name}': "
            f"{len(self._segments)} segments, {dur:.1f} s")

        # ── Pub / Sub ──────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(
            Twist, "/strafer/cmd_vel", 10)
        self.create_subscription(
            Odometry, "/strafer/odom", self._on_odom, 10)
        self.create_subscription(
            JointState, "/strafer/joint_states", self._on_joints, 10)
        self.create_subscription(
            OccupancyGrid, "/rtabmap/map", self._on_map, 1)

        # ── TF ─────────────────────────────────────────────────────
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        # ── Data ───────────────────────────────────────────────────
        self._odom: List[Pose2D] = []
        self._slam: List[Pose2D] = []
        self._joints: List[JointSample] = []
        self._maps: List[MapFrame] = []

        # ── State machine ──────────────────────────────────────────
        self._state = _State.WARMUP
        self._warmup_t0 = self.get_clock().now()
        self._pattern_start: Optional[float] = None
        self._seg_idx = 0
        self._seg_elapsed = 0.0
        self._settle_t0: Optional[float] = None
        self.done = False
        self._reported = False

        self._dt = 0.05  # 20 Hz tick
        self._timer = self.create_timer(self._dt, self._tick)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_sec(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def _quat_yaw(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _now(self) -> float:
        return self._to_sec(self.get_clock().now().to_msg())

    def _pub_twist(self, seg: Segment):
        msg = Twist()
        msg.linear.x = seg.vx
        msg.linear.y = seg.vy
        msg.angular.z = seg.omega
        self._cmd_pub.publish(msg)

    def _pub_stop(self):
        self._cmd_pub.publish(Twist())

    # ── Callbacks ───────────────────────────────────────────────────

    def _on_odom(self, msg: Odometry):
        self._odom.append(Pose2D(
            stamp=self._to_sec(msg.header.stamp),
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            yaw=self._quat_yaw(msg.pose.pose.orientation),
        ))

    def _on_joints(self, msg: JointState):
        if msg.velocity:
            self._joints.append(JointSample(
                stamp=self._to_sec(msg.header.stamp),
                velocities=tuple(msg.velocity[:4]),
            ))

    def _on_map(self, msg: OccupancyGrid):
        self._maps.append(MapFrame(
            stamp=self._to_sec(msg.header.stamp),
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
            data=np.array(msg.data, dtype=np.int8),
        ))

    def _sample_slam(self):
        try:
            tf = self._tf_buf.lookup_transform(
                "map", "base_link", Time())
            tr = tf.transform.translation
            self._slam.append(Pose2D(
                stamp=self._to_sec(tf.header.stamp),
                x=tr.x, y=tr.y,
                yaw=self._quat_yaw(tf.transform.rotation),
            ))
        except (LookupException, ConnectivityException,
                ExtrapolationException):
            pass

    # ── State machine ──────────────────────────────────────────────

    def _tick(self):
        self._sample_slam()
        if self._state == _State.WARMUP:
            self._do_warmup()
        elif self._state == _State.RUNNING:
            self._do_running()
        elif self._state == _State.SETTLING:
            self._do_settling()
        elif self._state == _State.REPORTING:
            self._do_reporting()

    def _do_warmup(self):
        elapsed = (self.get_clock().now() -
                   self._warmup_t0).nanoseconds * 1e-9
        if self._odom:
            self.get_logger().info(
                f"Odom received \u2014 warmup done ({elapsed:.1f} s)")
            self._begin_pattern()
            return
        if elapsed > self._warmup_sec:
            self.get_logger().warn(
                "No odom during warmup \u2014 proceeding anyway")
            self._begin_pattern()

    def _begin_pattern(self):
        self._state = _State.RUNNING
        self._pattern_start = self._now()
        self._seg_idx = 0
        self._seg_elapsed = 0.0
        self.get_logger().info("Pattern started")

    def _do_running(self):
        if self._seg_idx >= len(self._segments):
            self._pub_stop()
            self._state = _State.SETTLING
            self._settle_t0 = self._now()
            self.get_logger().info(
                "Pattern complete \u2014 settling (2 s)...")
            return
        seg = self._segments[self._seg_idx]
        self._pub_twist(seg)
        self._seg_elapsed += self._dt
        if self._seg_elapsed >= seg.duration:
            self._seg_idx += 1
            self._seg_elapsed = 0.0

    def _do_settling(self):
        self._pub_stop()
        if self._now() - self._settle_t0 > 2.0:
            self._state = _State.REPORTING

    def _do_reporting(self):
        if self._reported:
            return
        self._reported = True
        self._timer.cancel()
        os.makedirs(self._output_dir, exist_ok=True)

        dur = total_duration(self._segments)

        # ── Metrics ────────────────────────────────────────────────
        metrics = [
            compute_odom_drift(self._odom),
            compute_slam_deviation(self._odom, self._slam),
            compute_encoder_health(
                self._joints, self._segments,
                self._pattern_start or 0.0),
            compute_map_quality(self._maps),
        ]

        # ── Report ─────────────────────────────────────────────────
        report = format_report(metrics, self._pattern_name, dur)
        self.get_logger().info("\n" + report)
        rpt_path = os.path.join(self._output_dir,
                                "validation_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report)

        # ── CSV ────────────────────────────────────────────────────
        if self._odom:
            save_poses_csv(self._odom,
                           os.path.join(self._output_dir, "odom_path.csv"))
        if self._slam:
            save_poses_csv(self._slam,
                           os.path.join(self._output_dir, "slam_path.csv"))

        # ── Video ──────────────────────────────────────────────────
        if _HAS_CV2 and self._odom:
            vpath = os.path.join(self._output_dir, "validation.mp4")
            self.get_logger().info(f"Generating video \u2192 {vpath}")
            ok = generate_video(
                self._odom, self._slam, self._maps,
                vpath, self._pattern_name, self._video_fps)
            if ok:
                self.get_logger().info(f"Video saved: {vpath}")
            else:
                self.get_logger().warn("Video generation failed")
        elif not _HAS_CV2:
            self.get_logger().warn(
                "cv2 not available \u2014 skipping video")

        self.get_logger().info(f"Output \u2192 {self._output_dir}/")
        self.done = True


# ═════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = ValidateDriveNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info(
            "Interrupted \u2014 generating partial report...")
    finally:
        if not node._reported:
            node._do_reporting()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
