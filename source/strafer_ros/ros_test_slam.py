#!/usr/bin/env python3
"""ROS2 SLAM + motion verification for the Strafer robot.

Publishes Twist commands while recording odometry, SLAM pose (map→base_link
TF), and the RTAB-Map occupancy grid.  Generates a video showing the map
building up with both odom (blue) and SLAM (green) trails.

Requires the full navigation stack:
    make launch          # or: ros2 launch strafer_bringup navigation.launch.py

Usage:
    python3 ros_test_slam.py                                # verify topics only
    python3 ros_test_slam.py --drive forward --duration 3   # drive + record
    python3 ros_test_slam.py --drive rotate --duration 5    # rotate in place
    python3 ros_test_slam.py --drive all --duration 3       # all patterns
    python3 ros_test_slam.py --drive all --speed 1.5        # faster
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from strafer_shared.constants import DEPTH_CLIP_NEAR, DEPTH_CLIP_FAR

# ── Defaults ────────────────────────────────────────────────────────────
DEFAULT_BASE_LINEAR = 0.2   # m/s
DEFAULT_BASE_OMEGA = 0.3    # rad/s
PUBLISH_RATE_HZ = 50
TOPIC_TIMEOUT_SEC = 8.0
VIDEO_FPS = 10
CANVAS_PX = 800

# ── Data containers ─────────────────────────────────────────────────────

@dataclass
class Pose2D:
    stamp: float
    x: float
    y: float
    yaw: float


@dataclass
class MapFrame:
    stamp: float
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    data: np.ndarray


# ── ROS node ─────────────────────────────────────────────────────────────

class SlamTestNode(Node):
    """Subscribes to odom, SLAM TF, occupancy grid, and camera; publishes cmd_vel."""

    def __init__(self) -> None:
        super().__init__("ros_test_slam")
        self._lock = threading.Lock()
        self._bridge = CvBridge()

        # Publishers
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Subscribers
        self.create_subscription(Odometry, "/strafer/odom", self._on_odom, 10)
        self.create_subscription(OccupancyGrid, "/rtabmap/map", self._on_map, 1)
        self.create_subscription(
            Image, "/d555/color/image_sync", self._on_color, 1)
        self.create_subscription(
            Image, "/d555/aligned_depth_to_color/image_sync",
            self._on_depth, 1)

        # TF2
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)

        # Data stores
        self._odom: list[Pose2D] = []
        self._slam: list[Pose2D] = []
        self._maps: list[MapFrame] = []
        self._color_frames: list[tuple[float, np.ndarray]] = []  # (stamp, bgr)
        self._depth_frames: list[tuple[float, np.ndarray]] = []  # (stamp, 16UC1 mm)
        self._last_color_save: float = 0.0
        self._last_depth_save: float = 0.0
        self._FRAME_SAVE_INTERVAL: float = 0.5  # save a frame every 0.5s

        # Topic arrival flags
        self._got_odom = False
        self._got_map = False
        self._got_tf = False
        self._got_color = False
        self._got_depth = False

        # Latest odom for console feedback
        self._odom_vx = 0.0
        self._odom_vy = 0.0
        self._odom_omega = 0.0
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_odom(self, msg: Odometry) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self._lock:
            self._odom.append(Pose2D(stamp=t, x=p.x, y=p.y, yaw=yaw))
            self._odom_vx = msg.twist.twist.linear.x
            self._odom_vy = msg.twist.twist.linear.y
            self._odom_omega = msg.twist.twist.angular.z
            self._odom_x = p.x
            self._odom_y = p.y
            self._odom_theta = yaw
            self._got_odom = True

    def _on_map(self, msg: OccupancyGrid) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self._lock:
            self._maps.append(MapFrame(
                stamp=t,
                width=msg.info.width,
                height=msg.info.height,
                resolution=msg.info.resolution,
                origin_x=msg.info.origin.position.x,
                origin_y=msg.info.origin.position.y,
                data=np.array(msg.data, dtype=np.int8),
            ))
            self._got_map = True

    def _on_color(self, msg: Image) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._got_color = True
            if t - self._last_color_save >= self._FRAME_SAVE_INTERVAL:
                self._color_frames.append((t, img))
                self._last_color_save = t

    def _on_depth(self, msg: Image) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self._lock:
            self._got_depth = True
            if t - self._last_depth_save >= self._FRAME_SAVE_INTERVAL:
                self._depth_frames.append((t, img))
                self._last_depth_save = t

    def sample_slam_tf(self) -> None:
        """Try to look up the map→base_link TF and record it."""
        try:
            tf = self._tf_buf.lookup_transform("map", "base_link", Time())
            tr = tf.transform.translation
            q = tf.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            with self._lock:
                self._slam.append(Pose2D(stamp=t, x=tr.x, y=tr.y, yaw=yaw))
                self._got_tf = True
        except (LookupException, ConnectivityException, ExtrapolationException):
            pass

    # ── Accessors ────────────────────────────────────────────────────

    def get_topic_status(self) -> dict[str, bool]:
        with self._lock:
            return {
                "odom": self._got_odom,
                "map": self._got_map,
                "tf (map→base_link)": self._got_tf,
                "color": self._got_color,
                "depth": self._got_depth,
            }

    def get_odom_feedback(self) -> str:
        with self._lock:
            if not self._got_odom:
                return "(no odom yet)"
            return (
                f"vel=[{self._odom_vx:+.3f}, {self._odom_vy:+.3f}, "
                f"{self._odom_omega:+.3f}]  "
                f"pos=[{self._odom_x:+.3f}, {self._odom_y:+.3f}]  "
                f"yaw={math.degrees(self._odom_theta):+.1f}\u00b0"
            )

    def get_data(self):
        with self._lock:
            return (
                list(self._odom),
                list(self._slam),
                list(self._maps),
                [(t, img.copy()) for t, img in self._color_frames],
                [(t, img.copy()) for t, img in self._depth_frames],
            )

    def publish_twist(self, vx: float, vy: float, omega: float) -> None:
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = omega
        self._cmd_pub.publish(msg)

    def stop(self) -> None:
        self.publish_twist(0.0, 0.0, 0.0)

    def spin_for(self, seconds: float) -> None:
        end = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.01)
            self.sample_slam_tf()


# ── Topic verification ───────────────────────────────────────────────────

def verify_topics(node: SlamTestNode) -> bool:
    print(f"\n{'=' * 60}")
    print("  Strafer SLAM + Motion Test")
    print(f"{'=' * 60}")
    print(f"\n  Checking topics (timeout {TOPIC_TIMEOUT_SEC:.0f}s)...")

    start = time.monotonic()
    while time.monotonic() - start < TOPIC_TIMEOUT_SEC:
        rclpy.spin_once(node, timeout_sec=0.1)
        node.sample_slam_tf()
        status = node.get_topic_status()
        if all(status.values()):
            break

    status = node.get_topic_status()
    all_ok = True
    for name, ok in status.items():
        tag = "  OK  " if ok else " MISS "
        print(f"    [{tag}]  {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All topics active. Ready to drive.")
    else:
        missing = [n for n, ok in status.items() if not ok]
        print(f"\n  WARNING: Missing: {', '.join(missing)}")
        if not status.get("odom"):
            print("    Is the driver running?  make launch")
        if not status.get("map") or not status.get("tf (map→base_link)"):
            print("    RTAB-Map may need a moment to produce the first map.")
            print("    Move the robot slightly so SLAM gets visual features.")

    return all_ok


# ── Motion patterns ──────────────────────────────────────────────────────

def build_patterns(
    speed: float,
) -> dict[str, list[tuple[str, Callable[[float], tuple[float, float, float]]]]]:
    bl = DEFAULT_BASE_LINEAR * speed
    bo = DEFAULT_BASE_OMEGA * speed

    def forward(t):
        return (bl, 0.0, 0.0)

    def backward(t):
        return (-bl, 0.0, 0.0)

    def strafe_left(t):
        return (0.0, bl, 0.0)

    def strafe_right(t):
        return (0.0, -bl, 0.0)

    def rotate(t):
        period = 2.0
        phase = (t % (2 * period)) / period
        w = bo if phase < 1.0 else -bo
        return (0.0, 0.0, w)

    def circle(t):
        return (bl, 0.0, 0.3 * speed)

    return {
        "forward": [("Forward", forward)],
        "backward": [("Backward", backward)],
        "strafe_left": [("Strafe Left", strafe_left)],
        "strafe_right": [("Strafe Right", strafe_right)],
        "rotate": [("Rotate CW/CCW", rotate)],
        "circle": [("Circle", circle)],
        "all": [
            ("Forward", forward),
            ("Backward", backward),
            ("Strafe Left", strafe_left),
            ("Strafe Right", strafe_right),
            ("Rotate CW/CCW", rotate),
            ("Circle", circle),
        ],
    }


def run_patterns(
    node: SlamTestNode,
    patterns: list[tuple[str, Callable]],
    duration: float,
    speed: float,
) -> None:
    print(f"\n  Speed:    {speed:.1f}x  (base={DEFAULT_BASE_LINEAR} m/s)")
    print(f"  Duration: {duration}s per pattern")
    print(f"  Patterns: {[n for n, _ in patterns]}")
    print(f"  Press Ctrl+C to stop.\n")

    sleep_sec = 1.0 / PUBLISH_RATE_HZ

    for pattern_name, pattern_fn in patterns:
        print(f">>> {pattern_name}  ({duration}s)")
        print(
            f"    {'time':>6s}  {'cmd vx':>8s} {'cmd vy':>8s} {'cmd w':>8s}"
            f"  |  odom feedback"
        )
        print(
            f"    {'─' * 6}  {'─' * 8} {'─' * 8} {'─' * 8}"
            f"  |  {'─' * 50}"
        )

        start = node.get_clock().now()
        last_print_t = 0.0

        while rclpy.ok():
            t = (node.get_clock().now() - start).nanoseconds / 1e9
            if t >= duration:
                break

            vx, vy, omega = pattern_fn(t)
            node.publish_twist(vx, vy, omega)
            node.sample_slam_tf()

            if t - last_print_t >= 1.0:
                last_print_t = t
                fb = node.get_odom_feedback()
                print(
                    f"    {t:5.1f}s  {vx:+8.3f} {vy:+8.3f} {omega:+8.3f}"
                    f"  |  {fb}"
                )

            rclpy.spin_once(node, timeout_sec=sleep_sec)

        node.stop()
        if len(patterns) > 1:
            print("    ... pausing 2s")
            node.spin_for(2.0)

    node.stop()
    print("\n  Drive complete.")


# ── Video generation ─────────────────────────────────────────────────────

def _world_bounds(odom, slam, maps, margin=0.5):
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


def _pick_frame(frames, t):
    """Return the frame closest to timestamp t from a [(stamp, img)] list, or None."""
    if not frames:
        return None
    best = None
    best_dt = float("inf")
    for ft, img in frames:
        dt = abs(ft - t)
        if dt < best_dt:
            best_dt = dt
            best = img
    return best


def _colorize_depth(depth_16uc1: np.ndarray) -> np.ndarray:
    """Colorize 16UC1 depth (mm) with TURBO colormap."""
    depth_m = depth_16uc1.astype(np.float32) * 0.001
    depth_norm = np.clip(depth_m / DEPTH_CLIP_FAR, 0.0, 1.0)
    colored = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth_m < DEPTH_CLIP_NEAR] = 0
    return colored


def generate_video(
    odom: list[Pose2D],
    slam: list[Pose2D],
    maps: list[MapFrame],
    color_frames: list[tuple[float, np.ndarray]],
    depth_frames: list[tuple[float, np.ndarray]],
    output_path: str,
    pattern_label: str = "",
) -> bool:
    if not odom:
        print("  No odom data recorded — skipping video.")
        return False

    x0, y0, x1, y1 = _world_bounds(odom, slam, maps)
    span = max(x1 - x0, y1 - y0, 0.5)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = cx - span / 2, cx + span / 2
    y0, y1 = cy - span / 2, cy + span / 2
    ppm = CANVAS_PX / span

    # Video layout: map on left (800x800), camera + stats on right (400x800)
    map_w = CANVAS_PX
    side_w = 400
    total_w = map_w + side_w
    total_h = CANVAS_PX

    def w2p(x, y):
        return int((x - x0) * ppm), int((y1 - y) * ppm)

    t_start = odom[0].stamp
    t_end = odom[-1].stamp
    n_frames = max(1, int((t_end - t_start) * VIDEO_FPS)) + 1
    freeze_frames = VIDEO_FPS * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(VIDEO_FPS),
                             (total_w, total_h))
    if not writer.isOpened():
        print(f"  FAIL: Cannot open video writer for {output_path}")
        return False

    map_sorted = sorted(maps, key=lambda m: m.stamp)
    print(f"  Rendering {n_frames + freeze_frames} frames...")

    for fi in range(n_frames + freeze_frames):
        t = min(t_start + fi / VIDEO_FPS, t_end)

        # ── Left panel: map ──────────────────────────────────────────
        map_img = np.full((map_w, map_w, 3), 50, dtype=np.uint8)

        cur_map = None
        for mf in reversed(map_sorted):
            if mf.stamp <= t:
                cur_map = mf
                break
        if cur_map:
            _blit_map(map_img, cur_map, x0, y1, ppm)
        else:
            _draw_grid(map_img, x0, y0, x1, y1, ppm)

        # Odom trail (orange)
        odom_pts = [w2p(p.x, p.y) for p in odom if p.stamp <= t]
        for i in range(1, len(odom_pts)):
            cv2.line(map_img, odom_pts[i - 1], odom_pts[i], (255, 150, 0), 2)

        # SLAM trail (green)
        slam_pts = [w2p(p.x, p.y) for p in slam if p.stamp <= t]
        for i in range(1, len(slam_pts)):
            cv2.line(map_img, slam_pts[i - 1], slam_pts[i], (0, 220, 0), 2)

        # Start marker
        sp = w2p(odom[0].x, odom[0].y)
        cv2.drawMarker(map_img, sp, (0, 200, 0), cv2.MARKER_DIAMOND, 14, 2)

        # Current position + heading arrow
        cur_odom = [p for p in odom if p.stamp <= t]
        if cur_odom:
            cur = cur_odom[-1]
            cp = w2p(cur.x, cur.y)
            cv2.circle(map_img, cp, 7, (0, 0, 255), -1)
            al = 20
            dx = int(al * math.cos(cur.yaw))
            dy = int(-al * math.sin(cur.yaw))
            cv2.arrowedLine(map_img, cp, (cp[0] + dx, cp[1] + dy),
                            (0, 0, 255), 2, tipLength=0.35)

        # ── Right panel: RGB + depth + stats ─────────────────────────
        side_img = np.zeros((total_h, side_w, 3), dtype=np.uint8)

        # Layout: 3 rows — RGB (top third), depth (middle third), stats (bottom third)
        row_h = total_h // 3

        # RGB thumbnail
        cur_color = _pick_frame(color_frames, t)
        if cur_color is not None:
            thumb = cv2.resize(cur_color, (side_w, row_h))
            cv2.putText(thumb, "RGB", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            side_img[:row_h, :] = thumb
        else:
            cv2.putText(side_img, "No camera", (side_w // 2 - 60, row_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        # Depth colorized thumbnail
        cur_depth = _pick_frame(depth_frames, t)
        if cur_depth is not None:
            depth_col = _colorize_depth(cur_depth)
            depth_thumb = cv2.resize(depth_col, (side_w, row_h))
            cv2.putText(depth_thumb,
                        f"Depth ({DEPTH_CLIP_NEAR:.1f}-{DEPTH_CLIP_FAR:.0f}m)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            side_img[row_h:row_h * 2, :] = depth_thumb
        else:
            cv2.putText(side_img, "No depth",
                        (side_w // 2 - 50, row_h + row_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        # Stats (bottom third)
        elapsed = t - t_start
        y_text = row_h * 2 + 18
        line_h = 18
        stats_color = (200, 200, 200)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.42

        lines = [
            f"{pattern_label}  t={elapsed:.1f}s",
        ]

        if cur_odom:
            c = cur_odom[-1]
            lines += [
                f"Odom  x:{c.x:+.3f} y:{c.y:+.3f} yaw:{math.degrees(c.yaw):+.1f}",
            ]

        cur_slam = [p for p in slam if p.stamp <= t]
        if cur_slam:
            s = cur_slam[-1]
            lines += [
                f"SLAM  x:{s.x:+.3f} y:{s.y:+.3f} yaw:{math.degrees(s.yaw):+.1f}",
            ]

        if cur_map:
            lines.append(
                f"Map {cur_map.width}x{cur_map.height} @ {cur_map.resolution}m")

        lines.append(
            f"Samples: odom={len(odom_pts)} slam={len(slam_pts)}"
            f" maps={len([m for m in maps if m.stamp <= t])}")

        for line in lines:
            cv2.putText(side_img, line, (8, y_text), font, fs,
                        stats_color, 1, cv2.LINE_AA)
            y_text += line_h

        # Legend
        leg_y = total_h - 30
        cv2.rectangle(side_img, (6, leg_y - 4), (160, total_h - 4),
                      (30, 30, 30), -1)
        cv2.putText(side_img, "- Odom", (10, leg_y + 10),
                    font, 0.4, (255, 150, 0), 1)
        cv2.putText(side_img, "- SLAM", (80, leg_y + 10),
                    font, 0.4, (0, 220, 0), 1)

        # ── Compose ──────────────────────────────────────────────────
        frame = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        frame[:, :map_w] = map_img
        frame[:, map_w:] = side_img
        writer.write(frame)

        if (fi + 1) % (VIDEO_FPS * 2) == 0 or fi == n_frames + freeze_frames - 1:
            print(f"    {fi + 1}/{n_frames + freeze_frames} frames")

    writer.release()
    return True


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ROS2 SLAM + motion test for the Strafer robot",
    )
    parser.add_argument(
        "--drive",
        type=str,
        default=None,
        choices=[
            "forward", "backward",
            "strafe_left", "strafe_right",
            "rotate", "circle", "all",
        ],
        help="Motion pattern to execute (omit to only verify topics)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speed scale factor: 1.0 = 0.2 m/s base (default: 1.0)",
    )
    parser.add_argument(
        "--duration", type=float, default=3.0,
        help="Duration per pattern in seconds (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for output video (default: cwd)",
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip video generation after driving",
    )
    args = parser.parse_args()

    rclpy.init()
    node = SlamTestNode()

    try:
        ok = verify_topics(node)

        if args.drive is None:
            if not ok:
                sys.exit(1)
            print("\n  No --drive pattern specified. Use --drive forward (etc.) to move.")
            sys.exit(0)

        # Run motion patterns
        pattern_map = build_patterns(args.speed)
        patterns = pattern_map[args.drive]
        run_patterns(node, patterns, args.duration, args.speed)

        # Settle — keep sampling for 2s after stopping
        print("  Settling for 2s (collecting final SLAM data)...")
        node.spin_for(2.0)

        odom, slam, maps, color_frames, depth_frames = node.get_data()
        print(f"\n  Recorded: {len(odom)} odom, {len(slam)} SLAM, "
              f"{len(maps)} map updates, {len(color_frames)} camera, "
              f"{len(depth_frames)} depth frames")

        if not args.no_video:
            os.makedirs(args.output_dir, exist_ok=True)
            video_path = os.path.join(args.output_dir, "slam_test.mp4")
            print(f"\n  Generating video -> {video_path}")
            if generate_video(odom, slam, maps, color_frames, depth_frames,
                              video_path,
                              pattern_label=args.drive):
                print(f"  Video saved: {video_path}")
            else:
                print("  Video generation failed.")

    except KeyboardInterrupt:
        node.stop()
        print("\n[INTERRUPTED]")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
