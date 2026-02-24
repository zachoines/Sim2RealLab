#!/usr/bin/env python3
"""ROS2-based perception stack verification and recording.

Subscribes to the perception stack topics (must be running) and records
side-by-side videos showing what the robot actually sees — including the
downsampled 80x60 depth image that gets fed to the policy.

Requires the perception stack to be running:
    ros2 launch strafer_perception perception.launch.py

Topics subscribed:
    /d555/color/image_raw       — 640x360 BGR8 (RGB camera)
    /d555/depth/image_rect_raw  — 640x360 16UC1 (raw depth, mm)
    /d555/depth/downsampled     — 80x60 32FC1 (policy input, m)
    /d555/imu                   — sensor_msgs/Imu (accel + gyro)

Usage:
    python3 ros_test_perception.py                       # verify topics
    python3 ros_test_perception.py --record              # record 5s videos
    python3 ros_test_perception.py --record --duration 10
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge

from strafer_shared.constants import (
    DEPTH_WIDTH,
    DEPTH_HEIGHT,
    DEPTH_CLIP_NEAR,
    DEPTH_CLIP_FAR,
)

TOPIC_TIMEOUT_SEC = 5.0


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


class PerceptionTestNode(Node):
    """Subscribes to all perception topics and collects data for recording."""

    def __init__(self) -> None:
        super().__init__("ros_test_perception")
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        # Latest frames
        self._color_img: np.ndarray | None = None
        self._depth_raw_img: np.ndarray | None = None
        self._depth_ds_img: np.ndarray | None = None

        # IMU data
        self._imu_accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._imu_gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Counters
        self._counts = {"color": 0, "depth_raw": 0, "depth_ds": 0, "imu": 0}

        # Collected IMU for recording
        self._imu_log: list[tuple[float, float, float, float, float, float, float]] = []

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(
            Image, "/d555/color/image_raw", self._color_cb, qos)
        self.create_subscription(
            Image, "/d555/depth/image_rect_raw", self._depth_raw_cb, qos)
        self.create_subscription(
            Image, "/d555/depth/downsampled", self._depth_ds_cb, qos)
        self.create_subscription(
            Imu, "/d555/imu", self._imu_cb, qos)

    # ── Callbacks ──────────────────────────────────────────────────

    def _color_cb(self, msg: Image) -> None:
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._color_img = img
            self._counts["color"] += 1

    def _depth_raw_cb(self, msg: Image) -> None:
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self._lock:
            self._depth_raw_img = img
            self._counts["depth_raw"] += 1

    def _depth_ds_cb(self, msg: Image) -> None:
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self._lock:
            self._depth_ds_img = img
            self._counts["depth_ds"] += 1

    def _imu_cb(self, msg: Imu) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        a = msg.linear_acceleration
        g = msg.angular_velocity
        with self._lock:
            self._imu_accel = (a.x, a.y, a.z)
            self._imu_gyro = (g.x, g.y, g.z)
            self._counts["imu"] += 1
            self._imu_log.append((t, a.x, a.y, a.z, g.x, g.y, g.z))

    # ── Accessors ──────────────────────────────────────────────────

    def get_snapshot(self):
        """Return copies of current frames + IMU under lock."""
        with self._lock:
            return (
                self._color_img.copy() if self._color_img is not None else None,
                self._depth_raw_img.copy() if self._depth_raw_img is not None else None,
                self._depth_ds_img.copy() if self._depth_ds_img is not None else None,
                self._imu_accel,
                self._imu_gyro,
            )

    def get_counts(self) -> dict[str, int]:
        with self._lock:
            return self._counts.copy()

    def get_imu_log(self) -> list:
        with self._lock:
            return list(self._imu_log)

    def clear_imu_log(self) -> None:
        with self._lock:
            self._imu_log.clear()


def _colorize_depth_raw(depth_16uc1: np.ndarray) -> np.ndarray:
    """Colorize raw 16UC1 depth (mm) with TURBO colormap."""
    depth_m = depth_16uc1.astype(np.float32) * 0.001
    depth_norm = np.clip(depth_m / DEPTH_CLIP_FAR, 0.0, 1.0)
    colored = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth_m < DEPTH_CLIP_NEAR] = 0
    return colored


def _colorize_depth_ds(depth_32fc1: np.ndarray) -> np.ndarray:
    """Colorize downsampled 32FC1 depth (m), upscale for visibility."""
    depth_norm = np.clip(depth_32fc1 / DEPTH_CLIP_FAR, 0.0, 1.0)
    colored = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth_32fc1 < DEPTH_CLIP_NEAR] = 0
    return colored


def verify_topics(node: PerceptionTestNode) -> bool:
    """Wait for messages on all topics and report status."""
    _section("Topic Verification")
    topics = {
        "color": "/d555/color/image_raw",
        "depth_raw": "/d555/depth/image_rect_raw",
        "depth_ds": "/d555/depth/downsampled",
        "imu": "/d555/imu",
    }

    start = time.monotonic()
    all_ok = False
    while time.monotonic() - start < TOPIC_TIMEOUT_SEC:
        rclpy.spin_once(node, timeout_sec=0.1)
        counts = node.get_counts()
        if all(counts[k] > 0 for k in topics):
            all_ok = True
            break

    counts = node.get_counts()
    for key, topic in topics.items():
        status = "OK" if counts[key] > 0 else "MISSING"
        print(f"  {status:7s}  {topic}  ({counts[key]} msgs)")

    # Print resolution info
    snap = node.get_snapshot()
    if snap[0] is not None:
        h, w = snap[0].shape[:2]
        print(f"\n  Color resolution:          {w}x{h}")
    if snap[1] is not None:
        h, w = snap[1].shape[:2]
        print(f"  Depth (raw) resolution:    {w}x{h}")
    if snap[2] is not None:
        h, w = snap[2].shape[:2]
        print(f"  Depth (policy) resolution: {w}x{h}")

    if all_ok:
        print("\n  PASS: All perception topics active")
    else:
        missing = [t for k, t in topics.items() if counts[k] == 0]
        print(f"\n  FAIL: Missing topics: {', '.join(missing)}")
        print("  Is the perception stack running?")
        print("    ros2 launch strafer_perception perception.launch.py")

    return all_ok


def record_perception_video(
    node: PerceptionTestNode,
    duration: float,
    output_dir: str,
) -> str:
    """Record a composite video showing all perception data.

    Layout (top row: camera view, bottom row: policy view):
      ┌──────────────────┬──────────────────┐
      │  RGB (640x360)   │  Depth raw col.  │
      ├──────────────────┼──────────────────┤
      │ Downsampled depth│  IMU overlay     │
      │ (80x60 upscaled) │  (text readout)  │
      └──────────────────┴──────────────────┘

    Returns the output file path, or empty string on failure.
    """
    _section(f"Recording Perception Video ({duration:.0f}s)")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "perception.mp4")

    PANEL_W, PANEL_H = 640, 360
    CANVAS_W, CANVAS_H = PANEL_W * 2, PANEL_H * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 30, (CANVAS_W, CANVAS_H))
    if not writer.isOpened():
        print(f"  FAIL: Cannot create video writer for {output_path}")
        return ""

    frame_count = 0
    start = time.monotonic()

    while time.monotonic() - start < duration:
        rclpy.spin_once(node, timeout_sec=0.03)
        color, depth_raw, depth_ds, imu_accel, imu_gyro = node.get_snapshot()

        # Skip until we have at least color
        if color is None:
            continue

        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # Top-left: RGB
        rgb_panel = cv2.resize(color, (PANEL_W, PANEL_H))
        cv2.putText(rgb_panel, "RGB (640x360)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        canvas[:PANEL_H, :PANEL_W] = rgb_panel

        # Top-right: Raw depth colorized
        if depth_raw is not None:
            depth_col = _colorize_depth_raw(depth_raw)
            depth_col = cv2.resize(depth_col, (PANEL_W, PANEL_H))
            cv2.putText(depth_col, f"Depth Raw (0-{DEPTH_CLIP_FAR:.0f}m)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            canvas[:PANEL_H, PANEL_W:] = depth_col

        # Bottom-left: Downsampled depth (upscaled for visibility)
        if depth_ds is not None:
            ds_col = _colorize_depth_ds(depth_ds)
            ds_upscaled = cv2.resize(ds_col, (PANEL_W, PANEL_H),
                                     interpolation=cv2.INTER_NEAREST)
            cv2.putText(ds_upscaled,
                        f"Policy Depth ({DEPTH_WIDTH}x{DEPTH_HEIGHT} upscaled)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # Draw grid to show actual pixel boundaries
            step_x = PANEL_W // DEPTH_WIDTH
            step_y = PANEL_H // DEPTH_HEIGHT
            for gx in range(0, PANEL_W, step_x):
                cv2.line(ds_upscaled, (gx, 0), (gx, PANEL_H),
                         (80, 80, 80), 1)
            for gy in range(0, PANEL_H, step_y):
                cv2.line(ds_upscaled, (0, gy), (PANEL_W, gy),
                         (80, 80, 80), 1)

            canvas[PANEL_H:, :PANEL_W] = ds_upscaled

        # Bottom-right: IMU text readout
        imu_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        ax, ay, az = imu_accel
        gx, gy, gz = imu_gyro
        accel_norm = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_norm = math.sqrt(gx * gx + gy * gy + gz * gz)

        lines = [
            "IMU (ROS frame)",
            "",
            f"Accel (m/s2):",
            f"  X: {ax:+8.3f}",
            f"  Y: {ay:+8.3f}",
            f"  Z: {az:+8.3f}",
            f"  |a|: {accel_norm:.3f}",
            "",
            f"Gyro (rad/s):",
            f"  X: {gx:+8.4f}",
            f"  Y: {gy:+8.4f}",
            f"  Z: {gz:+8.4f}",
            f"  |w|: {gyro_norm:.4f}",
            "",
            f"t = {time.monotonic() - start:.1f}s / {duration:.0f}s",
        ]
        y0 = 35
        for j, line in enumerate(lines):
            cv2.putText(imu_panel, line, (20, y0 + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)
        canvas[PANEL_H:, PANEL_W:] = imu_panel

        writer.write(canvas)
        frame_count += 1

    writer.release()
    elapsed = time.monotonic() - start
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"  Saved {frame_count} frames ({fps:.1f} fps)")
    print(f"  Output: {output_path}")
    return output_path


def record_imu_video(
    node: PerceptionTestNode,
    duration: float,
    output_dir: str,
) -> str:
    """Record IMU visualization from ROS2 topics (with calibration).

    Same 4-panel visualization as test_d555_camera.py but using
    ROS2 /d555/imu topic instead of direct pyrealsense2 access.
    IMU data arrives in ROS optical_frame → we transform to robot body.

    Returns the output file path, or empty string on failure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _section(f"Recording IMU Visualization ({duration:.0f}s)")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "perception_imu.mp4")

    # ── Calibration: collect gravity at rest ───────────────────────
    cal_duration = 2.0
    print(f"  Calibrating — keep robot STILL for {cal_duration:.0f}s...")
    node.clear_imu_log()

    cal_start = time.monotonic()
    while time.monotonic() - cal_start < cal_duration:
        rclpy.spin_once(node, timeout_sec=0.01)

    cal_data = node.get_imu_log()
    if len(cal_data) < 50:
        print("  FAIL: Insufficient IMU data during calibration")
        return ""

    cal_arr = np.array(cal_data)
    # ROS realsense2_camera publishes in optical frame: X=right, Y=down, Z=forward
    # Transform to body: X_rob=Z_rs, Y_rob=-X_rs, Z_rob=-Y_rs
    cal_accel_body = np.column_stack([
        cal_arr[:, 3],   # Z_rs → X_rob
        -cal_arr[:, 1],  # -X_rs → Y_rob
        -cal_arr[:, 2],  # -Y_rs → Z_rob
    ])
    cal_gyro_body = np.column_stack([
        cal_arr[:, 6],   # Z_rs → X_rob
        -cal_arr[:, 4],  # -X_rs → Y_rob
        -cal_arr[:, 5],  # -Y_rs → Z_rob
    ])

    cal_g = cal_accel_body.mean(axis=0)
    cal_gyro_bias = cal_gyro_body.mean(axis=0)
    cal_roll = math.atan2(cal_g[1], cal_g[2])
    cal_pitch = math.atan2(-cal_g[0], math.sqrt(cal_g[1]**2 + cal_g[2]**2))

    print(f"  Calibration complete:")
    print(f"    Gravity (body): [{cal_g[0]:+.3f}, {cal_g[1]:+.3f}, {cal_g[2]:+.3f}] m/s²")
    print(f"    Mount offsets:  roll={math.degrees(cal_roll):+.1f}°  "
          f"pitch={math.degrees(cal_pitch):+.1f}°")
    print(f"    Gyro bias:      [{cal_gyro_bias[0]:+.5f}, {cal_gyro_bias[1]:+.5f}, "
          f"{cal_gyro_bias[2]:+.5f}] rad/s")

    input("  Press ENTER to start recording (move robot now)...")

    # ── Phase 1: Collect IMU data ──────────────────────────────────
    print(f"  Phase 1/3: Collecting IMU data ({duration:.0f}s)...")
    node.clear_imu_log()

    rec_start = time.monotonic()
    while time.monotonic() - rec_start < duration:
        rclpy.spin_once(node, timeout_sec=0.005)

    imu_data = node.get_imu_log()
    if len(imu_data) < 50:
        print("  FAIL: Insufficient IMU data")
        return ""

    arr = np.array(imu_data)
    t0 = arr[0, 0]
    timestamps = arr[:, 0] - t0

    # Transform to body frame
    accel_body = np.column_stack([
        arr[:, 3], -arr[:, 1], -arr[:, 2]
    ])
    gyro_body = np.column_stack([
        arr[:, 6], -arr[:, 4], -arr[:, 5]
    ]) - cal_gyro_bias

    print(f"  Collected {len(arr)} IMU samples")

    # ── Phase 2: Orientation estimation ────────────────────────────
    print("  Phase 2/3: Estimating orientation...")
    video_fps = 30
    total_t = timestamps[-1]
    n_frames = max(int(total_t * video_fps), 1)
    t_vid = np.linspace(0, total_t, n_frames)

    accel_i = np.column_stack([
        np.interp(t_vid, timestamps, accel_body[:, k]) for k in range(3)
    ])
    gyro_i = np.column_stack([
        np.interp(t_vid, timestamps, gyro_body[:, k]) for k in range(3)
    ])

    comp_alpha = 0.98
    roll = np.zeros(n_frames)
    pitch = np.zeros(n_frames)
    yaw = np.zeros(n_frames)

    ax0, ay0, az0 = accel_i[0]
    roll[0] = math.atan2(ay0, az0) - cal_roll
    pitch[0] = math.atan2(-ax0, math.sqrt(ay0**2 + az0**2)) - cal_pitch

    for i in range(1, n_frames):
        dt = t_vid[i] - t_vid[i - 1]
        axv, ayv, azv = accel_i[i]
        gx, gy, gz = gyro_i[i]
        r_acc = math.atan2(ayv, azv) - cal_roll
        p_acc = math.atan2(-axv, math.sqrt(ayv**2 + azv**2)) - cal_pitch
        roll[i] = comp_alpha * (roll[i - 1] + gx * dt) + (1 - comp_alpha) * r_acc
        pitch[i] = comp_alpha * (pitch[i - 1] + gy * dt) + (1 - comp_alpha) * p_acc
        yaw[i] = yaw[i - 1] + gz * dt

    roll_d, pitch_d, yaw_d = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    # ── Phase 3: Render video frames ──────────────────────────────
    print(f"  Phase 3/3: Rendering {n_frames} frames...")

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_eul = fig.add_subplot(2, 2, 2)
    ax_acc = fig.add_subplot(2, 2, 3)
    ax_gyr = fig.add_subplot(2, 2, 4)
    fig.suptitle("D555 IMU — Robot Body Frame (X=fwd, Y=left, Z=up)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    fig.canvas.draw()
    canvas_w, canvas_h = fig.canvas.get_width_height()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, video_fps,
                             (canvas_w, canvas_h))

    euler_lo = min(roll_d.min(), pitch_d.min(), yaw_d.min()) - 5
    euler_hi = max(roll_d.max(), pitch_d.max(), yaw_d.max()) + 5

    for i in range(n_frames):
        s = slice(0, i + 1)
        t = t_vid[i]

        # 3D orientation
        ax_3d.cla()
        _draw_orientation(ax_3d, roll[i], pitch[i], yaw[i])
        ax_3d.set_title(f"Orientation  t = {t:.2f} s")

        # Euler angles
        ax_eul.cla()
        ax_eul.plot(t_vid[s], roll_d[s], "r-", label="Roll", linewidth=1.2)
        ax_eul.plot(t_vid[s], pitch_d[s], "g-", label="Pitch", linewidth=1.2)
        ax_eul.plot(t_vid[s], yaw_d[s], "b-", label="Yaw", linewidth=1.2)
        ax_eul.set_xlim(0, total_t)
        ax_eul.set_ylim(euler_lo, euler_hi)
        ax_eul.set_xlabel("Time (s)")
        ax_eul.set_ylabel("Angle (deg)")
        ax_eul.set_title("Euler Angles")
        ax_eul.legend(loc="upper right", fontsize=8)
        ax_eul.grid(True, alpha=0.3)

        # Accelerometer
        ax_acc.cla()
        ax_acc.plot(t_vid[s], accel_i[s, 0], "r-", label="X (fwd)", linewidth=1)
        ax_acc.plot(t_vid[s], accel_i[s, 1], "g-", label="Y (left)", linewidth=1)
        ax_acc.plot(t_vid[s], accel_i[s, 2], "b-", label="Z (up)", linewidth=1)
        ax_acc.set_xlim(0, total_t)
        ax_acc.set_xlabel("Time (s)")
        ax_acc.set_ylabel("m/s²")
        ax_acc.set_title("Accelerometer (robot frame)")
        ax_acc.legend(loc="upper right", fontsize=8)
        ax_acc.grid(True, alpha=0.3)

        # Gyroscope
        ax_gyr.cla()
        ax_gyr.plot(t_vid[s], gyro_i[s, 0], "r-", label="X (fwd)", linewidth=1)
        ax_gyr.plot(t_vid[s], gyro_i[s, 1], "g-", label="Y (left)", linewidth=1)
        ax_gyr.plot(t_vid[s], gyro_i[s, 2], "b-", label="Z (up)", linewidth=1)
        ax_gyr.set_xlim(0, total_t)
        ax_gyr.set_xlabel("Time (s)")
        ax_gyr.set_ylabel("rad/s")
        ax_gyr.set_title("Gyroscope (robot frame)")
        ax_gyr.legend(loc="upper right", fontsize=8)
        ax_gyr.grid(True, alpha=0.3)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba()).reshape(canvas_h, canvas_w, 4)
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))

        if (i + 1) % 30 == 0 or i == n_frames - 1:
            print(f"    {i + 1}/{n_frames} frames rendered")

    writer.release()
    plt.close(fig)
    print(f"  Saved IMU visualization → {output_path}")
    return output_path


def _draw_orientation(ax3d, r, p, y):
    """Draw rotated 3D coordinate axes in robot body frame."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    rot = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ])

    origin = np.zeros(3)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    labels = ["X (fwd)", "Y (left)", "Z (up)"]

    for j in range(3):
        axis = np.zeros(3)
        axis[j] = 1.0
        end = rot @ axis
        ax3d.quiver(*origin, *end, color=colors[j],
                    arrow_length_ratio=0.15, linewidth=2.5)
        ax3d.text(*(end * 1.25), labels[j], color=colors[j],
                  fontsize=9, fontweight="bold", ha="center")

    ax3d.quiver(0, 0, 0, 0, 0, -1, color="gray", alpha=0.3,
                arrow_length_ratio=0.1, linewidth=1.5)
    ax3d.text(0, 0, -1.3, "g ↓", color="gray", alpha=0.5,
              fontsize=9, ha="center")

    ax3d.set_xlim(-1.5, 1.5)
    ax3d.set_ylim(-1.5, 1.5)
    ax3d.set_zlim(-1.5, 1.5)
    ax3d.set_xlabel("X (fwd)")
    ax3d.set_ylabel("Y (left)")
    ax3d.set_zlabel("Z (up)")
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.view_init(elev=25, azim=-135)


def main():
    parser = argparse.ArgumentParser(
        description="ROS2 perception stack verification & recording",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record perception and IMU videos after topic verification",
    )
    parser.add_argument(
        "--record-only", action="store_true",
        help="Skip topic verification, record videos only",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Recording duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for recorded videos (default: cwd)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = PerceptionTestNode()

    print("=" * 60)
    print("  ROS2 Perception Stack Verification")
    print("  (requires: ros2 launch strafer_perception perception.launch.py)")
    print("=" * 60)

    try:
        if not args.record_only:
            ok = verify_topics(node)
            if not ok and not args.record:
                sys.exit(1)

        if args.record or args.record_only:
            # Spin briefly to fill buffers
            for _ in range(30):
                rclpy.spin_once(node, timeout_sec=0.05)

            perc_path = record_perception_video(
                node, args.duration, args.output_dir)
            imu_path = record_imu_video(
                node, args.duration, args.output_dir)

            _section("Recorded Files")
            if perc_path:
                print(f"  Perception: {perc_path}")
            if imu_path:
                print(f"  IMU:        {imu_path}")
            if not perc_path and not imu_path:
                print("  No videos were recorded.")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
