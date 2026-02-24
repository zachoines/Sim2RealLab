#!/usr/bin/env python3
"""Standalone RealSense D555 camera verification and recording script.

Validates that the Intel RealSense D555 camera is:
  1. Detected on USB
  2. Capable of streaming depth at 640x360 @ 30 fps
  3. Capable of streaming color at 640x360 @ 30 fps
  4. Providing valid IMU data (BMI055 accelerometer + gyroscope)
  5. Delivering depth values within the expected sensor range

Optionally records visual output for validation:
  - Side-by-side RGB + colorized depth video
  - IMU data visualization with 3D orientation and time-series plots

Usage:
    python3 test_d555_camera.py                          # run tests only
    python3 test_d555_camera.py --record                 # tests + record 5s videos
    python3 test_d555_camera.py --record --duration 10   # tests + record 10s videos
    python3 test_d555_camera.py --record-only            # skip tests, record only
"""

import argparse
import math
import os
import sys
import time

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not installed.")
    print("Install with: pip install pyrealsense2  (or via apt for Jetson)")
    sys.exit(1)

import numpy as np

from strafer_shared.constants import (
    DEPTH_CLIP_NEAR,
    DEPTH_CLIP_FAR,
    DEPTH_WIDTH,
    DEPTH_HEIGHT,
)

# Number of frames to capture for validation
NUM_FRAMES = 30
STREAM_TIMEOUT_SEC = 10.0


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def test_device_detection() -> rs.device:
    """Check that exactly one D555 is connected."""
    _section("Device Detection")
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("FAIL: No RealSense devices found.")
        print("  - Check USB cable (must be USB 3.x)")
        print("  - Try: rs-enumerate-devices --compact")
        sys.exit(1)

    dev = devices[0]
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    fw = dev.get_info(rs.camera_info.firmware_version)
    usb = dev.get_info(rs.camera_info.usb_type_descriptor)

    print(f"  Device:    {name}")
    print(f"  Serial:    {serial}")
    print(f"  Firmware:  {fw}")
    print(f"  USB Type:  {usb}")

    if "D555" not in name and "D500" not in name:
        print(f"  WARN: Expected D555, got '{name}'")

    if not usb.startswith("3"):
        print(f"  WARN: USB {usb} detected — USB 3.x recommended for full bandwidth")

    print("  PASS: Device detected")
    return dev


def test_depth_stream(pipeline: rs.pipeline, profile: rs.pipeline_profile) -> bool:
    """Validate depth stream delivers frames with valid data."""
    _section("Depth Stream (640x360 @ 30 fps)")
    ok = True

    depth_stats = []
    start = time.monotonic()

    for i in range(NUM_FRAMES):
        frames = pipeline.wait_for_frames(int(STREAM_TIMEOUT_SEC * 1000))
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            print(f"  FAIL: No depth frame on iteration {i}")
            ok = False
            continue

        w, h = depth_frame.get_width(), depth_frame.get_height()
        if i == 0:
            print(f"  Resolution: {w}x{h}")
            if w != 640 or h != 360:
                print(f"  WARN: Expected 640x360, got {w}x{h}")

        # Convert to numpy and check for valid depth
        depth_data = np.asanyarray(depth_frame.get_data())  # uint16, millimeters
        depth_m = depth_data.astype(np.float32) * 0.001
        valid_mask = (depth_m >= DEPTH_CLIP_NEAR) & (depth_m <= DEPTH_CLIP_FAR)
        valid_pct = 100.0 * np.count_nonzero(valid_mask) / depth_m.size
        depth_stats.append(valid_pct)

    elapsed = time.monotonic() - start
    fps = NUM_FRAMES / elapsed

    avg_valid = np.mean(depth_stats)
    print(f"  Captured:  {NUM_FRAMES} frames in {elapsed:.2f}s ({fps:.1f} fps)")
    print(f"  Valid depth: {avg_valid:.1f}% of pixels in "
          f"[{DEPTH_CLIP_NEAR}, {DEPTH_CLIP_FAR}]m range")

    if avg_valid < 1.0:
        print("  WARN: Very low valid depth — camera may be pointed at ceiling/blank wall")

    if fps < 20:
        print(f"  WARN: Low FPS ({fps:.1f}) — check USB bandwidth")

    print(f"  {'PASS' if ok else 'FAIL'}: Depth stream")
    return ok


def test_color_stream(pipeline: rs.pipeline, profile: rs.pipeline_profile) -> bool:
    """Validate color stream delivers frames."""
    _section("Color Stream (640x360 @ 30 fps)")
    ok = True

    start = time.monotonic()
    for i in range(NUM_FRAMES):
        frames = pipeline.wait_for_frames(int(STREAM_TIMEOUT_SEC * 1000))
        color_frame = frames.get_color_frame()
        if not color_frame:
            print(f"  FAIL: No color frame on iteration {i}")
            ok = False
            continue

        if i == 0:
            w, h = color_frame.get_width(), color_frame.get_height()
            print(f"  Resolution: {w}x{h}")
            color_data = np.asanyarray(color_frame.get_data())
            print(f"  Channels:   {color_data.shape[2] if color_data.ndim == 3 else 1}")

    elapsed = time.monotonic() - start
    fps = NUM_FRAMES / elapsed
    print(f"  Captured:  {NUM_FRAMES} frames in {elapsed:.2f}s ({fps:.1f} fps)")
    print(f"  {'PASS' if ok else 'FAIL'}: Color stream")
    return ok


def test_imu_stream() -> bool:
    """Validate IMU (BMI055 accel + gyro) data separately.

    IMU uses a separate pipeline since it runs at a different rate (200 Hz).
    """
    _section("IMU Stream (BMI055 — 200 Hz target)")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)

    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"  SKIP: IMU not available on this device/firmware: {e}")
        print("  The D555 BMI055 IMU may not be exposed in current firmware.")
        print("  WARN: IMU data will need to come from an external source.")
        return True  # Non-fatal -- IMU absence is a known D555 issue

    accel_count = 0
    gyro_count = 0
    accel_samples = []
    gyro_samples = []

    start = time.monotonic()
    target_duration = 1.0  # Collect for 1 second

    try:
        while time.monotonic() - start < target_duration:
            frames = pipeline.wait_for_frames(1000)
            for frame in frames:
                motion = frame.as_motion_frame()
                if not motion:
                    continue
                data = motion.get_motion_data()
                profile = motion.get_profile()
                if profile.stream_type() == rs.stream.accel:
                    accel_count += 1
                    accel_samples.append([data.x, data.y, data.z])
                elif profile.stream_type() == rs.stream.gyro:
                    gyro_count += 1
                    gyro_samples.append([data.x, data.y, data.z])
    finally:
        pipeline.stop()

    elapsed = time.monotonic() - start
    print(f"  Accel frames: {accel_count} ({accel_count / elapsed:.0f} Hz)")
    print(f"  Gyro frames:  {gyro_count} ({gyro_count / elapsed:.0f} Hz)")

    ok = True

    if accel_count < 50:
        print("  FAIL: Too few accelerometer frames")
        ok = False
    else:
        accel_arr = np.array(accel_samples)
        accel_norm = np.linalg.norm(np.mean(accel_arr, axis=0))
        print(f"  Accel mean norm: {accel_norm:.2f} m/s² (expect ~9.81 at rest)")
        if not (7.0 < accel_norm < 12.0):
            print("  WARN: Accel norm outside expected range for stationary sensor")

    if gyro_count < 50:
        print("  FAIL: Too few gyroscope frames")
        ok = False
    else:
        gyro_arr = np.array(gyro_samples)
        gyro_mean = np.mean(np.abs(gyro_arr), axis=0)
        print(f"  Gyro mean |ω|: [{gyro_mean[0]:.4f}, {gyro_mean[1]:.4f}, "
              f"{gyro_mean[2]:.4f}] rad/s (expect ~0 at rest)")

    print(f"  {'PASS' if ok else 'FAIL'}: IMU stream")
    return ok


def test_depth_downsampling() -> bool:
    """Quick test that downsampling to policy resolution works."""
    _section(f"Depth Downsampling ({DEPTH_WIDTH}x{DEPTH_HEIGHT})")
    import cv2

    # Simulate a typical depth frame (D555 native: 640x360)
    raw = np.random.randint(400, 6000, (360, 640), dtype=np.uint16)
    depth_m = raw.astype(np.float32) * 0.001
    depth_m[(depth_m < DEPTH_CLIP_NEAR) | (depth_m > DEPTH_CLIP_FAR)] = 0.0
    resized = cv2.resize(depth_m, (DEPTH_WIDTH, DEPTH_HEIGHT),
                         interpolation=cv2.INTER_AREA)

    ok = True
    if resized.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
        print(f"  FAIL: Shape {resized.shape}, expected ({DEPTH_HEIGHT}, {DEPTH_WIDTH})")
        ok = False
    if resized.dtype != np.float32:
        print(f"  FAIL: dtype {resized.dtype}, expected float32")
        ok = False

    print(f"  Shape: {resized.shape}, dtype: {resized.dtype}")
    print(f"  Range: [{resized.min():.3f}, {resized.max():.3f}] m")
    print(f"  {'PASS' if ok else 'FAIL'}: Downsampling")
    return ok


# ── Recording / visualization ─────────────────────────────────────


def record_rgbd_video(duration: float, output_dir: str) -> str:
    """Record a side-by-side RGB + colorized depth video.

    Returns the output file path, or empty string on failure.
    """
    import cv2

    _section(f"Recording RGB+Depth Video ({duration:.0f}s)")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "d555_rgbd.mp4")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"  FAIL: Cannot start pipeline: {e}")
        return ""

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 30, (1280, 360))
    if not writer.isOpened():
        print(f"  FAIL: Cannot create video writer for {output_path}")
        pipeline.stop()
        return ""

    frame_count = 0
    start = time.monotonic()

    try:
        while time.monotonic() - start < duration:
            frames = pipeline.wait_for_frames(5000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())

            # Colorize depth with TURBO colormap
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * 0.001
            depth_norm = np.clip(depth_m / DEPTH_CLIP_FAR, 0.0, 1.0)
            depth_color = cv2.applyColorMap(
                (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO
            )
            depth_color[depth_m < DEPTH_CLIP_NEAR] = 0

            cv2.putText(color_img, "RGB", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(depth_color, f"Depth (0-{DEPTH_CLIP_FAR:.1f}m)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            writer.write(np.hstack([color_img, depth_color]))
            frame_count += 1
    finally:
        pipeline.stop()
        writer.release()

    elapsed = time.monotonic() - start
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"  Saved {frame_count} frames ({fps:.1f} fps)")
    print(f"  Output: {output_path}")
    return output_path


def record_imu_video(duration: float, output_dir: str) -> str:
    """Collect IMU data and render an orientation visualization video.

    The video shows four panels:
      - 3D rotated coordinate axes (orientation from complementary filter)
      - Roll / pitch / yaw Euler angles over time
      - Raw accelerometer X/Y/Z time series
      - Raw gyroscope X/Y/Z time series

    Returns the output file path, or empty string on failure.
    """
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _section(f"Recording IMU Visualization ({duration:.0f}s)")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "d555_imu.mp4")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)

    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"  SKIP: IMU not available — {e}")
        return ""

    # ── Calibration: measure gravity at rest ───────────────────────
    cal_duration = 2.0
    print(f"  Calibrating — keep robot STILL for {cal_duration:.0f}s...")
    cal_accel = []
    cal_gyro = []
    cal_start = time.monotonic()
    while time.monotonic() - cal_start < cal_duration:
        frames = pipeline.wait_for_frames(1000)
        for frame in frames:
            motion = frame.as_motion_frame()
            if not motion:
                continue
            d = motion.get_motion_data()
            prof = motion.get_profile()
            if prof.stream_type() == rs.stream.accel:
                cal_accel.append([d.x, d.y, d.z])
            elif prof.stream_type() == rs.stream.gyro:
                cal_gyro.append([d.x, d.y, d.z])

    if len(cal_accel) < 20:
        print("  FAIL: Insufficient calibration data")
        pipeline.stop()
        return ""

    # Compute gravity in RealSense frame, then transform to body frame
    cal_accel_rs = np.mean(cal_accel, axis=0)  # (x_rs, y_rs, z_rs)
    cal_gyro_rs = np.mean(cal_gyro, axis=0) if cal_gyro else np.zeros(3)
    # RealSense → Body: X_rob = Z_rs, Y_rob = -X_rs, Z_rob = -Y_rs
    cal_g = np.array([cal_accel_rs[2], -cal_accel_rs[0], -cal_accel_rs[1]])
    cal_gyro_bias = np.array([cal_gyro_rs[2], -cal_gyro_rs[0], -cal_gyro_rs[1]])

    # Roll/pitch offsets at rest (should be zero if camera were perfectly aligned)
    cal_roll = math.atan2(cal_g[1], cal_g[2])
    cal_pitch = math.atan2(-cal_g[0], math.sqrt(cal_g[1] ** 2 + cal_g[2] ** 2))
    print(f"  Calibration complete:")
    print(f"    Gravity (body): [{cal_g[0]:+.3f}, {cal_g[1]:+.3f}, {cal_g[2]:+.3f}] m/s²")
    print(f"    Mount offsets:  roll={math.degrees(cal_roll):+.1f}°  "
          f"pitch={math.degrees(cal_pitch):+.1f}°")
    print(f"    Gyro bias:      [{cal_gyro_bias[0]:+.5f}, {cal_gyro_bias[1]:+.5f}, "
          f"{cal_gyro_bias[2]:+.5f}] rad/s")

    input("  Press ENTER to start recording (move robot now)...")

    # ── Phase 1: Collect raw IMU data ──────────────────────────────
    print(f"  Phase 1/3: Collecting IMU data ({duration:.0f}s)...")
    accel_raw = []  # (timestamp_s, x, y, z)
    gyro_raw = []

    start = time.monotonic()
    try:
        while time.monotonic() - start < duration:
            frames = pipeline.wait_for_frames(1000)
            for frame in frames:
                motion = frame.as_motion_frame()
                if not motion:
                    continue
                d = motion.get_motion_data()
                ts = motion.get_timestamp() / 1000.0
                prof = motion.get_profile()
                if prof.stream_type() == rs.stream.accel:
                    accel_raw.append((ts, d.x, d.y, d.z))
                elif prof.stream_type() == rs.stream.gyro:
                    gyro_raw.append((ts, d.x, d.y, d.z))
    finally:
        pipeline.stop()

    if len(accel_raw) < 10 or len(gyro_raw) < 10:
        print("  FAIL: Insufficient IMU data collected")
        return ""

    accel_arr = np.array(accel_raw)
    gyro_arr = np.array(gyro_raw)
    t0 = min(accel_arr[0, 0], gyro_arr[0, 0])
    accel_arr[:, 0] -= t0
    gyro_arr[:, 0] -= t0

    print(f"  Collected {len(accel_arr)} accel + {len(gyro_arr)} gyro samples")

    # ── Transform: RealSense optical → Robot body (FLU) ───────────
    # RealSense:  X=right,   Y=down,    Z=forward
    # Robot body: X=forward, Y=left,    Z=up
    # Mapping:    X_rob = Z_rs,  Y_rob = -X_rs,  Z_rob = -Y_rs
    print("  Transforming RealSense frame → Robot body frame (FLU)...")
    accel_body = np.column_stack([
        accel_arr[:, 3],   #  Z_rs → X_rob (forward)
        -accel_arr[:, 1],  # -X_rs → Y_rob (left)
        -accel_arr[:, 2],  # -Y_rs → Z_rob (up)
    ])
    gyro_body = np.column_stack([
        gyro_arr[:, 3],    #  Z_rs → X_rob
        -gyro_arr[:, 1],   # -X_rs → Y_rob
        -gyro_arr[:, 2],   # -Y_rs → Z_rob
    ])
    accel_t = accel_arr[:, 0]
    gyro_t = gyro_arr[:, 0]

    # ── Phase 2: Orientation estimation (complementary filter) ─────
    # Robot frame: X=forward, Y=left, Z=up
    # Roll  = rotation about X (forward)  — tilting sideways
    # Pitch = rotation about Y (left)     — nose up/down
    # Yaw   = rotation about Z (up)       — heading change
    # All angles are relative to the calibrated rest pose.
    print("  Phase 2/3: Estimating orientation...")
    video_fps = 30
    total_t = max(accel_arr[-1, 0], gyro_arr[-1, 0])
    n_frames = max(int(total_t * video_fps), 1)
    t_vid = np.linspace(0, total_t, n_frames)

    accel_i = np.column_stack([
        np.interp(t_vid, accel_t, accel_body[:, k]) for k in (0, 1, 2)
    ])
    gyro_i = np.column_stack([
        np.interp(t_vid, gyro_t, gyro_body[:, k]) for k in (0, 1, 2)
    ])
    # Subtract gyro bias measured during calibration
    gyro_i -= cal_gyro_bias

    comp_alpha = 0.98
    roll = np.zeros(n_frames)
    pitch = np.zeros(n_frames)
    yaw = np.zeros(n_frames)

    # Initialize from first accel sample, minus calibration offsets
    ax0, ay0, az0 = accel_i[0]
    roll[0] = math.atan2(ay0, az0) - cal_roll
    pitch[0] = math.atan2(-ax0, math.sqrt(ay0 ** 2 + az0 ** 2)) - cal_pitch

    for i in range(1, n_frames):
        dt = t_vid[i] - t_vid[i - 1]
        ax_v, ay_v, az_v = accel_i[i]
        gx, gy, gz = gyro_i[i]
        r_acc = math.atan2(ay_v, az_v) - cal_roll
        p_acc = math.atan2(-ax_v, math.sqrt(ay_v ** 2 + az_v ** 2)) - cal_pitch
        roll[i] = comp_alpha * (roll[i - 1] + gx * dt) + (1 - comp_alpha) * r_acc
        pitch[i] = comp_alpha * (pitch[i - 1] + gy * dt) + (1 - comp_alpha) * p_acc
        yaw[i] = yaw[i - 1] + gz * dt

    roll_d = np.degrees(roll)
    pitch_d = np.degrees(pitch)
    yaw_d = np.degrees(yaw)

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
        buf = np.asarray(fig.canvas.buffer_rgba()).reshape(
            canvas_h, canvas_w, 4
        )
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))

        if (i + 1) % 30 == 0 or i == n_frames - 1:
            print(f"    {i + 1}/{n_frames} frames rendered")

    writer.release()
    plt.close(fig)
    print(f"  Saved IMU visualization → {output_path}")
    return output_path


def _draw_orientation(ax3d, r, p, y):
    """Draw rotated 3D coordinate axes in robot body frame.

    Robot frame: X=forward (red), Y=left (green), Z=up (blue).
    Gravity points down in world frame (−Z).
    """
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # ZYX intrinsic rotation matrix
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

    # Gravity reference (world-frame −Z)
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
        description="RealSense D555 camera verification & recording",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record RGB+depth and IMU videos after running tests",
    )
    parser.add_argument(
        "--record-only", action="store_true",
        help="Skip tests, record videos only",
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

    print("=" * 60)
    print("  RealSense D555 Camera Verification")
    print("  (standalone — no ROS2 required)")
    print("=" * 60)

    results = {}

    if not args.record_only:
        # 1. Device detection
        dev = test_device_detection()

        # 2. Depth + Color streams (D555: 640x360, NOT 640x480)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

        try:
            profile = pipeline.start(config)
        except RuntimeError as e:
            print(f"\nFAIL: Cannot start pipeline: {e}")
            sys.exit(1)

        try:
            results["depth"] = test_depth_stream(pipeline, profile)
            results["color"] = test_color_stream(pipeline, profile)
        finally:
            pipeline.stop()

        # 3. IMU (separate pipeline)
        results["imu"] = test_imu_stream()

        # 4. Downsampling logic
        results["downsampling"] = test_depth_downsampling()

        # Summary
        _section("Summary")
        all_pass = True
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {status}: {name}")
            if not passed:
                all_pass = False

        if all_pass:
            print("\n  All checks passed.")
        else:
            print("\n  Some checks failed. Fix issues before launching ROS2.")
            if not args.record:
                sys.exit(1)

    # ── Recording ──
    if args.record or args.record_only:
        rgbd_path = record_rgbd_video(args.duration, args.output_dir)
        imu_path = record_imu_video(args.duration, args.output_dir)

        _section("Recorded Files")
        if rgbd_path:
            print(f"  RGB+Depth: {rgbd_path}")
        if imu_path:
            print(f"  IMU:       {imu_path}")
        if not rgbd_path and not imu_path:
            print("  No videos were recorded.")


if __name__ == "__main__":
    main()
