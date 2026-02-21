#!/usr/bin/env python3
"""Test script to verify motion patterns on the real Strafer robot.

Real-robot counterpart of Scripts/test_strafer_env.py (Isaac Sim).
Talks directly to RoboClaw hardware -- no ROS2 driver node needed.

Usage:
    python3 test_motion_patterns.py                          # all patterns, 0.2 m/s
    python3 test_motion_patterns.py --pattern forward        # forward only
    python3 test_motion_patterns.py --speed 0.5              # faster (50% of max)
    python3 test_motion_patterns.py --pattern circle --duration 10
    python3 test_motion_patterns.py --dry-run                # print actions without moving
"""

from __future__ import annotations

import argparse
import signal
import time

import numpy as np

from strafer_shared.constants import (
    ROBOCLAW_FRONT_ADDRESS,
    ROBOCLAW_REAR_ADDRESS,
    ROBOCLAW_BAUD_RATE,
    ROBOCLAW_PID_P,
    ROBOCLAW_PID_I,
    ROBOCLAW_PID_D,
    ROBOCLAW_QPPS,
    ENCODER_TICKS_TO_RADIANS,
    MAX_LINEAR_VEL,
    MAX_ANGULAR_VEL,
)
from strafer_shared.mecanum_kinematics import (
    twist_to_wheel_velocities,
    wheel_vels_to_ticks_per_sec,
    encoder_ticks_to_body_velocity,
)
from strafer_driver.roboclaw_interface import (
    RoboClawInterface,
    RoboClawError,
    detect_roboclaws,
)

# Safety: default base speed is ~13% of max (~0.2 m/s out of ~1.57 m/s)
DEFAULT_BASE_LINEAR = 0.2  # m/s
DEFAULT_BASE_OMEGA = 0.3  # rad/s
COMMAND_RATE_HZ = 50

# Shutdown flag for signal handler
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    print("\n\n!!! Ctrl+C — stopping motors !!!")
    _shutdown = True


def _read_encoder_feedback(front: RoboClawInterface, rear: RoboClawInterface) -> str:
    """Read encoder speeds and compute body velocity."""
    try:
        fl_speed, _ = front.read_speed_m1()
        fr_speed, _ = front.read_speed_m2()
        rl_speed, _ = rear.read_speed_m1()
        rr_speed, _ = rear.read_speed_m2()
        ticks = np.array([fl_speed, fr_speed, rl_speed, rr_speed], dtype=np.float64)
        vx, vy, omega = encoder_ticks_to_body_velocity(ticks)
        return (
            f"vel=[{vx:+.3f}, {vy:+.3f}, {omega:+.3f}] "
            f"ticks=[{fl_speed:+d}, {fr_speed:+d}, {rl_speed:+d}, {rr_speed:+d}]"
        )
    except RoboClawError as e:
        return f"encoder read error: {e}"


def _send_twist(
    front: RoboClawInterface,
    rear: RoboClawInterface,
    vx: float,
    vy: float,
    omega: float,
) -> None:
    """Convert body twist to wheel commands and send to hardware."""
    wheel_vels = twist_to_wheel_velocities(vx, vy, omega)
    wheel_ticks = wheel_vels_to_ticks_per_sec(wheel_vels)
    front.drive_m1m2_speed(int(round(wheel_ticks[0])), int(round(wheel_ticks[1])))
    rear.drive_m1m2_speed(int(round(wheel_ticks[2])), int(round(wheel_ticks[3])))


def _stop_motors(front: RoboClawInterface, rear: RoboClawInterface) -> None:
    """Stop all motors safely."""
    try:
        front.stop_motors()
    except Exception:
        pass
    try:
        rear.stop_motors()
    except Exception:
        pass


def run_patterns(
    front: RoboClawInterface,
    rear: RoboClawInterface,
    args: argparse.Namespace,
) -> None:
    """Execute the selected motion patterns."""
    global _shutdown
    speed = args.speed
    base_lin = DEFAULT_BASE_LINEAR
    base_omega = DEFAULT_BASE_OMEGA
    duration = args.duration
    dry_run = args.dry_run

    # --- Pattern definitions (same structure as sim script) ---

    def pattern_forward(t: float) -> tuple[float, float, float]:
        return (base_lin * speed, 0.0, 0.0)

    def pattern_backward(t: float) -> tuple[float, float, float]:
        return (-base_lin * speed, 0.0, 0.0)

    def pattern_strafe_left(t: float) -> tuple[float, float, float]:
        return (0.0, base_lin * speed, 0.0)

    def pattern_strafe_right(t: float) -> tuple[float, float, float]:
        return (0.0, -base_lin * speed, 0.0)

    def pattern_strafe(t: float) -> tuple[float, float, float]:
        period = 2.0
        phase = (t % (2 * period)) / period
        vy = base_lin * speed if phase < 1.0 else -base_lin * speed
        return (0.0, vy, 0.0)

    def pattern_rotate(t: float) -> tuple[float, float, float]:
        period = 2.0
        phase = (t % (2 * period)) / period
        w = base_omega * speed if phase < 1.0 else -base_omega * speed
        return (0.0, 0.0, w)

    def pattern_circle(t: float) -> tuple[float, float, float]:
        return (base_lin * speed, 0.0, 0.3 * speed)

    def pattern_figure8(t: float) -> tuple[float, float, float]:
        period = 4.0
        phase = (t % (2 * period)) / period
        w = 0.4 * speed if phase < 1.0 else -0.4 * speed
        return (0.4 * speed, 0.0, w)

    pattern_map = {
        "forward": [("Forward", pattern_forward), ("Backward", pattern_backward)],
        "strafe": [("Strafe Left/Right", pattern_strafe)],
        "strafe_left": [("Strafe Left", pattern_strafe_left)],
        "strafe_right": [("Strafe Right", pattern_strafe_right)],
        "rotate": [("Rotate CW/CCW", pattern_rotate)],
        "circle": [("Circle", pattern_circle)],
        "figure8": [("Figure-8", pattern_figure8)],
        "all": [
            ("Forward", pattern_forward),
            ("Backward", pattern_backward),
            ("Strafe Left/Right", pattern_strafe),
            ("Rotate CW/CCW", pattern_rotate),
            ("Circle", pattern_circle),
            ("Figure-8", pattern_figure8),
        ],
    }

    patterns = pattern_map[args.pattern]

    print()
    print("=" * 60)
    print("Strafer Real-Robot Motion Pattern Test")
    print("=" * 60)
    print(f"  Speed scale:  {speed:.1f}x  (base linear={base_lin} m/s)")
    print(f"  Duration:     {duration}s per pattern")
    print(f"  Patterns:     {[n for n, _ in patterns]}")
    print(f"  Max linear:   {MAX_LINEAR_VEL:.2f} m/s")
    print(f"  Max angular:  {MAX_ANGULAR_VEL:.2f} rad/s")
    if dry_run:
        print("  *** DRY RUN — no commands sent ***")
    print(f"\n  Press Ctrl+C to stop at any time.")
    print("=" * 60)

    sleep_sec = 1.0 / COMMAND_RATE_HZ
    total_steps = 0

    for pattern_name, pattern_fn in patterns:
        if _shutdown:
            break

        print(f"\n>>> {pattern_name}  ({duration}s)")
        print(
            f"    {'time':>6s}  {'cmd vx':>8s} {'cmd vy':>8s} {'cmd w':>8s}  |  encoder feedback"
        )
        print(f"    {'─' * 6}  {'─' * 8} {'─' * 8} {'─' * 8}  |  {'─' * 50}")

        start = time.monotonic()
        last_print = 0.0

        while not _shutdown:
            t = time.monotonic() - start
            if t >= duration:
                break

            vx, vy, omega = pattern_fn(t)

            if not dry_run:
                try:
                    _send_twist(front, rear, vx, vy, omega)
                except RoboClawError as e:
                    print(f"    !!! Motor command error: {e}")

            # Print feedback once per second
            if t - last_print >= 1.0:
                last_print = t
                if dry_run:
                    fb = "(dry run)"
                else:
                    fb = _read_encoder_feedback(front, rear)
                print(
                    f"    {t:5.1f}s  {vx:+8.3f} {vy:+8.3f} {omega:+8.3f}  |  {fb}"
                )

            total_steps += 1
            time.sleep(sleep_sec)

        # Stop between patterns
        if not dry_run:
            _stop_motors(front, rear)

        if not _shutdown and len(patterns) > 1:
            print(f"    ... pausing 2s before next pattern")
            pause_end = time.monotonic() + 2.0
            while time.monotonic() < pause_end and not _shutdown:
                if not dry_run:
                    _stop_motors(front, rear)
                time.sleep(0.1)

    # Final stop
    if not dry_run:
        _stop_motors(front, rear)

    status = "INTERRUPTED" if _shutdown else "COMPLETED"
    print(f"\n[{status}] {total_steps} total command steps sent.")


def main():
    parser = argparse.ArgumentParser(
        description="Test motion patterns on the real Strafer robot"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="all",
        choices=[
            "forward",
            "strafe",
            "strafe_left",
            "strafe_right",
            "rotate",
            "circle",
            "figure8",
            "all",
        ],
        help="Motion pattern to test (default: all)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed scale factor: 1.0 = 0.2 m/s base, 2.0 = 0.4 m/s, etc.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration per pattern in seconds (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without sending to the robot",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)

    # Auto-detect RoboClaws
    print("Detecting RoboClaws...")
    detected = detect_roboclaws(
        ROBOCLAW_FRONT_ADDRESS, ROBOCLAW_REAR_ADDRESS, ROBOCLAW_BAUD_RATE
    )

    if not args.dry_run:
        if ROBOCLAW_FRONT_ADDRESS not in detected:
            print("ERROR: Front RoboClaw (0x80) not detected.")
            return 1
        if ROBOCLAW_REAR_ADDRESS not in detected:
            print("ERROR: Rear RoboClaw (0x81) not detected.")
            return 1

    front_port = detected.get(ROBOCLAW_FRONT_ADDRESS, "/dev/null")
    rear_port = detected.get(ROBOCLAW_REAR_ADDRESS, "/dev/null")
    print(f"  Front: {front_port}  Rear: {rear_port}")

    front = RoboClawInterface(front_port, ROBOCLAW_FRONT_ADDRESS, ROBOCLAW_BAUD_RATE)
    rear = RoboClawInterface(rear_port, ROBOCLAW_REAR_ADDRESS, ROBOCLAW_BAUD_RATE)

    try:
        if not args.dry_run:
            front.open()
            rear.open()
            # Set PID on every run (values are stored in RAM only)
            for rc, label in [(front, "Front"), (rear, "Rear")]:
                rc.set_velocity_pid_m1(ROBOCLAW_PID_P, ROBOCLAW_PID_I, ROBOCLAW_PID_D, ROBOCLAW_QPPS)
                rc.set_velocity_pid_m2(ROBOCLAW_PID_P, ROBOCLAW_PID_I, ROBOCLAW_PID_D, ROBOCLAW_QPPS)
                print(f"  {label} PID set: P={ROBOCLAW_PID_P} I={ROBOCLAW_PID_I} D={ROBOCLAW_PID_D} QPPS={ROBOCLAW_QPPS}")
        run_patterns(front, rear, args)
    finally:
        _stop_motors(front, rear)
        front.close()
        rear.close()


if __name__ == "__main__":
    main()
