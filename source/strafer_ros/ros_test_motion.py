#!/usr/bin/env python3
"""ROS2-based motion test for the Strafer robot.

Publishes Twist commands to /strafer/cmd_vel and reads feedback from
/strafer/odom.  Requires the driver node to be running:

    ros2 run strafer_driver roboclaw_node

Usage:
    python3 ros_test_motion.py                                # all patterns
    python3 ros_test_motion.py --pattern forward              # forward only
    python3 ros_test_motion.py --pattern forward --duration 5
    python3 ros_test_motion.py --speed 2.0                    # 0.4 m/s base
    python3 ros_test_motion.py --pattern circle --duration 10
"""

from __future__ import annotations

import argparse
import math
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# Safety: same defaults as test_motion_patterns.py
DEFAULT_BASE_LINEAR = 0.2  # m/s
DEFAULT_BASE_OMEGA = 0.3  # rad/s
PUBLISH_RATE_HZ = 50


class MotionTestNode(Node):
    """Node that publishes cmd_vel patterns and monitors odom feedback."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ros_test_motion")
        self._args = args
        self._shutdown = False

        # Latest odom feedback
        self._odom_vx = 0.0
        self._odom_vy = 0.0
        self._odom_omega = 0.0
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0
        self._odom_received = False
        self._odom_lock = threading.Lock()

        # Publisher
        self._cmd_pub = self.create_publisher(
            Twist,
            "/strafer/cmd_vel",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )

        # Subscriber
        self._odom_sub = self.create_subscription(
            Odometry, "/strafer/odom", self._odom_callback, 10
        )

    def _odom_callback(self, msg: Odometry) -> None:
        """Store latest odometry for feedback display."""
        with self._odom_lock:
            self._odom_vx = msg.twist.twist.linear.x
            self._odom_vy = msg.twist.twist.linear.y
            self._odom_omega = msg.twist.twist.angular.z
            self._odom_x = msg.pose.pose.position.x
            self._odom_y = msg.pose.pose.position.y
            # Extract yaw from quaternion
            q = msg.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self._odom_theta = math.atan2(siny, cosy)
            self._odom_received = True

    def get_odom_feedback(self) -> str:
        """Format the latest odom as a string."""
        with self._odom_lock:
            if not self._odom_received:
                return "(no odom yet)"
            return (
                f"vel=[{self._odom_vx:+.3f}, {self._odom_vy:+.3f}, {self._odom_omega:+.3f}] "
                f"pos=[{self._odom_x:+.3f}, {self._odom_y:+.3f}] "
                f"yaw={math.degrees(self._odom_theta):+.1f}°"
            )

    def publish_twist(self, vx: float, vy: float, omega: float) -> None:
        """Publish a Twist message."""
        try:
            msg = Twist()
            msg.linear.x = vx
            msg.linear.y = vy
            msg.angular.z = omega
            self._cmd_pub.publish(msg)
        except Exception:
            pass  # context may be torn down

    def stop(self) -> None:
        """Publish zero velocity."""
        self.publish_twist(0.0, 0.0, 0.0)

    def run_patterns(self) -> None:
        """Execute selected motion patterns (runs in the main thread)."""
        args = self._args
        speed = args.speed
        base_lin = DEFAULT_BASE_LINEAR
        base_omega = DEFAULT_BASE_OMEGA
        duration = args.duration

        # --- Pattern definitions ---

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
        print("Strafer ROS2 Motion Test")
        print("=" * 60)
        print(f"  Topics:   /strafer/cmd_vel -> /strafer/odom")
        print(f"  Speed:    {speed:.1f}x  (base={base_lin} m/s)")
        print(f"  Duration: {duration}s per pattern")
        print(f"  Patterns: {[n for n, _ in patterns]}")
        print(f"\n  Press Ctrl+C to stop.")
        print("=" * 60)

        # Wait briefly for odom subscription to connect
        self._spin_for(0.5)
        if not self._odom_received:
            print("\n  WARNING: No odom received yet. Is the driver node running?")
            print("           ros2 run strafer_driver roboclaw_node")

        sleep_sec = 1.0 / PUBLISH_RATE_HZ
        total_steps = 0

        for pattern_name, pattern_fn in patterns:
            if self._shutdown:
                break

            print(f"\n>>> {pattern_name}  ({duration}s)")
            print(
                f"    {'time':>6s}  {'cmd vx':>8s} {'cmd vy':>8s} {'cmd w':>8s}"
                f"  |  odom feedback"
            )
            print(
                f"    {'─' * 6}  {'─' * 8} {'─' * 8} {'─' * 8}"
                f"  |  {'─' * 55}"
            )

            start = self.get_clock().now()
            last_print_t = 0.0

            while not self._shutdown and rclpy.ok():
                t = (self.get_clock().now() - start).nanoseconds / 1e9
                if t >= duration:
                    break

                vx, vy, omega = pattern_fn(t)
                self.publish_twist(vx, vy, omega)

                # Print feedback once per second
                if t - last_print_t >= 1.0:
                    last_print_t = t
                    fb = self.get_odom_feedback()
                    print(
                        f"    {t:5.1f}s  {vx:+8.3f} {vy:+8.3f} {omega:+8.3f}  |  {fb}"
                    )

                total_steps += 1

                # Spin to process odom callbacks, then sleep remainder
                self._spin_for(sleep_sec)

            # Stop between patterns
            self.stop()

            if not self._shutdown and len(patterns) > 1:
                print("    ... pausing 2s before next pattern")
                self._spin_for(2.0)

        # Final stop
        self.stop()
        status = "INTERRUPTED" if self._shutdown else "COMPLETED"
        print(f"\n[{status}] {total_steps} cmd_vel messages published.")

    def _spin_for(self, seconds: float) -> None:
        """Spin the node for a given duration (processes callbacks)."""
        end = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        try:
            while self.get_clock().now().nanoseconds < end and not self._shutdown:
                rclpy.spin_once(self, timeout_sec=0.01)
        except Exception:
            self._shutdown = True


def main():
    parser = argparse.ArgumentParser(
        description="ROS2 motion test for the Strafer robot",
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
    args = parser.parse_args()

    rclpy.init()
    node = MotionTestNode(args)

    try:
        node.run_patterns()
    except KeyboardInterrupt:
        node._shutdown = True
        node.stop()
        print("\n[INTERRUPTED] Ctrl+C")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
