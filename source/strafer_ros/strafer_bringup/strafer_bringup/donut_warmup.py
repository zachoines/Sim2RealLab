#!/usr/bin/env python3
"""One-shot donut-coverage warmup for sim-in-the-loop bringup.

On a cold-mapped session the global costmap has an unknown-cell donut
around base_link: the D555 doesn't observe any cell inside its
``DEPTH_MIN`` (~0.4 m) and any cell behind the robot's footprint at the
spawn pose. Cells outside the donut are populated from a single depth
frame at spawn, which (combined with RTAB-Map's grid discretization)
produces a thin striated wedge of known-free cells in front of the
camera. Nav2 then plans a wavy path through that input on the first
mission of a session.

This node publishes a slow rotation on ``/cmd_vel`` once at bringup so
RTAB-Map collects ~one depth frame per ~3 degrees of yaw across a full
360 deg sweep. After RayTracing fills the per-frame gaps, the resulting
static map at the start pose is far closer to a uniform free disc.

Why a separate node and not a BT decorator:
  - Once-per-session semantics live naturally in a process lifecycle:
    the node spins, then exits.
  - BT.CPP doesn't reliably preserve decorator state across the
    halt/reset that ``nav2_bt_navigator::BtActionServer`` applies
    between goals, so a SingleTrigger inside the navigate-to-pose BT
    fires on every goal in practice rather than once per launch.
  - Rotation velocity is controlled here directly, not via
    ``behavior_server.max_rotational_vel`` (which would also slow the
    recovery Spin in the BT).

Velocity policy:
  - ``angular_vel`` default 0.4 rad/s is well under the ~1.9 rad/s
    sim cap. Slow enough that the mecanum chassis doesn't bounce on
    accel/decel transients.
  - Ramp-up / ramp-down at ``ramp_s`` keeps the start and stop smooth
    rather than stepping straight to the target rate.

Usage (typically launched from ``bringup_sim_in_the_loop.launch.py``):

    ros2 run strafer_bringup donut_warmup --ros-args -p use_sim_time:=true

To disable, pass ``--ros-args -p enabled:=false`` (or use the launch
arg). To skip on a per-run basis without rebuilding, set the
``STRAFER_DONUT_WARMUP_DISABLE=1`` environment variable before launch.

The node exits with status 0 after a successful rotation. If the
bridge never publishes ``/strafer/odom`` within
``odom_wait_timeout_s``, it exits with status 0 too (the warmup is
best-effort — a missing bridge is the operator's problem to fix, not
this node's blocker).
"""

from __future__ import annotations

import math
import os
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


# Same QoS the rest of the strafer stack uses for telemetry topics.
_TELEMETRY_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


class DonutWarmup(Node):
    """Publish a slow one-shot rotation on /cmd_vel, then shut down."""

    def __init__(self) -> None:
        super().__init__("donut_warmup")

        self.declare_parameter("enabled", True)
        self.declare_parameter("angular_vel", 0.4)         # rad/s
        self.declare_parameter("total_rotation", 2.0 * math.pi)  # rad
        self.declare_parameter("ramp_s", 0.75)             # ramp up + ramp down each
        self.declare_parameter("startup_delay_s", 5.0)     # wait for stack to settle
        self.declare_parameter("odom_wait_timeout_s", 20.0)
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        env_disable = os.environ.get("STRAFER_DONUT_WARMUP_DISABLE", "")
        if env_disable.strip() in ("1", "true", "True", "yes"):
            self.get_logger().info(
                "STRAFER_DONUT_WARMUP_DISABLE set — skipping warmup."
            )
            self._disabled = True
            return
        if not self.get_parameter("enabled").get_parameter_value().bool_value:
            self.get_logger().info("enabled:=false — skipping warmup.")
            self._disabled = True
            return
        self._disabled = False

        self._angular_vel = float(
            self.get_parameter("angular_vel").get_parameter_value().double_value
        )
        self._total_rotation = float(
            self.get_parameter("total_rotation").get_parameter_value().double_value
        )
        self._ramp_s = float(
            self.get_parameter("ramp_s").get_parameter_value().double_value
        )
        self._startup_delay_s = float(
            self.get_parameter("startup_delay_s").get_parameter_value().double_value
        )
        self._odom_wait_timeout_s = float(
            self.get_parameter("odom_wait_timeout_s").get_parameter_value().double_value
        )
        self._publish_hz = float(
            self.get_parameter("publish_hz").get_parameter_value().double_value
        )
        cmd_vel_topic = (
            self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        )

        self._odom_seen = False
        self.create_subscription(
            Odometry, "/strafer/odom", self._on_odom, _TELEMETRY_QOS,
        )
        self._cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

    def _on_odom(self, _msg: Odometry) -> None:
        self._odom_seen = True

    def _wait_for_odom(self) -> bool:
        """Block (spinning rclpy) until /strafer/odom is publishing or
        the timeout expires. Returns True iff at least one message was
        received.
        """
        deadline = self.get_clock().now().nanoseconds + int(
            self._odom_wait_timeout_s * 1e9,
        )
        while rclpy.ok() and not self._odom_seen:
            if self.get_clock().now().nanoseconds >= deadline:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._odom_seen

    def _sleep_simtime(self, seconds: float) -> None:
        """Sleep using the node's clock so use_sim_time is honored."""
        end = self.get_clock().now().nanoseconds + int(seconds * 1e9)
        while rclpy.ok() and self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _target_rate_at(self, elapsed_s: float, hold_s: float) -> float:
        """Triangle-trapezoid: ramp up over ramp_s, hold at angular_vel
        for hold_s, ramp down over ramp_s. Total duration is
        2 * ramp_s + hold_s.
        """
        if elapsed_s < self._ramp_s:
            return self._angular_vel * (elapsed_s / max(self._ramp_s, 1e-6))
        if elapsed_s < self._ramp_s + hold_s:
            return self._angular_vel
        ramp_down_elapsed = elapsed_s - (self._ramp_s + hold_s)
        if ramp_down_elapsed < self._ramp_s:
            return self._angular_vel * (
                1.0 - ramp_down_elapsed / max(self._ramp_s, 1e-6)
            )
        return 0.0

    def run(self) -> int:
        if self._disabled:
            return 0

        self.get_logger().info(
            f"Warmup queued: rotate {self._total_rotation:.2f} rad "
            f"at {self._angular_vel:.2f} rad/s (target ~"
            f"{self._estimate_total_duration_s():.1f}s of motion), "
            f"after {self._startup_delay_s:.1f}s startup delay."
        )

        # Wait for the bridge to come up. /strafer/odom is the most
        # reliable readiness signal — if it isn't publishing, /cmd_vel
        # has nothing to drive.
        if not self._wait_for_odom():
            self.get_logger().warn(
                "Timed out waiting for /strafer/odom — skipping warmup."
            )
            return 0
        self._sleep_simtime(self._startup_delay_s)

        # Compute hold duration to deliver the full rotation. Integral
        # of the trapezoid: angular_vel * (hold_s + ramp_s) = total_rotation
        # → hold_s = total_rotation / angular_vel - ramp_s.
        hold_s = max(
            self._total_rotation / max(self._angular_vel, 1e-6) - self._ramp_s,
            0.0,
        )
        total_duration_s = 2.0 * self._ramp_s + hold_s
        dt = 1.0 / max(self._publish_hz, 1.0)

        self.get_logger().info(
            f"Warmup spinning: hold {hold_s:.2f}s, ramps {self._ramp_s:.2f}s, "
            f"total {total_duration_s:.2f}s."
        )

        start_ns = self.get_clock().now().nanoseconds
        msg = Twist()
        while rclpy.ok():
            elapsed_s = (self.get_clock().now().nanoseconds - start_ns) / 1e9
            if elapsed_s >= total_duration_s:
                break
            msg.angular.z = self._target_rate_at(elapsed_s, hold_s)
            self._cmd_pub.publish(msg)
            self._sleep_simtime(dt)

        # Explicit zero hold so the smoother / bridge sees the stop.
        msg.angular.z = 0.0
        for _ in range(int(self._publish_hz)):
            self._cmd_pub.publish(msg)
            self._sleep_simtime(dt)

        self.get_logger().info("Warmup complete.")
        return 0

    def _estimate_total_duration_s(self) -> float:
        hold_s = max(
            self._total_rotation / max(self._angular_vel, 1e-6) - self._ramp_s,
            0.0,
        )
        return 2.0 * self._ramp_s + hold_s


def main(args=None) -> int:
    rclpy.init(args=args)
    try:
        node = DonutWarmup()
        return node.run()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
