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
import time

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
        self.declare_parameter("angular_vel", 0.6)         # rad/s
        self.declare_parameter("total_rotation", 2.0 * math.pi)  # rad
        self.declare_parameter("ramp_s", 0.75)             # ramp up + ramp down each
        self.declare_parameter("startup_delay_s", 5.0)     # wait for stack to settle
        self.declare_parameter("odom_wait_timeout_s", 20.0)
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        # Sim-clock stall detector. The warmup integrates rotation in
        # sim time, so at sub-unity RTF (e.g., ~0.04-0.1 for headless
        # Isaac Sim with cameras) the wall-clock wait per rotation grows
        # proportionally. An absolute wall-clock cap would cut the
        # rotation short in low-RTF runs. Instead we bail only if /clock
        # stops advancing AT ALL over a wall window — that catches a
        # genuinely paused / disconnected bridge without penalizing slow
        # RTF. Default 15s wall is generous: at RTF=0.01 sim still
        # advances 0.15s per 15s wall, well above the 0.001s "essentially
        # zero" threshold.
        self.declare_parameter("clock_stall_bail_wall_s", 15.0)

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
        self._clock_stall_bail_s = float(
            self.get_parameter("clock_stall_bail_wall_s")
            .get_parameter_value()
            .double_value
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
        the wall-clock timeout expires. Wall-clock here because if the
        bridge is up enough to publish /clock it's almost certainly
        publishing /strafer/odom too — and if not, we want to bail out
        cleanly, not hang on a stuck sim clock.
        """
        deadline = time.monotonic() + self._odom_wait_timeout_s
        while rclpy.ok() and not self._odom_seen:
            if time.monotonic() >= deadline:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._odom_seen

    def _sleep_dt(self, seconds: float) -> None:
        """Sleep one publish period while servicing rclpy callbacks.

        Uses wall-clock duration via rclpy.spin_once's timeout — so a
        stuck sim clock can never wedge this sleep. Returns promptly if
        a callback fires.
        """
        rclpy.spin_once(self, timeout_sec=max(seconds, 0.0))

    def _sleep_wall(self, seconds: float) -> None:
        """Sleep ``seconds`` of wall time while servicing callbacks.

        Wall clock by design: the warmup's startup delay and final
        zero-publish hold don't need to scale with sim RTF. At RTF=0.1,
        a 5-second sim-time startup delay would block the operator for
        50 wall seconds before the rotation begins — that's worse UX
        than just waiting 5 real seconds.
        """
        end = time.monotonic() + max(seconds, 0.0)
        while rclpy.ok() and time.monotonic() < end:
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
        self._sleep_wall(self._startup_delay_s)

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

        # Loop termination policy:
        #   - Primary: sim-clock elapsed reaches total_duration_s. The
        #     rotation is integrated in sim time so the robot covers
        #     `total_rotation` rad in the sim view regardless of RTF.
        #     At low RTF this just means the wall-clock wait is longer
        #     (Isaac Sim headless with cameras at ~0.04-0.1 RTF takes
        #     2-5 minutes for a 360 deg sweep — that's the cost of the
        #     warmup at this performance envelope).
        #   - Safety: if /clock stops advancing for clock_stall_bail_wall_s
        #     of wall time, bail. This catches a paused or disconnected
        #     bridge without falsely tripping on slow RTF.
        sim_start_ns = self.get_clock().now().nanoseconds
        last_sim_progress_ns = sim_start_ns
        last_sim_progress_wall_s = time.monotonic()
        # Tiny threshold for "did sim time advance at all" — sub-millisecond
        # noise from /clock republish jitter shouldn't count as progress.
        sim_progress_epsilon_ns = 1_000_000  # 1 ms

        msg = Twist()
        terminated_by = "sim_time_complete"
        while rclpy.ok():
            now_sim_ns = self.get_clock().now().nanoseconds
            sim_elapsed_s = (now_sim_ns - sim_start_ns) / 1e9

            if sim_elapsed_s >= total_duration_s:
                break

            # Stall detection on sim time. If now_sim has advanced past
            # the last-seen mark, update the high-water mark and reset
            # the stall window. If it hasn't, check whether wall clock
            # has run past the bail window since the last advance.
            if now_sim_ns > last_sim_progress_ns + sim_progress_epsilon_ns:
                last_sim_progress_ns = now_sim_ns
                last_sim_progress_wall_s = time.monotonic()
            elif (
                time.monotonic() - last_sim_progress_wall_s
                >= self._clock_stall_bail_s
            ):
                terminated_by = "clock_stall"
                self.get_logger().warn(
                    f"/clock stalled — no sim-time progress in the "
                    f"last {self._clock_stall_bail_s:.1f}s of wall "
                    f"clock. Sim elapsed {sim_elapsed_s:.2f}s of "
                    f"{total_duration_s:.2f}s. Stopping the spin so "
                    f"the launch isn't wedged."
                )
                break

            msg.angular.z = self._target_rate_at(sim_elapsed_s, hold_s)
            self._cmd_pub.publish(msg)
            self._sleep_dt(dt)

        # Explicit zero hold so anything sampling /cmd_vel sees the
        # stop. Uses wall-clock sleeping so it always terminates even
        # if sim time is paused.
        msg.angular.z = 0.0
        zero_hold_iters = max(int(self._publish_hz), 5)
        for _ in range(zero_hold_iters):
            self._cmd_pub.publish(msg)
            self._sleep_dt(dt)

        self.get_logger().info(
            f"Warmup complete (terminated_by={terminated_by})."
        )
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
