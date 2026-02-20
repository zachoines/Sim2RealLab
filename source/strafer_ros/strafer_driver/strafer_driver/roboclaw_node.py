"""RoboClaw driver node for the Strafer mecanum robot.

Bridges ROS2 and hardware:
  - Subscribes to ``/strafer/cmd_vel`` (Twist) and converts to per-wheel
    motor commands via ``strafer_shared.mecanum_kinematics``.
  - Publishes ``/strafer/joint_states`` (JointState) at 50 Hz from encoders.
  - Publishes ``/strafer/odom`` (Odometry) at 50 Hz from wheel odometry.
  - Broadcasts ``odom`` -> ``base_link`` TF at 50 Hz.
  - Publishes ``/diagnostics`` with connection state and error counts.

Threading model: single-threaded executor.  The 50 Hz timer callback handles
ALL serial I/O.  The ``cmd_vel`` subscription callback only stores the latest
Twist -- no serial I/O in the callback.

ALL constants from ``strafer_shared.constants``.
ALL kinematics from ``strafer_shared.mecanum_kinematics``.
Do NOT apply ``WHEEL_AXIS_SIGNS`` in this node -- the kinematics functions
handle sign correction internally.
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from tf2_ros import TransformBroadcaster

from strafer_shared.constants import (
    ROBOCLAW_FRONT_ADDRESS,
    ROBOCLAW_REAR_ADDRESS,
    ROBOCLAW_BAUD_RATE,
    ROBOCLAW_FRONT_PORT,
    ROBOCLAW_REAR_PORT,
    ENCODER_TICKS_TO_RADIANS,
    WHEEL_JOINT_NAMES,
)
from strafer_shared.mecanum_kinematics import (
    twist_to_wheel_velocities,
    wheel_vels_to_ticks_per_sec,
    encoder_ticks_to_body_velocity,
)

from strafer_driver.roboclaw_interface import (
    RoboClawInterface,
    RoboClawError,
)


# Watchdog timeout: stop motors if no cmd_vel received within this period.
WATCHDOG_TIMEOUT_SEC = 0.5

# After this many consecutive serial failures, go into error state.
MAX_CONSECUTIVE_FAILURES = 10

# Reconnect interval when in error state.
RECONNECT_INTERVAL_SEC = 2.0


class RoboClawNode(Node):
    """ROS2 driver node for two RoboClaw motor controllers."""

    def __init__(self) -> None:
        super().__init__("roboclaw_node")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("front_port", ROBOCLAW_FRONT_PORT)
        self.declare_parameter("rear_port", ROBOCLAW_REAR_PORT)
        self.declare_parameter("baud_rate", ROBOCLAW_BAUD_RATE)
        self.declare_parameter("publish_rate", 50.0)

        front_port = self.get_parameter("front_port").get_parameter_value().string_value
        rear_port = self.get_parameter("rear_port").get_parameter_value().string_value
        baud_rate = self.get_parameter("baud_rate").get_parameter_value().integer_value
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value

        self.get_logger().info(
            f"RoboClawNode starting: front={front_port} rear={rear_port} "
            f"baud={baud_rate} rate={publish_rate}Hz"
        )

        # ------------------------------------------------------------------
        # RoboClaw interfaces
        # ------------------------------------------------------------------
        self._front = RoboClawInterface(front_port, ROBOCLAW_FRONT_ADDRESS, baud_rate)
        self._rear = RoboClawInterface(rear_port, ROBOCLAW_REAR_ADDRESS, baud_rate)

        try:
            self._front.open()
            self.get_logger().info(f"Front RoboClaw opened on {front_port}")
        except Exception as e:
            self.get_logger().error(f"Failed to open front RoboClaw: {e}")

        try:
            self._rear.open()
            self.get_logger().info(f"Rear RoboClaw opened on {rear_port}")
        except Exception as e:
            self.get_logger().error(f"Failed to open rear RoboClaw: {e}")

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._latest_twist = Twist()  # zero velocity
        self._last_cmd_vel_time = self.get_clock().now()
        self._watchdog_active = False

        # Odometry integration (in odom frame)
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0
        self._last_odom_time: float | None = None

        # Error tracking
        self._consecutive_failures = 0
        self._total_errors = 0
        self._last_success_time = time.monotonic()
        self._in_error_state = False
        self._last_reconnect_time = 0.0

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self._joint_pub = self.create_publisher(
            JointState, "/strafer/joint_states", 10
        )
        self._odom_pub = self.create_publisher(
            Odometry, "/strafer/odom", 10
        )
        self._diag_pub = self.create_publisher(
            DiagnosticStatus, "/diagnostics", 10
        )

        # TF broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # ------------------------------------------------------------------
        # Subscriber
        # ------------------------------------------------------------------
        self._cmd_vel_sub = self.create_subscription(
            Twist,
            "/strafer/cmd_vel",
            self._cmd_vel_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )

        # ------------------------------------------------------------------
        # Timer (main control loop)
        # ------------------------------------------------------------------
        timer_period = 1.0 / publish_rate
        self._timer = self.create_timer(timer_period, self._timer_callback)

        self.get_logger().info("RoboClawNode ready.")

    # ==================================================================
    # Subscription callback (lightweight -- no serial I/O)
    # ==================================================================

    def _cmd_vel_callback(self, msg: Twist) -> None:
        """Store the latest Twist. Serial I/O happens in the timer."""
        self._latest_twist = msg
        self._last_cmd_vel_time = self.get_clock().now()
        self._watchdog_active = False

    # ==================================================================
    # Timer callback (all serial I/O)
    # ==================================================================

    def _timer_callback(self) -> None:
        """50 Hz control loop: send commands, read encoders, publish."""
        now = self.get_clock().now()

        # ------------------------------------------------------------------
        # Error state: attempt reconnect periodically
        # ------------------------------------------------------------------
        if self._in_error_state:
            mono_now = time.monotonic()
            if mono_now - self._last_reconnect_time >= RECONNECT_INTERVAL_SEC:
                self._last_reconnect_time = mono_now
                self.get_logger().warn("Attempting RoboClaw reconnect...")
                front_ok = self._front.reconnect()
                rear_ok = self._rear.reconnect()
                if front_ok and rear_ok:
                    self.get_logger().info("RoboClaw reconnected successfully.")
                    self._in_error_state = False
                    self._consecutive_failures = 0
                else:
                    self.get_logger().error("Reconnect failed.")
            self._publish_diagnostics()
            return

        # ------------------------------------------------------------------
        # Watchdog: stop motors if no cmd_vel within timeout
        # ------------------------------------------------------------------
        time_since_cmd = (now - self._last_cmd_vel_time).nanoseconds / 1e9
        if time_since_cmd > WATCHDOG_TIMEOUT_SEC:
            if not self._watchdog_active:
                self.get_logger().warn(
                    f"No cmd_vel for {time_since_cmd:.2f}s -- stopping motors."
                )
                self._watchdog_active = True
            twist = Twist()  # zero velocity
        else:
            twist = self._latest_twist

        # ------------------------------------------------------------------
        # Convert Twist -> per-wheel ticks/sec
        # ------------------------------------------------------------------
        vx = twist.linear.x
        vy = twist.linear.y
        omega = twist.angular.z

        wheel_vels_rad = twist_to_wheel_velocities(vx, vy, omega)
        wheel_ticks = wheel_vels_to_ticks_per_sec(wheel_vels_rad)
        # wheel_ticks: [FL, FR, RL, RR]

        # ------------------------------------------------------------------
        # Send motor commands
        # ------------------------------------------------------------------
        try:
            self._front.drive_m1m2_speed(
                int(round(wheel_ticks[0])),  # FL = M1
                int(round(wheel_ticks[1])),  # FR = M2
            )
            self._rear.drive_m1m2_speed(
                int(round(wheel_ticks[2])),  # RL = M1
                int(round(wheel_ticks[3])),  # RR = M2
            )
        except (RoboClawError, Exception) as e:
            self._handle_serial_failure(f"Motor command failed: {e}")
            return

        # ------------------------------------------------------------------
        # Read encoders
        # ------------------------------------------------------------------
        try:
            fl_speed, _ = self._front.read_speed_m1()
            fr_speed, _ = self._front.read_speed_m2()
            rl_speed, _ = self._rear.read_speed_m1()
            rr_speed, _ = self._rear.read_speed_m2()

            fl_enc, _ = self._front.read_encoder_m1()
            fr_enc, _ = self._front.read_encoder_m2()
            rl_enc, _ = self._rear.read_encoder_m1()
            rr_enc, _ = self._rear.read_encoder_m2()
        except (RoboClawError, Exception) as e:
            self._handle_serial_failure(f"Encoder read failed: {e}")
            return

        # Serial I/O succeeded
        self._consecutive_failures = 0
        self._last_success_time = time.monotonic()

        # Speed in ticks/sec -> rad/s (for JointState)
        speeds_ticks = np.array([fl_speed, fr_speed, rl_speed, rr_speed], dtype=np.float64)
        speeds_rad = speeds_ticks * ENCODER_TICKS_TO_RADIANS

        # Encoder counts -> cumulative radians (for JointState position)
        positions_rad = [
            fl_enc * ENCODER_TICKS_TO_RADIANS,
            fr_enc * ENCODER_TICKS_TO_RADIANS,
            rl_enc * ENCODER_TICKS_TO_RADIANS,
            rr_enc * ENCODER_TICKS_TO_RADIANS,
        ]

        # ------------------------------------------------------------------
        # Publish JointState
        # ------------------------------------------------------------------
        stamp = now.to_msg()

        js = JointState()
        js.header.stamp = stamp
        js.header.frame_id = ""
        js.name = list(WHEEL_JOINT_NAMES)
        js.velocity = speeds_rad.tolist()
        js.position = positions_rad
        js.effort = []
        self._joint_pub.publish(js)

        # ------------------------------------------------------------------
        # Compute and publish odometry
        # ------------------------------------------------------------------
        vx_body, vy_body, omega_body = encoder_ticks_to_body_velocity(speeds_ticks)

        mono_now = time.monotonic()
        if self._last_odom_time is not None:
            dt = mono_now - self._last_odom_time

            # Integrate in the odom frame
            cos_theta = math.cos(self._odom_theta)
            sin_theta = math.sin(self._odom_theta)
            self._odom_x += (vx_body * cos_theta - vy_body * sin_theta) * dt
            self._odom_y += (vx_body * sin_theta + vy_body * cos_theta) * dt
            self._odom_theta += omega_body * dt
        self._last_odom_time = mono_now

        # Odometry message
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self._odom_x
        odom.pose.pose.position.y = self._odom_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = _yaw_to_quaternion(self._odom_theta)
        odom.twist.twist.linear.x = vx_body
        odom.twist.twist.linear.y = vy_body
        odom.twist.twist.angular.z = omega_body
        self._odom_pub.publish(odom)

        # ------------------------------------------------------------------
        # Broadcast odom -> base_link TF
        # ------------------------------------------------------------------
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = self._odom_x
        tf.transform.translation.y = self._odom_y
        tf.transform.translation.z = 0.0
        tf.transform.rotation = _yaw_to_quaternion(self._odom_theta)
        self._tf_broadcaster.sendTransform(tf)

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------
        self._publish_diagnostics()

    # ==================================================================
    # Helpers
    # ==================================================================

    def _handle_serial_failure(self, msg: str) -> None:
        """Track consecutive failures and escalate to error state."""
        self._consecutive_failures += 1
        self._total_errors += 1
        self.get_logger().warn(
            f"{msg} (consecutive failures: {self._consecutive_failures})"
        )

        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self.get_logger().error(
                f"{MAX_CONSECUTIVE_FAILURES} consecutive failures -- "
                "stopping motors and entering error state."
            )
            try:
                self._front.stop_motors()
            except Exception:
                pass
            try:
                self._rear.stop_motors()
            except Exception:
                pass
            self._in_error_state = True
            self._last_reconnect_time = time.monotonic()

    def _publish_diagnostics(self) -> None:
        """Publish a DiagnosticStatus message."""
        diag = DiagnosticStatus()
        diag.name = "roboclaw_driver"
        diag.hardware_id = "roboclaw_front_rear"

        if self._in_error_state:
            diag.level = DiagnosticStatus.ERROR
            diag.message = "Serial communication failure -- reconnecting"
        elif self._consecutive_failures > 0:
            diag.level = DiagnosticStatus.WARN
            diag.message = f"Intermittent failures ({self._consecutive_failures})"
        else:
            diag.level = DiagnosticStatus.OK
            diag.message = "OK"

        diag.values = [
            KeyValue(key="front_port", value=self._front.port),
            KeyValue(key="rear_port", value=self._rear.port),
            KeyValue(key="front_connected", value=str(self._front.is_open)),
            KeyValue(key="rear_connected", value=str(self._rear.is_open)),
            KeyValue(key="consecutive_failures", value=str(self._consecutive_failures)),
            KeyValue(key="total_errors", value=str(self._total_errors)),
            KeyValue(key="last_success_age_sec", value=f"{time.monotonic() - self._last_success_time:.1f}"),
            KeyValue(key="watchdog_active", value=str(self._watchdog_active)),
        ]

        self._diag_pub.publish(diag)

    def destroy_node(self) -> None:
        """Ensure motors are stopped on shutdown."""
        self.get_logger().info("Shutting down -- stopping motors.")
        try:
            self._front.stop_motors()
        except Exception:
            pass
        try:
            self._rear.stop_motors()
        except Exception:
            pass
        self._front.close()
        self._rear.close()
        super().destroy_node()


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert a yaw angle (radians) to a Quaternion message."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def main(args=None):
    rclpy.init(args=args)
    node = RoboClawNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
