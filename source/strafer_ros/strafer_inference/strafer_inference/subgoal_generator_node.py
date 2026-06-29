"""Rolling-subgoal generator node: Nav2 ``/plan`` -> rolling subgoal pose.

Subscribes to Nav2's global plan (``nav_msgs/Path`` in the ``map`` frame),
looks up the robot pose via the same ``map -> base_link`` TF the inference
node uses, runs the pure :class:`RollingSubgoalGenerator` each tick, and
publishes the rolling subgoal as a ``geometry_msgs/PoseStamped`` on a
dedicated topic.

A dedicated topic (not ``/strafer/goal``) is deliberate: the inference
node resets the policy's hidden state when its goal moves past a
threshold, and a rolling subgoal advances every tick -- routing it through
the goal topic would reset a recurrent policy continuously. The published
pose is in the ``map`` frame; the body-frame observation transform the
policy consumes lives in the inference node.

ROS glue only -- all selection math is in :mod:`strafer_inference.generator`.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node

from strafer_shared.constants import SUBGOAL_LOOKAHEAD_M

from .generator import RollingSubgoalGenerator


def _default_update_period() -> float:
    """Read the policy step period from strafer_shared at call time.

    Indirected through a function (not a module-level import) so a test
    patching ``POLICY_SIM_DT`` / ``POLICY_DECIMATION`` changes the value
    the node picks up -- mirrors the inference node.
    """
    from strafer_shared.constants import POLICY_DECIMATION, POLICY_SIM_DT

    return POLICY_SIM_DT * POLICY_DECIMATION


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion for a planar yaw rotation."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class SubgoalGeneratorNode(Node):
    """Follows Nav2's ``/plan`` and emits the rolling subgoal pose."""

    def __init__(self, *, parameter_overrides: Optional[list] = None) -> None:
        super().__init__(
            "strafer_subgoal_generator",
            parameter_overrides=parameter_overrides or [],
        )

        self.declare_parameter("plan_topic", "/plan")
        self.declare_parameter("subgoal_topic", "/strafer/subgoal")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("update_period_s", _default_update_period())
        # Sourced from the shared constant so train and deploy cannot drift.
        self.declare_parameter("lookahead_m", SUBGOAL_LOOKAHEAD_M)
        # 0 (or negative) means "use the path as published" (no truncation).
        self.declare_parameter("max_path_points", 0)

        self._map_frame: str = self.get_parameter("map_frame").value
        self._base_frame: str = self.get_parameter("base_frame").value
        lookahead_m = float(self.get_parameter("lookahead_m").value)
        max_points_param = int(self.get_parameter("max_path_points").value)
        max_points = max_points_param if max_points_param >= 2 else None

        self._generator = RollingSubgoalGenerator(
            lookahead_m=lookahead_m, max_points=max_points
        )

        # Monotonic receipt time of the latest valid plan. Recorded here so
        # a plan-staleness watchdog source can consume it later; this node
        # does not act on it.
        self._last_plan_rx_t: Optional[float] = None

        plan_topic = self.get_parameter("plan_topic").value
        subgoal_topic = self.get_parameter("subgoal_topic").value

        self._subgoal_pub = self.create_publisher(PoseStamped, subgoal_topic, 10)
        self._plan_sub = self.create_subscription(
            Path, plan_topic, self._on_plan, 10
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        update_period = float(self.get_parameter("update_period_s").value)
        self._timer = self.create_timer(update_period, self._on_tick)

        self.get_logger().info(
            f"strafer_subgoal_generator up: plan_topic={plan_topic} "
            f"subgoal_topic={subgoal_topic} lookahead={lookahead_m:.4f} m "
            f"update_period={update_period:.4f} s "
            f"max_path_points={max_points if max_points is not None else 'unbounded'}"
        )

    def _on_plan(self, msg: Path) -> None:
        """Install a freshly published global plan and rewind the cursor."""
        if not msg.poses:
            self.get_logger().warning(
                "Received an empty /plan (planner produced no path); "
                "keeping the previous plan."
            )
            return
        path_xy = np.array(
            [(p.pose.position.x, p.pose.position.y) for p in msg.poses],
            dtype=np.float64,
        )
        self._generator.set_path(path_xy)
        self._last_plan_rx_t = time.monotonic()
        self.get_logger().debug(
            f"New /plan: {len(msg.poses)} poses, total arc "
            f"{self._generator.total_arc:.3f} m; cursor rewound."
        )

    def _on_tick(self) -> None:
        if not self._generator.has_path:
            return  # No plan yet -- do not publish a subgoal.

        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().debug(f"TF lookup failed: {exc}")
            return

        robot_xy = np.array(
            [tf.transform.translation.x, tf.transform.translation.y],
            dtype=np.float64,
        )
        state = self._generator.update(robot_xy)
        if state is None:
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.pose.position.x = float(state.subgoal_xy[0])
        msg.pose.position.y = float(state.subgoal_xy[1])
        msg.pose.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quaternion(state.subgoal_heading)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self._subgoal_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SubgoalGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
