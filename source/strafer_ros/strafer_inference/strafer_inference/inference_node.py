"""Trained-policy execution backend for strafer_direct missions.

Pins the topic and action surface only: IMU / joint-state / odometry /
depth subscriptions, a ``/strafer/cmd_vel`` publisher, and a
``navigate_to_pose`` action server parallel to Nav2's. Observation
assembly, model loading, action interpretation, and safety logic are
added in separate commits.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState


class InferenceNode(Node):
    """Node skeleton for the strafer_direct trained-policy backend."""

    def __init__(self) -> None:
        super().__init__("strafer_inference")

        self.declare_parameter("model_path", "")
        self.declare_parameter("policy_variant", "DEPTH")
        self.declare_parameter("infer_period_s", 1.0 / 30.0)
        self.declare_parameter("goal_topic", "/strafer/goal")
        self.declare_parameter("cmd_vel_topic", "/strafer/cmd_vel")
        self.declare_parameter("depth_topic", "/d555/depth/image_rect_raw")
        self.declare_parameter("imu_topic", "/d555/imu/filtered")
        self.declare_parameter("joint_states_topic", "/strafer/joint_states")
        self.declare_parameter("odom_topic", "/strafer/odom")
        self.declare_parameter("tf_max_age_s", 0.5)
        self.declare_parameter("goal_timeout_s", 1.0)
        self.declare_parameter("obs_timeout_s", 0.2)
        self.declare_parameter("depth_timeout_s", 0.5)

        goal_topic = self.get_parameter("goal_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        joint_states_topic = self.get_parameter("joint_states_topic").value
        odom_topic = self.get_parameter("odom_topic").value

        # Stub: created so the topic graph is visible, never published
        # to until the inference loop lands.
        self._cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self._imu_sub = self.create_subscription(
            Imu, imu_topic, self._on_imu, 10,
        )
        self._joint_states_sub = self.create_subscription(
            JointState, joint_states_topic, self._on_joint_states, 10,
        )
        self._odom_sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, 10,
        )
        self._depth_sub = self.create_subscription(
            Image, depth_topic, self._on_depth, 10,
        )
        self._goal_sub = self.create_subscription(
            PoseStamped, goal_topic, self._on_goal, 10,
        )

        # Relative name: the launched node namespace keeps this from
        # colliding with Nav2's /navigate_to_pose when both are up.
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            "navigate_to_pose",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )

        self.get_logger().info(
            "strafer_inference node up: goal=%s cmd_vel=%s depth=%s imu=%s "
            "joint_states=%s odom=%s policy_variant=%s",
            goal_topic, cmd_vel_topic, depth_topic, imu_topic,
            joint_states_topic, odom_topic,
            self.get_parameter("policy_variant").value,
        )

    def _on_imu(self, msg: Imu) -> None:
        del msg

    def _on_joint_states(self, msg: JointState) -> None:
        del msg

    def _on_odom(self, msg: Odometry) -> None:
        del msg

    def _on_depth(self, msg: Image) -> None:
        del msg

    def _on_goal(self, msg: PoseStamped) -> None:
        del msg

    def _goal_callback(self, goal_request):
        del goal_request
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        del goal_handle
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        self.get_logger().warning(
            "navigate_to_pose received but the trained-policy backend is "
            "not wired yet; aborting goal."
        )
        goal_handle.abort()
        return NavigateToPose.Result()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
