"""Trained-policy execution backend for strafer_direct missions.

Wires the DEPTH observation pipeline end-to-end: subscriptions to the
contract inputs, TF-driven map→body-frame goal transform, depth
downsample matching the sim's preprocessing, and an
``assemble_observation(PolicyVariant.DEPTH)`` pass that produces the
4819-dim vector the trained policy expects. Holds the
``navigate_to_pose`` action server parallel to Nav2's and the
``/strafer/cmd_vel`` publisher; cmd_vel is published as zero twist
while the obs pipeline is validated. Model loading, action
interpretation, and safety logic land in separate commits.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState

from strafer_shared.policy_interface import (
    PolicyVariant,
    assemble_observation,
)

from .obs_pipeline import (
    body_frame_goal,
    build_raw_obs_dict,
    downsample_depth,
    joint_state_to_wheel_vels,
    quaternion_to_yaw,
)


def _default_infer_period() -> float:
    """Read the policy step period from strafer_shared at call time.

    Indirected through a function (instead of a module-level import)
    so mock-patching ``strafer_shared.constants.POLICY_SIM_DT`` /
    ``POLICY_DECIMATION`` in tests changes the value the node picks up.
    """
    from strafer_shared.constants import POLICY_DECIMATION, POLICY_SIM_DT

    return POLICY_SIM_DT * POLICY_DECIMATION


class InferenceNode(Node):
    """DEPTH-variant trained-policy execution node for strafer_direct."""

    def __init__(self) -> None:
        super().__init__("strafer_inference")

        self.declare_parameter("model_path", "")
        self.declare_parameter("policy_variant", "DEPTH")
        self.declare_parameter("infer_period_s", _default_infer_period())
        self.declare_parameter("goal_topic", "/strafer/goal")
        self.declare_parameter("cmd_vel_topic", "/strafer/cmd_vel")
        self.declare_parameter("depth_topic", "/d555/depth/image_rect_raw")
        self.declare_parameter("imu_topic", "/d555/imu/filtered")
        self.declare_parameter("joint_states_topic", "/strafer/joint_states")
        self.declare_parameter("odom_topic", "/strafer/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tf_max_age_s", 0.5)
        self.declare_parameter("goal_timeout_s", 1.0)
        self.declare_parameter("obs_timeout_s", 0.2)
        self.declare_parameter("depth_timeout_s", 0.5)

        variant_str = self.get_parameter("policy_variant").value
        try:
            self._variant = PolicyVariant[variant_str]
        except KeyError as exc:
            raise ValueError(
                f"policy_variant={variant_str!r} is not a PolicyVariant; "
                f"expected one of {[v.name for v in PolicyVariant]}"
            ) from exc

        goal_topic = self.get_parameter("goal_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        joint_states_topic = self.get_parameter("joint_states_topic").value
        odom_topic = self.get_parameter("odom_topic").value

        self._map_frame: str = self.get_parameter("map_frame").value
        self._base_frame: str = self.get_parameter("base_frame").value

        self._last_imu: Optional[Imu] = None
        self._last_joint_states: Optional[JointState] = None
        self._last_odom: Optional[Odometry] = None
        self._last_depth_meters: Optional[np.ndarray] = None
        self._last_depth_stamp: Optional[float] = None
        self._last_goal_map: Optional[PoseStamped] = None
        # Raw [-1, 1]^3 policy output, cached for the next tick's
        # last_action obs term. Zero on first tick.
        self._last_action: np.ndarray = np.zeros(3, dtype=np.float32)

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

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

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

        infer_period = float(self.get_parameter("infer_period_s").value)
        self._timer = self.create_timer(infer_period, self._on_tick)

        self.get_logger().info(
            "strafer_inference node up: variant=%s tick=%.4fs goal=%s "
            "cmd_vel=%s depth=%s imu=%s joint_states=%s odom=%s "
            "map_frame=%s base_frame=%s",
            self._variant.name, infer_period, goal_topic, cmd_vel_topic,
            depth_topic, imu_topic, joint_states_topic, odom_topic,
            self._map_frame, self._base_frame,
        )

    # ------------------------------------------------------------------
    # Subscription callbacks. Depth is decoded once on receive so the
    # tick path is branchless; everything else caches the raw msg.
    # ------------------------------------------------------------------

    def _on_imu(self, msg: Imu) -> None:
        self._last_imu = msg

    def _on_joint_states(self, msg: JointState) -> None:
        self._last_joint_states = msg

    def _on_odom(self, msg: Odometry) -> None:
        self._last_odom = msg

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding != "32FC1":
            self.get_logger().warning(
                "Dropping depth frame with encoding=%r; expected 32FC1",
                msg.encoding,
            )
            return
        arr = np.frombuffer(msg.data, dtype=np.float32)
        try:
            arr = arr.reshape(msg.height, msg.width)
        except ValueError:
            self.get_logger().warning(
                "Depth frame data length %d does not match %dx%d",
                arr.size, msg.height, msg.width,
            )
            return
        self._last_depth_meters = arr
        self._last_depth_stamp = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )

    def _on_goal(self, msg: PoseStamped) -> None:
        self._last_goal_map = msg

    # ------------------------------------------------------------------
    # Action server (Nav2-compatible surface)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Inference tick
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        obs = self._try_assemble_observation()
        if obs is None:
            return
        # cmd_vel publishes zero twist until model loading + action
        # interpretation land; the obs pipeline is validated against a
        # sim-in-the-loop rosbag with the policy off.
        self._cmd_vel_pub.publish(Twist())

    def _try_assemble_observation(self) -> Optional[np.ndarray]:
        if (
            self._last_imu is None
            or self._last_joint_states is None
            or self._last_odom is None
            or self._last_depth_meters is None
            or self._last_goal_map is None
        ):
            return None

        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().debug("TF lookup failed: %s", exc)
            return None

        base_in_map_xy = (
            tf.transform.translation.x,
            tf.transform.translation.y,
        )
        rot = tf.transform.rotation
        base_in_map_yaw = quaternion_to_yaw(rot.x, rot.y, rot.z, rot.w)

        goal = self._last_goal_map.pose.position
        goal_rel, goal_dist, goal_head = body_frame_goal(
            goal_map_xy=(goal.x, goal.y),
            base_in_map_xy=base_in_map_xy,
            base_in_map_yaw=base_in_map_yaw,
        )

        try:
            wheel_vels = joint_state_to_wheel_vels(
                list(self._last_joint_states.name),
                list(self._last_joint_states.velocity),
            )
        except (KeyError, ValueError) as exc:
            self.get_logger().warning("JointState parse failed: %s", exc)
            return None

        depth_flat = downsample_depth(self._last_depth_meters)

        imu = self._last_imu
        odom = self._last_odom

        raw = build_raw_obs_dict(
            imu_accel=(
                imu.linear_acceleration.x,
                imu.linear_acceleration.y,
                imu.linear_acceleration.z,
            ),
            imu_gyro=(
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ),
            wheel_vels_rad_s=wheel_vels,
            goal_relative_xy=goal_rel,
            goal_distance=goal_dist,
            goal_heading_to_goal=goal_head,
            body_velocity_xy=(
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
            ),
            last_action=self._last_action,
            depth_flat_normalized=depth_flat,
        )

        t0 = time.monotonic_ns()
        obs = assemble_observation(raw, self._variant)
        t_assemble_ns = time.monotonic_ns() - t0

        if self.get_logger().get_effective_level() <= 10:  # DEBUG
            self._log_obs_summary(obs, depth_flat, t_assemble_ns)

        return obs

    def _log_obs_summary(
        self,
        obs: np.ndarray,
        depth_flat: np.ndarray,
        t_assemble_ns: int,
    ) -> None:
        digest = hashlib.sha1(obs.tobytes()).hexdigest()[:12]
        self.get_logger().debug(
            "obs_summary hash=%s dim=%d depth_mean=%.4f depth_min=%.4f "
            "action=%s t_assemble_ns=%d",
            digest, obs.shape[0],
            float(depth_flat.mean()), float(depth_flat.min()),
            np.array2string(self._last_action, precision=4),
            t_assemble_ns,
        )


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
