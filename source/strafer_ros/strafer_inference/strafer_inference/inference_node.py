"""Trained-policy execution backend for strafer_direct missions.

End-to-end runtime: model load at startup (fatal on failure — the
action server is only advertised when a valid policy is in hand and
the autonomy-side dispatcher falls back to nav2 otherwise), DEPTH
observation pipeline, six-source watchdog (goal / IMU /
joint_states / odom / depth / TF), recurrent ``policy.reset()``
triggers on action-server goal accept and on mid-mission goal pose
updates, L1 velocity clamp before ``/strafer/cmd_vel`` publish, and a
``ready`` parameter that flips to ``True`` only after the first
successful inference so operator health checks distinguish "warming
up" (TRT engine cold-start) from "wedged".

Thread safety: the action server lives in a ``ReentrantCallbackGroup``
so ``execute_callback`` can block on the mission while the timer keeps
ticking; the policy mutex serializes ``policy(obs)`` and
``policy.reset()`` per the recurrent hidden-state contract's
thread-safety point.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, Imu, JointState

from strafer_shared.constants import (
    MAX_ANGULAR_VEL,
    MAX_LINEAR_VEL,
    NAV_ANGULAR_VEL,
    NAV_LINEAR_VEL,
)
from strafer_shared.policy_interface import (
    LoadedPolicy,
    PolicyVariant,
    assemble_observation,
    interpret_action,
    load_policy,
)

from .obs_pipeline import (
    body_frame_goal,
    build_raw_obs_dict,
    downsample_depth,
    joint_state_to_wheel_vels,
    l1_clamp_velocity,
    quaternion_to_yaw,
)
from .watchdog import WatchdogTimeouts, stale_sources


def _default_infer_period() -> float:
    """Read the policy step period from strafer_shared at call time.

    Indirected through a function (instead of a module-level import)
    so mock-patching ``strafer_shared.constants.POLICY_SIM_DT`` /
    ``POLICY_DECIMATION`` in tests changes the value the node picks up.
    """
    from strafer_shared.constants import POLICY_DECIMATION, POLICY_SIM_DT

    return POLICY_SIM_DT * POLICY_DECIMATION


def _resolve_vel_caps() -> tuple[float, float]:
    """Mirror strafer_navigation's STRAFER_NAV_VEL_SCALE handling.

    Unset env var keeps the indoor-safety NAV_VEL_SCALE=0.5 cap;
    ``STRAFER_NAV_VEL_SCALE=1.0`` lifts to hardware max for sim.
    Non-numeric / non-positive overrides silently fall back to the
    constants-derived defaults — same behavior as the Nav2 launch's
    ``_resolved_nav_velocities`` so the two backends agree on the
    sim/real envelope without operator intervention.
    """
    raw = os.environ.get("STRAFER_NAV_VEL_SCALE")
    if not raw:
        return NAV_LINEAR_VEL, NAV_ANGULAR_VEL
    try:
        scale = float(raw)
    except ValueError:
        return NAV_LINEAR_VEL, NAV_ANGULAR_VEL
    if scale <= 0.0:
        return NAV_LINEAR_VEL, NAV_ANGULAR_VEL
    return (
        round(MAX_LINEAR_VEL * scale, 4),
        round(MAX_ANGULAR_VEL * scale, 4),
    )


_DEFAULT_VEL_CAP_LINEAR, _DEFAULT_VEL_CAP_ANGULAR = _resolve_vel_caps()


class InferenceNode(Node):
    """DEPTH-variant trained-policy execution node for strafer_direct."""

    def __init__(self, *, parameter_overrides: Optional[list] = None) -> None:
        super().__init__(
            "strafer_inference",
            parameter_overrides=parameter_overrides or [],
        )

        self.declare_parameter("model_path", "")
        self.declare_parameter("policy_variant", "DEPTH")
        self.declare_parameter("infer_period_s", _default_infer_period())
        self.declare_parameter("goal_topic", "/strafer/goal")
        self.declare_parameter("cmd_vel_topic", "/strafer/cmd_vel")
        self.declare_parameter("depth_topic", "/d555/depth/image_rect_raw")
        self.declare_parameter("subgoal_topic", "/strafer/subgoal")
        self.declare_parameter("imu_topic", "/d555/imu/filtered")
        self.declare_parameter("joint_states_topic", "/strafer/joint_states")
        self.declare_parameter("odom_topic", "/strafer/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tf_max_age_s", 0.5)
        self.declare_parameter("goal_timeout_s", 1.0)
        self.declare_parameter("obs_timeout_s", 0.2)
        self.declare_parameter("depth_timeout_s", 0.5)
        self.declare_parameter("vel_cap_linear_m_s", _DEFAULT_VEL_CAP_LINEAR)
        self.declare_parameter("vel_cap_angular_rad_s", _DEFAULT_VEL_CAP_ANGULAR)
        self.declare_parameter("is_mid_mission_reset", True)
        self.declare_parameter("mid_mission_reset_distance_m", 0.05)
        self.declare_parameter("goal_reached_distance_m", 0.25)
        self.declare_parameter("mission_timeout_s", 60.0)
        self.declare_parameter(
            "onnx_providers", [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        # Operator health check: stays False until the first successful
        # inference. Distinguishes "TRT engine still building" from
        # "wedged" — paired with the cold-start log line in _load_policy_from_param.
        self.declare_parameter("ready", False)

        variant_str = self.get_parameter("policy_variant").value
        try:
            self._variant = PolicyVariant[variant_str]
        except KeyError as exc:
            raise ValueError(
                f"policy_variant={variant_str!r} is not a PolicyVariant; "
                f"expected one of {[v.name for v in PolicyVariant]}"
            ) from exc

        # Variant-agnostic feature gating: the depth subscriber, the depth
        # watchdog source, and the goal-shaped obs keys key off what the
        # loaded variant's fields contain rather than a hardcoded DEPTH
        # assumption, so a no-camera / subgoal variant composes without a
        # per-variant branch.
        self._has_depth = any(
            f.key == "depth_image" for f in self._variant.fields
        )
        self._uses_subgoal = any(
            f.key.startswith("subgoal_") for f in self._variant.fields
        )

        self._map_frame: str = self.get_parameter("map_frame").value
        self._base_frame: str = self.get_parameter("base_frame").value
        self._timeouts = WatchdogTimeouts(
            goal=float(self.get_parameter("goal_timeout_s").value),
            imu=float(self.get_parameter("obs_timeout_s").value),
            joint_states=float(self.get_parameter("obs_timeout_s").value),
            odom=float(self.get_parameter("obs_timeout_s").value),
            depth=float(self.get_parameter("depth_timeout_s").value),
            tf=float(self.get_parameter("tf_max_age_s").value),
        )
        self._vel_cap_linear = float(
            self.get_parameter("vel_cap_linear_m_s").value
        )
        self._vel_cap_angular = float(
            self.get_parameter("vel_cap_angular_rad_s").value
        )
        self._is_mid_mission_reset = bool(
            self.get_parameter("is_mid_mission_reset").value
        )
        self._mid_mission_reset_distance_m = float(
            self.get_parameter("mid_mission_reset_distance_m").value
        )
        self._goal_reached_distance_m = float(
            self.get_parameter("goal_reached_distance_m").value
        )
        self._mission_timeout_s = float(
            self.get_parameter("mission_timeout_s").value
        )

        self._last_imu: Optional[Imu] = None
        self._last_imu_rx_t: Optional[float] = None
        self._last_joint_states: Optional[JointState] = None
        self._last_joint_states_rx_t: Optional[float] = None
        self._last_odom: Optional[Odometry] = None
        self._last_odom_rx_t: Optional[float] = None
        self._last_depth_meters: Optional[np.ndarray] = None
        self._last_depth_rx_t: Optional[float] = None
        self._last_goal_map: Optional[PoseStamped] = None
        self._last_goal_rx_t: Optional[float] = None
        self._last_subgoal_map: Optional[PoseStamped] = None
        self._last_subgoal_rx_t: Optional[float] = None
        self._last_action: np.ndarray = np.zeros(3, dtype=np.float32)

        self._policy_lock = threading.Lock()
        self._policy: Optional[LoadedPolicy] = None
        self._policy_load_error: Optional[str] = None
        self._ready_flag = False

        # Subscriber + tick share the default mutex group; the action
        # server lives in its own ReentrantCallbackGroup so a blocking
        # execute_callback does not starve the tick.
        self._default_cb_group = MutuallyExclusiveCallbackGroup()
        self._action_cb_group = ReentrantCallbackGroup()

        goal_topic = self.get_parameter("goal_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        joint_states_topic = self.get_parameter("joint_states_topic").value
        odom_topic = self.get_parameter("odom_topic").value

        self._cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self._imu_sub = self.create_subscription(
            Imu, imu_topic, self._on_imu, 10,
            callback_group=self._default_cb_group,
        )
        self._joint_states_sub = self.create_subscription(
            JointState, joint_states_topic, self._on_joint_states, 10,
            callback_group=self._default_cb_group,
        )
        self._odom_sub = self.create_subscription(
            Odometry, odom_topic, self._on_odom, 10,
            callback_group=self._default_cb_group,
        )
        # Depth is subscribed only when the variant consumes it; a no-camera
        # variant skips the subscriber (and its decode cost) entirely rather
        # than caching frames it never reads.
        self._depth_sub = None
        if self._has_depth:
            self._depth_sub = self.create_subscription(
                Image, depth_topic, self._on_depth, 10,
                callback_group=self._default_cb_group,
            )
        self._goal_sub = self.create_subscription(
            PoseStamped, goal_topic, self._on_goal, 10,
            callback_group=self._default_cb_group,
        )
        # Subgoal variants follow a rolling subgoal pose; it advances every
        # tick, so caching it must NOT trigger the mid-mission hidden-state
        # reset that _on_goal does for the final goal.
        self._subgoal_sub = None
        if self._uses_subgoal:
            subgoal_topic = self.get_parameter("subgoal_topic").value
            self._subgoal_sub = self.create_subscription(
                PoseStamped, subgoal_topic, self._on_subgoal, 10,
                callback_group=self._default_cb_group,
            )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._load_policy_from_param()

        # Action server is advertised only when a policy is loaded;
        # under strafer_direct selection the autonomy client's
        # wait_for_server then times out and the dispatcher falls back
        # to nav2 for the mission.
        self._action_server: Optional[ActionServer] = None
        if self._policy is not None:
            self._action_server = ActionServer(
                self,
                NavigateToPose,
                "navigate_to_pose",
                execute_callback=self._execute_callback,
                goal_callback=self._goal_callback,
                cancel_callback=self._cancel_callback,
                callback_group=self._action_cb_group,
            )

        infer_period = float(self.get_parameter("infer_period_s").value)
        self._timer = self.create_timer(
            infer_period, self._on_tick,
            callback_group=self._default_cb_group,
        )

        self.get_logger().info(
            f"strafer_inference node up: variant={self._variant.name} "
            f"tick={infer_period:.4f}s "
            f"vel_cap=({self._vel_cap_linear:.4f} m/s, "
            f"{self._vel_cap_angular:.4f} rad/s) "
            f"policy_loaded={self._policy is not None} "
            f"is_mid_mission_reset={self._is_mid_mission_reset}"
        )

    def _load_policy_from_param(self) -> None:
        """Load the model artifact pointed at by ``model_path``.

        Empty path or load failure leaves ``self._policy = None`` with
        a recorded error string; the action server is not advertised
        in that state.
        """
        model_path_raw = self.get_parameter("model_path").value
        if not model_path_raw:
            self._policy_load_error = "model_path is empty"
            self.get_logger().error(
                "model_path is empty; refusing to advertise the "
                "navigate_to_pose action server. JetsonRosClient will "
                "see the server as unavailable and must fall back to nav2."
            )
            return

        model_path = Path(model_path_raw)
        if not model_path.is_file():
            self._policy_load_error = f"model_path not found: {model_path}"
            self.get_logger().error(
                f"model_path={model_path} does not exist; refusing to "
                "advertise the navigate_to_pose action server."
            )
            return

        providers_param = self.get_parameter("onnx_providers").value
        if isinstance(providers_param, str):
            onnx_providers = [providers_param]
        else:
            onnx_providers = list(providers_param) if providers_param else None

        # Cold-start surfacing: ONNX Runtime's TRT EP builds the engine
        # on first inference if no .engine sidecar is shipped. On
        # Jetson Orin Nano that takes 10-30 s and looks identical to a
        # wedged node from the outside; log so the operator knows what
        # to expect, then flip the ready param to True after the first
        # successful inference in _on_tick.
        if self._has_depth and model_path.suffix == ".onnx" and onnx_providers and (
            "TensorrtExecutionProvider" in onnx_providers
        ):
            self.get_logger().info(
                f"Building TensorRT engine for {model_path}, may take "
                "~30 s on Jetson Orin Nano if no pre-built .engine "
                "sidecar is shipped. The `ready` parameter flips to True "
                "after the first successful inference."
            )

        try:
            self._policy = load_policy(
                model_path, self._variant,
                onnx_providers=(
                    onnx_providers if model_path.suffix == ".onnx" else None
                ),
            )
            self.get_logger().info(
                f"Loaded policy from {model_path} "
                f"(recurrent={self._policy.is_recurrent})"
            )
        except Exception as exc:
            self._policy_load_error = repr(exc)
            self.get_logger().error(
                f"load_policy({model_path}, {self._variant.name}) failed: "
                f"{exc}. Refusing to advertise the navigate_to_pose "
                "action server."
            )

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_imu(self, msg: Imu) -> None:
        self._last_imu = msg
        self._last_imu_rx_t = time.monotonic()

    def _on_joint_states(self, msg: JointState) -> None:
        self._last_joint_states = msg
        self._last_joint_states_rx_t = time.monotonic()

    def _on_odom(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_t = time.monotonic()

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding != "32FC1":
            self.get_logger().warning(
                f"Dropping depth frame with encoding={msg.encoding!r}; "
                "expected 32FC1"
            )
            return
        arr = np.frombuffer(msg.data, dtype=np.float32)
        try:
            arr = arr.reshape(msg.height, msg.width)
        except ValueError:
            self.get_logger().warning(
                f"Depth frame data length {arr.size} does not match "
                f"{msg.height}x{msg.width}"
            )
            return
        self._last_depth_meters = arr
        self._last_depth_rx_t = time.monotonic()

    def _on_subgoal(self, msg: PoseStamped) -> None:
        # Cache only; the rolling subgoal advances every tick and must not
        # drive the mid-mission hidden-state reset (that is the final goal's
        # job in _on_goal). The rx time is recorded for a later staleness
        # watchdog source.
        self._last_subgoal_map = msg
        self._last_subgoal_rx_t = time.monotonic()

    def _on_goal(self, msg: PoseStamped) -> None:
        previous = self._last_goal_map
        self._last_goal_map = msg
        self._last_goal_rx_t = time.monotonic()

        if (
            self._policy is not None
            and self._is_mid_mission_reset
            and previous is not None
            and self._goal_pose_changed(previous, msg)
        ):
            self.get_logger().info(
                "Mid-mission goal pose update; resetting policy hidden state."
            )
            with self._policy_lock:
                self._policy.reset()

    def _goal_pose_changed(
        self, previous: PoseStamped, current: PoseStamped,
    ) -> bool:
        dx = current.pose.position.x - previous.pose.position.x
        dy = current.pose.position.y - previous.pose.position.y
        return math.hypot(dx, dy) >= self._mid_mission_reset_distance_m

    # ------------------------------------------------------------------
    # Action server
    # ------------------------------------------------------------------

    def _goal_callback(self, goal_request):
        del goal_request
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        del goal_handle
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        if self._policy is None:
            goal_handle.abort()
            return NavigateToPose.Result()

        goal_pose: PoseStamped = goal_handle.request.pose
        self._last_goal_map = goal_pose
        self._last_goal_rx_t = time.monotonic()

        # New mission boundary → reset hidden state per recurrent
        # contract trigger 4.1.
        with self._policy_lock:
            self._policy.reset()

        mission_started_t = time.monotonic()
        feedback = NavigateToPose.Feedback()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.get_logger().info("navigate_to_pose canceled by client.")
                goal_handle.canceled()
                return NavigateToPose.Result()

            distance = self._current_goal_distance()
            if distance is not None:
                feedback.distance_remaining = float(distance)
                goal_handle.publish_feedback(feedback)
                if distance <= self._goal_reached_distance_m:
                    goal_handle.succeed()
                    return NavigateToPose.Result()

            if time.monotonic() - mission_started_t > self._mission_timeout_s:
                self.get_logger().warning(
                    f"navigate_to_pose mission timed out after "
                    f"{self._mission_timeout_s:.1f} s"
                )
                goal_handle.abort()
                return NavigateToPose.Result()

            time.sleep(0.05)

        goal_handle.abort()
        return NavigateToPose.Result()

    def _current_goal_distance(self) -> Optional[float]:
        if self._last_goal_map is None:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None
        dx = self._last_goal_map.pose.position.x - tf.transform.translation.x
        dy = self._last_goal_map.pose.position.y - tf.transform.translation.y
        return float(math.hypot(dx, dy))

    # ------------------------------------------------------------------
    # Inference tick
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        tf_age = self._tf_age_s()
        stale = stale_sources(
            now_monotonic_s=time.monotonic(),
            last_goal_rx_t=self._last_goal_rx_t,
            last_imu_rx_t=self._last_imu_rx_t,
            last_joint_states_rx_t=self._last_joint_states_rx_t,
            last_odom_rx_t=self._last_odom_rx_t,
            last_depth_rx_t=self._last_depth_rx_t,
            tf_age_s=tf_age,
            timeouts=self._timeouts,
            depth_enabled=self._has_depth,
        )
        if stale:
            self.get_logger().warning(
                f"Watchdog tripped, publishing zero twist: stale={stale}"
            )
            self._cmd_vel_pub.publish(Twist())
            return

        if self._policy is None:
            # Watchdog clean but no model loaded: hold the channel idle
            # (no publish). Action server is unadvertised so no missions
            # arrive here.
            return

        obs = self._assemble_observation_or_none()
        if obs is None:
            self._cmd_vel_pub.publish(Twist())
            return

        t0 = time.monotonic_ns()
        with self._policy_lock:
            action = self._policy(obs)
        t_inference_ns = time.monotonic_ns() - t0

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            self.get_logger().error(
                f"policy output shape {action.shape} != (3,); "
                "publishing zero twist."
            )
            self._cmd_vel_pub.publish(Twist())
            return

        self._last_action = action

        vx, vy, omega = interpret_action(action)
        vx, vy, omega = l1_clamp_velocity(
            vx, vy, omega,
            vel_cap_linear_m_s=self._vel_cap_linear,
            vel_cap_angular_rad_s=self._vel_cap_angular,
        )
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = omega
        self._cmd_vel_pub.publish(twist)

        if not self._ready_flag:
            self._ready_flag = True
            self.set_parameters(
                [Parameter("ready", Parameter.Type.BOOL, True)]
            )
            self.get_logger().info(
                f"strafer_inference ready (first inference complete, "
                f"t={t_inference_ns / 1e6:.2f} ms)."
            )

        if self.get_logger().get_effective_level() <= 10:  # DEBUG
            self._log_obs_summary(obs, action, t_inference_ns)

    def _assemble_observation_or_none(self) -> Optional[np.ndarray]:
        # The obs referent is the rolling subgoal for subgoal variants, the
        # final goal otherwise. Depth is required only when the variant has
        # a depth field.
        referent = (
            self._last_subgoal_map if self._uses_subgoal else self._last_goal_map
        )
        if (
            self._last_imu is None
            or self._last_joint_states is None
            or self._last_odom is None
            or referent is None
            or (self._has_depth and self._last_depth_meters is None)
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
            self.get_logger().debug(f"TF lookup failed: {exc}")
            return None

        base_in_map_xy = (
            tf.transform.translation.x,
            tf.transform.translation.y,
        )
        rot = tf.transform.rotation
        base_in_map_yaw = quaternion_to_yaw(rot.x, rot.y, rot.z, rot.w)

        ref_pos = referent.pose.position
        ref_rel, ref_dist, ref_head = body_frame_goal(
            goal_map_xy=(ref_pos.x, ref_pos.y),
            base_in_map_xy=base_in_map_xy,
            base_in_map_yaw=base_in_map_yaw,
        )

        try:
            wheel_vels = joint_state_to_wheel_vels(
                list(self._last_joint_states.name),
                list(self._last_joint_states.velocity),
            )
        except (KeyError, ValueError) as exc:
            self.get_logger().warning(f"JointState parse failed: {exc}")
            return None

        depth_flat = (
            downsample_depth(self._last_depth_meters) if self._has_depth else None
        )

        imu = self._last_imu
        odom = self._last_odom

        raw = build_raw_obs_dict(
            variant=self._variant,
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
            goal_relative_xy=ref_rel,
            goal_distance=ref_dist,
            goal_heading_to_goal=ref_head,
            body_velocity_xy=(
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
            ),
            last_action=self._last_action,
            depth_flat_normalized=depth_flat,
        )
        return assemble_observation(raw, self._variant)

    def _tf_age_s(self) -> Optional[float]:
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        return max(0.0, now - stamp)

    def _log_obs_summary(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        t_inference_ns: int,
    ) -> None:
        digest = hashlib.sha1(obs.tobytes()).hexdigest()[:12]
        depth_stats = ""
        if self._has_depth:
            offset = sum(
                f.dims for f in self._variant.fields if f.key != "depth_image"
            )
            depth_slice = obs[offset:]
            depth_stats = (
                f"depth_mean={float(depth_slice.mean()):.4f} "
                f"depth_min={float(depth_slice.min()):.4f} "
            )
        self.get_logger().debug(
            f"obs_summary hash={digest} dim={obs.shape[0]} "
            f"{depth_stats}"
            f"action={np.array2string(action, precision=4)} "
            f"t_inference_ns={t_inference_ns}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    # CLI / launch parameter overrides come through rclpy.init's argv
    # parser and are picked up automatically by the Node constructor.
    node = InferenceNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
