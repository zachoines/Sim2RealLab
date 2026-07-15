"""Trained-policy execution backend for strafer_direct missions.

End-to-end runtime: model load at startup (fatal on failure — the
action server is only advertised when a valid policy is in hand and
the autonomy-side dispatcher falls back to nav2 otherwise), DEPTH
observation pipeline, six-source watchdog (goal / IMU /
joint_states / odom / depth / TF), recurrent ``policy.reset()``
triggers on every action-server goal accept — goal updates arrive as
newest-goal-wins preempting goals, each its own mission boundary —
L1 velocity clamp before ``/strafer/cmd_vel`` publish, and a
``ready`` parameter that flips to ``True`` only after the first
successful inference so operator health checks distinguish "warming
up" (TRT engine cold-start) from "wedged".

Depth-freshness gate: a depth variant runs at most one inference per
fresh depth frame — a tick whose depth-frame counter has not advanced
since the last policy call is skipped, holding the channel. Skipped
ticks advance no hidden state, matching training's one-depth-one-step
alignment; without it, catch-up ticks under the slow sim depth feed
replay a stale frame into the recurrent state.

Thread safety: the action server lives in a ``ReentrantCallbackGroup``
so ``execute_callback`` can block on the mission while the timer keeps
ticking; the policy mutex serializes ``policy(obs)`` and
``policy.reset()`` per the recurrent hidden-state contract's
thread-safety point. Depth runs in its own callback group, so
``_on_depth`` and the tick execute on separate executor threads: a lock
guards the depth ``(array, rx_t, seq)`` triple, and each tick snapshots
it once so the watchdog, the freshness gate, and obs assembly all read
one coherent frame — a bumped seq paired with the previous array would
silently infer on the wrong frame. Every other cached source is written
by a callback in the default mutex group, serialized against the tick,
so only depth needs the lock.
"""

from __future__ import annotations

import hashlib
import json
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
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu, JointState

from strafer_shared.constants import (
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


# The L1 velocity clamp defaults to the indoor-safety Nav2 cap (a
# NAV_VEL_SCALE fraction of the chassis maximum) on every lane, matching
# strafer_navigation. Override per-node via the vel_cap_* parameters.
_DEFAULT_VEL_CAP_LINEAR = NAV_LINEAR_VEL
_DEFAULT_VEL_CAP_ANGULAR = NAV_ANGULAR_VEL

# Depth uses a SENSOR_DATA-style profile: the freshness gate wants only the
# newest frame, so a dropped frame should skip a tick, not trigger a reliable
# retransmit that worsens congestion while large fragmented frames are already
# being dropped. BEST_EFFORT receives from both a RELIABLE publisher (the sim
# bridge, and realsense2_camera's SYSTEM_DEFAULT image streams) and a
# BEST_EFFORT one (a depth_qos:=SENSOR_DATA override).
_DEPTH_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


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
        self.declare_parameter("cmd_vel_topic", "/strafer/cmd_vel")
        self.declare_parameter(
            "active_goal_topic", "/strafer_inference/active_goal"
        )
        self.declare_parameter("active_goal_keepalive_period_s", 1.0)
        self.declare_parameter("depth_topic", "/d555/depth/image_rect_raw")
        self.declare_parameter("subgoal_topic", "/strafer/subgoal")
        self.declare_parameter("imu_topic", "/d555/imu/filtered")
        self.declare_parameter("joint_states_topic", "/strafer/joint_states")
        self.declare_parameter("odom_topic", "/strafer/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tf_max_age_s", 0.5)
        self.declare_parameter("obs_timeout_s", 0.2)
        self.declare_parameter("depth_timeout_s", 0.5)
        self.declare_parameter("path_timeout_s", 1.0)
        self.declare_parameter("vel_cap_linear_m_s", _DEFAULT_VEL_CAP_LINEAR)
        self.declare_parameter("vel_cap_angular_rad_s", _DEFAULT_VEL_CAP_ANGULAR)
        self.declare_parameter("goal_reached_distance_m", 0.25)
        self.declare_parameter("mission_timeout_s", 60.0)
        self.declare_parameter(
            "onnx_providers", [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.declare_parameter("onnx_intra_op_threads", 1)
        # TRT engine-cache knobs (see _resolve_onnx_providers); the path + its
        # default live in inference.yaml, not code.
        self.declare_parameter("trt_engine_cache_enable", False)
        self.declare_parameter("trt_engine_cache_path", "")
        # Operator health check: stays False until the first successful
        # inference. Distinguishes "TRT engine still building" from
        # "wedged" — paired with the cold-start log line in _load_policy_from_param.
        self.declare_parameter("ready", False)
        # Diagnostic obs dump: when set to a path, each assembled obs is written
        # as one JSONL line for offline train↔deploy parity checks. Empty
        # (default) disables it with zero per-tick overhead. The write happens
        # after the cmd_vel publish so it can never delay the control path; a
        # DEPTH variant still writes a full depth vector per line, so it is not
        # for normal missions.
        self.declare_parameter(
            "obs_dump_path",
            "",
            ParameterDescriptor(
                description=(
                    "Diagnostic only. File for assembled-obs JSONL consumed by "
                    "the parity tooling (scripts/obs_parity.py); empty disables. "
                    "Truncated per launch. Do not enable for normal missions."
                )
            ),
        )

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

        # Diagnostic obs dump (opened once; line-buffered so each tick's line
        # is durable). Disabled leaves _obs_dump_enabled False so _on_tick and
        # obs assembly do no extra work.
        self._obs_dump_fh = None
        self._obs_dump_enabled = False
        self._last_obs_referent_xy: Optional[tuple[float, float]] = None
        obs_dump_path = str(self.get_parameter("obs_dump_path").value).strip()
        if obs_dump_path:
            try:
                # Truncate per launch (not append): under use_sim_time a relaunch
                # resets t_sim, and concatenating two runs into one file would
                # silently contaminate the sim-time join in the parity gate.
                self._obs_dump_fh = open(obs_dump_path, "w", buffering=1)
                self._obs_dump_enabled = True
                self.get_logger().info(
                    f"Diagnostic obs dump ENABLED → {obs_dump_path} (one JSONL "
                    "line per assembled obs, truncated per launch; do NOT enable "
                    "for normal missions)."
                )
            except OSError as exc:
                self.get_logger().error(
                    f"obs_dump_path={obs_dump_path!r} not writable ({exc}); "
                    "obs dump disabled."
                )

        obs_timeout_s = float(self.get_parameter("obs_timeout_s").value)
        # Env override for slow sim sensor feeds; the yaml default is the
        # real-robot value.
        env_obs_timeout = os.environ.get("STRAFER_OBS_TIMEOUT_S", "")
        if env_obs_timeout:
            obs_timeout_s = float(env_obs_timeout)
            self.get_logger().info(
                f"obs_timeout_s overridden to {obs_timeout_s} via "
                "STRAFER_OBS_TIMEOUT_S"
            )
        depth_timeout_s = float(self.get_parameter("depth_timeout_s").value)
        # Depth gets its own override: the sim bridge publishes depth much
        # slower (~3 Hz) than the other feeds, so the 0.5 s default false-trips.
        # Real-robot bringup leaves it unset.
        env_depth_timeout = os.environ.get("STRAFER_DEPTH_TIMEOUT_S", "")
        if env_depth_timeout:
            depth_timeout_s = float(env_depth_timeout)
            self.get_logger().info(
                f"depth_timeout_s overridden to {depth_timeout_s} via "
                "STRAFER_DEPTH_TIMEOUT_S"
            )
        self._timeouts = WatchdogTimeouts(
            imu=obs_timeout_s,
            joint_states=obs_timeout_s,
            odom=obs_timeout_s,
            depth=depth_timeout_s,
            tf=float(self.get_parameter("tf_max_age_s").value),
            path=float(self.get_parameter("path_timeout_s").value),
        )
        self._vel_cap_linear = float(
            self.get_parameter("vel_cap_linear_m_s").value
        )
        self._vel_cap_angular = float(
            self.get_parameter("vel_cap_angular_rad_s").value
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
        # Depth-frame counter (bumped in _on_depth): the tick runs at most
        # one inference per fresh frame. _last_inferred_depth_seq holds the
        # value consumed by the last policy call; -1 lets the first through.
        # It is written and read only by the tick, so it needs no lock.
        self._depth_seq = 0
        self._last_inferred_depth_seq = -1
        # _on_depth runs in its own callback group, concurrent with the tick.
        # This lock guards the (array, rx_t, seq) triple so the tick reads a
        # consistent snapshot; a fresh seq paired with a stale array would
        # silently infer on the wrong frame. Writes replace the array ref
        # (never mutate it), so a snapshotted ref stays valid without a copy.
        self._depth_lock = threading.Lock()
        # Snapshot of _last_depth_meters taken under _depth_lock at the top of
        # each tick; obs assembly reads this, not the live field.
        self._tick_depth_meters: Optional[np.ndarray] = None
        self._last_goal_map: Optional[PoseStamped] = None
        # Count, not a bool: a preempted or cancel-draining goal briefly
        # overlaps its successor, and the exiting goal must not clear
        # goal-source freshness under the newer mission.
        self._active_goal_count = 0
        # Newest accepted goal; a superseded execute aborts. Never
        # cleared, only replaced — a finished successor still supersedes.
        self._current_goal_handle = None
        self._goal_count_lock = threading.Lock()
        self._last_subgoal_map: Optional[PoseStamped] = None
        self._last_subgoal_rx_t: Optional[float] = None
        self._last_action: np.ndarray = np.zeros(3, dtype=np.float32)

        self._policy_lock = threading.Lock()
        self._policy: Optional[LoadedPolicy] = None
        self._policy_load_error: Optional[str] = None
        self._ready_flag = False

        # Small subs + tick share the default mutex group. Depth gets its own
        # group so its ~921 KB deserialize/take runs on a separate executor
        # thread instead of contending for the tick's single serialized slot;
        # the action server lives in its own ReentrantCallbackGroup so a
        # blocking execute_callback does not starve the tick.
        self._default_cb_group = MutuallyExclusiveCallbackGroup()
        self._depth_cb_group = MutuallyExclusiveCallbackGroup()
        self._action_cb_group = ReentrantCallbackGroup()

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        active_goal_topic = self.get_parameter("active_goal_topic").value
        self._active_goal_keepalive_period_s = float(
            self.get_parameter("active_goal_keepalive_period_s").value
        )
        depth_topic = self.get_parameter("depth_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        joint_states_topic = self.get_parameter("joint_states_topic").value
        odom_topic = self.get_parameter("odom_topic").value

        self._cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        # Status telemetry (accepted goal + keep-alive) for the subgoal
        # generator's replan ownership. navigate_to_pose remains the only
        # command channel into this node — nothing subscribes goals here.
        self._active_goal_pub = self.create_publisher(
            PoseStamped, active_goal_topic, 10
        )

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
                Image, depth_topic, self._on_depth, _DEPTH_QOS,
                callback_group=self._depth_cb_group,
            )
        # Subgoal variants follow a rolling subgoal pose that advances
        # every tick. The final goal is not a topic — it arrives through
        # the navigate_to_pose action only.
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
                handle_accepted_callback=self._handle_accepted,
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
            f"policy_loaded={self._policy is not None}"
        )

    def _resolve_onnx_providers(self) -> Optional[list]:
        """Build the ORT provider list from ``onnx_providers`` + TRT cache config.

        The ``onnx_providers`` param stays a plain preference list of strings
        (the pre-existing contract). When ``trt_engine_cache_enable`` is set and
        ``trt_engine_cache_path`` is non-empty, the ``TensorrtExecutionProvider``
        entry is upgraded to a ``(name, options)`` tuple carrying
        ``trt_engine_cache_enable`` / ``trt_engine_cache_path`` so the TRT EP
        persists its first engine build across relaunches. ORT accepts the mixed
        string/tuple list; ``load_policy`` forwards it verbatim. Returns ``None``
        when no providers are configured (ORT's session-wide default).
        """
        providers_param = self.get_parameter("onnx_providers").value
        if isinstance(providers_param, str):
            providers = [providers_param]
        else:
            providers = list(providers_param) if providers_param else None
        if not providers:
            return None

        if not bool(self.get_parameter("trt_engine_cache_enable").value):
            return providers
        cache_path = str(self.get_parameter("trt_engine_cache_path").value).strip()
        if not cache_path:
            return providers
        cache_path = os.path.expanduser(cache_path)
        try:
            os.makedirs(cache_path, exist_ok=True)
        except OSError as exc:
            self.get_logger().warning(
                f"trt_engine_cache_path {cache_path!r} not creatable ({exc}); "
                "TRT will rebuild the engine on every launch."
            )
            return providers

        return [
            (
                p,
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_path,
                },
            )
            if p == "TensorrtExecutionProvider"
            else p
            for p in providers
        ]

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

        dims_error = self._sidecar_obs_dim_mismatch(model_path)
        if dims_error is not None:
            self._policy_load_error = dims_error
            self.get_logger().error(
                f"{dims_error}; refusing to advertise the navigate_to_pose "
                "action server."
            )
            return

        onnx_providers = self._resolve_onnx_providers()
        # A cache-augmented TRT entry is a (name, options) tuple; unwrap for
        # membership checks / logging.
        provider_names = [
            p[0] if isinstance(p, tuple) else p for p in (onnx_providers or [])
        ]

        # Cold-start surfacing: ONNX Runtime's TRT EP builds the engine
        # on first inference if the engine cache is cold. On Jetson Orin
        # Nano that takes ~90 s for DEPTH_SUBGOAL and looks identical to a
        # wedged node from the outside; log so the operator knows what
        # to expect, then flip the ready param to True after the first
        # successful inference in _on_tick.
        if (
            self._has_depth
            and model_path.suffix == ".onnx"
            and "TensorrtExecutionProvider" in provider_names
        ):
            self.get_logger().info(
                f"TensorRT engine for {model_path} builds on first inference "
                "(~90 s for DEPTH_SUBGOAL on Jetson Orin Nano) when the engine "
                "cache is cold; a warm trt_engine_cache_path skips it. The "
                "`ready` parameter flips to True after the first successful "
                "inference."
            )

        try:
            self._policy = load_policy(
                model_path, self._variant,
                onnx_providers=(
                    onnx_providers if model_path.suffix == ".onnx" else None
                ),
                onnx_intra_op_threads=int(
                    self.get_parameter("onnx_intra_op_threads").value
                ),
            )
            self.get_logger().info(
                f"Loaded policy from {model_path} "
                f"(recurrent={self._policy.is_recurrent})"
            )
            # Active-provider surfacing: ORT silently drops a requested
            # TRT/CUDA EP to CPU when its libs are missing. Log what actually
            # bound so an operator can tell GPU-engaged from CPU-fallback.
            active = getattr(self._policy, "active_providers", None)
            if active is not None:
                self.get_logger().info(
                    f"ONNX Runtime active providers (priority order): {active}"
                )
        except Exception as exc:
            self._policy_load_error = repr(exc)
            self.get_logger().error(
                f"load_policy({model_path}, {self._variant.name}) failed: "
                f"{exc}. Refusing to advertise the navigate_to_pose "
                "action server."
            )

    def _sidecar_obs_dim_mismatch(self, model_path: Path) -> Optional[str]:
        """Error string if the model's ``<stem>.json`` sidecar records an
        obs_dim disagreeing with the loaded variant's, else ``None``.

        A stale-resolution artifact keeps its variant name, so load_policy's
        name check passes and the dim mismatch would otherwise surface only at
        the first inference. No sidecar / no recorded obs_dim is not an error;
        a malformed sidecar degrades to an ``unreadable`` string, never a raise
        (this runs during node construction — a raise would kill the process).
        """
        sidecar = model_path.with_suffix(".json")
        if not sidecar.is_file():
            return None
        try:
            payload = json.loads(sidecar.read_text())
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            recorded = payload.get("obs_dim")
            recorded = None if recorded is None else int(recorded)
        except (OSError, ValueError, TypeError) as exc:
            return f"sidecar {sidecar} is unreadable: {exc}"
        if recorded is not None and recorded != self._variant.obs_dim:
            return (
                f"sidecar {sidecar} records obs_dim={recorded} but variant "
                f"{self._variant.name} expects {self._variant.obs_dim} — stale "
                "artifact; re-export for the current depth resolution"
            )
        return None

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
        rx_t = time.monotonic()
        # Publish the frame as one atomic (array, rx_t, seq) update so the
        # concurrent tick never pairs a bumped seq with the previous array.
        # Validation/decode above stays outside the lock — only the triple
        # write is contended.
        with self._depth_lock:
            self._last_depth_meters = arr
            self._last_depth_rx_t = rx_t
            self._depth_seq += 1

    def _on_subgoal(self, msg: PoseStamped) -> None:
        # Cache only; the rolling subgoal advances every tick and must not
        # drive a hidden-state reset (mission boundaries — new or
        # preempting action goals — own that). The rx time is recorded
        # for the subgoal staleness watchdog source.
        self._last_subgoal_map = msg
        self._last_subgoal_rx_t = time.monotonic()

    # ------------------------------------------------------------------
    # Action server
    # ------------------------------------------------------------------

    @property
    def _goal_active(self) -> bool:
        return self._active_goal_count > 0

    def _goal_callback(self, goal_request):
        del goal_request
        return GoalResponse.ACCEPT

    def _handle_accepted(self, goal_handle) -> None:
        # Newest goal wins: taking ownership makes a superseded execute
        # loop abort, so a goal update is just a new action goal.
        with self._goal_count_lock:
            self._current_goal_handle = goal_handle
        goal_handle.execute()

    def _cancel_callback(self, goal_handle):
        del goal_handle
        return CancelResponse.ACCEPT

    def _superseded(self, goal_handle) -> bool:
        current = self._current_goal_handle
        return current is not None and current is not goal_handle

    def _execute_callback(self, goal_handle):
        if self._policy is None:
            goal_handle.abort()
            return NavigateToPose.Result()

        goal_pose: PoseStamped = goal_handle.request.pose

        # Gate, obs-referent write, and counter bump share one lock (also
        # held by _handle_accepted): otherwise a gated goal could write
        # _last_goal_map after a newer goal's, steering it at the wrong
        # pose.
        with self._goal_count_lock:
            superseded = (
                self._current_goal_handle is not None
                and self._current_goal_handle is not goal_handle
            )
            if not superseded:
                self._last_goal_map = goal_pose
                self._active_goal_count += 1
        if superseded:
            # A newer goal was accepted while this execute task was still
            # queued: abort without touching the live mission's goal pose
            # or hidden state.
            goal_handle.abort()
            return NavigateToPose.Result()

        try:
            # New mission boundary → reset hidden state per recurrent
            # contract trigger 4.1.
            with self._policy_lock:
                self._policy.reset()

            self._active_goal_pub.publish(goal_pose)
            last_keepalive_t = time.monotonic()

            # Deadline on the node clock (sim time under use_sim_time) so
            # a low RTF does not shrink the mission budget; the client's
            # /clock-stall detector handles a frozen clock.
            mission_started = self.get_clock().now()
            feedback = NavigateToPose.Feedback()

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info(
                        "navigate_to_pose canceled by client."
                    )
                    goal_handle.canceled()
                    return NavigateToPose.Result()

                if self._superseded(goal_handle):
                    self.get_logger().info(
                        "navigate_to_pose goal preempted by a newer goal."
                    )
                    goal_handle.abort()
                    return NavigateToPose.Result()

                distance = self._current_goal_distance(goal_pose)
                if distance is not None:
                    feedback.distance_remaining = float(distance)
                    goal_handle.publish_feedback(feedback)
                    if distance <= self._goal_reached_distance_m:
                        goal_handle.succeed()
                        return NavigateToPose.Result()

                now_monotonic = time.monotonic()
                if (
                    now_monotonic - last_keepalive_t
                    >= self._active_goal_keepalive_period_s
                    # Re-checked at publish time: a keep-alive landing
                    # after a preempting goal's accept-publish would
                    # briefly retarget the generator at the old goal.
                    and not self._superseded(goal_handle)
                ):
                    self._active_goal_pub.publish(goal_pose)
                    last_keepalive_t = now_monotonic

                elapsed_s = (
                    self.get_clock().now() - mission_started
                ).nanoseconds * 1e-9
                if elapsed_s > self._mission_timeout_s:
                    self.get_logger().warning(
                        f"navigate_to_pose mission timed out after "
                        f"{self._mission_timeout_s:.1f} s (node clock)"
                    )
                    goal_handle.abort()
                    return NavigateToPose.Result()

                time.sleep(0.05)

            goal_handle.abort()
            return NavigateToPose.Result()
        finally:
            with self._goal_count_lock:
                self._active_goal_count -= 1
            # Explicit stop on mission end: consumers hold the last
            # /cmd_vel, so without this the robot keeps the last policy
            # velocity. Skipped on preemption — the successor owns
            # /cmd_vel and a stop would fight its commands.
            if not self._superseded(goal_handle):
                self._cmd_vel_pub.publish(Twist())

    def _current_goal_distance(self, goal_pose: PoseStamped) -> Optional[float]:
        # Distance to the caller's own captured goal, not the shared
        # _last_goal_map: a superseded loop draining for <=50 ms must not
        # evaluate its succeed check against the successor's pose.
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
        dx = goal_pose.pose.position.x - tf.transform.translation.x
        dy = goal_pose.pose.position.y - tf.transform.translation.y
        return float(math.hypot(dx, dy))

    # ------------------------------------------------------------------
    # Inference tick
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        # Consistent snapshot of the depth triple (array, rx_t, seq) under the
        # lock: _on_depth may replace all three concurrently from its own
        # callback group. The rest of the tick reads only these locals plus
        # the tick-owned _last_inferred_depth_seq, so the frame the watchdog,
        # the gate, and obs assembly all see is one coherent frame.
        if self._has_depth:
            with self._depth_lock:
                self._tick_depth_meters = self._last_depth_meters
                depth_rx_t = self._last_depth_rx_t
                depth_seq = self._depth_seq
        else:
            self._tick_depth_meters = None
            depth_rx_t = None
            depth_seq = self._depth_seq  # 0; the gate is unreachable here

        tf_age = self._tf_age_s()
        stale = stale_sources(
            now_monotonic_s=time.monotonic(),
            last_imu_rx_t=self._last_imu_rx_t,
            last_joint_states_rx_t=self._last_joint_states_rx_t,
            last_odom_rx_t=self._last_odom_rx_t,
            last_depth_rx_t=depth_rx_t,
            tf_age_s=tf_age,
            timeouts=self._timeouts,
            depth_enabled=self._has_depth,
            last_subgoal_rx_t=self._last_subgoal_rx_t,
            subgoal_enabled=self._uses_subgoal,
            goal_active=self._goal_active,
        )
        if stale:
            if self._goal_active:
                # A mission is executing but a source is stale — a real
                # fault (e.g. the subgoal stream stopped). Throttled so a
                # persistent stall does not spam at the tick rate.
                self.get_logger().warning(
                    f"Watchdog tripped mid-mission, publishing zero twist: "
                    f"stale={stale}",
                    throttle_duration_sec=1.0,
                )
                self._cmd_vel_pub.publish(Twist())
            else:
                # Idle: publish nothing — /cmd_vel is shared, so between
                # missions it belongs to Nav2 / rotate / teleop.
                self.get_logger().debug(
                    f"Idle, holding cmd_vel; absent sources: {stale}"
                )
            return

        if self._policy is None:
            # Watchdog clean but no model loaded: hold the channel idle
            # (no publish). Action server is unadvertised so no missions
            # arrive here.
            return

        # After the watchdog, so a stale-source zero-twist still preempts; a
        # skipped tick (depth frame unchanged) holds the channel, no publish.
        # Uses the snapshot seq, not the live counter, so a frame landing
        # mid-tick is picked up next tick rather than tearing this one.
        if self._has_depth and depth_seq == self._last_inferred_depth_seq:
            return

        obs = self._assemble_observation_or_none()
        if obs is None:
            self._cmd_vel_pub.publish(Twist())
            return

        # Stamp the dump at assembly time, not post-publish: the obs describes
        # the state now, and TRT cold-start can put ~90 ms between here and the
        # publish. Captured only when dumping.
        dump_t_sim = (
            self.get_clock().now().nanoseconds * 1e-9
            if self._obs_dump_enabled
            else 0.0
        )

        t0 = time.monotonic_ns()
        with self._policy_lock:
            action = self._policy(obs)
        t_inference_ns = time.monotonic_ns() - t0

        # The policy call advanced the recurrent hidden state — consume the
        # snapshotted depth frame so the gate skips ticks until a fresher one
        # arrives. Records the snapshot seq (the frame obs was built from),
        # not the live counter which a mid-tick _on_depth may have advanced.
        # Recorded even on the shape-error path below (the state advanced
        # regardless); depth variants only, others never read it.
        if self._has_depth:
            self._last_inferred_depth_seq = depth_seq

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

        # Diagnostic dump strictly after the publish so it cannot delay control.
        if self._obs_dump_enabled:
            self._dump_obs(obs, dump_t_sim)

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
            or (self._has_depth and self._tick_depth_meters is None)
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

        ref_pos = referent.pose.position
        if self._obs_dump_enabled:
            self._last_obs_referent_xy = (float(ref_pos.x), float(ref_pos.y))
        ref_rel, ref_dist, ref_head = body_frame_goal(
            goal_map_xy=(ref_pos.x, ref_pos.y),
            base_in_map_xy=base_in_map_xy,
            base_in_map_quat=(rot.x, rot.y, rot.z, rot.w),
        )

        try:
            wheel_vels = joint_state_to_wheel_vels(
                list(self._last_joint_states.name),
                list(self._last_joint_states.velocity),
            )
        except (KeyError, ValueError) as exc:
            self.get_logger().warning(f"JointState parse failed: {exc}")
            return None

        # The tick's coherent snapshot, not the live field a concurrent
        # _on_depth may be replacing.
        depth_flat_meters = (
            downsample_depth(self._tick_depth_meters) if self._has_depth else None
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
            depth_flat_meters=depth_flat_meters,
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

    def _dump_obs(self, obs: np.ndarray, t_sim: float) -> None:
        # One JSONL line per assembled obs (t_sim = node clock = sim time under
        # use_sim_time, captured at assembly). Full variant vector, never
        # truncated. A write failure is logged (throttled) but never propagates
        # into the control path.
        if self._obs_dump_fh is None:
            return
        ref = self._last_obs_referent_xy
        record = {
            "t_sim": t_sim,
            "variant": self._variant.name,
            "obs": np.asarray(obs, dtype=np.float32).tolist(),
            "referent": (
                {"x": ref[0], "y": ref[1], "frame": self._map_frame}
                if ref is not None
                else None
            ),
        }
        try:
            self._obs_dump_fh.write(json.dumps(record) + "\n")
        except (OSError, ValueError) as exc:
            self.get_logger().warning(
                f"obs dump write failed: {exc}", throttle_duration_sec=5.0
            )

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
    # Three threads: the blocking action execute_callback, the tick + small
    # subs (default group), and _on_depth (its own group) each get a slot so
    # the heavy depth take never stalls the tick. A burst of overlapping
    # preempting goals (Reentrant execute_callbacks) can transiently occupy all
    # three for one ~50 ms sleep quantum; self-clearing.
    executor = MultiThreadedExecutor(num_threads=3)
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
