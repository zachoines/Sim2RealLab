"""Pure-Python configuration for the Isaac Sim ROS2 bridge.

This module is importable from any Python environment — it has zero
runtime dependency on Isaac Sim, Kit, or ``omni.*``. The OmniGraph builder
in :mod:`strafer_lab.bridge.graph` consumes an instance of
:class:`BridgeConfig` to drive ``og.Controller.edit(...)`` calls.

Separating config from graph wiring lets us unit-test the config builder
(topic names, frame IDs, prim path resolution) in a lightweight
pxr-free test env without pulling in Kit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from strafer_shared.constants import (
    FRAME_BASE_LINK,
    FRAME_D555_COLOR_OPTICAL,
    FRAME_D555_LINK,
    FRAME_ODOM,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
    TOPIC_CLOCK,
    TOPIC_CMD_VEL,
    TOPIC_COLOR_CAMERA_INFO,
    TOPIC_COLOR_IMAGE,
    TOPIC_DEPTH_CAMERA_INFO,
    TOPIC_DEPTH_IMAGE,
    TOPIC_IMU_FILTERED,
    TOPIC_JOINT_STATES,
    TOPIC_ODOM,
)


@dataclass(frozen=True)
class CameraStreamConfig:
    """Wiring for a single Isaac Sim camera prim → ROS2 image + camera_info.

    ``width`` / ``height`` are the explicit render-product resolution.
    ``IsaacCreateRenderProduct`` does NOT inherit the camera prim's USD
    ``width`` / ``height`` attributes — its own ``inputs:width`` /
    ``inputs:height`` default to Hydra's 1280×720 unless set, which
    overrides the configured TiledCameraCfg resolution and breaks
    ``camera_info`` (publishes 1280×720 with fx/fy at 2× the configured
    pinhole intrinsics). Carrying the resolution on the stream config
    forces the OmniGraph builder to set both fields explicitly.
    """

    camera_prim_path: str
    image_topic: str
    camera_info_topic: str
    frame_id: str
    stream_type: str  # "rgb" or "depth"
    width: int
    height: int


@dataclass(frozen=True)
class BridgeConfig:
    """Full set of topics, prims and frame IDs the bridge should wire.

    Prim paths are resolved against a live USD stage — the env runner is
    expected to pass the concrete ``{ENV_REGEX_NS}`` substitution before
    constructing this config. For a single-env sim-in-the-loop run that
    typically means ``/World/envs/env_0``.
    """

    graph_path: str
    chassis_prim_path: str
    color_camera: CameraStreamConfig
    depth_camera: CameraStreamConfig
    odom_topic: str
    cmd_vel_topic: str
    clock_topic: str
    odom_frame_id: str
    base_frame_id: str
    camera_mount_frame_id: str
    # Proprioception telemetry the trained-policy obs pipeline needs. The
    # bridge publishes the sim articulation's wheel-joint state and the D555
    # IMU sensor here so the Jetson inference node reaches ``ready`` in
    # sim-in-the-loop just as it does on hardware.
    joint_states_topic: str = TOPIC_JOINT_STATES
    imu_topic: str = TOPIC_IMU_FILTERED
    imu_frame_id: str = FRAME_D555_LINK
    # Additional prim paths that should appear in the robot TF tree
    # (published as base_link → d555_link etc.). Empty tuple means the
    # builder only publishes odom → base_link.
    tf_extra_target_prims: tuple[str, ...] = field(default_factory=tuple)
    # Bridge ticks dropped between publishes: the publisher pushes an
    # Image / CameraInfo every ``camera_frame_skip + 1``-th bridge tick (0 =
    # every tick). The runner derives the value so the publish cadence lands on
    # the policy period (set by sim.dt x decimation, NOT render_interval); this
    # field only carries the resolved value.
    camera_frame_skip: int = 0
    # Stop-on-silence watchdog window for the /cmd_vel stream, in **sim
    # seconds** (not wall-clock: under use_sim_time a low real-time factor
    # spaces healthy commands out in wall-time, so a wall window would
    # false-trip between them). Mirrors the RoboClaw driver's 0.5 s command
    # watchdog. 0 disables. Consumed by StraferAsyncPublisher, not the graph.
    cmd_watchdog_sim_s: float = 0.5


def build_default_bridge_config(
    *,
    env_ns: str = "/World/envs/env_0",
    graph_path: str = "/World/ROS2Bridge",
    camera_frame_skip: int = 0,
) -> BridgeConfig:
    """Construct the canonical Strafer bridge config for a single-env run.

    Parameters
    ----------
    env_ns:
        The USD prim path the Isaac Lab env places the robot under. For a
        single-env run this is ``/World/envs/env_0``. Tests pass a custom
        value to verify prim path substitution.
    graph_path:
        Where the OmniGraph prim is created on the stage. Kept configurable
        so a second graph can be spun up without clashing.
    """

    robot_root = f"{env_ns}/Robot"
    chassis_prim = f"{robot_root}/strafer/body_link"
    # Matches D555_PERCEPTION_CAMERA_PRIM_PATH in strafer_lab.tasks.navigation.d555_cfg.
    # The TiledCameraCfg spawns the camera USD prim directly under body_link at
    # this leaf name, so the bridge must target the same path.
    camera_prim = f"{chassis_prim}/d555_camera_perception"

    color_camera = CameraStreamConfig(
        camera_prim_path=camera_prim,
        image_topic=TOPIC_COLOR_IMAGE,
        camera_info_topic=TOPIC_COLOR_CAMERA_INFO,
        frame_id=FRAME_D555_COLOR_OPTICAL,
        stream_type="rgb",
        width=PERCEPTION_WIDTH,
        height=PERCEPTION_HEIGHT,
    )
    depth_camera = CameraStreamConfig(
        camera_prim_path=camera_prim,
        image_topic=TOPIC_DEPTH_IMAGE,
        camera_info_topic=TOPIC_DEPTH_CAMERA_INFO,
        frame_id=FRAME_D555_COLOR_OPTICAL,
        stream_type="depth",
        width=PERCEPTION_WIDTH,
        height=PERCEPTION_HEIGHT,
    )

    return BridgeConfig(
        graph_path=graph_path,
        chassis_prim_path=chassis_prim,
        color_camera=color_camera,
        depth_camera=depth_camera,
        odom_topic=TOPIC_ODOM,
        cmd_vel_topic=TOPIC_CMD_VEL,
        clock_topic=TOPIC_CLOCK,
        odom_frame_id=FRAME_ODOM,
        base_frame_id=FRAME_BASE_LINK,
        camera_mount_frame_id=FRAME_D555_LINK,
        tf_extra_target_prims=(camera_prim,),
        camera_frame_skip=camera_frame_skip,
    )
