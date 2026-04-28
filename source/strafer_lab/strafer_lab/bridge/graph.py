"""OmniGraph builder that wires the Strafer bridge onto a live USD stage.

This module imports ``omni.graph.core`` and therefore must only be invoked
inside a live Kit process (i.e. after ``isaaclab.app.AppLauncher`` has
booted Isaac Sim). The pure-Python config layer lives in
:mod:`strafer_lab.bridge.config` and is safe to import anywhere.

What gets wired
---------------
- **Sim clock** → ``/clock`` (``rosgraph_msgs/Clock``). Drives the
  whole cross-host stack when Jetson-side nodes run with
  ``use_sim_time:=True``, keeping TF stamps monotonic in sim time
  instead of mixing wall-time (RTAB-Map) and sim-time (bridge).
- **Color camera** → ``/d555/color/image_raw`` (``sensor_msgs/Image``)
  + ``/d555/color/camera_info`` (``sensor_msgs/CameraInfo``).
- **Depth camera** → ``/d555/depth/image_rect_raw``
  + ``/d555/depth/camera_info``. Both streams share the same
  ``isaacsim.sensors.camera`` prim via one ``IsaacCreateRenderProduct``
  node per stream.
- **Chassis odometry** → ``/strafer/odom`` (``nav_msgs/Odometry``) via
  ``IsaacComputeOdometry`` → ``ROS2PublishOdometry``.
- **TF tree** → odom → base_link (raw) plus base_link → d555_link via
  ``ROS2PublishTransformTree``.
- **/cmd_vel subscription** → ``ROS2SubscribeTwist``. The subscribed
  linear/angular velocity is read out in the sim runner's env-step loop
  and injected into the Isaac Lab action tensor; this module only creates
  the node.

Deferred (documented, not a bug)
--------------------------------
- **IMU publishing.** Isaac Sim ships no dedicated ``ROS2PublishImu``
  node, and wiring ``IsaacReadIMU`` outputs into the generic
  ``ROS2Publisher`` requires splitting each vec3d / quatd into nested
  dynamic attributes (``inputs:linear_acceleration:x``, ...), which is
  brittle across Isaac Sim versions. RTAB-Map's visual-inertial pipeline
  can run visual-only, which is sufficient to validate the sim-in-the-loop
  harness. Real-robot runs use the D555's embedded IMU unchanged.
- **Virtual /scan.** ``depthimage_to_laserscan`` already runs on the
  Jetson side as part of ``strafer_slam``; we publish the raw depth
  stream and let the Jetson synthesize the scan so the sim path
  exercises the same code as the real path.
"""

from __future__ import annotations

from typing import Any

from strafer_lab.bridge.config import BridgeConfig, CameraStreamConfig


def build_bridge_graph(
    config: BridgeConfig,
    *,
    skip_cameras: bool = False,
    skip_telemetry: bool = False,
) -> Any:
    """Create the ROS2 bridge OmniGraph described by ``config``.

    Returns the ``og.Controller.edit`` graph handle so callers can
    subsequently read or write attributes (e.g. poll the subscribed
    ``/cmd_vel`` linear/angular velocity).

    ``skip_cameras`` omits the render-product + camera publisher chain
    for both color and depth. Useful for a bridge-mode performance
    baseline where the env is spawned without a perception camera — no
    render product means no GPU readback sync, so env.step throughput is
    no longer coupled to the render pipeline.

    ``skip_telemetry`` omits the Clock / Odometry / TransformTree /
    SubscribeTwist nodes. Use when a Python-side publisher
    (``strafer_lab.bridge.async_publisher.StraferAsyncPublisher``) is
    driving those topics instead; leaves the camera chain on OmniGraph
    (cameras are render-product-bound and cannot be driven from Python
    without re-implementing the image serialization path).
    """

    import omni.graph.core as og

    keys = og.Controller.Keys

    graph_path = config.graph_path

    if not skip_telemetry:
        (graph_handle, *_) = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    # Clock — authoritative sim-time for every cross-host ROS 2 node
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    # Odometry
                    ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                    # TF: odom → base_link (raw) and base_link → d555_link (tree)
                    ("PublishOdomToBaseTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                    ("PublishBaseLinkTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                    # /cmd_vel subscription
                    ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ],
                keys.SET_VALUES: [
                    # /clock
                    ("PublishClock.inputs:topicName", config.clock_topic),
                    # Odometry
                    ("ComputeOdometry.inputs:chassisPrim", config.chassis_prim_path),
                    ("PublishOdometry.inputs:topicName", config.odom_topic),
                    ("PublishOdometry.inputs:chassisFrameId", config.base_frame_id),
                    ("PublishOdometry.inputs:odomFrameId", config.odom_frame_id),
                    # Raw odom → base_link
                    ("PublishOdomToBaseTF.inputs:parentFrameId", config.odom_frame_id),
                    ("PublishOdomToBaseTF.inputs:childFrameId", config.base_frame_id),
                    # base_link → children (d555_link + whatever else is in tf_extra_target_prims)
                    ("PublishBaseLinkTF.inputs:parentPrim", [config.chassis_prim_path]),
                    (
                        "PublishBaseLinkTF.inputs:targetPrims",
                        list(config.tf_extra_target_prims),
                    ),
                    # /cmd_vel
                    ("SubscribeTwist.inputs:topicName", config.cmd_vel_topic),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                    # Clock — tick every frame, stamp from the same sim-time source
                    # as every other publisher so consumers using use_sim_time:=True
                    # see monotonic sim-time across the whole stack.
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                    ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                    ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                    ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                    ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                    ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                    # odom → base_link raw TF reuses the odometry pose
                    ("OnPlaybackTick.outputs:tick", "PublishOdomToBaseTF.inputs:execIn"),
                    ("ComputeOdometry.outputs:orientation", "PublishOdomToBaseTF.inputs:rotation"),
                    ("ComputeOdometry.outputs:position", "PublishOdomToBaseTF.inputs:translation"),
                    # base_link TF tree ticks every frame
                    ("OnPlaybackTick.outputs:tick", "PublishBaseLinkTF.inputs:execIn"),
                    # Subscribe twist
                    ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                    # Context wiring
                    ("Context.outputs:context", "PublishOdometry.inputs:context"),
                    ("Context.outputs:context", "PublishOdomToBaseTF.inputs:context"),
                    ("Context.outputs:context", "PublishBaseLinkTF.inputs:context"),
                    ("Context.outputs:context", "SubscribeTwist.inputs:context"),
                    # Sim time on all publishers
                    ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishOdomToBaseTF.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishBaseLinkTF.inputs:timeStamp"),
                ],
            },
        )
    else:
        # Camera-only mode: still need OnPlaybackTick to drive the
        # render-product chain, RunOnce to gate one-shot initialization,
        # and a ROS2Context for the camera publishers. No sim-time node
        # is required because the camera helpers pull their own
        # timestamps via the IsaacSimulationGate.
        (graph_handle, *_) = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                ],
            },
        )

    if not skip_cameras:
        _add_camera_stream(
            graph_handle, graph_path, config.color_camera,
            suffix="Color", frame_skip=config.camera_frame_skip,
        )
        _add_camera_stream(
            graph_handle, graph_path, config.depth_camera,
            suffix="Depth", frame_skip=config.camera_frame_skip,
        )

    return graph_handle


def _add_camera_stream(
    graph_handle: Any,
    graph_path: str,
    stream: CameraStreamConfig,
    *,
    suffix: str,
    frame_skip: int = 0,
) -> None:
    """Wire one camera prim as both an image publisher and camera_info publisher.

    Each stream gets its own ``IsaacCreateRenderProduct`` configured at
    ``stream.width`` × ``stream.height``. ``IsaacCreateRenderProduct``
    does NOT inherit resolution from the camera prim — its
    ``inputs:width`` / ``inputs:height`` default to Hydra's 1280×720
    unless set explicitly, which silently overrides the camera prim's
    spawn-time resolution and publishes a wrong ``camera_info`` (width,
    height, and fx/fy all 2× the configured TiledCameraCfg). Sourcing
    the resolution from ``stream`` keeps it pinned to the same
    ``PERCEPTION_WIDTH`` / ``PERCEPTION_HEIGHT`` constants the
    perception camera uses on the env side.

    ``graph_path`` threads through so cross-block references (``RunOnce``,
    ``Context`` — created in the main ``build_bridge_graph`` edit block)
    can be addressed by full USD path. OmniGraph's short-name resolution
    only finds nodes created in the current ``Controller.edit`` call; a
    bare ``"RunOnce.outputs:step"`` falls through to being parsed as a
    literal USD path and then fails the source/destination type check.
    """

    import omni.graph.core as og

    keys = og.Controller.Keys

    render_node = f"RenderProduct{suffix}"
    image_node = f"CameraHelper{suffix}"
    info_node = f"CameraInfoHelper{suffix}"

    run_once_step = f"{graph_path}/RunOnce.outputs:step"
    context_out = f"{graph_path}/Context.outputs:context"

    og.Controller.edit(
        graph_handle,
        {
            keys.CREATE_NODES: [
                (render_node, "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                (image_node, "isaacsim.ros2.bridge.ROS2CameraHelper"),
                (info_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.SET_VALUES: [
                (f"{render_node}.inputs:cameraPrim", [stream.camera_prim_path]),
                (f"{render_node}.inputs:width", int(stream.width)),
                (f"{render_node}.inputs:height", int(stream.height)),
                (f"{image_node}.inputs:topicName", stream.image_topic),
                (f"{image_node}.inputs:frameId", stream.frame_id),
                (f"{image_node}.inputs:type", stream.stream_type),
                (f"{image_node}.inputs:resetSimulationTimeOnStop", True),
                (f"{image_node}.inputs:frameSkipCount", int(frame_skip)),
                (f"{info_node}.inputs:topicName", stream.camera_info_topic),
                (f"{info_node}.inputs:frameId", stream.frame_id),
                (f"{info_node}.inputs:resetSimulationTimeOnStop", True),
                (f"{info_node}.inputs:frameSkipCount", int(frame_skip)),
            ],
            keys.CONNECT: [
                (run_once_step, f"{render_node}.inputs:execIn"),
                (f"{render_node}.outputs:execOut", f"{image_node}.inputs:execIn"),
                (f"{render_node}.outputs:renderProductPath", f"{image_node}.inputs:renderProductPath"),
                (f"{render_node}.outputs:execOut", f"{info_node}.inputs:execIn"),
                (f"{render_node}.outputs:renderProductPath", f"{info_node}.inputs:renderProductPath"),
                (context_out, f"{image_node}.inputs:context"),
                (context_out, f"{info_node}.inputs:context"),
            ],
        },
    )


def read_cmd_vel(
    graph_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Sample the latest /cmd_vel from the SubscribeTwist node.

    Returns ``((linear_x, linear_y, linear_z), (angular_x, angular_y, angular_z))``.
    The env runner calls this each step and maps the Twist into the Isaac
    Lab action tensor (for the Strafer's mecanum base, that is typically
    ``[linear_x, linear_y, angular_z]``).
    """

    import omni.graph.core as og

    lin_attr = og.Controller.attribute(f"{graph_path}/SubscribeTwist.outputs:linearVelocity")
    ang_attr = og.Controller.attribute(f"{graph_path}/SubscribeTwist.outputs:angularVelocity")
    lin = lin_attr.get()
    ang = ang_attr.get()
    return (
        tuple(float(v) for v in lin),
        tuple(float(v) for v in ang),
    )
