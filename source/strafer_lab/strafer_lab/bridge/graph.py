"""OmniGraph builder for the residual sim-bridge nodes that still need Kit.

This module imports ``omni.graph.core`` and therefore must only be invoked
inside a live Kit process (i.e. after ``isaaclab.app.AppLauncher`` has
booted Isaac Sim). The pure-Python config layer lives in
:mod:`strafer_lab.bridge.config` and is safe to import anywhere.

What the OmniGraph still hosts
------------------------------
Nothing. The bridge migrated both telemetry (``/clock``, ``/odom``, TF,
``/cmd_vel`` subscribe) and cameras (``/d555/color/...``,
``/d555/depth/...``) onto Python rclpy threads
(:class:`strafer_lab.bridge.async_publisher.StraferAsyncPublisher` and
:class:`strafer_lab.bridge.async_camera_publisher.StraferCameraAsyncPublisher`).
The Kit graph this module builds carries only the
``OnPlaybackTick`` → ``RunOnce`` scaffolding plus a ``ROS2Context``, kept
as a small, dependable place to attach future Kit-bound bridge nodes
without re-wiring the whole graph creation flow. The Python publishers
do not consume any of these nodes' outputs.

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

from strafer_lab.bridge.config import BridgeConfig


def build_bridge_graph(config: BridgeConfig) -> Any:
    """Create the residual ROS2 bridge OmniGraph described by ``config``.

    Returns the ``og.Controller.edit`` graph handle. The graph holds only
    ``OnPlaybackTick`` → ``RunOnce`` + a ``ROS2Context``; all publish /
    subscribe work lives on the Python publishers
    (``StraferAsyncPublisher`` for telemetry,
    ``StraferCameraAsyncPublisher`` for the camera streams).
    """

    import omni.graph.core as og

    keys = og.Controller.Keys

    graph_path = config.graph_path

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

    return graph_handle


def read_cmd_vel(graph_path: str):
    """Compatibility shim — ``/cmd_vel`` is no longer wired into the OmniGraph.

    The original implementation read out the bridge graph's
    ``SubscribeTwist`` node, but that node moved onto the Python rclpy
    publisher (:class:`strafer_lab.bridge.async_publisher.StraferAsyncPublisher`)
    when telemetry came off OmniGraph. Callers should obtain the latest
    twist by injecting ``StraferAsyncPublisher.get_cmd_vel`` as the
    ``cmd_vel_reader`` argument on the relevant adapter. Kept here as a
    named symbol so the lazy import in
    :mod:`strafer_lab.sim_in_the_loop.runtime_env` does not raise
    ``ImportError`` at module load; calling it raises a clear error
    pointing at the migration.
    """
    raise RuntimeError(
        "strafer_lab.bridge.graph.read_cmd_vel is no longer wired — the "
        "OmniGraph SubscribeTwist node was migrated to "
        "StraferAsyncPublisher (Python rclpy). Inject "
        "StraferAsyncPublisher.get_cmd_vel as the cmd_vel_reader on "
        "IsaacLabEnvAdapter."
    )
