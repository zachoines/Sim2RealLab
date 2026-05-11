"""Python-side ROS 2 publisher for the sim-in-the-loop perception camera.

Companion to :mod:`strafer_lab.bridge.async_publisher`: that module moved
the chassis telemetry (``/clock``, ``/odom``, TF, ``/cmd_vel`` subscribe)
off the Kit OmniGraph onto a Python rclpy thread; this module does the
same for the perception camera streams. Together they leave the bridge
OmniGraph with just the ``OnPlaybackTick`` / ``ROS2Context`` scaffolding
for the few remaining nodes the bridge still runs.

What ships here
---------------
* ``StraferCameraAsyncPublisher`` — owns the rclpy Node, four publishers
  (``/d555/color/image_raw``, ``/d555/color/camera_info``,
  ``/d555/depth/image_rect_raw``, ``/d555/depth/camera_info``), a worker
  thread, and a single-slot handoff queue.

* Per-frame protocol: the bridge mainloop calls
  :meth:`StraferCameraAsyncPublisher.notify_frame` after every
  ``env.step()``. That call snapshots the perception camera's GPU output
  tensors with a same-stream clone (cheap, ~hundreds of microseconds for
  640×360) so the next ``env.step`` is free to mutate the TiledCamera
  output buffer in place. The worker then does the D→H readback,
  ``sensor_msgs/Image`` construction, and ``Publisher.publish`` on its
  own thread, off the bridge critical path.

Why a separate CUDA stream
--------------------------
The worker reads the snapshot back to CPU on its own ``torch.cuda.Stream``
gated by a ``torch.cuda.Event`` recorded on the bridge stream right after
the clone. That sequences the D→H copy strictly after the snapshot
without forcing the bridge stream to wait for the copy to finish, so the
next env.step's PhysX / scene-update kernels overlap with the readback
instead of serializing behind it. With a shared stream the move-to-Python
would still pay the readback wall time on the critical path; the explicit
stream is the throughput win.

What stays on OmniGraph
-----------------------
Nothing camera-related. The OmniGraph still hosts
``OnPlaybackTick`` + ``ROS2Context`` for the wider bridge scaffolding,
but the ``IsaacCreateRenderProduct`` / ``ROS2CameraHelper`` /
``ROS2CameraInfoHelper`` chain is gone — this module owns the camera
publish path end-to-end.

Importable only inside a Kit process that has activated
``isaacsim.ros2.bridge`` (which transitively loads the bundled 3.12
rclpy at ``isaacsim.ros2.core/humble/rclpy``). Importing outside Kit
will fail at the ``rclpy`` line.
"""

from __future__ import annotations

import array
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import rclpy
import torch
import warp as wp
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from strafer_lab.bridge.config import CameraStreamConfig


def _to_torch(arr: Any) -> torch.Tensor:
    """Wrap an Isaac Lab 3.0 sensor-data output in a torch tensor.

    ``TiledCamera.data.output["..."]`` returns either a raw ``torch.Tensor``
    (Isaac Lab develop on the perception path) or a ``wp.array`` (older
    builds). ``wp.to_torch`` is a zero-copy view on CUDA; passing a tensor
    through it is a no-op via the duck-typed fall-through below.
    """
    if isinstance(arr, torch.Tensor):
        return arr
    return wp.to_torch(arr)


@dataclass
class _PendingFrame:
    """Snapshot handed from the bridge thread to the worker thread.

    Holds GPU clones of the RGB / depth tensors plus a CUDA event recorded
    on the bridge stream right after the clones, so the worker's D→H copy
    can wait on the event from its own stream without serializing the
    bridge stream behind the readback.
    """

    sim_time_s: float
    rgb_gpu: torch.Tensor | None  # (H, W, 3) uint8, on CUDA
    depth_gpu: torch.Tensor | None  # (H, W) float32 metres, on CUDA
    ready_event: Any  # torch.cuda.Event recorded after the clones


def _as_uint8_array(buf: np.ndarray) -> "array.array[int]":
    """Wrap a numpy buffer in ``array.array('B', ...)`` for the Image.data fast path.

    The rclpy-generated ``sensor_msgs/Image`` setter (see
    ``isaacsim.ros2.core/humble/rclpy/sensor_msgs/msg/_image.py``)
    has a single-branch fast path that accepts ``array.array`` with
    typecode ``'B'`` and assigns it in O(1). **Every other input type**
    — including ``bytes`` — falls through to a debug-mode assert chain
    that iterates the entire buffer twice (``all(isinstance(v, int) for v in value)``
    + ``all(0 <= val < 256 for val in value)``) before constructing the
    same ``array.array`` internally. For a 640×360×3 uint8 RGB image
    that is ~1.4 M Python-level operations per frame; for 640×360×4
    float-as-bytes depth, ~1.8 M. Measured cost in this codebase: ~80–100 ms
    per stream, i.e. ~180 ms for a (color + depth) pair — which is what
    ``camera :: rclpy publish`` was reporting before this helper landed.

    ``array.array('B', bytes_obj)`` uses the buffer protocol to memcpy
    the payload in C, bypassing the Python iteration entirely. Passing
    the result to ``img.data`` keeps the setter on its fast path.

    Accepts any contiguous numpy buffer; reads it as raw bytes
    (encoding interpretation lives in ``img.encoding``, not in the data
    type — sensor_msgs/Image's ``data`` field is always ``uint8[]``
    regardless of whether the payload is ``rgb8``, ``32FC1``, etc.).
    """
    return array.array("B", buf.tobytes())


def _build_camera_info(
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    frame_id: str,
) -> CameraInfo:
    """Construct a ``plumb_bob`` ``CameraInfo`` matching the prior OmniGraph output.

    The OmniGraph ``ROS2CameraInfoHelper`` paired with the
    ``opencvPinhole`` lens-distortion stamp authored by
    :func:`strafer_lab.tasks.navigation.mdp.events.stamp_d555_perception_opencv_pinhole`
    publishes a ``plumb_bob`` CameraInfo with zero distortion coefficients
    and intrinsics derived from the camera prim's focal length / aperture.
    Reproduce the same fields so Jetson-side consumers (RTAB-Map,
    ``depthimage_to_laserscan``, ``goal_projection_node``) cannot tell the
    publisher swapped under them.
    """
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.height = int(height)
    msg.width = int(width)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [
        float(fx), 0.0, float(cx),
        0.0, float(fy), float(cy),
        0.0, 0.0, 1.0,
    ]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [
        float(fx), 0.0, float(cx), 0.0,
        0.0, float(fy), float(cy), 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    msg.binning_x = 0
    msg.binning_y = 0
    return msg


class StraferCameraAsyncPublisher:
    """rclpy-backed publisher for the perception camera color + depth streams.

    Parameters
    ----------
    camera_sensor:
        Isaac Lab ``TiledCamera`` (or ``Camera``) handle from
        ``env.scene["d555_camera_perception"]``. The publisher reads
        ``camera_sensor.data.output["rgb"]`` and
        ``...["distance_to_image_plane"]`` each
        :meth:`notify_frame` call.
    color_stream / depth_stream:
        Per-stream wiring (topic, frame_id, resolution, encoding hint).
        Reused from :mod:`strafer_lab.bridge.config` so the OmniGraph and
        the Python path agree on names without a second source of truth.
    focal_length_mm / horizontal_aperture_mm:
        Pinhole intrinsics used to derive ``fx``/``fy`` for ``CameraInfo``.
        Match the values stamped on the camera prim by
        ``stamp_d555_perception_opencv_pinhole`` so the published
        ``CameraInfo`` is bit-identical to the prior OmniGraph output.
    frame_skip:
        Number of bridge ticks dropped between publishes. ``0`` publishes
        every tick (matches pre-optimization behavior); ``3`` publishes
        every 4th tick (matches ``--camera-frame-skip`` default and
        ``sim.render_interval``). Mirrors the
        ``ROS2CameraHelper.inputs:frameSkipCount`` field the OmniGraph
        used.
    node_name:
        ROS 2 node name. Defaults to ``strafer_sim_bridge_camera_publisher``.
    on_readback_ms / on_publish_ms:
        Optional callbacks invoked from the worker thread with the
        per-frame wall time (in milliseconds) for the GPU→CPU readback
        and the rclpy publish path respectively. Used by the
        ``--profile`` harness in ``run_sim_in_the_loop.py`` to report
        camera-thread cost alongside the bridge mainloop phases.
    """

    # Match isaacsim.ros2.bridge.ROS2CameraHelper defaults so Jetson-side
    # QoS-matched subscribers don't need re-tuning.
    _IMAGE_QOS = QoSProfile(
        depth=10,
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.VOLATILE,
    )

    def __init__(
        self,
        *,
        camera_sensor: Any,
        color_stream: CameraStreamConfig,
        depth_stream: CameraStreamConfig,
        focal_length_mm: float,
        horizontal_aperture_mm: float,
        frame_skip: int = 0,
        node_name: str = "strafer_sim_bridge_camera_publisher",
        on_readback_ms: Callable[[float], None] | None = None,
        on_publish_ms: Callable[[float], None] | None = None,
    ) -> None:
        self._camera = camera_sensor
        self._color_stream = color_stream
        self._depth_stream = depth_stream
        self._frame_skip = max(0, int(frame_skip))
        self._tick_counter = 0
        self._on_readback_ms = on_readback_ms
        self._on_publish_ms = on_publish_ms

        # ``rclpy`` may already be up from the telemetry publisher; guard
        # so both can coexist in one process.
        if not rclpy.ok():
            rclpy.init()

        self._node = Node(node_name)
        self._color_image_pub = self._node.create_publisher(
            Image, color_stream.image_topic, self._IMAGE_QOS,
        )
        self._color_info_pub = self._node.create_publisher(
            CameraInfo, color_stream.camera_info_topic, self._IMAGE_QOS,
        )
        self._depth_image_pub = self._node.create_publisher(
            Image, depth_stream.image_topic, self._IMAGE_QOS,
        )
        self._depth_info_pub = self._node.create_publisher(
            CameraInfo, depth_stream.camera_info_topic, self._IMAGE_QOS,
        )

        # Compute and cache CameraInfo from the OpenCV pinhole intrinsics.
        # Both streams share the perception camera prim, so they share
        # intrinsics — depth and color are co-registered at the same
        # resolution by design.
        fx = fy = color_stream.width * float(focal_length_mm) / float(horizontal_aperture_mm)
        cx = color_stream.width / 2.0
        cy = color_stream.height / 2.0
        self._color_info_template = _build_camera_info(
            width=color_stream.width,
            height=color_stream.height,
            fx=fx, fy=fy, cx=cx, cy=cy,
            frame_id=color_stream.frame_id,
        )
        self._depth_info_template = _build_camera_info(
            width=depth_stream.width,
            height=depth_stream.height,
            fx=fx, fy=fy, cx=cx, cy=cy,
            frame_id=depth_stream.frame_id,
        )

        self._device = self._infer_device()
        # Dedicated CUDA stream for D→H copies. Sequenced after each
        # bridge-thread clone via a ``torch.cuda.Event`` so the env.step
        # stream is not blocked by the readback.
        if self._device.type == "cuda":
            self._worker_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=self._device)
        else:
            self._worker_stream = None

        self._cv = threading.Condition()
        self._pending: _PendingFrame | None = None
        self._stop = threading.Event()

        # SingleThreadedExecutor mirrors StraferAsyncPublisher's pattern.
        # Camera node has no subscriptions or timers today, but keeping a
        # spinner running gives rclpy a place to service discovery / QoS
        # bookkeeping on its own thread without bloating the bridge loop.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._spin, daemon=True, name="strafer_camera_rclpy_spin",
        )
        self._spin_thread.start()

        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="strafer_camera_worker",
        )
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # Bridge-thread side
    # ------------------------------------------------------------------

    def notify_frame(self, sim_time_s: float) -> None:
        """Hand off the latest perception-camera frame to the worker.

        Called from the bridge mainloop after each ``env.step()``. Pulls
        the rendered RGB / depth GPU tensors out of the TiledCamera output
        dict, clones them (GPU→GPU, cheap) so the next ``env.step`` can
        mutate the underlying TiledCamera buffer in place without
        corrupting the in-flight publish, records a CUDA event for the
        worker stream to wait on, and signals the worker.

        ``--camera-frame-skip`` semantics: every ``frame_skip + 1``-th tick
        actually queues a publish; intermediate ticks return immediately.
        Drops (instead of queueing) if the worker is still busy with the
        previous frame, so a slow readback cannot wedge the bridge loop.
        """
        self._tick_counter += 1
        if self._frame_skip > 0 and (self._tick_counter - 1) % (self._frame_skip + 1) != 0:
            return

        try:
            rgb_gpu = _to_torch(self._camera.data.output["rgb"])[0]
            depth_gpu = _to_torch(self._camera.data.output["distance_to_image_plane"])[0]
        except (KeyError, IndexError):
            # Camera data not ready yet (typically the first tick before
            # the sensor has rendered). Skip silently — the next call will
            # retry once the output dict is populated.
            return

        rgb_clone = rgb_gpu.detach().clone()
        depth_clone = depth_gpu.detach().clone()

        if self._worker_stream is not None:
            ready_event: Any = torch.cuda.Event()
            ready_event.record()
        else:
            ready_event = None

        with self._cv:
            self._pending = _PendingFrame(
                sim_time_s=float(sim_time_s),
                rgb_gpu=rgb_clone,
                depth_gpu=depth_clone,
                ready_event=ready_event,
            )
            self._cv.notify()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while self._pending is None and not self._stop.is_set():
                    self._cv.wait(timeout=0.5)
                if self._stop.is_set():
                    return
                pending = self._pending
                self._pending = None

            try:
                self._process(pending)
            except Exception as exc:  # pragma: no cover — log and keep running
                self._node.get_logger().error(
                    f"camera worker frame failed: {type(exc).__name__}: {exc}"
                )

    def _process(self, pending: _PendingFrame) -> None:
        # D→H copy on the worker's own CUDA stream, gated by the event
        # the bridge thread recorded after the clone. The env.step stream
        # is free to enqueue new work in parallel.
        t_readback = time.perf_counter()
        if self._worker_stream is not None and pending.ready_event is not None:
            with torch.cuda.stream(self._worker_stream):
                pending.ready_event.wait()
                rgb_cpu = pending.rgb_gpu.to("cpu", non_blocking=False)
                depth_cpu = pending.depth_gpu.to("cpu", non_blocking=False)
        else:
            rgb_cpu = pending.rgb_gpu.cpu()
            depth_cpu = pending.depth_gpu.cpu()
        readback_ms = (time.perf_counter() - t_readback) * 1000.0
        if self._on_readback_ms is not None:
            self._on_readback_ms(readback_ms)

        t_publish = time.perf_counter()
        stamp = Time(seconds=pending.sim_time_s).to_msg()
        self._publish_color(rgb_cpu.numpy(), stamp)
        self._publish_depth(depth_cpu.numpy(), stamp)
        publish_ms = (time.perf_counter() - t_publish) * 1000.0
        if self._on_publish_ms is not None:
            self._on_publish_ms(publish_ms)

    def _publish_color(self, rgb: np.ndarray, stamp: Any) -> None:
        # TiledCamera's "rgb" channel is HWC uint8; drop the alpha if the
        # sensor returned RGBA (data_types="rgb" should already give HWC=3
        # but a defensive squeeze is cheap insurance against an Isaac Lab
        # API change that adds the alpha plane).
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[0], rgb.shape[1]

        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = self._color_stream.frame_id
        img.height = int(h)
        img.width = int(w)
        img.encoding = "rgb8"
        img.is_bigendian = 0
        img.step = int(w * 3)
        img.data = _as_uint8_array(rgb)
        self._color_image_pub.publish(img)

        info = self._color_info_template
        info.header.stamp = stamp
        self._color_info_pub.publish(info)

    def _publish_depth(self, depth: np.ndarray, stamp: Any) -> None:
        # TiledCamera's "distance_to_image_plane" arrives as float32 in
        # metres, shape (H, W) or (H, W, 1). Match the prior OmniGraph
        # output's ``32FC1`` encoding so Jetson-side consumers (RTAB-Map,
        # ``depthimage_to_laserscan``, ``goal_projection_node``) see no
        # change. NaN / inf for sky and out-of-frustum samples flow
        # through unaltered — that is the documented invariant in
        # ``context/bridge-runtime-invariants.md``: depth past the 6 m
        # sensor saturation reaches the bridge and Jetson consumers cap
        # on their end.
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        h, w = depth.shape[0], depth.shape[1]

        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = self._depth_stream.frame_id
        img.height = int(h)
        img.width = int(w)
        img.encoding = "32FC1"
        img.is_bigendian = 0
        img.step = int(w * 4)
        img.data = _as_uint8_array(depth)
        self._depth_image_pub.publish(img)

        info = self._depth_info_template
        info.header.stamp = stamp
        self._depth_info_pub.publish(info)

    def _spin(self) -> None:
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers / lifecycle
    # ------------------------------------------------------------------

    def _infer_device(self) -> torch.device:
        try:
            rgb = self._camera.data.output["rgb"]
        except (KeyError, AttributeError):
            return torch.device("cpu")
        tensor = _to_torch(rgb)
        return tensor.device

    def shutdown(self) -> None:
        """Stop both the worker and the rclpy spinner.

        Does NOT call ``rclpy.shutdown()`` — see StraferAsyncPublisher;
        other Kit components hold the rclpy context. Process exit is the
        clean shutdown trigger.
        """
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        try:
            self._executor.shutdown()
        except Exception:
            pass
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        try:
            self._node.destroy_node()
        except Exception:
            pass
