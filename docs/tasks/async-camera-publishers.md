# Async camera publishers (move RGB/depth off `OnPlaybackTick`)

**Type:** task / refactor
**Owner:** DGX agent
**Priority:** P1
**Estimate:** L (~3-5 days; mirrors the existing `StraferAsyncPublisher`
refactor that moved telemetry off the OmniGraph)

## Story

As a **bridge operator**, I want **camera publishing to happen on its
own Python rclpy thread instead of on Kit's `OnPlaybackTick`**, so that
**env.step throughput is decoupled from camera-publish work and the
bridge can sustain a higher mission-execution rate without sacrificing
camera publish cadence**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)

## Context

Profiling the bridge in headless + cameras-on (the configuration where
the editor viewport doesn't muddy the picture) measured:

| Phase                    | p50 (ms) |
|--------------------------|---------:|
| `simulation_app.update`  | 74.2     |  ← camera-bridge OG eval + render-product readback
| `env.step (sim.step)`    | 22.3     |
| `env.step` total         | 42.0     |
| Loop wall total          | ~117     |  → 8.5 Hz

`simulation_app.update` is the only thing on the loop that scales with
camera resolution / channel count. It's currently inline in the bridge
mainloop because the camera publish chain is wired into the
OmniGraph as `OnPlaybackTick` nodes — the only way to evaluate them is
to pump Kit's main loop with `playSimulations=True`, which serializes
camera publishing with env.step.

Telemetry (`/clock`, `/odom`, `/tf`, `/cmd_vel`) was already moved off
the OmniGraph onto a Python rclpy thread in commit `f912d73`
(`StraferAsyncPublisher`). That decoupled telemetry rate from env.step
rate and gave us /clock at 50 Hz regardless of bridge throughput. The
camera path should follow the same architectural pattern.

The technical wrinkle: cameras need GPU render-product readback (the
RGB / depth tensor lives on the GPU after Replicator renders the
view) and a sync to CPU before serialization to ROS Image messages.
That sync is the bulk of the 74 ms cost, NOT the message
construction or rclpy publish. Moving the publish to a Python thread
doesn't avoid the readback — but it lets the readback overlap with
the next env.step, which is the throughput win.

## Scope of impact

- **Bridge throughput** (headless): ~74 ms removed from the critical
  path per loop. Loop wall could drop from ~117 ms (8.5 Hz) toward
  ~45 ms (22 Hz) if readback overlap is achieved cleanly. Bounded by
  whichever phase becomes the new ceiling — likely PhysX (~22 ms)
  plus IsaacLab manager loop (~18 ms) ≈ 40 ms / 25 Hz.
- **Bridge headed mode**: smaller relative win because the editor
  viewport (~88 ms in `sim.render`) is the bottleneck under `--viz kit`.
  Best to land the [Kit-pump-redundancy fix](kit-pump-redundancy-investigation.md)
  alongside this one — together they would put headed mode in the
  same ballpark as headless.
- **Real-robot deployment**: zero impact. The real robot's camera
  drivers publish on their own threads natively; this brings sim into
  parity with that architecture.

## Acceptance criteria

- [ ] Color and depth publishers run on a Python rclpy thread, mirroring
      the structure of `StraferAsyncPublisher`.
- [ ] The OmniGraph no longer contains color / depth / camera_info
      publisher nodes.
- [ ] `ros2 topic hz /d555/color/image_raw` and
      `/d555/depth/image_rect_raw` match or exceed the prior bridge's
      rates under `make sim-bridge --enable_cameras`.
- [ ] `--profile` shows `simulation_app.update` p50 drops materially
      (target: from ~74 ms to ≤ 20 ms) and the new camera thread's
      readback time is reported separately.
- [ ] Camera frame timestamps still align with `/clock` to within
      one frame period — RTAB-Map and goal projection rely on
      time-aligned RGB-D pairs.
- [ ] `--camera-frame-skip` semantics preserved (or replaced with a
      per-thread rate divider); existing CLI surface intact.
- [ ] Headless smoke test: a navigate_to_pose mission completes against
      the bridge with at least the same success rate as today.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_lab/strafer_lab/bridge/async_publisher.py` —
  `StraferAsyncPublisher` is the prior-art template for the camera
  thread. Note its rclpy Node + thread + ScheduledTimer structure.
- `source/strafer_lab/strafer_lab/bridge/graph.py` — the camera
  OmniGraph nodes that need to be removed from the graph build.
- `source/strafer_lab/strafer_lab/bridge/config.py` — `CameraStreamConfig`
  carries the prim path, topic name, frame_id, and stream type. Reuse
  for the async path.
- Render-product readback: `Camera` / `TiledCamera` exposes
  `data.output["rgb"]` and `data.output["distance_to_image_plane"]` as
  GPU tensors. The async thread reads these (probably needs a
  CUDA-stream-aware copy to avoid blocking the env.step thread).
- ROS 2 Image serialization: `cv_bridge` or manual `sensor_msgs/Image`
  construction with `numpy.tobytes()`. Match the existing OmniGraph
  output's encoding (likely `rgb8` + `32FC1` for float depth, or
  `16UC1` for 1mm-quantized depth — check `bridge/graph.py` for which).
- Synchronization: render product readback must happen AFTER the sim
  has rendered to that product. With `--viz kit` the KitVisualizer.step
  triggers camera renders inside `sim.render`. Without it, the
  `simulation_app.update()` call (or its replacement) is what
  populates render products. The async thread needs a signal /
  semaphore from the bridge mainloop after each render.

## Risks / open questions

- **CUDA stream conflicts**: env.step uses CUDA via PyTorch + Warp;
  the async thread copying camera tensors must use a separate CUDA
  stream or pinned-memory staging buffer to avoid serializing onto
  the env.step stream.
- **/clock alignment**: today camera timestamps are stamped by
  OmniGraph using `IsaacReadSimulationTime`. The async thread will
  need to read sim time from `unwrapped.sim` or get it passed in
  per-frame. Investigate whether reading sim time off-thread is
  thread-safe in Isaac Lab develop.
- **Frame drops**: if the async thread can't keep up (e.g. heavy GPU
  contention with another sim render), do we drop frames or block?
  Real D555 publishes at fixed cadence with frame drops; bridge
  should mirror that.

## Out of scope

- Lowering perception camera resolution. The sim-mirrors-real
  constraint and reasoning are spelled out in
  [`context/bridge-runtime-invariants.md`](context/bridge-runtime-invariants.md#camera-resolutions-sim-mirrors-real).
- Touching the policy 80×60 camera (`d555_camera`) — it's not bridged
  and stays inside the env on GPU.
- Removing the second Kit pump itself — that's the
  [Kit-pump-redundancy task](kit-pump-redundancy-investigation.md).
  The two changes are complementary and either can land first.
