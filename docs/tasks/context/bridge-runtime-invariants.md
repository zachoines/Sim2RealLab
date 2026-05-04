# Bridge runtime invariants

Facts about the sim-in-the-loop ROS 2 bridge that any task touching
it must hold. The bridge is the DGX side of the architecture — it
makes Isaac Sim look like a real D555 + mecanum chassis to the
Jetson stack.

Source code: [`source/strafer_lab/scripts/run_sim_in_the_loop.py`](../../../source/strafer_lab/scripts/run_sim_in_the_loop.py)
+ [`source/strafer_lab/strafer_lab/bridge/`](../../../source/strafer_lab/strafer_lab/bridge/)
+ [`source/strafer_lab/strafer_lab/sim_in_the_loop/`](../../../source/strafer_lab/strafer_lab/sim_in_the_loop/).
Re-run the `--profile` harness on the bridge mainloop (see
[Phase-level profiler](#phase-level-profiler---profile) below) to
re-derive per-phase wall-time attribution if you suspect a regression.

## Two run modes

| Mode | Drives env from | Used by |
|------|-----------------|---------|
| `--mode bridge` (default) | `/cmd_vel` from the Jetson, written into the action tensor in the mainloop | Manual ops, Nav2 missions, `make sim-bridge[-gui]` |
| `--mode harness` | Mission generator walking scene metadata + autonomy executor over `execute_mission` | Reachability-labeled dataset capture |

`--mode bridge` and `--mode harness` share the same env, the same
bridge OmniGraph, and the same telemetry async publisher. They
differ only in **what writes the action**: the bridge mainloop in
the first case, the harness's `IsaacLabEnvAdapter` in the second.

## cmd_vel normalization contract (BOTH paths)

`MecanumWheelAction.process_actions` consumes actions in the
**normalized `[-1, 1]`** contract and scales by
`_velocity_scale = [MAX_LINEAR_VEL, MAX_LINEAR_VEL, MAX_ANGULAR_VEL]`
(values in `strafer_shared.constants`).

`/cmd_vel` arrives in **physical units (m/s, rad/s)**. Both bridge
paths divide by `MAX_LINEAR_VEL` / `MAX_ANGULAR_VEL` and clamp to
`[-1, 1]` before writing into the action tensor:

- `--mode bridge` mainloop: see
  [`run_sim_in_the_loop.py`](../../../source/strafer_lab/scripts/run_sim_in_the_loop.py)
  near the `_clamp_unit(...)` call sites.
- `--mode harness` adapter:
  [`runtime_env._build_action`](../../../source/strafer_lab/strafer_lab/sim_in_the_loop/runtime_env.py).

If you add a new code path that writes `/cmd_vel` into the action
tensor, **mirror this normalization**. Skipping it amplifies every
Nav2 command by ~1.57× linearly / ~4.19× angularly and saturates
the per-wheel motor cap on diagonal motion (the bug commits
`d642bff` + `70c4ba9` fixed).

## Telemetry vs. cameras (split publishers)

| Channel | Publisher | Tick driver | Why |
|---------|-----------|-------------|-----|
| `/clock` (50 Hz), `/odom`, `/tf`, `/cmd_vel` (subscribe) | `StraferAsyncPublisher` (Python rclpy thread) | independent of `env.step` | Decouples telemetry rate from env throughput; lets `/clock` advance smoothly even at low bridge FPS |
| `/d555/color/image_raw`, `/d555/color/camera_info`, `/d555/depth/image_rect_raw`, `/d555/depth/camera_info` | OmniGraph nodes inside `isaacsim.ros2.bridge` | `OnPlaybackTick`, fired by `simulation_app.update()` after each `env.step` | Camera publish requires the render product, which lives on the GPU and ticks with Kit's main loop |

Moving cameras off `OnPlaybackTick` onto a Python thread is open
work, tracked at
[`async-camera-publishers.md`](../active/async-camera-publishers.md).

## Camera resolutions (sim mirrors real)

| Camera | Resolution | Bridged? | Consumer |
|--------|------------|----------|----------|
| `d555_camera` (policy) | 80×60 | NO | RL policy depth observation only — never leaves the env |
| `d555_camera_perception` | 640×360 | YES | VLM grounding, RTAB-Map, depthimage_to_laserscan, goal projection, RViz/Foxglove |

640×360 is **locked to the real D555 native rate** (see
`strafer_shared.constants.PERCEPTION_WIDTH/HEIGHT` comment).
Lowering it sim-side introduces a deliberate sim-to-real gap, not an
optimization — the camera-publish OmniGraph cost (~74 ms/loop on
this stack) is a fact of the bridge, not an axis to tune away.

Bridge enforcement: the resolution flows through
`CameraStreamConfig.width` / `height` →
`IsaacCreateRenderProduct.inputs:width` / `inputs:height` in
`bridge/graph.py`. Setting these explicitly is mandatory —
`IsaacCreateRenderProduct` does NOT inherit resolution from the
camera prim, so an unset render product silently falls through to
Hydra's 1280×720 default and publishes a wrong `camera_info`
(width/height + fx/fy all 2× the configured pinhole intrinsics).

## Renderer frustum vs. depth-sensor saturation

The D555 camera's `PinholeCameraCfg.clipping_range` uses
`(DEPTH_SIM_CLIP_NEAR=0.01, D555_RENDER_FAR_CLIP_M=50.0)` —
**generous renderer frustum**, decoupled from the depth-sensor
saturation model. The 6 m sensor-saturation limit
(`DEPTH_CLIP_FAR=6.0`) is applied in software in
[`mdp/observations.py:depth_image`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
via `torch.clamp`, not at the renderer. Wiring the sensor limit
into the frustum (as the original code did) blackens RGB beyond
6 m too. Don't conflate the two.

The bridge's `/d555/depth/image_rect_raw` stream comes straight
off the rendered camera with no software hook, so depth values past
6 m DO reach the bridge. Jetson-side consumers wanting sim/real
parity at the real saturation point cap depth on their end.

## Headless vs. `--viz kit` defaults

| Make target | Visibility | When to use |
|-------------|------------|-------------|
| `make sim-bridge` | Headless (no editor viewport) | **Daily-driver for missions.** Saves ~85 ms/loop vs `--viz kit` on a DISPLAY-attached host (~2 ms/loop on a no-DISPLAY DGX, where the editor viewport collapses to a no-op). Visual debugging via the Jetson-side `foxglove_bridge` (`bringup_sim_in_the_loop.launch.py` `viewer:=true`, default), connected through an SSH tunnel from the operator's workstation; see [INTEGRATION_SIM_IN_THE_LOOP.md Stage 3.5](../../INTEGRATION_SIM_IN_THE_LOOP.md). |
| `make sim-bridge-gui` | Headed, Kit editor viewport | Visual debugging only. The bridge mainloop sets `env.render_enabled = False` so `KitVisualizer.step`'s `app.update()` is short-circuited; the explicit `simulation_app.update()` after each `env.step()` is the sole Kit pump per loop, refreshing the viewport, firing `OnPlaybackTick`, and driving the camera-publish OmniGraph in one pass. |

Default decimation in bridge: **`decimation=1`** (matches
`run_sim_in_the_loop.py --decimation` default). RL training default
is `decimation=4`; the bridge override is the perf fix from
`74979d6`. `collect_demos.py` does NOT override decimation, which
is why it appears slower than `make sim-bridge-gui` at first
glance.

## Sim-time-aware navigation timeout (Jetson side)

The executor's nav-timeout enforcement uses
`node.get_clock().now()`, which respects `use_sim_time`. On the
sim-in-the-loop bringup launch (`use_sim_time:=true`), 90 s of
sim-time wait is 90 s of `/clock` advance — not 90 s of wall
clock. Real-robot bringup leaves `use_sim_time=False`, so the same
code path uses the system wall clock natively. Set via
`STRAFER_NAVIGATION_TIMEOUT_S` env var (default 90 s; bumped to
600 in `env_sim_in_the_loop.env` for slow-RTF sim runs).

Source: commit `f60456e`, lives in
[`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py).

## Phase-level profiler (`--profile`)

`run_sim_in_the_loop.py --profile` wraps the bridge mainloop's outer
phases (`cmd_vel read`, `env.step (total)`, `publish_state`,
`simulation_app.update`) and monkey-patches `sim.step` /
`sim.render` to attribute their per-tick wall time. p50/p99 reported
on a rolling window every `--profile-interval` seconds. Lives behind
the flag so normal runs are untouched. Use for any bridge perf
regression hunt — added in `70c4ba9`.

Reference per-phase numbers on this stack
(InfinigenPerception, decimation=1, render_interval=1, no DISPLAY
attached) at the time the headless default was chosen:

| Configuration | `sim.step` p50 | `sim.render` p50 | `simulation_app.update` p50 | Loop total p50 | Throughput |
|---|---|---|---|---|---|
| `make sim-bridge-gui` (cameras on) | 21.8 ms | 88.6 ms | 83.1 ms | ~213 ms | 4.7 Hz |
| `make sim-bridge-gui` (cameras off) | 21.7 ms | 88.6 ms | 40.8 ms | ~125 ms | 8.0 Hz |
| `make sim-bridge` headless (cameras on) | 22.3 ms | 0.05 ms | 74.2 ms | ~117 ms | 8.5 Hz |

Read: `sim.render`'s 88 ms is editor-viewport RTX work (vanishes
headless); `simulation_app.update`'s ~74 ms is the camera-publish
OmniGraph evaluation; PhysX (`sim.step`) is rock-stable at ~22 ms
across configs. The bridge's bottleneck is rendering pipeline cost,
not physics.

## Scene-side prerequisites the bridge assumes

These are baked at scene-generation time, not at bridge launch:

- **Infinigen floor mesh colliders are stripped** — see
  [`scripts/postprocess_scene_usd.py`](../../../source/strafer_lab/scripts/postprocess_scene_usd.py).
  Robot collision is delegated to the `/World/ground/terrain` plane
  raised to floor height by the env's
  `lift_ground_plane_to_floor` startup event. If you re-bake an
  Infinigen scene, this strip is automatic; manual USD imports
  need to honor the same convention.
- **`floor_top_z` per scene** lives in
  `Assets/generated/scenes/scenes_metadata.json`, authored by
  `generate_scenes_metadata.py`. The startup event reads it via
  `_get_infinigen_active_scene_floor_top_z(scene_stem)` and lifts
  the ground plane to `floor_top_z - 0.002` (2 mm below to avoid
  z-fighting).
