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

## Telemetry vs. cameras (both on Python publishers)

| Channel | Publisher | Tick driver | Why |
|---------|-----------|-------------|-----|
| `/clock` (50 Hz), `/odom`, `/tf`, `/cmd_vel` (subscribe) | `StraferAsyncPublisher` (Python rclpy thread) | independent of `env.step` | Decouples telemetry rate from env throughput; lets `/clock` advance smoothly even at low bridge FPS |
| `/d555/color/image_raw`, `/d555/color/camera_info`, `/d555/depth/image_rect_raw`, `/d555/depth/camera_info` | `StraferCameraAsyncPublisher` (Python rclpy thread + dedicated CUDA stream) | bridge mainloop calls `notify_frame(sim_time_s)` after each `env.step`; worker thread does the GPU→CPU readback + serialize + publish | Same architecture as the real-robot driver (camera frames published on their own thread). Decouples readback wall time from `simulation_app.update`. |

Per-frame protocol for cameras: the bridge mainloop snapshots the
TiledCamera output tensors with a same-stream `clone()` (microseconds
for 640×360) and records a CUDA event; the worker waits on that event
from its own `torch.cuda.Stream`, runs the D→H copy, and publishes
the four ROS 2 messages. The next `env.step`'s GPU work queues onto
the renderer stream in parallel with the readback. `--camera-frame-skip`
is honored on the publish side: `frame_skip=N` means the publisher
queues a frame once every `N+1` bridge ticks (default 3, matching
`sim.render_interval`).

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

Bridge enforcement: the perception camera prim itself is spawned at
`PERCEPTION_WIDTH × PERCEPTION_HEIGHT` by the `TiledCameraCfg` in
[`d555_cfg.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py)
(`make_d555_perception_camera_cfg`), so the GPU render product the
async camera publisher reads from is already at the right resolution.
`StraferCameraAsyncPublisher` reuses the same `PERCEPTION_WIDTH` /
`PERCEPTION_HEIGHT` (via `CameraStreamConfig.width` / `height`) to
populate `CameraInfo` and to size the published `sensor_msgs/Image`,
so width / height / fx / fy stay consistent end-to-end. The historic
`IsaacCreateRenderProduct` resolution footgun (silent fall-through to
Hydra's 1280×720, doubling fx / fy) is moot now that the bridge no
longer creates a second render product — the migration off OmniGraph
deleted that code path.

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
code path uses the system wall clock natively.

This convention is project-wide for motion deadlines:
`navigate_to_pose` (via `_wait_for_future` / `_wait_for_nav_result`)
and `rotate_in_place` both compute their primary deadline as
`clock.now() + Duration(seconds=timeout)`. To stop a *frozen* `/clock`
(crashed / paused bridge) from wedging the executor, each wait also
runs a **sim-time stall detector** (`_ClockStallDetector` in
[`ros_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)):
it records the wall instant at which `/clock` was last seen advancing
and bails only if wall time runs `clock_stall_bail_wall_s`
(`STRAFER_CLOCK_STALL_BAIL_WALL_S`, default 15 s) past that mark with
no sim-time progress.

This replaced an earlier absolute `2 * timeout` wall-clock cap. That
cap was an absolute *wall* bound and therefore wrong under sub-unity
RTF: a 90 s sim deadline maps to ~1800 s of wall time at RTF=0.05, but
a `2 * 90 = 180` s wall cap fired mid-motion while `/clock` was still
advancing legitimately, aborting rotations and translations partway
through. The stall detector instead distinguishes "no progress" (frozen
clock → bail) from "slow progress" (live clock → keep waiting), so it
tolerates any RTF while still catching a genuinely dead bridge. On real
hardware (`use_sim_time=False`) `/clock` tracks wall time, so sim time
always advances and the detector never fires — the primary deadline
governs, identical to the pre-detector behavior. Mirrors the reference
loop in
[`donut_warmup.py`](../../../source/strafer_ros/strafer_bringup/strafer_bringup/donut_warmup.py)
(`clock_stall_bail_wall_s`). Set `STRAFER_CLOCK_STALL_BAIL_WALL_S=0` to
disable the detector and leave the sim-clock deadline as the sole bound.

Nav2's own internal deadlines (`controller_frequency`,
`*_tolerance`, `cycle_frequency`, `movement_time_allowance`,
`failure_tolerance`, …) are measured against each server's node clock,
so they tick on `/clock` automatically once `use_sim_time:=true` flows
through. That flow depends on the per-server `use_sim_time: false`
lines in
[`nav2_params.yaml`](../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml)
**staying present**: Humble's `RewrittenYaml` only rewrites leaf keys
that already exist, so those literal `false` entries are the scaffold
the launch arg (`use_sim_time:=true`) is rewritten onto at runtime —
deleting them or hardcoding them `true` both break sim/real switching.

`STRAFER_NAVIGATION_TIMEOUT_S` (default 90 s; 180 s in
`env_sim_in_the_loop.env`) is the operator's per-mission ceiling.
Per-step budgets are derived in the executor:

- **Progress-aware mode (default, `STRAFER_NAV_PROGRESS_AWARE=1`).**
  `_translate`, `_rotate_by_degrees`, `_orient_to_direction`, and
  `_dispatch_nav_goal` synthesize per-step budgets from the requested
  displacement (or robot→goal straight-line distance) divided by
  `NAV_LINEAR_VEL` / `NAV_ANGULAR_VEL`, scaled by `safety_factor`
  and offset by `setup_overhead_s`. The result is capped at
  `STRAFER_NAVIGATION_TIMEOUT_S`. Additionally,
  `ros_client.navigate_to_pose` registers a `feedback_callback` and
  runs a stall watchdog on Nav2's `distance_remaining`: if no
  ≥ `nav_stall_progress_m` of progress occurs over
  `nav_stall_window_s` of sim-time, the goal is canceled with
  `error_code=navigation_stalled`. Tunables (env / dataclass default):
  - `STRAFER_NAV_BUDGET_SAFETY_FACTOR` / `nav_budget_safety_factor` (2.0)
  - `STRAFER_NAV_BUDGET_SETUP_OVERHEAD_S` / `nav_budget_setup_overhead_s` (5.0)
  - `STRAFER_NAV_STALL_PROGRESS_M` / `nav_stall_progress_m` (0.10 m)
  - `STRAFER_NAV_STALL_WINDOW_S` / `nav_stall_window_s` (20.0 s)
- **Legacy mode (`STRAFER_NAV_PROGRESS_AWARE=0`).** Every motion
  step uses `STRAFER_NAVIGATION_TIMEOUT_S` as the single deadline,
  no stall watchdog. Bisection escape hatch only.

Sources: commit `f60456e` (sim-clock deadline for `navigate_to_pose`),
the `progress-aware-nav-timeouts` brief (per-step budgets + watchdog),
`rotate-in-place-sim-clock-deadline` (sim-clock deadline for
`rotate_in_place`, completing the convention), and
`nav-deadline-sim-time-audit` (replaced the `2 * timeout` wall caps
with the `_ClockStallDetector`, confirmed the Nav2 `use_sim_time`
flow-through).
Live in
[`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
and
[`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`](../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py).

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
attached), measured before the camera-publisher migration:

| Configuration | `sim.step` p50 | `sim.render` p50 | `simulation_app.update` p50 | Loop total p50 | Throughput |
|---|---|---|---|---|---|
| `make sim-bridge-gui` (cameras on) | 21.8 ms | 88.6 ms | 83.1 ms | ~213 ms | 4.7 Hz |
| `make sim-bridge-gui` (cameras off) | 21.7 ms | 88.6 ms | 40.8 ms | ~125 ms | 8.0 Hz |
| `make sim-bridge` headless (cameras on) | 22.3 ms | 0.05 ms | 74.2 ms | ~117 ms | 8.5 Hz |

Pre-migration read: `sim.render`'s 88 ms is editor-viewport RTX work
(vanishes headless); `simulation_app.update`'s ~74 ms was the
camera-publish OmniGraph evaluation; PhysX (`sim.step`) is rock-stable
at ~22 ms across configs.

After moving cameras to `StraferCameraAsyncPublisher`, the
camera-publish work no longer runs inside `simulation_app.update` —
that phase collapses to a small Kit-pump cost (~few ms; the
`OnPlaybackTick` + `ROS2Context` scaffolding only). The GPU→CPU
readback and rclpy serialization run on the camera worker thread,
overlapping with the next `env.step`. Use `--profile` to confirm —
new phases `camera :: GPU→CPU readback` and `camera :: rclpy publish`
report camera-worker p50/p99 alongside the bridge mainloop phases.

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
