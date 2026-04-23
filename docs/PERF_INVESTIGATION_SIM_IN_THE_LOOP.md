# Sim-in-the-loop performance investigation

Living log of experiments and findings for the low-throughput issue on
the DGX Spark sim-in-the-loop bridge. Update as new data lands.

## Symptom

- `make sim-bridge-gui` with the HQ Infinigen scene: `/d555/color/image_raw`
  publishing at **~5 Hz** on the Jetson side.
- Config target in `strafer_env_cfg` is 30 Hz (`sim.dt = 1/120`,
  `sim.render_interval = 4`, render period = 33.3 ms).
- Real-time factor ≈ 0.17 at the start of the investigation.
- Scene swap to `fast_singleroom` moved the bar to ~6 Hz; full scene
  deletion moved it to ~7 Hz. Scene complexity is **not** the bottleneck.

## Test plan

Three diagnostic threads:

- **(a)** Is the GB10 actually the limiter?
- **(b)** Is there a single-thread CPU bottleneck?
- **(c)** Does the same slowdown show up *without* the ROS 2 bridge graph?

## Results — subsystem utilization (bridge running, fast_singleroom)

### GPU utilization

```
nvidia-smi dmon -s u -c 30
nvidia-smi pmon -c 15
```

| Metric | Observed |
|---|---|
| GPU SM (dmon) | 62–77 %, avg ~70 % |
| GPU memory bandwidth | 0 % (sustained) |
| Kit process SM | 61–66 % (≈ total SM, Kit is sole consumer) |
| Encoder / decoder / JPG / OFA | 0 % |

**GB10 has ~30 % headroom.** Not pinned. Memory bandwidth is essentially
idle so we're not bottlenecked on VRAM traffic either.

### CPU utilization

```
pidstat -t -p $KIT_PID 1 5
```

| Metric | Observed |
|---|---|
| Main process total CPU | **139 %** (~1.4 cores) |
| Hottest single thread | 24–29 % (the Python main thread) |
| Other worker threads | 5–11 % each across ~6 threads |
| System load avg (over 20 cores) | 2.17 |
| `%wait` / IO wait | ~1 % |

**No single-thread pinning.** No disk-I/O bottleneck. System has 18+
idle cores. Kit isn't CPU-bound in any obvious sense.

### Interpretation: sync-barrier hypothesis

Both GPU and CPU have abundant headroom but per-step time is ~120 ms.
Signature of a **synchronization barrier** — CPU submits, waits for
GPU, GPU finishes, waits for CPU, repeat. Neither saturates because
each is gated on the other across a fence.

## Results — without the ROS 2 bridge (test c)

Benchmark script: `/tmp/bridge_perf_rollout.py`. Boots Isaac Sim at
`Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0`, does 10 warmup
steps, times 100 `env.step(zero)` calls. **No bridge graph, no ROS 2
publishing.**

| Resolution | num_envs | ms/step | steps/s | env-steps/s |
|---|---|---|---|---|
| 640 × 360 | 1 | 120 | 8.30 | 8.30 |
| 320 × 180 | 1 | 115 | 8.68 | 8.68 |
| 160 × 90  | 1 | 119 | 8.41 | 8.41 |
| 640 × 360 | 2 | 165 | 6.07 | 12.13 |
| 640 × 360 | 4 | 174 | 5.76 | 23.02 |

### Finding 1 — resolution is not the bottleneck

16× fewer pixels (640×360 → 160×90) leaves step time unchanged. Kit's
DLSS pipeline clamps render resolution to a minimum (`DLSS increasing
input dimensions: Render resolution of (371, 209) is below minimal
input resolution of 300` appears in the startup log), so below a
threshold there's no work reduction. Render rasterization is **not**
dominating the frame budget even at native resolution.

### Finding 2 — per-env cost is nearly free

- 1 → 2 envs: step time +37 %, throughput +46 % (env-steps/s)
- 2 → 4 envs: step time +5 %,  throughput +90 %
- Marginal cost per additional env at steady state: **2–10 ms**.

Almost all of the ~120 ms single-env step time is a **fixed per-step
overhead** paid regardless of env count. Candidates:

- Physics step pump (kernel launch overhead, Fabric propagation)
- OmniGraph / Replicator SDG pipeline per-step evaluation
- Kit app.update() scheduling + event dispatch
- Render product readback barrier (sync to CPU even if image is unused)

### Finding 3 — bridge adds noticeable but not dominant overhead

| Config | Hz |
|---|---|
| Bridge on + perception env | ~5 |
| No bridge, same env, num_envs=1 | 8.7 |

The ROS 2 bridge publish chain drops throughput by ~40 %. Secondary
factor; the ~8 Hz ceiling is the real headline.

## Results — isolating camera cost (NoCam vs Depth vs Perception)

| Task | Cameras | ms/step | steps/s |
|---|---|---|---|
| `Isaac-Strafer-Nav-Real-NoCam-Play-v0` | none | **97** | 10.32 |
| `Isaac-Strafer-Nav-Real-Depth-Play-v0` | depth 80×60 | 107 | 9.35 |
| `Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0` | perception 640×360 + policy 80×60 | 118 | 8.45 |

### Finding 4 — rendering costs only ~21 ms; the real cost is elsewhere

Even with zero cameras and no rendering, the env takes **97 ms per step**.
Cameras on top cost ~10–21 ms depending on resolution. This ruled out
rendering as the dominant cost and pointed the investigation at the
physics + manager loop inside `env.step()`.

## Results — physics decimation override (NoCam floor)

| sim.dt | decimation | ms/step | steps/s |
|---|---|---|---|
| 1/120 | 4 (default) | 81 | 12.28 |
| 1/120 | 2 | 44 | 22.95 |
| 1/120 | **1** | **23** | **43.33** |
| 1/60 | 4 | 121 | 8.26 |

### Finding 5 — decimation is the dominant cost

**Each `env.step()` runs `decimation` physics ticks serially.** Going
from 4 → 1 tripled throughput. At decimation=1, one env.step advances
sim-time by a single physics tick (~8.3 ms), but wall-time drops to
~23 ms — you get ~43 env-steps/s on NoCam.

Increasing `sim.dt` while holding decimation constant got *slower*
(PhysX solver does more convergence work per tick at larger dt), so
the lever is decimation, not dt.

## Results — bridge-on with new default decimation=1

After wiring `--decimation` into `run_sim_in_the_loop.py` with default 1,
measured `/clock`, `/d555/color/image_raw`, and `/strafer/odom` from a
second shell via rclpy for a 10-second window.

| Topic | Messages | Wall-rate |
|---|---|---|
| `/clock` | 64 | **6.40 Hz** |
| `/d555/color/image_raw` | 64 | **6.40 Hz** |
| `/strafer/odom` | 64 | **6.40 Hz** |

(All three topics publish in lockstep since they tick on the same
`OnPlaybackTick` in the bridge graph.)

### Finding 7 — decimation=1 helps but the bridge graph is now the ceiling

| Config | Hz |
|---|---|
| Bridge-off, decimation=1 | 28.89 |
| Bridge-off, decimation=4 | 8.47 |
| Bridge-on, decimation=1 | **6.40** |
| Bridge-on, decimation=4 | ~5 |

Decimation=1 only raised bridge-on from ~5 → 6.4 Hz. The per-step cost
now breaks down roughly as:

- 35 ms env.step (physics + managers + render)
- ~120 ms OmniGraph evaluation of the bridge publishers

Bridge overhead is ~120 ms **independent of decimation** — the graph
evaluates on every `OnPlaybackTick` regardless of how many physics
ticks a decimation=1 step compresses. At decimation=4 this is paid
once per 4 physics ticks; at decimation=1 it's paid once per single
tick, so reducing decimation doesn't amortize the bridge cost.

## Results — perception env with lower decimation

| Task | Decimation | ms/step | steps/s |
|---|---|---|---|
| Perception (default) | 4 | 118 | 8.47 |
| Perception | 2 | 66 | 15.16 |
| Perception | **1** | **35** | **28.89** |

### Finding 6 — perception env at decimation=1 hits ~29 Hz

Nearly at the 30 Hz target the bridge's `sim.render_interval=4`
assumed. The tradeoff is sim-time per env.step: at decimation=1 each
step advances sim time by only 8.3 ms (vs 33.3 ms at decimation=4).
RL training used decimation=4 to compress sim-time and keep motor
dynamics / command delay at their tuned ranges — bridge mode has no
such constraint.

## Ruled out

- Scene geometry complexity (fast_singleroom vs HQ: <1 Hz difference).
- Per-robot USD complexity (would scale strongly with num_envs; it
  doesn't).
- Texture / material cost (same conclusion — num_envs scales cheap).
- Perception camera resolution.
- GPU render saturation.
- Memory bandwidth.
- Single-thread CPU pinning.
- Disk I/O.

## Remaining hypotheses

Ordered by likelihood / investigation cost:

1. ~~**Replicator SDG pipeline evaluating per step even when unused.**~~
   Partially disproved by Finding 4 — the NoCam floor already pins
   the cost at 97 ms with no rendering at all. Replicator may still
   contribute a few ms on the perception env but it is not the
   dominant cost.

2. ~~**TiledCamera render-product readback barrier per step.**~~
   Bounded by Finding 4 at ≤21 ms, not dominant.

3. **Bridge publish chain under-investigated.** No-bridge perception
   env at decimation=1 is 29 Hz; bridge-on at decimation=4 is 5 Hz.
   The delta at matched decimation has not been measured. Open
   question: how much does the bridge graph add at decimation=1?

## Async / decoupling options

The Kit event loop is single-stream and `env.step()` is a blocking
Python call. Three levers buy us effective async:

1. **Decouple publish rate from sim step rate.** Gate
   `ROS2CameraHelper` on every Nth tick using a frame-skip node or
   separate ticking. Ships `/cmd_vel` and physics at full step rate
   but images at half or quarter rate. Reduces bridge publish cost
   without touching the per-step fixed overhead.

2. **`publish_multithreading_disabled = False`** in the
   `isaacsim.ros2.bridge` extension config. Default is False (so
   threading IS enabled) — confirmed from `extension.toml`. Already
   active.

3. **`sim.render_interval` tuning.** Increase from 4 (render every
   4 physics steps) to 8. Halves render cost at the cost of camera
   rate. Directly attacks the fixed render cost.

Not supported in this stack:

- Returning from `env.step()` before GPU completion.
- `/cmd_vel` fire-and-forget mid-step (SubscribeTwist polled once
  per sim tick).

## Open next actions

1. ✅ Added `--decimation N` override to `run_sim_in_the_loop.py`
   (default 1). Bridge-on goes from 5 → 6.4 Hz.

2. ✅ Benchmarked bridge-on at decimation=1 — confirmed ~120 ms of
   fixed OmniGraph evaluation cost per tick, **independent of
   decimation**. This is now the ceiling.

3. ✅ Wired `camera_frame_skip` through `BridgeConfig` →
   `ROS2CameraHelper.inputs:frameSkipCount` +
   `ROS2CameraInfoHelper.inputs:frameSkipCount`, defaulted to 3
   (publish every 4th tick, matching `render_interval=4`). Exposed
   `--camera-frame-skip N` on `run_sim_in_the_loop.py`.

## Results — bridge-on with camera_frame_skip=3

| Topic | Messages in 10 s | Rate |
|---|---|---|
| `/clock` | 64 | 6.40 Hz |
| `/d555/color/image_raw` | 16 | **1.60 Hz** ← correctly gated to 1/4 |
| `/strafer/odom` | 64 | 6.40 Hz |

### Finding 8 — image publishing was NOT the ~120 ms/tick bottleneck

Frame-skip took effect (images dropped from 6.4 → 1.6 Hz exactly as
expected for `frameSkipCount=3`), but **`/clock` and `/strafer/odom`
stayed at 6.40 Hz**. The overall tick rate did not improve.

This rules out image serialization + DDS send as the dominant
per-tick cost. The ~120 ms gap between bridge-off (29 Hz) and
bridge-on (6.4 Hz) is being paid somewhere else in the graph or in
a Kit-side synchronization that activates whenever the bridge
extension is loaded.

### New hypothesis — render-product readback is syncing env.step to render cadence

`IsaacCreateRenderProduct` attaches hydra textures to the stage
whenever the bridge graph instantiates. With a render product
attached, Kit may force every `app.update()` (every physics tick with
decimation=1) to synchronize against the render pipeline — even
though `sim.render_interval=4` should only *render* every 4 ticks, the
*sync* might still happen per tick. If so, env.step is capped at the
render completion rate (~7 Hz on this scene) regardless of physics
speed.

## Results — render_interval sweep (decimation=1, bridge on)

Added `--render-interval` CLI override to `run_sim_in_the_loop.py`
(default leaves env value alone). Swept across 2 / 4 / 8.

| `render_interval` | `/clock` Hz | `/image` Hz | ms/tick | renders/s | ms/render |
|---|---|---|---|---|---|
| 2 | **10.12** | 2.50 | 99 | 5.06 | 198 |
| 4 | 6.38 | 1.62 | 157 | 1.60 | 625 |
| 8 | 3.75 | 0.88 | 267 | 0.47 | 2128 |

### Finding 9 — env.step is render-paced AND per-render cost scales with render_interval

Two observations, both surprising:

1. **Halving `render_interval` roughly doubles tick rate.** env.step
   completion is gated on render work, not physics.
2. **Per-render wall time grows non-linearly with `render_interval`.**
   A single render takes ~200 ms at RI=2 but ~2100 ms at RI=8. The
   render pipeline is not just firing less often — it's doing more
   work per fire as RI grows.

Fitting `ms/tick ≈ 28 × RI + 44` — a fixed ~44 ms per-tick floor
plus ~28 ms per `render_interval` unit. Doubling RI adds that much
proportionally.

Whatever the accumulation mechanism is, it makes higher `render_interval`
actively worse for sim-in-the-loop throughput. The instinct of "render
less frequently to save work" backfires: the infrequent renders become
slow enough to dominate anyway. Sweet spot is `render_interval=2` at
this scene on this hardware.

## Results — render_interval=1 (full fit check)

| config | /clock | /image |
|---|---|---|
| RI=1 | **15.88 Hz** | 4.00 Hz |
| RI=2 | 10.12 Hz | 2.50 Hz |
| RI=4 | 6.38 Hz | 1.62 Hz |
| RI=8 | 3.75 Hz | 0.88 Hz |

### Finding 10 — `render_interval=1` is the best bridge config on this stack

Actual 15.88 Hz beat the linear-fit prediction of 14 Hz. The
per-tick fixed floor is lower than the fit suggested; per-RI-unit
cost is the dominant term.

**3.2× improvement** over our starting 5 Hz with zero graph
rewrites — just two env-cfg overrides (`decimation=1`,
`render_interval=1`) threaded through `run_sim_in_the_loop.py`.

## Results — cameras-off bridge

Test: Depth env (80×60 render product) + `--no-camera-bridge`
(skip `_add_camera_stream` calls so the bridge has no
`ROS2CameraHelper` / `ROS2CameraInfoHelper` / render products of its
own, though the env's own render product still exists).

| config | /clock |
|---|---|
| Depth env + no-camera-bridge + RI=4 | 7.00 Hz |
| Perception env + camera bridge + RI=4 | 6.38 Hz |

### Finding 11 — bridge camera chain adds ~10% overhead; env-side render product is the coupling source

Skipping every `ROS2CameraHelper` + `ROS2CameraInfoHelper` in the
bridge only raised tick rate from 6.4 → 7.0 Hz. The majority of the
render-pacing comes from **Kit's render pipeline operating on any
`TiledCameraCfg` in the env** — regardless of whether the bridge
reads its output.

Confirms Finding 9's framing: the coupling lives inside Kit, not in
the bridge graph. Removing `ROS2CameraHelper` does not give us a
cameras-free baseline because the env's TiledCamera still forces the
render pipeline to run each tick.

A true cameras-off baseline would need:

1. An env cfg with no `TiledCameraCfg` (e.g., the `NoCam` variant).
2. `--enable_cameras` still on (Kit's Replicator SDG graph fails to
   wrap without the rendering kit).
3. …but `NoCam` + `--enable_cameras` currently crashes in
   `rep.set_global_seed()` because Kit's Replicator setup needs a
   render product to seed against. This is an Isaac Lab quirk; a
   true no-render bridge would require a custom env cfg with an
   invisible sentinel camera or monkey-patching Replicator seed.

## Closing summary

Starting point: **5 Hz** bridge-on with default training cfg (HQ
Infinigen scene, `decimation=4`, `render_interval=4`).

End state: **15.88 Hz** with `decimation=1` and `render_interval=1`
defaults on `run_sim_in_the_loop.py`. 3.2× speedup with zero changes
to scene geometry, camera resolution, or bridge graph structure.

Diagnostic lessons worth keeping:

- **Physics decimation** (not render cost) was the biggest leveraged
  win — RL training's `decimation=4` doesn't fit bridge mode where
  wall-clock rate beats sim-time compression.
- **Render coupling happens inside Kit**, gated on the env's
  TiledCamera render-product regardless of who consumes its output.
- **`render_interval` higher actively hurts**: per-render wall-time
  grows faster than the savings of fewer renders.
- **Bridge camera publish (`ROS2CameraHelper`) is a small contributor**
  — the frame-skip knob works correctly (image publish rate
  does divide) but doesn't unblock the per-tick stall.

Further speedup paths beyond this investigation's scope:

1. Move cameras off the per-tick path via a second OmniGraph driven
   only on `omni.replicator.core`'s render event (non-trivial; OmniGraph
   doesn't have a clean render-event source out of the box).
2. Reduce per-render Kit work — DLSS off, denoiser off, RTX settings
   tuning. Possibly significant on DGX Spark's Blackwell GB10.
3. Custom env cfg without a `TiledCamera` in the scene, paired with
   bridge-side explicit RenderProduct attach on a separate graph.
   Most work, most upside.

Done for now — 15.88 Hz is enough to drive Nav2 + RTAB-Map cleanly.

## Benchmark reproduction

Current script at `/tmp/bridge_perf_rollout.py`. Accepts `--res WxH`,
`--num-envs N`, `--steps N`. Run with no bridge running:

```
TERM=xterm ~/Workspace/IsaacLab/isaaclab.sh -p /tmp/bridge_perf_rollout.py \
    --headless --enable_cameras --res 640x360 --num-envs 1
```

Should move into `source/strafer_lab/scripts/` once this investigation
stabilizes — it's a generally useful tool, not just a one-off.

## Results — Isaac Lab 3.0 / Isaac Sim 6 migration

Measured `make sim-bridge` on the `phase_15-isaaclab3` branch
(phase_15 + `feature/isaaclab-3.0-migration`) against the phase_15
baseline. Same host, same scene pool, same subscriber script.

### Initial result — 3.4× regression

| Stack | `/clock` Hz | `/image` Hz |
|---|---:|---:|
| phase_15 tip (Isaac Lab 2.3.2, Sim 5.1) | 15.88 | 4.00 |
| phase_15-isaaclab3, first measurement (Lab develop, Sim 6) | 4.70 | 1.05 |

Stable across three 10 s samples (4.60 / 4.70 / 4.70 Hz `/clock`). GPU
SM utilization sat at 59–69 %, within 5 pp of the phase_15 envelope, so
the GB10 wasn't more-pinned — per-iteration wall time grew while SM
occupancy stayed put.

### Knob sweep — the usual levers didn't move the needle

| Config | `/clock` Hz |
|---|---:|
| default (decimation=1, render_interval=1) | 4.70 |
| decimation=1, render_interval=4 | 4.67 |
| decimation=4, render_interval=4 | 5.13 |
| `--no-camera-bridge` (skip ROS2 camera publish) | 5.30 |
| `RenderCfg(antialiasing="Off", dl_denoiser=False)` | 3.55 |
| `RenderCfg(antialiasing="DLSS", dlss_mode=0)` | 4.53 |
| Kit-GUI runtime DLSS 4×/Perf toggle | 8.07 |
| kit_args `--/omni/replicator/asyncRendering=1`, `--/app/asyncRendering=1`, `--/app/omni.usd/asyncHandshake=1` | 5.07 |
| kit_args async + `/app/hydraEngine/waitIdle=0` + `/app/runLoops/main/rateLimitEnabled=false` | 5.00 |

None of the render/physics/carb knobs that worked on 5.1 had an effect
on 3.0. Even disabling DLSS actively hurt, and the GUI DLSS toggle's
partial gain turned out to come from a carb path the `RenderCfg` API
doesn't expose. That was the signal that the bottleneck wasn't in the
render path at all.

### Step profiler — env.step is fine

Instrumented `ManagerBasedRLEnv.step` phase-by-phase (standalone, no
bridge graph, same task and overrides):

| phase | mean ms | share |
|---|---:|---:|
| `sim.step` (physics, TGS solver w/ 16 vel iters) | 30.5 | 58.5 % |
| `obs.compute` (incl. lazy tiled-camera render) | 15.7 | 30.3 % |
| all others | ~6 | 11.2 % |
| **total env.step** | **52.2** | — |

Implied rate **19.17 Hz**. Attaching the full bridge graph without
external subscribers: 63 ms, **15.83 Hz**. So env.step itself on 3.0 is
perfectly healthy — within the 5.1 envelope.

### Root cause — Kit main-loop no longer ticked per env.step

On Isaac Lab 2.x, `SimulationContext.render()` internally advanced
Kit's main loop, which fired `OnPlaybackTick` and ran the ROS2 bridge
OmniGraph once per env.step. On Isaac Lab 3.0, `render()` was
refactored to a lightweight "update visualizers" that no longer calls
`simulation_app.update()`. Result: the bridge graph falls back to
Kit's background cadence (~4 Hz) regardless of env.step rate.

Direct confirmation via per-bucket timing inside the bridge's while
loop: env.step ran steady at 15–17 Hz, but `/clock` only reached the
subscriber at 4.6 Hz — a clean 4× under-publication tied to Kit's
background main-loop cadence rather than to any render or physics cost.

### Fix — one-line `simulation_app.update()` in `_run_bridge_mode`

`source/strafer_lab/scripts/run_sim_in_the_loop.py` now calls
`simulation_app.update()` after `env.step()` in the bridge while-loop.
That explicitly ticks Kit once per iteration, firing `OnPlaybackTick`
and advancing the ROS2 OmniGraph.

| Stack | `/clock` Hz | `/image` Hz |
|---|---:|---:|
| phase_15 tip (Isaac Lab 2.3.2, Sim 5.1) | 15.88 | 4.00 |
| phase_15-isaaclab3 **before** fix | 4.70 | 1.05 |
| phase_15-isaaclab3 **after** `simulation_app.update()` fix | **10.20** | **2.40** |

10.2 Hz `/clock` and 2.4 Hz `/image` across three back-to-back samples
(10.10 / 10.30 / 10.20 Hz). That's 2.17× over the regression baseline
and 64 % of the phase_15 tip. The remaining gap is `simulation_app.update()`
overhead itself (~38 ms per iteration on top of the 62 ms env.step),
which is the cost of evaluating the entire OmniGraph and flushing DDS
publishers every tick.

### Branch status

Shippable. 642 autonomy/VLM tests pass, bridge boots clean, Infinigen
scene setup survives, and the perf regression has been reduced from
-70 % to -36 %. The remaining gap is a real but smaller tax of moving
from Isaac Sim 5.1 to Isaac Sim 6 — closing it further would require
either (a) making Kit's main-loop tick cheaper (batch or skip expensive
extensions), or (b) moving the ROS2 publishers off the env.step hot
path via a dedicated async publisher thread. Both are follow-on work;
neither blocks merging this branch into `phase_15`.
