# Bridge throughput toward the predicted 25 Hz ceiling

**Type:** task / investigation + refactor
**Owner:** DGX agent
**Priority:** P2 (follow-up to `async-camera-publishers`; lifts bridge
mainloop throughput from ~13.5 Hz toward the per-phase ceiling the
async-camera brief identified at ~25 Hz / 40 ms loop)
**Estimate:** M (~2-3 days: a measurement pass + scoped refactor for
the lever that pays out; longer if multiple levers land in series)
**Branch:** task/bridge-throughput-toward-25hz

## Story

As a **bridge operator running `make sim-bridge --enable_cameras` who
just landed the async-camera-publisher migration**, I want **the
remaining ~31 ms `simulation_app.update` residual and the ~20 ms
IsaacLab manager-loop overhead inside `env.step` trimmed**, so that
**the bridge mainloop reaches the throughput ceiling the
`async-camera-publishers` brief predicted (~22 ms PhysX + ~18 ms
manager loop ≈ 40 ms / 25 Hz)** without breaking any of the
in-process harness drivers (teleop, oracle, bridge-driver) or
training.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)
- [`context/conventions.md`](../../context/conventions.md)

Sibling briefs whose contract this brief must not break:
- [`teleop-driver`](../harness/teleop-driver.md) — in-process
  driver that calls `env.step()` from a gamepad. Captures
  `(frame, depth, pose, cmd_vel, mission_text)` via direct scene-
  handle access (not via the observation tensor). Trimming the
  observation manager is safe **for this driver**.
- [`oracle-driver`](../../parked/harness/oracle-driver.md) — in-process
  driver that runs the `NOCAM_SUBGOAL` RL policy. **Consumes the
  observation tensor.** Trimming the observation manager is NOT
  safe for this driver. Any manager-loop trim must be scoped so
  the oracle's env cfg keeps the full observation pipeline.
- [`trajectory-first-captioning`](../harness/trajectory-first-captioning.md) —
  random-target driver. Captures via scene handles like teleop.
  Safe.
- [`behavior-cloning-data-expansion`](../harness/behavior-cloning-data-expansion.md) —
  schema definition. Per-tick capture cadence must not regress.
- [`subgoal-env`](../trained-policy/subgoal-env.md) — the
  training env that produces the NoCam waypoint policy the oracle
  consumes. Training cfg must not be touched.

Prior art:
- [`completed/kit-pump-redundancy-investigation.md`](../../completed/kit-pump-redundancy-investigation.md) —
  the precursor that eliminated one Kit pump per loop by setting
  `env.render_enabled = False` in `_run_bridge_mode`. This brief
  takes the next step: reducing the cadence of the *remaining* pump.
- `f912d73` (telemetry async migration) + this PR (camera async
  migration) — the two refactors that put us at the current state
  where `simulation_app.update` no longer has any camera or
  telemetry work inside it. The 31 ms residual is pure Kit-pump
  scaffolding cost.

## Context

Post-`async-camera-publishers` measured state (DGX, headless,
InfinigenPerception, `decimation=1`, `render_interval=1`,
`--camera-frame-skip=3`):

| Phase | p50 (ms) | Notes |
|---|---:|---|
| `loop :: env.step (total)` | 42 | 22 PhysX + ~20 IsaacLab manager loop |
| `env.step :: sim.step (PhysX)` | 22 | locked — physics fidelity required by training |
| `loop :: simulation_app.update` | 31 | Kit pump residual; no camera or telemetry work |
| `loop :: publish_state` | 1 | StraferAsyncPublisher overhead |
| `loop :: camera notify_frame` | <1 | bridge-side GPU clone + event |
| `camera :: rclpy publish` | <1 | worker thread, post `array.array` fast path |
| `camera :: GPU→CPU readback` | 0.2 | worker thread on its own CUDA stream |
| **Loop total** | **~74** | → **13.5 Hz** |

The brief's predicted ceiling was **22 ms PhysX + 18 ms manager loop
≈ 40 ms / 25 Hz**. Gap to close: ~34 ms across `simulation_app.update`
(31 ms) and the in-`env.step` manager loop (~20 ms of the 42 ms
total).

### Three independent levers, measured by win-per-risk

**Lever 1 — Reduce `simulation_app.update` cadence.**
The remaining 31 ms is Kit's main-loop pump: extension tick
dispatch, USD-event flush, and the residual `OnPlaybackTick` →
`RunOnce` + `ROS2Context` node evaluation that the bridge graph
still hosts (kept as scaffolding by the camera-migration commit).
Today `_run_bridge_mode` calls `simulation_app.update()` every
loop iteration. Pumping every Nth iteration instead amortizes the
cost. Cadence N=4 → ~8 ms/tick → loop drops to ~51 ms → **~20 Hz**.

Risks:
- **Kit liveness.** Some Kit extensions assume periodic ticks for
  internal timers / GC. After ~1 s of no pump, things can wedge.
  Need to verify the minimum pump rate that keeps Kit healthy.
- **TiledCamera render-product freshness.** The async camera
  publisher reads `camera.data.output[...]` after each `env.step`.
  `TiledCamera` writes its output tensor during `sim.render`
  (which `env.step` already calls), not during
  `simulation_app.update`. So reducing the pump cadence should
  not stale camera tensors — but this needs explicit verification
  in the headless and headed paths.
- **USD event handlers.** Reset events (`stamp_d555_perception_opencv_pinhole`,
  `lift_ground_plane_to_floor`, etc.) fire on reset, not every tick.
  Should be unaffected, but verify with one full
  `navigate_to_pose` mission after the change.

**Lever 2 — Trim the IsaacLab manager loop inside `env.step`
(bridge-driver-scoped).**
Of the 42 ms in `env.step`, ~22 ms is PhysX and ~20 ms is the
observation / event / termination / reward manager dispatch. In
**bridge driver mode**, the policy isn't running — the action comes
from `/cmd_vel`, and the observation tensor is computed and
discarded. Skipping the observation manager (and possibly the
reward manager) would save ~5-15 ms with no behavior change for
this driver.

This lever is **bridge-driver-scoped**, NOT harness-wide:
- Safe for `--mode bridge` (the `_run_bridge_mode` runner).
- Safe for `--mode harness` with the bridge driver (autonomy
  executor publishes `/cmd_vel`; observation tensor unused).
- Safe for `harness-teleop-driver` (gamepad action; observation
  tensor unused; captures via scene handles).
- Safe for `harness-trajectory-first-captioning` (random-target;
  captures via scene handles).
- **NOT safe** for `harness-oracle-driver` — that driver runs
  the `NOCAM_SUBGOAL` RL policy in-process and feeds it the
  observation tensor. The oracle's env cfg must keep the
  full observation manager.

Implementation note: this should be a config-level toggle (e.g.
`env_cfg.observations = None` or a "lean bridge mode" env cfg
class), not a runtime monkey-patch. The toggle is applied by the
runner / driver that knows it doesn't need observations.

**Lever 3 — Delete the residual bridge OmniGraph.**
`OnPlaybackTick` / `RunOnce` / `ROS2Context` were kept in
[`graph.py`](../../../../source/strafer_lab/strafer_lab/bridge/graph.py)
as scaffolding by the async-camera-publishers commit. After the
migration, nothing downstream consumes them — they exist purely
for "future Kit-bound nodes might want to attach here," which is
hypothetical. Removing them saves the OmniGraph evaluation slice
of `simulation_app.update` (estimated 2-3 ms). Marginal on top
of Lever 1; mostly a cleanup that's worth doing before Lever 1
to give Lever 1 a cleaner baseline.

### Estimated combined ceiling

22 ms PhysX + ~5 ms thinned manager loop + ~8 ms amortized Kit
pump (N=4 cadence) + ~1 ms publish ≈ **~36 ms / ~28 Hz**.
Crosses the 25 Hz line predicted by `async-camera-publishers`.
Whether all three levers compose cleanly is the investigation
question.

## Scope of impact

- **Bridge headless throughput (`make sim-bridge`)**: 13.5 Hz →
  target ~25-28 Hz.
- **Bridge headed (`make sim-bridge-gui`, DISPLAY-attached)**:
  smaller relative win — `sim.render` editor-viewport RTX work
  (~88 ms p50) becomes the new ceiling, untouched by any lever
  here.
- **Harness drivers**:
  - Teleop, bridge, trajectory-first: same env throughput win as
    the bridge driver (~25 Hz), if they share the same lean env
    cfg.
  - Oracle: unaffected (keeps full manager loop).
- **Training (RL)**: untouched. `train_strafer_navigation.py` does
  not invoke `simulation_app.update` and depends on the full
  manager loop. See companion brief
  [`training-throughput-profile-and-investigate`](../investigations/training-throughput-profile-and-investigate.md)
  for training-side perf work.
- **Real-robot deployment**: zero impact (no bridge in the loop).

## Acceptance criteria

- [ ] `--profile` after the change shows `loop :: simulation_app.update`
      p50 ≤ 10 ms in `make sim-bridge` headless (target ≤ 8 ms with
      cadence N=4).
- [ ] `--profile` shows `loop :: env.step (total)` p50 < 30 ms when
      the bridge runs against the lean env cfg (i.e. Lever 2 applied
      to the bridge-driver path).
- [ ] Total loop p50 ≤ 50 ms in `make sim-bridge` headless (≥ 20 Hz).
      Stretch: ≤ 40 ms (≥ 25 Hz).
- [ ] `ros2 topic hz /d555/color/image_raw` and
      `/d555/depth/image_rect_raw` continue to match or exceed the
      pre-`async-camera-publishers` rates. Camera worker is at
      ~1400 Hz capacity; the constraint is bridge tick rate × frame
      skip. No frame drops introduced.
- [ ] A `navigate_to_pose` mission completes against the bridge
      with at least the same success rate as today. Run the
      [`next-integration-round`](../investigations/next-integration-round.md)
      sequence to validate.
- [ ] Harness driver coverage:
  - [ ] `harness-teleop-driver` (or `collect_demos.py` smoke as a
        proxy) runs against the lean env cfg without observation-
        related errors and writes per-tick frames.
  - [ ] `harness-oracle-driver` (or a manual smoke that loads the
        NoCam waypoint policy and runs `env.step` for one episode)
        continues to work against the **full** env cfg — confirms
        Lever 2's scoping holds.
- [ ] No regression in
      [`completed/kit-pump-redundancy-investigation`](../../completed/kit-pump-redundancy-investigation.md)'s
      acceptance state (`env.render_enabled = False` still in place;
      no second pump per loop).
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- **Pump cadence** —
  [`source/strafer_lab/scripts/run_sim_in_the_loop.py`](../../../../source/strafer_lab/scripts/run_sim_in_the_loop.py)
  in `_run_bridge_mode`: introduce a counter that calls
  `simulation_app.update()` every N iterations. Default N=1 (no
  change); promote to N=4 once Kit liveness is verified.
- **Manager-loop scoping** —
  [`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py):
  look at the `Realistic` env cfg variants. Either subclass with
  `observations = None` for the bridge path, or expose a
  "lean_bridge" env cfg registered as a separate task name
  (`Isaac-Strafer-Nav-Real-InfinigenPerception-LeanBridge-v0`).
- **OmniGraph deletion** —
  [`source/strafer_lab/strafer_lab/bridge/graph.py`](../../../../source/strafer_lab/strafer_lab/bridge/graph.py)
  in `build_bridge_graph`: drop the function body or make it a
  no-op. Remove the call site if the OmniGraph adds no value.
- **Profiler attribution** — Lever 1 will move work into
  `env.step :: sim.render (Kit)` if Kit's pump is partly invoked
  from there. Confirm with a fresh `--profile` run.
- **Camera tensor freshness** — under reduced pump cadence, check
  that `camera.data.output["rgb"]` advances on every `env.step`,
  not just on pump ticks. The async camera publisher's
  `notify_frame` is called every env.step — if tensors don't
  advance, the publisher will re-publish the same frame, visible
  as a frozen image in Foxglove.

## Risks / open questions

- **Kit pump cadence floor.** What's the minimum pump rate Kit
  tolerates before extensions wedge? May be Kit-version-dependent.
  Need empirical verification (run with N=8, N=16, look for stalls).
- **Lever 2 interaction with harness oracle driver.** If Lever 2
  is applied as a *new* env cfg (rather than a runtime toggle),
  the oracle brief at pickup time must explicitly select the full-
  manager cfg. Document the boundary in
  `context/bridge-runtime-invariants.md` so the oracle agent does
  not accidentally inherit the lean cfg.
- **OmniGraph deletion ordering.** If Lever 1 lands first with the
  OmniGraph still in place, the cadence reduction may make
  `OnPlaybackTick` firing irregular. Land Lever 3 (OmniGraph
  deletion) before or simultaneously with Lever 1 to keep
  causality clean.
- **The `loop :: camera notify_frame` p99 of 40 ms** (observed in
  the post-fast-path profile) suggests occasional CUDA stream
  back-pressure when the renderer stream is busy. Not a blocker
  for this brief, but worth a glance — may be cheap to mitigate
  with a `torch.cuda.Event` on the renderer stream that the
  bridge thread waits on once per loop.

## Out of scope

- **PhysX simplification.** The 22 ms `sim.step` cost reflects
  mecanum joint constraints + collision shapes tuned for sim-to-
  real fidelity. Touching it crosses into training-quality
  territory and belongs in a dedicated physics-tuning brief, not
  here.
- **DDS reliability QoS switch.** Switching the image topic from
  `RELIABLE` to `BEST_EFFORT` (to match Isaac Sim's
  `ROS2CameraHelper` default and the real RealSense driver)
  doesn't affect loop throughput — the camera worker is already
  at ~1400 Hz capacity post-`array.array` fast path. It belongs
  in a separate cross-lane brief because the Jetson's
  `timestamp_fixer` subscribes RELIABLE; switching requires a
  paired Jetson-side change.
- **Training throughput.** Companion brief
  [`training-throughput-profile-and-investigate`](../investigations/training-throughput-profile-and-investigate.md)
  covers RL training perf separately. The two paths share only
  the in-`env.step` IsaacLab manager loop, and even there the
  binding constraints differ (training is dominated by GPU
  rendering + policy forward pass at high env counts).
