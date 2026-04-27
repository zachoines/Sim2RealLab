# Kit-pump redundancy investigation

**Type:** investigation
**Owner:** DGX agent
**Priority:** P1
**Estimate:** M (a day or two of measurement + small refactor if confirmed)

## Story

As a **bridge / `--viz kit` operator**, I want **only one Kit `app.update()`
per env.step iteration**, so that **headed bridge-mode and `--video`
training runs aren't paying ~80 ms/loop for what looks like a
redundant viewport refresh on top of the OmniGraph publish pump**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)

## Context

Phase-level profiling of the sim-in-the-loop bridge with
`--viz kit --enable_cameras` (perception 640×360 + policy 80×60,
decimation=1) measured this attribution per loop:

| Phase                                   | p50 (ms) | What |
|-----------------------------------------|---------:|------|
| `env.step :: sim.render (Kit)`          | 88.6     | KitVisualizer.step calls `app.update()` with `playSimulations=False` (per `kit_visualizer.py:118-128`) |
| `simulation_app.update` (after env.step)| 83.1     | Bridge loop manually pumps Kit again with `playSimulations=True` (commit `0deda2b` rationale: drive camera-publish OmniGraph) |
| `env.step :: sim.step (PhysX)`          | 21.8     | Physics |
| Loop wall total                         | ~213     | 4.7 Hz |

Decomposition (cross-checked with the camera-bridge-off and headless
profile runs in `docs/PERF_INVESTIGATION_SIM_IN_THE_LOOP.md`):

- The 88 ms in `sim.render` is **100% editor viewport rendering** — when
  KitVisualizer is unregistered (headless), `sim.render` collapses to
  ~0.05 ms.
- The 83 ms in `simulation_app.update` is mostly **camera-bridge
  OmniGraph evaluation** (~74 ms) + a small viewport refresh (~9 ms).
- The two pumps look like they each refresh the editor viewport once
  per loop iteration, which is redundant.

The architectural reason both pumps exist:

- KitVisualizer.step pumps with `playSimulations=False` to avoid Kit's
  main loop double-stepping physics (env.step's PhysX call already
  advanced sim time). With `playSimulations=False`, `OnPlaybackTick`
  OmniGraph nodes do NOT fire — so the camera publish chain is silent.
- The bridge's `simulation_app.update()` pumps with `playSimulations=True`
  to wake up `OnPlaybackTick` so cameras publish per env.step (commit
  `0deda2b`). Side-effect: the editor viewport is refreshed a second
  time.

So the redundancy is between the two pumps' viewport-refresh side
effects, not between their primary jobs (one drives UI, one drives
OmniGraph publish under different timeline-play conditions).

The investigation should answer:

1. Is there a way to drive the camera-publish OmniGraph without
   `playSimulations=True` (e.g. retargeting nodes off `OnPlaybackTick`
   to a clock-driven evaluator)? If yes, we can drop the second pump.
2. Alternatively, can KitVisualizer.step be reconfigured to pump with
   `playSimulations=True` safely (Isaac Lab's `SimulationContext.step`
   already advanced PhysX, so we'd need to verify Kit's main-loop tick
   doesn't double-step)? If yes, we can drop `simulation_app.update`.
3. Failing both, can we make one of the two pumps cheaper — e.g. lower
   editor viewport resolution / DLSS-by-default / disable RTX features
   in the Kit settings the visualizer pumps?

## Scope of impact

- **Bridge GUI mode (`make sim-bridge-gui`)**: ~80 ms/loop saved if
  redundancy resolves, doubling effective throughput from ~5 Hz to ~9-10 Hz.
- **`Scripts/play_strafer_navigation.py --viz kit`** (inference rollouts
  with the editor viewport open): same savings.
- **`Scripts/train_strafer_navigation.py --video`**: training auto-injects
  `--viz kit` for the env-relative video camera positioning. So `--video`
  training runs ALSO pay this cost. Speeds up video-on training.
- **Pure RL training (no `--video`, no `--viz kit`)**: NO impact — the
  KitVisualizer is never registered in headless training, so the
  redundant pump doesn't exist. This is not an RL-training perf win.

## Acceptance criteria

- [ ] Empirically confirm whether the two pumps each render the editor
      viewport (instrument the Kit settings or use NVIDIA Nsight to
      attribute GPU work to each `app.update()` call).
- [ ] One of the following lands:
  - The bridge drops `simulation_app.update()` and camera publishing
    still works (acceptance: `ros2 topic hz /d555/color/image_raw` ≥
    target rate after the change).
  - KitVisualizer.step pumps with `playSimulations=True` and PhysX
    isn't double-stepped (acceptance: `/clock` advance per env.step
    matches `physics_dt * decimation` to within 1 %).
  - Both pumps stay but one is made significantly cheaper (acceptance:
    `--profile` p50 of the targeted phase drops by ≥ 50 %).
- [ ] No regression in headless bridge mode (`make sim-bridge`): same
      throughput as before, same camera publish rates.
- [ ] No regression in headless RL training: same iteration time as
      before.
- [ ] `docs/PERF_INVESTIGATION_SIM_IN_THE_LOOP.md` updated with the
      finding and the chosen fix.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

Files most likely involved:

- `source/strafer_lab/scripts/run_sim_in_the_loop.py` — the
  `simulation_app.update()` call site in the bridge mainloop, the
  comment explaining why it was added (commit `0deda2b`).
- `source/strafer_lab/strafer_lab/bridge/graph.py` — where the camera
  OmniGraph nodes are wired. Check what trigger their publishers use
  (`OnPlaybackTick` vs `OnImpulseEvent` vs others).
- `~/Documents/repos/IsaacLab/source/isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer.py:108-130` —
  KitVisualizer.step; the `playSimulations` toggle that governs whether
  `OnPlaybackTick` fires during the visualizer's pump.

Useful diagnostic command (the `--profile` harness from
`run_sim_in_the_loop.py`):

```bash
$ISAACLAB -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
    --mode bridge --enable_cameras --viz kit \
    --profile --profile-interval 10 --profile-window 200
```

## Out of scope

- Lowering perception camera resolution (creates a sim-to-real gap;
  see resolution analysis in `STRAFER_AUTONOMY_NEXT` thread).
- Moving camera publish off OnPlaybackTick onto a Python rclpy thread —
  that's the [async-camera-publishers task](async-camera-publishers.md);
  larger refactor, complementary to this one.
