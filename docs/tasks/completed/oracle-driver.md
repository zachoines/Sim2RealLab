# In-process oracle-policy driver for parallel scale-out data capture

**Status:** Retired 2026-05-24. Never picked up. Reason: folded
into the consolidated
[`harness-architecture`](../../active/harness/harness-architecture.md)
brief as
[Driver: scripted](../../active/harness/harness-architecture.md#driver-scripted)
+ the
[Scripted × queue](../../active/harness/harness-architecture.md#scripted--queue-oracle-path)
mission-source cell. The 2026-05-24 audit found that this driver
shares its action source (scripted RL controller + proportional
fallback) with the trajectory-first regime and the new "coverage"
regime for room-state eval — three mission sources, one driver.
Treating them as one driver with three mission-source variants is
the simplification. The `subgoal-env` pickup gate, the
proportional fallback, the quality-assessment metrics
(curvature variance / stop accuracy / hesitation / action
smoothness), and the parallel-env throughput-measurement gate
are preserved in the consolidated brief.

**Type:** new feature (sketch — not yet ready to pick up)
**Owner:** DGX agent
**Priority:** P3 (filed-on-trigger; do not pick up until the
trigger condition fires — see below)
**Estimate:** L (~week+; in-process Isaac Lab + scripted policy +
parallel-env orchestration + writer adaption)
**Branch:** task/harness-oracle-driver

## Story

As an **operator who has hit the throughput ceiling of teleop
data collection** (typically when scaling beyond ~10k
trajectories or ablating across many scene seeds), I want **a
scripted oracle policy that runs in-process Isaac Lab across
hundreds of parallel envs and emits the same harness schema as
teleop / bridge drivers**, so that **VLA training data scale is
no longer operator-bottlenecked, at the deliberate cost of demo
quality**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)

Parent design doc:
[`MISSION_VALIDATION_ARCHITECTURE.md` §3.6.c](../../../MISSION_VALIDATION_ARCHITECTURE.md#36c-in-process-oracle-future-for-scale-supplements-only).

Sibling drivers + the mission generator (read first to
understand the schema this brief consumes and emits against):
- [`teleop-driver`](../../active/harness/teleop-driver.md)
- [`behavior-cloning-data-expansion`](../../active/harness/behavior-cloning-data-expansion.md)
- [`mission-generator`](../../active/harness/mission-generator.md) —
  produces the `mission_queue.yaml` rows whose `planned_path`
  this brief consumes.
- [`scene-connectivity-validation`](../../active/multi-room/scene-connectivity-validation.md) —
  produces the connectivity graph the A* fallback consumes for
  multi-room paths.
- [`harness-throughput-measurement`](harness-throughput-measurement.md) —
  audit-filed; the parallel-env count this brief plans against
  must be grounded in the measurement before pickup.

## Trigger condition — when to pick this brief up

**Do not pick up this brief until BOTH:**

1. [`subgoal-env`](../../active/trained-policy/subgoal-env.md) has
   shipped, producing the NoCam RL waypoint-following checkpoint
   the oracle uses as its tracking layer.
2. **And** at least one of:
   - The teleop driver has shipped and a v2 VLA training run
     has demonstrated that throughput (not quality) is the
     binding constraint — i.e., the model would benefit from
     10× more demos but operator time isn't available.
   - A scene-coverage ablation requires sweeping >100 scene
     seeds and teleop coverage at that scale is impractical.
   - A specific downstream brief (e.g.,
     `mvp-teacher-vla-distillation.md`) explicitly requires
     parallel-env data collection.
3. **Recommended:**
   [`harness-throughput-measurement`](harness-throughput-measurement.md)
   has shipped or is shipping in the same PR, so this brief's
   `num_envs` argument and the "≥ 100× teleop throughput"
   acceptance criterion are anchored to a measured per-env
   throughput, not the legacy "Isaac Lab supports thousands"
   assertion. The perception scene config caps at ~1–8 parallel
   envs by Isaac Sim's per-env memory budget
   (`strafer_env_cfg.py:335-337`); the 100× factor is unreachable
   at that env count unless paired with a NoCam-driver + replay
   architecture (see throughput-measurement brief for the
   options).

If neither side of the AND has fired, this brief stays parked.
The sketch exists to claim the slot so that when the triggers do
fire, the brief slot is ready.

**Why the RL-controller dependency is hard.** The proportional
fallback exists for debugging, but the *actual* value of the
oracle is producing demos whose action distribution matches
deployment. That requires the trained NoCam waypoint follower
from `subgoal-env`. Picking up this brief without that
prerequisite produces low-quality demos that don't justify the
implementation effort.

## Architectural sketch

The driver is **in-process Isaac Lab + the MVP RL controller +
the existing `actions.jsonl` writer**, parallelizable via Isaac
Lab's native multi-env infrastructure (`ManagerBasedRLEnv`
supports many parallel envs in general; **on the specific
640×360-perception scene the cap is ~1–8 envs** per
`strafer_env_cfg.py:335-337`). Multi-room is supported by
default — A* on the navigable mask handles cross-room paths
through the connectivity graph from
[`scene-connectivity-validation`](../../active/multi-room/scene-connectivity-validation.md).

Pipeline:

```
mission_queue.yaml row  → planned_path (LLM-emitted)  → next waypoint → NoCam RL controller → cmd_vel → env
(from mission-              (consumed directly,            (consumer of the    (loaded via
generator)                  no in-oracle planning)         waypoint, fed as     strafer_shared.policy_interface
                                                            the policy goal)    .load_policy())
```

The architectural split: **the mission generator owns the
*what* (waypoint sequence, possibly path-shape-aware via the
LLM); the RL controller owns the *how* (waypoint tracking).**
Path-shape constraints affect waypoints only — they don't touch
the controller.

Fallback: when the mission generator has not pre-computed
`planned_path` for a queue row (e.g., `--mode endpoint`), the
oracle computes its own A* shortest-path on the navigable mask.
This is the legacy oracle behavior and is retained for backward
compatibility.

Components:

- **Path source (primary):** `planned_path` from the mission
  queue. Already validated against the navigable mask + scene
  bounds + connectivity graph at generation time.
- **Path source (fallback):** A* on the scene's navigable mask
  when the queue row has no `planned_path`. Multi-room paths
  use the connectivity graph from
  [`scene-connectivity-validation`](../../active/multi-room/scene-connectivity-validation.md).
- **Path tracking:** the MVP NoCam RL controller produced by
  [`subgoal-env`](../../active/trained-policy/subgoal-env.md) — a
  goal-conditioned waypoint follower trained on the actual
  mecanum dynamics. Loaded via
  `strafer_shared.policy_interface.load_policy()` from
  `~/.strafer/models/policy_*.pt` or `.onnx`. Recorded
  `actions.jsonl` rows are the controller's `(vx, vy, ωz)`
  output, which is what a deployed VLA would learn to emit at
  the controller-output level.
  - **Controller fallback:** `--controller proportional`
    selects a simple proportional `(vx, vy)` toward the next
    waypoint. Useful as a debug option if the RL checkpoint has
    issues at pickup time, or for sanity-check runs without the
    policy dependency.
- **Stop heuristic:** within R = 0.5 m of `target_position_3d`,
  emit `stop=True`.
- **Hard-negative injection:** reuse the harness's
  `--inject-bad-grounding` flag — perturb the goal post-projection
  so the oracle drives to the wrong target deliberately.

The oracle is *intentionally a scale supplement, not a v1
replacement*. Demos are tagged `source: "oracle"` in
`actions.jsonl` so downstream training can weight or filter them
differently from teleop or bridge demos. Quality comes from
teleop; scale comes from oracle.

## Acceptance criteria (preliminary; expand at pickup time)

- [ ] **Parallel-env orchestration.** Driver runs N parallel
      Isaac Lab envs; **target N is set by
      [`harness-throughput-measurement`](harness-throughput-measurement.md)'s
      measured ceiling for the chosen scene config**, not by
      assertion. Each env gets its own scene seed and mission
      queue.
- [ ] **Schema parity.** Output matches the
      [`behavior-cloning-data-expansion`](../../active/harness/behavior-cloning-data-expansion.md)
      schema bit-for-bit (or whatever
      [`output-format-alignment`](output-format-alignment.md)
      lands on); `actions.jsonl` rows tagged
      `source: "oracle"`.
- [ ] **RL controller integration.** Default `--controller rl`
      loads the NoCam waypoint-following checkpoint from
      `~/.strafer/models/` via
      `strafer_shared.policy_interface.load_policy()` and feeds
      it the next A* waypoint as the goal observation. Recorded
      `actions.jsonl` rows are the controller's
      `(vx, vy, ωz)` output. Smoke test: ≥ 1 episode per scene
      reaches its target via the RL controller.
- [ ] **Proportional fallback.** `--controller proportional`
      runs a simple proportional `(vx, vy)` controller toward
      the next waypoint, no policy load. Smoke test: 1 episode
      per scene completes.
- [ ] **Throughput target.** Reach the *measured ceiling*
      reported by
      [`harness-throughput-measurement`](harness-throughput-measurement.md).
      The legacy "≥ 100× teleop" target was not grounded in
      measurement and the perception-scene memory budget caps
      single-process parallelism at single-digit envs; the
      honest target is "reach what the scene config allows,"
      not "100×." Report measured throughput in the PR;
      proportional-fallback throughput reported separately for
      reference.
- [ ] **Quality assessment with falsifiable metrics.** Brief PR
      includes a side-by-side comparison: 30 oracle demos vs.
      30 teleop demos on the same missions, scored on:
  - **Path smoothness.** Trajectory curvature variance (sum of
    squared angular accelerations along the path, lower is
    smoother). Report mean ± std for both sources.
  - **Stop accuracy.** Distance from final pose to
    `target_position_3d` (lower is better). Report mean ± std.
  - **Hesitation.** Count of `|cmd_vel| < 0.05` ticks during
    the navigation phase divided by total navigation ticks
    (lower means fewer pauses).
  - **Action smoothness.** Mean ‖cmd_vel(t) − cmd_vel(t-1)‖
    over the episode (lower is smoother control).
  Report each metric per source as a table; flag where oracle
  diverges from teleop by >2σ as evidence that oracle demos
  should be filtered or weighted lower in downstream training.
- [ ] **Doc surface.**
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md)
      gains a Stage 5c — Oracle data collection section.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.

## Out of scope

- **Replacing teleop.** Oracle is a *supplement*, not a
  replacement. Teleop demos remain the quality anchor.
- **Real-robot use.** Sim-only. The oracle policy depends on
  full scene metadata that's only available in sim.
- **Training a new controller.** This brief reuses the NoCam
  waypoint-following policy from
  [`subgoal-env`](../../active/trained-policy/subgoal-env.md);
  it does not retrain the controller. If the trained controller
  underperforms, the proportional fallback is the safety net,
  and a separate brief decides whether to retrain.
- **MPPI or behavior-tree controllers.** Out of scope. The RL
  controller (or proportional fallback) is the bar. MPPI lives
  in the v1 stack; teleop and the v1 stack cover that path.
- **Curriculum / active sampling.** Future brief if the simple
  random-mission distribution underdelivers.
