# In-process oracle-policy driver for parallel scale-out data capture

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
- [`harness-teleop-driver`](harness-teleop-driver.md)
- [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md)
- [`harness-mission-generator`](harness-mission-generator.md) —
  produces the `mission_queue.yaml` rows whose `planned_path`
  this brief consumes.
- [`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md) —
  produces the connectivity graph the A* fallback consumes for
  multi-room paths.

## Trigger condition — when to pick this brief up

**Do not pick up this brief until BOTH:**

1. [`strafer-lab-subgoal-env`](strafer-lab-subgoal-env.md) has
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

If neither side of the AND has fired, this brief stays parked.
The sketch exists to claim the slot so that when the triggers do
fire, the brief slot is ready.

**Why the RL-controller dependency is hard.** The proportional
fallback exists for debugging, but the *actual* value of the
oracle is producing demos whose action distribution matches
deployment. That requires the trained NoCam waypoint follower
from `strafer-lab-subgoal-env`. Picking up this brief without
that prerequisite produces low-quality demos that don't justify
the implementation effort.

## Architectural sketch

The driver is **in-process Isaac Lab + the MVP RL controller +
the existing `actions.jsonl` writer**, parallelizable via Isaac
Lab's native multi-env infrastructure (`ManagerBasedRLEnv`
already supports thousands of parallel envs on the DGX).
Multi-room is supported by default — A* on the navigable mask
handles cross-room paths through the connectivity graph from
[`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md).

Pipeline:

```
mission_queue.yaml row  → planned_path (LLM-emitted)  → next waypoint → NoCam RL controller → cmd_vel → env
(from harness-mission-      (consumed directly,            (consumer of the    (loaded via
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
  [`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md).
- **Path tracking:** the MVP NoCam RL controller produced by
  [`strafer-lab-subgoal-env`](strafer-lab-subgoal-env.md) — a
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
      Isaac Lab envs (target N=64 minimum on the DGX); each env
      gets its own scene seed and mission queue.
- [ ] **Schema parity.** Output matches the
      [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md)
      schema bit-for-bit; `actions.jsonl` rows tagged
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
- [ ] **Throughput target.** ≥ 100× teleop throughput in
      success-episodes-per-hour (single operator vs. parallel
      DGX run, with the RL controller). Measured + reported in
      the PR. Proportional-fallback throughput reported
      separately for reference.
- [ ] **Honest quality assessment.** Brief PR includes a
      side-by-side comparison: 30 oracle demos vs. 30 teleop
      demos on the same missions. Operator notes which
      qualitative differences are visible (path smoothness,
      stop accuracy, hesitation patterns).
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
  [`strafer-lab-subgoal-env`](strafer-lab-subgoal-env.md);
  it does not retrain the controller. If the trained controller
  underperforms, the proportional fallback is the safety net,
  and a separate brief decides whether to retrain.
- **MPPI or behavior-tree controllers.** Out of scope. The RL
  controller (or proportional fallback) is the bar. MPPI lives
  in the v1 stack; teleop and the v1 stack cover that path.
- **Curriculum / active sampling.** Future brief if the simple
  random-mission distribution underdelivers.
