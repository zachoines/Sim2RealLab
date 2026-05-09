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
- [`context/repo-topology.md`](../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
- [`context/conventions.md`](../context/conventions.md)

Parent design doc:
[`MISSION_VALIDATION_ARCHITECTURE.md` §3.6.c](../../MISSION_VALIDATION_ARCHITECTURE.md#36c-in-process-oracle-future-for-scale-supplements-only).

Sibling drivers (read first to understand the schema this brief
emits against):
- [`harness-teleop-driver`](harness-teleop-driver.md)
- [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md)

## Trigger condition — when to pick this brief up

**Do not pick up this brief until at least one of:**

- The teleop driver has shipped and a v2 VLA training run has
  demonstrated that throughput (not quality) is the binding
  constraint — i.e., the model would benefit from 10× more demos
  but operator time isn't available.
- A scene-coverage ablation requires sweeping >100 scene seeds
  and teleop coverage at that scale is impractical.
- A specific downstream brief (e.g.,
  `mvp-teacher-vla-distillation.md`) explicitly requires
  parallel-env data collection.

If none of those have fired, this brief stays parked. The
sketch exists to claim the slot so that when the trigger does
fire, the brief slot is ready.

## Architectural sketch

The driver is **in-process Isaac Lab + a scripted policy + the
existing `actions.jsonl` writer**, parallelizable via Isaac Lab's
native multi-env infrastructure (`ManagerBasedRLEnv` already
supports thousands of parallel envs on the DGX).

The scripted policy is the smallest thing that works:

- **Path planning:** A* on the scene's navigable mask (Infinigen
  metadata already exposes navigable space).
- **Path tracking:** simple proportional controller `(vx, vy)`
  toward the next waypoint; no MPPI, no Nav2.
- **Stop heuristic:** within R = 0.5 m of `target_position_3d`,
  emit `stop=True`.
- **Hard-negative injection:** reuse the harness's
  `--inject-bad-grounding` flag — perturb the goal post-projection
  so the oracle drives to the wrong target deliberately.
- **Path-shape constraint** (case 3, optional):
  ingest `path_constraints[]` from the procedural generator
  (filed as
  [`harness-procedural-path-shape-generator`](harness-procedural-path-shape-generator.md))
  and bias the A* cost function — wall-following becomes a soft
  cost on distance-from-wall, "via room R" becomes a waypoint.

The policy is *intentionally crude*. The point isn't to produce
expert demos — it's to produce *more* demos. Quality comes from
teleop; scale comes from oracle.

## Acceptance criteria (preliminary; expand at pickup time)

- [ ] **Parallel-env orchestration.** Driver runs N parallel
      Isaac Lab envs (target N=64 minimum on the DGX); each env
      gets its own scene seed and mission queue.
- [ ] **Schema parity.** Output matches the
      [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md)
      schema bit-for-bit; `actions.jsonl` rows tagged
      `source: "oracle"`.
- [ ] **Throughput target.** ≥ 100× teleop throughput in
      success-episodes-per-hour (single operator vs. parallel
      DGX run). Measured + reported in the PR.
- [ ] **Honest quality assessment.** Brief PR includes a
      side-by-side comparison: 30 oracle demos vs. 30 teleop
      demos on the same missions. Operator notes which
      qualitative differences are visible (path smoothness,
      stop accuracy, hesitation patterns).
- [ ] **Doc surface.**
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
      gains a Stage 5c — Oracle data collection section.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.

## Out of scope

- **Replacing teleop.** Oracle is a *supplement*, not a
  replacement. Teleop demos remain the quality anchor.
- **Real-robot use.** Sim-only. The oracle policy depends on
  full scene metadata that's only available in sim.
- **Sophisticated controllers.** Proportional control on A*
  waypoints is the bar. MPPI / RL / behavior-tree controllers
  are over-scoped here — that's what teleop and the v1 stack
  cover.
- **Curriculum / active sampling.** Future brief if the simple
  random-mission distribution underdelivers.
