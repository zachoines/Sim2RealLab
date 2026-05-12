# Measure harness parallel-env throughput before scale-out briefs commit to numbers

**Type:** investigation
**Owner:** DGX agent
**Priority:** P2 (blocks
[`oracle-driver`](oracle-driver.md) acceptance bar +
[`trajectory-first-captioning`](../../active/harness/trajectory-first-captioning.md)
acceptance bar; not strictly required before teleop ships but
should ship before any "scale up to thousands of trajectories"
work begins)
**Estimate:** S–M (~1–2 days; phase profiler + parallel-env sweep
+ VRAM measurement + writeup; no implementation of new drivers)
**Branch:** task/harness-throughput-measurement

## Story

As a **DGX operator about to commit weeks of compute to either
the oracle driver or trajectory-first captioning at "thousands of
parallel envs" scale**, I want **a single measurement pass that
characterizes actual harness throughput across realistic env
counts on the actual scene configs (NoCam vs. perception cam),
GPU memory, and concurrent-load conditions**, so that **the
scale-out briefs' `num_envs` arguments and trajectories/hour
acceptance criteria are anchored to measurement, not assertion,
and so the operator can pick between the single-config vs.
two-pass (NoCam-driver + perception-replay) architectures with
real numbers**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)

Adjacent perf work:
- [`training-throughput-profile-and-investigate`](../../active/investigations/training-throughput-profile-and-investigate.md) —
  RL-training throughput profiling; same machinery, different
  consumer. Don't duplicate; reuse the phase-profiler scaffold
  this brief shares.
- [`bridge-throughput-toward-25hz`](../../active/sim-performance/bridge-throughput-toward-25hz.md) —
  bridge mainloop perf; this brief is about *data-collection*
  parallel throughput, not the cross-host bridge path.

Sibling briefs whose acceptance depends on this:
- [`oracle-driver`](oracle-driver.md) — `num_envs` target +
  "100× teleop" target are unmeasured assertions today.
- [`trajectory-first-captioning`](../../active/harness/trajectory-first-captioning.md) —
  "64 parallel envs → 2k trajectories / hour" is asserted.

## Context

### The assertion the audit can't verify

The current oracle-driver and trajectory-first briefs both
assume **≥ 64 parallel envs on the DGX**, producing **~2k
trajectories / hour**. Neither has measurement to back this up.
The scene config they'd actually run against —
`Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0` — flags
itself in
[`strafer_env_cfg.py:335-337`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py#L335-L337)
as capping at **~1–8 envs** because `replicate_physics: bool =
False` (Infinigen prim layout is per-env) and the 640×360
perception camera VRAM footprint dominates.

There are two scene configs the brief should measure:

1. **`Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0`** —
   what the captioning + perception-feature consumers actually
   need camera frames from. Expected to top out at single-digit
   envs.
2. **`Isaac-Strafer-Nav-Real-NoCam-Play-v0`** — the NoCam
   variant used for RL training. Expected to support 256+ envs
   per the same scene-config comment. Useful for the
   "drive-then-replay" two-pass architecture
   ([`trajectory-first-captioning`](../../active/harness/trajectory-first-captioning.md)
   names this option).

Two architectures sit downstream of the measurement:

- **Single-config (perception scene only).** Run the driver on
  the perception scene at whatever `num_envs` the measurement
  allows; live capture; one pass.
- **Two-pass (NoCam-driver + perception-replay).** Run the
  driver on the NoCam scene at high parallelism to *generate
  trajectories* (no cameras); replay each trajectory under the
  perception scene at low parallelism to *harvest frames* for
  captioning. Trades complexity (requires deterministic replay)
  for throughput.

This brief measures both branches so the scale-out briefs can
pick.

### What the measurement reports

A short doc / table per scene config + env count, capturing:

| Scene config | num_envs | Steps/sec (per env, mean) | Steps/sec (aggregate) | VRAM (GB) | RAM (GB) | Boot time (s) | Notes |
|---|---|---|---|---|---|---|---|

Plus:

- **Phase profile per env-count point**: PhysX, render, action,
  recorder, RAM allocator — surface where the per-env-step time
  goes so the brief can flag whether parallelism is
  render-bound, PhysX-bound, or recorder-bound.
- **Failure / instability points**: at what env count does Isaac
  Sim OOM, crash, or thrash? Run until failure on at least one
  scene config.
- **Aggregate trajectories/hour**: derive from steps/sec ×
  num_envs / (mean trajectory length in steps). Use existing
  harness mission generator output to set realistic trajectory
  length.

### What this brief does NOT change

- No new drivers. No new training. No new recorder. This is a
  measurement pass that writes a doc + posts numbers, then
  files follow-up briefs against measured hot spots.
- No changes to scene configs. The 1–8 env cap on the
  perception scene is the measurement *result*, not something
  to "fix" — fixing it would require re-architecting the scene
  config or accepting smaller cameras / lower per-env memory.

## Acceptance criteria

- [ ] **Phase profiler hooked into the harness.** The
      `_PhaseProfiler` scaffold in
      [`run_sim_in_the_loop.py:258-350`](../../../../source/strafer_lab/scripts/run_sim_in_the_loop.py#L258-L350)
      already exists for the bridge mode; extend (or reuse) for
      a no-bridge in-process harness measurement script
      (`Scripts/measure_harness_throughput.py` or similar).
- [ ] **Sweep across scene configs and env counts.** Measure
      at least:
      - `Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0`:
        `num_envs ∈ {1, 2, 4, 8, 16}` (continue until OOM /
        crash; report failure point).
      - `Isaac-Strafer-Nav-Real-NoCam-Play-v0`:
        `num_envs ∈ {1, 16, 64, 256, 1024}` (continue until
        OOM / crash).
- [ ] **Per-env metrics.** Steps/sec/env (mean + p99), aggregate
      steps/sec, VRAM peak, RAM peak, boot time. Report in a
      table.
- [ ] **Trajectory-per-hour projection.** Multiply measured
      steps/sec × env count / mean-trajectory-length-in-steps;
      cross-reference with mean trajectory length from a recent
      teleop session (or mission-generator output, if teleop
      hasn't shipped).
- [ ] **Two-pass replay feasibility note.** Smoke whether a
      trajectory recorded on the NoCam config can be replayed
      under the perception config: record one trajectory at
      `num_envs=1` on NoCam, then re-load the env at the
      perception config and replay the `actions.jsonl` rows.
      Compare the camera output to a fresh run with the same
      seed. Pass = visually identical frames; fail = drift.
- [ ] **Writeup.** A short doc — either a new `docs/HARNESS_
      THROUGHPUT_<date>.md` or an appendix in
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md) —
      with the measurement table, the dominant-bottleneck call
      per env count, and a recommendation for the scale-out
      briefs (single-config vs. two-pass).
- [ ] **Update the scale-out briefs** in the same PR (or file a
      follow-up if the recommendation requires architectural
      changes). Specifically, the `num_envs` target and the
      throughput-acceptance bullet in
      [`oracle-driver`](oracle-driver.md) and
      [`trajectory-first-captioning`](../../active/harness/trajectory-first-captioning.md)
      are reset to the measured ceiling.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- Phase-profiler scaffold:
  [`run_sim_in_the_loop.py:258-350`](../../../../source/strafer_lab/scripts/run_sim_in_the_loop.py#L258-L350).
- Scene config self-doc on env-count caps:
  [`strafer_env_cfg.py:335-337`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py#L335-L337).
- Isaac Lab `ManagerBasedRLEnv` num_envs scaling is documented
  in the published Isaac Lab paper (arXiv:2511.04831);
  cross-check expectations against published Isaac Lab Arena
  measurements (Isaac Lab Arena ran 4096 envs on 8× 6000D GPUs
  in 0.76h vs. 34.9h sequential — but that was a manipulation
  task, not the Infinigen-perception scene).
- VRAM headroom: DGX Spark unified-memory budget is shared
  across training and rendering. Concurrent training jobs will
  contend; the measurement should record idle-DGX numbers and
  flag the concurrency caveat.

## Out of scope

- **Driver implementation.** This is investigation only. The
  oracle / trajectory-first / replay drivers are filed
  separately.
- **Re-architecting the perception scene to support more envs.**
  If the measurement shows the perception scene is fundamentally
  capped, file a follow-up; don't try to fix it inside this
  brief.
- **Real-robot data capture.** Sim-only; the parallel-env
  question doesn't exist on hardware.
- **Bridge mainloop throughput.** Covered by
  [`bridge-throughput-toward-25hz`](../../active/sim-performance/bridge-throughput-toward-25hz.md);
  this brief is about in-process-only drivers (teleop, oracle,
  trajectory-first).
