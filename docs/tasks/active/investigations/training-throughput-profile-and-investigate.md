# Profile RL training throughput and file follow-up briefs

**Type:** investigation
**Owner:** DGX agent
**Priority:** P2 (no known training-blocking regression; this brief
exists to *measure* before any "training is slow" optimization is
attempted, because every optimization filed against an unmeasured
bottleneck is a coin flip)
**Estimate:** S-M (~1-2 days: add a phase profiler to the training
loop, run a representative training pass on the procroom env, file
findings as sub-briefs; no implementation work in this brief itself)
**Branch:** task/training-throughput-profile-and-investigate

## Story

As a **DGX operator running RL training on the strafer procroom env
who wants to know where wall-clock time is going during a training
run before deciding what to optimize**, I want **a phase-level
profiler hooked into `train_strafer_navigation.py` that attributes
per-iteration cost across PhysX, GPU rendering, IsaacLab manager
loop, policy forward / backward, and rsl_rl bookkeeping**, so that
**any subsequent throughput-optimization brief is filed against a
*measured* hot spot, not a hypothesized one, and so that the
bridge-side perf work this brief is paired with doesn't have its
training-throughput hypotheses left untested**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/conventions.md`](../../context/conventions.md)

Adjacent perf work:
- [`bridge-throughput-toward-25hz`](../sim-performance/bridge-throughput-toward-25hz.md) —
  the companion brief for *bridge* throughput. This brief is the
  *training* counterpart; the two paths share only the in-`env.step`
  IsaacLab manager loop, and the binding constraints differ
  (training scales with `num_envs` and GPU rendering, bridge is
  num_envs=1 wall-clock-paced).
- [`completed/kit-pump-redundancy-investigation`](../../completed/kit-pump-redundancy-investigation.md) —
  shows the pattern of profile-first-then-fix this brief follows.
  That brief added the `--profile` harness to the bridge runner;
  this brief adds an analogous one to the training runner.

Prior art (the bridge `--profile` harness this brief mirrors):
[`source/strafer_lab/scripts/run_sim_in_the_loop.py`](../../../../source/strafer_lab/scripts/run_sim_in_the_loop.py)
class `_PhaseProfiler`. Reusable for the training script with a
thin adapter — the per-phase context manager is generic; only the
phase names and the hook points differ.

## Context

Training throughput on the strafer procroom envs is not
currently measured at phase granularity. We know steady-state
training takes hours-to-days per run, but the wall-time budget is
not attributed across:

| Suspect phase | Why it matters |
|---|---|
| `sim.step` (PhysX) | Scales with `num_envs` and collision complexity. Procroom adds furniture clutter vs. the cleaner standard envs. |
| `sim.render` (GPU) | TiledCamera (80×60 policy cam) renders per env per step. At `num_envs=256` this is non-trivial GPU bandwidth. |
| IsaacLab manager loop | Observation / event / termination / reward dispatch every step. At high env counts the per-env cost amortizes but the per-step Python overhead remains. |
| Policy forward (`act`) | rsl_rl's per-step action sampling. CPU-GPU sync; potentially blocks on PhysX queueing if streams are entangled. |
| Policy backward + optimizer step | PPO update phase. Runs once per `num_steps_per_env` collection cycle. |
| rsl_rl bookkeeping | Reward logging, value normalization, GAE compute. |
| Video / video-writer | When `--video` is on, encoding cost shows up in `env.step` via the wrapped env. |

The brief filed against the *wrong* phase is the failure mode
this brief exists to prevent. A "training is slow" intuition
might lead someone to file "trim the manager loop" when the
actual bottleneck is GPU memory bandwidth at high env counts — or
vice versa. The right shape is: measure first, file targeted
sub-briefs second.

### What's known *without* measurement, and what's deliberately NOT acted on

- At `num_envs=256+` with the 80×60 policy camera, GPU rendering
  is *probably* a significant fraction of wall time — but
  "probably" is not actionable.
- The IsaacLab manager loop at 20 ms per step amortizes to
  ~0.08 ms/env at num_envs=256 — *probably* invisible — but
  whether that holds depends on which observation terms are
  vectorized vs. Python-iterating.
- Procroom-specific suspects: procedurally-generated scene
  geometry (Infinigen output) can have high triangle counts that
  hurt RTX render time per env. The
  [`source/strafer_lab/scripts/postprocess_scene_usd.py`](../../../../source/strafer_lab/scripts/postprocess_scene_usd.py)
  already strips floor mesh colliders; there may be analogous
  opportunities for walls / furniture.
- The video recording path (`--video`) is widely suspected to
  cost wall time disproportionate to its information value
  during long training runs, but no measurement exists today.

This brief produces the measurement. Acting on findings is
out-of-scope here; each finding worth fixing becomes its own
sub-brief.

## Scope of impact

- **Training**: zero behavior change. This brief adds a profiler
  flag; runs the script with the flag set; reads the output; files
  follow-up briefs.
- **Bridge**: no impact. The bridge already has its own profiler.
- **Real-robot**: no impact.

The deliverables are:
1. A `--profile` flag on
   [`source/strafer_lab/scripts/train_strafer_navigation.py`](../../../../source/strafer_lab/scripts/train_strafer_navigation.py)
   that records per-phase p50 / p99 over a rolling window and
   prints periodic reports.
2. A measurement report (committed alongside the profiler) of a
   representative procroom training pass at the default
   `num_envs` and decimation.
3. One or more follow-up briefs filed against measured hot spots,
   each scoped narrowly enough to be a single PR.

## Acceptance criteria

- [ ] **Profiler implemented.** `source/strafer_lab/scripts/train_strafer_navigation.py`
      gains a `--profile` flag mirroring the bridge runner's
      `--profile-interval` / `--profile-window` contract. Reuse
      `_PhaseProfiler` from
      [`run_sim_in_the_loop.py`](../../../../source/strafer_lab/scripts/run_sim_in_the_loop.py)
      directly or extract it into a shared module under
      `strafer_lab.tools` if the duplication is uncomfortable.
- [ ] **Phase coverage.** The profiler attributes wall time
      across at least: `env.step (PhysX)`, `env.step (render +
      manager loop)`, `policy act`, `policy update` (every
      `num_steps_per_env` ticks), and `video writer` (when
      `--video` is on). Manager-loop sub-attribution (observations
      vs. rewards vs. terminations vs. events) is a stretch; if
      hard to instrument, file it as a follow-up.
- [ ] **Representative measurement run.** Run a procroom training
      pass at the default `num_envs` (likely 256) and decimation
      (likely 4) for at least 200 PPO iterations — long enough
      for steady-state numbers, short enough to not consume a
      full training budget. With `--video off` first, then a
      second run with `--video on` to attribute video cost.
      Capture the last steady-state `[profile]` block from each.
- [ ] **Follow-up briefs filed.** For each measured phase with
      p50 ≥ 10% of the loop, file a `docs/tasks/active/<epic>/...md`
      brief with:
      - the measured numbers,
      - a hypothesis for why it costs what it does,
      - acceptance criteria scoped to that phase only,
      - explicit "out of scope" callout for the other phases.
      If no single phase clears 10%, file a single
      `training-throughput-no-low-hanging-fruit.md` summary
      brief noting that and pointing at the measurement.
- [ ] **No env / cfg changes.** This brief ships the profiler and
      the report. No training behavior changes. No cfg edits
      beyond adding the flag. If the profiler reveals a config
      mistake (e.g. an obs term that's accidentally expensive),
      that's a finding to file, not a fix to land here.
- [ ] **Docs surface.** Add a short note to the training operator docs
      ([`source/strafer_lab/README.md` → Run](../../../../source/strafer_lab/README.md#run)
      or the cheatsheet) on how to invoke `--profile` and how to read the
      output. Reuse the bridge runner's docs as the template.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- **Training entry point** —
  [`source/strafer_lab/scripts/train_strafer_navigation.py`](../../../../source/strafer_lab/scripts/train_strafer_navigation.py).
  The main loop is wrapped by `rsl_rl`'s `OnPolicyRunner`; the
  inner `env.step` call is what to instrument. The PPO update
  phase fires once every `num_steps_per_env` collection ticks
  inside `OnPolicyRunner.learn`.
- **Procroom env cfg** — `StraferNavEnvCfg_Real_ProcRoom_*` in
  [`strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py).
  Check what observation / reward terms are active; they show up
  in the manager-loop time.
- **Phase profiler reuse** —
  `_PhaseProfiler` in `run_sim_in_the_loop.py` already supports
  context-manager phase timing and monkey-patching `sim.step` /
  `sim.render` for sub-attribution. Lift it into
  `source/strafer_lab/strafer_lab/tools/phase_profiler.py` for
  clean reuse if the duplication bothers you; otherwise import-by-
  path is fine for a one-investigation brief.
- **Video cost suspicion** — if `--video on` shows a large jump
  in `env.step`, the suspect is the video wrapper inside
  Isaac Lab's `gym.make` path. Look at
  `RecordVideo` / `VideoRecorder` callsites.
- **GPU bandwidth measurement** — `nvidia-smi dmon -s u` running
  alongside the training process gives a rough GPU utilization
  signal. Not phase-attributed but useful as a sanity check that
  the GPU is the bottleneck (high util) vs. the CPU side (high
  CPU, low GPU util means something is starving the GPU).

## Risks / open questions

- **PPO update phase frequency.** The update fires every
  `num_steps_per_env` collection ticks. The profiler must accept
  that some phases sample rarely; report n alongside p50 so the
  consumer can judge confidence.
- **rsl_rl internals.** Some bookkeeping happens inside rsl_rl,
  not Isaac Lab. Instrumenting it may require a thin patch around
  `OnPolicyRunner.learn` rather than purely external timing.
  Acceptable; the bridge profiler already monkey-patches `sim.step`
  / `sim.render`, same pattern.
- **Measurement variance.** A single 200-iteration run can be
  noisy. The rolling p50 / p99 window mitigates within-run noise;
  cross-run variance (different seeds, different scene seeds)
  is out of scope — one representative run is enough to identify
  a > 10% phase.

## Out of scope

- **Any optimization implementation.** This brief is *pure
  measurement*. If `sim.render` turns out to be the bottleneck,
  the brief filed against it is the one that touches the render
  path; this brief just files that follow-up.
- **Bridge-mode work.** The companion brief
  [`bridge-throughput-toward-25hz`](../sim-performance/bridge-throughput-toward-25hz.md)
  covers bridge perf. The bridge `--profile` harness already
  exists; reuse it there.
- **Cross-env comparisons.** "Procroom is slower than Plane" is
  not a question this brief answers. We measure procroom because
  that's the env the operator is running; comparing across envs
  is a separate exercise.
- **rsl_rl algorithm changes.** No swapping PPO for SAC, no
  reward-shape edits, no `num_envs` tuning. Throughput
  measurement and follow-up filing only.
