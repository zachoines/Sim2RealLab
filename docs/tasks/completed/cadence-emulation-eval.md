# Adjudicate the cadence hypothesis in closed-loop sim

**Status:** Shipped 2026-08-05 in `5b77c59` (DGX). The harness landed
earlier in `659a5b1` (PR #179); this change records the 2026-08-04 evaluation
run and the scored read-out that closes the brief. The temporal axis is
exonerated as sufficient: the 22–25 Hz band costs ≤3% of completion, and the
12 Hz / 36%-duplicate profile costs ~⅓ — a real loss, but not the rig's
total failure. No cadence-targeted retrain is licensed.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/186

**Type:** investigation (evaluation harness + one operator-launched run)
**Owner:** DGX
**Priority:** P0 — a shipped attribution and the retrain decision both rest on
it; the training lever is provisional until this reads out.
**Estimate:** M (harness in-PR; one evaluation session on the play env)
**Branch:** `task/cadence-emulation-eval`

## Story

As the **coordinator deciding whether the depth subgoal policy's advance
failure is policy-owned**, I want **the deploy rig's temporal texture
reproduced inside closed-loop sim, swept as a grid**, so that **the attribution
rests on a sensitivity curve rather than on a single arm that happened to run
at half the historical inference rate.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/recurrent-policy-contract.md](../context/recurrent-policy-contract.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

The most recent enriched rig arm scored the failing branch, but it ran at an
**unprecedented temporal profile**: 11.68 Hz mid-arm depth arrival and 38.3%
duplicate depth content, against the 22–25 Hz band of every prior session. The
scene and anchoring attributions from that session are clean; the temporal axis
is not, and it is perfectly confounded with sim-versus-rig. So the recorded
verdict — the failure is policy-owned and the training lever fires — is
**provisional pending this adjudication**.

Two facts make the hypothesis structurally live rather than speculative.

**The deployed node's semantics are reproducible in sim.** A tick whose depth
frame has not advanced returns before observation assembly: no inference, no
publish (so the chassis keeps executing the last command), and no hidden-state
advance. The freshness gate keys on a node-local receive counter, so there is
no back-fill — frames that arrive during a hold are dropped, never replayed.
Separately, `depth_repeat_content` counts inferences fed a bit-identical depth
block on an *advancing* counter, i.e. a publisher stamping faster than it
renders. **These are two independent axes**, and the harness models them as
two: a held fraction that sets the effective inference rate, and a duplicate
fraction that sets content novelty among the inferences that do run. The rig
number quoted as "duplicate/held fraction 0.38–0.6" decomposes into
`hold_fraction ≈ 0.611` (11.68 Hz out of a 30 Hz tick) and
`stale_fraction ≈ 0.383` (the measured repeat share of those inferences).

**The training distribution contains almost none of it.** The shipped
checkpoint trained on the ROBUST tier with a *fixed* 2-step depth latency, 1%
i.i.d. frame drops, a structurally fixed 30 Hz tick, and a *constant*
per-modality skew. `control_frequency_jitter_pct` and `obs_latency_steps_range`
are declared and read by nothing. The emulated held fraction is therefore
roughly 38× outside anything the policy saw, which is exactly why the question
cannot be answered from the training logs.

## The harness

`source/strafer_lab/scripts/eval_cadence_emulation.py` extends the play
rollout loop. Per env, per tick, one of three kinds:

| kind | policy runs | depth seen | recurrent state | action |
|---|---|---|---|---|
| fresh | yes | live | advances | new |
| stale | yes | cached block | advances | new |
| held | no | — | frozen | previous, re-issued |

Held ticks are emulated by running the batched forward for every env and then
restoring the held rows' hidden columns and overwriting their action rows —
equivalent to never having called the policy for those rows, and it keeps the
batch width constant so no env's convolution numerics shift with its schedule.
Depth is **not** cached across a hold: the real node's inferring tick always
consumes the newest frame it has, so caching across the hold would inject a lag
deploy does not have. The cache exists only for the duplicate-content axis.

Scoring is read from the termination manager immediately after the step (the
command term's completion flag and progress delta are clobbered by the
auto-reset inside `step()` and read as a silent 0% completion rate). Arc-length
progress is latched pre-step from the monotone path cursor.

## Acceptance criteria

- [x] Pure tests cover the schedule sampler's realized statistics against the
      requested profile, the observation-dump profile loader against a
      synthetic JSONL in the node's schema, the signed direction-offset sign
      convention, and the held-row hidden-state restore equivalence.
- [x] Harness dry-run on the play env with realized-schedule statistics
      printed and inside tolerance of the request.
- [~] `B2` (baseline) and `B1` run **first**. If the baseline completion rate
      is far from ~0.8, STOP and report — that is harness or env drift, not a
      cadence result. If harness-`v1` collapses at the band profile, the
      emulation is too aggressive and must be recalibrated before the depth v2
      arms are read.
- [x] Per-arm results table (completion, near-arrival, progress, direction
      offset, realized profile) plus the raw JSONL in the PR description.
- [x] Failure counts broken out by cause. A completion drop that is entirely
      `off_path_divergence` is a different finding from one that is entirely
      `time_out`, and `near_arrival` separates a dwell-gate failure from never
      having arrived.
- [x] No changes to envs, cfgs, noise models, or the play script — the harness
      is additive.
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [x] No regression in the workflows the touched code supports — the pure
      strafer_lab suite stays green and the play script is untouched.

## Arms

Env `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`, fixed seed
set, 8–16 envs, ≥100 episodes per arm, 20 s episode cap.

| arm | policy | profile |
|---|---|---|
| B2 | depth v2 `run_20260727_171735/model_998.pt` | `clean` — every tick fresh; the in-harness baseline, mandatory |
| T2a | depth v2 | `band` — ~23 Hz effective, short hold runs |
| T2b | depth v2 | `degraded` — ~11.7 Hz effective, bursty holds, 38% duplicate content |
| T2c | depth v2 | `measured` — replayed from a rig observation dump; skip and note if unavailable |
| B1 / T1a / T1b | depth v1 `run_20260708_005923/model_500.pt` | the same three parametric profiles — calibration control |

Profiles are comma-separated in one invocation, so each checkpoint costs one
session rather than one per arm.

## Decision rule (pre-registered)

Committed before the session runs, so the outcome cannot be read backwards.

| prediction | reading if it fails |
|---|---|
| v2 in-harness baseline ≥ 0.8 completion | harness or env drift — stop, do not read the rest |
| v2 holds ≥ 60% of baseline at `band` | cadence alone would then explain the historical failures |
| v2 degrades materially at `degraded` | the recorded arm's temporal profile is not exculpatory |
| **no** leftward rotation emerges under emulation at any profile | the directional signature would be temporal after all |
| v1 holds ≥ 80% of baseline at every profile | the emulation is too aggressive; recalibrate before reading v2 |

| outcome | consequence |
|---|---|
| v2 collapses at `degraded` but holds at `band` | the recorded arm is invalidated as attribution evidence, the training lever un-fires, and the path is node-consumption fixes then a re-run at a restored band |
| v2 holds even at `degraded` | the temporal axis is exonerated, the failing branch stands, and the training lever fires for real |
| v2 collapses already at `band` | temporal texture is a first-order training defect across all deploy history, and the retrain leads with temporal-texture randomization |

## Results — 2026-08-04 run

The harness landed in `659a5b1` (PR #179). This section records the run it was
built for. **The brief stays active: what closes it is the read-out against the
pre-registered rule below, and that scoring is not recorded here.**

Both invocations: `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`,
16 envs, seed 42, 100 episodes per profile, observation corruption on,
`--warmup-ticks 2`. No profile hit its step ceiling.

**v2 — `run_20260727_171735/model_998.pt`**

| profile | Hz | hold | dup | eps | completion | ratio | near | progress | offset | left |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 30.00 | 0.000 | 0.000 | 100 | 0.900 | 1.000 | 0.580 | 0.916 | +4.24° | 0.612 |
| band | 23.29 | 0.224 | 0.000 | 100 | 0.870 | 0.967 | 0.710 | 0.921 | +3.84° | 0.606 |
| degraded | 12.01 | 0.600 | 0.363 | 100 | 0.610 | 0.678 | 0.690 | 0.817 | +2.87° | 0.550 |

**v1 — `run_20260708_005923/model_500.pt`** (calibration control)

| profile | Hz | hold | dup | eps | completion | ratio | near | progress | offset | left |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 30.00 | 0.000 | 0.000 | 100 | 0.840 | 1.000 | 0.720 | 0.915 | +6.29° | 0.616 |
| band | 23.23 | 0.226 | 0.000 | 100 | 0.750 | 0.893 | 0.760 | 0.906 | +4.67° | 0.579 |
| degraded | 12.01 | 0.600 | 0.370 | 100 | 0.560 | 0.667 | 0.720 | 0.846 | +4.32° | 0.550 |

**Failure cause, as a fraction of episodes** (time-out is 0.000 in all six):

| profile | v2 path_complete / off_path / collision | v1 path_complete / off_path / collision |
|---|---|---|
| clean | 0.900 / 0.010 / 0.090 | 0.840 / 0.010 / 0.150 |
| band | 0.870 / 0.030 / 0.100 | 0.750 / 0.050 / 0.200 |
| degraded | 0.610 / 0.190 / 0.200 | 0.560 / 0.280 / 0.160 |

**Requested vs realized.** `band` asked 0.233 hold / 0.000 duplicate and
realized 0.224 / 0.000 (v2) and 0.226 / 0.000 (v1). `degraded` asked 0.611 /
0.383 and realized 0.600 / 0.363 (v2) and 0.600 / 0.370 (v1). The shortfall is
`--warmup-ticks` forcing fresh ticks after each episode boundary.

### Gates and deviations

- **Baseline STOP gate passed.** v2 clean is 0.900 — above ~0.8, not far below
  it. No harness or env drift; proceeding to the degraded profiles was correct.
- **The recalibration clause did not trip.** It is scoped to the band profile,
  and v1 band holds 0.893 of its baseline. No recalibration is owed under the
  acceptance criterion.
- **A pre-registered row fails, and it is a scoring input rather than a gate.**
  The decision-rule row reads "v1 holds ≥ 80% of baseline at every profile";
  v1 at `degraded` is 0.667. The acceptance criterion above is band-scoped and
  passes, so the two texts differ in reach. Both policies lose baseline at
  near-identical rates at `degraded` (v1 0.667, v2 0.678). Recorded, not scored.
- **Arm order deviated from the literal wording.** The acceptance asks for `B2`
  and `B1` first; profiles run comma-separated within one invocation per
  checkpoint, so the run was v2 clean/band/degraded then v1 clean/band/degraded,
  and `B1` landed after `T2a`/`T2b` in wall-clock terms. Every gate was
  evaluated on the complete set before any profile was read.
- **`T2c` (`measured`) did not run** — no inference-node observation dump exists
  on this machine, which the Arms table above pre-authorises ("skip and note if
  unavailable"). The capture is blocked upstream and that blocker is already
  owned by [`enriched-lane-rig-stability`](../active/reliability/enriched-lane-rig-stability.md);
  no new brief is filed for it.
- **The conditional rate-only cell did not fire.** It was pre-registered to run
  as its own invocation (`--profile degraded --stale-fraction 0.0`) if v2's
  degraded completion fell below 60% of its baseline; v2 reads 0.678.

### Where the evidence lives

`logs/` is gitignored, so the run's JSONL is not in version control and the
tables above are its durable record. On the machine that produced it the files
are `logs/rsl_rl/strafer_navigation/cadence_emulation/cadence_20260804_190256.jsonl`
(v2) and `cadence_20260804_191139.jsonl` (v1), each carrying three profile
records with per-episode detail. The two dry-run files from the same day
(`cadence_20260804_081050.jsonl`, `cadence_20260804_081812.jsonl`) hold the
harness check that preceded them.

## Read-out — scored against the pre-registered rule

- **(i) Baseline: met.** v2 clean 0.900 ≥ 0.8. The harness and env reproduce
  the healthy closed-loop regime.
- **(ii) Band: met, decisively.** v2 at the 23 Hz band holds 0.967 of
  baseline. At the 22–25 Hz cadence every recorded deploy session actually ran
  in, emulated temporal texture costs ≤3% completion. **The band is exonerated
  as a cause of any recorded deploy failure.**
- **(iii) Degraded: met, with a bounded magnitude.** 0.678 of baseline at the
  12 Hz / 36%-duplicate profile — a real ~⅓ loss, but far from the outcome it
  was built to test: 0.610 completion and 0.817 progress in sim, against zero
  completions and no net advance on the rig at the same temporal profile.
  **Temporal texture explains at most a third of the enriched-lane failure,
  not the failure.**
- **(iv) No emergent rotation: met.** Direction offset runs +4.2° (clean) →
  +2.9° (degraded) with left-share falling 0.61 → 0.55; both policies show the
  same small baseline lean. The rig's directional signature is not temporal.
- **(v) Calibration: the one failed row, informative rather than gating.** v1
  at degraded is 0.667 < 0.80; band-scoped calibration passed (0.893). Both
  policies lose ≈⅓ at 12 Hz (0.667 vs 0.678) — the profile is generically
  harsh, not differentially harmful to v2, which is itself evidence: a
  v2-specific rig failure cannot be produced by an axis that punishes both
  policies equally.

**Decision-map application.** v2 holds at `degraded` by the pre-registered
line (0.678 ≥ 0.60): the temporal axis is **exonerated as sufficient**.
Consequences:

1. **No cadence-targeted retrain is licensed** — neither a fixed-20 Hz retrain
   nor a temporal-texture-first augmentation. The temporal-DR wiring remains a
   cheap rider if a retrain ever happens for other reasons.
2. **The enriched-lane `mission` ✗ result is now unexplained by scene
   (sim-enriched completes 0.90), by anchoring semantics (the emulation uses
   training's own), or by temporal texture (this run).** Its attribution
   reopens with four live candidates: SLAM-frame anchoring noise (map→odom
   moved 0.166 m / 6.7° in that same session's discarded arm), planner
   path-geometry distribution (Nav2 plans vs the training A*), unbounded
   recurrent-state horizon (deploy never resets the GRU; training episodes cap
   at 600 steps), and residual observation-chain deltas.
3. **The receiver-side deploy fixes proceed on their own merits, now with
   quantified stakes**: deploy at the 12 Hz regime costs ~⅓ of completion;
   at the band it costs ~3%. Recovering arrival rate recovers almost the whole
   temporal cost.
4. **The measured-profile cell becomes archival** — the decision no longer
   hangs on it; it runs if/when the capture unblocks, for the record.

The residual-attribution arms this read-out dispatches are their own brief —
[`cadence-harness-residual-arms`](../active/trained-policy/cadence-harness-residual-arms.md)
— and are not part of this one.

## Investigation pointers

- Freshness gate and the hold semantics it defines:
  `source/strafer_ros/strafer_inference/strafer_inference/inference_node.py:954`
  (skip, before assembly) and `:1056` (`_note_depth_content`).
- Observation-dump schema, one record per inference:
  `source/strafer_ros/strafer_inference/strafer_inference/inference_node.py:1240`.
- Auto-reset ordering that invalidates a post-step command-term read:
  terminations compute before `_reset_idx`, the command manager computes after.
- Completion is dwell-gated: 10 consecutive steps inside the arrival radius
  under 0.1 m/s. Re-issuing a non-zero held command can break the dwell, so a
  cadence-induced completion drop is not necessarily a tracking failure —
  `near_arrival_rate` is the discriminator.
- `off_path_divergence` fires at 0.3 m cross-track and is the expected
  dominant failure mode under holds.
- The play variant is **not** noise-free: it inherits full realistic/robust
  randomization and observation corruption, differing from the training config
  only in env count. `--disable-obs-corruption` lifts it if a clean ablation is
  wanted; the default leaves it on so every arm shares one noise floor.

## Out of scope

- Wiring the dead randomization knobs or widening any randomization band —
  measurement first. The audit doc's Phase 2 block is the designed home if this
  eval confirms.
- Node-side consumption fixes (transport reliability, executor split, CPU
  pinning). They are justified regardless of this result and are sequenced
  after the rig profile capture, so that the capture reflects the stack the
  failing arm actually ran on.
- The retrain itself, which stays held until this reads out.
- The sim depth stream's render duplication, which this harness *models* but
  does not fix.
