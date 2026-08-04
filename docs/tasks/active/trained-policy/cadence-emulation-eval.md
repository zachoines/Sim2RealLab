# Adjudicate the cadence hypothesis in closed-loop sim

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

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

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

- [ ] Pure tests cover the schedule sampler's realized statistics against the
      requested profile, the observation-dump profile loader against a
      synthetic JSONL in the node's schema, the signed direction-offset sign
      convention, and the held-row hidden-state restore equivalence.
- [ ] Harness dry-run on the play env with realized-schedule statistics
      printed and inside tolerance of the request.
- [ ] `B2` (baseline) and `B1` run **first**. If the baseline completion rate
      is far from ~0.8, STOP and report — that is harness or env drift, not a
      cadence result. If harness-`v1` collapses at the band profile, the
      emulation is too aggressive and must be recalibrated before the depth v2
      arms are read.
- [ ] Per-arm results table (completion, near-arrival, progress, direction
      offset, realized profile) plus the raw JSONL in the PR description.
- [ ] Failure counts broken out by cause. A completion drop that is entirely
      `off_path_divergence` is a different finding from one that is entirely
      `time_out`, and `near_arrival` separates a dwell-gate failure from never
      having arrived.
- [ ] No changes to envs, cfgs, noise models, or the play script — the harness
      is additive.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports — the pure
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
