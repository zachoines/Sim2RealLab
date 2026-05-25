# Use NVIDIA Cosmos for video-domain replay-with-perturbation

**Type:** new feature / investigation (filed-on-trigger)
**Owner:** DGX agent
**Priority:** P3 (filed-on-trigger; pick up only after teleop has
shipped + produced ≥ 500 trajectories AND Cosmos Predict / Cosmos
Transfer access is available on the DGX)
**Estimate:** L (~week+; Cosmos integration + per-trajectory
conditioning prep + augmented-corpus emission + quality
benchmark)
**Branch:** task/harness-cosmos-replay-perturbation

## Story

As an **operator who has shipped a teleop corpus and wants to
10× the effective training-corpus volume without burning more
operator hours**, I want **a video-domain replay-with-perturbation
pass that re-renders each captured trajectory under different
lighting / texture / weather seeds using NVIDIA Cosmos as the
world model**, so that **the corpus multiplies in scale without
re-running the simulator, with sim-to-real-relevant
augmentations (lighting / texture / weather variations) that the
in-sim domain-randomization pipeline cannot produce as
realistically as a frontier world-model can**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)

Parent design context:
[`MISSION_VALIDATION_ARCHITECTURE.md` §3.6.a](../../../MISSION_VALIDATION_ARCHITECTURE.md#36a-teleop-demos-primary-canonical) —
teleop budgets explicitly include a ~5× replay-perturbation
multiplier; this brief is the implementation.

Sibling briefs:
- [`harness-architecture`](../../active/harness/harness-architecture.md) —
  defines the LeRobot v3 canonical schema this brief emits
  augmented variants against, the
  [Driver: teleop](../../active/harness/harness-architecture.md#driver-teleop)
  that records the gamepad event stream this brief needs for
  replay, and the
  [Scripted × captioner](../../active/harness/harness-architecture.md#scripted--captioner-trajectory-first-path)
  complementary multiplier this brief stacks with. Subsumes the
  retired
  [`teleop-driver`](../../completed/teleop-driver.md),
  [`behavior-cloning-data-expansion`](../../completed/behavior-cloning-data-expansion.md),
  [`trajectory-first-captioning`](../../completed/trajectory-first-captioning.md),
  and
  [`output-format-alignment`](../../completed/output-format-alignment.md)
  briefs.

## Trigger condition — when to pick this brief up

Pick up only when **all** of:

1. [`harness-architecture`](../../active/harness/harness-architecture.md)
   Tier 1 (teleop) has shipped and the teleop corpus has ≥ 500
   trajectories. Smaller corpora aren't worth augmenting yet.
2. **Cosmos Predict 2.5 (or successor) is accessible on the DGX.**
   Either via Cosmos's HF release, NVIDIA NIM, or a local
   checkpoint. As of audit (2026-05), Cosmos Predict 2.5 /
   Transfer 2.5 / Reason 2 are NVIDIA-public; verify access at
   pickup time.
3. **DGX VRAM headroom exists** for Cosmos's 2B–14B-parameter
   inference passes on top of whatever else is running. Cosmos
   Transfer 2.5 is "3.5× smaller than its predecessor" per
   NVIDIA's release notes; even so, plan for ≥ 30 GB of VRAM
   during inference.

Until all three fire, this brief stays parked.

## Context

### Why Cosmos vs. in-sim domain randomization

The Strafer training pipeline already has in-sim domain
randomization via `EventTerm` in
`strafer_env_cfg.py` (lighting, friction, mass, motor strength,
D555 mount). That's fine for the *training-time* DR loop. But
for *post-collection corpus multiplication*, in-sim DR has
limits:

- **Lighting/texture diversity** in Isaac Sim is bounded by what
  Replicator + dome lights can express. Cosmos was trained on
  200M curated video clips and can re-render the same trajectory
  under daylight / nighttime / overcast / rain / dawn / dusk
  with vastly higher photometric realism than a Replicator
  randomization sweep.
- **Texture diversity** in Isaac Sim requires a texture library
  the artist team maintains. Cosmos has a learned distribution
  of textures from its training corpus.
- **Out-of-distribution sim-to-real gap** is reduced more by
  realistic photometric augmentation than by random
  permutations of an in-sim palette. NVIDIA's own messaging
  positions Cosmos-augmented data as the bridge between sim
  and real for physical AI.

The trade: Cosmos hallucinates. Replay-with-Cosmos can produce
frames that diverge from the underlying scene geometry (a
hallucinated chair behind a real wall). The brief must validate
against this — accepted Cosmos frames must remain trajectory-
consistent.

### Pipeline sketch

```
Captured trajectory (RGB frames + actions + mission_text)
    ↓
For each perturbation seed S ∈ {daylight, dusk, dawn, ...}:
    Cosmos Transfer 2.5:
      Input: original RGB sequence + seed S (text/condition)
      Output: re-rendered RGB sequence under condition S
    ↓
Validation:
  - Trajectory consistency: depth (still recorded from original
    sim run) is reused; the Cosmos output must not contradict
    the recorded depth.
  - Action consistency: actions are reused as-is; if Cosmos
    output suggests a different action (model-detected via a
    consistency check), reject the augmentation.
  - Caption consistency: original mission_text + paraphrases
    are reused; if Cosmos output makes the target invisible,
    the variant is rejected.
    ↓
Augmented variant emitted with metadata:
  source: "cosmos-replay-perturbation"
  cosmos_model: "cosmos-transfer-2.5"
  cosmos_seed: S
  cosmos_consistency_validated: true
```

Each accepted trajectory produces a few variants per original
(typical: 3–5 lighting variants × 2–3 texture variants = 6–15
augmented trajectories per original). With a teleop corpus of
500 trajectories, that's ~5k augmented variants.

### What this brief does NOT replace

- **In-sim DR during training.** Continue. Cosmos is a corpus
  multiplier, not a training-time augmentation; the cost
  profile is wrong for online use.
- **Trajectory-first captioning.** A complementary regime.
  Trajectory-first creates more trajectories; this brief creates
  more *variants* per trajectory. Stack them.
- **Real-robot data collection.** Sim-side only. Real-robot
  augmentation via Cosmos is a future brief.

### Risks / unknowns

- **Cosmos hallucination rate** on the strafer-style indoor
  scene distribution is untested. The Cosmos release training
  corpus skews automotive / outdoor; indoor + wheeled-mecanum
  may be out-of-distribution and produce visible artifacts.
  Plan a hold-out hand-inspection pass.
- **Latency.** Cosmos inference is not real-time. The brief
  must commit to an offline-batch architecture; no online
  use during data collection.
- **Cost.** Cosmos at scale may require renting GPU cloud;
  the DGX alone may be too small. Brief PR must report
  measured throughput and decide.

## Acceptance criteria (preliminary; expand at pickup time)

- [ ] **Cosmos integration script.**
      `source/strafer_lab/strafer_lab/tools/cosmos_replay.py`
      consumes a teleop / oracle / trajectory-first corpus
      directory and emits augmented variants under the same
      canonical schema. Per-variant metadata includes
      `cosmos_model` + `cosmos_seed` + consistency-validation
      results.
- [ ] **Consistency validation.** Trajectory-depth consistency
      + caption-target-visibility consistency checks run on
      every Cosmos output; failed variants are rejected with a
      logged reason.
- [ ] **Quality benchmark.** Hand-inspect a stratified sample
      of ≥ 50 augmented variants (5 lighting × 5 texture × 2
      trajectory length bins). Report hallucination rate per
      seed. Iterate on Cosmos prompt / conditioning until rate
      is < 10% per seed. Document the final conditioning
      pattern.
- [ ] **Throughput report.** Wall-time per augmented
      trajectory; total wall-time for the existing teleop
      corpus's worth of augmentations; VRAM peak; concurrent-
      DGX-load impact.
- [ ] **Schema parity.** Augmented variants are consumable by
      the same HF `LeRobotDataset` loader that consumes the
      original corpus — Cosmos outputs land as additional MP4s
      under `videos/.../observation.images.perception_cosmos_{seed}.mp4`
      referencing the same parquet rows with new `episodes.jsonl`
      entries (`generator_metadata.cosmos_seed` set), per
      [`harness-architecture`](../../active/harness/harness-architecture.md)'s
      output format.
- [ ] **Doc surface.**
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md)
      gains a Stage 5d — Cosmos replay-with-perturbation
      section.
      [`source/strafer_lab/README.md`](../../../../source/strafer_lab/README.md)
      "Scripts and tools inventory" gains `cosmos_replay.py`.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- NVIDIA Cosmos product page:
  [https://www.nvidia.com/en-us/ai/cosmos/](https://www.nvidia.com/en-us/ai/cosmos/)
- Cosmos Predict 2.5 / Transfer 2.5 / Reason 2 release notes
  (NVIDIA newsroom, 2026 announcements).
- Cosmos arXiv: [arXiv:2511.00062](https://arxiv.org/abs/2511.00062)
  ("World Simulation with Video Foundation Models for Physical
  AI") — methodology baseline.
- "Curating Synthetic Datasets to Train Physical AI Models with
  NVIDIA Cosmos Reason" (NVIDIA developer blog) — Cosmos Reason
  is the curation companion, useful for filtering
  hallucinated outputs.
- For the conditioning prompt design, mirror the
  trajectory-first captioning brief's instructive-prompt
  approach — Cosmos Predict takes text + image / video
  conditioning per its API docs.

## Out of scope

- **Online (training-time) Cosmos augmentation.** Cost profile
  is wrong; in-sim DR + offline Cosmos batch is the right
  split.
- **Real-robot Cosmos augmentation.** Sim-side first.
- **Cosmos training / fine-tuning.** This brief uses
  Cosmos out-of-the-box. Fine-tuning Cosmos on the strafer
  corpus is a future brief if the hallucination rate
  underdelivers.
- **Cross-host Cosmos service.** Cosmos runs on the DGX
  locally; if VRAM doesn't fit, the brief picks "rent cloud
  GPU" or "drop a Cosmos variant," not "build a remote
  Cosmos microservice."
