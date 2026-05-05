# Train a small learned mid-mission validator

**Type:** task / new feature
**Owner:** DGX agent (training pipeline + dataset assembly are
DGX-side; deployment lives behind the same env-var the CLIP path
uses, no Jetson code changes beyond a model swap)
**Priority:** P2 (gated on
[`clip-mid-mission-validator-evaluation.md`](clip-mid-mission-validator-evaluation.md)
shipping `fail` or `in-between` after the §3.1 cheap-fix re-run)
**Estimate:** L (~multi-day; new training script, harness label
augmentation, model export, eval reuse)
**Branch:** task/learned-mid-mission-validator

## Story

As an **operator who has measured that the CLIP transit-monitor
path doesn't reach the ≥ 0.7 TPR / ≤ 0.1 FPR bar even after the
cheap-fix pass**, I want **a small frozen-DINOv2 + temporal-head
validator trained on the harness's `(frame_window, mission_text,
on_course?)` tuples**, so that **mid-mission validation gets the
purpose-built signal the open-vocab CLIP recipe could not provide,
without paying the small-VLA training and co-tenancy cost**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
- [`context/conventions.md`](../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md)

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md),
§3.2 (small learned validator) and §4 (recommendation).

Prerequisite brief (must ship first):
[`clip-mid-mission-validator-evaluation.md`](clip-mid-mission-validator-evaluation.md).

## Context

### Why this brief exists

The architecture doc's §3.2 candidate: a frozen-feature backbone
(DINOv2-S/14 or the existing OpenCLIP visual tower used as a frozen
feature extractor) plus a small temporal head (Conv1D /
Transformer over 5–10 captures) emitting a scalar
"on-course / off-course" decision and a per-frame visualizable
attention.

The §3.2 case is exactly: *the runtime wants room-classification
behavior; CLIP's open-vocab loss trains the wrong contrast; the
right loss on the right architecture should win.* The CLIP-eval
brief tests whether CLIP-the-cheap-option wins anyway. This brief
fires only if it doesn't.

### What it builds

- **Dataset assembly** at
  `source/strafer_lab/strafer_lab/tools/build_validator_dataset.py`
  that walks `data/sim_in_the_loop/<scene>/episode_NNNN/`,
  consumes the `root_cause`-augmented `frames.jsonl` produced by
  the prerequisite brief's `--root-cause-pass`, and emits
  `(frame_window, mission_text, on_course)` tuples to a parquet
  dataset. Same-mission frames within `R_window=5` consecutive
  captures form a window. Negative examples come from
  `wrong_locale` legs and from deliberately-perturbed missions
  (the harness gets a new `--inject-bad-grounding` flag to
  produce hard negatives).
- **Training script** at
  `source/strafer_lab/scripts/train_validator.py`. Backbone =
  frozen DINOv2-S/14 (cached HuggingFace `facebook/dinov2-small`).
  Temporal head = 3-layer Transformer over patch features, output
  is a 2-class logit + a per-frame attention map for diagnostics.
  AdamW, cross-entropy on the binary label, with class re-weighting
  to compensate for the harness's mostly-succeeded mission mix.
- **Export** to ONNX (visual tower + head) under
  `~/.strafer/models/validator/` matching the path convention the
  CLIP encoder uses.
- **Runtime swap** behind a `STRAFER_VALIDATOR_BACKEND` env var:
  `clip` (current path), `learned` (new path). The
  `BackgroundMapper`'s capture loop calls a `Validator.check()`
  protocol instead of `TransitMonitor.check()`; both implement
  the same return shape (`{"on_track", "abort", "reason"}`).
- **Measurement reuse.** The eval script
  `Scripts/eval_transit_monitor.py` from the prerequisite brief
  is parameterized by validator backend, so the same
  pre-registered metrics (TPR, FPR, time-to-decision) are
  recomputed for the learned validator on the same episode set.

### Dataset budget

The current single-room scene set
(`scene_fast_singleroom_000_seed0`,
`scene_high_quality_dgx_000_seed0`) plus 1–2 additional Infinigen
seeds gives ~30 missions per scene at the harness's defaults. With
`R_window=5` and one window per leg-second, that's ~5–10k labeled
windows per scene — enough for a frozen-backbone fine-tune with
class re-weighting but tight for a from-scratch backbone (which
this brief avoids). The hard-negative injection flag on the
harness multiplies effective negatives without re-running the full
sweep.

### Compute envelope

- **Training (DGX).** Frozen DINOv2-S/14 encodes once per frame;
  cached features can fit in DGX RAM for a 5k-window dataset.
  Head training is single-GPU minutes-to-an-hour.
- **Inference (Jetson Orin Nano).** Frozen DINOv2-S/14 ~60–80 ms
  at FP16. Temporal head ~10–20 ms. Comfortably under the
  `BackgroundMapper` poll interval. VRAM impact ~120–180 MB,
  acceptable alongside Nav2 / RTAB / executor.

## Acceptance criteria

- [ ] **Dataset builder.**
      `source/strafer_lab/strafer_lab/tools/build_validator_dataset.py`
      consumes the `root_cause`-augmented harness output and emits
      a parquet dataset with documented schema. Unit tests under
      `source/strafer_autonomy/tests/` (mirroring how
      `dataset_export` is tested today) cover the window-construction
      + class-balancing logic.
- [ ] **Hard-negative injection.**
      `source/strafer_lab/scripts/run_sim_in_the_loop.py` gains a
      `--inject-bad-grounding` flag that perturbs goal positions
      by a sampled offset to produce labelled `wrong_locale`
      missions deliberately. Behavior documented in
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)'s
      Stage 5 section.
- [ ] **Training script.**
      `source/strafer_lab/scripts/train_validator.py` trains the
      frozen-DINOv2 + temporal-head model, exports a checkpoint +
      ONNX under `~/.strafer/models/validator/`, and writes a
      JSON metadata sidecar (variant, backbone, head config,
      dataset commit, repo SHA). MLflow tracking optional via the
      same `--mlflow-experiment` flag pattern that
      [`finetune_clip.py`](../../../source/strafer_lab/scripts/finetune_clip.py)
      uses.
- [ ] **Runtime swap.**
      `BackgroundMapper` accepts a `Validator` protocol; both
      `TransitMonitor` (CLIP) and a new `LearnedValidator`
      implement it. The executor's main.py reads
      `STRAFER_VALIDATOR_BACKEND` and constructs the right one.
      Default stays `clip` until this brief's eval declares the
      swap.
- [ ] **Eval reuse.** `Scripts/eval_transit_monitor.py` from the
      prerequisite brief is parameterized by `--backend
      {clip,learned}` so the same metrics are reported on both.
- [ ] **Decision report.** Append a §4.4 to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md)
      with the learned validator's TPR/FPR/time-to-decision on
      the same episode set, alongside the CLIP path's numbers
      from §4.3. If TPR ≥ 0.85 at FPR ≤ 0.1, flip the runtime
      default to `learned` in the same PR; otherwise leave the
      default at `clip` and record the gap.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports
      (the existing CLIP backend must keep working with
      `STRAFER_VALIDATOR_BACKEND=clip`; smoke this in the PR).

## Investigation pointers

- DINOv2 caching path: `~/.cache/huggingface/hub/models--facebook--dinov2-small/`.
  Avoid hard-coded HF tokens; the model is public.
- Per-mission temporal grouping is by `mission_id` in
  `frames.jsonl` — see the schema in
  [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
  Stage 5.
- ONNX export pattern that exists in the codebase:
  [`finetune_clip.export_towers_to_onnx`](../../../source/strafer_lab/scripts/finetune_clip.py#L262).
  Mirror the structure (separate visual + head exports + a
  metadata JSON) for parity with the CLIP path.
- Sim-to-real notes for backbone choice:
  [`docs/SIM_TO_REAL_TUNING_GUIDE.md`](../../SIM_TO_REAL_TUNING_GUIDE.md).
- The validator runs alongside the existing TRT-EP CLIP path on
  the Jetson if the operator wants to A/B them — make sure the
  ONNX runtime providers don't fight (the current path uses the
  ORT default provider list; document if the validator needs TRT
  vs. CUDA EP explicitly).

## Out of scope

- **Replacing CLIP for `verify_arrival`.** This brief only ships
  the *transit monitoring* validator. `verify_arrival` keeps
  using the CLIP-top-k path until a separate brief evaluates
  whether the learned validator should also handle arrival
  verification.
- **Training a small VLA (§3.3).** Stays deferred per
  [`MISSION_VALIDATION_ARCHITECTURE.md` §4.2](../../MISSION_VALIDATION_ARCHITECTURE.md#42-whats-explicitly-not-recommended).
- **Real-robot data collection.** Sim-side training only.
  Real-robot evaluation is a future brief.
- **Replacing the frozen backbone.** DINOv2-S/14 is the chosen
  backbone for this brief. Backbone-shopping is out of scope —
  if the head reaches the bar, the backbone is fine; if it
  doesn't, the next brief decides whether to climb to DINOv2-B/14
  or move to a different family.
