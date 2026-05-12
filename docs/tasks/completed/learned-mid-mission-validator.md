# Train a small learned mid-mission validator

**Status:** Retired 2026-05-09. Never picked up. Reason: the
proposed architecture (frozen DINOv2-S/14 + frozen sentence
encoder + ~5 M trainable fusion head) is structurally
underpowered against the CLIP cascade-arbiter alternative the
project is shipping via
[`clip-mid-mission-validator-evaluation`](../active/clip-validation/validator-evaluation.md).
The cascade includes Qwen2.5-VL-3B as the arbiter plus the
planner LLM as judge — ~75× more pretrained capacity and 3
orders of magnitude more pretraining data than a small frozen-head
fine-tune. Empirical literature (Cao et al. 2023, OpenVLA,
Speaker-Follower analysis) consistently shows small-from-scratch
heads losing to large pretrained zero-shot baselines in this
regime. Retiring the brief reduces complexity without losing
capability — the alternative escalation path is now the
end-to-end VLA research arm
([`strafer-vla-v2-architecture`](../parked/experimental/vla-v2-architecture.md)),
which has billions of pretrained parameters as starting capital
and is already filed for sim-side exploration.
**Replaced-by:** [`clip-cotrained-retrieval-augmented`](../parked/clip-validation/cotrained-retrieval-augmented.md)
captures the directions for *improving* the CLIP cascade
itself (co-trained speaker integration + retrieval-augmented
inference) which dominate the small-head-validator approach
on the same evaluation metrics.

**Type:** task / new feature
**Owner:** DGX agent (training pipeline + dataset assembly are
DGX-side; deployment lives behind the same env-var the CLIP path
uses, no Jetson code changes beyond a model swap)
**Priority:** P2 (gated on
[`clip-mid-mission-validator-evaluation.md`](../active/clip-validation/validator-evaluation.md)
shipping `fail` or `in-between` after the §3.1 cheap-fix re-run)
**Estimate:** L (~multi-day; new training script, harness label
augmentation, model export, eval reuse)
**Branch:** task/learned-mid-mission-validator

## Story

As an **operator who has measured that the CLIP transit-monitor
path doesn't reach the per-case TPR / FPR bar even after the §3.1
cheap-fix pass**, I want **a small frozen-DINOv2 + temporal-head
validator with mission-text fusion, trained on the harness's
five-way categorical labels (`on_course` / `wrong_room` /
`wrong_instance` / `trajectory_violation` / `ambiguous`)**, so that
**mid-mission validation gets the purpose-built signal the
open-vocab CLIP recipe could not provide for case 2 (instance-grain)
in particular, without paying the small-VLA training and co-tenancy
cost**.

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
[`clip-mid-mission-validator-evaluation.md`](../active/clip-validation/validator-evaluation.md).

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
  consumes the `root_cause`-augmented harness output (the canonical
  schema lives in
  [`harness-behavior-cloning-data-expansion`](../active/harness/behavior-cloning-data-expansion.md);
  reads `frames_skill.jsonl` + `frames_tick.jsonl` when present,
  falls back to legacy `frames.jsonl`) produced by the
  CLIP-eval brief's `--root-cause-pass`, and emits
  `(frame_window, mission_text, category_label)` tuples to a
  parquet dataset. The label is **five-way categorical**, not
  binary: `on_course` / `wrong_room` / `wrong_instance` /
  `trajectory_violation` (always-empty in this brief; reserved
  field) / `ambiguous` (excluded from train / eval). Same-mission
  frames within `R_window=5` consecutive captures form a window.
  **Hard negatives** are consumed from deliberately-perturbed
  missions produced by the harness's `--inject-bad-grounding`
  flag — that flag is owned by
  [`harness-behavior-cloning-data-expansion`](../active/harness/behavior-cloning-data-expansion.md),
  not added here. This brief documents which submodes
  (`wrong_room`, `wrong_instance`) it relies on and runs them
  during dataset assembly.
- **Training script** at
  `source/strafer_lab/scripts/train_validator.py`. Backbone =
  frozen DINOv2-S/14 (cached HuggingFace `facebook/dinov2-small`).
  **Mission-text fusion is mandatory** (a vision-only validator
  collapses to the case-1 signal §3.1 already provides; see
  [`MISSION_VALIDATION_ARCHITECTURE.md` §3.2](../../MISSION_VALIDATION_ARCHITECTURE.md)).
  Mission text encoded by a frozen sentence encoder
  (`sentence-transformers/all-MiniLM-L6-v2`, 22 M params, ~5 ms on
  Orin Nano) — the runtime caches the embedding per mission so
  the cost is paid once. Fusion head = 3-layer Transformer over
  concatenated `(visual_patch_tokens, text_token)` sequence.
  Output: 4-class logit (`on_course` / `wrong_room` /
  `wrong_instance` / `trajectory_violation`; `ambiguous` is a
  training-time exclusion only) + per-frame attention map for
  diagnostics. AdamW, weighted cross-entropy with class weights
  inversely proportional to class frequency in the dataset.
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

The harness corpus is multi-room by default (per
[`multi-room-autonomy-stack`](../active/multi-room/autonomy-stack.md) +
[`multi-room-scene-connectivity-validation`](../active/multi-room/scene-connectivity-validation.md)).
This brief's *training* corpus uses both single-room and
multi-room missions; the *eval* corpus is **single-room first**
to match the single-room measurement scope of
[`clip-mid-mission-validator-evaluation`](../active/clip-validation/validator-evaluation.md).

Volume math:
- Single-room scenes (`scene_fast_singleroom_000_seed0`,
  `scene_high_quality_dgx_000_seed0` rendered single-room) plus
  1–2 multi-room Infinigen seeds gives ~30 missions per scene at
  the harness's defaults. With `R_window=5` and one window per
  leg-second, that's ~5–10k labeled windows per scene —
  enough for a frozen-backbone fine-tune with class re-weighting
  but tight for a from-scratch backbone (which this brief
  avoids).
- Multi-room missions are longer (~60–90s vs. ~30s), so windows
  per mission are 2–3× higher. Net dataset size grows roughly
  proportionally.

The hard-negative consumption from
[`harness-behavior-cloning-data-expansion`](../active/harness/behavior-cloning-data-expansion.md)'s
`--inject-bad-grounding` flag plus teleop-tagged hard negatives
multiply effective negatives without re-running the full sweep.

A multi-room eval re-test is filed as a follow-up brief
(`learned-validator-multi-room-remeasure.md`) once the
clip-eval's multi-room re-test ships and a multi-room bar is
established.

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
- [ ] **Hard-negative consumption.** Two equally-valid sources;
      either or both:
  - **Teleop hard negatives** (cleanest, recommended).
    [`harness-teleop-driver`](../active/harness/teleop-driver.md)'s `X` and
    `SELECT` buttons let the operator commit to specific
    failure modes (`wrong_instance` / `wrong_room` /
    `trajectory_violation`) at capture time, no post-hoc
    inference needed. ≥ 30 tagged hard-negative episodes per
    scene gets the dataset to a workable balance.
  - **Bridge-injected hard negatives** via the harness's
    `--inject-bad-grounding` flag (owned by
    [`harness-behavior-cloning-data-expansion`](../active/harness/behavior-cloning-data-expansion.md)).
    Useful for scaling beyond what an operator can demonstrate;
    requires the v1 stack reliable enough to run.

      Pipe whichever source(s) through the dataset builder so
      train / eval class balance reflects them. **Do not
      re-implement either flag here** — both are owned by
      upstream briefs.
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
- [ ] **Decision report.** Append a §4.5 to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md)
      with the learned validator's **per-case** TPR / FPR /
      time-to-decision (case 1: `wrong_room`; case 2:
      `wrong_instance`) on the same episode set, alongside the
      CLIP path's numbers from the eval brief's §4.4 addendum. If
      both case-1 + case-2 TPR ≥ 0.85 at FPR ≤ 0.1, flip the
      runtime default to `learned` in the same PR; otherwise
      leave the default at `clip` and record the per-case gap.
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
- Per-mission temporal grouping is by `mission_id` in the
  harness output. Canonical schema:
  [`harness-behavior-cloning-data-expansion`](../active/harness/behavior-cloning-data-expansion.md)
  (after that brief ships) or
  [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
  Stage 5 (legacy).
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
