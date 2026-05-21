# Learned VPR descriptors for semantic-graph loop closure

**Type:** new feature (filed-on-trigger)
**Owner:** DGX agent (ONNX export + runtime swap in
`semantic_map/`; cross-lane like every other backbone-touching
brief)
**Priority:** P3 — filed-on-trigger. Becomes pickable only if
[`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)
v1.5 ships AND its calibration sweep (now an acceptance
criterion on that brief) either can't find an operating point
that meets the precision-recall floor on the multi-bedroom
adversarial scene, OR finds one too narrow to survive sim-to-
real (precision degrades > 20% between sim and real on the
same scene topology). Do NOT pre-empt — raw CLIP cosine may
be sufficient at strafer's deployment scale.
**Estimate:** M (~half-week to a week; ONNX export of the
chosen descriptor + bakeoff-style eval + runtime swap behind
`STRAFER_VPR_DESCRIPTOR`)
**Branch:** task/learned-vpr-loop-closure

## Story

As an **operator running the v1.5 semantic-graph loop closure
on a real home where two physically-distinct bedrooms with
similar furniture vocabulary keep getting loop-closed into one
cluster (CLIP cosine ≥ 0.75 between them) AND the calibration
sweep can't find a `(similarity_threshold, distance_threshold_m)`
pair that holds precision and recall simultaneously**, I want
**the loop-closure detector's descriptor swapped from raw
OpenCLIP cosine to a learned Visual Place Recognition
descriptor (SALAD / MegaLoc / AnyLoc)**, so that **the
`same_place` edge set becomes instance-discriminating by
construction rather than by threshold tuning, the brittleness
v1.5 papered over with `distance_threshold_m` goes away, and
the same ANN store and `same_place` edge protocol continue
to work — only the embedding tower changes**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)
  — the v1.5 substrate. This brief swaps its
  `detect_loop_closures` descriptor while preserving the
  detection contract (candidate pairs in,
  `same_place` edges annotated). The
  `similarity_threshold` / `distance_threshold_m` knobs
  retire.
- [`semantic-map-lifecycle-merge`](semantic-map-lifecycle-merge.md)
  — primary consumer of loop-closure candidates; pooled-
  anchor path benefits from cleaner detections.
- [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
  — the multi-bedroom adversarial scene + repeated-traversal
  trajectories are the measurement substrate.
- [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
  — sibling backbone-selection brief. SALAD / MegaLoc / AnyLoc
  are VPR-specific; they do NOT replace the CLIP backbone
  that the rest of the system uses for room labeling, text
  queries, or validator cosine. This brief deploys VPR
  *alongside* the CLIP backbone, with a second ONNX session
  in `clip_encoder.py`.

## Trigger detail — when to un-park

This brief is medium-sized but adds a second backbone runtime
to maintain. Pre-empting it doubles the ONNX surface area
without evidence it's needed. File active only when **at least
one** of:

1. **v1.5 calibration sweep fails on the multi-bedroom
   scene.** The sweep over
   `similarity_threshold ∈ [0.60, 0.90]` and
   `distance_threshold_m ∈ [0.5, 3.0]` (per
   [`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)'s
   amended acceptance criterion) returns no operating point
   where multi-bedroom precision and same-room recall both
   exceed 0.7. v1.5's "tune the spatial filter" remediation
   ran out of room.
2. **Real-robot deployment shows the calibrated operating
   point doesn't transfer.** v1.5 ships an operating point
   tuned in sim; the real-robot D555 + lighting + clutter
   shift pushes the optimal point outside what v1.5's
   thresholds cover. The sim→real degradation on the same
   home is measurable (precision drops > 20% on the same
   topology).
3. **The lifecycle-merge brief's pooled anchors are
   consistently merging distinct places.** Loop-closure
   false positives propagate through the spatial-pool
   step in
   [`semantic-map-lifecycle-merge`](semantic-map-lifecycle-merge.md)
   and produce visibly wrong long-term anchors on the
   eval-harness 7-day replay.

If none of these fire within ~3 months of v1.5 shipping, the
likely correct action is to **delete this brief** and accept
raw CLIP cosine as the floor. Don't pre-empt.

## Context

### What's brittle about raw CLIP cosine for loop closure

OpenCLIP ViT-B/32 was trained for contrastive image-text
alignment over LAION-2B. It learned to put images and their
captions in similar embedding regions; it did NOT learn to
discriminate "this kitchen" from "that kitchen with a
different layout." Two physically-distinct rooms with similar
furniture (the multi-bedroom adversarial case) routinely
produce cosine similarities in `[0.75, 0.85]` — overlapping
the range that "same physical spot from different headings"
also occupies. The v1.5 brief acknowledges this and adds a
spatial filter (`distance_threshold_m`) as the
discrimination knob; the spatial filter does most of the
work because CLIP cosine alone is too coarse.

VPR descriptors (NetVLAD's lineage) are trained for *exactly*
this discrimination task on geo-tagged datasets (Pittsburgh,
Tokyo24/7, MSLS, GSV-Cities). The 2025 generation:

- **SALAD** (CVPR 2024) — DINOv2 backbone + Sinkhorn
  optimal-transport aggregation. Strong supervised baseline
  on the standard benchmarks.
- **MegaLoc** (2024) — foundation-model-driven, lifts raw
  DINOv2 / CLIP into VPR-grade descriptors without fine-
  tuning. Closest fit for "drop in without a training run."
- **AnyLoc** (ICRA 2024) — zero-shot VPR via DINOv2
  features + clustering. Lightest deployment cost.

All three produce a single per-image descriptor that supports
cosine ranking; the ANN-store contract is identical to v1.5's
CLIP path.

### Why this doesn't replace the CLIP backbone

The CLIP backbone serves at least four consumers in this
project:

1. **Room labeling** — `RoomClassifier` text↔image cosine.
2. **Text queries** —
   [`query-room-by-text-v1`](../../active/multi-room/query-room-by-text-v1.md).
3. **Validator (case 2)** — text-vs-image cosine on the
   mission target.
4. **`query_by_text` / `query_by_embedding`** — generic
   semantic-map retrieval.

VPR descriptors don't have a text tower; they don't speak
"kitchen" the way CLIP does. They speak "this place vs.
that place." Swapping the system-wide backbone to a VPR
descriptor would break consumers 1–3.

This brief deploys VPR **alongside** the CLIP backbone — a
second ONNX session in `clip_encoder.py`, loaded when
`STRAFER_VPR_DESCRIPTOR` is set, used **only** by the
loop-closure detector. CLIP keeps serving everything else.
Two backbones loaded; ~80MB extra memory; acceptable on the
DGX, measure on the Jetson.

### Approach

Three pieces, in dependency order:

**1. Pick a descriptor.** Recommended: **MegaLoc** for v1 of
this brief — foundation-model-driven, no fine-tune needed,
ONNX-exportable, and the closest fit for "drop in and
measure." SALAD is the supervised baseline if MegaLoc
underperforms on the eval set. AnyLoc is the lightest-weight
fallback. All three are evaluated head-to-head against the
v1.5 raw-CLIP baseline on the same multi-bedroom adversarial
scene.

**2. ONNX export + runtime swap.** A new ONNX file at
`~/.strafer/models/vpr/<descriptor>.onnx` consumed by an
extended `clip_encoder.py` runtime path. The loop-closure
detector calls a new `encode_for_loop_closure(image_rgb)`
method that returns the VPR descriptor when
`STRAFER_VPR_DESCRIPTOR` is set, falling back to the existing
`encode_image` (raw CLIP) when not. v1.5's detection pass
needs no other changes — `detect_loop_closures` is
descriptor-agnostic.

**3. Re-run the v1.5 calibration sweep.** Same sweep over
`similarity_threshold ∈ [0.60, 0.90]`, but the threshold
shifts (VPR descriptors typically sit in `[0.4, 0.8]` for
same-place pairs, much lower for distinct places). Report
the new operating point and the precision-recall lift over
raw CLIP. The `distance_threshold_m` filter should be
*loosable* — VPR's instance discrimination removes much of
the spatial-filter dependency.

### Tie-in to state-of-the-art

- **SALAD** (CVPR 2024) — DINOv2 + Sinkhorn aggregation.
  Public code, ONNX-friendly export. Strong supervised
  baseline.
- **MegaLoc** (2024) — foundation-model-driven VPR; lifts
  raw DINOv2 / CLIP without fine-tuning. Recommended for
  this brief's v1.
- **AnyLoc** (ICRA 2024, arXiv:2308.00688) — zero-shot VPR
  via DINOv2 features + VLAD clustering. Lightest-weight
  fallback.
- **CosPlace** (CVPR 2022) — classification-over-geocells
  approach; mature baseline.
- **EigenPlaces** (ICCV 2023) — viewpoint-tolerant; useful
  for the "same room different heading" case the v1.5
  detector targets.
- **MixVPR** (WACV 2023) — feature-mixer global descriptor.
- **Survey context:** [Improving VPR with Sequence-Matching
  Receptiveness Prediction (arXiv:2503.06840)](https://arxiv.org/abs/2503.06840),
  [VPR pair-retrieval evaluation (arXiv:2603.13917)](https://arxiv.org/abs/2603.13917).

The pattern is settled enough that "raw CLIP cosine for VPR"
is a 2023-era expedient and "learned VPR descriptor as a
sibling backbone to CLIP" is the 2025 production shape. This
brief lifts loop closure to that floor when the trigger
fires.

## Acceptance criteria

- [ ] **Trigger condition met before pickup.** The PR
      description names which of the three triggers
      (`a`, `b`, `c` in "Trigger detail" above) motivated the
      un-park. Numbers attached: v1.5's calibration-sweep
      operating-point coverage OR the sim-to-real
      degradation measurement OR the lifecycle-merge wrong-
      anchor count.
- [ ] **Descriptor pick + rationale.** SALAD / MegaLoc /
      AnyLoc selected and reasoned. Recommended MegaLoc for
      v1; document divergence.
- [ ] **ONNX export.** Descriptor exported to
      `~/.strafer/models/vpr/<descriptor>.onnx` via a
      script at
      `source/strafer_lab/scripts/export_vpr_onnx.py`,
      parameterized by `--descriptor`. Metadata JSON
      sidecar with checkpoint URL + export-time torch
      version + ONNX SHA-256, mirroring
      [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)'s
      export pattern.
- [ ] **Runtime swap.** `clip_encoder.py` gains an
      `encode_for_loop_closure(image_rgb)` method that
      reads `STRAFER_VPR_DESCRIPTOR` and dispatches to the
      VPR ONNX when set, else falls back to `encode_image`
      (raw CLIP). The loop-closure detector
      (`detect_loop_closures` in `room_state.py`) calls
      this new method exclusively. Other consumers of
      `encode_image` are unaffected.
- [ ] **Re-calibration on the eval set.** PR description
      includes the precision-recall sweep on the
      [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
      multi-bedroom adversarial scene with the new
      descriptor, comparing v1.5 raw-CLIP baseline vs. the
      VPR descriptor. The VPR path ships if it lifts the
      operating-point precision by ≥ one CI-width AND
      recovers same-room recall to ≥ 0.85 at that
      precision.
- [ ] **`distance_threshold_m` re-tuning.** PR description
      reports whether the spatial filter can be relaxed
      (or removed entirely) under the VPR descriptor. v1.5
      used the spatial filter to compensate for CLIP's
      poor instance discrimination; VPR may make the
      filter optional.
- [ ] **No regression** in non-loop-closure consumers of
      `encode_image`. Smoke test: with
      `STRAFER_VPR_DESCRIPTOR` unset, all v1.5 behavior is
      byte-for-byte identical.
- [ ] **Latency budget.** Per-frame VPR encode latency on
      the Jetson (Orin Nano FP16) reported as median + p95
      over ≥ 1000 captures. Budget: ≤ 200 ms (similar to
      the existing CLIP visual encode); above that, file a
      follow-up to evaluate AnyLoc or a smaller variant.
- [ ] **§4.8 addendum** to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
      with the descriptor pick, the operating-point
      comparison, and the latency measurement.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit.

## Investigation pointers

- v1.5 loop-closure detector:
  [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
  — `detect_loop_closures` (lands with the v1.5 brief).
- CLIP encoder pattern to mirror:
  [`semantic_map/clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py)
  — graceful-degrade, ONNX loading, preprocessing.
- ONNX export reference:
  [`source/strafer_lab/scripts/finetune_clip.py`](../../../../source/strafer_lab/scripts/finetune_clip.py#L262)
  — `export_towers_to_onnx`. Same shape.
- Eval scene set: the multi-bedroom adversarial scene from
  [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md).
- Reference VPR descriptors:
  - SALAD (CVPR 2024) — DINOv2 + Sinkhorn.
  - MegaLoc (2024) — foundation-model-driven, no fine-tune.
  - AnyLoc (ICRA 2024, arXiv:2308.00688) — zero-shot.
  - CosPlace (CVPR 2022), EigenPlaces (ICCV 2023),
    MixVPR (WACV 2023) — supervised baselines.
- Calibration-sweep procedure: same shape as v1.5's
  amended acceptance criterion in
  [`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md).

## Out of scope

- **Replacing the CLIP backbone system-wide.** CLIP keeps
  serving room labeling, text queries, validator cosine, and
  generic semantic-map retrieval. VPR is loop-closure only.
- **Fine-tuning the VPR descriptor on the harness corpus.**
  v1 of this brief uses off-the-shelf weights. A fine-tune
  follow-up lands only if the off-the-shelf descriptors
  underperform.
- **Replacing the same-place edge protocol.**
  [`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)'s
  edge annotations (`same_place` edge type with weight =
  similarity) are unchanged.
- **Lifecycle-merge spatial pooling.** That work lives in
  [`semantic-map-lifecycle-merge`](semantic-map-lifecycle-merge.md);
  this brief feeds it cleaner candidates, doesn't replace it.
- **Real-robot VPR data collection.** Sim-only training and
  eval; sim-to-real degradation measurement is the trigger,
  not new training data.
- **Multi-floor / cross-story VPR.** Strafer is single-story.
