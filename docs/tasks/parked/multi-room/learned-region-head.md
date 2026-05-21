# Learned region head — retire the prompt-set + clustering + smoothing pipeline

**Type:** investigation / new feature (filed-on-trigger)
**Owner:** DGX agent (training pipeline + ONNX export + runtime
swap inside `semantic_map/`; cross-lane like the other
backbone-touching briefs)
**Priority:** P3 — filed-on-trigger. Becomes pickable when the
v2 room-state series ships
([`room-state-temporal-smoothing`](../../active/multi-room/room-state-temporal-smoothing.md),
[`room-state-uncertainty-calibration`](../../active/multi-room/room-state-uncertainty-calibration.md),
[`room-label-vlm-refinement`](../../active/multi-room/room-label-vlm-refinement.md))
AND the
[`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
shows the symbolic-layer pipeline plateauing below the
V-measure floor on the multi-bedroom adversarial scene (or
similar adversarial home seeds). Do NOT pre-empt — v2 may
clear the bar.
**Estimate:** XL (~multi-week; training pipeline + held-out
eval + runtime swap + ablation against the v2 stack)
**Branch:** task/learned-region-head

## Story

As an **operator running the v2 symbolic room-state stack
(per-node CLIP top-1 → graph clustering → smoothing →
uncertainty calibration → VLM prototype refinement) on a
real home where the multi-bedroom case and the rooms-that-
don't-fit-the-seven-classes case keep flickering across
re-runs**, I want **a single learned region head that takes
a CLIP image embedding (or a small temporal window of them)
and outputs `(region_id, label, confidence)` directly, trained
on the eval-harness corpus**, so that **the four threshold-
tuned briefs in the v2 series (smoothing, calibration, VLM
refinement, parts of loop closure) collapse into one forward
pass, the per-node classification stops being i.i.d. CLIP
top-1, and the symbolic-layer brittleness the multi-room
audit (PR #43) flagged at the prompt-set / threshold boundary
goes away — at the cost of one new training pipeline and a
held-out scene-seed protocol**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)
  — v1 substrate. The region head replaces v1's
  `RoomClassifier` + `cluster_nodes` + `aggregate_room_entries`
  pipeline.
- [`room-state-temporal-smoothing`](../../active/multi-room/room-state-temporal-smoothing.md)
  — v2 smoothing pass. Subsumed by this brief if the head
  ships (one forward pass handles temporal aggregation as
  the architecture chooses; no separate smoothing step).
- [`room-state-uncertainty-calibration`](../../active/multi-room/room-state-uncertainty-calibration.md)
  — v2 calibration pass. Subsumed: the head outputs
  calibrated logits directly (cross-entropy loss + held-out
  ECE is part of training).
- [`room-label-vlm-refinement`](../../active/multi-room/room-label-vlm-refinement.md)
  — v2 VLM prototype refinement. Subsumed: per-node
  `detected_objects` are an input feature to the head, not
  a downstream override.
- [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
  — produces both the training corpus and the held-out eval
  set. Held-out seed protocol is the trigger gate.
- [`query-room-by-text-v1`](../../active/multi-room/query-room-by-text-v1.md)
  — the per-room-query API. The head produces region-level
  embeddings the query path consumes; the API surface is
  unchanged.
- [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)
  — sibling sub-symbolic primitive. This brief is the
  *symbolic* layer's brittleness-killer (still produces
  discrete `RoomEntry`); cotrained is the *sub-symbolic*
  layer's threshold-free path (memory bank + cross-
  attention).

## Trigger detail — when to un-park

This brief is XL and partially obsoletes work the v2 series
ships. Pre-empting it costs the v2 PRs review attention they
deserve. File active only when **at least one** of:

1. **The v2 series ships and the eval harness's cluster-purity
   V-measure on the multi-bedroom adversarial scene stays
   below 0.6 after smoothing + calibration + VLM-refinement
   land.** That's the "we tuned every threshold we could
   reach and it still doesn't work" bar.
2. **A room type appears in real-robot deployment that isn't
   in `DEFAULT_ROOM_PROMPTS` and the operator can't add a
   prompt + prototype pair without measurable regression on
   the existing seven classes.** Open-vocab labeling (per
   [`query-room-by-text-v1`](../../active/multi-room/query-room-by-text-v1.md))
   patches the *query* side but the *stamp* side is still
   stuck on the seven classes — if that's the bottleneck,
   pick this up.
3. **The eval harness shows per-node classification accuracy
   is i.i.d.-bound** (every node classified independently;
   neighbors disagree even after smoothing because the CLIP
   top-1 signal is noisy at the input). A learned head with
   a small temporal/spatial context window beats this by
   construction.

If none of these bite within ~6 months of the v2 series
shipping, the likely correct action is to **delete this
brief**, not pick it up. v2's threshold-tuning may be
sufficient; the head is the escape valve, not the goal.

## Context

### What this brief replaces

The v2 symbolic room-state stack today:

```
add_observation(rgb, detected_objects):
    clip_emb = clip_encoder.encode_image(rgb)
    raw_label, raw_conf = clip_classifier.classify(clip_emb)
                                                       # 7-class argmax
    if detected_objects:
        vlm_label, vlm_score = refine_from_objects(...)
                                                       # Jaccard threshold
    metadata["room_label"] = chosen
    metadata["room_label_dist"] = calibrated_softmax   # temperature T
    # ... cluster cache invalidated ...

known_rooms():
    clusters = cluster_nodes(graph)                    # modularity / CC
    smooth_labels(graph)                               # BP, two passes
    entries = aggregate_room_entries(graph, clusters)  # majority vote
    return entries
```

Six tuning points: prompt set (size + content), temperature
`T`, smoothing `neighbor_weight`, smoothing `n_passes`,
Jaccard threshold, cluster cache invalidation fraction. All
hand-set; each fails in some adversarial regime.

The replacement:

```
add_observation(rgb, detected_objects):
    clip_emb = clip_encoder.encode_image(rgb)
    metadata["clip_emb_id"] = store(clip_emb)
    # NO per-node classification; the head does it at query time

known_rooms():
    # ONE forward pass over the per-node CLIP embeddings +
    # detected_objects, conditioned on the graph topology.
    entries = region_head(graph)                       # learned
    return entries
```

The head's architecture choice is itself a tunable, but the
**number of consumer-visible knobs goes to zero** — the
threshold-tuning surface area collapses inside one trained
artifact that the eval harness measures end-to-end.

### Head architecture options

Three viable shapes, escalating in capacity:

**(a) Per-node MLP head + graph aggregation.** Small MLP
(~2 layers, 512→256→K classes) on each node's CLIP embedding
+ detected-object one-hot. Region assignment via greedy
max-cut over predicted labels, no separate clustering. ~1M
params; trains on a few thousand harness frames. Closest to
v2's shape; cheapest to ship. Likely floor.

**(b) Graph neural network over the proximity graph.** GNN
(GCN / GAT, ~3 layers) consumes the node features + the
proximity edge structure; outputs per-node (label, region_id,
confidence) jointly with neighbor context baked in. Replaces
smoothing and clustering simultaneously. ~5M params; trains
on the harness corpus + held-out seeds. Recommended for
v1 of this brief — the structural fit (graph in, graph out)
is right.

**(c) Transformer over a temporal window of recent
observations.** Reads the last N frames as a sequence,
attention picks up persistent regions vs. transient frames.
~20M params; trains on full trajectories. Highest ceiling,
most data-hungry. Defer to v3 unless (b) is data-bound.

The brief picks (b) by default; (a) is the floor; (c) is the
ceiling. Each costs one training pipeline.

### Training corpus

The
[`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
brief already specifies trajectories captured per scene with
raw RGB + pose + timestamp. Ground-truth per-pose `room_idx`
comes from
[`scene_labels.get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py)
over the Infinigen room polygons. That **is** the training
corpus — same data, used for training in this brief and for
held-out eval in the eval-harness brief. Held-out seed
protocol is non-negotiable: the same protocol the eval
harness defines (2/3 seeds train, 1/3 eval) carries over
here, and the per-seed disaggregation rules out the seven-
prompt prior leaking via the prompt set.

The detected-object features come from the existing VLM
detections on mission-driven nodes (`scan_for_target`,
`verify_arrival`); background-mapper-only nodes have no
detections (input feature is zero-padded). This is realistic
— the deployment-time distribution matches the training-
time distribution.

### Deployment

The head ships as an ONNX artifact in
`~/.strafer/models/region_head/` consumed by an extended
`room_state.py` runtime path. The v1 `RoomClassifier` stays
present as a fallback when the head ONNX is missing — same
graceful-degrade pattern as the CLIP encoder. The v2 series'
ONNX-less path is preserved.

The runtime swap is gated on `STRAFER_REGION_HEAD_ENABLED`
(default `0` for safety until calibration completes; flips
to `1` once the per-scene V-measure on the held-out eval set
matches or beats the v2 stack).

### Why this is the symbolic layer's brittleness killer

Every tuning point in v2 exists because the input signal
(per-node CLIP top-1) is too weak to support the downstream
operations without help. Smoothing fixes the i.i.d. noise.
Calibration fixes the cosine-isn't-a-probability problem.
VLM-refinement fixes the missing-context problem. Each adds
one knob.

A learned head trained on the actual joint distribution
`P(label | clip_emb, detected_objects, graph_neighborhood)`
handles all three failure modes inside the network. The
*existence* of a learned head doesn't guarantee it beats v2 —
that's the eval-harness bar — but the *failure mode* is
different. v2 fails because the threshold is wrong; the head
fails because the training data is wrong. The latter is
fixable by adding more / better-distributed scenes to the
harness; the former requires a per-deployment tuning pass.

### Tie-in to state-of-the-art

- **OpenScene** (CVPR 2023, arXiv:2211.15654) — open-vocab
  3D scene understanding. Learned per-point feature head;
  same shape at a different granularity.
- **ConceptFusion** (arXiv:2302.07241) — per-pixel CLIP
  feature head trained on the same backbone. The 2D
  equivalent of this brief's per-graph-node head.
- **HOV-SG** (RSS 2024, arXiv:2403.17846) — hierarchical
  scene-graph nodes consumed by a learned head for room /
  region prediction. Direct architectural precedent at
  option (b) (GNN over the graph).
- **GRES / GRES++** (NeurIPS 2024) — referring expression
  segmentation via a graph head. Same shape: input is
  graph + features; output is region assignment.
- **Khronos** (RSS 2024) — explicit spatio-temporal scene
  graph with learned region prediction. Direct precedent
  for option (c) (transformer over a temporal window).
- **NaVid / NaVILA** (CoRL 2024 / 2025) — VLA + learned
  region prior. Less directly applicable (they consume
  the prior, not produce it), but informs option (c)'s
  context-window choice.

The pattern is well-established. The 2025 floor is "a small
learned head trained on a held-out scene split"; the ceiling
is "a transformer over a memory bank shared with the VLA."
v2's hand-set thresholds were the 2022 floor.

## Acceptance criteria

- [ ] **Trigger condition met before pickup.** The PR
      description names which of the three triggers
      (`a`, `b`, `c` in "Trigger detail" above) motivated
      the un-park. Numbers attached: v2's V-measure on the
      adversarial scene OR the missing room type OR the
      i.i.d. noise diagnostic.
- [ ] **Head architecture picked.** Option (a) / (b) / (c)
      selected and rationale documented. Recommended (b) for
      v1 of this brief; reasons for diverging are stated.
- [ ] **Training pipeline.**
      `source/strafer_lab/scripts/train_region_head.py`
      consumes the
      [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
      corpus (raw RGB → CLIP embeddings + detected_objects +
      graph topology + ground-truth `room_idx`). Held-out
      seed protocol is identical to the eval harness's
      `held_out_seeds` field. MLflow tracking per the
      existing fine-tune scripts' pattern.
- [ ] **ONNX export + runtime swap.** Head ships as
      `~/.strafer/models/region_head/head.onnx`. Runtime path
      in `room_state.py` reads the env var
      `STRAFER_REGION_HEAD_ENABLED` (default `0`); when set,
      replaces `RoomClassifier` + `cluster_nodes` +
      `aggregate_room_entries`. v1 path preserved when the
      ONNX is missing or the flag is unset.
- [ ] **End-to-end eval lift.** PR description includes
      per-scene V-measure / label P/R / connectivity P/R
      on the held-out seeds, comparing:
      v1 baseline → v2 stack → region head. The head ships
      iff it beats v2 by ≥ one CI-width on cluster purity
      (V-measure) AND label precision on the multi-bedroom
      adversarial scene.
- [ ] **Threshold-tuning surface area collapses.** PR
      description enumerates the v2 tunables retired (prompt
      set, temperature `T`, smoothing weights, Jaccard
      threshold, cluster cache invalidation fraction) and
      confirms the head consumer exposes zero hand-set
      thresholds at the consumer boundary.
- [ ] **Backward compat.** `RoomEntry` shape unchanged.
      `known_rooms` / `current_room` / `connectivity` /
      `room_anchor` / `query_room_by_text` API surface
      unchanged. v2 callers swap automatically when the
      env flag flips.
- [ ] **No regression** in the v1 path with
      `STRAFER_REGION_HEAD_ENABLED=0`. Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites pass.
- [ ] **Sunset which v2 briefs (if any).** PR description
      names whether
      [`room-state-temporal-smoothing`](../../active/multi-room/room-state-temporal-smoothing.md),
      [`room-state-uncertainty-calibration`](../../active/multi-room/room-state-uncertainty-calibration.md),
      [`room-label-vlm-refinement`](../../active/multi-room/room-label-vlm-refinement.md)
      remain shipped-and-supported (head as a sibling) or
      get retired (head as a replacement). Decision is per-
      brief; the head ships either way.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit.

## Investigation pointers

- v1 substrate that the head replaces:
  [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
  — `RoomClassifier`, `cluster_nodes`,
  `aggregate_room_entries`.
- Training corpus shape:
  [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
  — same trajectories, same held-out seed protocol.
- Sibling fine-tune scripts:
  [`source/strafer_lab/scripts/finetune_clip.py`](../../../../source/strafer_lab/scripts/finetune_clip.py)
  — pattern for export + MLflow + CLI flags.
- Reference architectures:
  - OpenScene (arXiv:2211.15654) — open-vocab feature head.
  - ConceptFusion (arXiv:2302.07241) — per-pixel CLIP feature.
  - HOV-SG (arXiv:2403.17846) — GNN-over-graph head.
  - Khronos (RSS 2024) — spatio-temporal scene graph.
  - GRES / GRES++ (NeurIPS 2024) — graph referring head.

## Out of scope

- **Replacing the CLIP backbone.** Backbone choice stays at
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md).
  This brief consumes whichever embedding tower is selected.
- **Sub-symbolic / VLA path.** The implicit-mapping primitive
  lives at
  [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md);
  this brief is the symbolic-layer counterpart. The two are
  parallel; ship one, the other, or both as triggers fire.
- **Multi-floor / cross-story regions.** Strafer is
  single-story; the head outputs single-floor regions only.
- **Real-robot data.** Sim-only training corpus, same rule
  as every other v2 brief. Real-robot transfer is a future
  follow-up if the head sim-to-real-fails materially.
- **Online fine-tuning during deployment.** The head ships
  as a fixed ONNX; periodic re-training off accumulated
  harness data is a separate follow-up.
- **Replacing the loop-closure detector.** That's at
  [`learned-vpr-loop-closure`](learned-vpr-loop-closure.md);
  region head and VPR are orthogonal — region head decides
  *what* a node is, VPR decides *whether two nodes are the
  same place*.
