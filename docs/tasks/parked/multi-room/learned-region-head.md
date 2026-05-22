# Learned region head — trained partition when v2 feature-clustering falls short

**Type:** investigation / new feature (filed-on-trigger)
**Owner:** DGX agent (training pipeline + ONNX export + runtime
swap inside `semantic_map/`; cross-lane like the other
backbone-touching briefs)
**Priority:** P3 — filed-on-trigger. The **escape valve** to v2
([`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md),
the unsupervised feature+space clustering). Becomes pickable
only when v2 ships AND the
[`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
shows v2's single `α` knob or its sim-to-real transfer can't
clear the V-measure floor on the adversarial scenes. Do NOT
pre-empt — v2's unsupervised clustering may be sufficient, in
which case this brief is deleted, not picked up.
**Estimate:** XL (~multi-week; training pipeline + held-out
eval + runtime swap + ablation against the v2 clustering
baseline)
**Branch:** task/learned-region-head

## Story

As an **operator running v2's feature+space region clustering
on a real home where the single `α` (feature/space weight)
knob can't simultaneously hold the open-plan split and the
multi-bedroom split, or where the clustering's operating point
doesn't transfer sim→real**, I want **a trained region head
that learns the partition + region embeddings end-to-end from
the harness corpus, rather than HDBSCAN over a hand-weighted
joint metric**, so that **the partition adapts to the data
instead of to a hand-set `α`, the per-node region assignment
stops being a fixed-metric clustering decision, and the
symbolic-layer quality lifts past where unsupervised
clustering plateaus — at the cost of one training pipeline and
a held-out scene-seed protocol the harness epic supports**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md) — where this brief sits in the v1 / v1.5 / v2 / v3 / escape-valve stack, and which planner-side consumers depend on its output.
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
  — v1 substrate (shipped). The original `RoomClassifier` +
  `cluster_nodes` + `aggregate_room_entries` pipeline.
- [`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md)
  — **v2, this brief's predecessor.** v2 replaces v1 with
  unsupervised feature+space clustering + open-vocab labels
  (one `α` knob, no training). This brief replaces v2's
  HDBSCAN-over-a-hand-weighted-metric with a *learned*
  partition when that single knob proves insufficient. The
  `RoomEntry` / `known_rooms` API surface is preserved across
  both.
- [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
  — produces both the training corpus and the held-out eval
  set. Held-out seed protocol is the trigger gate.
- [`query-room-by-text-v1`](../../active/multi-room/query-room-by-text-v1.md)
  — the per-room-query API. The head produces region-level
  embeddings the query path consumes; the API surface is
  unchanged.
- [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md)
  — sibling sub-symbolic primitive. This brief is the
  *symbolic* layer's learned partition (still produces
  discrete `RoomEntry`); the implicit memory map is the
  *sub-symbolic* layer's threshold-free path (memory bank +
  cross-attention), consumed by the validator and the VLA.

## Trigger detail — when to un-park

This brief is XL and supersedes v2's clustering. Pre-empting
it costs v2 the chance to prove the unsupervised approach is
sufficient. File active only when **at least one** of:

1. **v2 ships and the eval harness's cluster-purity V-measure
   on the open-plan or multi-bedroom adversarial scene stays
   below 0.6 across the full sweep of v2's single `α` knob.**
   That's the "the one knob can't hold both splits at once"
   bar — the failure mode where the feature/space weight that
   splits the open-plan kitchen/living also fragments a single
   bedroom, or vice versa.
2. **v2's calibrated `α` doesn't transfer sim→real.** The
   operating point tuned on the harness corpus degrades
   measurably on real-robot D555 captures (V-measure drop
   > 0.1 on the same home topology). A learned head trained
   with domain randomization can absorb the shift the fixed
   metric can't.
3. **The eval harness shows the partition is feature-distance
   bound** — regions that should split share too much CLIP
   feature mass (e.g., two near-identical bedrooms) for any
   `α` to separate them spatially without over-fragmenting
   elsewhere. A learned head with object-evidence + graph
   context as additional inputs beats a pure feature+space
   metric here.

If none of these bite within ~6 months of v2 shipping, the
likely correct action is to **delete this brief**, not pick it
up. v2's unsupervised clustering may be sufficient; the
trained head is the escape valve, not the goal.

## Context

### What this brief replaces

v2 ([`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md))
clusters nodes by a hand-weighted joint metric:

```
known_rooms():
    # HDBSCAN over d(a,b) = α·(1 − cos(clip_a, clip_b))
    #                     + (1 − α)·spatial_dist(a, b)
    clusters = partition_regions(graph, alpha=α_fixed)
    # open-vocab labels via CLIP-text similarity to centroids
    return [RoomEntry(...) per cluster]
```

v2 already collapsed v1's six tuning points to **one**: the
feature/space weight `α`. But `α` is a single global scalar —
it can't simultaneously hold a tight open-plan split (which
wants the feature term to dominate) and a clean multi-bedroom
split (which wants the spatial term to dominate) if a home has
both. And a fixed metric doesn't adapt sim→real.

The replacement — a learned partition:

```
known_rooms():
    # ONE forward pass over per-node CLIP embeddings +
    # detected_objects + graph topology, learned end-to-end.
    entries = region_head(graph)                       # learned partition
    return entries
```

The head learns the feature/space/object-evidence tradeoff
per-region from the harness corpus, instead of applying one
global `α` everywhere. The `RoomEntry` output and open-vocab
labeling are preserved from v2; only the partition step
changes from fixed-metric clustering to a learned model.

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
confidence) jointly with neighbor context baked in. Learns the
partition that v2's fixed-metric HDBSCAN approximates with one
global `α`. ~5M params; trains on the harness corpus +
held-out seeds. Recommended for v1 of this brief — the
structural fit (graph in, graph out) is right.

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
`room_state.py` runtime path. v2's unsupervised
feature+space clustering stays present as a fallback when the
head ONNX is missing — same graceful-degrade pattern as the
CLIP encoder. v2's no-ONNX path is preserved.

The runtime swap is gated on `STRAFER_REGION_HEAD_ENABLED`
(default `0` for safety until calibration completes; flips
to `1` once the per-scene V-measure on the held-out eval set
matches or beats v2's feature+space clustering).

### Why this is the symbolic layer's brittleness killer

v2's feature+space clustering already removed the worst
brittleness (the fixed prompt set, the threshold pile). What
survives in v2 is the single global `α` and the fixed-metric
assumption: one feature/space weight applied to the whole
home, and a clustering decision that can't use object
evidence or graph context as more than a distance term.

A learned head trained on the joint distribution
`P(region | clip_emb, detected_objects, graph_neighborhood)`
learns a *per-region* tradeoff instead of one global `α`, and
can absorb sim→real shift via domain randomization. The
*existence* of a learned head doesn't guarantee it beats v2 —
that's the eval-harness bar — but the *failure mode* is
different. v2 fails because one global knob can't fit a home
with both open-plan and multi-bedroom structure; the head
fails because the training data is wrong. The latter is
fixable by adding more / better-distributed scenes to the
harness; the former is structural to a single-scalar metric.

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
      replaces v2's `partition_regions` clustering step. v2's
      feature+space clustering preserved when the ONNX is
      missing or the flag is unset.
- [ ] **End-to-end eval lift.** PR description includes
      per-scene V-measure / label P/R / connectivity P/R
      on the held-out seeds, comparing:
      v1 baseline → v2 clustering → learned head. The head
      ships iff it beats v2 by ≥ one CI-width on cluster
      purity (V-measure) AND label precision on the open-plan
      and multi-bedroom adversarial scenes.
- [ ] **The single `α` knob is retired.** PR description
      confirms the head consumer exposes zero hand-set
      thresholds at the consumer boundary — no feature/space
      weight to tune.
- [ ] **Backward compat.** `RoomEntry` shape unchanged.
      `known_rooms` / `current_room` / `connectivity` /
      `room_anchor` / `query_room_by_text` API surface
      unchanged. v2 callers swap automatically when the
      env flag flips.
- [ ] **No regression** with `STRAFER_REGION_HEAD_ENABLED=0`
      (falls back to v2 clustering). Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites pass.
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
  [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md);
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
