# Learned spatial encoder — one frozen trunk, a place-recognition head and a region head

**Type:** investigation / research / new feature (filed-on-trigger)
**Owner:** DGX agent (trunk + heads + training + ONNX export are
DGX-side; runtime swap behind env flags is a small Jetson-lane
edit, cross-lane like every other backbone-touching brief)
**Priority:** P3 — filed-on-trigger. The shared **escape valve**
for v2's two unsupervised mechanisms: the feature+space region
partition
([`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md))
and the raw-CLIP loop closure
([`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)).
Un-parks if **either** mechanism measurably falls short — see
"Trigger detail." Do NOT pre-empt; the unsupervised v2 may
clear both bars.
**Estimate:** L–XL (~1.5–3 wk; trunk wiring + two heads +
training + held-out eval + runtime swap + ablation against
both v2 baselines)
**Branch:** task/learned-spatial-encoder

**Pickup gate:** Un-parks when **either** trigger below fires.
Blocked-on-deps until
[`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
ships (it is both the training corpus and the measurement
substrate) and the corresponding v2 mechanism has shipped and
been measured.

## Story

As an **operator running v2's unsupervised room-state stack on
a real home with long sessions**, where **(a) the region
partition's single `α` knob can't simultaneously split the
open-plan zones and keep revisited spots together, and/or
(b) raw-CLIP loop closure can't tell two similar bedrooms apart
without over-merging revisits**, I want **one learned spatial
encoder — a frozen foundation trunk feeding a place-recognition
head and a region-partition head — that produces
viewpoint-invariant same-place descriptors and data-driven
region assignments from the same forward pass**, so that
**better place recognition feeds cleaner `same_place` edges,
those edges let the region partition be aggressive enough to
split open-plan zones without fragmenting revisits, and the two
v2 brittleness points collapse into one trained artifact rather
than two hand-tuned thresholds — at the cost of one training
run on the harness corpus**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md) — the v3 escape valve for the symbolic layer; this brief replaces what were two separate escape valves with one.
- [`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md)
  — v2 region partition (unsupervised feature+space HDBSCAN +
  `α`). This brief's **region head** replaces its
  fixed-metric clustering with a learned partition.
- [`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)
  — v2 loop closure (raw-CLIP cosine + `similarity_threshold` /
  `distance_threshold_m`). This brief's **place-recognition
  head** replaces the raw-CLIP descriptor with a VPR descriptor;
  the `same_place` edge protocol is unchanged.
- [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
  — training corpus + held-out eval set + the open-plan and
  multi-bedroom adversarial scenes.
- [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
  — the trunk decision point. DINOv2 is the natural pick (it is
  what AnyLoc / SALAD / ConceptGraphs all build on); coordinate
  the trunk choice there.
- [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md)
  — the *sub-symbolic* sibling. That primitive is a memory bank
  + cross-attention for the validator / VLA; this brief is the
  *symbolic* layer's learned encoder. Both can share the same
  frozen trunk — note the co-tenancy below.

## Trigger detail — when to un-park

This brief is L–XL and supersedes v2's two unsupervised
mechanisms with a trained encoder. Pre-empting it denies v2 the
chance to prove the unsupervised approach is enough. File
active when **at least one** of:

1. **Region partition: the single `α` can't hold both splits.**
   v2 ships, and the eval-harness V-measure stays below 0.6 on
   the open-plan *or* multi-bedroom adversarial scene across the
   full `α` sweep — the failure mode where the feature/space
   weight that splits open-plan also fragments revisited spots
   (the long-session tension; see
   [`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)).
2. **Loop closure: raw-CLIP can't discriminate places.** v2's
   loop-closure calibration sweep can't find an operating point
   where multi-bedroom precision and same-room recall both
   exceed 0.7, OR the calibrated point degrades > 20% sim→real.
3. **Either falls to long-session drift.** On the eval-harness
   repeated-traversal trajectory (≥ 3× the same scene), region
   V-measure or loop-closure recall degrades materially as nodes
   accumulate — the regime the single-traversal smoke test
   doesn't exercise and that matures as sessions stop being
   one-shot.

**Why one brief covers all three:** triggers 1 and 2 are the
two ends of the *same* coupling. Better place recognition
(trigger 2's fix) produces cleaner `same_place` edges, which
directly relaxes the `α` tension (trigger 1's fix). They share
a trunk and feed each other; fixing one in isolation leaves
value on the table. If only one trigger fires, the brief ships
only that head (see staging) — but the trunk and eval are
built once.

If none fire within ~6 months of v2 shipping, **delete this
brief**; the unsupervised v2 was sufficient.

## Context

### Architecture — one frozen trunk, two heads

State-of-the-art spatial perception has converged on a single
frozen foundation trunk feeding multiple lightweight task
heads, rather than separate networks per task:

```
RGB frame
   │
   ▼
DINOv2 trunk (FROZEN)  ── one forward pass per frame
   │   ├─ dense patch tokens  (N_patches × D)
   │   └─ CLS / pooled token  (D)
   │
   ├──────────────► PLACE-RECOGNITION HEAD
   │                  VLAD / SALAD aggregation over dense
   │                  tokens → global descriptor (R^d_vpr)
   │                  → ChromaDB ANN → same-place candidates
   │                  → semantic-graph-loop-closure's
   │                    `same_place` edges
   │
   └──────────────► REGION HEAD
                      lightweight per-node classifier (MLP or
                      a GNN over the proximity graph) on the
                      pooled token + detected_objects one-hot
                      → per-node region assignment + label +
                        confidence → known_rooms
```

Concrete shapes (DINOv2 ViT-S/14 at 224², the Jetson-budget
pick; ViT-B/14 if latency allows):

- **Trunk:** DINOv2 ViT-S/14 — 256 patch tokens × 384-d + a
  384-d CLS token. **Frozen by default** (see training scheme).
- **Place-recognition head:** AnyLoc-style VLAD aggregation
  over the value-facet dense tokens against a vocabulary of
  K=64–128 cluster centers built unsupervised on indoor
  frames → a `K×384`-d VLAD descriptor, PCA-whitened to
  ~512–4096-d. Zero-shot at the floor; a learned Sinkhorn
  aggregation (SALAD) is the escalation. Output replaces the
  raw-CLIP embedding the v2 loop-closure detector ranks on —
  same ANN-store + `same_place` contract, different descriptor.
- **Region head:** a 2-layer MLP (`384 + |obj-vocab| → 256 →
  n_regions` logits) per node at the floor; a 2–3-layer GNN
  (GCN/GAT over the proximity + `same_place` edges) as the
  escalation that bakes in neighbor context. Output per node:
  region id + open-vocab-compatible label embedding +
  confidence. Replaces v2's `partition_regions` HDBSCAN.

Both heads consume the **same** frozen DINOv2 forward pass, so
the per-frame cost is one trunk encode + two cheap heads.

### Why the trunk is frozen (and when to unfreeze)

The dominant 2024-2025 pattern freezes the foundation trunk and
trains only lightweight heads. The reason is a gradient
conflict: fine-tuning the trunk for region *classification*
pulls features toward category discrimination and **away** from
the viewpoint invariance place recognition needs — the same
objective tension
[`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)
flags for action-vs-contrastive. AnyLoc demonstrates VPR works
**zero-shot** on a frozen DINOv2 trunk; ConceptGraphs / HOV-SG
run their semantic *and* cross-view-merge tasks off one frozen
trunk. So the clean v1 is **frozen trunk + two independently
trained heads** — no multi-task balancing, no conflict, the
heads don't share parameters beyond the frozen features.

Unfreezing (joint multi-task fine-tune of the trunk) is an
**escalation only**, gated on heads-only plateauing. If
attempted: multi-task loss = region cross-entropy + VPR
contrastive, with gradient surgery (PCGrad) or uncertainty
weighting, and a hard guard that VPR recall must not regress as
region accuracy climbs. Document the trade if pursued.

### Training scheme — three regimes, escalating

| Regime | Trunk | VPR head | Region head | When |
|---|---|---|---|---|
| **R0 — zero-shot floor** | frozen | AnyLoc VLAD (no training; vocab built on indoor frames) | — (v2's HDBSCAN still runs) | Quick check: does a frozen DINOv2 VPR descriptor alone clear trigger 2 without any training? |
| **R1 — heads-only (recommended v1)** | frozen | AnyLoc VLAD floor, or light SALAD aggregation fine-tune | learned MLP/GNN region head, supervised | The default. Two heads train independently on the frozen trunk; no multi-task balancing. |
| **R2 — co-fine-tune (escalation)** | unfrozen | SALAD | GNN | Only if R1 plateaus. Multi-task loss + gradient surgery + the VPR-recall guard above. |

**Data + supervision:**

- **Region head:** harness trajectories; ground-truth region per
  node from
  [`scene_labels.get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py)
  over Infinigen `room_idx`, **re-grouped for open-plan** (zones
  with no demarcating wall that the planner should treat as one
  region get one label; the eval-harness open-plan adversarial
  defines the grouping). Loss: cross-entropy + an optional
  intra-region contrastive term.
- **VPR head (if trained, SALAD):** same-place pairs mined from
  the harness — two observation nodes within R metres whose
  RTAB-Map poses agree (true same-place), contrasted against
  far-apart nodes (true different-place). Multi-bedroom pairs
  are the hard negatives. Loss: triplet / multi-similarity, the
  standard VPR recipe.
- **Held-out protocol:** the eval-harness `held_out_seeds`.
  For VPR, hold out **whole scenes** (place recognition must
  generalise to unseen homes, not just unseen trajectories in a
  seen home). For the region head, the held-out seeds suffice.
- **Domain randomisation:** lighting / texture / camera
  perturbation during training — load-bearing for VPR, which is
  the sim-to-real-fragile head (trigger 2's failure mode is
  exactly sim→real descriptor drift).

### Metric gathering

Three metric families, all computed on the
[`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
held-out set. Each head has its own bar; the **coupled metric**
is what justifies one brief over two.

**Place-recognition metrics (standard VPR protocol):**

| Metric | Definition | Bar |
|---|---|---|
| Recall@1 / @5 / @10 | Is the true same-place node in the top-K ANN retrievals? | Recall@1 ≥ v2 raw-CLIP + 1 CI-width |
| Same-place precision-recall | PR curve over the descriptor-similarity sweep; pick the loop-closure operating point | precision ≥ 0.9 at recall ≥ 0.7 on the multi-bedroom scene |
| Multi-bedroom false-merge rate | Fraction of distinct-bedroom pairs incorrectly same-placed | Strictly below v2's |
| Sim→real Recall@1 gap | Recall@1 on real D555 captures vs sim, same home topology | < 10% degradation (the trigger-2 floor was > 20% for v2) |

**Region metrics (the eval-harness room-state metrics):**

| Metric | Definition | Bar |
|---|---|---|
| Cluster purity (V-measure) | per-node region vs Infinigen `room_idx` | ≥ v2 clustering + 1 CI-width on open-plan AND multi-bedroom |
| Label P/R | per `RoomEntry.label` vs ground truth | ≥ v2 |
| Open-plan split | open-plan scene yields ≥ 2 coherent regions | holds without fragmenting revisits |

**The coupled metric (justifies the collapse):**

- **Region V-measure with VPR-`same_place`-edges ON vs OFF.**
  Run the region head twice on the repeated-traversal
  trajectory: once with the VPR head's `same_place` edges fed
  into the graph, once without. The lift from VPR-on is the
  evidence that the two heads are functionally coupled (cleaner
  place recognition → better partition under long sessions). If
  the lift is zero, the heads are independent after all and the
  brief can be re-split — but the SOTA prior is a clear positive
  lift, which is why this ships as one brief.

**Latency:** one DINOv2 forward pass + VLAD + region head on the
Jetson Orin; report median + p95 per frame. DINOv2 ViT-S/14 is
~tens of ms at FP16; both heads are cheap. Budget: under the
`BackgroundMapper` 2 s poll interval with comfortable margin.

### Tie-in to state-of-the-art

- **AnyLoc** (Keetha et al., RA-L 2023 / ICRA 2024,
  [arXiv:2308.00688](https://arxiv.org/abs/2308.00688)) —
  frozen DINOv2 facet features + VLAD → SOTA VPR **zero-shot**.
  The R0 floor and the proof that the trunk already carries
  place-discriminative features.
- **SALAD** (Izquierdo & Civera, CVPR 2024) — DINOv2 +
  optimal-transport (Sinkhorn) aggregation, lightly fine-tuned.
  The R1/R2 VPR-head upgrade.
- **DINOv2** (Oquab et al. 2023) — the frozen trunk both heads
  share.
- **ConceptGraphs** ([ICRA 2024](https://arxiv.org/abs/2309.16650)),
  **HOV-SG** ([RSS 2024](https://arxiv.org/abs/2403.17846)) —
  one foundation trunk feeding semantic region/object features
  AND cross-view merging; the precedent for "two tasks, one
  trunk."
- **MegaLoc** (2024) — foundation-model-driven VPR; the
  no-fine-tune alternative to SALAD if R1's VPR fine-tune
  underdelivers.

### Co-tenancy with the implicit memory map

[`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md)
(the sub-symbolic primitive for the validator / VLA) can share
**the same frozen DINOv2 trunk** loaded on the Jetson — one
trunk, three heads total (VPR, region, memory-bank
cross-attention). If both this brief and the implicit memory
map ship, coordinate the trunk load so it's resident once. The
[`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
brief is the single trunk-choice decision point for all three.

## Approach — staged

Ship only the head(s) the firing trigger demands; build the
trunk + eval once.

1. **Trunk wiring.** Load DINOv2 (from the bakeoff's pick) as a
   second frozen ONNX session in `clip_encoder.py` behind
   `STRAFER_SPATIAL_ENCODER`; expose the dense + pooled features
   to both heads. CLIP stays the system backbone for room
   labeling / text queries / validator cosine (the spatial
   encoder is additive, same co-tenancy rule as the retired VPR
   brief).
2. **R0 zero-shot VPR check.** Build the VLAD vocabulary on
   harness indoor frames; measure VPR metrics. If R0 clears
   trigger 2 with no training, ship the VPR head as-is.
3. **R1 heads-only training.** Train the region head
   (supervised); fine-tune the VPR aggregation (SALAD) if R0
   fell short. Independent training runs; frozen trunk.
4. **Eval + coupled-metric report.** Full VPR + region + coupled
   metrics vs both v2 baselines on the held-out set.
5. **R2 co-fine-tune** only if R1 plateaus, with the gradient
   guard.

## Acceptance criteria

- [ ] **Trigger recorded.** PR description names which
      trigger(s) fired with the eval-harness numbers (region
      V-measure under `α` sweep / loop-closure PR / sim→real
      gap / long-session degradation).
- [ ] **Frozen trunk, two heads.** DINOv2 (bakeoff's pick)
      loaded once, frozen by default; VPR head + region head
      consume the same forward pass. Trunk choice coordinated
      with
      [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md).
- [ ] **Place-recognition head.** VLAD (R0 floor) or SALAD
      (R1) aggregation → global descriptor → the existing
      ChromaDB ANN + `semantic-graph-loop-closure` `same_place`
      protocol. VPR metrics reported per the table; beats v2
      raw-CLIP by ≥ 1 CI-width on Recall@1 and on multi-bedroom
      false-merge rate.
- [ ] **Region head.** Learned MLP/GNN per-node partition,
      supervised on the harness corpus with open-plan-regrouped
      labels. Replaces v2's `partition_regions` behind
      `STRAFER_REGION_HEAD_ENABLED`; v2 HDBSCAN is the fallback.
      Region metrics beat v2 clustering by ≥ 1 CI-width on the
      open-plan AND multi-bedroom scenes.
- [ ] **Frozen-first.** R1 ships with a frozen trunk and two
      independently trained heads — no multi-task balancing. R2
      (co-fine-tune) is attempted only if R1 plateaus, and only
      with gradient surgery + a VPR-recall non-regression guard.
- [ ] **Coupled-metric report.** Region V-measure with VPR
      `same_place` edges ON vs OFF on the repeated-traversal
      trajectory. A positive ON-vs-OFF lift is the evidence the
      heads belong in one brief; report it explicitly.
- [ ] **Sim→real VPR.** Recall@1 gap < 10% on real D555 vs sim
      for the same home, with domain randomisation in training.
- [ ] **Backward compat.** `RoomEntry` shape + `known_rooms` /
      `current_room` / `connectivity` / `room_anchor` /
      `query_room_by_text` unchanged. `same_place` edge protocol
      unchanged. Both fallbacks (v2 HDBSCAN, v2 raw-CLIP loop
      closure) preserved behind the env flags.
- [ ] **Latency.** One-trunk + two-head per-frame latency on
      Orin (median + p95) under the `BackgroundMapper` poll
      interval. Reported.
- [ ] **Unit tests.** Synthetic graphs for the region head
      (open-plan split, multi-bedroom split, single room);
      same-place / different-place fixtures for the VPR head;
      frozen-trunk no-grad assertion.
- [ ] **No regression** with both env flags off (falls back to
      v2). Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites pass.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.

## Investigation pointers

- v2 region partition replaced by the region head:
  [`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md)
  — `partition_regions`.
- v2 loop closure whose descriptor the VPR head replaces:
  [`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)
  — `detect_loop_closures`, the `same_place` edge protocol.
- CLIP encoder co-tenancy pattern (load a second frozen ONNX
  session):
  [`semantic_map/clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py).
- Training corpus + held-out seeds + adversarial scenes:
  [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md).
- Ground-truth region labels:
  [`scene_labels.get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py).
- Reference architectures:
  - AnyLoc (arXiv:2308.00688) — frozen-DINOv2 + VLAD, zero-shot VPR.
  - SALAD (CVPR 2024) — DINOv2 + Sinkhorn aggregation.
  - DINOv2 (Oquab et al. 2023) — the shared trunk.
  - ConceptGraphs (arXiv:2309.16650), HOV-SG (arXiv:2403.17846) — one trunk, semantic + merge.
  - MegaLoc (2024) — no-fine-tune VPR alternative.

## Out of scope

- **The unsupervised v2 mechanisms themselves.** v2's
  feature+space partition
  ([`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md))
  and raw-CLIP loop closure
  ([`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md))
  remain the floor and the fallback; this brief is their shared
  escape valve, not a rewrite of them.
- **Task-driven dynamic granularity.** That's
  [`dynamic-region-granularity`](dynamic-region-granularity.md)
  (CLIO-style); orthogonal — it changes *granularity per task*,
  this brief changes *how regions and places are computed*.
- **The sub-symbolic memory map.**
  [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md)
  is the validator/VLA primitive; it can share this brief's
  frozen trunk but has a different head and consumer.
- **System-wide CLIP backbone swap.** Backbone selection stays
  at
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md);
  this brief consumes the trunk it picks and adds the spatial
  encoder alongside CLIP.
- **Multi-floor.** Strafer is single-story.
- **Real-robot training data.** Sim-only training corpus +
  domain randomisation for transfer; real-robot fine-tune is a
  future brief if the sim→real gap persists past the bar.
