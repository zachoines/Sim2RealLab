# Learned spatial encoder — VPR + region heads on the shared text-capable trunk (dedicated DINOv2 trunk is the escape valve)

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
without over-merging revisits**, I want **a learned spatial
encoder — a place-recognition head and a region-partition head
on the shared text-capable trunk the bake-off picks (a dedicated
DINOv2 trunk only as the escape valve) — that produces
viewpoint-invariant same-place descriptors and data-driven
region assignments from the shared frozen forward pass**, so that
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
- [`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md)
  — the FROZEN spine. By default this brief's VPR + region heads
  run on the **shared text-capable trunk** the bake-off picks
  (one trunk, many heads). The dedicated DINOv2 trunk below is
  the spine's **escape valve**, un-parked only via trigger 2.
- [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
  — the trunk decision point. It picks a **text-capable** shared
  trunk (SigLIP-2-Base lead, MobileCLIP-2-S, OpenCLIP baseline);
  it does **not** offer DINOv2. This brief's VPR Recall@K and
  region V-measure jobs are folded into the bake-off so the
  shared trunk is scored on them. A dedicated DINOv2 trunk is
  reachable only if that shared trunk underdelivers on VPR
  (trigger 2).
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
   This is also the trigger for the **dedicated DINOv2 escape
   valve**: if the VPR head on the shared text-capable trunk
   underdelivers on the bake-off's widened VPR Recall@K eval,
   trigger 2 un-parks the second resident DINOv2 trunk per the
   spine.
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

### Architecture — three heads on the shared trunk; DINOv2 is the escape valve

State-of-the-art spatial perception has converged on a single
frozen foundation trunk feeding multiple lightweight task
heads, rather than separate networks per task. **The default
trunk here is the shared text-capable trunk the bake-off
picks** (per the spine): the VPR head and the region head are
two more heads on the same frozen forward pass the validator
and the co-trained CLIP already consume — "one trunk, three
heads" means the *shared* trunk, not a new one. A **dedicated
DINOv2 trunk is the escape valve** (a second resident trunk),
un-parked only via trigger 2 when the shared trunk's visual
features can't meet the VPR bar.

```
RGB frame
   │
   ▼
SHARED text-capable trunk (FROZEN, bake-off's pick)  ── one forward pass per frame
   │   ├─ dense patch tokens  (N_patches × D)
   │   └─ CLS / pooled token  (D)
   │
   ├──────────────► PLACE-RECOGNITION HEAD  (warm-map case-1 booster)
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

The VPR head is the **warm-map case-1 booster** on the shared
trunk: per the spine, image-vs-text stays the cold-start safe
primary; cleaner `same_place` retrieval is what the VPR head
adds once the map has warmed.

**On the shared trunk (default).** The trunk is whatever the
bake-off picks (SigLIP-2-Base lead). Its visual features feed
both heads. The spine's load-bearing bet is that a text-capable
trunk's visual features are good enough for VPR; **MegaLoc** is
the backbone-flexible VPR-aggregation hedge that makes VPR on a
non-DINOv2 shared trunk viable (AnyLoc/SALAD VLAD vocabularies
are DINOv2-tuned, so a MegaLoc-style aggregation is the path
when the trunk is not DINOv2). Concrete head shapes:

- **Place-recognition head:** VLAD / SALAD-style aggregation
  over the shared trunk's dense tokens against a vocabulary of
  K=64–128 cluster centers built unsupervised on indoor
  frames → a `K×D`-d descriptor, PCA-whitened to ~512–4096-d.
  Zero-shot at the floor; a learned Sinkhorn aggregation
  (SALAD) or **MegaLoc** is the escalation when the trunk is
  not DINOv2. Output replaces the raw-CLIP embedding the v2
  loop-closure detector ranks on — same ANN-store +
  `same_place` contract, different descriptor.
- **Region head:** a 2-layer MLP (`D + |obj-vocab| → 256 →
  n_regions` logits) per node at the floor; a 2–3-layer GNN
  (GCN/GAT over the proximity + `same_place` edges) as the
  escalation that bakes in neighbor context. The object channel
  (`detected_objects` one-hot) depends on the **R1 detections
  column** — the harness per-frame detections the Tier-1 writer
  emits; without it the region head trains **vision-only** (the
  pooled token alone). Output per node: region id +
  open-vocab-compatible label embedding + confidence. Replaces
  v2's `partition_regions` HDBSCAN.

Both heads consume the **same** frozen shared-trunk forward
pass, so the per-frame cost is one trunk encode (already paid
by the validator / co-trained heads) + two cheap heads.

**The DINOv2 escape valve (trigger 2 only).** If the shared
trunk's VPR Recall@K underdelivers on the bake-off's widened
eval, this brief un-parks a **dedicated frozen DINOv2 ViT-S/14
trunk** as a *second* resident encoder, loaded behind
`STRAFER_SPATIAL_ENCODER`. Concrete shapes for that fallback:
DINOv2 ViT-S/14 at 224² — 256 patch tokens × 384-d + a 384-d
CLS token; AnyLoc/SALAD VLAD over the value-facet tokens
(`K×384`-d, PCA-whitened). This is the spine's escape valve and
costs real Jetson memory (two foundation trunks resident), so
it is a measured trade, not the baseline.

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
heads don't share parameters beyond the frozen features. The
freeze invariant holds whether the trunk is the shared
text-capable one (default) or the DINOv2 escape valve; it is
the spine's training-discipline invariant, and unfreezing the
shared trunk for one head would invalidate the ChromaDB index
and every other consumer's features at once.

Unfreezing (joint multi-task fine-tune of the trunk) is an
**escalation only**, gated on heads-only plateauing. If
attempted: multi-task loss = region cross-entropy + VPR
contrastive, with gradient surgery (PCGrad) or uncertainty
weighting, and a hard guard that VPR recall must not regress as
region accuracy climbs. Document the trade if pursued.

### Training scheme — three regimes, escalating

| Regime | Trunk | VPR head | Region head | When |
|---|---|---|---|---|
| **R0 — zero-shot floor** | frozen shared trunk | VLAD / MegaLoc aggregation (no training; vocab built on indoor frames) | — (v2's HDBSCAN still runs) | Quick check: does a frozen shared-trunk VPR descriptor alone clear trigger 2 without any training? (This is also the bake-off's VPR Recall@K signal.) |
| **R1 — heads-only (recommended v1)** | frozen shared trunk | VLAD / MegaLoc floor, or light SALAD aggregation fine-tune | learned MLP/GNN region head, supervised | The default. Two heads train independently on the frozen shared trunk; no multi-task balancing. |
| **R2 — co-fine-tune (escalation)** | unfrozen | SALAD | GNN | Only if R1 plateaus. Multi-task loss + gradient surgery + the VPR-recall guard above. |

**Data + supervision:**

- **Region head:** harness trajectories; ground-truth region per
  node from
  [`scene_labels.get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py)
  over Infinigen `room_idx`, **re-grouped for open-plan** (zones
  with no demarcating wall that the planner should treat as one
  region get one label; the eval-harness open-plan adversarial
  defines the grouping). Loss: cross-entropy + an optional
  intra-region contrastive term. The head's **object-channel
  input** (`detected_objects` one-hot) depends on the **R1
  detections column** — the harness per-frame detections the
  Tier-1 writer emits. If that column is absent, the region head
  trains **vision-only** on the pooled token alone; the object
  channel is additive, not required for a v1.
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

**Latency:** on the default shared-trunk path, the trunk encode
is **already paid** by the validator / co-trained heads, so this
brief adds only VLAD/MegaLoc + region head — both cheap. Report
median + p95 per frame for the two heads. On the DINOv2
escape-valve path, a *second* trunk encode is added (DINOv2
ViT-S/14 is ~tens of ms at FP16) and is the resident-memory cost
trigger 2 weighs. Budget either way: under the `BackgroundMapper`
2 s poll interval with comfortable margin.

### Tie-in to state-of-the-art

- **AnyLoc** (Keetha et al., RA-L 2023 / ICRA 2024,
  [arXiv:2308.00688](https://arxiv.org/abs/2308.00688)) —
  frozen DINOv2 facet features + VLAD → SOTA VPR **zero-shot**.
  The R0 floor and the proof that the trunk already carries
  place-discriminative features.
- **SALAD** (Izquierdo & Civera, CVPR 2024) — DINOv2 +
  optimal-transport (Sinkhorn) aggregation, lightly fine-tuned.
  The R1/R2 VPR-head upgrade.
- **DINOv2** (Oquab et al. 2023) — the trunk AnyLoc/SALAD build
  on; here it is the **escape-valve** second trunk (trigger 2),
  not the default. The default trunk is the bake-off's
  text-capable pick.
- **ConceptGraphs** ([ICRA 2024](https://arxiv.org/abs/2309.16650)),
  **HOV-SG** ([RSS 2024](https://arxiv.org/abs/2403.17846)) —
  one foundation trunk feeding semantic region/object features
  AND cross-view merging; the precedent for "two tasks, one
  trunk."
- **MegaLoc** (2024) — foundation-model-driven, **backbone-flexible**
  VPR aggregation. This is the hedge that makes VPR viable on a
  **non-DINOv2 shared trunk**: AnyLoc/SALAD vocabularies are
  DINOv2-tuned, so MegaLoc-style aggregation is the path when the
  shared trunk is SigLIP-2 / MobileCLIP-2. Also the no-fine-tune
  alternative to a SALAD fine-tune if R1's VPR head underdelivers.

### Co-tenancy with the implicit memory map

[`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md)
(the sub-symbolic primitive for the validator / VLA) shares
**the same frozen shared trunk** the bake-off picks — "one
trunk, three heads" total (VPR, region, memory-bank
cross-attention), all on the shared text-capable trunk by
default per the spine's consumer table. If both this brief and
the implicit memory map ship, coordinate the trunk load so it's
resident once. The
[`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
brief is the single trunk-choice decision point for all three.
(Only on the DINOv2 escape-valve path is a *second* trunk
resident — and trigger 2 weighs that memory cost explicitly.)

## Approach — staged

Ship only the head(s) the firing trigger demands; build the
trunk + eval once.

1. **Trunk wiring.** *Default:* expose the **shared trunk's**
   dense + pooled features (the bake-off's text-capable pick,
   already loaded for the validator / co-trained heads) to both
   the VPR head and the region head — no new trunk, no second
   ONNX session. *Escape valve (trigger 2 only):* if the shared
   trunk's VPR underdelivers, load a **dedicated frozen DINOv2
   ONNX session** in `clip_encoder.py` behind
   `STRAFER_SPATIAL_ENCODER` as a second resident trunk. The
   shared text-capable trunk stays the system backbone for room
   labeling / text queries / validator cosine in both cases; the
   DINOv2 second trunk is additive and memory-weighed.
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
- [ ] **Frozen trunk, two heads.** By default the VPR head +
      region head consume the **shared text-capable trunk** the
      [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
      picks — the same frozen forward pass the validator already
      runs, no second trunk. The dedicated DINOv2 trunk is loaded
      **only** if trigger 2 fires (shared-trunk VPR below the
      bake-off bar); when loaded it is frozen and resident
      alongside the shared trunk. Trunk choice coordinated with
      the bake-off and consistent with
      [`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md).
- [ ] **Place-recognition head.** VLAD (R0 floor) or SALAD
      (R1) aggregation → global descriptor → the existing
      ChromaDB ANN + `semantic-graph-loop-closure` `same_place`
      protocol. VPR metrics reported per the table; beats v2
      raw-CLIP by ≥ 1 CI-width on Recall@1 and on multi-bedroom
      false-merge rate.
- [ ] **Region head.** Learned MLP/GNN per-node partition,
      supervised on the harness corpus with open-plan-regrouped
      labels. The object channel (`detected_objects` one-hot)
      consumes the **R1 detections column**; if that column is
      absent the head trains vision-only on the pooled token and
      the PR notes it. Replaces v2's `partition_regions` behind
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
- CLIP encoder co-tenancy pattern (the escape-valve path loads
  a second frozen ONNX session here; the default path reuses the
  shared trunk's features):
  [`semantic_map/clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py).
- Training corpus + held-out seeds + adversarial scenes:
  [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md).
- Ground-truth region labels:
  [`scene_labels.get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py).
- Reference architectures:
  - AnyLoc (arXiv:2308.00688) — frozen-DINOv2 + VLAD, zero-shot VPR.
  - SALAD (CVPR 2024) — DINOv2 + Sinkhorn aggregation.
  - DINOv2 (Oquab et al. 2023) — the escape-valve second trunk, not the default.
  - ConceptGraphs (arXiv:2309.16650), HOV-SG (arXiv:2403.17846) — one trunk, semantic + merge.
  - MegaLoc (2024) — backbone-flexible VPR aggregation; the hedge for VPR on a non-DINOv2 shared trunk.

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
  is the validator/VLA primitive; it shares the same shared
  trunk but has a different head and consumer.
- **Choosing the shared trunk.** Trunk selection stays at
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md);
  this brief consumes the text-capable trunk it picks and adds
  two heads on it. The only trunk this brief may *add* is the
  dedicated DINOv2 escape valve, and only when trigger 2 fires.
- **Multi-floor.** Strafer is single-story.
- **Real-robot training data.** Sim-only training corpus +
  domain randomisation for transfer; real-robot fine-tune is a
  future brief if the sim→real gap persists past the bar.
