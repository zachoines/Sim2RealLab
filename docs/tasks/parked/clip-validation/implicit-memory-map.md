# Implicit memory map — retrieval-augmented embedding primitive

**Type:** new feature / research (filed-on-trigger)
**Owner:** DGX agent (cross-attention module + RAG-aware
training + ONNX export; minimal Jetson-side glue to load the
module into the inference path)
**Priority:** P3 — filed-on-trigger. The shared sub-symbolic
primitive both the CLIP cascade validator and the v2 VLA
consume. Build it when the first consumer commits to it (see
"Trigger detail").
**Estimate:** L (~1–1.5 wk; cross-attention module + RAG-aware
training loop + cold-deployment augmentation + ONNX export +
ablation against the bare-encoder baseline)
**Branch:** task/implicit-memory-map

**Pickup gate:** Blocked-on-deps until
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
ships (provides the v1 cascade baseline + the ChromaDB-backed
`SemanticMapManager` retrieval primitive this layer wraps).
Un-park by `git mv` per
[`README.md`'s Directory layout](../../README.md#directory-layout).

## Story

As an **operator who wants the project's sub-symbolic scene
representation to be one shared primitive rather than two
parallel implementations (one buried in the CLIP validator,
one re-derived for the VLA)**, I want **a standalone
retrieval-augmented embedding module — a memory bank of past
CLIP/backbone embeddings indexed by pose, consumed via
cross-attention on the live query — with RAG-aware training
and cold-deployment robustness baked in**, so that **the CLIP
cascade validator and the v2 VLA both consume the same
memory-augmented embedding, the cold-start collapse problem is
solved once, and the implicit / sub-symbolic half of the
project's map architecture has a single home instead of living
as Step B of a validator brief**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md) — the symbolic-vs-sub-symbolic split. This brief is the **sub-symbolic** primitive; the symbolic counterpart is [`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md).
- [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
  — the v1 cascade + the ChromaDB retrieval primitive this
  module wraps. Consumer #1.
- [`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md)
  — consumes this primitive in its Step B (retrieval-augmented
  inference). Formerly defined the cross-attention layer
  inline; now references this brief.
- [`vla-v2-map-conditioning`](../experimental/vla-v2-map-conditioning.md)
  — consumes this primitive in its Option B (cross-attention
  over a memory bank). Consumer #2.

## Trigger detail — when to un-park

This is shared infrastructure; build it when the first
consumer is ready to commit, not speculatively. Un-park when
**either**:

1. **[`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md)
   reaches its Step B** (Step A co-training shipped; the
   cascade wants retrieval-augmented inference). The validator
   is consumer #1 and the most likely first mover.
2. **[`vla-v2-map-conditioning`](../experimental/vla-v2-map-conditioning.md)
   picks Option B** (the ablation shows cross-attention over a
   memory bank beats text-serialization / no-consumption on
   cross-room mission success). The VLA is consumer #2.

Whichever consumer fires first un-parks this brief and builds
the primitive; the second consumer then reuses it. If neither
fires — the cascade clears its bar without retrieval AND the
VLA picks Option A or C — this brief stays parked or is
deleted.

## Context

### Why a standalone primitive, not a validator feature

The retrieval-augmented embedding layer was originally Step B
of [`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md),
framed as a CLIP-validator improvement. But it is
architecturally **the project's implicit-mapping primitive** —
the sub-symbolic counterpart to the symbolic
[`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md).
It has two distinct consumers with different objectives, which
is the textbook signal to factor it out:

| Consumer | Objective | What it does with the augmented embedding |
|---|---|---|
| CLIP cascade validator | Binary off-course detection | Cosine vs. mission-text / semantic-map nodes; fires the arbiter |
| v2 VLA | Action prediction | Conditions the policy's forward pass on scene memory |

Same primitive (memory bank + cross-attention), same training
pattern (RAG-aware), same cold-start problem — built once,
consumed twice.

### The primitive

```
live frame → backbone visual tower → query embedding (R^512)
  → ChromaDB retrieval: top-K past observations near current
                        pose and high-cosine to the query
  → cross-attention(query, retrieved) → augmented embedding (R^512)
  → consumer head (validator cosine OR VLA policy)
```

**Cross-attention module.** Standard transformer cross-attention
block: query ∈ R^512 (live frame embedding); keys + values ∈
R^(K × 512) from retrieved past observations. ~5–10 M trainable
parameters at hidden_dim=256, 4 heads. Output replaces the bare
query in the downstream consumer.

**Retrieval.** Reuses
[`SemanticMapManager.query_by_embedding`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
for top-K = 8 (default; tunable) past observations from
ChromaDB. Each contributes its embedding + metadata (pose,
timestamp, source). Same index at train and inference time —
no train-test mismatch on the memory side.

### RAG-aware training + the cold-deployment problem

The cross-attention layer trains **with retrieval in the
forward pass**, not bolted on at inference. Headline finding
from the lineage (Re-CLIP, MemoryBank-CLIP, Atlas, RA-DIT):
training-with-retrieval beats inference-only retrieval.

**Cold-deployment collapse + mitigation.** The training-time
pool is the full harness corpus (the
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
episode set — same `held_out_seeds` protocol, minus the
held-out trajectory); the deployment-time pool is whatever the
`SemanticMapManager` has accumulated on this house — empty on
a fresh deployment, growing to ~50–200 nodes. A layer trained
only against a dense pool collapses at K=0 (no keys) or K=1
(degenerate softmax). Mitigation, from
[Atlas (Izacard et al., 2022)](https://arxiv.org/abs/2208.03299)
and [RA-DIT (Lin et al., ICLR 2024)](https://arxiv.org/abs/2310.01352):
**vary the retrieved-pool size during training** — sample
`K_train ∈ {0, 1, 2, 4, 8}` uniformly per batch, truncate (or
pad with a learned "no-retrieval" token at K=0). The model
learns to function across the full pool-size range; the
inference path picks `K` per the current node count.

### Empirical precedent

| Source | Contribution |
|---|---|
| **Re-CLIP / REACT** (Liu et al., NeurIPS 2024) | Retrieval-augmented CLIP trained to consume retrieved context; training-with-retrieval beats inference-only |
| **MemoryBank-CLIP** (2024) | Learnable memory bank trained jointly with the encoder |
| **Atlas** (Izacard et al., 2022), **RA-DIT** (Lin et al., ICLR 2024) | Retrieval-aware training; the `K_train` pool-size augmentation |
| **OpenScene** ([CVPR 2023](https://arxiv.org/abs/2211.15654)), **3D-VLA** (CVPR 2024) | The implicit / feature-bank scene-representation lineage this primitive sits in |

## Approach

1. **Module** at
   `source/strafer_autonomy/strafer_autonomy/semantic_map/retrieval_augmented_encoder.py`
   implementing the cross-attention(query, retrieved) →
   augmented-embedding forward pass. Backbone-agnostic (works
   on whichever embedding tower is loaded).
2. **RAG-aware training script** at
   `source/strafer_lab/scripts/train_implicit_memory_map.py`.
   `K_train ∈ {0, 1, 2, 4, 8}` augmentation; held-out-trajectory
   holdout protocol to prevent leakage.
3. **ONNX export** to `~/.strafer/models/memory_map.onnx`,
   consumable by both the validator path and the VLA path.
4. **Consumer interfaces** documented for both:
   `augment(query_emb, retrieved) -> augmented_emb` is the
   single entry point.

## Acceptance criteria

- [ ] **Standalone module.**
      `semantic_map/retrieval_augmented_encoder.py` implements
      the cross-attention forward pass; backbone-agnostic;
      `augment(query, retrieved)` is the single consumer
      entry point.
- [ ] **RAG-aware training with `K_train` augmentation.** The
      training script samples `K_train ∈ {0, 1, 2, 4, 8}`
      uniformly per batch; a learned no-retrieval token
      handles K=0. Holdout protocol excludes the training
      sample's own trajectory from its retrieval pool.
- [ ] **Cold-deployment graceful degradation.** With the
      retrieval index forcibly empty, the augmented embedding
      degrades to within one CI-width of the bare-query
      baseline on whatever metric the first consumer uses
      (ROC-AUC for the validator; mission success for the
      VLA). Below that floor, the module ships disabled until
      re-tuned.
- [ ] **ONNX export.** `~/.strafer/models/memory_map.onnx`
      loadable by both consumer paths. Latency budget: ≤ 10 ms
      cross-attention + ≤ 5 ms ChromaDB top-K on the Jetson;
      reported median + p95.
- [ ] **Two-consumer interface documented.** README note (in
      [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md))
      covering `augment(query, retrieved)` and how the
      validator (cosine head) and the VLA (policy
      conditioning) each consume it.
- [ ] **No regression** in the bare-encoder path. With the
      module disabled, validator + any VLA baseline behave
      identically.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same commit.

## Investigation pointers

- The retrieval primitive this wraps:
  [`SemanticMapManager.query_by_embedding`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py).
- Consumer #1 (validator): the Step B section of
  [`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md).
- Consumer #2 (VLA): Option B of
  [`vla-v2-map-conditioning`](../experimental/vla-v2-map-conditioning.md).
- RAG-aware training reference: Re-CLIP / REACT (NeurIPS 2024),
  Atlas (arXiv:2208.03299), RA-DIT (arXiv:2310.01352).
- Cross-attention block: standard transformer cross-attention,
  ~5–10 M params at hidden_dim=256, 4 heads.

## Out of scope

- **Co-training the backbone.** The contrastive fine-tune
  (Step A) lives at
  [`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md);
  this brief is the retrieval layer only. It works on whatever
  backbone is loaded (v1 OpenCLIP, or the bakeoff's pick).
- **The symbolic region partition.** That's the explicit-layer
  counterpart at
  [`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md).
  Shared infrastructure (ChromaDB, backbone) but different
  output (dense augmented embedding vs. discrete `RoomEntry`).
- **VLA training.** The VLA itself lives at
  [`vla-v2-architecture`](../experimental/vla-v2-architecture.md);
  this brief produces a conditioning input it consumes.
- **Online memory-bank updates during deployment.** The index
  grows via the existing `SemanticMapManager` add path; this
  brief does not add online learning of the cross-attention
  weights.
- **Real-robot retrieval index.** Sim-only training + eval;
  real-robot transfer is a future brief.
