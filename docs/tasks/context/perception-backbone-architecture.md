# Perception backbone architecture — one frozen text-capable trunk, per-consumer heads

The spine every perception-learning brief references instead of
re-deriving. It records the **settled** decision (operator +
orchestrator, after the epic inter-consistency audit) for how the
project's learned perception components share a single visual
backbone on the Jetson, why, and which brief owns which piece.

For the symbolic-vs-sub-symbolic split of the map work, see
[`multi-room-architecture.md`](multi-room-architecture.md). For the
mid-mission deviation contract, see
[`../parked/clip-validation/clip-multi-room-validator-remeasure.md`](../parked/clip-validation/clip-multi-room-validator-remeasure.md)
and `docs/MISSION_VALIDATION_ARCHITECTURE.md`. This module is the
backbone/trunk layer those sit on top of.

---

## The decision

**One modern, *text-capable*, frozen trunk (SigLIP-2-Base lead),
with per-consumer LoRA adapters + light heads.** One forward pass
per frame feeds every learned perception consumer. A dedicated
DINOv2 trunk is an **escape valve only**, un-parked if the shared
trunk's visual features prove insufficient for visual place
recognition (VPR).

This reverses the prior default (OpenCLIP ViT-B/32 by inheritance)
and reorders the roadmap: the backbone choice is made **first**,
before the v1 cascade ships. It is affordable because the project
is greenfield on weights — there is no exported ONNX on the DGX, the
CLIP fine-tune has never run, and `finetune_clip.py` is retired by
[`../active/harness/harness-architecture.md`](../active/harness/harness-architecture.md)
(verified: `MISSION_VALIDATION_ARCHITECTURE.md §2.6`). There is no
sunk cost in ViT-B/32 to migrate off.

---

## Why one trunk — the hinge

The system has two representational needs, and no 2021-era model
serves both:

| Need | Used by | Requires |
|---|---|---|
| **(A) text-aligned image↔text** | validator case-2 (image-vs-text), `query_room_by_text`, zero-shot room labeling, co-trained CLIP heads 1 & 3 | a **text tower** |
| **(B) pure visual place / region features** | VPR (`same_place` loop closure), the learned region head | DINOv2-family; vision-only suffices |

Case-1 (place) is served by both families; case-2 (instance/text)
needs (A); VPR is dominated by (B). A single shared backbone is
possible **only if a text-capable model's visual features are also
good enough for VPR/region** — the load-bearing empirical bet (see
Risks). A vision-only trunk (DINOv2/DINOv3) is therefore disqualified
as the *shared* default: it would gut case-2.

The Jetson Orin Nano (8 GB unified, already running Nav2 + RTAB-Map +
executor) cannot comfortably hold two foundation trunks resident.
The learned spatial encoder runs **Jetson-side at inference** (it
loads in `clip_encoder.py` behind `STRAFER_SPATIAL_ENCODER`), so a
second trunk is real memory pressure, not a DGX-only training cost.
That is why "share one trunk" is a hard requirement, not a
preference.

---

## The consumers — one forward pass, many heads

```
RGB frame
   │
   ▼
FROZEN shared trunk (SigLIP-2-B)  ── one encode per frame
   │   ├─ image embedding / dense + pooled tokens
   │   └─ text tower (text-capable)
   │
   ├─► validator case-1 (image-vs-image place rec) + case-2 (image-vs-text)   [validator LoRA/head]
   ├─► region partition  (pooled token + detected_objects → region)           [region head]
   ├─► VPR / same_place descriptor  (VLAD/SALAD over dense tokens)             [VPR head]
   └─► Step B memory cross-attention (query + retrieved → augmented embedding) [memory head]
```

| Consumer | Brief | Head on the shared trunk |
|---|---|---|
| Mid-mission validator (case-1 + case-2) | [`../active/clip-validation/validator-evaluation.md`](../active/clip-validation/validator-evaluation.md) | image-vs-text (primary) + image-vs-image; LoRA via cotrained |
| Co-trained CLIP (representation shaping) | [`../parked/clip-validation/cotrained-retrieval-augmented.md`](../parked/clip-validation/cotrained-retrieval-augmented.md) | LoRA + per-case projection heads (case-agnostic tower) |
| Implicit memory map (Step B) | [`../parked/clip-validation/implicit-memory-map.md`](../parked/clip-validation/implicit-memory-map.md) | cross-attention over retrieved neighbors |
| Region partition (learned variant) | [`../parked/multi-room/learned-spatial-encoder.md`](../parked/multi-room/learned-spatial-encoder.md) | region head |
| VPR / loop closure (learned variant) | [`../parked/multi-room/learned-spatial-encoder.md`](../parked/multi-room/learned-spatial-encoder.md) | VPR head |

The unsupervised v2 mechanisms
([`../active/multi-room/semantic-region-partition.md`](../active/multi-room/semantic-region-partition.md),
`semantic-graph-loop-closure`) run on the same frozen trunk's
features without training; the learned heads above are their escape
valve.

---

## Training discipline (the invariant that keeps it coherent)

**Freeze the base trunk; train per-consumer LoRA + light heads.**
"Trained alongside" means **independent heads on shared frozen
features** (trained in parallel, no shared trainable parameters
beyond the frozen trunk), **not** a joint multi-task fine-tune of the
trunk. Joint trunk fine-tuning is disallowed by default because the
objectives conflict — VPR wants viewpoint-*invariant* features, the
region head wants region-*discriminative* features, the validator
wants contrastive case-1/case-2 separation — and unfreezing the
shared trunk would invalidate the ChromaDB index and every other
head's features at once.

The reconciliation with "co-trained CLIP wants to shape the validator
tower": `cotrained` already defaults to **LoRA adapters with the base
towers frozen**. So the validator gets its discrimination via LoRA +
per-case projection heads *on top of* the frozen base, while VPR /
region / Step-B consume the same frozen base. Anyone who unfreezes
for one head breaks the others — LoRA is the escape hatch and this is
a stated invariant.

**Contrastive loss form: sigmoid, not InfoNCE.** SigLIP-2 is
pretrained with the **sigmoid (pairwise) image-text loss**, not
softmax/InfoNCE. So the *image-text* heads (the validator's case-2
and `cotrained` heads 1 & 3) train with the **sigmoid loss** — it
keeps the LoRA-adapted embeddings in the pretrained geometry and is
**batch-size-robust** (no in-batch all-gather of negatives), which
matters for the small harness corpus where InfoNCE's few-negatives
regime degrades. Image-image objectives (the same-region head, VPR)
are unaffected.

---

## Deployment context — single-home, long-lived

The deployment model is **one robot living in one home for weeks**.
The semantic map is persisted (`~/.strafer/semantic_map/`) and warms
up within a mission (`BackgroundMapper` populates it as the robot
moves) and across missions. So the **warm map is the steady state;
cold-start is the first hour, a transient.** This elevates warm-map
techniques (VPR head, Step B memory) from "marginal / parked-maybe"
to **planned consumers**.

It does **not** demote the cold-start path: **image-vs-text stays the
safe primary signal** (map-free; the only thing that works on a
first visit), and image-vs-image / VPR / Step-B layer on as **warm
boosters**, never replacing it. On a cold/empty map every
retrieval-based consumer degrades to its no-retrieval baseline by
construction.

---

## The escape valve — a dedicated DINOv2 second trunk

If the widened bake-off (below) shows the shared trunk's visual
features cannot meet the VPR bar,
[`../parked/multi-room/learned-spatial-encoder.md`](../parked/multi-room/learned-spatial-encoder.md)
un-parks a **dedicated, frozen DINOv2 trunk** as a second resident
encoder (its trigger 2). At that point the second-trunk memory cost
is a measured, justified trade — not a blind default. The VPR head
runs on the shared trunk by default; the DINOv2 trunk is the
fallback, not the baseline.

---

## The three risks (named, not buried)

1. **The VPR bet.** AnyLoc/SALAD VPR is DINOv2-specific; VPR on
   SigLIP-2 features is unproven. What makes the bet plausible
   (rather than a long shot): SigLIP-2's pretraining adds
   **self-distillation + masked prediction (SILC/TIPS-style)** — the
   same DINO-family objective that gives DINOv2 the dense-feature
   quality AnyLoc exploits — so its visual features carry more
   place-discriminative signal than CLIP/SigLIP-1. Hedge: **MegaLoc**
   is a more backbone-flexible VPR aggregation than AnyLoc's
   DINOv2-specific facets. The widened bake-off still measures it;
   treat the shared-trunk-serves-VPR claim as a *principle with a
   measured gate*, not a settled fact.
2. **Freeze discipline.** The whole architecture holds only with a
   frozen base. Unfreezing for any single head breaks the rest (and
   the persisted index). LoRA is the only sanctioned adaptation path.
3. **Roadmap reorder.** The backbone decision moves *before*
   `validator-evaluation` v1, reversing the current "v1 on existing
   ONNX → bake-off later" dependency. Greenfield makes it affordable.

---

## Backbone-dim hygiene (one trunk, one dim, decided early)

Because the trunk is chosen once, the embedding dimension is fixed
once (SigLIP-2-B = 768; not 512). The two live 512-dim sentinels —
`clip_encoder.py:17` (`_EMBEDDING_DIM = 512`) and `manager.py:590`
(`np.zeros(512)`) — must be made backbone-dim-aware **inside the
backbone-switch PR**, and the persisted ChromaDB store at
`~/.strafer/semantic_map/chroma` (dim-pinned at first insert) needs a
**re-index / fresh-store step** on any backend change. Owned by
[`../parked/clip-validation/backbone-bakeoff.md`](../parked/clip-validation/backbone-bakeoff.md).

---

## Roadmap / sequencing

1. **`backbone-bakeoff` (narrowed + widened) runs first** — picks the
   shared trunk on the full job set (case-1, case-2, `query_room_by_text`,
   VPR Recall@K, region V-measure). Candidates: SigLIP-2-B (lead),
   MobileCLIP-2-S; OpenCLIP ViT-B/32 as baseline-to-beat; DINOv3-S as
   the vision half of a hybrid only if single-tower VPR fails.
2. **`validator-evaluation` v1** ships the cascade on the chosen
   trunk (pickup gate now *after* the bake-off).
3. Warm-map consumers (`cotrained` Step A, then Step B / VPR head) are
   sequenced behind v1 on measured warm-leg residual.

---

## Maintenance contract

Same rule as [`multi-room-architecture.md`](multi-room-architecture.md)
and [`../BOARD.md`](../BOARD.md): a brief that adds, removes, or
re-homes a consumer of the shared trunk updates the consumer table
here in the same PR. A change to the **decision** itself (the trunk
family, the freeze invariant, the escape-valve trigger) is an
architecture change — it revises this module and is reviewed like any
other PR. If a brief touches `clip_encoder.py`'s backbone loading,
the dim hygiene section is its checklist.
