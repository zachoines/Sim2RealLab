# Backbone bake-off: SigLIP-2 (lead) vs. MobileCLIP-2 vs. OpenCLIP baseline — picks the one shared text-capable trunk

**Type:** investigation / new feature
**Owner:** DGX agent (ONNX exports + offline eval are all
DGX-side; the runtime swap behind `STRAFER_CLIP_BACKEND` is a
small Jetson-lane edit, cross-lane like
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md))
**Priority:** P2 (the project should not re-commit to a
2021-era backbone for the case-1 + case-2 tripwire and the
downstream cascade improvements without a defensible
alternative-considered comparison; this brief picks the one
frozen shared trunk per the spine and now runs *before*
`validator-evaluation` v1 ships on it)
**Estimate:** M (~half-week to a week; ONNX export of the three
candidates + the shared eval script run + a short write-up)
**Branch:** task/backbone-bakeoff

This brief implements the trunk-choice decision recorded in
[`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md)
(the frozen spine). Read the spine first; this brief is the
measurement that turns its lead pick into a numbers-backed
choice, and it must stay consistent with the spine's decision,
consumer table, and dim-hygiene section.

## Story

As an **operator about to ship the v1 CLIP cascade and then
spend XL training cycles on
[`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md),
[`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md),
or
[`vla-v2-architecture`](../experimental/vla-v2-architecture.md)**,
I want **a documented head-to-head comparison of the
candidate shared trunks (SigLIP-2-Base as the lead,
MobileCLIP-2-S as the lower-latency option, OpenCLIP
ViT-B/32 as the baseline-to-beat) scored across the *full*
consumer job set — case-1 + case-2 ROC-AUC, VPR Recall@K,
and region V-measure — not just the validator's eval set**,
so that **the v1 cascade ships on the trunk that serves ALL
consumers per the spine, with the alternative-considered
trail in writing, rather than inheriting ViT-B/32 by default
because that's what `finetune_clip.py` happens to export
today**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
§3.1 (better CLIP usage; the alternatives this brief actually
measures).

Sibling briefs:
- [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md) —
  **now downstream of this brief, not upstream.** v1 ships its
  cascade *on the trunk this bake-off picks*, so this brief's
  pickup gate is now *before* v1's (see "Pickup gate / reorder"
  below). The eval scaffolding — statistics framework, held-out
  episode set, bias-mitigation protocol — must be available to
  the bake-off first; whether it splits out of
  `validator-evaluation` or this brief stands it up is an open
  coordination point for the orchestrator (flagged below).
- [`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md) —
  consumes the chosen trunk for the fine-tune init. Same epic,
  same eval framework.
- [`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md) —
  consumes the chosen trunk for its VPR + region heads. Its
  VPR Recall@K and region V-measure eval jobs are folded into
  this bake-off's per-candidate scoring so the trunk is picked
  on the full consumer job set, not the validator's alone.
- [`vla-v2-architecture`](../experimental/vla-v2-architecture.md) —
  the VLA backbone (OpenVLA's DINOv2+SigLIP fusion, π0's tower,
  Octo's encoder, ...) is a constraint on this brief's choice:
  the cascade and the v2 stack should ideally share a visual
  tower so co-tenancy memory + ablation comparability are
  preserved.

## Context

### Pickup gate / reorder — this brief now runs *before* v1

**This brief picks the one shared trunk that
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
v1 ships on, so it runs BEFORE v1**, per the spine's roadmap.
This **inverts** the relationship the older revision of this
brief described ("triggered when `validator-evaluation` ships,
reuses its eval script"): the dependency now points the other
way.

- **New prerequisite:** harness episodes (Tier 1) + the shared
  eval scaffolding (statistics framework, held-out seeds,
  bias-mitigation protocol). It no longer waits on v1.
- **Open coordination point for the orchestrator.** The eval
  scaffolding must exist for the bake-off *before* v1 ships.
  Whether it **splits out of `validator-evaluation`** into a
  shared substrate both briefs consume, or **the bake-off
  stands it up** and v1 inherits it, is an explicit
  coordination decision left to the orchestrator — this brief
  does not unilaterally pick.
- **Why the reorder is affordable: greenfield on weights.**
  There is no exported ONNX on the DGX, the CLIP fine-tune has
  never run, and `finetune_clip.py` is being retired. There is
  no sunk cost in ViT-B/32 to migrate off, so choosing the
  trunk first costs nothing that an after-the-fact swap would
  have saved. The spine records this under "The decision".

### Why this brief exists

The CLIP fine-tune target in
[`finetune_clip.py`](../../../../source/strafer_lab/scripts/retired/finetune_clip.py)
starts from OpenCLIP ViT-B/32 + `laion2b_s34b_b79k` weights —
the same OpenAI-style 2021-era backbone the field has moved
past on essentially every axis that matters for case-1 and
case-2:

| Axis | OpenCLIP ViT-B/32 (current) | What's available in 2025 |
|---|---|---|
| **Pretrain objective** | Symmetric InfoNCE on image-text pairs | SigLIP-2 (sigmoid loss + caption pretraining + self-distillation, outperforms SigLIP-1 at every scale per [Tschannen et al., 2025](https://arxiv.org/abs/2502.14786)) |
| **Visual-only feature quality** | ViT-B/32 patches; lossy on indoor scenes | SigLIP-2 / MobileCLIP-2 carry stronger visual features at this budget; DINOv3-S/14 is the strongest pure-vision option but is text-incapable, so it is **not** a standalone shared-trunk candidate (it would gut case 2) — it is held in reserve as the vision half of a hybrid only if single-tower VPR fails the bar |
| **Jetson-class latency** | ~150 – 200 ms at FP16 (CUDA EP); plausibly 80 – 120 ms at TRT-EP after a tuning pass | MobileCLIP-2-S4 matches SigLIP-SO400M/14 at **2× fewer parameters** and runs at 2.5× lower latency than DFN ViT-L/14 on iPhone 12 Pro Max ([Faghri et al., TMLR 2025](https://arxiv.org/abs/2508.20691)) — strict superset of "fits Orin Nano" |
| **Patch count at 224²** | 49 patches (ViT-B/32) | 256 (DINOv3-S/14), 196 (SigLIP-2-B/16), comparable (MobileCLIP-2 variants) — directly addresses the §2.2 patch-count limitation |
| **VLA-backbone overlap** | None (no shipping wheeled VLA uses ViT-B/32) | OpenVLA fuses DINOv2 + SigLIP visual features; π0 / NaVid / NaVILA use SigLIP-family encoders. Sharing a tower with the v2 VLA is structurally cleaner. |

[`MISSION_VALIDATION_ARCHITECTURE.md` §3.1](../../../MISSION_VALIDATION_ARCHITECTURE.md#31-better-clip-usage)
names these candidates in prose. This brief is the measurement
that turns prose into a chosen backbone with numbers attached.

### What this brief produces

Five artifacts in one PR:

1. **ONNX exports** of the three candidate trunks under
   `~/.strafer/models/backbones/<name>/{visual,text}.onnx` —
   all three are text-capable, so each ships a text tower. One
   sub-directory per backbone. (A vision-only DINOv3-S export is
   produced only if the hybrid fallback is triggered; see
   candidates below.)
2. **A runtime backbone switch** behind a
   `STRAFER_CLIP_BACKEND` env var consumed by
   [`clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py).
   The current `clip_visual.onnx` / `clip_text.onnx` filenames
   become the `openclip` backend; new backends live under their
   own subdirs. Default stays `openclip` until this brief
   declares a swap. **The switch PR also lands the backbone-dim
   hygiene work** (see "Backbone-dim hygiene" below) — the
   512-dim sentinels become backbone-dim-aware and the persisted
   ChromaDB store is re-indexed / freshened on a backend change.
   These are not separable: switching the backbone without them
   silently corrupts embeddings and the dim-pinned store.
3. **A per-candidate run of the shared eval scaffolding** (the
   statistics framework + held-out episode set; `--backbone`
   selects the trunk) emitting one
   `data/transit_monitor_eval/<backbone>/report.json` per
   candidate, scored on the **full consumer job set**: case-1 /
   case-2 ROC-AUC (the validator), VPR Recall@K, and region
   V-measure (the
   [`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md)
   jobs). Same statistics framework, same scene seeds, same
   case-1 / case-2 split — only the backbone changes. "Works
   with all approaches" is measured here, not assumed: the
   chosen trunk must serve every consumer in the spine's
   consumer table, not just the validator.
4. **A bakeoff write-up** at
   `docs/artifacts/backbone_bakeoff/<run_id>/report.md`
   summarizing per-backbone case-1 / case-2 ROC-AUC + 95 % CI,
   VPR Recall@K, region V-measure, latency on Orin Nano FP16,
   VRAM footprint, and a recommendation.
5. **The default backbone flip** in `executor/main.py` and
   `finetune_clip.py`'s default init, plus a §4.6 addendum
   to
   [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
   recording the chosen backbone and the comparison table. If
   the choice changes any fact in the spine's consumer table,
   dim-hygiene, or roadmap sections, update
   [`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md)
   in the same PR.

### Candidates and rationale

The shared trunk **must be text-capable** — vision-only guts the
validator's case-2 (image-vs-text) and `query_room_by_text`, per
the spine's "Why one trunk — the hinge". So the standalone
candidates are narrowed to three text-capable trunks, and the
strongest pure-vision model is demoted to a conditional hybrid
half:

| Backbone | Role | Why measure | Pretrain corpus | Has text tower? |
|---|---|---|---|---|
| **SigLIP-2-Base/16** | **lead / default** | 2025 SOTA for zero-shot image-text alignment at this parameter budget; the spine's lead pick; serves case-2, `query_room_by_text`, and the co-trained heads | WebLI + multilingual + self-distillation | Yes |
| **MobileCLIP-2-S** | lower-latency option | Explicit Jetson / mobile target; the lower-latency contender that may still hit the AUC bar | Reinforced training (synthetic captions + ensemble teachers) | Yes |
| **OpenCLIP ViT-B/32 (current)** | baseline-to-beat | Incumbent; what `finetune_clip.py` exports today. The bar each candidate must clear; not a candidate to ship forward on its own merits | LAION-2B (`laion2b_s34b_b79k`) | Yes |

**DINOv3-S is demoted from a standalone candidate.** It is
text-incapable, so it cannot be the shared trunk without gutting
case-2. It is held in reserve as **the vision half of a hybrid**
(DINOv3-S for the case-1 / VPR image-vs-image signal, paired with
the winning text-capable trunk for case-2), and is exported and
evaluated **only if** a single text-capable trunk fails the VPR
bar — i.e. the spine's escape valve / `learned-spatial-encoder`
trigger 2. The hybrid is a fallback, not a default candidate.

### Input resolution — evaluate the NaFlex variant

SigLIP-2 ships **NaFlex** variants that ingest **native aspect ratio
at a variable sequence length** instead of forcing a square
center-crop. The perception camera is 640×360 (16:9), and the live
`clip_encoder` center-crops to 224², dropping ~27% of the horizontal
FoV (`MISSION_VALIDATION_ARCHITECTURE.md §2.2`) — the same loss
behind the letterbox-vs-center-crop reconciliation. A NaFlex
SigLIP-2 variant could take the 16:9 frame natively, **preserving the
side context and making the preprocessing reconciliation moot.**
Evaluate the NaFlex variant alongside the fixed-224 one on the same
job set, report the FoV / preprocessing trade in the write-up, and if
NaFlex clears the bars make it the preferred input path.

### Eval methodology — shared scaffolding, widened jobs

This brief runs on the **shared eval scaffolding** — the
statistics framework, the held-out episode set, the
bias-mitigation protocol — that
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
otherwise carries. Because this brief now runs **before** v1
(it picks the trunk v1 ships on), that scaffolding must be
available to the bake-off first. Whether it splits out of
`validator-evaluation` into a shared substrate or this brief
stands it up is an **open coordination point for the
orchestrator** (see "Pickup gate / reorder"). The statistics
framework itself is reused without reinventing it. That means:

- Same held-out scene seeds (per the shared eval's
  `held_out_seeds` field). Backbone choice cannot game the
  eval by drawing from training-distribution seeds.
- Same per-window + per-leg disaggregation.
- Same warm-vs-cold map disaggregation.
- Same arbiter bias-mitigation protocol.
- Same Brier calibration + McNemar pairwise comparisons.

**Widened beyond case-1 / case-2.** The candidate trunk serves
*every* consumer in the spine, so each candidate is scored on
two additional job families drawn from
[`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md):

- **VPR Recall@K** — the same-place retrieval the VPR head
  ranks on. The spine's load-bearing bet is that a text-capable
  trunk's visual features are good enough for VPR; this is where
  it is measured (with MegaLoc as the backbone-flexible
  aggregation hedge). A trunk that fails the VPR bar is what
  triggers the DINOv3 hybrid fallback.
- **Region V-measure** — the partition quality the region head
  produces on the open-plan and multi-bedroom scenes.

A trunk wins only if it serves the validator AND the spatial
encoder's heads; a high case-2 AUC with a failing VPR Recall@K
is not a shippable shared trunk on its own.

The decision rule for "this trunk is better" is the
**same ≥ 1-CI-width margin** the cotrained brief uses for its
ablation table: the chosen trunk must beat OpenCLIP ViT-B/32
(the baseline-to-beat) by at least one CI width on case-1
(warm-map) and case-2 ROC-AUC, must not regress VPR Recall@K
or region V-measure below the baseline, AND stay inside the
latency budget. SigLIP-2-Base is the lead going in; MobileCLIP-2-S
is preferred if it clears every bar at materially lower latency.
If no text-capable trunk clears the VPR bar, the brief records
that and triggers the DINOv3-hybrid fallback path rather than
forcing a single-tower pick. If nothing displaces the baseline,
the brief records "OpenCLIP not displaced; reasons here."

### Open-vocab room-state API — measured here, owned elsewhere

The `query_room_by_text` API surface ships on v1's OpenCLIP
ViT-B/32 via
[`query-room-by-text-v1`](../../active/multi-room/query-room-by-text-v1.md);
it does NOT block on this bakeoff and the API itself is
backbone-agnostic. This brief retains the **per-backbone
quality measurement** of that API:

- [ ] **Per-backbone open-vocab eval.** Each candidate
      trunk's bakeoff report includes a precision@5 /
      MRR measurement of `query_room_by_text` against
      ground-truth room labels on the eval set. The
      open-vocab path's quality is reported per trunk,
      not assumed; the v1 brief sets the floor and this
      brief measures whether SigLIP-2 / MobileCLIP-2
      lift it. All three candidates are text-capable, so
      every candidate has a `query_room_by_text` number.
- [ ] **Hybrid-fallback text path (conditional).** The
      DINOv3-S vision half has no text tower. It is only
      exported / measured if a single text-capable trunk
      fails the VPR bar. In that hybrid, a *paired* text
      tower path (vision: DINOv3-S, text: the winning
      text-capable trunk) requires a learned projection
      head trained on the eval set's paired
      `(image, room_label)` data — embeddings from
      heterogeneous towers are not directly comparable via
      cosine. Filed as a follow-up here only if the hybrid
      fallback is triggered.

### Latency budget

| Backbone | Expected FP16 latency on Orin Nano (per-frame visual encode) | Source |
|---|---|---|
| SigLIP-2-Base/16 (lead) | ~150 – 220 ms (same architecture class, slightly larger patch tokens) | Architecture parity; measure to confirm |
| MobileCLIP-2-S | ~30 – 80 ms — the explicit mobile target | [Apple ML team's iPhone numbers](https://machinelearning.apple.com/research/mobileclip), scaled |
| OpenCLIP ViT-B/32 (baseline) | 150 – 200 ms (CUDA EP, today); 80 – 120 ms (TRT-EP, projected) | [`MISSION_VALIDATION_ARCHITECTURE.md` §3.1](../../../MISSION_VALIDATION_ARCHITECTURE.md#31-better-clip-usage) |
| DINOv3-S/14 (hybrid half — conditional) | ~60 – 80 ms (matches DINOv2-S/14 architecture) | [`MISSION_VALIDATION_ARCHITECTURE.md` §3.1](../../../MISSION_VALIDATION_ARCHITECTURE.md#31-better-clip-usage) |

All sit comfortably under the `BackgroundMapper`'s 2 s
poll interval. The brief reports measured numbers in the
addendum, not projected.

### Backbone-dim hygiene

The embedding dimension is pinned to the trunk (OpenCLIP
ViT-B/32 = 512; SigLIP-2-Base = 768). Two consequences ship
**inside the backbone-switch PR**, not as follow-ups:

- **(a) Make the two live 512-dim sentinels backbone-dim-aware.**
  Both currently hardcode 512:
  - [`clip_encoder.py:17`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py#L17)
    — `_EMBEDDING_DIM = 512` (the zero-vector fallback dim and
    the encoder's declared output dim).
  - [`manager.py:590`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py#L590)
    — `np.zeros(512, dtype=np.float32)` (the empty-result
    fallback in `get_clip_embedding`).
  Both must derive the dim from the active backend, not a
  literal, so a non-512 trunk doesn't emit mis-shaped vectors.
  (Cites verified against the working tree.)
- **(b) Re-index or freshen the persisted store on a backend
  change.** The ChromaDB store at
  `~/.strafer/semantic_map/chroma` (created in
  [`manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  from `storage_dir`) is **dim-pinned at first insert** —
  inserting a different-dim embedding into an existing
  collection fails or corrupts ANN results. A backend change
  must either re-index the store under the new dim or start
  from a fresh store. The switch ships this step; it is not
  optional.

This section is the checklist the spine's "Backbone-dim
hygiene" delegates to this brief. If `clip_encoder.py`'s
backbone loading changes, this is the gate.

## Acceptance criteria

- [ ] **ONNX exports.** Each text-capable candidate exports
      `visual.onnx` + `text.onnx` to
      `~/.strafer/models/backbones/<name>/`. Export script
      lives at `source/strafer_lab/scripts/export_backbone_onnx.py`
      and is parameterized by `--backbone {siglip2-base,
      mobileclip2-s, openclip-vit-b32}`; `dinov3-s` (vision-only)
      is supported by the script but exported only if the hybrid
      fallback is triggered. The script writes a metadata JSON
      sidecar with the source checkpoint URL, the export-time
      torch version, the embedding dim, and a SHA-256 of the
      ONNX file for reproducibility.
- [ ] **Runtime backbone switch (+ dim hygiene).**
      [`clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py)
      reads `STRAFER_CLIP_BACKEND` and resolves the ONNX path
      under `~/.strafer/models/backbones/<name>/`. The existing
      `clip_visual.onnx` / `clip_text.onnx` top-level files
      remain the `STRAFER_CLIP_BACKEND=openclip-vit-b32` path
      (backward-compat). **The same PR lands the backbone-dim
      hygiene work** (see "Backbone-dim hygiene"): the two 512-dim
      sentinels at
      [`clip_encoder.py:17`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py#L17)
      and
      [`manager.py:590`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py#L590)
      become backbone-dim-aware, and the dim-pinned ChromaDB
      store at `~/.strafer/semantic_map/chroma` is re-indexed /
      freshened on a backend change. (Hybrid only) a vision-only
      half degrades the image-vs-text monitor gracefully — if no
      text tower is available, only the image-vs-image monitor
      runs, and the executor logs this at startup.
- [ ] **Per-candidate eval on the full job set.** Run the
      shared eval scaffolding with `--backbone <name>` for each
      text-capable candidate. Same held-out seeds, same
      per-window / per-leg / map-state disaggregation. Each
      candidate is scored on case-1 + case-2 ROC-AUC **and** VPR
      Recall@K **and** region V-measure (the
      [`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md)
      jobs). Each emits a
      `data/transit_monitor_eval/<run_id>__<backbone>/report.json`
      and the committed copy under
      `docs/artifacts/backbone_bakeoff/<run_id>/`.
- [ ] **Latency measurements** on the actual Orin Nano (not
      the DGX). Each backbone's visual + text encode latency
      is reported as median + p95 from ≥ 1000 captures.
- [ ] **Bakeoff write-up.**
      `docs/artifacts/backbone_bakeoff/<run_id>/report.md`
      contains a single table with one row per candidate and
      columns: case-1 ROC-AUC + CI (warm-map only), case-2
      ROC-AUC + CI, cascade-end-to-end ROC-AUC + CI, Brier
      score per case, **VPR Recall@K, region V-measure**,
      median + p95 latency, VRAM at FP16, and a one-line
      "displaces baseline?" verdict per the ≥ 1-CI-width rule
      above. If no text-capable trunk clears the VPR bar, the
      write-up records that and recommends the DINOv3 hybrid
      fallback.
- [ ] **§4.6 addendum** to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
      records the chosen trunk, the comparison table, and
      a one-paragraph rationale. If no trunk displaces
      OpenCLIP, the addendum records that explicitly so the
      cotrained brief inherits the default knowingly. If the
      choice changes any spine fact, update
      [`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md)
      in the same PR.
- [ ] **Default flip (if applicable).** If a backbone
      displaces OpenCLIP, `executor/main.py` and
      `finetune_clip.py`'s default `--pretrained` value flip
      in the same PR. The legacy ONNX path remains accessible
      via env var so operators can A/B test.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the v1 cascade with
      `STRAFER_CLIP_BACKEND=openclip-vit-b32`. Smoke this in
      the PR description with the same one-line excerpt the
      validator-evaluation brief uses.

## Investigation pointers

- OpenCLIP export precedent:
  [`finetune_clip.export_towers_to_onnx`](../../../../source/strafer_lab/scripts/retired/finetune_clip.py#L262).
  Mirror the structure for the new backbones.
- DINOv3 weights (hybrid fallback only): [`facebook/dinov3-small`](https://huggingface.co/facebook/dinov3-small)
  on HuggingFace; loads via `transformers` cleanly. Exported
  only if a single text-capable trunk fails the VPR bar.
- SigLIP-2: `google/siglip2-base-patch16-224`. Tokenizer
  differs from OpenCLIP; the runtime side needs the right
  tokenizer per backbone — encode at `clip_encoder.encode_text`'s
  fallback path.
- MobileCLIP-2: [`apple/ml-mobileclip`](https://github.com/apple/ml-mobileclip).
  Distinct export path; the maintainers provide ONNX-friendly
  recipes.
- ONNX-providers preference for each: TRT-EP if the export
  graph is fully fused (OpenCLIP / MobileCLIP-2 likely yes);
  CUDA-EP as the fallback. Document per backbone in the
  metadata sidecar.

## Out of scope

- **Fine-tuning any of the candidate backbones.** This brief
  measures **off-the-shelf** zero-shot performance. Fine-tuning
  lives in
  [`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md),
  which consumes whichever backbone the bakeoff selects.
- **Inventing a new statistics framework.** The held-out scene
  seeds, the per-window / per-leg labels, the calibration, and
  the bias-mitigation protocol are reused from the shared eval
  scaffolding, not reinvented. (Where that scaffolding *lives*
  — split out of `validator-evaluation` or stood up here — is a
  coordination decision, not new methodology; see "Pickup gate
  / reorder".) Folding in the VPR Recall@K and region V-measure
  jobs reuses the
  [`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md)
  metric definitions rather than authoring new ones.
- **Real-robot measurement.** Sim-side only. Real-robot
  transfer is a future brief.
- **Visual tower architectures outside the named candidates.**
  EVA-CLIP, Florence-2, BEiT-3, ConvNeXt-CLIP and friends are
  deliberately deferred — the three text-capable candidates
  (plus the conditional DINOv3 hybrid half) are the
  highest-leverage 2025 options given the cascade's Orin Nano
  latency budget and the project's preference for
  HuggingFace-hosted, ONNX-exportable, deployment-friendly
  weights. A follow-up brief can extend the bake-off if a
  candidate loses by a small margin and a new candidate becomes
  obviously relevant.
- **Replacing the cascade architecture.** The bakeoff stays
  inside the cascade-with-arbiter pattern; replacing it with
  an end-to-end VLA lives in
  [`vla-v2-architecture`](../experimental/vla-v2-architecture.md).
