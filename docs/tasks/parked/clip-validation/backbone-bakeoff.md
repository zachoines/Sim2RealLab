# Backbone bake-off: OpenCLIP vs. DINOv3 vs. SigLIP-2 vs. MobileCLIP-2 for the cascade tripwire

**Type:** investigation / new feature
**Owner:** DGX agent (ONNX exports + offline eval are all
DGX-side; the runtime swap behind `STRAFER_CLIP_BACKEND` is a
small Jetson-lane edit, cross-lane like
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md))
**Priority:** P2 (filed-on-trigger; the project should not
re-commit to a 2021-era backbone for the case-1 + case-2
tripwire and the downstream cascade improvements without a
defensible alternative-considered comparison)
**Estimate:** M (~half-week to a week; ONNX export of 3 – 4
backbones + the existing eval script re-run + a short
write-up)
**Branch:** task/backbone-bakeoff

## Story

As an **operator who has shipped the v1 CLIP cascade on the
codebase's existing OpenCLIP ViT-B/32 export and is about to
spend XL training cycles on
[`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md)
or
[`vla-v2-architecture`](../experimental/vla-v2-architecture.md)**,
I want **a documented head-to-head comparison of the 2025-era
visual+text backbones (OpenCLIP ViT-B/32, SigLIP-2-Base,
MobileCLIP-2-S, DINOv3-S) on the same case-1 + case-2 eval
set the v1 cascade was measured on**, so that **the
cascade-improvements brief and the VLA v2 brief both pick a
backbone with the alternative-considered trail in writing,
rather than inheriting ViT-B/32 by default because that's
what `finetune_clip.py` happens to export today**.

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
  prerequisite. Ships the wiring, the eval script, the harness
  episode set, and the pre-registered statistics framework
  this brief reuses without modification.
- [`cotrained-retrieval-augmented`](cotrained-retrieval-augmented.md) —
  consumes the bakeoff's chosen backbone for the fine-tune
  init. Sibling brief: same epic, same eval framework.
- [`vla-v2-architecture`](../experimental/vla-v2-architecture.md) —
  the VLA backbone (OpenVLA's DINOv2+SigLIP fusion, π0's tower,
  Octo's encoder, ...) is a constraint on this brief's choice:
  the cascade and the v2 stack should ideally share a visual
  tower so co-tenancy memory + ablation comparability are
  preserved.

## Context

### Why this brief exists

The CLIP fine-tune target in
[`finetune_clip.py`](../../../../source/strafer_lab/scripts/finetune_clip.py)
starts from OpenCLIP ViT-B/32 + `laion2b_s34b_b79k` weights —
the same OpenAI-style 2021-era backbone the field has moved
past on essentially every axis that matters for case-1 and
case-2:

| Axis | OpenCLIP ViT-B/32 (current) | What's available in 2025 |
|---|---|---|
| **Pretrain objective** | Symmetric InfoNCE on image-text pairs | SigLIP-2 (sigmoid loss + caption pretraining + self-distillation, outperforms SigLIP-1 at every scale per [Tschannen et al., 2025](https://arxiv.org/abs/2502.14786)) |
| **Visual-only feature quality** | ViT-B/32 patches; lossy on indoor scenes | DINOv3-S/14 — 7B teacher distilled to small students; explicit robotics applications per [Meta DINOv3, 2025](https://ai.meta.com/dinov3/) and [DINOv3-Diffusion-Policy](https://arxiv.org/abs/2509.17684); pure-vision so it cannot do case 2 alone but dominates case 1 |
| **Jetson-class latency** | ~150 – 200 ms at FP16 (CUDA EP); plausibly 80 – 120 ms at TRT-EP after a tuning pass | MobileCLIP-2-S4 matches SigLIP-SO400M/14 at **2× fewer parameters** and runs at 2.5× lower latency than DFN ViT-L/14 on iPhone 12 Pro Max ([Faghri et al., TMLR 2025](https://arxiv.org/abs/2508.20691)) — strict superset of "fits Orin Nano" |
| **Patch count at 224²** | 49 patches (ViT-B/32) | 256 (DINOv3-S/14), 196 (SigLIP-2-B/16), comparable (MobileCLIP-2 variants) — directly addresses the §2.2 patch-count limitation |
| **VLA-backbone overlap** | None (no shipping wheeled VLA uses ViT-B/32) | OpenVLA fuses DINOv2 + SigLIP visual features; π0 / NaVid / NaVILA use SigLIP-family encoders. Sharing a tower with the v2 VLA is structurally cleaner. |

[`MISSION_VALIDATION_ARCHITECTURE.md` §3.1](../../../MISSION_VALIDATION_ARCHITECTURE.md#31-better-clip-usage)
names these candidates in prose. This brief is the measurement
that turns prose into a chosen backbone with numbers attached.

### What this brief produces

Five artifacts in one PR:

1. **ONNX exports** of the candidate backbones under
   `~/.strafer/models/backbones/<name>/{visual,text}.onnx` (or
   `visual.onnx` only for DINOv3-S — vision-only family). One
   sub-directory per backbone.
2. **A runtime backbone switch** behind a
   `STRAFER_CLIP_BACKEND` env var consumed by
   [`clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py).
   The current `clip_visual.onnx` / `clip_text.onnx` filenames
   become the `openclip` backend; new backends live under their
   own subdirs. Default stays `openclip` until this brief
   declares a swap.
3. **A re-run of the validator-evaluation script** (no logic
   changes; only `--backbone` flag added) against the same
   held-out episode set, emitting one
   `data/transit_monitor_eval/<backbone>/report.json` per
   candidate. Same statistics framework, same scene seeds,
   same case-1 / case-2 split — only the backbone changes.
4. **A bakeoff write-up** at
   `docs/artifacts/backbone_bakeoff/<run_id>/report.md`
   summarizing per-backbone ROC-AUC + 95 % CI, latency on
   Orin Nano FP16, VRAM footprint, and a recommendation.
5. **The default backbone flip** in `executor/main.py` and
   `finetune_clip.py`'s default init, plus a §4.6 addendum
   to
   [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
   recording the chosen backbone and the comparison table.

### Candidates and rationale

| Backbone | Why measure | Pretrain corpus | Has text tower? |
|---|---|---|---|
| **OpenCLIP ViT-B/32 (current)** | Baseline; what the v1 cascade shipped on | LAION-2B (`laion2b_s34b_b79k`) | Yes |
| **SigLIP-2-Base/16** | 2025 SOTA for zero-shot image-text alignment at this parameter budget; drop-in replacement for the case-2 image-vs-text head | WebLI + multilingual + self-distillation | Yes |
| **MobileCLIP-2-S/B variants** | Explicit Jetson / mobile target; the lower-latency option that may still hit the AUC bar | Reinforced training (synthetic captions + ensemble teachers) | Yes |
| **DINOv3-S/14** | 2025 SOTA pure-vision; dominates case 1 place-recognition where text is irrelevant | LVD-1689M (1.7 B images, no captions) | **No** — paired with one of the above for case 2 |

DINOv3 is **explicitly vision-only**. The bakeoff treats it as
a hybrid candidate: DINOv3-S/14 for the case-1 image-vs-image
retrieval signal, paired with whichever of {SigLIP-2,
MobileCLIP-2} wins the case-2 image-vs-text head.

### Eval methodology — reuse, don't reinvent

This brief reuses the
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
brief's eval script, episode set, and statistics framework
**unchanged**. The only difference is which ONNX gets loaded
in `clip_encoder.py`. That means:

- Same held-out scene seeds (per
  validator-evaluation's `held_out_seeds` field). Backbone
  choice cannot game the eval by drawing from
  training-distribution seeds.
- Same per-window + per-leg disaggregation.
- Same warm-vs-cold map disaggregation.
- Same arbiter bias-mitigation protocol.
- Same Brier calibration + McNemar pairwise comparisons.

The decision rule for "this backbone is better" is the
**same ≥ 1-CI-width margin** the cotrained brief uses for its
ablation table: each new backbone must beat OpenCLIP ViT-B/32
by at least one CI width on both case-1 (warm-map) and case-2
ROC-AUC, AND stay inside the same latency budget, to displace
the default. Otherwise the default stays put and the brief
records "OpenCLIP not displaced; reasons here" as the outcome.

### Latency budget

| Backbone | Expected FP16 latency on Orin Nano (per-frame visual encode) | Source |
|---|---|---|
| OpenCLIP ViT-B/32 | 150 – 200 ms (CUDA EP, today); 80 – 120 ms (TRT-EP, projected) | [`MISSION_VALIDATION_ARCHITECTURE.md` §3.1](../../../MISSION_VALIDATION_ARCHITECTURE.md#31-better-clip-usage) |
| SigLIP-2-Base/16 | ~150 – 220 ms (same architecture class, slightly larger patch tokens) | Architecture parity; measure to confirm |
| MobileCLIP-2-S | ~30 – 80 ms — the explicit mobile target | [Apple ML team's iPhone numbers](https://machinelearning.apple.com/research/mobileclip), scaled |
| DINOv3-S/14 | ~60 – 80 ms (matches DINOv2-S/14 architecture) | [`MISSION_VALIDATION_ARCHITECTURE.md` §3.1](../../../MISSION_VALIDATION_ARCHITECTURE.md#31-better-clip-usage) |

All four sit comfortably under the `BackgroundMapper`'s 2 s
poll interval. The brief reports measured numbers in the
addendum, not projected.

## Acceptance criteria

- [ ] **ONNX exports.** Each candidate backbone exports to
      `~/.strafer/models/backbones/<name>/visual.onnx` (+
      `text.onnx` if the family includes a text tower).
      Export script lives at
      `source/strafer_lab/scripts/export_backbone_onnx.py` and
      is parameterized by `--backbone {openclip-vit-b32,
      siglip2-base, mobileclip2-s, dinov3-s}`. The script
      writes a metadata JSON sidecar with the source checkpoint
      URL, the export-time torch version, and a SHA-256 of the
      ONNX file for reproducibility.
- [ ] **Runtime backbone switch.**
      [`clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py)
      reads `STRAFER_CLIP_BACKEND` and resolves the ONNX path
      under `~/.strafer/models/backbones/<name>/`. The existing
      `clip_visual.onnx` / `clip_text.onnx` top-level files
      remain the `STRAFER_CLIP_BACKEND=openclip-vit-b32` path
      (backward-compat). Vision-only backbones (DINOv3-S)
      degrade the image-vs-text monitor gracefully — if no text
      tower is available, only the image-vs-image monitor
      runs, and the executor logs this at startup.
- [ ] **Re-run validator-evaluation script** with
      `--backbone <name>` for each candidate. Same held-out
      seeds, same per-window / per-leg / map-state
      disaggregation. Each emits a
      `data/transit_monitor_eval/<run_id>__<backbone>/report.json`
      and the committed copy under
      `docs/artifacts/backbone_bakeoff/<run_id>/`.
- [ ] **Latency measurements** on the actual Orin Nano (not
      the DGX). Each backbone's visual + text encode latency
      is reported as median + p95 from ≥ 1000 captures.
- [ ] **Bakeoff write-up.**
      `docs/artifacts/backbone_bakeoff/<run_id>/report.md`
      contains a single table with one row per backbone and
      columns: case-1 ROC-AUC + CI (warm-map only), case-2
      ROC-AUC + CI, cascade-end-to-end ROC-AUC + CI, Brier
      score per case, median + p95 latency, VRAM at FP16, and
      a one-line "displaces openclip?" verdict per the
      ≥ 1-CI-width rule above.
- [ ] **§4.6 addendum** to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
      records the chosen backbone, the comparison table, and
      a one-paragraph rationale. If no backbone displaces
      OpenCLIP, the addendum records that explicitly so the
      cotrained brief inherits the default knowingly.
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
  [`finetune_clip.export_towers_to_onnx`](../../../../source/strafer_lab/scripts/finetune_clip.py#L262).
  Mirror the structure for the new backbones.
- DINOv3 weights: [`facebook/dinov3-small`](https://huggingface.co/facebook/dinov3-small)
  on HuggingFace; loads via `transformers` cleanly.
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
- **A new eval methodology.** The eval script, the held-out
  scene seeds, the per-window / per-leg labels, and the
  bias-mitigation protocol are all inherited verbatim from
  [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md).
- **Real-robot measurement.** Sim-side only, same as the
  parent eval brief. Real-robot transfer is a future brief.
- **Visual tower architectures outside the four named
  candidates.** EVA-CLIP, Florence-2, BEiT-3, ConvNeXt-CLIP
  and friends are deliberately deferred — the four chosen are
  the highest-leverage 2025 candidates given the cascade's
  Orin Nano latency budget and the project's preference for
  HuggingFace-hosted, ONNX-exportable, deployment-friendly
  weights. A follow-up brief can extend the bakeoff if any of
  the four lose by a small margin and a fifth candidate
  becomes obviously relevant.
- **Replacing the cascade architecture.** The bakeoff stays
  inside the cascade-with-arbiter pattern; replacing it with
  an end-to-end VLA lives in
  [`vla-v2-architecture`](../experimental/vla-v2-architecture.md).
