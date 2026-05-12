# Co-trained CLIP fine-tune with retrieval-augmented inference (cascade improvements)

**Type:** investigation / research / new feature
**Owner:** DGX agent (training pipeline + cross-attention layer
deployment via `clip_encoder.py` ONNX swap; minimal Jetson-side
glue for the new module)
**Priority:** P3 (research follow-up; filed-on-trigger after
[`clip-mid-mission-validator-evaluation`](clip-mid-mission-validator-evaluation.md)
ships, regardless of whether the cascade meets its AUC bar.
Mandatory if the bar fails; optional but well-motivated if it
passes — both phases compound on top of a passing baseline.)
**Estimate:** XL (~multi-week; two-step research with
multi-task training, retrieval-aware training, deployment
integration, and ablation reports)
**Branch:** task/clip-cotrained-retrieval-augmented

## Story

As an **operator who has shipped the v1 CLIP cascade and wants
to push its statistics higher using the harness corpus and
SemanticMapManager memory primitive we've built**, I want **a
two-step research effort that (Step A) co-trains the CLIP
fine-tune with the trajectory-first speaker model on the
harness corpus and (Step B) adds a retrieval-augmented
cross-attention layer that uses the SemanticMapManager's
ChromaDB index at inference**, so that **the cascade's case-1
and case-2 ROC-AUC improve by leveraging the same data and
memory primitives the rest of the project already produces,
with clean ablation between co-training and retrieval
contributions**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
§3.1 (better CLIP usage), §3.5 (cascade with arbiter), §4
(recommendation; this brief is the named cascade-improvement
follow-up).

Sibling briefs this brief depends on or composes with:
- [`clip-mid-mission-validator-evaluation`](clip-mid-mission-validator-evaluation.md) —
  the v1 cascade ships first; its statistics framework is the
  baseline this brief improves on.
- [`harness-trajectory-first-captioning`](harness-trajectory-first-captioning.md) —
  produces the speaker model + caption corpus this brief
  co-trains against.
- [`harness-behavior-cloning-data-expansion`](harness-behavior-cloning-data-expansion.md) —
  defines the canonical harness output schema both phases
  consume.
- The SemanticMapManager's ChromaDB infrastructure already
  exists in
  [`source/strafer_autonomy/strafer_autonomy/semantic_map/`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/)
  — this brief reuses it as the retrieval index.

Retired brief this one replaces:
- [`completed/learned-mid-mission-validator.md`](../../completed/learned-mid-mission-validator.md) —
  small frozen-head validator was structurally underpowered
  against the cascade; cascade improvements are the better
  investment.

## Context

### Why two steps, why combine

The two phases address different layers of the cascade:

| Layer | Step A (co-training) | Step B (retrieval-augmented inference) |
|---|---|---|
| Training-time | **Shapes representations** via multi-task losses (InfoNCE on speaker captions + same-region contrastive + hard-negative target-text) | Trains a cross-attention layer that consumes retrieved memories alongside the live frame |
| Inference-time | Frozen, used like any CLIP fine-tune (drop-in ONNX replacement) | **Adds memory lookup** via SemanticMapManager + cross-attention; ~5–10 ms additional latency |
| Corpus | Harness frames + trajectory-first speaker captions | Same corpus, indexed in ChromaDB |
| Result | Better representations | Better use of those representations at inference |

The two enhancements compound. Co-training makes the CLIP
encoder produce representations that are *retrieval-friendly*
(because it's been trained on the same speaker corpus that
lives in the retrieval index). The retrieval layer then *uses*
those representations to actually pull useful neighbors at
inference. Each improvement makes the other more effective.

### Empirical precedent

The lineage this brief draws from:

| Source | Contribution |
|---|---|
| **Speaker-Follower** (Fried et al., NeurIPS 2018) | Speaker-augmented data improves follower; cross-task representation transfer |
| **Re-CLIP / REACT** (Liu et al., NeurIPS 2024) | Retrieval-augmented CLIP trained to consume retrieved context; demonstrates training-with-retrieval beats inference-only retrieval |
| **MemoryBank-CLIP** (2024) | Learnable memory bank trained jointly with the encoder |
| **Self-RAG for vision-language** (2024) | Model learns when to retrieve, not just how |
| **Atlas / RA-DIT** | Retrieval-aware-training pattern (text-only origin, ported to vision-language) |

Headline empirical finding across these: **training with
retrieval in the forward pass beats training without retrieval
and bolting it on at inference.** Co-training also makes
representations memory-aware, which is what the cross-attention
layer needs at inference.

### Step A — Co-trained CLIP + speaker

**The candidate.** Multi-task fine-tune of the OpenCLIP visual +
text towers on the harness corpus. Three loss heads:

1. **Image-vs-instructive-caption InfoNCE.** Positive pairs:
   `(frame, instructive_caption)` from the
   trajectory-first speaker. Negative pairs: in-batch + a hard
   sampling of `(frame, mismatched_caption)` from the speaker's
   negative pairs. Aligns the visual encoder with
   *instruction-following* language, not just description.
2. **Same-region image-image contrastive (case-1).** Positives:
   two views from the same `scene_name` within R = 1 m. Negatives:
   views from different scenes. Pushes the visual tower toward
   place-recognition-friendly representations.
3. **Hard-negative target-text alignment (case-2).** Positives:
   `(frame, target_phrase)` where the target is the focal
   object. Hard negatives: `(frame, alternate_phrase)` where
   the alternate is a same-label sibling drawn from
   `scene_metadata.json` (per the renamed mission generator's
   negative-mining rules). Forces instance-level discrimination.

Optional auxiliary head 4: **caption-reconstruction loss** —
the visual tower's pooled embedding is decoded by a small head
to reconstruct the speaker's caption. Forces the visual tower
to encode language-relevant content. Increases multi-task
complexity; treat as a stretch goal.

Loss weighting: start at `1.0 * loss_1 + 0.5 * loss_2 + 0.5 *
loss_3` and tune via per-loss validation curves. Standard
multi-task balancing (gradient surgery, uncertainty weighting)
applies if losses fight.

Backbone: continue from OpenCLIP ViT-B/32 + `laion2b_s34b_b79k`
weights (the existing fine-tune init). LoRA adapters on
attention + projection layers; full towers stay frozen by
default. ~50 M trainable params via LoRA vs. ~150 M for full
fine-tune. Choice between LoRA and full is a brief-execution
decision; document it.

**Deployment:** the resulting `clip_visual.onnx` +
`clip_text.onnx` drop in via the existing
[`clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py)
ONNX-loading path. No runtime code changes needed for Step A
alone.

**Acceptance:** re-run the cascade statistics from
[`clip-mid-mission-validator-evaluation`](clip-mid-mission-validator-evaluation.md)
on the same episode set. Report ROC-AUC + 95 % CI per case
+ signal, comparing the Phase-1 model against the v1 baseline.
Step A ships if it improves both case-1 and case-2 ROC-AUC by
at least one CI-width on the v1 baseline; it regresses (does
not ship) if either case worsens.

### Step B — Retrieval-augmented cross-attention layer

**The candidate.** A cross-attention module placed between the
visual tower's output and the cosine head. At inference:

```
live frame → CLIP visual tower (Phase-1 fine-tuned) → query embedding
  → ChromaDB retrieval: top-K past observations near current pose
                        and high-cosine to the query embedding
  → cross-attention(query, retrieved) → augmented embedding
  → cosine vs. semantic-map nodes (case-1) OR
    cosine vs. mission-text embedding (case-2)
```

**Cross-attention module.** Standard transformer cross-attention
block: query ∈ R^512 (the live frame's CLIP embedding); keys +
values ∈ R^(K × 512) from the retrieved past observations.
~5–10 M trainable parameters. Output is a 512-dim
"memory-augmented embedding" that replaces the bare query in
the downstream cosine.

**Retrieval.** Reuses
[`SemanticMapManager.query_by_embedding`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
to fetch top-K = 8 (default; tunable) past observations from
ChromaDB. Each observation contributes its CLIP embedding +
metadata (pose, timestamp, source). Both at training time and
at inference time the retrieval pull comes from the same kind
of index — no train-test distribution mismatch on the memory
side.

**Training (RAG-aware).** The cross-attention layer trains
*with retrieval in the forward pass*. For each training-time
sample `(frame, label, scene)`:

1. Sample top-K retrievals from a "scene-snapshot" ChromaDB
   index built from the harness corpus minus the held-out
   trajectory. Holdout protocol: the trajectory the training
   sample comes from is excluded from the retrieval pool to
   avoid leakage.
2. Forward pass: Phase-1 encoder → query embedding →
   cross-attention(query, retrieved) → augmented embedding →
   loss head.
3. Loss head: same as Step A (case-1 contrastive or case-2
   target-text alignment) but with the augmented embedding as
   the query. Optionally, an additional **retrieval-aware
   InfoNCE** that contrasts augmented vs. bare query —
   encourages the layer to *use* the memory rather than ignore
   it.

Phase-1 encoder can stay frozen during Phase-2 training (only
the cross-attention layer learns), or be jointly fine-tuned
(harder, more data-hungry, potentially higher ceiling). Brief
execution picks based on Phase-1's measured ROC-AUC; if
Step A already converges cleanly, freeze; if not, joint
fine-tune.

**Deployment.** A new ONNX file
`~/.strafer/models/cross_attn.onnx` consumed by an extended
`clip_encoder.py` (or a sibling
`semantic_map/retrieval_augmented_encoder.py`). The
SemanticMapManager's existing query path is reused; the
cross-attention call adds ~5–10 ms latency at FP16. Total
encoder budget: ~150 ms (Step A) + ~5–10 ms (cross-attention)
+ ~5 ms (ChromaDB top-K). Comfortably under the
`BackgroundMapper`'s 2 s poll interval.

**Acceptance:** re-run cascade statistics. Report ROC-AUC + CIs
comparing:
- v1 baseline
- Step A alone
- Step A + Step B (combined)

Step B ships if **adding retrieval** improves both case-1 and
case-2 ROC-AUC by at least one CI-width over Phase-1-alone. If
the combined system regresses on either case vs. Step A, Step B
does *not* ship and the brief documents why (typical failure
mode: cross-attention layer overfits to retrieval distribution
mismatch).

### Combined ablation framework

The brief produces a single addendum (§4.5) to
[`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
reporting:

| System | Case-1 ROC-AUC + 95% CI | Case-2 ROC-AUC + 95% CI | Time-to-decision (median, p95) |
|---|---|---|---|
| v1 baseline (vanilla OpenCLIP ViT-B/32) | from clip-eval brief | from clip-eval brief | from clip-eval brief |
| Step A (co-trained) | this brief | this brief | this brief |
| Step B added (co-trained + retrieval-augmented) | this brief | this brief | this brief (incl. ChromaDB latency) |

Plus McNemar's test comparing each system pair on the same
held-out trial set, and Brier-score calibration curves.

### Sim-to-real considerations

Both phases are trained on Infinigen sim data. The same caveats
that apply to the v2 VLA brief apply here:
- Sim-to-real gap on visual features (Infinigen rendering ≠ real
  D555 capture).
- Retrieval index distribution shift (real-robot deployment
  builds a different ChromaDB index than the training index).
- Cross-attention layer may overfit to the sim retrieval
  pattern.

Real-robot transfer is filed as a future brief once sim
results justify it. This brief stops at sim eval.

## Acceptance criteria

### Step A

- [ ] **Multi-task training script** at
      `source/strafer_lab/scripts/cotrain_clip_with_speaker.py`
      that consumes the harness corpus + the trajectory-first
      speaker captions; trains via the three-loss multi-task
      recipe; emits checkpoint + ONNX exports for visual + text
      towers; MLflow tracking via the same `--mlflow-experiment`
      flag pattern that
      [`finetune_clip.py`](../../../../source/strafer_lab/scripts/finetune_clip.py)
      uses.
- [ ] **LoRA vs. full fine-tune decision documented** in the
      training-config sidecar (the same JSON pattern existing
      training scripts emit).
- [ ] **Cascade re-eval with Phase-1 ONNX** using the same
      script + statistics framework as
      [`clip-mid-mission-validator-evaluation`](clip-mid-mission-validator-evaluation.md).
      Report ROC-AUC + CI per case + signal alongside v1
      baseline.
- [ ] **Phase-1 ship decision** recorded in addendum: ships iff
      ≥ one-CI-width improvement on both case-1 and case-2
      vs. v1.

### Step B

- [ ] **Cross-attention layer module** at
      `source/strafer_autonomy/strafer_autonomy/semantic_map/retrieval_augmented_encoder.py`
      implementing the cross-attention(query, retrieved) →
      augmented embedding forward pass.
- [ ] **RAG-aware training script** at
      `source/strafer_lab/scripts/train_retrieval_augmented_clip.py`
      that trains the cross-attention layer with retrieval in
      the forward pass. Holdout protocol prevents
      same-trajectory leakage.
- [ ] **ONNX export** of the cross-attention module to
      `~/.strafer/models/cross_attn.onnx`; consumable by the
      extended `clip_encoder.py` runtime path.
- [ ] **Cascade re-eval with combined system.** Same statistics
      framework, comparing:
      v1 baseline / Step A / Step A + Step B / cascade
      end-to-end (with arbiter on top of each).
- [ ] **Phase-2 ship decision** recorded in addendum: ships iff
      ≥ one-CI-width improvement on both case-1 and case-2
      vs. Step A.

### Cross-cutting

- [ ] **Latency budget verified.** Total encoder forward pass
      stays under 200 ms on Orin Nano FP16; total
      tripwire-decision latency stays under 500 ms (encoder +
      retrieval + cross-attention + cosine).
- [ ] **§4.5 addendum** to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
      with the full ablation table + McNemar's test pairwise
      comparisons + Brier calibration plots + time-to-decision
      CDFs.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.
- [ ] No regression in the v1 cascade with `STRAFER_VALIDATOR_BACKEND`
      pointing at the v1 ONNX (Step A + Step B ONNX deploy
      under separate filenames; legacy path unchanged).

## Investigation pointers

- The OpenCLIP fine-tune scaffold:
  [`finetune_clip.py`](../../../../source/strafer_lab/scripts/finetune_clip.py).
  Most of the multi-task recipe extends this.
- The trajectory-first speaker (and its caption corpus):
  [`harness-trajectory-first-captioning`](harness-trajectory-first-captioning.md).
- The retrieval primitive:
  [`SemanticMapManager.query_by_embedding`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py).
- LoRA implementation: PEFT library is the standard choice;
  drop-in for OpenCLIP attention + projection layers.
- Re-CLIP / REACT paper for the RAG-aware training recipe is
  the closest published precedent. Their public code (when
  available) is a reasonable starting reference.
- Cross-attention layer architecture: standard transformer
  block with one cross-attention layer; query from frame; key /
  value from retrieved. ~5–10 M parameters at hidden_dim=256
  with 4 heads.

## Out of scope

- **Real-robot retrieval index.** Sim-only for both phases.
  Real-robot transfer is a future brief.
- **Online retrieval-index updates.** The retrieval index is
  rebuilt offline from the harness corpus per training run; no
  online learning during deployment.
- **Replacing the cascade architecture with a single end-to-end
  validator.** That's the
  [`strafer-vla-v2-architecture`](strafer-vla-v2-architecture.md)
  brief's territory; this one stays inside the
  cascade-with-arbiter pattern.
- **Training a new speaker model.** This brief consumes
  whatever the trajectory-first brief ships. Speaker
  fine-tuning is its own follow-up if needed.
- **Retiring the v1 cascade ONNX.** Both Step A and Step B
  ONNX deploy under separate filenames so operators can A/B
  test or roll back via env var.
- **Reducing the cross-attention layer's parameter count
  beyond ~5–10 M.** Optimization for tighter Jetson budgets is
  a future concern; v1 of this brief picks a conservative size
  and verifies it fits.
