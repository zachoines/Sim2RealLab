# Co-trained CLIP + retrieval-augmented inference (the project's implicit-mapping primitive)

**Type:** investigation / research / new feature
**Owner:** DGX agent (training pipeline + cross-attention layer
deployment via `clip_encoder.py` ONNX swap; minimal Jetson-side
glue for the new module)
**Priority:** P3 (research follow-up; filed-on-trigger after
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
ships, regardless of whether the cascade meets its AUC bar.
Mandatory if the bar fails; optional but well-motivated if it
passes — both phases compound on top of a passing baseline.)
**Estimate:** XL (~multi-week; two-step research with
multi-task training, retrieval-aware training, deployment
integration, and ablation reports)
**Branch:** task/clip-cotrained-retrieval-augmented

## What this brief actually is — the project's implicit-mapping primitive

This brief sells itself as "cascade-validator improvements,"
but architecturally it **is** the project's implicit-mapping
primitive — the sub-symbolic counterpart to the explicit
semantic-graph work in the
[multi-room epic](../../active/multi-room/). Reading the brief
that way clarifies its scope and its downstream consumers:

- **Step B's cross-attention over a memory bank IS implicit
  mapping.** The 2025 production pattern for sub-symbolic
  scene representation is "indexed bank of past CLIP /
  DINOv2 / SigLIP embeddings consumed via cross-attention
  on the live query," not a learned dense feature field
  (CLIP-Fields / VLMaps / LERF). See
  [OpenScene (CVPR 2023)](https://arxiv.org/abs/2211.15654),
  [3D-VLA (CVPR 2024)](https://arxiv.org/pdf/2403.17846),
  [HOV-SG (RSS 2024)](https://arxiv.org/pdf/2403.17846),
  [OK-Robot (ICRA 2024)](https://arxiv.org/abs/2401.12202).
  This brief's Step B is the memory-bank-with-cross-attention
  shape — same pattern, applied to validator scoring as the
  first consumer.
- **The cascade validator is consumer #1; the v2 VLA is
  consumer #2.** When
  [`vla-v2-map-conditioning`](../experimental/vla-v2-map-conditioning.md)
  picks **Option B** (cross-attention over a memory bank),
  it inherits this brief's infrastructure verbatim — same
  ChromaDB index, same cross-attention module, same RAG-
  aware training pattern (the `K_train ∈ {0, 1, 2, 4, 8}`
  augmentation for cold-deployment robustness). Building it
  here first means the VLA path lands as a swap of the
  consumer head, not a from-scratch implementation.
- **Symbolic vs. sub-symbolic layer split.** The MVP's LLM
  planner needs the *explicit* semantic graph (it can't
  reason over a feature field); the v2 VLA stretch needs
  the *implicit* memory bank (an end-to-end policy doesn't
  consume discrete `RoomEntry` objects). The two coexist,
  share the ANN store, and serve different consumers. This
  brief is the implicit half of that split.

The framing changes nothing about the acceptance criteria
below — they were already correct. It changes how the brief
fits into the project's architecture story, which the
multi-room v2 audit (PR #43) surfaced as a gap.

## Story

As an **operator who has shipped the v1 CLIP cascade and wants
to push its statistics higher using the harness corpus and
SemanticMapManager memory primitive we've built — while
laying the implicit-mapping foundation the v2 VLA's
map-conditioning path will inherit**, I want **a two-step
research effort that (Step A) co-trains the CLIP fine-tune
with the trajectory-first speaker model on the harness corpus
and (Step B) adds a retrieval-augmented cross-attention layer
that uses the SemanticMapManager's ChromaDB index at
inference**, so that **the cascade's case-1 and case-2
ROC-AUC improve by leveraging the same data and memory
primitives the rest of the project already produces, the
implicit-mapping primitive is in place when the VLA stretch
goal picks it up, and the project has a clean symbolic-vs-
sub-symbolic split rather than two parallel sub-symbolic
efforts**.

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
- [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md) —
  the v1 cascade ships first; its statistics framework is the
  baseline this brief improves on.
- [`trajectory-first-captioning`](../../active/harness/trajectory-first-captioning.md) —
  produces the speaker model + caption corpus this brief
  co-trains against.
- [`behavior-cloning-data-expansion`](../../active/harness/behavior-cloning-data-expansion.md) —
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

Sibling research arm this brief is *not* duplicated by:
- [`vla-v2-architecture`](../experimental/vla-v2-architecture.md) —
  end-to-end VLA. See "Relationship to VLA v2" below for why
  the two efforts are kept separate even though both consume
  the trajectory-first speaker corpus.

Backbone-selection brief that gates the choice of visual /
text towers this brief fine-tunes:
- [`backbone-bakeoff`](backbone-bakeoff.md) — measures the
  competing 2025-era backbones (DINOv3-S, MobileCLIP-2-S,
  SigLIP-2-B, OpenCLIP ViT-B/32) on the same case-1 / case-2
  eval set. **This brief inherits whichever backbone the
  bakeoff picks; it does not pre-commit to ViT-B/32.**

## Context

### Relationship to VLA v2

Once [`vla-v2-architecture`](../experimental/vla-v2-architecture.md)
has a working speaker model and a fine-tuned VLA backbone on
the harness corpus, the natural question is: *does the v2 VLA's
visual+text tower deliver the case-1 / case-2 discrimination
this brief is fine-tuning for, "for free"?* Reading both briefs
together, the answer is **no** — the two efforts are
structurally distinct and the cascade still wants the
contrastive-objective tower this brief produces:

| Axis | VLA v2 (action-prediction) | This brief (contrastive-discrimination) |
|---|---|---|
| Training loss | Behavior cloning on `(frame, text, action)` tuples; SFT objective is `argmax P(action_t)` | Three InfoNCE / contrastive heads (image-vs-caption, same-region, hard-negative target-text); objective is `argmin distance(positive_pair) - distance(negative_pair)` |
| What the visual tower learns | Features that predict "drive forward", "turn", "stop" given context | Features that **discriminate** same-room-different-instance ("the south window" vs. "the north window") |
| Inference role | Drives `cmd_vel` end-to-end | Scores a tripwire signal that wakes the arbiter |
| Failure modes shared with the other | Sim-to-real gap; corpus distribution shift; action-space drift (v2 only) | None — different objectives mean different overfit failures |

A VLA's visual tower may **incidentally** acquire instance
discrimination if the action-prediction loss correlates strongly
with target identity, but the literature is clear that
action-prediction does not guarantee zero-shot
instance-discrimination ([OpenVLA §4 fine-tune ablations](https://arxiv.org/abs/2406.09246)
report that OpenVLA's frozen visual tower scores below SigLIP
on object-classification-style retrieval; the action head's
gradients pull representations toward control, not toward
contrastive separation). The cascade tripwire wants the second
behavior, not the first.

**Shared infrastructure, separate models.** The two briefs
genuinely overlap in two places:

1. **Dataset pipeline.** Both consume the trajectory-first
   speaker corpus from
   [`trajectory-first-captioning`](../../active/harness/trajectory-first-captioning.md)
   and the canonical schema from
   [`behavior-cloning-data-expansion`](../../active/harness/behavior-cloning-data-expansion.md).
   This brief commits to **sharing the dataset assembly tooling
   with v2** — the loader at
   `source/strafer_lab/strafer_lab/tools/dataset_export.py`
   gains a `dump_contrastive_pairs()` method alongside v2's
   `build_vla_dataset.py`, and both consume the same on-disk
   schema.

2. **Backbone choice.** If v2 ships first with DINOv2+SigLIP
   (OpenVLA's default), this brief's fine-tune should target
   the **same backbone** rather than diverge to a different
   visual tower. Sharing the backbone keeps the runtime memory
   footprint manageable (one tower loaded, two heads consume it)
   and makes the cascade-end-to-end vs. v2-end-to-end ablation
   comparable. The
   [`backbone-bakeoff`](backbone-bakeoff.md) brief is the
   coordinated decision point; both briefs reference its
   chosen backbone at execution time.

**What this brief is NOT doing:** training a VLA. Step A and
Step B both produce a CLIP-style tower with contrastive heads;
no action head, no `cmd_vel` decoder. If the cascade
underdelivers even after Step A + Step B, the escalation path
is v2 (end-to-end VLA), not "make this brief produce one."

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

Backbone: **inherited from
[`backbone-bakeoff`](backbone-bakeoff.md)** — defaults to
whatever that brief selected for production at the time this
brief picks up. If the bakeoff has not shipped, the fallback is
OpenCLIP ViT-B/32 + `laion2b_s34b_b79k` weights (the existing
fine-tune init), but the brief execution must record the
divergence in the training-config sidecar and run a one-shot
backbone-comparison head-to-head before committing the
Phase-1 ship decision. LoRA adapters on attention + projection
layers; full towers stay frozen by default. ~50 M trainable
params via LoRA vs. ~150 M for full fine-tune (numbers scale
with the chosen backbone). Choice between LoRA and full is a
brief-execution decision; document it.

**Deployment:** the resulting `clip_visual.onnx` +
`clip_text.onnx` drop in via the existing
[`clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py)
ONNX-loading path. No runtime code changes needed for Step A
alone.

**Acceptance:** re-run the cascade statistics from
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
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

**Retrieval-pool-size augmentation (deploy-distribution match).**
The training-time retrieval pool is the full harness corpus
(minus the held-out trajectory). The *deployment-time* retrieval
pool is whatever the SemanticMapManager has accumulated **on
this house so far** — empty on a fresh deployment, ~3 – 5
nodes after the first scan, growing to ~50 – 200 nodes across
typical operation. Training only against a fully-populated pool
produces a cross-attention layer that **collapses** when the
pool is small, because the attention pattern that was useful at
K = 8 with a dense index has no analogue at K = 0 (no keys to
attend over) or K = 1 (degenerate softmax).

Mitigation, drawn from the retrieval-aware-training pattern in
[Atlas (Izacard et al., 2022)](https://arxiv.org/abs/2208.03299)
and [RA-DIT (Lin et al., ICLR 2024)](https://arxiv.org/abs/2310.01352):
**vary the retrieved-pool size during training**. For each
training batch sample a `K_train ∈ {0, 1, 2, 4, 8}` uniformly,
truncate the retrieved set to that size (or pad with learned
"no-retrieval" tokens at K = 0), and forward through the
cross-attention layer. The model learns to function across the
full pool-size range; the inference path picks `K` per the
SemanticMapManager's current node count.

The combined-system acceptance gains an additional
**cold-deployment-eval bullet**: run the v2 cascade statistics
with the retrieval index **forcibly empty** (`map_state = cold`
per the validator-evaluation brief's disaggregation). The
cascade-end-to-end ROC-AUC must **degrade gracefully** to
Step-A-alone performance (within one CI-width); if Step B's
cold-eval falls *below* Step A's bare-encoder performance, the
cross-attention layer has overfit to the warm-map regime and
ships disabled until the augmentation is re-tuned.

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
      [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md).
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

- [ ] **Backbone inherited from [`backbone-bakeoff`](backbone-bakeoff.md).**
      The visual + text tower this brief fine-tunes is whichever
      the bakeoff selects. The Phase-1 init weights come from
      that brief's chosen public checkpoint (e.g.,
      `dinov3-small`, `siglip-2-base`, or `mobileclip-2-s`).
      If the bakeoff has not shipped at brief pickup, default
      to whatever
      [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
      shipped to production and document the divergence in the
      training-config sidecar.
- [ ] **Dataset pipeline shared with v2.** Both Step A's
      contrastive-pairs loader and
      [`vla-v2-architecture`](../experimental/vla-v2-architecture.md)'s
      action-tuple loader live alongside each other in
      `source/strafer_lab/strafer_lab/tools/dataset_export.py`.
      Adding `dump_contrastive_pairs()` does not break the
      existing `export_clip_csv()` consumers; smoke-test by
      running both loaders against the same harness episode
      directory.
- [ ] **Retrieval-pool-size augmentation (Step B).** Training
      samples `K_train ∈ {0, 1, 2, 4, 8}` uniformly per batch
      and truncates the retrieval pool accordingly. The model
      consumes a learned `no-retrieval` token at K = 0 so the
      forward pass is well-defined on a cold map.
- [ ] **Cold-deployment eval (Step B).** Combined-system
      statistics are re-run with the SemanticMapManager
      forcibly cold (no prior nodes). Cascade-end-to-end
      ROC-AUC must degrade to within one CI-width of
      Step-A-alone on cold maps; below that floor, the
      cross-attention layer ships disabled until the
      augmentation is re-tuned. Reported in the §4.5 addendum
      as a "cold vs. warm" disaggregated row alongside the
      v1 baseline / Step A / Step A+B columns.
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
  [`trajectory-first-captioning`](../../active/harness/trajectory-first-captioning.md).
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
  [`vla-v2-architecture`](../experimental/vla-v2-architecture.md)
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
- **Choosing the visual / text backbone.** Backbone selection
  (OpenCLIP ViT-B/32 vs. DINOv3-S vs. SigLIP-2-B vs.
  MobileCLIP-2-S vs. ...) lives in
  [`backbone-bakeoff`](backbone-bakeoff.md), filed parked and
  triggered when
  [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
  ships. This brief inherits whichever backbone the bakeoff
  selects; it does not re-run the comparison.
