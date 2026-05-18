# DeFM antialias preprocessing — measure deployment vs training delta and decide alignment

**Type:** investigation
**Owner:** DGX (`strafer_lab` lane)
**Priority:** P3 — depth images on this robot are nearly piecewise-smooth,
so the embedding delta is bounded; this brief quantifies the bound and
decides whether to (a) leave preprocessing asymmetric, (b) align
deployment to training (re-enable antialias in deployment), or (c)
align training to deployment (turn off antialias in training and
re-train the projection head). Filed off
[`export-onnx-depth`](../../completed/export-onnx-depth.md)'s
landing — the ONNX-side preprocessing disables antialiasing because
`aten::_upsample_bilinear2d_aa` isn't supported through opset 21.
**Estimate:** S–M (~½–1 day: extract held-out depth corpus, run both
preprocessings through the trained projection head, write up the
embedding-delta histogram + a decision).
**Branch:** task/defm-preprocess-antialias-audit

## Story

As a **maintainer of the DEPTH policy export path**, I want **a
measured delta between the trained DeFM preprocessing (antialiased
`torchvision.transforms.v2.Resize`) and the ONNX-safe deployment
preprocessing (`F.interpolate(..., antialias=False)`)**, so that **the
decision to ship asymmetric pipelines is anchored to a number rather
than a hand-wave**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [export-onnx-depth.md](../../completed/export-onnx-depth.md) — the
  shipped ONNX path that introduced the asymmetry. Read
  `_onnx_safe_defm_preprocess` and the inline comment about the
  precision delta.

## Context

### Symptom

`DeFMDepthEncoder.forward()` runs `defm.utils.utils.preprocess_depth_batch`,
which builds its norm transform via
`torchvision.transforms.v2.Resize((224, 224))` with the default
`antialias=True`. That traces to `aten::_upsample_bilinear2d_aa`, which
`torch.onnx.export` doesn't support at any opset through 21.

The shipped workaround in
[`_onnx_safe_defm_preprocess`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py)
substitutes `F.interpolate(mode='bilinear', antialias=False)` —
ONNX-clean, but numerically different on inputs with spatial
high-frequency content.

### What state of the art does

- TensorRT 8/10 + ORT TRT-EP do not support antialiased bilinear
  upsampling natively; production deployments commonly drop antialias
  for ONNX/TRT export. The cost is documented (see
  [PyTorch issue #142854](https://github.com/pytorch/pytorch/issues/142854)
  for the upstream limitation summary).
- Anti-aliased vs non-antialiased bilinear differ most on textured /
  high-frequency content; depth images of indoor surfaces (walls,
  floors, large planar objects) have substantially less
  spatial-frequency content than RGB.

### What we don't know

- The actual L2 / cosine delta in the 128-d projection space on a
  representative held-out depth corpus from harness rollouts.
- Whether the delta is uniform across distance ranges (close-up clutter
  vs open hallway).
- Whether the downstream MLP + GRU policy is sensitive enough to that
  delta to change actions meaningfully on the deployment loop.

## Approach

### Phase 1 — Extract held-out depth corpus

Pick a representative sample of depth images from existing harness
output (or from a one-shot `Scripts/test_strafer_env.py` capture if
no harness corpus is on disk). Target ~500–2000 frames spanning
indoor / hallway / cluttered scenes — proportional to the
training distribution.

### Phase 2 — Measure projection-space delta

For each frame, run:
1. `DeFMDepthEncoder.forward()` with the original DeFM antialiased
   preprocess → embedding `e_train` ∈ ℝ^128.
2. `_OnnxSafeDeFMDepthEncoder.forward()` with the deployment
   non-antialiased preprocess → embedding `e_deploy` ∈ ℝ^128.

Report distributions of:
- L2 norm `||e_train - e_deploy||_2`.
- Cosine similarity `cos(e_train, e_deploy)`.
- Per-distance-bucket breakdown (group by mean depth in the frame).

Embed the histograms in the brief's write-up.

### Phase 3 — Sensitivity-check the policy

If the embedding delta is non-trivial, run the trained DEPTH GRU
policy with both embeddings on a fixed observation trace and report
the per-action delta (L_inf on the 3-d normalized action vector).
Threshold for "matters": `> 0.01` (~1% of normalized action range)
sustained across multiple frames in the same episode.

### Phase 4 — Decision + follow-up

Based on the measurement, pick one of:

- **(a) Leave asymmetric.** Document the delta in the
  strafer_lab README's deferred section as "measured negligible" and
  call this audit shipped.
- **(b) Align deployment to training.** File a brief to ship a custom
  ONNX op or a `torch.export`-based exporter path that supports
  antialiased upsample. Likely needs the new
  `torch.onnx.export(..., dynamo=True)` path; estimate L.
- **(c) Align training to deployment.** Re-train the DEPTH baseline
  with `antialias=False` in the preprocessing pipeline. File a brief
  for the retraining run; estimate L.

## Acceptance criteria

- [ ] A written report (added to this brief as a `## Findings`
      section or shipped as a new `docs/` note) with the embedding
      delta distributions and the policy-action sensitivity number.
- [ ] A decision (a / b / c) recorded, with the follow-up brief
      filed if the decision is (b) or (c).
- [ ] If decision is (a): the deferred-section entry in the
      strafer_lab README is updated to cite the measured number, not
      the hand-wave "small precision delta."
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py)
  — `_onnx_safe_defm_preprocess` and the inline note this audit
  resolves.
- `~/.cache/torch/hub/leggedrobotics_defm_main/defm/utils/utils.py`
  `preprocess_depth_batch` + `make_norm_transform` — the training-time
  pipeline this audit compares against.
- [PyTorch issue #142854](https://github.com/pytorch/pytorch/issues/142854)
  — upstream tracking for the antialiased-upsample ONNX export gap.

## Out of scope

- **Implementing alignment.** This brief decides; follow-ups
  implement. Don't ship (b) or (c)'s code change here.
- **Re-evaluating DeFM choice.** Backbone bake-offs are
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md);
  this brief assumes the trained DeFM checkpoint.
