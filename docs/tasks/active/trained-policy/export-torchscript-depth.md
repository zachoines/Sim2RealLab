# DEPTH TorchScript export — work around DeFM `BiFPN` scriptability

**Type:** bug / refactor
**Owner:** DGX (`strafer_lab` lane)
**Priority:** P2 — DEPTH ONNX ships via [`export-onnx-depth`](export-onnx-depth.md);
TorchScript is the redundant deployment path (CPU-EP fallback on the
Jetson). The brief still has consumers — `play_strafer_navigation.py`'s
`--policy_path` smoke test currently expects a `.pt` for sim-side
verification of the exported artifact, and `strafer_shared.policy_interface
.load_policy()` accepts both formats. Lift to P1 if `load_policy()` ends
up needing a `.pt` round-trip in `loader-recurrent-state`.
**Estimate:** S–M (~½–1 day: replace the DeFM backbone in
`_DepthGRUExportModel` with a traced copy so `torch.jit.script` doesn't
recurse into `BiFPN`, add a real-checkpoint test, re-run
`Scripts/export_policy.py --variant DEPTH --formats pt,onnx`).
**Branch:** task/export-torchscript-depth

## Story

As an **operator promoting a trained DEPTH PPO checkpoint to robot
deployment**, I want **`Scripts/export_policy.py --variant DEPTH
--formats pt,onnx` to land both artifacts**, so that **the sim-side
TorchScript smoke test in `play_strafer_navigation.py --policy_path` is
exercisable against a real DEPTH checkpoint instead of only the ONNX
path**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [export-onnx-depth.md](export-onnx-depth.md) — sibling brief that
  shipped the ONNX path and the export-pipeline plumbing; the same
  encoder-substitution pattern (`_OnnxSafeDeFMDepthEncoder`) is the
  template for this brief's TorchScript-safe variant.

## Context

### Symptom

`Scripts/export_policy.py --variant DEPTH --formats pt` against a real
DEPTH checkpoint (e.g.
`logs/rsl_rl/strafer_navigation/run_20260504_053331/model_1000.pt`)
fails inside `torch.jit.script(_DepthGRUExportModel)` with:

```
RuntimeError: Arguments for call are not valid.
  ...
The original call is:
  File "/.../torch/hub/leggedrobotics_defm_main/defm/models/bifpn.py", line 22
        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-6)
        return sum(w[i] * f for i, f in enumerate(features))
               ~~~ <--- HERE
```

TorchScript cannot infer the return type of `sum(generator)` over
tensors. DeFM's `BiFPN.WeightedFusion.forward` lives in the `torch.hub`
cache (upstream `leggedrobotics/defm`), so a direct upstream patch is
fragile (next `hub.load(force_reload=True)` reverts it).

The ONNX-export path doesn't see this — `torch.onnx.export` *traces*
the graph rather than *scripting* it; generators evaluate concretely
during the trace. So ONNX ships fine and TorchScript is the residual
gap.

### Why this matters

- `Scripts/play_strafer_navigation.py --policy_path <.pt>` is the
  sim-side smoke test for verifying the export didn't drift; without a
  working DEPTH TorchScript artifact, this fast-feedback loop is
  missing on the same robot config the Jetson deploys.
- `strafer_shared.policy_interface.load_policy()` accepts both formats;
  a DGX side check that doesn't require ONNX Runtime is operationally
  useful.

### Why not "just file an upstream PR to DeFM"

- Rebuilding via `torch.hub.load(force_reload=True)` replaces the patch
  on every fresh checkout. The local Strafer repo doesn't pin a DeFM
  fork.
- TorchScript-trace-not-script for the frozen backbone is the standard
  workaround pattern; it's already how `_OnnxSafeDeFMDepthEncoder` in
  `depth_rnn_model.py` handles the corresponding ONNX-side limitation
  (antialiased resize). Symmetric solution.

## Approach

In
[`source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py):

1. Add `_TorchSafeDeFMDepthEncoder` (or extend the existing
   `_OnnxSafeDeFMDepthEncoder` pattern) that replaces the DeFM backbone
   with `torch.jit.trace(backbone, dummy_3ch_224x224)` on `__init__`.
   The traced backbone is a fixed-shape concrete graph that bypasses
   the un-scriptable `sum(generator)` because the trace records the
   concrete computation.
2. Substitute the encoder inside `_DepthGRUExportModel.__init__` only
   when the underlying encoder is a `DeFMDepthEncoder`; the legacy
   CNN encoder is scriptable as-is.
3. Carry the runtime preprocessing the same way as the ONNX side does
   — either reuse `_onnx_safe_defm_preprocess` (already
   trace-friendly) or wrap DeFM's `preprocess_depth_batch` via
   `torch.jit.trace` if matching training-time pipeline precisely
   matters. Note that aligning preprocessing across `.pt` and `.onnx`
   is the easier-to-reason-about default.

## Acceptance criteria

- [ ] `Scripts/export_policy.py --variant DEPTH --formats pt,onnx`
      against a real DEPTH checkpoint writes both artifacts + sidecar
      end-to-end, on the DGX in `env_isaaclab3`.
- [ ] The exported `.pt` round-trips through
      `strafer_shared.policy_interface.load_policy()` and produces
      deterministic actions (matches the export-time round-trip the
      sibling brief established for ONNX).
- [ ] `python -m pytest source/strafer_lab/tests/test_export_policy.py`
      passes — including a new test exercising the
      `_TorchSafeDeFMDepthEncoder` path against a tiny synthetic
      DeFM-shaped stub.
- [ ] `Scripts/play_strafer_navigation.py --policy_path <exported.pt>`
      runs one rollout without raising (sim-side smoke; deterministic
      sample obs / fixed seed).
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py)
  — `_DepthGRUExportModel` (TorchScript path) and the ONNX-side
  `_OnnxSafeDeFMDepthEncoder` whose trace-the-backbone pattern this
  brief mirrors.
- `~/.cache/torch/hub/leggedrobotics_defm_main/defm/models/bifpn.py`
  line 22 — the un-scriptable `sum(generator)` over a list of
  tensors. Read once, don't patch in place — upstream owns this file.
- [`Scripts/export_policy.py`](../../../../Scripts/export_policy.py)
  `export_torchscript()` and `_verify_torchscript_determinism()` —
  the round-trip helpers the new artifact must satisfy.
- [`Scripts/play_strafer_navigation.py`](../../../../Scripts/play_strafer_navigation.py)
  `--policy_path` — the consumer this brief unblocks.

## Out of scope

- **Upstream DeFM patch.** Filing an upstream PR to
  `leggedrobotics/defm` is the long-term fix but not gating; the
  local workaround ships independently.
- **TorchScript LSTM support for DEPTH.** Current config uses GRU; if
  a future config switches to LSTM, this brief's wrapper raises and
  a follow-up adds the LSTM path. Don't speculatively implement it.
- **Aligning preprocessing precision with training.** That's
  [`defm-preprocess-antialias-audit`](../investigations/defm-preprocess-antialias-audit.md);
  this brief reuses whatever preprocessing the ONNX side ships.
