# DEPTH-aware actor: ONNX export support for `StraferDepthRNNModel`

**Status:** Shipped 2026-05-18 (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/38
**Follow-ups:**
[`export-torchscript-depth`](export-torchscript-depth.md) — DEPTH TorchScript export on real checkpoints still fails on DeFM's `BiFPN` scriptability bug;
[`defm-preprocess-antialias-audit`](../active/investigations/defm-preprocess-antialias-audit.md) — measure projection-space delta between training-time antialiased preprocessing and the deployment ONNX-safe non-antialiased version, then decide alignment;
[`export-sidecar-training-preset`](../active/trained-policy/export-sidecar-training-preset.md) — sidecar `training_preset` records the configclass name instead of the rsl_rl preset variable.

**Type:** task / feature
**Owner:** DGX (`strafer_lab` lane — RL agent code)
**Priority:** P1 — gates the Jetson TRT-EP path for the DEPTH MVP in
[`inference-package`](inference-package.md). Without
ONNX, DEPTH inference on Jetson Orin Nano falls back to TorchScript-CPU,
which misses the latency target.
**Estimate:** M (~1–2 days: write `_OnnxDepthGRUModel` mirror of
`_DepthGRUExportModel`, integrate into `Scripts/export_policy.py`, add
round-trip + parity tests, rerun benchmark on a real checkpoint).
**Branch:** task/policy-export-onnx-depth

## Story

As an **operator promoting a trained DEPTH PPO checkpoint to robot
deployment**, I want **`Scripts/export_policy.py --variant DEPTH
--formats pt,onnx` to actually emit a working ONNX artifact**, so that
**the Jetson inference node can load it through the TensorRT execution
provider and meet the strafer-inference brief's ≤ 10 ms p95 end-to-end
latency target — instead of falling back to TorchScript-CPU and
plateauing well above the budget on the 4819-dim observation vector**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/recurrent-policy-contract.md](../context/recurrent-policy-contract.md) — the
  canonical spec for hidden-state shape, reset semantics, and the
  determinism contract this brief's producer side must satisfy. Read
  before writing `_OnnxDepthGRUModel`; the contract pins what `h_in`
  / `h_out` look like at the seam (points 1 + 3).
- [policy-export-tooling.md](policy-export-tooling.md) — the export
  tooling whose `--formats pt,onnx` path errors today on DEPTH
  variants because of the gap this brief closes.
- [strafer-inference-package.md](inference-package.md) — the
  Jetson consumer whose Phase 3 latency target gates on the TRT-EP
  path, which gates on this brief.

## Context

### What already exists

- The TorchScript export path for `StraferDepthRNNModel` is shipped:
  [`source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py)
  defines `_DepthGRUExportModel`, a scriptable wrapper that fuses the
  depth encoder + GRU + MLP + deterministic-output module into a single
  module with hidden state in a `register_buffer`.
- `Scripts/export_policy.py` already drives both TorchScript and ONNX
  through `policy_model.as_jit()` / `policy_model.as_onnx()` — the
  ONNX path works for `MLPModel`-derived NoCam policies.

### What's missing

`StraferDepthRNNModel.as_onnx()` raises `NotImplementedError`
([source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py:235-242](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py)):

```python
def as_onnx(self, verbose: bool = False) -> nn.Module:
    """Return an ONNX-exportable version of this model."""
    if not self._has_depth:
        return super().as_onnx(verbose)
    raise NotImplementedError(
        "ONNX export not yet implemented for StraferDepthRNNModel with depth. "
        "Use as_jit() for deployment."
    )
```

The blockers behind that `NotImplementedError`:

1. **The depth encoder.** `defm_efficientnet_b0` (per
   `STRAFER_PPO_DEPTH_RUNNER_CFG.actor.depth_encoder_type = "defm"`) is
   a `timm`-backed CNN with operations (depthwise convs, SE blocks,
   etc.) that may need opset-18 support. Need to verify the encoder is
   ONNX-traceable as-is and the resulting graph is consumable by both
   ORT-CPU (DGX validation) and ORT-TRT (Jetson deployment).

   **Sub-blockers under the DeFM encoder specifically**
   ([`depth_encoders.py:103-131`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_encoders.py)):

   - **Runtime preprocessing pipeline.** `DeFMDepthEncoder.forward()`
     calls `_preprocess(...)` (a function loaded via
     `_get_defm_preprocess()` which mutates `sys.path` at module load
     time to import from the `torch.hub` cache directory). This is
     not script-friendly; tracing should still work because the
     function is just a tensor pipeline at runtime, but verify the
     traced graph captures the upsample 60×80 → 224×224 + 3-channel
     log-normalization steps as expected. If the function contains
     Python-level control flow, trace will silently bake in the
     branch taken on the dummy input.
   - **Frozen backbone with dict output.** The backbone is wrapped in
     `with torch.no_grad():` and returns `out["global_backbone"]`. The
     tracer should follow the `global_backbone` key only; verify the
     resulting ONNX graph has the expected single-output backbone
     (not the full dict). The official DeFM repo
     ([leggedrobotics/defm](https://github.com/leggedrobotics/defm))
     does NOT publish ONNX export examples — verify locally on a
     1-batch dummy before relying on TensorRT consumption.
   - **EfficientNet-B0 SE blocks** require opset 17+ to export
     cleanly; `_DEFAULT_ONNX_OPSET = 18` in `export_policy.py` is
     sufficient.

2. **Recurrent I/O signature.** rsl_rl's stock `_OnnxRNNModel` uses
   `(obs, h_in[, c_in]) → (actions, h_out[, c_out])`. The DEPTH ONNX
   wrapper needs the same signature + the depth-encoder split inside
   `forward`, since hidden state can't live in an internal buffer once
   the model is ONNX (ONNX nodes are stateless by construction).

3. **The existing export pipeline can't handle multi-input ONNX
   today — and this is not DEPTH-specific.** `Scripts/export_policy.py
   :export_onnx` hardcodes
   `torch.onnx.export(module, (dummy_obs,), ..., input_names=["obs"],
   output_names=["actions"])` and `_verify_onnx_determinism` only
   feeds `sess.get_inputs()[0].name`. This means even the existing
   rsl_rl stock `_OnnxRNNModel` (used by `STRAFER_PPO_LSTM_RUNNER_CFG`
   for NoCam LSTM) would fail to export through this script — its
   `(obs, h_in, c_in)` signature wouldn't be satisfied by the
   hardcoded single-input dummy. The Phase 2 work in this brief
   therefore unlocks BOTH the DEPTH GRU path AND the NoCam LSTM path,
   not just DEPTH. Make the export pipeline consult
   `module.input_names` / `module.output_names` /
   `module.get_dummy_inputs()` when present, fall back to the current
   stateless signature otherwise.

4. **The deterministic head is Beta-distribution.** Unlike rsl_rl's
   stock Gaussian-based `_BetaDeterministicOutput` is composed of
   `softplus`, division, scalar multiply, scalar subtract — all
   standard ONNX ops in opset 17+. But the MLP outputs shape
   `[..., 2, output_dim]` (see
   [`distributions.py:96`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/distributions.py))
   which is reshaped from a flat `2 * output_dim` linear output. Verify
   the reshape exports as a concrete `Reshape` op, not an opaque
   `Constant + Reshape` chain that TensorRT might struggle with. Spot
   check the ONNX graph in Netron after export.

## Approach

### Phase 1 — Author `_OnnxDepthGRUModel`

In
[`source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py),
add `_OnnxDepthGRUModel` mirroring `_DepthGRUExportModel` but:

- `forward(obs, h_in)` → `(actions, h_out)` (GRU only — LSTM raises
  `NotImplementedError` until/unless we ever switch DEPTH to LSTM).
- `obs` shape `(1, scalar_obs_dim + depth_obs_dim)` flat — split into
  `scalar` and `depth`, encode depth, concat, normalize, GRU, MLP,
  deterministic head.
- Provide `get_dummy_inputs()`, `input_names = ["obs", "h_in"]`,
  `output_names = ["actions", "h_out"]` (matches rsl_rl's
  `_OnnxRNNModel` so `Scripts/export_policy.py` doesn't need a
  DEPTH-specific export path).
- Update `StraferDepthRNNModel.as_onnx()` to return the new wrapper
  for GRU + depth, raise for unsupported (LSTM, no-depth + non-MLPModel
  fallthroughs).

### Phase 2 — Update the export pipeline (applies to all recurrent ONNX, not just DEPTH)

`Scripts/export_policy.py` currently expects `as_onnx()` to return a
stateless module with `input_names = ["obs"]` /
`output_names = ["actions"]`. The recurrent ONNX wrapper has a
multi-input/output signature, so `export_onnx()` and the round-trip
verifier need to handle:

- `policy_model.as_onnx()` returning `_OnnxDepthGRUModel`
  (multi-input/output), an `_OnnxRNNModel` (multi-input/output — used
  today by the existing NoCam LSTM config that has no shipping ONNX
  path either), OR a stateless `_OnnxMLPModel`.
- ONNX export call: forward `dummy_inputs = onnx_module.get_dummy_inputs()`
  and `input_names` / `output_names` from the wrapper instead of
  hardcoding. The contract surface for both `_OnnxRNNModel` and the
  new `_OnnxDepthGRUModel` exposes these as attributes — consult the
  module rather than treating DEPTH as a special case.
- Round-trip verification needs to feed both `obs` and `h_in` (zeros)
  to `ort.InferenceSession.run(...)`, assert determinism with `h_in`
  reset between calls. Read the input/output names from
  `sess.get_inputs()` rather than hardcoding `"obs"`.

The `is_recurrent` flag in the sidecar already records the contract;
this brief just makes the artifact actually loadable.

The "fixes NoCam LSTM too" property is a free win, not a goal — the
NoCam LSTM ONNX path was never in active use because the existing
NoCam MLP config is faster on CPU. Don't expand scope to validate
NoCam LSTM end-to-end here; that's a separate brief if/when the LSTM
config becomes a deployment target.

### Phase 3 — Tests

Extend
[`source/strafer_lab/tests/policy_tooling/test_export_policy.py`](../../../source/strafer_lab/tests/policy_tooling/test_export_policy.py):

- Build a tiny `_OnnxDepthGRUModel`-shaped dummy (depth encoder = 1
  linear layer; GRU = `nn.GRU(...)`; MLP = `nn.Linear`). Verify ONNX
  export, sidecar marks `is_recurrent: true`, and round-trip determinism
  holds when `h_in` is reset between calls.
- Verify the multi-input ONNX file is loadable via ORT-CPU on the DGX
  (TRT EP verification stays Jetson-side per
  [`inference-package`](inference-package.md)).

### Phase 4 — Real-checkpoint smoke test

Once a deployable DEPTH checkpoint exists, run:

```
$ISAACLAB -p Scripts/export_policy.py \
    --checkpoint logs/rsl_rl/strafer_navigation/<run>/model_<step>.pt \
    --output models/strafer_depth_v0 \
    --variant DEPTH \
    --formats pt,onnx
```

Confirm both artifacts land, the sidecar has `formats: ["pt", "onnx"]`
+ `is_recurrent: true`, and the ONNX file passes the export-time
round-trip + benchmark on the DGX (CPU EP). Jetson TRT-EP latency
sweep happens after rsync per the strafer-inference brief.

## Acceptance criteria

### Build / structure

- [ ] `python -m pytest source/strafer_lab/tests/policy_tooling/test_export_policy.py`
      passes — including the new `_OnnxDepthGRUModel` tests.
- [ ] `StraferDepthRNNModel.as_onnx()` no longer raises
      `NotImplementedError` for the GRU + depth case; LSTM still raises
      with a clear message.
- [ ] `Scripts/export_policy.py --variant DEPTH --formats pt,onnx`
      succeeds end-to-end against a real DEPTH checkpoint and writes
      both artifacts + sidecar.

### Determinism contract

- [ ] Recurrent-ONNX round-trip: same `(obs, h_in=0)` fed twice through
      ORT-CPU produces byte-identical `actions` (and identical `h_out`),
      asserted at export time before the artifact is written.

### Cross-brief consumer parity

- [ ] The exported `.onnx` is loadable by ONNX Runtime via the CPU
      execution provider on the DGX. TRT EP loadability on the Jetson
      is owned by the strafer-inference brief's TRT-EP step (this
      brief produces the artifact; consumption is downstream).
- [ ] Sidecar metadata for DEPTH includes `formats: ["pt", "onnx"]`,
      `onnx_opset` (matches the value
      `Scripts/export_policy.py` chose), and `is_recurrent: true`.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context module
      or in [`policy-export-tooling.md`](policy-export-tooling.md)'s
      out-of-scope notes, update those in the same commit.

## Investigation pointers

- [`source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py)
  — `_DepthGRUExportModel` is the TorchScript reference; the new
  ONNX wrapper mirrors its forward path with hidden state lifted to
  inputs / outputs.
- [`source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_encoders.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_encoders.py)
  — `DeFMDepthEncoder` and `_get_defm_preprocess`; the runtime-loaded
  preprocessing function and the dict-returning frozen backbone. Trace
  this submodule alone first (`torch.onnx.export` on a 1-batch dummy
  depth tensor) to confirm the encoder is ONNX-exportable in
  isolation before integrating into the full DEPTH wrapper.
- `rsl_rl.models.rnn_model._OnnxRNNModel` (vendored at
  `~/miniconda3/envs/env_isaaclab3/lib/python3.12/site-packages/rsl_rl/models/rnn_model.py:180`)
  — the canonical recurrent-ONNX template (no depth encoder; same I/O
  shape). Exposes `get_dummy_inputs()`, `input_names`, `output_names`
  as the contract for `export_policy.py` to consume.
- `rsl_rl` upstream issue
  [isaac-sim/IsaacLab#3008](https://github.com/isaac-sim/IsaacLab/issues/3008)
  — historical GRU-export bug in Isaac Lab's exporter (hardcoded LSTM
  assumption). Confirm the pinned `rsl_rl` version on
  `env_isaaclab3` has the GRU fix; the in-repo `_DepthGRUExportModel`
  works around the TorchScript side already.
- [`Scripts/export_policy.py`](../../../Scripts/export_policy.py)
  `export_onnx()` and `_verify_onnx_determinism()` — the export and
  round-trip helpers that need to handle the multi-input signature.
- [`source/strafer_lab/strafer_lab/tasks/navigation/agents/distributions.py:114-124`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/distributions.py)
  — `_BetaDeterministicOutput`; the export-friendly Beta-mean module
  that sits at the head of the DEPTH ONNX graph.

## Out of scope

- **Pre-built `.engine` generation.** Recording the path in the
  sidecar (`tensorrt_engine_path` / `tensorrt_version`) is supported
  today; building the engine itself is operator-side via NVIDIA's
  `trtexec` against the JetPack-pinned TRT runtime, not part of this
  brief.
- **LSTM support for the DEPTH actor.** The current
  `STRAFER_PPO_DEPTH_RUNNER_CFG` uses GRU; if a future config switches
  to LSTM, this brief's `as_onnx()` raises and a follow-up adds the
  LSTM path. Don't speculatively implement it.
- **Loader-side stateful inference contract.** That's
  [`loader-recurrent-state`](loader-recurrent-state.md)
  (shipped). This brief produces the artifact; the loader brief consumes it.
- **Re-validating NoCam ONNX.** NoCam ONNX export already works through
  the rsl_rl stock path; don't refactor that surface as part of this
  brief.
- **DEPTH TorchScript export on real checkpoints.** The brief's Phase 4
  smoke test surfaced that `--formats pt` for DEPTH fails on
  `torch.jit.script` because DeFM's `BiFPN.WeightedFusion.forward` uses
  `sum(generator)`. ONNX traces (not scripts) so the brief's primary
  goal is unaffected; the TorchScript residual ships separately as
  [`export-torchscript-depth`](export-torchscript-depth.md).
- **Matching the antialiased DeFM preprocessing exactly.**
  `_onnx_safe_defm_preprocess` substitutes
  `F.interpolate(..., antialias=False)` for the torchvision Resize
  because `aten::_upsample_bilinear2d_aa` isn't ONNX-supported through
  opset 21. The bounded projection-space delta and the alignment
  decision (leave / align deployment / align training) are owned by
  [`defm-preprocess-antialias-audit`](../active/investigations/defm-preprocess-antialias-audit.md).
- **Sidecar `training_preset` correctness.** The smoke-test sidecar
  records the configclass name (`RslRlOnPolicyRunnerCfg`) instead of
  the rsl_rl preset variable (`STRAFER_PPO_DEPTH_RUNNER_CFG`); this is
  a pre-existing bug in `policy-export-tooling`'s call site, surfaced
  by this brief and filed separately as
  [`export-sidecar-training-preset`](../active/trained-policy/export-sidecar-training-preset.md).
