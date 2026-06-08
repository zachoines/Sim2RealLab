# Migrate policy export off deprecated TorchScript / legacy-ONNX APIs

**Type:** task / trained-policy (forward-compat / tech-debt)
**Owner:** DGX primary (`source/strafer_lab/scripts/export_policy.py` + the `strafer_shared.policy_interface` loader). Crosses into the Jetson runtime — see Out of scope.
**Priority:** P3 today (legacy path still works, warnings only); **P2 / blocking** once a torch upgrade removes the legacy APIs — gated by [`isaac-lab-upgrade`](../tooling/isaac-lab-upgrade.md).
**Estimate:** M–L — re-export both variants + re-validate the determinism / recurrent / cross-format-parity contracts + Jetson load + TRT-EP.
**Branch:** task/policy-export-deprecation-migration

## Story

As **the policy export/deploy chain owner**, I want **policy export off the
deprecated `torch.jit.*` / legacy `torch.onnx.export` APIs**, so that **a
future torch (post Isaac-Lab bump) that drops them doesn't break export —
while the artifacts stay loadable by the Jetson inference node.**

## Context bundle

- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md) —
  this crosses DGX (export) / shared (loader) / Jetson (runtime).
- [`context/recurrent-policy-contract.md`](../../context/recurrent-policy-contract.md)
  — the hidden-state contract the migration must preserve.
- Gated by [`isaac-lab-upgrade`](../tooling/isaac-lab-upgrade.md).

## Context (measured)

PyTorch 2.9+ deprecates the TorchScript and legacy-TorchScript-ONNX export
APIs (warnings now, removal later). Our export tooling sits squarely on them:

- `export_policy.py::export_torchscript` → `torch.jit.script(module)` → `<output>.pt`.
- `export_policy.py::export_onnx` → `torch.onnx.export(..., opset_version=18)`
  (the legacy TorchScript exporter) → `<output>.onnx`.
- Determinism verify → `torch.jit.load`.
- Loader `strafer_shared.policy_interface.load_policy` → `.pt` via
  `torch.jit.load`, `.onnx` via `onnxruntime.InferenceSession`.

These artifacts are **deploy contracts**, not internal: the Jetson inference
node loads the `.pt` via `torch.jit.load` and the `.onnx` (opset 18) via ONNX
Runtime's CUDA/TRT execution providers. So the migration can't just flip the
exporter — whatever it produces must remain loadable by `load_policy` + the
Jetson runtime, and must preserve:

- **byte-identical determinism** (two same-obs calls identical — the inference
  node asserts this at startup),
- the **recurrent hidden-state contract** (`(obs, h_in) -> (actions, h_out)`
  threading, reset semantics),
- **cross-format `.pt` ↔ `.onnx` parity**.

The guards already exist (and are exactly the files emitting these deprecation
warnings today): `tests/policy_tooling/test_export_policy.py`,
`tests/policy_tooling/test_load_policy.py`,
`tests/contracts/test_recurrent_contract_e2e.py`.

## Approach

- Evaluate the replacements per artifact:
  - **`.onnx`**: move to the new `torch.export` / dynamo ONNX exporter; pick an
    opset the Jetson's ORT / TRT-EP supports; keep the recurrent multi-input
    signature.
  - **`.pt`**: decide whether TorchScript stays (if `torch.jit.load`'s removal
    lags the exporter's) or moves to `torch.export`'s save format — coupled to
    what the Jetson loader can consume.
- Update `load_policy` + the Jetson consumer in lockstep with the new format.
- Re-validate the three contracts above + re-run a Jetson load + TRT-EP
  latency check.

## Acceptance

- [ ] `export_policy.py` produces `.pt` + `.onnx` (or their successors) with
      **no** `torch.jit.*` / legacy-`torch.onnx.export` calls — the
      deprecation warnings in `tests/policy_tooling/` +
      `tests/contracts/test_recurrent_contract_e2e.py` are gone.
- [ ] Artifacts load via `strafer_shared.policy_interface.load_policy`;
      determinism + recurrent + cross-format-parity tests green.
- [ ] Per-format decision recorded (e.g. `.pt` stays TorchScript if its
      deprecation lags; `.onnx` moves to the new exporter).
- [ ] If your work invalidates a fact in any context module, README, or guide,
      update it in the same commit.

## Out of scope

- **The Jetson runtime re-verification** (inference-node load + TRT-EP
  on-device) — peer-side; this brief lands the DGX export + shared loader, the
  Jetson lane re-verifies on hardware. Flag in the end-of-session summary.
- The Pillow 16-bit-PNG depth-sidecar deprecation — a separate effort (moving
  the LeRobot depth sidecar to a native video format).
- The Isaac Lab bump itself ([`isaac-lab-upgrade`](../tooling/isaac-lab-upgrade.md));
  this brief consumes its torch as the trigger.

## Triggered by

Deprecation warnings surfaced while reviewing the pure-Python test output
(test-tree-unification PR): the export tooling rides `torch.jit.script` /
`torch.jit.trace` / legacy `torch.onnx.export`, all deprecated in torch 2.9+.
Filed proactively so the eventual torch bump (`isaac-lab-upgrade`) doesn't
break the deploy artifact chain.
