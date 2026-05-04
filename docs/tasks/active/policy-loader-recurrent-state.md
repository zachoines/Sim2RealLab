# `load_policy()`: recurrent / stateful artifact support

**Type:** task / feature
**Owner:** Either (`strafer_shared` shared boundary — append-friendly
edits to `policy_interface.py`; coordinate with the operator if a
breaking signature change is needed)
**Priority:** P1 — without this, exported recurrent policies (DEPTH
GRU, NoCam LSTM, future RNN variants) load but cannot be driven
deterministically on the Jetson, blocking the strafer-inference
DEPTH MVP.
**Estimate:** S–M (~1 day: extend the loader return type, add an
ONNX recurrent path, episode-boundary `reset()` plumbing, tests).
**Branch:** task/policy-loader-recurrent-state

## Story

As a **Jetson inference-node operator running an exported recurrent
policy**, I want **`strafer_shared.policy_interface.load_policy()` to
return an object that exposes a `.reset()` hook and that, for ONNX
artifacts, properly threads hidden state across calls**, so that **the
inference node can clear hidden state at episode boundaries / mission
starts and the policy produces deterministic actions instead of
drifting under uncontrolled hidden-state evolution**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
  — particularly the **Shared boundary** section: `strafer_shared`
  edits should be append-friendly; coordinate any signature changes
  with the operator.
- [policy-export-tooling.md](policy-export-tooling.md) — the producer
  side. Sidecar already records `is_recurrent` and (post-
  [`policy-export-onnx-depth.md`](policy-export-onnx-depth.md)) ONNX
  artifacts will carry `(obs, h_in[, c_in]) → (actions, h_out[, c_out])`.
- [strafer-inference-package.md](strafer-inference-package.md) — the
  consumer that calls `load_policy()` at startup and asserts
  determinism per tick.

## Context

### What already exists

[`source/strafer_shared/strafer_shared/policy_interface.py`](../../../source/strafer_shared/strafer_shared/policy_interface.py)
ships `load_policy(path, variant)` returning a Python callable
`(obs) → action` for both `.pt` and `.onnx` formats. The loader is
stateless by contract — no `.reset()` exposed, no hidden-state I/O.

### What's missing

1. **No `.reset()` on the returned callable.** The exported recurrent
   TorchScript module (rsl_rl's `_TorchGRUModel` / `_TorchLSTMModel`
   and the strafer-side `_DepthGRUExportModel`) carries hidden state
   in a `register_buffer` and exposes `.reset()` to zero it. The
   loader returns a closure that hides the underlying module, so the
   inference node has no way to reset hidden state at episode
   boundaries.
2. **No support for the multi-input recurrent ONNX signature.**
   rsl_rl's `_OnnxRNNModel` (and the forthcoming
   `_OnnxDepthGRUModel`) advertise inputs `(obs, h_in[, c_in])` and
   outputs `(actions, h_out[, c_out])`. Today's `_infer_onnx`
   ignores everything past the first input/output.

These limitations are silently broken right now: a TorchScript
recurrent export *does* load, but consecutive same-obs calls give
different actions because hidden state evolves uncontrolled — failing
the determinism contract the inference brief asserts. ONNX recurrent
exports load with the wrong shape and either error or return
nonsensical actions.

## Approach

### Phase 1 — Loader return type

Replace the bare-callable return type with a small `LoadedPolicy`
class (still callable, so existing call sites keep working):

```python
class LoadedPolicy:
    def __call__(self, obs: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...  # no-op for stateless artifacts
    is_recurrent: bool
```

`load_policy()` returns `LoadedPolicy`. Stateless `.pt` / `.onnx`
artifacts get a `LoadedPolicy` whose `reset()` is a no-op and whose
`is_recurrent` is `False`. Recurrent artifacts get a subclass /
instance whose `reset()` zeros the underlying state.

`is_recurrent` is read from the sidecar JSON when present
(`<model>.json`); when the sidecar is missing (legacy artifacts),
infer from the loaded module's exposed `reset` method (TorchScript)
or from the ONNX graph's input names (presence of `h_in`).

### Phase 2 — TorchScript recurrent path

For `.pt` whose loaded module has a callable `reset` attribute:
- `LoadedPolicy.__call__` runs the module forward as today.
- `LoadedPolicy.reset()` calls `model.reset()`.

This is the simpler half — the recurrent state lives inside the
scripted module already, so the loader just exposes it.

### Phase 3 — ONNX recurrent path

For `.onnx` with multi-input signature:
- On load, inspect `sess.get_inputs()` for `h_in` (and `c_in` if
  present) — those name the recurrent state ports.
- Allocate persistent NumPy buffers for `h_in` (and `c_in`) sized from
  the ONNX input shapes.
- `__call__(obs)` feeds `obs` + the cached hidden state, captures the
  returned `h_out` (`c_out`) and writes back into the buffers.
- `reset()` zeros the buffers.

Determinism contract: with `h_in = 0`, two consecutive calls feeding
`(obs, h_in=0)` produce byte-identical `actions`. The loader must
enforce this on first load via the same probe the export tooling uses
at write time.

### Phase 4 — Inference-node call-site updates

The `strafer-inference-package` brief's Phase 3 already plans a
determinism unit test and an episode-boundary `reset()` semantics.
This brief makes the surface available; the inference brief's PR is
where it actually gets called from `inference_node.py`. Update the
inference brief in the same commit if the contract surface (method
names, return type) ends up different from what that brief assumed.

### Phase 5 — Tests

Extend
[`source/strafer_lab/tests/test_export_policy.py`](../../../source/strafer_lab/tests/test_export_policy.py)
or add a sibling `test_load_policy.py` (the strafer_shared lane gains
no test infra otherwise; piggybacking on the export-tooling test dir
keeps things in one place):

- Stateless `.pt` round-trip: `LoadedPolicy.is_recurrent is False`,
  `reset()` is a no-op, two same-obs calls byte-identical.
- Stateless `.onnx` round-trip: same.
- Recurrent `.pt` round-trip (build a tiny GRU-shaped dummy via
  `nn.GRU` + a register_buffer hidden state): two same-obs calls
  byte-identical *with `reset()` between them*; *different* without.
- Recurrent `.onnx` round-trip: same shape, with `h_in` cached
  between calls. Reset zeros it; same-obs after reset is
  byte-identical.

## Acceptance criteria

### Build / structure

- [ ] `load_policy(path, variant)` returns an object that is callable
      and exposes `.reset()` and `.is_recurrent`. Existing call sites
      that did `policy = load_policy(...); policy(obs)` continue to
      work unchanged.
- [ ] `python -m pytest source/strafer_lab/tests/test_export_policy.py
      [test_load_policy.py]` covers stateless + recurrent paths for
      both formats.

### Determinism contract

- [ ] **Stateless artifacts**: two same-obs calls byte-identical. No
      regression.
- [ ] **Recurrent TorchScript**: with `policy.reset()` between calls,
      two same-obs calls byte-identical. Without `reset()`, behavior
      reflects hidden-state evolution (this isn't a bug — it's the
      semantic the inference node uses for tick-to-tick state).
- [ ] **Recurrent ONNX**: same as TorchScript above; `reset()` zeros
      the cached `h_in` (`c_in`) buffers.

### Sidecar consumption

- [ ] When `<model>.json` is present, `is_recurrent` and `policy_variant`
      are read from it. Mismatch between sidecar variant and the
      `variant` arg passed to `load_policy()` raises a clear startup
      error (the cross-brief invariant the inference node enforces —
      enforce it at the loader so all consumers benefit).

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit. In particular,
      [`strafer-inference-package.md`](strafer-inference-package.md)'s
      Phase 3 references the loader contract — update it if the
      surface differs from what that brief assumed.

## Investigation pointers

- [`source/strafer_shared/strafer_shared/policy_interface.py`](../../../source/strafer_shared/strafer_shared/policy_interface.py)
  `load_policy()` (lines 155–202), `benchmark_policy()` (lines 211+).
  The new wrapper class lives alongside these.
- `rsl_rl.models.rnn_model._OnnxRNNModel` (vendored under
  `env_isaaclab3`) — defines the canonical recurrent ONNX I/O shape
  the loader must consume.
- [`Scripts/export_policy.py`](../../../Scripts/export_policy.py)
  `_verify_torchscript_determinism()` and `_verify_onnx_determinism()`
  — the export-time round-trip checks. The loader's runtime checks
  should mirror their semantics so divergence is impossible.

## Out of scope

- **The producer of recurrent ONNX artifacts.** Filed as
  [`policy-export-onnx-depth.md`](policy-export-onnx-depth.md) (DGX
  lane). This brief is the *consumer* side: it can be authored before
  that brief ships, since stateless ONNX + recurrent TorchScript paths
  are independently useful.
- **Renaming or removing existing constants in `strafer_shared`.** The
  shared-boundary rule says no removals without operator coordination.
  Prefer additive APIs (a new class beside the existing function, with
  the function preserved as a thin wrapper if needed).
- **Inference-node integration of `reset()` at episode boundaries.**
  That's [`strafer-inference-package.md`](strafer-inference-package.md)
  Phase 3's responsibility; this brief makes the surface available,
  the inference brief calls it.
- **Multi-env / batched inference.** The exported recurrent modules
  are deployment-shape (batch size 1). Anything past `(1, obs_dim)` is
  a separate concern — file a follow-up brief if it ever comes up.
