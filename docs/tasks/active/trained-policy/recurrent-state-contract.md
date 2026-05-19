# Pin the end-to-end recurrent hidden-state contract across train / export / inference

**Type:** docs / refactor
**Owner:** Either (anchored in `strafer_shared`, surfaces in `strafer_lab`
+ `strafer_inference`)
**Priority:** P1 — three sibling briefs
([`export-onnx-depth`](../../completed/export-onnx-depth.md),
[`loader-recurrent-state`](../../completed/loader-recurrent-state.md),
[`inference-package`](inference-package.md)) each describe *their side*
of the recurrent contract. The whole-pipeline shape (hidden-state
allocation, reset semantics, episode boundary, thread-safety, byte
identity expectations) is currently distributed across three documents
that nobody owns end-to-end. The first time the contract drifts will be
the first time recurrent DEPTH inference ships actions that look
plausible but are silently wrong.
**Estimate:** S–M (~1 day: write the spec doc into a `strafer_shared`
docstring + a `context/` module, file follow-up edits to the three
implementer briefs to reference it, and add a single end-to-end
determinism integration test that pins all three sides at once).
**Branch:** task/recurrent-state-contract

## Story

As a **maintainer touching any of the train / export / inference paths
for a recurrent policy**, I want **one canonical spec for how hidden
state is shaped, initialized, threaded across calls, and reset across
episode boundaries**, so that **the next agent who edits one of the
three sides can verify their change against a single authoritative
contract rather than reading three sibling briefs and guessing what the
other two assume**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [export-onnx-depth.md](../../completed/export-onnx-depth.md) — DGX-side producer of
  recurrent ONNX.
- [loader-recurrent-state.md](../../completed/loader-recurrent-state.md) — Either-side
  consumer that exposes `.reset()` and threads hidden state.
- [inference-package.md](inference-package.md) — Jetson-side caller
  that decides *when* `.reset()` fires.
- [policy-export-tooling.md](../../completed/policy-export-tooling.md) —
  shipped baseline; sidecar already records `is_recurrent`.

## Context

### The three handoffs today

```
┌────────────────┐   .pt / .onnx   ┌────────────────┐   load_policy()    ┌─────────────────────┐
│ DGX training   │ ───────────────▶│ DGX export     │ ──────────────────▶│ Jetson inference    │
│ (rsl_rl)       │   + sidecar     │ (Scripts/      │   + sidecar        │ (strafer_inference) │
│                │                 │  export_policy)│                    │                     │
└────────────────┘                 └────────────────┘                    └─────────────────────┘
   T training-time                    bytes on disk                          episode loop
```

Each handoff has assumptions about hidden-state shape, reset semantics,
and frame timing. Today those assumptions live in three different
briefs:

| Concern | Brief that owns it today | Failure mode if mis-aligned |
|---|---|---|
| Hidden-state tensor shape (layers × batch × hidden) | implicit in rsl_rl `_TorchGRUModel` / `_OnnxRNNModel`; not re-stated anywhere | Wrong-shape exception at inference startup — visible early, low impact |
| Initial state on first tick | TorchScript: zero buffer at module load. ONNX: zero numpy buffer in loader. | Aligned by convention but never asserted; drift would be silent |
| Per-tick state threading | TorchScript: in-module buffer. ONNX: loader-side persistent numpy. | If the loader forgets to write `h_out` back, the policy resets every tick → behaves like a feedforward MLP without diagnostic |
| Episode-boundary reset trigger | Inference brief Phase 3 says "episode boundaries / mission starts" but doesn't define what those are at the topic level | A mission with two `/strafer/goal` updates (VLM re-grounding mid-mission) could either be (a) treated as one episode w/ carried state — policy is biased by the previous goal — or (b) two episodes w/ reset between — policy starts cold on the new goal. Both are defensible; only one matches training. |
| Determinism contract | Export brief: byte-identical with `h_in=0`. Inference brief: byte-identical "across calls" but doesn't condition on reset. | Determinism unit test can pass at every layer while end-to-end determinism fails |
| Thread safety | TorchScript module is **stateful** via `register_buffer` — not thread-safe. Loader brief Phase 2 just exposes `.reset()`. Inference brief doesn't mention. | If inference node ever uses async callbacks / multi-threaded executor, parallel `model(obs)` calls corrupt hidden state silently |

### What state of the art does

- **RSL-RL upstream** (`rsl_rl.models.rnn_model`) defines
  `is_recurrent: bool`, `reset(dones)`, `get_hidden_state()`,
  `detach_hidden_state()`. The export wrappers (`_TorchGRUModel`,
  `_TorchLSTMModel`, `_OnnxRNNModel`) re-implement the contract for
  TorchScript and ONNX respectively. Hidden-state is `(num_layers, 1,
  hidden_size)` for batch-1 inference.
- **ANYmal locomotion** ([leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym))
  resets hidden state at every episode termination (`done=True`) during
  training and at every nav-action-server goal acceptance at
  deployment.
- **NavDP** ([arxiv:2505.08712](https://arxiv.org/html/2505.08712v2))
  uses Markovian single-frame design — no recurrent state, no
  episode-boundary question. Not directly applicable here but
  illustrates the alternative.
- **GR00T N1.7** [export pipeline](https://github.com/NVIDIA/Isaac-GR00T)
  exports full pipeline to ONNX + TensorRT; the deployed runtime
  treats one mission as one episode (state is freed when the policy
  server receives a new task goal).

### Why a spec brief, not more code

Each of the three implementer briefs is correct *as far as its scope
goes*. The gap is at the seams. Filing this as a contract doc + a single
end-to-end integration test means:

- One place a future maintainer reads to understand the whole picture.
- One test that catches a regression at any of the three layers (the
  loader brief's tests cover loader-only round-trip; the export brief's
  cover export-only round-trip; neither covers end-to-end).
- Cross-references that prevent the three implementer briefs from
  drifting out of sync as they get edited independently.

## Approach

### Phase 1 — Write the contract spec

In
[`source/strafer_shared/strafer_shared/policy_interface.py`](../../../../source/strafer_shared/strafer_shared/policy_interface.py),
add a module-level docstring section "Recurrent hidden-state contract"
covering:

1. **Hidden-state shape.** `(rnn_num_layers, 1, rnn_hidden_dim)` for
   batch-1 deployment inference. LSTM carries `(h, c)`; GRU carries `h`
   only. Numbers anchored to the rsl_rl runner config (e.g.
   `STRAFER_PPO_DEPTH_RUNNER_CFG.actor.rnn_hidden_dim=128`).
2. **Initial state.** Always zero on `load_policy()` return.
3. **Per-tick threading.** TorchScript: `register_buffer` is the
   owner; the loader holds a reference to the scripted module and
   never copies the buffer. ONNX: the loader owns the persistent
   `(h, [c])` numpy buffers and feeds them through `sess.run` each
   tick.
4. **Reset trigger.** The Jetson inference node calls
   `policy.reset()` exactly once per "mission episode," defined as:
   - On action server goal-accepted (`/navigate_to_pose` callback
     fires for a NEW goal handle).
   - On goal pose update mid-mission **if** `is_mid_mission_reset`
     in the inference config is True (default: True). Rationale: VLM
     regrounding produces a new goal; the policy's hidden state
     learned to expect monotonic progress toward the old goal.
   - On watchdog trip (any of the 6 sources stale → policy is
     paused, hidden state stays frozen).
5. **Determinism contract.** Two consecutive calls with the same
   observation produce byte-identical actions **iff** `reset()` is
   called between them. This is the *only* determinism statement the
   inference node should assert; "byte-identical without reset" is
   the wrong contract for a recurrent policy (it conflicts with the
   semantic the recurrent state exists for).
6. **Thread safety.** The loader is **NOT thread-safe**. The
   inference node must serialize all policy calls through a single
   thread. If the rclpy executor uses a MultiThreadedExecutor,
   policy calls must be guarded by a mutex.

### Phase 2 — Author a context module

Add `docs/tasks/context/recurrent-policy-contract.md` (cross-epic
context module, flat per
[`context/README.md`](../../context/README.md)) summarizing the same
spec. The three implementer briefs reference this module instead of
re-stating the contract.

Module content is the same as Phase 1's docstring but linked from
multiple briefs.

### Phase 3 — Edit the three implementer briefs to reference the contract

Update each of:
- [`export-onnx-depth.md`](../../completed/export-onnx-depth.md)
- [`loader-recurrent-state.md`](../../completed/loader-recurrent-state.md)
- [`inference-package.md`](inference-package.md)

with one line in the Context bundle pointing at
`context/recurrent-policy-contract.md`, and remove any contract
re-statement that's already in the module.

### Phase 4 — End-to-end integration test

Add a single test that exercises all three sides:
`source/strafer_lab/tests/test_recurrent_contract_e2e.py`. With a
tiny dummy DEPTH-shaped GRU module (1-linear depth encoder, 1-layer
GRU, 1-linear head):

1. Export to both `.pt` and `.onnx` via `Scripts/export_policy.py`
   helpers.
2. Load each via `strafer_shared.policy_interface.load_policy()` (the
   updated version from `loader-recurrent-state`).
3. Assert: two calls with the same obs and `reset()` between → byte-
   identical actions in both formats.
4. Assert: two calls with the same obs and **no** `reset()` between →
   **different** actions (state evolved as expected).
5. Assert: `.pt` and `.onnx` produce numerically-close actions on
   the same `(obs, h_in)` (≤ 1e-5 max abs delta) — proves the two
   formats agree on the same underlying model.

The cross-format parity check (5) is the load-bearing piece — neither
sibling brief asserts it, and divergence here is exactly the failure
mode this brief exists to catch.

## Acceptance criteria

### Spec

- [ ] `policy_interface.py` module docstring contains a "Recurrent
      hidden-state contract" section covering the six points in Phase 1.
- [ ] `docs/tasks/context/recurrent-policy-contract.md` exists and
      summarizes the same spec.
- [ ] [`export-onnx-depth.md`](../../completed/export-onnx-depth.md),
      [`loader-recurrent-state.md`](../../completed/loader-recurrent-state.md), and
      [`inference-package.md`](inference-package.md) each reference the
      context module in their Context bundle.

### Integration test

- [ ] `python -m pytest source/strafer_lab/tests/test_recurrent_contract_e2e.py`
      passes — both formats round-trip, both honor the determinism +
      reset semantics, and cross-format parity holds.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_shared/strafer_shared/policy_interface.py`](../../../../source/strafer_shared/strafer_shared/policy_interface.py)
  `load_policy()` — the loader signature this brief pins.
- [`source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py:250`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py)
  `_DepthGRUExportModel` — TorchScript-side state owner.
- `rsl_rl.models.rnn_model._OnnxRNNModel` (vendored under
  `env_isaaclab3`, lines 180-249) — canonical ONNX-side multi-input
  contract.
- [`Scripts/export_policy.py:_verify_torchscript_determinism`](../../../../Scripts/export_policy.py)
  — pattern for the determinism probe with `is_recurrent` branching.
- ANYmal upstream reset semantics: `legged_gym/envs/base/legged_robot.py`
  `reset_idx()` — the per-episode hidden-state zeroing pattern.

## Out of scope

- **Implementing the producer side.** That's
  [`export-onnx-depth`](../../completed/export-onnx-depth.md). This brief documents
  the contract that producer must satisfy.
- **Implementing the consumer side.** That's
  [`loader-recurrent-state`](../../completed/loader-recurrent-state.md). Same.
- **Implementing the caller side.** That's
  [`inference-package`](inference-package.md). Same.
- **Mutex around policy calls in `strafer_inference`.** Mentioned in
  the spec as a constraint; the actual mutex implementation is owned
  by the inference brief's Phase 3.
- **Designing a stateless / Markovian alternative.** The Strafer
  DEPTH policy is recurrent by configuration choice
  (`STRAFER_PPO_DEPTH_RUNNER_CFG.actor.rnn_type="gru"`). Re-evaluating
  that choice is a separate research question, not a contract-doc
  brief.
