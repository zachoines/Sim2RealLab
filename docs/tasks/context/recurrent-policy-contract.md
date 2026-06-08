# Recurrent policy hidden-state contract

The Strafer DEPTH policy is recurrent (rsl_rl PPO actor with `rnn_type="gru"`,
`rnn_hidden_dim=128`, `rnn_num_layers=1` per
[`STRAFER_PPO_DEPTH_RUNNER_CFG`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/rsl_rl_ppo_cfg.py)).
Hidden state is owned at three places along the train → export → inference
chain (rsl_rl trainer; the export wrappers in
[`depth_rnn_model.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py)
and [`source/strafer_lab/scripts/export_policy.py`](../../../source/strafer_lab/scripts/export_policy.py); the
inference-side loader in
[`strafer_shared.policy_interface`](../../../source/strafer_shared/strafer_shared/policy_interface.py)).
This module pins the seam-level contract so each layer can be edited
without re-deriving what the other two assume.

The authoritative in-code statement of the same contract is the
"Recurrent hidden-state contract" section of
`source/strafer_shared/strafer_shared/policy_interface.py`'s module
docstring. This file is the cross-task mirror; if the two disagree, the
docstring wins.

## The six pinned points

### 1. Hidden-state tensor shape

`(rnn_num_layers, 1, rnn_hidden_dim)` for batch-1 deployment inference.
The trailing-1 is a fixed batch axis. LSTM artifacts carry the pair
`(h, c)` of identically-shaped tensors; GRU carries `h` only.

- **TorchScript**: scripted module's `register_buffer("hidden_state", ...)`
  is allocated at this shape on module load.
- **ONNX**: input port `h_in` (and `c_in` on LSTM) is declared with this
  shape; symbolic batch dims collapse to 1 via
  `policy_interface._resolve_hidden_shape`.

Numbers come from the rsl_rl runner config (DEPTH today: 128 / 1; NOCAM
LSTM variant: 256 / 1). Treat them as parameterized, not pinned to a
specific value.

### 2. Initial state

Always zero on `load_policy()` return.

- **TorchScript**: `register_buffer(..., torch.zeros(...))`.
- **ONNX**: `np.zeros(shape, dtype=np.float32)` in
  `_RecurrentOnnxPolicy.__init__`.

No warm-start codepath. If an episode benefits from carried state, the
caller is responsible for not calling `reset()`.

### 3. Per-tick state threading

| Format | State owner | Per-tick flow |
|---|---|---|
| TorchScript | scripted module (in-module buffer) | `forward(obs)` reads the buffer, runs the RNN, writes new state back in-place. Loader holds a reference to the module; never copies the buffer. |
| ONNX | loader (persistent numpy buffers) | `__call__` feeds cached `h_in` (and `c_in`) into the session, reads `h_out` (and `c_out`) from output, writes them back. |

Skipping the ONNX-side write-back makes the policy effectively reset
every tick (feedforward MLP) with no error raised — a silent-failure
mode this contract exists to prevent.

### 4. Reset trigger

The inference node calls `policy.reset()` on every "mission episode"
boundary. The trigger set the inference-side caller commits to:

- **Action-server goal accepted** — a NEW `/navigate_to_pose` goal
  handle (not a re-statement of the current one). Hidden state from
  the previous mission is no longer relevant.
- **Mid-mission goal pose update** — VLM re-grounding produces a new
  goal pose mid-mission. Controlled by an `is_mid_mission_reset`
  config flag in the inference node (default: `True`). Rationale: the
  hidden state learned to expect monotonic progress toward the *old*
  goal; carrying it across re-grounding biases the policy.
- **Watchdog trip** — any of the contracted-stale topic sources fires.
  The policy is paused; hidden state stays frozen until the watchdog
  clears, then the *next* mission boundary calls `reset()` as normal.
  The pause itself does not reset.

Stateless policies expose a no-op `reset()` so callers can invoke it
unconditionally without an `is_recurrent` branch.

### 5. Determinism contract

Two consecutive calls with the same `obs` produce byte-identical
actions **iff** `reset()` is called between them.

Without an intervening `reset()`, the hidden state has evolved by
construction and the two actions differ — that's the recurrent-model
definition, not a bug. Determinism probes
([`source/strafer_lab/scripts/export_policy.py`](../../../source/strafer_lab/scripts/export_policy.py)'s
`_verify_torchscript_determinism` is the canonical example) must
condition on this. "Byte-identical between consecutive calls" is the
wrong assertion for a recurrent artifact and would force the scripted
module into a stateless mode that defeats its purpose.

### 6. Thread safety

`LoadedPolicy` and all its subclasses are **NOT thread-safe**.
TorchScript recurrent modules mutate `register_buffer("hidden_state",
...)` in `forward`; the ONNX variant mutates `self._h_in` (and
`self._c_in`) in `__call__`. Concurrent `policy(obs)` calls from
multiple threads race on those buffers.

Inference-node implications:
- Serialize all policy calls through a single thread, OR
- Guard `policy(obs)` and `policy.reset()` with the same mutex if the
  rclpy executor is a `MultiThreadedExecutor`.

The `LoadedPolicy` class itself does not provide the mutex — the
caller is in the best position to decide whether one thread is
sufficient.

## Cross-format parity (the load-bearing seam-level invariant)

The same underlying recurrent model exported to both `.pt` and `.onnx`
must produce numerically-close actions on the same `(obs, h_in)`
sequence. Tolerance: ≤ 1e-5 max abs delta per action component on
float32 weights.

This is the failure mode no single-format test catches. The TorchScript
loader test asserts TS round-trip; the ONNX loader test asserts ONNX
round-trip; only a cross-format test catches export divergence (e.g.
one path applying obs normalization with a different epsilon than the
other, or one path emitting a different activation than the trained
graph).

The integration test that pins this is
[`source/strafer_lab/tests/contracts/test_recurrent_contract_e2e.py`](../../../source/strafer_lab/tests/contracts/test_recurrent_contract_e2e.py).

## What this module is NOT

- **Not the implementation.** The producer side is
  [`source/strafer_lab/scripts/export_policy.py`](../../../source/strafer_lab/scripts/export_policy.py) +
  [`depth_rnn_model.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/agents/depth_rnn_model.py)
  (TorchScript) and the same export script + ONNX wrappers
  (`_OnnxDepthGRUModel`). The consumer side is
  [`policy_interface.load_policy`](../../../source/strafer_shared/strafer_shared/policy_interface.py).
  The caller side is the Jetson inference node (out of tree until
  `inference-package` ships).
- **Not the model architecture.** Width, depth, RNN choice are runner-
  config concerns; this contract holds across LSTM/GRU and across any
  `(num_layers, hidden_dim)` choice.
- **Not the obs / action contract.** Those live in `policy_interface.py`
  (`PolicyVariant`, `assemble_observation`, `interpret_action`).

## When to update

Update this module — and the in-code mirror — when ANY of the six
points changes. Specifically:

- New format added (e.g. TensorRT engine) → add a per-tick state
  threading row.
- New reset trigger added on the inference side → update point 4.
- `LoadedPolicy` gains internal locking → update point 6.
- Hidden-state shape convention changes (e.g. fixed batch becomes
  dynamic) → update point 1.

Briefs that touch the train / export / inference chain cite this
module from their `## Context bundle`. If you invalidate a fact here,
update both this file and the docstring in the same commit.
