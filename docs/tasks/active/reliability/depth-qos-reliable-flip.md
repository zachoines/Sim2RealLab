# Flip the inference node's depth subscription to RELIABLE

**Type:** task (deploy runtime)
**Owner:** Jetson
**Priority:** P1 — it costs the policy ~20% of its training cadence on every
mission, and the loss is at the receiver, so no amount of node-side work
recovers it.
**Estimate:** S (one parameter, one QoS profile, plus a re-measure)
**Branch:** `task/depth-qos-reliable-flip`

## Story

As the **operator running the depth-subgoal policy against the sim bridge**, I
want **the inference node to receive every depth frame the bridge publishes**,
so that **the policy runs at the 30 Hz cadence it was trained at instead of the
22–25 Hz it currently sees.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

The 2026-08-01 four-arm re-validation measured the node's own cadence counters
per arm (straddler-free pooled rate, `depth_rx` against `inferences`):

| arm | achieved | depth arrival AT THE NODE | consumption |
|---|---|---|---|
| v2 × `rolling` | 23.70 Hz sim (79% of target) | 22.20 Hz sim | 97.9% |
| v2 × `mission` | 25.06 Hz sim (84% of target) | 25.54 Hz sim | 98.1% |

Consumption near 98% says the node converts almost every frame it *receives*
into an inference. The shortfall is therefore in arrival, not processing — the
node is starved, not slow.

**The discriminating measurement.** A second subscriber was run **concurrently
with the inference node**, on exactly the node's own depth QoS (BEST_EFFORT,
`KEEP_LAST` depth 1), for 60 s during a live arm:

```
depth 134 frames, 2.25 Hz wall, RTF 0.075  ->  29.84 Hz SIM
consecutive-identical content 0/133
```

The bridge supplies the full ~30 Hz sim cadence and a second subscriber on the
same QoS receives it, at the moment the inference node is receiving 22–25 Hz.
That excludes a sim render shortfall (`sim-depth-render-rate-parity` is a
different defect) and excludes the transport itself.

**Mechanism.** With `history=KEEP_LAST, depth=1`, a frame arriving while the
node is busy — mid-inference, at 30 Hz, on a 3619-dim observation — overwrites
the queued frame, and the older one is lost at the receiver before any callback
runs. The probe does almost no per-frame work and so never overwrites. The
`inference-cadence-shortfall` session measured `reliable` at 30.15 Hz over
container loopback, consistent with this.

Do not be misled by the node's own shortfall attribution, which names
*"this node: frames arrived but did not become inferences"*. Its `gate` skips
are the expected steady-state skip on a depth variant (no new frame since the
last inference) and its `watchdog` skips are inter-mission idle. That line
attributes correctly for a *processing* shortfall and misattributes an *arrival*
one. Only the concurrent independent subscriber separates them.

## Why this matters beyond the Hz number

A `skip_gate` tick publishes **nothing** — it returns without a Twist, so the
previous command stays latched on `/cmd_vel` while the recurrent hidden state
stays frozen. At 22 Hz against a 30 Hz timer, roughly a quarter of ticks latch.
Training has no analogue: it steps the policy exactly once per environment step
with exactly one depth render per step, so a held action paired with a frozen
recurrent state is a deploy-only regime. This is a train↔deploy parity gap on
the control path, not merely a throughput loss.

## Acceptance criteria

- [ ] `depth_reliability` defaults to `reliable` for the sim-bridge lane, or the
      depth history depth is raised, with the choice justified against measured
      numbers rather than asserted.
- [ ] Re-measure with the same method: node `depth_rx` in sim Hz, and a
      **concurrent** independent subscriber. Both must read within 10% of each
      other; the node must reach ≥ 95% of the 30 Hz target.
- [ ] Confirm no regression in end-to-end latency — a reliable subscription that
      buffers stale frames would trade a rate loss for a staleness loss. Report
      the depth age at inference time before and after.
- [ ] Re-run at least one arm of the anchoring re-validation goal set and report
      whether the advance numbers move at all. **The expectation is that they do
      not** — the advance failure is policy-owned — so a change here is itself a
      finding.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Investigation pointers

- The parameter is `depth_reliability`, declared in
  `strafer_inference/inference_node.py` and shipped as `"best_effort"` in
  `strafer_inference/config/inference.yaml`.
- The node already prints the QoS in its shortfall warning, so the re-measure
  needs no new instrumentation.
- The concurrent-probe method is `tools/bridge_probe.py` in the 2026-08-01
  session artifacts (`~/strafer_v2_validation/`); it subscribes on the node's
  exact QoS and reports both wall and sim-time rates plus RTF.

## Out of scope

- The sim's duplicate depth content (byte-identical consecutive frames;
  intermittent, 24.1% in one arm and 0% in another). A DGX brief owns it.
- The policy's advance failure. Training lane owns it.
