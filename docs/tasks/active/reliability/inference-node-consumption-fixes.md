# Stop the node dropping depth while it infers

**Type:** task (deploy runtime — throughput)
**Owner:** Jetson
**Priority:** P1 — the cost is now measured rather than assumed: at the 12 Hz
arrival regime the emulated policy loses ~⅓ of its completion rate, and at the
22–25 Hz band it loses ~3%, so recovering arrival rate recovers almost the whole
temporal cost.
**Estimate:** M (three independent levers, each with its own measurement; no
policy or training change)
**Branch:** `task/inference-node-consumption-fixes`

## Story

As the **deployed inference node**, I want **the tick's policy call to stop
occupying the thread that has to receive the next depth frame**, so that **the
frames the bridge actually publishes are the frames the policy actually sees.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/recurrent-policy-contract.md](../../context/recurrent-policy-contract.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

The node receives depth at 22–25 Hz sim against a 30 Hz publish, at ~98%
consumption, while a **concurrent** subscriber on the identical
`BEST_EFFORT`/depth-1 QoS sees 29.8 Hz. Frames are on the wire; this process is
not taking them. In the same instrumented arm the node missed **11 146** timer
deadlines mid-mission against **1** when idle.

Those two readings are one mechanism seen from both ends. At queue depth 1 an
un-drained sample is overwritten, so a thread that is busy elsewhere when the
frame lands loses it at the receiver — which is exactly what the concurrent
subscriber's 29.8 Hz proves it was not a transport problem. What occupies the
thread is the tick itself: TensorRT inference runs inside the callback, and the
prior round of work on this node already went 2→3 executor threads for depth
reception and then 3→5 so each blocking callback group has a slot
([`depth-reception-reliability`](../../completed/depth-reception-reliability.md),
[`inference-cadence-shortfall`](../../completed/inference-cadence-shortfall.md)),
leaving a ≤2.9% residual that `timer_deadline_missed` now measures directly. The
mid-arm number says the residual is not 2.9% under load.

**This work was sequenced behind the rig profile capture and is no longer.** It
was held only so a capture would reflect the stack the failing arm ran on; the
temporal axis has since been adjudicated on synthetic profiles and the
measured-profile cell is archival, so nothing is gained by holding a cheap
recovery behind a blocked capture. See
[`cadence-emulation-eval`](../../completed/cadence-emulation-eval.md)'s read-out
for the stakes this brief's `Priority` quotes.

**It also does not own the QoS flip, and is not an alternative to it.**
[`depth-qos-reliable-flip`](depth-qos-reliable-flip.md) buys queueing; this
brief buys drain rate. They are complementary remedies for one symptom and the
measurement below must be able to attribute a recovery to one of them, which
means they are measured separately before either is judged.

## The three levers

Independent, and worth keeping independent so a recovery is attributable.

1. **Executor / callback split.** Take the policy call out of the callback that
   holds a thread the depth subscription needs — either its own callback group
   with a dedicated thread, or off-executor entirely with the tick publishing
   the result. The thread-safety pattern the depth path already uses (lock plus
   snapshot, so a tick reads one coherent observation) is the model to follow.
2. **CPU pinning.** The Orin's cores are shared with the bridge client, SLAM and
   Nav2. Pin the executor's threads and record what was pinned; an unpinned
   measurement is not reproducible across sessions.
3. **Transport reliability.** Receive-buffer and `rmem_max` sysctl are already
   checked in for the fragmented-depth path; confirm they are still in force on
   the current image and that nothing in the container bringup resets them.

## Acceptance criteria

- [ ] Baseline first: `timer_deadline_missed`, depth arrival Hz, consumption
      share and the node's own inference latency distribution, captured idle
      and under a full mission, **before** any lever moves. Without the loaded
      baseline no lever can be scored.
- [ ] Each lever measured on its own arm against that baseline, and the arms
      reported separately. A combined-only result cannot attribute the recovery.
- [ ] The concurrent-subscriber probe re-run on each arm — it is the
      discriminator between "the frame never arrived" and "we did not take it".
- [ ] Inference latency reported as a distribution, not a mean. The tail is what
      overruns a 33 ms tick.
- [ ] No change to the freshness gate, the depth-age watchdog, or the planner
      refusal and starvation guards. Those are a separate decision (see
      Out of scope).
- [ ] Thread-safety preserved: whatever moves off the tick thread, a tick still
      reads one coherent observation snapshot, and the guard is exercised by a
      test rather than argued.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports — the
      `strafer_inference` suites stay green and the containerized hybrid smoke
      still comes up.

## Decision rule (pre-registered)

| measurement | reading |
|---|---|
| the executor arm recovers arrival to ≥ 28 Hz with `timer_deadline_missed` back near its idle value | node-side consumption was the shortfall; the QoS flip is then a second-order improvement, not the fix |
| the executor arm moves arrival by < 2 Hz | the loss is not the tick blocking the receiver; hand the shortfall to the QoS flip and record the executor arm as a negative result rather than dropping it |
| inference tail latency alone exceeds the 33 ms tick period | no amount of thread rearrangement recovers 30 Hz, and the shortfall is a model or runtime problem — file it rather than absorbing it here |

## Investigation pointers

- The measured symptom, per arm, with the concurrent-subscriber probe:
  [`depth-qos-reliable-flip`](depth-qos-reliable-flip.md) and the arm table in
  [`subgoal-anchoring-rig-revalidation`](../../completed/subgoal-anchoring-rig-revalidation.md).
- The mid-arm deadline-miss count and the attribution it carries:
  [`enriched-lane-rig-stability`](enriched-lane-rig-stability.md)'s
  `## Out of scope`, which currently routes this scope to the QoS brief.
- Prior rounds on the same node and what they already bought: the dedicated
  depth callback group and 2→3 threads in
  [`depth-reception-reliability`](../../completed/depth-reception-reliability.md);
  5 executor threads and the `timer_deadline_missed` counter in
  [`inference-cadence-shortfall`](../../completed/inference-cadence-shortfall.md).
- **The coupling to watch.**
  [`cadence-harness-residual-arms`](../trained-policy/cadence-harness-residual-arms.md)
  scores a proposal to drive inference off a 30 Hz timer that reuses the last
  depth frame. If that is adopted, the node infers 30 times a second instead of
  once per arrival, which makes this brief's contention **worse**, not better.
  Whichever lands second inherits the other's measurement.

## Out of scope

- **The freshness gate's semantics.** Whether the node should infer on a reused
  depth frame is pre-registered as an arm in
  [`cadence-harness-residual-arms`](../trained-policy/cadence-harness-residual-arms.md)
  and belongs to that read-out, not to a throughput brief.
- **The QoS flip** — [`depth-qos-reliable-flip`](depth-qos-reliable-flip.md)
  owns it.
- **The sim-side render duplication.** A DGX brief owns the publisher stamping
  faster than it renders.
- **Any policy, checkpoint or training change.**
