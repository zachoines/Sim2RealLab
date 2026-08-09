# Tick inference on the timer and reuse the newest depth frame

**Status:** Shipped 2026-08-09 in `1c551c8` (Jetson). `make test-ros` reads
755 passed / 11 skipped across all seven `strafer_ros` packages, against 735
before.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/198

**Type:** task (deploy runtime — inference semantics)
**Owner:** Jetson
**Priority:** P1 — adoption was ordered by a scored, pre-registered rule, and
the change is the first half of a gate the enriched-lane attribution cannot
discharge until both halves ship.
**Estimate:** S (one parameter, one branch in the tick, one counter split)
**Branch:** `task/inference-timer-reuse-semantics`

## Story

As the **deployed recurrent policy**, I want **a tick to step me on the newest
depth frame it has rather than skipping when no new one arrived**, so that **a
degraded depth arrival regime costs me stale pixels instead of freezing my
hidden state, my proprioception and my command along with it.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/recurrent-policy-contract.md](../context/recurrent-policy-contract.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

The node ran at most one inference per fresh depth frame. A tick whose
depth-frame counter had not advanced returned before observation assembly: no
inference, no publish, no hidden-state advance. That preserved training's
one-depth-one-step alignment exactly, and it pinned the inference rate to the
depth arrival rate — in the failing 12 Hz regime the gate suppressed ~61% of
ticks, so the node advanced its recurrent state ~700 times across a 60 s
mission against training's 1800.

That trade was taken before anything measured which half of it was expensive.
[`inference-cadence-shortfall`](inference-cadence-shortfall.md) shipped with
"the freshness gate is NOT loosened", and named the reasoning: stale content
into a recurrent state is a known hazard. What it could not know is that rate
parity, not content novelty, carries almost the whole cost.

### The measurement that ordered this

[`cadence-harness-residual-arms`](cadence-harness-residual-arms.md) scored the
proposal as a pre-registered arm before any node code was written, on the
closed-loop cadence-emulation harness. Holding distinct depth content fixed and
moving only the inference rate:

| operating point | gated (as shipped) | timer reuse | of the 0.900 reference |
|---|---|---|---|
| degraded, novelty-matched (7.6–7.8 Hz distinct) | 0.610 | **0.910** | 1.011 |
| full novelty (11.7–12.2 Hz distinct) | — | 0.860 | 0.956 |
| band-equivalent (23.1–23.2 Hz distinct) | 0.870 | 0.850 | 0.944 |

The novelty-matched cell is the one the adoption rule reads, and it cleared the
0.85 line by a wide margin; the band cell is neutral, which is the shape the
mechanism predicts. Both committed predictions held. The explanation the
read-out gives for duplicate content being nearly free is that training itself
ran amid render duplication, so the policy has seen stale pixels at full rate
and has not seen a frozen state at a live one.

### The second motivation, independent of that incident

The arrival regime is not a fixed property of the rig. The transport
measurement in
[`depth-receiver-host-capacity`](depth-receiver-host-capacity.md) bracketed the
sim-host uplink at 60 Mbit/s loss-free at the start of a window and 20 Mbit/s
~50 minutes later, with RSSI and negotiated PHY rate unchanged across the drop
— a 3× swing inside one hour, attributed to airtime contention rather than
signal quality, and the reason
[`sim-bridge-link-transport-capacity`](../active/reliability/sim-bridge-link-transport-capacity.md)
requires every arm to be bracketed at both ends.

So the depth-arrival regime a deployed policy sees is a function of transport
weather. Decoupling the inference rate from arrival is insurance against that
variability, not only a remedy for the one regime that was measured. It is also
why no separate reuse budget was introduced: a knob sized against one observed
arrival rate would need re-sizing every time the link moved.

### What bounds the reuse

Nothing new. The depth-age watchdog already runs ahead of the freshness gate in
the tick, so a cached frame older than `depth_timeout_s` holds the tick before
any reuse decision is reached — zero twist mid-mission, silence when idle,
exactly as before. The bound follows the deployed override
(`STRAFER_DEPTH_TIMEOUT_S`) wherever a lane sets one, and the node logs the
value it is bounded at.

## Acceptance criteria

- [x] Ticks run on the timer at the artifact-resolved step period. The period
      is whatever `_resolve_infer_period` settled on, so the artifact's
      `trained_period_s` still wins over the configured value and the cadence
      line's target follows it.
- [x] The freshness gate's skip path becomes bounded stale reuse: an inferring
      tick consumes the newest cached frame whether or not its seq advanced.
- [x] `depth_timeout_s` bounds the staleness; beyond it the watchdog holds the
      tick exactly as today. No separate reuse budget exists.
- [x] The planner refusal and starvation guards are untouched — nothing in
      `subgoal_generator_node.py` changed.
- [x] Config-gated: `depth_tick_semantics: timer_reuse | gated`, shipped
      `timer_reuse`, validated against the named set with an unknown value
      raising at construction. The same shape as the anchoring flip.
- [x] The gate-skip counter splits into fresh vs reuse inferring ticks, and the
      cadence line reports the split.
- [x] `depth_age` is recorded on reuse ticks, so the shipped instrument reads
      what the reuse cost.
- [x] Tests: both modes' semantics, reuse bounded by the timeout and resuming
      after it, hidden-state advance on every inferring tick, the counter
      split, `unconsumed` against the fresh count, and the repeat-content
      instrument in both directions.
- [x] Docs swept: `source/strafer_ros/README.md`'s trained-policy execution
      entry, `strafer_inference/scripts/PARITY_SCHEMA.md`'s cadence report,
      `inference.yaml`'s parameter documentation, and the node's module
      docstring. `docs/tasks/DEFERRED_WORK.md`, the top-level `Readme.md` and
      the cheatsheet carry no fact this invalidates.
- [x] No regression: `make test-ros` 755 passed / 11 skipped, against 735.

## Decisions taken

Recorded so the next reader does not re-litigate them.

- **The semantics own the scheduler; `tick_on_depth` does not compose with
  them.** Under `gated`, the gate is what caps the rate, so ticking on arrival
  only changes the tick's phase. Without a gate it changes the rate: every
  arrival would raise an inference *on top of* the timer's, putting a recurrent
  policy above its trained cadence — the exact silent failure the artifact
  cadence contract exists to prevent. `timer_reuse` therefore builds no wake
  handle, `tick_on_depth` is read under `gated` only, and the node logs that
  the parameter is inert rather than leaving the composition to be discovered.
- **No separate reuse budget.** The adoption note anticipated one ("strictly
  shorter than `depth_timeout_s`"). It would be a second staleness threshold
  sitting under an existing one, needing its own default, its own override and
  its own re-sizing whenever the link moves. The watchdog already enforces a
  staleness bound at exactly the point the reuse decision is made; a budget
  under it can only ever fire first for a reason no measurement has supplied.
- **`depth_repeat_content` counts fresh frames only.** It exists to catch a
  publisher stamping faster than it renders, which is the question
  [`sim-depth-render-rate-parity`](../active/sim-performance/sim-depth-render-rate-parity.md)
  is open on. A reused frame repeats by construction, so counting reuse ticks
  would read as publisher duplication and, at a degraded arrival rate against a
  30 Hz tick, would swamp the signal entirely. Node-side reuse is read from the
  cadence line's own `reuse=` count instead.
- **`unconsumed` is read against the fresh count.** `depth_rx - inferences`
  goes negative under reuse. `depth_rx - infer_fresh` is the frames that
  arrived and were overwritten before any tick took them, which is what the
  figure meant all along, and it is identical under `gated`.
- **The shortfall attribution keys on held ticks under reuse.** A low arrival
  count is no longer a shortfall by itself, so attributing one to transport on
  that basis would mis-blame the link for a node-side tick overrun. Under
  `timer_reuse` the transport branch keys on depth-stale ticks — depth absent
  long enough for the watchdog to hold the tick — and reports both counts.

## Consequences carried forward

- **The recurrent horizon lengthens.** Deploy moves from ~700 hidden-state
  advances per 60 s mission in the degraded regime toward the full 1800. The
  chained-horizon arm is flat to ~920 advances and untested beyond; a
  long-horizon chained arm is already pre-registered for the next acceptance
  grid.
- **Thread contention rises.** The tick now calls the policy on every timer
  fire rather than once per arrival, so inference occupies its thread more
  often. `timer_deadline_missed` is the shipped instrument and is unchanged;
  the node-side consumption levers remain their own scope.
- **This is one half of an ordered gate.** The composition read-out is
  discharged only after both this change and the subgoal-drift randomization
  ship, then the retrain runs and is scored, then the rig is re-validated
  running that policy under these semantics. A rig read taken between the two
  merges tests half the remedy.

## Out of scope

- **The depth-age watchdog's budget.** `depth_timeout_s` stays 0.5 s and the
  per-lane override stays as it is; this change makes the watchdog load-bearing
  for a second reason without moving it.
- **The planner refusal and starvation guards**, whose whole purpose is
  breaking the zero-twist self-lock.
- **The depth QoS default and history depth**, and the node-side consumption
  levers (executor split, pinning, sysctl).
- **The cadence setpoint.** 30 Hz stands; the read-out that ordered this
  dissolved the 20 Hz branch.
- **The subgoal-drift randomization**, the other half of the ordered gate, and
  a `strafer_lab` change.
- **The sim's duplicate depth content**, which is a publisher question and is
  why the repeat-content instrument was preserved rather than repurposed.

## Investigation pointers

- The scored arm, its pre-registered adoption rule and the per-arm tables:
  [`cadence-harness-residual-arms`](cadence-harness-residual-arms.md)'s Arm D
  and its read-out item (d).
- The ruling this supersedes on measurement rather than precedent:
  [`inference-cadence-shortfall`](inference-cadence-shortfall.md)'s "the
  freshness gate is NOT loosened".
- The step-period contract the tick runs at:
  [`recurrent-policy-contract.md` §7](../context/recurrent-policy-contract.md#7-trained-step-cadence),
  and [`policy-artifact-cadence-contract`](policy-artifact-cadence-contract.md).
- The transport bracket behind the second motivation:
  [`depth-receiver-host-capacity`](depth-receiver-host-capacity.md)'s
  measurement section, carried forward by
  [`sim-bridge-link-transport-capacity`](../active/reliability/sim-bridge-link-transport-capacity.md).
- The `depth_age` instrument this change turns into a cost meter shipped in
  [`depth-qos-reliable-flip`](depth-qos-reliable-flip.md) and had never yielded
  a rig reading, because no frame reached the node.
