# Drive inference off depth arrival so the deploy cadence matches training

**Status:** Shipped 2026-07-31 in `<ship-commit>` (Jetson). Filed and shipped in
the same PR — the work came from the 2026-07-31 obs-parity decision session, not
from the queue.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/<N>

**Type:** bug (train↔deploy cadence)
**Owner:** Jetson
**Priority:** P1 — the policy is recurrent, so a missing step is a missing
hidden-state advance.
**Estimate:** M
**Branch:** `task/inference-cadence-shortfall`

## Story

As a **recurrent policy trained at 30 Hz**, I need **one inference per fresh
depth frame at the rate the frames actually arrive**, so that **my hidden state
advances on deploy at the cadence it advanced on during training.**

## Context bundle

- [context/conventions.md](../context/conventions.md)
- [context/recurrent-policy-contract.md](../context/recurrent-policy-contract.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Symptom

The inference node ran **23.49 Hz sim** against training's 30 Hz. The inter-tick
mode was exactly 33.33 ms and **24.47%** of intervals stretched to ≥ 2 periods:
the node held the right period and *dropped* ticks.

(Two corrections to the session's own figures, both measured here: the headline
"23.79 Hz" is the **mission-p2 segment alone**; the aggregate over the whole dump
is 23.49 Hz, and the 24% figure is stable across idle-gap segmentation thresholds
of 0.5 / 1 / 2 / 5 s. The three mission segments differ — p1 78.82%, p2 79.25%,
p3 76.04% — and p3's loss has a materially different mix, ~12% of its lost slots
being watchdog trips against 0.4% for p2, on a degraded subgoal stream.)

## Diagnosis — done before the fix, and adjudicated between two candidates

Nothing upstream of the node lost anything. Over the bag window every sensor
topic is complete: depth / odom / imu / joint_states each 2245 messages,
expected 2245, **zero** missing sim slots, every header delta exactly one
period. `/clock` steps at exactly 30.000000 Hz sim (only two step values,
33333333 ns ×1497 and 33333334 ns ×748).

**The key join:** over sim slots 5895–7060 (1166 slots, 38.83 s sim), 1166/1166
depth frames present, 919 node ticks (78.82%), and of the **247 missing ticks,
247 — 100% — land on slots where a depth frame WAS published.** Zero coincide
with a missing frame.

Candidates were ruled out by measurement, not argument:

| candidate | verdict | evidence |
|---|---|---|
| timer/clock aliasing | **refuted** | `/clock` has no sub-33.33 ms granularity to alias against; a replayed 1/30 s ROS_TIME timer under that exact clock stream covers 406/406 slots |
| watchdog trips | **refuted** for p1/p2 | 1 exact-zero-twist `/cmd_vel` row in 1200 slots; the sim lane runs `STRAFER_OBS_TIMEOUT_S=1.0` (`compose/sim_bridge.env`), not the 0.2 s yaml default. Real in p3 |
| ORT inference latency | **refuted** | TRT median 4.71 ms against a ~218 ms wall budget per sim step |
| executor starvation → lost timer deadlines | **minor, ≤ 2.9%** | see below |
| **depth-freshness gate × depth arrival phase** | **CONFIRMED** | see below |

**The mechanism.** The sim runs at RTF 0.15, so one 1/30 s sim step spans
~218 ms of wall, and that step's depth frame lands a long way into it — 25.8%
within 1 ms of the clock step, 59.0% at ≥ 90 ms. A timer-driven tick therefore
usually fires *before* its own step's frame; the gate correctly skips it, and the
frame is superseded before any later tick reads it.

Two independent measurements pin this rather than executor starvation:

1. A zero-free-parameter model — "the tick runs τ ms after the clock step; the
   gate passes iff an unconsumed frame has landed by then" — reproduces
   **97.0%** of the individual per-slot hit/miss decisions (TPR 0.999, TNR
   0.862) with **no starvation term**, against baselines of 78.8%
   (always-hit) and 66.6% (rate-matched random).
2. Reconstructing which frame each inference actually consumed — the repo's own
   `downsample_depth` applied to the raw 640×360 bag payloads, matched
   **bit-exactly** against the obs dump's depth block (919/919 at max|Δ| = 0.0)
   — shows **47.7%** of ticks consuming a one-slot-old frame and **34.2%** a
   two-slot-old one. A tick *delayed* by the 921 KB deserialize would consume
   the **current** frame. Starvation is left at most 34/1166 = 2.9% of slots.

## What shipped

- **The tick is driven by depth ARRIVAL** for depth variants — a guard condition
  triggered from `_on_depth`, running in the *default* callback group so the
  921 KB deserialize keeps its own executor slot. The freshness gate is
  untouched and is still the rate limiter: this fixes the tick's **phase**, not
  its cap, and makes "one inference per fresh depth frame" structural rather
  than emergent. `tick_on_depth: false` restores timer-only scheduling for an
  A/B.
- **The timer stays** — it is the watchdog heartbeat, the scheduler for
  camera-free variants, and the safety net if the depth stream stops. Without it
  a dead depth feed would produce no tick and therefore no zero-twist.
- **`executor_threads: 5`** (was a hardcoded 3) so each of the five blocking
  callback groups has a slot. This is the ≤ 2.9% residual, not the fix, and
  `timer_deadline_missed` now measures it directly.
- **Permanent fail-loud counters** in a periodic `cadence:` line: achieved Hz
  sim vs target, ticks by source, inferences, every skip by cause
  (`gate` / `watchdog` / `obs_none` / `no_policy` / `action_shape`), depth
  `rx` / `unconsumed` / `repeat_content` / `bad_encoding` / `bad_shape`,
  `timer_deadline_missed`, and a stale-source histogram. Below 90% of target it
  escalates to a WARNING that **names the owner**: `depth_rx` also short ⇒ the
  transport; `depth_rx` healthy ⇒ this node, with the skip histogram attached.
- **`depth_reliability`** exposes the depth subscription's QoS as a named lever,
  **default unchanged** (`best_effort`). The old BEST_EFFORT rationale — "a
  dropped frame should skip a tick rather than trigger a retransmit" — is
  inverted by arrival-driven ticking, but a RELIABLE subscriber is
  QoS-incompatible with a BEST_EFFORT publisher and would receive *nothing* from
  a camera brought up with `depth_qos:=SENSOR_DATA`. The lever exists so the rig
  can measure it; see the evidence below for why it is worth measuring.

## Evidence — bag replayed into the real node

The preserved bag's exact `/clock` + depth wall cadence replayed into a real
`InferenceNode` (stubbed policy at the measured 4.71 ms TRT median; the same
callback groups, executor, depth QoS, gate, and obs assembly). Node-side
consumption is the metric the fix owns — what fraction of the frames that
*reached* the node became inferences:

| arm | achieved Hz sim | depth_rx | inferences | consumed |
|---|---|---|---|---|
| timer-only, 3 threads (before) | 20.20 / 21.85 / 23.59 / 19.37 | 158 / 166 / 185 / 144 | 138 / 150 / 162 / 133 | **87.3 / 90.4 / 87.6 / 92.4%** |
| depth-driven, 5 threads (after) | 25.44 / 21.07 / 24.88 / 27.37 | 175 / 146 / 171 / 188 | 173 / 144 / 170 / 187 | **98.9 / 98.6 / 99.4 / 99.5%** |
| depth-driven + `reliable` | 30.15 | 209 | 206 | 98.6% |

`timer_deadline_missed` reads **0** in every fixed arm, consistent with the
adjudication that starvation was not the driver.

The residual in the shipped (`best_effort`) arm is **transport, not the node** —
`depth_rx` itself is short of the ~206 published slots, and the counters say so
by name. The `reliable` arm recovers it and lands at **30.15 Hz sim**, the
training cadence. **Caveat:** that replay runs over container-loopback DDS with
`ROS_LOCALHOST_ONLY=1`, which is not the field transport, and the QoS
incompatibility above is a real hazard. Flipping the default is a rig decision,
not an offline one; `subgoal-anchoring-rig-revalidation` reads exactly this pair.

## Separate finding — not a node defect, recorded

The sim publishes depth at 30 Hz stamps but **renders at 15 Hz**: over the join
window there are exactly 583 duplicate runs, every one of length 2 (54.1% of
messages byte-identical to their predecessor over the whole bag). The gate keys
on the message counter, so duplicates satisfy it. The training gym dump shows the
same 15 Hz novelty after t_sim ≈ 66 s, so this is parity-matched rather than a
deploy bug — but it means the 24% tick loss cost recurrent **hidden-state steps**
(30/sim-s training vs 23.49/sim-s deploy), not image information: 97.25% of
distinct images still reached the policy. The new `depth_repeat_content` counter
makes it visible at runtime. Whether the gym side's 30 Hz → 15 Hz transition is
an artifact or a config difference is filed as
[`sim-depth-render-rate-parity`](../active/sim-performance/sim-depth-render-rate-parity.md).

## Acceptance criteria

- [x] Cause diagnosed and *adjudicated* against competing candidates before any
      fix — with the discriminating measurement stated for each.
- [x] Minimal fix targeting the confirmed cause; the freshness gate is NOT
      loosened (one inference per fresh depth frame is preserved exactly, and
      is now structural).
- [x] Permanent fail-loud counters by cause, in a periodic log, with an
      owner-naming WARNING below 90% of target.
- [x] Node-side consumption 87–92% → 98.6–99.5% in a replay of the preserved
      capture into the real node.
- [x] Config-gated (`tick_on_depth`) so the rig can A/B it.
- [x] No obs-assembly change.
- [x] User-facing docs updated in the same commit.

## Out of scope

- Flipping the depth QoS default — measured, exposed, and left to the rig.
- The sim's 15 Hz depth render — a bridge-lane question, filed separately.
- p3's watchdog trips on a degraded subgoal stream — the counters now name it;
  the generator half is `subgoal-mission-anchoring`.
