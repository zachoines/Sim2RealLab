# Put the subgoal generator's freshness guards on the clock its timeouts are sized in

**Type:** bug (measurement validity, sim lane)
**Owner:** Jetson (`strafer_inference` lane)
**Priority:** P1 — it intermittently suppresses one of the three anchor
admission rules that `subgoal-mission-anchoring` (#174) and the drift-band work
(#200) shipped, on exactly the lane behavioural acceptance runs on. It does not
degrade a measurement visibly; it makes a rule fire about half the time instead
of always, and logs the suppression at a throttled WARN that reads as scene noise.
A duty-cycled admission rule is worse than an absent one: the anchoring records
it produces are neither the shipped behaviour nor a clean control.
**Estimate:** S (one clock accessor, twelve call sites, plus the reporting line)
**Branch:** `task/subgoal-generator-sim-clock-freshness`

## Story

As the **engineer running behavioural acceptance on the sim-bridge lane**, I
want **the subgoal generator's staleness guards measured on the same clock its
timeouts are expressed in**, so that **a costmap that is fresh in the robot's
own time is not declared stale, and the collision admission rule is actually
in force during the runs that are supposed to exercise it.**

## Context

`strafer_subgoal_generator` runs under `use_sim_time:=true` on the sim-bridge
lane (`docker-compose.override.sim-bridge.yml` sets it for `slam`/`navigation`;
`compose/sim_bridge.env` carries `STRAFER_USE_SIM_TIME=true` for the inference
service, which hosts this node). Its timeouts are sized in the same units as
the sim-time quantities they gate — a costmap publish interval, a replan
period.

Every one of its freshness guards is stamped and compared with
`time.monotonic()`, which is **wall** time and is unaffected by `use_sim_time`:

| guard | parameter | default | stamp / compare |
|---|---|---:|---|
| costmap staleness | `costmap_timeout_s` | 5.0 | [`:614`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L614) / [`:470`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L470) |
| plan staleness | `path_timeout_s` | 1.0 | [`:523`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L523) / [`:967`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L967) |
| goal telemetry | `goal_telemetry_timeout_s` | 2.5 | [`:640`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L640) / [`:667`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L667) |
| replan spacing | `replan_period_s` | 0.5 | [`:708`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L708) / [`:682`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L682) |
| `anchor_age` in the status line | — | — | [`:414`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L414) / [`:576`](../../../../source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py#L576) |

The node holds **12** `time.monotonic()` call sites against **1** use of
`get_clock().now()`. Its sibling `inference_node.py` is the counter-example and
the model: its staleness and cadence logic reads `get_clock().now()`
([`:1165`](../../../../source/strafer_ros/strafer_inference/strafer_inference/inference_node.py#L1165),
[`:1176`](../../../../source/strafer_ros/strafer_inference/strafer_inference/inference_node.py#L1176),
[`:1207`](../../../../source/strafer_ros/strafer_inference/strafer_inference/inference_node.py#L1207)),
which is why its `depth_age` is reported in sim seconds and is correct.

This is latent wherever real time and sim time agree — a real robot, or a sim at
RTF 1. It fires on any lane running slower than `timeout / interval` times real
time, and the enriched sim-bridge lane runs at **RTF 0.106**.

## Measured, 2026-08-17 rig gate

Bridge `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`. **RTF 0.106**,
taken session-wide from the ride-along (980.8 s sim in 9257.0 s wall); spot
readings ranged 0.106–0.115 (`/clock` advanced 2.18 s per 20.0 s wall on one
such sample). Every duty figure below uses the session-wide 0.106, so the
arithmetic is stated once and does not drift between paragraphs.

- `/global_costmap/costmap` publishes at **0.948 Hz sim** — inter-arrival
  min/mean/max **1.02 / 1.05 / 1.11 s sim**, so **0%** of gaps exceed the 5.0 s
  `costmap_timeout_s`. In wall time at RTF 0.106 the same gaps are **9.9 s**, so
  the guard fired for the larger part of every cycle and `Costmap is older than
  5.0 s; skipping the collision admission rule` was emitted at its 10 s throttle
  throughout.
- Consequence: of the three admission rules the status line advertises
  (`cross_track>0.50 m`, `collision_check=/global_costmap/costmap@cost>=99`,
  `goal_changed`), the collision rule is **intermittently suppressed, not
  disabled**. Each costmap arrival opens a 5.0 s **wall** window in which the
  rule is live; the next arrival is 1.05 s sim ÷ 0.106 = **9.9 s wall** away, so
  the rule is unavailable for **49.5%** of the interval — call it half. (At the
  fastest spot RTF observed, 0.115, the gap is 9.1 s and the figure is 45.2%;
  the session-wide number is the one quoted here.) The suppression is invisible
  in the log because the warning is throttled to 10 s and therefore reads as
  continuous. Both outcomes were observed in the same session: the warning fired
  for the larger part of every cycle, and collision admissions still landed in
  the windows where the guard happened to be open. `anchor_in_collision` in the
  `anchor status:` line is a **cumulative counter**, not a flag — it climbs
  1 → 43 across this session over 13 distinct values — so a single reading of
  `anchor_in_collision=2` is a snapshot early in the run, not a total. The
  session total is **43**, of which **18** fall inside the six scored mission
  windows (14 of those in one mission). The other two rules are unaffected
  (`cross_track_exceeded` and `goal_changed` both fired normally).
- `plan is stale (older than 1.0 s); suppressing rolling-subgoal output` also
  fired: the replan period is 0.5 s sim ≈ 4.7 s wall at RTF 0.106, against a
  1.0 s wall guard.
- `anchor_age` in the `anchor status:` line is wall seconds while `cursor` and
  `cross_track` beside it are spatial and the mode is sim-driven. It read
  `anchor_age=794.0s` on a mission that had run ~50 s sim, which is a live trap
  for anyone reading that line to time an anchor.

## Acceptance

- [ ] Freshness guards read the node clock (`get_clock().now()`), so they follow
      `use_sim_time` and are expressed in the units their parameters name.
- [ ] `anchor_age` in the status line is sim seconds, or is labelled with its
      clock.
- [ ] Wall-clock stays where wall-clock is correct: the status-line log cadence
      is a human-facing interval and should remain monotonic-based.
- [ ] A regression test drives the node with a clock slower than real time and
      asserts the collision admission rule stays in force while the costmap is
      fresh in sim time.
- [ ] Re-run the enriched sim-bridge lane and confirm `Costmap is older than
      5.0 s` no longer appears while the costmap publishes at ~0.95 Hz sim.

## Scope

The fix is a clock swap inside one node. It changes no admission thresholds and
no default. It does change behaviour on slow sim lanes — the collision rule goes
from firing about half the time to always — so the first re-run after it lands is
expected to show **more** `anchor_in_collision` admissions than this session's 43,
and anchoring records taken before and after the fix are not directly comparable.
