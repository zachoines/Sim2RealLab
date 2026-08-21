# Depth-subgoal v2 mission gate on the rebuilt deploy stack, 2026-08-17

Six scored missions for the `DEPTH_SUBGOAL` v2 artifact against the deploy stack
at `e7ea7bd`, run over the direct DGX↔robot cable. **The stack met every
contract it owns; the policy reached 0 of 6 goals.** Acceptance thresholds fixed
before the runs were ≥4/6 pass, 1–3/6 partial, 0/6 fail, so this set **fails**.
The failure shape is a consistent *under-advance*, not an absence of motion: net
straight-line advance was positive in five of six missions (+0.15 to +1.10 m of
2.3–3.7 m) and −0.03 m in the sixth.

## Setup

| | |
|---|---|
| scene | `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, `Environment seed : 42` |
| Kit log | `kit_20260816_234250.log` |
| SLAM key | `enrich_riggate2` (fresh; the previous token's 13 MB database belongs to an earlier procedural draw) |
| images | `strafer-cpu:humble` / `strafer-gpu:humble`, both `e7ea7bd62f85`, no `-dirty` |
| artifact | `strafer_depth_subgoal_v2_998.onnx`, sha256 `855e1df7d0dac3be7229f933b59546b26f18959f97966b9e2f2f22e752bf5165` |
| | `policy_variant: DEPTH_SUBGOAL`, `obs_dim 3619`, recurrent, from `run_20260727_171735/model_998.pt` |
| lane | `hybrid_nav2_strafer` + `DEPTH_SUBGOAL`, `anchoring=mission`, `depth tick semantics=timer_reuse` |
| cadence contract | `publish 30.00 Hz sim`, `frame_skip=3 (derived, derived 3)`, bridge tick 120 Hz, script defaults |
| RTF | **0.106** over the whole run (980.8 s sim in 9257.0 s wall) |

## Files

| file | holds |
|---|---|
| `missions.jsonl` | one record per mission: goal, start pose, bearing, full track, verdict. Tracks decimated to 10 Hz sim from ~120 Hz; `track_raw_samples` records the original count |
| `tf_drift.jsonl` | map→odom **change points** — 1095 rows retained from 86 393 raw 10 Hz samples. `map→odom` is piecewise constant between SLAM corrections, so dropping the repeats is lossless *for the statistics reported here*: re-running `analyze_tf.py` on the compacted file reproduces the correction count (395), the jump quantiles, and τ to the printed precision. It was not diffed sample-by-sample against the raw series, which was not retained |
| `analyze_tf.py` | the drift analysis, so the compaction claim above and `drift_summary.json` can both be re-derived |
| `drift_summary.json` | correction statistics and autocorrelation |
| `anchor_per_mission.json` | cross-track, cursor advance and admission reasons per mission |
| `repeatability.jsonl` | the 2026-08-19 fixed-goal repeats (see the addendum), same record shape |
| `probes.txt` | raw output of the addendum's read-only probes (guidance, command character, costmap neighbourhood, open-loop actuation), each stamped with the pose it was taken at |

Velocity fields in both JSONLs (`track.v_par`, `track.speed`, and the
`mission.v_par_*` aggregates) are **derived offline from the position series** —
the runner's own per-sample differencing compared consecutive TF lookups, which
return the same transform between TF updates, and collapsed to zero. Per-sample
values are raw finite differences at the ~0.1 s sim track spacing and are
quantisation-noisy; the aggregates, and every velocity quoted below, use a 0.5 s
sim uniform resample of the same series. `mission._velocity_fields` restates this
in each record.

## Transport and cadence — the stack delivered its contract

The link is no longer the constraint. iperf3 across the cable: **943/941 Mbit/s**
forward, **941/940 Mbit/s** reverse (the depth direction), **0 retransmits**. At
`608 × RTF` the deployed census is ~64 Mbit/s against 940 delivered.

The configuration under test is the one that collapsed on WiFi:
`ros2 topic info /d555/depth/image_rect_raw --verbose` reported
`Subscription count: 2` — `strafer_inference` **and** `timestamp_fixer`, both
`RELIABLE`. Every figure below is differenced from the **complete** container log
for this session — 886 counter lines spanning the node's whole life, 948.8 s sim
(~2.6 h wall). Cumulative counters make the span load-bearing: a figure taken
from a partial capture is smaller but not wrong, so each is stated with the span
it covers. Under that load:

| quantity | value |
|---|---|
| cadence, `d(inferences)/d(span_sim)` over 787 windows | p05 **28.18** · p50 **30.00** · p95 **31.19** Hz sim |
| windows below the 27 Hz floor | 18 / 787 (2.3%), all straddling a mission boundary |
| inferences on a **fresh** frame | **24 892 / 24 892** (`reuse = 0`) |
| `depth_age` | p50 **0.025** · p95 **0.025** · max **0.033** s sim (one frame period is 0.0333 s) |
| `timer_deadline_missed` | **0** |
| `bad_encoding` / `bad_shape` / `obs_none` / `gate` skips | **0 / 0 / 0 / 0** |
| `Cadence disagreement` lines | **0** (expected: the sidecar carries no `trained_period_s`) |

**Do not read the node's own `rate` field for cadence.** `_cadence_t0_sim` is set
at the first inference and never reset, so `rate` is a lifetime average that
inter-mission idle drags down; this run logged **220** `CADENCE SHORTFALL`
warnings ramping smoothly 11.46 → 26.0 Hz while the instantaneous rate was
30.00 Hz. The warning's *attribution* is correct throughout — it names
`watchdog` skips, never depth.

**Duplicate depth content is bimodal, not a rate.** Differencing
`repeat_content` against `depth rx` over the 885 windows with `d(depth rx) > 0`
gives either 0% or ~50%, never in between (8 windows land in a 5–40% band).
Blocks of at least 20 s sim, on the same 948.8 s sim axis as the counters:

| block (s sim from the first counter line) | duplicate fraction |
|---|---:|
| 2.3 – 52.4 | ~50% |
| 52.4 – 142.7 | 0% |
| 142.7 – 448.7 | ~50% (three short 0% interruptions) |
| **449.7 – 948.6** | **0%** |

Session mean 17.3%. The 50% regime is every other published frame repeated —
30 Hz sim publish carrying 15 Hz of new content. Frames arrive; some carry no new
information, so this is a render-side content defect, not a delivery one.

## Missions

Fixed start heading **130°**, held to within 2.6–2.8° on every mission by a
closed-loop in-place rotation. Goals were selected from a plannability probe
(`ComputePathToPose`, 102 candidates, 40 reachable) rather than chosen by eye, so
each carries a measured path/straight-line ratio and costmap neighbourhood.
Bearing is measured from the actual start pose to the goal, relative to the
start heading; negative is right of heading, positive left.

| mission | bearing | start dist | final dist | min dist | v_par while moving | net advance | cursor adv. | action |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M1 dead-ahead | **+0.8°** | 2.95 m | 1.85 m | 1.85 m | 0.026 m/s | **+1.096 m** | 1.86 m | ABORTED |
| M2 right, region N | −107.2° | 3.38 m | 2.85 m | 2.73 m | 0.035 m/s | +0.537 m | 0.00 m | ABORTED |
| M3 right, furniture | −57.4° | 3.69 m | 3.72 m | 3.59 m | 0.026 m/s | **−0.028 m** | 0.01 m | ABORTED |
| M4 left, region W | +47.7° | 2.48 m | 1.79 m | 1.79 m | 0.045 m/s | +0.687 m | −0.01 m | ABORTED |
| M5 left, region SW | +89.5° | 2.98 m | 2.83 m | 2.81 m | 0.033 m/s | +0.154 m | 0.22 m | ABORTED |
| M6 left, corner | +90.6° | 2.31 m | 1.88 m | 1.84 m | **0.083 m/s** | +0.435 m | 0.31 m | ABORTED |

Tolerance is 0.30 m (`GOAL_ARRIVAL_RADIUS_M`, which is also the node's declared
`goal_reached_distance_m`). Closest approach across the six was **1.79 m** — no
mission came within 6× the tolerance.

The set covers what it was meant to: one dead-ahead (+0.8°), three right, three
left, four map regions, a furniture-standoff goal (`cost_goal 0` with
`cost_near 99`, i.e. lethal cost within 0.5 m) and a wrapping goal
(path/straight-line ratio 1.48).

**Every mission ended `ABORTED` at exactly 60.0 s sim**, which is
`mission_timeout_s` (`strafer_inference/config/inference.yaml:58`), enforced on
the node clock. The timeout is not the whole story: at cut-off M1 was still
closing but at 0.013 m/s, needing ~120 s more at that rate.

### Cross-track developed, and was mostly consumed

| mission | anchored arc | cross-track start / max / end | developed | consumed | collision admissions |
|---|---:|---|---|---|---:|
| M1 | 3.73 m | 0.030 / 0.427 / 0.078 | yes | yes | 1 |
| M2 | 3.43 m | 0.038 / 0.463 / 0.329 | yes | yes | 0 |
| M3 | 3.72 m | 0.023 / 0.184 / 0.183 | yes | no | 2 |
| M4 | 3.47 m | 0.029 / 0.359 / 0.110 | yes | yes | **14** |
| M5 | 3.66 m | 0.040 / 0.055 / 0.005 | no | no | 0 |
| M6 | 2.48 m | 0.015 / 0.078 / 0.044 | yes | no | 1 |

Cursor advance along the anchored path is the sharper reading: M2 advanced
**0.00 m** of a 3.43 m anchored path while still closing 0.54 m of straight-line
distance, so it moved goalward without tracking the path at all.

## map→odom drift — measured, and ruled out as the cause

395 corrections over 980.8 s sim (24.2 per sim minute).

| | value |
|---|---|
| translation jump | p50 **0.0126** · p95 **0.0770** · max **0.6279** m |
| yaw jump | p50 **0.230** · p95 **1.395** · max **18.32** ° |
| inter-correction gap | p50 0.517 · p95 9.17 · max 56.5 s sim |
| autocorrelation | **τ(1/e) = 18.5 s sim**; lag-1 0.972, lag-10 0.739, lag-60 0.086 |
| total map→odom travel / net shift | 9.26 m / 0.051 m |

The two largest corrections (0.628 m with 13.0°, and 0.337 m with 18.3°) both
landed inside unscored transits, never inside a scored mission. Restricted to the
**six gate missions**: **100 corrections, max 0.168 m, p95 0.069 m, and zero at
or above the 0.30 m tolerance.** (Adding the pilot's own window brings the count
to 137 with the same max; the pilot is not one of the six, so 100 is the
denominator for the gate.) The map frame was stable to well inside the pass
margin for every scored mission, so drift does not explain the outcome.

## What this set does and does not attribute

Ruled out by measurement, not by argument — with the scope of each stated, since
"ruled out" is only as wide as the instrument:

- **Transport.** 940 Mbit/s against ~64 Mbit/s of demand; 24 892 of 24 892
  inferences on fresh frames; `depth_age` max 0.033 s sim, i.e. one frame period.
- **Cadence.** 30.00 Hz sim median, 2.3% of windows below the 27 Hz floor and
  those only at mission boundaries; zero missed timer deadlines.
- **Observation assembly — the *mechanics*, not the *content*.** Zero
  `obs_none`, `gate`, `bad_encoding`, `bad_shape`, so the vector was assembled
  every tick without a malformed input. This says nothing about whether the
  assembled values match what training produced: no obs-parity capture was taken,
  and residual observation-chain deltas remain a live attribution candidate.
- **Drift — magnitude within scored windows.** Zero corrections at or above the
  tolerance across the six gate missions. Not excluded: the effect of the
  wall-clock freshness defect on the subgoal stream those corrections feed.
- **Duplicate depth content.** The final 499 s sim (~78 min wall) ran at 0%
  duplicates and covered M4, M5, M6 and M3; those advanced *less* than M1 and M2,
  which ran in the ~50% regime. Anti-correlated with outcome.
- **Goal bearing.** Advance does not order by bearing: the dead-ahead mission
  (+0.8°) advanced most of the six, and the two best `v_par` values sit at +90.6°
  and +47.7°.

What remains, stated as the measurement rather than a diagnosis: **the policy
closes distance at roughly 0.026–0.083 m/s while moving, against goals 2.3–3.7 m
away and a 60 s sim budget.** Training uses 20.0 s episodes
(`_DEFAULT_NAV_EPISODE_LENGTH_S`) with `min_goal_distance = 2.0` m, which demands
≥0.1 m/s of net closing; the observed rate is roughly half that at best and a
quarter at worst.

One data point contradicts a blanket reading, and is retained in
`missions.jsonl` as `PILOT_uncontrolled_heading`: an earlier run on the same
stack, same artifact and same scene **reached tolerance** — 3.03 m goal, final
0.299 m, 53.6 s sim, `v_par` 0.107 m/s while moving, `SUCCEEDED`. It is not one
of the six because it ran before the fixed-heading correction landed (its start
heading was 14.5°, not 130°). It establishes that this stack and artifact can
reach a ~3 m goal inside the 60 s budget, and that the six-mission result is
therefore about consistency rather than capability.

## Addendum, 2026-08-19 — the mechanism, isolated

A follow-up run on a fresh bridge (`kit_20260819_190207.log`, same task and seed,
SLAM key `enrich_pingpong1`) repeated **one fixed goal** (−2.00, 2.25) from a
fixed start pose and heading, which the six-mission set could not do: each of its
missions used a different goal, so it measures coverage, not reliability.

**Repeatability: 0/3, and the three runs converge on the same place.** All three
ended within 4 cm of each other — (−0.36, −0.49), (−0.37, −0.48), (−0.35, −0.46).
The first drove there from the origin; the second and third began there and never
left. None finished within 3.1 m of a goal 0.30 m in tolerance. Net change in
distance to goal was **+0.071 m (R1), −0.036 m (R2), +0.085 m (R3)** — two ended
farther than they started and one ended 3.6 cm nearer, so the honest statement is
that **none of the three made material progress**, not that all three retreated.

**The guidance the policy receives is correct.** Measured live, with the robot
stationary and a goal active:

| | measured |
|---|---|
| `active_goal` | (−2.00, 2.25) — the submitted goal, 3.12 m away |
| `/plan` | 147 poses, 3.69 m, terminating **0.00 m** from the goal |
| `/strafer/subgoal` | 0.81 m ahead of the robot, into free space |

So the goal is right, the plan reaches it, and the subgoal is a sane local
target — at the pose sampled. Nothing upstream of the policy was misdirecting it
there.

> **These probes are a juxtaposition, not a simultaneous capture.** The guidance
> table above and the command table below were taken by **separate** read-only
> probes, minutes apart, at **different robot poses** — the guidance probe at
> (−0.26, −0.35) facing 111°, the command probe after an intervening open-loop
> test had moved the robot, at (−0.345, −0.335) facing 122°. Both poses sit in
> the same attractor and both carried an active goal on the same fixed target, so
> the pairing is fair, but no single instant was captured showing correct guidance
> and an off-goal command together. A simultaneous capture is the obvious
> strengthening of this result and was not taken. The raw probe outputs are in
> `probes.txt`.

**The policy's command is sustained and points the wrong way.** Over 240
consecutive `/cmd_vel` messages (30.1 Hz sim), with the goal essentially dead
ahead (robot facing 122°, goal bearing 122.6°):

| axis | mean (signed) | mean abs | duty | sign flips |
|---|---:|---:|---:|---:|
| `vx` | **−0.114** | 0.121 | 0.94 | 1.5 Hz sim |
| `vy` | **+0.224** | 0.224 | **1.00** | **0** |
| `wz` | +0.060 | 0.119 | 0.50 | 8.0 Hz sim |

`vy` is a *perfectly* sustained left strafe — duty 1.00, not one sign change in
8 s sim — combined with a **backward** `vx`. That is a **~117° sustained
directional error in the robot frame** toward a goal that is straight ahead. It
is not dither, not under-commanding, and not a magnitude problem.

**Backward is into an obstacle.** At the parked pose the nearest lethal costmap
cell is **0.21 m at −174° relative to heading** — directly behind the robot —
with 37 lethal cells inside 0.5 m. The policy reverses into it.

**The chassis is not the fault, and neither is wedging.** Driven open-loop with
the policy holding no goal, the robot moves freely on every axis:

| axis | achieved over 3.5 s sim | of commanded |
|---|---:|---:|
| forward / backward | 0.339 / 0.320 m | ~32% / ~30% |
| left / right | 0.468 / 0.335 m | ~45% / ~32% |
| yaw | +45.0° | ~37% |

Under the policy in free space the same ratio holds (**36.3%** translation,
64.9% rotation), so the ~3× command-tracking deficit is the modelled actuation
(motor dynamics τ=0.05 s, 1–3 step command delay, slew limit, command hold) and
is present identically whoever is driving. It lowers the policy's effective
speed to a third of what it commands; it does not steer it. The one place the
ratio collapses to **5.6%** is at the parked pose, which is exactly where the
robot is pressed against the obstacle behind it.

**Map noise is not supported at short timescales.** Sampled per costmap update
for 144 s wall with the robot stationary (0.007 m of drift), the global costmap
was stable: the robot's own cell held constant, and the ≥99 cell count varied by
7 in 5125 (0.1%). Over the longer run the robot did enter the inscribed band 131
times and spent **64% of 44 min** inside it — but that is a consequence of
parking against an obstacle, not evidence of a flickering map.

> **Reading note for anyone re-running this.** `/global_costmap/costmap` is
> published in nav2's **0–100 display scale** (100 lethal, 99 inscribed), not the
> raw 0–254 cost. A `>= 254` test on that topic silently returns zero obstacles
> in a fully furnished room. This session made that mistake before catching it.

**What this adds, and where it sits in the attribution record.** The 0/6 stands,
and its shape is now narrower than "under-advance": the deploy stack delivers a
correct goal, a correct plan, a correct subgoal, fresh depth at 30 Hz sim, and a
chassis that tracks commands the same for any driver — and the policy answers
with a sustained strafe roughly 117° off the goal direction, reversing into
mapped geometry.

**This is new evidence on a reopened question, not a reproduction of a standing
call.** The 2026-08-01 four-arm session's `policy-owned` conclusion has been
superseded twice and should not be cited as current:
[`enriched-scene-anchoring-addendum`](../../tasks/completed/enriched-scene-anchoring-addendum.md)
records that "the `mission` ✗ result stands; the **policy-owned attribution does
not**, and **no retrain is licensed on it**", and retires the 2026-08-01
attribution outright; [`cadence-harness-residual-arms`](../../tasks/completed/cadence-harness-residual-arms.md)
then states the enriched-lane
advance failure is "**unattributed on every axis that has been tested**", naming
four live candidates — SLAM-frame anchoring noise, planner path-geometry
distribution, unbounded recurrent-state horizon, and residual observation-chain
deltas.

Against that register, this session's contribution is specific. It supplies the
enriched × `mission` cell **at a clean temporal profile** — the thing the
2026-08-02 arm could not, having run at 11.68 Hz depth arrival, 38.3% repeat
content and 11 146 deadline misses, against this run's 30.00 Hz, `reuse = 0` and
zero deadline misses. It therefore bears on two of the four candidates and
excludes neither of the others:

| candidate | what this session says |
|---|---|
| SLAM-frame anchoring noise | **Bounded, not excluded.** Within the six gate windows the map frame moved at most 0.168 m, none at or above tolerance. But the wall-clock freshness defect duty-cycled the collision admission rule for roughly half of its evaluations, and the effect of that on the subgoal stream was **not** measured. |
| planner path-geometry | **Sampled, not characterised.** `/plan` was verified to reach the goal at the poses sampled; no distribution over plans was taken. |
| unbounded recurrent horizon | **Untested, and this run sits squarely in it.** Every mission ran to the 60 s `mission_timeout_s`, i.e. up to 1800 hidden-state advances against training's 600. |
| residual observation-chain deltas | **Untested.** No obs-parity capture was taken; the assembled observation *content* was never compared against the training pipeline. |

Two further deploy-side items are also outside what was measured: the **TRT
execution path** on this device (the export was exonerated at the constants level
in the addendum, but the on-device execution was not re-checked here), and the
subgoal stream under the freshness defect above. So the correct summary is that
**transport, cadence and temporal texture, chassis actuation, map-frame
displacement magnitude, and goal/plan/subgoal correctness at the sampled poses
are excluded — not that every deploy-side alternative is.** The residual is
consistent with the directional bias
[`depth-camera-vfov-parity`](../../tasks/completed/depth-camera-vfov-parity.md)
described on 2026-08-01, but that call is retired, and nothing here re-establishes
it; what this run does is remove the temporal confound that made the 2026-08-02
cell unusable, leaving the remaining candidates to be discriminated.

**Operator observations from ad-hoc testing, 2026-08-19** (reported, not
instrumented, and recorded as such): ~5 natural-language missions, none advanced
correctly and most drifted in unrelated directions; 4 direct pose goals, 2
reached, with a difficult initial start on the successes. The NL failures are
plausibly a *different* mechanism — a policy that under-advances toward a correct
goal does not produce motion in an unrelated direction — and the grounding →
projection → dispatch chain was never exercised by this set, which submitted
poses directly. Isolating it needs one NL mission captured with
`docker logs strafer_autonomy` alongside the dispatched pose.

## Confounds this set carries

1. **The collision admission rule was duty-cycled, not in force.** The subgoal
   generator's freshness guards compare `time.monotonic()` (wall) against
   timeouts sized in sim units. The costmap arrives every ~1.05 s sim ≈ 9.1 s
   wall against a 5.0 s guard, so the rule was unavailable for roughly 45% of
   evaluations. Both states were observed: the warning fired throughout, and
   `anchor_in_collision` admissions still occurred (40 across the run, 14 of
   them inside M4). Filed as
   [`subgoal-generator-sim-clock-freshness`](../../tasks/active/reliability/subgoal-generator-sim-clock-freshness.md).
2. **Start pose varied; only start heading was fixed.** Nav2 cannot reposition
   from a just-parked pose in this scene — `controller_server` reports
   `Resulting plan has 0 poses in it` and the goal fails, once after 202 s wall
   with the robot never moving. Transits therefore ran on the policy itself and
   were aborted by the same 60 s timeout, landing 0.20–3.53 m from the nominal
   start. Every mission's actual start pose and bearing is recorded.
3. **A minimum-start-distance floor was added to the harness mid-set.** One run
   scored `REACHED` from a start 0.201 m from its goal — inside the 0.30 m
   tolerance — with `sim_elapsed 0.01 s` and a single track sample: a pass
   recorded without the robot moving. A 1.20 m floor was added, after which any
   mission starting nearer is refused as `INVALID_START` rather than scored. The
   floor fired once, on the original furniture-standoff goal (item 4). No scored
   mission in this record began inside it, and the discarded run is not counted
   in the 0/6.
4. **The furniture-standoff goal was moved outward.** Its original coordinates
   sat 1.21 m from the start pose, so a successful transit left the robot 0.686 m
   away — inside a distance that measures nothing. It was moved to a goal the
   probe placed in the same class (`cost_goal 0`, `cost_near 99`) at 3.17 m. The
   refused run is retained in the raw logs.
5. **Duplicate-content regime changed mid-run**, 50% → 0% for the final 499 s sim (~78 min wall),
   so M1/M2 and M3–M6 were not taken in the same render regime.
6. **No secondary artifact arm was run.** A v1 `DEPTH_SUBGOAL` artifact is
   present and runnable (`policy.onnx`, `run_20260708_005923/model_500.pt`,
   `obs_dim 3619`), but the fixed acceptance criteria stop the set on a fail
   rather than continuing to spend rig time. That comparison is the obvious first
   arm of the next run and would separate a policy-specific under-advance from a
   lane-wide one.
