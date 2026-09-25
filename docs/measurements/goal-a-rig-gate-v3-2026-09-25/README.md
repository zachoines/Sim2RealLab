# Depth-subgoal v3 mission gate on the sim-bridge lane, 2026-09-25

The v3 depth-subgoal artifact reached **4 of 6** scored missions within the 0.30 m tolerance,
and **3 of 3** repeats of the fixed goal. The lane was the enriched ProcRoom sim bridge over the
direct cable, and the protocol was the one
[`goal-a-rig-gate-2026-08-17`](../goal-a-rig-gate-2026-08-17/README.md) ran for v2. The thresholds
were written down and deposited before the first mission: ≥ 4 of 6 reads as **PASS**, so the
sim gate for goal-a closes on this set. The two misses advanced 2.25 m and 2.38 m and stopped
0.35 m and 0.51 m short. The fixed-goal leg, which measures reliability at one goal, reached on
every repeat. In the same session, from the same starts, v2 reached neither of its two
descriptive runs, closing at 0.014–0.027 m/s against 0.026–0.083 m/s on 2026-08-17. No threshold
reads v2.
This is the gate the brief
[`depth-subgoal-v3-retrain`](../../tasks/active/trained-policy/depth-subgoal-v3-retrain.md) is
accepted against.

## Setup

| | |
|---|---|
| scene | `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, `Environment seed : 42`, one bridge launch for the whole session |
| Kit log | `kit_20260924_222850.log` |
| cadence contract | `publish 30.00 Hz sim`, `frame_skip=3 (derived, derived 3)`, bridge tick 120 Hz, script defaults (no cadence flags) |
| SLAM key | `enrich_rigv3gate1` (fresh for this bridge launch) |
| images | `strafer-cpu:humble` / `strafer-gpu:humble`, both `bd240ba2c2bf`, built clean at the merge of #229 |
| candidate | `strafer_depth_subgoal_v3_999.onnx`, sha256 `c866bfd54ec1a8352159e33d7875d41e3f07a442ff8301ba3700867932e2eb91`, verified in the running container |
| descriptive arm | `strafer_depth_subgoal_v2_998.onnx`, sha256 `855e1df7d0dac3be7229f933b59546b26f18959f97966b9e2f2f22e752bf5165`, swapped in-container and verified |
| lane | `hybrid_nav2_strafer` + `DEPTH_SUBGOAL`, `anchoring=mission`, `depth_tick_semantics: timer_reuse`, `mission_timeout_s` 60 (node clock), tolerance 0.30 m (`GOAL_ARRIVAL_RADIUS_M`) |
| subgoal generator | freshness windows on the node clock ([`subgoal-generator-sim-clock-freshness`](../../tasks/completed/subgoal-generator-sim-clock-freshness.md), #229): the collision admission rule was in force for the whole session |
| transport | direct cable, 943/941 Mbit/s forward and 943/940 reverse with 0 retransmits (measured the same day); the Cyclone interface pin selected the cable in every running container, including the recreated v2 container; two RELIABLE depth subscribers (`strafer_inference`, `timestamp_fixer`) |
| start | nominal (0.075, −0.035), heading 130°, reached before every run by a scripted `cmd_vel` transit and closed-loop heading hold, not by Nav2 or the policy |
| RTF | 0.1306 over the scored session (769.3 s sim in 5892.9 s wall) |
| window | first goal 03:34:47 UTC, last result 05:09:21 UTC |

## Pre-registration

`preregistration.md` was committed to the evidence deposit at 03:30:11 UTC and pushed at 03:30:13.
The first goal was sent at 03:34:47. The file carries:

- the thresholds, verbatim;
- how PARTIAL's qualifiers are read over the set;
- the definition of a reach (the node's own `NavigateToPose` returns `SUCCEEDED`, with the TF
  distance as a cross-check);
- the narrow causes for which a run is unscored;
- the start handling, the floor, the goal table, the order, and an answer to each of the
  2026-08-17 set's six confounds.

Its sha256 is `69a6807fb828bf88eca15a1af2c5d7acdcffbaab831133baecf1b01277c4c00c`. It was not
edited afterwards.

The table is quoted verbatim from that file:

| outcome | reading |
|---|---|
| ≥ 4 of 6 missions reach their goal within 0.30 m | **PASS — G-sim closes** (goal-a's sim gate) |
| 1–3 of 6, with positive `v_par` and cross-track that develops and is consumed | **PARTIAL** — full capture, per-mission attribution, no tuning in-session |
| 0 of 6, or no net advance | **FAIL** — full capture, STOP, no tuning |
| fixed-goal leg | reported alongside: ≥ 2 of 3 reach = reliable at that goal; < 2 named as the reliability finding |

**The goals** came from a plannability probe: `ComputePathToPose` from the nominal start over
472 candidates, all of them plannable. The probe ran on the post-warm-up map of a separate
dry-run bridge launch with the same task and seed. The six are:

| id | kind |
|---|---|
| G1 | dead-ahead; its path wraps the block in front of the start |
| R1 | a wrapping goal: its straight line crosses the block |
| L1 | left, W region |
| R2 | a furniture standoff: lethal cost 0.39 m from the goal |
| L2 | left, SW region |
| R3 | open floor, NE region |

The split is one dead-ahead, two left and three right, over four regions. G1 is also the
fixed goal. A read-only re-check on the scored map found all six plannable before the first
mission.

## Verdict

| set | reached | reading |
|---|---:|---|
| the six | **4 of 6** (G1, R1, L1, L2) | **PASS** |
| fixed-goal leg | **3 of 3** (F1, F2, F3) | **reliable at that goal** |
| v2 (descriptive) | 0 of 2 | — |

Every reach agrees with its TF cross-check: the TF distance at the result was ≤ 0.30 m in each
case. No run was unscored, and no mission was re-run.

## The six

Bearing is taken from the actual start, relative to the actual heading; + is left.

Column key:
- *start off* is the start's offset from the nominal pose (position, then heading).
- *v_par* is the while-moving mean velocity along the start→goal line, from a 0.5 s sim
  resample of the position series.
- *dup* is the in-window duplicate-content fraction, `repeat_content / depth rx`.
- *coll* is the in-window count of `anchor_in_collision` admissions.
- *cross-track* is given as start / max / end, and *dev/con* says whether it developed and
  whether it was consumed, by the rule the pre-registration fixes (±0.05 m).

| mission | goal | bearing | start dist | start off | status | final | min | net advance | sim s | v_par | dup | coll | cross-track s/max/end (m) | dev/con |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| G1 dead-ahead | (−2.00, 2.25) | +0.8° | 3.11 | 0.043 m / 2.2° | **SUCCEEDED** | 0.300 | 0.300 | +2.808 | 19.0 | +0.257 | 0.00 | 0 | 0.155 / 0.218 / 0.026 | y / y |
| R1 wrap, N | (−0.05, 2.15) | −38.1° | 2.16 | 0.027 m / 1.2° | **SUCCEEDED** | 0.290 | 0.290 | +1.873 | 6.6 | +0.318 | 0.41 | 2 | 0.013 / 0.275 / 0.267 | y / n |
| L1 left, W | (−2.00, 0.00) | +50.5° | 2.07 | 0.020 m / 0.9° | **SUCCEEDED** | 0.299 | 0.299 | +1.768 | 11.1 | +0.219 | 0.44 | 0 | 0.075 / 0.161 / 0.018 | y / y |
| R2 standoff, N | (1.00, 2.35) | −60.5° | 2.60 | 0.066 m / 1.8° | ABORTED | 0.350 | 0.347 | +2.250 | 60.0 | +0.250 | 0.49 | 0 | 0.040 / 0.223 / 0.110 | y / y |
| L2 left, SW | (−2.75, −2.25) | +88.8° | 3.63 | 0.037 m / 0.7° | **SUCCEEDED** | 0.300 | 0.300 | +3.327 | 15.4 | +0.256 | 0.45 | 0 | 0.069 / 0.094 / 0.024 | n / n |
| R3 open, NE | (2.65, 1.25) | −105.3° | 2.89 | 0.031 m / 1.2° | ABORTED | 0.508 | 0.499 | +2.383 | 60.0 | +0.185 | 0.48 | 6 | 0.088 / 0.314 / 0.079 | y / y |

The set-level reads are the pre-registration's own, reported although PASS does not need them:
- the median while-moving `v_par` is **+0.253 m/s**;
- the median net advance is **+2.32 m**;
- cross-track developed and was consumed in **4 of 6**.

On 2026-08-17, v2 closed at 0.026–0.083 m/s on the same protocol, and its closest approach
over six missions was 1.79 m.

### The two misses

The record does not attribute either miss. What each run did:

**R2, the furniture standoff.**
- The robot was within 0.37 m of the goal 9.9 s sim into the mission.
- For the remaining ~50 s sim it held 0.347–0.359 m from the goal, commanding a mean
  0.020 m/s over the last 10 s sim.
- The goal cell is free, and lethal cost lies 0.39 m from it. On the pre-mission costmap
  snapshot, the nearest lethal cell to the robot's final pose was 0.64 m away.
- The node aborted at 60.0 s sim.
- `map→odom` corrections in the window were each ≤ 0.160 m. Summed over 60 s sim, they
  moved `map→odom` by 0.33 m.

**R3, open floor, NE — run on a displaced SLAM estimate.**
- During the transit that brought the robot to the start for R3, `map→odom` stepped by
  **0.953 m and −75.7°**. It held there for the whole mission, at about (−0.8, +0.1, −77°)
  against about (0.08, −0.05, +1.3°) for the runs before and after.
- It stepped back by 0.941 m and +77.5° during the next transit, before F1.
- At the first step, rtabmap's map-update time rose from ~0.005 s to 0.17 s for three
  iterations. Its log records no accepted loop closure at INFO level.
- In R3 the robot came within 0.53 m of the goal by 17.7 s sim and ended 0.499–0.508 m away,
  commanding a mean 0.017 m/s over the last 10 s sim.
- The generator admitted six new anchors on the collision rule, the most of any run, and
  the anchored arcs ranged from 0.2 m to 8.05 m for a 2.89 m goal.
- The generator's one mid-mission `plan is stale` episode in this run coincided with a
  planner refusal: `ComputePathToPose` status 6, after which the relaxed planner engaged. The
  subgoal went stale for 13 policy ticks there, and the inference node skipped them on its
  watchdog; no other scored window had more than one such skip.
- None of the pre-registration's unscored causes applies: a start outside tolerance or inside
  the floor, a broken stack contract (depth stall, container restart, SLAM FATAL), or a
  wall-guard cancel. It names no cause for a displaced SLAM estimate, so R3 is scored as run.

## Fixed-goal leg

G1 (−2.00, 2.25) was run three times from the nominal start and heading, after the six. G1's
own run inside the six counts only toward the six.

| repeat | start off | bearing | status | final | net advance | sim s | v_par | dup | coll | cross-track s/max/end |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| F1 | 0.006 m / 2.0° | +0.3° | **SUCCEEDED** | 0.297 | +2.796 | 10.7 | +0.309 | 0.42 | 0 | 0.057 / 0.280 / 0.213 |
| F2 | 0.004 m / 0.8° | +2.9° | **SUCCEEDED** | 0.292 | +2.793 | 6.8 | +0.433 | 0.00 | 0 | 0.016 / 0.196 / 0.032 |
| F3 | 0.019 m / 0.6° | +1.6° | **SUCCEEDED** | 0.284 | +2.784 | 6.8 | +0.429 | 0.00 | 0 | 0.090 / 0.285 / 0.009 |

**3 of 3 reached**, so v3 is reliable at this goal. On 2026-08-19 v2 scored 0 of 3 at the same
goal, all three runs ending within 4 cm of each other.

**F1's window carries the scored session's one in-window correction at or above the
tolerance.** `map→odom` stepped **0.497 m and −17.35°** at 2.12 s into the 10.72 s sim mission.
The corrections after it were 0.20 m or less, and none exceeded 0.022 m in the last 3 s sim
before the result, so the jump did not deliver the reach.

**F2's first transit settled at 3.00° off the 130° heading.** That is inside the runner's
enforced 3.0° but outside the transit's own 2.5° target, so the transit was repeated before
any goal was sent.

## The v2 descriptive arm

v2 was swapped in after the fixed-goal leg: the host artifact key was changed, `inference` was
recreated, and the sha was verified in the container. It ran on the same bridge, map and start
handling. v3 was restored and verified afterwards.

| run | goal | bearing | status | final | min | net advance | v_par | cross-track s/max/end |
|---|---|---:|---|---:|---:|---:|---:|---|
| V2_G1 | (−2.00, 2.25) | +2.6° | ABORTED | 2.097 | 2.097 | +0.967 | +0.027 | 0.058 / 0.515 / 0.351 |
| V2_L1 | (−2.00, 0.00) | +48.3° | ABORTED | 2.129 | 1.831 | −0.052 | +0.014 | 0.120 / 0.499 / 0.438 |

V2_G1 repeats the 2026-08-17 M1 shape: that run ended 1.85 m from the same goal with `v_par`
0.026 m/s. The generator admitted cross-track re-anchors on both goals (two each). It also
recorded mid-mission planner refusals: one on V2_G1, and 13 on V2_L1, where three starvation
holds started.

## The stack's own contract

These figures come from the v3 inference container's complete log. It hosted the nine v3 runs
and ran for 542 s sim.

Restricted to the six scored windows (`table.md`), the same reads are: cadence p50 30.00–30.01 Hz
sim in every window and p05 ≥ 28.57; 5145 of 5145 inferences on a fresh frame; `depth_age` max
0.033 s sim; 0 missed deadlines; no `bad_encoding`, `bad_shape`, `obs_none`, `gate` or
`action_shape` skip. Mid-mission watchdog skips were 0–1 per window, at goal acceptance, except
R3's 13 (above).

| quantity | value |
|---|---|
| cadence, `d(inferences)/d(span_sim)` over 146 clean windows | p05 **28.57** · p50 **30.00** · p95 **31.54** Hz sim (pooled 29.94); 2 windows below 27 Hz |
| inferences on a fresh frame | **5871 / 5871** (`reuse = 0`) |
| `depth_age` | p50 **0.025** · max **0.033** s sim (one frame period) |
| `timer_deadline_missed` | **0** |
| `bad_encoding` / `bad_shape` / `obs_none` / `gate` / `action_shape` skips | **0 / 0 / 0 / 0 / 0** |
| `Cadence disagreement` lines | **0**: the sidecar's `trained_period_s` 0.0333 equals the configured period |
| `Costmap is older than` lines | **0** over the whole session |

- The v2 container reads the same way: 3468 of 3468 inferences on a fresh frame, cadence p50
  29.99 Hz sim, `depth_age` max 0.042 s sim, 0 deadline misses, 0 disagreement lines.
- The node's lifetime `rate` field (10.8 Hz) and its 110 `CADENCE SHORTFALL` lines are the
  lifetime average dragged down by idle time between missions. They are not cadence, as the
  2026-08-17 record explains.
- **`plan is stale` fired 70 times in the v3 container, and once inside a scored window** (R3,
  above). The rest fall outside every mission window: after each mission's result, and after
  the scripted transits, whose `ComputePathToPose` results `planner_server` echoes on `/plan`
  to an idle generator.
- Every in-window occurrence in this session coincided with a planner refusal: R3 once, V2_G1
  once, and V2_L1 three times.

## Duplicate content and collision admissions, per mission

Both are reported beside every mission above. **Neither is comparable to the 2026-08-17 set.**
That set changed duplicate regime mid-run, where this one is a single launch; and since #229
the collision admission rule is in force for the whole interval rather than about half of it.

- **Duplicate content.**
  - The launch was at 0% for G1 (03:34–03:47 UTC), at ~41–49% for R1 through F1, and at 0%
    again from F2 onward, including both v2 runs.
  - Outcome does not order by regime: the 0% runs are G1, F2, F3 and the two v2 runs; the ~50%
    runs include three of the four reaches in the six, and both misses.
  - The regime is render-side and is not set from the deploy stack.
- **Collision admissions** inside the six windows: 8 (R1 2, R3 6). The v3 container's
  cumulative counter ended at 11; the other three fall outside mission windows.

## map→odom

A ride-along sampled `map→odom` at ~10 Hz sim for the whole scored session: 7406 samples. A
correction is a step of more than 1e-4 m or 0.057° between samples. The statistics use the
2026-08-17 record's analysis, run on the same series.

| | corrections | largest | p95 | at or above 0.30 m |
|---|---:|---:|---:|---:|
| the six scored windows | 62 | 0.160 m | 0.121 m | **0** |
| fixed-goal leg windows | 35 | 0.497 m (F1, 2.1 s into the mission) | 0.221 m | 1 |
| v2 windows | 76 | 0.029 m | 0.011 m | 0 |
| whole session | 534 | 0.953 m | 0.093 m | 4 |

- In the last 1 s sim before every result, no correction exceeded 0.037 m, so no reach was
  delivered by a map step.
- Of the four session steps at or above 0.30 m, three fell in transits: 0.315 m before L2,
  0.953 m / −75.7° before R3, and 0.941 m / +77.5° before F1. The fourth is F1's.
- Drift autocorrelation τ(1/e) is 63.5 s sim, and the net shift over the session is 0.05 m.

## Confounds of the 2026-08-17 set, as answered before the runs

1. **Collision rule duty-cycled.**
   - Fixed by #229 and in the images.
   - The generator logged no `Costmap is older` line in the session.
   - Collision admissions are counted per window (above).
2. **Start pose varied.** Every run started from the fixed nominal pose via the scripted
   transit, with offsets of 0.004–0.066 m and 0.3–2.2° against the runner's enforced
   0.15 m / 3.0°.
   - The transits took 7.8–78.6 s sim, used no escape, and were refused no plan.
   - The one repeat (F2) is described above.
3. **Floor added mid-set.** The 1.20 m floor was in force from the first mission and never
   fired: the shortest start distance was 2.07 m.
4. **Goal moved.** No goal was moved; all were fixed before the first mission.
5. **Duplicate regime changed mid-run.** This set ran on one launch at script defaults. The
   regime still changed within the launch; it is measured per mission above.
6. **No secondary arm.** v2 ran on two of the six goals in the same session.

## What this record does not claim

- **Real-sensor behaviour.** Every depth frame here is bridge depth. The real-sensor gate
  depends on the D555 decode and texture work and is not addressed.
- **A cause for either miss.** In particular, it does not establish whether R3's displaced
  SLAM estimate caused its miss, or what produced the step.
- **Comparability of the per-mission duplicate fraction or collision-admission counts** with
  the 2026-08-17 set.
- **Anything about v2 beyond the two descriptive runs.**

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `goal-a-rig-gate-v3-2026-09-25/record-files/` |
| deposit commit | `b513b2890abf137a78730f1ec07daca2a7d4ace5` |
| file deposit | `15cea6163fc25d77cf21fad6cf8d557ff6f6e9eb` (every file below; `b513b289` changes only `DEPOSIT.md`'s re-derive command) |
| pre-registration commit | `185d2bc30c4d5acb14e7307c72e05633b693b19d` (the file alone, before the first mission) |

The deposit holds everything this record names:
- the per-mission table and its JSON (`table.md`, `table.json`) and the drift summary
  (`drift_summary.json`);
- one runner record per run (`missions/`) and one transit record per run (`transits/`);
- the counter scrapes (`scrape/`) and the `map→odom` ride-along (`ride_along/`);
- the stage checks (`stage0/`) and every container log plus the bridge console (`logs/`);
- the probes that chose the goals (`goal_selection/`);
- the preparation session on its own bridge launch (`dry_run/`), which also carries the lane
  re-run #229's brief cites;
- the image builds and the test lane (`build/`), and the scripts that produced and analysed
  all of it (`tools/`).

Two text files had rig network addresses replaced with placeholders before deposit. Its
`DEPOSIT.md` gives the commands that re-derive `table.json` and `drift_summary.json`
byte-for-byte from the deposited files.

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/goal-a-rig-gate-v3-2026-09-25/record-files/. \
      docs/measurements/goal-a-rig-gate-v3-2026-09-25/
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/goal-a-rig-gate-v3-2026-09-25/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
8d4cf6394d1afb9354e9c24496a1ca820df46b33414509edc0226a1f90021440  build/make_images_branchhead.log
02e2cd1ce7245ba95a7e030359a9537c50cffff3acedfb122e79ee26517b01ff  build/make_images_mergehead.log
978d97f29d7f2b6ff9168504b03c08567ce28fe78620eb50b23c931e2b1fb32b  build/test_ros_mergehead.log
92c78820b83516925d24af8be674715471b26a54c472a8ebbe84087549a7994b  drift_summary.json
9e5b268a1911cab9e32e63a58f4b86ab007e1c390313722cfa486a68c2ac23d8  dry_run/DRY_T1.json
3879b774d8ec66276e51c86b5985922e20aeffedf09466feda7a93794c4f80af  dry_run/DRY_T1.out
c5da33154235aab6e27414050d691fbf48a31f3b78e1025ccdc991fddd912fc6  dry_run/DRY_T1_scrape.json
57d98d1f89cbeba2acfb416db43fd6f9d625e2d8ab1aa20fa9af99e240fc2f17  dry_run/a2_startup_facts.txt
4523284f0542e74b758a0f846ee3ac76171b1a00261e80fc4f1c24df172a407d  dry_run/cyclone_pin_in_containers_dry.txt
50fb7d30861f8debf37c288243334c42c12fc3f57c8f24905440582529e98bf5  dry_run/depth_topic_info_dry.txt
5852e8ce7e951b498c6ef7b2266687c58b413fd6ff998829064bc3f9fa2c94cc  dry_run/link_iperf3.txt
373dd57843d6a8a8eebfa5b97e25bbfe825ce5ca5d53ff388f7092cb608c274d  dry_run/rig_gate_v3_dry_bridge.log
9efd57ffb31e99c2cb3b5889a3a4f977099a4ffc70f41e7c3608fdcad814c6d2  dry_run/strafer_inference.cold.log
c862e60bcbd494223e83387b4c53975f195833a3408a61a3e71e2566aab55655  dry_run/strafer_inference.v2swap.log
ce5b6a72dd7be309ee2c3ca11cec90596815bbecc6ef18181b576565d2457ce0  dry_run/strafer_inference.v3restore.log
114060ede75064cfba66aa4fb5afc138d5c0c34f4dc6344608579f4980db7200  dry_run/strafer_inference.warm.final.log
e6f5942ab528dc07bb01235621b942de295eb7b47cea9227694f6f20ab0a5152  dry_run/strafer_slam.log
2ffde4c24db5fceb3271373752cef208af9602adf2096b0b55d72f34df559316  dry_run/tf_dry.jsonl
36e9bec758e34bf2e9c5f073a8dd3c9f553afe097a931de7b59b04aee907ac3c  dry_run/to_start_dry1.json
72237423ca54c1aa8d3968761cb4a61fd21fc477a5c5fa9dfa78c7b5cb9d450d  dry_run/to_start_dry1.out
a7b1bbb2064182205284a3e2e93a8c69c69f5968961d34e94f532caf3b4dc0c7  dry_run/to_start_dry2.json
0d9ac881044f39ed291dcba627a460453413ac0f9c0896c89f3cb8ae02715ef9  dry_run/to_start_dry2.out
073861b090f8ee4d980e8889f5c9549f0f59fb2fa34da6c883bc2f30e8a7351f  goal_selection/goal_selection.json
545eb5c716ac1f3f160dcc0aab3709fbbf61129c4964d8740a4d78a5b5afb5cc  goal_selection/probe_dry.json
f74a79b7af023ab793d84783a5867a9026779442bb07bbb5bfaf9953397ae611  goal_selection/probe_dry.stdout
210078d85fa13cdb1115757df676e7aa6c9fc42ab321551c9f975327d5c52374  goal_selection/probe_dry_costmap.npz
6c65257fd96fc9d3c22e377d6ea05053d2778c73516ea40300bb0edb18289d39  goal_selection/probe_dry_dense.json
e0fbade843121fc1536eb99b13f98ba3666ec1e87c4a67872fd69d402568a1f3  goal_selection/probe_dry_dense.stdout
d9875e4820cbe9b8bde4b9a6efcaf6e5915f04afd1173cc003af0c437aa76e2c  goal_selection/probe_dry_dense_costmap.npz
3dd00d5e884f3624ad6a97704681d519202bab27ed41f3b6b5ebfe6bf30a9197  logs/rig_gate_v3_bridge.log
7e43bf2bacacf81fec6f94186066ada10273f80b98d0305aa567bb850495ed7a  logs/rig_gate_v3_envsetup.log
d1f332db1add8e3fd41c5723dc8751939f17250c536da48ab736cab2b786f3c8  logs/runs/F1.mission.out
d4aa85f5c514eb091bffd98c0b96591d1160671c4282c27b025ee1026b369f0e  logs/runs/F1.to_start.out
5d8e76b5b4ab02381d028dc0bc5d9f319fb7c38137632b1d6c15a96a8c1aa3c0  logs/runs/F2.mission.out
e3e11cbdef83074bd24a970bfb2e6867bf49464a1d2dc53a478c53a6c87f5cbc  logs/runs/F2.to_start.out
5ee6bddaf4b6d3d701c6c9420f1bdc1aa59ac8522c5da3942a1992e2d62e1a7c  logs/runs/F3.mission.out
3fac75f393454f4d965a25e595ccc7fdebf0ca028751f8e5f4984de493d3dc9c  logs/runs/F3.to_start.out
900eccadf5fadebb7c7f377622bbcc9172cab4e26d78b2f649089a96a4194464  logs/runs/G1.mission.out
9bef0c5f6859ddd98218f04f814c06fe194527d9c1f2db43bb116af766d9d1d4  logs/runs/G1.to_start.out
b73b187218e96ba9af618bc21c30dd0769e969ee7a43726c0dd6afc951418056  logs/runs/L1.mission.out
9b645768c6474ea74ae775b31601f6a2e114e1d7746731f3da462c9d966a1b35  logs/runs/L1.to_start.out
16fe3ac465f9490d653e2af19cc660c755045d13f624f43780fcd521da5a8294  logs/runs/L2.mission.out
22945f1f7650c6388d18545a737e1f7a5fa234e692e03f5735ccfe695cbccb40  logs/runs/L2.to_start.out
edff57e59df5f51fdca57bc1442a5e5daa895953959356dcd175d445165629db  logs/runs/R1.mission.out
68f1f3a50a17149f4d9a35920d4c75f535912384c7be98f4cfebf8520a06cb53  logs/runs/R1.to_start.out
b34f50f91f1e1be5d208b3dadb95e799033d33e38b19122cb1a6c475dbba4a7a  logs/runs/R2.mission.out
e9a59eded334e9fa814da2d08b08ea6d4d8f0ee45376cd9396a4d4a04609bf1d  logs/runs/R2.to_start.out
19197a065d2eb92411c51615c5db95102d875e686686cf80c1f3f1718860c720  logs/runs/R3.mission.out
61bf0e1cb638b063e1a3b768139aa53d7115bee7e73484a026f66bd9089163fc  logs/runs/R3.to_start.out
c5f4d2aee96293a8bd3deb5f950cdce292fc3b6bf0df600aaf47e6b7fc4ece4b  logs/runs/V2_G1.mission.out
dfe5852e75aa50dbe615609fff2cbf06baaeaff3b8c91891e9e41101b48ca083  logs/runs/V2_G1.to_start.out
14e7b9b39b64d1d234bdd13f34c815482d2977ca577f01e40d0bb4638d2a154c  logs/runs/V2_L1.mission.out
03c45145bbc121c69b47b41033f53bef7a929f59a1cd64ec13081cd08ade5909  logs/runs/V2_L1.to_start.out
73f16aa76d556f0a8dd40e09d9f9efc7f441c973d2b27aae29032e16ee2849c8  logs/strafer_inference.v2.log
cf918ac99a3b84f74267c5601dc6ee3dca91971e3cd14c406154e0a1e85581d9  logs/strafer_inference.v3.log
65dedef44aa27f452c04ea946b594f865053217260754179096df16c6e923347  logs/strafer_inference.v3restore.log
9952c280e1aca28f9f0f54129b95f3b2ee6f393f42d05492c6848752bfae87ef  logs/strafer_navigation.log
d6cb0989553dddec41f64f573999054b7b25cb8552ae443d89bbce33339e5fdc  logs/strafer_sim_perception.log
a612a848cba04630ad9be0a8fd9ee504ffed89cc0f9b55db726624cc9f2ac2ea  logs/strafer_slam.log
a722e42aea87530d3a8810bc099259127ff83f98565e807f7b999c2c184da29e  missions/F1.json
5b14ffc5dcc6db3b9bad748ff1b31fe62e0defaf7481d9bdc61a17d4881415f3  missions/F2.json
a1aa319ad3208b3ea5065e93adc91e11e054a3c2b6ffdce97b93afea293f4283  missions/F3.json
3613e2b65b9b34890e566934efa8755703c73d45e71a2990371abbd24cae7159  missions/G1.json
7c173295f05334192bf942c2808909bd6e90692b011670886de39f6216faeda3  missions/L1.json
5253bc80d7b4d0590e54c304a5ee543604497297a7d4f17c187c3e290fa98849  missions/L2.json
bfb8cfab8b029d13f70a3fade2ba1869a6e542516669a1338a17a0e8833f34c3  missions/R1.json
ebd11d176a57acd6b0ae1fc2025bfbf8f7ec5bd31ae254e3ca2e412c6876aafe  missions/R2.json
4566db70905a0f358fa19b6991295bb26d2c5b060e39c01df3d5e293a379d128  missions/R3.json
6c0e2341807570afc208b77d6eec46b594ae4b9b436c7db5e8c9c73946ee08ab  missions/V2_G1.json
cc90b62096705017ef836b4f5267060a00515056a8cae8ee3672ca0151cc572c  missions/V2_L1.json
69a6807fb828bf88eca15a1af2c5d7acdcffbaab831133baecf1b01277c4c00c  preregistration.md
1f7e4c08d3e22e5f6949a93bf158e2ac55f5d29ef0ac3cb6aa346e0f6d6b7593  ride_along/tf_gate.jsonl
1dcf54927b8525935d350fb8e76538b800c1d0cddd3b717dac1c7290fd9e21ce  ride_along/tf_logger_gate.out
befb15fa58ec7023a18222dbfb1732d41bd28a2280c3614d26ee8f874c5b8b59  scrape/scrape_v2.json
be53935178160203c06879413f82d69d1e8fd43820dc26d780d2100fb2589205  scrape/scrape_v2.txt
899a19fc4f044704949c3812570ae04846b1e2c8d6f58376571516165cc24c1f  scrape/scrape_v3.json
937e27a69f6b792afb343dbce0f81017c69a6b6668a62fce4836d0c666a85a3d  scrape/scrape_v3.txt
38f2bb82eb928576d41cd7f90c9e77d51c28c9e70d5e4387d80d07721402e8ad  stage0/bridge_identity.txt
37a6d740e7ad47172fbfe6b103c2d1d21ebffb8926a5a34a51a5956f36c91ec5  stage0/compose_down.txt
635dd8730e1635994df72b01fd88b63b53c199475d4cf4d40bffe4204f12d082  stage0/cyclone_pin_in_containers.txt
4f08ff0a77ed3cfacdc26a6ece2b53a1c695e4e6cab76fd58108a9ae95b78619  stage0/depth_topic_info.txt
312c94239e2edd8e8a8f6e690212683eca8aaaa99b52475c38c53e6d69dfc4bd  stage0/harness_sha256.txt
9d5412a6b9a5f5ba10daafe0ea8f279eac4e2cd97e9695735ae7b07d18ad6e76  stage0/probe_gate_recheck.json
63526281ffb6120bba86ae92b11012328aece8f367d701f4590422d4cfc3d92e  stage0/probe_gate_recheck.stdout
ab99b92cd62e072e926a37321f6df8e66bbd83fba4e076ad3db12cfab7877f5c  stage0/probe_gate_recheck_costmap.npz
179b39e7784c2b913aeafb93ceac58890cbfdcbda32452eb1c6a22f73369963a  stage0/up_t0.txt
b9225db57e9b10239106087761971719d44d7b39f9a1ad296a34b2f1a07a8680  table.json
bad9d38c80ca13257a48a5260375a02f9b14d86364d40052b0b71c2fa3c0f47a  table.md
e2e9cad2fe91064954c93f10018bf6d7122a5e8c2cb5a64ae0d72cc037280e6b  tools/analyze_tf.py
efec4ece1d68fbd8dff072e36a60b47e2fa13e46eed9184de3cfba2bb46c77a6  tools/build_table.py
cd663222fc4060ce908d1434f49a0bbad05ec6b4585904f3f3d5762e9b650e50  tools/drift_windows.py
2e0ecafe86ac323734682a01b5bb2bd6e025d6ba7afda97fdc7b0a20d4ed1f13  tools/gate_mission.py
89a1296abba90e4c29b0c03a70a8b459aa7e8bd527cea0b0097734e8cc1e3f40  tools/probe_goals.py
ca66c078b8331fa81a7f7194c2d632d056b3c0a3bb6fd8ce83ae68d4682089c5  tools/run_one.sh
59849e7cccefb971431c30ed7c669fe5860dff4aca0a5d9a0ad695b3a17b6a8c  tools/scrape_windows.py
8f47148d0c32260650377ec70a9055fda02a3aa32bda807eb95508fc8804e4e4  tools/tf_logger.py
8cbeaa661eb6e2618543864502a41a5c3c6f5677ee62952b8bd5437477bcabf0  tools/to_start.py
0a38027d705b9511e04c2bdc3daa03948346b1bf00f5c46cf2e8340954341431  tools/topic_rate_sim.py
1500be958fbaea868589cc69960a5f245859b4722de5822952dbdc00643efeae  transits/F1.to_start.json
7cf4168e6ddbdb5887dc8cae93747c1756bfa9184618eeb50c607a86b3c25308  transits/F2.to_start.json
5782642ab3f89be419046c68f5ba27a72dbf79658870a6bf18b0398785170a82  transits/F2.to_start_retry.json
dd0f6337ddc58670591f380ae657bc6c8d748a3379b4f56d1e168e9ea61203ce  transits/F3.to_start.json
f1e96b6734807007e479255dc88335bbafea608f60743f0be42163b690aef5e0  transits/G1.to_start.json
5672df572001975b33d16b66a2f46aedcf445fb7b2a54f7d23fcc228c5a2a83a  transits/L1.to_start.json
c16945ce9e46b217228568967ea4f98f2810fc4c67ed8828fd8726d9af9e539e  transits/L2.to_start.json
12711ea34bed29cee3807dab692ecddce7f6e1fd04e4e85b7cbfceed7b2b6fec  transits/R1.to_start.json
21a4c024a370b747f5e7a0d05f70425a5848d00d95b5702a94321ab6af0ee814  transits/R2.to_start.json
38d953a7d013444f1a3564ed2019e3d285fffc8a087903f9895e769b11c50f9c  transits/R3.to_start.json
7a5d6ba3f9f2c48436ac0be07bb0d1d327dac445bb400f41f1ff1502642409b0  transits/V2_G1.to_start.json
0dc92f1470f29ffafca42c4b6c2af33133218f8b09fc4fe103d8aba0f1ae3da1  transits/V2_L1.to_start.json
```
