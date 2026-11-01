# The node's speed cap on the sim-bridge lane, 2026-10-14

In v3's training env the inference node's speed cap roughly doubles the rate at which the policy goes
quiet short of the goal. That is the pooled read over four launches, computed after that record's
registered rule, which itself read the cap a minor term
([`policy-time-to-goal-baseline-2026-10-11`](../policy-time-to-goal-baseline-2026-10-11/README.md) §9).
On the sim-bridge lane, that stop-short happens on about 7 of 10 band entries. This record asks
whether lifting the cap on the lane itself shortens the terminal band and removes the stop-short. v3
ran the video set's two goals, alternating the deployed cap (**C**, 0.7841 m/s and 2.0933 rad/s) with
training's envelope (**U**, 1.5683 m/s and 4.1866 rad/s), through the full lane. The reads and the
decision rule were committed before the first reach.

**By the pre-registered rule the cap is not the lane's main term. At six reaches per arm the lane
measures no cap effect** (stop-short U 4 of 6 against C 5 of 6: Fisher p 1.0, rate ratio 1.25, 95 %
0.64–2.44; band medians Mann–Whitney p 0.82; computed after the run, §5). The training-env doubling is
that record's figure. The interval includes the training env's rate ratio, 2.17, so six reaches per arm
cannot exclude a doubling. The verdict comes from the second session, twelve reaches, six per arm, all
`SUCCEEDED`, none void:
- **Band time.** From the first pose sample within 0.42 m of the goal to the node's result, the median
  is 2.65 s on U against 1.93 s on C. The rule needed U at half of C or less.
- **Stop-short.** 4 of the 6 U reaches went quiet before the first 0.30 m crossing, against 5 of 6 on
  C. The rule needed 2 of 6 or fewer on U.
- **The manipulation held.** Every U reach commanded |vx| + |vy| up to 1.5683 m/s, and every C reach
  stayed at 0.7841 or below (+1e-4). Before the band, U commanded a median of 0.85 m/s and C 0.65 m/s.

**Lifting the cap does not remove the lane's stop-short.** 9 of the 12 reaches went quiet 0.304–0.395 m
from the goal before crossing 0.30 m, uncapped and capped alike (§5). The lane's like-for-like rate
on earlier records is 7 of 10; the referent-offset A/B's was 11 of 12. In the training env the rate is
2.94 % uncapped and 6.40 % capped (pooled).

> **Correction (2026-10-29).** As the Setup's goals row says, G1 and L2 lie outside the room. Of this
> record's clauses, only the lane clause rests on that: the second session's twelve scored reaches, all
> at G1 and L2, read the cap as not the lane's main term at those two goals and say nothing of goals
> inside the room. The earlier lane rates quoted above are likewise outside-goal reads: 8 of the 10
> band entries behind the 7 of 10, and all 12 reaches behind the 11 of 12. The training-env figures
> cited above (the doubling, 2.94 % and 6.40 %), on which `depth-subgoal-v4-scoping` carries its
> speed-envelope item, involve no lane reach and stand. See
> [`lane-outside-room-goals-2026-10-23`](../lane-outside-room-goals-2026-10-23/README.md).

**Computed after the run: with SLAM's frame held at goal-sent the terminal reads change** (§5).
- **The held read.** The goal is mapped into odom once and the robot read from the odom series.
- **Five reaches never came within 0.30 m.** Of the twelve `SUCCEEDED` reaches, five never did:
  B01 0.337, B02 0.305, B03 0.349, B04 0.457 and B07 0.338 m.
- **The stop-shorts.** The pre-crossing stop-shorts read U 2 of 6 against C 2 of 6.
- **The consequence.** The lane's terminal reads move with the frame at the scale of the effect, so
  this record's headline stop-short figures are live-frame reads, with in-reach goal excursions of
  0.076–0.327 m.
- **The rule's verdict is unchanged under the held frame.** The stop-short clause would pass (U 2 of
  6), but the band clause fails: U 2.70 s against C 4.47 s.

**The first session did not run the lane the pre-registration specifies, and its reading is void.**
- At 75 s sim, after its first reach, SLAM's `map→odom` stepped 1.25 m and 88°, and it never returned
  for more than a few seconds (§3). On the sim bridge odom is ground truth, so the step is SLAM error.
  From 01:04:20 UTC rtabmap logged 68 "wrong loop closure … after graph optimization" warnings. That
  cascade is the mechanism lead for why the frame never returned. The frame was away before it and
  stayed away after it (§3).
- Six of the seven goals submitted after the step went to a different physical point, 0.89–5.23 m
  from where the referent-offset A/B sent them. S05's went 0.050 m off, during the frame's brief
  return (§3).
- By its own rule the session is void: 4 `SUCCEEDED` reaches per arm, fewer than 5. Two reaches were
  refused by the planner dry-check, and two transits pinned against the room's doorway.
- An addendum committed before the second session added a frame check to every reach (§4).
  - Before the second session's third reach, the frame gate read 0.44 m of goal drift and stopped
    the chain, as the addendum provides.
  - After one relaunch, every gate and acceptance read stayed within 0.29 m for the remaining ten.
  - Between those reads, during B04's transit, the ride-along's goal drift reached 0.45 m (§4).

## 1. Setup

| | |
|---|---|
| question and rule | `PREREGISTRATION.md`, committed at `74127b0` (19:34:52 CDT) before the first reach; the second session's addendum at `b9a6683` (21:29:51) before its boot, amended at `8d7c5fe` (21:38:34) before its first reach |
| arms | the node's `vel_cap_linear_m_s` / `vel_cap_angular_rad_s`, 0.7841 / 2.0933 (C, `NAV_*`) and 1.5683 / 4.1866 (U, `MAX_*`), set by an `inference.yaml` mounted read-only over the installed copy (`node/`). The node reads the caps once at start, so each slot recreated the `inference` container with its arm's file and checked the new container's `node up` line for that arm's caps and `policy_loaded=True` |
| images | `strafer-gpu:humble` and `strafer-cpu:humble`, revision label `9f7d20758d5b`, in all five running containers (`stage0/images.txt`, `session2/stage0*/images.txt`) |
| policy | `strafer_depth_subgoal_v3_999.onnx`, sha256 `c866bfd5…`, verified inside the running `inference` container |
| scene | `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, seed 42, headless, through `tools/kit_boot_watchdog.sh`; `publish 30.00 Hz sim`, `frame_skip=3`. One boot for the first session; two for the second (§4). Every boot started on its first attempt |
| SLAM keys | `lanetermA3` (first session), `lanetermA3b` and `lanetermA3c` (second session's two legs), each fresh |
| planner / VLM | Qwen3-4B and Qwen2.5-VL-3B from the sim host's main checkout at `6cd6fd7`, serving `SYSTEM_PROMPT` sha256 `5f8d805d…` (`session1/logs/dgx/planner_served.txt`); one service process each for every session |
| lane | `hybrid_nav2_strafer` + `DEPTH_SUBGOAL`; the subgoal generator as on `main` (`anchoring=mission`, lookahead 1.0 m, the anchored plan's end as the referent); the node's goal bound 60 s and arrival radius 0.30 m; the executor on sim time with a 65 s policy budget |
| harness | the CLI confirmation set's pinned tools (`stage0/harness_sha256.txt`). Per reach, `run_cli_one.sh` runs the transit to the nominal start (0.075, −0.035, 130°), the passive observer, the planner dry-check, then one `make submit-deploy` |
| goals | the video set's G1 (−2.00, 2.25) and L2 (−2.75, −2.25), map frame. Both lie outside the room, past its doorway |
| order | G1 L2 L2 G1 G1 L2 L2 G1 G1 L2 L2 G1, arms U C U C …: three reaches per goal per arm (`manifest.json`, `session2/manifest.json`) |
| sessions | first: 19:35–21:24 CDT; second: 21:38–23:22 CDT, both 2026-10-14 |

**Definitions** (`lane_summary.py`, as the pre-registration states them):
- **Band time**: from the first pose sample within 0.42 m of the goal to the node's result, s sim.
- **Pre-crossing stop-short**: the first quiet stop in the band (the 10th consecutive `/cmd_vel` tick
  with hypot(vx, vy) < 0.05 m/s) strictly before the first pose sample within 0.30 m, or with no
  crossing. Its goal distance is read from the step-held 10 Hz pose.
- **`vx` flip share**: consecutive band ticks of opposite `vx` sign, both above 0.01 m/s in magnitude,
  over all consecutive band ticks.
- **Commanded speed before the band**: mean hypot(vx, vy) from the node's acceptance to band entry.

## 2. The second session: the reaches

| reach | goal | arm | band (s) | quiet stop (m from goal) | pre-crossing stop-short | flip share | cmd speed before band (m/s) | max \|vx\|+\|vy\| |
|---|---|---|---:|---|---|---:|---:|---:|
| B01 | G1 | U | 0.83 | none in band | no | 0.00 | 0.877 | 1.5683 |
| B02 | L2 | C | 8.17 | 0.367 at +0.8 s | yes | 0.20 | 0.594 | 0.7841 |
| B03 | L2 | U | 2.04 | 0.308 at +1.6 s | yes | 0.00 | 0.814 | 1.5683 |
| B04 | G1 | C | 1.26 | none in band | no | 0.00 | 0.652 | 0.7841 |
| B05 | G1 | U | 5.39 | 0.345 at +1.1 s | yes | 0.20 | 0.813 | 1.5683 |
| B06 | L2 | C | 5.54 | 0.370 at +1.2 s | yes | 0.09 | 0.617 | 0.7841 |
| B07 | L2 | U | 2.29 | none in band | no | 0.00 | 0.884 | 1.5683 |
| B08 | G1 | C | 1.79 | 0.313 at +1.2 s | yes | 0.00 | 0.661 | 0.7841 |
| B09 | G1 | U | 3.01 | 0.331 at +1.7 s | yes | 0.00 | 0.900 | 1.5683 |
| B10 | L2 | C | 2.07 | 0.368 at +1.1 s | yes | 0.00 | 0.651 | 0.7841 |
| B11 | L2 | U | 6.92 | 0.395 at +0.4 s | yes | 0.16 | 0.717 | 1.5683 |
| B12 | G1 | C | 1.54 | 0.304 at +1.5 s | yes | 0.00 | 0.685 | 0.7841 |

- **Every reach is valid.** Each ended on the node's own `SUCCEEDED`, the CLI reported `succeeded`, and
  every join reads `node_result_executor_agrees`. Every transit reached the start on its first
  attempt, every dry-check matched, no reach fell back to Nav2, and the largest SLAM step inside any
  band was 0.207 m (B10).
- **Per goal.** L2: band median U 2.29 s, C 5.54 s; stop-shorts U 2 of 3, C 3 of 3. G1: U 3.01 s,
  C 1.54 s; U 2 of 3, C 2 of 3.
- **The terminal reads of `reach_terminal.py`** (beside the rule): every reach entered 0.42 m once, so
  its last-inward band time equals the first-entry one (medians U 2.65 s, C 1.93 s). Its flip-share
  median is 0.030 on U and 0.000 on C.

## 3. The first session, as run

**The frame.** The session's ride-along samples `map→odom` at 10 Hz.
- It started at (−0.025, −0.091, +0.5°).
- At 75.1 s sim (00:41:14 UTC) it stepped 1.25 m and 88°.
- Steps of 0.81, 0.48, 0.67 and 1.01 m followed, the last at 481.6 s.
- After 75 s it was back within 0.3 m and 5° of its start only between 218.6 and 230.1 s, which covers
  S05's acceptance.

**Where the goals went** (computed after the run; `session1/frames_posthoc_session1.txt`). Each
submitted reach's map-frame goal, mapped to odom through `map→odom` at acceptance, against where the
referent-offset A/B sent the same goal (the means of its six acceptances per goal, which stayed within
0.05 m for G1 and 0.10 m for L2). That session's ride-along stayed within 0.219 m and 6.5° of its
first sample, and its twelve observer records' `map_odom` within 0.287 m and 4.93° of their first
sample (`posthoc_frame_excursion.txt`, computed after the run):

| reach | goal | `map→odom` at acceptance | from the referent A/B's goal |
|---|---|---|---:|
| S01 | G1 | (−0.005, −0.143, +3.0°) | 0.252 m |
| S03 | L2 | (+0.239, +0.846, +88.8°) | 5.229 m |
| S04 | G1 | (+0.218, +0.860, +90.3°) | 3.368 m |
| S05 | G1 | (+0.030, −0.011, +0.4°) | 0.050 m |
| S06 | L2 | (+0.532, +0.392, +30.4°) | 2.255 m |
| S07 | L2 | (+0.359, +0.504, +30.2°) | 2.049 m |
| S08 | G1 | (+0.391, +0.491, +29.9°) | 0.894 m |
| S10 | L2 | (+0.418, +0.109, +18.1°) | 1.501 m |

**The pins.** The nominal start is a map-frame point too, and the moved frame put it against the room's
doorway.
- S11's transit pinned at map (−0.549, −0.070) and ran out its 150 s sim budget on both attempts.
- The chain was stopped, and the video set's `nudge.py` moved the robot 0.30 m south, off the
  doorway's geometry, in 3.6 s sim (`session1/logs/nx/S11_nudge.out`).
- S12's transit then pinned at (−0.360, −0.273). It was stopped, and the session ended.
- At S12's pinned pose the Nav2 costmap's nearest lethal cell was 0.50 m away
  (`session1/logs/nx/S12_costmap_window.txt`), while the robot, commanded at 0.3 m/s, did not move.

**The refusals.** Two reaches were not submitted, because the planner dry-check refused their compiled
plans (§6). The re-runs the pre-registration provides were never reached.

**The reading, for the record.** By the rule the session is void: U 4 and C 4 `SUCCEEDED`. Descriptively,
the band medians are U 4.45 s and C 5.97 s, the pre-crossing stop-shorts U 2 of 4 and C 3 of 4, and the
caps bound as designed (U 1.568 on every reach, C 0.7841).

**What the first session does not show.** Its SLAM failure has no identified cause.
- By the concurrent Jetson session's own account (a message to this session), its test containers
  ran on the default ROS domain, against the lane's 42, and none ran between 00:25 and 00:42 UTC.
  So none was running at 00:41:14, when the 88° step came. Its deposit
  (`chassis-driver-port-fence-2026-10-14` in the evidence repository) stamps its runs
  00:44:03–00:48:51 UTC, ending 12 s before the 0.81 m step.
- **The SLAM log** (`session1/logs/nx/strafer_slam.log`):
  - No warning falls at the 88° step or the 0.81 m step.
  - From 01:04:20 UTC (line 640) to 01:29:41 UTC (line 1190) rtabmap logged 68 warnings that a
    "wrong loop closure has been detected after graph optimization". Each one rejected that
    iteration's added loop closures, and each came with a companion "Loop closure … rejected!" line
    (lines 641–1191).
  - Separately, 18 loop closures were rejected for too few inliers (lines 410–1147). That makes 154
    lines naming a rejection.
  - This wrong-loop-closure cascade is the mechanism lead for the frame never returning. It began 23
    minutes after the 88° step, and it is not shown to have caused that step. It covers 01:04–01:29
    UTC, after the frame's one brief return (00:59:09–01:00:41). The frame was away before the
    cascade, and it stayed away after it to the ride-along's end at 02:24:42, with no further warning.
  - The counts and line numbers are in `posthoc_round_reads.txt` (computed after the run).

## 4. The second session's frame checks

**The addendum** (`session2/PREREGISTRATION_SESSION2.md`). The question, arms, goals, order, reads and
rule are unchanged; the labels are `B01` … `B12`. Two checks are added:
- **Before each reach, a frame gate.** Under the latest `map→odom`, each goal must map to within 0.30 m
  (the arrival radius) of its position under the session's reference frame.
- **After each reach, a frame void.** The same test on `map→odom` at acceptance; a failure voids the
  reach.
- **On a frame stop**, the bridge and the stack relaunch, with the robot back at its spawn and a fresh
  SLAM key. At most two relaunches are allowed.

**The amendment before the first reach.** As committed, the reference was the referent A/B's goal
positions, not the session's.
- The fresh frame settled at (+0.121, −0.083, +2.64°). Under it L2 sat 0.410 m from the referent
  position: 0.31 m from this frame's own rotation and translation, and 0.11 m from the referent
  session's.
- That gate would have refused every reach on a frame that had not moved.
- `8d7c5fe` changed the reference to the session's own settled frame (`session2/frame_ref*.json`).
  Replayed on the first session's eight submitted reaches, the amended check passes 2 (S01 0.162 m,
  S05 0.100 m) and fails 6 (S03 5.076, S04 3.361, S06 2.080, S07 1.874, S08 0.840, S10 1.329 m).
  The addendum listed S01, S04, S05 and S10. The other four were computed after the run
  (`posthoc_round_reads.txt`).

**What the frame did.**

| leg | reaches | reference `map→odom` | drift read |
|---|---|---|---|
| 1 (key `lanetermA3b`) | B01, B02 | (+0.121, −0.083, +2.64°) | 0.232 and 0.286 m at acceptance; the gate before B03 read 0.441 m and stopped the chain |
| 2 (key `lanetermA3c`) | B03–B12 | (+0.056, −0.012, +1.98°) | at most 0.285 m at any gate; 0.027–0.270 m at acceptance. Between reads the ride-along's goal drift exceeded 0.30 m on 525 of its 10 Hz samples, peaking at 0.446 m (G1, 138.1 s sim, during B04's transit) |

Leg 1's drift came as a slope and three steps.
- It began at 66.2 s sim (02:39:30 UTC), as the robot first moved on B01's transit.
- The steps were 0.15 m (77.3 s), 0.12 m (80.2 s), and 0.29 m and −4.3° (137.3 s, 02:48:27 UTC,
  during B02's reach).
- It reached 0.52 m and 5° by 148 s.
- rtabmap logged four rejected loop closures at 02:46:12–02:46:53 UTC
  (`session2/logs/nx_leg1/strafer_slam.log`, lines 345–360), none at the steps, and no wrong-loop-closure
  warning.

Leg 2's rtabmap log has 89 rejected loop closures (`session2/logs/nx_leg2/strafer_slam.log`, lines
247–1503) and no wrong-loop-closure warning. The scene-axis boot's has 31
(`lane-term-scene-axis-2026-10-14`).

**Where the goals sat** (computed after the run; `session2/frames_posthoc_session2.txt`).
- B04–B12 sat 0.053–0.211 m from the referent A/B's goals.
- B01–B03 sat 0.369–0.573 m off, with leg 1's start-up frame and leg 2's early drift.
- Read on B04–B12 alone, the subset is void-sized: four U reaches, below the series minimum of five.
  As a descriptive read, the band median is U 4.20 s against C 1.79 s and the stop-shorts U 3 of 4
  against C 4 of 5.
- One goal moved more than 0.30 m during its reach: B04, by 0.327 m. The addendum reads the frame at
  acceptance only, so B04 stands.

## 5. Beside the rule

- **Where the stop-shorts stop.** Uncapped 0.308–0.395 m from the goal, capped 0.304–0.370 m. The rig's
  earlier pre-crossing stop-shorts went quiet 0.295–0.411 m from their held subgoal
  (`policy-time-to-goal-baseline-2026-10-11` §9). That subgoal sat 0.012–0.065 m from the goal
  ([`subgoal-referent-offset-ab-2026-10-11`](../subgoal-referent-offset-ab-2026-10-11/README.md)).
- **The cap does change the approach.** U commanded a median 0.85 m/s before the band against C's
  0.65 m/s, and U's band was not shorter.
- **The two goals differ.** On L2 the capped band was longer (median 5.54 s against U's 2.29 s); on G1
  the uncapped one was (3.01 s against 1.54 s). With three reaches per cell, these per-goal medians
  are descriptive.
- **The verdict's statistics** (computed after the run; `posthoc_round_reads.txt`):
  - Fisher's exact test, two-sided, on the stop-short counts.
  - The rate ratio is capped over uncapped, with a Katz log interval.
  - The Mann–Whitney test is exact and two-sided, on the twelve band times.

**Computed after the run: the frame held at goal-sent** (`posthoc_frame_held.py` →
`posthoc_frame_held.txt`). The node scores each reach on the live map frame, so a `map→odom` change
during the reach moves the goal under the robot. This read holds the frame instead:
- the map-frame goal is mapped into odom once, through the `map→odom` sample nearest goal-sent;
- the robot is read from the record's odom series (on the bridge, ground truth);
- `lane_summary.py`'s band, crossing and stop-short definitions are then repeated, over goal-sent to
  the result.

| reach | arm | live-frame distance at the result (m) | held-frame distance at the result (m) | held-frame band time (s) | held-frame crossing | held-frame pre-crossing stop-short | in-reach goal excursion (m) |
|---|---|---:|---:|---:|---|---|---:|
| B01 | U | 0.300 | 0.337 | 0.93 | no | no | 0.216 |
| B02 | C | 0.298 | 0.305 | 8.38 | no | yes | 0.076 |
| B03 | U | 0.300 | 0.349 | 1.83 | no | yes | 0.116 |
| B04 | C | 0.300 | 0.457 | never entered 0.42 m | no | no | 0.327 |
| B05 | U | 0.299 | 0.222 | 6.25 | yes | no | 0.162 |
| B06 | C | 0.300 | 0.173 | 6.37 | yes | no | 0.164 |
| B07 | U | 0.275 | 0.338 | 2.29 | no | no | 0.173 |
| B08 | C | 0.300 | 0.213 | 2.00 | yes | no | 0.162 |
| B09 | U | 0.300 | 0.241 | 3.11 | yes | no | 0.166 |
| B10 | C | 0.278 | 0.286 | 2.58 | yes | yes | 0.245 |
| B11 | U | 0.300 | 0.265 | 7.12 | yes | yes | 0.135 |
| B12 | C | 0.300 | 0.261 | 1.54 | yes | no | 0.118 |

- **The crossings.** Five of the twelve never came within 0.30 m under the held frame, and each
  one's closest approach was its distance at the result.
- **The stop-shorts.** They read U 2 of 6 (B03, B11) against C 2 of 6 (B02, B10), against the live
  frame's 4 of 6 and 5 of 6.
- **The rule under the held frame.** The band medians are U 2.70 s against C 4.47 s, with B04, which
  never entered, ranked last. The stop-short clause would pass, but the band clause fails, so the
  verdict is the same.
- **The consequence.** The lane's terminal reads move with the frame at the scale of the effect, so
  the headline stop-short figures above are live-frame reads, with in-reach goal excursions of
  0.076–0.327 m.

## 6. The planner refusals

Both of the first session's refusals are the shipped planner mis-signing the lateral component of a
two-axis translate. The compile-first dry-check caught each, and nothing was submitted.

| reach | command | compiled | transit end |
|---|---|---|---|
| S02 (L2) | `move 0.038 meters backward and 3.588 meters left` | dx −0.038, dy **−3.588** | (0.044, 0.002, 128.26°) |
| S09 (G1) | `move 3.077 meters forward and 0.035 meters left` | dx 3.077, dy **−0.035** | (0.046, −0.048, 131.03°) |

- **S02** is the backward-left template that
  [`planner-translate-sign-2026-10-11`](../planner-translate-sign-2026-10-11/README.md) records as
  mis-signed on the shipped tree. The pre-registration named the hazard: L2's dx crosses zero inside
  the transit's tolerance.
- **S09** is a forward-left command with a 0.035 m lateral component. That record's sweep compiled the
  forward-left anchors exactly, with lateral components of 0.021 m and 0.191 m and a forward component
  of 0.032 m, and did not measure 0.035 m. This is one observation outside its sweep, from one
  deterministic answer.
- No command in the second session was refused.

## 7. Deviations and defects

- **The second session and its amendment** (§4) were not in the first pre-registration. They were
  committed before the reaches they govern. The first session's reading is reported, not dropped.
- **Operator moves.** The first session's chain was stopped twice by its process group, and one
  `nudge.py` move freed the robot between reaches (§3). The second session needed none.
- **The chain's wait was shorter than two pinned transits.** `n2_session.sh wait` gave up after 1800 s
  wall, while two pinned transit attempts take about 2300 s. The chain was stopped by hand before it
  mattered. The second session's chain waits 3600 s.
- **`stop-local` did not stop Kit.** The watchdog runs Isaac Lab's launcher in a process group of its
  own, so signalling the recorded group left Kit running. The first session's bridge was stopped
  through the launcher's group by number. The second session's script signals the watchdog's child
  groups (`stop-bridge`).
- **The second session's chain logs "chain done" after a frame STOP.** The `exit` inside the
  manifest's pipeline leaves only its subshell. It did stop: no reach ran after the STOP line.
- **Every reach recreated the `inference` container.** The node reads its caps once at start, so each
  slot recreated the container with its arm's file. The referent-offset A/B switched its arm with
  `ros2 param set`, without a recreate. As the deposits show, the recreate is not a candidate cause of
  the frame failures:
  - it happened before every reach in all four of the evening's boots, including the one whose frame
    held;
  - no step coincides with one (computed after the run, `posthoc_round_reads.txt`). Of the 51
    `map→odom` steps of 0.1 m or more between consecutive ride-along samples, none came within 42 s
    after a recreate's start; the nearest came 18.0 s before one, and the median distance to the
    nearest recreate start is 81 s;
  - rtabmap consumes nothing the `inference` container publishes. rtabmap's inputs are the synced
    colour and depth relays, the scan, the bridge's odom and the IMU; the container publishes
    `/cmd_vel`, the active goal and the subgoal.

  One path the deposits do not test: both containers run on the robot host, so they share its
  compute.

## 8. What is not claimed

- **Not the cap's effect in the training env.** That is
  `policy-time-to-goal-baseline-2026-10-11` §9's measurement. This record reads the lane only, with
  six reaches per arm and two goals.
- **Not a rate.** Six reaches per arm read a direction, not a rate. As pre-registered, the stop-short
  clause passes 42 % of the time at a true uncapped rate of 0.46 and 7 % at 0.70.
- **Not why the lane stops short.** By the pre-registered rule the cap is not the main term on this
  lane. The rule's power against a major term is not stated (the pre-registration). The scene-axis
  session (`lane-term-scene-axis-2026-10-14`) was to read the next candidate; it is void.
- **Not the cause of the first session's SLAM failure.** The wrong-loop-closure cascade (§3) is a
  lead, not a shown cause.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `lane-term-speedcap-ab-2026-10-14/record-files/` |
| deposit commit | `38e3bf9b62a91c9154ec7189efcb62538f1ec3e4` |

The pre-registration, scorer, cap files and session scripts are in `74127b0`, before the first reach.
The second session's addendum, frame check and manifest are in `b9a6683`, before its boot. The
amendment and stage-0 checks are in `8d7c5fe`, before its first reach. The first relaunch's reference
frames and stage-0 checks are in `e2575cd`, before its next reach. The runs are in `76a489b`. The
post-hoc reads were added at `3f1debf` (frame held, frame excursion) and `38e3bf9` (the round reads, and
the held-frame band times). `DEPOSIT.md` lists every file, how each number above is re-derived, and the
redactions.
