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

**By the pre-registered rule, the cap stays "a doubling, not the main term".** The verdict comes from
the second session, twelve reaches, six per arm, all `SUCCEEDED`, none void:
- **Band time.** From the first pose sample within 0.42 m of the goal to the node's result, the median
  is 2.65 s on U against 1.93 s on C. The rule needed U at half of C or less.
- **Stop-short.** 4 of the 6 U reaches went quiet before the first 0.30 m crossing, against 5 of 6 on
  C. The rule needed 2 of 6 or fewer on U.
- **The manipulation held.** Every U reach commanded |vx| + |vy| up to 1.5683 m/s, and every C reach
  stayed at 0.7841 or below (+1e-4). Before the band, U commanded a median of 0.85 m/s and C 0.65 m/s.

**Lifting the cap does not remove the lane's stop-short.** 9 of the 12 reaches went quiet 0.304–0.395 m
from the goal before crossing 0.30 m, uncapped and capped alike (§5). The lane's like-for-like rate
on earlier records is 7 of 10; the referent-offset A/B's was 11 of 12. In the training env the rate is
2.94 % uncapped and 6.40 % capped (pooled). By the rule the cap is not the lane's main term. Six
reaches per arm cannot exclude a contribution of the size seen in training.

**The first session did not run the lane the pre-registration specifies, and its reading is void.**
- At 75 s sim, after its first reach, SLAM's `map→odom` stepped 1.25 m and 88°, and it never returned
  for more than a few seconds (§3). On the sim bridge odom is ground truth, so the step is SLAM error.
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
0.05 m for G1 and 0.10 m for L2; that session's `map→odom` stayed within 0.22 m and 7°):

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
- The rtabmap log carries no warning at either step.

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
  Replayed on the first session, the amended check passes S01 (0.162 m) and S05 (0.100 m) and fails
  S04 (3.361 m) and S10 (1.329 m).

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
- rtabmap logged four rejected loop closures at 02:46:12–02:46:53 UTC and no warning at the steps.

**Where the goals sat** (computed after the run; `session2/frames_posthoc_session2.txt`).
- B04–B12 sat 0.053–0.211 m from the referent A/B's goals.
- B01–B03 sat 0.369–0.573 m off, with leg 1's start-up frame and leg 2's early drift.
- Read on B04–B12 alone, the rule's terms read U band median 4.20 s against C 1.79 s, and stop-shorts
  U 3 of 4 against C 4 of 5. The verdict is the same.
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

## 8. What is not claimed

- **Not the cap's effect in the training env.** That is
  `policy-time-to-goal-baseline-2026-10-11` §9's measurement. This record reads the lane only, with
  six reaches per arm and two goals.
- **Not a rate.** Six reaches per arm read a direction, not a rate. As pre-registered, the stop-short
  clause passes 42 % of the time at a true uncapped rate of 0.46 and 7 % at 0.70.
- **Not why the lane stops short.** By the pre-registered rule the cap is not the main term on this
  lane. The rule's power against a major term is not stated (the pre-registration). The scene-axis
  session (`lane-term-scene-axis-2026-10-14`) was to read the next candidate; it is void.
- **Not the cause of the first session's SLAM failure.**

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `lane-term-speedcap-ab-2026-10-14/record-files/` |
| deposit commit | `76a489bc33ed19e40b825790b6686373cf3b88f0` |

The pre-registration, scorer, cap files and session scripts are in `74127b0`, before the first reach.
The second session's addendum, frame check and manifest are in `b9a6683`, before its boot. The
amendment and stage-0 checks are in `8d7c5fe`, before its first reach. The first relaunch's reference
frames and stage-0 checks are in `e2575cd`, before its next reach. `DEPOSIT.md` lists every file, how
each number above is re-derived, and the redactions.
