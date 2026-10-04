# CLI-submitted missions on the v3 policy with the executor in the loop, 2026-10-03

Four of five pre-registered missions were submitted through the autonomy CLI with the
executor in the loop. Each ended on the policy node's own `SUCCEEDED`, and the executor
reported each as `succeeded`. No mission ended in `navigation_timeout`, and the node
reported no cancel. The policy-backend navigate budget of commit `9b80e99` (the node's
`mission_timeout_s` 60 s + 5 s = 65 s sim) therefore **passes** the rule written down
before the first mission. The fifth mission, R1, was not submitted: the planner compiled
its command, "… 1.372 meters right", as a move 1.372 m to the left, and the dry-check
that precedes each submission stopped it.

**None of the four reaches would have tripped the old budget either.** They took
8.5–12.5 s sim, against the 10.2–14.0 s the Nav2-sized budget would have given them.
So the set shows the CLI path working end to end under the new budget, with the node
deciding every step. It contains no mission the old budget would have cancelled. The
case the change addresses is the v3 gate's: three of its seven reaches outlasted that
budget ([`executor-policy-nav-budget`](../../tasks/completed/executor-policy-nav-budget.md)).

## Setup

| | |
|---|---|
| change under test | `9b80e99` on `task/executor-policy-nav-budget`. A goal that `strafer_direct` or `hybrid_nav2_strafer` executes gets `POLICY_MISSION_TIMEOUT_S` + `policy_budget_margin_s` = 60 + 5 = **65 s** on the executor's node clock, capped by `STRAFER_NAVIGATION_TIMEOUT_S` (unset, so 90 s). Nav2 keeps `2·d / 0.7841 + 5 s` and its stall watchdog |
| images | `strafer-cpu:humble` and `strafer-gpu:humble`, both built clean at `9b80e99` (revision label `9b80e9976961` read from all five running containers) |
| scene | `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, `Environment seed : 42`, one bridge launch for the set, no other Kit actor on the sim host |
| Kit log | `kit_20261003_215506.log` |
| cadence contract | `publish 30.00 Hz sim`, `frame_skip=3 (derived, derived 3)`, bridge tick 120 Hz, script defaults |
| SLAM key | `enrich_cliconfirm1` (fresh for this bridge launch) |
| policy | `strafer_depth_subgoal_v3_999.onnx`, sha256 `c866bfd54ec1a8352159e33d7875d41e3f07a442ff8301ba3700867932e2eb91`, sidecar `d22e3504…33ab5`, both verified inside the running `inference` container |
| lane | `hybrid_nav2_strafer` + `DEPTH_SUBGOAL`, `anchoring=mission`, `depth_tick_semantics: timer_reuse`, node `mission_timeout_s` 60 s (node clock), tolerance 0.30 m (`GOAL_ARRIVAL_RADIUS_M`) |
| executor | `strafer-executor` in the `autonomy` service. `use_sim_time` true on both its nodes; backend `hybrid_nav2_strafer`; its budget constants read back in the container: `POLICY_MISSION_TIMEOUT_S` 60.0, margin 5.0, policy budget 65.0 s, Nav2 budget for 3.1 m 12.907 s |
| planner / VLM | Qwen3-4B and Qwen2.5-VL-3B on the sim host, reached over the direct cable; `/health` from the `autonomy` container returned `model_loaded` true for both in 5–6 ms before the first mission |
| transport | direct cable; the Cyclone interface pin selected it in all five containers; two RELIABLE depth subscribers (`strafer_inference`, `timestamp_fixer`) |
| start | nominal (0.075, −0.035), heading 130°, reached before every mission by the v3 gate's scripted `cmd_vel` transit, never by Nav2 or the policy; enforced 0.15 m / 3.0° and the 1.20 m floor |
| plannability | all four distinct goals plannable from the nominal start on the scored map (`ComputePathToPose` re-check before the first mission; it sends no goal, and the planner's echo of its paths reached the idle subgoal generator before the first transit) |
| RTF | 0.1333 over the scored session (294.9 s sim in 2212.9 s wall) |
| window | first acceptance 03:01:29 UTC, last result 03:38:22 UTC (2026-10-04; 22:01–22:38 CDT on 2026-10-03) |

## Pre-registration

`preregistration.md` was committed to the evidence deposit at 02:59:45 UTC, with the
observer harness pinned by digest beside it (`tools_sha256.txt`, 34 files). The first
transit started at 02:59:58 and the first command was submitted at 03:01:23. Its sha256
is `60dcd060a62f099963cfee7999c4db7fb1416d6cc07445e7aa398b5ff935c806`. It was not
edited afterwards.

It fixes:
- the question;
- the lane and setup;
- the five goals and their order;
- the mission form and its dry-check;
- the scored outcome;
- the decision rule and how it is read;
- the narrow unscored causes;
- the descriptive columns;
- the distinctness checks.

The decision rule, verbatim:

> The budget fix **PASSES** if no mission ends in `navigation_timeout` on a goal the node
> then reaches (an executor-attributable cancel = **FAIL** of the fix, whatever else
> happens). A node-level miss is recorded and not counted against the fix; the policy is
> the gate's policy and the gate already characterised it. Expect at least three
> node-level reaches of the five; fewer is a note for the record, not a verdict.

How it is read was also fixed beforehand:
- **Executor-attributable cancel:** the CLI's `navigation_timeout`, or the node reporting
  `CANCELED` / `CANCELING` for the executor's goal. Nothing else in the setup sends or
  cancels policy goals.
- **Exception:** a cancel by the executor's `/clock` stall detector is unscored, and is
  neither a PASS nor a FAIL. It applies only if the observer measured a wall gap of
  13.5 s or more with no sim progress inside the mission window, and the cancel came
  before 64.5 s sim after acceptance. Every other executor cancel is a FAIL.
- **PASS:** every mission that reached the policy's action server ended on the node's own
  result (`SUCCEEDED` or `ABORTED`), and the executor's result agreed with it (`succeeded`
  with `SUCCEEDED`; a failed step carrying `navigation_failed` with `ABORTED`).
- **INCONCLUSIVE:** fewer than three of the five missions reaching the policy's action
  server.

## Mission form

The CLI takes natural language only. The executor's runbook form for a navigate-to-pose
step is a translate command ("move forward 1 meter"), which the planner compiles to one
`translate {dx_m, dy_m}` step in the robot's frame. Per mission:

1. At the start pose the observer read `map→base_link` and formed
   `move {dx} meters forward and {|dy|} meters left|right` to three decimals, with
   (dx, dy) the goal's offset in the robot frame there.
2. The same command was sent once to the planner's `/plan` with the executor's request
   shape. The planner decodes greedily. The command was submitted only if the plan was
   one `translate` step matching (dx, dy) to 0.001 m.
3. `make submit-deploy CMD="<command>"` at the repo root.
4. The executor read its own `map→base_link`, turned in place first when the bearing was
   0.3 rad or more off the heading (its documented rotate-then-translate rule), and sent
   one `NavigateToPose` to the policy's action server.
5. A passive observer took acceptance, the dispatched goal and the result from the policy
   action's status topic and the node's `active_goal` echo. It subscribes only.

## Verdict

| reading | missions |
|---|---|
| executor-attributable cancel | **none**. The node's statuses for each executor goal were `EXECUTING` then `SUCCEEDED` only; the CLI's `error_code` was empty every time |
| reached the policy's action server | **4 of 5** (G1, L1, L2, FX); R1 not run (planner transcription) |
| node result, executor agreeing | 4 of 4 `SUCCEEDED` with `final_state succeeded` |
| node-level reaches against the ≥ 3 expectation | **4**, met |
| clock-stall exception | did not apply: the largest in-window wall gap without sim progress was 0.16–0.20 s, against 13.5 s |
| **set** | **PASS** |

## The five missions

Bearing is relative to the heading, + left. The command was formed from the measured
start. The policy's start is the pose at goal acceptance, after any pre-rotation.

| mission | goal | start off (pos / yaw) | command | dry-check | pre-rotation (expected / turned) | policy start bearing | dispatched-goal error | node result | reach, s sim | executor | old budget, s | new budget, s |
|---|---|---|---|---|---|---:|---:|---|---:|---|---:|---:|
| G1 | (−2.00, 2.25) | 0.021 m / +1.7° | `move 3.107 meters forward and 0.021 meters left` | pass | none (0.4°) | +0.4° | 0.4 mm | **SUCCEEDED** | **8.833** | succeeded | 12.925 | 65 |
| R1 | (−0.05, 2.15) | 0.033 m / +1.1° | `move 1.712 meters forward and 1.372 meters right` | **fail**: plan `dy_m` +1.372 | (−38.7°) | — | — | not run | — | — | — | — |
| L1 | (−2.00, 0.00) | 0.035 m / +0.3° | `move 1.354 meters forward and 1.529 meters left` | pass | +48.5° / +43.6° | +4.5° | 0.2 mm | **SUCCEEDED** | **8.475** | succeeded | 10.209 | 65 |
| L2 | (−2.75, −2.25) | 0.057 m / −1.3° | `move 0.032 meters forward and 3.533 meters left` | pass | +89.5° / +83.6° | +6.2° | 1.1 mm | **SUCCEEDED** | **12.525** | succeeded | 14.012 | 65 |
| FX | (−2.00, 2.25) | 0.006 m / −1.3° | `move 3.074 meters forward and 0.191 meters left` | pass | none (3.6°) | +3.6° | 0.5 mm | **SUCCEEDED** | **10.825** | succeeded | 12.856 | 65 |

- **Reach** is the node's result minus its acceptance, on the sim clock.
- **Old budget** is what the executor before `9b80e99` would have passed:
  `min(90, max(5, 2·d / 0.7841 + 5))` with `d = hypot(dx_m, dy_m)` of the translate step.
  Every reach is under it, by 1.5–4.1 s.
- **Dispatched goal.** The column is the node's `active_goal` echo against the map goal.
  The executor's `Sending hybrid goal` lines name the same four goals to within 1.4 mm.
- **Pre-rotation.** The executor turned in place at 0.5 rad/s for 1.76 s sim (L1) and
  3.37 s sim (L2). It stopped 5.6–5.7° short in odom, at the edge of its 0.1 rad
  tolerance. The policy therefore started L1 and L2 facing the goal to within 4.5° and
  6.2°. The v3 gate started them at +50.5° and +88.8°.
- **The executor's step.** Planning took 4.0–4.5 s wall in every mission. The step then
  ran 8.83 / 10.24 / 15.90 / 10.83 s sim: pre-rotation plus the node's reach, from
  the first rotation twist or acceptance to the result. That is 69.5–117.0 s wall, inside
  the CLI's own status-poll brackets.

### R1: the planner dropped the sign

- **What the planner returned.** For `move 1.712 meters forward and 1.372 meters right`,
  Qwen3-4B returned one `translate` step with `dx_m = 1.712` and `dy_m = +1.372`. That is
  1.372 m to the robot's left: the mirror image of the goal about the robot's heading.
- **Disposition.** The dry-check failed on the sign. Nothing was submitted, and the
  mission was not rephrased.
- **The other commands.** The four two-axis commands to the left compiled with the right
  sign and magnitudes to 0.001 m.
- **The prompt.** Its only right-hand translate example is single-axis ("strafe right 2
  meters" → `[0.0, -2.0]`).
- **Follow-up.** Filed as
  [`planner-translate-two-axis-sign`](../../tasks/active/reliability/planner-translate-two-axis-sign.md).
  The executor composes and dispatches a translate faithfully, so a submitted R1 would have
  driven the robot toward (−2.12, 0.35), not (−0.05, 2.15).

## How the reaches were reached (descriptive)

| mission | final (map) | net advance | v_par while moving | terminal dither: last inward 0.42 m → result, s sim | band held, m | mean cmd, m/s | `vx` sign flips | dwell read |
|---|---:|---:|---:|---:|---|---:|---:|---|
| G1 | 0.2996 | +2.807 | +0.374 | 3.46 | 0.300–0.415 | 0.049 | 9.7 % | holds (≤ 0.299 m, ≤ 0.015 m/s) |
| L1 | 0.2994 | +1.743 | +0.265 | 5.37 | 0.299–0.413 | 0.144 | 23.1 % | holds (≤ 0.298 m, ≤ 0.011 m/s) |
| L2 | 0.3004 | +3.233 | +0.331 | 6.50 | 0.300–0.411 | 0.112 | 24.2 % | **does not hold** (≤ 0.306 m, ≤ 0.044 m/s) |
| FX | 0.3001 | +2.780 | +0.309 | 4.84 | 0.300–0.414 | 0.083 | 25.7 % | holds (≤ 0.299 m, ≤ 0.025 m/s) |

- **Set reads.** Median while-moving `v_par` +0.320 m/s, median net advance +2.794 m;
  cross-track developed and was consumed in 4 of 4.
- **G1, L1 and L2 were faster than on the gate.** Their terminal approach took 3.5, 5.4
  and 6.5 s sim. On the v3 gate it took 13.9, 7.0 and 7.8 s, and the reaches 19.0, 11.1 and
  15.4 s sim. The cause is not attributed: L1 and L2 also started facing the goal here.
  FX's 4.8 s falls inside the gate's fixed-goal repeats (1.3–5.1 s).
- **L2 and FX ended on the line.** The observer's own TF reads at the result were 0.3004
  and 0.3001 m, while the node's own lookup returned `SUCCEEDED`. The cross-check agrees
  within its 0.05 m.
- **The dwell read.** It uses the samples after the result, when the node has stopped
  commanding. It cannot say whether the policy would have held. On the v3 gate all seven
  dwell reads held; L2's does not here.

**Final distance in stable frames.** Odom is ground truth on the sim bridge, so every
`map→odom` change is SLAM error. The final odom pose is mapped through two frames:

| mission | map frame at the result | run's own start frame | session-median frame |
|---|---:|---:|---:|
| G1 | 0.300 | 0.276 | 0.253 |
| L1 | 0.299 | **0.375** | **0.353** |
| L2 | 0.300 | **0.353** | **0.370** |
| FX | 0.300 | **0.520** | **0.403** |

- **Which reaches hold.** G1 ends inside 0.30 m in both frames. L1, L2 and FX end
  outside in both.
- **What moved the map frame.** `map→odom` moved 0.08, 0.26 and 0.43 m inside the L1, L2
  and FX windows. FX's window holds a 0.261 m / −4.9° correction 3.5 s sim after
  acceptance.
- **The session-median frame is skewed.** `map→odom` held one value for 134.5 s sim, 43 %
  of the ride-along, while FX's first transit sat in one place. The third frame the gate
  used (the median without a displaced interval) does not apply: no step of 0.9 m or more
  occurred.
- **Scoring.** The scored reach is the map-frame one, as the pre-registration defines it.

## The stack's own contract and SLAM health

From the complete `strafer_inference` log, differenced over each `[acceptance, result]`
window:
- cadence p50 30.00 Hz sim in every window, p05 ≥ 28.67;
- 1216 of 1216 inferences on a fresh frame;
- `depth_age` max 0.033 s sim;
- 0 deadline misses;
- no `gate`, `obs_none`, `action_shape`, `bad_encoding` or `bad_shape` skips;
- mid-mission watchdog skips 0–2 per window;
- 0 `Costmap is older` and 0 in-window `plan is stale` lines.

Duplicate content was 0.000 for G1 and 0.435–0.450 for the other three, the same split the
v3 gate showed. There were no collision admissions. Each container started once, with
no FATAL and no restart.

| | `map→odom` corrections in window | largest | ≥ 0.30 m | SLAM-health flag |
|---|---:|---|---:|---|
| G1 | 5 | 0.030 m | 0 | set: `map→odom` at acceptance 0.40 m from the session median |
| L1 | 10 | 0.080 m | 0 | set: 0.50 m from the median |
| L2 | 11 | 0.111 m | 0 | set: a 0.320 m / +4.1° step during its transit, and 0.33 m from the median |
| FX | 13 | 0.261 m | 0 | not set (0.08 m from the median) |

- **What the flag means.** It is descriptive under the pre-registration, not an unscored
  cause.
- **The median skew drives it.** The displacement test reads against the skewed session
  median, so it flags G1, L1 and L2, but not FX, whose in-window correction was the
  largest.
- **Steps of 0.30 m or more.** The session's only one fell in L2's transit, before its
  goal.

## Transits

- **G1, R1, L1, L2.** Each transit took one attempt: 7.9, 46.0, 1.7 and 13.0 s sim. L1's
  start was already inside the target.
- **FX's first attempt.** It ran out its 150 s sim budget ending 0.41 m / −109° off, with
  29 replans. It then held within 0.05 m of one spot for its last 126.6 s sim.
- **FX's retry.** The one allowed retry reached the start in 13.0 s sim, and FX started
  0.006 m / −1.3° off nominal.
- **Why the session ran long.** The two FX attempts account for 1227 s of the session's
  2213 s wall.

## Distinct runs

- **Windows.** All five missions ran on one bridge launch. The four scored windows do not
  overlap (gaps of 21–169 s sim).
- **Goal ids.** Each submitted mission produced exactly one new goal id on the node, the
  four ids are distinct, and R1 produced none.
- **G1 and FX.** They share coordinates, but their pose series differ point-wise by up to
  0.29 m in the map frame and 0.45 m in odom. Their starts were 0.03 m and 3.1° apart in
  the map frame, and 0.45 m apart in ground truth.

## Deviations

- **The harness's own reading line.** `run_cli_one.sh` checks a record's freshness by
  `observer.started_wall`, a field `cli_mission.py` does not write. So every `<label>.done`
  reads `verdict=STALE_RECORD reading=NA`, and the wrapper skipped its join.
  - After each mission, the deposited join (`cli_mission.py --prereg-read`) was run by hand
    on the record and the CLI output.
  - Freshness was checked from the record's own wall times against the submission's.
  - The joins re-run byte-identically from the working root. The deposit's copies embed
    absolute input paths.
  - Nothing that measures was changed, and the pinned tools are as deposited.
- **Analysis helpers written after the set.** `analysis/tools/session_summary.py`,
  `distinct.py` and `decision.py` were written after the last mission. They are not in the
  pre-registration's tool list or its pinned digests. They apply the pre-registered
  definitions mechanically. The per-mission readings also come from the pinned join, and
  `verify/` recomputes the headline figures independently from the raw evidence.
- **Images at the branch head, not a merge.** The change is unmerged, so the images were
  built from a clean checkout of `9b80e99`.
- **Service URLs.** The executor reached the planner and VLM over the direct cable,
  through a session-local compose file applied after the host's own override. The host's
  override names the LAN address.
- **Stage-0 policy-load lines.** The stage-0 script's grep for the inference node's
  `Loaded policy` lines looked in the wrong directory. Those four lines were appended from
  the container log after the fact; the in-container `sha256sum` output is the script's
  own.

## What this record does not claim

- **That the old budget would have cancelled any of these missions.** It would not have;
  see the lead. The old budget's effect is the v3 gate's reach times set against it.
- **Anything about the policy** beyond the four runs: no re-score of the gate, no
  attribution of the shorter terminal approaches here.
- **That a two-axis "right" always mis-signs.** One observation, with the planner's prompt
  at `ed0d5af`.
- **Real-sensor behaviour.** Every depth frame here is bridge depth.
- **The Nav2 backend's budget.** It is unchanged, and pinned by the unit tests in
  `9b80e99`.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `goal-a-cli-confirmation-2026-10-03/record-files/` |
| deposit commit | `7dd63db32a7531cbde9d378dddb2851c0721abcb` |
| earlier commits | `90179bc77757dbbedf412aaab1878104d25c8d20` deposited `preregistration.md`, `tools/` and `tools_sha256.txt` before the first mission; `281d8239c9d1e6e77d966e3bfd3f189d61a3e674` the other files and `DEPOSIT.md`; `5c3e03a2` and `7dd63db3` corrected `DEPOSIT.md` (list formatting; the two `COMMANDS.md` steps that name paths outside the deposit) |
| pre-registration commit | `90179bc77757dbbedf412aaab1878104d25c8d20` |

The deposit holds everything this record names:
- the observer's records, the transit records and the ride-along;
- per mission, the command, the planner dry-check, the CLI's full output and the joined
  reading;
- the five container logs and the sim host's bridge, planner and VLM logs;
- the stack checks before the first mission;
- the test lanes and the image build at `9b80e99`;
- the analysis with its exact commands, and an independent recomputation;
- the harness.

Thirteen text files had rig network addresses replaced with placeholders before deposit.
`DEPOSIT.md` lists them. Every analysis output re-derives byte-for-byte from the deposited
copies with the commands in `analysis/COMMANDS.md`, except the absolute input paths that a
few of them embed (`DEPOSIT.md` names them).

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/goal-a-cli-confirmation-2026-10-03/record-files/. \
      docs/measurements/goal-a-cli-confirmation-2026-10-03/
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/goal-a-cli-confirmation-2026-10-03/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
62d6ec0d285aa62618a996b11e71492a372f25969158776b6c3da96d0c30e943  analysis/COMMANDS.md
bf82c92d776027736ab5b39cfe66389f90790c1a4bd5536b3f97231f8b3a3e1c  analysis/SUMMARY.md
8e256b7dfacfde0c57d43f4b3fa8c362cece8c00842fec612663ff77f327789d  analysis/decision.json
cb9a68abef6be6c7de219294de1163fba4c395a33963fc24ac4c4820ee465be0  analysis/distinct.json
8f77e428358a43b87339dd8060ae5c37cbb91b6647c9d6da8ab86552141a9a97  analysis/drift_summary.json
3ea4ddcfc5cbed7aa3581a299a6098e7815deb814a00733f67c9342bd8b8a5d9  analysis/drift_summary.stdout
6abf128f87a7f2b288484a186fcb22d9e8ff2a571b551118cb319a184d3cb163  analysis/prereg_rerun/FX.prereg.json
01c1a18c7f7f828bdcfba55406b528e6c3aca38625ed2c7594755f6f2a5cab0c  analysis/prereg_rerun/FX.stdout
430c55a8c13917203392ddf84abb64b762728641b7a02ef1a23feea10a17639c  analysis/prereg_rerun/G1.prereg.json
881943cdfe6586de7af95aa7f9be49620eb53586a33b0af5ac1276ce85539e60  analysis/prereg_rerun/G1.stdout
2d52accc8425043ee493814e620a24ea656d54ab32a106ab24c6c400b9b9918d  analysis/prereg_rerun/L1.prereg.json
b5db59187ab4348260d95df5b7868b481de2a59716eb2a0272594cb83358a421  analysis/prereg_rerun/L1.stdout
f7387ec09ffd223d98a8c7bd5b5e200a3145d71fd2aa1651efc3c3bbf51010fe  analysis/prereg_rerun/L2.prereg.json
3dfdedcad38a740526f16d306d2f8da85a972274511d9b38a9f4ecd0adbd5256  analysis/prereg_rerun/L2.stdout
a171aaeb9ff84911dc1c59424bf013d6f230518efe5eba6ba8b4c9a0eb8c325e  analysis/reach_terminal.json
b780ee63f7d7184c88ec8470b832dc8a4b1a550abdda7324ea73e324a34fdefc  analysis/reach_terminal.stdout
ac3b868253eac6a0a11b9ef087eab2bffb74077d16d8eba45b36701cb30e009e  analysis/scrape.json
8492297776f8b5887edf5585a601a28c90246b1c6d1d2b512f43713c709f5533  analysis/scrape.txt
e083d2bb1aac0a7163e29aad3bdc9c70f69754d04ec71345d12ea293dfffdd67  analysis/session.json
dbe333cf835217522b9ff7ed3831c4a6d680d90028da99db74db087a1ca44c9c  analysis/slam_health.json
4bd44925cb4512613c70444e51fe0c682e95d5dbd4ae9f2a2026d035709d48b7  analysis/slam_health.stdout
8fba79e8aecb291b4800c4ad7695b70055fa6b6dc2bf4fc9998d0c7cee7a8616  analysis/table.json
93b04b8b2819159f4db8c878c16a7d7b2b35f52771a5bfe947e0dd7a0ea51587  analysis/table.md
9faef24a8dd557b35c313ceec4cd648f8b0a2a8aac903b9bb7232b43ca207314  analysis/tools/decision.py
00b2ba3fe23c204d79bec7c914ad1ddf98d9dd8a2bde6e128d85092a8c6691ab  analysis/tools/distinct.py
7436778cee6388d367eb1910dfd96cc8587e16f160d049b1037f28636ac90cfa  analysis/tools/session_summary.py
861dac4385a936fc67ccb17e599478b81b76f3b35db37c27ce633c8f29ba9a78  logs/build/make_images_branchhead.log
628084cf4c50b7ef8d992e410b6c68c07bf035d5c1281b3dcd18570b430a8a3a  logs/build/test_autonomy_requires_ros_wt-main.log
628084cf4c50b7ef8d992e410b6c68c07bf035d5c1281b3dcd18570b430a8a3a  logs/build/test_autonomy_requires_ros_wt-pr.log
1b9b8b39a756ce0c83481983d357eb5a371795beceb7eb1b32e67f4b32e57868  logs/build/test_autonomy_ros_files_wt-main.log
5d591dcc89169605c0672592cb93ca2524a2227facf97871a2ef9e94a5f19571  logs/build/test_autonomy_ros_files_wt-pr.log
6899814aa183eff21285c71d5c1e7fa73c91e667962a16c9ede8a85df25f058a  logs/build/test_autonomy_wt-main.log
5f87f46fd79dcc31c966594682678330c29f053bf59fabe38767b471d802efef  logs/build/test_autonomy_wt-pr.log
19e7c567e91ed22695617f0da0ac30a74e84ef6284d8bd5d64bb8eefe01e6821  logs/build/test_jetson_branchhead.log
e1cbfbb25690a29098ff99aca0be5aa2a8292210da358356c446d0c2ec0990d7  logs/dgx/cli_confirm_bridge.log
120590a338d1076e6321523c3e4998292d59038c0b99169af1169d7010f42250  logs/dgx/cli_confirm_envsetup.log
93fb92a76a034a10dadd98b83d64bb4eaffa649cf571be0744e74c697b46ca70  logs/dgx/cli_confirm_serve_planner.log
3c09557b8a07106e974070b87da2252d3332ecb8b89a8c46ded4245078934e95  logs/dgx/cli_confirm_serve_vlm.log
3f753638d7f27559a0f351bf02500ac071447a39a16e7ae93853d2e99efa1e43  logs/fix_round/adv_failed_set.final.log
03019782c411522e2279cdc7c47df9d3b1cb263e1bfbe50c863024e896958942  logs/fix_round/adv_failed_set.log
93ff7811a209e2a8479230bbb9b6bc19f7f311d3af383ec350c1db2a7e7d5494  logs/fix_round/adv_failed_set.rc
04834ed0ea8cb6a03f997d40e55e7c8d9f07537c5f52164852d5a0da558c8a51  logs/fix_round/adv_regression_set.final.log
fb657aae3908e90cdd8910660f56db28cb26ab01bad5a2e232a9894b83f7a9ef  logs/fix_round/adv_regression_set.log
93ff7811a209e2a8479230bbb9b6bc19f7f311d3af383ec350c1db2a7e7d5494  logs/fix_round/adv_regression_set.rc
28c716e04248a341a9a82bd434daeda87a88895d498bf5c908825711574dc3e1  logs/fix_round/byte_identity.final.log
93ff7811a209e2a8479230bbb9b6bc19f7f311d3af383ec350c1db2a7e7d5494  logs/fix_round/final_adv_failed.rc
93ff7811a209e2a8479230bbb9b6bc19f7f311d3af383ec350c1db2a7e7d5494  logs/fix_round/final_adv_regression.rc
93ff7811a209e2a8479230bbb9b6bc19f7f311d3af383ec350c1db2a7e7d5494  logs/fix_round/final_gate.rc
93ff7811a209e2a8479230bbb9b6bc19f7f311d3af383ec350c1db2a7e7d5494  logs/fix_round/final_mock_1.rc
4f82a2f127886ad936dc6f4291ed4a72e15433eb1c5e9c525a2bb95764805bad  logs/fix_round/final_mock_1.start
93ff7811a209e2a8479230bbb9b6bc19f7f311d3af383ec350c1db2a7e7d5494  logs/fix_round/final_mock_2.rc
e8cd5d7d0021b311a789192c3be908b23cc5afc47a01c918b94a404300799743  logs/fix_round/final_mock_2.start
d117fa006ba9208500b2930ce69cbde436c647afa917cb7396a9bc9111a46dd2  logs/fix_round/final_runs.done
5cca0dd17bcb6800dda1e257e7f5c8ab9b777a1098f96e0606fe7b301004aae5  logs/fix_round/run2.rc
d359f19dd162c413adb09f7f12afacf7402e9b76fbc74cf396e61936ac7a1f3a  logs/fix_round/run2.start
894eeb0440e78018ea892cfe8352ea70008729a0818f8fedb7cf88bf564edb15  logs/fix_round/selftest.final.log
94e404f0bec96f1a10128466f51af2019d234f507298a4e3ef67ba8460a63cbb  logs/fix_round/selftest.log
78c7833cbee897ebafadb6d7f7343eca08059a5f496ff62110f037b0a871b364  logs/fix_round/test_cli_mission_mock.final1.log
6c1bd4c9427bc402492d70743280580c3bd8fe4783adc6c293c8c2cd34e41ccf  logs/fix_round/test_cli_mission_mock.final2.log
79d8ff4f730e51afc7ebfd247e01f4b83cca3aa3f3adf891050a9b0da5df436c  logs/fix_round/test_cli_mission_mock.run1.log
2f564e1ae4df6bf7dcdf1f36fc07c5e3c075270a5e68caa5a2fe00eb9c40535d  logs/fix_round/test_cli_mission_mock.run2.log
87f5664482978e197b9080a664b5437558a35e1c46bba0f942c25929b71ddf45  logs/fix_round/test_gate_mission_mock.final.log
9ed1c995ea127195a1b7c5bf05b2c82b7e1213b80cc94b58ce3810da4063187e  logs/fix_round/test_gate_mission_mock.regression.log
9f9302d66512c4d41bff5b4d93980bf06a7cb989f05fc70a9f61b3e44fbfd242  logs/fix_round/test_run_cli_one.final.log
d778e5eb3aa133b6a2e79d04007f3781bb7cbc53c015215f29de86e8a7cc9384  logs/runs/FX.cli.meta
d2141e00df5db03d80f070d12f76331247bca88409e9f3b8c6c1eb80656b4402  logs/runs/FX.cli.out
d09644eec8b809e2a0e28a0bbe6c65c51b18fdc2bc57fa7a9539efae30f7a157  logs/runs/FX.cmd
e46db1287e13bdb04ba861d4f6e96b1474af9040e40fab51171d99b40c412185  logs/runs/FX.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/runs/FX.done
d9b195ebb226c477d8298674e9faddd0fb72bc6e4286345f234c18381d0a377f  logs/runs/FX.plancheck.out
6abf128f87a7f2b288484a186fcb22d9e8ff2a571b551118cb319a184d3cb163  logs/runs/FX.prereg.json
9ad0e5027d2a90f925b0b0193f70e28499752d48f306fc981d9dd2a3c4ad1b8d  logs/runs/FX.run.log
56939763ea3c5e588732c0602642db3df5a18fd87061c7bdb7836a7cc44bbc35  logs/runs/FX.to_start.out
6e197a8098354eb982eb6922a685af7dc0205b7bf2c71fac7eebb33fda47a294  logs/runs/G1.cli.meta
8f4a1c76bed098cb88df696baed6af8b2bd695518abad7471aa8fc48a1643442  logs/runs/G1.cli.out
6f1ec1d853559fc02969a73ab6753b50fbf05dfc06e5c6ba321d7735fb885ae1  logs/runs/G1.cmd
6444065dcb417a90c4eb41c9317c42a3a99a894dc6d315d7566311a7e1e20dd8  logs/runs/G1.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/runs/G1.done
92e8dc0320131d4e0545c5d6f457159d8991e49f62d39a8a1215d82d223e6436  logs/runs/G1.plancheck.out
430c55a8c13917203392ddf84abb64b762728641b7a02ef1a23feea10a17639c  logs/runs/G1.prereg.json
0277ac118134eb1b266093762954e63935285529102f35adddceb9040d3c9bf8  logs/runs/G1.run.log
046388bb79970a3e31cffe9a5b54d6cf0435f49106d0881e2294ef14fb84844d  logs/runs/G1.to_start.out
7deb8149a447c4b2d982272a0c1b0ff937a5e07c937c441cee7de7907d16d2f3  logs/runs/L1.cli.meta
3aabe7cd7877766e81729a68a691b8edf313e7e4d89b80b92ac2157a315d0b56  logs/runs/L1.cli.out
17efd60e8244f85a81244aca631acaf941d6fcbc66da68ccbdc4d17f1ccd29f4  logs/runs/L1.cmd
96d108efed38645e89469fb0805dd2bfe39312864386422c875f6fe13060baec  logs/runs/L1.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/runs/L1.done
19ecf43c941d1155b48486970367759fb24f49c793df0b5cefcfafd551498cd0  logs/runs/L1.plancheck.out
2d52accc8425043ee493814e620a24ea656d54ab32a106ab24c6c400b9b9918d  logs/runs/L1.prereg.json
9610c99b97ea838205948426b24a422817b809046ce1f6d3536ee73d73eb278c  logs/runs/L1.run.log
07bdf334a33db661095c23ebc04c29b55fa890ef5a99819c999c53ecb6598396  logs/runs/L1.to_start.out
a4c45cce0d27d0b71083990a419ed041946c82c87e9672dcf94f485bc71eccde  logs/runs/L2.cli.meta
01eac07071723107f0fa9ad994ed26ddec57ec84658dac8157fbe470718b6338  logs/runs/L2.cli.out
ae4e2c3678f3958cdc2f0b41ef16d6d8c50fa4378d210a6889d603a7ea19500f  logs/runs/L2.cmd
0d22399c4fd7bdb15fab890657b9d76443f8ea4a7c09654efbf4d6a1780905eb  logs/runs/L2.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/runs/L2.done
d8601f20d5583c260db2e07aa7755e486860dfcc90252cac5389fcea142e5bda  logs/runs/L2.plancheck.out
f7387ec09ffd223d98a8c7bd5b5e200a3145d71fd2aa1651efc3c3bbf51010fe  logs/runs/L2.prereg.json
57ef8a7995c2c7afe918b2c3e18b2b5df0610d5b09f4d14bb1fac7ed7b9a27f6  logs/runs/L2.run.log
808926fc2ea3d22dbaa13f67d0f0c62aacff7e08889ed26907eb83273b9e8a9c  logs/runs/L2.to_start.out
c026ff65f9b703ddd3041a96eec29e0cd7cd9db4e3a80dcdb91d545ef3ed80a4  logs/runs/R1.cmd
c3d14ddb13e657adb4d9b001198765c2466d47f39103dd3d636ce69e1382d4f1  logs/runs/R1.cmd.json
af2977789ae1d2bc2775831aab90be05ea44c0b5fab910b4835f0026c5bc6d30  logs/runs/R1.done
17f2d5bea3fbfc81b2c6a58592010c6c5f38faa5dee2efc246fbef9f7c6ebc8e  logs/runs/R1.plancheck.out
1298b51c2bbd052b2b66455ec7c458ac5f08496eebc58eb9be4e8478a1f49318  logs/runs/R1.plancheck_failed
a0b6bd2a1c0adc990569c87e87fb6d772ef0fef7a585aa468cfd8a43ba924991  logs/runs/R1.run.log
5c5244fb709e26fdaa111a1f20bec267281dcb2f575fe226d356b7a3b5ec26a5  logs/runs/R1.to_start.out
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate/FX.cli_exited
d09644eec8b809e2a0e28a0bbe6c65c51b18fdc2bc57fa7a9539efae30f7a157  logs/runs/container_gate/FX.cmd
e46db1287e13bdb04ba861d4f6e96b1474af9040e40fab51171d99b40c412185  logs/runs/container_gate/FX.cmd.json
893d59af442304a4daadf4b9f50488ba5f8213c7d33148728c96a3cb5b4f4afb  logs/runs/container_gate/FX.json
1b70462d4a24f5fe70d3dd4f7fedc2706ca871335c58be9c64a17ccca8c29146  logs/runs/container_gate/FX.mission.out
8bfe4fbc9526c12b1e8590107fd00b0eb7fc0cbef68e4f358aaba5cdec85fd10  logs/runs/container_gate/FX.progress.jsonl
3d3c424e17324f87eb38658c130c37e5fa6f8476430982189b016429c6af263a  logs/runs/container_gate/FX.to_start.json
33e806576a9c6516023694d00182e77b34d1d8d58e11f6279da5ec0e0ba60fd7  logs/runs/container_gate/FX.to_start_retry.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate/G1.cli_exited
6f1ec1d853559fc02969a73ab6753b50fbf05dfc06e5c6ba321d7735fb885ae1  logs/runs/container_gate/G1.cmd
6444065dcb417a90c4eb41c9317c42a3a99a894dc6d315d7566311a7e1e20dd8  logs/runs/container_gate/G1.cmd.json
033d25646dae0bbbdc6762fce1fd2bf1b32fb64821fe30ddad4ab81f06f5167b  logs/runs/container_gate/G1.json
1d9dc5714d7ae9cc1e1726efd8c62472407b94a7d21f06700561a5ea464449a1  logs/runs/container_gate/G1.mission.out
8ac304d059cd7b9eb1038625545599846e038cce1e25f63c7fef367aa561a682  logs/runs/container_gate/G1.progress.jsonl
ebbdd67c75d2803389fb249408c79e512242404917a338a961cf1a7eda14f014  logs/runs/container_gate/G1.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate/L1.cli_exited
17efd60e8244f85a81244aca631acaf941d6fcbc66da68ccbdc4d17f1ccd29f4  logs/runs/container_gate/L1.cmd
96d108efed38645e89469fb0805dd2bfe39312864386422c875f6fe13060baec  logs/runs/container_gate/L1.cmd.json
fbe45d23f9a3daa8571e0016615b2297fbbd46f41aefa7fa95a490bd557ec368  logs/runs/container_gate/L1.json
9f34b5cb68b574fa6b10ba35c4f022d1dda8505c9c1e4e5ad3469658cb4f01ed  logs/runs/container_gate/L1.mission.out
0b77a89beff1c9c440b762b7b5533baa19b9660a0810dd25dd9556f26ad0bb89  logs/runs/container_gate/L1.progress.jsonl
65f0871d00d3b0d7d54d7361d6748b0e44514da9bdbdf4566025f55baf311ce4  logs/runs/container_gate/L1.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate/L2.cli_exited
ae4e2c3678f3958cdc2f0b41ef16d6d8c50fa4378d210a6889d603a7ea19500f  logs/runs/container_gate/L2.cmd
0d22399c4fd7bdb15fab890657b9d76443f8ea4a7c09654efbf4d6a1780905eb  logs/runs/container_gate/L2.cmd.json
3165e0404f1225f14a857e8a81a474096af7643f025d0ee505035314bb863125  logs/runs/container_gate/L2.json
36ebc9c192cd0cddf38cb1c9f48a8228b3717b356df05695848e6df430ba2102  logs/runs/container_gate/L2.mission.out
962e8fe60f65bbe702bd7238b1d2ca36ab35ef3c0ab6d9b94f2df15dfef002d9  logs/runs/container_gate/L2.progress.jsonl
fb5f26917999c50678c552c7971eb5f6f4badb37119f606d863ba9df53f7b70a  logs/runs/container_gate/L2.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate/R1.cli_exited
c026ff65f9b703ddd3041a96eec29e0cd7cd9db4e3a80dcdb91d545ef3ed80a4  logs/runs/container_gate/R1.cmd
c3d14ddb13e657adb4d9b001198765c2466d47f39103dd3d636ce69e1382d4f1  logs/runs/container_gate/R1.cmd.json
415d6361afeec4a95916d107ea62551ee1a400bffece327afd6a42572b9671df  logs/runs/container_gate/R1.json
97273462cb05cfbc987e370b9650b39e51e9573e822921e95f590421515669ef  logs/runs/container_gate/R1.mission.out
842f727873617b992a38ffa7072150497dab91ab2785d12e15fc9ac596322585  logs/runs/container_gate/R1.progress.jsonl
1e4af6b92b52288b128460a882729daef55ae9bfc2d7eba7fdd729c711e95baa  logs/runs/container_gate/R1.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate_all/FX.cli_exited
d09644eec8b809e2a0e28a0bbe6c65c51b18fdc2bc57fa7a9539efae30f7a157  logs/runs/container_gate_all/FX.cmd
e46db1287e13bdb04ba861d4f6e96b1474af9040e40fab51171d99b40c412185  logs/runs/container_gate_all/FX.cmd.json
893d59af442304a4daadf4b9f50488ba5f8213c7d33148728c96a3cb5b4f4afb  logs/runs/container_gate_all/FX.json
1b70462d4a24f5fe70d3dd4f7fedc2706ca871335c58be9c64a17ccca8c29146  logs/runs/container_gate_all/FX.mission.out
8bfe4fbc9526c12b1e8590107fd00b0eb7fc0cbef68e4f358aaba5cdec85fd10  logs/runs/container_gate_all/FX.progress.jsonl
3d3c424e17324f87eb38658c130c37e5fa6f8476430982189b016429c6af263a  logs/runs/container_gate_all/FX.to_start.json
33e806576a9c6516023694d00182e77b34d1d8d58e11f6279da5ec0e0ba60fd7  logs/runs/container_gate_all/FX.to_start_retry.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate_all/G1.cli_exited
6f1ec1d853559fc02969a73ab6753b50fbf05dfc06e5c6ba321d7735fb885ae1  logs/runs/container_gate_all/G1.cmd
6444065dcb417a90c4eb41c9317c42a3a99a894dc6d315d7566311a7e1e20dd8  logs/runs/container_gate_all/G1.cmd.json
033d25646dae0bbbdc6762fce1fd2bf1b32fb64821fe30ddad4ab81f06f5167b  logs/runs/container_gate_all/G1.json
1d9dc5714d7ae9cc1e1726efd8c62472407b94a7d21f06700561a5ea464449a1  logs/runs/container_gate_all/G1.mission.out
8ac304d059cd7b9eb1038625545599846e038cce1e25f63c7fef367aa561a682  logs/runs/container_gate_all/G1.progress.jsonl
ebbdd67c75d2803389fb249408c79e512242404917a338a961cf1a7eda14f014  logs/runs/container_gate_all/G1.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate_all/L1.cli_exited
17efd60e8244f85a81244aca631acaf941d6fcbc66da68ccbdc4d17f1ccd29f4  logs/runs/container_gate_all/L1.cmd
96d108efed38645e89469fb0805dd2bfe39312864386422c875f6fe13060baec  logs/runs/container_gate_all/L1.cmd.json
fbe45d23f9a3daa8571e0016615b2297fbbd46f41aefa7fa95a490bd557ec368  logs/runs/container_gate_all/L1.json
9f34b5cb68b574fa6b10ba35c4f022d1dda8505c9c1e4e5ad3469658cb4f01ed  logs/runs/container_gate_all/L1.mission.out
0b77a89beff1c9c440b762b7b5533baa19b9660a0810dd25dd9556f26ad0bb89  logs/runs/container_gate_all/L1.progress.jsonl
65f0871d00d3b0d7d54d7361d6748b0e44514da9bdbdf4566025f55baf311ce4  logs/runs/container_gate_all/L1.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate_all/L2.cli_exited
ae4e2c3678f3958cdc2f0b41ef16d6d8c50fa4378d210a6889d603a7ea19500f  logs/runs/container_gate_all/L2.cmd
0d22399c4fd7bdb15fab890657b9d76443f8ea4a7c09654efbf4d6a1780905eb  logs/runs/container_gate_all/L2.cmd.json
3165e0404f1225f14a857e8a81a474096af7643f025d0ee505035314bb863125  logs/runs/container_gate_all/L2.json
36ebc9c192cd0cddf38cb1c9f48a8228b3717b356df05695848e6df430ba2102  logs/runs/container_gate_all/L2.mission.out
962e8fe60f65bbe702bd7238b1d2ca36ab35ef3c0ab6d9b94f2df15dfef002d9  logs/runs/container_gate_all/L2.progress.jsonl
fb5f26917999c50678c552c7971eb5f6f4badb37119f606d863ba9df53f7b70a  logs/runs/container_gate_all/L2.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/runs/container_gate_all/R1.cli_exited
c026ff65f9b703ddd3041a96eec29e0cd7cd9db4e3a80dcdb91d545ef3ed80a4  logs/runs/container_gate_all/R1.cmd
c3d14ddb13e657adb4d9b001198765c2466d47f39103dd3d636ce69e1382d4f1  logs/runs/container_gate_all/R1.cmd.json
415d6361afeec4a95916d107ea62551ee1a400bffece327afd6a42572b9671df  logs/runs/container_gate_all/R1.json
97273462cb05cfbc987e370b9650b39e51e9573e822921e95f590421515669ef  logs/runs/container_gate_all/R1.mission.out
842f727873617b992a38ffa7072150497dab91ab2785d12e15fc9ac596322585  logs/runs/container_gate_all/R1.progress.jsonl
1e4af6b92b52288b128460a882729daef55ae9bfc2d7eba7fdd729c711e95baa  logs/runs/container_gate_all/R1.to_start.json
3e8b252f2c61467b71355b2209e7515e46aa93422fde708c9eafaa4a7f55b86a  logs/runs/container_gate_all/tf_cli.jsonl
711d0978621abcc7c25745445a66764b7e48cce961d9b847b137f667c1d59a30  logs/runs/container_gate_all/tf_logger_cli.out
afeb9a0b5741bf4d7797eed9d6b95387cc93780515ae1fefa4a0c3ebe76f0773  logs/stage0/bridge_identity.txt
417bcc69cead7640131a17e18eb4fc32acf2c4a366c12df9afd8b0a6a8da546f  logs/stage0/cyclone_pin_in_containers.txt
2253534a9fe56ac1678e9ac89923e88ab5b4873685ef01b27b11ed2655cbd0aa  logs/stage0/depth_topic_info.txt
c38424f0046e3921fcb1bad0bf6296d4369682e77ecb273e4a45b34fc5482cd8  logs/stage0/executor_checks.txt
c61cce874211758d0f7dbadd6fb919d315cae29fb57d41e5cac42cd666db7544  logs/stage0/harness_sha256.txt
276f72bd1b7966757f57263b3b1316952d8bbed3716337d6844f97c2c9841ab6  logs/stage0/images.txt
9585187cabbb30b54e7fd180f12caf1fd3ecf9a7e15fb17a6e85fcc88f25b605  logs/stage0/in_container_artifact_checks.txt
bfd4ecbb9077ad699ae76ed8fe7d77996dd7139de6850151940eaf2d0912fdd4  logs/stage0/probe_cli_recheck.json
5c283afc50d924885a5638c1bf218a91a28c93df88c6720d87351a90f3aaa2ed  logs/stage0/probe_cli_recheck.stdout
77049be5e189672025ab6c0804674065d72dd8eb164670d2b38c7f2f5beaa804  logs/stage0/probe_cli_recheck_costmap.npz
bfd4ecbb9077ad699ae76ed8fe7d77996dd7139de6850151940eaf2d0912fdd4  logs/stage0/probe_container/probe_cli_recheck.json
77049be5e189672025ab6c0804674065d72dd8eb164670d2b38c7f2f5beaa804  logs/stage0/probe_container/probe_cli_recheck_costmap.npz
dca3a06400986da7b965bbdc1688fe5ddbbf809c9331a61671dd8c502bedec24  logs/stage0/service_health.txt
71a2ba136e0130894cbed1a2e6136094e569f96ee18dced0c2ea8d9fc0afc962  logs/strafer_autonomy.log
3ecd9476fda74c52a4030b6c92ec1727d8137ea892d7efb2eb08a22ce84ebc9a  logs/strafer_inference.log
2e7081c2cb33f97c49729389a1d6b2737366ed06b70234c7c7bd55d01594069a  logs/strafer_navigation.log
077ceab71587c11c0062b7bdd7ca5d4923ca0d30471020d7347d53c23dd19da9  logs/strafer_sim_perception.log
e52da80041f33e4a81960ee157c91d432928dd57af583b601fab76d5afb53f03  logs/strafer_slam.log
b249474e6952d4fb87ba2bb893c8d266843be142c715633395ebdd16d72a6382  logs/test_cli_mission_mock.log
6de7af822fac9fa705a1ffda72caada275b8c51b935624b934b254b0739f8519  logs/test_cli_mission_mock.mine.log
cb69466a399bfc87ddc607b8247af5b264213dd27051749acee1e8b532d06b4c  logs/test_gate_mission_mock.regression.log
ff7536bf891f9d3bf3d1ae8f2214b8b0d1f70d9528dc528e3e874cf9f511d56b  logs/up_t0.txt
60dcd060a62f099963cfee7999c4db7fb1416d6cc07445e7aa398b5ff935c806  preregistration.md
99d93c8ddc8ca43c51447392cb984a4115498048a6546eab494e25a8b5d17158  session/autonomy-urls-direct.yml
f26df9cde00844103c74802c318da102f646d5199218c573e44a180f10510999  session/dgx_up.sh
8b0a0283b3a53f47c6bcd71da91ac539b64d88c881687b937e5d09a1f1d51f39  session/nx_up.sh
be6e8898bb6e6514c36b4adddf2054753b28bbe4a4c2d5539bdec20666396839  session/stage0.sh
8dd83146796438b995ecfa88a77af638dccaf5803eb7bea6f7fb1178bd8d9e7f  tools/action_present.py
e2e9cad2fe91064954c93f10018bf6d7122a5e8c2cb5a64ae0d72cc037280e6b  tools/analyze_tf.py
e2e9cad2fe91064954c93f10018bf6d7122a5e8c2cb5a64ae0d72cc037280e6b  tools/analyze_tf_0817.py
074b003aa44d6fed9182f95a65d50c261a4f8cda8656f767ef9a302a3f7f004a  tools/build_table.py
4dd6e18489e885fb795b3b27c728b14eb03c7ff77549e1948c29a70a86287aa5  tools/cli_mission.py
1e1c467f0e84dddf5887c92543d803487457fcce10e785ed1aed83657a624a79  tools/drift_windows.py
85c9cb6a6a288398ce0e0acd690a58c5d2c16c49f964f07ccb94179694667036  tools/fake_executor.py
2e0ecafe86ac323734682a01b5bb2bd6e025d6ba7afda97fdc7b0a20d4ed1f13  tools/gate_mission.py
2cab0b4528489f614b66d759bccab853979439411838646c1f3781b7495f33e7  tools/mock_policy_stack.py
4e669d84171a821ccb2cdbe29c447770c4e95c79244ca51e5e08c394938ddce1  tools/plan_check.py
89a1296abba90e4c29b0c03a70a8b459aa7e8bd527cea0b0097734e8cc1e3f40  tools/probe_goals.py
eb4c4f3eedcf8d96288b4f4e15ac86595286e62caf606d44744a8c740c74cfdd  tools/reach_terminal.py
571b66b447ed8154411dfb2216ee373cadae5e765b6e0ffc980f235f1cf65699  tools/run_cli_one.sh
ca66c078b8331fa81a7f7194c2d632d056b3c0a3bb6fd8ce83ae68d4682089c5  tools/run_one.sh
59849e7cccefb971431c30ed7c669fe5860dff4aca0a5d9a0ad695b3a17b6a8c  tools/scrape_windows.py
2ad148f8cf5eb28b1ba25c0bea16852e6d6103934f372354d1b8eaf41482882c  tools/slam_health.py
8e1378bc6b8196a436a7f5779f1e27297d282c372b01b7c209b08a5cb2416e24  tools/test/adv/adv_run.sh
bf5f9d5f145e9d7a4d0be1a1577dcb99958e7d8fc91ad6c86874e80a0dc0030f  tools/test/adv/adv_run.sh.orig_reviewer_b
a2b777a146060beb372f5ebba4b956e1eeab981648b6e4d43090b75a92900ae9  tools/test/adv/adv_sender.py
0be2ce1a7b86aaeb59d491c1a96579f5910b8626fab0a958b4f1d04476a5603e  tools/test/adv/check_adv.py
559e5b4bb33760287961601208e98323fb7fc7c71d937e975cb76fecf19a8498  tools/test/adv/status_script.py
50ae567177cafd81250f44f404339e0eab731843fce9210380a5441edfc62aab  tools/test/adv/summarize.py
df602d9767d8dc3992783ca84c17708b16e3d1a40d1bec15d68fb871160a1b97  tools/test/check_cli_records.py
aaee673ed9db4898f2caccd43315be8239ce8aea54570c6894d7cc5c70788b90  tools/test/fake_planner.py
8bd4e798d2cf1931d4e3c0164f4da1d8d9c24eaef955726be2f9b148c28cda27  tools/test/fake_sim_to_start.py
9870c957bca5de72b60393209078ec4d2099be2d7fa9ab7bdbe6470da9c4767b  tools/test/wrapper/bin/docker
ee144636607a63781920a2f2e3c1075b5c6dfdc6d4e5bd44cc1380a8a47aeb9b  tools/test/wrapper/bin/make
b97be9809801e98c0b1badcaecc8a14d2c8caf922493ae6db261bd1a8f3ba562  tools/test/wrapper/fake_observer.py
3d1c5129d6bd7526ab087ecdf34e03470d4cba81e1a912ad3af96006942ab19e  tools/test/wrapper/test_run_cli_one.sh
6c7881d10e4517f135908e2a0cac46d7711ea1bf0e92b01d7951dff2344fae7e  tools/test_cli_mission_mock.sh
55dd62dcb75f935c9422c927cec1e885c65c95ea5189321fcb6213f1f8e785fc  tools/test_gate_mission_mock.sh
8f47148d0c32260650377ec70a9055fda02a3aa32bda807eb95508fc8804e4e4  tools/tf_logger.py
8cbeaa661eb6e2618543864502a41a5c3c6f5677ee62952b8bd5437477bcabf0  tools/to_start.py
0a38027d705b9511e04c2bdc3daa03948346b1bf00f5c46cf2e8340954341431  tools/topic_rate_sim.py
73179029e156bb428fe16596d7be709b309ec01c2ffda825fdbdacb7675c0cd9  tools_sha256.txt
4d2625ed9d99f1a2804cb1c9d8dfcb29724c76cb10a35fd11d7f1a7e2cca2538  verify/commands_digests.txt
f3f87eb5aad007be5730a6a1727633a22d9f12189c1d87f879a1ebc6549a50d8  verify/compare.json
b8c62b26d849d11e42bb38f22d50f374a47d9d0811292bd0bd15643657906372  verify/compare.py
223902603504a554d24973a170ebb184dfd251a5f350798d58ba04b0c294a65d  verify/verify.json
51b3956770afb930781a531db0475328b2d395d0e4e88e9e4c8db6cc86031ed8  verify/verify_all.py
20944091d96bf7402715ea2e66805aeb325d6092d50da52fb98ef3ed718bac07  verify/verify_extra.json
9d482701564258bdd1005678619bff16664547ed3fc1c60d5ef8a9777c588807  verify/verify_extra.py
```
