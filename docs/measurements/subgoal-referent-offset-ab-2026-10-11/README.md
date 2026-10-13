# The anchored plan's end against the goal pose as the policy's referent on the sim-bridge lane, 2026-10-11

On the sim-bridge lane, the subgoal the DEPTH_SUBGOAL policy holds through its terminal approach
is the last point of the anchored Nav2 plan. On the earlier rig reaches, in
[`goal-a-rig-gate-v3-2026-09-25`](../goal-a-rig-gate-v3-2026-09-25/README.md) and
[`goal-a-cli-video-2026-10-09`](../goal-a-cli-video-2026-10-09/README.md), that point sat
0.012–0.065 m from the goal pose (the pre-registration's figures), while the inference node ends a
goal on the TF distance to the goal pose itself. In training, the completion referent is the
path's last point. This record asks whether making the anchored plan end at the goal pose shortens
the terminal band and moves the policy's quiet stop inside the 0.30 m arrival radius.

Twelve CLI-submitted reaches of v3 ran in one session, alternating two arms of the subgoal
generator: **C**, as on `main`, holds the plan's own last point; **T**, a local patch, moves that
point onto the goal. They are scored against reads and a decision rule committed before the first
reach.

**By the pre-registered rule, the offset is recorded as a minor term.**
- **Band time.** From the first pose sample within 0.42 m of the goal to the node's result, the
  median over the six T reaches is 4.37 s sim, against 6.33 s over the six C reaches, a ratio of
  0.69. The rule needed 0.5 or less.
- **Quiet stop.** None of the six T reaches had its quiet stop inside 0.30 m; the rule needed at
  least four. Every T reach went quiet 0.309–0.352 m from the goal, which on T is its referent:
  the subgoal held through the band sat 0.000–0.001 m from the goal.
- **The reaches.** All twelve ended on the node's own `SUCCEEDED`. The CLI reported each as
  `succeeded`, with no executor cancel, and every join reads `node_result_executor_agrees`. None
  is void: every manipulation check passed, every transit reached the start on its first attempt,
  and every planner dry-check matched.

**Beside the rule, the `vx` sign-flip share was lower on T than on C, on both definitions.** The
flip share is a read the pre-registration reports beside the rule, not part of it.
- By `band.py`'s definition the median is 0.009 on T against 0.147 on C. By `reach_terminal.py`'s,
  over the window from the last inward crossing of 0.42 m, it is 0.095 against 0.204.
- On both definitions, five of the six C reaches sit above every T reach. The sixth, D10, reads
  0.000.
- There are six reaches per arm in one session, and the two definitions differ by about a factor
  of ten on T. The difference is suggestive, not established (§5).

**Computed after the run, with n = 6: where the plan end is the referent, the band time tracks how
far short of the goal the referent sits** (§5a). The signed offset at band entry is the goal distance
minus the held referent's distance; positive means the referent is short of the goal. In the C arm:

| reach | offset | band |
|---|---:|---:|
| D10 | −0.051 m | 1.62 s |
| D12 | +0.004 m | 4.62 s |
| D02 | +0.037 m | 5.64 s |
| D06 | +0.026 m | 7.02 s |
| D04 | +0.026 m | 9.37 s |
| D08 | +0.061 m | 41.81 s |

Spearman's rank correlation is +0.77.
- **Out of sample.** The same relation holds on the 13 earlier sim-bridge reaches with the plan end
  as referent that succeeded: Spearman +0.76.
  - The fourteenth, R2, ran to the bound and held its band quietly (flip share 0.00, 0.025 m/s
    commanded) at an offset of 0.000 m.
- **The flip share follows the same relation.** The reaches with the most back-and-forth are the
  high-offset, high-commanded-speed ones: offset against flip share is +0.77 here and +0.88 on the
  earlier reaches. So the flip-share drop with the referent on the goal reads as the same
  phenomenon.
- **The rule's outcome stands.** By the pre-registered rule the offset is a minor term. This is what
  it governs: the tail of the band and its back-and-forth, not the stop-short.

**The policy still goes quiet before the radius with the referent on the goal.** Here 11 of the 12
reaches went quiet outside 0.30 m, all six T reaches among them. Quiet is ten ticks commanding
under 0.05 m/s, not a standstill. Every reach still ended on the node's first tick within 0.30 m,
the T reaches 0.30–4.21 s after going quiet (§5).

As context: the pre-registration's prior cites v3's training-env median quiet-stop distance,
0.174 m. That median counts stops made after the dwell, inside 0.30 m, which this lane cannot
observe. The comparable figure is the share of band entries that go quiet before the first 0.30 m
crossing ([`policy-time-to-goal-baseline-2026-10-11`](../policy-time-to-goal-baseline-2026-10-11/README.md)
§3, rows 0–599):
- 4 of 179 (2.23 %) without the node's caps;
- 37 of 513 (7.21 %) under them;
- here, 11 of the 12 reaches.

That difference is the finding. This record does not identify which lane condition produces it (§7).

## 1. Setup

| | |
|---|---|
| change on the rig | a local, unpushed patch of the subgoal generator, `node/subgoal_generator_node.py` (sha256 `c8085992…`): `main`'s file (`52c9beed…`, the same bytes as `9f7d207`'s `strafer_inference/subgoal_generator_node.py`) plus `referent_arm.patch`. It is bind-mounted read-only over the image's installed copy in the `inference` container (`node/inference-referent-node.yml`), and its sha256 inside the container matched (`stage0/referent_node_mount.txt`). No tree change |
| images | `strafer-gpu:humble` (`inference`) and `strafer-cpu:humble` (the other four), revision label `9f7d20758d5b` (`main` at `9f7d207`), read from all five running containers (`stage0/images.txt`) |
| scene | `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, seed 42 (`logs/dgx/bridge.log`, line 42), one bridge launch for all twelve reaches |
| cadence contract | `publish 30.00 Hz sim`, `frame_skip=3 (derived, derived 3)`, bridge tick 120 Hz (`logs/dgx/bridge.log`) |
| bridge launch | headless (`run_sim_in_the_loop.py --mode bridge --headless --enable_cameras`) through `tools/kit_boot_watchdog.sh`, from the sim host's main checkout at `9f7d207` (`session/d_session.sh`, `bridge`). It booted on attempt 1 and ran the whole session |
| SLAM key | `enrich_referentab1`, fresh: its database was 0 MB at start (`logs/nx/strafer_slam.log`) |
| policy | `strafer_depth_subgoal_v3_999.onnx`, sha256 `c866bfd5…`, sidecar `d22e3504…`, verified inside the running `inference` container |
| lane | `hybrid_nav2_strafer` + `DEPTH_SUBGOAL`. The generator anchors one path per mission (`anchoring=mission`) with a 1.0 m lookahead. The node's goal bound is 60 s, its arrival radius 0.30 m and its velocity cap (0.7841 m/s, 2.0933 rad/s). The executor runs on sim time with a 65.0 s policy budget (`stage0/`) |
| planner / VLM | Qwen3-4B and Qwen2.5-VL-3B on the sim host's GPU beside Kit, from the main checkout, over the direct link; health round-trips of 2 ms (planner) and 6 ms (VLM) from the `autonomy` container, both `model_loaded` true (`stage0/service_health.txt`) |
| transport | the direct link, with the Cyclone pin selected in all five containers; one RELIABLE depth publisher and two RELIABLE subscribers |
| harness | the tools of the CLI confirmation set ([`goal-a-cli-confirmation-2026-10-03`](../goal-a-cli-confirmation-2026-10-03/README.md)), unchanged. `stage0/harness_sha256.txt` holds 22 digests. The first 21 are the harness tools', and each matches its entry in that set's `tools_sha256.txt` (`goal-a-cli-confirmation-2026-10-03/record-files/` in the evidence repository); the 22nd is `session/stage0.sh`'s and matches the deposited copy. Per reach, `run_cli_one.sh` runs the scripted transit to the nominal start (0.075, −0.035, 130°), the passive observer, the planner dry-check, then one `make submit-deploy` |
| goals | the video set's (`goal-a-cli-video-2026-10-09`) G1 (−2.00, 2.25), a straight reach, and L2 (−2.75, −2.25), where the executor first turns the robot in place |
| order | arms alternate every reach and each goal gets three of each: T C T C … with goals G1 L2 L2 G1 G1 L2 L2 G1 G1 L2 L2 G1 (`manifest.json`) |
| session | 2026-10-11 CDT: the first slot started at 20:13:33 and the last result came at 21:59:09 (`provenance.md`) |

**Definitions** (the pre-registration's; `ab_summary.py` implements them):
- **Band time** runs from the first pose sample within 0.42 m of the goal to the node's result, in
  s sim. A reach that ends `ABORTED` at the node's bound ranks above every completed reach in the
  median. None did.
- The **quiet stop** is the 10th consecutive `/cmd_vel` tick with planar speed below 0.05 m/s
  inside the band window. Its **goal distance** is read from the 10 Hz pose.
  - The node succeeds on its first tick within 0.30 m, so the band window ends at the crossing. A
    quiet stop recorded in it is therefore at or outside 0.30 m, apart from the pose's sampling.
  - A reach that succeeds with no 10-tick quiet run in its band **counts as inside 0.30 m**, as does
    one whose quiet stop is below 0.30 m.
- The **`vx` sign-flip share** counts consecutive band ticks of opposite `vx` sign, both above
  0.01 m/s in magnitude, over all consecutive band ticks (`band.py`). The variant over the last
  inward crossing (`reach_terminal.py`) is reported beside it (§5).

**What was written down first.** The pre-registration, the manifest, the scorer, the three read
tools (fixed by the commit; §6 on their digests), the node patch and its files, three session
scripts and the stage-0 checks were committed to the evidence repository at `8a7648f`, at
20:13:09 CDT on 2026-10-11. The first slot
started at 20:13:33 (`logs/dgx/chain.log`), and its goal was accepted at 20:15:38. None of those
22 files changed afterwards except by the redaction of a network address (Evidence).

## 2. The arms and the manipulation check

**The patch.** One parameter, `referent_arm`, read at each anchoring:
- `plan_end` (arm C) behaves as `main`.
- `snap` (arm T): when a plan is anchored while a goal is active, the plan's last point is replaced
  by the active goal's position in the map frame. The plan keeps its point count; only its last
  segment changes. The node logs the move as a `referent_arm=snap` line.

**The switch.** Before each reach, `session/d_chain.sh` set the parameter with `ros2 param set` and
read it back with `ros2 param get`. All twelve read-backs match the slot's arm
(`logs/dgx/chain.log`). Each reach anchored once, on its goal's arrival (`goal_changed`). At the
end of the session the generator's status line counts 13 admissions: 12 `goal_changed` and the
session's first plan (`no_anchor`), before D01's goal. There was no cross-track or
`anchor_in_collision` admission (`logs/nx/strafer_inference.log`).

**The six `snap` lines**, one per T reach, each within 3 ms of that reach's anchoring line and
14–81 ms wall after its goal's acceptance, by the node's own log stamps
(`logs/nx/strafer_inference.log`; `chain.log`'s running count rose only after T slots):

| reach | plan end (map) | goal | moved |
|---|---|---|---:|
| D01 (G1) | (−1.974, 2.224) | (−2.000, 2.250) | 0.036 m |
| D03 (L2) | (−2.787, −2.205) | (−2.750, −2.250) | 0.058 m |
| D05 (G1) | (−1.998, 2.292) | (−2.000, 2.249) | 0.043 m |
| D07 (L2) | (−2.763, −2.299) | (−2.750, −2.250) | 0.051 m |
| D09 (G1) | (−1.963, 2.201) | (−2.000, 2.250) | 0.062 m |
| D11 (L2) | (−2.787, −2.246) | (−2.750, −2.251) | 0.037 m |

Before the snap, the T plans ended 0.036–0.062 m from the goal. The C plans' held ends were
0.051–0.064 m from it, apart from D12's 0.014 m.

**The manipulation check** passed on all twelve (`ab_summary.txt`, last column):
- **T:** the band held one subgoal position, 0.0002–0.0006 m from the intended goal, against the
  rule's 0.005 m, and each reach's `snap` line appeared at its anchoring (above). The goal the
  executor dispatched differs from the intended one by under 1 mm (the observer's
  `dispatched_goal_err_m`), which accounts for what remains.
- **C:** one subgoal position per band, 0.014–0.064 m from the goal, inside the expected
  0.012–0.065 m.

## 3. The twelve reaches

**The reads** (`ab_summary.txt` and `ab_summary.json`, except three columns):
- The last-inward band and its flip share are from `reach_terminal.json`'s
  `terminal_dither_prereg`, a key named for the CLI confirmation set's pre-registration. Here the
  flip share is this pre-registration's variant beside the rule, and the last-inward band time was
  computed after the run.
- The snap's move is from the node's log.

| slot | reach | goal | arm | band, first entry (s) | band, last inward crossing (s; computed after the run) | quiet stop, m from goal (s after band entry) | inside 0.30 m | flip share, `band.py` | flip share, `reach_terminal.py` | band subgoal from goal (m) | snap moved (m) |
|---:|---|---|---|---:|---|---|---|---:|---:|---:|---:|
| 1 | `D01_G1_T` | G1 | T | 5.34 | 5.34 | 0.352 (+1.1) | no | 0.031 | 0.100 | 0.000 | 0.036 |
| 2 | `D02_L2_C` | L2 | C | 5.64 | 5.64 | none in band | yes | 0.094 | 0.196 | 0.058 | — |
| 3 | `D03_L2_T` | L2 | T | 3.43 | 3.43 | 0.346 (+1.4) | no | 0.010 | 0.059 | 0.000 | 0.058 |
| 4 | `D04_G1_C` | G1 | C | 9.37 | 9.37 | 0.373 (+0.7) | no | 0.203 | 0.269 | 0.056 | — |
| 5 | `D05_G1_T` | G1 | T | 4.24 | 4.03 (2 entries) | 0.344 (+1.2) | no | 0.008 | 0.108 | 0.001 | 0.043 |
| 6 | `D06_L2_C` | L2 | C | 7.02 | 7.02 | 0.350 (+0.9) | no | 0.199 | 0.290 | 0.064 | — |
| 7 | `D07_L2_T` | L2 | T | 1.96 | 1.96 | 0.309 (+1.7) | no | 0.000 | 0.018 | 0.000 | 0.051 |
| 8 | `D08_G1_C` | G1 | C | 41.81 | 16.33 (2 entries) | 0.394 (+0.7) | no | 0.244 | 0.178 | 0.062 | — |
| 9 | `D09_G1_T` | G1 | T | 4.72 | 4.72 | 0.341 (+1.0) | no | 0.035 | 0.106 | 0.000 | 0.062 |
| 10 | `D10_L2_C` | L2 | C | 1.62 | 1.62 | 0.303 (+1.6) | no | 0.000 | 0.000 | 0.051 | — |
| 11 | `D11_L2_T` | L2 | T | 4.50 | 4.50 | 0.348 (+1.0) | no | 0.007 | 0.090 | 0.001 | 0.037 |
| 12 | `D12_G1_C` | G1 | C | 4.62 | 4.62 | 0.317 (+1.9) | no | 0.086 | 0.212 | 0.014 | — |

**The reaches** (the observer's records, `logs/nx/runs/container_gate/<label>.json`, and the joins,
`joined/<label>.prereg.json`). The command is the one the observer formed from the measured start,
as metres forward / left. The start offset is from the nominal start, in the map frame, at that
moment. The pre-rotation is the executor's in-place turn before it sends the policy its goal
(`cli.pre_rotation`, `yaw_change_deg` over `nonzero_span_sim`). The reach runs from the policy
goal's acceptance to the node's result. The final distance is the observer's TF lookup at the
result. The `map→odom` shift is SLAM's net movement of the map frame over the reach
(`mission.slam_map_odom_shift_m`).

| slot | reach | arm | command (m fwd / left) | start offset | executor pre-rotation | reach (s sim) | final distance, TF (m) | node / CLI / join | `map→odom` shift (m) | RTF |
|---:|---|---|---|---|---|---:|---:|---|---:|---:|
| 1 | `D01_G1_T` | T | 3.088 / 0.018 | 0.040 m / +1.16° | none | 10.97 | 0.2998 | `SUCCEEDED` / `succeeded` / agrees | 0.057 | 0.128 |
| 2 | `D02_L2_C` | C | 0.171 / 3.599 | 0.044 m / +1.49° | 80.35° in 3.26 s | 11.99 | 0.2996 | `SUCCEEDED` / `succeeded` / agrees | 0.100 | 0.139 |
| 3 | `D03_L2_T` | T | 0.075 / 3.578 | 0.021 m / −0.98° | 82.49° in 3.35 s | 9.83 | 0.2880 | `SUCCEEDED` / `succeeded` / agrees | 0.030 | 0.139 |
| 4 | `D04_G1_C` | C | 3.104 / 0.161 | 0.037 m / −1.28° | none | 15.21 | 0.2996 | `SUCCEEDED` / `succeeded` / agrees | 0.112 | 0.137 |
| 5 | `D05_G1_T` | T | 3.063 / 0.044 | 0.062 m / +0.34° | none | 9.43 | 0.3003 | `SUCCEEDED` / `succeeded` / agrees | 0.110 | 0.137 |
| 6 | `D06_L2_C` | C | 0.152 / 3.558 | 0.044 m / +1.08° | 81.35° in 3.30 s | 12.88 | 0.2997 | `SUCCEEDED` / `succeeded` / agrees | 0.093 | 0.138 |
| 7 | `D07_L2_T` | T | 0.074 / 3.577 | 0.032 m / −1.19° | 82.50° in 3.33 s | 7.85 | 0.2957 | `SUCCEEDED` / `succeeded` / agrees | 0.096 | 0.138 |
| 8 | `D08_G1_C` | C | 3.066 / 0.110 | 0.024 m / −0.10° | none | 47.98 | 0.2993 | `SUCCEEDED` / `succeeded` / agrees | 0.140 | 0.131 |
| 9 | `D09_G1_T` | T | 3.061 / 0.065 | 0.025 m / +0.98° | none | 10.31 | 0.3000 | `SUCCEEDED` / `succeeded` / agrees | 0.113 | 0.128 |
| 10 | `D10_L2_C` | C | 0.151 / 3.547 | 0.040 m / +0.65° | 82.57° in 3.31 s | 7.39 | 0.2996 | `SUCCEEDED` / `succeeded` / agrees | 0.004 | 0.129 |
| 11 | `D11_L2_T` | T | 0.058 / 3.583 | 0.029 m / −0.52° | 84.11° in 3.34 s | 10.75 | 0.2994 | `SUCCEEDED` / `succeeded` / agrees | 0.086 | 0.127 |
| 12 | `D12_G1_C` | C | 3.068 / 0.182 | 0.022 m / −1.47° | none | 9.97 | 0.3006 | `SUCCEEDED` / `succeeded` / agrees | 0.086 | 0.128 |

- "agrees" is the join's reading `node_result_executor_agrees`. Every join's `VERDICT` is
  `REACHED`, and none records an executor-attributable cancel.
- Every planner dry-check compiled the command to one `translate` step matching to 0.001 m
  (`logs/nx/runs/<label>.plancheck.out`).
- Every transit reached the start on its first attempt: each `to_start.out` holds one
  `TO_START RESULT=OK` line and no retry.
- Three final distances read 0.3000–0.3006 m (D05, D09, D12). That is the observer's TF lookup at
  the result, not the node's own tick, which ended the goal inside 0.30 m. The observer's
  cross-check agrees with the status within its 0.05 m tolerance, and none of the three records
  carries a flag.

## 4. The decision rule and its result

**The rule, as committed.** "The arm is worth a PR only if the median band time over the six T
reaches is at most half the median over the six C reaches, and the quiet-stop goal distance is
inside 0.30 m on at least 4 of the 6 T reaches. Otherwise the offset is recorded as a minor term."

**Result** (`ab_summary.txt`):

| | T | C | condition | met |
|---|---:|---:|---|---|
| band time, pooled median | 4.37 s | 6.33 s | T ≤ 0.5 × C = 3.16 s | no (ratio 0.69) |
| quiet stop inside 0.30 m | 0 of 6 | 1 of 6 | T on ≥ 4 of 6 | no |

**Decision: the offset is recorded as a minor term.** Both conditions fail; either alone would have
decided it.

Per-goal medians, reported beside the pooled ones as registered:

| goal | band median, T | band median, C | inside, T | inside, C |
|---|---:|---:|---|---|
| G1 | 4.72 s | 9.37 s | 0 of 3 | 0 of 3 |
| L2 | 3.43 s | 5.64 s | 0 of 3 | 1 of 3 |

- C's one reach counted inside 0.30 m is D02, under the rule's convention: it succeeded with no
  10-tick quiet run in its band.
- Computed after the run: the pooled C median does not depend on D08's 41.81 s. Any value at or
  above 7.02 s gives the same 6.33 s.
- The pre-registration notes that, with three reaches per goal and arm, and G1's band time ranging
  from 3.46 s (`goal-a-cli-confirmation-2026-10-03`) to 18.72 s (`goal-a-cli-video-2026-10-09`)
  across the earlier sessions, the rule can miss a real effect. A pass would have been a reason to
  write a PR, not a measured effect size; this fail is not a measured effect size either.

## 5. Beside the rule

**The `vx` sign-flip share** (pre-registered as a read, not part of the rule).
- **The two definitions.**
  - `band.py`: consecutive band ticks of opposite `vx` sign, both above 0.01 m/s in magnitude,
    over all consecutive band ticks, from the first entry into 0.42 m to the result.
  - `reach_terminal.py`: consecutive commands of opposite `vx` sign among the commands with |`vx`|
    above 0.0001 m/s, over the consecutive pairs of those commands, from the last inward crossing
    of 0.42 m to the result.
- **The values.**

  | definition | T median (range) | C median (range) |
  |---|---|---|
  | `band.py` (`ab_summary.txt` medians; ranges from `ab_summary.json`) | 0.009 (0.000–0.035) | 0.147 (0.000–0.244) |
  | `reach_terminal.py`, last inward crossing (`reach_terminal.json`) | 0.095 (0.018–0.108) | 0.204 (0.000–0.290) |

- **The pattern.** On both definitions, five of the six C reaches lie above the highest T reach.
  The sixth, D10, reads 0.000 on both; D07 (T) also reads 0.000 on `band.py`'s.
- **Why it is not established.** There are six reaches per arm, in one session, in one room, on
  two goals. The two definitions differ by about a factor of ten on T, so the size of the
  difference depends on the definition.
- **Computed after the run, not pre-registered: why the two definitions part.**
  - The T arm's band commands are small. Over the band, the T reaches' mean commanded planar speed
    was 0.038–0.046 m/s, against 0.055–0.209 m/s on C. The chassis moved at 0.025–0.032 m/s on T
    and 0.021–0.071 m/s on C. Both are from `reach_terminal.json`, `terminal_approach`.
  - Most of T's sign flips involve a tick commanding |`vx`| at or under 0.01 m/s, which `band.py`
    does not count. Over `band.py`'s window (first entry to result, from the observer's
    `series.cmd`), counting flips among the commands with |`vx`| above 0.0001 m/s, its floor
    discards 67–100 % of each T reach's flips against 21–59 % of each C reach's. D10 has none.
  - The two denominators differ by 1–8 ticks per band, at most 2 %. Ticks with
    0 < |`vx`| ≤ 0.01 m/s are a similar share of the band on both arms: 9–31 % on T, 0–24 % on C.

**Band times.**
- **Spread.** T's band times are 1.96–5.34 s; C's are 1.62–41.81 s.
- **From the last inward crossing.** `reach_terminal.py` also gives the band time from the last
  inward crossing of 0.42 m. Its medians, computed after the run, are 4.265 s on T and 6.33 s on C,
  a ratio of 0.67.
- **D08** (C, G1) entered 0.42 m 6.17 s after its goal's acceptance, left the band and re-entered it
  at 31.66 s. The node's result came at 47.98 s. Its band time is therefore 41.81 s from the first
  entry and 16.33 s from the last. Over that last stretch its mean commanded planar speed was
  0.043 m/s (`reach_terminal.json`; computed after the run).
- **D05** (T, G1) also entered twice, at 5.18 and 5.39 s: 4.24 s from the first entry and 4.03 s
  from the last.

**The quiet stop on T.** The distances are the pre-registered read; the median and the time to
the result are computed after the run from `ab_summary.json`.
- The six T reaches went quiet 1.0–1.7 s after entering the band, 0.309–0.352 m from the goal
  (median 0.345 m). The node's result followed 0.30–4.21 s later.
- "Quiet" is a run of ten ticks commanding under 0.05 m/s, not a standstill. The T arm's mean
  commanded speed over the band was itself under 0.05 m/s (above).
- The pre-registration's prior expected the T quiet stops at about 0.30 m from the goal, if the
  video set's stop distances (0.295 and 0.310 m from the subgoal) held against the snapped
  referent. They sat 0.009–0.052 m beyond that.

**Confounds the pre-registration asked to report.**
- **`anchor_in_collision`:** none in the session (above, §2).
- **Start offsets:** 0.021–0.062 m and −1.47° to +1.49° from the nominal start.
- **SLAM:** `map→odom` moved 0.004–0.140 m over a reach (net, per reach; §6).
- **RTF:** 0.127–0.139 per reach, headless, with no livestream.

**Also observed, not pre-registered.**
- **Two planner failures**, in D05 and D09 (both T; two events, no arm comparison is read from
  them). In each, a fresh Nav2 plan failed (status 6) and the generator kept its plan. Under
  `anchoring=mission` a fresh plan does not re-anchor, so neither changed a referent.
- **One watchdog trip per reach**, at the goal's start: `Watchdog tripped mid-mission, publishing
  zero twist: stale=['subgoal']`, within 0.3 s wall of the acceptance (12 lines).
- **Cadence.** The node's `CADENCE SHORTFALL` warnings average over windows that include the idle
  time between goals ("over 100.2 s sim"). Computed after the run from its cadence lines inside each
  goal: inferences rose at 29.88–30.20 per s sim, all fresh, with no further watchdog skip.

## 5a. Band time against the referent's offset (computed after the run)

`posthoc_offset_band.py` → `posthoc_offset_band.txt`, which was not pre-registered. For each reach
with the anchored plan's end as the referent, at band entry (the first pose sample within 0.42 m of
the goal):
- **The signed offset.** The goal distance minus the distance to the subgoal held through the band.
  Positive means the referent sits short of the goal, on the robot's side.
- **The reads beside it.** The band time, the `vx` flip share (`band.py`'s definition) and the mean
  commanded planar speed in the band, as `planend.py` and `band.py` read the same records.

**This A/B's C arm** (n = 6):

| reach | signed offset (m) | band time (s) | `vx` flip share | mean commanded speed (m/s) |
|---|---:|---:|---:|---:|
| D10 | −0.051 | 1.62 | 0.00 | 0.073 |
| D12 | +0.004 | 4.62 | 0.09 | 0.054 |
| D02 | +0.037 | 5.64 | 0.09 | 0.070 |
| D06 | +0.026 | 7.02 | 0.20 | 0.149 |
| D04 | +0.026 | 9.37 | 0.20 | 0.142 |
| D08 | +0.061 | 41.81 | 0.24 | 0.208 |

Spearman's rank correlation, using average ranks for the tied offsets:
- offset against band time, +0.77;
- offset against flip share, +0.77;
- mean commanded speed against flip share, +0.77.

On the T arm every offset is within 0.001 m of zero, so no correlation is computed there.

**The earlier sim-bridge reaches with the plan end as the referent**: the gate
([`goal-a-rig-gate-v3-2026-09-25`](../goal-a-rig-gate-v3-2026-09-25/README.md)), the CLI
confirmation set ([`goal-a-cli-confirmation-2026-10-03`](../goal-a-cli-confirmation-2026-10-03/README.md))
and the video set ([`goal-a-cli-video-2026-10-09`](../goal-a-cli-video-2026-10-09/README.md)).
Two of their records are not in the table:
- gate R3 never entered 0.42 m;
- the CLI set's R1 was not submitted, so its record has no result.

| reach | status | signed offset (m) | band time (s) | `vx` flip share | mean commanded speed (m/s) |
|---|---|---:|---:|---:|---:|
| gate G1 | succeeded | +0.014 | 13.88 | 0.26 | 0.226 |
| gate R1 | succeeded | −0.088 | 2.29 | 0.00 | 0.107 |
| gate L1 | succeeded | +0.051 | 7.05 | 0.24 | 0.222 |
| gate L2 | succeeded | +0.001 | 7.85 | 0.15 | 0.092 |
| gate R2 | aborted at the bound | −0.000 | 52.68 | 0.00 | 0.025 |
| gate F1 | succeeded | −0.000 | 5.08 | 0.01 | 0.045 |
| gate F2 | succeeded | −0.039 | 1.34 | 0.00 | 0.082 |
| gate F3 | succeeded | −0.013 | 1.26 | 0.00 | 0.080 |
| CLI FX | succeeded | +0.005 | 4.84 | 0.13 | 0.082 |
| CLI G1 | succeeded | −0.008 | 3.46 | 0.03 | 0.048 |
| CLI L1 | succeeded | +0.028 | 5.37 | 0.21 | 0.144 |
| CLI L2 | succeeded | +0.016 | 6.50 | 0.14 | 0.112 |
| video G1 | succeeded | +0.025 | 18.87 | 0.25 | 0.181 |
| video L2 | succeeded | +0.033 | 11.18 | 0.31 | 0.236 |

- **The 13 that succeeded.** Offset against band time is +0.76, and offset against flip share is
  +0.88.
- **All 14.** Offset against band time is +0.64, offset against flip share +0.87, and mean commanded
  speed against flip share +0.83.
- **R2 is the exception.** It ran to the bound, holding its band quietly (flip share 0.00,
  0.025 m/s) at an offset of 0.000 m. That is a hold, not a dither.

**How to read this.** These are rank correlations on small samples, computed after the run. The
earlier reaches were not run to test them.
- **The pattern.** Where the referent sits short of the goal, the policy's band runs long and its
  `vx` changes sign often. Where the referent is at or past the goal, the band is short and quiet.
- **A reading of why, not tested here.** The policy stops some 0.31–0.35 m from its referent. The
  node ends a goal at 0.30 m from the goal. With the referent short of the goal, the node's line
  lies inside the policy's own stopping distance, so the policy has to close in on a referent it
  already treats as reached.

## 6. Deviations and notes

- **None of the void conditions fired.** There was no bridge or robot-stack restart, and no reach
  was re-run. The bridge booted on its watchdog's first attempt and ran all twelve reaches; its
  final line reads `attempt 1: exit=0 wall=6701s` (`logs/dgx/bridge.log`).
- **The GPU rule held as registered.** The planner and VLM ran on the sim host's GPU beside Kit.
  Before the bridge launched, the GPU listed exactly their two server processes, PIDs 3881866 and
  3881865 (`logs/dgx/gpu_before_bridge.txt`; the PIDs are in each service's log).
- **The SLAM key.** The robot host's deploy `.env` carried the session's SLAM token,
  `referentab1`, for the session and was restored from its backup afterwards. The restore is not
  recorded in the deposit. The SLAM log records `scene_key=enrich_referentab1` with a 0 MB
  database at start.
- **Base and perception were not started.** `session/nx_up.sh` brings up only the five named
  services, `stage0/images.txt` (taken before the first reach) shows exactly those five containers,
  and each container log holds one launch.
- **The read tools are fixed by commit, not by a digest list.** `tools/planend.py`, `band.py` and
  `reach_terminal.py` are in `8a7648f`. The pre-registration's "pinned by digest below" carries no
  list, and of the three only `reach_terminal.py`'s digest appears in that commit, in
  `stage0/harness_sha256.txt`.
- **The SLAM confound is reported as a net shift.** The pre-registration asked for `map→odom`
  corrections as in the CLI sets, which give a count and the largest per window. This record gives
  each reach's net `map→odom` shift from the observer's record (`mission.slam_map_odom_shift_m`)
  instead; the corrections were not tallied.
- **`session/d_chain.sh`**, which drove the twelve reaches, was committed with the runs, not with
  the pre-registration.
- **The wrapper's join.** `run_cli_one.sh` reads `STALE_RECORD`, the CLI sets' known defect, so the
  joins were run by hand with the CLI confirmation set's pinned `cli_mission.py`
  (`goal-a-cli-confirmation-2026-10-03/record-files/tools/cli_mission.py` in the evidence
  repository, sha256 `4dd6e184…`; the command is in `DEPOSIT.md`). Re-running them gives the same
  joins apart from their embedded input paths.
- **A teardown defect.** `d_session.sh stop-local` matches processes by command-line text. At
  teardown it also signalled the shell that invoked it. The bridge and both services still shut
  down cleanly (`DEPOSIT.md`).
- **No tree change.** The node patch is local and unpushed, and this record proposes none.

## 7. What this does not show

- **A small effect.** Six reaches per arm, in one session, one room (seed 42), from one start, to
  two goals. The rule tests for a large effect (band time halved, and the quiet stop inside
  0.30 m). A smaller one is not excluded, and the band-time ratio of 0.69 is not an effect size.
- **The flip share's difference** as established, or what produces it (§5).
- **The stop-short's cause**, the policy going quiet outside the radius. With the referent on the
  goal, it still went quiet 0.31–0.35 m out.
  - In its training env, over rows 0–599, 4 of 179 band entries (2.23 %) went quiet before their
    first 0.30 m crossing without the node's caps (run 2). Under the node's speed cap and 60 s
    bound (run 3), 37 of 513 (7.21 %) did. Here 11 of 12 reaches did.
  - Both figures are from
    [`policy-time-to-goal-baseline-2026-10-11`](../policy-time-to-goal-baseline-2026-10-11/README.md)
    §3. All three are read before the first 0.30 m crossing, the only side of the line the node
    lets this lane observe.
  - Computed after the run: `planend.py` (in `8a7648f`) also prints each quiet stop's distance to
    the held subgoal. The C reaches went quiet 0.312–0.354 m from theirs and the T reaches
    0.308–0.352 m, so all 11 quiet stops were also at or beyond 0.30 m from the referent the
    policy held.
  - With the offset removed, the stop-short persisted on all six T reaches, so an offset of
    0.036–0.062 m does not account for it there. The training env has the same behaviour at a much
    lower rate, so the lane raises the rate of something the policy already does. This record does
    not identify which lane condition raises it.
- **A larger offset.** The snap moved the plan's end 0.036–0.062 m. A plan that ends farther from
  its goal is not tested.
- **Real hardware.** Everything ran on the sim bridge, with the rendered depth stream.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `subgoal-referent-offset-ab-2026-10-11/record-files/` |
| deposit commit | `c3168a0df072e84a8d909d0d9180708821a38cfa` |
| pre-registration commit | `8a7648fad150d40cb7230eb0d16e038c61e36996` |

The pre-registration commit, `8a7648f`, was made at 20:13:09 CDT, before the first slot at
20:13:33. It holds 22 files:
- `PREREGISTRATION.md`, `manifest.json` and the scorer `ab_summary.py`;
- `tools/` (`planend.py`, `band.py`, `reach_terminal.py`);
- the node patch and its files;
- `session/d_session.sh`, `nx_up.sh` and `stage0.sh`;
- the stage-0 checks.

`a213f88` adds the other 239 files. Of the 22, it changes three, each only by replacing a
direct-link address with a placeholder. `ccec8bc` corrects `DEPOSIT.md`'s account of the
pre-registration commit and changes no other file. `c3168a0` adds the post-hoc offset read
(`posthoc_offset_band.*`, §5a).

The deposit holds:
- the launch inputs: the pre-registration, the manifest, the patched and `main` node files, the
  patch, the mount and the session scripts;
- the stage-0 checks (`stage0/`);
- per reach, the harness's files, the observer's record and the join (`logs/nx/runs/`, `joined/`),
  and the session's `map→odom` ride-along;
- the five container logs, and the sim host's bridge, watchdog, service, env-setup, GPU and chain
  logs;
- the reads: `ab_summary.py` with `ab_summary.txt` and `ab_summary.json`, and `reach_terminal.json`
  with its stdout.

**Redactions.** Rig network addresses were replaced with placeholders in 20 text files before the
final commit, and `DEPOSIT.md` lists them. The session scripts do not run as deposited without the
address put back. In the pre-registration commit three files still carry a direct-link address:
`session/d_session.sh` (the robot host's), and `stage0/executor_checks.txt` and
`stage0/stage0.out` (the sim host's, in the planner and VLM URLs).

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/subgoal-referent-offset-ab-2026-10-11/record-files/. \
      docs/measurements/subgoal-referent-offset-ab-2026-10-11/
```

Re-derive the reads with Python 3 and the standard library (the deposited outputs reproduce byte
for byte with Python 3.12.13):

```
cd docs/measurements/subgoal-referent-offset-ab-2026-10-11
python3 ab_summary.py manifest.json logs/nx/runs/container_gate | diff - ab_summary.txt
python3 tools/reach_terminal.py logs/nx/runs/container_gate \
    logs/nx/runs/container_gate/tf_referent.jsonl /tmp/reach_terminal.json \
    --labels $(python3 -c "import json;print(','.join(e['label'] for e in json.load(open('manifest.json'))))") \
    | diff - reach_terminal.stdout
diff /tmp/reach_terminal.json reach_terminal.json
```

None of the three `diff`s shows anything. `ab_summary.py` also rewrites `ab_summary.json` in the
working directory, with the same bytes as the deposited file. The restored files are not part of
the record and want clearing before the next commit.

Verify digests with:

```
cd Sim2RealLab-Artifacts/subgoal-referent-offset-ab-2026-10-11/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit (`DEPOSIT.md` lists the same):

```
1afa721f968a34d3b6fb45a2993415f6fb57447ed8c6425b9281b03305bee806  PREREGISTRATION.md
698d67089697481959a0a4197fdc5263659a879be979d7d4f04a5b8cbc330186  ab_summary.json
7a057fae58ad29dfcf8d04ce6b60a6c069325d8491369f8db5549c2f828f2cd9  ab_summary.py
d31bf537845d168d2900d105ef7d9f08243098b0aa38ee3d0a168738c562e29e  ab_summary.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D01_G1_T.prereg.err
52a64078a9cb71b44c8c55f987d8b6cb68d5d1d55a0780c5dc98e5d14c02f7a8  joined/D01_G1_T.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D02_L2_C.prereg.err
5ba11733401d3d3c4205f54aa678725e1762122aeb1e97d03ca02df05061f7e8  joined/D02_L2_C.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D03_L2_T.prereg.err
08f89aa4d7b7a3e6fb6a82adbc47ffa8eb330660b434f59d8bd57a60d1bc7c7a  joined/D03_L2_T.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D04_G1_C.prereg.err
202385789585826692afa83272014dba0a1e0c82dd66c41169585d9d9d247196  joined/D04_G1_C.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D05_G1_T.prereg.err
4ca1d94c2cdf5dae251a40c719cae006972d101bd86c6d042522f92cfb7ee695  joined/D05_G1_T.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D06_L2_C.prereg.err
9a9d5bcbad7a01c51279b75127bb9ace15f51effd97b9a3b465691d3402f80dc  joined/D06_L2_C.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D07_L2_T.prereg.err
6d8d62f3985ca2e55cf1e1b1aa9663904b9092ff2d1e73da3d984f50e44d27d3  joined/D07_L2_T.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D08_G1_C.prereg.err
6d745bf9f99528df253a2f5f5f7c845d3bfe4d6fe61240d061def9cf55057d84  joined/D08_G1_C.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D09_G1_T.prereg.err
073cea51c41d7974e7492f60dd81e46c1f9841611b440d2646a9e86553c36a38  joined/D09_G1_T.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D10_L2_C.prereg.err
983f235b5ae769b53e7432bce069d87953c4bfe7c0bf65236d4850d69f871843  joined/D10_L2_C.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D11_L2_T.prereg.err
53c252f0913c8e88030037a063cdd38530c359fab0b227c2520b06c91b3e29e8  joined/D11_L2_T.prereg.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  joined/D12_G1_C.prereg.err
68217bc0476368e1a376e9e89dc516babf4c677880e03ff42a4a5e1df3e9c2ee  joined/D12_G1_C.prereg.json
f61a522adca4674848e74f7df6829a959c44b907fcd2b20263deb7ba1a6f233f  logs/dgx/bridge.log
fbadf106431f29b40b9920a49396413140adda380044c3846208293aaaf4c7b3  logs/dgx/chain.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/dgx/chain.out
b3b0e46037bffbb2c5de1ad805105416645b397f3ed2082a997a8451331aad48  logs/dgx/envsetup.log
edc85ad637372275567916edd9765edb6725161a6e2158ae933ce3f2331d9186  logs/dgx/gpu_before_bridge.txt
b4c8bb609769ac69b8d77cfea22102e519c1c7e0a640d1ec58df583e99d50c46  logs/dgx/serve_planner.log
694fdec572a3fa8408b28def894a9af5a9003031816a0a4a15e0caf594816983  logs/dgx/serve_vlm.log
75558d6e5d95640cf51cfd244e7d8393a75533930dc2141afa3533dc2a0dc701  logs/dgx/watchdog_bridge.log
75558d6e5d95640cf51cfd244e7d8393a75533930dc2141afa3533dc2a0dc701  logs/dgx/watchdog_bridge.log.attempt1
3d3bb039f83e0a4d7d2ce160ce6ef51e069092d663a259cd995fd4365586d7e6  logs/nx/runs/D01_G1_T.cli.meta
10043c21d262efadba8fa8e950e0d62115a05691b9c20c295d08bd595d9c9393  logs/nx/runs/D01_G1_T.cli.out
ebca0183b62f639ea1c575be85674caf4109c98cc33b37c84b3f42dd64397779  logs/nx/runs/D01_G1_T.cmd
d7286bf352f00cfe400b6d90d5be7d9df697a34b0fc601dbfe578cbb860e6b94  logs/nx/runs/D01_G1_T.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D01_G1_T.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D01_G1_T.driver.out
6aac3208cbdfdf3095a74b8fbcb45e2cbaa5283e1588d43e71ead3842f78f039  logs/nx/runs/D01_G1_T.plancheck.out
e4b115c83e99339b5619289088eca8a6fe1771c65cbc4aac3cc7bac1172d5d79  logs/nx/runs/D01_G1_T.run.log
551f8e55b920e05852dcfddae1685944f14e7db96bda3edd0d62579d47c58c2a  logs/nx/runs/D01_G1_T.to_start.out
f78c0abea3b979d20fc1cce645bcdbaf5181633eae619d4b807595a5a5b1abfa  logs/nx/runs/D02_L2_C.cli.meta
5c0dbee25b70b6e2f755e87d981087ff637a1cd2f8d0db11803a05a04d113827  logs/nx/runs/D02_L2_C.cli.out
d5ec26b76cad8c8bf5010bed2b1f99b43b06c61f0fda7191c3d449c47fd30a61  logs/nx/runs/D02_L2_C.cmd
d2c722364bd5de09f35cee8cfe4afcfe3c5120716e99402e8218cbe2174b4310  logs/nx/runs/D02_L2_C.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D02_L2_C.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D02_L2_C.driver.out
a20d980ce04887963de2ad52c2491ce53c8527029576c51a5c2acfdc1c25b0b4  logs/nx/runs/D02_L2_C.plancheck.out
91f70dd0224f334e3b917341b04dd33df04fb7beed81ff1f486a98636e6e0db6  logs/nx/runs/D02_L2_C.run.log
efe82961205fbf98ffa322d223308b8419010a0a92a3b4fa8f10ef7599fb9dbe  logs/nx/runs/D02_L2_C.to_start.out
6672362877fb1c7bfdaacb28c4b46b2b16007fa43f22bde835923fde01732240  logs/nx/runs/D03_L2_T.cli.meta
160f5f2e12f9e09bffd9aff6e04d474a61932c38f33129703d807216919eb549  logs/nx/runs/D03_L2_T.cli.out
b8a09677edd47f4a1a74d1cfcbebe31439bef4994cd6324b9a97a935318d73a5  logs/nx/runs/D03_L2_T.cmd
4b7521aa45039a59ba605e42c58832f0ccd4b94c8958814471947b41b4bd04fe  logs/nx/runs/D03_L2_T.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D03_L2_T.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D03_L2_T.driver.out
ab119d88837e09229a1a7cc29847fca3702bd25d2c5be37617fd50aeae223a8c  logs/nx/runs/D03_L2_T.plancheck.out
0f02af3c57bf6423d9856033e6d5ccd4a4fa8330fc1bf12532d360f3da60d04a  logs/nx/runs/D03_L2_T.run.log
0cd600a5aafe9fed003cb2cf0020db78e255543555a198e2dca81950ec99a360  logs/nx/runs/D03_L2_T.to_start.out
87f8673686f8ae65cc9b9efb0e14733c54968b36c3569c81426cfc1b54951de0  logs/nx/runs/D04_G1_C.cli.meta
734a7e1d36b04a84b040f1cc47a797049b56b4fd12d2ff51af4417c52c8d468e  logs/nx/runs/D04_G1_C.cli.out
717baaad86fd4a94d6b6e73c0da93d19d62a61324e3fb3bb56a9fa1ea1c68743  logs/nx/runs/D04_G1_C.cmd
3cac01d9cfcdacae716e752ba7ff13d0e1058559c6552cdc27d0f1c79f705185  logs/nx/runs/D04_G1_C.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D04_G1_C.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D04_G1_C.driver.out
1f263a8f125e5fceb7c5543fee4be416a20f187176fafdf199fdf4059ebf5e36  logs/nx/runs/D04_G1_C.plancheck.out
345d502898a398a80c3e4fbdeca910b067ec0b14ecf0c4e911e8b96740d7c210  logs/nx/runs/D04_G1_C.run.log
0b54f4a319db61e5d9390a3ef3d78593c5cbbbfa4f7b5c54da9018b1f5b54ae3  logs/nx/runs/D04_G1_C.to_start.out
66fae531842d8df79b6603a43f8bd5d5ee0cf5c87478336bd240529a31ab8661  logs/nx/runs/D05_G1_T.cli.meta
f10a82220f703a0ae2e3846d03fcdb1c6f0d155bda9bf4c2f64ed2c104f07e4e  logs/nx/runs/D05_G1_T.cli.out
ab13fd0cb401d82bcb9966d3efaaa2fe404d3cbb72f0512af9f6cbff249d6ea4  logs/nx/runs/D05_G1_T.cmd
a5f87efd3fdbd0da971128f4b48783358bdcee3084caaf603454dc8971f816c5  logs/nx/runs/D05_G1_T.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D05_G1_T.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D05_G1_T.driver.out
6ef878852153dce30712d5b46bdaa260c71bc56c3f0a51062de7809ea9cfa483  logs/nx/runs/D05_G1_T.plancheck.out
206b3c1788515062c50dd9cb4c8ff1ce0ad632fdbda26c3dbccae2cfeec81974  logs/nx/runs/D05_G1_T.run.log
b54444f7f332aa0facfa34134b0de35953b09281c38bcb583c91e39dcea61a09  logs/nx/runs/D05_G1_T.to_start.out
d31003abdda2a55825b9766bebc8024ca8aa60c4bf02ed7006bdfede05de1df0  logs/nx/runs/D06_L2_C.cli.meta
92a4d5ad374d52334e28983852bd384df25c53554fc8b541ee265ca38a53a7ce  logs/nx/runs/D06_L2_C.cli.out
773de0486ccf7cd1f0788406f3bf39ebec9a5af326e41acbcd6acbd7b1a19548  logs/nx/runs/D06_L2_C.cmd
60586601b6700e17f8d36951822459710ab72732ad51cd8306a6520629a313de  logs/nx/runs/D06_L2_C.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D06_L2_C.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D06_L2_C.driver.out
1ebfc31048f7e756cec8f5ce947e875395874e39b35e77d4809951034618979a  logs/nx/runs/D06_L2_C.plancheck.out
d4aff3869b4388cd3c90159353ffea3d568e8ca4e8189b8d1d4e16752f6fc190  logs/nx/runs/D06_L2_C.run.log
365582964d36035b19270afc3eec6682efad9ca7754f07de90a7c12fec842459  logs/nx/runs/D06_L2_C.to_start.out
ae2b382260af1d7cb6346c95a4bf3c4f28d8d9955ffb0eb914dce1cc89d8089e  logs/nx/runs/D07_L2_T.cli.meta
a0f7e1d16925ef1966e4b0e569e858e089ba359320f25dc984942412c9c24137  logs/nx/runs/D07_L2_T.cli.out
bd2dc6f518f2de3a4fdfa6c2108fcdc7de8ea6b7b3f19a4a057fe91cf2b3497d  logs/nx/runs/D07_L2_T.cmd
ed1b8a5f3aa1763b100683a55e26d8c04315ca9993be368cb0e7b9972f70ce8a  logs/nx/runs/D07_L2_T.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D07_L2_T.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D07_L2_T.driver.out
be9402b4a497ad9140b731ee18c16904110035c8870e1d9cf3412e9a074babb0  logs/nx/runs/D07_L2_T.plancheck.out
06d6585814451bb3e166c57a4a2ac03e5ea417b9c80c42579e968814bb0388fb  logs/nx/runs/D07_L2_T.run.log
271f79ca68d8f3d0fc75c5ef456cb0e382b3d22ce7f94e91c18313a7f5d1aeba  logs/nx/runs/D07_L2_T.to_start.out
4366e2e4475e99e491fec3e35920daf046168ee8d305415860e88a155bd81e82  logs/nx/runs/D08_G1_C.cli.meta
0e50122493d2640a4e26ff1d9dc85ec1900d701821fa7f8e448a7f9ff15be6cf  logs/nx/runs/D08_G1_C.cli.out
e07cf2985f424029f098863b4003a9433eda1e17d4a148710ba2d4e4511e2731  logs/nx/runs/D08_G1_C.cmd
31c3e9582e5a0937f98dd99c52e03c5263672ba4ae6c14e5f2a130553ae3c578  logs/nx/runs/D08_G1_C.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D08_G1_C.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D08_G1_C.driver.out
8ea4714c004ec4e45b90e038af3631351930e8887b62b8f4d7d2ffc9da985488  logs/nx/runs/D08_G1_C.plancheck.out
ca1bc0a46138616e1ecd68632ab9f40691445f5e7ba265bb618402a2fd4f7855  logs/nx/runs/D08_G1_C.run.log
797fdb3f2a5a3013140c582fbf2fe8c70cce6b930cd439beee5af22659bfbd28  logs/nx/runs/D08_G1_C.to_start.out
342e75b5d84f7e47d327d6a46437a77a03477cf6dfa6b4c53de845ed17d2817f  logs/nx/runs/D09_G1_T.cli.meta
968d9becc45c4d0a4ecad54fb7c0f5633e5c0e38651e5a1838165019c96adc14  logs/nx/runs/D09_G1_T.cli.out
0ef9efb9291ca994d90204582b92682f4819061d14ece0426fd4d9239ff030d8  logs/nx/runs/D09_G1_T.cmd
af9d783004912068dd34e88f173638cf64e6425277ab5b024d8bccf9d8faf118  logs/nx/runs/D09_G1_T.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D09_G1_T.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D09_G1_T.driver.out
7df7bff154a66601fe5d021e55727204b08bf3ffa87ce365405bb31414a0f4b8  logs/nx/runs/D09_G1_T.plancheck.out
7eee9a4e1b0096396186c413fe2ad59eb72fbdd88936c3691c8d81c6cc1b6d74  logs/nx/runs/D09_G1_T.run.log
12eb139ab2b1028720409db79a27128bae9f246c907b2903cf4e8dfe9c8cd097  logs/nx/runs/D09_G1_T.to_start.out
c599dad273aa8544f21bbc3b78e979300307f4006d8f4585f6d3966d62b890ee  logs/nx/runs/D10_L2_C.cli.meta
9e1a6f454a7d7491b9e3a23833dee0186b104c042b8450244f542d49c5c2f656  logs/nx/runs/D10_L2_C.cli.out
5de7204ff5a54be25112afe408f82c6ed08df714b88828d5a9e9ef5eab0ed291  logs/nx/runs/D10_L2_C.cmd
5431f7f8d50e8fafe2ba296224c4f1a3106b2a08336fe77ed19f85ff6972626f  logs/nx/runs/D10_L2_C.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D10_L2_C.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D10_L2_C.driver.out
d7e41c2e2daa190bd64df32284bfa88fd93641e35e7f0ab142ec04090e76bef2  logs/nx/runs/D10_L2_C.plancheck.out
e7bdb13d76b1e05733001f898edb318c3309b41bbb3c81214815f88c14a7108f  logs/nx/runs/D10_L2_C.run.log
28fd7588575d6c09a89ff7c378fb5c647664f22b4287a8d480bbf13ed11e8adf  logs/nx/runs/D10_L2_C.to_start.out
6fd8ef9e2c19e6e388fa47d93f4778173f29e6e896bb9004ff6e1bc43ca58aba  logs/nx/runs/D11_L2_T.cli.meta
8faa6d944834be42ba7facb2f17492862355a101bc69d9d360ea1b940b1d6b8a  logs/nx/runs/D11_L2_T.cli.out
f4d3acdd905f0a127a1e5d9651b15d252dc9879d5cccd54bcfbdbd42ccd701f0  logs/nx/runs/D11_L2_T.cmd
3f85762560408b2bc3920b36f5af7d46b9c1073f3d257814f45f297fb088dd4b  logs/nx/runs/D11_L2_T.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D11_L2_T.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D11_L2_T.driver.out
c63c39d1beeea435bfe6a241ce1390bb7ea17e8e9f36f651f1dfe8fde09dc1ce  logs/nx/runs/D11_L2_T.plancheck.out
2fa89a9499d58a4f8018c59c0f23ca83c9cf6ccbe124124a1a38944b7daef045  logs/nx/runs/D11_L2_T.run.log
bf701097b8ddbc9c8c3a2693513e2723a1a93128f2466c8521361eb3a721bd75  logs/nx/runs/D11_L2_T.to_start.out
f2c00631c6fcb0f4666f65692e8dbb5745dc96d2c520362b3f29544da88ef9c1  logs/nx/runs/D12_G1_C.cli.meta
bd88adaf6bf49a1ac3c51f9d62a8989c39ddd54fbf7e34f4eb08e9f41bbea10d  logs/nx/runs/D12_G1_C.cli.out
0ba81c5d0e37204ff110d6e8b0d1c23583c7da7a7f1eec8a7c5f159cf4bd94f4  logs/nx/runs/D12_G1_C.cmd
c622cd37189f9d2675db9a0307a6671ecbfd2e27d8d8d9390062b8cbc7597b33  logs/nx/runs/D12_G1_C.cmd.json
e13acd17c2814661b4a436c31d625006b2024c6e790dc25ab06c34c540c47322  logs/nx/runs/D12_G1_C.done
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/D12_G1_C.driver.out
ff246138f621e7642812e5464e022abdba1a5a8aa683ae84ce1584bb7c01837c  logs/nx/runs/D12_G1_C.plancheck.out
e2d0411c8099899ca50b1815be3f6b4ad087b4cb73667b4d024450d0d101dd9f  logs/nx/runs/D12_G1_C.run.log
d8cea8c9accebbf72c337c1a8aa599adf69f255ca0f7d381ac63f91e7f734a15  logs/nx/runs/D12_G1_C.to_start.out
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D01_G1_T.cli_exited
ebca0183b62f639ea1c575be85674caf4109c98cc33b37c84b3f42dd64397779  logs/nx/runs/container_gate/D01_G1_T.cmd
d7286bf352f00cfe400b6d90d5be7d9df697a34b0fc601dbfe578cbb860e6b94  logs/nx/runs/container_gate/D01_G1_T.cmd.json
395d5ae5325f061a5676647bf36c4b9411db2795e8fb10452e021da7a46419bd  logs/nx/runs/container_gate/D01_G1_T.json
aef66b88381518da35d0c8621cc7ffed064cb9cdbe096398fafe39a3e77cec2f  logs/nx/runs/container_gate/D01_G1_T.mission.out
6aa489d19ed730cce0c1e24d382bc56e1e14913e5f5027c09b2662829b30307e  logs/nx/runs/container_gate/D01_G1_T.progress.jsonl
376875682644fef688cfa69a30b6d0fcf72cd5c5239193c5db0419df387c3d34  logs/nx/runs/container_gate/D01_G1_T.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D02_L2_C.cli_exited
d5ec26b76cad8c8bf5010bed2b1f99b43b06c61f0fda7191c3d449c47fd30a61  logs/nx/runs/container_gate/D02_L2_C.cmd
d2c722364bd5de09f35cee8cfe4afcfe3c5120716e99402e8218cbe2174b4310  logs/nx/runs/container_gate/D02_L2_C.cmd.json
13c1f1e949b1a17e37514c2728adf02c511e89dfcc3522d0656c154d2bb8af96  logs/nx/runs/container_gate/D02_L2_C.json
933197a05c59565c13948544c7e3c4a04c7218e754972ff66f17b8828692cef9  logs/nx/runs/container_gate/D02_L2_C.mission.out
725623d637f523987a93592364579233d699e1164cf6ce7ce8ee94fab90d2b7e  logs/nx/runs/container_gate/D02_L2_C.progress.jsonl
6c571b121862e42c6cb0ca98e648763bb7778a900933cc08fb059c616f537eea  logs/nx/runs/container_gate/D02_L2_C.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D03_L2_T.cli_exited
b8a09677edd47f4a1a74d1cfcbebe31439bef4994cd6324b9a97a935318d73a5  logs/nx/runs/container_gate/D03_L2_T.cmd
4b7521aa45039a59ba605e42c58832f0ccd4b94c8958814471947b41b4bd04fe  logs/nx/runs/container_gate/D03_L2_T.cmd.json
0a1b65840f7e2563860f9656b96ae18dd999c2f9fd45ab8fe0d1e987c71ba651  logs/nx/runs/container_gate/D03_L2_T.json
20ca69dfed360d7c8ba4d42cadecd7f30a918462b7894759b00015180b64186a  logs/nx/runs/container_gate/D03_L2_T.mission.out
89423ec7c14889cdb8f1e2ea8fa6aed6a0096ac13e08366b93f152e64835a238  logs/nx/runs/container_gate/D03_L2_T.progress.jsonl
e7d90c40d56ab0d73893b145f8fb84351c5c7f8a1a743ddef912d562da09cb02  logs/nx/runs/container_gate/D03_L2_T.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D04_G1_C.cli_exited
717baaad86fd4a94d6b6e73c0da93d19d62a61324e3fb3bb56a9fa1ea1c68743  logs/nx/runs/container_gate/D04_G1_C.cmd
3cac01d9cfcdacae716e752ba7ff13d0e1058559c6552cdc27d0f1c79f705185  logs/nx/runs/container_gate/D04_G1_C.cmd.json
2f51847d60dc576c0b8b712461cd1fcb5f9ba060f6dd23145c5d61d117f6519f  logs/nx/runs/container_gate/D04_G1_C.json
85deeff0a734958f1971374a88f29cdfb40d57ceb7a08f629f92faf40aada747  logs/nx/runs/container_gate/D04_G1_C.mission.out
8805d6c348f2501327851863745240dc90b974bdba1c6a6aa06d19304b3e885b  logs/nx/runs/container_gate/D04_G1_C.progress.jsonl
955fc873ca30fd92afe77c08d1273bcb56a3851c4a0ff2ddf38bbed1487b581b  logs/nx/runs/container_gate/D04_G1_C.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D05_G1_T.cli_exited
ab13fd0cb401d82bcb9966d3efaaa2fe404d3cbb72f0512af9f6cbff249d6ea4  logs/nx/runs/container_gate/D05_G1_T.cmd
a5f87efd3fdbd0da971128f4b48783358bdcee3084caaf603454dc8971f816c5  logs/nx/runs/container_gate/D05_G1_T.cmd.json
7f4c8ae74efac094369bc1141fd2f7a3acb92b9c5533ae9a5734de94d40fec5a  logs/nx/runs/container_gate/D05_G1_T.json
78848d9082cfae7dbcf39473c8a979476ca05c0a7428e0eaaf568b695811267a  logs/nx/runs/container_gate/D05_G1_T.mission.out
989f5b7484bbfa708e343c228e96c535d5706fd9440a903ac6ab98af6f1c0e46  logs/nx/runs/container_gate/D05_G1_T.progress.jsonl
7e97a8d155b76352fa7abd364be80835153bfb9834a9d09e9c5488d332c5264f  logs/nx/runs/container_gate/D05_G1_T.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D06_L2_C.cli_exited
773de0486ccf7cd1f0788406f3bf39ebec9a5af326e41acbcd6acbd7b1a19548  logs/nx/runs/container_gate/D06_L2_C.cmd
60586601b6700e17f8d36951822459710ab72732ad51cd8306a6520629a313de  logs/nx/runs/container_gate/D06_L2_C.cmd.json
32367e397c7f02a1485fc4ff4302ec5316e2e7ccba070c5a06239f36e848a127  logs/nx/runs/container_gate/D06_L2_C.json
739fa5bd84745c34eda8ce416e6425f5a31a3b0bbab6fa50d68f186df24c6a30  logs/nx/runs/container_gate/D06_L2_C.mission.out
ab8ff23184e7e7a7d07e624b2ea1b9d2e77c8513b5db5ebfec41d6ced3946e44  logs/nx/runs/container_gate/D06_L2_C.progress.jsonl
e6a17d9f4e348e8de0a5c583091c5bffad27b8eb41b0239f85477e9614d8f92b  logs/nx/runs/container_gate/D06_L2_C.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D07_L2_T.cli_exited
bd2dc6f518f2de3a4fdfa6c2108fcdc7de8ea6b7b3f19a4a057fe91cf2b3497d  logs/nx/runs/container_gate/D07_L2_T.cmd
ed1b8a5f3aa1763b100683a55e26d8c04315ca9993be368cb0e7b9972f70ce8a  logs/nx/runs/container_gate/D07_L2_T.cmd.json
eb369f7e1e3fb3779dcca19dd2f4756e6a88d20d08a7057a317d465ac4472e31  logs/nx/runs/container_gate/D07_L2_T.json
c9cea022bbc695be579259b012fc751398d91342edced100668d4ae0fa47c8d4  logs/nx/runs/container_gate/D07_L2_T.mission.out
3e22f20e87d701ade17d551efb1607aac681d5eeb89783829c6891724c8c9c17  logs/nx/runs/container_gate/D07_L2_T.progress.jsonl
a48bef8c8a941437932a6c54a9a7cf52b9077ae708a80d61f4b1e26c587f9678  logs/nx/runs/container_gate/D07_L2_T.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D08_G1_C.cli_exited
e07cf2985f424029f098863b4003a9433eda1e17d4a148710ba2d4e4511e2731  logs/nx/runs/container_gate/D08_G1_C.cmd
31c3e9582e5a0937f98dd99c52e03c5263672ba4ae6c14e5f2a130553ae3c578  logs/nx/runs/container_gate/D08_G1_C.cmd.json
d2e34504efb56eaf23afc785822b3e634f41f27f02f187d673d46826fd53e50f  logs/nx/runs/container_gate/D08_G1_C.json
cc67b9bd09014709f55f9cd4ac83462c796efb873702fca11891fc37ed66c0f2  logs/nx/runs/container_gate/D08_G1_C.mission.out
be496be7edfb3e8409f7bf14b0e6d1bc05dbbba80b38f5fd410c0019e3359c28  logs/nx/runs/container_gate/D08_G1_C.progress.jsonl
b0360350ddb6a73c9db10a7eba65e445891d2c392cac7e0a03e509bf5126ebe9  logs/nx/runs/container_gate/D08_G1_C.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D09_G1_T.cli_exited
0ef9efb9291ca994d90204582b92682f4819061d14ece0426fd4d9239ff030d8  logs/nx/runs/container_gate/D09_G1_T.cmd
af9d783004912068dd34e88f173638cf64e6425277ab5b024d8bccf9d8faf118  logs/nx/runs/container_gate/D09_G1_T.cmd.json
27df06cb3dd8264f123c3bc1cf8da4a2ddd523046426b6c5019282d4ec884d2e  logs/nx/runs/container_gate/D09_G1_T.json
105936c3b94665b18a9486a9aff8e4e2154b973f7402e3c3e35ee6b085e4c5c7  logs/nx/runs/container_gate/D09_G1_T.mission.out
71d90412904e6323fa5b4b496329de3b60fdf9ecdb562a6cf6de38c66910ad87  logs/nx/runs/container_gate/D09_G1_T.progress.jsonl
aa865e9ac0267fa84cbf8a4e46d1c2f7297fb0b6e5ee065f1387c0557dcf8341  logs/nx/runs/container_gate/D09_G1_T.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D10_L2_C.cli_exited
5de7204ff5a54be25112afe408f82c6ed08df714b88828d5a9e9ef5eab0ed291  logs/nx/runs/container_gate/D10_L2_C.cmd
5431f7f8d50e8fafe2ba296224c4f1a3106b2a08336fe77ed19f85ff6972626f  logs/nx/runs/container_gate/D10_L2_C.cmd.json
3ed21e71e12c8f77da3a6aa971e64a391db73f38ce41bf6f5e49ac3d57e2d219  logs/nx/runs/container_gate/D10_L2_C.json
14d5dc95cbc03e495b36de0f3559d97c5aa4b29e42be99c3c914722faf96ed63  logs/nx/runs/container_gate/D10_L2_C.mission.out
6ac79699fe27c81bf6f9483c61de6dd79cfd7efd846761ee8b04c77d62de0787  logs/nx/runs/container_gate/D10_L2_C.progress.jsonl
b5bde3a94f9d2f0298a2ce05bf16b11595f6f23170dd3e96b44b08c41624eaee  logs/nx/runs/container_gate/D10_L2_C.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D11_L2_T.cli_exited
f4d3acdd905f0a127a1e5d9651b15d252dc9879d5cccd54bcfbdbd42ccd701f0  logs/nx/runs/container_gate/D11_L2_T.cmd
3f85762560408b2bc3920b36f5af7d46b9c1073f3d257814f45f297fb088dd4b  logs/nx/runs/container_gate/D11_L2_T.cmd.json
6fe53b9d1c6909f35498dca88c2e9b3c0ec497b83da4cefa41835ef9b2312a39  logs/nx/runs/container_gate/D11_L2_T.json
0b4b0eeeef28a67694ec136a30c8936447ee74ead3743ef7681a15e4b679bc14  logs/nx/runs/container_gate/D11_L2_T.mission.out
4091af716e95dd928e735dc681155fe10ed4a733c44ff18f46999d5988d846a1  logs/nx/runs/container_gate/D11_L2_T.progress.jsonl
65cbe63c61773643d3ae82656fc4e752246bf167447bb0b476031a01e9abb392  logs/nx/runs/container_gate/D11_L2_T.to_start.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/nx/runs/container_gate/D12_G1_C.cli_exited
0ba81c5d0e37204ff110d6e8b0d1c23583c7da7a7f1eec8a7c5f159cf4bd94f4  logs/nx/runs/container_gate/D12_G1_C.cmd
c622cd37189f9d2675db9a0307a6671ecbfd2e27d8d8d9390062b8cbc7597b33  logs/nx/runs/container_gate/D12_G1_C.cmd.json
cbbbcc9e9051c856cc589983bd2a59dd2c6f400f3ea2ff130c789ca4cc9f0102  logs/nx/runs/container_gate/D12_G1_C.json
dcbc09499b7c639de7e0753800a7ff63bdae38db344a9f5236cc48161e0e80ca  logs/nx/runs/container_gate/D12_G1_C.mission.out
163ace5dc4b78c5ff4f4bbe7974977e1f232356b3b8b723a5809ef9fb7f8514b  logs/nx/runs/container_gate/D12_G1_C.progress.jsonl
41aff6537d62bc6a43d52def767ae7f151c2f741007118180b66f73143e2a500  logs/nx/runs/container_gate/D12_G1_C.to_start.json
2aaaa895c1d43916fdd6bec475102605790a4d36932fcc87fd7db79248e36806  logs/nx/runs/container_gate/tf_logger_referent.out
4e5ae0c87eace70bbd8eda2f0ea074ef5e5b3c73573ac17c04ca64649e69b109  logs/nx/runs/container_gate/tf_referent.jsonl
68f578bf89a76ea5689a7d14dc9c3eb4338a04885878b9dc2a0bf77938343330  logs/nx/strafer_autonomy.log
df9de06ab4f6815af6167ad8c0ff4431029d16c5df38233bf0b65d7de5fd1973  logs/nx/strafer_inference.log
c1e3766e9530fd0232aba47854528d665f663460483ecc7a2883ccdb8e234e9b  logs/nx/strafer_navigation.log
9a8a693061dc569d91eb9148ebc8f9f8ee2d5d655120f4fa22f134b36485232a  logs/nx/strafer_sim_perception.log
520a2d7cf2d8e4c88428264e80961e6fd37bf8c1b29cc6114f304c6f83925d5e  logs/nx/strafer_slam.log
93638da45630ad5be4668187486685de4abab760b47ad54209db3b8507b50c2c  logs/nx/up_t0.txt
f4251f310ca6e72b353610beffe9a36564c417ee2af2b43af1ed6300dc4599ab  manifest.json
39a25b551747c9bccf3197fc2787bb056b0449a320ce4a3f7e17b911b601f15f  node/inference-referent-node.yml
52c9beed6bcec7392a6f625ee44b3cad8c8e6594b898924b5cb7564ea04fa9a5  node/subgoal_generator_node.main.py
c8085992c6b1492c117ae103591152615f0031f9f35b94d0bac2fc7f48322fb5  node/subgoal_generator_node.py
d45ae310388c4cfc4336de02e8f77af07ca025f0f9dc25de25986d82fed3daa0  posthoc_offset_band.py
46658be0f6bc50e3101f725a54073c3a64c98cad5210519592e3b629125bee96  posthoc_offset_band.txt
b34b9759c6e36bfc342ae6d583900afcb18c2a707e325bb1588f396aca1b8c1d  reach_terminal.json
ec1cea9e83eed8f56a2f91c1a222d2184a4fecd5a8bf4217dc64fd95a210733c  reach_terminal.stdout
ee2b1d9643a449feee3cd7fea7b15b130f592a32ca27408f994e0fdb0c218016  referent_arm.patch
99a470eb6307b7b8437c25bd6485e0066dd23074eb594e29e4be0a1b7c3469a1  session/d_chain.sh
91b20b582b81bef7638d14ec18afdcc877e86a88372b2999216c6b0ca77439e4  session/d_session.sh
7ae3cbbf44e3a0e4a70f4d77638058b8a5095b5e1bf8ef39996f66f63d5cba50  session/nx_up.sh
dbf31eb56f34ac4bd9963a1166fbe529bb3da3329390f18f04dcbcc107e2c1e6  session/stage0.sh
417bcc69cead7640131a17e18eb4fc32acf2c4a366c12df9afd8b0a6a8da546f  stage0/cyclone_pin_in_containers.txt
69b04393dedcb7e237416a98c985273e72d8f970d77bc89f2d618f37a4fe8da1  stage0/depth_topic_info.txt
ce6c21b3f9fb55d003984a50f38349f11695070b69b05adddcf4778e27f2b056  stage0/executor_checks.txt
2fc1e7152218299d3340bd1b2d01e32be6ea3f5068fb3da80d5dbe9a445b5b63  stage0/harness_sha256.txt
eafd124b1a6f8c217b0306afe90b658f262665809323838e772762070a1564f6  stage0/images.txt
e3cf55593ac1fee8b37b7fb2bbc7626bf4357f43acab3af81ad4c2810e499779  stage0/in_container_artifact_checks.txt
baae0276a7e6ac9127dc1032580e4a2247407d9bfb97d863679f814a529da02a  stage0/referent_node_mount.txt
2a007000c2318be829313475b9039e624ce5722f849c2a99ee23326253b664f8  stage0/service_health.txt
a4bae5e3609485c2186c1f87172c105aea649fcd5617945249cb278f9be39045  stage0/stage0.out
568badfda0e21e97e4b9dc8cfd52f20de4dbf7776ddb099b30db288cbda44f7f  tools/band.py
714c8e037ab2987ea3228a37734427d514b5fd72a95bef8e5f8c8e04f344002e  tools/planend.py
eb4c4f3eedcf8d96288b4f4e15ac86595286e62caf606d44744a8c740c74cfdd  tools/reach_terminal.py
```
