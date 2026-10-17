# Training-style goals on the sim-bridge lane (the scene axis), 2026-10-14

On the sim-bridge lane, v3 goes quiet short of the goal before the 0.30 m crossing on about 7 of 10
band entries. In its training env the rate is 2.94 % uncapped and 6.40 % capped
([`policy-time-to-goal-baseline-2026-10-11`](../policy-time-to-goal-baseline-2026-10-11/README.md) §9).
This record asks which side of the lane raises that rate:
- its pipeline (SLAM referent, body-velocity source, actuation tier, lookahead); or
- the scene and depth content at the goals the lane has used.

To separate them it keeps the pipeline and moves the goals to where training puts them.

**The series is void by its pre-registered clause.** One Kit boot produced four scored reaches of the
twelve the rule needs.
- During the fourth slot's transit, SLAM's `map→odom` drifted 1.35 m and 18° from where it started,
  peaking at 1.36 m just after. On the sim bridge odom is ground truth, so the drift is SLAM error.
- The fifth slot's transit then stalled at the room's west side on both attempts. Nav2's costmap
  showed free space at the stalled pose, and the transit's target, the spawn, sat in a lethal cell.
  That is consistent with a map built in the moving frame (inferred, §4).
- **The operator stopped the series after five slots** (§5). With `map→odom` about a metre off, later
  transits looked set to pin as S05's did, and a nudge (which the amendment allows) would not repair
  Nav2's map. Neither was tested.
- Stopping before the re-runs left fewer than 12 scored reaches. That voids the series by its clause:
  "fewer than 12 scored reaches or fewer than 3 goals among them, after the re-runs". What ran is
  deposited.

**No verdict on the scene axis follows.** Beside the void, descriptively:
- **The reaches.** All four scored reaches ended on the node's `SUCCEEDED`, at four different goals.
- **The stop-short.** Two of the four went quiet before the 0.30 m crossing, 0.396 m and 0.325 m from
  the goal.
- **The plan end.** It sat 0.030–0.053 m from the goal at band entry, the same as on the video set's
  goals.

Four band entries separate no two of the rule's cases. The pre-registration's operating
characteristics need twelve.

**What the session did establish:**
- **The lane already runs the generator v3 trained on.** It uses the enriched ProcRoom generator at the env's fixed seed, so every boot builds the same room. Computed after the run on this
  boot's training grid (`posthoc_video_goals_on_grid.py`): at the physical points the referent-offset
  A/B sent them to, the video set's G1 and L2 lie outside the room's free space, 1.682 m and 2.372 m from
  the nearest free cell centre. At their nominal map coordinates the distances are 1.718 m and
  2.475 m. L2 lies outside the training grid's 8 m extent altogether (room x −4.29 m against the
  grid's −4.0 m edge), so its "in free space: no" is an array-bounds decision, not a grid cell.
  Training never puts a goal there.
- **Training-style goals go through the executor's CLI path.**
  - Six goals were drawn inside the room by training's own rule (§2).
  - Every reach was drawn to start at the fixed spawn with its goal 45° to the left of the heading.
    S01–S03 began 0.07–0.21 m from it. S04 began 1.35 m from it, with the goal 49.7° to the left,
    because the frame moved (§4).
  - That is not training's start: training draws the robot's start cell and yaw at random.
  - It kept all four submitted commands in the forward-left template. All four compiled exactly, and
    none was refused.
- **Anchoring each goal in odom, mapped to the map frame at the reach, kept the goals physical through
  the frame's drift.** S04's goal was mapped under a frame 1.35 m and 18° off and moved at most 0.119 m
  during its reach. The starts were not kept physical (§4): S04 began 1.35 m from the spawn.

## 1. Setup

| | |
|---|---|
| question and rule | `PREREGISTRATION.md`, committed at `e6e0b61` (19:43:05 CDT) before the boot, with a time correction at `7317155`; `PREREGISTRATION_AMENDMENT.md` at `5100a79` (23:26:17) before the boot; the drawn goals at `cb480a5` (23:33:40) before the first reach |
| scene | `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, seed 42, headless, one boot through `tools/kit_boot_watchdog.sh` and the deposit-only wrapper `dgx/bridge_grid_dump.py`. It booted on its first attempt; `publish 30.00 Hz sim`, `frame_skip=3` |
| departure from the measurement as specified | the scene-axis measurement was specified on `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0`, the vanilla generator (1.0 m walls, no ceiling, difficulty 7 only), whose top-11-row depth sits at the 6 m clamp 58.9 % of the time against 7.3 % on the enriched one. It asks for a training-distribution scene, and v3 trained on the enriched one, so the session ran the lane's own enriched variant and moved the goals instead |
| node | v3 (`strafer_depth_subgoal_v3_999.onnx`, `c866bfd5…`); the deployed caps, 0.7841 m/s and 2.0933 rad/s, through the speed-cap A/B's mounted cap file; the subgoal generator as on `main` |
| robot stack | the five bridge-lane services, images `9f7d20758d5b`; SLAM key `lanetermA4`, fresh |
| planner / VLM | the speed-cap A/B's two service processes, from the main checkout at `6cd6fd7` (that record's `session1/logs/dgx/`) |
| reach | `tools/run_cli_one_start.sh`, the pinned `run_cli_one.sh` (`571b66b4…`) with one change (`tools/run_cli_one_start.diff`). It takes the start and the goal in the odom frame and maps each to the map frame with the latest `map→odom` sample of the session's ride-along: the start before the transit, the goal after it. A sample older than 5 s wall refuses the reach. The samples used are in each reach's `<label>.frames.json` |
| goals and order | six goals A–F (§2), drawn after the boot from its own room and committed before the first reach; order A B C D E F A B C D E F (`goals/manifest.json`) |
| session | 2026-10-14 23:33 CDT to 2026-10-15 00:35 CDT |

## 2. The goals

`dgx/bridge_grid_dump.py` saved the room's grid after the bridge's one reset (`goals/grid_dump.npz`,
sha256 `36929c68…`):
- the spawn at world (−1.65, −0.55), as on every earlier boot of this lane;
- the env origin at (0, 0);
- 200 BFS-reachable pool cells and 1565 free cells, spanning room x −2.15…1.95 m and y −2.45…2.45 m.

`a4_goals.py` (seed 20261014) applied training's sampler (`SubgoalCommand._resample_command`):
- **Candidates:** the pool cells at least 2.0 m from the robot (119), shuffled.
- **Acceptance:** each candidate is kept if `plan_path` plans to it from the spawn and it lies at least
  0.60 m from every goal already kept. The separation is the one addition to training's rule.
- **Result:** nine candidates were tried. Every one planned, and three were skipped for separation.
- **Start yaw:** each goal's start yaw puts the goal 45° to the left of the heading.

| goal | room (m) | odom (m) | from the spawn (m) | start yaw (odom) |
|---|---|---|---:|---:|
| A | (−0.25, 2.15) | (1.40, 2.70) | 3.04 | 17.59° |
| B | (1.45, −0.55) | (3.10, 0.00) | 3.10 | −45.00° |
| C | (0.85, −2.35) | (2.50, −1.80) | 3.08 | −80.75° |
| D | (1.45, 1.05) | (3.10, 1.60) | 3.49 | −17.70° |
| E | (0.05, −1.75) | (1.70, −1.20) | 2.08 | −80.22° |
| F | (1.95, −2.25) | (3.60, −1.70) | 3.98 | −70.28° |

## 3. The reaches

| reach | goal | result | band (s) | quiet stop (m from goal) | pre-crossing stop-short | plan end at band entry (m) | cmd speed before band (m/s) | goal moved during the reach (m) |
|---|---|---|---:|---|---|---:|---:|---:|
| S01 | A | `SUCCEEDED` | 2.27 | none in band | no | 0.043 | 0.519 | 0.050 |
| S02 | B | `SUCCEEDED` | 6.75 | 0.396 at +3.0 s | yes | 0.030 | 0.662 | 0.093 |
| S03 | C | `SUCCEEDED` | 2.33 | none in band | no | 0.053 | 0.592 | 0.043 |
| S04 | D | `SUCCEEDED` | 8.82 | 0.325 at +6.7 s | yes | 0.051 | 0.534 | 0.119 |
| S05 | E | void: transit failed both attempts | — | — | — | — | — | — |

- **Every scored reach is valid.** The CLI reported each as `succeeded`, and every join reads
  `node_result_executor_agrees`. Every dry-check matched, and no reach fell back to Nav2.
- **The caps held.** |vx| + |vy| reached 0.7841 m/s and no more.
- **The commands.** All four compiled to the forward-left template; S05's transit never reached the
  command. S01's was `move 2.062 meters forward and 2.140 meters left`.
- **The in-reach frame check** (the amendment's void, at 0.30 m) passed all four.

## 4. The frame and the void

**The frame** (the session's ride-along, `tf_a4.jsonl`):
- It started at (−0.002, +0.003, +0.1°) and settled after SLAM's warm-up at (+0.044, −0.006, +1.7°).
- During S03's transit it moved away, to as much as 0.78 m at 135.3 s. It then stepped 0.73 m and
  +10.2° back to (+0.033, +0.024, −0.5°) at 136.8 s sim (04:44:43 UTC).
- During S04's transit (171–207 s) it drifted to (−0.465, −1.262, +18.0°), 1.35 m. It peaked at
  (−0.506, −1.262, +17.8°), 1.36 m, at 211 s.
- It stepped 0.58 m and −16.3° at 261.9 s (05:00:37 UTC), and ended at (+0.002, −0.961, −1.7°).

**The goals stayed where training drew them.** The reach wrapper maps every goal through the latest
sample, so each goal went to its drawn odom point whatever the frame:
- S04's under (−0.465, −1.262, +18.0°);
- S02's under (+0.016, +0.057, −0.8°).

During the reaches themselves the frame moved each goal 0.043–0.119 m.

**The starts did not stay physical.** Each start was mapped before its transit.
- S04's frame moved 1.37 m during the transit. S04 therefore began at odom (0.86, 1.04, −35.8°),
  1.35 m from the spawn, and ran 2.31 m to D with the goal 49.7° to the left, not the drawn 3.49 m
  at 45°.
- S01–S03 began 0.07–0.21 m from the spawn (`start.ob` in each observer record).

**The transits did not.** The transit plans on Nav2's map, and that map was built in a frame that kept
moving.
- S05's start, the spawn, mapped to map (−0.723, −1.084).
- The retry stalled at map (−0.424, +0.031). Under the frame at that moment that is room
  (−2.11, +0.43), 0.17 m outside the training grid's inflated free space (nearest free cell centre
  (−1.95, +0.35)). The first attempt had ended at map (−0.010, +0.042), inside the free space.
- Nav2's costmap showed zero cost at the stalled pose (`logs/nx/S05_costmap_window.txt`). The
  transit logged its target in a lethal cell (`target_pooled_cost=100`, `S05_E_C.to_start.out`).
- Both attempts ran out the 150 s sim budget (1134.9 s and 1143.3 s wall), and nothing was submitted.

**The stop** was the operator's (§5).
- The chain was stopped at 00:34:56 CDT.
- The chain had already begun slot 6's arm switch, so the `inference` container was recreated once
  more. No slot-6 reach started.

**Across the evening's four boots of the lane** (this record and
[`lane-term-speedcap-ab-2026-10-14`](../lane-term-speedcap-ab-2026-10-14/README.md)):
- three saw `map→odom` move more than 0.4 m from its ride-along's first sample (1.495, 0.530 and
  1.362 m);
- the fourth stayed within 0.311 m of it, though between that record's frame reads its goal drift
  reached 0.45 m;
- the referent-offset A/B's boot, on 2026-10-11, stayed within 0.219 m and 6.5° of its ride-along's
  first sample over twelve reaches. Over its twelve observer records' `map_odom` it stayed within
  0.287 m and 4.93°.

These figures are computed after the run, in the speed-cap record's `posthoc_frame_excursion.txt`,
deposited at `3f1debf`.

This boot's rtabmap log has 31 rejected loop closures (`logs/nx/strafer_slam.log`, lines 250–630) and no
wrong-loop-closure warning.

Every reach this evening recreated the `inference` container before it started, to set the node's caps;
the referent A/B changed its arm with `ros2 param set` and did not. As the deposits show, the recreate
is not a candidate cause of the drift, except through the robot host's shared compute, which they do not
test. The speed-cap record (§7) gives the timing and the topics.

## 5. Deviations

- **The task id** (§1), stated in the pre-registration.
- **The operator stopped the series after five slots.** That judgment is the one in the summary, and
  it was not tested. No re-run was attempted.
- **The amendment's replay statement is wrong.** It says `frame_check_a4.py` flags one of the
  speed-cap A/B's reaches. It flags three: that record's first-session S05 (0.402 m) and S08
  (0.396 m), and B04 (0.327 m). The amendment had read five reaches. The check as run on this session
  is unaffected.
- **The pin detector missed S05.** It compared pose strings exactly, and the stalled pose's heading
  still changed by 0.1°. The stall was found by hand 38 minutes into the slot.
- **The grid wrapper's docstring gives the spawn quaternion's order wrong** (`DEPOSIT.md`). Nothing
  reads the field.

## 6. What is not claimed

- **Not a verdict on the scene axis, either way.** Two of four is no share the rule can read.
- **Not that training-style goals are easier or harder on the lane.** The four scored reaches cover four
  goals once each.
- **Not the cause of the SLAM drift.**

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `lane-term-scene-axis-2026-10-14/record-files/` |
| deposit commit | `30e034a70b9f1857c510a1a6e8432d3025e08842` |

The pre-registration and its tools are in `e6e0b61`, before the boot; the amendment in `5100a79`, before
the boot; the room's grid, the drawn goals and the stage-0 checks in `cb480a5`, before the first reach.
`DEPOSIT.md` lists every file, how each number above is re-derived, and the redactions. The runs are in
`76a489b`, and the post-hoc grid read of the video set's goals (the summary) was added at `30e034a`.
