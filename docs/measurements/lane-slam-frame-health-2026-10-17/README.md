# The lane's SLAM frame, 2026-10-17

On 2026-10-14 SLAM's `map→odom` on the sim-bridge lane moved more than 0.4 m from its first sample in three of four
boots, and the lane's terminal reads moved with it
([`lane-term-speedcap-ab-2026-10-14`](../lane-term-speedcap-ab-2026-10-14/README.md),
[`lane-term-scene-axis-2026-10-14`](../lane-term-scene-axis-2026-10-14/README.md)). This record measures the frame
itself.
- **Read in odom.** On the bridge odom is ground truth, so the frame is read against it, continuously.
- **Two rtabmap arms.** **A**, as deployed. **B**, with loop-closure acceptance off: `Rtabmap/LoopThr` 1.1,
  `RGBD/AggressiveLoopThr` 1.1 and `RGBD/ProximityBySpace` false. Each arm's values were checked in the running node's
  parameter dump before its reaches.
- **Six boots, arms alternating.** Twelve reaches per arm, each of four goals three times. The deployed inference
  node, unchanged. Goals and starts anchored in odom.
- **Terminal reads scored in odom,** beside the node's own map-frame result.

The reads, the decision rule and its operating characteristics were committed (`ea69464`) before the robot stack came up
(00:04:26 Z) and before the first reach (00:08:26 Z); F1's Kit log, which would date the bridge's launch, was lost with
the sim host's scratch (§7).

**The frame holds in neither arm. Arm A's failures are closure-driven: one accepted wrong closure, 142→16, rotated
F1's frame 86.84°. Arm B's 0.40–0.90 m drifts are the accumulated error of visual neighbour-link refinement
(`RGBD/NeighborLinkRefining` true, `Reg/Strategy` 0, `Vis/MinInliers` 15) replacing the bridge's ground-truth odometry
links. B does not isolate the bridge's publication, so the pre-registered both-fail branch does not follow** (§5).

B's "100 % of the motion in refinement-only iterations" is what rtabmap 0.23.7 produces by construction once closures
are off: the refinement branch is then the only remaining writer of the map correction. It is a manipulation check,
not a finding.

- **Arm A failed in all three boots** (F1, F3, F5): largest drift 5.140, 0.734 and 0.502 m, and nine steps of 0.5 m or
  more, seven of them in F1.
  - **F1's frame rotated 86.84° at 75.2 s sim.** The rtabmap iteration that accepted the global loop closure 142→16
    carries that frame: its frame matches the step's after-sample to 0.000 m.
  - **Only F1 rotated far.** Against their references, F3 and F5 stayed within 9.00° and 3.81°.
  - **A's frame motion by iteration class differs by boot.** In F1, 54.3 % came in global-closure iterations. In F3
    and F5, refinement-only iterations carried the most (60.5 % and 58.9 %).
- **Arm B failed too,** although no B segment accepted a global loop closure or a proximity detection.
  - **Drift.** Its largest drift was 0.404–0.898 m in the four segments with reaches, with two steps of 0.5 m or
    more.
  - **All of it in refinement iterations, by construction.** Both steps, every excursion above 0.30 m, and 100 % of
    the frame's motion fell in iterations whose only accepted change was a neighbour-link refinement. With closures
    off, that is what rtabmap 0.23.7 produces: a manipulation check. Its 875 iterations that accepted nothing moved
    the frame 0.0002 m in all.
  - **Per link (read after the runs),** the refined links' error was isotropic, a median 11.4 mm and 0.220°, and
    barely tied to the inlier count (§5).
  - **Without F1 the arms overlap.** F3 and F5 drifted 0.734 and 0.502 m against B's 0.404–0.898 m, and B spent more
    of its samples above 0.30 m (91.2 % and 86.6 % in F4 and F6, against 31.0 % and 15.5 % in F3 and F5). The
    comparison is descriptive; three boots per arm support no test.
- **The node's `SUCCEEDED` overstates arrival.** All 24 scored reaches `SUCCEEDED` on the node's live map frame. In
  odom, 7 of 24 never came within 0.30 m of the goal fixed at goal-sent: A 5 of 12, B 2 of 12 (§4).
- **The sim host lost power twice during the session,** with no cause logged in its journal, and F2 was relaunched
  twice (§7). Each relaunch and tool fix is recorded in an amendment committed before the reaches it governs.

## 1. Setup

- **The lane,** as the speed-cap A/B's second session.
  - **Sim host:** the bridge `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0` at seed 42, through the
    watchdog and the scene-axis session's grid-dump wrapper. Every boot whose grid check survives hashed
    `36929c68…`, the scene-axis boot's room: F1 (`goals/grid_dump_F1.npz`), and F2r–F6
    (`boots/<RUN>/dgx/grid_dump_sha256.txt`). F2's check was lost (§7). Its R1 reach stays scored on the seed-42 build,
    as `AMENDMENT_F2.md` provides, although the pre-registered check could not be shown. The planner and VLM services
    ran from `main` at `0a975ef`.
  - **Robot host:** the five bridge-lane services, without `base` or `perception`.
    - Images labelled `9f7d20758d5b`, and the v3 policy artifact.
    - The deployed inference node with no cap overlay: every node-up line read
      `vel_cap=(0.7841 m/s, 2.0933 rad/s) policy_loaded=True`.
    - The inference container was recreated before every reach, as on 2026-10-14.
    - Each reach went through the scene-axis session's odom-anchored `run_cli_one_start.sh`.
- **The arms.** Each arm's `rtabmap_params.yaml` was mounted read-only over the installed file through the same
  overlay. A mounted a byte-equal copy of the image's file. In every segment the host and container sha256 agreed,
  and `ros2 param dump /rtabmap/rtabmap` read the arm's values for seven keys (`PARAMS OK`;
  `boots/*/robot/logs/rtabmap_params_check.txt`).
  - **What B switches off.** From rtabmap 0.23.7's source:
    - a global closure needs a hypothesis ≥ `Rtabmap/LoopThr`, and the hypothesis is at most 1;
    - `RGBD/AggressiveLoopThr` can lower that threshold only when it is the smaller;
    - proximity detection by space runs only under `RGBD/ProximityBySpace`;
    - proximity detection by time is off in both arms (`RGBD/ProximityByTime` false, no intermediate nodes).

    B's manipulation check found no detection of either kind.
  - **What B leaves on.** Neighbour-link refining (`RGBD/NeighborLinkRefining` true) stays on in both arms. With
    `Reg/Strategy` 0 it is visual registration, not ICP as the file's comments say. Every accepted refinement
    recomputes the map correction, and so moves `map→odom`.
- **Boots and segments.** A *segment* is one bridge boot under one reference. Each boot is one segment, except F2:
  its two relaunches after the sim host's power losses add F2r and F2r2 (§7).

  | boot | arm | goal order |
  |---|---|---|
  | F1 | A | G1 R1 L2 R2 |
  | F2 → F2r → F2r2 | B | G1 R1 · L2 (cut) · L2 R2 |
  | F3 | A | R1 L2 R2 G1 |
  | F4 | B | R1 L2 R2 G1 |
  | F5 | A | L2 R2 G1 R1 |
  | F6 | B | L2 R2 G1 R1 |

  Each segment had a fresh bridge boot, a fresh SLAM scene token and a fresh map (rtabmap's first line
  `local map=1`).
- **The goals,** in odom.
  - **G1 and L2:** the goals of the [`goal-a-cli-video-2026-10-09`](../goal-a-cli-video-2026-10-09/README.md) set,
    at its coordinates, (−2.00, 2.25) and (−2.75, −2.25).
  - **R1 and R2:** the first two goals of the scene-axis session's sampler (training's rule) on F1's grid,
    (1.4, 2.7) and (3.1, 0.0).
  - **Starts:** G1 and L2 start from (0.075, −0.035), R1 and R2 from the spawn. Each start yaw puts the goal 45° to
    the left, so a reach whose transit holds the frame compiles through the forward-left template. The executor
    faces the goal before the policy drives.
  - **Mapping:** the wrapper maps the start into the map frame through the live `map→odom` before the transit, and
    the goal after it.
- **The reference.** The sim-bridge lane's `sim-perception` service spins the robot through 360° at bring-up
  (`donut_warmup` in `strafer_bringup/launch/sim_bridge_support.launch.py`, default on; about 11 s sim).
  - **The spin settles early.** The frame's last change before the reference came at 12.3–16.2 s sim in every
    segment.
  - **The reference sample** is the one taken once the spin's process had finished and the frame had held for
    10 s sim: 22.9–28.1 s sim.
  - **Warm-up offset.** From the first sample to the reference the frame moved 0.030–0.121 m and 0.25–3.10°.
- **The ride-alongs.**
  - `map→odom` at a nominal 20 Hz sim: 17.8–18.3 samples per sim second delivered.
  - One line per rtabmap iteration from `/rtabmap/info`, with the accepted closures, rejections and refinements.
    rtabmap writes `Loop/MapToOdom` only on closure iterations, so this ride-along also reads `map→odom` 0.5 s wall
    after each message.
- **The terminal reads** are scored in odom against the node's goal mapped into odom at goal-sent (§4).

## 2. The frame against each segment's reference (read 1)

- **Drift** is the largest displacement, over the four goals, of a goal's map point as the frame moves it. It is the
  speed-cap A/B's frame-gate quantity over four goals instead of two, so it is stricter.
- **A step** is a move of 0.5 m or more, between consecutive samples, in where the frame puts any of the four goals:
  the largest over the goals of |T_s⁻¹g_k − T_{s−1}⁻¹g_k|. It is not the change in the drift value; F4's 0.666 m step
  changed drift by 0.227 m.
- **The window** runs from the reference to the segment's last node result. F2's and F2r's end at the sim's last
  sample, before the power losses.

| segment | arm | window (s sim) | largest drift (m) | share > 0.30 m | steps ≥ 0.5 m | holds | G1 and L2 only: largest drift (m), share | largest translation (m), \|yaw\| (°) vs reference | global closures | proximity | refinements | nodes added | distance (m) |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| F1 | A | 197.2 | 5.140 | 68.6 % | 7 | no | 5.140, 68.6 % | 0.950, 88.77 | 21 | 23 | 162 | 394 | 46.0 |
| F3 | A | 164.3 | 0.734 | 31.0 % | 1 | no | 0.692, 28.1 % | 0.400, 9.00 | 18 | 31 | 131 | 328 | 44.5 |
| F5 | A | 195.0 | 0.502 | 15.5 % | 1 | no | 0.502, 14.4 % | 0.374, 3.81 | 43 | 42 | 159 | 389 | 45.3 |
| F2 | B | 111.4 | 0.404 | 2.2 % | 0 | no | 0.404, 2.2 % | 0.236, 3.93 | 0 | 0 | 80 | 222 | 23.3 |
| F2r | B | 14.5 | 0.172 | 0.0 % | 0 | yes | 0.172, 0.0 % | 0.115, 1.87 | 0 | 0 | 11 | 28 | 0.3 |
| F2r2 | B | 99.6 | 0.652 | 8.9 % | 1 | no | 0.652, 8.9 % | 0.267, 7.25 | 0 | 0 | 71 | 198 | 20.5 |
| F4 | B | 164.1 | 0.898 | 91.2 % | 1 | no | 0.898, 91.2 % | 0.464, 8.84 | 0 | 0 | 134 | 327 | 46.2 |
| F6 | B | 274.8 | 0.663 | 86.6 % | 0 | no | 0.663, 78.5 % | 0.449, 8.21 | 0 | 0 | 152 | 548 | 44.3 |

Notes on the columns:
- **Closures, proximity detections and refinements** count the window's iterations that accepted one, and one
  iteration can count in more than one column. F1 has 21 global closures and 23 proximity detections, in 27
  iterations with either.
- **Distance** is rtabmap's own `Memory/Distance_travelled` over the window.
- **F2r** ran 14.5 s sim past its reference. Its first reach was cut by the second power loss before its goal was
  sent.
- **The G1-and-L2-only drift** is the like-for-like comparison with the 2026-10-14 sessions, whose frame gate read
  those two goals.

## 3. What moved the frame (read 2)

- **Each step's iteration.** Each step is attributed to an rtabmap iteration: among those whose message arrived in
  the 6 s wall before the step's after-sample, the one whose frame (read 0.5 s wall after its message) lies closest
  to the after-sample. Ten of the eleven steps match their attributed iteration to 0.000 m.
  - **The eleventh,** F1's 0.785 m at 164.7 s, is attributed to iteration 320 at 0.785 m (its before-sample). Its own
    iteration, closure 321→19, sent its message 0.11 s wall after the step's sample, outside the window, and that
    iteration's frame matches the after-sample to 0.000 m.
  - **Arm A's nine steps** fell in global-closure iterations (six) or proximity iterations (three), and each has an
    accepted closure within 2 s sim. In order: F1 75.2 s (away from the reference, 4.919 m), 106.2 (toward, 0.936),
    163.3 (toward, 4.372), 164.7 (away, 0.785), 165.2 (away, 0.579), 207.3 (away, 0.805), 211.3 (toward, 1.924); F3
    131.9 (toward, 0.575); F5 124.8 (toward, 0.528). No sample gap exceeded 0.6 s sim.
  - **F1's path.** F1's 86.84° step at 75.2 s is closure 142→16, 0.02 s before the step's sample.
    - The frame came back to 0.117 m of its reference at 163.3 s (closure 318→18).
    - It moved away again in steps of 0.785 m and 0.579 m at 164.7 and 165.2 s (closures 321→19 and 322→20), and
      0.805 m at 207.3 s (proximity 406→62).
    - It came back to 0.103 m at 211.3 s (proximity 414→4).
  - **Arm B's two steps,** 0.513 m (F2r2, 115.2 s, away) and 0.666 m (F4, 143.8 s, toward), each fall in an iteration
    that accepted only a neighbour-link refinement.
- **The frame's motion by iteration class.** Each iteration's change of the frame is measured at the goals as a step
  is, then summed per segment; the sum is a path length, not net drift.
  - The classes are exclusive, in the order global closure, proximity (no global closure), refinement only,
    nothing accepted.
  - An iteration that accepted both a global closure and a proximity detection counts as a global closure, so A's
    proximity class is smaller than §2's proximity counts.

  | segment | arm | global closure | proximity (no global closure) | refinement only | nothing accepted | total |
  |---|---|---:|---:|---:|---:|---:|
  | F1 | A | 21 it., 12.09 m (54.3 %) | 6 it., 3.21 m (14.4 %) | 138 it., 6.95 m (31.2 %) | 229 it., 0.00 m | 22.25 m |
  | F3 | A | 18 it., 1.08 m (12.3 %) | 13 it., 2.38 m (27.2 %) | 105 it., 5.30 m (60.5 %) | 192 it., 0.00 m | 8.76 m |
  | F5 | A | 43 it., 3.24 m (35.9 %) | 4 it., 0.47 m (5.2 %) | 115 it., 5.32 m (58.9 %) | 227 it., 0.00 m | 9.03 m |
  | A pooled | | 82 it., 16.41 m (41.0 %) | 23 it., 6.05 m (15.1 %) | 358 it., 17.58 m (43.9 %) | 648 it., 0.0064 m | 40.04 m |
  | F2, F2r, F2r2, F4, F6 | B | — | — | 448 it., 19.32 m (100 %) | 875 it., 0.0002 m | 19.32 m |

  - **Inside each excursion above 0.30 m.** B had five runs above 0.30 m, every one 100 % refinement-only: F2 at
    42.6–43.6 s and 110.1–111.6 s, F2r2 at 115.2–123.9 s, F4 at 39.3–189.1 s (6.86 m) and F6 at 62.8–299.8 s
    (5.20 m).
  - **A's excursions.** A had twelve. F1's two, peaking at 5.140 m and 1.882 m, were mostly global-closure motion
    (60 % and 69 %). The onset, end, peak and class split of each are in `analysis/frame_health_<RUN>.txt`.
  - **Iterations that accepted nothing** moved the frame 0.0064 m in all over A's 648 and 0.0002 m over B's 875,
    with no single move above 0.001 m. On this lane, then, the frame moved only in iterations that accepted a closure
    or a refinement. rtabmap can also re-optimise on an iteration that retrieves nodes, reduces the graph, or adds a
    prior, gravity or landmark link; none of these moved the frame here.
- **Where the graph strained (the pre-registered strain read).** rtabmap created nodes 1–24 before the warm-up spin
  ended, in every segment.
  - **Closures.** A closure's largest-error edges lay among those nodes in 10 of F1's 27 closure iterations (all on
    edge 18→19), 0 of F3's 31 and 0 of F5's 47.
  - **Steps.** They did for 3 of A's nine steps (F1 at 106.2, 163.3 and 164.7 s).
  - **The 86.84° step's closure,** 142→16, strains edge 84→85, outside the spin.
- **Read after the runs (`arm_summary.py`, descriptive).** A closure's *matched* node was a spin node in 18 of F1's 27
  closure iterations, 20 of F3's 31 and 20 of F5's 47. That includes F1's 86.84° step (node 16), and its return at
  163.3 s (node 18). Every reach's start anchor lies within 0.09 m of the spawn, where those nodes sit. This read
  therefore cannot separate the spin from the start point that every reach returns to.
- **Rejections.** No "wrong loop closure … after graph optimization" rejection was logged in any segment; the
  2026-10-14 speed-cap A/B's first session logged 68. Registration rejections were logged in arm A only (7, 16 and 14
  in F1, F3 and F5). Arm B had none, by construction: no hypothesis passes B's threshold, so B never attempts a
  closure's registration.
  - **Near three of A's steps.** One rejected hypothesis falls within 2 s sim of each of three steps (F1 at 211.3 s, F3
    at 131.9 s, F5 at 124.8 s), seen both as the info stream's flag and as its registration log line. Each of those
    steps also has accepted closures within 2 s, and its attributed iteration accepted one.
  - **Chance.** In each A window, 10.9–17.7 % of samples lie within 2 s sim of a rejection, so some steps would sit
    near one by chance.
  - **Mechanism.** A closure rejected at registration, the only kind logged here, never adds a link, so the
    rejection alone does not move the frame.
- **Other WARN lines,** listed apart: one `Memory.cpp:6380` per segment at start-up, and one `Transform.cpp:584` in
  F2r, F3 and F4. All fall outside the windows.

## 4. The terminal reads in odom (reads 3 and 4)

Per reach, the node's map-frame goal is mapped into odom once, at goal-sent, and the robot is read from the odom series
from goal-sent to the node's result. The definitions are the lane-term sessions':
- band entry: within 0.42 m;
- crossing: within 0.30 m;
- band time: from band entry to the node's result;
- pre-crossing stop-short: 10 consecutive `/cmd_vel` ticks below 0.05 m/s in the band, before the crossing or with no
  crossing.

The live reads use the same definitions on the node's map frame.

| arm | scored | node `SUCCEEDED` | within 0.30 m in odom at the result | never within 0.30 m in odom | pre-crossing stop-shorts, odom (live) | band median, odom (live) | largest in-reach goal excursion |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 12 | 7 | 5 | 6 of 12 (7 of 12) | 3.38 s (3.34 s) | 0.290 m |
| B | 12 | 12 | 10 | 2 | 4 of 12 (6 of 12) | 2.75 s (1.68 s) | 0.183 m |

| reach | node: distance at result, live (m) | odom: at result (m) | odom: closest (m) | band, odom (live) (s) | stop-short, odom (quiet stop, m); live | goal excursion (m) | goal at sent vs anchor (m) | start vs anchor (m) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| F1_S1_G1_A | 0.299 | 0.368 | 0.368 | 1.94 (2.65) | no; no | 0.102 | 0.013 | 0.069 |
| F1_S2_R1_A_R | 0.299 | 0.208 | 0.208 | 2.43 (1.57) | no; no | 0.092 | 0.022 | 0.087 |
| F1_S3_L2_A | 0.299 | 0.348 | 0.348 | 3.78 (4.22) | yes (0.399); yes | 0.070 | 0.035 | 0.369 |
| F1_S4_R2_A | 0.300 | 0.285 | 0.285 | 5.40 (5.18) | yes (0.338); yes | 0.023 | 0.000 | 0.649 |
| F3_S1_R1_A | 0.300 | 0.272 | 0.272 | 10.74 (9.92) | yes (0.393); yes | 0.051 | 0.007 | 0.003 |
| F3_S2_L2_A | 0.286 | 0.313 | 0.313 | 1.38 (1.38) | no; no | 0.078 | 0.066 | 0.118 |
| F3_S3_R2_A | 0.300 | 0.297 | 0.297 | 2.98 (2.87) | no; no | 0.050 | 0.006 | 0.129 |
| F3_S4_G1_A | 0.300 | 0.354 | 0.313 | 7.95 (7.84) | yes (0.320); yes | 0.290 | 0.043 | 0.030 |
| F5_S1_L2_A | 0.300 | 0.324 | 0.324 | 1.58 (1.58) | yes (0.343); yes | 0.087 | 0.047 | 0.061 |
| F5_S2_R2_A | 0.300 | 0.273 | 0.273 | 4.57 (3.94) | yes (0.337); yes | 0.209 | 0.015 | 0.112 |
| F5_S3_G1_A | 0.284 | 0.218 | 0.216 | 4.24 (3.82) | no; yes | 0.167 | 0.042 | 0.075 |
| F5_S4_R1_A | 0.300 | 0.170 | 0.170 | 2.88 (1.28) | no; no | 0.242 | 0.070 | 0.225 |
| F2_S1_G1_B | 0.298 | 0.313 | 0.313 | 0.40 (0.29) | no; no | 0.093 | 0.028 | 0.037 |
| F2_S2_R1_B | 0.300 | 0.272 | 0.272 | 1.98 (1.67) | no; no | 0.083 | 0.072 | 0.156 |
| F2r2_S3_L2_B | 0.300 | 0.360 | 0.360 | 1.17 (1.68) | yes (0.373); yes | 0.094 | 0.020 | 0.108 |
| F2r2_S4_R2_B | 0.300 | 0.227 | 0.227 | 9.42 (9.01) | no; no | 0.118 | 0.023 | 0.050 |
| F4_S1_R1_B | 0.300 | 0.145 | 0.145 | 2.74 (1.49) | no; no | 0.183 | 0.016 | 0.045 |
| F4_S2_L2_B | 0.293 | 0.282 | 0.282 | 6.06 (5.75) | yes (0.354); yes | 0.052 | 0.021 | 0.270 |
| F4_S3_R2_B | 0.300 | 0.283 | 0.283 | 6.75 (6.96) | yes (0.380); yes | 0.089 | 0.019 | 0.407 |
| F4_S4_G1_B | 0.299 | 0.164 | 0.164 | 1.12 (0.80) | no; no | 0.159 | 0.006 | 0.305 |
| F6_S1_L2_B | 0.299 | 0.264 | 0.264 | 1.89 (1.58) | no; yes | 0.045 | 0.017 | 0.061 |
| F6_S2_R2_B | 0.300 | 0.261 | 0.261 | 10.06 (8.78) | no; yes | 0.067 | 0.030 | 0.157 |
| F6_S3_G1_B | 0.299 | 0.281 | 0.281 | 8.74 (9.38) | yes (0.321); yes | 0.085 | 0.021 | 0.074 |
| F6_S4_R1_B | 0.300 | 0.171 | 0.171 | 2.76 (1.62) | no; no | 0.167 | 0.010 | 0.275 |

- **Every node result is `SUCCEEDED`,** at 0.284–0.300 m on the live frame. In odom the same results lie
  0.145–0.368 m from the goal fixed at goal-sent.
- **The gap is the frame's motion during the reach.** The odom fixed point is the node's own map goal at goal-sent.
  At the result, the live and odom distances therefore differ by at most how far the frame has carried the goal
  since then. In all 24 reaches the difference stays within the reach's in-reach goal excursion (largest
  difference 0.155 m, F4_S1_R1_B, against its 0.183 m excursion). The largest excursions, 0.290 m in A and 0.183 m in
  B, approach the 0.30 m arrival radius.
- **Odom anchoring kept the goals, not the starts.** Every goal at goal-sent lay within 0.072 m of its anchor. The
  starts lay up to 0.649 m from theirs (F1_S4_R2_A), because a frame that moves during the transit moves the
  transit's target.
- **Not scored:**
  - `F1_S2_R1_A`, refused at the dry-check (`plancheck_rc=1`). F1's 86.84° step came during its transit, so its start
    was displaced, and its command left the forward-left template and was mis-signed. Re-run at the end of F1 as
    `F1_S2_R1_A_R`, scored.
  - `F2_S3_L2_B` and `F2r_S3_L2_B`, cut by the sim host's power losses.
- **No Nav2 fallback, no pins.** No reach fell back to Nav2, and none pinned. For F2, whose chain log was lost, the
  Nav2 check is its autonomy container log: no `falling back to nav2` line.

## 5. The rule's outcome

The frame holds in an arm if every segment of that arm has a largest drift of 0.30 m or less and no step of 0.5 m or
more, over at least 12 scored reaches in the arm.
- **Arm A:** 12 scored reaches, and it fails in all three boots.
- **Arm B:** 12 scored reaches only with F2r2's two.
  - **The relaunch.** F2r2 is a second relaunch of F2, beyond the pre-registered one relaunch per boot. Under the
    void condition (fewer than 12 scored reaches after one relaunch per boot), B would have 10 and its outcome would be
    void, so no branch would apply. `AMENDMENT_CRASH.md`, committed at `4e7e324` before F2r2's first reach, admits
    the second relaunch because both interruptions were the sim host's power losses. B's twelve scored reaches rest
    on that amendment.
  - **B's failure does not rest on it.** F4 (0.898 m, one step) and F6 (0.663 m) each fail on their own, and F2
    fails through its F2 (0.404 m) and F2r2 (0.652 m, one step) segments. F2r's "yes" is a 14.5 s segment whose only
    reach was cut before its goal was sent.

**The conclusion.** The frame holds in neither arm.
- **Arm A's failures are closure-driven.** One accepted wrong closure, 142→16, rotated F1's frame 86.84°.
- **Arm B's 0.40–0.90 m drifts are the accumulated error of visual neighbour-link refinement**
  (`RGBD/NeighborLinkRefining` true, `Reg/Strategy` 0, `Vis/MinInliers` 15) replacing the bridge's ground-truth
  odometry links.
- **The both-fail branch does not follow.** Its premise was that an arm without loop closure isolates the bridge's
  publication. B does not: refinement still writes the map correction.
- **B's 100 % is a manipulation check.** That 100 % of its motion falls in refinement-only iterations is what
  rtabmap 0.23.7 produces by construction once closures are off, because the refinement branch is the only remaining
  writer of the map correction.

**The per-link error (read after the runs, not pre-registered; `posthoc_link_error.py`,
`analysis/posthoc_link_error.txt`).**
- **The method.** For each refinement-only link between consecutive nodes, over each segment's whole boot, the link
  error in the new node's body frame is E = odom_new⁻¹ · (T[j−1]⁻¹ · T[j]) · odom_new. Here T is the frame after each
  iteration, and odom_new = T[j]⁻¹ · `Loop/MapToBase`.
- **Validation.** odom_new matches the observer records' `pose_odom` to a median 0.2 mm (95th percentile 1.9 mm,
  max 5.8 mm) and 0.023° over 647 iterations.
- **Per-link error.** Arm B, 545 links: median 11.4 mm and 0.220° (95th percentile 48.4 mm and 1.232°). Arm A, 417
  links: 13.3 mm and 0.230°.
- **Isotropic.** In B the body-frame spreads are 16.3 mm along x and 16.8 mm along y, with means near −1 mm.
- **Barely tied to inliers in B.** The Spearman correlation with `NeighborLinkRefining/Inliers` (median 32, range
  15–75) is −0.03 for translation and −0.15 for rotation; in A it is −0.18 and −0.34.
- **Mildly tied to the link's own motion:** +0.25 for translation and +0.24 for rotation, in B.
- **A stamp-lag signature, if images lagged odometry by τ,** would make E ≈ −τ · Δv, with Δv the change of
  body-frame velocity across the link. The fitted slope gives τ = 38 ms in B (95 % 21–55 ms; R² 0.018). The lag term
  carries 8.1 % of B's summed error. In A, τ = 7 ms (−16 to +31 ms) and the share is 1.4 %.
- **What that bounds.** It puts a stamp lag's share of B's error at under 13 %. It does not show the lag to be under
  one 33 ms camera frame: B's estimate lies between about two thirds of a frame and five thirds.

**The next measurements (a proposal).**
- **Not B with refinement off.** With closures already off there is then no refinement and no optimisation
  trigger, so `_mapCorrection` never changes. The frame would be constant by construction, which is not a measurement.
- **The candidate arms build on the deployed configuration:**
  - **A′:** `RGBD/NeighborLinkRefining` false, with closures on. This is the deploy candidate.
  - **A″:** refinement on, with `Vis/MinInliers` raised (for example to 40).

  Both need no new tool; only the mounted file changes.
- **The bridge stamp check** of its camera, odometry and `/clock` is a static, lower-priority item. The read above
  already bounds a lag's share of B's error at under 13 %, at a lag estimated at 21–55 ms.

**The arm-A spin pattern is descriptive, and not one of the rule's branches.**
- **The large rotations.** F1's two large rotations (+86.84° at 75.2 s, −71.57° at 163.3 s) came in iterations that
  accepted closures matched to spin nodes 16 and 18.
- **The strain read.** By the pre-registered strain read, 10 of A's 105 closure iterations strained edges among the
  spin's nodes, all of them in F1. The read's condition, that most of A's steps follow such closures, is not met:
  3 of 9.
- **On the robot.** The robot's container stack does not start the spin. On this lane it comes from the sim-bridge
  `sim-perception` service, and the native bring-up (`strafer_bringup/launch/navigation.launch.py`) defaults
  `donut_warmup` to false. A loop-closure finding from this lane should therefore be shown without the spin before
  it changes the robot's configuration.

## 6. The real robot

On the robot there is no ground-truth odom: the frame is the robot's only estimate of where it is, so the frame
question becomes the localisation question. G-real's pre-registration should therefore carry SLAM health as a scored
void, not an unscored cause:
- **The read.** `map→odom` against the session's settled reference, by this record's definition, read continuously by
  a ride-along.
- **The events.** The steps, with their times, and the loop-closure log, aligned to them through `/rtabmap/info`.
- **The void,** stated before the first reach.

G-real's scored void is two counts:
- the count of `map→odom` steps of 0.5 m or more;
- the loop-closure log: accepted closures, rejections and wrong-closure warnings.

Each reach's goal excursion is reported beside them. A drift threshold on wheel odometry is not set from this record.
The terminal band is scored descriptively against the live frame, with each reach's frame excursion beside it.

## 7. Deviations, interruptions and defects

- **The pre-registered departures from the speed-cap A/B's second session** (`PREREGISTRATION.md`, *Departures*):
  - goals and starts anchored in odom;
  - start yaws that put the goal 45° left;
  - the `/rtabmap/info` ride-along, and the `map→odom` ride-along at 20 Hz;
  - a SIGINT stop of `strafer_slam` at each segment's end;
  - arm A mounting a byte-equal copy of its file;
  - no SLAM-step void inside a band, and no nudge;
  - the shared robot host.

  Every segment's SIGINT stop saved rtabmap's database (46–272 MB) and finished cleanly, with no launch escalation
  (`boots/*/robot/logs/slam_stop.txt`). The databases stay on the robot host; their digests are in `DEPOSIT.md`.
- **A tool defect in the reach launch** (`AMENDMENT_F1.md`). The launch backgrounded its whole `&&` list, so the
  start command's ssh did not return.
  - F1's chain stopped after its first reach, as its rules provide, while the reach itself ran normally.
  - The launch was fixed, and F1 resumed under its own reference.
  - The 2026-10-14 sessions' script had the same construct; their chain did not check the start command's return
    code.
- **The sim host lost power twice,** needing a manual power-on each time (`AMENDMENT_CRASH.md`).
  - **When.** At about 00:54:55Z, 18–21 min into F2's bridge (its launch time was lost); and at about 01:51:19Z,
    6 min into F2r's bridge.
  - **No logged cause.** Neither boot's journal shows a shutdown, or a thermal, power, GPU (Xid), out-of-memory or
    panic message. Each journal ends 1–5 min before the loss, and `last` marks both sessions `crash`
    (`host/journal_tail_boot-2.txt`, `host/journal_tail_boot-1.txt`, `host/last_x.txt`).
  - **The first loss was misread at the time.** `AMENDMENT_F2.md`, written before the second loss, attributes the
    stop to the end of the session that drove the campaign; `AMENDMENT_CRASH.md` corrects it.
  - **What it cost.** The loss cleared `/tmp`, which held F1's and F2's sim-host logs. F1's chain log is transcribed
    from the console output captured while it ran (`boots/F1/dgx/chain_log_reconstructed.txt`, with its uncaptured
    lines named). F2's sim-host logs, including its grid-hash check, are lost.
- **What changed after each loss.**
  - **After the first:** every sim-host process ran as a systemd user unit, with its logs in a persistent directory
    (`AMENDMENT_F2.md`). That answered the misreading and does not bear on a power loss.
  - **After the second:** a 1 Hz telemetry log ran from 02:06:42Z to the end of the session (`AMENDMENT_CRASH.md`).
    Over its 2 h 23 min the GPU peaked at 60 °C and 73.6 W and the ACPI zones at 67.3 °C, with no clock-event reason
    active.
  - **The result:** no third loss occurred. The telemetry covers neither loss.
- **F2 was relaunched twice,** against the pre-registered one relaunch per boot, because both interruptions were the
  sim host's (§5). F2 kept slots 1–2, F2r scored no reach, and F2r2 ran slots 3–4 under a fresh map and reference.
- **The robot host was shared** with another test campaign, by agreement.
  - **Before this session's first bring-up,** that campaign's compose containers ran on ROS domain 87 and were removed
    by 19:40:12Z.
  - **During this session** it ran network-less (`--network none`) containers and host-network test runners on ROS
    domain 0; this session ran on domain 42. At one load snapshot (F1's re-run) an unnamed container of that campaign
    used 98 % of a core.
  - **Its container timeline** is deposited (`host/jetson_session_timeline.txt`).
- **The wrapper's join** reads `STALE_RECORD` on every reach, as on earlier records using this wrapper. The scorers
  read each reach's observer record directly.

## 8. What is not claimed

- **No test between the arms.** Three boots per arm cannot support one; the comparisons above are descriptive.
- **No full cause for refinement's link errors.** The post-hoc per-link read finds them isotropic, of about 1 cm and
  0.2°, and barely tied to the inlier count. A stamp lag carries at most a small share of them. What produces the
  rest of the visual registration error is not shown.
- **The spin is not shown to cause anything.** It is not established that the spin, rather than the start point,
  explains arm A's closures to spin nodes, or that either causes F1's wrong closure.
- **The two power losses are unexplained.**
- **The policy is not measured.** The terminal reads describe this lane under each arm, not the policy.

## Evidence — deposit

`Sim2RealLab-Artifacts/lane-slam-frame-health-2026-10-17/record-files/`.
- **Commits.**
  - The pre-registration at `ea69464`, before the robot stack came up and before the first reach.
  - The goals at `514bdb8`, before the first reach.
  - The amendments at `849595a`, `01dd2a5` and `4e7e324`.
  - The runs at `f689d6d`.
  - The deposit's dated corrections at `c854a8f` (2026-10-18) and `715c61e` (2026-10-19), each described in
    `DEPOSIT.md`.
- **Changes after the pre-registration.** `frame_health.py` is unchanged from `ea69464`. `odom_terminal.py` differs by
  one not-scored row label, with no read changed (`DEPOSIT.md`). `arm_summary.py` and `reproduce.sh` were written
  after the runs.
- **`DEPOSIT.md`** lists every file's sha256 and the restore command.
- **Reproduce** from the restored record directory:

      bash reproduce.sh        # analysis/frame_health_<RUN>.{txt,json}, analysis/odom_terminal.{txt,json}, analysis/arm_summary.txt

  It regenerates all 22 analysis files byte for byte from a clean copy; `posthoc_link_error.txt` is the post-hoc
  per-link read.
