# The lane's goals outside the room, 2026-10-23

*Post hoc and descriptive. Nothing here was pre-registered, and no reach was run for this record. A read-only audit
ran on the sim host on 2026-10-23, 20:29–21:33 CDT, while the
[`lane-rtabmap-a3-2026-10-22`](../lane-rtabmap-a3-2026-10-22/README.md) campaign was running; no step of that
campaign cites it.*

**The question.** On 2026-10-23 a transit pinned at the room's doorway. Do the sim-bridge lane's goals lie outside the
room, and if so, what does that mean for SLAM, for the policy, and for the records that used them?

**The answer.**
- **Three of the lane's goals lie outside the room.** The lane's room is the ProcRoom-Enriched room at seed 42. G1 and
  L1 lie 1.18–1.20 m beyond its west wall, and L2 1.93–1.95 m beyond it, off the 8 m training grid as well. G1 and L2
  are the pair the lane has used since 2026-10-09; L1 ran on 2026-09-25 and 2026-10-03. R1, R2, the 2026-09-25 gate's
  R-goals as listed, the scene-axis goals and every nominal start lie inside.
- **How they came to be outside.** G1, L1 and L2 are the 2026-08-17 gate's M1, M4 and M5, reused as map coordinates
  relative to the spawn. Between the 2026-08-22 probe and the post-flip grid dump the seed-42 spawn moved 2.33 m.
  - G1 and L1 have lain outside since that move, which surfaced at the Isaac Lab pair flip.
  - **L2 lay outside before it too:** the 2026-08-17 gate already had at least one goal outside the room (L2; M6
    depends on the room's size).
  - Why the spawn moved is not documented.
- **The share of the evidence.** 56 of the 75 post-flip v3 lane reaches before 2026-10-23 went to goals outside the
  room. With the 2026-10-23 session's counted reaches, 69 of 100 did.
- **This was on record before.** Both 2026-10-14 records say the goals lie outside the room. What is new here is how
  they came to be there, how SLAM's failures split by location, and what follows for the records.
- **The exterior is repetitive, not short of features.** Outside, rtabmap extracts more words per node than inside (a
  median of 390.5 against 132). Facing outward, the camera sees one periodic grid plane under a uniform sky, so the
  hazard is perceptual aliasing (the reading; §9).
- **SLAM's two failure modes split by location.**
  - **Metre-scale wrong closures.** In the reprocessing record's matrix, every deployed-set and A′ run with a
    drift-growing step of 1 m or more took its first such step with the robot beyond the west wall's centre line (57
    of 57; A″ and A‴ take none). The closures matched doorway and exterior views to start-point views that look out
    through the doorway. F1's 86.84° closure 142→16 had 33 inliers.
  - **Smaller steps and refinement drift are not concentrated outside.** The deployed set's drift-growing steps of
    0.5–1 m in the reprocessing matrix came mostly inside (13 of 16 under 0.23.7, 12 of 15 under 0.23.13), and per
    metre refined links err less outside (§5). The scene-axis session, which never left the room, moved its frame
    1.36 m.
  - **Failed transits stall at or near the doorway,** the routes' narrowest passage. Each of them came with the frame
    already moved, but the doorway has also pinned a transit whose frame had not moved (§5).
- **The policy.** Training never puts the robot, the goal or a subgoal outside a room (`proc_room.py` lines 575–591).
  At G1, L1 and L2 the policy is outside its training distribution.
  - At the node level it still reached every such goal it was sent to (48 of 48): out of distribution did not mean
    failed.
  - The lane's policy-side findings at those goals are confounded with being outside, and several were read almost
    entirely there.
  - The G-sim v3 PASS is unsupported on inside reaches, not overturned. Its inside evidence is R1 reached and R2
    missed.
- **What changes.** Eight earlier records carry dated lines of 2026-10-29 that points here: six corrections, and
  notes in the video and scene-axis records, which were already right (§7).
- **What is proposed:**
  - a lane preflight;
  - an inside-room goal set drawn by training's rule;
  - the next lane session on that set under A‴;
  - an inside-room acceptance set for v4.

## 1. Method

- **Four investigations, each re-derived by an independent verifier.** Each verifier recomputed the figures with its
  own code, then gave every claim a verdict ("holds" or "holds with correction") and listed what the investigation
  missed. The four:
  - `slam_by_location`: where SLAM's error arises, against the room;
  - `camera_by_location`: what the camera sees and how many features rtabmap extracts, by location;
  - `records_survey`: which records used outside goals, and how;
  - `training_and_geometry`: the room's geometry, and whether training ever goes outside a room.

  Their summaries, tables, verdicts and corrections are in `wf_digest.txt`, and their full output is
  `workflow1_exterior_audit.json`. `VERIFICATION.md` carries every correction and every missed item. This record states
  the findings with those corrections applied, and recomputes each figure it states.
- **Read-only.** The audit read the 2026-10-17 rtabmap databases, the reprocessing record's 770 matrix runs and its
  addendum's 180, the dumped training grid, and the earlier records and their deposits. It ran no bridge, container
  or robot-side process, and no reach. Of the 2026-10-23 session it read only B3's dumped grid, the grid hashes of
  B1–B3 and B2's `chain.log` FRAME lines. This record's count of that session's reaches (§3) also reads its
  `analysis/odom_terminal.json` (`provenance.md`).
- **Recomputed for this record.** Every figure in §§2–7 was recomputed from the deposits, by independent code or by
  rerunning a copy of the deposited script, before it was written here (`recompute_2026-10-29/`). Four kinds are marked
  where they appear instead: figures quoted from another record, §2's readings quoted from
  `wf_training_and_geometry/notes.txt`, §5's rotation-matched ratios, and the stated figures in "Figures the
  verification changed". Where a figure first stated elsewhere did not reproduce, the recomputed value is used, and the
  difference is named; where a verifier's figure differs from this record's, `VERIFICATION.md` names it (its first
  table and its *Record* notes).
- **Names.**
  - **The 2026-10-17 arms** are A, as deployed, and B, with loop-closure acceptance off. Their segments are F1–F6, with
    F2's relaunches F2r and F2r2 ([`lane-slam-frame-health-2026-10-17`](../lane-slam-frame-health-2026-10-17/README.md),
    called the frame-health record below).
  - **The reprocessing record's candidates**
    ([`lane-rtabmap-reprocess-2026-10-21`](../lane-rtabmap-reprocess-2026-10-21/README.md)) are A′,
    `RGBD/NeighborLinkRefining` false; A″, `Vis/MinInliers` 40; and A‴, both.
  - **The a3 record** is [`lane-rtabmap-a3-2026-10-22`](../lane-rtabmap-a3-2026-10-22/README.md), the live session
    of 2026-10-23.
  - **G-sim** is goal-a's sim gate; its PASS is the 2026-09-25 record's.
- **Frames.**
  - **world** is the environment's frame, with its origin at the room's centre.
  - **odom** is ground truth on the bridge: world = odom + (−1.65, −0.55).
  - The robot spawns at world (−1.65, −0.55) with yaw 0. The grid dump's quaternion (0, 0, 0, 1) is in Isaac Lab's
    (x, y, z, w) order, and so is the identity.
  - The map frame starts at the spawn, with the world's axes.
- **The wall line** is a wall's centre line. "Beyond the wall" means past it, "inside" within all four, "outside" not
  inside.
- **A goal is classed by where it physically lay:** its point in odom when it was sent. Where a frame had moved before
  goal-sent, the goal sent in map coordinates lay somewhere else (09-25's R3; the 2026-10-14 speed-cap first session,
  §3).
- **The first-cut box.** Some investigations classed locations against a first-cut box: odom x [−0.80, 3.90], y
  [−2.20, 3.30]. Its west edge is world x −2.45, one end of the west wall line's range (§2). It lies about 0.2 m inside
  the other three wall lines, which no recorded node approaches. Where a figure uses the box, it says so.
- **Drift, steps and reference.** These follow the frame-health and reprocessing records.
  - **Drift** is the largest displacement, over four goals anchored in odom, of a goal's map point against the
    segment's reference.
  - **A step** is a move of 0.5 m or more between consecutive iterations in where the frame puts any of those goals.
  - **A drift-growing step** is one after which the drift is larger than before.
  - **A segment's window** runs from its reference to its last node result.

## 2. The room

The training grid that the bridge dumps at reset: 80 × 80 cells of 0.1 m, centred on the environment's origin and
spanning ±4.0 m, sha256 `36929c68…`. It is the same in every boot that hashed it, from 2026-10-14 on. The generator
builds the room centred on that origin, with walls 0.15 m thick on the lines x = ±w/2 and y = ±h/2 (`proc_room.py` line
1169, at `af7d7985`). It then marks every cell whose centre lies beyond those lines as occupied (lines 575–591), and
inflates obstacles by a 3-cell disc. The room lies inside the grid on every side, so nothing is truncated.

- **The free space** spans cell centres x −2.15 to 1.95 and y −2.45 to 2.45 (world). The column at x −2.15 is free only
  in rows y −1.15 to −0.25: the doorway.
- **The west wall line lies at world x −2.45 to −2.475** (odom −0.80 to −0.825). Two constraints fix it:
  - the doorway's free column, three cells inside the exterior clip, puts it at or beyond −2.45;
  - the east wall's rasterisation, four cells beyond the last free column, puts it short of −2.475.

  The east line is its mirror. The north and south lines lie at y = ±(2.875 to 2.975).
- **One doorway, in the west wall.** The free rows at the doorway column place the jambs:
  - the wall's south piece ends at y −1.50 to −1.40;
  - its north piece starts at y 0.00 to 0.10;
  - so the opening is 1.40–1.60 m wide, full height.

  Replaying the generator's wall phase against the grid narrows these to −1.4725…−1.4025 and 0.000…0.0998, an opening of
  1.40–1.57 m, in a room 4.900–4.945 m by 5.805–5.945 m (35,110 feasible draws, all one structure;
  `wf_training_and_geometry/room_fit_feasible_final.json.gz`, `wf_training_and_geometry/notes.txt`). That fit also
  leaves a full-height gap of 0.73–0.90 m at the north-west corner. Furniture fills that corner in the grid, and whether
  the robot could pass it was not checked.
- **Every crossing went through the doorway.** In the 2026-10-17 databases the robot crossed the west wall line 22
  times, with the line taken at either end of its range. Every crossing lay at world y −0.95 to −0.25, inside the
  doorway. The robot never crossed the north, south or east line. Positions are `Node.pose`, which is odometry
  (`wf_training_and_geometry/node_odom_poses.json`; equal to the databases' Node table for F1 and F6).
- **Walls 2.7 m high, and no ceiling (inferred).** Over 91 of F1's stored depth frames (every fifth node), the highest
  points sit 2.704–2.707 m above the floor, and none lies above 2.75 m (`wf_training_and_geometry/heights.py`, which
  reads F1; `wf_training_and_geometry/notes.txt` reports the same for 120 of F6's frames). The investigation infers no
  ceiling from the same frames: no constant-height plane, and nothing returned in the top rows where a ceiling drawn
  2.2–2.9 m up would return. That inference is quoted from `wf_training_and_geometry/notes.txt`, not recomputed here.
- **The starts.** The spawn lies 0.80–0.825 m east of the west wall line, and the nominal start of the G1 and L2
  reaches, odom (0.075, −0.035), lies 0.875–0.90 m east of it. Both lie within the doorway's north–south span.

## 3. The goals: where they lie, and how they came to be there

**Classification** (lane goals as map coordinates; world = map + (−1.65, −0.55) when the frame is at its start). The
distance is to the nearest wall line, over the wall lines' ranges (§2).

| goal | map (x, y) | world | distance from the nearest wall line (m) | class | used by |
|---|---|---|---|---|---|
| G1 (= 08-17 M1; the fixed goal F1–F3 and FX) | (−2.00, 2.25) | (−3.65, 1.70) | 1.18–1.20 beyond | outside | 09-25 to 10-23 |
| L1 (= M4) | (−2.00, 0.00) | (−3.65, −0.55) | 1.18–1.20 beyond | outside | 09-25, 10-03 |
| L2 (= M5) | (−2.75, −2.25) | (−4.40, −2.80) | 1.93–1.95 beyond, and past the grid's −4.0 m edge | outside | 09-25 to 10-23 |
| gate R1, R2, R3 | (−0.05, 2.15), (1.00, 2.35), (2.65, 1.25) | (−1.70, 1.60), (−0.65, 1.80), (1.00, 0.70) | 0.75–0.78, 1.08–1.18, 1.45–1.48 inside | inside | 09-25 |
| R1 (= scene-axis A), R2 (= B) | (1.4, 2.7), (3.1, 0.0) | (−0.25, 2.15), (1.45, −0.55) | 0.73–0.83, 1.00–1.03 inside | inside | 10-14, 10-17, 10-23 |
| scene-axis C–F | (2.5, −1.8), (3.1, 1.6), (1.7, −1.2), (3.6, −1.7) | (0.85, −2.35), (1.45, 1.05), (0.05, −1.75), (1.95, −2.25) | 0.50 or more inside | inside | 10-14 |
| spawn; nominal start | (0, 0); (0.075, −0.035) | (−1.65, −0.55); (−1.575, −0.585) | 0.80–0.83; 0.88–0.90 inside | inside | all |

- **Every nominal start lies inside.** Under its moved frame, the speed-cap first session's S03 and S04 began in the
  doorway, 0.08–0.10 m beyond the wall line (their observer records' first `pose_odom`;
  `recompute_2026-10-29/checks_starts_views_steps.py`).
- **The straight lines.** From the nominal start, the straight line to G1 meets the west wall line at y 0.38–0.41,
  0.28–0.41 m north of the north jamb. That line meets solid wall, and every route driven bent south through the
  doorway. The north-west gap is furniture-filled in the grid, and whether it is passable was not checked. The
  straight line to L2 passes the south jamb's outer corner at 0.04–0.13 m, less than the chassis's 0.18 m half-width
  (0.432 × 0.360 m footprint). So a straight L2 route clips the jamb.
- **Before the flip.** G1, L1 and L2 are the 2026-08-17 gate's M1, M4 and M5. That gate's goals came from a
  plannability probe on the map of the time.
  - **Which goals.** The 09-25 pre-registration names the reuse ("= 2026-08-17 M1", "M4", "M5"). It also notes G1
    "unmapped at t0", and drops 08-17's two right-side goals, M2 and M3, as now "against the north wall"; M6 is not
    discussed (`goal-a-rig-gate-v3-2026-09-25`'s deposit, `preregistration.md`).
  - **The pre-flip spawn.** The 2026-08-22 same-pose probe anchored the spawn at world (0.35, −1.75), yaw 0
    (`goal-a-attribution-2026-08-22`'s deposit, `same-pose-probe/meta.json`). That the 08-17 gate spawned there is
    inferred from the probe's depth match.
  - **Where the goals lay then.** Enriched rooms span U[4.0, 7.5] m a side, centred on the origin
    (`strafer_env_cfg.py` line 1402). So a point inside the smallest room's walls (|x|, |y| < 1.925) is inside every
    room, and a point beyond the largest's (> 3.825) is outside every room. At that spawn:
    - G1 sat at (−1.65, 0.50) and L1 at (−1.65, −1.75): inside any enriched room;
    - **L2 sat at (−2.40, −4.00): outside any enriched room.**
    - M6 sat at (−1.65, −3.25); its class depends on the room of the time, whose size is not known.

    **The 2026-08-17 v2 gate therefore already had at least one goal outside the room (L2; M6 depends on the room's
    size). "Since the flip" applies to G1 and L1 only.**
- **The move.** From (0.35, −1.75) at the 2026-08-22 probe to (−1.65, −0.55) in the post-flip dump: 2.33 m.
  - **When.** The tooling brief
    [`pose-stable-sim-probe`](../../tasks/active/tooling/pose-stable-sim-probe.md) records that after the Isaac Lab
    pair flip of 2026-09-12 "the spawn had moved", about 2 m. It also records that seed determinism across a version
    change is not a property the repository holds.
  - **Why.** Not documented anywhere. Reading the move as an accident of the flip is an inference.
  - **The scored sessions ran in the current room from 2026-09-25 on.** The Nav2 costmaps deposited by the 2026-09-25,
    2026-10-03 and 2026-10-09 sessions fit the current room's walls at the post-flip spawn. There, 0.293–0.312 of
    their lethal cells (cost 99 or more) lie within 0.10 m of a wall line (hw 2.4625, hh 2.9375), against 0.038–0.039
    at the pre-flip spawn (`recompute_2026-10-29/cm.py`, over each deposit's `*costmap*.npz`).
  - **The 2026-09-25 record** compares v2 at G1 with 08-17's M1 as "the same goal". The coordinates are the same; the
    place is not.
- **The outside-ness was on record before 2026-10-23.**
  - The 2026-10-09 video record shows both goals outside, past the doorway.
  - The 2026-10-14 speed-cap record's setup row says G1 and L2 "both lie outside the room, past its doorway".
  - The 2026-10-14 scene-axis record measured G1 and L2 at 1.682 and 2.372 m from the nearest free cell, and wrote
    "Training never puts a goal there".

  None of them linked the outside routes to SLAM's failures, or traced how the goals came to be there.

**The reaches** (post-flip v3 lane reaches: each goal the v3 policy was sent through the lane, with its node result,
classed by where the goal physically lay):

| record | reaches | outside | inside | node `SUCCEEDED` at G1, L1 or L2 |
|---|---:|---:|---:|---|
| `goal-a-rig-gate-v3-2026-09-25` | 9 | 7: G1, L1, L2, F1–F3; R3 (physically) | 2: R1, R2 | 6 of 6 |
| `goal-a-cli-confirmation-2026-10-03` | 4 | 4: G1, L1, L2, FX | 0 | 4 of 4 |
| `goal-a-cli-video-2026-10-09` | 2 | 2 | 0 | 2 of 2 |
| `subgoal-referent-offset-ab-2026-10-11` | 12 | 12 | 0 | 12 of 12 |
| `lane-term-speedcap-ab-2026-10-14`, first session (void) | 8 | 7 | 1: S04 (physically) | 8 of 8; six at points the moved frame sent them to |
| `lane-term-speedcap-ab-2026-10-14`, second session | 12 | 12 | 0 | 12 of 12 |
| `lane-term-scene-axis-2026-10-14` | 4 | 0 | 4 | — |
| `lane-slam-frame-health-2026-10-17` | 24 | 12 | 12 | 12 of 12 |
| **before 2026-10-23** | **75** | **56** | **19** | |
| `lane-rtabmap-a3-2026-10-22` (run 2026-10-23), counted reaches: the 24 the two arms count (B2r in B2's place) and B7's one | 25 | 13 | 12 | 13 of 13 |
| **with that session** | **100** | **69** | **31** | |

- **The share.** 56 of 75 (75 %) before 2026-10-23, and 44 of 51 before 2026-10-17. With 2026-10-23's counted
  reaches, 69 of 100.
  - That session's count leaves out B2's three reaches, which its relaunch B2r replaces under the session's count
    rule. Counting them too gives 71 of 103.
  - Sources: each record's reach table, the speed-cap deposit's `session1/frames_posthoc_session1.txt`, and the a3
    deposit's `analysis/odom_terminal.json`.
- **Two reaches change class on the physical goal.**
  - **09-25's R3** was an inside goal, sent on a frame rotated −77.4°. It physically lay at world (−1.99, 3.08),
    0.11–0.21 m north of the north wall line. The robot left through the doorway 10.5 s sim after the goal was sent,
    and parked outside at (−2.18, 3.28), 0.27 m from the sent point in odom. The node returned `ABORTED`
    (`goal-a-rig-gate-v3-2026-09-25`'s deposit: `missions/R3.json` and `ride_along/tf_gate.jsonl`).
  - **The speed-cap first session's S04** was sent to G1 on a frame rotated +90.3°. It physically lay at world
    (−0.25, 1.66), inside (`session1/frames_posthoc_session1.txt`).
  - **That session's other goals.** A goal counts as moved when it lay 0.5 m or more from where the referent-offset
    session sent the same goal. Its five other moved goals (S03, S06, S07, S08, S10) stayed outside. S01 and S05 lay
    0.25 m and 0.05 m from their intended points.
- **48 of 48.** Every reach sent to G1, L1 or L2 ended on the node's `SUCCEEDED`. That holds in every post-flip session
  before 2026-10-23 whose frame kept the goal at its coordinates: all but the speed-cap first session.
  - The void first session's eight reaches also ended `SUCCEEDED`, six of them at points the moved frame had sent
    them to, 0.89–5.23 m from their intended points.
  - So did the 2026-10-23 session's 13 counted outside reaches.
  - The one reach that physically ended outside without `SUCCEEDED` is R3, whose goal was never an outside lane goal.

## 4. What the camera sees outside

**The view.** The stored frames, 640 × 360 RGB and depth from the perception camera
(`wf_camera_by_location/images/`, `wf_training_and_geometry/imgs/`):
- **The upper half** is a uniform bright sky with no depth return.
- **The lower half** is Isaac Lab's default ground plane: mid-grey, with light square grid lines running to the
  horizon. The same plane, with the same texture, lies inside the room.
- **The walls** are plain, untextured light grey on both faces.

**Facing outward, almost every feature outside lies on one periodic plane.**
- **The share.** The median outside node has all its features beyond the room (a median share of 1.000;
  `wf_camera_by_location/out/feature_world.txt`).
- **Facing north or south**, an outside view also sees the room's outer wall. The 155 outside nodes with part of the
  room in view (up to half the field) have a median of 101 words (`recompute_2026-10-29/checks_starts_views_steps.py`).
- **The far clamp.** Take the top 11/45 of each frame's rows: its top 88 of 360, the band the scene-axis record reads
  at the policy's 45 rows. There a median of 100 % of pixels lie at the far clamp (no return, or beyond 6 m) outside,
  against 3.1 % inside (n 282 and 665, every third node). Over the whole frame the medians are 54.6 % and 1.4 %.
  - These are the perception camera's frames, not the policy's reduced input.
  - Source: `wf_training_and_geometry/farclamp_by_location.txt`; outside = world x < −2.475, inside = x ≥ −2.45.

**More features, not fewer.** Over the 2026-10-17 databases (2,847 nodes), classed against the first-cut box with a
0.30 m margin (`wf_camera_by_location/out/compact_table.txt`, `summary_by_location.txt`):

| | inside | outside |
|---|---:|---:|
| nodes | 1,848 | 696 |
| words per node, median (`Keypoint/Current_frame/words`, capped at 500) | 132 | 390.5 |
| nodes under 50 words | 135 | 0 |
| neighbour-refinement inliers, median (attempts) | 25 (953) | 39 (315) |
| refinements accepted | 0.73 | 0.83 |

- **Classed by the wall's centre line instead**, with the same 0.30 m margin, the medians are 132 and 391 words.
- **The views that are short of features are inside:** a camera close to a plain wall or furniture slab.
- **So the hazard outside is perceptual aliasing, not tracking loss** (the reading; §9). The exterior offers many
  keypoints, all non-distinctive.
  - **Odometry.** On the bridge odometry is ground truth, so a feature-poor view cannot cause odometry drift. SLAM adds
    error only through the closures and refinements it accepts.
  - **Refinement.** A feature-poor view can still give a poor refinement. Arm B's two refinement steps of 0.5 m or
    more came from 17-inlier views of the plain east wall, inside the room: F2r2 node 229 and F4 node 285
    (`recompute_2026-10-29/checks_starts_views_steps.py`).
- **How the session faced.** It drove its return legs camera-outward, on the holonomic base. None of the 846–852 nodes
  beyond the west wall line, by the line's range, faces east (heading within ±45° of +x). The data therefore hold
  almost no exterior views facing the room.

## 5. SLAM's failure modes, by location

**Metre-scale wrong closures come at the doorway and outside.**
- **F1's closure 142→16** is the one accepted closure in the 2026-10-17 data that is grossly wrong.
  - **Its error.** Against ground truth its transform errs by 0.90 m and 89.8°, and it rotated F1's frame 86.84°.
  - **Where it fired.** Node 142 stands in the doorway, at world (−2.56, −0.33), 0.09–0.11 m past the west wall line,
    heading 113°. The robot was returning from G1, camera outward.
  - **What it matched.** Node 16 is a warm-up-spin node at the spawn, heading −160°, looking out through the doorway.
  - **Its strength.** It had 33 inliers of 63 matches (`Loop/Visual_inliers` at node 142 in the F1 database). That
    passes the deployed `Vis/MinInliers` of 15 and falls short of A‴'s 40. Whether rtabmap 0.23.13 would register
    it alike is not tested.
  - **The other closures.** Arm A's databases keep 105 accepted closure and proximity links. The other 104 err by at
    most 0.289 m and 3.7° (median 0.016 m), and none has an end beyond the wall line
    (`wf_camera_by_location/out/closures.txt`). Arm B keeps none.
- **In the reprocessing record's matrix, the first metre-scale step comes beyond the wall line in every deployed-set
  and A′ run that has one.** These are the runs on the restored-covariance inputs: 18 and 16 deployed-set runs, and 13
  and 10 A′ runs, under 0.23.7 and 0.23.13, 57 in all. In each, the first drift-growing step of 1 m or more came with
  the robot beyond the west wall's centre line, by at least 0.07 m. A″ and A‴ take no such step in their 320 runs. The
  addendum's runs are not counted.
  - **The places.** The 57 steps fall at 16 nodes in nine places.
    - Five are at the doorway: two in F1, and one each in F2, F4 and F5.
    - Two are near G1: F1 node 115, and F2 nodes 87–99.
    - Two are on the L2 route, about 1 m beyond the wall line: F2r2 node 123, and F6 nodes 120–121.

    Ten reprocessings of one database are not independent.
  - **What they matched.** Fourteen of the 16 nodes matched warm-up-spin nodes 13, 15, 16 or 17. Those sit at the
    spawn with headings from about 155° through 180° to −145°, inside the doorway's span of bearings from the spawn
    (144° through 180° to −131°). F5's two nodes matched node 80, a start-point node facing the doorway.
  - **The rotation.** At six of the places the frame rotated 75–91°, and at F5's 59–89°. At F4 node 153 and F1 node
    149 it rotated under 6.5°: a doorway view matched to the spawn 0.94–1.08 m away, a translation alias.
  - Sources: the reprocessing deposit's `runs/`; `wf_records_survey/rp_steps.txt`;
    `wf_verify_records_survey/rp_verify.py`; `recompute_2026-10-29/rp.py`, `rp_rot.py` and
    `checks_starts_views_steps.py`.
- **The concentration belongs to the 1 m tail.** The deployed set's drift-growing steps of 0.5–1 m came mostly inside:
  13 of 16 under 0.23.7 and 12 of 15 under 0.23.13. Its steps of 1 m or more came in the doorway band or outside: 20 of
  22 and 16 of 18 (`wf_records_survey/rp_steps.txt`).
  - **The doorway band** reaches 0.3 m inside the wall line and 0.4 m beyond it.
  - **The two inside under each build** are later steps in F1 runs, after a first step at the doorway.
- **A‴ accepts none of them.** In its 160 runs, over the whole run, A‴ accepts no closure while the robot is beyond the
  wall line (world x < −2.45).
  - **The deployed set** accepts 28 such closure and proximity iterations in 80 runs under 0.23.7, and 31 under
    0.23.13 (`recompute_2026-10-29/closures_beyond_wall.py`, over the reprocessing deposit's `runs/`).
  - **By the doorway and exterior zones of `wf_slam_by_location/zones_0.6.txt`**, which reach 0.6 m around the doorway,
    the deployed set accepts 37 and 39 (26 + 11 and 30 + 9), and A‴ none.
- **Location does not settle the deployed set's outcome.** Of its 70 failing runs, 52 fail with no step of 1 m or more
  under 0.23.7, and 54 under 0.23.13 (`wf_verify_slam_by_location/v_largest.py`; `recompute_2026-10-29/fail.py`). So
  the doorway's and the exterior's wrong closures do not account for most failures.

**Refinement drift is not concentrated outside per metre travelled** (per refined neighbour link, read from the
databases). Per degree turned and in per-link translation, outside links err more; matched for rotation, arm A's are
about equal.

| | arm A, inside | arm A, outside | arm B, inside | arm B, outside |
|---|---:|---:|---:|---:|
| median yaw error per link (°), windows, link midpoint against the first-cut box | 0.321 | 0.207 | 0.310 | 0.165 |
| summed translation error per metre travelled (mm/m) | 279.1 | 181.7 | 313.9 | 150.4 |
| summed yaw error per metre (°/m) | 8.15 | 2.33 | 8.32 | 2.24 |
| summed yaw error per degree turned | 0.058 | 0.237 | 0.048 | 0.185 |

Sources: `wf_slam_by_location/recorded_by_location_+0.00.txt` for the medians, and
`wf_camera_by_location/out/link_error_by_location.txt` for the rows below them. Those rows take whole databases, with
the to-node classed by the 0.30 m margin.

- **Per link and per metre, outside links err less in yaw.** They are straight runs.
- **Per degree turned they err more,** and their median translation error is larger: 15.7 mm against 11.0 mm in arm A.
- **Matched for rotation,** arm A's outside links carry 0.93–0.99 of the inside yaw error, and arm B's 0.64–0.67. These
  are the investigation's and its verifier's figures under their binning (`VERIFICATION.md`, `slam_by_location` C4), not
  re-derived here.
- **Drift without the exterior.** The 2026-10-14 scene-axis session never left the room: its robot came no closer to
  the west wall line than x −2.11. Its `map→odom` still moved 1.362 m from its first ride-along sample.
- **The exterior without drift.** The 2026-10-11 referent-offset session ran twelve reaches to G1 and L2, and its frame
  stayed within 0.219 m.
- Sources: each deposit's ride-along, `tf_a4.jsonl` and `tf_referent.jsonl`, and the scene-axis observer and transit
  records.

**The three live away-rotations of 76–88°.** These are `map→odom` steps between consecutive ride-along samples that
grow the frame's offset from its first sample. The robot's position comes from the observer or transit trace, mapped
into odom.

| session | step | robot at the step | closure |
|---|---|---|---|
| 2026-09-25 | 0.953 m, −75.7° (356.9→357.0 s sim) | world (−1.68, −0.84), 0.28 m from the nominal start, 0.77–0.79 m inside the west wall line; heading 169–173° across the step, rotating in place from −133° (355.0 s) through 180° | none accepted at INFO level (that record's README) |
| 2026-10-14, first session | 1.249 m, +88.1° (75.0→75.1 s) | world (−2.57, −0.30), 0.09–0.12 m past the wall line, heading 123°, on the transit back from G1 | — |
| 2026-10-17, F1 | 86.84° (75.2 s) | node 142, world (−2.56, −0.33), heading 113°, back from G1 | 142→16 |

- **Two came in the doorway, facing out, on the way back from G1.**
- **The 2026-09-25 rotation came at the start point, inside the room,** with no accepted closure logged. So the
  doorway concentration is not a property of every large rotation. It is a property of the reprocessing runs' tail of
  1 m or more.
- **Two return steps of 50° or more came with the robot inside:**
  - 2026-09-25 at 475.5 s (0.941 m, +77.5°, after R3), with the robot at (−1.80, −0.54);
  - the 2026-10-14 first session at 218.6 s (0.234 m, −52.2°, after S04), with the robot at (−1.70, −0.37).
- Sources: `goal-a-rig-gate-v3-2026-09-25`'s `ride_along/tf_gate.jsonl` and `transits/`; the speed-cap deposit's
  `session1/.../tf_a3.jsonl` and its transit and observer records; `recompute_2026-10-29/live_steps.py`.

**Failed transits stall at or near the doorway, the routes' narrowest passage.**
- **Failed** here means a transit that ran out its budget or was stopped.
- **Pinned** is the a3 record's rule: in transit, more than 0.5 m from the target, with no 0.03 m of progress in the
  straight-line distance to it for 24 s sim. That distance is read in the map frame.
- **Where.** The position is the mean over the longest span whose last sample lies within 0.05 m of its first
  (`recompute_2026-10-29/pins_rule.py`).
- **The frame offset** is `map→odom`'s distance from the session's first ride-along sample, over the hold.

| session | transit | where it stalled (world) | frame offset (m) | pinned by the rule |
|---|---|---|---|---|
| 2026-10-03 | FX, back from L2 | (−1.80, −0.98), 127 s: inside, 0.65–0.67 m east of the wall line | 0.45 | no: 0.41–0.46 m from its target |
| 2026-10-09 | to the L2 start, back from G1 | first (−2.92, 0.26) for 64 s, 0.45–0.47 m beyond the wall line, north of the north jamb; then (−2.76, 0.10) at the jamb from about 179 s sim until it ran out | 0.45 over the first hold; 0.59–0.63 at the jamb | yes |
| 2026-10-09 | its retry | (−2.76, 0.13), 44 s: at the north jamb, 0.29–0.31 m beyond the wall line | 0.63–0.66 | yes |
| 2026-10-14, first session | S11, both attempts | (−2.78, −1.56) for 44 s, then (−2.78, −1.37) for 149 s: 0.31–0.33 m beyond the wall line, at the south jamb's end | 1.25–1.30 | yes |
| 2026-10-14, first session | S12 | (−2.78, −1.65), 94 s: 0.31–0.33 m beyond the wall line, south of the jamb's end | 1.31–1.32 | no: it settled 0.496 m from its target |
| 2026-10-14, scene-axis | S05, both attempts | (−1.62, 0.47) for 102 s, then (−2.10, 0.41) for 111 s: inside, 0.35–0.85 m east of the wall line | 0.96–1.02, then 0.93–0.98 | yes |

Source: `recompute_2026-10-29/pins_rule.py`, over each deposit's `*.to_start*.json` and ride-along.

- **Where.** Those beyond the wall line sat at or near a jamb. The two inside sat within a metre of the doorway.
- **When.** Each came with the frame already 0.45–1.32 m off.
- **Downstream of frame error? A reading, not a test.** The doorway is hard under an unmoved frame too.
  - **A‴'s B5 transit** back from L2, on 2026-10-23, sat 54.9 s at one place with its frame within 0.015 m. That is a
    pin by the rule, and it then got through (the a3 record's §6, quoted).
  - **F6's 2026-10-17 pin** came with closures off.

  These stalls are evidence of the doorway as a bottleneck, not independent evidence that the exterior breaks SLAM.
- **Elsewhere.** The 2026-10-17 F6 transit's pin, which resolved, is described in that record's dated line. The
  2026-10-23 B2, B5 and B7 transits are in the a3 record's §6.

## 6. The policy side

**Training never places the robot, a goal, a path or a subgoal outside a room** (code at `af7d7985`; `proc_room.py` is
unchanged from `615fc14`, the tree v3 trained at).
- **Where they come from.** Spawn points, goal candidates and the A* path all come from the room's free space.
  - The robot spawns on reachable free cells (`reset_robot_proc_room`).
  - The goal is drawn from the same pool, at least 2.0 m from the robot (`SubgoalCommand`, `min_goal_distance`).
  - The subgoal is a point ahead on that path.
- **That free space ends inside the walls.** It marks every cell beyond the room's wall lines occupied
  (`proc_room.py` lines 575–591). So at a doorway it ends 0.30–0.40 m short of the wall line.
- **The robot cannot leave the room.** Straying 0.30 m from the path ends the episode with a −50 penalty
  (`_SUBGOAL_MAX_OFF_PATH_M`, the `off_path` term).
- **Every enriched room has walls.** Enriched training uses difficulty levels 4–7 only (`_ENRICH_MIN_LEVEL`,
  `_ENRICH_MAX_LEVEL`); the open-field levels, which skip the clip, are not drawn.

**What training does see.** Doorways and openings, from inside.
- **How often.** A walls-only model of the generator, which ignores furniture, puts the doorway in view within 6 m on
  0.424 of episodes and 0.294 of path samples, with the camera heading along the path. With a heading drawn afresh at
  each 0.1 m path sample, it is in view on 0.999 of episodes (`wf_training_and_geometry/mc_exterior.py`,
  `mc_exterior.json`; a model, not a measurement).
- **What lies beyond.** Training environments sit 10 m apart (`_PROCROOM_ENV_SPACING`). So beyond an interior
  environment's doorway should lie a neighbour room's outer wall, 2.5–6 m away. That is an inference from the spacing,
  not a render, and an environment on the array's perimeter faces open floor.
- **The sky.** 30 % of training rooms have no ceiling (`_ENRICH_P_CEIL` 0.7).
- **On the lane, looking out from inside.** The lane's own views out through the doorway are like the outside class.
  Inside but within 1 m of the west wall line, facing west (|heading| over 135°), the far-clamp medians are 0.812 in
  the top 11/45 of the rows and 0.445 over the frame. That is 167 nodes of the 2026-10-17 databases; the verifier's
  41-frame sample gives the same medians (`VERIFICATION.md`, `training_and_geometry` C1).
- **What training never had** is the robot itself outside the room, with its goal and subgoal outside, and a goal
  reached through a doorway.

**At G1, L1 and L2 the policy is outside its training distribution. Out of distribution did not mean failed.** At the
node level it reached every such goal it was sent to (48 of 48, §3). The findings read at those goals are confounded
with being outside, and several rest almost entirely on them.

- **2026-10-17's arrival split is a frame-direction statistic, not a policy one.**
  - **The split.** In odom, 5 of the 12 reaches at G1 and L2 came within 0.30 m of the goal fixed at goal-sent,
    against 12 of 12 at R1 and R2 (Fisher's exact p 0.0046, two-sided). All seven reaches that never came within
    0.30 m were at G1 or L2.
  - **The direction.** At the result, the odom distance exceeded the node's live one on exactly those seven, and fell
    below it on all twelve R1 and R2 reaches.
  - **Not larger frame motion.** The in-reach goal excursion's median is 0.090 m at G1 and L2 and 0.091 m at R1 and
    R2.
  - **The restatement.** That record's "the node's `SUCCEEDED` overstates arrival" becomes: the node misstates arrival
    by the in-reach frame motion projected on the approach. It overstates at the west goals, G1 and L2 (on 7 of their
    12 reaches), and understates at the east goals, R1 and R2.
  - Sources: the frame-health deposit's `analysis/odom_terminal.json`.
- **The 6 Hz limit cycle has been read almost only outside.**
  - **The windows.** In the time-to-goal baseline's rig row, the detector fired on 5 of the 6 outside windows long
    enough to read (at least 45 ticks): the gate's G1, L1 and L2, and the video's G1 and L2. It fired on neither of
    the 2 inside windows, R1 and R2 (Fisher p 0.11; the baseline deposit's `posthoc_terminal_modes.txt`).
  - **What that supports.** "It occurs only at outside goals" rests on two inside windows.
- **The stop-short's direction persists at inside goals; its size there is unknown.**
  - **In the rig row:** 6 of 8 outside band entries went quiet before the 0.30 m crossing, and 1 of 2 inside.
  - **On 2026-10-17:** 8 of 12 outside and 5 of 12 inside on the live frame (p 0.41), and 6 of 12 and 4 of 12 in odom
    (p 0.68).
  - **Pooled at inside goals:** 8 of 18 (gate R1 and R2, 1 of 2; scene-axis 2 of 4; 2026-10-17 live 5 of 12), against
    training's 2.23–7.21 %. The pool mixes three stop-short definitions: path-end distance, the scene-axis
    `lane_summary` read, and live-frame goal distance.
  - **One more inside stop-short.** The pool leaves out the speed-cap first session's S04. It was physically inside,
    and went quiet 0.414 m out, with a 38.53 s band, in a void session (`session1/lane_summary_a3.txt`).
- **The G-sim v3 PASS is unsupported on inside reaches, not overturned.**
  - **The counted successes.** Six of its seven were at outside goals: G1, L1 and L2 in the six missions, and the
    fixed-goal leg's F1–F3 at G1. The one inside success is R1.
  - **The inside evidence** is R1 reached and R2 missed.
  - **R3 was physically outside** (§3).
  - **The test.** By nominal goal the gate read 6 of 6 outside against 1 of 3 inside (p 0.083). By physical goal it is
    6 of 7 against 1 of 2 (p 0.42).

## 7. What changes, and what stands, in earlier records

Eight earlier records carry dated lines of 2026-10-29 that point here: six corrections, and notes in the video and
scene-axis records, which were already right.

| record | the dated line |
|---|---|
| [`goal-a-rig-gate-v3-2026-09-25`](../goal-a-rig-gate-v3-2026-09-25/README.md) | a correction: the PASS is unsupported on inside reaches, not overturned; six of seven counted successes outside; R3 physically outside |
| [`goal-a-cli-confirmation-2026-10-03`](../goal-a-cli-confirmation-2026-10-03/README.md) | a correction: all four goals outside; the budget PASS stands, since it does not depend on where the goal lies; its terminal reads are at outside goals |
| [`goal-a-cli-video-2026-10-09`](../goal-a-cli-video-2026-10-09/README.md) | a note: descriptive and right that the goals lie outside; the frame's offset during the doorway pin |
| [`policy-time-to-goal-baseline-2026-10-11`](../policy-time-to-goal-baseline-2026-10-11/README.md) | a correction: the rig row is 8 of 10 band entries outside; the limit cycle 5 of 6 against 0 of 2; the inside gap's size unknown; the training-env measurements stand; §6's rig comparison and Reading 3's attribution of the rig's limit cycle to the lane's feedback path rest on rig reaches almost all at outside goals |
| [`subgoal-referent-offset-ab-2026-10-11`](../subgoal-referent-offset-ab-2026-10-11/README.md) | a correction: all twelve reaches outside; the referent-offset reading stands at G1 and L2, where it was read, and says nothing of goals inside the room; its stop-short figures are outside-only |
| [`lane-term-speedcap-ab-2026-10-14`](../lane-term-speedcap-ab-2026-10-14/README.md) | a correction: only the lane clause rests on outside goals; the training-env clauses, which carry [`depth-subgoal-v4-scoping`](../../tasks/active/trained-policy/depth-subgoal-v4-scoping.md)'s speed-envelope item, stand |
| [`lane-term-scene-axis-2026-10-14`](../lane-term-scene-axis-2026-10-14/README.md) | a note: already right; a pointer |
| [`lane-slam-frame-health-2026-10-17`](../lane-slam-frame-health-2026-10-17/README.md) | a correction: arrival restated as a frame-direction statistic; "none pinned" corrected: F6's transit pinned and resolved |

**What stands.**
- **The 2026-08-22 v2 attribution.** G1 was an inside goal then.
- **The CLI confirmation's budget PASS** (the table above).
- **The frame-health and reprocessing records' refinement-drift finding** (§5).
- **The 2026-10-23 session's comparison of A‴ against the deployed set.** Both arms ran the same goals; whether the
  deployed set's failure rate there transfers to a room without this exterior is that record's caveat.
- **The reprocessing record's spin-node reading,** which the doorway complements rather than replaces. The spin's
  west-facing nodes are the views that look out through the doorway.

## 8. Proposals

- **A lane preflight,** deposited with the next registration. A goal or start outside the dumped grid's free space is
  refused unless the manifest labels it an exterior condition, and the check is repeated at every scene or spawn
  change.
- **An inside-room goal set,** drawn from the current room by training's own rule. The scene-axis generator
  (`a4_goals.py` in that record's deposit) already does this; R1 and R2 are its first two draws.
- **The next lane session on that set, under A‴.** It re-measures the stop-short gap and the G-sim question on goals
  the policy was trained for, with:
  - one stop-short definition and the limit-cycle detector;
  - the terminal reads in odom, and the SLAM read;
  - at least 2 boots and 12 reaches.
- **v4's acceptance gate on an inside-room goal set.** The September PASS is not its baseline.
- **Exterior conditions, when wanted, as labelled conditions.** A goal outside the room or a route through a doorway
  measures something training never showed the policy.

## 9. What is not claimed

- **No test.** Every contrast here is post hoc and descriptive. Inside and outside reaches differ in:
  - goal type (the outside goals sit on open floor; inside goals include a furniture standoff and a goal the path
    wraps a block to reach);
  - route (two doorway crossings per outside reach) and distance;
  - where the robot dwells, and the warm-up spin's views.
- **Not that the exterior causes SLAM's failures.** No causal test was run, such as texturing the exterior. The
  aliasing reading rests on one wrong closure in the recorded data, the reprocessing runs' matched nodes, and the
  stored frames.
- **Not the inbound view.** The 2026-10-17 data hold almost no exterior views facing the room.
- **Not the 2026-10-23 session's own data.** Its databases and ride-alongs were not read here. Of its deposit, the audit
  read B3's dumped grid, the grid hashes of B1–B3 and B2's `chain.log` FRAME lines (§1); this record reads its
  `analysis/odom_terminal.json` for the reach count (§3) and quotes B5's transit from its record (§5); the notes in
  `VERIFICATION.md` also quote B2's drift (`analysis/frame_health_B2*.txt`). Its record reads its reaches' classes.
- **Not a training rate of exterior views.** The doorway-view figures are a walls-only model, and what lies beyond a
  training doorway is inferred from the environment spacing.
- **Not that the policy arrives outside.** 48 of 48 is the node's verdict on its live frame. In odom, 7 of
  2026-10-17's 12 outside reaches never came within 0.30 m.
- **Not why the spawn moved.**
- **Nothing about the real robot.** The bridge supplies ground-truth odometry and a simulated exterior; the robot has
  neither.

## Figures the verification changed

Each row is a figure an investigation stated (`wf_digest.txt`, `workflow1_exterior_audit.json`) that its verifier
corrected or narrowed, and the figure this record uses. The verifiers' text, with every correction and every missed
item, is `VERIFICATION.md`.

| figure | stated | used | section |
|---|---|---|---|
| the west wall's doorway, world y | −1.4 to −0.05 | the wall's south piece ends at −1.50 to −1.40 and its north piece starts at 0.00 to 0.10; an opening of 1.40–1.60 m | §2 |
| the spawn and the starts, east of the west wall line | 0.88–0.90 m for both starts | the spawn 0.80–0.825 m; the G1 and L2 start 0.875–0.90 m | §2 |
| post-flip v3 lane reaches to outside goals | 57 of 76 (75 %); 45 of 52 before 2026-10-17 | 56 of 75 (75 %); 44 of 51 | §3 |
| 2026-10-09's reaches | 3 | 2, both outside | §3 |
| 2026-09-25's R3 | an inside goal | physically outside, world (−1.99, 3.08), 0.11–0.21 m north of the north wall line | §3 |
| 2026-10-14's first-session S04 | an outside goal (sent to G1) | physically inside, world (−0.25, 1.66) | §3 |
| the straight line to L2, at the south jamb | 0.11–0.20 m from the south door edge | 0.04–0.13 m from the south jamb's outer corner | §3 |
| words per node, outside against inside | about 3× more outside | a median of 390.5 against 132, from almost no views facing the room | §4, §9 |
| exterior views facing the room | 56 outside nodes | almost none: none of the 846–852 outside nodes faces east | §4 |
| walls in view outside | none | facing north or south, the room's outer wall | §4 |
| rotation-matched yaw error, outside over inside | arm A 0.93, arm B 0.64: less outside in both | arm A 0.93–0.99, about equal; arm B 0.64–0.67 | §5 |
| refinement error outside | not concentrated outside | not per metre travelled; per degree turned, more (arm A 0.237 against 0.058) | §5 |
| where the reprocessing's first steps of 1 m or more came | in the doorway band (from about 0.3 m inside the wall line) or outside, 57 of 57 | beyond the west wall line, by 0.07 m or more, 57 of 57 | §5 |
| rotation at the reprocessing's first steps of 1 m or more | about 80–90° | 75–91° at six places, 59–89° at F5's, under 6.5° at F4 node 153 and F1 node 149 | §5 |
| the nodes those closures matched | west-facing warm-up-spin nodes | spin nodes for 14 of 16; F5's two matched node 80, a start-point node | §5 |
| live frame steps of 50° or more on a return | 3 of 3, from outside goals, at or near the doorway | three that grow the offset (76–88°): two in the doorway, 2026-09-25's inside at the start; two more with the robot inside | §5 |
| failed transits | every pin a doorway crossing | of six, four beyond the wall line at or near a jamb, two inside within a metre of the doorway | §5 |
| the G-sim v3 gate by goal class | inside 1 of 3, outside 6 of 6 (p 0.083) | by physical goal, outside 6 of 7, inside 1 of 2 (p 0.42) | §6 |
| the limit-cycle detector's firings | 5 of 5 at outside goals, none inside | 5 of 6 outside windows, 0 of 2 inside (p 0.11) | §6 |
| stop-short at inside goals | 8 of 18, "far above training's 2.2–7.2 %" | 8 of 18 over three definitions, S04 left out; its size unknown | §6 |
| training's doorway free space, short of the wall line | 0.30–0.32 m | 0.30–0.40 m | §6 |
| training's doorway in view (walls-only model) | an upper bound of about 42 % of episodes | 0.424 with the heading along the path; 0.999 with it drawn afresh | §6 |

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `lane-outside-room-goals-2026-10-23/record-files/` |
| deposit commit | `ac0bf958130ac037043ba2a778d8d80e3f70b4c1` |

Unless it names the main repository's tree or another deposit, every path this record names is relative to that
directory. `DEPOSIT.md`'s `cp -a` restores it into this record directory, where the same paths resolve.

- **Where it came from:** `provenance.md`, "The deposit". At `1e2447e` the audit's files moved here unchanged, except
  that the 20 links became relative and `SOURCES.txt` was added (below).
- **`DEPOSIT.md`** describes the files.
- **`recompute_2026-10-29/`** holds the record's recomputation: 26 scripts and one output, `mc_exterior.json`.
- **`SHA256SUMS`** lists the sha256 of every regular file except its own and `DEPOSIT.md`'s (308 files; its own sha256
  `b276005ded11cfd6973aeefebef6dbcab7d41364fb297fc5dde4bedc485f9415`). The 20 links under
  `wf_records_survey/leg2_files/` are not hashed there; `SOURCES.txt` gives their targets' paths and sha256. To check,
  from the deposit directory: `sha256sum -c SHA256SUMS`.
- **What it holds:**
  - `VERIFICATION.md`, the verifiers' corrections and the items they found missed, with this record's notes (added at
    `1ad0032`);
  - `wf_digest.txt`, the four investigations' summaries, verdicts, corrections and tables;
  - `workflow1_exterior_audit.json`, their full output;
  - `wf_<investigation>/` and `wf_verify_<investigation>/`, each investigation's and each verifier's scripts and
    outputs;
  - `recompute_2026-10-29/`, above.
- **The links.** `wf_records_survey/leg2_files/` holds 20 relative links into `lane-term-speedcap-ab-2026-10-14`'s
  deposit. After a `cp -a` restore outside the evidence repository the links dangle, and `SOURCES.txt` identifies the
  files.
- **Rerunning the scripts.** They ran on the sim host with python3 (the standard library, numpy and sqlite3; OpenCV
  for the depth readers). Many hold the host's absolute paths to their inputs, and several write their outputs into
  the working directory. Copy a script into a scratch directory, set its paths, and run it there, never in place.
  - The inputs, and where each is deposited, are in `provenance.md`'s Inputs table.
  - For example, from a scratch copy, `python3 -I farclamp_by_location.py` and `python3 -I mc_exterior.py` print the
    figures in §4 and §6.
