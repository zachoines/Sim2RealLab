# The lane's rtabmap databases, reprocessed, 2026-10-21

[`lane-slam-frame-health-2026-10-17`](../lane-slam-frame-health-2026-10-17/README.md) (below, the frame-health
record) read SLAM's `map→odom` on the sim-bridge lane under two rtabmap arms:
- **arm A,** as deployed;
- **arm B,** with loop-closure acceptance off.

The frame held in neither: arm A failed through closures, and arm B drifted with the accumulated error of visual
neighbour-link refinement. That record proposed two candidates built on the deployed configuration:
- **A′:** `RGBD/NeighborLinkRefining` false, with closures on;
- **A″:** refinement on, with `Vis/MinInliers` 40 (deployed: 15).

The frame-health record's eight rtabmap databases keep their sensor data. Each database is one **segment**, one bridge
boot under one reference: F1, F3 and F5 under arm A, and F2, its relaunches F2r and F2r2, F4 and F6 under arm B. This
record reprocesses them offline with `rtabmap-reprocess`: the same images, depth, features and odometry under each
configuration, with no bridge and no robot. It also reads each neighbour link's error directly from the databases'
Link and Node tables.

The matrix, its reads, a distributional control and the decision rule were committed (`ff3336f` in the evidence
repository) before any matrix run; one mechanics run preceded the registration (§1). A third arm, A‴, makes both
changes. It was registered by amendment (`78b1227`) after the registered matrix's 0.23.7 runs had been read, and
before any of its own runs.

The reads are the frame-health record's. A segment's **window** runs from its **reference**, the frame once the
warm-up spin has settled, to its last node result. **Drift** is the largest displacement, over four goals anchored in
odom, of a goal's map point against the reference. A **step** is a move of 0.5 m or more between consecutive
iterations. A run **holds** when its largest drift is 0.30 m or less and it has no step. A segment is **valid** when
the recorded run passes the control (§2).

**By the registered rule, neither A′ nor A″ holds.** A′ holds in all ten runs on one of the seven valid segments (F6),
and A″ on two (F2 and F6). The registered next step is therefore the refinement's scale shortfall against the camera
intrinsics, which the same databases can test. The two arms fail in different ways. A‴ (the amendment, chosen after
A′ and A″ were read, on the same data) holds in all ten runs on all eight segments, under both rtabmap builds.
On this lane, though, odometry is ground truth, and A‴ accepts few closures. It moves the frame by at most 0.079 m,
which is not the same as correcting a drifting wheel odometry.

Figures below are rtabmap 0.23.7's unless marked.
- **The control.** Seven of eight segments are valid.
  - The recorded verdict is the majority verdict of ten reprocessings of its database in all eight segments.
  - The recorded largest drift lies within the ten runs' range on six segments. It lies 0.010 m above it on F2r2,
    where the recorded run also steps once and none of the ten does, and 0.059 m above it on F2r.
  - F2r, a 14.5 s segment with no reach, is therefore invalid: 0.059 m exceeds the 0.05 m tolerance.
  - The recorded frame differs from the ten runs' by more than they differ from each other in five of eight segments,
    by up to 1.58×.
  - The tool cannot reproduce the recorded frame iteration for iteration, by construction (§1).
- **The deployed set** (arm A's configuration, run on all eight databases) fails in 70 of its 80 runs.
  - In all 18 of its failing runs whose largest step is 1 m or more, that step closes on a node created before the
    lane's warm-up spin ended (15 of 16 under 0.23.13).
  - The warm-up spin is the 360° turn the lane's perception service makes at bring-up; it covers nodes 1–24.
  - On F2 its median largest drift is 7.34 m (0.381 m under 0.23.13).
- **A′ (refinement off)** removes the refinement drift but not the wrong closures.
  - Away from F1, its median largest drift is at most 0.194 m.
  - It fails in 15 of 80 runs, 9 of them on F1. In 13 of the 15 the largest step is 1 m or more, and in 12 of those
    13 it closes on a spin node.
  - It accepts F1's wrong closure 142→16 in 5 of 10 runs.
- **A″ (`Vis/MinInliers` 40)** removes the metre-scale wrong closures but not the refinement drift.
  - No run has a step of 1 m or more, and none accepts 142→16.
  - It fails in 29 of 80 runs, 27 of them with no step, mainly through the drift of the refinements it keeps. F4,
    where only refinement moved the frame, holds in 0 of 10 runs, at 0.57–0.83 m.
- **A‴ (the amendment)** holds in all 160 runs, under 0.23.7 and 0.23.13. Its largest drift is 0.074 m (0.079 m
  under 0.23.13).
  - Per window its median run accepts at most 6 global closures and 4 proximity detections, and no refinement. On the
    seven segments other than F2r, the deployed set's median run accepts 4.5–38.5, 10.5–42 and 71–163.5.
  - On F2, F2r and F4 its median run accepts nothing. On F2r and F4 the frame does not move in any run.
- **rtabmap 0.23.13,** the robot image's build since 2026-10-19, gives the same outcome by the rule. A′'s one
  all-ten segment moves from F6 to F2, and A″ fails 41 of 80 runs, against 29.
  - Its refinement-driven drift varies far less from run to run.
  - Its accepted closures still split from run to run, and under the deployed set they add metre-scale outliers on F5
    and F6.
- **Per link, from the databases.**
  - Arm B's 545 refined links match the frame-health record's per-link read to 0.0007 mm.
  - Arm A's 511 refined links err by a median 12.4 mm and 0.255°.
  - The stored registration covariance overstates the yaw uncertainty about eightfold in arm B, and sevenfold in arm
    A (§6).

## 1. Method

- **Inputs.** The eight databases, each checked against the frame-health deposit's digest before use.
- **The tool.** `rtabmap-reprocess` from rtabmap 0.23.7, the version that wrote the databases (`Admin.version`).
  - The image is built from the ROS 2 Humble snapshot of 2026-09-14, the last to carry 0.23.7.
  - It feeds every Node row in id order, including the small-movement nodes rtabmap discarded (weight −9).
  - It reuses each node's stored features, and detects closures afresh rather than replaying the recorded ones.
  - Each run is a sealed container, with its input mounted read-only.
- **No two runs need be identical, and none is expected to reproduce the live one.** rtabmap's FLANN kd-trees
  shuffle with `std::random_device`. In both builds that touches the vocabulary, and the matching for global closures
  and proximity detections. In 0.23.7 it also touches refinement's guess-window search: the search for matches within
  40 px of each feature projected through the odometry guess. The registration stated this more narrowly, naming the
  vocabulary alone for 0.23.13.
  - So no reprocessing can reproduce the recorded frame iteration for iteration. Each configuration runs ten times,
    and the recorded run is compared with the ten (§2).
  - A run that accepts no closure and no refinement is not affected. For example, A‴ on F4 drifts 0.000 m in all ten
    runs.
  - A mechanics run made before the registration differed from the recorded frame by 62.9 mm and 0.93° at its 24th
    iteration (`analysis/mechanics_run_check.txt`). It covered F2r's first 24 nodes, the warm-up spin, on the `asis`
    input below, so that difference is not randomness alone. The registration declares it.
- **The odometry covariance.**
  - Live rtabmap received an odometry variance of 0.001 on every iteration of every database
    (`per-link/link_read.txt`), that is, information 1000·I.
  - The tool instead rebuilds each node's odometry covariance from its stored backward neighbour link. For a refined
    link, that holds the registration's covariance.
  - The primary inputs (`odomcov`) restore the live value: in a copy, every neighbour link's information matrix is set
    to 1000·I. The link transforms are not inputs, because each link is rebuilt from the node poses.
  - The unmodified inputs (`asis`) run the control as a comparison.
- **The parameter sets.** Each is the database's stored parameters, overridden on the command line, and read back from
  the output's Info row.
  - **REC:** no override. That is the deployed set on F1, F3 and F5, and arm B on the others.
  - **DEP:** the deployed values of the three keys arm B changed: `Rtabmap/LoopThr` 0.11, `RGBD/AggressiveLoopThr`
    0.05 and `RGBD/ProximityBySpace` true. On F1, F3 and F5, DEP is REC. "The deployed set" below means REC on those
    three and DEP on the others.
  - **A′:** DEP with `RGBD/NeighborLinkRefining` false.
  - **A″:** DEP with `Vis/MinInliers` 40. That key gates every visual registration, so A″ also raises the bar for
    global closures and proximity detections.
  - **A‴ (amendment):** DEP with both changes.
- **The matrix.** Ten runs of each configuration:
  - the primary set (0.23.7, `odomcov`): REC, A′ and A″ on F1, F3 and F5, and REC, DEP, A′ and A″ on the five arm-B
    databases;
  - the control on the `asis` inputs;
  - a replicate under rtabmap 0.23.13 (DEP, A′, A″);
  - A‴ under both versions.

  All 770 runs exited 0, processed every input node, and stored the overrides their set names.
- **The reads.** `map→odom` after each iteration is `Loop/MapToBase` composed with the inverse of the node's odometry
  pose (`Node.pose`⁻¹), read from the output's own Statistics and Node tables.
  - Each segment's window and reference, and the four goals in odom, come from the frame-health record.
  - Drift, steps of 0.5 m or more, and "holds" are that record's definitions, applied per iteration.
- **The reads were validated before any matrix run,** as the registration states; `analysis/validate_reads.txt`,
  written after the runs, prints that check. Applied to the eight recorded databases, the reads reproduce the
  record's largest drift to the millimetre, its step counts and its largest |yaw| to 0.01° in all eight segments. The
  share of the window over 0.30 m agrees to within 0.4 points; it is time-weighted here, against the record's
  per-sample share.

## 2. The control: the recorded frame against ten reprocessings

Primary set, REC. A segment is valid when two conditions hold:
- **C1:** the recorded verdict is the majority verdict of the ten runs;
- **C2:** the recorded largest drift lies within the runs' range, widened by 0.05 m.

The **frame difference** between two frames is the largest distance, over the window and the four goals, between
where they place a goal at the same iteration. The table gives its median over the ten runs (recorded against a run)
or over the 45 pairs (run against run).

| segment | recorded: largest drift (m), steps, holds | ten REC runs: hold of 10; largest drift range (m), median | C1 | C2 | valid | frame difference, recorded vs a run (m) | frame difference, run vs run (m) |
|---|---|---|---|---|---|---:|---:|
| F1 | 5.140, 7, no | 0; 0.522–7.417, 3.315 | yes | yes | yes | 5.386 | 5.413 |
| F3 | 0.734, 1, no | 0; 0.570–0.936, 0.607 | yes | yes | yes | 0.778 | 0.669 |
| F5 | 0.502, 1, no | 0; 0.459–0.616, 0.560 | yes | yes | yes | 0.430 | 0.498 |
| F2 | 0.404, 0, no | 0; 0.370–0.689, 0.438 | yes | yes | yes | 0.304 | 0.355 |
| F2r | 0.172, 0, yes | 10; 0.081–0.113, 0.088 | yes | no | no | 0.187 | 0.118 |
| F2r2 | 0.652, 1, no | 1; 0.284–0.642, 0.412 | yes | yes | yes | 0.605 | 0.393 |
| F4 | 0.898, 1, no | 0; 0.625–1.708, 1.154 | yes | yes | yes | 1.122 | 1.015 |
| F6 | 0.663, 0, no | 0; 0.394–0.913, 0.593 | yes | yes | yes | 0.803 | 0.633 |

- **Within the reprocessings' spread, at its edge in places.**
  - The recorded verdict is the ten runs' majority in all eight segments.
  - Its frame differs from theirs by more than they differ from each other in five of eight segments, by up to 1.58×
    (F2r).
  - On F2r2 it steps once where none of the ten does, and its drift lies just above all ten. It passes C2 through the
    0.05 m widening.
  - On F1 both differences reach about 5.4 m, because the reprocessings split on F1's metre-scale closures to spin
    nodes. Five of the ten runs drift 5.2–7.4 m: three through 142→16, and two through closures to nodes 15 and 22.
- **F2r is invalid by C2** and drops out of the decision. It is 14.5 s long, with no reach.
- **With the inputs as recorded (`asis`),** REC gives the same verdicts: only on F2r do all ten runs hold.
  - F1 is worse: a median 5.260 m, against 3.315 m with the covariance restored.
  - F2r2 holds in 2 of 10 runs, against 1 of 10.

## 3. The candidate arms, primary set (0.23.7, restored odometry covariance)

Each cell gives the runs that hold out of ten, then the largest drift's median and maximum, in metres.

| segment | recorded under | deployed | A′ | A″ | A‴ (amendment) |
|---|---|---|---|---|---|
| F1 | A | 0; 3.315, 7.417 | 1; 4.925, 5.145 | 2; 0.364, 0.546 | 10; 0.065, 0.071 |
| F3 | A | 0; 0.607, 0.936 | 9; 0.194, 0.372 | 6; 0.245, 0.459 | 10; 0.020, 0.036 |
| F5 | A | 0; 0.560, 0.616 | 9; 0.112, 3.921 | 6; 0.270, 0.507 | 10; 0.059, 0.074 |
| F2 | B | 0; 7.341, 7.408 | 9; 0.091, 4.897 | 10; 0.160, 0.189 | 10; 0.000, 0.022 |
| F2r | B | 10; 0.085, 0.096 | 10; 0.000, 0.000 | 10; 0.057, 0.095 | 10; 0.000, 0.000 |
| F2r2 | B | 0; 0.463, 6.482 | 9; 0.046, 6.116 | 7; 0.280, 0.458 | 10; 0.040, 0.042 |
| F4 | B | 0; 1.016, 2.006 | 8; 0.161, 1.020 | 0; 0.660, 0.834 | 10; 0.000, 0.000 |
| F6 | B | 0; 0.482, 0.721 | 10; 0.102, 0.135 | 10; 0.187, 0.278 | 10; 0.042, 0.048 |

**What failed, run by run** (`analysis/failures.txt`, read after the runs). A step's node is the node that its accepted
closure or proximity detection matched. A spin node is one created before the warm-up spin ended: nodes 1–24, in every
segment. Spin nodes cannot be told apart from the start point every reach returns to, as the frame-health record noted.

| arm | runs that fail, of 80 | with no step | whose largest step is 1 m or more | of those, closing on a spin node |
|---|---:|---:|---:|---:|
| deployed | 70 | 30 | 18 | 18 |
| A′ | 15 | 1 | 13 | 12 |
| A″ | 29 | 27 | 0 | — |
| A‴ | 0 | — | — | — |

**What each arm accepts.** The range over segments of the median run's count, per window:

| arm | global closures | proximity detections | neighbour refinements |
|---|---|---|---|
| deployed | 0–38.5 | 0–42 | 11–163.5 |
| A′ | 0–39 | 0–42.5 | 0 |
| A″ | 0–4.5 | 0–4 | 3–70 |
| A‴ | 0–6 | 0–4 | 0 |

- **A′** removes the refinement drift. Away from F1 its median largest drift is at most 0.194 m.
  - Away from F1 it keeps about as many closures as the deployed set, wrong ones included.
  - On F1 its median run accepts 8 global closures and 10 proximity detections, against the deployed set's 19 and 23.
    Over F1's ten runs it logs 201 wrong-loop-closure rejections.
  - It accepts F1's 142→16 in 5 of 10 runs, against 3 of 10 for REC.
- **A″** accepts few closures, and no run has a step of 1 m or more.
  - It keeps 3–70 refinements per window in the median run.
  - Its 27 failures without a step come mainly from that refinement drift, with some closure motion on F1, F3, F5
    and F2r2.
  - On F4 only refinement moved the frame. F4's median run accepts no closure, and F4 holds in none of its runs.
- **A‴ (the amendment)** keeps few closures and no refinement.
  - In every run of every segment, its largest drift is at most 0.074 m.
  - On F2, F2r and F4 its median run accepts nothing. On F2r and F4 the frame does not move in any run; on F2 the
    largest drift is 0.022 m.

## 4. The registered outcome, and the amendment

**The rule.** An arm holds on a segment when all ten of its runs hold. It is a candidate when it holds on every valid
segment.
- **A′** holds on F6 only, of the seven valid segments.
- **A″** holds on F2 and F6.

**Neither is a candidate.** The registered next step for that outcome is to measure the refinement's scale shortfall
(3.61 % along the motion, the frame-health record's §5) against the camera intrinsics, which the same databases can
test.

**The amendment.** A‴ holds in all ten runs on all eight segments, including F2r.
- **How to read it.** It was registered after the primary set was read, because the two failures looked separable. It
  is tested on the same eight databases that showed them, so it is a candidate, not a confirmed configuration.
- **What remains of the registered next step.** A‴ turns refinement off, so the scale shortfall does not bear on it.
  The shortfall still matters for any configuration that keeps refinement.

## 5. rtabmap 0.23.13

The robot image rebuilt on 2026-10-19 carries rtabmap 0.23.13, not the 0.23.7 that wrote these databases. Among its
source changes, two lie on refinement's path (`RegistrationVis.cpp`, `OptimizerG2O.cpp`), and a third in the FLANN
index that the vocabulary uses (`FlannIndex.cpp`):
- refinement's guess-window search becomes exact;
- bundle-adjustment outliers are counted per observation;
- the vocabulary's index is rebuilt by a different rule.

| segment | deployed | A′ | A″ | A‴ |
|---|---|---|---|---|
| F1 | 0; 5.250, 5.312 | 4; 4.916, 5.101 | 0; 0.447, 0.499 | 10; 0.071, 0.078 |
| F3 | 0; 0.534, 0.635 | 9; 0.183, 0.387 | 9; 0.288, 0.318 | 10; 0.031, 0.037 |
| F5 | 0; 0.665, 5.117 | 9; 0.121, 3.734 | 0; 0.338, 0.341 | 10; 0.067, 0.079 |
| F2 | 0; 0.381, 7.377 | 10; 0.062, 0.080 | 10; 0.127, 0.127 | 10; 0.000, 0.042 |
| F2r | 10; 0.082, 0.082 | 10; 0.000, 0.000 | 10; 0.070, 0.070 | 10; 0.000, 0.000 |
| F2r2 | 0; 0.418, 0.424 | 9; 0.048, 6.062 | 0; 0.358, 0.358 | 10; 0.039, 0.043 |
| F4 | 0; 1.032, 1.172 | 9; 0.130, 1.029 | 0; 0.723, 0.772 | 10; 0.000, 0.000 |
| F6 | 0; 0.541, 6.584 | 9; 0.101, 6.238 | 10; 0.166, 0.169 | 10; 0.041, 0.051 |

- **The same outcome by the rule.**
  - The deployed set fails in 70 of 80 runs. In 16 of them the largest step is 1 m or more, and in 15 of those it
    closes on a spin node.
  - A′ fails in 11 of 80. In 10 the largest step is 1 m or more, and in 9 of those it closes on a spin node.
  - A″ fails in 41 of 80, every one without a step.
  - A‴ (the amendment) fails in none.
- **Less variation in refinement-driven drift.** Several configurations give one largest drift in all ten runs: A″ on
  F2r2, 0.358 m, and the deployed set on F2r, 0.082 m.
- **Not less in closures.** Accepted closures still split from run to run. Under the deployed set they add metre-scale
  outliers on F5 (up to 5.117 m) and F6 (6.584 m) that 0.23.7 did not show.
- **By the rule, again neither A′ nor A″ holds** on every segment; A′'s one all-ten segment is F2 here, not F6.
  A‴, the amendment, holds on all eight.

## 6. Each neighbour link's error, from the databases (read after the runs, not pre-registered)

`per-link/link_read.py` reads each neighbour link k → j from the databases' Link and Node tables:
- **O,** odometry's delta odom_k⁻¹ · odom_j, the true link;
- **R,** the stored link;
- **E = O⁻¹ · R,** the link's error in j's frame.

The frame-health record's per-link read took each link's frame from a ride-along that sampled `map→odom` 0.5 s after
each rtabmap iteration. This read uses neither.

**Classes.** A link is classed by its new node's iteration:
- refined;
- rejected, with the odometry information kept;
- rejected, with the registration covariance ×100;
- not attempted: a small-movement node kept because a closure fired.

| segment | links | refined | rejected, odometry kept | rejected, ×100 | not attempted | spanning discarded nodes (nodes spanned) |
|---|---:|---:|---:|---:|---:|---|
| F1 | 228 | 180 | 24 | 22 | 2 | 41 (212) |
| F3 | 207 | 152 | 37 | 16 | 2 | 39 (168) |
| F5 | 231 | 179 | 32 | 17 | 3 | 49 (206) |
| F2 | 122 | 100 | 20 | 2 | 0 | 20 (144) |
| F2r | 33 | 30 | 2 | 1 | 0 | 2 (36) |
| F2r2 | 121 | 91 | 15 | 15 | 0 | 25 (125) |
| F4 | 218 | 152 | 40 | 26 | 0 | 36 (156) |
| F6 | 234 | 172 | 33 | 29 | 0 | 60 (362) |

- **Arm B's 545 refined links are the frame-health record's 545.**
  - Link by link, the two errors agree to 0.0007 mm and 0.00002°, so the 0.5 s delay lost nothing.
  - The databases' per-iteration `map→odom` agrees with the ride-along's frame to 3.0e-5 m and 8.7e-6 rad over every
    iteration of arm B's five segments (3.4e-5 m and 8.8e-6 rad over all eight). That is the precision at which rtabmap
    stores its statistics as text.
  - The chain of stored links reproduces the databases' frame to 2.7e-5 m.
  - So the frame-health record's per-link figures stand as exact:
    - a median 11.4 mm and 0.220°, with a 95th percentile of 48.4 mm and 1.232°;
    - 3.61 % short along the motion over the 322 links of 20 mm or more (3.25 % weighted by length, new here);
    - barely tied to the inlier count (Spearman −0.027 for translation, −0.149 for rotation).
- **Arm A from the databases: 511 refined links.** The ride-along read saw 417, because it left out closure
  iterations.
  - The error is a median 12.4 mm and 0.255°, with a 95th percentile of 53.4 mm and 1.395°.
  - Along the motion, the mean over the links of 20 mm or more is −1.60 % (z −1.4), not distinguishable from zero, or
    −1.89 % weighted by length. By segment it is +1.54 % (long) on F1, −3.57 % on F3 and −3.09 % on F5.
  - Links with 40 inliers or more err by a median 11.0 mm, against 13.2 mm below 40.
- **The discarded nodes.** Links whose ends span discarded small-movement nodes err like the others in B: a median
  11.5 mm against 11.4 mm. In A they err less: a median 9.1 mm against 13.3 mm.
- **Unrefined links equal odometry** to float32 precision: 3.4e-7 m and 1.3e-7 rad.
- **The stored covariance against the error.** The refined links' stored σ is a median 23.4 mm along x and 4.26° in
  yaw in B (23.3 mm and 4.22° in A). The RMS of the error over σ is:
  - 0.48 along x and 0.57 along y in B (0.49 and 0.63 in A);
  - 0.126 in yaw in B (0.143 in A).

  So the registration overstates its translation uncertainty 1.6–2.1-fold, and its yaw uncertainty about eightfold
  in B and sevenfold in A. σ along x and the translation error are rank-correlated at +0.588 in B.

## 7. What this means for the next measurements (a proposal)

- **The registered next step is the scale shortfall** against the camera intrinsics, and the same databases can test
  it offline. It bears on any configuration that keeps refinement, which A‴ does not.
- **On this lane, A‴ (the amendment) moves the frame by at most 0.079 m.**
  - The lane's odometry is ground truth. A‴'s few accepted closures move the frame by at most 0.079 m, within the
    0.30 m tolerance. Where it accepts nothing, the frame does not move.
  - That shows it adds little error here. It does not show A‴ corrects anything: its median run accepts 0–6 global
    closures per window, and none on F2, F2r and F4.
  - A lane session with A‴ against the deployed set could confirm the first statement, and could not test the second.
- **On the robot, the frame is the localisation.** Wheel odometry drifts, and the closures are what bound it. So
  A‴'s cost would show there, not here: whether a handful of 40-inlier closures per session keeps the frame where the
  room is. Settling that needs the real-robot gate's pre-registered SLAM read (`lane-slam-frame-health-2026-10-17`,
  §6), with A‴'s accepted closures counted beside the steps.
- **The warm-up spin.** When a failing deployed-set run's largest step is 1 m or more, that step closes on a spin node
  in 18 of 18 runs (15 of 16 under 0.23.13); for A′, 12 of 13 (9 of 10). The spin is a lane fixture: the robot's
  container stack does not start it.
  - A lane boot without the spin, under the deployed set, would show how many of those metre-scale steps depend on the
    spin.
  - Neighbour links join consecutive nodes, and drift is read against a reference taken after the spin, so the
    refinement drift is not expected to depend on it.
- **The build.** The robot now runs 0.23.13. Its verdicts agree here with 0.23.7's, but its refinement differs. A lane
  or robot session should record the rtabmap build it ran.

## 8. What is not claimed

- **No independent confirmation of A‴.** It was chosen after the primary set was read, and tested on the same data.
- **Nothing about the real robot.** The robot has no ground-truth odometry, does not run the warm-up spin, and is
  not reprocessed here.
- **No exact reproduction of the session.** The tool is randomised, and the recorded run is compared with ten
  reprocessings.
- **Nothing beyond these eight segments' data.** The data was recorded under arm A or arm B, and reprocessing replays
  the recorded trajectories. It cannot show how another configuration would have moved the robot.
- **The causes of the spin-node closures** are not shown. Spin nodes cannot be told apart from the start point every
  reach returns to.

## Evidence — deposit

`Sim2RealLab-Artifacts/lane-rtabmap-reprocess-2026-10-21/record-files/`, at commit
`0380202ab415602e25c2eb40684d8974d62c3d66`.
- **Registration:**
  - `PREREGISTRATION.md` at `ff3336f`, before any matrix run;
  - `AMENDMENT_A3.md` at `78b1227`, before any A‴ run.
- **Changes after the registration.** They are listed in `DEPOSIT.md`; none changes a registered read.
- **`SHA256SUMS`** lists every file's sha256 except its own and `DEPOSIT.md`'s. Its own sha256 is
  `1b4733801d5e9f6f1924f381d644566a578a858709be8717f9fadf084744e41f`.
- **Reproduce** from a clone of the evidence repository, with python3 and the standard library only:

      bash lane-rtabmap-reprocess-2026-10-21/record-files/reproduce.sh

  It regenerates every file in `analysis/` and `per-link/link_read.txt` byte for byte. It reads `runs/` (770 runs'
  extracts and logs), `recorded/` and `per-link/db_links/`, plus the frame-health deposit beside it.
- **Not deposited:** the input databases, the prepared inputs and the output databases. Their digests are in the
  deposit (`inputs.sha256`, and each run's manifest line).
