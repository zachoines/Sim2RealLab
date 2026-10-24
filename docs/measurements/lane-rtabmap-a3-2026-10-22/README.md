# A‴ against the deployed rtabmap set, live on the lane, 2026-10-22

[`lane-rtabmap-reprocess-2026-10-21`](../lane-rtabmap-reprocess-2026-10-21/README.md) (below, the reprocess record)
reprocessed the lane's eight rtabmap databases of 2026-10-17 offline. There **A‴**, the deployed rtabmap configuration
with `RGBD/NeighborLinkRefining` false and `Vis/MinInliers` 40, held SLAM's `map→odom` in all 160 of its runs. It was
chosen after the other candidates had been read, on the same data, and it accepts few closures. This record asks
whether A‴ holds live: closed loop, on new trajectories, on the robot images' current build (rtabmap 0.23.13), with the
deployed set as control. It also counts one deployed-set boot without the bring-up's warm-up spin.

The lane is [`lane-slam-frame-health-2026-10-17`](../lane-slam-frame-health-2026-10-17/README.md)'s (below, the
frame-health record): the sim bridge on the sim host, the robot's bridge-lane services on the robot host over the
direct link, the v3 policy through the deployed inference node, and four goals anchored in odom. Six boots alternate the
arms, A‴ in B1, B3 and B5 and the deployed set in B2, B4 and B6, and each boot reaches each goal once. B7 runs the
deployed set's file without the spin. The reads, the rule and the tools were committed (`030ff40` in the evidence
repository) before the first boot. The campaign ran on 2026-10-23, 18:45–23:16 CDT.

The reads are the frame-health record's. A boot's **reference** is the frame once the warm-up spin has finished and the
frame has held for 10 s sim. **Drift** is the largest displacement, over the four goals, of a goal's map point against
the reference. A **step** is a move of 0.5 m or more, between consecutive ride-along samples, in where the frame puts
any of the goals. A boot's **window** runs from its reference to its last node result. A boot **holds** when its largest
drift over the window is 0.30 m or less and it has no step.

**By the registered rule, A‴ holds: its three boots drift at most 0.148, 0.083 and 0.109 m, with no step, over 12 scored
reaches. The deployed set fails every boot: 0.497, 0.642 and 0.993 m, with one step each in B4 and B6.**

Three qualifiers bound what that shows.
- **On this lane odometry is ground truth.** Any motion of `map→odom` is error that SLAM adds. "Holds" therefore means
  that A‴ added at most 0.148 m here. It does not show that A‴ corrects a drifting odometry.
- **A‴ accepts few closures.** Per window it accepted 7–10 global closures and 5–11 proximity detections, and no
  refinement, against the deployed set's 18–42 global closures, 34–42 proximity detections and 141–145 refinements.
- **Transfer to the real robot is a separate question.** The robot's wheel odometry drifts, and its stack runs no
  warm-up spin. Whether a few 40-inlier closures bound that drift is the real-robot gate's SLAM read.

The other reads, in brief:
- **What moved the frame (read 2).** All but 0.0008 m of A‴'s frame motion came in iterations that accepted a closure or
  a proximity detection (69.4 % and 30.5 % of it), by at most 0.140 m in any one, and none before its reference. The
  deployed set's frame moved mostly in refinement-only iterations (61.1 %). Its two steps were global closures that
  brought a refinement-driven excursion back toward the reference. No step moved any boot's frame away from its
  reference (§3).
- **The warm-up spin.** Each deployed boot's frame moved 0.018–0.137 m and 0.91–4.33° during the spin, before its
  reference. Every A‴ boot, and B7 without the spin, set its reference with the frame unmoved (§3).
- **The terminal reads in odom (reads 3 and 4).** With A‴'s frame nearly static within a reach (in-reach goal excursion
  at most 0.015 m), the node stops at its 0.30 m radius and the odom and live distances agree to within 6.8 mm. Its
  count "within 0.30 m in odom" (5 of 12) therefore splits mostly on sub-millimetre differences: 10 of the 12 odom
  distances lie within 1 mm of 0.30 m, and the other two 5.1 mm below it and 6.7 mm above. The deployed set's odom
  distances move with its frame (§4).
- **B7, counted.** Its first reach drifted 0.246 m, through refinement. Over its whole run 12 iterations accepted a
  closure or proximity detection (12 proximity detections, one of them with a global closure in the same iteration),
  none onto a start-point node. Its next
  two transits pinned at the doorway, so the robot never returned to the start point after its first reach. B7 cannot
  show what removing the spin does to the deployed set's closures there (§5).
- **Departures.** B2's fourth transit pinned, and the bridge then hung. B2 was relaunched once, as B2r, under a count
  rule stated before it ran, with the live frame checks of B1–B6 in view. After the campaign, before any scorer ran,
  `odom_terminal.py` was fixed mechanically and `arm_summary.py` was changed to apply that count rule. An outside stall
  guard stopped pinned transits: B2's re-run from 19:57 CDT, every remaining transit from 22:35. A viewer and read-only
  analyses also ran on the sim host. Each departure is logged with its time (§1).
- **Defects.** The lane's transit tool never detects a pinned transit while its re-plans succeed (§6). The bridge hung
  once mid-run, after B2's last scored reach (§7).
- **Post hoc (2026-10-23, run during the campaign).** Two of this session's four goals, G1 and L2, lie outside the
  room. G1 has lain outside it since the Isaac Lab pair flip of 2026-09-12 (lane reaches from 2026-09-25); L2 lay outside
  it before the flip too, since the goals were first set. Both arms ran the same goals, so the comparison stands. The
  deployed set's failure rate on this lane may not transfer to a room without that exterior (§8).

## 1. Method and departures

### The lane

As the frame-health session (`PREREGISTRATION.md`, *Lane*):
- **Sim host.** The bridge `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0` at seed 42, through the Kit boot
  watchdog and the grid-dump wrapper.
  - Every boot's grid hashed `36929c68…`, the frame-health session's room (`boots/<RUN>/dgx/grid_dump_sha256.txt`).
  - In B5 and B2r the bridge stalled 61 s into its boot, and the watchdog relaunched it once (their `bridge.log`).
  - The planner and VLM services ran from the main checkout at `0a975ef`.
- **Robot host.** The five bridge-lane services: `inference`, `slam`, `navigation`, `sim-perception` and `autonomy`.
  - Images labelled `0423a13fcde6`, whose `strafer_slam` carries rtabmap 0.23.13 (stage 0's package build strings,
    `boots/<RUN>/robot/stage0/images.txt`).
  - The v3 policy artifact through the deployed inference node. Every boot's node-up line reads
    `vel_cap=(0.7841 m/s, 2.0933 rad/s) policy_loaded=True`.
  - The inference container was recreated before every reach.
- **Goals and starts,** in odom: G1 (−2.00, 2.25), L2 (−2.75, −2.25), R1 (1.4, 2.7) and R2 (3.1, 0.0).
  - G1 and L2 start from (0.075, −0.035), R1 and R2 from the spawn. Each start yaw puts the goal 45° to the left.
  - Each reach went through the frame-health session's odom-anchored wrapper.
- **The reference.** The `sim-perception` service spins the robot through 360° at bring-up (`donut_warmup`: 6.28 rad at
  0.60 rad/s, 11.22 s with its ramps; `boots/<RUN>/robot/stage0/slam_mount.txt`).
  - B7 launched that service with `donut_warmup:=false` (`boots/B7/robot/logs/nospin.txt`). Its sim-perception log has
    no warm-up line, and its settle logged `warmup=disabled`.

### The arms

Each arm mounts its file read-only over the installed `rtabmap_params.yaml` (`node/slam_params_overlay.yml`):
- **the deployed set** (labels `_A`): `node/rtabmap_params_image.yaml`, byte-equal to the image's file (sha256
  `e67c97ff…`);
- **A‴** (labels `_C`): `node/rtabmap_params_C.yaml` (`c355d811…`), which differs from it in two lines,
  `RGBD/NeighborLinkRefining: "false"` and `Vis/MinInliers: "40"`.

In every boot:
- the host and container sha256 of the mounted file agreed (`boots/<RUN>/robot/stage0/slam_mount.txt`);
- the running node's parameter dump read the arm's values for eight keys, `Vis/MinInliers` 15 or 40 among them
  (`PARAMS OK`), beside rtabmap's own `Setting` lines (`boots/<RUN>/robot/logs/rtabmap_params_check.txt`);
- rtabmap started a fresh map (`local map=1`) under the boot's own token, `a31022<boot>`
  (`boots/<RUN>/robot/stage0/slam_mount.txt`, its "a fresh map" section, and `logs/strafer_slam.log`).

### Boots

| boot | arm | goal order | reference (s sim) | scored reaches |
|---|---|---|---:|---:|
| B1 | A‴ | G1 R1 L2 R2 | 14.6 | 4 |
| B2 | deployed | G1 R1 L2 R2 | 24.5 | 3 (replaced by B2r) |
| B3 | A‴ | R1 L2 R2 G1 | 13.8 | 4 |
| B4 | deployed | R1 L2 R2 G1 | 24.6 | 4 |
| B5 | A‴ | L2 R2 G1 R1 | 15.1 | 4 |
| B6 | deployed | L2 R2 G1 R1 | 24.4 | 4 |
| B7 | the deployed set's file, no spin | G1 R1 L2 R2 | 12.3 | 1 (counted, not scored) |
| B2r | deployed (B2's relaunch) | G1 R1 L2 R2 | 24.4 | 4 |

The scorers are the frame-health session's. The registration ran `frame_health.py` and `odom_terminal.py` unchanged,
each byte-equal to that deposit's, and `arm_summary.py` adapted to this session's boots and arms. After the campaign
`odom_terminal.py` took one mechanical fix, and `arm_summary.py` one change that applies the count rule (item 8 below).
`startpoint_count.py` reads B7's count.

### Departures from the registration

Each is logged with its time (CDT, 2026-10-23) in `campaign/interventions.log`, `campaign/campaign.log`,
`campaign/viewer_start.txt`, or the boot's `chain.log` and `boot.log`; the audit's times are in
`posthoc_outside_room/workflow1_exterior_audit.json`.
1. **Other work on the sim host,** which the registration did not provide for.
   - **A viewer, 19:13:48–23:16:38.** A viewer process ran on the sim host from the B1/B2 boundary to the campaign's end
     (`campaign/viewer_start.txt`). B1 ran without it. It ran during two of the three A‴ boots (B3, B5) and every
     deployed boot, so it does not line up with the arm.
   - **Read-only analyses, from 20:29.** The §8 audit ran from 20:29 to 21:33, during B3's last minutes, B4 and B5.
     Further read-only analyses, not part of this record, ran from 21:42, during B6, B7 and B2r (their files' times on
     the sim host; they are not deposited).
   - **Load, read after the runs.** The robot host's 1-min load over each boot's log: a mean of 9.07–10.15 and a maximum
     of 14.10–17.77 in the seven boots with the viewer, against 9.57 and 14.96 in B1
     (`boots/<RUN>/robot/logs/loadavg.log`). The sim host's 1-min load (`load1` in `host/telemetry_session.csv`)
     averaged 2.32 in B1 and 2.35–3.50 in the other boots, with a maximum of 8.21 (B5); over the audit's run it averaged
     3.57. Its GPU utilisation had a median of 38–40 % in every boot.
   - No read here measures the effect of either on rtabmap or on a reach.
2. **B2's pinned transit, stopped by hand at 19:56:42.** B2 scored three reaches. Slot 4's transit (`B2_S4_R2_A`), back
   from L2, pinned on the doorway's south jamb at map (−1.423, −0.455) from about 181 s sim.
   - Its first attempt ran to the tool's budget: `BUDGET_SIM` after 150.6 s sim and 1144 s wall, with 29 re-plans and
     no escape.
   - Its retry was stopped with SIGTERM, on the maintainer's word, after 24.7 s sim without movement (`INTERRUPTED`).
   - The protocol's own outcome followed: nothing submitted, not scored, and the reach re-run.
3. **A stall guard, stated at 19:57:28, before the re-run's transit.** `campaign/intervention/stall_guard.py` applies,
   from outside, a stall rule modelled on the transit tool's (three 8 s stalls without 0.03 m of progress), measured on
   the straight-line distance to the target rather than the re-planned path's remaining length.
   - The rule: an attempt in transit, more than 0.5 m from its target, whose straight-line distance to the target has
     not fallen by 0.03 m for 24 s sim, is stopped with SIGTERM.
   - It fired once, at 20:02:35, on the re-run's first attempt: best distance 1.417 m at 350.0 s sim, then 24.4 s sim
     without progress.
   - The re-run's retry ended in the bridge's mid-run hang (§7). Two failed transits in a row stopped B2's chain
     (exit 6, 20:06:01).
4. **B2, torn down by hand, 20:07:17–20:08:12,** with the boot's own steps: the frame read after the last reach,
   `fh_up.sh down` (rtabmap's database saved, 289 MB), and the bridge's stop. The hung Kit process needed SIGKILL
   (20:08:37). The campaign resumed with B3 at 20:08:57, as a new systemd unit.
5. **B2's relaunch, B2r, under a count rule stated at 22:12:29, before it ran** (`AMENDMENT_RELAUNCH.md`).
   - The rule was stated 24 s after B6's last frame check (22:12:05), with the live frame checks of B1–B6 in view
     (each boot's `chain.log` FRAME lines, at its slot boundaries). Every A‴ check read within 0.30 m; B2, B4 and B6 each
     had at least one read over 0.30 m.
   - With B2's three scored reaches and four each from B4 and B6, the deployed arm could reach at most 11 of the 12 the
     rule needs. The registration allows one relaunch per boot before an arm is void.
   - The count rule: B2r is B2's one relaunch, and a boot counts once. B2r's scored reaches take B2's place in the arm's
     count, and B2 is reported in full beside them.
   - B2r ran B2's manifest, so its records carry B2's labels, under its own run directory and token (`a31022B2r`) and
     with the spin, 22:48:10–23:16:10. It scored 4 of 4.
   - A‴'s outcome does not depend on B2r. Every deployed boot fails, B2 (0.588 m) and B2r (0.497 m) alike (§2). The
     relaunch decides only whether the control reaches the 12 scored reaches the void clause needs.
6. **The stall guard, extended at 22:35:36** to every remaining transit (B7 and B2r), after the maintainer reported B7
   pinned at the doorway's north jamb (`stall_guard_campaign.py`, the same thresholds).
   - It stopped all four of B7's slot-2 and slot-3 attempts. The first had made no progress for 113.9 s sim when the
     guard was armed.
   - It did not fire in B2r. Two failed transits in a row stopped B7's chain (exit 6, 22:46:50).
7. **B7, torn down by hand, 22:47:13–22:47:55,** as B2 was (database saved, 213 MB).
8. **The amendment (`AMENDMENT_RELAUNCH.md`), written after the campaign and before any scorer ran.** It and the
   scorer changes were written with every boot's live frame checks in view, B7's and B2r's included (B7 had one read
   over 0.30 m, B2r two). Neither scorer change touches a definition or a threshold. In the evidence repository,
   `git diff 030ff40 eeac8d9 -- lane-rtabmap-a3-2026-10-22/record-files/odom_terminal.py
   lane-rtabmap-a3-2026-10-22/record-files/arm_summary.py` shows both in full.
   - **`odom_terminal.py`, a mechanical fix.** Its summary named the frame-health session's arms, A and B only, so A‴
     (C) would have had no read-4 summary. After the fix it loops over C, A, N and B, and each row carries its boot.
   - **`arm_summary.py`, the count rule applied.** As registered it had no place for a relaunch. It gains B2r, and
     reads B2r in B2's place for the arm's boots, its 12-reach count and its verdict (`REPLACED`), with B2's row
     beside.
   - **The windows of the stopped chains.** B2's and B7's registered windows end at their last node results, slot 3's
     and slot 1's. Each is also read, descriptively, to its ride-along's last sample
     (`analysis/frame_health_<RUN>_to_ridealong_end.*`). Only the registered window enters the rule.
   - **B7's count** had no named tool. `startpoint_count.py` (new) reads it as the registration defines it (§5).

## 2. The frame against each boot's reference (read 1), and the rule

| boot | arm | window (s sim) | largest drift (m) | share > 0.30 m | steps ≥ 0.5 m | holds | largest translation (m), \|yaw\| (°) vs reference | global closures | proximity | refinements | nodes added | distance (m) |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| B1 | A‴ | 197.3 | 0.148 | 0.0 % | 0 | yes | 0.071, 1.53 | 10 | 11 | 0 | 393 | 43.2 |
| B3 | A‴ | 156.2 | 0.083 | 0.0 % | 0 | yes | 0.020, 1.09 | 8 | 5 | 0 | 311 | 42.2 |
| B5 | A‴ | 270.2 | 0.109 | 0.0 % | 0 | yes | 0.025, 1.37 | 7 | 8 | 0 | 539 | 43.5 |
| B2r | deployed | 167.8 | 0.497 | 15.5 % | 0 | no | 0.333, 3.95 | 27 | 36 | 141 | 334 | 44.9 |
| B4 | deployed | 160.8 | 0.642 | 43.2 % | 1 | no | 0.305, 6.51 | 18 | 34 | 145 | 321 | 44.3 |
| B6 | deployed | 215.3 | 0.993 | 43.1 % | 1 | no | 0.426, 11.27 | 42 | 42 | 144 | 429 | 43.5 |
| B2 (replaced) | deployed | 119.2 | 0.588 | 28.7 % | 0 | no | 0.485, 4.42 | 10 | 18 | 92 | 237 | 32.8 |
| B7 (counted) | no spin | 23.2 | 0.246 | 0.0 % | 0 | yes (counted, not scored) | 0.156, 1.45 | 0 | 0 | 20 | 45 | 6.5 |

Sources: `analysis/arm_summary.txt` and `analysis/frame_health_<RUN>.txt`.
- **The share above 0.30 m** is the share of the window's ride-along samples.
- **Closures, proximity detections and refinements** count the window's iterations that accepted one. One iteration can
  count in more than one column.
- **Distance** is rtabmap's own `Memory/Distance_travelled` over the window.
- **To the ride-along's last sample** (descriptive): B2, to 391.4 s sim, drifts 1.231 m at 200.3 s sim, with 77.5 % of
  samples above 0.30 m and no step. B7, to 259.6 s sim, drifts 0.362 m at 214.4 s sim, with 38.3 % above 0.30 m and no
  step.

**The rule** (`PREREGISTRATION.md`). A‴ holds when every A‴ boot (at least two) has a largest drift of 0.30 m or less
and no step of 0.5 m or more, over at least 12 scored reaches in the arm.
- **A‴ holds.** B1, B3 and B5 each hold, over 12 scored reaches.
- **The deployed set, read by the same rule as the control, fails every boot.** B2r, B4 and B6 each drift past 0.30 m,
  and B4 and B6 each take one step. Its 12 scored reaches count B2r in B2's place. B2 fails as well (0.588 m).
- **The control fails in all three boots,** so the registered branch for a control that holds in all three ("the lane
  cannot separate the arms") does not apply.
- **The registered consequence:** "A‴ holds: it becomes the lane's configuration, and the tracked
  `rtabmap_params.yaml` change follows in its own PR" (§10).
- **Live against offline (descriptive).** Offline, A‴'s largest drift was 0.079 m under 0.23.13 and 0.074 m under
  0.23.7, with the odometry covariance restored, and 0.084 m on the as-recorded inputs (0.23.7, the reprocess record's
  §9). Its median run accepted at most 6 global closures and 4 proximity detections per window (the reprocess
  record). Live, it drifted up to 0.148 m (B1) and accepted 7–10 and 5–11. The two read different trajectories and
  windows.

## 3. What moved the frame (read 2)

**Motion by iteration class.** Each iteration's change of the frame is measured at the goals, as a step is, and summed
over the window. The sum is a path length, not net drift. The classes are exclusive, in the order global closure,
proximity (no global closure), refinement only, nothing accepted.

| boot | arm | global closure | proximity | refinement only | nothing accepted | total |
|---|---|---|---|---|---|---:|
| B1 | A‴ | 10 it., 0.34 m (67.0 %) | 4 it., 0.17 m (32.9 %) | — | 379 it., 0.0003 m | 0.51 m |
| B3 | A‴ | 8 it., 0.20 m (99.9 %) | — | — | 303 it., 0.0003 m | 0.20 m |
| B5 | A‴ | 7 it., 0.24 m (57.6 %) | 3 it., 0.17 m (42.3 %) | — | 529 it., 0.0003 m | 0.41 m |
| A‴, pooled | | 25 it., 0.77 m (69.4 %) | 7 it., 0.34 m (30.5 %) | — | 1211 it., 0.0008 m | 1.11 m |
| B2r | deployed | 27 it., 1.92 m (25.9 %) | 12 it., 0.86 m (11.7 %) | 109 it., 4.62 m (62.4 %) | 186 it., 0.0014 m | 7.41 m |
| B4 | deployed | 18 it., 1.68 m (20.7 %) | 16 it., 0.91 m (11.2 %) | 112 it., 5.53 m (68.1 %) | 175 it., 0.0025 m | 8.12 m |
| B6 | deployed | 42 it., 3.72 m (43.1 %) | 3 it., 0.30 m (3.5 %) | 101 it., 4.61 m (53.4 %) | 283 it., 0.0011 m | 8.64 m |
| deployed, pooled | | 87 it., 7.32 m (30.3 %) | 31 it., 2.08 m (8.6 %) | 322 it., 14.76 m (61.1 %) | 644 it., 0.0050 m | 24.17 m |
| B2 (replaced) | deployed | 10 it., 0.50 m (8.7 %) | 8 it., 1.74 m (30.1 %) | 79 it., 3.55 m (61.3 %) | 140 it., 0.0004 m | 5.79 m |
| B7 (counted) | no spin | — | — | 20 it., 0.56 m (100 %) | 25 it., 0.0000 m | 0.56 m |

The per-boot rows are each `analysis/frame_health_<RUN>.json`'s `motion_by_iteration_class`, with the shares taken over
the boot's total; the pooled rows sum them over the arm's boots, as `analysis/arm_summary.txt` does. An iteration that
accepted both a global closure and a proximity detection counts as a global closure, so the proximity class is smaller
than §2's proximity count.
- **All but 0.0008 m of A‴'s frame motion came in iterations that accepted a closure or proximity detection.** Its 1211
  iterations that accepted nothing moved the frame 0.0008 m in all. Its largest move in one iteration was 0.140 m (B1,
  a global closure).
- **A‴ refined nothing.** With `RGBD/NeighborLinkRefining` false that holds by construction: a manipulation check.
- **The deployed set's frame moved mostly in refinement-only iterations,** 53.4–68.1 % of its motion per boot (a path
  length, as above).
  - Each deployed boot's largest drift came in an excursion above 0.30 m that moved in refinement-only iterations:
    B2r's 0.497 m (190.3–192.2 s sim), B4's 0.642 m (104.3–115.7 s sim) and B6's 0.993 m (114.8–122.3 s sim). So did
    every other excursion of B4's, and all but one of B2r's: its 184.3–184.7 s sim excursion also held four global
    closures (the excursion lines of `analysis/frame_health_<RUN>.txt`). The frame-health record's arm B (closures
    off) drifted through refinement on 2026-10-17 as well.
  - B6's first excursion (92.8–93.8 s sim, peak 0.386 m) came in global closures (0.358 m) and one proximity detection
    (0.109 m), and its three short excursions at 126.4–129.8 s sim mainly in global closures. Its last (136.3–216.3 s
    sim, peak 0.598 m) moved mainly in refinement-only iterations.
  - No closure moved the frame away from the reference by 0.5 m or more, as the frame-health record's F1 closure
    142→16 did under its arm A, the deployed set.
- **No wrong-closure rejection.** No "wrong loop closure … after graph optimization" line was logged in any boot
  (`boots/<RUN>/robot/logs/strafer_slam.log`).
- **More registration rejections under A‴,** 46, 93 and 62 per window, against the deployed set's 17, 66 and 22 (B2r,
  B4, B6). These count rtabmap's "Rejected loop closure" warnings in `strafer_slam.log` within the window
  (`frame_health.py`'s `registration_rejected`, under `warn_window` in `analysis/frame_health_<RUN>.json`).

**The deployed set's two steps** (`analysis/frame_health_B4.txt`, `analysis/frame_health_B6.txt`). Each moved the frame
toward the reference. Each fell in an iteration that accepted a global closure, and that iteration's frame matches the
step's after-sample to 0.000 m.
- **B4, 0.585 m at 115.8 s sim,** closure 229→20: 0.378 m of translation and +4.13° of yaw, taking the drift from
  0.642 m to 0.231 m. It ended an excursion above 0.30 m (104.3–115.7 s sim, peak 0.642 m) whose motion came in 27
  refinement-only iterations, 1.09 m in all.
- **B6, 0.801 m at 122.3 s sim,** closure 242→186: 0.447 m and +7.24°, the drift from 0.993 m to 0.259 m. It ended an
  excursion (114.8–122.3 s sim, peak 0.993 m) whose motion came in 12 refinement-only iterations, 1.29 m.
- **Both came between reaches.** B4's fell after its L2 reach (result at 88.9 s sim) and before its R2 goal was sent
  (124.0 s sim). B6's fell after its R2 reach (106.3 s sim) and before G1 (136.3 s sim).
- **Node 20 is a spin node and a start-point node** (both defined below). Node 186 is neither.
- **Neither step decides a verdict.** Both boots fail on drift alone (0.642 and 0.993 m).

**The warm-up spin.** The frame's motion from the ride-along's first sample to the reference ("before the reference" in
`analysis/frame_health_<RUN>.txt`):

| boot | arm | the frame's motion before the reference | its last change (s sim) |
|---|---|---|---:|
| B1, B3, B5 | A‴ | 0.000 m, 0.00° | none |
| B2 | deployed | 0.137 m, +1.47° | 13.3 |
| B2r | deployed | 0.063 m, +0.91° | 13.2 |
| B4 | deployed | 0.095 m, +2.17° | 13.3 |
| B6 | deployed | 0.018 m, +4.33° | 13.3 |
| B7 | no spin | 0.000 m, 0.00° | none |

- **No boot accepted a closure or proximity detection before its reference.** Every accepted one in
  `analysis/frame_health_<RUN>.json`'s `events` lies after it.
- **The deployed boots' spin motion came in refinements.** Each accepted 19 or 20 refinements before its reference.
  The A‴ boots and B7 accepted none (read after the runs, from `boots/<RUN>/robot/logs/ridealong/info_<run>.jsonl`).
- **A‴'s and B7's references hold `map→odom` at zero,** within 1e-8 m and 0.0001° (`boots/<RUN>/robot/frame_ref.json`).
  They came at 12.3–15.1 s sim. The deployed boots' came at 24.4–24.6 s sim, after 10 s sim without a frame change
  following the spin's last, at 13.2–13.3 s sim.

**Closures onto spin and start-point nodes.** A **spin node** is one rtabmap created before the warm-up spin ended:
nodes 1–24 in every boot with the spin. A **start-point node** is one whose odometry pose, read from the boot's
database, lies within 0.30 m of the spawn. The counts are of iterations that accepted a global closure or a proximity
detection, over each boot's whole info ride-along (`analysis/arm_summary.txt`, `analysis/startpoint_count.txt`).

| boot | arm | accepted iterations | onto a spin node | onto a start-point node |
|---|---|---:|---:|---:|
| B1 | A‴ | 14 | 11 | — |
| B3 | A‴ | 8 | 4 | — |
| B5 | A‴ | 10 | 9 | — |
| B2r | deployed | 39 | 19 | 39 |
| B4 | deployed | 34 | 14 | 30 |
| B6 | deployed | 45 | 34 | 38 |
| B2 (replaced) | deployed | 18 | 6 | 17 |
| B7 (counted) | no spin | 12 | (no spin) | 0 |

- **The start-point read was registered for B7 and the deployed boots only.** B2's spin-node count applies
  `arm_summary.py`'s definition to `analysis/frame_health_B2.json`; the replaced boot's summary does not print it.
- **A‴ also closes mostly onto spin nodes,** 24 of its 32 accepted iterations. No such closure moved its frame by more
  than 0.140 m.
- **The deployed boots' closures onto start-point nodes caused no step away from the reference this session.**
- **The spin turns the robot in place at the spawn,** within 0.09 m of every reach's start, so the boots with the spin
  cannot separate its nodes from the start point's (the frame-health record, §3).

## 4. The terminal reads in odom (reads 3 and 4)

Per reach, the node's map-frame goal is mapped into odom once, at goal-sent, and the robot is read from the odom series
from goal-sent to the node's result. The definitions are the frame-health record's (§4):
- band entry: within 0.42 m;
- crossing: within 0.30 m;
- band time: from band entry to the node's result;
- pre-crossing stop-short: 10 consecutive `/cmd_vel` ticks below 0.05 m/s in the band, before the crossing or with no
  crossing.

The live reads use the same definitions on the node's map frame.

| arm | scored | node `SUCCEEDED` | odom at the result (m): range, median | live at the result (m): range | largest \|odom − live\| (m) | within 0.30 m in odom at the result | pre-crossing stop-shorts, odom (live) | band median, odom (live) | largest in-reach goal excursion (m) |
|---|---:|---:|---|---|---:|---:|---|---|---:|
| A‴ (B1, B3, B5) | 12 | 12 | 0.295–0.307, 0.300 | 0.299–0.301 | 0.007 | 5 | 8 of 12 (8 of 12) | 3.58 s (3.58 s) | 0.015 |
| deployed (B2r, B4, B6) | 12 | 12 | 0.175–0.391, 0.280 | 0.278–0.302 | 0.125 | 8 | 8 of 12 (9 of 12) | 4.67 s (4.14 s) | 0.174 |
| B2 (replaced) | 3 | 3 | 0.122–0.198, 0.164 | 0.212–0.300 | 0.177 | 3 | 0 of 3 (1 of 3) | 2.73 s (1.54 s) | 0.177 |
| B7 (counted) | 1 | 1 | 0.157 | 0.300 | 0.143 | 1 | 0 of 1 (1 of 1) | 3.27 s (2.86 s) | 0.166 |

Source: `analysis/odom_terminal.json`, its rows keyed by `boot`, with its per-arm definitions. The A‴ row is
`odom_terminal.py`'s printed summary for arm C. The deployed, B2 and B7 rows were computed after the runs from
`odom_terminal.json`'s rows by the same definitions; no deposited tool prints them.
- **`odom_terminal.py`'s printed per-arm summary pools B2 with B2r.** For arm A it reads 15 scored reaches, 11 within
  0.30 m in odom at the result, 8 of 15 pre-crossing stop-shorts, a band median of 3.73 s (live 3.63 s) and a largest
  excursion of 0.177 m (`analysis/odom_terminal.txt`). The deployed row above leaves B2 out, as the rule's count does.
- **A‴'s count sits on a knife edge.** With its frame nearly static within a reach, its live and odom distances agree to
  under 1 mm in 9 of 12 reaches. In the other three they differ by 1.2, 4.4 and 6.8 mm, each within that reach's
  in-reach goal excursion.
  - The node ends each reach at its 0.30 m arrival radius on the live frame, so the odom distance at the result sits on
    the read's 0.30 m threshold. Five lie at or below it (0.2949–0.2999 m); seven lie 0.01–6.7 mm above it, six of them
    within 1 mm.
  - "5 of 12 within 0.30 m" is therefore a split at the node's own 0.30 m arrival radius, not a measure of arrival.
  - The same edge enters the stop-short count. A reach whose odom distance never falls to 0.30 m has no crossing, so its
    own arrival stop can count: B1_S1_G1_C's does, at 0.301 m. A‴'s other seven quiet stops lie at 0.318–0.413 m.
- **The deployed set's odom distances move with its frame.** Its frame carried the goal by up to 0.174 m during a reach.
  In every one of its reaches, the odom and live distances differ by no more than that reach's goal excursion.
  - Its lower odom distances are real on this lane, where odom is ground truth: in B4_S1_R1_A the robot ended 0.175 m
    from the goal in odom, 0.300 m live.
  - They come from the frame moving the goal the node tracked (by 0.160 m in that reach), not from the node or the
    policy stopping nearer: the node still stopped at its 0.30 m radius on the live frame.
- **Every node result was `SUCCEEDED`, and no reach fell back to Nav2** (`nav2_fallback_lines=0` in every chain log).
- **Not scored:** B2's slot 4 and its re-run (transits pinned, §1); B7's slots 2 and 3 (transits pinned) and slot 4,
  which never ran because the chain had stopped.
  - The registration re-runs a reach that is not scored once at the end of its boot. The registered chain stop (two
    failed transits in a row, exit 6) pre-empted B7's re-runs, and B7 was not relaunched. It is counted, not scored, so
    the void clause does not bind, and its count rests on one reach.
- **Odom anchoring kept the goals** within 0.062 m of their anchors at goal-sent, except B2_S1_G1_A (0.397 m), whose
  frame moved between the wrapper's mapping and goal-sent.

Each reach, with its goal's location (inside or outside the room, read after the runs, §8):

| reach | boot | goal | node: live at the result (m) | odom at the result (m) | in-reach goal excursion (m) | goal at sent vs anchor (m) | stop-short, odom (quiet stop, m); live |
|---|---|---|---:|---:|---:|---:|---|
| B1_S1_G1_C | B1 | G1, out | 0.3008 | 0.3008 | 0.000 | 0.000 | yes (0.301); yes |
| B1_S2_R1_C | B1 | R1, in | 0.2998 | 0.3010 | 0.007 | 0.047 | no; no |
| B1_S3_L2_C | B1 | L2, out | 0.3001 | 0.3000 | 0.000 | 0.000 | yes (0.400); yes |
| B1_S4_R2_C | B1 | R2, in | 0.2997 | 0.2997 | 0.000 | 0.010 | no; no |
| B3_S1_R1_C | B3 | R1, in | 0.2995 | 0.2995 | 0.000 | 0.000 | no; no |
| B3_S2_L2_C | B3 | L2, out | 0.2998 | 0.2998 | 0.000 | 0.000 | yes (0.327); yes |
| B3_S3_R2_C | B3 | R2, in | 0.2994 | 0.2949 | 0.006 | 0.001 | no; no |
| B3_S4_G1_C | B3 | G1, out | 0.2998 | 0.2999 | 0.000 | 0.000 | yes (0.326); yes |
| B5_S1_L2_C | B5 | L2, out | 0.3004 | 0.3004 | 0.000 | 0.000 | yes (0.329); yes |
| B5_S2_R2_C | B5 | R2, in | 0.3001 | 0.3000 | 0.000 | 0.028 | yes (0.318); yes |
| B5_S3_G1_C | B5 | G1, out | 0.3001 | 0.3001 | 0.000 | 0.000 | yes (0.398); yes |
| B5_S4_R1_C | B5 | R1, in | 0.2999 | 0.3067 | 0.015 | 0.017 | yes (0.413); yes |
| B2_S1_G1_A | B2r | G1, out | 0.3022 | 0.2761 | 0.082 | 0.011 | no; yes |
| B2_S2_R1_A | B2r | R1, in | 0.3000 | 0.3031 | 0.071 | 0.062 | no; no |
| B2_S3_L2_A | B2r | L2, out | 0.2989 | 0.2497 | 0.145 | 0.014 | yes (0.326); yes |
| B2_S4_R2_A | B2r | R2, in | 0.3000 | 0.2893 | 0.076 | 0.015 | no; no |
| B4_S1_R1_A | B4 | R1, in | 0.2996 | 0.1749 | 0.160 | 0.007 | no; no |
| B4_S2_L2_A | B4 | L2, out | 0.3000 | 0.2516 | 0.174 | 0.020 | yes (0.327); yes |
| B4_S3_R2_A | B4 | R2, in | 0.3000 | 0.2764 | 0.071 | 0.038 | yes (0.395); yes |
| B4_S4_G1_A | B4 | G1, out | 0.3002 | 0.2806 | 0.075 | 0.058 | yes (0.328); yes |
| B6_S1_L2_A | B6 | L2, out | 0.3000 | 0.3907 | 0.091 | 0.013 | yes (0.410); yes |
| B6_S2_R2_A | B6 | R2, in | 0.3000 | 0.3364 | 0.163 | 0.026 | yes (0.384); yes |
| B6_S3_G1_A | B6 | G1, out | 0.2779 | 0.3412 | 0.134 | 0.040 | yes (0.357); yes |
| B6_S4_R1_A | B6 | R1, in | 0.2999 | 0.2786 | 0.087 | 0.043 | yes (0.397); yes |
| B2_S1_G1_A | B2 | G1, out | 0.2118 | 0.1635 | 0.051 | 0.397 | no; no |
| B2_S2_R1_A | B2 | R1, in | 0.2993 | 0.1220 | 0.177 | 0.015 | no; no |
| B2_S3_L2_A | B2 | L2, out | 0.3000 | 0.1976 | 0.153 | 0.009 | no; yes |
| B7_S1_G1_N | B7 | G1, out | 0.3004 | 0.1569 | 0.166 | 0.013 | no; yes |

## 5. B7, the spin-less deployed boot (counted)

The registration counts, for B7, its accepted closures and proximity detections onto start-point nodes and its steps of
0.5 m and of 1 m or more, with the same counts for the deployed boots beside them.
- **The frame was unmoved at its reference.** B7 ran no warm-up process (§1). Its reference came at 12.3 s sim, with
  `map→odom` at zero and no change before it (§3).
- **Its window is one reach** (G1): 12.3–35.5 s sim, 23.2 s sim.
  - Largest drift 0.246 m at 34.8 s sim; no step; no sample above 0.30 m.
  - No closure or proximity detection was accepted. All of the frame's motion came in its 20 refinement-only
    iterations, 0.56 m in all.
- **The registered count** (`analysis/startpoint_count.txt`):

  | boot | iterations with an accepted closure or proximity detection: whole ride-along / window | of them onto a start-point node: whole / window | steps ≥ 0.5 m in the window | of them ≥ 1 m |
  |---|---|---|---:|---:|
  | B7 (no spin) | 12 / 0 | 0 / 0 | 0 | 0 |
  | B2 | 18 / 18 | 17 / 17 | 0 | 0 |
  | B2r | 39 / 39 | 39 / 39 | 0 | 0 |
  | B4 | 34 / 34 | 30 / 30 | 1 | 0 |
  | B6 | 45 / 45 | 38 / 38 | 1 | 0 |

- **B7 never returned to the start point.** Its slot-2 transit, back from G1, pinned at the doorway's north jamb, and
  slot 3's transit, from there toward L2's start, pinned there too.
  - The four attempts ended at map x −1.11 to −1.14 and y +0.36 to +0.41 (`boots/B7/robot/logs/runs/*.to_start.out`).
    Slot 2's first attempt made no progress from 56.9 s sim until the extended guard stopped it at 170.8 s sim.
  - B7's 12 accepted iterations came after its window, at 218.3–257.8 s sim, while the robot sat at the doorway. They
    matched nodes 106–247, none of them a start-point node (`analysis/frame_health_B7.txt`).
  - The deployed boots' closures onto start-point nodes came over all their reaches, each started from the start
    point. B7's zero therefore does not show that removing the spin removes those closures.
- **From the reference to the first node result** (descriptive, read after the runs, not registered). This is the
  largest drift, as `frame_health.py` defines it, over each boot's ride-along samples from its reference to slot 1's
  node result (`boots/<RUN>/robot/logs/ridealong/tf_<run>.jsonl`, with `frame_ref.json` and slot 1's result time in
  `analysis/odom_terminal.json`).
  - B7 (G1 first, no spin): 0.246 m.
  - With the spin: B2 0.326 m and B2r 0.309 m (G1 first), B4 0.373 m (R1 first) and B6 0.105 m (L2 first). Each A‴
    boot's was 0.000 m.
  - B2's came at 35.3 s sim, before its first goal was sent (40.3 s sim). Of the deployed boots, only B2 and B2r ran
    G1, B7's first goal, first; B1, an A‴ boot, did too (0.000 m).
  - B7's value lies within the deployed boots' range over the same window.
  - At slot 2's start (each boot's `chain.log` FRAME line) the same boots read B7 0.246 m, B2 0.171 m, B2r 0.303 m,
    B4 0.325 m and B6 0.063 m.
- **No metre-scale step to remove.** No deployed boot took a step of 1 m or more this session, with or without the
  spin. The offline cut of the reprocess record (§9) remains the only read of the spin's metre-scale steps.
- **To the ride-along's last sample** (descriptive, `analysis/frame_health_B7_to_ridealong_end.txt`): largest drift
  0.362 m at 214.4 s sim, 38.3 % of samples above 0.30 m, no step. 87.8 % of the frame's motion came in refinement-only
  iterations.

## 6. The transit tool's stall check (a defect)

Every lane reach drives to its start with `to_start.py` (sha256 `8cbeaa66…`,
`boots/<RUN>/robot/stage0/harness_sha256.txt`). It follows Nav2's planned path in the map frame. The tool is deposited
with `goal-a-rig-gate-v3-2026-09-25` and `goal-a-cli-confirmation-2026-10-03`; it is not in this repository.
- **The defect.** In `track()`, each successful periodic re-plan (every `--replan-sim`, 5 s sim) resets the stall
  check's best remaining path length and its time. The stall check needs `--stall-sim`, 8 s sim, without 0.03 m of
  progress. While re-plans succeed it never fires, so `STALLED` (three stalls in a row) cannot be reached.
- **What that costs.** A pinned transit runs to the tool's budget (`--budget-sim`, 150 s sim), and the reach wrapper's
  one retry does the same. B2's first slot-4 attempt took 1144 s wall for its 150.6 s sim.
- **Where it bit.**
  - **B2, slot 4** (the deployed set), back from L2. During that transit's first 26 s sim (152.6–179.0 s sim) the frame
    moved 0.43 m and 4.9° (read after the runs from `boots/B2/robot/logs/ridealong/tf_b2.jsonl`). It sat 0.876 m and
    7.14° off the reference by the re-run (`boots/B2/dgx/chain.log`). The robot was pinned from about 181 s sim
    (`campaign/interventions.log`).
  - **B7, slot 2,** back from G1: 113.9 s sim without progress before the extended guard stopped it.
  - **A precedent, 2026-10-17.** F6_S2_R2_B's transit, back from L2 under the frame-health record's arm B (closures
    off), sat at map (−1.39 to −1.37, −0.44) from 69.0 to 107.8 s sim, at a drift of 0.394 m, with 8 re-plans in that
    time, before it got through (`RESULT=OK`; 17 re-plans over the whole transit). Sources, in the frame-health
    deposit at `9325a49`: `record-files/boots/F6/robot/logs/runs/F6_S2_R2_B.to_start.out` and
    `record-files/boots/F6/robot/logs/runs/container_gate/F6_S2_R2_B.to_start.json` (its plan times). The drift is
    `slam_by_location`'s (§8), re-read after the runs from that deposit's `boots/F6/robot/logs/ridealong/tf_f6.jsonl`.
- **The fix belongs to the tool.**
  - Measure progress on the robot's distance to the target, and do not reset it on a re-plan; or keep the stall window
    below the re-plan period.
  - On `STALLED`, back off before re-planning. The tool's escape path exists, but it triggers only on a planner refusal.
  - This session applied an outside stall guard instead, on the straight-line distance (§1, items 3 and 6).

## 7. The bridge's mid-run hang

- **When.** During B2's re-run transit, its retry, the bridge's sim clock stopped at 391.4 s sim, at 20:04:39 CDT.
  - In the same second, the sim host's GPU utilisation fell from a median of 36 % over the preceding 100 s to 1 %
    (`host/telemetry_session.csv`).
- **The process** (`boots/B2/dgx/hang_2004/`, taken at 20:06:30):
  - Kit's process was running (state R) with 147 threads.
  - Its main thread was running, with no wait channel, and used 579 clock ticks of CPU between the two thread
    snapshots taken 5 s apart: about one core.
  - The other 146 threads were asleep. At the second snapshot 145 sat in a kernel wait: futex (123), sleep (9), poll
    (8), epoll (3) or socket receive (2). One showed state S and no wait channel (`threads_t0.txt`, `threads_t5.txt`).
  - The 131 of those threads whose snapshot rows parse cleanly used 204 ticks between the snapshots, at most 49 for any
    one. The other 15 have spaces in their names, which shifts their state and tick fields, so their CPU use is not
    read; their wait channels are.
  - No Python stack could be taken: `py-spy` was denied permission.
- **How it ended the boot.** The retry's transit ended with `CLOCK_STALLED` (no sim-time progress for 60 s wall), 71 s
  after the clock stopped. That was the chain's second failed transit in a row (exit 6, 20:06:01).
  - The bridge stop's SIGINT, and a SIGTERM to its process group, ended the watchdog and the wrapper. The Kit process
    ended only on SIGKILL (20:08:37).
- **What it changed.** No scored reach. It came after B2's last scored reach (143.8 s sim) and outside every registered
  window. The retry it ended had started 15.2 s sim earlier, from the pinned position.
- **Not a power loss.** The 1 Hz telemetry runs without a gap longer than 1.1 s from 18:40:00 to 23:16:47 CDT.
- **Not the boot watchdog's case.** The Kit boot watchdog acts on boot stalls: it relaunched B5's and B2r's bridges
  after a 61 s stall in boot. It did not act on this hang.
- **The cause is not known.**

## 8. The lane's outside-room goals (2026-10-23, post hoc)

*Not registered, and written up after the campaign. The maintainer asked, after B2's pin, whether routes that leave the
room confound the lane. A read-only audit answered it on the sim host on 2026-10-23, 20:29–21:33 CDT, while B3's last
minutes, B4 and B5 ran. No campaign step cites it: neither the count rule (22:12:29) nor the guard's extension
(22:35:36) does (`campaign/interventions.log`). It answered in four investigations, each re-derived by a second,
independent pass. The corrections that bear on the figures used here are folded in; all of them are in
`posthoc_outside_room/wf_digest.txt`, under the headings `slam_by_location`, `camera_by_location`, `records_survey` and
`training_and_geometry`. Its full output is `posthoc_outside_room/workflow1_exterior_audit.json`. It read the
2026-10-17 databases, the reprocess record's 770 runs and its addendum's runs, and the earlier records. Of this session
it read only B3's dumped grid, the grid hashes of B1–B3 and B2's `chain.log` FRAME lines, not its databases or
ride-alongs. It changes no registered read.*

**Two of this session's four goals lie outside the room.**
- **The room** (seed 42; fitted to the dumped grid, which every boot here hashed alike): 4.900–4.945 m by
  5.805–5.945 m, with walls 2.7 m high and no ceiling (inferred from the 2026-10-17 depth; this session's depth was not
  checked). One doorway, 1.40–1.57 m wide, opens in the west wall, whose line lies at odom x −0.80 to −0.82
  (`training_and_geometry`).
  - The wall pieces also leave a 0.73–0.90 m full-height gap at the north-west corner, wider than the 0.36 m chassis,
    which furniture blocks in the grid. Whether the robot could pass it was not checked. Every crossing of the west
    wall line on 2026-10-17 (22) went through the doorway.
- **The goals.** G1 lies 1.18–1.20 m west of that line, and L2 1.93–1.95 m, beyond training's 8 m grid as well. L1
  (−2.00, 0.00), used in earlier lane records, lies 1.19 m outside. R1, R2 and both starts are inside
  (`training_and_geometry`, `records_survey`).
- **How they came to be outside** (`records_survey`).
  - G1, L1 and L2 are the 2026-08-17 gate's goals M1, M4 and M5. The seed-42 spawn was then at world (0.35, −1.75),
    inferred from the 2026-08-22 same-pose probe. G1 and L1 lay inside the room, and L2 lay outside it for any room
    size.
  - The Isaac Lab pair flip of 2026-09-12 moved the spawn about 2.3 m, to (−1.65, −0.55). From 2026-09-25 on, the goals
    were reused relative to the spawn.
  - 56 of the 75 post-flip v3 lane reaches (75 %) went to an outside goal.

**The exterior is repetitive, not short of features** (`camera_by_location`). Outside the room the camera sees the
default grid ground plane under a uniform sky.
- In the 2026-10-17 databases rtabmap extracted a median of 390 words per node outside (696 nodes), against 132 inside
  (1848 nodes), under a cap of 500.
- Refinement found a median of 39 inliers outside against 25, and was accepted 83 % of the time against 73 %.
- No outside node had fewer than 50 words; 135 inside nodes did.
- **Scope.** The 2026-10-17 return legs were driven with the camera facing outward, so those data hold almost no
  exterior views facing the room.

**The two SLAM failure modes differ by location** (2026-10-17 data and the reprocess runs).
- **Metre-scale wrong closures come mostly at the doorway and outside.**
  - The reprocessing runs (the deployed set and A′, under both builds) with a drift-growing step of 1 m or more took
    their first such step with the robot at the doorway or outside in 57 of 57 runs. That is about nine distinct places,
    since the ten reprocessings of a database are not independent (`records_survey`).
  - Counting every such step of the deployed set by `slam_by_location`'s zones, 21 of 22 came at the doorway or
    outside under 0.23.7 and 16 of 18 under 0.23.13. The rest came in the start zone.
  - Drift-growing steps of 0.5–1 m came mostly inside: 13 of 16 for the deployed set under 0.23.7.
  - F1's closure 142→16, 0.90 m and 89.8° wrong against ground truth, matched two views dominated by the exterior grid,
    seen at or through the doorway. It had 33 inliers: above the deployed `Vis/MinInliers` 15, below A‴'s 40. Whether
    0.23.13 registers it alike is not tested (`camera_by_location`).
  - A‴ accepted no doorway or exterior closure in its 160 runs. In the addendum's post-spin cut of the deployed set
    (F1 and F2, 10 runs each), such closures fell from 23 to 1 on F1 and from 7 to 0 on F2 (`slam_by_location`).
- **Refinement drift is not higher outside per metre or matched for rotation; per degree turned it is.**
  - Matched for rotation, a refined link's yaw error outside is 0.93–0.99 of inside for 2026-10-17's arm A, and
    0.64–0.67 for arm B (`slam_by_location`).
  - Per degree turned and per link, outside links err more: 0.237 against 0.058° of yaw error per degree turned for
    arm A, and 0.185 against 0.048 for arm B; a mean translation error per refined link of 23.3 against 15.8 mm for A,
    and 17.0 against 15.8 mm for B (`camera_by_location`'s verifier, C12).
  - Arm B's inside-room link errors alone recompose to 0.49–0.90 m of drift (`records_survey`; 0.46–0.90 m by
    `slam_by_location`'s classes), and its outside-room link errors alone to 0.17–0.61 m, F4's 0.608 m among them
    (`slam_by_location`).
  - The 2026-10-14 scene-axis session never left the room, and its `map→odom` moved up to 1.36 m from its first
    ride-along sample (`records_survey`'s measure, not this record's drift).
  - 52 of the deployed set's 70 failing reprocessing runs under 0.23.7 have no step of 1 m or more
    (`slam_by_location`). Every 2026-10-17 segment but F2r (14 s, 0.34 m) includes G1 or L2, so time outside is
    confounded with route length.
- **This session's deployed boots first passed 0.30 m in different places.**
  - B2 (replaced) and B4 passed it inside the room, before the robot first left it, through refinement: B2 at 35.3 s
    sim, before its first goal was sent (40.3 s sim), and B4 at 39.3 s sim, during its first reach, to R1.
  - B2r passed it outside, at 46.2 s sim during its G1 reach, with the robot at odom (−1.92, 1.92), through
    refinement.
  - B6 passed it at 92.8 s sim at the spawn, after its L2 reach and as its R2 reach began, through global closures and
    one proximity detection (§3).
  - Sources: the excursions in `analysis/frame_health_<RUN>.txt`; the robot's position from the info ride-along's
    `Loop/MapToBase`, mapped into odom through the tf ride-along (read after the runs); reach times in
    `analysis/odom_terminal.json`.
- **Transit pins sit at or near the doorway,** the routes' narrowest passage.
  - Of the six transit failures `records_survey` located, B2's among them, four sat at a jamb beyond the wall line and
    two inside, within about 0.7 m of the doorway.
  - Earlier pins came with the frame already displaced: by 0.68 m on 2026-10-09 and 1.2–1.5 m on 2026-10-14, in
    `map→odom`'s offset from the session's first ride-along sample (`records_survey`,
    `wf_records_survey/session_steps.txt`). At B2's pin (181 s sim) its `map→odom` lay 0.87 m from its reference, a
    drift of 1.22 m (read after the runs from `tf_b2.jsonl`).
  - B7's slot-2 pin began with its frame 0.28 m off the reference (`map→odom` 0.278 m from it, a drift of 0.282 m, at
    56.9 s sim; `tf_b7.jsonl`, read after the runs). 2026-10-17's F6 transit sat at a drift of 0.394 m with closures
    off (§6). Doorway pins here came at drifts of 0.28–0.39 m as well as at B2's 1.22 m.
  - **B2's pin and F6's sit are different places.** Their map coordinates nearly coincide (§1, item 2; §6) only because
    their frames differed. In odom, B2's pin lay at about (−1.14, −0.90): at the doorway's south edge (odom y −0.92 to
    −0.85), 0.32–0.34 m west of the wall line. F6 sat at about (−1.15, −0.50), 0.39–0.40 m further north: within the
    doorway's north–south span, 0.35–0.42 m north of its south edge, and 0.33–0.35 m west of the wall line, just outside
    the opening.
  - Sources, read after the runs: each transit's map poses (`boots/B2/robot/logs/runs/B2_S4_R2_A.to_start.out`; the
    frame-health deposit's `boots/F6/robot/logs/runs/F6_S2_R2_B.to_start.out`), mapped into odom through the frame then
    in force (`boots/B2/robot/logs/ridealong/tf_b2.jsonl`; that deposit's `boots/F6/robot/logs/ridealong/tf_f6.jsonl`).
    B2's position is over 207–326 s sim (the rest of its first attempt, and its retry), at map (−1.42, −0.45); F6's
    spans its sit, 69.0–107.8 s sim. The door edge and the wall line, in odom, are
    `posthoc_outside_room/wf_training_and_geometry/notes.txt`'s.
  - L2's straight line from its start passes the south door edge's outer corner by 0.04–0.11 m, less than the robot's
    0.18 m half-width (`training_and_geometry`). That bears on the policy's outbound L2 reach; whether the transits'
    planned paths run that close was not read.

**The policy is outside its training distribution at G1 and L2** (`training_and_geometry`).
- **Training.** Spawn, goal, path and subgoal are drawn from a free space that marks everything outside the room
  occupied (`source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py`, lines 575–591 at `af7d7985`, unchanged
  since `0a975ef`). A doorway's free space therefore stops 0.30–0.40 m inside the wall line, and straying 0.30 m from
  the path ends an episode.
- **What lies beyond a doorway differs.** Training sees doorways from inside: about 42 % of episodes in a walls-only
  model, possibly more. Beyond a training doorway lies the next room, 2.5–6 m away by the 10 m environment spacing
  (inferred from the configuration), not open floor.
- **On the lane.** In the top 11 of 45 depth rows, a median of 100 % of pixels sit at the 6 m far limit outside,
  against 3.1 % inside. These are the 2026-10-17 perception depth images, not the policy's input.
  - The split mixes location with heading: no outside node faces east. Inside, near the door and facing west, the same
    median is 81 % (`training_and_geometry`'s verifier, C1). Doorway and exterior views also occur from inside, on the
    lane and in training (C2). What training never had is the robot outside the room with its subgoal outside.
- **Policy-side lane findings at these goals are confounded with it.** On 2026-10-17, 5 of 12 outside reaches and 12 of
  12 inside reaches ended within 0.30 m in odom (Fisher exact, two-sided p = 0.0046; `records_survey`). The odom
  distance at the result moves with the frame, and the audit could not separate this split from a direction-dependent
  frame bias at outside goals (`records_survey`); it does not isolate the policy.

**What it means for this record.**
- **The comparison stands.** Both arms ran the same four goals in the same three rotations. Half of each arm's 12 scored
  reaches went to an outside goal, and each complete boot left the room twice. A‴ held through those reaches.
- **The deployed set's failure rate may not transfer** to a room without this exterior, or to the robot. Its failures
  here were mainly refinement drift, which the 2026-10-17 data do not concentrate outside per metre or matched for
  rotation. B4, and the replaced B2, passed 0.30 m inside the room before the robot first left it; B2r passed it
  outside, during its G1 reach; B6 at the spawn, through global closures. Its metre-scale wrong closures, in the earlier
  data, sit mostly at the doorway or outside: of its drift-raising steps of 1 m or more in the reprocess runs, 11 came
  at the doorway and 10 outside under 0.23.7 (21 of 22), and 10 and 6 under 0.23.13 (16 of 18;
  `posthoc_outside_room/wf_slam_by_location/zones_0.6.txt`).
- **The terminal reads at G1 and L2** (§4) are reads at goals outside the policy's training distribution.
- **This session's two pin locations** were each first reached on a transit heading back through the doorway from an
  outside goal: B2's slot 4 from L2 and B7's slot 2 from G1. B7's slot-3 transit started where slot 2 had pinned, and
  pinned there too (`boots/B7/robot/logs/runs/B7_S3_L2_N.to_start.out`).
- **Each reach's class** is in §4's per-reach table: G1 and L2 out, R1 and R2 in. Every scored reach's goal at
  goal-sent lay within 0.397 m of its anchor. That is less than any anchor's distance from the wall lines
  (`training_and_geometry`'s odom wall lines): G1 and L2 lie at least 1.18 m outside, R1 and R2 at least 0.75 m
  inside. So the class holds for the physical goal too.

**Proposed, not decided here.**
- A lane preflight that refuses a goal or start outside the dumped grid's free space, unless the manifest labels it an
  exterior condition, re-checked at every scene or spawn change.
- An inside-room goal set, drawn from the current room.
- Dated correction lines in the records whose findings rest on outside goals: `goal-a-rig-gate-v3-2026-09-25`'s pass,
  `policy-time-to-goal-baseline-2026-10-11`'s stop-short gap and limit-cycle evidence,
  `subgoal-referent-offset-ab-2026-10-11`'s stop-short reading (11 of 12 reaches stopped short, all at G1 and L2; cause
  unidentified), `lane-slam-frame-health-2026-10-17`'s "SUCCEEDED overstates arrival", and
  `lane-term-speedcap-ab-2026-10-14`'s verdict.
- Each live reach's inside or outside class, recorded post hoc: done here, in §4.

## 9. What is not claimed

- **No test between the arms beyond the registered rule.** Three boots per arm support no test; every comparison beyond
  the rule is descriptive.
- **Not that A‴ corrects drift.** On this lane odometry is ground truth, and A‴ added at most 0.148 m.
- **Nothing about the real robot.** The bridge supplied ground-truth odometry and simulated sensors. On the robot,
  wheel odometry drifts, and its stack runs no warm-up spin.
- **Not the spin's role.** B7 drove one reach and never returned to the start point, and no deployed boot here took a
  metre-scale step for the spin's absence to remove.
- **Not that the deployed set fails at this rate on another lane or room** (§8).
- **Not the policy.** The terminal reads describe this lane under each arm. At G1 and L2 they lie outside the policy's
  training distribution, and A‴'s count within 0.30 m in odom is a split at the node's own 0.30 m arrival radius, not a
  measure of arrival.
- **Not why live A‴ drifted more than offline A‴** (0.148 m against 0.079 m offline under the same build, 0.23.13). The
  two read different trajectories.
- **Not the hang's cause.**

## 10. What follows

- **The tracked change.** The registration prescribes it: "A‴ holds: it becomes the lane's configuration, and the
  tracked `rtabmap_params.yaml` change follows in its own PR".
  - This record supports A‴ as the lane's configuration only.
  - The tracked file, `source/strafer_ros/strafer_slam/config/rtabmap_params.yaml`, is byte-equal to the deployed set
    at `af7d7985` (sha256 `e67c97ff…`), and the robot runs it too. That PR therefore scopes the change to the lane, or
    states that the robot takes it ahead of the real-robot gate's SLAM read.
- **The robot question** goes to the real-robot gate's pre-registration: the frame-health record's SLAM read (its §6),
  with A‴'s accepted closures counted beside the steps (the reprocess record, §7).
- **The scale shortfall** stays deferred, as the reprocess record set it. It bears on a configuration that keeps
  refinement, which A‴ does not, unless the real-robot gate's SLAM read calls for it.
- **The transit tool's fix** (§6) is for the tool's owner. Until it lands, a lane session needs an outside stall guard.
- **The outside-room proposals** (§8) wait on a decision.
- **The build.** Stage 0 recorded the robot images' rtabmap build strings in every boot (`provenance.md`). Pinning
  them is [`deploy-rtabmap-build-pin`](../../tasks/active/reliability/deploy-rtabmap-build-pin.md)'s.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `lane-rtabmap-a3-2026-10-22/record-files/` |
| deposit commit | `e4d1b20ae2d36c69a56e6333974fee3cff6ac37e` |

Unless it names the main repository's tree or another deposit, every path this record names is relative to that
directory. `DEPOSIT.md`'s `cp -a` restores it into this record directory, where the same paths resolve.

- **The registration.** `PREREGISTRATION.md`, committed at `030ff40` (2026-10-22, 18:45:36 CDT) before the first boot,
  with the session tools and the sim host's power soak (`soak/`).
- **The campaign,** at `eeac8d9`: the boots' robot and sim-host logs, the campaign and intervention logs, the session
  telemetry, `AMENDMENT_RELAUNCH.md` (written after the campaign and before any scorer ran), the two scorer changes
  (§1, item 8), and `analysis/`.
- **At `e4d1b20`:** `DEPOSIT.md`, `SHA256SUMS` and `posthoc_outside_room/` (§8), and the rig's network addresses
  redacted to `<addr>` in 63 text files (`DEPOSIT.md` lists the change; `analysis/` regenerates byte-identical).
- **Changes after the registration** are listed in `DEPOSIT.md`. `frame_health.py` is unchanged from `030ff40`, and
  byte-equal to the frame-health deposit's.
- **`SHA256SUMS`** lists every file's sha256 except its own and `DEPOSIT.md`'s. Its own sha256 is
  `693d687a25cf1766881a8c52757f632811f9cdf7552ce8e36afc763a6ced0598`.
- **Reproduce** from a clone of the evidence repository, with python3 and the standard library only:

      bash lane-rtabmap-a3-2026-10-22/record-files/reproduce.sh <absolute db_dir>

  With `<db_dir>`, an absolute path, holding the boots' rtabmap databases as `<db_dir>/<RUN>/logs/*.db`, it regenerates
  all 24 files of `analysis/` byte for byte; `reproduce.sh` changes into the deposit directory first, so a relative
  path would not resolve. Without it, it regenerates the 23 that do not read the databases. After it runs,
  `git status --porcelain lane-rtabmap-a3-2026-10-22/record-files/analysis` is empty. The post-hoc audit's files are
  deposited as they were produced; `reproduce.sh` does not regenerate them.
- **Not deposited:** the eight rtabmap databases. Their sha256 are in `host/databases_sha256.txt` and in
  `provenance.md`. Also the planner's and VLM service's log lines after the registration (`DEPOSIT.md`).
