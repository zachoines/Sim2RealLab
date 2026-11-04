# The policy at inside-room goals on the lane, under A‴, 2026-11-02

The lane's earlier policy reads came mostly from goals outside the room
([`lane-outside-room-goals-2026-10-23`](../lane-outside-room-goals-2026-10-23/README.md)). Of the 10 rig band entries in
[`policy-time-to-goal-baseline-2026-10-11`](../policy-time-to-goal-baseline-2026-10-11/README.md) (below, the baseline),
8 were at goals beyond the room's west wall, where training never places the robot or a subgoal. This record reads the
same v3 policy at six goals inside the room, drawn by training's own sampler from the lane's spawn. SLAM's frame is
under A‴ (`RGBD/NeighborLinkRefining` false, `Vis/MinInliers` 40), now the robot images' own configuration.

The lane is [`lane-rtabmap-a3-2026-10-22`](../lane-rtabmap-a3-2026-10-22/README.md)'s (below, the a3 record). The sim
bridge runs on the sim host and the robot's bridge-lane services on the robot host, over the direct link, with the
deployed inference node unchanged. Six goals, I1–I6, are anchored in odom.
- **The campaign:** three boots, IB1, IB2 and IB3, each reaching each goal once, 18 slots. IB1 was relaunched once as
  IB1r over its slots 4–6 after the sim host froze.
- **The registration:** `PREREGISTRATION.md` in the evidence repository, committed at `a7df5b0` before the first boot,
  with two additions at `bd9c3ae` and the recording at `01c6d30`, all before IB1.
- **When it ran:** IB1 on 2026-11-02, 14:14–14:34 UTC; IB1r, IB2 and IB3 on 2026-11-03, 22:08–23:31 UTC.

**By the registered rules:**
- **The stop-short (R1):** a pre-crossing stop-short on 9 of 18 band entries (exact 95 % interval 26.0–74.0 %), from
  five of the six goals, against capped training's 44 of 688 (6.40 %): one-sided exact binomial *p* = 5.1 × 10⁻⁷. The
  reading: "at inside goals on the lane the policy stops short more often than in capped training".
- **The time to the goal (R3):** 0 of 18 beyond the bound *T*(*L*): "not distinguished from capped training's rate
  beyond its bound at this *s*".
- **The limit cycle (R2):** no window of 18 fires; the exact one-sided 95 % upper bound on its rate is 15.3 %.
- **The frame (R5):** A‴ holds at inside goals. Each segment's largest drift is 0.049–0.077 m, with no step, over three
  boots and 18 scored reaches.
- **The G-sim question (R6, descriptive):** the node's SUCCEEDED on 18 of 18 reaches, each ending 0.284–0.307 m from
  its goal in odom.

Read together: every one of the 18 reaches arrived, within the bound, 0.284–0.307 m from its goal in odom (R3, R4,
R6). On 9 of them the commanded speed, from band entry, first stayed under 0.05 m/s for ten consecutive ticks before
the first 0.30 m crossing (R1).
That crossing is the earlier of the node's own result and the first pose sample inside 0.30 m, so the node's terminal
zero twist is not counted. The baseline's reading of the rig, with the crossing from the pose samples alone, gives
the same 9 (descriptive).

Three qualifiers bound what that shows.
- **The reaches are not independent draws.** They repeat six goals from one start, three times each, so the rules'
  *p* values treat correlated reaches as independent. The registration states its readings only when the events come
  from at least two goals; these came from five (§2).
- **On this lane odometry is ground truth.** A‴'s frame added at most 0.077 m here (§6). That shows nothing about
  correcting a drifting odometry.
- **Transfer to the real robot is a separate question.** The bridge supplied ground-truth odometry and simulated
  sensors; the robot's own read is the real-robot gate's.

## 1. Method and departures

### The lane

As the a3 record, except:
- **The images.** `strafer-cpu` `a40d973431e8` and `strafer-gpu` `659b7649552a`, labelled `f16d18d64c2f`, built from
  `main` on 2026-11-02. A‴ is the images' own `rtabmap_params.yaml` (sha256 `82025bc7…`), with nothing mounted over it
  (arm M). Every bring-up passed PARAMS OK in all of the following:
  - the eight running values;
  - rtabmap's own `Setting` lines for `RGBD/NeighborLinkRefining` false and `Vis/MinInliers` 40;
  - the five containers on the two image IDs;
  - no mount on the parameter file's path;
  - the installed file's sha256.
- **rtabmap** is 0.23.13 in every segment, with build strings identical to the a3 session's (`provenance.md`).
- **The sim side** ran from the main checkout at `0a975ef`. Its `strafer_lab`, `strafer_autonomy` and `strafer_vlm`
  are byte-identical to `main`'s at `f16d18d6`.

### The goals and the boots

- **I1–I6** are the draw of training's sampler from the spawn (seed 20261014, MIN_SEP 0.60 m, at least 2.0 m away), on
  the lane room's grid. They lie 2.08–3.98 m from the spawn and all east of it. The spawn sits 0.80–0.825 m east of
  the room's west wall, inside the doorway's span.
- **The starts.** Every reach starts at the spawn, at the start yaw that puts its goal 45° to the left. The doorway is
  at least 50° off every start heading, outside either camera's half field of view (43.6° for the training camera,
  44.9° for the real unit), so the exterior is not in view when a reach starts.
- **The boots.** IB1 runs I1…I6, IB2 starts at I4, and IB3 starts at I2. Each boot is a fresh bridge and a fresh SLAM
  map under its own token, with the frame's reference set once the warm-up spin had finished and the frame had held for
  10 s sim.

### Per reach

As the a3 session (the inference container recreated, the transit to the start in odom, the observer, one submitted
move), with the registration's changes:
- **The transit** is `to_start.py` v2.
- **The frame gate.** Before each reach the chain stops the boot if the frame has moved any goal more than 0.30 m.
- **The sim-clock watch** stops the chain if the sim clock stalls.
- **A reach is scored** only when the chain logged its done line with no Nav2 fallback.

None of those stopped a boot here. Every slot's transit succeeded, every gate read 0.077 m or less, and no slot was
re-run.

### Departures from the registration

- **The sim host froze during IB1** (§8). IB1's slots 1–3 were scored. Its slot 4 had been submitted, but the chain
  logged no done line for it, so it is not scored.
- **The relaunch.** By the registration's relaunch rule (its power-loss case: a freeze that needed a power cycle), IB1
  was relaunched once as IB1r. IB1r ran the unscored slots 4–6 with its own token and reference. IB1's first segment
  stands with slots 1–3, and IB1 counts once toward the boots.
- **Two operations on the sim host** for the segments after the freeze, recorded in `OPERATIONAL_NOTE_2026-11-03.md`
  before IB1r's first reach. Neither is a read:
  - the page cache was dropped before every bridge launch;
  - a 1 Hz memory log ran beside the telemetry.
- Each remaining boot ran as its own campaign unit. Nothing else departed from the registration.

## 2. The stop-short (R1)

Like-for-like with training's pre-crossing read, in the node's live frame (`analysis/inside_reads.txt`, `.json`). *d*
is the distance from the record's map pose (10 Hz) to the record's map goal.
- **Band entry** is the first pose sample with *d* ≤ 0.42 m.
- **The first crossing** is the first pose sample with *d* ≤ 0.30 m or, for a SUCCEEDED reach, the node's result if
  that comes first.
- **The quiet stop** is the 10th of the first 10 consecutive `/cmd_vel` ticks with planar command under 0.05 m/s,
  searched from band entry to the result.
- **A pre-crossing stop-short** is a quiet stop strictly before the first crossing, or with no crossing.

| reach | goal | quiet stop, from the goal (m) | first crossing less the quiet stop (s) | pre-crossing stop-short |
|---|---|---:|---:|---|
| IB1_S1_I1_M | I1 | — | — | no |
| IB1_S2_I2_M | I2 | 0.344 | 1.51 | yes |
| IB1_S3_I3_M | I3 | 0.309 | 0.16 | yes |
| IB1_S4_I4_M (IB1r) | I4 | — | — | no |
| IB1_S5_I5_M (IB1r) | I5 | — | — | no |
| IB1_S6_I6_M (IB1r) | I6 | 0.300 | 0.00 | no |
| IB2_S1_I4_M | I4 | 0.314 | 0.36 | yes |
| IB2_S2_I5_M | I5 | — | — | no |
| IB2_S3_I6_M | I6 | 0.303 | 0.11 | yes |
| IB2_S4_I1_M | I1 | 0.351 | 1.06 | yes |
| IB2_S5_I2_M | I2 | 0.302 | 0.03 | yes |
| IB2_S6_I3_M | I3 | — | — | no |
| IB3_S1_I2_M | I2 | 0.326 | 0.91 | yes |
| IB3_S2_I3_M | I3 | — | — | no |
| IB3_S3_I4_M | I4 | 0.307 | 0.12 | yes |
| IB3_S4_I5_M | I5 | — | — | no |
| IB3_S5_I6_M | I6 | 0.328 | 1.03 | yes |
| IB3_S6_I1_M | I1 | — | — | no |

- **All 18 entered the band.** Ten went quiet in it. IB1_S6_I6_M's quiet stop fell on its first crossing, so it is not
  a stop-short. The nine stop-shorts went quiet 0.302–0.351 m from the goal and crossed 0.03–1.51 s later.
- **Per goal:** I1 1 of 3, I2 3 of 3, I3 1 of 3, I4 2 of 3, I5 0 of 3, I6 2 of 3.
- **The rule.** 9 in 18 against capped training's pooled 44 in 688 (6.40 %; the baseline's run 3 and capped open-dwell
  launches, rows 0–599): one-sided exact binomial *p* = 5.1 × 10⁻⁷. At *n* = 18 the reading needs 4 or more. The events
  come from five goals, so the registered reading holds: "at inside goals on the lane the policy stops short more often
  than in capped training".
- **Beside it** (the registration's comparators, not tests): uncapped training's 21 in 714 (2.94 %), and the rig's 1 in
  2 inside and 6 in 8 outside.
- **The baseline's rig reading** of these records (descriptive), with the crossing from the pose samples alone, also
  gives 9 of 18.

## 3. The limit cycle (R2)

The baseline's detector, conditions (a)–(d) as its `ttg_summary.py` holds them, on each band entry's `/cmd_vel` ticks
from band entry to before the first crossing (`analysis/inside_reads.txt`, `.json`).
- **No window fires.** All 18 windows were read: each had at least 45 ticks (45–141).
- **The rule.** The detector fires on none of 18 windows. The exact one-sided 95 % upper bound on its rate is 15.3 %;
  the scorer also prints the two-sided exact interval, 0.0–18.5 %.
- **Beside it:** training's 0 in 127, and the rig's 0 in 2 inside, 5 in 6 outside (the baseline's correction of
  2026-10-29).

## 4. The time to the goal (R3)

For each SUCCEEDED reach, *t* is the result's sim time less acceptance's, and *L* is the first plan's arc length at
acceptance. The bound is *T*(*L*) = 11.8745 + 1.5319 *L* s, the baseline's capped 99 % bound
(`analysis/inside_reads.txt`, `.json`).

| reach | *L* (m) | *t* (s) | *T*(*L*) (s) | *t* / *T* |
|---|---:|---:|---:|---:|
| IB1_S1_I1_M | 3.048 | 8.62 | 16.54 | 0.52 |
| IB1_S2_I2_M | 3.100 | 11.05 | 16.62 | 0.66 |
| IB1_S3_I3_M | 3.173 | 7.75 | 16.74 | 0.46 |
| IB1_S4_I4_M (IB1r) | 3.499 | 8.00 | 17.23 | 0.46 |
| IB1_S5_I5_M (IB1r) | 2.090 | 5.57 | 15.08 | 0.37 |
| IB1_S6_I6_M (IB1r) | 4.003 | 9.91 | 18.01 | 0.55 |
| IB2_S1_I4_M | 3.499 | 8.76 | 17.23 | 0.51 |
| IB2_S2_I5_M | 2.012 | 5.57 | 14.96 | 0.37 |
| IB2_S3_I6_M | 4.003 | 10.04 | 18.01 | 0.56 |
| IB2_S4_I1_M | 3.048 | 9.25 | 16.54 | 0.56 |
| IB2_S5_I2_M | 3.100 | 10.19 | 16.62 | 0.61 |
| IB2_S6_I3_M | 3.172 | 7.85 | 16.73 | 0.47 |
| IB3_S1_I2_M | 3.100 | 13.14 | 16.62 | 0.79 |
| IB3_S2_I3_M | 3.124 | 7.26 | 16.66 | 0.44 |
| IB3_S3_I4_M | 3.499 | 8.60 | 17.23 | 0.50 |
| IB3_S4_I5_M | 2.090 | 5.54 | 15.08 | 0.37 |
| IB3_S5_I6_M | 4.003 | 10.43 | 18.01 | 0.58 |
| IB3_S6_I1_M | 3.047 | 9.52 | 16.54 | 0.58 |

- **The rule.** 0 of 18 reaches beyond *T*(*L*), against capped training's 5 completions in 506 beyond its own bound
  (0.99 %): one-sided exact binomial *p* = 1. The reading is "not distinguished from capped training's rate beyond its
  bound at this *s*". At *s* = 18 the other reading needs 2 or more. Per goal: 0 of 3 at each of I1–I6.
- **The end points differ from training's,** as registered. The node ends a goal at its first live 0.30 m crossing,
  where training's time runs to the dwell's firing. The executor turns the robot toward the goal before acceptance,
  where training's time includes the turn. Both make the lane's *t* shorter for the same approach.

## 5. The terminal reads in odom (R4)

Per reach (`analysis/inside_reads.json`'s rows, all 18 non-void; `analysis/odom_terminal.txt` prints the same values
per reach):
- **The odom distance from the robot at the result to the goal's anchor:** 0.284 m minimum, 0.298 m median, 0.307 m
  maximum.
- **The in-reach frame motion projected on the approach**, *p*: −0.017 to +0.014 m.
  - *p* is Δ·*u*, where Δ is the motion of the goal's odom position between acceptance and the result, and *u* is the
    unit vector from the robot to the goal at acceptance.
  - The largest |Δ| is 0.018 m.
- **The live distance less the odom distance at the result:** −0.014 to +0.016 m.
- **Within 0.35 m of the anchor:** 18 of 18. That is the node's 0.30 m radius plus the observer's 0.05 m agreement
  tolerance. With the frame held, the odom distance sits within 0.016 m of the node's radius, so a count split at
  0.30 m itself is not read.

`odom_terminal.py`'s own odom and live stop-short columns and its per-arm stop-short line are not read; R1 is the only
stop-short.

## 6. The SLAM read (R5) and the frame rule

Per segment, against the segment's reference, over the window from the reference to the segment's last node result
(`analysis/frame_health_<RUN>.txt`, `analysis/slam_reads.txt`, `analysis/arm_summary.txt`):

| segment | window (s sim) | largest drift (m) | steps ≥ 0.5 m | steps ≥ 1 m | accepted closures in the window (global + proximity) | `Visual_inliers`, min / median / max | largest goal motion across one (m) |
|---|---:|---:|---:|---:|---:|---|---:|
| IB1 | 105.1 | 0.049 | 0 | 0 | 11 (10 + 1) | 40 / 47 / 59 | 0.075 |
| IB1r | 98.3 | 0.050 | 0 | 0 | 17 (17 + 0) | 43 / 48 / 60 | 0.050 |
| IB2 | 232.7 | 0.058 | 0 | 0 | 40 (27 + 13) | 41 / 47.5 / 59 | 0.055 |
| IB3 | 223.3 | 0.077 | 0 | 0 | 32 (23 + 9) | 40 / 47 / 69 | 0.058 |

- **The share of samples above 0.30 m** is 0.0 % in every segment.
- **IB1's chain never logged done,** so its segment is also read, descriptively, to its ride-along's last sample
  (t_sim 149.1). The
  largest drift there is the same 0.049 m, with no step (`analysis/frame_health_IB1_to_ridealong_end.txt`).
- **The void rule** (`analysis/inside_reads.txt`, `.json`). Every scored reach's own goal moved at most 0.046 m from
  the reference during the reach. Every reach had at least 100 ride-along samples in its window, and every dispatched
  goal was observed, at most 0.016 m from the record's goal. So none is void.
- **The rule:** A‴ holds at inside goals. Every segment read has a largest drift of 0.30 m or less and no step of
  0.5 m or more, over three boots and 18 scored reaches.

## 7. The G-sim question (R6), descriptive

Per goal, the node's SUCCEEDED and the count ending within 0.35 m of the goal's odom anchor: 3 of 3 at each of
I1–I6; pooled, 18 of 18 and 18 of 18 (`analysis/inside_reads.txt`, `.json`).
- **Beside it,** the September gate (`goal-a-rig-gate-v3-2026-09-25`): six successes in seven at outside goals, one in
  two inside, as the outside-room record reads it.
- **The September rule is not applied,** as registered. These goals are the inside-room draw of training's own
  sampler, not the gate's goals.

## 8. The sim host's freeze, and the operations after it

The sim host stopped during IB1's slot 4, with no clean shutdown (`OPERATIONAL_NOTE_2026-11-03.md`):
- **Its 1 Hz telemetry** (fsync'd line by line) ends at 2026-11-02 14:34:43 UTC. The journal's last line is 14:31:55 UTC
  (`campaign/journal_reads.txt`). Over IB1 the GPU ran at 36–58 °C and 10.5–62.6 W, in short bursts at about one per
  reach (the last at 14:34:28–29, 55–57 °C and 54–56 W) between quieter stretches near 50 °C and 20 W. No clock-event
  reason was set at any sample. The thermal zones read 35.8–62.6 °C, and the 1-minute load 0.6–5.9 (1.8–2.8 in the last
  minute).
- **The one kernel line.** `journalctl -b -1 -k` holds one GPU-driver line besides the module's load,
  `NVRM: … Out of memory [NV_ERR_NO_MEMORY]`, at 14:14:27 UTC, 12 s after IB1's bridge launched
  (`campaign/journal_reads.txt`). The bridge came up and IB1 ran for 20 minutes after it.
- **The power cycle** was on 2026-11-03 at 16:40 UTC.
- **The robot host's stack ran on for about 31 hours.** IB1 was torn down on 2026-11-03 at 22:04 UTC. The database saved
  cleanly, the recorder stopped with its `metadata.yaml`, and no container was left.

The maintainer's lead is about 80 GB of the sim host's memory in the page cache at the out-of-memory line, on the GB10,
where the CPU and the GPU share memory. **It is not shown as the cause of the freeze**: the run continued for 20
minutes after the line.

For IB1r, IB2 and IB3, the page cache was dropped before each bridge launch, without root (`posix_fadvise`
`POSIX_FADV_DONTNEED` over the home directory), and a 1 Hz memory log ran (`campaign/evict.log`,
`campaign/meminfo.csv`; the eviction tool prints kB/1024², GiB, labelled "GB"):

| before | Cached (GiB) | MemAvailable (GiB) |
|---|---|---|
| IB1r | 17.17 → 1.87 | 100.45 → 100.37 |
| IB2 | 4.91 → 2.09 | 100.47 → 100.47 |
| IB3 | 4.56 → 2.09 | 100.48 → 100.47 |

- **From the first eviction to IB3's end,** MemAvailable stayed at 86.3 GiB or more and Cached at 6.9 GiB or less.
- **No further out-of-memory or Xid line** appeared in the kernel log of the boot that ran IB1r, IB2 and IB3, read after
  IB3 (`campaign/journal_reads.txt`), **and no freeze.** That is three boots, not a test; the eviction is not shown to
  prevent a stop. The freeze is filed as its own brief (§11).

## 9. The recording (no read depends on it)

A `ros2 bag` in sqlite3 storage ran in every boot (`boots/<RUN>/robot/logs/viz/viz_log.txt`). It was given nine topics,
`/rtabmap/map`, `/tf`, `/tf_static`, `/robot_description`, `/strafer_inference/active_goal`, `/strafer/subgoal`,
`/cmd_vel`, `/odom` and `/strafer/odom`, and recorded eight. Its log shows no subscription to `/odom`, and no bag holds
it.
- **The check.** Each bring-up's 10 s check of the two transient-local topics read one message each.
- **The stop.** Each stop wrote its `metadata.yaml` within 2 s.
- **The bags** are 81.5 MB (IB1, including the hours after the freeze), 26.3 MB (IB1r), 59.0 MB (IB2) and 52.9 MB
  (IB3). They are kept on the sim host by digest (`campaign/databases_sha256.txt`).
- **The cost (descriptive).** The bring-up's load log averaged 10.50 (IB1, to 14:34:44 UTC), 9.17 (IB1r), 9.52 (IB2)
  and 9.04 (IB3), against 9.57, 9.07 and 9.78 in the a3 session's A‴ boots without a recorder. The navigation container,
  where the recorder runs, drew 137–178 % CPU at slot starts, against 139–167 %. Nothing tests a difference; a cost is
  not ruled out.
- **For viewing,** the deposit's `viz/` holds an offline collage for Foxglove: one MCAP per segment (3.0–12.4 MB) with
  every recorded message and a derived scene of the goals, the spawn and R1's events. `viz/build_collage.py` writes it
  from the deposit and the bags kept on the sim host, and `viz/HOWTO.md` says how to lay it out (`viz/README.md`). It
  was checked with the `mcap` readers, not opened in the app.

## 10. What is not claimed

- **Not why the policy goes quiet near the goal.** The record counts the stop-shorts by the registered definition and
  reports where and when they happened. It does not explain them.
- **Not that the stop-short rate is the policy's rate in general.** The reaches are three repeats of six goals from one
  start. The rule's *p* treats them as independent, which they are not, and the reading is stated because the events
  come from five goals.
- **Not that the policy is as fast as capped training.** "Not distinguished at this *s*" is a non-rejection, at 18
  reaches whose end points are shorter than training's.
- **Not that the limit cycle cannot occur at inside goals:** 0 of 18, with a one-sided upper bound of 15.3 %.
- **Nothing about the real robot,** and not that A‴ corrects drift: on this lane odometry is ground truth.
- **Not the freeze's cause.**

## 11. What follows

- **The real-robot gate's pre-registration** can draw its goal set from the real room's free space by the same rule;
  this record is the lane's inside-goal read beside it.
- **The freeze** is filed as [`dgx-freeze-under-kit`](../../tasks/active/reliability/dgx-freeze-under-kit.md) (P1). The
  memory log is its first evidence; its decisions are the maintainer's.
- **The recording's tree version** is filed as
  [`campaign-viz-recording`](../../tasks/active/tooling/campaign-viz-recording.md) (P2). Its topic set, its storage
  plugin and a tracked layout are the maintainer's decisions.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `lane-inside-goals-2026-11-02/record-files/` |
| deposit commit | `901557d33ba778fafe970e946bccc26421d7755e` |

Unless it names the main repository's tree or another deposit, every path this record names is relative to that
directory. `DEPOSIT.md`'s `cp -a` restores it into this record directory, where the same paths resolve.

- **The registration:**
  - `PREREGISTRATION.md`, committed at `a7df5b0` (2026-11-02) before the first boot;
  - its additions at `bd9c3ae` (both cameras' half fields; the counts-once guard) and `01c6d30` (the recording);
  - the operational note at `cffc8ad`, before IB1r's first reach.
- **The campaign** at `e7755be`:
  - the four segments' robot and sim-host logs;
  - IB1's container logs cut at the interruption and the teardown (`DEPOSIT.md` names the cut points and line counts);
  - the campaign, telemetry, memory and eviction logs;
  - `analysis/` from `reproduce.sh`.
- **At `901557d`:** `DEPOSIT.md`, the collage (`viz/`), the recorder's own logs (`boots/<RUN>/robot/logs/viz/`,
  left out at `e7755be`), the journal reads, the units' start lines (`campaign/units_journal.txt`) and a dated
  correction to the operational note's telemetry and memory figures.
- **`SHA256SUMS`** lists every file's sha256 except its own and `DEPOSIT.md`'s (`DEPOSIT.md` gives its digest).
- **Reproduce** from a `cp -a` copy of the deposit directory, with python3 carrying numpy and scipy (on the sim host,
  `env_isaaclab3`'s):

      PATH=~/miniconda3/envs/env_isaaclab3/bin:$PATH bash reproduce.sh

  It regenerates all 18 files of `analysis/` byte for byte.
- **Not deposited:** the four rtabmap databases, the recording's bags and IB1's full container logs. They are on the sim
  host, with the databases' and bags' sha256 in `campaign/databases_sha256.txt` (`DEPOSIT.md`). The journal reads the
  record cites are deposited as `campaign/journal_reads.txt`.
