# Provenance — the lane's goals outside the room, 2026-10-23

captured_date: 2026-10-23 (the audit ran 20:29–21:33 CDT, 2026-10-24 01:29–02:33 UTC); written 2026-10-29

## Host

| role | host | kernel | notes |
|---|---|---|---|
| sim | gx10-d1d8 (NVIDIA GB10, aarch64) | 7.0.0-1019-nvidia | Read-only. Python 3.12.13 with the standard library, numpy 2.3.1, OpenCV 4.13.0 and SQLite 3.53.2 (versions read on the host on 2026-10-29); no bridge, no Kit, no container, no GPU process, no access to the robot host. |

**At the same time.** The [`lane-rtabmap-a3-2026-10-22`](../lane-rtabmap-a3-2026-10-22/README.md) campaign ran on
both hosts. The audit's window covered B3's last minutes, all of B4 and most of B5 (`campaign/campaign.log` in that
deposit: B3 done 20:33:03, B4 20:59:38, B5 21:39:25 CDT). No campaign step cites the audit.

## Method

- **Four investigations,** `slam_by_location`, `camera_by_location`, `records_survey` and `training_and_geometry`,
  ran in parallel from 20:29.
- **Four independent verifiers** followed, one per investigation, finishing at 21:33. Each recomputed the
  investigation's figures with its own code, gave each claim a verdict, and listed what was missed.
- **What was kept.** Each investigation's and each verifier's scripts and outputs are the deposit's `wf_<name>/` and
  `wf_verify_<name>/`. Their summaries, tables, verdicts, corrections and missed items are `wf_digest.txt`, and the
  structured output is `workflow1_exterior_audit.json`.
- **No registration.** Nothing was pre-registered. No reach, bridge or robot-side process was run for the audit.

**Recomputation for the record (2026-10-29).** Every figure the README states was recomputed on the same host before
it was written. Four kinds are marked in the README instead: figures quoted from another record, §2's
readings quoted from `wf_training_and_geometry/notes.txt`, §5's rotation-matched ratios, and Appendix A's verifier text.
- **How.** By independent code read against the deposits, or by rerunning a copy of a deposited script in a scratch
  directory. No deposited file was run in place or changed.
- **The databases.** The 2026-10-17 databases were opened read-only (`mode=ro&immutable=1`). Their sha256 were
  re-checked against `lane-slam-frame-health-2026-10-17`'s `DEPOSIT.md` first, and all eight match.
- **Deposited as `recompute_2026-10-29/`:** the recomputation's scripts, as run. They read the deposits and the
  2026-10-17 databases by absolute paths on the sim host, named in each script. Each figure in the README names the
  deposited file that carries it, or the deposited input it was recomputed from.

## Inputs

| input | where it was read | deposited as |
|---|---|---|
| the eight 2026-10-17 rtabmap databases | sim host, `~/Workspace/rig_slam_frame_dgx/databases/` (copies made 2026-10-19) | not deposited; sha256 in `lane-slam-frame-health-2026-10-17`'s `DEPOSIT.md`; `wf_training_and_geometry/node_odom_poses.json` and `wf_camera_by_location/out/nodes_<seg>.csv` are extracts |
| the frame-health deposit: windows, goals, ride-alongs, transit traces, parameters | `lane-slam-frame-health-2026-10-17/record-files/`, as corrected at `c0c02e6` | that deposit |
| the reprocessing runs | sim host, `~/Workspace/rtabmap_reprocess_dgx/runs/` | `lane-rtabmap-reprocess-2026-10-21/record-files/runs/` (the 770 matrix extracts byte-equal) and `addendum-2026-10-22/runs/`, at `df39404` |
| the per-link extracts | `lane-rtabmap-reprocess-2026-10-21/record-files/per-link/db_links/` | that deposit |
| the dumped training grid | sim host, B3's `~/Workspace/rig_lane_a3_dgx/B3/grid_dump.npz`, sha256 `36929c68a9e48a74eed89cfeb27c4dec0d938603e5ee8be2f24321f0b4adbbcd` | `lane-rtabmap-a3-2026-10-22`'s `boots/grid_dump_B3.npz`; byte-equal to `lane-slam-frame-health-2026-10-17`'s `goals/grid_dump_F1.npz` |
| the 2026-10-23 session | the audit: B3's dumped grid, the grid hashes of B1–B3 and B2's `chain.log` FRAME lines only; the record's reach count (2026-10-29): `analysis/odom_terminal.json`; the README's quoted figures: that record's §6 and its `analysis/frame_health_B2*.txt` | `lane-rtabmap-a3-2026-10-22`'s `boots/` and `analysis/` |
| the earlier sessions' deposits | `goal-a-rig-gate-2026-08-17`, `goal-a-attribution-2026-08-22`, `goal-a-rig-gate-v3-2026-09-25`, `goal-a-cli-confirmation-2026-10-03`, `goal-a-cli-video-2026-10-09`, `policy-time-to-goal-baseline-2026-10-11`, `subgoal-referent-offset-ab-2026-10-11`, `lane-term-speedcap-ab-2026-10-14`, `lane-term-scene-axis-2026-10-14` | those deposits; `wf_records_survey/leg2_files/` links twenty of the speed-cap deposit's files |
| the records | the sim host's main checkout at `0a975ef`; the frame-health and reprocessing READMEs from their pull-request branches (`44e56ab`, `1df7935`), merged since | this repository |
| the source | the sim host's checkout at `0a975ef`, for `proc_room.py`, `commands.py`, `events.py`, `strafer_env_cfg.py` and `bridge_grid_dump.py`; Isaac Lab v3.0.0-beta2.patch1 for the root quaternion's order | the README cites lines at `af7d7985`; `proc_room.py` is unchanged between `615fc14` (v3's training tree) and `af7d7985` |

## The deposit

- **First deposited** at `e4d1b20`, on 2026-10-24, as `posthoc_outside_room/` in `lane-rtabmap-a3-2026-10-22`'s
  deposit.
- **Moved** at `1e2447ef1cd8b307d418267aeacb4d91fa4d00c3` to `lane-outside-room-goals-2026-10-23/record-files/`. The
  move changed nothing but the 20 links under `wf_records_survey/leg2_files/`, which were absolute and became relative,
  with a `SOURCES.txt` of their targets' paths and sha256.
- **The layout:**
  - `wf_digest.txt` and `workflow1_exterior_audit.json` at the top;
  - one directory per investigation (`wf_slam_by_location/`, `wf_camera_by_location/`, `wf_records_survey/`,
    `wf_training_and_geometry/`);
  - one per verifier (`wf_verify_<name>/`).
- **Completed** at `e604a87` with `DEPOSIT.md`, the record's recomputation scripts (`recompute_2026-10-29/`) and
  `SHA256SUMS`. Four more recomputation scripts were added at `6b53956`: `closures_beyond_wall.py`, `pins_rule.py`,
  `live_steps.py` and `checks_starts_views_steps.py` (named so at `3a5a9a9`); `SHA256SUMS` lists 307 files.
- **`DEPOSIT.md`** describes the files.
- **`SHA256SUMS`** lists the sha256 of every regular file except its own and `DEPOSIT.md`'s. The 20 links under
  `wf_records_survey/leg2_files/` are not hashed there; `SOURCES.txt` gives their targets' paths and sha256.
