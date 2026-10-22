# Provenance — the lane's rtabmap databases, reprocessed, 2026-10-21

captured_date: 2026-10-21 (the runs ran 2026-10-22 03:38–04:11 UTC; 22:38–23:11 local, UTC−5)

## Host

| role | host | kernel | notes |
|---|---|---|---|
| sim | gx10-d1d8 (NVIDIA GB10, aarch64, 20 cores; read from the host, not deposited) | 7.0.0-1019-nvidia | CPU only: no bridge, no Kit, no GPU process, no robot host. Each run a sealed container (`--network none`); the matrix ran as systemd user units with its logs on persistent disk. |

## Inputs

- **The eight rtabmap databases** of `lane-slam-frame-health-2026-10-17`, from the copies on the sim host, each checked
  against that record's deposit digest before use (`tools/prepare_inputs.py`). The databases themselves are not
  deposited; the reprocessing inputs made from them (`asis` and `odomcov`) are listed with their digests in
  `inputs.sha256`.
- **The record's own deposit** at `c0c02e6`: each segment's window (`analysis/frame_health_<RUN>.json`), the goals
  (`goals/goals_odom.json`) and the ride-along (for the per-link read's agreement checks).

## Tool and images

| image | rtabmap | source | image id |
|---|---|---|---|
| `rtabmap-reprocess:0.23.7` | `ros-humble-rtabmap 0.23.7-1jammy.20260907.223800` | ROS 2 Humble snapshot 2026-09-14 | `sha256:70f47d2c…` |
| `rtabmap-reprocess:0.23.13` | `ros-humble-rtabmap 0.23.13-1jammy.20261002.091850` | ROS 2 Humble snapshot 2026-10-12 | `sha256:89417926…` |

Both on `ros@sha256:5417e569…`, with the same OpenCV (4.5.4), libg2o (2020.5.29) and gtsam (4.2.0) builds; full package
lists in `image/<version>/dpkg.txt`. The databases were written by rtabmap 0.23.7 on the robot image of revision
`9f7d20758d5b` (built 2026-10-11, no longer on the robot host); which 0.23.7 rebuild it carried is not known. The robot
image rebuilt on 2026-10-19 carries 0.23.13 at the build listed above.

## Registration

- `PREREGISTRATION.md`, evidence repository commit `ff3336f`, before any matrix run.
- `AMENDMENT_A3.md`, commit `78b1227`, after the primary set was read and before any A‴ run.
- One mechanics run preceded the registration (F2r, `-stop 24`), declared in it.
