# Provenance — A‴ against the deployed rtabmap set, live on the lane, 2026-10-22

captured_date: 2026-10-23 (registered 2026-10-22; the campaign ran 2026-10-23 18:45–23:16 local, UTC−5, that is
2026-10-23T23:45Z–2026-10-24T04:16Z)

## Hosts

| role | host | kernel | notes |
|---|---|---|---|
| sim | gx10-d1d8 (NVIDIA GB10, aarch64) | 7.0.0-1019-nvidia (Kit's banner in `boots/<RUN>/dgx/bridge.log`) | Driver 580.173.02 (the same banner). Runs the bridge, the planner and VLM services, the session's drivers (`dgx/`) as systemd user units with their logs on persistent disk, the 1 Hz telemetry (`dgx/fh_telemetry.py`), the stall guards and the viewer; from 20:29 on 2026-10-23, read-only analyses as well (the record's §1, item 1). |
| robot | strafer-nx | not read in this deposit | Runs the deploy stack in five containers (`session/fh_up.sh`), the reach harness, stage 0 (`session/stage0_fh.sh`) and the frame check (`tools/frame_check4.py`). Its hardware, as read on 2026-10-10 for `goal-a-cli-video-2026-10-09`: a Jetson Orin NX 16 GB, L4T R36.4.3, kernel 5.15.148-tegra. |

**The link.** The direct link carried DDS and the planner and VLM calls; the sim host drove each boot over it by ssh.
- **Sim side:** the bridge ran on ROS domain 42 with the direct-link Cyclone configuration (line 1 of each
  `bridge.log`).
- **Robot side:** every container selected the direct-link interface
  (`boots/<RUN>/robot/stage0/cyclone_pin_in_containers.txt`).

**The robot host was held by this session** from bring-up to its handback at 23:18:04 local
(`campaign/interventions.log`).

## Trees and stack

| side | tree | stack |
|---|---|---|
| sim | the main checkout at `0a975ef` (`git=0a975ef5` on line 1 of each `bridge.log`). By the registration, the lane's sim-side sources there equal `main`'s at `af7d7985`. | conda `env_isaaclab3` (Kit's configuration path in each `bridge.log` reads Isaac Lab 3.0), the environment of the 2026-10-17 session (Isaac Sim 6.0.1.0 + Isaac Lab v3.0.0-beta2.patch1 by that record's provenance); the planner and VLM services from the same checkout, one process each, started for the power soak (2026-10-22 22:41Z) and stopped at 23:16:38 local on 2026-10-23, so the same processes served the campaign (their logs up to the registration are `soak/services/`) |
| robot | images labelled `0423a13fcde6`, built 2026-10-22 (the registration); the checkout at `6715cf3` with the lane's two host-local env-file edits, `source/strafer_ros/deploy/compose/autonomy.env` and `source/strafer_ros/strafer_bringup/config/env_autonomy.env` (stage 0's "robot checkout" lines) | `strafer-gpu` `sha256:2183f6b18fdf…` (`inference`); `strafer-cpu` `sha256:c30981464f22…` (the other four) (`boots/<RUN>/robot/stage0/images.txt`, identical in all eight boots) |

**rtabmap, read from inside `strafer_slam` in every boot** (stage 0, `images.txt`; identical in all eight):

| package | build |
|---|---|
| `ros-humble-rtabmap` | `0.23.13-1jammy.20261002.091850` |
| `-slam` | `0.23.13-1jammy.20261008.162341` |
| `-ros` | `0.23.13-1jammy.20261008.165507` |
| `-launch`, `-demos`, `-examples` | `0.23.13-1jammy.20261008.165222`, `.165226`, `.165236` |
| `-conversions`, `-costmap-plugins`, `-msgs`, `-odom`, `-python`, `-rviz-plugins`, `-sync`, `-util`, `-viz` | `0.23.13-1jammy.20261002.*` |

These are not the 2026-10-17 session's images: those carried rtabmap 0.23.7.

**Scorers.** python3 3.12 with the standard library only. The byte-cache files removed at `eeac8d9` were CPython
3.12's. A re-run on the sim host with Python 3.12.13 (conda `env_isaaclab3`) reproduced all 24 files of `analysis/` at
`eeac8d9` byte for byte.

**The policy.** `/models/strafer_depth_subgoal_v3_999.onnx`, sha256 `c866bfd5…` (its `.json` `d22e3504…`), loaded
`recurrent=True` in every boot (`boots/<RUN>/robot/stage0/in_container_artifact_checks.txt`). The deployed inference
node's caps, unchanged: `vel_cap=(0.7841 m/s, 2.0933 rad/s) policy_loaded=True`.

**The arms.**
- `node/rtabmap_params_image.yaml` (sha256 `e67c97ff…`) is byte-equal to the image's installed file and to `main`'s at
  `af7d7985`.
- `node/rtabmap_params_C.yaml` (`c355d811…`) differs from it in `RGBD/NeighborLinkRefining` and `Vis/MinInliers`.
- `node/slam_params_overlay.yml` mounts the boot's file read-only over the installed copy;
  `node/nospin_overlay.yml` sets `donut_warmup:=false` for B7.
- In every boot `stage0/slam_mount.txt` reads one sha256 on the host and in the container, and
  `logs/rtabmap_params_check.txt` reads `PARAMS OK` for eight keys in the running node's parameter dump.
- `node/rtabmap_params_B.yaml` is the frame-health session's arm B, copied with its tools; no boot here mounted it.

**The robot's compose chain** is the frame-health session's (`session/fh_up.sh`): profiles `policy` and `sim-bridge`;
`docker-compose.yml` and `docker-compose.override.sim-bridge.yml` from the robot host's checkout, plus its two
host-local overrides, which are not in the tree; the CLI set's direct-link autonomy URLs; and the arm's params overlay
(`node/slam_params_overlay.yml`), with `node/nospin_overlay.yml` added for B7. It runs five named services:
`inference`, `slam`, `navigation`, `sim-perception`, `autonomy`. Each boot's resolved configuration is
`boots/<RUN>/robot/logs/compose_config.yml`.

**The harness.** The robot host's reach harness ran unchanged; stage 0's `harness_sha256.txt` holds its digests and
the session files'. The transit tool `to_start.py` is `8cbeaa66…`. The deposited reach wrapper
`tools/run_cli_one_start.sh` is byte-equal to the frame-health deposit's (`a49bf7b2…`).

**The bridge.** `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0` (`dgx/fh_session.sh`), configuration
`StraferNavCfg_BridgeAutonomy_ProcRoomEnriched`, seed 42, through the main checkout's `tools/kit_boot_watchdog.sh` (at
`0a975ef`) and the grid-dump
wrapper (`dgx/bridge_grid_dump.py`). Every boot's dumped grid hashed
`36929c68a9e48a74eed89cfeb27c4dec0d938603e5ee8be2f24321f0b4adbbcd`, equal to the frame-health session's F1
(`boots/<RUN>/dgx/grid_dump_sha256.txt`; B3's array is deposited as `boots/grid_dump_B3.npz`).

## The power soak (before the registration)

On 2026-10-22, from 22:41:13Z to 23:44:16Z, the bridge and both services ran under a request load that began at
22:42:51Z (`soak/`; `soak/soak_summary.txt`, by `soak/soak_summary.py`):
- 1 Hz telemetry over 63.0 min, with no gap longer than 1.1 s;
- 103 `/plan` and 52 `/describe` requests, none failed (non-200), with mean latencies of 4.2 s and 3.9 s;
- GPU at most 62 °C and 65.0 W (median 20.7 W); the hottest ACPI zone at most 71.3 °C; no clock-event reason and no
  hardware slowdown ever active.

There was no power loss. The registration was committed at `030ff40`, 23:45:36Z.

## The session

**Telemetry.** `host/telemetry_session.csv`, 1 Hz, from 23:40:00Z on 2026-10-23 to 04:16:47Z on 2026-10-24: 16,606
samples, with no gap longer than 1.1 s. By `soak/soak_summary.py`'s definitions, read after the runs: GPU at most 64 °C
and 62.8 W (median 19.5 W); the hottest ACPI zone at most 82.8 °C (20:50:06 local, during B4); no clock-event reason
and no hardware slowdown ever active. The sim host's 1-min load (`load1`) averaged 2.32–3.50 per boot, with a maximum
of 8.21 (B5); read-only analyses ran on it from 20:29 (the record's §1, item 1).

**Timeline** (local, UTC−5, 2026-10-23; `campaign/campaign.log`, `campaign/interventions.log`,
`campaign/viewer_start.txt`, the boots' `boot.log` and `chain.log`; the evidence repository's `eeac8d9` commit time):

| time | event |
|---|---|
| 18:40:00 | the session's telemetry starts |
| 18:45:01 | the first campaign unit starts: B1 |
| 19:13:44 | B2 |
| 19:13:48 | the viewer starts |
| 19:56:42 | B2's slot-4 retry stopped by hand |
| 19:57:28 | the stall guard, stated for B2's re-run |
| 20:04:39 | the bridge's sim clock stops (B2, 391.4 s sim) |
| 20:06:01 | B2's chain stops (exit 6); the first unit ends |
| 20:07:17–20:08:37 | B2 torn down by hand; the hung Kit process killed |
| 20:08:57 | unit `a3-campaign-b3` starts: B3 |
| 20:29–21:33 | the post-hoc outside-room audit runs on the sim host, read-only (its own record, `lane-outside-room-goals-2026-10-23`) |
| 20:33:03 | B4 |
| 20:59:38 | B5 (the watchdog relaunched its bridge once) |
| 21:39:25 | B6 |
| 21:42 on | further read-only analyses on the sim host, not part of the record (their files' times; not deposited) |
| 22:12:29 | the count rule for B2's relaunch stated |
| 22:12:48 | B7 |
| 22:35:36 | the stall guard extended to every remaining transit |
| 22:46:50 | B7's chain stops (exit 6) |
| 22:47:13–22:47:55 | B7 torn down by hand |
| 22:48:10 | unit `a3-campaign-b2r` starts: B2r (the watchdog relaunched its bridge once) |
| 23:16:10 | B2r done; the campaign done |
| 23:16:38 | the stall guard stopped; the viewer removed |
| 23:16:47 | the telemetry ends |
| 23:18:04 | the robot host handed back |
| 23:22:07 | the amendment, the two scorer changes and `analysis/` committed (`eeac8d9`) |

**Registration and amendment.** `PREREGISTRATION.md` at `030ff40`, before the first boot. `AMENDMENT_RELAUNCH.md` at
`eeac8d9`, committed with the two scorer changes and `analysis/` six minutes after the campaign ended; its count rule
for B2r was logged at 22:12:29, before B2r ran.

## The rtabmap databases (not deposited)

Each boot's `fh_up.sh down` saved rtabmap's database (167–289 MB as rtabmap reports them; `boots/<RUN>/dgx/boot.log`).
They were copied from the robot host to the sim host (`~/Workspace/rig_lane_a3_dgx/robot/<RUN>/logs/`), with digests
equal on both hosts at the handback (`campaign/interventions.log`, 23:18:04), and re-checked against
`host/databases_sha256.txt` on 2026-10-24.

| sha256 | bytes | path |
|---|---:|---|
| `f36f8f23ef82e9b8e19cf4fee1141b274e025784baa7f1cb85ab94a8823f25cc` | 201994240 | `B1/logs/rtabmap_enrich_a31022B1.db` |
| `cdca5fc61a5f8457619975c2127758aba2f8afe9cf6e395bf7fd1473205a34eb` | 303263744 | `B2/logs/rtabmap_enrich_a31022B2.db` |
| `f7020c3ea447c3608ae023bf8bf78430858acf1cea9103193d1bde8e73c28042` | 193007616 | `B2r/logs/rtabmap_enrich_a31022B2r.db` |
| `a99041ed2fc528f9209f511347fb6eeaec925bae2375d2389daf17d5d708b662` | 175587328 | `B3/logs/rtabmap_enrich_a31022B3.db` |
| `dfe89d11a7cad13a179f941a0c75fb460a66e96de065b0e65ff31ed2c8be8fb6` | 194965504 | `B4/logs/rtabmap_enrich_a31022B4.db` |
| `9a7644fd69611bf92ec0cb360f242b91911e3a2a348aad9a977cbd7162a3d274` | 266354688 | `B5/logs/rtabmap_enrich_a31022B5.db` |
| `7b8e1fae07f841b28781d2e441b859a4bf396a5ce5e8ff05ec29a6568dddc5a0` | 242077696 | `B6/logs/rtabmap_enrich_a31022B6.db` |
| `d24be2e069677fe0ebe28df78c62a801adf5cdfbe60d6cf35cfc7a886e69a37d` | 223428608 | `B7/logs/rtabmap_enrich_a31022B7.db` |
