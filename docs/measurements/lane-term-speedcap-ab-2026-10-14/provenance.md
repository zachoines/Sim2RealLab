# Provenance — the node's speed cap on the sim-bridge lane, 2026-10-14

captured_date: 2026-10-14

## Hosts

| role | host | kernel | notes |
|---|---|---|---|
| sim | gx10-d1d8 (NVIDIA GB10; Kit reports processor Cortex-A725, 20 cores; aarch64) | 7.0.0-1019-nvidia | driver 580.173.02, Ubuntu 24.04.5 (Kit's banner, `session1/logs/dgx/bridge.log`); the bridge, the planner and VLM services, and the drivers (`dgx/n2_session.sh`, `dgx/n2_chain.sh`, `session2/n2_chain_b.sh`) |
| robot | strafer-nx | not recorded in this deposit | the deploy stack in five containers (`session/nx_up.sh`), the CLI sets' harness, the stage-0 checks (`session/stage0.sh`) and the frame check (`session2/frame_check.py`) |

The two hosts were joined by the direct link, which carried DDS and the planner and VLM calls; the sim
host drove each session over it by ssh. The bridge used the direct-link Cyclone configuration (line 1
of each `bridge.log`), and every robot container selected the direct-link interface
(`stage0*/cyclone_pin_in_containers.txt`). The robot host's module, L4T and kernel were read on
2026-10-10 for [`goal-a-cli-video-2026-10-09`](../goal-a-cli-video-2026-10-09/provenance.md): a Jetson
Orin NX 16 GB, L4T R36.4.3, kernel 5.15.148-tegra.

## Tree and stack

| side | tree | stack |
|---|---|---|
| sim | the main checkout at `6cd6fd7` (`git=6cd6fd7` on line 1 of each `bridge.log`) | conda `env_isaaclab3`, the canonical Isaac Sim 6.0.1.0 + Isaac Lab v3.0.0-beta2.patch1 pair; the planner and VLM in `.venv_vlm` (`make serve-planner`, `make serve-vlm`), serving `SYSTEM_PROMPT` sha256 `5f8d805d…` (`session1/logs/dgx/planner_served.txt`) |
| robot | images built at `9f7d207`, revision label `9f7d20758d5b` | `strafer-gpu:humble` `sha256:85133267…` (`inference`); `strafer-cpu:humble` `sha256:52d829c3…` (the other four) (`stage0/images.txt`, `session2/stage0*/images.txt`) |

**The node.** `node/inference_cap_C.yaml` (sha256 `528f81ba…`) and `node/inference_cap_U.yaml`
(`ddc3d02f…`) are the image's own `inference.yaml` (`node/inference_image.yaml`, `b2842100…`, the same
file as `main`'s) plus four lines setting the caps. `node/cap_overlay.yml` mounts the arm's file
read-only over the installed copy; `stage0/cap_mount.txt` reads the same sha256 on the host and in the
container. The subgoal generator is the image's, as on `main`: no `referent_arm` parameter,
`subgoal_anchoring` `mission`, `lookahead_m` 1.0.

**The robot's compose chain** (`session/nx_up.sh`): profiles `policy` and `sim-bridge`;
`docker-compose.yml` and `docker-compose.override.sim-bridge.yml` from the robot host's checkout and
its two host-local overrides, which are not in the tree; the CLI set's direct-link autonomy URLs; the
cap overlay; five named services: `inference`, `slam`, `navigation`, `sim-perception`, `autonomy`.
The robot host's checkout was at `6715cf3` with two host-local env-file edits, as recorded by every
stage-0 `cap_mount.txt`; the containers run the images, not the checkout's sources.

**The harness.** `~/rig_cli_confirm/tools` ran unchanged: `stage0/harness_sha256.txt` holds the 21
harness digests of the CLI confirmation set's `tools_sha256.txt` and `session/stage0.sh`'s.

## Artifact

| file | sha256 |
|---|---|
| `strafer_depth_subgoal_v3_999.onnx` | `c866bfd54ec1a8352159e33d7875d41e3f07a442ff8301ba3700867932e2eb91` |
| `strafer_depth_subgoal_v3_999.json` | `d22e3504e6aa2f0cefbcf2af41561c8acbab4ecf87a8c2e3a4b494294bf33ab5` |

Both read inside the running `inference` container (`stage0*/in_container_artifact_checks.txt`).

## GPU and Kit

There was one Kit actor at a time, the bridge.
- **The check before each launch** refused any compute process other than the `.venv_vlm` services
  (`*/gpu_before_bridge.txt`: PIDs 38892 and 38888 only, the VLM's and the planner's server
  processes).
- **The boots**, each on the watchdog's first attempt and each ending `exit=0` (the watchdog's last line,
  at the end of each boot's `bridge.log`):

| boot | Kit log begins (UTC) | robot stack up (UTC) | SLAM key | watchdog |
|---|---|---|---|---|
| first session | 00:30:59 | 00:31:22 | `lanetermA3` | attempt 1: exit=0 wall=7102s |
| second session, leg 1 | 02:30:18 | 02:30:44 | `lanetermA3b` | attempt 1: exit=0 wall=1444s |
| second session, leg 2 | 02:54:30 | 02:54:44 | `lanetermA3c` | attempt 1: exit=0 wall=5485s |

- **The services.** One planner and one VLM process served both lane-term records' sessions. The
  planner served 50 `/plan` requests over the evening: per submitted reach a dry-check and a mission,
  per refused reach a dry-check. The VLM served health checks only. Both logs end `Application
  shutdown complete` (`session1/logs/dgx/serve_*.log`).
- **Teardown.** The first bridge did not stop on a signal to its recorded process group; Kit runs in
  the launcher's own group, which was signalled by number. Later bridges were stopped by
  `stop-bridge`, which signals the watchdog's child groups. The services were stopped by their recorded
  groups at the end of the scene-axis session.

## Timeline (2026-10-14 CDT; UTC is 5 h later)

**First session.** Times from `session1/logs/dgx/chain.log`, the transit logs and the observer records.

| slot | reach | arm set and read back | transit (s sim) | goal accepted | result | slot done |
|---:|---|---|---|---|---|---|
| 1 | `S01_G1_U` | 19:35:29 (1.5683 m/s, 4.1866 rad/s) | 9.82 | 19:37:33 | 19:38:28 | 19:38:47 |
| 2 | `S02_L2_C` | 19:39:09 (0.7841 m/s, 2.0933 rad/s) | 19.94 | — | — | 19:42:27 |
| 3 | `S03_L2_U` | 19:42:50 (1.5683 m/s, 4.1866 rad/s) | 1.69 | 19:44:11 | 19:45:47 | 19:45:53 |
| 4 | `S04_G1_C` | 19:46:15 (0.7841 m/s, 2.0933 rad/s) | 31.58 | 19:50:47 | 19:56:27 | 19:56:42 |
| 5 | `S05_G1_U` | 19:56:59 (1.5683 m/s, 4.1866 rad/s) | 21.99 | 20:00:26 | 20:01:32 | 20:01:49 |
| 6 | `S06_L2_C` | 20:02:12 (0.7841 m/s, 2.0933 rad/s) | 20.34 | 20:05:48 | 20:07:38 | 20:07:47 |
| 7 | `S07_L2_U` | 20:08:10 (1.5683 m/s, 4.1866 rad/s) | 46.15 | 20:14:47 | 20:15:47 | 20:16:03 |
| 8 | `S08_G1_C` | 20:16:26 (0.7841 m/s, 2.0933 rad/s) | 20.84 | 20:19:39 | 20:20:25 | 20:20:45 |
| 9 | `S09_G1_U` | 20:21:07 (1.5683 m/s, 4.1866 rad/s) | 46.80 | — | — | 20:27:44 |
| 10 | `S10_L2_C` | 20:28:07 (0.7841 m/s, 2.0933 rad/s) | 1.68 | 20:29:27 | 20:30:32 | 20:30:39 |
| 11 | `S11_L2_U` | 20:30:57 (1.5683 m/s, 4.1866 rad/s) | 150.63 FAIL, 150.63 FAIL | — | — | — |
| 12 | `S12_G1_C` | 21:11:49 (0.7841 m/s, 2.0933 rad/s) | 95.27 INTERRUPTED (stopped while pinned) | — | — | — |

S12's transit was stopped while pinned: its `.out` log has no result line, and its `to_start.json` records
95.27 s sim, `INTERRUPTED`. The chain was stopped by its
process group at 21:02:05 (during S11's wait), relaunched at 21:11:32 for S12 and the re-runs after the
nudge, and stopped again at 21:24:15.

**Second session.** Times from `session2/chain_all.log`, the transit logs and the observer records.

| slot | reach | arm set and read back | transit (s sim) | goal accepted | result | slot done |
|---:|---|---|---|---|---|---|
| 1 | `B01_G1_U` | 21:39:04 (1.5683 m/s, 4.1866 rad/s) | 7.78 | 21:40:46 | 21:41:26 | 21:41:36 |
| 2 | `B02_L2_C` | 21:41:55 (0.7841 m/s, 2.0933 rad/s) | 40.24 | 21:48:01 | 21:49:44 | 21:50:03 |
| 3 | `B03_L2_U` | 22:02:01 (1.5683 m/s, 4.1866 rad/s) | 7.87 | 22:04:13 | 22:05:05 | 22:05:19 |
| 4 | `B04_G1_C` | 22:05:37 (0.7841 m/s, 2.0933 rad/s) | 63.95 | 22:14:14 | 22:15:01 | 22:15:17 |
| 5 | `B05_G1_U` | 22:15:41 (1.5683 m/s, 4.1866 rad/s) | 29.34 | 22:19:59 | 22:21:11 | 22:21:32 |
| 6 | `B06_L2_C` | 22:21:50 (0.7841 m/s, 2.0933 rad/s) | 60.37 | 22:30:26 | 22:31:53 | 22:32:01 |
| 7 | `B07_L2_U` | 22:32:25 (1.5683 m/s, 4.1866 rad/s) | 23.44 | 22:36:25 | 22:37:17 | 22:37:30 |
| 8 | `B08_G1_C` | 22:37:48 (0.7841 m/s, 2.0933 rad/s) | 103.28 | 22:51:14 | 22:52:05 | 22:52:19 |
| 9 | `B09_G1_U` | 22:52:42 (1.5683 m/s, 4.1866 rad/s) | 31.09 | 22:57:14 | 22:58:04 | 22:58:18 |
| 10 | `B10_L2_C` | 22:58:42 (0.7841 m/s, 2.0933 rad/s) | 41.27 | 23:05:25 | 23:06:30 | 23:06:50 |
| 11 | `B11_L2_U` | 23:07:09 (1.5683 m/s, 4.1866 rad/s) | 41.17 | 23:13:46 | 23:15:25 | 23:15:33 |
| 12 | `B12_G1_C` | 23:15:56 (0.7841 m/s, 2.0933 rad/s) | 34.54 | 23:21:14 | 23:22:06 | 23:22:18 |

The frame gate stopped leg 1 before slot 3 at 21:50:05; leg 2's chain started at 22:01:43 and logged
`chain done` at 23:22:18.
