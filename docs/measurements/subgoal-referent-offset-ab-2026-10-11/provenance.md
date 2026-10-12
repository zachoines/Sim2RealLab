# Provenance — the anchored plan's end against the goal pose as the policy's referent on the sim-bridge lane, 2026-10-11

captured_date: 2026-10-11

## Hosts

| role | host | kernel | notes |
|---|---|---|---|
| sim | gx10-d1d8 (NVIDIA GB10; Kit reports processor Cortex-A725, 20 cores; aarch64) | 7.0.0-1019-nvidia | driver 580.173.02, Ubuntu 24.04.5; the bridge, the planner and VLM services, and the session's driver scripts (`session/d_session.sh`, `session/d_chain.sh`) |
| robot | strafer-nx | not recorded in this deposit | the deploy stack in five containers (`session/nx_up.sh`), the CLI sets' harness and the stage-0 checks (`session/stage0.sh`) |

The two hosts were joined by the direct link. It carried DDS and the planner and VLM calls, and
the sim host drove the session over it by ssh.

Sources:
- `DEPOSIT.md` names both hosts.
- Kit's banner in `logs/dgx/bridge.log` (lines 11–29) records the sim host's GPU, driver, OS,
  kernel, processor and cores at session time.
- The architecture, aarch64, is from `logs/dgx/envsetup.log`: its libgomp and Isaac Sim build
  paths are aarch64 ones (lines 2 and 7).
- The robot host's name also appears in the `inference` container's launch log path
  (`logs/nx/strafer_inference.log`, line 2).
- This deposit does not record the robot host's module, L4T or kernel. They were read on 2026-10-10
  for [`goal-a-cli-video-2026-10-09`](../goal-a-cli-video-2026-10-09/provenance.md): a Jetson Orin
  NX 16 GB, L4T R36.4.3, kernel 5.15.148-tegra.
- The bridge used the direct-link Cyclone configuration (`direct-link.xml`, `logs/dgx/bridge.log`
  line 1). Every robot container selected the direct-link interface
  (`stage0/cyclone_pin_in_containers.txt`).

## Tree and stack

| side | tree | stack |
|---|---|---|
| sim | the main checkout at `9f7d207` (`git=9f7d207` on line 1 of `logs/dgx/bridge.log`). `d_session.sh services` refuses to start while that checkout has tracked changes | conda `env_isaaclab3`, with Kit's IsaacLab 3.0 user config (`logs/dgx/bridge.log` line 4). This is the canonical Isaac Sim 6.0.1.0 + Isaac Lab v3.0.0-beta2.patch1 pair recorded by `isaac-lab-upgrade-landing-2026-09-12`; this deposit does not print the versions. The planner and VLM ran from the same checkout in `.venv_vlm` (`make serve-planner`, `make serve-vlm`) |
| robot | images built at `9f7d207`, revision label `9f7d20758d5b` | `strafer-gpu:humble`, image `sha256:85133267…` (`inference`); `strafer-cpu:humble`, image `sha256:52d829c3…` (`slam`, `navigation`, `sim-perception`, `autonomy`) (`stage0/images.txt`) |

**The node.**
- `node/subgoal_generator_node.main.py` has sha256
  `52c9beed6bcec7392a6f625ee44b3cad8c8e6594b898924b5cb7564ea04fa9a5`. That is the image's
  installed copy, and the same bytes as `9f7d207`'s
  `source/strafer_ros/strafer_inference/strafer_inference/subgoal_generator_node.py`, as checked
  with `git show` for this record.
- `node/subgoal_generator_node.py`, the mounted file, has sha256
  `c8085992c6b1492c117ae103591152615f0031f9f35b94d0bac2fc7f48322fb5`. It is that file plus
  `referent_arm.patch`.
- `stage0/referent_node_mount.txt` reads the same sha256 on the host and inside the `inference`
  container, and shows the mount read-only. It also shows that the module imports from the
  mounted path, and the generator's parameters at stage 0: `referent_arm` `plan_end`,
  `subgoal_anchoring` `mission`, `lookahead_m` 1.0.

**The robot's compose chain** (`session/nx_up.sh`):
- profiles `policy` and `sim-bridge`;
- `docker-compose.yml` and `docker-compose.override.sim-bridge.yml` from the robot host's
  checkout, and its two host-local overrides, `docker-compose.override.yml` and
  `docker-compose.override.autonomy-local.yml`, which are not in the tree;
- the CLI set's direct-link autonomy URLs, `~/rig_cli_confirm/compose/autonomy-urls-direct.yml`;
- the node mount, `node/inference-referent-node.yml`;
- five named services: `inference`, `slam`, `navigation`, `sim-perception`, `autonomy`.

**The harness.** `~/rig_cli_confirm/tools` ran unchanged. `stage0/harness_sha256.txt` holds 22
digests. Its first 21 are the harness tools', and each matches its entry in the CLI confirmation
set's `tools_sha256.txt` (`goal-a-cli-confirmation-2026-10-03/record-files/` in the evidence
repository), as checked for this record. The 22nd is `session/stage0.sh`'s and matches the
deposited copy.

## Artifact

| file | sha256 | matches |
|---|---|---|
| `strafer_depth_subgoal_v3_999.onnx` (in the `inference` container) | `c866bfd54ec1a8352159e33d7875d41e3f07a442ff8301ba3700867932e2eb91` | the exported-artifact row of `depth-subgoal-v3-retrain-2026-09-21`, as in the CLI sets |
| `strafer_depth_subgoal_v3_999.json` | `d22e3504e6aa2f0cefbcf2af41561c8acbab4ecf87a8c2e3a4b494294bf33ab5` | the same |

Both are read inside the running container (`stage0/in_container_artifact_checks.txt`), where the
node logged `Loaded policy from /models/strafer_depth_subgoal_v3_999.onnx (recurrent=True)`.

## GPU and Kit

There was one Kit actor on the sim host, the bridge.
- **The check before launch.** `d_session.sh bridge` refuses to launch while any compute process
  other than the `.venv_vlm` services is listed, or while any bridge or watchdog process runs.
  `logs/dgx/gpu_before_bridge.txt` lists exactly two compute processes, PIDs 3881866 and 3881865.
  Those are the VLM's and the planner's server processes (`Started server process` in
  `logs/dgx/serve_vlm.log` and `logs/dgx/serve_planner.log`).
- **The bridge.** Kit's log begins at 01:10:44 UTC (20:10:44 CDT, `logs/dgx/bridge.log` line 5).
  The bridge booted on the watchdog's first attempt and ran all twelve reaches. The last line of
  `logs/dgx/bridge.log` (line 280), the watchdog's, reads `attempt 1: exit=0 wall=6701s`.
- **The services.** The planner served 24 `/plan` requests: twelve dry-checks and twelve missions.
  The VLM served health checks only. Both logs end with `Application shutdown complete`.
- **Teardown.** `d_session.sh stop-local` stopped the bridge and both services by signal. It also
  signalled the shell that invoked it, the defect that `DEPOSIT.md` records.

## Timeline (2026-10-11 CDT; UTC is 5 h later)

| | |
|---|---|
| bridge's Kit log begins | 20:10:44 (`logs/dgx/bridge.log`) |
| robot stack up, five services, fresh SLAM key `enrich_referentab1` | 20:11:10 (`logs/nx/up_t0.txt`) |
| policy loaded in the `inference` container | 20:11:17 |
| stage-0 service health round-trip | 20:11:55 (`stage0/service_health.txt`) |
| pre-registration written, and committed as `8a7648f` | 20:13:09 |
| the twelve reaches | 20:13:33 to 21:59:25 (below) |
| bridge exits, 6701 s wall after launch | about 22:02 |
| deposit committed as `a213f88` | 22:05:41 |

Per reach, all times to the second, truncated. The arm set and read back, and slot done, are from
`logs/dgx/chain.log`. Transit is the robot's scripted transit to the start, from
`logs/nx/runs/<label>.to_start.out`. Accepted and result are the policy goal's, from the observer's
record. The last column is `chain.log`'s running count of the node's `referent_arm=snap` lines.

| slot | reach | arm set and read back | transit (s sim) | goal accepted | result | slot done | snap lines so far |
|---:|---|---|---:|---|---|---|---:|
| 1 | `D01_G1_T` | 20:13:33 `snap` | 11.30 | 20:15:38 | 20:17:04 | 20:17:22 | 1 |
| 2 | `D02_L2_C` | 20:17:25 `plan_end` | 79.73 | 20:28:13 | 20:29:40 | 20:29:55 | 1 |
| 3 | `D03_L2_T` | 20:29:58 `snap` | 42.65 | 20:36:09 | 20:37:20 | 20:37:37 | 2 |
| 4 | `D04_G1_C` | 20:37:41 `plan_end` | 38.21 | 20:42:51 | 20:44:41 | 20:44:48 | 2 |
| 5 | `D05_G1_T` | 20:44:52 `snap` | 47.58 | 20:51:14 | 20:52:23 | 20:52:30 | 3 |
| 6 | `D06_L2_C` | 20:52:34 `plan_end` | 32.38 | 20:57:28 | 20:59:01 | 20:59:11 | 3 |
| 7 | `D07_L2_T` | 20:59:15 `snap` | 46.00 | 21:05:46 | 21:06:43 | 21:06:54 | 4 |
| 8 | `D08_G1_C` | 21:06:57 `plan_end` | 46.17 | 21:13:03 | 21:19:11 | 21:19:26 | 4 |
| 9 | `D09_G1_T` | 21:19:30 `snap` | 53.67 | 21:27:08 | 21:28:29 | 21:28:40 | 5 |
| 10 | `D10_L2_C` | 21:28:44 `plan_end` | 55.89 | 21:37:11 | 21:38:08 | 21:38:25 | 5 |
| 11 | `D11_L2_T` | 21:38:28 `snap` | 50.94 | 21:46:10 | 21:47:35 | 21:47:54 | 6 |
| 12 | `D12_G1_C` | 21:47:57 `plan_end` | 70.51 | 21:57:51 | 21:59:09 | 21:59:25 | 6 |

`chain.log` reads `chain done` at 21:59:25.
