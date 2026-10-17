# Provenance — training-style goals on the sim-bridge lane (the scene axis), 2026-10-14

captured_date: 2026-10-14

## Hosts

| role | host | kernel | notes |
|---|---|---|---|
| sim | gx10-d1d8 (NVIDIA GB10; Kit reports processor Cortex-A725, 20 cores; aarch64) | 7.0.0-1019-nvidia | driver 580.173.02, Ubuntu 24.04.5 (Kit's banner, `logs/dgx/bridge.log`); the bridge through `dgx/bridge_grid_dump.py`, the planner and VLM services, and the drivers (`dgx/n2_session.sh`, `dgx/n2_chain.sh`) |
| robot | strafer-nx | not recorded in this deposit | the deploy stack in five containers (`session/nx_up.sh`), the CLI sets' harness with `tools/run_cli_one_start.sh`, the stage-0 checks |

The direct link carried DDS and the planner and VLM calls, with the direct-link Cyclone configuration on
the bridge and the direct-link interface selected in every robot container
(`stage0/cyclone_pin_in_containers.txt`). The robot host's module, L4T and kernel are as read for
[`goal-a-cli-video-2026-10-09`](../goal-a-cli-video-2026-10-09/provenance.md).

## Tree and stack

| side | tree | stack |
|---|---|---|
| sim | the main checkout at `6cd6fd7` (`git=6cd6fd7` on line 1 of `logs/dgx/bridge.log`); the wrapper and the sampler are deposit-only | conda `env_isaaclab3`; the planner and VLM are the speed-cap A/B's processes, serving `SYSTEM_PROMPT` sha256 `5f8d805d…` |
| robot | images labelled `9f7d20758d5b` | as the speed-cap A/B's (`stage0/images.txt`) |

**The goal draw.** `a4_goals.py` loads `planner.py` from `6cd6fd7` by path (sha256 `e922c2ab…`,
`goals/inputs_sha256.txt`) and reads `goals/grid_dump.npz` (`36929c68…`), the file the wrapper wrote at
the boot's reset (`goals/grid_dump_line.txt`).

**The node.** The deployed caps through `node/inference_cap_C.yaml` (`528f81ba…`), mounted read-only
(`stage0/cap_mount.txt`); the subgoal generator as on `main`.

## Artifact

`strafer_depth_subgoal_v3_999.onnx` sha256 `c866bfd5…` and its sidecar `d22e3504…`, read inside the
running `inference` container (`stage0/in_container_artifact_checks.txt`).

## GPU and Kit

- **Before the launch** the GPU listed only the two `.venv_vlm` service processes, 38892 and 38888
  (`logs/dgx/gpu_before_bridge.txt`).
- **The boot.** Kit's log begins at 04:26:29 UTC; the robot stack came up at 04:26:57 with the fresh SLAM
  key `lanetermA4`. The watchdog's last line, at the end of `logs/dgx/bridge.log`, reads
  `attempt 1: exit=0 wall=4320s`.
- **Teardown.** `stop-bridge` stopped Kit through the watchdog's child group; the services were stopped
  by their recorded groups afterwards, with 0 GPU compute processes left. The robot host's deploy
  `.env` was restored from the backup made before the evening's first token change and compared equal;
  the backup is not deposited.

## Timeline (2026-10-14 to 2026-10-15 CDT; UTC is 5 h later)

| slot | reach | arm set and read back | transit (s sim) | goal accepted | result | slot done |
|---:|---|---|---|---|---|---|
| 1 | `S01_A_C` | 23:34:11 (0.7841 m/s, 2.0933 rad/s) | 4.23 | 23:35:43 | 23:36:44 | 23:36:58 |
| 2 | `S02_B_C` | 23:37:15 (0.7841 m/s, 2.0933 rad/s) | 22.38 | 23:41:00 | 23:42:32 | 23:42:51 |
| 3 | `S03_C_C` | 23:43:14 (0.7841 m/s, 2.0933 rad/s) | 26.27 | 23:47:27 | 23:48:27 | 23:48:34 |
| 4 | `S04_D_C` | 23:48:51 (0.7841 m/s, 2.0933 rad/s) | 34.90 | 23:54:12 | 23:55:50 | 23:55:59 |
| 5 | `S05_E_C` | 23:56:16 (0.7841 m/s, 2.0933 rad/s) | 150.62 FAIL, 150.62 FAIL | — | — | 00:34:44 |

S05's two transit attempts each ran out the 150 s sim budget. The chain was stopped at 00:34:56, 12 s
after S05's done line, while slot 6's arm switch was under way; the `inference` container came up again
at 00:35:02, and no slot-6 reach started.
