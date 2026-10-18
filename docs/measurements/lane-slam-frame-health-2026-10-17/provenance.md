# Provenance — the lane's SLAM frame, 2026-10-17

captured_date: 2026-10-17 (the boots ran 2026-10-18 00:02–04:27 UTC)

## Hosts

| role | host | kernel | notes |
|---|---|---|---|
| sim | gx10-d1d8 (NVIDIA GB10, aarch64) | 7.0.0-1019-nvidia | driver 580.173.02 (Kit's banner in each `boots/*/dgx/bridge.log`). Runs the bridge, the planner and VLM services, and the session's drivers (`dgx/`). From F2r on these ran as systemd user units (`AMENDMENT_F2.md`). From F2r2 on, a 1 Hz telemetry log also ran (`host/telemetry.csv`, from 02:06:42Z). The sim-side reads from `bridge.log` cover F2r–F6; F1's and F2's bridge logs were lost (see Interruptions). |
| robot | strafer-nx | not recorded in this deposit | Runs the deploy stack in five containers (`session/fh_up.sh`), the CLI sets' harness, stage 0 (`session/stage0_fh.sh`) and the frame check (`tools/frame_check4.py`). |

**The link.** The two hosts were joined by the direct link, which carried DDS and the planner and VLM calls; the sim
host drove each boot over it by ssh.
- **Sim side:** the bridge ran on ROS domain 42 with the direct-link Cyclone configuration (line 1 of each
  `bridge.log`).
- **Robot side:** every container selected the direct-link interface (`boots/*/robot/stage0/
  cyclone_pin_in_containers.txt`).
- **The robot host's hardware,** read on 2026-10-10 for `goal-a-cli-video-2026-10-09`: a Jetson Orin NX 16 GB,
  L4T R36.4.3, kernel 5.15.148-tegra.

## Tree and stack

| side | tree | stack |
|---|---|---|
| sim | the main checkout at `0a975ef` (`git=0a975ef5` on line 1 of each `bridge.log`); the lane's sources are unchanged since `6cd6fd7` except two comments in `constants.py` | conda `env_isaaclab3` (Isaac Sim 6.0.1.0 + Isaac Lab v3.0.0-beta2.patch1); the planner and VLM in `.venv_vlm` (`make serve-planner`, `make serve-vlm`), Qwen3-4B and Qwen2.5-VL-3B-Instruct |
| robot | images built at `9f7d207`, revision label `9f7d20758d5b`; the checkout at `6715cf3` with its two host-local env-file edits, as on 2026-10-14 | `strafer-gpu:humble` `sha256:85133267…` (`inference`), `strafer-cpu:humble` `sha256:52d829c3…` (the other four) (`boots/*/robot/logs/up_checks.txt`); rtabmap 0.23.7 |

**The arms.**
- `node/rtabmap_params_image.yaml` (sha256 `e67c97ff…`) is byte-equal to the image's installed file and to `main`'s.
- `node/rtabmap_params_B.yaml` (`4ebd1807…`) differs from it in three lines.
- `node/slam_params_overlay.yml` mounts the boot's file read-only over the installed copy.
- In every boot, `logs/up_checks.txt` reads one sha256 on the host and in the container, and
  `logs/rtabmap_params_check.txt` reads `PARAMS OK` for seven keys in the node's parameter dump.

**The robot's compose chain** (`session/fh_up.sh`):
- profiles `policy` and `sim-bridge`;
- `docker-compose.yml` and `docker-compose.override.sim-bridge.yml` from the robot host's checkout, plus its two
  host-local overrides, which are not in the tree;
- the CLI set's direct-link autonomy URLs, and the params overlay;
- five named services: `inference`, `slam`, `navigation`, `sim-perception`, `autonomy`.

The containers run the images, not the checkout's sources. `logs/compose_config.yml` holds each boot's resolved
configuration.

**The harness.** `~/rig_cli_confirm/tools` ran unchanged, and stage 0's `harness_sha256.txt` holds its digests and the
session files'. The deposited reach wrapper `tools/run_cli_one_start.sh` is byte-equal to the scene-axis session's
(`a49bf7b2…`).

## Interruptions

- **Power.** The sim host lost power at about 00:54:55Z and 01:51:19Z on 2026-10-18 (`AMENDMENT_CRASH.md`).
- **Lost logs.** F1's and F2's sim-host logs were in `/tmp`, which the first loss cleared (`host/journal_tail_boot-2.txt`, `host/last_x.txt`). F1's chain log is a
  transcription; F2's sim-host logs are lost. The robot host's files for every boot are complete.
