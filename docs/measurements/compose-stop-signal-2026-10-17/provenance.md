# Provenance: the compose stop signal, 2026-10-17

captured_date: 2026-10-17 (UTC and CDT); the test runs at the work commit on 2026-10-18 UTC

## Host

| role | host | system | kernel |
|---|---|---|---|
| every run | strafer-nx | Jetson Orin NX 16 GB, L4T R36.4.3 | 5.15.148-tegra |

- **No hardware in any run.** No container bound `/dev` or had the D555's `/dev/ttyACM0`, and no
  RoboClaw was connected.
- **The timed `slam` runs** used host networking, as the service does, under the compose project
  `stopsig` with `ROS_DOMAIN_ID` 87. No container existed after either arm
  (`slam/*/docker_ps_after.txt`).
- **Every probe** ran with `--network none`.

## Image and runner

| | |
|---|---|
| image | `strafer-cpu:humble`, revision label `9f7d20758d5b` (image `sha256:52d829c3…`, `slam/*/arm.txt`, `probes/probe_env.txt`) |
| launch | ros-humble-launch 1.0.14, ros-humble-launch-ros 0.19.13, ros-humble-ros2launch 0.19.13, Python 3.10.12 (`image_reads/launch_signal_paths.txt`) |
| test runner | `tools/run_ros_tests.sh`: `docker run --rm --network host`, mounting the tree's `source/strafer_ros` and `source/strafer_shared` only |
| driver probe | mounts `main`'s `strafer_driver` package, launch and config, `strafer_bringup/launch` and `strafer_shared` over the image's copies, and `strafer_driver/test` at `/probe/fence_test`. The image itself holds `9f7d207`'s driver |

## Trees

| run | tree |
|---|---|
| `slam/before/` | `main` at `0a975ef`, a clean detached worktree |
| `slam/after/` | the task branch's working tree before its work commit, with the compose `stop_signal` lines. Its rendered `slam` config equals the work commit `e7e6e2b`'s byte for byte (`slam/after/compose_config_slam_at_e7e6e2b.yml`) |
| `probes/` | `main` at `0a975ef` for the mounted driver and launch files; the probe files are in `probe/` |
| `tests/main/` | `main` at `0a975ef`, with `e7e6e2b`'s `test_bringup.py` copied in (sha256 in `tests/main/tree.txt`) |
| `tests/fix/` | `task/compose-stop-signal-sigint` at the work commit `e7e6e2b`, with this record's docs uncommitted; no test reads them (`tests/fix/tree.txt`) |

## Timeline (UTC, 2026-10-17)

| event | time |
|---|---|
| `slam` before: up, stop sent, stop returned, down | 19:35:38.8, 19:36:24.5, 19:39:24.8, 19:39:25.5 |
| `slam` after: up, stop sent, stop returned, down | 19:39:25.9, 19:40:11.5, 19:40:12.2, 19:40:12.9 |
| sealed probes, `sigterm_child` through `driver_sigint` | 19:40:51–19:42:33 |
| the two `--init` probes | 19:52:26–19:52:35 |
| tests on `main`, then the whole ROS suite, at an earlier, unpushed work commit (in the deposit's first commit; replaced since) | 19:50:14–19:51:00 |
| tests on `main`, then the whole ROS suite at `e7e6e2b`, then `make env-check` | 2026-10-18 00:03:38–00:04:30 |
| `slam` config rendered from `e7e6e2b` (`docker compose config`; no container) | 2026-10-18 00:05 |
