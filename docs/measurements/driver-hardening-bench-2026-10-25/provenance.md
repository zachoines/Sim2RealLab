# Provenance: the driver hardening on the robot, 2026-10-25

captured_date: 2026-10-25 CDT (2026-10-25 UTC)

## Host and hardware

| | |
|---|---|
| host | strafer-nx: Jetson Orin NX 16 GB, L4T R36.4.3, kernel 5.15.148-tegra, Docker 29.1.3 |
| controllers | two RoboClaw ST 2x45A, USB `03eb:2404`, on the hub at port `1-2`: front 0x80 on `1-2.3`, rear 0x81 on `1-2.4`, each with a 0.5 s serial timeout saved. Their `/dev/roboclaw*` names followed the enumeration order, which changed across the power cycles (the record's §4) |
| depth camera | D555, CDC interface at `/dev/ttyACM0`, attached throughout |
| webcam | Logitech BRIO on hub port `1-2.2`, `/dev/video6`, on FR's hub as in `chassis-stop-modes-2026-10-20`; uvcvideo with `hwtimestamps=0` |
| power | a 4S LiPo, a 300 W 12 V buck converter, a Y harness to both controllers; main power cuts the RoboClaws only |
| chassis | on blocks, all four wheels off the ground, for every drive |

The questions put to the maintainer before each step, and the answers, are verbatim in the
deposit's `operator_notes.txt`.

## Tree and images

| | |
|---|---|
| tree | `main` at `0423a13`, the merge of #253, in a clean detached worktree |
| images | `strafer-cpu:humble` `c30981464f22` and `strafer-gpu:humble` `2183f6b18fdf`, both labelled `0423a13fcde6`, built by `make images` in that tree on 2026-10-22 |
| previous images | `4473c937e6a3` and `c11bedfb3979`, tagged `:pre-2026-10-21` before the build |
| compose | `source/strafer_ros/deploy` of that tree, with no override file; in its `docker-compose.yml`, `base` takes `restart: unless-stopped` and `stop_signal: SIGINT` from the `x-cpu-svc` anchor |
| publisher | `drive_probe.py` in a separate `strafer-cpu:humble` container on domain 42: `Twist(angular.z=0.8)` on `/cmd_vel` at 20 Hz, recording `/strafer/joint_states` |

## Timeline (UTC, 2026-10-25)

| event | time |
|---|---|
| both RoboClaws enumerate, rear first | 16:04:09–16:04:10 |
| pre-registration commit `0b4a7665` | 16:09:28 |
| step 0: `--read-only` | 16:10:13 |
| census; `base` up (entrypoint; driver ready) | before 16:10:29; 16:10:29.788; 16:10:32.168 |
| pause_3b: pause; unpause | 16:11:13.030; 16:11:16.086 |
| pause_3c: pause; unpause | 16:13:24.514; 16:13:27.572 |
| `base` stopped (SIGINT); the 1.0 s set | 16:18:37.349; 16:18:38 |
| `base` up (driver ready) | 16:18:48.127 |
| pause_dose: pause; unpause | 16:18:59.149; 16:19:02.212 |
| the dose test's power cycle, with `base` up: off USB; back, rear first | 16:20:40.973; 16:21:27.277–16:21:30.973 |
| the driver reconnects, gains read back | 16:21:31.390 |
| `base` stopped (SIGINT); `--read-only` | 16:22:04.264; 16:22:04 |
| `base` up for step 3 (driver ready) | 16:22:19.983 |
| first cut, USB in: off USB; back, front first, swapped | 16:24:41.617; 16:24:51.569–16:24:52.021 |
| `base` restarted: stopped; driver ready | 16:25:33.143; 16:25:35.806 |
| second cut, USB in: off USB; back, front first | 16:26:22.741; 16:26:34.741–16:26:35.193 |
| the driver reconnects, gains read back | 16:26:36.968 |
| the spin: first and last message | 16:28:17.225; 16:28:19.579 |
| `base` stopped (SIGINT); `--read-only` | 16:28:39.852; 16:28:40 |
| deposit commits `8242cafb`; `862cf84d` (the late sample's positions) | 16:31:51; 16:34:16 |
| deposit correction `5953ef77`, and `520d984f`, which dates it in UTC (`DEPOSIT.md`'s correction of 2026-10-30) | 2026-10-30 03:00:11; 03:00:41 |

## Departures from the pre-registration

- **pause_3b ran without the camera.** `pause_run.sh` was launched by a relative path and resolved
  `wheelcam.py` against the compose tree it had changed into. The script was then made to resolve
  its own directory, and pause_3c repeated the run with the camera. Both versions are deposited.
- **The step-0 read is recovered, filtered.** Its log file was not written, because `diag.sh` was
  given a relative path. The command's terminal output, filtered to the lines that carry the
  figures, is deposited.
- **Step 2.4's power cycle ran with `base` up.** The question to the maintainer announced the stop
  of `base`, but it came only after the cycle. With no publisher, nothing could drive, though no
  encoder read covers the cycle; it is kept as an unplanned reconnect. That the USB went out before
  main power rests on the maintainer's answer: the kernel log is equally consistent with main
  power first.
- **Step 3 needed a second cut.** The first came back with the names swapped, the pre-registered
  competing outcome. `base` was restarted, and a second cut, asked of the maintainer then, gave
  the planned case.
- **The spin's publisher ran 2.35 s, not 4 s.** The probe container's start-up used the rest of
  the 4 s window, so the means are taken over 1.75 s, from 0.6 s after the first message to the
  last, which includes the ramp (the record's §4).
