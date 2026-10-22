# Provenance: the chassis bring-up on the real RoboClaws, 2026-10-19

captured_date: 2026-10-19 CDT (2026-10-19 UTC)

## Host and hardware

| | |
|---|---|
| host | strafer-nx: Jetson Orin NX 16 GB, L4T R36.4.3, kernel 5.15.148-tegra, Docker 29.1.3 |
| controllers | two RoboClaw ST 2x45A, USB `03eb:2404`, on the hub at USB port `1-2`: front 0x80, rear 0x81 |
| camera | D555, CDC interface at `/dev/ttyACM0` (`8086:0b56`), attached throughout |
| power | one LiPo feeding both controllers through a main power switch that cuts the RoboClaws only; the Jetson has its own supply. *2026-10-21: as built, the LiPo is a 4S feeding both through a 300 W 12 V buck converter and a Y harness ([`chassis-stop-modes-2026-10-20`](../chassis-stop-modes-2026-10-20/README.md) §1).* |
| chassis | on blocks, all four wheels off the ground, from the first `base` start on |
| sim host | no container and no ROS process running (read over SSH before the first drive) |

The controllers' ports followed enumeration order.
- Until the proof's power cycle, the front was `ttyACM2` (`/dev/roboclaw2`) and the rear
  `ttyACM1`.
- After it, the front was `ttyACM1` (`/dev/roboclaw1`) and the rear `ttyACM2`.
- After the negative half the rear came back on hub port `1-2.4`, as `ttyACM2`.

Every driver and tool run found both by address.

## Tree and images

| | |
|---|---|
| tree | `main` at `371e848`, which merges `compose-stop-signal-sigint` (#248) and `roboclaw-serial-timeout` (#249), in a clean detached worktree (`images/tree.txt`) |
| images | `strafer-cpu:humble` `4473c937e6a3`, `strafer-gpu:humble` `c11bedfb3979`, both labelled `371e8487533a`, built by `make images` in that tree (`images/`) |
| previous images | `52d829c3656c` and `8513326715ec`, labelled `9f7d20758d5b`; after the build neither ID exists on the host, so there is no rollback image |
| compose | `source/strafer_ros/deploy` of that tree, with no override file; `base` with `stop_signal: SIGINT` (`host/compose_config_base.yml`) |
| udev | `99-strafer.rules` from that tree in `/etc/udev/rules.d/`, its sha256 equal to the tree's (`host/host_udev_mm.txt`) |

## Timeline (UTC)

| event | time |
|---|---|
| both RoboClaws enumerate, with main power already on; each USB then re-plugged once | 20:57:46–20:57:58 |
| `make images` | 21:19:35–21:36:53 |
| `99-strafer.rules` installed; reload; trigger of `ttyACM1` and `ttyACM2` only | 21:20:27 |
| the image's `--help` check | 21:37:10 |
| step 0: `--read-only` | 21:39:46 |
| step 0: front set and save; rear set and save | 22:49:43; 22:50:14 |
| a power cycle reported done by the maintainer, with no USB event in the kernel log; not counted | about 22:51 |
| the power cycle: USB out and back in (kernel log); main power off and on between, the maintainer's report, which the host cannot observe | 22:52:19–22:53:13 |
| step 0: the proof read | 22:53:20 |
| DDS census of domain 42 | 22:53:48 |
| §7's positive half: `base` up | 22:54:16 |
| the rear's USB out | 22:58:08 |
| §7's negative half: `base` recreated; 10 refusals | 22:58:43; 22:58:47–23:00:02 |
| the rear's USB back (port `1-2.4`); `base` ready | 23:00:16; 23:00:56 |
| `base` stopped for the duty test; the duty test; `base` up | 23:06:07; 23:06:19; 23:06:47 |
| `cmd_vel`: 0.1 m/s; 0.1 m/s again; 0.3 m/s | 23:12:22; 23:15:55; 23:16:10 |
| the first attempt at the compose-stop test, whose stop command was not sent (see below) | 23:17:24–23:17:36 |
| the spin with `docker compose stop base` | 23:18:19; stop at 23:18:25.271 |
| `base` up | about 23:19 |
| the spin with SIGKILL | 23:19:57; kill at 23:20:03.977 |
| `base` up, gates and encoder read | about 23:20:30 |
| `base` stopped | 23:22:11 |
| deposit commit `f4135db1` | after the runs |

## Departures from the planned procedure

- **The udev trigger.** `make udev` triggers every device. Here the trigger named the two RoboClaw
  ttys, so ModemManager would not handle a change event on the camera's CDC port. The rule file
  and the reload are `make udev`'s own.
- **The plug-order step ran before the rule was installed.** The controllers were powered and
  then plugged in at 20:57:46, and each re-plugged with power on. The rule's trigger then created
  the names.
- **The duty test ran before the first closed-loop drive,** not last, at the maintainer's choice.
  It was the polarity check in the record's §4.
- **The drive speed.** At 0.1 m/s two of the four wheels did not turn, so the stops (§6) used a
  0.8 rad/s spin. That turns each wheel at the same speed as the 0.3 m/s run, where all four
  moved.
- **The first attempt at the compose-stop test sent no stop.** It ran `docker compose stop base`
  from the wrong directory, which failed with "no configuration file provided".
  - The spin ran its full 12 s and ended with the publisher.
  - The driver's watchdog stopped the wheels 0.50 s after the last message.
  - That run is kept as `drive/step6b_attempt1_*` and shows the joint-position wrap in §7. The
    stop was then run again.
