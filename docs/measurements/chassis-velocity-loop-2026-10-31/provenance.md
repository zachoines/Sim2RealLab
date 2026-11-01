# Provenance: the chassis velocity loop on blocks, 2026-10-31

captured_date: 2026-10-30 and 2026-10-31 CDT (2026-10-31 and 2026-11-01 UTC)

## Host and hardware

| | |
|---|---|
| host | strafer-nx: Jetson Orin NX 16 GB, L4T R36.4.3, kernel 5.15.148-tegra |
| controllers | two RoboClaws on the host's USB2 hub `1-2` (`0424:2744`): front 0x80 on `1-2.3` (`ttyACM1`, `/dev/roboclaw1`), rear 0x81 on `1-2.4` (`ttyACM2`, `/dev/roboclaw2`), in both sessions and after both flash saves; each with a 0.5 s serial timeout saved |
| wheels | four mecanum wheels, 537.7 encoder ticks per revolution, radius 0.048 m |
| power | a 4S LiPo, a 300 W 12 V buck converter and a Y harness to both controllers. By the maintainer's meter at idle: session 1, the pack 15.33 V and both controllers' leads 12 V; session 2, after a recharge, 16.68 V and 12.07 V |
| webcam | Logitech BRIO on hub port `1-2.1`, `/dev/video6`, about 15 cm from FR; its box and threshold fixed before any motor command, and re-aimed at FR for session 2. It recorded no run |
| chassis | on blocks, all four wheels off the ground, throughout |

The questions put to the maintainer before each step, and the answers, are verbatim in the
deposit's `operator_notes.txt`.

## Tree and images

| | |
|---|---|
| tree | `task/chassis-velocity-pid-tuning` in a worktree: at `main`'s `061a5fb` through step 5; the constants commit `b77b06d` (QPPS 2657) for step 6, amended after it as `ac39454`; `4624293`, which adds `diagnose_roboclaw.py --set-gains`, for the flash save |
| images | `strafer-cpu:humble` `c30981464f22` and `strafer-gpu:humble` `2183f6b18fdf`, both labelled `0423a13fcde6`, built 2026-10-22 (the GPU image's ID as in `driver-hardening-bench-2026-10-25`'s provenance); `source/` is unchanged from `0423a13` to `061a5fb`. Every container here ran the CPU image |
| bench tool | `motor_bench.py` in one-off containers of the `base` service, named `strafer_bench`, through compose from the worktree; `bench.sh` refuses while any `base`-service container is up. Session 1 ran the version kept as `scripts/motor_bench_steps1-4.py`, session 2 `scripts/motor_bench.py` as deposited at `8d0dc89a`, which refuses duty 127 |
| step 1a | `diagnose_roboclaw.py --drive` from the image, in a one-off `base` container |
| step 6 | `base` through `docker-compose.dev.yml`, which mounts the worktree's source over the image's (`s6/base_mounts.txt`); `drive_probe.py` in a separate `strafer-cpu:humble` container on domain 42 |
| flash save | `diagnose_roboclaw.py` from the worktree at `4624293`, through the same overlay, in a one-off `base` container with `base` stopped. Its wrapper, `flash.sh`, refuses if the worktree's tool or constants differ from `4624293` |

## Timeline (UTC)

| event | time |
|---|---|
| 2026-10-31: both RoboClaws enumerate, the USB plugged in after main power | 04:05:43 |
| gate 0.2: the read | 04:35:34 |
| the webcam's box and threshold fixed | 04:39:20 |
| gate 0.1: the meter readings; the session set to steps 0 to 4 | 04:49:52 |
| pre-registration commit `c19fe7de` | 05:08:33 |
| step 1a: read before; `--drive`; read after | 05:08:43; 05:19:13; 05:19:21 |
| step 1b: first command; last sample | 05:22:25.271; 05:25:13.069 |
| steps 2 and 3 dropped, by the maintainer's answer | 05:30:51 |
| step 4a: read; first command; last sample; read | 05:30:51; 05:31:00.971; 05:31:38.219; 05:31:38 |
| step 4b: first command; last sample | 05:37:20.682; 05:41:28.029 |
| main power cut: both controllers off USB | 05:44:00 |
| the last read of session 1, which found no controller | 05:48:57 |
| deposit commit `8d0dc89a`: session 1 and the amendment after step 4b | 05:51:13 |
| the front's hub port sees a device, then the rear's; neither enumerates (the kernel gives up at 23:22:44 and 23:23:52) | 23:21:36; 23:22:45 |
| session 2's first command, with no motor command; the webcam re-aimed at FR | 23:53:29; 23:55:09–23:56:28 |
| 2026-11-01: the amendment before session 2's first motor command, `64704ebe` | 00:06:07 |
| the USB re-plugged with main power on: the rear's port sees it; the front's, when the kernel reaches it; neither enumerates | 00:06:08; 00:07:16 |
| the USB2 hub `1-2` reset; both controllers enumerate | 00:09:55; 00:09:57 |
| gate 0.2: the read | 00:10:21 |
| gate 0.1: the meter, remeasured; S2a's first command; its last sample | 00:21:44; 00:21:51.603; 00:24:39.430 |
| S2b: read; first command; last sample; read | 00:26:05; 00:26:15.415; 00:27:31.528; 00:27:31 |
| S2c: first command; last sample | 00:31:04.811; 00:35:12.125 |
| step 5: C0; C1 (first command to last sample) | 00:38:44.799–00:39:17.470; 00:41:23.795–00:41:56.450 |
| step 5: C1 backward; C1 in the spin | 00:48:56.373–00:49:03.357; 00:49:11.221–00:49:18.206 |
| the constants commit `b77b06d` | 00:50:47 |
| step 6: `base` up; `RoboClawNode ready.` | 01:23:17.738; 01:23:20.945 |
| step 6: the first run's first message; the last run's last message | 01:23:33.626; 01:25:13.686 |
| `base` stopped | 01:25:43.580 |
| `ac39454`, the constants commit amended (committer date; its author date stays 00:50:47) | 01:49:02 |
| deposit commit `c75fc3f7`: session 2's data | 02:00:12 |
| task-branch commit `4624293`; the tool's `--help` from the worktree | 02:28:26; 02:29:38.814 |
| the flash question's answer recorded (received after 01:25:44); the amendment, `35428e1e` | 02:30:04; 02:30:05 |
| the front's save: start; off USB; back | 03:04:47.226; 03:04:49; 03:04:51 |
| the rear's save: start; off USB; back | 03:11:17.869; 03:11:19; 03:11:22 |
| the final `--read-only` | 03:12:19.885 |
| deposit commit `4cc75b84`: the flash outputs, their USB events, the last answers, `scripts/analysis.py` and `analysis.txt` | 04:04:55 |
| step 7: the D555 off USB; the host boots; the controllers enumerate as `ttyACM0` and `ttyACM1` | 04:09:21; 04:12:47; 04:12:59 |
| step 7: the read; the D555 plugged in again | 04:19:14; 04:25:01 |
| deposit commit `6cce8172`: step 7 | 04:38:18 |

Commit times are the commits' own; every other time is from `bench_log.txt`, the CSVs, the kernel
log (`host/`), the step 6 events (`s6/`), the flash outputs (`flash/`), step 7's state files (`s7/`) or
`operator_notes.txt`.

## Departures from the pre-registration

- **Steps 2 and 3 were not run,** by the maintainer's decision after step 1b, which found no steady
  deficit to locate. The swap's decision (2.1) stays unrun. `hold`'s wiring check, which gate 3.1
  would have repeated, passed before every closed-loop call.
- **Step 4b's duty-127 steps ran backward on the forward ladders,** so step 4b gives full speed
  backward only. The amendment after step 4b records it. `motor_bench.py` was changed to refuse 127;
  the version that ran steps 1a to 4b is kept as `scripts/motor_bench_steps1-4.py`.
- **The FR hub reseat, and the runs planned after it, were replaced.** The amendment after step 4b
  planned FR's hub off and on, then FR alone at duty 32 (5a) and a forward full-speed run (5b). The
  maintainer's work went further: FR's miter gears re-seated, every hub off and every wheel oiled.
  So the amendment before session 2 made 1b, 4a and 4b the "before" for all four wheels, ran S2a
  to S2c in place of 5a and 5b, took C1's QPPS from S2c alone, and withdrew step 5's C0 prediction
  at 178 ticks/s.
- **Whether the wobble remained was asked and not answered.** The amendment put both the hub and
  the wobble to the maintainer before S2a; the answer covered the hubs and the miter gears.
- **The re-plug did not enumerate the controllers; a hub reset did.** The amendment had both cables
  re-plugged, main power on, before gate 0.2. The re-plug failed as the power-up had, and the reset
  of hub `1-2` enumerated both (`host/s2_kernel_usb_events.txt`).
- **Session 2's gate 0.1 was asked twice.** The first answer gave the pack as about 16 V and each
  controller as getting 12 V, so the meter readings were asked for again, and remeasured before
  S2a.
- **The flash-save question came before its amendment,** at the maintainer's request. Step 6 had
  said the tool and its sequence would be committed in an amendment before the question was put.
  The amendment was committed before the tool touched a controller.
- **The flash sequence sends command 95 before the set,** departing from step 6's "command 94 with
  no 95", so the save changes nothing but the gains. Nothing is sent between the read-back and 94,
  and no 95 after it.
- **The constants commit had no question of its own.** The order puts it behind one; the
  maintainer's answer before C0, "We can continue through the gains", was taken to cover it, and
  the question before step 6 reported it.
- **Step 6's constants commit was amended after the runs.** Step 6 ran at `b77b06d`. `ac39454`
  changes only `docs/SIM_TO_REAL_TUNING_GUIDE.md` against it, so `source/` is the same.
