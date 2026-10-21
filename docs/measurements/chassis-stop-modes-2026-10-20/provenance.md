# Provenance: the chassis stop modes on the real RoboClaws, 2026-10-20

captured_date: 2026-10-20 CDT (2026-10-21 UTC)

## Host and hardware

| | |
|---|---|
| host | strafer-nx: Jetson Orin NX 16 GB, L4T R36.4.3, kernel 5.15.148-tegra, Docker 29.1.3 |
| controllers | two RoboClaw ST 2x45A, USB `03eb:2404`, on the hub at USB port `1-2`: front 0x80 on `1-2.3` (`ttyACM1`, `/dev/roboclaw1`), rear 0x81 on `1-2.4` (`ttyACM2`, `/dev/roboclaw2`), each with a 0.5 s serial timeout saved |
| depth camera | D555 on USB port `2-2`, CDC interface at `/dev/ttyACM0`, attached throughout: the kernel logged no USB event for it from 20:50 to 21:40 CDT |
| webcam | Logitech BRIO (`046d:085e`) on hub port `1-2.1`, `/dev/video6`, about 10 cm from FR's outer face; uvcvideo with `hwtimestamps=0` |
| power | a 4S LiPo charged to 16.8 V, a 300 W 12 V buck converter, a Y harness to both controllers; main power cuts the RoboClaws only, and the Jetson has its own supply |
| chassis | on blocks, all four wheels off the ground, for every drive |

The maintainer's reports and each checkpoint's wording are verbatim in the deposit's
`operator_notes.txt`.

## Tree and images

| | |
|---|---|
| tree | `main` at `371e848`, in the clean worktree the bring-up built from |
| images | `strafer-cpu:humble` `4473c937e6a3`, labelled `371e8487533a` (`host/image.txt`); no build for this record |
| compose | `source/strafer_ros/deploy` of that tree, with no override file; `base` with `stop_signal: SIGINT` and `restart: unless-stopped` (`gate/base_up.txt`) |
| publisher | `drive_probe.py` in a separate `strafer-cpu:humble` container on domain 42, with the deploy's Cyclone config: `Twist(angular.z=0.8)` on `/cmd_vel` at 20 Hz until a stop file appears, recording `/strafer/joint_states` |
| webcam capture | `wheelcam.py` on the host (OpenCV 4.8), raw MJPEG frames written as they arrive, each with its V4L2 buffer timestamp |

## Timeline (UTC, 2026-10-21)

| event | time |
|---|---|
| front and rear enumerate | 01:54:33; 01:54:34 |
| both drop off USB, 0.14 s apart; both back | 01:56:27; 01:56:40 |
| the webcam enumerates on `1-2.1` | 02:08:54 |
| pre-registration commit `42a810c` | 02:11:16 |
| step 0: `--read-only` | 02:11:25 |
| the webcam's analysis parameters fixed | 02:13:55 |
| census; `base` up | before 02:14:31; 02:14:31 |
| the webcam streams 10 s with `base` up | 02:14:52–02:15:02 |
| pause_1: pause; unpause | 02:18:20.006; 02:18:23.074 |
| main power cut: front and rear USB disconnect | 02:18:23.082; 02:18:23.230 |
| main power back on, then the USB back in (the maintainer's order): both re-enumerate on their old ports; the driver reconnects | 02:19:47; 02:19:49 |
| `base` stopped (SIGINT) | 02:19:53 |
| census; `base` up | before 02:21:59; 02:21:59 |
| pause_2: pause; unpause | 02:22:20.433; 02:22:23.500 |
| sigkill_1: kill | 02:24:46.268 |
| `base` up; encoders read | 02:25:29; 02:25:34 |
| usbpull_1: the rear's USB disconnect; its USB device back; `ttyACM2` | 02:27:22.230; 02:27:32.186; 02:27:32.298 |
| prockill_1: kill; `base` restarted by Docker | 02:31:27.435; 02:31:28.972 |
| pause_3 addendum commit `a923c14` | 02:34:37 |
| pause_3: pause; publisher stop; unpause | 02:34:52.340; 02:34:54.394; 02:34:55.395 |
| `base` stopped | 02:35:50 |
| deposit commit `de2addf8` | 02:39:03 |
| deposit correction `365e754f`: `analysis.py` prints the rest of the cited figures; the manifest's wording | 02:55:44 |
| `base`'s whole log read from Docker's retained log | 03:28:07 |
| deposit correction `613caf68`: more checks in `analysis.py`; `base`'s whole log, the operator notes, and `pause_run.sh` as pause_1 and pause_2 ran it | 03:31:27 |

## Departures from the pre-registration

- **The wall clock is a webcam, not a phone.** The amendment, written before any run, put a BRIO
  on FR.
  - The run scripts issue the pause or kill at a fixed time, so no key press is filmed.
  - Stops are counted from the command's issue time, not from a key press.
  - The frames carry arrival times, not capture times; the record says so.
- **The webcam also filmed the USB pull and the process kill.** The pre-registration timed those
  by encoders and logs alone.
- **The power-cut check.** The pre-registration's §0 planned a hand-turned FL, then a cut. The
  amendment, written before any run, took the pack swap as the cut instead, with the bring-up's
  last counts as the "before".
  - The amendment's "before" counts, FR about +11,200 and RR about +11,190, come from the
    bring-up's 6b read. The bring-up's last read was 6d: FR +13,622 and RR +13,578. Either way
    the counts show a reset.
  - Which kernel event was the pack swap is not recorded. Both controllers enumerated at
    01:54:33–34, dropped off USB 0.14 s apart at 01:56:27, and came back at 01:56:40.
  - RR read 3 ticks, one past the ±2 expected.
  - pause_1's unplanned cut gave the clean check: no USB event between `base`'s first start and
    the cut, and every count at 0 after it.
- **pause_1's main-power cut.** The checkpoint before the run said "Your abort during the pause
  is the main-power cut.", without a condition.
  - The maintainer cut main power near the end of the pause.
  - Under the pre-registered rule no abort was due: FR was at rest at +0.609 s.
  - That left 1.4, the re-arm, undecided. pause_2 repeated the run, its checkpoint saying "Main
    power stays on unless a wheel keeps turning ~1 s after the pause."
- **pause_3 was added.** pause_2 showed no re-arm with nothing waiting. pause_3 kept the publisher
  sending into the pause, to test the competing prediction. It was pre-registered in `a923c14`
  before it ran, and it ran last.
- **The order.** The pre-registration listed the pack, the pause, SIGKILL, the USB pull, then the
  process kill. The runs went pause_1, pause_2, SIGKILL, the USB pull, the process kill, then
  pause_3.
- **`pause_run.sh` changed between the pauses.** For pause_3, one text replacement at 02:34:38
  added `STOP_DELAY`.
  - pause_1 and pause_2 ran the version before it, which the deposit rebuilds as
    `scripts/pause_run_before_pause3.sh`.
  - With `STOP_DELAY` unset, both touch the stop file as soon as `docker pause` returns.
