# The chassis stop modes on the real RoboClaws, 2026-10-20

The bring-up record,
[`chassis-bringup-2026-10-19`](../chassis-bringup-2026-10-19/README.md), showed the controllers'
0.5 s serial timeout stopping the wheels after a SIGKILL of `base`. This record covers three other
ways the driver can lose the controllers mid-drive:
- a hang (`docker pause`);
- a pulled USB cable;
- a kill of the driver process inside its container.

Stops are timed two ways:
- **by the encoders**, read after the run;
- **contactlessly, by a webcam** on FR whose frames carry host timestamps. It also re-times the
  SIGKILL case.

Every expectation was pre-registered in the deposit before its run. The chassis stood on blocks.
Every drive was a 0.8 rad/s spin, 5.6–6.2 rad/s at the wheels, published on `/cmd_vel` from a
separate container. The deposit's `scripts/analysis.py` recomputes every figure from the runs.

**Result.**
- **Each of the four ways tested stops the wheels.**
  - Whether `base` is frozen, killed, or loses its driver process, the controllers' 0.5 s timeout
    stops the wheels.
  - By camera, FR's last motion came 0.55 s after the driver's last sample in all five such runs,
    and FR was at rest 0.60–0.62 s after the command.
  - When one controller's USB is pulled, its own timeout stops its wheels, and the driver stops
    the other controller: FR was at rest 0.225 s after the kernel saw the pull. After the replug
    the driver reconnected on the same name and held the wheels at zero.
  - After the process kill, Docker restarted `base`, and the new driver held the wheels at zero.
- **After a stall, the driver can re-run a stale command.**
  - If `cmd_vel` arrives while the driver is frozen, and the publisher then falls silent, the
    driver runs one of those messages as fresh once it resumes. Here that message was at least
    1.0 s old.
  - The wheels turned about 0.4 of a turn more, until the watchdog stopped them again 0.5 s
    later. Filed as
    [`driver-stale-cmd-vel-rearm`](../../tasks/active/reliability/driver-stale-cmd-vel-rearm.md).
- **The supply is 12 V at both controllers, and their voltage readings run low.** Both terminals
  meter 12 V, yet the controllers read 10.2 V and 11.2 V.
- **A main-power cut shows in the counts.** A cut during the first pause reset every count to 0.

## 1. Setup

- **Software.**
  - **The tree:** images built at `371e848`, `strafer-cpu:humble` `4473c937e6a3`. Compose came
    from that tree, with `base` on `stop_signal: SIGINT` and `restart: unless-stopped`.
  - **The pre-registration:** the deposit's `PREREGISTRATION.md` was committed before the first
    run, and its pause_3 addendum before that run.
  - **Isolation:** a census of DDS domain 42 found no publisher on `/cmd_vel` before `base`'s
    first start and before its start after the power cut. Each run's publisher saw one
    subscriber.
  - **The trigger** came 4.3–4.5 s after the spin's first message, in every run.
- **The supply.** The power chain is a 4S LiPo, then a 300 W 12 V buck converter, then a Y
  harness to both RoboClaws. Main power cuts the RoboClaws only; the Jetson has its own supply.
  - After a charge to 16.8 V, the maintainer metered the buck's output at 12 V and each
    controller's `+`/`−` terminals at 12 V.
  - At that supply the controllers read 10.2 V (front) and 11.2 V (rear). So their own voltage
    sensing reads about 1.8 V and 0.8 V low.
  - In the bring-up they read 10.2–10.3 V and 11.1 V, within 0.1 V of these. So the
    bring-up's readings most likely carry the same sensing error.
- **The camera.** A Logitech BRIO on the Jetson filmed FR's hub from about 10 cm, at 1280x720
  MJPEG and 60 fps.
  - Each frame carries its V4L2 buffer timestamp, put on the host's wall clock. The run scripts
    stamp their commands with the same clock.
  - With uvcvideo's `hwtimestamps=0`, that timestamp is when the frame's first USB packet reached
    the host. So every camera time is late by the webcam's latency, which is not measured, and
    each stop came no later than stated.
  - **Motion** is the mean absolute difference of consecutive frames in a box around the hub. At
    rest it measures a mean of 1.08 and a maximum of 2.99.
  - The threshold of 4.0, and "at rest" meaning under it for 0.5 s, were fixed before the drives.
    Every stop below is reported at thresholds 3.5, 4.0 and 5.0, and only pause_3's stop at the
    pause differs between them.
  - The webcam shares the controllers' USB hub. In 10 s of it streaming with `base` up, the
    driver logged nothing.
- **Power losses, shown by the counts.** A controller's encoder count resets only when it loses
  power.
  - At `base`'s first start the counts read 0, 0, 0 and 3. At the bring-up's last read, FR and RR
    stood near +13,600, and FL and RL near 2³². RR's 3 ticks are one past the pre-registered ±2;
    from +13,578 they still show a reset.
  - The controllers had lost power in between. The kernel shows both dropping off USB 0.14 s
    apart at 20:56:27 CDT, as in a main-power cut, and both back 13 s later.
  - **pause_1's cut is the clean case.** No USB event for either controller came between `base`'s
    first start and that cut. Before the pause FR stood at 2,075 and RR at 2,038, and FL and RL
    near 2³². At the next start every count read 0.

## 2. The hang analogue: `docker pause` mid-spin, `docker unpause` 3 s later

`docker pause` freezes every process in `base`, so the ports stay open and no packet goes out.
The pause takes effect within the 46–56 ms the command takes to return. In each run two
`joint_states` samples fall inside that window, and none after it until the unpause.

| | pause_1 | pause_2 | pause_3 |
|---|---|---|---|
| publisher's last message, from the pause | +0.053 s | +0.030 s | **+2.047 s** (1.0 s before the unpause) |
| the driver's last take before the freeze | +0.000 to +0.010 s | +0.031 to +0.041 s | −0.002 to +0.008 s |
| a message waiting at the unpause | yes | no | **yes** |
| FR at rest, camera (last motion / first still frame) | +0.593 / +0.609 s | +0.586 / +0.602 s | +0.579 / +0.599 s |
| encoders across the pause, time at the last speed | — (power cut) | 0.53–0.57 s | 0.51–0.54 s |
| first line after the unpause | +0.038 s, `No cmd_vel for 3.10s -- stopping motors.` | +0.041 s, `No cmd_vel for 3.07s` | +0.039 s, `No cmd_vel for 3.09s` |
| after the unpause | power cut: I/O errors and the error state | 327 samples, all at 0.00 rad/s; camera max 2.80 | **counts moving from +0.089 to +0.667 s; peak 5.27 rad/s** |

- **Which runs had a message waiting.**
  - Each watchdog line gives the age of the driver's last take, so its stamp less that age dates
    the take, to ±0.005 s.
  - A message the publisher sent after that take was still waiting when the driver resumed.
- **pause_1.** FR was at rest 0.609 s after the pause.
  - The maintainer then cut main power. The checkpoint before the run had named the main-power
    cut as the abort during the pause, without saying when to use it.
  - The pre-registered rule called for an abort only if a wheel was not at rest 1 s after its
    expected stop. FR had been at rest since +0.609 s.
  - The controllers dropped off USB at +3.076 s (front) and +3.224 s (rear), around the unpause
    at +3.068 s. The driver's first commands after the unpause failed with I/O errors. A message
    was waiting, so the cut hid whether it would have re-armed.
  - When main power was back on and their USB plugged back in, both controllers re-enumerated
    on their old names. The driver
    reconnected at its first attempt after both had.
- **pause_2: no re-arm, with nothing waiting.** The driver took the last message before the
  freeze. After the unpause all four wheels held at zero, with one watchdog line, and main power
  stayed on.
- **pause_3: a re-arm.** The publisher kept sending for 2 s into the pause, then stopped.
  - At the unpause the driver's first tick logged the watchdog's zero.
  - By the second watchdog line, the driver then took a message at +0.048 to +0.058 s. It was one
    sent into the pause, at least 1.0 s old.
  - It drove the spin again. The counts moved from +0.089 to +0.667 s, and the camera saw FR
    move from +0.109 to +0.629 s. The wheels turned 2.47–2.64 rad.
  - The driver logged `No cmd_vel for 0.51s -- stopping motors.` at +0.565 s.
  - At threshold 5.0 the camera puts the stop at the pause at +0.563/+0.579 s.
- **Why the re-arm happens.**
  - The subscription keeps one message (`depth=1`), so a message sent into the pause is still
    waiting when the driver resumes.
  - `_cmd_vel_callback` stamps it with the time it takes it; a `Twist` carries no stamp.
  - rclpy runs an overdue timer before pending subscriptions. So the first tick after the unpause
    zeroes the motors, the callback then stores the old message as new, and the next ticks drive
    it until the watchdog fires 0.5 s later.

## 3. SIGKILL, timed contactlessly

`docker kill -s SIGKILL strafer_base`, 4.4 s into a spin.
- **The container** exited 137 and was not restarted.
- **The driver's last sample** came +0.039 s after the kill.
- **The camera:** FR's last motion at +0.591 s, 0.552 s after the driver's last sample, and at
  rest from +0.607 s.
- **The encoders, read after a restart:** FL −3.14, FR +3.05, RL −3.20 and RR +3.18 rad, which is
  0.49–0.54 s at the last speed.
- **The two agree within the encoder figure's precision.**
  - The encoder figure times the travel at one sampled speed.
  - Across the runs it puts FR's stop from −0.064 to +0.025 s off the camera's last motion; here
    it is −0.064 s.
  - So the bring-up's encoder estimate stands, now checked by a clock that sends nothing to the
    controllers.

## 4. A pulled USB cable: the rear controller's, mid-spin

The maintainer pulled the rear controller's USB 4.5 s into a spin and plugged it back in about
10 s later. Times are from the kernel's disconnect at 21:27:22.230 CDT.

| t | event |
|---|---|
| −0.221 s | the driver's last `joint_states` sample |
| +0.003 s | `Encoder read failed: device reports readiness to read but returned no data` |
| +0.155 s | `10 consecutive failures -- stopping motors and entering error state`: a stop to both |
| +0.222 s | the publisher's last message |
| +0.225 s | FR, on the front controller, at rest (camera; last motion +0.205 s) |
| +2.2 to +8.2 s | `Attempting RoboClaw reconnect...`, then `Reconnect failed.`, every 2 s |
| +9.956 s | the rear's USB device is back |
| +10.068 s | `ttyACM2`, the rear's old `/dev/roboclaw2` |
| +10.263 s | `RoboClaw reconnected successfully.`, then `No cmd_vel for 10.05s -- stopping motors.` |

- **The rear wheels stopped on their own timeout.** RL and RR turned 3.32 and 3.25 rad after the
  last sample, 0.55–0.56 s at speed, as with a frozen driver.
- **The front wheels were stopped by the driver.** FL and FR turned 2.45 and 2.33 rad after the
  last sample, 0.38–0.41 s at speed. That puts their stop at about +0.15 to +0.19 s, at the
  driver's stop at +0.155 s. The camera's last motion came at +0.205 s.
- **The driver's loop waited on the rear.** No tick completed in the 0.22 s between its last
  sample and the disconnect. It was waiting out the rear's 0.1 s timeouts (`read_timeout=0.1`,
  `max_retries=1`).
- **The D555's serial port** (`/dev/ttyACM0`) had no holder in any of 86 samples.
  - The samples came 0.28–0.43 s apart, from 6 s before the pull to 10 s after the replug.
  - A probe holds a port for milliseconds, so the samples rule out only a lasting hold.
  - That the reconnect opens only the configured paths is from the code: `reconnect()` reopens
    `self.port`.
- **After the reconnect,** all 2,263 samples were at 0.00 rad/s.

## 5. The process kill: `pkill -KILL` of `roboclaw_node` inside the container, mid-spin

The kill came 4.3 s into a spin.
- **Launch ended.** At +0.090 s it logged `roboclaw_node exited -9; ending the launch`.
- **Docker restarted `base`.** Its events show the container dying with exit code 1. It started
  again at +1.536 s, and the restart count went from 0 to 1.
- **The controllers stopped the wheels in the gap.**
  - The camera put FR's last motion at +0.607 s, 0.554 s after the old driver's last sample at
    +0.053 s, and rest at +0.623 s.
  - The encoders show 3.13–3.21 rad, 0.53–0.58 s at speed.
- **The restarted driver commanded no motion.**
  - It logged `RoboClawNode ready.` at +3.782 s, and its first sample came at +3.807 s, at
    0.00 rad/s.
  - All 847 of its samples stayed at zero, with no position change.
  - It logged `No cmd_vel for 0.50s -- stopping motors.` 0.48 s after ready.
- **Its first motor command is zero by construction.** At startup `_latest_twist` is `Twist()`,
  and no `cmd_vel` reached it. No packet was captured.

## 6. Findings

1. **After a stall, the driver runs a stale command as fresh** (§2, pause_3).
   - Its watchdog counts from when it takes a message, not from when the message was sent. So a
     message that waited through a stall drives for another 0.5 s.
   - Here that restarted a spin at least 1.0 s old, after the watchdog's stop.
   - It takes a publisher that sends into the stall and then falls silent without a final zero.
     pause_2, with nothing sent into the stall, did not re-arm.

   Filed as
   [`driver-stale-cmd-vel-rearm`](../../tasks/active/reliability/driver-stale-cmd-vel-rearm.md).
2. **One controller that does not answer blocks the driver's whole loop** in its 0.1 s timeouts.
   Here no tick completed for 0.22 s before the disconnect (§4). That is under the controllers'
   0.5 s timeout, so here it stopped nothing.
3. **The controllers' voltage readings run low,** by about 1.8 V (front) and 0.8 V (rear) at a
   metered 12 V.
4. **A main-power cut drops both RoboClaws off USB,** with the cables in. They dropped 0.15 s
   apart in pause_1. A pair 0.14 s apart at 20:56:27, whose cause is not recorded, matches. So
   the host sees a main-power cut as two USB disconnects.

## What this does not show

- **Driving on the floor.** Every run had the wheels in the air.
- **A real hang,** such as a deadlock or a scheduling stall, rather than the cgroup freezer that
  `docker pause` uses.
- **A controller that stops answering but stays on USB.**
  - From the code, each tick would then wait out about 0.3 s of timeouts. So the 10 failures
    before the driver's stop would take about 3 s.
  - All that time it would keep commanding the other controller.
- **The webcam's latency.** Its frame times are arrivals, so every camera time is late by an
  unmeasured amount.
- **A pull of the front controller.** Only the rear's USB was pulled.
- **A replug onto a different `/dev/roboclaw*` name.** The rear came back on its old one.

## Evidence

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `chassis-stop-modes-2026-10-20/record-files/` |
| deposit commit | `613caf68f04b698dfba6ee2ef9263b9e4f3fa938` |
| pre-registration commits | `42a810ca` (the record's), `a923c14` (pause_3's) |

**What the deposit holds:**
- the step-0 read and the two `base` starts, with their censuses and counts;
- the webcam's framing, its at-rest frames and its fixed parameters;
- per run:
  - the events;
  - the publisher's summary;
  - every `joint_states` message;
  - the container log;
  - every frame's timestamp;
  - the raw frames around each event;
- `base`'s whole log for the runs, and the kernel's USB events;
- the USB pull's per-sample list;
- the process kill's Docker events and restart samples;
- each checkpoint and each of the maintainer's reports, verbatim;
- the scripts, and `analysis.py` with its output.

Hosts, trees and times are in [`provenance.md`](provenance.md). Verify the digests, then
recompute the figures (with OpenCV):

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/chassis-stop-modes-2026-10-20/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
python3 scripts/analysis.py | diff - analysis.txt
```
