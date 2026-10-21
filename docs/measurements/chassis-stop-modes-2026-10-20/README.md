# The chassis stop modes on the real RoboClaws, 2026-10-20

The bring-up record,
[`chassis-bringup-2026-10-19`](../chassis-bringup-2026-10-19/README.md), showed the controllers'
0.5 s serial timeout stopping the wheels after a SIGKILL of `base`. This record covers three other
ways the driver can lose the controllers mid-drive:
- a hang (`docker pause`);
- a pulled USB cable;
- a kill of the driver process inside its container.

Stops are timed two ways:
- **from `joint_states`,** or from a read of the encoders after the run;
- **contactlessly, by a webcam** on FR whose frames carry host timestamps. It also re-times the
  SIGKILL case.

Every expectation was pre-registered in the deposit before its run. The chassis stood on blocks.
Every drive was a 0.8 rad/s spin, 6.24 rad/s commanded at each wheel, published on `/cmd_vel` from
a separate container. At the last sample before each trigger the wheels read 5.55–6.24 rad/s. The deposit's `scripts/analysis.py` recomputes every figure from the runs.

**Result.**
- **Each of the four ways tested stops the wheels.**
  - Whether `base` is frozen, killed, or loses its driver process, the controllers' 0.5 s timeout
    stops the wheels.
  - By camera, FR's last motion came 0.55–0.56 s after the driver's last sample in all five such
    runs, and FR was at rest 0.60–0.62 s after the command.
  - When one controller's USB is pulled, its own timeout stops its wheels, and the driver stops
    the other controller: FR was at rest 0.225 s after the kernel saw the pull. After the replug
    the driver reconnected on the same name and held the wheels at zero.
  - After the process kill, Docker restarted `base`, and the new driver held the wheels at zero.
- **After a whole-executor stall, the driver can re-run a stale command.**
  - If `cmd_vel` arrives while the driver is frozen, and the publisher then falls silent, the
    driver runs the newest of those messages as fresh once it resumes. Here it was 1.05–1.06 s
    old when the driver took it.
  - The wheels turned about 0.4 of a turn more, until the watchdog stopped them again 0.5 s
    later. Filed as
    [`driver-stale-cmd-vel-rearm`](../../tasks/active/reliability/driver-stale-cmd-vel-rearm.md).
- **A reconnect after a main-power cut restores the link, not the gains.** Filed as
  [`driver-reconnect-reapplies-gains`](../../tasks/active/reliability/driver-reconnect-reapplies-gains.md).
- **The controllers read their supply low.** The maintainer metered 12 V at both controllers'
  terminals; the controllers read 10.2 V and 11.2 V.
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
  - **The trigger** came 4.26–4.48 s after the spin's first message, in every run.
  - **One clock.** Commands, frames and samples are stamped on the host's wall clock.
    - A `joint_states` sample is timed by its arrival at the probe unless marked "by stamp". Its
      own stamp, the driver's clock read at the tick's start, is a few milliseconds earlier.
    - Log lines are timed by their ROS stamps, the node's own reads of that clock. Launch's own
      lines carry none and are timed by Docker's stamp.
- **The supply.**
  - **The chain,** as the maintainer described it: a 4S LiPo, then a 300 W 12 V buck converter,
    then a Y harness to both RoboClaws. Main power cuts the RoboClaws only; the Jetson has its own
    supply.
  - **The meter readings are the maintainer's report, to the volt.** They were given at 02:09:16
    UTC, two minutes before the controllers' own read (command 24) at idle. No instrument log or
    photo is deposited.
    - After a charge, the pack read 16.8 V.
    - The buck's output read 12 V, as did each controller's `+`/`−` terminals.
  - **At that supply the controllers read 10.2 V (front) and 11.2 V (rear).**
    - The evidence favours their own sensing running low, by about 1.8 V and 0.8 V.
    - The terminal reading rules out a drop in the harness at idle. Which instrument is right in
      absolute terms stays open.
  - In the bring-up they read 10.2–10.3 V and 11.1 V, within 0.1 V of these. So the bring-up's
    readings most likely carry the same sensing error.
- **The camera.** A Logitech BRIO on the Jetson filmed FR's hub from about 10 cm, at 1280x720
  MJPEG and 60 fps.
  - Each frame carries its V4L2 buffer timestamp, put on the host's wall clock.
  - With uvcvideo's `hwtimestamps=0`, that timestamp is when the frame's first USB packet reached
    the host. So every camera time is late by the webcam's latency, which is not measured, and
    each stop came no later than stated.
  - **Motion** is the mean absolute difference of consecutive frames in a box around the hub. At
    rest it measures a mean of 1.08 and a maximum of 2.99.
  - **The threshold** of 4.0, and "at rest" meaning under it for 0.5 s, are stated in
    `analysis_params.txt`, whose header puts their fixing before the drives (02:13:55 UTC). The
    file itself was deposited after the runs.
  - **Every stop is also computed over thresholds 3.0 to 6.0.** Only pause_2's and pause_3's stops
    at the pause move, by 0.016 s and 0.036 s.
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
| the driver's last take before the freeze | +0.000 to +0.010 s (by the line's stamp) | +0.030 to +0.040 s | −0.003 to +0.007 s |
| a message waiting at the unpause | yes | no | **yes** |
| FR at rest, camera (last motion / first still frame) | +0.593 / +0.609 s | +0.586 / +0.602 s | +0.579 / +0.599 s |
| encoders across the pause, time at the last speed | — (power cut) | 0.53–0.57 s | 0.51–0.54 s |
| first line after the unpause | +0.037 s, `No cmd_vel for 3.10s -- stopping motors.` | +0.040 s, `No cmd_vel for 3.07s` | +0.039 s, `No cmd_vel for 3.09s` |
| after the unpause | power cut: I/O errors and the error state | 327 samples, all at 0.00 rad/s; camera max 2.80 | **counts moving from +0.089 to +0.667 s by arrival at the probe (+0.081 to +0.661 s by stamp); peak 5.27 rad/s** |

- **Which runs had a message waiting.**
  - Each watchdog line gives the age of the driver's last take (to ±0.005 s), computed from the
    tick's clock read. That read is also the stamp of the sample the tick published, so the read
    less the age dates the take. Where the tick published no sample, the line's own stamp stands
    in, about 1–2 ms late: pause_1's two takes, dated this way, may be up to 2 ms earlier.
  - A message the publisher sent after that take was still waiting when the driver resumed.
- **pause_1.** FR was at rest 0.609 s after the pause.
  - The maintainer then cut main power. The question put to the maintainer before the run had
    named the main-power cut as the abort during the pause, without saying when to use it.
  - The pre-registered rule called for an abort only if a wheel was not at rest 1 s after its
    expected stop. FR, the one wheel filmed, had been at rest since +0.609 s; pause_1 has no
    reading of FL, RL or RR after +0.037 s.
  - The controllers dropped off USB at +3.076 s (front) and +3.224 s (rear), around the unpause
    at +3.068 s. The driver's first commands after the unpause failed with I/O errors.
  - The waiting message was taken: the line after the reconnect, `No cmd_vel for 86.66s`, dates
    the take to +0.040–0.050 s after the unpause. Its drive commands failed with the rest, so the
    cut hid whether it would have re-armed.
  - **The reconnect restored the link, not the configuration.**
    - When main power was back on and the USB back in, both controllers re-enumerated on their
      old names, and the driver reconnected at its first attempt after both had.
    - A power-up loads the controllers' saved settings, which step 0 read before the driver's
      first start: the front's velocity gains P 14999, I 749 on M1 (15000/750 on M2), and the
      rear's P, I, D and QPPS all 0.
    - The driver writes its gains only at start (P 15000, I 750, D 0, QPPS 2796), so the rear's
      were gone. The tick after the reconnect sent the watchdog's zero
      (`base_log_full_after.txt`). The next `cmd_vel` would have driven the rear on zero gains,
      had `base` not been stopped 4 s later.
    - The names repeated because the enumeration order repeated; the bring-up saw the two swap.
      `reconnect()` reopens the path detection chose at start (`self.port`) and probes it at that
      controller's address; it does not detect again.
- **pause_2: no re-arm, with nothing waiting.** The driver took the last message, sent at
  +0.030 s, before the freeze. After the unpause all four wheels held at zero, with one watchdog
  line, and main power stayed on.
- **pause_3: a re-arm.** The publisher kept sending for 2 s into the pause, then stopped.
  - At the unpause the driver's first tick logged the watchdog's zero.
  - By the second watchdog line, the driver then took a message at +0.046 to +0.056 s. The
    publisher's last message was 1.05–1.06 s old at that take.
  - It drove the spin again. The counts moved from +0.089 to +0.667 s by arrival at the probe,
    +0.081 to +0.661 s by the samples' stamps. The camera saw FR move from +0.109 to +0.629 s. The
    wheels turned 2.47–2.64 rad.
  - The driver logged `No cmd_vel for 0.51s -- stopping motors.` at +0.563 s.
  - The camera's stop at the pause depends on the threshold. From 3.0 to 6.0 the last motion
    moves from +0.599 to +0.563 s and the first still frame from +0.615 to +0.579 s.
- **Why the re-arm happens.**
  - `_cmd_vel_callback` stamps a message with the time it takes it; a `Twist` carries no stamp.
  - rclpy runs an overdue timer before pending subscriptions, and rcl schedules ticks on the
    period grid, so a catch-up tick can follow the overdue one within milliseconds. In pause_3
    two ticks, with clock reads at +0.037 and +0.045 s, sent the watchdog's zero before the take
    (pause_2 shows the same pair, +0.039 and +0.047 s). The callback then stored the old message
    as new, and the next ticks drove it until the watchdog fired 0.5 s later.
  - The subscription keeps one message (`depth=1`), which is why only the newest is taken. Where
    the messages waited during the freeze, in the socket buffer or in the publisher's
    retransmission, is a reading, not measured.
  - **The precondition is a stall of the whole executor, longer than the watchdog window.** The
    USB pull (§4) is the contrast. Its loop waited on the rear's timeouts for about 0.2 s, under the
    window. In the 10 s of error state that followed, the executor kept serving the subscription
    between ticks: the reconnect's watchdog line dates the last take to +0.217–0.227 s, at the
    publisher's last message. So nothing waited through the outage, and nothing re-armed after the
    reconnect.

## 3. SIGKILL, timed contactlessly

`docker kill -s SIGKILL strafer_base`, 4.37 s into a spin.
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

The kernel logged the rear controller's USB disconnect 4.48 s into a spin; the plug was out about
0.2 s before that (below). The maintainer plugged it back in about 10 s later. Times are from the
kernel's disconnect at 21:27:22.230 CDT.

| t | event |
|---|---|
| −0.228 s | the stamp of the driver's last `joint_states` sample (it reached the probe at −0.221 s) |
| −0.208 s | the next tick starts and completes no sample |
| +0.001 s | `Encoder read failed: device reports readiness to read but returned no data` |
| +0.155 s | `10 consecutive failures -- stopping motors and entering error state`: a stop command to both, the rear's failing |
| +0.222 s | the publisher's last message |
| +0.225 s | FR, on the front controller, at rest (camera; last motion +0.205 s) |
| +2.2 to +8.2 s | `Attempting RoboClaw reconnect...`, then `Reconnect failed.`, every 2 s |
| +9.956 s | the rear's USB device is back |
| +10.068 s | `ttyACM2`, the rear's old `/dev/roboclaw2` |
| +10.263 s | `RoboClaw reconnected successfully.`, then at +10.273 s `No cmd_vel for 10.05s -- stopping motors.` |

- **The plug was out about 0.2 s before the kernel's line.** A tick sends both drive commands
  before its reads. The tick due at −0.208 s completed no sample, so the rear's ACK to its drive
  command never came. That is a reading of the code against the timing.
- **The rear wheels stopped on their own timeout.** RL and RR turned 3.32 and 3.25 rad after the
  last sample, 0.55–0.56 s at speed, as with a frozen driver.
- **The front wheels were stopped by the driver.** FL and FR turned 2.45 and 2.33 rad after the
  last sample, 0.38–0.41 s at speed. That puts their stop at about +0.15 to +0.19 s, at the
  driver's stop at +0.155 s. The camera's last motion came at +0.205 s.
- **The driver's loop waited on the rear.** No tick completed between the last sample and the
  disconnect. It was waiting out the rear's 0.1 s timeouts (`read_timeout=0.1`,
  `max_retries=1`).
- **The D555's serial port** (`/dev/ttyACM0`) had no holder in any of 86 samples.
  - The samples came 0.28–0.43 s apart, from 6 s before the pull to 10 s after the replug.
  - A probe holds a port for milliseconds, so the samples rule out only a lasting hold.
  - That the reconnect opens only the paths detection chose at start is from the code:
    `reconnect()` reopens `self.port`.
- **After the reconnect,** all 2,263 samples were at 0.00 rad/s. The USB pull left the rear's
  main power on, so its gains were not lost.

## 5. The process kill: `pkill -KILL` of `roboclaw_node` inside the container, mid-spin

The kill came 4.26 s into a spin.
- **Launch ended.** At +0.090 s, by Docker's stamp, it logged
  `roboclaw_node exited -9; ending the launch`.
- **Docker restarted `base`.** Its events show the container dying with exit code 1. It started
  again at +1.536 s, and the restart count went from 0 to 1.
- **The controllers stopped the wheels in the gap.**
  - The camera put FR's last motion at +0.607 s, 0.554 s after the old driver's last sample at
    +0.053 s, and rest at +0.623 s.
  - The encoders show 3.13–3.21 rad, 0.53–0.58 s at speed.
- **The restarted driver commanded no motion.**
  - It logged `RoboClawNode ready.` at +3.781 s, and its first sample came at +3.807 s, at
    0.00 rad/s.
  - All 847 of its samples stayed at zero, with no position change.
  - It logged `No cmd_vel for 0.50s -- stopping motors.` 0.48 s after ready.
- **Its first motor command is zero by construction.** At startup `_latest_twist` is `Twist()`,
  and no `cmd_vel` reached it. No packet was captured.

## 6. Findings

Finding 1 tests the pre-registered competing prediction. Findings 2–5, and the dating of takes
from the watchdog lines, are post-hoc.

1. **After a whole-executor stall, the driver runs a stale command as fresh** (§2, pause_3).
   - Its watchdog counts from when it takes a message, not from when the message was sent. So a
     message that waited through a stall drives for another 0.5 s.
   - Here that restarted a spin 1.05–1.06 s old at the take, after the watchdog's stop.
   - It takes a publisher that sends into the stall and then falls silent without a final zero.
     pause_2, with nothing sent into the stall, did not re-arm. Nor did the USB pull, whose
     executor kept serving the subscription through its 10 s outage.

   Filed as
   [`driver-stale-cmd-vel-rearm`](../../tasks/active/reliability/driver-stale-cmd-vel-rearm.md).
2. **One controller that does not answer blocks the driver's whole loop** in its 0.1 s timeouts.
   Here no tick completed for about 0.2 s before the disconnect (§4). That is under the
   controllers' 0.5 s timeout, so here it stopped nothing. Filed, with the longer case from the
   code, as
   [`driver-unanswered-controller-stall`](../../tasks/active/reliability/driver-unanswered-controller-stall.md).
3. **The controllers' voltage readings run low,** by about 1.8 V (front) and 0.8 V (rear) against
   the maintainer's metered 12 V at their terminals. That reading rules out a harness drop at
   idle. Which instrument is right in absolute terms stays open.
4. **A main-power cut drops both RoboClaws off USB,** with the cables in. They dropped 0.15 s
   apart in pause_1. A pair 0.14 s apart at 20:56:27, whose cause is not recorded, matches. So
   the host sees a main-power cut as two USB disconnects.
5. **A reconnect restores the link, not the configuration** (§2, pause_1).
   - After a main-power cut the controllers come back with their saved gains: zeros on the rear.
   - The driver writes its own gains only at start, so its next `cmd_vel` would drive the rear on
     zero gains until `base` restarts.
   - `reconnect()` probes a fixed path, so a swap in the enumeration order would leave it failing
     until a restart.

   Filed as
   [`driver-reconnect-reapplies-gains`](../../tasks/active/reliability/driver-reconnect-reapplies-gains.md).

## What this does not show

- **Driving on the floor.** Every run had the wheels in the air.
- **A real hang,** such as a deadlock or a scheduling stall, rather than the cgroup freezer that
  `docker pause` uses.
- **A controller that stops answering but stays on USB.**
  - From the code, each tick would then wait out about 0.3 s of timeouts, so the 10 failures
    before the driver's stop would take about 3 s.
  - All that time it would keep commanding the other controller with the last command.
  - On a policy mission the inference node zeroes `/cmd_vel` about 0.2 s after `joint_states`
    stops. The zero lands mid-tick, and the overdue next tick runs on the old command before the
    take, so the other controller would get its zero about 0.6–0.75 s after the last good tick.
    The 3 s stands for Nav2 and bench publishers.
- **The webcam's latency.** Its frame times are arrivals, so every camera time is late by an
  unmeasured amount.
- **A pull of the front controller.** Only the rear's USB was pulled.
- **A replug onto a different `/dev/roboclaw*` name.** Both replugs came back on their old names.

## Evidence

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `chassis-stop-modes-2026-10-20/record-files/` |
| deposit commit | `f2ed7f80c51fee3bd6265bbae09b1d81782f3be4` |
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
- each question put to the maintainer before a run, and each of the maintainer's reports,
  verbatim;
- the scripts, and `analysis.py` with its output.

Hosts, trees and times are in [`provenance.md`](provenance.md). Verify the digests, then
recompute the figures (with OpenCV and PIL):

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/chassis-stop-modes-2026-10-20/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
python3 scripts/analysis.py | diff - analysis.txt
```
