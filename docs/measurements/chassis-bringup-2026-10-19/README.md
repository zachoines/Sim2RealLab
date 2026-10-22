# The chassis bring-up on the real RoboClaws, 2026-10-19

On 2026-10-19 the chassis stack ran on the robot with both RoboClaws connected for the first time:
images built from `main` at `371e848`, with
[`compose-stop-signal-sigint`](../../tasks/completed/compose-stop-signal-sigint.md) and
[`roboclaw-serial-timeout`](../../tasks/completed/roboclaw-serial-timeout.md) merged, and the D555
attached.

This record runs the
[port-fence record's §7](../chassis-driver-port-fence-2026-10-14/README.md#7-the-on-site-check)
on-site check, its step 0 (the controllers' serial timeout) included. It then drives the wheels:
- the driver's response to `cmd_vel`;
- its stop when `base` is stopped;
- the controllers' own stop when the driver is killed.

From the first `base` start on, the chassis stood on blocks with all four wheels off the ground.
Every figure here is recomputed from the deposit by its `scripts/analysis.py`.

**Result.**
- **The serial timeout is set and saved on both controllers.** Both first read 0.0 s, which
  disables it. Each acknowledged the reload (command 95), took 0.5 s, read it back and
  acknowledged the save (command 94). After a power cycle with the USB unplugged first, both read
  0.5 s. The kernel log shows the USB gaps; the main-power cut between them is the maintainer's
  action, which the host cannot observe.
- **The port fence holds on the robot.**
  - With both controllers it starts on two different `/dev/roboclaw*` names and leaves the
    camera's port alone.
  - With one unplugged, `base` refused 10 times with exit 1 and restarted each time. Over the
    first eight refusals, a loop ran `fuser` on the camera's port 149 times and reported no
    process. It kept only that summary, not a per-sample list.
- **The driver drives the wheels the commanded way at 0.3 m/s.** All four turned forward, within
  −4 % to +15 % of the commanded speed over the hold, from 0.6 s after the first message to the
  last. At 0.1 m/s the front-right wheel did not move, a tuning
  gap filed as
  [`chassis-velocity-pid-tuning`](../../tasks/active/reliability/chassis-velocity-pid-tuning.md).
- **A stop of `base` stops the motors.**
  - `docker compose stop base` during a spin: SIGINT, exit 0, the command back in 0.87 s, and the
    driver's teardown at +0.18 s.
  - The wheels turned for about 0.1 s more.
- **A SIGKILL leaves the stop to the controllers.** Killed mid-spin, `base` exited 137 with no
  teardown. The encoders show 3.25–3.37 rad more after the last sample, 0.55–0.56 s at their last
  speed. The stop fell about 0.5 s after the driver's last packet, at the controllers' saved
  timeout, and the rest is spin-down.
- **Two more findings.**
  - The driver reports wrapped joint positions when a wheel turns backward:
    [`roboclaw-encoder-count-wrap`](../../tasks/active/reliability/roboclaw-encoder-count-wrap.md).
  - The driver's command topic is `/cmd_vel`, not `/strafer/cmd_vel` (§6).

## 1. Setup

- **The images.** `make images` in a clean detached worktree at `371e848` built
  `strafer-cpu:humble` `4473c937e6a3` and `strafer-gpu:humble` `c11bedfb3979`, both labelled
  `371e8487533a`. Every run printed that revision. Compose ran from the same worktree, with no
  override file. `base` carries `stop_signal: SIGINT`.
- **The udev rule.** `99-strafer.rules` from that tree was installed as `make udev` does.
  - The trigger was limited to the two RoboClaw ttys. A full trigger would very likely have made
    ModemManager probe the camera's CDC port again, which it handles as a modem candidate.
  - `/dev/roboclaw1` and `/dev/roboclaw2` appeared, with `ID_MM_DEVICE_IGNORE=1`.
  - ModemManager logged nothing.
- **The controllers.**
  - Two RoboClaw ST 2x45A on one USB hub: the front at address 0x80, the rear at 0x81.
  - One LiPo feeds both through a main power switch that cuts the RoboClaws only.
    *2026-10-21: as built, the LiPo is a 4S feeding both through a 300 W 12 V buck converter and
    a Y harness ([`chassis-stop-modes-2026-10-20`](../chassis-stop-modes-2026-10-20/README.md)
    §1).*
  - Both enumerated at 15:57:46 CDT with main power already on, and each USB was then re-plugged
    once.
- **Nothing else could command the wheels.** Before `base` first started, a census of DDS domain
  42 found no node but its own. While `base` ran, `/cmd_vel` had no publisher and one
  subscriber, the driver. The sim host ran no container and no ROS process.

## 2. Step 0: the serial timeout

`diagnose_roboclaw.py` from the image, by the commands in §7 step 0:

| run (UTC) | front 0x80 | rear 0x81 | exit |
|---|---|---|---|
| `--read-only` 21:39:46 | 0.0 s, 10.3 V | 0.0 s, 11.1 V | 0 |
| `--front-only --set-serial-timeout 0.5 --write-nvm` 22:49:43 | 95 acked; 0.0 → set 0.5, read back 0.5; 94 acked | — | 0 |
| `--rear-only --set-serial-timeout 0.5 --write-nvm` 22:50:14 | — | 95 acked; 0.0 → set 0.5, read back 0.5; 94 acked | 0 |
| `--read-only` 22:53:20, after the power cycle | **0.5 s**, 10.2 V | **0.5 s**, 11.1 V | 0 |

- **Each save reset its controller and re-enumerated its USB.** The port was back 3 s later
  (front `1-2.3` at 17:49:45–17:49:48 CDT, rear `1-2.1` at 17:50:16–17:50:19). These are three of
  the facts the serial-timeout brief left unverified on hardware: both controllers acknowledge 95,
  each acknowledged 94 within the tool's wait, and a save re-enumerates USB.
- **The proof's power cycle.** The kernel log shows both USB cables out first, the rear at
  17:52:19 and the front at 17:52:22, and back in with new device numbers, the front at 17:52:57
  and the rear at 17:53:13. Those are gaps of 35 s and 54 s.
  - **The main-power cut is the maintainer's report.** Main power was cut and restored inside
    those gaps. The host cannot observe that step.
  - **An earlier "done" was not counted.** The maintainer reported the cycle done once before.
    The kernel log showed no USB event for it, so it was not counted and the cycle was repeated.
  - **What the proof read shows.** With main power cut, the read gives the value loaded at
    power-up.
- **The velocity PID read** shows the front's flash holding P≈15000, I≈750, QPPS 2796, and the
  rear's holding zeros. The driver writes its own constants to both at every start.

## 3. The fence on the robot

**The positive half**, `docker compose up -d base` with the camera attached:
- the log shows `revision=371e8487533a`;
- then `Auto-detecting RoboClaw ports on /dev/roboclaw*...` and
  `Auto-detect: front (0x80) -> /dev/roboclaw1, rear (0x81) -> /dev/roboclaw2`;
- then `RoboClawNode starting`, both PID writes, `RoboClawNode ready`, and the watchdog holding
  zero;
- `fuser` finds only `roboclaw_node` on ttyACM1 and ttyACM2, and nothing on the camera's
  ttyACM0;
- `/strafer/odom` published at 49.99 Hz.

**The negative half.** The rear's USB was unplugged at 17:58:08 CDT, with power on.
- **The running node did not re-detect.** It logged a failed reconnect every 2 s.
- **The recreated `base` refused 10 times,** 22:58:47 to 23:00:02 UTC. Each attempt logged
  `Auto-detect: found 1/2 controllers on /dev/roboclaw* (0x80 on /dev/roboclaw1). Missing
  addresses: 0x81. Refusing to start`, then `roboclaw_node exited 1; ending the launch`.
- **The restarts slowed as Docker doubled its delay.** For 40 s from the recreate, to about
  22:59:24 UTC, a loop ran `fuser` on ttyACM0. That window spans the first eight of the ten
  refusals. The loop reported 149 samples and 0 hits, and kept only that summary; no per-sample
  list is deposited.
- **The rear came back on a different hub port.** It was plugged back in at 18:00:16 CDT on port
  `1-2.4`, not `1-2.1`. The next start, at 23:00:56 UTC, found both by address and came up.

## 4. The wheels' polarity

**Why this check comes first.** A motor whose encoder counts backward against its own drive gives
the controller's velocity loop positive feedback, and the wheel runs away. So before the first
closed-loop drive, `diagnose_roboclaw.py --drive` ran the open-loop duty test with `base`
stopped. That is §7's optional last step, run first at the maintainer's choice.

| wheel | encoder before → after | change | speed read at 1 s |
|---|---|---|---|
| FL (front M1) | 0 → 253 | +253 | 6 ticks/s |
| FR (front M2) | 0 → 229 | +229 | 4 ticks/s |
| RL (rear M1) | 0 → 253 | +253 | 7 ticks/s |
| RR (rear M2) | 33 → 228 | +195 | 6 ticks/s |

- **Every encoder moved with its duty.** The test holds each motor at duty 32/127 for 1 s and
  sends nothing during the hold.
- **The speed at 1 s is near zero, so the controllers stopped each motor at their 0.5 s timeout.**
  The roughly 200–250 ticks each turned fit about half a second of running.

## 5. The driver's `cmd_vel`

Each run published `Twist` on `/cmd_vel` at 20 Hz for 2 s (40 messages) from a separate container,
recording `/strafer/joint_states`. The commanded wheel speed is +2.08 rad/s at 0.1 m/s and
+6.25 rad/s at 0.3 m/s, all four positive.

| run (UTC) | FL | FR | RL | RR | readings zero after the last message |
|---|---|---|---|---|---|
| 0.1 m/s, 23:12:22 | +1.11 | **0.00** | +1.17 | **0.00** | +0.91 s |
| 0.1 m/s again, 23:15:55 | +1.71 | **0.00** | +1.54 | +0.22 | +1.04 s |
| 0.3 m/s, 23:16:10 | +6.02 | +7.18 | +6.14 | +6.81 | +1.46 s |

Mean wheel speed in rad/s over each hold, from 0.6 s after the first message to the last.

- **At 0.3 m/s all four wheels rolled forward.** The maintainer watched them.
- **At 0.1 m/s the front-right wheel never moved, and the rear-right started late in the second
  run.** The deposit shows three things about it:
  - the commanded 178 ticks/s is 6 % of QPPS, a speed the driver's gains (P 15000, I 750) were
    not tuned for;
  - in the duty test the right-side motors (M2) turned less than the left under equal duty: FR
    229 and RR 195 ticks against 253 for FL and RL;
  - the two controllers read 10.3 V and 11.1 V from one pack at no load.

  At 0.3 m/s the right side overshoots by 9–15 %.
- **The watchdog fires 0.50–0.51 s after the last message,** as designed.
- **The wheels stop sooner than their speed readings say.**
  - At 0.3 m/s, encoder positions show FL, RL and RR turning only 0.37–0.65 rad after the
    watchdog, nearly all of it within 0.3 s. FR turned 1.10 rad over 0.7 s.
  - The RoboClaw's speed reading lags the count, which is why the readings take 0.9–1.5 s to
    reach zero.

**The spin used for the stops.** The spin at 0.8 rad/s turns FL and RL backward and FR and RR
forward, at about 6.25 rad/s each. The maintainer saw exactly that.

## 6. The stops

**SIGINT: `docker compose stop base` about 6 s into a spin.**
- **The command returned in 0.87 s, and the container exited 0.**
- **The log, timed from the command:**
  - at +0.17 s, launch logged "user interrupted with ctrl-c (SIGINT)" and passed SIGINT to both
    nodes;
  - at +0.18 s, `roboclaw_node` logged "Shutting down -- stopping motors.";
  - at +0.61 s, both had finished cleanly.
- **The wheels turned about 0.1 s more.** The last `joint_states` sample, at +0.155 s, read
  ±6.2 rad/s. The encoders read at the next start show each wheel turned 0.39–0.54 rad after it.
  That is 0.06–0.09 s at full speed.

**SIGKILL: `docker kill -s SIGKILL strafer_base` about 6 s into a spin.**
- **No teardown ran.** The container exited 137 and logged no teardown.
- **Docker did not restart it.** A kill counts as a manual stop, so `restart: unless-stopped`
  left it exited at +10 s.
- **Nothing could re-command the wheels.** The publishing container stopped 0.14 s after the kill.
- **The controllers stopped the wheels.**
  - The last sample, at +0.031 s, read ±6.0 rad/s.
  - The encoders read after the restart show each wheel turned 3.25–3.37 rad after it: 0.55 s
    at that speed for FL and RL, 0.56 s for FR and RR.
  - The maintainer counted about half a second. That is the saved 0.5 s timeout running from the
    driver's last packet, plus the spin-down.

**Restart.** `docker compose up -d base` passed §3's positive gate again: both names, no restart,
`StopSignal` SIGINT, the camera's port free, and odometry at 49.99 Hz. `base` was stopped at
23:22 UTC, with exit 0 and the teardown's motor stop.

**The command topic.** `driver.launch.py` remaps the node's `/strafer/cmd_vel` to `/cmd_vel`, and
`ros2 node info` lists the driver's one subscription as `/cmd_vel`. Anything that drives the
chassis, or checks that nothing does, uses `/cmd_vel`.

## 7. Findings

1. **Low-speed breakaway.** At 0.1 m/s the front-right wheel does not turn, and the right side
   overshoots at 0.3 m/s.
   - The driver writes its P, I, D and QPPS to both controllers at every start. The front-right
     wheel is on the front controller.
   - The deposit shows a right-side (M2) open-loop deficit under equal duty (§4), gains untuned
     for 6 % of QPPS, and 10.3 V at one controller against 11.1 V at the other.

   Filed as
   [`chassis-velocity-pid-tuning`](../../tasks/active/reliability/chassis-velocity-pid-tuning.md).
2. **Joint positions wrap.** `read_encoder_m1`/`m2` unpack the count as unsigned 32-bit. A wheel
   that turns backward past its power-up count jumps by 2³² ticks: 50,187,977 rad in
   `joint_states`, seen on FL and RL during a spin. Wheel speeds, and the odometry built from
   them, are signed and unaffected. Filed as
   [`roboclaw-encoder-count-wrap`](../../tasks/active/reliability/roboclaw-encoder-count-wrap.md).
3. **The two controllers read different battery voltages from one pack at no load:** front
   10.2–10.3 V, rear 11.1 V. A meter on the pack and on each controller's terminals settles which
   is right. A charged 3S pack reads 12.6 V.
   *2026-10-21: metered, the supply is 12 V at both controllers' terminals, behind a 12 V buck from
   a 4S pack, so the 3S comparison does not apply. At that metered supply, in
   [`chassis-stop-modes-2026-10-20`](../chassis-stop-modes-2026-10-20/README.md) §1, the
   controllers read 10.2 V (front) and 11.2 V (rear), about 1.8 V and 0.8 V low. The readings here
   are within 0.1 V of those.*
4. **ModemManager and the camera.** On this host ModemManager takes up the D555's CDC port as a
   modem candidate at each enumeration, and the RoboClaw rule does not cover it. The narrowed
   trigger in §1 avoided adding another such probe. See `roboclaw-serial-timeout`'s out-of-scope
   note.

## What this does not show

- **Driving on the floor, with load.** Every run had the wheels in the air.
- **Whether tuning brings the front-right wheel round at 0.1 m/s** (§5).
- **That `99-strafer.rules` survives a reboot.** The host did not reboot.
- **The stop through a daemon stop or a host reboot,** and `slam`'s save.
- **A controller left powered with its USB unplugged for longer than the 54 s here.**

## Evidence

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `chassis-bringup-2026-10-19/record-files/` |
| deposit commit | `f4135db124c447702a3363af7496ffff5d3374ac` |

**What the deposit holds:**
- the image build;
- `diagnose_roboclaw.py`'s five runs;
- the kernel's USB events, the udev and ModemManager reads, and the compose config;
- the DDS census and `/cmd_vel`'s endpoints;
- both fence halves' logs, and the camera-port sampling summary;
- the duty test;
- every drive run's `joint_states` CSV and probe output, including the first attempt at the
  compose-stop test, whose stop command was not sent;
- the `base` container's whole log;
- the tools these runs used, and `analysis.py` with its output.

Hosts, trees and times are in [`provenance.md`](provenance.md). Verify the digests, then
recompute the figures:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/chassis-bringup-2026-10-19/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
python3 scripts/analysis.py | diff - analysis.txt
```
