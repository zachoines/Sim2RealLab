# The driver hardening on the robot, 2026-10-25

#253 (`0423a13`) changed three things in the RoboClaw driver:
- **The stall mark.** A message that waited through a stall of the executor is dropped. The mark
  holds through rcl's catch-up tick until the first take.
- **The reconnect.** It re-writes the velocity gains and reads them back before the error state
  clears.
- **Startup.** The gains are read back at startup too.

This record checks them on the robot. It also measures the stop's dose-response on the
controllers' serial timeout.

Every expectation was pre-registered in the deposit before the first run. The chassis stood on
blocks. Every drive was a 0.8 rad/s spin, 6.24 rad/s commanded at each wheel, published on
`/cmd_vel` from a separate container. The webcam on FR was framed and analysed as in
[`chassis-stop-modes-2026-10-20`](../chassis-stop-modes-2026-10-20/README.md). The deposit's
`scripts/analysis.py` recomputes every figure from the runs.

**Result.**
- **A message that waited through a freeze no longer re-arms the wheels.**
  - The run was 2026-10-20's pause_3, repeated twice: the publisher sent 2 s into a 3 s
    `docker pause` and then stopped.
  - Each run gave one watchdog line after the unpause. All 325 and 326 ticks after it read
    0.00 rad/s with no count change. In pause_3c, the run with the camera, the wheel stayed still
    (maximum motion 0.76).
  - The tick after the overdue one came 8 ms later in pause_3b, as on 2026-10-20, and 16 ms later
    in pause_3c (17 ms in the dose run): the tick due next on the timer's 20 ms grid. The stale
    message was dropped both times.
  - In pause_3c the freeze landed inside a tick's body, in an executor pass that did not hold the
    subscription. A pass that held it is the fix's residual case, and it stays unmeasured.
- **The stop follows the controllers' packet-serial timeout.**
  - The dose run raised the controllers' active timeout from 0.5 s to 1.0 s. FR's last motion
    came 1.055 s after the driver's last whole sample, against 0.552–0.567 s at 0.5 s. On that
    basis and from the stamp of the last tick that commanded the controllers, the +0.5 s moved the
    stop by +0.49 to +0.51 s (§3).
  - The encoders turned 1.03–1.08 s at speed.
  - A power cycle reverts the active value to the saved 0.5 s, and did.
- **The driver writes its gains and reads them back, at startup and after a reconnect.**
  - Every start, and both reconnects that found the controllers on their old names, read back
    P 15000, I 750, D 0, QPPS 2796 on all four motors, with no restart of `base`.
  - At the end the rear held those gains, though its saved gains are zero.
- **A main-power cut can swap the controllers' names.** After the first of two cuts with the USB
  in, the controllers came back swapped. The driver's 25 reconnect attempts all failed, so it
  stayed stopped, as pre-registered, until a restart of `base` detected them again.
  - The kernel enumerated the front first after both cuts, and the rear first at the day's first
    power-up. The names follow that order.

## 1. Setup

- **Software.** Images built at `0423a13`: `strafer-cpu:humble` `c30981464f22`, labelled
  `0423a13fcde6`. Compose came from a clean worktree at that commit, with `base` on
  `stop_signal: SIGINT` and `restart: unless-stopped`. The pre-registration is the deposit's
  `PREREGISTRATION.md`, committed before the first run.
- **The controllers.** Both keep a 0.5 s serial timeout saved.
  - Today they first enumerated rear first: rear `/dev/roboclaw1`, front `/dev/roboclaw2`.
  - Their saved gains read front M1 P 14999, I 749 (M2 15000, 750) and rear all 0.
  - They read their supply as 10.2 V and 11.1 V, the low readings of
    `chassis-stop-modes-2026-10-20` §1.
- **Isolation.** Before `base`'s first start, a census of DDS domain 42 found no publisher on
  `/cmd_vel`. `diagnose_roboclaw.py` ran only with `base` stopped.
- **The camera.** At rest it measured a mean motion of 0.83 and a maximum of 2.65, so the
  pre-registered threshold of 4.0 stands. Every frame time is an arrival on the host, late by the
  webcam's unmeasured latency.
- **Samples are timed by their stamps,** the tick's clock read.
  - A sample can arrive after the unpause although its tick began before the freeze. pause_3c has
    one, stamped +0.032 s from the pause.
  - So the analysis classes samples by stamp, and takes the travel across a pause from the last
    tick whose sample arrived before the unpause: the last whole tick.

## 2. pause_3 again, on the fixed driver

| | pause_3b | pause_3c |
|---|---|---|
| publisher's last message, from the unpause | −1.026 s | −1.005 s |
| last whole tick before the freeze (stamp, from the pause) | +0.017 s | +0.012 s |
| first ticks after the unpause (stamps) | +0.036, **+0.044**, +0.061 s | +0.039, **+0.055**, +0.075 s |
| the second tick, after the overdue one | 8.3 ms | 15.8 ms |
| travel across the pause, time at the last speed | 0.53–0.56 s | 0.50–0.56 s |
| watchdog lines after the unpause | 1: +0.037 s, `No cmd_vel for 3.06s` | 1: +0.040 s, `No cmd_vel for 3.09s` |
| ticks after the unpause | 325, all 0.00 rad/s, no count change | 326, all 0.00 rad/s, no count change |
| camera | not recorded | at rest +0.601 s after the pause; after the unpause max 0.76, no frame over 4.0 |

- **The catch-up tick.**
  - pause_3b's second tick after the unpause came 8 ms after the overdue one, as on 2026-10-20.
    pause_3c's came 16 ms after it, and the dose run's (§3) 17 ms.
  - Each is the tick due next on the timer's 20 ms grid, as the ticks before the pause set it. In
    pause_3b it ran 3 ms past its point, behind the overdue tick's body. From the third tick on,
    every run is back on the grid.
  - So how soon the catch-up tick follows depends on where the overdue tick falls on the grid. A
    mark that every tick reset would have been cleared by that tick, before any take that came
    after it, as pause_3's take did on 2026-10-20.
- **A freeze inside a tick body.**
  - In pause_3c the tick stamped +0.032 s read the rear's speeds before the freeze and its encoder
    counts after it.
  - Its sample arrived 0.038 s after the unpause. Against the first tick after the unpause, its
    front positions are 3.13 and 3.01 rad short and its rear positions match: the rear's counts
    were read after the stop.
  - The next tick began more than the watchdog window later, so the mark was set and the message
    still dropped.
  - That executor pass did not hold the subscription. A pass that held it would have taken the
    waiting message after the frozen tick, before the overdue tick could set the mark, and the
    message would have re-armed the wheels. Nothing re-armed.
  - So the residual that `driver-stale-cmd-vel-rearm` names, a freeze inside a tick body in a pass
    that already held the subscription, stays unmeasured.
- **No line marks the drop.** The single watchdog line and the stillness are its evidence.
- **The controllers' stop at the pause** is unchanged by the fix. FR's last motion came 0.567 s
  after the last whole tick's sample. A tick sends both drive commands before it reads a speed or
  a count, so the tick the freeze cut had already commanded both controllers, 20 ms later: FR's last motion
  came 0.553 s after that tick's stamp.
- **pause_3b has no camera.** It was launched by a relative path, which left the webcam
  unstarted. pause_3c repeated it with the camera.

## 3. The dose-response: a 1.0 s serial timeout

- **The set.** With `base` stopped, `diagnose_roboclaw.py --set-serial-timeout 1.0`, without
  `--write-nvm`, set both controllers' active packet-serial timeout and read back 1.0 s. The saved
  value stayed 0.5 s.
- **One pause run** with the publisher stopped at the pause, as 2026-10-20's pause_2:

| | at 0.5 s (2026-10-20) | at 1.0 s |
|---|---|---|
| FR's last motion after the pause, camera (thr 4.0) | +0.579 to +0.593 s | **+1.077 s** |
| FR's last motion after the driver's last sample | 0.552–0.556 s | **1.055 s** |
| travel across the pause, time at the last speed | 0.49–0.58 s | **1.03–1.08 s** |
| after the unpause | one watchdog line, no motion | one watchdog line, 325 ticks at 0.00, camera max 0.74 |

- **Across thresholds 3.0 to 6.0** FR's last motion moves from +1.077 to +1.057 s.
- **+0.5 s on the active timeout moved the stop by +0.49 to +0.51 s.**
  - Timed from the arrival of the driver's last whole sample: +0.488 s against pause_3c's 0.567 s
    the same day, and +0.499 to +0.503 s against 2026-10-20's five freeze and kill runs.
  - Timed from the stamp of the last tick that commanded the controllers: +0.509 s against
    pause_3c, whose last commanding tick is the one the freeze cut (§2). In the dose run it is the
    last whole tick.
  - So the controllers' own timeout, not anything in the driver, is what stops a frozen driver's
    wheels.
- **The power cycle.** `base` was then to be stopped before a main-power cycle with the USB out
  first, but it was stopped only after the cycle (§4).
  - A power cycle reverts the active value to the saved one: the following `--read-only` read
    0.5 s on both. That holds whichever went first, the USB or main power.
  - The gains read P 15000, I 750, D 0, QPPS 2796 on all four motors.

## 4. The reconnect

Three main-power cycles ran with `base` up and no publisher.

| | the dose test's cycle | first cut, USB in | second cut, USB in |
|---|---|---|---|
| controllers off USB, apart | 0.148 s | 0.148 s | 0.152 s |
| the driver's error state | +0.153 s, `10 consecutive failures over 0.30s` | +0.166 s, over 0.38 s | +0.145 s, over 0.41 s |
| back on USB, as the kernel enumerated them | rear, then front 3.58 s later | front, then rear 0.34 s later | front, then rear 0.34 s later |
| back on | the old names | **swapped**: front `ttyACM1`, rear `ttyACM2` | the names `base` was using |
| reconnect | +0.53 s after the re-enumeration; gains read back | 25 attempts, all failed | +1.89 s after the re-enumeration; gains read back |
| restarts of `base` | 0 | 0, then a restart by hand | 0 |

- **The names follow the enumeration order.** `99-strafer.rules` names each RoboClaw
  `/dev/roboclaw%n` after its `ttyACM` number, and the D555 holds `ttyACM0`. The kernel enumerated
  the rear first at the day's first power-up and at the dose test's cycle, and the front first
  after both USB-in cuts.
- **The dose test's cycle** was not planned as a reconnect check. `base` had not been stopped
  before it.
  - The maintainer's answer puts the USB out before main power. The kernel log is equally
    consistent with main power first: the controllers left USB 0.148 s apart, as in both USB-in
    cuts, and came back 3.6 s apart.
  - They came back on the old names, and the driver re-wrote and read back its gains, then
    cleared its error state.
  - With no publisher nothing could drive, but no encoder read covers the cycle: none was
    recorded before the first cut reset the counts.
- **The first cut: the competing outcome.**
  - The front enumerated first, against the rear-first order `base` had detected at its start, so
    each controller came back on the other's name.
  - `reconnect()` reopens the paths detection chose at start and probes each at its own address.
    All 25 attempts, one every 2 s from the error state to the restart, failed. The first five
    failed within 2.4 ms each, four of them before either controller re-enumerated. The other 20
    met the wrong controller and took 0.42–0.43 s each, both probes timing out.
  - The driver stayed stopped in its error state. With no publisher nothing could drive; as for
    the dose test's cycle, no encoder read covers the cut.
  - A restart of `base` detected front `/dev/roboclaw1` and rear `/dev/roboclaw2`, and its
    startup read-back passed with both controllers fresh from power-up.
- **The second cut: the planned case.** The front enumerated first again, matching the restart's
  detection, so the names held. The driver's next attempt re-wrote the gains, read back P 15000,
  I 750, D 0, QPPS 2796 on all four motors, and logged `RoboClaw reconnected successfully.` There
  was no restart of `base`. The encoders cover this cut: at the spin's first sample, 102 s after
  the controllers came back, all four counts read 0.
- **The spin on the reconnected driver.**
  - All four wheels turned. The publisher ran 2.35 s, not the planned 4 s (`provenance.md`).
    From 0.6 s after the first message to the last, 0.6–2.35 s, the means were FL −5.37,
    FR +5.49, RL −5.59 and RR +5.42 rad/s: −10 % to −14 % of 6.24, within the pre-registered
    ±15 %.
  - That window includes the ramp: each wheel first reached 90 % of 6.24 rad/s 1.0–2.3 s after
    the first message.
  - Over the same 0.6–2.35 s 2026-10-20's six spins ran −9 % to −14 %. So −10 % to −14 % is
    like-for-like, not an effect of the reconnect's gains. Over their whole holds, to their last
    message 4.3–6.4 s in, they ran −7 % to −10 %, the figure the pre-registration quotes.
  - Over the last second of publishing the means were −9 % to −12 %.
  - The watchdog fired 0.506 s after the last message, and all four wheels were under 0.05 rad/s
    by +1.05 s.
- **The final read,** with `base` stopped: P 15000, I 750, D 0, QPPS 2796 on all four motors, and
  0.5 s on both.

## 5. Findings

The pre-registered checks are findings 1–3. Finding 4 comes from the competing outcome that was
pre-registered for step 3.

1. **The stall drop holds on the robot,** through the catch-up tick and through a freeze inside a
   tick body (§2). That freeze came in an executor pass that did not hold the subscription. The
   residual case, a pass that did and would take the message before the overdue tick, stays
   unmeasured.
2. **The stop of a frozen driver's wheels is the controllers' packet-serial timeout.** Raising
   the active value from 0.5 s to 1.0 s moved the stop by +0.49 to +0.51 s (§3).
3. **The gains are re-applied and read back on hardware** at every start and at every reconnect
   that found the controllers. The rear, whose saved gains are zero, ends holding the driver's
   (§4).
4. **The enumeration order is not stable across main-power cuts.** The kernel enumerated the rear
   first at the day's first power-up and at the dose test's cycle, and the front first after both
   USB-in cuts. So the first cut swapped the names `base` had detected, and the driver stayed
   stopped until `base` restarted. That is safe, but it needs a person. Filed as
   [`driver-reconnect-redetects-by-address`](../../tasks/active/reliability/driver-reconnect-redetects-by-address.md):
   on a failed reconnect, detect the controllers again by address. Until it ships, the restart
   stays the recovery.

## What this does not show

- **Driving on the floor.** Every run had the wheels in the air.
- **A real hang,** rather than the cgroup freezer that `docker pause` uses.
- **A controller that stops answering but stays on USB.** On the robot a power cut drops a
  RoboClaw off USB, so that case is covered by the fake-bus tests only.
- **More than one stale message after a stall.** Neither run showed one.
- **A freeze inside a tick body in an executor pass that already held the subscription,** the
  residual `driver-stale-cmd-vel-rearm` names. pause_3c's freeze came in a pass that did not
  (§2).
- **Whether anything moved through the dose test's cycle or the first cut.** Nothing could drive,
  with no publisher, but no encoder read covers either (§4).
- **The webcam's latency,** which makes every camera time late by an unmeasured amount.

## Evidence

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `driver-hardening-bench-2026-10-25/record-files/` |
| deposit commit | `520d984fe98bc8e907e545faf0e16a5d5b0a29c2` |
| pre-registration commit | `0b4a7665` |

**What the deposit holds:**
- the gates;
- every run's events, publisher summary, `joint_states` samples and container log;
- the webcam frames around pause_3c's and the dose run's pause and unpause (pause_3b has none);
- the kernel's USB events and `base`'s whole log;
- each question put to the maintainer, verbatim;
- the harness, including `pause_run.sh` as pause_3b ran it;
- `analysis.py` with its output, and a dated correction in `DEPOSIT.md`.

`analysis.py` also reads the deposit of `chassis-stop-modes-2026-10-20`, beside this one in the
same clone, for the comparisons with that day's runs. Hosts, trees and times are in
[`provenance.md`](provenance.md). Verify the digests, then recompute the figures (with OpenCV and
PIL):

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/driver-hardening-bench-2026-10-25/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
python3 scripts/analysis.py | diff - analysis.txt
```
