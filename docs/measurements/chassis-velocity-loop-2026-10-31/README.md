# The chassis velocity loop on blocks, 2026-10-31

On 2026-10-19 the front-right wheel did not turn at 0.1 m/s
([`chassis-bringup-2026-10-19`](../chassis-bringup-2026-10-19/README.md) §5). This record runs the
bench [`chassis-velocity-pid-tuning`](../../tasks/active/reliability/chassis-velocity-pid-tuning.md)
asks for, wheels lifted: the M2 open-loop deficit at a metered 12 V, each motor's breakaway speed
and full speed (QPPS), gains chosen in a bench tool, those gains through the driver, their save to
flash, and a reboot of the host.

It ran in two sessions, on 2026-10-31 UTC and from 23:53 that day to 03:12 on 2026-11-01; between
them the maintainer re-seated front-right's miter gears and oiled every wheel. The plan was
pre-registered in the deposit before the first motor command; three dated amendments, each
committed before the runs it covers, added to it or changed it. FL and FR are the front controller's (0x80) M1 and M2, RL and RR the rear's (0x81).
At 537.7 ticks per revolution and a 0.048 m wheel radius, 0.3 m/s is 535 ticks/s (6.25 rad/s) and
0.1 m/s is 178 ticks/s. The deposit's `scripts/analysis.py` recomputes every figure from the runs.

**Result.**
- **There is no steady deficit on FR.** At duty 32/127 over 4 s holds, FR ran 2.8 % over the FL/RL
  mean forward and 1.5 % over it backward. RR ran 0.8 % over it forward; backward it ran 4.1 %
  under, within three repeat spreads, and 3.1 % under in session 2, beyond them.
- **FR had a tight spot locked to its wheel angle, and its low-speed starts failed.** One 45° bin
  ran −9.2 to −9.5 % forward and −13.2 to −14.2 % backward, the same bin in every hold. At the old
  gains FR took 2.04 s to turn at 178 ticks/s, where the others took 0.16–0.34 s; at 150, 2.96 s
  (FL 0.28 s, RL 0.24 s, RR 2.28 s).
- **It was gone after the maintainer's work between the sessions,** which re-seated FR's miter
  gears and oiled every wheel. FR's worst bin then read −0.5 to −1.0 %, and FR turned within
  0.12 s at 178 in both repeats.
- **QPPS measured 2630.5–2672.0 ticks/s,** +1.58 % largest over smallest; their mean, 2657, is
  4.97 % under the nominal 2796. **The chosen gains are P 15000, I 750, D 0, QPPS 2657, shared by all four.** At
  2796 FL ran −6.00 % and −5.97 % at 0.3 m/s; at 2657 every wheel held −3.8 to −1.9 %.
- **Through the driver, on blocks, both tests passed:** every wheel within −4.0 to −0.2 % at
  0.3 m/s in both runs, and turning within 0.099–0.123 s at 0.1 m/s.
- **The gains are saved in both controllers' flash,** each save verified by what the controller
  loaded after its reset; the final read showed 15000/750/0/2657 on all four motors.
- **Duty 127 drives these controllers full reverse.** Every forward ladder's duty-127 step ran its
  wheel backward at full speed. The bench tool now stops at 126.
- **The host's setup survives a reboot.** After it the controllers held their gains and timeouts on
  new names, and the D555's kernel modules, HID binding and masked sensor service were as before.

## 1. Setup

- **Software.** Images built at `0423a13`: `strafer-cpu:humble` `c30981464f22`, labelled
  `0423a13fcde6`. The bench tool, `motor_bench.py`, ran in one-off containers of the `base`
  service with no other `base`-service container up. `base` ran only in step 6, with the task
  branch's source mounted over the image's ([`provenance.md`](provenance.md)).
- **The supply.** A 4S LiPo, a 300 W 12 V buck and a Y harness feed both controllers. The
  maintainer's meter at idle read the pack at 15.33 V and both leads at 12 V in session 1, and
  16.68 V and 12.07 V in session 2, after a recharge. The controllers read 10.2–10.3 V (front) and
  11.1–11.2 V (rear) throughout, ladder ramps down included.
- **The controllers** keep a 0.5 s serial timeout saved, and at power-up load their saved gains:
  front M1 14999/749/0/2796 and M2 15000/750/0/2796, rear all 0.
- **The bench tool** sends only duty commands open loop. Closed loop (`hold`) it checks the wiring
  (duty 32 for 0.4 s per channel), writes the gains to RAM and reads them back. Each hold starts
  cold, after a silence that trips each controller's timeout, and drives with command 37, as the
  driver does.
- **Metrics.** *S*: the count rate over the most whole revolutions in [1.0, 4.0] s of a hold. *The
  0.5 s count*: ticks in the first 0.5 s from rest. *Bins*: eight 45° bins of wheel angle, each
  bin's speed against the hold's mean. *Error*: the mean speed over [0.6 s, end of hold] against
  the target. *Breakaway*: the first sample with the count 10 ticks from its start.

## 2. Session 1: the M2 deficit, open loop

**1a, the 2026-10-19 test as it was:** `diagnose_roboclaw.py --drive`, duty 32/127 for 1 s per
motor with nothing sent, so the 0.5 s timeout ends each. From counts of 0, FL turned 276 ticks, FR
251, RL 264 and RR 224: FR −7.0 % and RR −17.0 % against the FL/RL mean (2026-10-19: FL and RL
253, FR 229, RR 195 or 228 from 0). **The expectation, FR about 10 % and RR 10 to 23 % below FL
and RL, was mixed:** RR is inside its band, FR short of it. 1a's counts start from 0, so its RR
figure carries no doubt about the starting count.

**1b, the longer hold:** duty 32/127, 4.0 s holds, forward and backward, 3 repeats. Session 2's S2a
(§5) repeated it after the maintainer's work.

| 1b / S2a | FL | FR | RL | RR |
|---|---|---|---|---|
| S forward, ticks/s (repeat SD) | 580.8 (1.4) / 616.9 (0.6) | 606.3 (4.5) / 636.0 (3.4) | 599.0 (8.0) / 624.4 (1.7) | 594.8 (12.1) / 634.9 (2.7) |
| against the FL/RL mean | −1.5 / −0.6 % | **+2.8** / +2.5 % | +1.5 / +0.6 % | +0.8 / +2.3 % |
| S backward | 583.2 (0.9) / 618.9 (0.0) | 607.1 (3.7) / 636.1 (1.4) | 613.7 (4.8) / 632.3 (3.5) | 574.0 (11.4) / 606.2 (4.4) |
| against the FL/RL mean | −2.5 / −1.1 % | +1.5 / +1.7 % | +2.5 / +1.1 % | −4.1 / −3.1 % |
| 0.5 s count against the FL/RL mean, fwd; back | −0.1 / +0.9 %; −1.3 / −0.2 % | +6.3 / +2.0 %; +7.1 / +1.6 % | +0.1 / −0.9 %; +1.3 / +0.2 % | −5.5 / −0.1 %; −10.3 / −6.1 % |
| worst 45° bin of the 3 holds, fwd | −0.8 to −1.2 / −0.9 % | **−9.2 to −9.5** / −0.5 to −0.8 % | −1.3 to −1.7 / −0.8 to −1.1 % | −1.5 to −2.9 / −0.4 to −0.7 % |
| worst 45° bin, back | −0.2 to −0.7 / −1.1 to −1.7 % | **−13.2 to −14.2** / −0.7 to −1.0 % | −1.1 to −2.0 / −0.7 to −0.8 % | −0.8 to −2.3 / −0.3 to −1.0 % |

- **1.1, FL and RL within 3 % of each other: failed,** at −3.04 % forward and −4.96 % backward,
  so each deficit is also given against each.
- **1.2, FR's forward S at least 3 % below the FL/RL mean, by more than 3 repeat spreads: failed.**
  FR ran +2.8 % over the mean (+4.4 % over FL, +1.2 % over RL), and +1.5 % backward (+4.1 % over
  FL, −1.1 % against RL). RR ran +0.8 % forward (+2.4 % over FL, −0.7 % against RL) and −4.1 %
  backward (−1.6 % and −6.5 %). By 1.2's test RR's backward deficit does not hold in 1b: its
  per-repeat spread was 1.6 points, so three spreads are 4.8. In S2a it does: −3.1 %, against
  three spreads of 1.2 points. On the 0.5 s count FR ran +6.3 % forward and +7.1 % backward, RR
  −5.5 % and −10.3 %.
- **1.3, one FR bin at least 8 % slow at the same angle (±1 bin) in every forward repeat, no other
  wheel's below −5 %: held.** FR's worst bins were −9.2, −9.5 and −9.4 %, all bin 2; the
  others' worst was −2.9 % (RR). **1.4, backward, had no prediction:** FR's bin 2 read −14.2,
  −13.5 and −13.2 %.
- **Steps 2 and 3, the FR↔FL lead swap, did not run,** by the maintainer's decision after 1b. With
  no steady deficit to locate, the swap could only have placed FR's +4.4 % over FL, and shown an
  angle-locked spot following the motor, which it must.

## 3. Session 1: breakaway and the duty ladder

**4a, breakaway at the old gains:** P 15000, I 750, D 0, QPPS 2796 in RAM, all four wheels at once,
3 s per speed, each from a cold start. S2b's two repeats (§5) stand beside it.

| ticks/s | FL: 4a / S2b r1 / r2 | FR | RL | RR |
|---|---|---|---|---|
| 60 | never / 2.62 / never | 2.20 / 1.94 / 1.96 | never / never / never | never / never / never |
| 90 | 2.68 / 1.08 / 1.40 | 0.52 / 0.38 / 0.28 | 2.52 / 2.30 / 1.50 | never / 1.94 / 1.74 |
| 120 | 1.34 / 0.22 / 0.18 | 0.36 / 0.16 / 0.18 | 0.90 / 0.32 / 0.28 | 2.70 / 0.30 / 0.68 |
| 150 | 0.28 / 0.16 / 0.14 | **2.96** / 0.14 / 0.14 | 0.24 / 0.16 / 0.14 | 2.28 / 0.16 / 0.14 |
| 178 | 0.16 / 0.12 / 0.12 | **2.04** / 0.12 / 0.12 | 0.16 / 0.14 / 0.14 | 0.34 / 0.13 / 0.12 |
| 240 | 0.12 / 0.08 / 0.10 | 0.12 / 0.08 / 0.08 | 0.10 / 0.09 / 0.10 | 0.12 / 0.10 / 0.10 |

Seconds from the command to 10 ticks; "never" is not within the 3 s hold. At 30 no wheel turned.

- **4.1, FR's breakaway speed above 178 ticks/s, or FR later than the other three at every speed:
  failed.** By 4a's definition, the lowest speed at which a wheel turns within 0.5 s, FR's was 120
  (0.36 s), and FR was later than all three others only at 150 and 178.
- **FR turned, then stalled, or turned late.** At 120 it moved its 10 ticks at 0.36 s, yet its
  mean error over [0.6, 3.0] s was −92.6 %; at 150 it took 2.96 s and ran −97.9 %; at 178, 2.04 s
  and −72.4 %, where the others ran −30.3 to −34.9 %. At 240 its last second ran −43.3 %, the
  others −18.9 to −20.7 %. That fits 1.3's tight spot catching the wheel. Its strict breakaway
  speed, turning within 0.5 s there and at every higher speed, was 240.
- **Within 3 s,** the lowest speeds turned were FL 90, FR 60, RL 90 and RR 120 ticks/s.

**4b, the duty ladder:** each channel forward and backward through duties 8, 12, 16, 20, 24, 32, 64
and 127, 3 s each, then a 4 s ramp down; S over [1.5, 3.0] s of each step.
- **Duty 127 drives these controllers full reverse.** Every forward ladder's 127 step ran its wheel
  backward at full speed (FL −2589, FR −2592, RL −2644, RR −2553 ticks/s), where the manual says
  "127 = full speed forward", and the ramp's first steps, 126 and down, drove it forward again.
  The backward ladders' 127 steps ran backward, as asked. The maintainer heard the reversals and
  found the wheels fine. `motor_bench.py` now refuses 127 on commands 0, 1, 4 and 5.
- **So 4b gives full speed backward only:** FL 2597, FR 2588, RL 2644, RR 2566 ticks/s, a mean of
  2598.75, 7.1 % under 2796. **4.3, measured QPPS below 2796 on every motor: held.** **4.2,
  whether the controllers' reading gap is a supply difference: within 3 %, so none:** FL against
  RL −1.8 %, FR against RR +0.9 %.
- **Low duty.** Every motor first moved at duty 8, RL forward only after 1.04 s. FR backward then
  stopped at 8 (S 0), so its lowest kept duty was 12; every other's was 8.

## 4. Between the sessions

- **The maintainer's work.** After step 4b the maintainer saw a slight wobble on FR. Before
  session 2 the maintainer found FR's miter gears slightly off centre, loosened them, turned them a
  few turns so they could set, and tightened them again. Every hub came off, every wheel's bearings
  and miter gears were oiled and cleaned, and the pack was recharged. So the amendment before
  session 2 made 1b, 4a and 4b the "before" for all four wheels, with S2a to S2c in place of the
  runs planned after 4b.
- **The controllers did not enumerate.** Both left USB at 05:44:00, when main power was cut. At
  23:21:36 the front's hub port saw a device again, and the rear's at 23:22:45, once the kernel had
  given up on the front's at 23:22:44; it gave up on the rear's at 23:23:52. That fits main power
  coming on with the USB plugged in (session 1 had powered first, then plugged). A re-plug with
  main power on failed the same way: the rear's port saw it at 00:06:08 and the front's at
  00:07:16, when the kernel reached it.
- **A reset of the host's USB2 hub fixed it.** `USBDEVFS_RESET` on hub `1-2` (`0424:2744`) at
  00:09:55 re-enumerated every device on it; both controllers came up at 00:09:57 on their
  session-1 names. The read at 00:10:21 found both by address, with 0.5 s timeouts, the saved
  gains and every count 0. Why the ports failed, and why the reset cleared them, is not shown.

## 5. Session 2: the wheels after the maintainer's work

**S2a, step 1b's command again** (§2's table).
- **S2a.1, FR's slow spot gone, every hold's worst bin at −5.0 % or above each way: gone, both
  ways.** **S2a.2, no bin below −5.0 % on the other three: held;** the worst was −1.7 %.
- **Reported.** 1.1 now holds: FL against RL −1.20 % and −2.12 %. Every wheel ran faster, S +4.2
  to +6.7 % forward and +3.0 to +6.1 % backward, and the 0.5 s count +1.9 to +13.1 %. RR's forward
  spread fell from 12.1 to 2.7 ticks/s. FR's excess over FL went from +4.4 to +3.1 % forward and
  from +4.1 to +2.8 % backward; over RL it read +1.9 % and +0.6 % (1b: +1.2 % and −1.1 %). The
  bins count from session 2's power-up zero, so their indices are not compared.

**S2b, breakaway at the old gains again,** with 2 repeats (§3's table).
- **S2b.1, FR turning within 0.5 s at 178 and 240 ticks/s in both repeats: held:** 0.12 s at 178
  and 0.08 s at 240 both times. Its error at 178 was −17.6 and −16.6 %, from −72.4 %. Step 5's C0
  prediction "At 178: FR late or still", written before the maintainer's work, was withdrawn.
- **Every wheel's breakaway speed fell.** By 4a's definition, 4a against S2b's two repeats: FL 150
  to 120 and 120, FR 120 to 90 and 90, RL 150 to 120 and 120, RR 178 to 120 and 150. Turning
  within 0.5 s there and at every higher speed in every repeat: FL 150 to 120, **FR 240 to 90**,
  RL 150 to 120, RR 178 to 150. Within 3 s: FL 60 and 90, FR 60, RL and RR 90.

**S2c, the ladder both ways to duty 126.**

| ticks/s | FL | FR | RL | RR |
|---|---|---|---|---|
| 4b, duty 127, backward ladder | 2597 | 2588 | 2644 | 2566 |
| S2c, duty 126, forward; backward | 2622; 2639 | 2659; 2657 | 2639; 2705 | 2725; 2607 |
| **QPPS for C1, the mean of the two** | **2630.5** | **2658.0** | **2672.0** | **2666.0** |
| against 2796 | −5.9 % | −4.9 % | −4.4 % | −4.6 % |

- **The shared value** is the mean of the four, 2656.62, so 2657: 4.97 % under 2796. Largest over
  smallest is +1.58 %, under the 5 % rule, so one set is shared. C1 ran: it would have been
  skipped within 3 % of 2796.
- **4.2 again: within 3 % in all four comparisons, so no supply difference:** FL against RL −0.6 %
  forward and −2.4 % backward, FR against RR −2.4 % and +1.9 %. **4.3 again: held.**
- **The friction offsets** went from 2.83–4.77 duty steps in 4b to 2.32–3.46; every forward one
  fell, FR's most, 4.17 to 2.32 (3.87 to 2.36 backward). The slopes, 21.12–22.08 ticks/s per duty
  step in 4b, read 21.30–22.19. Every motor kept turning from duty 8 both ways (4b: FR backward
  from 12).

## 6. Step 5: the gains, in the bench tool

6 s cold holds, 2 repeats at 535 ticks/s and 2 at 178. The pass: at 535, each wheel within ±5 %
over [0.6, 6.0] s in both repeats, with a peak of at most +15 %; at 178, each wheel's breakaway
within 0.5 s in both.

| error, r1 / r2 | FL | FR | RL | RR |
|---|---|---|---|---|
| **535, C0 (QPPS 2796)** | **−6.0 / −6.0** | −4.2 / −4.2 | −4.8 / −4.6 | −4.6 / −4.6 |
| **535, C1 (QPPS 2657)** | −3.8 / −3.8 | −1.9 / −2.0 | −2.5 / −2.4 | −2.3 / −2.3 |
| C1's error over C0's (5.1) | 0.634 | **0.466** | 0.527 | 0.513 |
| **178, C0** | −16.5 / −16.3 | −11.8 / −11.8 | −13.7 / −14.4 | −15.6 / −15.9 |
| **178, C1** | −13.7 / −13.7 | −9.7 / −9.5 | −11.9 / −11.2 | −13.3 / −12.8 |
| C1 backward, −535 | −3.6 | −2.0 | −2.0 | −3.9 |
| C1 spin, −534/+534/−534/+534 | −3.7 | −2.0 | −2.1 | −2.3 |

% of the target's magnitude; P 15000, I 750, D 0 throughout.

- **C0's expectation at 535, "−4 to −10 %, no overshoot, FL the worst": held,** with peaks of
  −2.1 to −0.6 %. **C0 fails the pass:** FL ran −6.00 % and −5.97 %.
- **C1 passes, and is chosen.** At 535 it ran −3.8 to −1.9 %, peaks −0.7 to +0.4 %, and reached
  90 % in 0.34–0.42 s (C0: 0.66–1.14 s). At 178 every wheel broke away within 0.10–0.12 s, as
  under C0. C2 (I raised) and C3 (P raised) were not run. The QPPS spread is +1.58 % and C1 is one
  set for all four motors, so it is shared, and the driver, which writes one set, needs no change.
- **5.1, C1 at least halving C0's shortfall at 535: failed. It held on FR only.** Over all four
  wheels the shortfall went from −4.87 % to −2.65 %, a ratio of 0.543.
- **Reported.** C1 backward and in the spin peaked at −0.9 to +0.7 % and −0.7 to +0.9 %, and
  broke away at 0.06 s. At 178 the cold holds ran low over the window, as the pass allows. No hold
  oscillated: the within-run SD at 535 was 1.1–1.8 % for C0 and 0.8–1.3 % for C1.

## 7. Step 6: through the driver

- **Setup and gate.** QPPS 2657 was committed in `strafer_shared.constants` on the task branch,
  and `base` ran with the branch's source mounted over the image's. Domain 42 had no `/cmd_vel` or
  `/strafer/cmd_vel` publisher before the start. The start log shows both `PID set` lines with
  `P=15000 I=750 D=0 QPPS=2657`, both read-backs identical, then `RoboClawNode ready.` No count
  at the start was below zero: FL 14050, FR 21202, RL 13652, RR 21675.
- **The runs.** `drive_probe.py` published on `/cmd_vel` at 20 Hz for 6 s from a separate
  container, recording `/strafer/joint_states`; the six ran back to back, as the maintainer asked.

| run, in order (first message, UTC) | FL | FR | RL | RR | 10 ticks after the first message |
|---|---|---|---|---|---|
| 0.3 m/s, 1 (01:23:33.626) | −4.0 % | −2.1 % | −2.6 % | −2.5 % | 0.082 s |
| 0.3 m/s, 2 (01:23:54.884) | −0.4 % | −0.2 % | −0.4 % | −0.2 % | 0.084 s |
| 0.1 m/s, 1 (01:24:34.185) | −3.0 % | −3.7 % | −4.6 % | −6.4 % | 0.123 s |
| 0.1 m/s, 2 (01:24:45.429) | −0.6 % | −0.8 % | −1.5 % | −1.2 % | 0.099 s; RR 0.119 s |
| −0.3 m/s (01:24:56.598) | −8.5 % | −5.2 % | −6.1 % | −8.7 % | 0.070 s |
| spin, 0.8 rad/s (01:25:07.735) | −1.3 % | −3.4 % | −0.7 % | −5.4 % | 0.073 s |

Each wheel's mean `joint_states` velocity over [first message + 0.6 s, last message], by arrival,
against its target's magnitude. The spin turns FL and RL backward.

- **6.1, each wheel within ±5 % of 6.25 rad/s in both 0.3 m/s runs: held,** at −4.0 to −0.2 %.
  **6.2, each wheel's count moving 10 ticks within 0.5 s of the first message in both 0.1 m/s
  runs: held,** at 0.099–0.123 s.
- **Backward and the spin, reported.** Backward ran −5.2 to −8.7 % over the window and −2.2 to
  −3.9 % over its last second. The spin ran −0.7 to −5.4 %; 2026-10-20's six spins, at QPPS 2796,
  ran −7 to −10 % over the same window, from 0.6 s to their last message
  ([`driver-hardening-bench-2026-10-25`](../driver-hardening-bench-2026-10-25/README.md) §4). The driver logged `No cmd_vel` 0.504–0.519 s after each last
  message, and all four wheels were under 0.05 rad/s by +0.95 to +1.07 s.
- **In step 6, each run in the same direction as the one before it ran closer to the target.** The
  pre-registration named C0 for this comparison; step 6 ran C1's gains, so it is made against C1's
  cold bench holds (§6):
  - the first 0.3 m/s run, the first after `base`'s start wrote the gains, ran −4.0 to −2.1 %, as
    the cold holds did (−3.8 to −1.9 %), and the second, after it, ran −0.4 to −0.2 %;
  - both 0.1 m/s runs, after the forward runs, ran nearer than the cold −13.7 to −9.5 %;
  - the backward run, after four forward runs, ran further short than the cold −3.9 to −2.0 %;
  - in the spin, FL and RL, still turning backward, beat their cold −3.7 and −2.1 %; FR and RR,
    reversed, fell short of their cold −2.0 and −2.3 %.
- **This fits the controllers keeping part of their integral term across zero-speed commands,**
  and clearing it at a gains write or a serial-timeout trip. Between the runs the controllers took
  the watchdog's zero-speed commands every tick, with no trip and no write (the driver's watchdog,
  `roboclaw_node.py`); every bench hold began after a trip, and its repeats agree. The integral
  term was not read, neither the manual nor any record says what clears it, and the bench's window
  runs from the first command, the driver's from the first message. 6.1 holds on the first run
  alone.

## 8. The flash save

After step 6 the maintainer chose to save the gains to both controllers' flash, with a tool kept in
the tree: `diagnose_roboclaw.py --set-gains --write-nvm` (`4624293`), committed with its sequence
in an amendment before it touched a controller. It ran on one controller at a time, `base`
stopped, and then `--read-only` read both.

| | front, 03:04:47.226 | rear, 03:11:17.869 |
|---|---|---|
| active before (commands 55, 56) | 15000/750/0/2657 on M1 and M2 | 15000/750/0/2657 on M1 and M2 |
| after command 95, as saved | timeout 0.5 s; M1 14999/749/0/2796, M2 15000/750/0/2796 | timeout 0.5 s; all 0 |
| set (28, 29) and read back | 15000/750/0/2657, identical | 15000/750/0/2657, identical |
| command 94: the port dropped after; back on its name | 0.1 s; 2.8 s later | 0.2 s; 2.8 s later |
| loaded at the reset (55, 56, 15) | 15000/750/0/2657 on both, 0.5 s; exit 0 | 15000/750/0/2657 on both, 0.5 s; exit 0 |

- **F.1, each save exiting 0 with its port seen to drop, the driver's gains and a 0.5 s timeout
  loaded at the reset: held** on both; each printed "Saved and verified." The kernel logged
  `1-2.3` out at 03:04:49 and back at 03:04:51, and `1-2.4` at 03:11:19 and 03:11:22. **F.2, the
  final read showing 15000/750/0/2657 on all four motors and both timeouts at 0.5 s: held,** at
  03:12:19.885.
- **Command 95 reloads the gains, not only the timeout.** Before it both controllers held QPPS 2657
  in active memory; after it, the saved 2796 (front) and zeros (rear). Sent before the set, it made
  the save change nothing but the gains.
- **The reset is not a power cycle.** A main-power cycle with the USB out first was offered as the
  stronger check, and the maintainer chose the read without it.

## 9. Step 7: the reboot

With `base` stopped and the D555 unplugged (off USB at 04:09:21), the host rebooted at 04:12:47.
The controllers were left on main power.
- **The controllers came back on new names, found by address.** With the D555 out at boot they
  enumerated at 04:12:59 as `ttyACM0` and `ttyACM1`: front `/dev/roboclaw0`, rear `/dev/roboclaw1`,
  both with `ID_MM_DEVICE_IGNORE=1`, as step 7 expected. The hub enumerated them at once.
- **The read at 04:19:14:** both serial timeouts 0.5 s, and 15000/750/0/2657 on all four motors;
  every count 0, as after the flash saves' resets.
- **The clock** read synchronized.
- **The D555, plugged in again at 04:25:01:** `/sys/module/uvcvideo/srcversion`
  `D0E2A5944399A529814A98C`, its HID device bound to `hid-sensor-hub`, `iio-sensor-proxy`
  `masked`. Its serial interface took `ttyACM2`, with no `roboclaw` name. It first enumerated at
  SuperSpeed Plus, dropped at once, and came back at SuperSpeed.
- **Every step-7 expectation held.** This closes the "survives a reboot" box of
  [`d555-l4t-stream-integrity`](../../tasks/active/reliability/d555-l4t-stream-integrity.md), and
  the reboot half of [`jetson-untether`](../../tasks/active/reliability/jetson-untether.md)'s udev
  box.

## 10. Findings

Findings 1–5, 7 and 9 rest on pre-registered tests; 6 comes from step 4b's runs, as the amendment
after it records; 8 is an observation.

1. **No steady deficit on FR at a metered 12 V** (1.2 failed; §2). The 2026-10-19 figures came
   from the 1 s duty test, a start from rest plus the spin-down; 1a, the same test, gave FR −7.0 %
   and RR −17.0 % (2026-10-19: −9.5 % and −22.9 %). RR's backward shortfall, −4.1 % in 1b and
   −3.1 % in S2a, holds by 1.2's test in S2a only, and had no prediction.
2. **FR's low-speed fault was mechanical, and it is gone** (§3, §5). With the tight spot locked to
   its wheel angle, FR turned at 0.36 s at 120 ticks/s and then ran −92.6 %, and took 2.96 s at 150
   and 2.04 s at 178. It went after the maintainer's work between the sessions, FR's miter gears
   re-seated and every wheel oiled (S2a.1, S2b.1).
3. **QPPS is 2657, shared** (§5, §6). Every motor runs below the nominal 2796 (4.3), within
   +1.58 % of each other. C1, the lowest-gain candidate that passed, is chosen. The lower QPPS cut
   every wheel's shortfall at 535 but halved it on FR only, so 5.1 failed (§6).
4. **On blocks, through the driver, the chosen gains pass at 0.3 m/s and 0.1 m/s** (6.1, 6.2),
   and the first 0.3 m/s run, the first after `base`'s start, passes on its own (§7).
5. **The gains live in `strafer_shared.constants` and in both controllers' flash** (§8), so a tool
   that drives the controllers without the driver finds them after a reset; a power cycle is not
   shown.
6. **Duty 127 on the forward commands drives these controllers full reverse** (§3).
7. **The controllers' reading gap is not a supply difference** (4.2; §3, §5).
8. **In step 6, each run in the same direction as the one before it ran closer to the target**
   (§7). It is an observation, not a measured mechanism.
9. **The D555's host setup and the controllers' udev names and gains survive a reboot** (§9).

## What this does not show

- **Driving on the floor.** Every run had the wheels lifted. The brief's "on the floor at 0.1 m/s"
  box, with the heading drift over a measured distance, and the breakaway on the floor stay open.
- **That the flash survives a power cycle.** Command 94's reset is not one, the main-power cycle
  with the USB out first was declined, and the controllers kept main power through the reboot.
- **Backward through the driver from a cold start.** The one backward run followed four forward
  runs and ran −5.2 to −8.7 %. No backward test was pre-registered.
- **The integral term itself.** It was not read; §7's carry-over is inferred from the speeds.
- **Higher gains.** C2 and C3 were not run, because C1 passed. Whether they would track 178 ticks/s
  from a cold start better than C1's −13.7 to −9.5 % is open.
- **The lead swap.** Steps 2 and 3 did not run, so the motor and M2's output stage are not
  separated; with no steady deficit, the swap had nothing to place.
- **The D555's stream after the reboot.** The IMU rate and the aligned depth were not re-read; they
  are read at the next perception start.
- **Which battery reading is right,** and why the controllers' USB failed to enumerate.
- **`make test-ros` and the documentation box,** which belong to the constants change's pull
  request, not to this bench.

## Evidence

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `chassis-velocity-loop-2026-10-31/record-files/` |
| deposit commit | `6cce8172`, the final one |
| pre-registration commit | `c19fe7de` |
| session 1, and the amendment after step 4b | `8d0dc89a` |
| the amendment before session 2's first motor command | `64704ebe` |
| session 2's data | `c75fc3f7` |
| the amendment before the flash save | `35428e1e` |
| the flash outputs (`flash/`), their USB events, `scripts/analysis.py` and `analysis.txt` | `4cc75b84` |
| step 7, the reboot (`s7/`, `reads/s7_read.txt`), and its section of `analysis.py` | `6cce8172` |

**What the deposit holds:**
- `PREREGISTRATION.md` and its amendments;
- each question put to the maintainer, answered verbatim (`operator_notes.txt`);
- every bench call's output, read and run CSV, with the summaries;
- step 6's census, start log and mounts, and each run's `joint_states`, publisher output, `base` log
  and summary;
- the flash outputs and the kernel's USB events;
- step 7's state before and after the reboot (`s7/`) and its read;
- the tools, with `analysis.py` (standard library only) and its output.

Hosts, trees and times are in [`provenance.md`](provenance.md). Verify the digests, then recompute
the figures:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/chassis-velocity-loop-2026-10-31/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
python3 scripts/analysis.py | diff - analysis.txt
```
