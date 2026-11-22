#!/usr/bin/env python3
"""PID tuning script for RoboClaw velocity controllers.

Programmatically sets PID/QPPS values and runs step-response tests that
measure each motor: it drives with command 37, as the driver does, reads the
encoder counts (commands 16 and 17), passes a motor whose mean speed tracks
the target within ±5 %, and reports the dead time and time constant fitted
to the count. The robot is tuned for tracking and then measured; nothing
here aims at a time constant.

Usage:
    # Read current PID on both controllers
    python3 tune_pid.py --read

    # Set PID values and run step-response test
    python3 tune_pid.py --tune

    # Set specific PID values without running a test
    python3 tune_pid.py --set --p 15000 --i 1000 --d 0

    # Run step-response only (after PID is already set)
    python3 tune_pid.py --step-response

    All commands accept --controller front|rear|both (default: both)
    and --motor m1|m2|both (default: both)
"""

from __future__ import annotations
from strafer_shared.constants import (
    RADIANS_TO_ENCODER_TICKS,
    ROBOCLAW_BAUD_RATE,
    ROBOCLAW_FRONT_ADDRESS,
    ROBOCLAW_PID_D,
    ROBOCLAW_PID_I,
    ROBOCLAW_PID_P,
    ROBOCLAW_QPPS,
    ROBOCLAW_REAR_ADDRESS,
    WHEEL_RADIUS,
)
from strafer_driver.roboclaw_interface import (
    RoboClawInterface,
    detect_roboclaws,
)

import argparse
import atexit
import bisect
import math
import signal
import time
from typing import NamedTuple


# Global list of open controllers for cleanup on exit
_open_controllers: list[RoboClawInterface] = []

# ---------------------------------------------------------------------------
# Step-response parameters
# ---------------------------------------------------------------------------

# The gains the driver writes at every start; --tune writes them unless given others.
DEFAULT_P = ROBOCLAW_PID_P
DEFAULT_I = ROBOCLAW_PID_I
DEFAULT_D = ROBOCLAW_PID_D

# The tracking box: at 0.3 m/s, each wheel's mean speed from 0.6 s after the
# first command to the last is within ±5 % of its target.
STEP_SPEED_MPS = 0.3
STEP_VELOCITY_TICKS = round(STEP_SPEED_MPS / WHEEL_RADIUS * RADIANS_TO_ENCODER_TICKS)  # 535
TRACK_START_S = 0.6
TRACK_TOLERANCE = 0.05

STEP_HOLD_S = 6.0
MIN_HOLD_S = TRACK_START_S + 0.4
MAX_HOLD_S = 30.0
SAMPLE_PERIOD_S = 0.02  # the driver's 50 Hz: command 37 re-sent and the counts read each cycle
PRE_S = 0.2  # counts read at rest before the first command
SETTLE_TIME_S = 0.2  # Wait after stopping before next test
REST_TICKS = 2  # a count that moves more than this over PRE_S is not at rest

# A step drives only while the controller's packet-serial timeout (command 15)
# is armed, so the controller stops the wheels if this tool dies mid-step.
SERIAL_TIMEOUT_RANGE_S = (0.1, 1.0)
STOP_TRIES = 3  # each motor's duty-0 stop is retried until acknowledged
# After a final stop that is not acknowledged, no packet for longer than the
# timeout, so the controller's timeout stops the wheels.
SILENT_S = 1.2

# Aborts: a motor turning against its target (reversed motor or encoder
# leads), running away from it, or stalling (a dead encoder, or a jammed wheel
# or one not broken away). Both motors are watched, the one held at 0 too.
AGAINST_TPS = 150  # ticks/s against the target ...
AGAINST_SAMPLES = 3  # ... for this many samples in a row
RUNAWAY_MARGIN_TPS = 300  # over 2 * |target| + this ...
RUNAWAY_SAMPLES = 10  # ... for this many samples in a row
STALL_AFTER_S = 0.5  # from this long after the command, a stepped motor whose count ...
STALL_WINDOW_S = 0.3  # ... over the last this long ...
STALL_FRACTION = 0.1  # ... moved under this fraction of its target's travel has stalled
MAX_MISSED = 2  # count reads failed, or command 37 unacknowledged, this many cycles in a row
# ROBOCLAW_QPPS is the motors' mean full speed: the largest step keeps the
# runaway limit 10 % under it, for the slowest motor and the speed's scatter.
MAX_STEP_TICKS = (int(0.9 * ROBOCLAW_QPPS) - RUNAWAY_MARGIN_TPS) // 2

# The fit covers the rise, before the tracking window opens: after it, the
# integral term's slow approach to the target is no first-order lag, and over a
# whole hold it pins the dead time to 0 and stretches tau.
FIT_END_S = TRACK_START_S
# The fit: dead time on [0, 0.3] s and tau on [5 ms, 3 s], coarse to fine.
DEAD_TIME_MAX_S = 0.3
TAU_MIN_S = 0.005
TAU_MAX_S = 3.0
MIN_FIT_SAMPLES = 10


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class StepFit(NamedTuple):
    """A dead time and a first-order lag fitted to a step's encoder count."""

    dead_time_s: float
    tau_s: float
    speed: float  # the level the rise settles toward, ticks/s
    rms_ticks: float  # the count's rms residual about the fit


class StepTrace:
    """One motor's step: its count from rest, timed from the first command."""

    def __init__(self, motor: str, target: int) -> None:
        self.motor = motor
        self.target = target
        self.times: list[float] = []  # s, when each count read returned
        self.counts: list[int] = []  # ticks from the count at rest
        self.abort: str | None = None


def count_delta(a: int, b: int) -> int:
    """Signed change from count *a* to count *b*, folding the 32-bit wrap."""
    return ((b - a + 2**31) % 2**32) - 2**31


def _step_sse(
    times: list[float], counts: list[float], sum_cc: float, dead: float, tau: float
) -> tuple[float, float]:
    """(sum of squared residuals, best final speed) for one dead time and tau."""
    sum_gg = sum_gc = 0.0
    for t, c in zip(times, counts):
        x = t - dead
        if x <= 0.0:
            continue
        g = x - tau * (1.0 - math.exp(-x / tau))
        sum_gg += g * g
        sum_gc += g * c
    if sum_gg == 0.0:
        return sum_cc, 0.0
    return sum_cc - sum_gc * sum_gc / sum_gg, sum_gc / sum_gg


def fit_dead_time_and_tau(
    times: list[float], counts: list[float], end_s: float = FIT_END_S
) -> StepFit | None:
    """Fit a dead time and a time constant to a step's encoder count.

    The speed is a first-order lag behind a dead time; its integral is the count:

        count(t) = V * (x - tau * (1 - exp(-x / tau))),  x = max(0, t - dead_time)

    with *times* in seconds from the first command and *counts* in ticks from
    the count at rest, over the samples before *end_s*. For each (dead time,
    tau) the best V is linear least squares. The pair comes from a coarse grid,
    then a pattern search from its best point: a 3x3 stencil that moves while
    it improves and halves its steps when it does not, so it follows the
    valley where more dead time trades for less tau. The dead time is bounded
    below by 0. Fitting the count, not the reported speed (commands 18 and 19),
    avoids the speed reading's lag behind the wheel.

    The dead time plus tau (the time to 63 % of the level) is better resolved
    than the split between them. From whole-tick counts at 50 Hz alone, for a
    535 ticks/s step with tau near the sample period, the sum scatters by
    under 1 ms (one standard deviation) and the split by about 2 ms; both grow
    as the level falls, to about 2 and 5 ms for a 178 ticks/s step, whose
    rise levels off near 140.

    Returns None with fewer than MIN_FIT_SAMPLES samples, or a count that
    never leaves rest.
    """
    pts = [(t, float(c)) for t, c in zip(times, counts) if 0.0 <= t < end_s]
    if len(pts) < MIN_FIT_SAMPLES or max(abs(c) for _, c in pts) <= REST_TICKS:
        return None
    ts = [t for t, _ in pts]
    cs = [c for _, c in pts]
    sum_cc = sum(c * c for c in cs)

    def trial(dead, tau, best):
        dead = min(max(dead, 0.0), DEAD_TIME_MAX_S)
        tau = min(max(tau, TAU_MIN_S), TAU_MAX_S)
        sse, v = _step_sse(ts, cs, sum_cc, dead, tau)
        return (sse, dead, tau, v) if sse < best[0] else best

    n_tau = 60
    ratio = (TAU_MAX_S / TAU_MIN_S) ** (1.0 / (n_tau - 1))
    best = (math.inf, 0.0, TAU_MIN_S, 0.0)
    for i in range(round(DEAD_TIME_MAX_S / 0.01) + 1):
        for k in range(n_tau):
            best = trial(i * 0.01, TAU_MIN_S * ratio**k, best)

    dead_step, tau_step = 0.01, ratio  # tau steps are factors
    while dead_step > 1e-6:
        center = best
        for dd in (-1, 0, 1):
            for dt in (-1, 0, 1):
                best = trial(center[1] + dd * dead_step, center[2] * tau_step**dt, best)
        if best is center:
            dead_step /= 2.0
            tau_step = math.sqrt(tau_step)

    sse, dead, tau, v = best
    return StepFit(
        dead_time_s=dead,
        tau_s=tau,
        speed=v,
        rms_ticks=math.sqrt(max(sse, 0.0) / len(ts)),
    )


def tracking_speed(
    times: list[float], counts: list[float], start: float, end: float
) -> float | None:
    """Mean speed over [start, end] from the count, in ticks/s.

    The count's change between the first and last samples in the window, over
    the time between them. None with fewer than two samples there.
    """
    win = [(t, c) for t, c in zip(times, counts) if start <= t <= end]
    if len(win) < 2 or win[-1][0] <= win[0][0]:
        return None
    (t0, c0), (t1, c1) = win[0], win[-1]
    return (c1 - c0) / (t1 - t0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hard_stop(rc: RoboClawInterface) -> bool:
    """Force-stop both motors using duty-cycle commands (bypasses PID).

    This sends forward_m1(0) and forward_m2(0) which switch the RoboClaw
    out of velocity PID mode into duty-cycle mode with 0% duty = stopped.
    This is critical because drive_m*_speed(0) only sets the PID target
    to 0, and with accumulated I-term or reversed motors, the PID may
    continue driving.

    Each motor's stop is sent on its own and retried until acknowledged, up
    to STOP_TRIES times. A Ctrl-C, SIGTERM or SIGHUP meanwhile is held and
    raised once both have been sent; a second one raises at once, abandoning
    the stops (a write that hangs, say) to the controller's serial timeout.
    Returns True when both were acknowledged.
    """
    held: list[int] = []

    def hold(signum: int, _frame) -> None:
        if held:
            raise KeyboardInterrupt(f"signal {signum}")
        held.append(signum)

    handlers = {}
    try:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, hold)
        except ValueError:
            pass  # not the main thread: the stops go out unheld
        acked = True
        for stop in (rc.forward_m1, rc.forward_m2):
            for _ in range(STOP_TRIES):
                try:
                    if stop(0):
                        break
                except Exception:
                    pass  # Best-effort during cleanup
            else:
                acked = False
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)
    if held:
        raise KeyboardInterrupt(f"signal {held[0]}")
    return acked


def _atexit_stop_all() -> None:
    """Emergency stop all motors on script exit."""
    for rc in _open_controllers:
        hard_stop(rc)
        try:
            rc.close()
        except Exception:
            pass
    _open_controllers.clear()


atexit.register(_atexit_stop_all)


def open_controller(port: str, address: int) -> RoboClawInterface:
    """Open and return a RoboClawInterface for the given port."""
    rc = RoboClawInterface(
        port=port,
        address=address,
        baud_rate=ROBOCLAW_BAUD_RATE,
        read_timeout=0.1,
        max_retries=2,
    )
    rc.open()
    hard_stop(rc)  # Ensure motors are stopped on connect
    _open_controllers.append(rc)
    return rc


def close_controller(rc: RoboClawInterface) -> None:
    """Hard-stop and close a controller, removing it from the cleanup list."""
    hard_stop(rc)
    try:
        rc.close()
    except Exception:
        pass
    if rc in _open_controllers:
        _open_controllers.remove(rc)


# Cached auto-detected port mapping
_detected_ports: dict[int, str] | None = None


def _get_detected_ports() -> dict[int, str]:
    """Auto-detect RoboClaw ports (cached)."""
    global _detected_ports
    if _detected_ports is None:
        _detected_ports = detect_roboclaws(
            ROBOCLAW_FRONT_ADDRESS, ROBOCLAW_REAR_ADDRESS, ROBOCLAW_BAUD_RATE
        )
        if _detected_ports:
            print(
                f"Auto-detected: {', '.join(f'0x{a:02X}->{p}' for a, p in _detected_ports.items())}"
            )
        else:
            print(
                "WARNING: No RoboClaws found on /dev/roboclaw* (the only names scanned; "
                "`make udev` installs 99-strafer.rules)."
            )
    return _detected_ports


def get_controllers(which: str) -> list[tuple[str, str, int]]:
    """Return list of (label, port, address) based on --controller arg."""
    detected = _get_detected_ports()
    controllers = []
    if which in ("front", "both"):
        port = detected.get(ROBOCLAW_FRONT_ADDRESS)
        if port:
            controllers.append(("Front", port, ROBOCLAW_FRONT_ADDRESS))
        else:
            print(f"WARNING: Front RoboClaw (0x{ROBOCLAW_FRONT_ADDRESS:02X}) not detected.")
    if which in ("rear", "both"):
        port = detected.get(ROBOCLAW_REAR_ADDRESS)
        if port:
            controllers.append(("Rear", port, ROBOCLAW_REAR_ADDRESS))
        else:
            print(f"WARNING: Rear RoboClaw (0x{ROBOCLAW_REAR_ADDRESS:02X}) not detected.")
    return controllers


def get_motor_list(which: str) -> list[str]:
    """Return list of motor names based on --motor arg."""
    if which == "m1":
        return ["M1"]
    if which == "m2":
        return ["M2"]
    return ["M1", "M2"]


def _read_count(rc: RoboClawInterface, motor: str) -> tuple[float, int]:
    """(monotonic time the read returned, raw count) for one motor."""
    count, _ = rc.read_encoder_m1() if motor == "M1" else rc.read_encoder_m2()
    return time.monotonic(), count


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_read(args: argparse.Namespace) -> None:
    """Read and display current PID values."""
    for label, port, addr in get_controllers(args.controller):
        print(f"\n{'='*50}")
        print(f"  {label} RoboClaw (port={port}, addr=0x{addr:02X})")
        print(f"{'='*50}")
        try:
            rc = open_controller(port, addr)
            pid_m1 = rc.read_velocity_pid_m1()
            pid_m2 = rc.read_velocity_pid_m2()
            rc.close()
            print(
                f"  M1 PID: P={pid_m1['P']:>6}  I={pid_m1['I']:>6}  D={pid_m1['D']:>6}  QPPS={pid_m1['QPPS']}"
            )
            print(
                f"  M2 PID: P={pid_m2['P']:>6}  I={pid_m2['I']:>6}  D={pid_m2['D']:>6}  QPPS={pid_m2['QPPS']}"
            )
        except Exception as e:
            print(f"  ERROR: {e}")


def cmd_set(args: argparse.Namespace) -> None:
    """Set PID values on selected controllers/motors."""
    p = args.p
    i = args.i
    d = args.d
    qpps = args.qpps or ROBOCLAW_QPPS
    motors = get_motor_list(args.motor)

    for label, port, addr in get_controllers(args.controller):
        print(f"\n{label} RoboClaw (0x{addr:02X}):")
        try:
            rc = open_controller(port, addr)

            for motor in motors:
                if motor == "M1":
                    ok = rc.set_velocity_pid_m1(p, i, d, qpps)
                else:
                    ok = rc.set_velocity_pid_m2(p, i, d, qpps)
                status = "OK" if ok else "FAILED"
                print(f"  {motor} SET P={p} I={i} D={d} QPPS={qpps} -> {status}")

            # Verify by reading back
            pid_m1 = rc.read_velocity_pid_m1()
            pid_m2 = rc.read_velocity_pid_m2()
            print(
                f"  Readback M1: P={pid_m1['P']} I={pid_m1['I']} D={pid_m1['D']} QPPS={pid_m1['QPPS']}"
            )
            print(
                f"  Readback M2: P={pid_m2['P']} I={pid_m2['I']} D={pid_m2['D']} QPPS={pid_m2['QPPS']}"
            )

            rc.close()
        except Exception as e:
            print(f"  ERROR: {e}")


def run_step_response(
    rc: RoboClawInterface,
    motors: list[str],
    speed_ticks: int,
    hold_s: float,
) -> list[StepTrace]:
    """Step the selected motors of one controller to *speed_ticks* and record their counts.

    Command 37 sets both motors at once, as the driver does, and is re-sent
    every SAMPLE_PERIOD_S for *hold_s*; a motor not selected is held at 0.
    Each cycle reads both motors' counts (commands 16 and 17). The step does
    not start unless the controller's serial timeout is armed, the stop is
    acknowledged and both counts are at rest. It stops, for both motors, when
    a motor turns against its target, runs away from it, or (if stepped)
    stalls, or when reads fail or command 37 goes unacknowledged MAX_MISSED
    cycles in a row. An unacknowledged final stop is followed by SILENT_S
    without a packet, and a motor still moving after the stop fails.
    """
    traces = {m: StepTrace(motor=m, target=speed_ticks) for m in motors}
    target = {m: speed_ticks if m in motors else 0 for m in ("M1", "M2")}

    def abort(reason: str, by: str | None = None) -> list[StepTrace]:
        for m, tr in traces.items():
            tr.abort = reason if by in (None, m) else f"stopped with {by}: {reason}"
        return list(traces.values())

    try:
        timeout = rc.read_serial_timeout()
    except Exception as e:
        return abort(f"serial timeout unread ({e}); not stepped")
    lo, hi = SERIAL_TIMEOUT_RANGE_S
    if not lo <= timeout <= hi:
        return abort(
            f"serial timeout {timeout:g} s (0 disables it); a step needs {lo:g} to {hi:g} s, "
            "so the controller stops the wheels if this tool dies. Not stepped; set it with "
            "diagnose_roboclaw.py --set-serial-timeout."
        )

    # Hard-stop both motors before test (duty-cycle, bypasses PID)
    if not hard_stop(rc):
        return abort("the duty-0 stop went unacknowledged; not stepped")
    time.sleep(SETTLE_TIME_S)

    # The counts at rest: the last before the first command is each motor's zero.
    rest: dict[str, list[int]] = {m: [] for m in target}
    end_pre = time.monotonic() + PRE_S
    while time.monotonic() < end_pre:
        for m in target:
            rest[m].append(_read_count(rc, m)[1])
        time.sleep(SAMPLE_PERIOD_S)
    for m in target:
        moved = max(abs(count_delta(rest[m][0], c)) for c in rest[m])
        if moved > REST_TICKS:
            hard_stop(rc)
            return abort(
                f"{m} not at rest before the step (count moved {moved} ticks); not stepped", m
            )

    zero = {m: rest[m][-1] for m in target}
    last = {m: (0.0, 0) for m in target}
    against = {m: 0 for m in target}
    runaway = {m: 0 for m in target}
    missed_reads = missed_acks = 0
    try:
        t0 = time.monotonic()
        next_cycle = t0
        while time.monotonic() - t0 < hold_s:
            missed_acks = 0 if rc.drive_m1m2_speed(target["M1"], target["M2"]) else missed_acks + 1
            if missed_acks >= MAX_MISSED:
                abort(f"command 37 unacknowledged {MAX_MISSED} cycles in a row")
                break
            try:
                reads = {m: _read_count(rc, m) for m in target}
            except Exception:
                missed_reads += 1
                if missed_reads >= MAX_MISSED:
                    abort(f"{MAX_MISSED} count reads failed in a row")
                    break
                reads = None
            stop = None
            if reads is not None:
                missed_reads = 0
                for m, (t_read, raw) in reads.items():
                    t, c = t_read - t0, count_delta(zero[m], raw)
                    if m in traces:
                        traces[m].times.append(t)
                        traces[m].counts.append(c)
                    t_prev, c_prev = last[m]
                    last[m] = (t, c)
                    if t <= t_prev:
                        continue
                    v = (c - c_prev) / (t - t_prev)
                    goal = target[m]
                    backward = goal != 0 and v * math.copysign(1, goal) < -AGAINST_TPS
                    against[m] = against[m] + 1 if backward else 0
                    fast = abs(v) > 2 * abs(goal) + RUNAWAY_MARGIN_TPS
                    runaway[m] = runaway[m] + 1 if fast else 0
                    if against[m] >= AGAINST_SAMPLES:
                        stop = (
                            f"{m} turns against its target ({v:+.0f} ticks/s for "
                            f"{goal:+d}): REVERSED motor/encoder direction? "
                            f"Swap {m}A / {m}B."
                        )
                    elif runaway[m] >= RUNAWAY_SAMPLES:
                        stop = f"{m} ran away: {v:+.0f} ticks/s for a target of {goal:+d}"
                    elif goal != 0 and t >= STALL_AFTER_S:
                        stop = _stalled(traces[m], goal)
                    if stop:
                        hard_stop(rc)
                        abort(stop, m)
                        break
            if stop:
                break
            next_cycle += SAMPLE_PERIOD_S
            time.sleep(max(0.0, next_cycle - time.monotonic()))
    finally:
        # Hard-stop motor (duty-cycle, not PID)
        stopped = hard_stop(rc)
        if not stopped:
            print(
                f"    STOP NOT ACKNOWLEDGED: silent {SILENT_S:g} s so the controller's serial "
                "timeout stops the wheels."
            )
            time.sleep(SILENT_S)

    # Verify the motors stopped: a wheel still turning SETTLE_TIME_S after the
    # stop is stopped again, and fails.
    faults = [] if stopped else [f"the duty-0 stop went unacknowledged (silent {SILENT_S:g} s)"]
    try:
        time.sleep(SETTLE_TIME_S)
        before = {m: _read_count(rc, m)[1] for m in target}
        time.sleep(SETTLE_TIME_S)
        after = {m: _read_count(rc, m)[1] for m in target}
        moving = [m for m in target if abs(count_delta(before[m], after[m])) > REST_TICKS]
        if moving:
            hard_stop(rc)
            faults.append(f"{', '.join(moving)} still moving after the stop")
    except Exception as e:
        faults.append(f"the stop could not be checked ({e})")
    for tr in traces.values():
        tr.abort = "; ".join(([tr.abort] if tr.abort else []) + faults) or None

    return list(traces.values())


def _stalled(trace: StepTrace, goal: int) -> str | None:
    """Why a stepped motor has stalled, or None.

    It has, when its count over the last STALL_WINDOW_S moved under
    STALL_FRACTION of the travel its target asks for.
    """
    t, c = trace.times[-1], trace.counts[-1]
    k = bisect.bisect_right(trace.times, t - STALL_WINDOW_S) - 1
    if k < 0:
        return None
    moved, span = abs(c - trace.counts[k]), t - trace.times[k]
    if moved >= STALL_FRACTION * abs(goal) * span:
        return None
    return (
        f"{trace.motor}'s count moved {moved} ticks in {span:.2f} s at {goal:+d} ticks/s: "
        "a dead encoder, a jammed wheel, or one not broken away at this speed"
    )


def _report_step_result(trace: StepTrace, hold_s: float, verbose: bool) -> bool:
    """Print one motor's tracking verdict and its fit. True on a pass."""
    speed = trace.target
    if trace.abort:
        print(f"    ABORT: {trace.abort}")
        return False

    if not trace.times:
        print(f"    No data collected -- check motor/encoder connection")
        return False

    tracked = tracking_speed(trace.times, trace.counts, TRACK_START_S, hold_s)
    print(f"    Samples:      {len(trace.times)}")
    if tracked is None:
        print(f"    FAIL -- too few samples in [{TRACK_START_S}, {hold_s}] s to measure tracking")
        return False
    error = tracked / speed - 1.0
    print(
        f"    Tracking:     {tracked:.0f} ticks/s over [{TRACK_START_S}, {hold_s:g}] s "
        f"(target={speed}, error={error*100:+.1f}%)"
    )

    fit = fit_dead_time_and_tau(trace.times, trace.counts)
    if fit is None:
        print(
            f"    Fit:          none (too few samples before {FIT_END_S} s, "
            "or the count did not leave rest)"
        )
    else:
        dead_at = (
            " (the fit's floor)" if fit.dead_time_s <= 0.0
            else " (the fit's bound)" if fit.dead_time_s >= DEAD_TIME_MAX_S else ""
        )
        tau_at = " (the fit's bound)" if not TAU_MIN_S < fit.tau_s < TAU_MAX_S else ""
        print(
            f"    Dead time:    {fit.dead_time_s*1000:.1f} ms{dead_at}   "
            f"tau: {fit.tau_s*1000:.1f} ms{tau_at}   "
            f"63% at {(fit.dead_time_s + fit.tau_s)*1000:.1f} ms"
        )
        print(
            f"                  (fit to the count before {FIT_END_S} s: "
            f"level {fit.speed:.0f} ticks/s, rms {fit.rms_ticks:.1f} ticks)"
        )
        if tau_at or fit.dead_time_s >= DEAD_TIME_MAX_S:
            print(
                f"    WARNING: The fit sits at a bound: the rise before {FIT_END_S} s is not "
                "measured. Do not record these."
            )

    if abs(tracked) < abs(speed) * 0.1:
        print("    WARNING: Mean speed is <10% of target.")
        print(f"             Motor may be stalled, disconnected, or wheels loaded.")
        print(f"             Lift the robot so wheels spin freely for tuning.")
    elif abs(tracked) < abs(speed) * 0.5:
        print("    WARNING: Mean speed is <50% of target.")
        print(f"             Check motor wiring and ensure wheels are unloaded.")

    passed = abs(error) <= TRACK_TOLERANCE
    if passed:
        print(f"    PASS -- tracks within {TRACK_TOLERANCE*100:.0f}% of target")
    else:
        side = "under" if error < 0 else "over"
        print(
            f"    FAIL -- tracks {abs(error)*100:.1f}% {side} target "
            f"(box: ±{TRACK_TOLERANCE*100:.0f}%)"
        )

    if verbose:
        print(f"\n    {'Time(ms)':>10}  {'Count':>10}")
        for t, c in zip(trace.times, trace.counts):
            print(f"    {t*1000:10.1f}  {c:10d}")
    return passed


def _step_controller(
    rc: RoboClawInterface, motors: list[str], speed: int, hold_s: float, verbose: bool
) -> tuple[bool, bool]:
    """Step one controller's selected motors and report each.

    Returns (every motor passed, the step was aborted).
    """
    print(f"  Testing {', '.join(motors)} (command 37, {hold_s:g} s hold)...")
    traces = run_step_response(rc, motors, speed, hold_s)
    passed = True
    for trace in traces:
        print(f"  {trace.motor}:")
        passed &= _report_step_result(trace, hold_s, verbose)
    time.sleep(SETTLE_TIME_S)
    return passed, any(trace.abort for trace in traces)


def _selected_controllers(which: str) -> list[tuple[str, str, int]]:
    """The controllers --controller names, or none unless every one was detected."""
    controllers = get_controllers(which)
    if len(controllers) < (2 if which == "both" else 1):
        print("ERROR: not every selected controller was detected; nothing is stepped.")
        return []
    return controllers


STOP_AFTER_ABORT = "\n  Not stepping the remaining controllers after the abort."


def cmd_step_response(args: argparse.Namespace) -> bool:
    """Run the step-response test; report tracking, dead time and tau. True if all passed."""
    motors = get_motor_list(args.motor)
    speed = args.speed if args.speed is not None else STEP_VELOCITY_TICKS
    controllers = _selected_controllers(args.controller)
    passed = bool(controllers)

    for label, port, addr in controllers:
        print(f"\n{'='*50}")
        print(f"  {label} RoboClaw (0x{addr:02X}) -- Step Response")
        print(f"{'='*50}")

        try:
            rc = open_controller(port, addr)

            # Show current PID
            pid_m1 = rc.read_velocity_pid_m1()
            pid_m2 = rc.read_velocity_pid_m2()
            print(
                f"  Current M1 PID: P={pid_m1['P']} I={pid_m1['I']} D={pid_m1['D']} QPPS={pid_m1['QPPS']}"
            )
            print(
                f"  Current M2 PID: P={pid_m2['P']} I={pid_m2['I']} D={pid_m2['D']} QPPS={pid_m2['QPPS']}"
            )
            print(f"  Step target: {speed} ticks/s ({speed / ROBOCLAW_QPPS * 100:.0f}% of QPPS)")
            print()

            ok, aborted = _step_controller(rc, motors, speed, args.hold, args.verbose)
            passed &= ok
            close_controller(rc)
            if aborted:
                print(STOP_AFTER_ABORT)
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            passed = False
            print(STOP_AFTER_ABORT)
            break
    print()
    print("  The dead time and tau above are measurements, not targets: nothing here")
    print("  passes or fails on them.")
    return passed


def cmd_tune(args: argparse.Namespace) -> bool:
    """Set PID values and immediately run the step-response test. True if all passed."""
    p = args.p or DEFAULT_P
    i_val = args.i if args.i is not None else DEFAULT_I
    d = args.d if args.d is not None else DEFAULT_D
    qpps = args.qpps or ROBOCLAW_QPPS
    motors = get_motor_list(args.motor)
    speed = args.speed if args.speed is not None else STEP_VELOCITY_TICKS
    controllers = _selected_controllers(args.controller)
    passed = bool(controllers)

    for label, port, addr in controllers:
        print(f"\n{'='*60}")
        print(f"  {label} RoboClaw (0x{addr:02X}) -- Tune + Step Response")
        print(f"{'='*60}")

        try:
            rc = open_controller(port, addr)

            # Set PID
            written = True
            for motor in motors:
                if motor == "M1":
                    ok = rc.set_velocity_pid_m1(p, i_val, d, qpps)
                else:
                    ok = rc.set_velocity_pid_m2(p, i_val, d, qpps)
                written &= bool(ok)
                status = "OK" if ok else "FAILED"
                print(f"  {motor} SET P={p} I={i_val} D={d} QPPS={qpps} -> {status}")

            # Verify
            for motor in motors:
                if motor == "M1":
                    pid = rc.read_velocity_pid_m1()
                else:
                    pid = rc.read_velocity_pid_m2()
                written &= (pid["P"], pid["I"], pid["D"], pid["QPPS"]) == (p, i_val, d, qpps)
                print(
                    f"  {motor} readback: P={pid['P']} I={pid['I']} D={pid['D']} QPPS={pid['QPPS']}"
                )
            if not written:
                print("  ERROR: the gains were not confirmed; not stepping on other gains.")
                close_controller(rc)
                passed = False
                print(STOP_AFTER_ABORT)
                break

            print(
                f"\n  Step target: {speed} ticks/s ({speed / qpps * 100:.0f}% of QPPS)"
            )
            print()

            ok, aborted = _step_controller(rc, motors, speed, args.hold, args.verbose)
            passed &= ok
            close_controller(rc)
            if aborted:
                print(STOP_AFTER_ABORT)
                break
        except Exception as e:
            print(f"  ERROR: {e}")
            passed = False
            print(STOP_AFTER_ABORT)
            break

    print(f"\n{'='*60}")
    print("  Tuning Summary")
    print(f"{'='*60}")
    print(f"  Gains:       P={p} I={i_val} D={d} QPPS={qpps}")
    print(
        f"  Tracking:    {'PASS' if passed else 'FAIL'} -- the box is "
        f"±{TRACK_TOLERANCE*100:.0f}% of the target, from {TRACK_START_S} s after "
        "the first command to the last"
    )
    print()
    print("  The dead time and tau above are measurements: record them with the gains.")
    print("  They are not targets, and nothing here passes or fails on them.")
    print()
    print("  Iterate by running:  python3 tune_pid.py --tune --p <new_P> --i <new_I>")
    print("  PID values stored in RAM only — set automatically on startup")
    print("  by the driver node and test scripts.")
    return passed


def cmd_stop(args: argparse.Namespace) -> bool:
    """Emergency stop all motors. True if every selected controller acknowledged its stop."""
    controllers = get_controllers(args.controller)
    stopped = len(controllers) == (2 if args.controller == "both" else 1)
    for label, port, addr in controllers:
        print(f"  Stopping {label} (0x{addr:02X})...", end=" ")
        try:
            rc = open_controller(port, addr)
            # hard_stop is called by open_controller, but call again explicitly
            acked = hard_stop(rc)
            # Also read back speed to verify
            spd1, _ = rc.read_speed_m1()
            spd2, _ = rc.read_speed_m2()
            close_controller(rc)
            print(f"{'OK' if acked else 'FAILED, unacknowledged'} (M1={spd1}, M2={spd2} ticks/s)")
            stopped &= acked
        except Exception as e:
            print(f"ERROR: {e}")
            stopped = False
    return stopped


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RoboClaw velocity PID tuning for Strafer chassis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tune_pid.py --read                          # Show current PID
  python3 tune_pid.py --tune                          # Set defaults + test
  python3 tune_pid.py --tune --p 20000 --i 500        # Custom PID + test
  python3 tune_pid.py --set --p 15000 --i 750 --d 0   # Set PID only (RAM)
  python3 tune_pid.py --step-response                  # Test only
  python3 tune_pid.py --step-response -v               # Test with raw data
  python3 tune_pid.py --tune --controller front        # Front only
  python3 tune_pid.py --tune --motor m1                # M1 only (all controllers)

The step drives the wheels: lift the chassis so every wheel is off the ground.
It needs each controller's serial timeout armed (0.1 to 1.0 s).
Exit status of --step-response and --tune: 1 if a motor fails tracking, a step is
aborted or errs, a selected controller is missing or (--tune) the gains are not
confirmed. Of --stop: 1 if a selected controller is missing or a stop is not
acknowledged. 130 if interrupted (Ctrl-C, SIGTERM, SIGHUP); else 0.
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read", action="store_true", help="Read current PID values")
    mode.add_argument("--tune", action="store_true", help="Set PID + run step-response")
    mode.add_argument("--set", action="store_true", help="Set PID values only")
    mode.add_argument(
        "--step-response", action="store_true", help="Run step-response only"
    )
    mode.add_argument("--stop", action="store_true", help="Emergency stop all motors")

    parser.add_argument(
        "--controller",
        choices=["front", "rear", "both"],
        default="both",
        help="Which controller(s) to operate on (default: both)",
    )
    parser.add_argument(
        "--motor",
        choices=["m1", "m2", "both"],
        default="both",
        help=(
            "Which motor(s) to operate on (default: both); a step holds the other at 0 "
            "with its speed loop"
        ),
    )

    parser.add_argument(
        "--p", type=int, default=None, help=f"P gain (default for --tune: {DEFAULT_P})"
    )
    parser.add_argument(
        "--i", type=int, default=None, help=f"I gain (default for --tune: {DEFAULT_I})"
    )
    parser.add_argument(
        "--d", type=int, default=None, help=f"D gain (default for --tune: {DEFAULT_D})"
    )
    parser.add_argument(
        "--qpps", type=int, default=None, help=f"QPPS (default: {ROBOCLAW_QPPS})"
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=None,
        help=(
            f"Step-response target speed in ticks/s "
            f"(default: {STEP_VELOCITY_TICKS}, {STEP_SPEED_MPS} m/s)"
        ),
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=STEP_HOLD_S,
        help=f"Step hold in seconds (default: {STEP_HOLD_S:g}; {MIN_HOLD_S:g} to {MAX_HOLD_S:g})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print raw step-response counts"
    )
    args = parser.parse_args()

    # Validate --set requires at least one PID value
    if (
        args.set
        and args.p is None
        and args.i is None
        and args.d is None
        and args.qpps is None
    ):
        parser.error("--set requires at least one of --p, --i, --d, --qpps")
    stepping = args.tune or args.step_response
    if stepping and not MIN_HOLD_S <= args.hold <= MAX_HOLD_S:
        parser.error(
            f"--hold must be from {MIN_HOLD_S:g} s, to measure tracking from {TRACK_START_S} s, "
            f"to {MAX_HOLD_S:g} s"
        )
    if stepping and args.speed is not None and not 0 < abs(args.speed) <= MAX_STEP_TICKS:
        parser.error(
            f"--speed must be nonzero and at most {MAX_STEP_TICKS} ticks/s either way, "
            "so the runaway abort fires 10 % under full speed"
        )

    # A terminal hang-up or a kill takes the same path as Ctrl-C, so the
    # motors are stopped on the way out.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)

    try:
        if args.stop:
            return 0 if cmd_stop(args) else 1
        elif args.read:
            cmd_read(args)
        elif args.set:
            # For --set, default unspecified values to 0 rather than the tuning defaults
            if args.p is None:
                parser.error("--set requires --p")
            if args.i is None:
                args.i = 0
            if args.d is None:
                args.d = 0
            cmd_set(args)
        elif args.step_response:
            return 0 if cmd_step_response(args) else 1
        elif args.tune:
            return 0 if cmd_tune(args) else 1
    except KeyboardInterrupt:
        print("\nInterrupted: stopping the motors.")
        _atexit_stop_all()
        return 130
    return 0


def _interrupt(signum: int, _frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


if __name__ == "__main__":
    raise SystemExit(main())
