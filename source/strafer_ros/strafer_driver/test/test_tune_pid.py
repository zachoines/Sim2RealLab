"""Tests for tune_pid.py: the step it drives, the fit to the count, and the tracking verdict.

The tuner drives each controller with command 37, as the driver does, and reads the encoder
counts (commands 16 and 17). On the default step a motor passes when its mean speed from 0.6 s
after the first command to the end of the hold is within ±5 % of the target; off it, the tuner
reports the tracking with no verdict. It reports a dead time and a time constant fitted to the
count before 0.6 s, and aims at neither. The fit runs on
synthetic steps and on two bench holds of the real wheels (data/step_holds/SOURCE.md). The
step runs on test_port_fence's fake bus with a simulated motor behind each channel.
"""

import csv
import functools
import importlib.util
import inspect
import math
import os
import re
import signal
import statistics
import threading
import struct
import sys
from unittest.mock import MagicMock, patch

import pytest

from strafer_driver.roboclaw_interface import _crc16
from strafer_shared.constants import (
    ROBOCLAW_PID_D,
    ROBOCLAW_PID_I,
    ROBOCLAW_PID_P,
    ROBOCLAW_QPPS,
    ROBOCLAW_SERIAL_TIMEOUT_S,
)

# test/ has no __init__.py, so pytest puts it on sys.path; the imported autouse fixture
# applies here.
from test_port_fence import (  # noqa: F401
    CAMERA_PORT,
    FRONT,
    REAR,
    FakeBus,
    FakeCamera,
    FakeRoboClaw,
    patched,
)

TUNER = os.path.join(os.path.dirname(__file__), "..", "..", "tune_pid.py")
TRACES = os.path.join(os.path.dirname(__file__), "data", "step_holds")
CHANNELS = ("F1", "F2", "R1", "R2")
HOLD_S = 6.0  # the bench's hold for these runs

# The tracking box, the one the driver's gains passed, and the fit's end.
BOX_TOLERANCE = 0.05
BOX_START_S = 0.6

# The manual's numbers, written out as in test_port_fence.
DRIVE_M1M2_SPEED = 37
DRIVE_ONE_SPEED = {35, 36}
READ_COUNTS = {16, 17}
READ_SPEEDS = {18, 19}
DUTY_FORWARD = {0, 4}
SET_PID = {28, 29}
MOTOR_COMMANDS = {0, 1, 4, 5, 35, 36, DRIVE_M1M2_SPEED}


def load_tuner():
    spec = importlib.util.spec_from_file_location("tune_pid", TUNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tuner = load_tuner()


def test_the_box_and_the_default_step():
    """±5 % from 0.6 s, the fit before it, and 0.3 m/s at the wheel: the bench's 535 ticks/s."""
    assert (tuner.TRACK_TOLERANCE, tuner.TRACK_START_S) == (BOX_TOLERANCE, BOX_START_S)
    assert tuner.FIT_END_S == BOX_START_S
    assert tuner.STEP_VELOCITY_TICKS == 535


def test_the_largest_step_leaves_the_runaway_rule_room():
    """At the largest step the runaway limit stays 10 % under the motors' mean full speed."""
    limit = 2 * tuner.MAX_STEP_TICKS + tuner.RUNAWAY_MARGIN_TPS
    assert limit <= 0.9 * ROBOCLAW_QPPS < limit + 2


# ---------------------------------------------------------------------------
# The fit and the tracking measure
# ---------------------------------------------------------------------------


def synthetic_step(dead, tau, level, period=0.02, n=300, shift=0.0, rest=0.5):
    """A first-order lag behind a dead time, sampled as the tuner samples it.

    Each read lands 1-3 ms into its cycle, plus *shift*; counts are whole ticks, the encoder
    at rest *rest* of a tick past its last edge.
    """
    times, counts = [], []
    for k in range(n):
        t = k * period + 0.001 + 0.0005 * ((7 * k) % 5) + shift
        x = max(0.0, t - dead)
        times.append(t)
        counts.append(math.floor(level * (x - tau * (1.0 - math.exp(-x / tau))) + rest))
    return times, counts


STEPS = [(0.012, 0.022, 510.0), (0.005, 0.035, 145.0), (0.05, 0.08, 1200.0)]


@pytest.mark.parametrize("dead, tau, level", STEPS)
def test_fit_is_exact_on_an_exact_count(dead, tau, level):
    times, _ = synthetic_step(dead, tau, level)
    xs = [max(0.0, t - dead) for t in times]
    exact = [level * (x - tau * (1.0 - math.exp(-x / tau))) for x in xs]

    fit = tuner.fit_dead_time_and_tau(times, exact)

    assert fit.dead_time_s == pytest.approx(dead, abs=1e-4)
    assert fit.tau_s == pytest.approx(tau, rel=1e-3)
    assert fit.speed == pytest.approx(level, rel=1e-4)


@pytest.mark.parametrize(
    "dead, tau, level, sum_sd, dead_bias",
    [
        (0.012, 0.022, 510.0, 0.0009, 0.0011),
        (0.005, 0.035, 145.0, 0.0030, 0.0032),
        (0.05, 0.08, 1200.0, 0.0005, 0.0005),
    ],
)
def test_fit_in_whole_ticks_scatters_as_documented(dead, tau, level, sum_sd, dead_bias):
    """Over 24 read phases and rest positions: unbiased, the sum tighter than the split.

    The scatter grows as the level falls, as the fit's docstring states.
    """
    fits = []
    for k in range(24):
        shift, rest = 0.02 * k / 24, (7 * k % 24) / 24
        times, counts = synthetic_step(dead, tau, level, shift=shift, rest=rest)
        fits.append(tuner.fit_dead_time_and_tau(times, counts))
    sums = [f.dead_time_s + f.tau_s - dead - tau for f in fits]
    deads = [f.dead_time_s - dead for f in fits]

    assert abs(statistics.mean(sums)) < sum_sd / 2
    assert statistics.pstdev(sums) < sum_sd
    assert abs(statistics.mean(deads)) < dead_bias
    assert statistics.pstdev(deads) > statistics.pstdev(sums)
    assert all(f.speed == pytest.approx(level, rel=0.01) for f in fits)


def test_fit_ignores_the_count_after_the_rise():
    """Samples from 0.6 s on, the integral term's slow approach, do not move the fit."""
    times, counts = synthetic_step(0.012, 0.022, 510.0)
    crept = [c + (round(40 * (t - BOX_START_S) ** 2) if t >= BOX_START_S else 0)
             for t, c in zip(times, counts)]

    assert tuner.fit_dead_time_and_tau(times, crept) == tuner.fit_dead_time_and_tau(times, counts)


def test_fit_needs_motion():
    times, _ = synthetic_step(0.0, 0.02, 0.0)

    assert tuner.fit_dead_time_and_tau(times, [0] * len(times)) is None
    assert tuner.fit_dead_time_and_tau(times[:5], list(range(5))) is None


def test_tracking_speed_is_the_count_over_the_window():
    times = [0.02 * k for k in range(301)]
    counts = [round(500 * t) if t >= 0.5 else 0 for t in times]

    assert tuner.tracking_speed(times, counts, 0.6, 6.0) == pytest.approx(500.0, rel=1e-3)
    assert tuner.tracking_speed(times, counts, 6.5, 7.0) is None


def test_count_delta_folds_the_wrap():
    assert tuner.count_delta(2**32 - 5, 3) == 8
    assert tuner.count_delta(3, 2**32 - 5) == -8
    assert tuner.count_delta(100, 40) == -60


def report(trace, hold_s, capsys):
    """(passed, printed lines) of the tuner's report on *trace*."""
    passed = tuner._report_step_result(trace, hold_s, False)
    return passed, capsys.readouterr().out


def trace_of(times, counts, target):
    trace = tuner.StepTrace(motor="M1", target=target)
    trace.times, trace.counts = list(times), list(counts)
    return trace


@pytest.mark.parametrize("dead, tau, flagged", [
    (0.4, 0.02, "Dead time:    300.0 ms (the fit's bound)"),
    (0.0, 0.002, "tau: 5.0 ms (the fit's bound)"),
])
def test_a_fit_at_its_bound_is_flagged_not_recorded(capsys, dead, tau, flagged):
    """A wheel that starts after the dead time's bound, or a lag shorter than tau's, gives no
    measurement to record."""
    _, out = report(trace_of(*synthetic_step(dead, tau, 535.0), 535), HOLD_S, capsys)

    assert flagged in out
    assert "The fit sits at a bound" in out


# ---------------------------------------------------------------------------
# Two bench holds of the real wheels (data/step_holds/)
# ---------------------------------------------------------------------------

# scipy.optimize.least_squares on the same samples, continuous in all three parameters and
# started from 16 points: (dead time ms, tau ms) of the count before 0.6 s, and the tracking
# error (%) over [0.6, 6.0] s.
REFERENCE = {
    ("all535_r1", "F1"): (12.83, 22.34, -3.71),
    ("all535_r1", "F2"): (13.70, 20.96, -1.87),
    ("all535_r1", "R1"): (15.99, 18.57, -2.48),
    ("all535_r1", "R2"): (11.51, 28.12, -2.31),
    ("all178_r1", "F1"): (5.98, 32.58, -13.43),
    ("all178_r1", "F2"): (0.00, 30.82, -9.25),
    ("all178_r1", "R1"): (3.59, 39.49, -11.67),
    ("all178_r1", "R2"): (4.51, 37.80, -12.82),
}
# The bench's own summary of the 535 hold: each motor's mean reported speed over
# [0.6, 6.0] s against its target, %.
BENCH_MEAN_535 = {"F1": -3.8, "F2": -1.9, "R1": -2.5, "R2": -2.3}


@functools.lru_cache(maxsize=None)
def bench_trace(run, channel):
    """(times, counts) of one channel's drive rows, as the bench timed them.

    Times are tr_F or tr_R, when that controller's count reads returned, in s from the
    first command; counts are from the last count before it.
    """
    with open(os.path.join(TRACES, run + ".csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    key = "tr_" + channel[0]
    zero = [int(r["c_" + channel]) for r in rows if r["phase"] == "pre"][-1]
    drive = [r for r in rows if r["phase"] == "drive" and r[key] and r["c_" + channel]]
    times = tuple(float(r[key]) for r in drive)
    counts = tuple(tuner.count_delta(zero, int(r["c_" + channel])) for r in drive)
    return times, counts


def target(run):
    return int(re.search(r"all(\d+)_", run).group(1))


@pytest.mark.parametrize("run, channel", sorted(REFERENCE))
def test_fit_on_the_bench_holds(run, channel):
    dead_ms, tau_ms, _ = REFERENCE[(run, channel)]

    fit = tuner.fit_dead_time_and_tau(*bench_trace(run, channel))

    assert fit.dead_time_s * 1000 == pytest.approx(dead_ms, abs=0.1)
    assert fit.tau_s * 1000 == pytest.approx(tau_ms, abs=0.1)
    assert fit.rms_ticks < 0.5, "the model should follow the rise to within a tick"


@pytest.mark.parametrize("run, channel", sorted(REFERENCE))
def test_tracking_on_the_bench_holds(run, channel):
    """535 ticks/s (0.3 m/s) tracks inside the ±5 % box on every motor; 178 ticks/s does not."""
    _, _, error_pct = REFERENCE[(run, channel)]

    speed = tuner.tracking_speed(*bench_trace(run, channel), BOX_START_S, HOLD_S)
    error = speed / target(run) - 1.0

    assert error * 100 == pytest.approx(error_pct, abs=0.01)
    assert (abs(error) <= BOX_TOLERANCE) == (target(run) == 535)
    if target(run) == 535:
        # At the box's speed the count agrees with the bench's mean reported speed.
        assert error * 100 == pytest.approx(BENCH_MEAN_535[channel], abs=0.2)


@pytest.mark.parametrize("run, channel", sorted(REFERENCE))
def test_report_on_the_bench_holds(run, channel, capsys):
    """The printed verdict, dead time and tau are the window's and the rise's."""
    dead_ms, tau_ms, error_pct = REFERENCE[(run, channel)]

    passed, out = report(trace_of(*bench_trace(run, channel), target(run)), HOLD_S, capsys)

    error = float(re.search(r"error=([+-][\d.]+)%", out).group(1))
    assert error == pytest.approx(error_pct, abs=0.06)
    dead, tau = measured([out])
    assert dead * 1000 == pytest.approx(dead_ms, abs=0.06)
    assert tau * 1000 == pytest.approx(tau_ms, abs=0.06)
    assert passed == (target(run) == 535)


# ---------------------------------------------------------------------------
# The step, on the fake bus
# ---------------------------------------------------------------------------


class Clock:
    """time.monotonic and time.sleep for the tuner: sleeping advances the time.

    A run past ten simulated minutes fails the test, however the tool handles exceptions, so a
    step that never ends cannot hang the suite.
    """

    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)
        if self.now > 100.0 + 600.0:
            pytest.fail("the tuner ran for ten simulated minutes")


class Motor:
    """A wheel behind its speed target: a first-order lag behind a dead time.

    *gain* scales the speed it settles at; 0 is a dead encoder or a jammed wheel, and a
    negative gain turns it against its target.
    """

    def __init__(self, dead=0.012, tau=0.022, gain=1.0):
        self.dead, self.tau, self.gain = dead, tau, gain
        self.targets = [(-math.inf, 0)]  # (time, ticks/s)
        self.driven = False  # set once command 37 has reached it
        self.t = None
        self.speed = 0.0
        self.count = 0.0

    def command(self, now, ticks_per_s):
        self.targets.append((now, ticks_per_s))

    def advance(self, now, step=0.001):
        if self.t is None:
            self.t = now
        while self.t < now:
            dt = min(step, now - self.t)
            when = self.t - self.dead
            goal = self.gain * next(v for s, v in reversed(self.targets) if s <= when)
            decay = math.exp(-dt / self.tau)
            self.count += goal * dt + (self.speed - goal) * self.tau * (1.0 - decay)
            self.speed = goal + (self.speed - goal) * decay
            self.t += dt


class Drifting(Motor):
    """A wheel turning at *drift* ticks/s whatever its target: from the start, or once
    command 37 has reached its controller."""

    def __init__(self, drift, from_command=False, **kw):
        super().__init__(**kw)
        self.drift = drift
        self.driven = not from_command

    def advance(self, now, step=0.001):
        if self.driven and self.t is not None and now > self.t:
            self.count += self.drift * (now - self.t)
        super().advance(now, step)


class SteppedRoboClaw(FakeRoboClaw):
    """test_port_fence's fake with a simulated motor on each channel.

    Command 37 and the duty commands set the motors' targets (the tuner sends only duty 0);
    commands 16 and 17 read their counts. Each packet takes WIRE_S. The serial timeout is
    armed at 0.5 s unless given, and as on the controller, no packet for that long stops both
    motors; here any packet, answered or not, restarts it.
    """

    WIRE_S = 0.0005

    def __init__(self, address, clock, m1, m2, serial_timeout=5):
        super().__init__(address, serial_timeout=serial_timeout)
        self.clock = clock
        self.motors = {1: m1, 2: m2}
        self.count_reads = 0
        self.log = []  # (time, command) of every packet
        self.last_packet = None

    def respond(self, packet):
        self.clock.now += self.WIRE_S
        now = self.clock.now
        timeout = self.serial_timeout / 10
        if timeout and self.last_packet is not None and now - self.last_packet > timeout:
            for motor in self.motors.values():
                motor.command(self.last_packet + timeout, 0)
        self.last_packet = now
        for motor in self.motors.values():
            motor.advance(now)
        valid = (
            len(packet) >= 4
            and packet[0] in self.addresses
            and _crc16(packet[:-2]) == struct.unpack(">H", packet[-2:])[0]
        )
        command = packet[1] if valid else None
        self.log.append((now, command))
        if command == DRIVE_M1M2_SPEED:
            return self.drive(now, *struct.unpack(">ii", packet[2:10]))
        if command in DUTY_FORWARD:
            assert packet[2] == 0, "the tuner's only duty command is the stop"
            return self.stop(now, 1 if command == 0 else 2)
        if command in READ_COUNTS:
            self.count_reads += 1
            return self.read_count(packet, 1 if command == 16 else 2)
        return super().respond(packet)

    def drive(self, now, m1, m2):
        for motor, ticks_per_s in ((self.motors[1], m1), (self.motors[2], m2)):
            motor.command(now, ticks_per_s)
            motor.driven = True
        return b"\xff"

    def stop(self, now, channel):
        self.motors[channel].command(now, 0)
        return b"\xff"

    def read_count(self, packet, channel):
        payload = struct.pack(">I", round(self.motors[channel].count) % 2**32) + b"\x00"
        return payload + struct.pack(">H", _crc16(packet[:2] + payload))


FRONT_PORT, REAR_PORT = "/dev/roboclaw2", "/dev/roboclaw1"


def stepped_robot(clock, front=None, rear=None, front_cls=SteppedRoboClaw,
                  rear_cls=SteppedRoboClaw, present=("front", "rear"), **claw_kw):
    """The robot's serial devices, each RoboClaw with two simulated motors."""
    devices = {CAMERA_PORT: FakeCamera()}
    for name, tty, address, motors, cls in (
        ("front", "/dev/ttyACM2", FRONT, front, front_cls),
        ("rear", "/dev/ttyACM1", REAR, rear, rear_cls),
    ):
        if name not in present:
            continue
        claw = cls(address, clock, *(motors or (Motor(), Motor())), **claw_kw)
        devices[tty] = claw
        devices["/dev/roboclaw" + tty.removeprefix("/dev/ttyACM")] = claw
    return FakeBus(devices)


def run_tuner(clock, *argv, module=None, confirmed=True):
    """Run a fresh copy of the tuner's main() on the simulated clock; return its exit status.

    Its signal handlers go to a mock, not to this process. A step is run with --no-confirm
    unless *confirmed* is False.
    """
    module = module or load_tuner()
    if confirmed and {"--tune", "--step-response"} & set(argv):
        argv = (*argv, "--no-confirm")
    with patch.object(sys, "argv", ["tune_pid.py", *argv]), patch.object(
        module, "time", clock
    ), patch.object(module, "signal", MagicMock()):
        try:
            return module.main()
        except KeyboardInterrupt:
            pytest.fail("a KeyboardInterrupt escaped main()")


def gains_written(bus):
    """(D, P, I, QPPS) of every gains write (commands 28 and 29), on any port."""
    return [struct.unpack(">IIII", data[2:18]) for _, data in bus.packets() if data[1] in SET_PID]


def commands(bus, port):
    return [data[1] for _, data in bus.packets(port)]


def reports(out):
    """{(controller, motor): its report's lines}, in the order printed."""
    found, controller, motor = {}, None, None
    for line in out.splitlines():
        if m := re.search(r"(Front|Rear) RoboClaw \(0x", line):
            controller, motor = m.group(1), None
        elif m := re.fullmatch(r"\s+(M[12]):", line):
            motor = m.group(1)
            found[(controller, motor)] = []
        elif motor is not None and line.startswith("    "):
            found[(controller, motor)].append(line.strip())
    return found


def measured(lines):
    """(dead time s, tau s) from a motor's report."""
    m = re.search(r"Dead time:\s+([\d.]+) ms.*tau: ([\d.]+) ms", " ".join(lines))
    return float(m.group(1)) / 1000, float(m.group(2)) / 1000


def passes(lines):
    return any(line.startswith("PASS -- tracks") for line in lines)


def aborted(lines, text):
    return any(line.startswith("ABORT:") and text in line for line in lines)


def faulted(lines, text):
    """A step that ran its hold, measured, and then failed on *text*."""
    return any(line.startswith("Tracking:") for line in lines) and any(
        line.startswith("FAIL --") and text in line for line in lines
    )


def no_verdict(lines):
    return any(line.startswith("NO VERDICT --") for line in lines)


HOLD = "6"  # the default hold, the only one the box judges
CYCLES = 300  # a 6 s hold at the driver's 50 Hz
PLANT_63 = 0.012 + SteppedRoboClaw.WIRE_S + 0.022  # dead + tau, plus the command on the wire


def test_tune_drives_with_37_reads_counts_and_reports_tau(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    rc = run_tuner(clock, "--tune", "--hold", HOLD)

    assert rc == 0
    for port in (FRONT_PORT, REAR_PORT):
        sent = commands(bus, port)
        assert CYCLES - 1 <= sent.count(DRIVE_M1M2_SPEED) <= CYCLES + 1
        assert not set(sent) & DRIVE_ONE_SPEED, "drives with command 37, as the driver does"
        assert not set(sent) & READ_SPEEDS, "measures the count, not the reported speed"
        assert READ_COUNTS <= set(sent)
        assert SET_PID <= set(sent), "--tune writes both motors' gains"
        drives = {
            struct.unpack(">ii", data[2:10]) for _, data in bus.packets(port) if data[1] == 37
        }
        assert drives == {(535, 535)}, "the default step is 0.3 m/s on both motors"
        for _, data in bus.packets(port):
            if data[1] in SET_PID:
                d, p, i, qpps = struct.unpack(">IIII", data[2:18])
                assert (p, i, d, qpps) == (
                    ROBOCLAW_PID_P,
                    ROBOCLAW_PID_I,
                    ROBOCLAW_PID_D,
                    ROBOCLAW_QPPS,
                )
        motor = [c for c in sent if c in MOTOR_COMMANDS]
        assert motor[-2:] == [0, 4], "the port ends in the duty-0 stop"

    out = capsys.readouterr().out
    found = reports(out)
    assert sorted(found) == [(c, m) for c in ("Front", "Rear") for m in ("M1", "M2")]
    for lines in found.values():
        assert passes(lines), lines
        dead, tau = measured(lines)
        # Up to one of the simulation's 1 ms steps on top of the plant and the wire.
        assert dead + tau == pytest.approx(PLANT_63, abs=0.0015)
        assert dead == pytest.approx(0.012, abs=0.003)
        assert tau == pytest.approx(0.022, rel=0.12)
    assert "still moving" not in out


def test_the_step_stops_in_the_cycle_its_hold_ends(patched, capsys):
    """The step's own stop, before the settle reads and the controller's close."""
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    for port in (FRONT_PORT, REAR_PORT):
        sent = commands(bus, port)
        last37 = len(sent) - 1 - sent[::-1].index(DRIVE_M1M2_SPEED)
        between = sent[last37 + 1:sent.index(0, last37)]
        assert set(between) <= READ_COUNTS and len(between) <= 2
    assert "still moving" not in capsys.readouterr().out


def test_tune_writes_the_constants_themselves(patched):
    """The written gains follow strafer_shared.constants, not literals that equal them today."""
    clock = Clock()
    bus = patched(stepped_robot(clock))
    with patch("strafer_shared.constants.ROBOCLAW_QPPS", 2001), patch(
        "strafer_shared.constants.ROBOCLAW_PID_P", 15001
    ), patch("strafer_shared.constants.ROBOCLAW_PID_I", 751), patch(
        "strafer_shared.constants.ROBOCLAW_PID_D", 1
    ):
        module = load_tuner()
    run_tuner(clock, "--tune", "--hold", HOLD, module=module)

    written = gains_written(bus)
    assert len(written) == 4
    assert all((p, i, d, qpps) == (15001, 751, 1, 2001) for d, p, i, qpps in written)


def test_set_writes_the_constants_qpps(patched):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    run_tuner(clock, "--set", "--p", str(ROBOCLAW_PID_P))

    written = gains_written(bus)
    assert written and all(qpps == ROBOCLAW_QPPS for _, _, _, qpps in written)
    assert (tuner.DEFAULT_P, tuner.DEFAULT_I, tuner.DEFAULT_D) == (
        ROBOCLAW_PID_P,
        ROBOCLAW_PID_I,
        ROBOCLAW_PID_D,
    )
    assert not hasattr(tuner, "TARGET_TAU_S")


@pytest.mark.parametrize("mode", ["--step-response", "--tune"])
@pytest.mark.parametrize("short", [("Front", "M1"), ("Rear", "M2")])
def test_a_motor_short_of_its_target_fails_tracking(patched, capsys, mode, short):
    clock = Clock()
    motors = {"Front": [Motor(), Motor()], "Rear": [Motor(), Motor()]}
    motors[short[0]][int(short[1][1]) - 1] = Motor(gain=0.9)
    patched(stepped_robot(clock, front=motors["Front"], rear=motors["Rear"]))

    assert run_tuner(clock, mode, "--hold", HOLD) == 1

    found = reports(capsys.readouterr().out)
    failed = re.search(r"FAIL -- tracks ([\d.]+)% under target", " ".join(found[short]))
    assert failed and float(failed.group(1)) == pytest.approx(10.0, abs=0.3)
    assert measured(found[short])[1] == pytest.approx(0.022, rel=0.12), "tau is still reported"
    assert all(passes(lines) for key, lines in found.items() if key != short)


@pytest.mark.parametrize("gain, rc", [(0.94, 1), (0.96, 0)])
def test_the_box_edge(patched, gain, rc):
    clock = Clock()
    patched(stepped_robot(clock, rear=(Motor(), Motor(gain=gain))))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == rc


def test_a_reverse_step_tracks(patched, capsys):
    """Measured, with no verdict: the box judges only the default step, forward."""
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--hold", HOLD, "--speed", "-535") == 3

    drives = [struct.unpack(">ii", data[2:10]) for _, data in bus.packets() if data[1] == 37]
    assert set(drives) == {(-535, -535)}
    out = capsys.readouterr().out
    for lines in reports(out).values():
        assert no_verdict(lines) and measured(lines)[1] == pytest.approx(0.022, rel=0.12)
        assert re.search(r"error=[+-]0\.0%", " ".join(lines))
    assert "measurements, not targets" in out


@pytest.mark.parametrize("mode", ["--step-response", "--tune"])
@pytest.mark.parametrize("speed", ["535", "-535"])
def test_a_reversed_motor_stops_the_run(patched, capsys, speed, mode):
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Motor(gain=-1.0), Motor())))

    assert run_tuner(clock, mode, "--hold", HOLD, "--speed", speed) == 1

    sent = commands(bus, FRONT_PORT)
    assert sent.count(DRIVE_M1M2_SPEED) < 10, "the step stops within a few cycles"
    first_stop = next(k for k, c in enumerate(sent) if c == 0 and k > sent.index(DRIVE_M1M2_SPEED))
    assert DRIVE_M1M2_SPEED not in sent[first_stop:], "nothing drives after the stop"
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT), "the run ends at the abort"
    out = capsys.readouterr().out
    found = reports(out)
    assert aborted(found[("Front", "M1")], "turns against its target")
    assert aborted(found[("Front", "M2")], "stopped with M1")
    assert "Not stepping the remaining controllers" in out


def test_a_runaway_motor_stops_the_run(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Motor(gain=3.0), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) < 25
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "ran away")


def test_a_count_that_never_moves_stops_the_run(patched, capsys):
    """A dead encoder: no abort fires on its speed, so its stillness does, 0.5 s in."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Motor(gain=0.0), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert 25 <= commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) <= 27  # stopped 0.5 s in
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "a dead encoder")


@pytest.mark.parametrize("gain, stalls", [(0.08, True), (0.13, False)])
def test_a_slow_wheel_stalls_under_a_tenth_of_its_target(patched, capsys, gain, stalls):
    clock = Clock()
    patched(stepped_robot(clock, front=(Motor(gain=gain), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "a dead encoder") == stalls


@pytest.mark.parametrize("run, channel", sorted(REFERENCE))
def test_the_stall_rule_passes_the_real_wheels(run, channel):
    trace = tuner.StepTrace(motor=channel, target=target(run))
    for t, c in zip(*bench_trace(run, channel)):
        trace.times.append(t)
        trace.counts.append(c)
        if t >= 0.5:
            assert tuner._stalled(trace, target(run)) is None, t


class Jamming(Motor):
    """A wheel that turns, then jams *after* s after its first command."""

    def __init__(self, after, **kw):
        super().__init__(**kw)
        self.after, self.jam_at = after, None

    def command(self, now, ticks_per_s):
        if self.jam_at is None and ticks_per_s:
            self.jam_at = now + self.after
        super().command(now, ticks_per_s)

    def advance(self, now, step=0.001):
        if self.jam_at is not None and now >= self.jam_at:
            self.t, self.speed = now, 0.0
            return
        super().advance(now, step)


def test_a_wheel_that_jams_mid_hold_stops_the_run(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Jamming(1.0), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert 63 <= commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) <= 69  # stopped 1.3 s in
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "a jammed wheel")


def test_the_motor_held_at_0_is_watched(patched, capsys):
    """With --motor m1, M2 is held at 0 by command 37, and still read and guarded."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Motor(), Drifting(400.0, from_command=True))))

    assert run_tuner(clock, "--step-response", "--controller", "front", "--motor", "m1",
                     "--hold", HOLD) == 1

    sent = commands(bus, FRONT_PORT)
    first, last = sent.index(DRIVE_M1M2_SPEED), len(sent) - sent[::-1].index(DRIVE_M1M2_SPEED)
    assert sent[first:last].count(17) >= sent.count(DRIVE_M1M2_SPEED) - 1, "M2 read every cycle"
    found = reports(capsys.readouterr().out)
    assert sorted(found) == [("Front", "M1")]
    assert aborted(found[("Front", "M1")], "stopped with M2: M2 ran away")


def test_a_turning_wheel_is_not_stepped(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front=(Drifting(25.0), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert DRIVE_M1M2_SPEED not in commands(bus, FRONT_PORT)
    found = reports(capsys.readouterr().out)
    assert aborted(found[("Front", "M1")], "not at rest")
    assert aborted(found[("Front", "M2")], "stopped with M1")


class BumpedClaw(SteppedRoboClaw):
    """M1's count steps 2 ticks at its 5th count read, mid-rest, as a settling wheel."""

    def read_count(self, packet, channel):
        if channel == 1:
            self.m1_reads = getattr(self, "m1_reads", 0) + 1
            if self.m1_reads == 5:
                self.motors[1].count += 2
        return super().read_count(packet, channel)


def test_the_zero_is_the_last_count_at_rest(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front_cls=BumpedClaw))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    for key, lines in reports(capsys.readouterr().out).items():
        dead, tau = measured(lines)
        assert dead + tau == pytest.approx(PLANT_63, abs=0.0015), key


class SlowWire(SteppedRoboClaw):
    WIRE_S = 0.004


def test_reads_are_timed_when_they_return(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front_cls=SlowWire, rear_cls=SlowWire))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    for key, lines in reports(capsys.readouterr().out).items():
        dead, tau = measured(lines)
        assert dead + tau == pytest.approx(0.012 + SlowWire.WIRE_S + 0.022, abs=0.0015), key


class DroppingClaw(SteppedRoboClaw):
    """Count reads go unanswered from the 40th on, about 10 cycles into the step."""

    def read_count(self, packet, channel):
        return b"" if self.count_reads >= 40 else super().read_count(packet, channel)


class Unacked37(SteppedRoboClaw):
    def drive(self, now, m1, m2):
        return b""


@pytest.mark.parametrize("cls, reason", [
    (DroppingClaw, "count reads failed"),
    (Unacked37, "command 37 unacknowledged"),
])
def test_a_link_that_stops_answering_stops_the_run(patched, capsys, cls, reason):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=cls))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) < 20
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT)
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], reason)


class IgnoresStops(SteppedRoboClaw):
    """Acknowledges the duty-0 stop but keeps driving."""

    def stop(self, now, channel):
        return b"\xff"


def test_a_wheel_still_turning_after_the_stop_fails(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=IgnoresStops))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    found = reports(capsys.readouterr().out)
    assert faulted(found[("Front", "M1")], "still moving after the stop")
    sent = commands(bus, FRONT_PORT)
    tail = sent[len(sent) - sent[::-1].index(DRIVE_M1M2_SPEED):]
    assert tail.count(0) == tail.count(4) == 3, "the step's stop, the stop again, the close"


@pytest.mark.parametrize("raw", [0, 11, 20])
def test_no_step_without_an_armed_serial_timeout(patched, capsys, raw):
    """0 disables the timeout; 1.1 and 2.0 s are past the range: neither controller is driven."""
    clock = Clock()
    bus = patched(stepped_robot(clock, serial_timeout=raw))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    for port in (FRONT_PORT, REAR_PORT):
        assert DRIVE_M1M2_SPEED not in commands(bus, port)
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "serial timeout")


@pytest.mark.parametrize("raw", [1, 10])
def test_the_serial_timeout_range_is_inclusive(patched, raw):
    clock = Clock()
    patched(stepped_robot(clock, serial_timeout=raw))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0


class NoTimeoutRead(SteppedRoboClaw):
    def respond(self, packet):
        return b"" if packet[1] == 15 else super().respond(packet)


def test_an_unread_serial_timeout_is_no_step(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=NoTimeoutRead))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert DRIVE_M1M2_SPEED not in [data[1] for _, data in bus.packets()]
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "serial timeout unread")


class StickyGains(SteppedRoboClaw):
    """Acknowledges commands 28 and 29 but keeps its old gains."""

    def respond(self, packet):
        kept = dict(self.gains)
        reply = super().respond(packet)
        self.gains = kept
        return reply


def test_tune_does_not_step_on_unconfirmed_gains(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=StickyGains))

    assert run_tuner(clock, "--tune", "--hold", HOLD) == 1

    for port in (FRONT_PORT, REAR_PORT):
        assert DRIVE_M1M2_SPEED not in commands(bus, port)
    assert "gains were not confirmed" in capsys.readouterr().out


class Interrupting(SteppedRoboClaw):
    """Ctrl-C arrives during a count read, about 10 cycles into the step."""

    def read_count(self, packet, channel):
        if self.count_reads == 40:
            raise KeyboardInterrupt
        return super().read_count(packet, channel)


def test_ctrl_c_stops_the_motors(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=Interrupting))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 130

    sent = commands(bus, FRONT_PORT)
    last37 = len(sent) - 1 - sent[::-1].index(DRIVE_M1M2_SPEED)
    assert {0, 4} <= set(sent[last37:]), "both motors are stopped after the interrupt"
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT)
    assert "Interrupted" in capsys.readouterr().out


def test_a_hangup_a_kill_or_a_ctrl_z_takes_the_ctrl_c_path(patched):
    module = load_tuner()
    clock = Clock()
    bus = patched(stepped_robot(clock))
    with patch.object(sys, "argv", ["tune_pid.py", "--read"]), patch.object(
        module, "signal"
    ) as sig:
        opened = []
        sig.signal.side_effect = lambda signum, handler: (
            opened.append(len(bus.opens)) if handler is module._interrupt else None
        )
        module.main()

    assert opened == [0, 0, 0], "installed before any port opens"
    handlers = {
        call.args[0]: call.args[1]
        for call in sig.signal.call_args_list
        if call.args[1] is module._interrupt
    }
    assert set(handlers) == {sig.SIGTERM, sig.SIGHUP, sig.SIGTSTP}
    for handler in handlers.values():
        with pytest.raises(KeyboardInterrupt):
            handler(15, None)


@pytest.mark.parametrize("present", [(), ("front",)])
def test_a_missing_controller_fails_and_steps_nothing(patched, present):
    clock = Clock()
    bus = patched(stepped_robot(clock, present=present))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert DRIVE_M1M2_SPEED not in [data[1] for _, data in bus.packets()]


@pytest.mark.parametrize(
    "speed", ["0", str(tuner.MAX_STEP_TICKS + 1), str(-tuner.MAX_STEP_TICKS - 1)]
)
def test_a_step_the_aborts_cannot_guard_is_refused(patched, speed):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    with pytest.raises(SystemExit) as exit_:
        run_tuner(clock, "--step-response", "--hold", HOLD, "--speed", speed)

    assert exit_.value.code == 2
    assert bus.packets() == []


class AckOnRetry(SteppedRoboClaw):
    """Leaves every other duty-0 stop unacknowledged (and unapplied)."""

    def stop(self, now, channel):
        self.stops = getattr(self, "stops", 0) + 1
        return super().stop(now, channel) if self.stops % 2 == 0 else b""


def test_each_stop_is_retried_until_acknowledged(patched):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=AckOnRetry))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    sent = commands(bus, FRONT_PORT)
    last37 = len(sent) - 1 - sent[::-1].index(DRIVE_M1M2_SPEED)
    first_stop = sent.index(0, last37)
    assert sent[first_stop:first_stop + 4] == [0, 0, 4, 4], "each motor's stop sent again"


class NeverAcksStops(SteppedRoboClaw):
    def stop(self, now, channel):
        return b""


def test_a_controller_that_does_not_acknowledge_the_stop_is_not_stepped(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=NeverAcksStops))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert DRIVE_M1M2_SPEED not in [data[1] for _, data in bus.packets()]
    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "stop went unacknowledged")


class StopsFailOnceDriven(SteppedRoboClaw):
    """Acknowledges stops until command 37 arrives, then neither applies nor acknowledges one."""

    def stop(self, now, channel):
        return b"" if self.motors[channel].driven else super().stop(now, channel)


def test_an_unacknowledged_final_stop_goes_silent_then_fails(patched, capsys):
    """No packet for 1.2 s, longer than any armed serial timeout, so the controller's timeout
    stops the wheels that ignored the stops; the run still fails on the unacknowledged stop."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=StopsFailOnceDriven))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    log = bus.devices[FRONT_PORT].log
    last37 = max(k for k, (_, command) in enumerate(log) if command == DRIVE_M1M2_SPEED)
    tries = [k for k in range(last37, len(log)) if log[k][1] in DUTY_FORWARD][:6]
    assert [log[k][1] for k in tries] == [0, 0, 0, 4, 4, 4], "each stop tried 3 times"
    # The next packet is the check's first read: the silence, then the check's own wait.
    quiet = log[tries[-1] + 1][0] - log[tries[-1]][0]
    assert tuner.SILENT_S + tuner.STILL_CHECK_S <= quiet < tuner.SILENT_S + tuner.STILL_CHECK_S + 0.01
    assert log[tries[-1] + 1][1] in READ_COUNTS
    out = capsys.readouterr().out
    assert "STOP NOT ACKNOWLEDGED: silent 1.2 s" in out
    found = reports(out)[("Front", "M1")]
    assert faulted(found, "stop went unacknowledged (silent 1.2 s)")
    assert "still moving" not in out, "the controller's timeout stopped them"


class IgnoresFirstStopOnceDriven(SteppedRoboClaw):
    """Acknowledges, but does not apply, the first stop after command 37."""

    def stop(self, now, channel):
        motor = self.motors[channel]
        if motor.driven and not getattr(motor, "ignored", False):
            motor.ignored = True
            return b"\xff"
        return super().stop(now, channel)


def test_a_wheel_the_second_stop_stops_still_fails(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front_cls=IgnoresFirstStopOnceDriven))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 1

    assert faulted(reports(capsys.readouterr().out)[("Front", "M1")], "still moving after the stop")


HELD = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP)


@pytest.fixture
def handlers_kept():
    """SIGTERM, SIGHUP and SIGTSTP take the Ctrl-C path, as main() sets them, and hard_stop
    leaves the handlers as it found them. The process's own handlers come back after the test."""
    before = {sig: signal.getsignal(sig) for sig in HELD}
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP):
        signal.signal(sig, tuner._interrupt)
    installed = {sig: signal.getsignal(sig) for sig in HELD}
    try:
        yield
        assert {sig: signal.getsignal(sig) for sig in HELD} == installed, "hard_stop restores them"
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)


def signalling_stops(*signals):
    """A controller whose M1 stop receives *signals*, from this process, as it is sent."""
    sent = []
    rc = MagicMock()

    def m1(duty):
        sent.append(1)
        for sig in signals:
            os.kill(os.getpid(), sig)
        return True

    rc.forward_m1.side_effect = m1
    rc.forward_m2.side_effect = lambda duty: sent.append(2) or True
    return rc, sent


@pytest.mark.parametrize("sig", HELD)
def test_hard_stop_holds_a_signal_until_both_stops_are_sent(handlers_kept, sig):
    """Python runs signal handlers in the main thread, so this holds with native threads too."""
    rc, sent = signalling_stops(sig)

    with pytest.raises(KeyboardInterrupt, match=f"signal {int(sig)}"):
        tuner.hard_stop(rc)

    assert sent == [1, 2]


def test_a_second_signal_abandons_the_stops(handlers_kept):
    """A write that hangs can still be interrupted: the controller's timeout stops the wheels."""
    rc, sent = signalling_stops(signal.SIGINT, signal.SIGINT)

    with pytest.raises(KeyboardInterrupt):
        tuner.hard_stop(rc)

    assert sent == [1]


def test_hard_stop_restores_the_handlers(handlers_kept):
    rc = MagicMock()
    rc.forward_m1.return_value = rc.forward_m2.return_value = True

    assert tuner.hard_stop(rc) is True


def test_hard_stop_off_the_main_thread_sends_both_stops(handlers_kept):
    rc = MagicMock()
    rc.forward_m1.return_value = rc.forward_m2.return_value = True
    result = []
    worker = threading.Thread(target=lambda: result.append(tuner.hard_stop(rc)))
    worker.start()
    worker.join()

    assert result == [True] and rc.forward_m2.called


def test_hard_stop_sends_m2_when_m1_fails(handlers_kept):
    rc = MagicMock()
    rc.forward_m1.side_effect = OSError("gone")
    rc.forward_m2.return_value = True

    assert tuner.hard_stop(rc) is False

    assert rc.forward_m1.call_count == 3 and rc.forward_m2.call_count == 1


def test_the_count_wraps_through_the_step(patched, capsys):
    clock = Clock()
    m1, m2 = Motor(), Motor()
    m1.count = m2.count = 2**32 - 60.0  # the 32-bit count wraps during the rise
    patched(stepped_robot(clock, front=(m1, m2)))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    for key, lines in reports(capsys.readouterr().out).items():
        dead, tau = measured(lines)
        assert dead + tau == pytest.approx(PLANT_63, abs=0.0015), key


class FlakyReads(SteppedRoboClaw):
    """Count reads go unanswered on every other cycle of the step, retries included."""

    def drive(self, now, m1, m2):
        self.flaky = not getattr(self, "flaky", True)
        return super().drive(now, m1, m2)

    def stop(self, now, channel):
        self.flaky = False
        return super().stop(now, channel)

    def read_count(self, packet, channel):
        return b"" if getattr(self, "flaky", False) else super().read_count(packet, channel)


def test_reads_failing_on_alternate_cycles_do_not_stop_the_step(patched, capsys):
    """Only MAX_MISSED failed cycles in a row stop it; one, then a good one, does not."""
    clock = Clock()
    patched(stepped_robot(clock, front_cls=FlakyReads))

    assert run_tuner(clock, "--step-response", "--hold", HOLD) == 0

    assert all(passes(lines) for lines in reports(capsys.readouterr().out).values())


class NoCounts(SteppedRoboClaw):
    def read_count(self, packet, channel):
        return b""


@pytest.mark.parametrize("mode", ["--step-response", "--tune"])
def test_an_error_ends_the_run(patched, capsys, mode):
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=NoCounts))

    assert run_tuner(clock, mode, "--hold", HOLD) == 1

    assert DRIVE_M1M2_SPEED not in [data[1] for _, data in bus.packets()]
    out = capsys.readouterr().out
    assert "ERROR:" in out and "Not stepping the remaining controllers" in out


@pytest.mark.parametrize("hold", ["0.5", "0.99", "31", "nan", "inf"])
def test_a_hold_out_of_range_is_refused(patched, hold):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    with pytest.raises(SystemExit) as exit_:
        run_tuner(clock, "--step-response", "--hold", hold)

    assert exit_.value.code == 2
    assert bus.packets() == []


def test_stop_fails_on_an_unacknowledged_stop(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front_cls=NeverAcksStops))

    assert run_tuner(clock, "--stop") == 1

    assert "FAILED, unacknowledged" in capsys.readouterr().out


@pytest.mark.parametrize("motor, payload", [("m1", (535, 0)), ("m2", (0, 535))])
def test_one_motor_is_stepped_and_the_other_held_at_0(patched, capsys, motor, payload):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--controller", "front", "--motor", motor,
                     "--hold", HOLD) == 0

    drives = {struct.unpack(">ii", data[2:10]) for _, data in bus.packets() if data[1] == 37}
    assert drives == {payload}
    found = reports(capsys.readouterr().out)
    assert sorted(found) == [("Front", motor.upper())] and passes(found[("Front", motor.upper())])


@pytest.mark.parametrize("hold", ["1", "30"])
def test_the_hold_range_is_inclusive(patched, hold):
    clock = Clock()
    patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--controller", "front", "--hold", hold) == 3


def test_a_stop_ignores_the_step_options(patched):
    """An emergency stop is never refused over the step options, which it does not use."""
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--stop", "--hold", "0.5", "--speed", "0", "--settle", "-1") == 0

    assert {0, 4} <= set(commands(bus, FRONT_PORT)) and {0, 4} <= set(commands(bus, REAR_PORT))


def test_stop_fails_when_a_controller_is_missing(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, present=("rear",)))

    assert run_tuner(clock, "--stop") == 1

    assert {0, 4} <= set(commands(bus, REAR_PORT)), "the one found is still stopped"


# ---------------------------------------------------------------------------
# The step's options, the constants behind them, and the box's verdict
# ---------------------------------------------------------------------------


def test_the_options_default_to_the_constants():
    """Each step option defaults to the constant that carries its reason, not to a literal."""
    with patch.object(tuner, "STEP_VELOCITY_TICKS", 534), patch.object(
        tuner, "STEP_HOLD_S", 5.5
    ), patch.object(tuner, "SETTLE_TIME_S", 0.3):
        args = tuner.build_parser().parse_args(["--tune"])

    assert (args.speed, args.hold, args.settle) == (534, 5.5, 0.3)


def test_the_silence_outlasts_every_timeout_a_step_accepts():
    """The bring-up's serial timeout is in the accepted range, and the silence after an
    unacknowledged stop is longer than any timeout in it."""
    low, high = tuner.SERIAL_TIMEOUT_RANGE_S

    assert low <= ROBOCLAW_SERIAL_TIMEOUT_S <= high < tuner.SILENT_S


def test_the_cycle_is_the_drivers():
    """The step's cycle is the driver's publish rate, which has no shared constant."""
    import yaml

    with open(os.path.join(os.path.dirname(__file__), "..", "config", "driver_params.yaml")) as f:
        rate = yaml.safe_load(f)["roboclaw_node"]["ros__parameters"]["publish_rate"]

    assert tuner.SAMPLE_PERIOD_S == pytest.approx(1.0 / rate)


@pytest.mark.parametrize("argv", [["--hold", "2"], ["--speed", "534"], ["--speed", "-535"]])
@pytest.mark.parametrize("mode", ["--step-response", "--tune"])
def test_only_the_default_step_gets_a_verdict(patched, capsys, mode, argv):
    """Off the default step the box does not judge: the tracking is printed, even for a wheel
    short of its target, and the run exits 3."""
    clock = Clock()
    patched(stepped_robot(clock, rear=(Motor(), Motor(gain=0.9))))

    assert run_tuner(clock, mode, *argv) == 3

    out = capsys.readouterr().out
    found = reports(out)
    assert all(no_verdict(lines) and not passes(lines) for lines in found.values())
    short = re.search(r"error=([+-][\d.]+)%", " ".join(found[("Rear", "M2")]))
    assert float(short.group(1)) == pytest.approx(-10.0, abs=0.3)
    assert "NO VERDICT -- the box is" in out and "has no verdict" in out


def test_a_step_off_the_default_still_fails_on_an_abort(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front=(Motor(gain=-1.0), Motor())))

    assert run_tuner(clock, "--step-response", "--hold", "2") == 1

    assert aborted(reports(capsys.readouterr().out)[("Front", "M1")], "turns against its target")


def test_the_box_judges_the_hold_the_gains_passed_on():
    """The integral term closes the last few percent over seconds: on the bench hold the
    front-left wheel misses the box over a 2 s hold and passes it over the 6 s one."""
    times, counts = bench_trace("all535_r1", "F1")
    error = {
        hold: tuner.tracking_speed(times, counts, tuner.TRACK_START_S, hold) / 535 - 1
        for hold in (2.0, tuner.STEP_HOLD_S)
    }

    assert error[2.0] < -tuner.TRACK_TOLERANCE < error[tuner.STEP_HOLD_S]


def test_a_dead_time_on_its_floor_is_flagged_not_recorded(capsys):
    """The bench's front-right wheel at 178 ticks/s: a fit on the 0 floor, whose split from tau
    is not measured."""
    _, out = report(trace_of(*bench_trace("all178_r1", "F2"), 178), HOLD_S, capsys)
    _, interior = report(trace_of(*bench_trace("all535_r1", "F1"), 535), HOLD_S, capsys)

    assert "(the fit's floor)" in out and "its split from tau is not measured" in out
    assert "Do not record these" in out
    assert "WARNING" not in interior


class SleepLog(Clock):
    """The simulated clock, keeping every sleep."""

    def __init__(self):
        super().__init__()
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        super().sleep(seconds)


def test_settle_is_the_wait_before_the_rest_check_not_the_still_check(patched):
    """--settle sets the wait after a stop, before each step and between controllers; the check
    that the wheels stopped keeps STILL_CHECK_S, twice per controller."""
    clock = SleepLog()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--hold", HOLD, "--settle", "0.7") == 0

    assert clock.sleeps.count(0.7) == 4
    assert clock.sleeps.count(tuner.STILL_CHECK_S) == 4
    log = bus.devices[FRONT_PORT].log
    first37 = next(k for k, (_, c) in enumerate(log) if c == DRIVE_M1M2_SPEED)
    stop = max(k for k in range(first37) if log[k][1] == 4)
    assert log[stop + 1][1] in READ_COUNTS and log[stop + 1][0] - log[stop][0] >= 0.7
    last37 = max(k for k, (_, c) in enumerate(log) if c == DRIVE_M1M2_SPEED)
    stop = next(k for k in range(last37, len(log)) if log[k][1] == 4)
    assert log[stop + 1][0] - log[stop][0] == pytest.approx(tuner.STILL_CHECK_S, abs=0.01)


@pytest.mark.parametrize("argv", [
    ["--settle", "-0.1"], ["--settle", "31"], ["--settle", "nan"], ["--settle", "inf"],
])
def test_an_option_out_of_range_is_refused(patched, argv):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    with pytest.raises(SystemExit) as exit_:
        run_tuner(clock, "--step-response", *argv)

    assert exit_.value.code == 2
    assert bus.packets() == []


@pytest.mark.parametrize("argv, rc", [
    (["--settle", "0"], 0), (["--settle", "30"], 0), (["--hold", "1"], 3), (["--hold", "30"], 3),
])
def test_an_option_at_its_bound_is_accepted(patched, argv, rc):
    """The settle is no part of the box: the default step is still judged."""
    clock = Clock()
    patched(stepped_robot(clock))

    assert run_tuner(clock, "--step-response", "--controller", "front", *argv) == rc


def test_the_controllers_are_opened_as_the_driver_opens_them(patched):
    """The interface's own read timeout and retries, as roboclaw_node opens it."""
    clock = Clock()
    patched(stepped_robot(clock))
    module = load_tuner()
    real, made = module.RoboClawInterface, []
    signature = inspect.signature(real)

    def spy(*args, **kwargs):
        made.append(signature.bind(*args, **kwargs).arguments)
        return real(*args, **kwargs)

    with patch.object(module, "RoboClawInterface", spy):
        assert run_tuner(clock, "--step-response", "--hold", HOLD, module=module) == 0

    assert made and not any({"read_timeout", "max_retries"} & set(bound) for bound in made)


def test_the_help_gives_each_step_option_its_default_and_range():
    """--help renders (its %% escapes hold), with each default's reason and each bound."""
    text = " ".join(tuner.build_parser().format_help().split())

    for part in (
        "default: 535, 0.3 m/s at the wheel", "at most 1045 either way", "10 % under full speed",
        "default: 6, the hold the driver's gains were chosen on", "from 1,", "to 30.",
        "default: 0.2: a lifted wheel stopped at duty 0", "Another hold gets no verdict",
        "about 1.8 m, and up to 17.6 m", "armed (0.1 to 1 s)", "±5 % of its target",
        "3 if the step was not the default one", "130 if interrupted (Ctrl-C, Ctrl-Z",
    ):
        assert part in text, part


@pytest.mark.parametrize("mode", ["--tune", "--set"])
@pytest.mark.parametrize("qpps", ["0", "-1"])
def test_a_qpps_that_is_not_positive_is_refused(patched, mode, qpps):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    with pytest.raises(SystemExit) as exit_:
        run_tuner(clock, mode, "--p", "15000", "--qpps", qpps)

    assert exit_.value.code == 2
    assert bus.packets() == []


def test_a_gain_of_0_is_written_as_0(patched):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--tune", "--controller", "front", "--p", "0") == 0

    assert [p for _, p, _, _ in gains_written(bus)] == [0, 0]


def test_the_step_is_shown_as_a_share_of_the_qpps_read_back(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock, present=("front",)))
    bus.devices[FRONT_PORT].gains = {1: (15000, 750, 0, 1070), 2: (15000, 750, 0, 1070)}

    assert run_tuner(clock, "--step-response", "--controller", "front") == 0

    assert "Step target: 535 ticks/s (50% of QPPS 1070)" in capsys.readouterr().out


def test_an_abort_prints_its_counts_with_verbose(patched, capsys):
    clock = Clock()
    patched(stepped_robot(clock, front=(Jamming(1.0), Motor())))

    assert run_tuner(clock, "--step-response", "-v") == 1

    lines = reports(capsys.readouterr().out)[("Front", "M1")]
    assert aborted(lines, "a jammed wheel") and "Time(ms) Count" in " ".join(" ".join(lines).split())


# ---------------------------------------------------------------------------
# A paused step, and the typed confirmation that the wheels are lifted
# ---------------------------------------------------------------------------


class Pausing(SteppedRoboClaw):
    """The host stalls for *pause* s while the 100th command 37 is sent, 2 s into the step."""

    pause = 1.0

    def drive(self, now, m1, m2):
        self.drives = getattr(self, "drives", 0) + 1
        if self.drives == 100:
            self.clock.now += self.pause
        return super().drive(now, m1, m2)


def test_a_pause_past_the_serial_timeout_ends_the_step(patched, capsys):
    """The controller's timeout stopped the wheels during the pause; the step is not resumed."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=Pausing))

    assert run_tuner(clock, "--step-response") == 1

    assert commands(bus, FRONT_PORT).count(DRIVE_M1M2_SPEED) == 100
    assert DRIVE_M1M2_SPEED not in commands(bus, REAR_PORT)
    out = capsys.readouterr().out
    assert aborted(reports(out)[("Front", "M1")], "the step is not resumed")
    assert "still moving" not in out


class ShortPause(Pausing):
    pause = 0.3


def test_a_late_cycle_does_not_replay_the_missed_ones(patched):
    """Under the serial timeout the step goes on, one cycle at a time, not a burst of them."""
    clock = Clock()
    bus = patched(stepped_robot(clock, front_cls=ShortPause))

    assert run_tuner(clock, "--step-response") == 0

    sent = [t for t, c in bus.devices[FRONT_PORT].log if c == DRIVE_M1M2_SPEED]
    assert min(b - a for a, b in zip(sent, sent[1:])) >= tuner.SAMPLE_PERIOD_S - 0.002
    missed = round(ShortPause.pause / tuner.SAMPLE_PERIOD_S)
    assert abs(len(sent) - (CYCLES - missed)) <= 1, "the cycles the pause took are not run"


def test_no_step_without_a_terminal_to_confirm_it(patched, capsys):
    clock = Clock()
    bus = patched(stepped_robot(clock))

    assert run_tuner(clock, "--tune", confirmed=False) == 1

    assert bus.packets() == []
    assert "about 1.80 m of travel" in capsys.readouterr().out


@pytest.mark.parametrize("answer, rc", [("yes", 0), ("y", 1), (EOFError, 1)])
def test_only_a_typed_yes_steps(patched, answer, rc):
    clock = Clock()
    bus = patched(stepped_robot(clock))
    module = load_tuner()
    reply = {"side_effect": answer} if answer is EOFError else {"return_value": answer}

    with patch.object(module, "sys") as fake_sys, patch.object(module, "termios"), patch(
        "builtins.input", **reply
    ):
        fake_sys.stdin.isatty.return_value = True
        assert run_tuner(clock, "--step-response", module=module, confirmed=False) == rc

    assert (DRIVE_M1M2_SPEED in [data[1] for _, data in bus.packets()]) == (rc == 0)
