"""Tests for strafer_lab.bridge.cmd_watchdog — the pure stop-on-silence gate.

Runs in the pxr-free autonomy suite via the ``strafer_lab`` namespace stub
installed by :mod:`conftest` (no rclpy / warp / Isaac Sim). The watchdog owns
the sim-time staleness accumulator; the message-side wiring in
:mod:`strafer_lab.bridge.async_publisher` (``get_cmd_vel`` returning zeros,
the ROS warn line) is smoke-tested in-process by ``run_sim_in_the_loop.py``
inside the Isaac Lab env.

The window is measured in **sim seconds**: under ``use_sim_time`` a low
real-time factor spaces healthy commands out in wall-time, so a wall-clock
window would false-trip between them. All ``sim_time_s`` values below use
0.25-multiples so the accumulation is exact in binary floating point.
"""

from __future__ import annotations

import pytest

from strafer_lab.bridge.cmd_watchdog import CmdVelWatchdog


class TestFreshCommandResetsAccumulator:
    def test_new_seq_zeroes_the_age(self):
        w = CmdVelWatchdog(0.5)
        w.observe(1, 0.0)  # anchor
        w.observe(1, 0.4)  # silence: age 0.4
        assert not w.stale()
        w.observe(2, 0.5)  # fresh command → age reset
        assert not w.stale()
        w.observe(2, 0.9)  # silence again from 0 → age 0.4
        assert not w.stale()

    def test_first_observation_does_not_accumulate(self):
        # The anchoring call establishes seq + time; no elapsed delta yet.
        w = CmdVelWatchdog(0.5)
        w.observe(0, 100.0)
        assert not w.stale()


class TestSilenceTripsAtWindow:
    def test_trips_only_past_the_window(self):
        w = CmdVelWatchdog(0.5)
        w.observe(1, 0.0)
        w.observe(1, 0.25)  # age 0.25
        assert not w.stale()
        w.observe(1, 0.50)  # age 0.50 == window → strict > means not yet
        assert not w.stale()
        w.observe(1, 0.75)  # age 0.75 > window
        assert w.stale()


class TestTripLogsOncePerSilence:
    def test_take_trip_log_latches_and_rearms(self):
        w = CmdVelWatchdog(0.5)
        w.observe(1, 0.0)
        w.observe(1, 1.0)  # age 1.0 > window → stale
        assert w.stale()
        assert w.take_trip_log() == pytest.approx(1.0)  # logged once
        assert w.take_trip_log() is None  # not again while still silent
        w.observe(2, 1.1)  # fresh command → un-trips + re-arms the log
        assert not w.stale()
        assert w.take_trip_log() is None
        w.observe(2, 1.8)  # silence again: age 0.7 > window
        assert w.stale()
        assert w.take_trip_log() == pytest.approx(0.7)  # logs again next silence


class TestCommandResumptionUntrips:
    def test_resume_clears_stale(self):
        w = CmdVelWatchdog(0.5)
        w.observe(1, 0.0)
        w.observe(1, 1.0)  # stale
        assert w.stale()
        w.observe(2, 1.25)  # new command
        assert not w.stale()


class TestDisabled:
    def test_zero_window_never_trips(self):
        w = CmdVelWatchdog(0.0)
        assert not w.enabled
        w.observe(1, 0.0)
        w.observe(1, 100.0)  # arbitrarily long silence
        assert not w.stale()
        assert w.take_trip_log() is None


class TestStartupNoCommandEverReceived:
    def test_constant_seq_still_ages_and_trips(self):
        # No /cmd_vel ever arrives → the sequence counter never changes. The
        # held command is the publisher's initial zero, so the applied action
        # is zeros both before and after the trip; the watchdog just makes the
        # zeroing explicit once the window elapses.
        w = CmdVelWatchdog(0.5)
        w.observe(0, 0.0)  # anchor on the initial seq
        w.observe(0, 0.25)  # age 0.25 → still within window
        assert not w.stale()
        w.observe(0, 0.75)  # age 0.75 > window
        assert w.stale()
