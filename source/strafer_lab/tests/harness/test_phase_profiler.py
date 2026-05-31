"""Unit tests for the teleop-loop phase profiler.

Pure stdlib — runs in ``.venv_harness`` with no Isaac Sim. Timings are
exercised with monkeypatched clocks so assertions are deterministic
rather than wall-clock-dependent.
"""

from __future__ import annotations

import pytest

from strafer_lab.tools.phase_profiler import PhaseProfiler


class _FakeClock:
    """Deterministic perf_counter / perf_counter_ns stand-in."""

    def __init__(self) -> None:
        self.ns = 0

    def advance_ms(self, ms: float) -> None:
        self.ns += int(ms * 1e6)

    def perf_counter_ns(self) -> int:
        return self.ns

    def perf_counter(self) -> float:
        return self.ns / 1e9


@pytest.fixture
def clock(monkeypatch):
    c = _FakeClock()
    monkeypatch.setattr("strafer_lab.tools.phase_profiler.time.perf_counter_ns",
                        c.perf_counter_ns)
    monkeypatch.setattr("strafer_lab.tools.phase_profiler.time.perf_counter",
                        c.perf_counter)
    return c


class TestDisabledIsNoOp:
    def test_disabled_phase_does_not_raise_or_record(self):
        p = PhaseProfiler(enabled=False)
        with p.phase("env_step"):
            pass
        p.add_render(5_000_000)
        p.tick()
        # Disabled profiler never reports.
        assert p.maybe_report() is None

    def test_disabled_phase_context_yields(self):
        p = PhaseProfiler(enabled=False)
        ran = False
        with p.phase("driver"):
            ran = True
        assert ran


class TestPhaseAccounting:
    def test_phase_times_sum_into_window(self, clock):
        p = PhaseProfiler(enabled=True, report_period_s=1.0)
        # One tick: 2ms driver + 40ms env_step + 5ms writer + 3ms overhead.
        with p.phase("driver"):
            clock.advance_ms(2)
        with p.phase("env_step"):
            clock.advance_ms(40)
        with p.phase("writer"):
            clock.advance_ms(5)
        with p.phase("overhead"):
            clock.advance_ms(3)
        p.tick()
        line = p.format_window()
        assert "1 ticks" in line
        assert "total=50.0ms" in line
        assert "driver=2.0" in line
        assert "writer=5.0" in line
        assert "overhead=3.0" in line
        # ~20 fps at 50ms/tick.
        assert "~20.0 fps" in line

    def test_render_is_subset_of_env_step(self, clock):
        p = PhaseProfiler(enabled=True)
        with p.phase("env_step"):
            clock.advance_ms(40)
        # Of the 40ms env_step, 30ms was render.
        p.add_render(30_000_000)
        p.tick()
        line = p.format_window()
        assert "render=30.0" in line
        assert "sim+mgr=10.0" in line
        assert "render_calls/tick=1.00" in line

    def test_means_average_over_ticks(self, clock):
        p = PhaseProfiler(enabled=True)
        for _ in range(4):
            with p.phase("env_step"):
                clock.advance_ms(20)
            p.tick()
        line = p.format_window()
        assert "4 ticks" in line
        # mean env_step = 20ms; total = 20ms; ~50 fps.
        assert "env_step=20.0" in line
        assert "~50.0 fps" in line


class TestReportCadence:
    def test_no_report_before_period(self, clock):
        p = PhaseProfiler(enabled=True, report_period_s=2.0)
        with p.phase("env_step"):
            clock.advance_ms(40)
        p.tick()
        # Only 40ms elapsed; period is 2s.
        assert p.maybe_report() is None

    def test_reports_after_period_then_resets(self, clock):
        p = PhaseProfiler(enabled=True, report_period_s=0.1)
        with p.phase("env_step"):
            clock.advance_ms(40)
        p.tick()
        clock.advance_ms(100)  # cross the 0.1s report period
        first = p.maybe_report()
        assert first is not None
        assert "1 ticks" in first
        # Window reset: a fresh report needs new ticks + elapsed time.
        assert p.maybe_report() is None
        with p.phase("env_step"):
            clock.advance_ms(10)
        p.tick()
        clock.advance_ms(100)
        second = p.maybe_report()
        assert second is not None
        assert "1 ticks" in second
        assert "env_step=10.0" in second

    def test_zero_ticks_never_reports(self, clock):
        p = PhaseProfiler(enabled=True, report_period_s=0.0)
        clock.advance_ms(100)
        assert p.maybe_report() is None
