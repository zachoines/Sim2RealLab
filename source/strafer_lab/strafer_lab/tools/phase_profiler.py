"""Rolling per-phase wall-time profiler for the teleop env-step loop.

The teleop capture loop serializes several phases on one thread —
gamepad / pose read, ``env.step`` (which itself folds physics + the Kit
render pump + the manager loop), the LeRobot writer, and the
viewport / HUD / PIP overhead. When felt FPS is low, the operator needs
to know *which* phase dominates before choosing a remedy: a render-bound
loop, a manager-bound loop, and an ``env.step``-serialization-bound loop
each call for a different fix.

:class:`PhaseProfiler` accumulates nanosecond timings per named phase
(plus a render-call count and a separate render-time sub-total) over a
rolling window and formats a mean-ms-per-tick breakdown every
``report_period_s``. It is pure stdlib so it imports without the Isaac
Sim runtime and is unit-testable; the driver wires it in under a
``--profile`` flag and leaves it a no-op otherwise.

The render sub-total is tracked separately from the phase totals because
the render happens *inside* ``env.step``: it is reported as a breakdown
of the ``env_step`` phase (``render`` vs ``sim+mgr``), not as an
additional phase, so the per-phase numbers still sum to the loop total.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator


class PhaseProfiler:
    """Accumulate per-phase wall-time over a rolling window and report it.

    Enable with ``PhaseProfiler(enabled=True)``; when disabled every method
    is a cheap no-op so the caller can wire it in unconditionally. Time a
    phase with the :meth:`phase` context manager, call :meth:`tick` once per
    loop iteration, and call :meth:`maybe_report` each iteration to get a
    formatted breakdown string back at most once per ``report_period_s``
    (``None`` otherwise).
    """

    def __init__(self, enabled: bool, report_period_s: float = 2.0) -> None:
        self.enabled = bool(enabled)
        self.report_period_s = float(report_period_s)
        self._reset_window()
        self._last_report = time.perf_counter()

    def _reset_window(self) -> None:
        self._phase_ns: dict[str, int] = {}
        self._render_ns: int = 0
        self._render_calls: int = 0
        self._physics_ns: int = 0
        self._physics_calls: int = 0
        self._ticks: int = 0

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time the wrapped block and add it to ``name``'s window total."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self._phase_ns[name] = (
                self._phase_ns.get(name, 0) + (time.perf_counter_ns() - t0)
            )

    def add_render(self, elapsed_ns: int) -> None:
        """Record one render call and its wall-time (a subset of env_step)."""
        if not self.enabled:
            return
        self._render_ns += int(elapsed_ns)
        self._render_calls += 1

    def add_physics(self, elapsed_ns: int) -> None:
        """Record one PhysX step call and its wall-time (a subset of env_step).

        Called once per decimation substep; the manager-loop remainder is
        ``env_step - render - physics``, which isolates the Python-side
        manager dispatch (rewards / terminations / events / curriculum) from
        the C++ PhysX cost.
        """
        if not self.enabled:
            return
        self._physics_ns += int(elapsed_ns)
        self._physics_calls += 1

    def tick(self) -> None:
        """Mark one completed loop iteration."""
        if self.enabled:
            self._ticks += 1

    def maybe_report(self) -> str | None:
        """Return a breakdown line if ``report_period_s`` elapsed, else None.

        Resets the window after emitting so each line is the mean over the
        ticks since the previous report.
        """
        if not self.enabled or self._ticks == 0:
            return None
        now = time.perf_counter()
        if now - self._last_report < self.report_period_s:
            return None
        line = self.format_window()
        self._reset_window()
        self._last_report = now
        return line

    def format_window(self) -> str:
        """Format the current window as a mean-ms-per-tick breakdown."""
        ticks = self._ticks
        if ticks == 0:
            return "[profile] no ticks recorded"
        per = 1.0 / 1e6 / ticks  # ns-sum -> mean ms/tick

        def ms(name: str) -> float:
            return self._phase_ns.get(name, 0) * per

        total_ms = sum(self._phase_ns.values()) * per
        render_ms = self._render_ns * per
        physics_ms = self._physics_ns * per
        env_ms = ms("env_step")
        # Manager-loop remainder once render + physics are carved out of
        # env.step. Clamped at 0 in case sub-timer wrappers were not installed.
        mgr_ms = max(0.0, env_ms - render_ms - physics_ms)
        fps = 1000.0 / total_ms if total_ms > 0 else float("inf")
        rc = self._render_calls / ticks
        pc = self._physics_calls / ticks
        ordered = ("driver", "env_step", "writer", "overhead")
        parts: list[str] = []
        for name in ordered:
            if name == "env_step":
                parts.append(
                    f"env_step={env_ms:.1f} "
                    f"[render={render_ms:.1f} physics={physics_ms:.1f} "
                    f"mgr={mgr_ms:.1f}]"
                )
            else:
                parts.append(f"{name}={ms(name):.1f}")
        # Any phases the caller named outside the canonical order.
        for name in sorted(set(self._phase_ns) - set(ordered)):
            parts.append(f"{name}={ms(name):.1f}")
        return (
            f"[profile] {ticks} ticks  total={total_ms:.1f}ms (~{fps:.1f} fps)  "
            + "  ".join(parts)
            + f"  render_calls/tick={rc:.2f}  physics_steps/tick={pc:.2f}"
        )
