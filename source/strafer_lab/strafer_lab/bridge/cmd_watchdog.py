"""Sim-time stop-on-silence watchdog for the bridge ``/cmd_vel`` stream.

Mirrors the real RoboClaw driver's command watchdog — motors are zeroed
after ``WATCHDOG_TIMEOUT_SEC = 0.5`` s of silence
(``strafer_driver.roboclaw_node``) — but measures the window in **sim
seconds**, not wall-clock. Under ``use_sim_time`` the inference node ticks
on the sim clock, so at a low real-time factor healthy policy commands
arrive only every few hundred wall-milliseconds; a wall-clock window would
false-trip *between* healthy commands mid-mission. Sim-time is the domain
the command stream lives in and is real-time-factor-independent.

Pure — no ``rclpy`` / ``warp`` / Isaac Sim imports — so it unit-tests in the
pxr-free autonomy suite. :class:`~strafer_lab.bridge.async_publisher.StraferAsyncPublisher`
increments a command sequence counter on every ``/cmd_vel`` and feeds
``(cmd_seq, sim_time_s)`` to :meth:`CmdVelWatchdog.observe` each
``publish_state``; :meth:`CmdVelWatchdog.stale` then reports whether the held
command should be zeroed.
"""

from __future__ import annotations


class CmdVelWatchdog:
    """Accumulate ``/cmd_vel`` silence in sim time and trip past a window.

    Single-threaded by contract: the publisher calls :meth:`observe` and
    :meth:`stale` / :meth:`take_trip_log` from the bridge's main step thread.
    The command sequence counter it consumes is the only value shared with
    the rclpy spinner thread, and the publisher guards that with its own
    lock before handing the snapshot here.
    """

    def __init__(self, window_sim_s: float) -> None:
        self._window = float(window_sim_s)
        self._last_seq: int | None = None
        self._last_sim_time_s: float | None = None
        self._age_sim_s = 0.0
        # Once-per-silence latch so the caller logs a single "stopping" line
        # per silence episode rather than every step it stays tripped.
        self._logged = False

    @property
    def enabled(self) -> bool:
        """A window of ``0`` (or negative) disables the watchdog entirely."""
        return self._window > 0.0

    def observe(self, cmd_seq: int, sim_time_s: float) -> None:
        """Advance the staleness accumulator for one bridge step.

        ``cmd_seq`` increments whenever a fresh ``/cmd_vel`` lands; a change
        since the previous call means the stream is alive, so the age resets
        to zero. An unchanged counter accumulates the sim-time elapsed since
        the previous observation.
        """
        if not self.enabled:
            return
        if self._last_seq is None:
            # First observation: anchor seq + time, no elapsed delta yet.
            self._last_seq = cmd_seq
            self._last_sim_time_s = sim_time_s
            return
        if cmd_seq != self._last_seq:
            self._age_sim_s = 0.0
            self._last_seq = cmd_seq
            self._logged = False
        elif self._last_sim_time_s is not None:
            self._age_sim_s += sim_time_s - self._last_sim_time_s
        self._last_sim_time_s = sim_time_s

    def stale(self) -> bool:
        """True once the stream has been silent longer than the window."""
        return self.enabled and self._age_sim_s > self._window

    def take_trip_log(self) -> float | None:
        """Return the silence age to log once per silence episode, else ``None``.

        Latches on the first stale step after a fresh command and re-arms when
        :meth:`observe` next sees a new command. Lets the caller emit a single
        "No cmd_vel ..." warning per silence, mirroring the RoboClaw driver.
        """
        if self.stale() and not self._logged:
            self._logged = True
            return self._age_sim_s
        return None
