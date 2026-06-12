"""Sim-in-the-loop orchestration loop.

Pure-Python harness with all sim / ROS access abstracted behind adapter
protocols passed in by the caller. The launch script that owns the live
Isaac Sim env and the ROS context is responsible for constructing the
adapters; this module never imports ``omni`` or ``rclpy``, which keeps
the orchestration logic unit-testable against plain fakes.

Per-mission lifecycle (one mission = one recorded episode)::

    for spec, meta in missions:
        env.reset()
        recorder.begin_episode(spec, meta, start_bundle)
        recorder.add_frame(start_bundle)
        mission_id = submit(meta.dispatch_command)
        while not terminal:
            env.step()                       # bridge applies /cmd_vel
            every Nth step: recorder.add_frame(env.capture())
            poll status; check abort signal
        recorder.add_frame(final_bundle)
        recorder.end_episode(status=final_status)        # kept
        # — or, on abort signal / executor loss / crash —
        recorder.end_episode(status=..., discard=True)   # never reaches disk

Discard paths (the episode is closed with ``discard=True`` and the
recorder guarantees nothing reaches disk):

- the injected ``abort_signal`` reports a reason (e.g. /cmd_vel silence
  beyond the operator's grace window),
- ``status()`` fails ``max_consecutive_status_failures`` times in a row
  (executor crashed or became unreachable mid-mission),
- any exception inside the mission loop (re-raised after the discard).

Pure Python — testable by passing fake env / mission / recorder objects.
No Isaac Sim or rclpy imports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

from strafer_lab.sim_in_the_loop.mission import MissionSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome / status records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionStatus:
    """Snapshot of the executor's mission state, sampled by the harness."""

    terminal: bool
    state: str
    error_code: str = ""
    elapsed_s: float = 0.0
    message: str = ""

    @property
    def succeeded(self) -> bool:
        return self.terminal and self.state.lower() in {"succeeded", "success"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "error_code": self.error_code,
            "elapsed_s": float(self.elapsed_s),
            "message": self.message,
        }


@dataclass(frozen=True)
class EpisodeOutcome:
    """Per-mission summary, returned by ``run_one_mission``."""

    mission_id: str
    reachability: bool
    final_status: MissionStatus
    frames_written: int
    elapsed_s: float
    discarded: bool = False


# ---------------------------------------------------------------------------
# Frame bundle returned by the env adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameBundle:
    """Everything the recorder needs from one env step.

    The harness does not interpret the array contents, only forwards
    them. The runtime env adapter pulls these from the live camera /
    articulation handles and from the /cmd_vel subscription:

    - ``action`` is the normalized ``[-1, 1]`` 3-vector most recently
      written into the env's action tensor (the same contract the
      mecanum action term consumes and the teleop driver records).
    - ``achieved_vel`` is ``(vx, vy, omega_z)`` from sim body state.
    - ``detections`` is the perception camera's parsed bbox list for
      this frame (``None`` when detections capture is disabled).
    """

    rgb: Any  # np.ndarray (H, W, 3) uint8, or None when not captured
    depth: Any | None  # np.ndarray (H, W) float32 meters, or None
    robot_pos: tuple[float, float, float]
    robot_quat: tuple[float, float, float, float]  # (qx, qy, qz, qw)
    achieved_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    action: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sim_time_s: float = 0.0
    depth_policy: Any | None = None  # np.ndarray (h, w) float32, or None
    rgb_policy: Any | None = None  # np.ndarray (h, w, 3) uint8, or None
    detections: Sequence[Any] | None = None  # DetectedBbox records, or None


# ---------------------------------------------------------------------------
# Per-mission capture metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeMeta:
    """What to record vs. what to dispatch for one mission.

    Splitting the two is what makes hard-negative injection possible:
    ``mission_text`` (recorded as the episode task) keeps naming the
    original target while ``dispatch_command`` may steer the executor
    toward a deliberately-wrong object. For ordinary missions the two
    agree and the injection fields are ``None``.
    """

    mission_text: str
    dispatch_command: str
    source_mission_source: str
    target_label: str | None = None
    target_object_id: str | None = None
    target_position_3d: tuple[float, float, float] | None = None
    injection_mode: str | None = None
    injection_mode_actual: str | None = None
    original_target_position_3d: tuple[float, float, float] | None = None
    paraphrases: tuple[str, ...] = ()
    generator_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: MissionSpec, *, source_mission_source: str = "scene-metadata") -> "EpisodeMeta":
        return cls(
            mission_text=spec.raw_command,
            dispatch_command=spec.raw_command,
            source_mission_source=source_mission_source,
            target_label=spec.target_label,
            target_object_id=str(spec.target_instance_id),
            target_position_3d=spec.target_position_3d,
        )


# ---------------------------------------------------------------------------
# Injected interfaces
# ---------------------------------------------------------------------------


class EnvAdapter(Protocol):
    def reset(self, *, scene_name: str) -> None:
        """Reset the env and load ``scene_name`` if scene-switching is supported."""

    def step(self) -> None:
        """Advance the simulator by one step (the bridge applies /cmd_vel)."""

    def capture(self) -> FrameBundle:
        """Pull sensors + state + action for the current frame."""


class MissionApi(Protocol):
    def submit(self, *, raw_command: str, request_id: str) -> str:
        """Submit a mission, return the mission_id assigned by the executor."""

    def status(self) -> MissionStatus:
        """Snapshot the current mission status."""

    def cancel(self) -> None:
        """Cancel the active mission."""


class EpisodeRecorder(Protocol):
    """Narrow recording contract the harness drives.

    Implementations own the on-disk format; the bridge driver's
    LeRobot-backed implementation lives in
    :mod:`strafer_lab.sim_in_the_loop.lerobot_recorder`.
    """

    def begin_episode(
        self, *, spec: MissionSpec, meta: EpisodeMeta, start_bundle: FrameBundle,
    ) -> None: ...

    def add_frame(self, bundle: FrameBundle) -> None: ...

    def end_episode(
        self, *, status: MissionStatus | None, discard: bool = False,
    ) -> None: ...


class AbortSignal(Protocol):
    """Mid-episode discard trigger, polled once per env step.

    ``arm()`` is called right after a mission is accepted by the
    executor, so implementations can scope their window to the active
    mission (e.g. ignore /cmd_vel traffic from the previous one).
    """

    def arm(self) -> None: ...

    def check(self) -> str | None: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessConfig:
    """Per-run knobs for the sim-in-the-loop harness."""

    # Stop polling and close the episode after this many seconds.
    mission_timeout_s: float = 60.0
    # How often (in env steps) to capture a frame during navigation.
    capture_every_n_steps: int = 5
    # Hard cap on env steps per mission to prevent runaway loops.
    max_steps_per_mission: int = 2000
    # Sleep between status polls in seconds (real-time, not sim time).
    status_poll_period_s: float = 0.2
    # Consecutive status() failures (no response / exception) before the
    # executor is declared unreachable and the episode discarded.
    max_consecutive_status_failures: int = 5


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class SimInTheLoopHarness:
    """Drives an env + executor through a stream of missions.

    The harness owns no sim or ROS state of its own. All sim access
    goes through ``env_adapter`` and all executor access goes through
    ``mission_api``. The ``recorder`` produces the on-disk dataset.
    ``abort_signal``, when provided, is polled once per env step and
    returns a short reason string when the episode must be discarded
    (e.g. /cmd_vel silence past the grace window); ``None`` means keep
    going.
    """

    def __init__(
        self,
        *,
        env_adapter: EnvAdapter,
        mission_api: MissionApi,
        recorder: EpisodeRecorder,
        config: HarnessConfig | None = None,
        abort_signal: AbortSignal | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._env = env_adapter
        self._mission = mission_api
        self._recorder = recorder
        self._config = config or HarnessConfig()
        self._abort_signal = abort_signal
        self._clock = clock
        self._sleep = sleep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        missions: Iterable[tuple[MissionSpec, EpisodeMeta] | MissionSpec],
    ) -> list[EpisodeOutcome]:
        """Execute every mission in order, returning per-mission outcomes."""

        outcomes: list[EpisodeOutcome] = []
        for item in missions:
            if isinstance(item, tuple):
                spec, meta = item
            else:
                spec, meta = item, None
            outcomes.append(self.run_one_mission(spec, meta=meta))
        return outcomes

    def run_one_mission(
        self, spec: MissionSpec, meta: EpisodeMeta | None = None,
    ) -> EpisodeOutcome:
        """Drive one mission end-to-end and return the outcome."""

        if meta is None:
            meta = EpisodeMeta.from_spec(spec)

        logger.info(
            "[%s] starting mission '%s' (target=%s)",
            spec.scene_name, spec.mission_id, meta.target_label,
        )

        self._env.reset(scene_name=spec.scene_name)
        start_bundle = self._env.capture()
        self._recorder.begin_episode(spec=spec, meta=meta, start_bundle=start_bundle)
        frames_written = 0
        start = self._clock()

        try:
            # The starting view is part of the episode even if the
            # executor rejects the command immediately.
            self._recorder.add_frame(start_bundle)
            frames_written += 1

            mission_id = self._mission.submit(
                raw_command=meta.dispatch_command, request_id=spec.mission_id,
            )
            logger.debug("[%s] executor accepted as mission_id=%s", spec.mission_id, mission_id)
            if self._abort_signal is not None:
                self._abort_signal.arm()

            final_status, frames_written, abort_reason = self._drive_until_terminal(
                spec, frames_written=frames_written,
            )

            if abort_reason is not None:
                self._recorder.end_episode(status=final_status, discard=True)
                elapsed = self._clock() - start
                logger.warning(
                    "[%s] episode DISCARDED (%s) after %d frames / %.1fs",
                    spec.mission_id, abort_reason, frames_written, elapsed,
                )
                return EpisodeOutcome(
                    mission_id=spec.mission_id,
                    reachability=False,
                    final_status=final_status,
                    frames_written=frames_written,
                    elapsed_s=elapsed,
                    discarded=True,
                )

            # Final frame at the terminal pose.
            self._recorder.add_frame(self._env.capture())
            frames_written += 1

        except Exception:
            logger.exception("[%s] harness crashed mid-mission", spec.mission_id)
            self._recorder.end_episode(status=None, discard=True)
            raise

        self._recorder.end_episode(status=final_status)
        reachability = final_status.succeeded
        elapsed = self._clock() - start
        outcome = EpisodeOutcome(
            mission_id=spec.mission_id,
            reachability=reachability,
            final_status=final_status,
            frames_written=frames_written,
            elapsed_s=elapsed,
        )
        logger.info(
            "[%s] mission complete reachable=%s state=%s frames=%d elapsed=%.1fs",
            spec.mission_id, reachability, final_status.state, frames_written, elapsed,
        )
        return outcome

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _drive_until_terminal(
        self,
        spec: MissionSpec,
        frames_written: int,
    ) -> tuple[MissionStatus, int, str | None]:
        """Step the env + poll status until the executor reports terminal.

        Returns ``(terminal_status, frames_written, abort_reason)``;
        ``abort_reason`` is non-None when the episode must be discarded.
        """

        cfg = self._config
        steps = 0
        last_poll = self._clock()
        consecutive_status_failures = 0

        # Sentinel terminal status if the outer timeout fires before the
        # executor publishes one.
        last_status = MissionStatus(
            terminal=True, state="timeout", error_code="harness_timeout",
        )

        deadline = self._clock() + cfg.mission_timeout_s
        terminated_naturally = False
        while self._clock() < deadline and steps < cfg.max_steps_per_mission:
            self._env.step()
            steps += 1

            if self._abort_signal is not None:
                reason = self._abort_signal.check()
                if reason is not None:
                    self._cancel_quietly(spec)
                    return (
                        MissionStatus(
                            terminal=True, state="discarded", error_code=reason,
                        ),
                        frames_written,
                        reason,
                    )

            if steps % cfg.capture_every_n_steps == 0:
                self._recorder.add_frame(self._env.capture())
                frames_written += 1

            now = self._clock()
            if now - last_poll >= cfg.status_poll_period_s:
                last_poll = now
                try:
                    status = self._mission.status()
                except Exception:
                    logger.exception("[%s] status() raised", spec.mission_id)
                    status = MissionStatus(
                        terminal=False, state="unknown",
                        error_code="status_no_response",
                    )
                if status.error_code == "status_no_response":
                    consecutive_status_failures += 1
                    if consecutive_status_failures >= cfg.max_consecutive_status_failures:
                        self._cancel_quietly(spec)
                        return (
                            MissionStatus(
                                terminal=True, state="discarded",
                                error_code="executor_unreachable",
                            ),
                            frames_written,
                            "executor_unreachable",
                        )
                else:
                    consecutive_status_failures = 0
                    if status.terminal:
                        last_status = status
                        terminated_naturally = True
                        break

        if not terminated_naturally:
            # Timed out or hit the step cap. Cancel so the executor
            # doesn't keep running; re-poll once for the post-cancel state.
            self._cancel_quietly(spec)
            try:
                last_status = self._mission.status()
            except Exception:
                pass
            if not last_status.terminal:
                last_status = MissionStatus(
                    terminal=True,
                    state="timeout",
                    error_code="harness_timeout",
                    elapsed_s=cfg.mission_timeout_s,
                )

        return last_status, frames_written, None

    def _cancel_quietly(self, spec: MissionSpec) -> None:
        try:
            self._mission.cancel()
        except Exception:
            logger.exception("[%s] cancel() raised", spec.mission_id)
