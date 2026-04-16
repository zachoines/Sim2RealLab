"""Sim-in-the-loop orchestration loop.

Phase 1 ships this pure-Python harness with all sim / ROS access
abstracted behind two callables passed in by the caller. Phase 2 will
provide a thin runtime adapter that boots Isaac Sim + opens an
``rclpy.action.ActionClient`` for ``execute_mission`` and passes those
through. None of phase 1's code needs to change for that.

Per-mission lifecycle::

    for spec in mission_generator:
        writer.begin_episode()
        capture_initial_frame()           # one frame at the env reset pose
        mission_id = submit_mission(spec.raw_command)
        while True:
            env_step()
            status = poll_status(mission_id)
            capture_frame(extras={..., reachability=None})
            if status.terminal:
                break
            if elapsed > timeout: cancel + break
        capture_final_frame(extras={..., reachability=succeeded})
        writer.end_episode(keep=True)

Pure Python — testable in ``.venv_vlm`` by passing fake env / mission
callables. No Isaac Sim or rclpy imports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol

from strafer_lab.sim_in_the_loop.extras import make_episode_extras
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


# ---------------------------------------------------------------------------
# Frame bundle returned by the env adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameBundle:
    """Everything the writer needs from one env step.

    The harness does not interpret the array contents, only forwards
    them to ``PerceptionFrameWriter.save_frame``. The runtime adapter
    in phase 2 fills these from the live ``d555_camera_perception``
    handle and the robot articulation root pose.
    """

    rgb: Any  # np.ndarray (H, W, 3) uint8
    depth: Any | None  # np.ndarray (H, W) float32 or None
    robot_pos: tuple[float, float, float]
    robot_quat: tuple[float, float, float, float]
    cam_pos: tuple[float, float, float] | None = None
    cam_quat: tuple[float, float, float, float] | None = None
    bboxes: list[Mapping[str, Any]] = field(default_factory=list)
    image_width: int | None = None
    image_height: int | None = None


# ---------------------------------------------------------------------------
# Injected interfaces
# ---------------------------------------------------------------------------
#
# Both protocols are runtime-checkable so the harness can validate the
# adapter at construction time, but tests can pass plain functions /
# lambdas and they will satisfy the structural type.


class EnvAdapter(Protocol):
    def reset(self, *, scene_name: str) -> None:
        """Reset the env and load ``scene_name`` if scene-switching is supported."""

    def step(self) -> None:
        """Advance the simulator by one step (the bridge applies /cmd_vel)."""

    def capture(self) -> FrameBundle:
        """Pull RGB + depth + pose for the current frame."""


class MissionApi(Protocol):
    def submit(self, *, raw_command: str, request_id: str) -> str:
        """Submit a mission, return the mission_id assigned by the executor."""

    def status(self) -> MissionStatus:
        """Snapshot the current mission status."""

    def cancel(self) -> None:
        """Cancel the active mission."""


# ---------------------------------------------------------------------------
# Writer protocol — narrow contract from PerceptionFrameWriter
# ---------------------------------------------------------------------------


class FrameWriter(Protocol):
    def begin_episode(self) -> Any: ...
    def end_episode(self, *, keep: bool) -> None: ...
    def save_frame(
        self,
        *,
        frame_id: int | str,
        rgb: Any,
        depth: Any | None,
        scene_name: str,
        scene_type: str,
        robot_pos: Any,
        robot_quat: Any,
        cam_pos: Any | None = ...,
        cam_quat: Any | None = ...,
        bboxes: Any | None = ...,
        image_width: int | None = ...,
        image_height: int | None = ...,
        extras: Mapping[str, Any] | None = ...,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessConfig:
    """Per-run knobs for the sim-in-the-loop harness."""

    # Stop polling and label the episode unreachable after this many seconds.
    mission_timeout_s: float = 60.0
    # How often (in env steps) to capture a frame during navigation.
    capture_every_n_steps: int = 5
    # Hard cap on env steps per mission to prevent runaway loops.
    max_steps_per_mission: int = 2000
    # Sleep between status polls in seconds (real-time, not sim time).
    status_poll_period_s: float = 0.2
    # The scene_type string written into every frame's record. Matches
    # what generate_descriptions / prepare_vlm_finetune_data expect.
    scene_type: str = "infinigen_sim_in_the_loop"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class SimInTheLoopHarness:
    """Drives an env + executor through a stream of missions.

    The harness owns no sim or ROS state of its own. All sim access goes
    through ``env_adapter`` and all executor access goes through
    ``mission_api``. The ``writer`` produces the on-disk dataset.

    Phase 2 wires:
      - ``env_adapter`` to a wrapper around ``run_sim_in_the_loop.py``'s
        env handle + ``strafer_lab.bridge.graph.read_cmd_vel``.
      - ``mission_api`` to ``rclpy.action.ActionClient`` against the
        Jetson's ``execute_mission`` action.
    """

    def __init__(
        self,
        *,
        env_adapter: EnvAdapter,
        mission_api: MissionApi,
        writer: FrameWriter,
        config: HarnessConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._env = env_adapter
        self._mission = mission_api
        self._writer = writer
        self._config = config or HarnessConfig()
        self._clock = clock
        self._sleep = sleep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, missions: Iterable[MissionSpec]) -> list[EpisodeOutcome]:
        """Execute every mission in order, returning per-mission outcomes."""

        outcomes: list[EpisodeOutcome] = []
        for spec in missions:
            outcomes.append(self.run_one_mission(spec))
        return outcomes

    def run_one_mission(self, spec: MissionSpec) -> EpisodeOutcome:
        """Drive one mission end-to-end and return the outcome."""

        logger.info(
            "[%s] starting mission '%s' (target=%s)",
            spec.scene_name, spec.mission_id, spec.target_label,
        )

        self._env.reset(scene_name=spec.scene_name)
        self._writer.begin_episode()
        frames_written = 0
        start = self._clock()

        try:
            # Initial frame at reset pose, captured before submitting the
            # mission so the dataset includes the starting view even if
            # the executor rejects the command immediately.
            self._save_frame(spec=spec, frame_id=frames_written, reachability=None, status=None)
            frames_written += 1

            mission_id = self._mission.submit(
                raw_command=spec.raw_command, request_id=spec.mission_id,
            )
            logger.debug("[%s] executor accepted as mission_id=%s", spec.mission_id, mission_id)

            final_status, frames_written = self._drive_until_terminal(
                spec, frames_written=frames_written,
            )
            reachability = final_status.succeeded

            # Final frame, this time tagged with the terminal reachability + status.
            self._save_frame(
                spec=spec,
                frame_id=frames_written,
                reachability=reachability,
                status=final_status,
            )
            frames_written += 1

        except Exception:
            logger.exception("[%s] harness crashed mid-mission", spec.mission_id)
            self._writer.end_episode(keep=False)
            raise

        self._writer.end_episode(keep=True)
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
    ) -> tuple[MissionStatus, int]:
        """Step the env + poll status until the executor reports terminal.

        Captures intermediate frames every ``capture_every_n_steps``
        steps, with ``reachability=None`` because we don't yet know the
        outcome. Returns ``(terminal_status, updated_frames_written)``.
        """

        cfg = self._config
        steps = 0
        last_poll = self._clock()

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

            if steps % cfg.capture_every_n_steps == 0:
                self._save_frame(
                    spec=spec, frame_id=frames_written, reachability=None, status=None,
                )
                frames_written += 1

            now = self._clock()
            if now - last_poll >= cfg.status_poll_period_s:
                last_poll = now
                status = self._mission.status()
                if status.terminal:
                    last_status = status
                    terminated_naturally = True
                    break

        if not terminated_naturally:
            # Timed out or hit the step cap. Cancel so the executor
            # doesn't keep running; re-poll once for the post-cancel state.
            try:
                self._mission.cancel()
            except Exception:
                logger.exception("[%s] cancel() raised", spec.mission_id)
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

        return last_status, frames_written

    def _save_frame(
        self,
        *,
        spec: MissionSpec,
        frame_id: int,
        reachability: bool | None,
        status: MissionStatus | None,
    ) -> None:
        bundle = self._env.capture()
        extras = make_episode_extras(
            spec=spec,
            reachability=reachability,
            mission_status=status.as_dict() if status is not None else None,
        )
        self._writer.save_frame(
            frame_id=frame_id,
            rgb=bundle.rgb,
            depth=bundle.depth,
            scene_name=spec.scene_name,
            scene_type=self._config.scene_type,
            robot_pos=bundle.robot_pos,
            robot_quat=bundle.robot_quat,
            cam_pos=bundle.cam_pos,
            cam_quat=bundle.cam_quat,
            bboxes=bundle.bboxes,
            image_width=bundle.image_width,
            image_height=bundle.image_height,
            extras=extras,
        )

