"""Tests for strafer_lab.sim_in_the_loop.harness.SimInTheLoopHarness.

Pure Python — runs in .venv_vlm via the strafer_lab namespace stub.
The harness is exercised against fake env / mission / writer adapters
so no Isaac Sim or rclpy is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from strafer_lab.sim_in_the_loop.harness import (
    EpisodeOutcome,
    FrameBundle,
    HarnessConfig,
    MissionStatus,
    SimInTheLoopHarness,
)
from strafer_lab.sim_in_the_loop.mission import MissionSpec


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeEnv:
    """Records reset/step calls and emits a deterministic FrameBundle."""

    resets: list[str] = field(default_factory=list)
    step_count: int = 0

    def reset(self, *, scene_name: str) -> None:
        self.resets.append(scene_name)
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1

    def capture(self) -> FrameBundle:
        return FrameBundle(
            rgb=f"rgb_{self.step_count}",
            depth=None,
            robot_pos=(float(self.step_count), 0.0, 0.0),
            robot_quat=(0.0, 0.0, 0.0, 1.0),
            image_width=640,
            image_height=360,
        )


@dataclass
class FakeMission:
    """Mission API that walks through a scripted status sequence.

    ``status_sequence`` is consulted on each ``status()`` call. The last
    element repeats indefinitely, which simulates a stuck non-terminal
    state until an external timeout fires.
    """

    status_sequence: list[MissionStatus] = field(default_factory=list)
    submitted: list[tuple[str, str]] = field(default_factory=list)
    cancels: int = 0
    _index: int = 0

    def submit(self, *, raw_command: str, request_id: str) -> str:
        self.submitted.append((raw_command, request_id))
        return f"executor_{request_id}"

    def status(self) -> MissionStatus:
        idx = min(self._index, len(self.status_sequence) - 1)
        self._index += 1
        return self.status_sequence[idx]

    def cancel(self) -> None:
        self.cancels += 1


@dataclass
class FakeWriter:
    """Records every begin/end/save_frame call for assertion."""

    begins: int = 0
    ends_keep: list[bool] = field(default_factory=list)
    saved_frames: list[dict[str, Any]] = field(default_factory=list)

    def begin_episode(self) -> None:
        self.begins += 1

    def end_episode(self, *, keep: bool) -> None:
        self.ends_keep.append(keep)

    def save_frame(
        self,
        *,
        frame_id,
        rgb,
        depth,
        scene_name,
        scene_type,
        robot_pos,
        robot_quat,
        cam_pos=None,
        cam_quat=None,
        bboxes=None,
        image_width=None,
        image_height=None,
        extras: Mapping[str, Any] | None = None,
    ) -> str:
        self.saved_frames.append(
            {
                "frame_id": frame_id,
                "rgb": rgb,
                "scene_name": scene_name,
                "scene_type": scene_type,
                "extras": dict(extras or {}),
            }
        )
        return f"frame_{frame_id:04d}.jpg"


class ManualClock:
    """Monotonic clock under test control. Advances only on ``advance()``."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _spec(label: str = "Chair", instance_id: int = 1) -> MissionSpec:
    return MissionSpec(
        mission_id=f"scene_a__{label.lower()}__{instance_id}",
        scene_name="scene_a",
        target_label=label,
        target_instance_id=instance_id,
        target_position_3d=(1.0, 2.0, 0.0),
        target_room_idx=0,
        raw_command=f"go to the {label.lower()}",
    )


def _terminal(state: str = "succeeded", **kwargs) -> MissionStatus:
    return MissionStatus(terminal=True, state=state, **kwargs)


def _running() -> MissionStatus:
    return MissionStatus(terminal=False, state="running")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunOneMissionSuccess:
    """Mission that the executor reports as terminal-succeeded immediately."""

    @pytest.fixture
    def harness_and_fakes(self):
        env = FakeEnv()
        mission = FakeMission(status_sequence=[_terminal("succeeded", elapsed_s=2.0)])
        writer = FakeWriter()
        clock = ManualClock()
        cfg = HarnessConfig(
            mission_timeout_s=10.0,
            capture_every_n_steps=2,
            max_steps_per_mission=10,
            status_poll_period_s=0.0,  # poll every iteration so the test is deterministic
        )
        harness = SimInTheLoopHarness(
            env_adapter=env,
            mission_api=mission,
            writer=writer,
            config=cfg,
            clock=clock,
            sleep=lambda _s: None,
        )
        return harness, env, mission, writer, clock

    def test_outcome_marked_reachable(self, harness_and_fakes):
        harness, *_ = harness_and_fakes
        outcome = harness.run_one_mission(_spec())
        assert outcome.reachability is True
        assert outcome.final_status.state == "succeeded"

    def test_writer_begins_and_keeps_episode(self, harness_and_fakes):
        harness, _, _, writer, _ = harness_and_fakes
        harness.run_one_mission(_spec())
        assert writer.begins == 1
        assert writer.ends_keep == [True]

    def test_initial_frame_captured_before_submit(self, harness_and_fakes):
        harness, _, mission, writer, _ = harness_and_fakes
        harness.run_one_mission(_spec())
        # First frame extras should NOT carry reachability (mission not done yet).
        assert "reachability" not in writer.saved_frames[0]["extras"]
        # And it must precede the executor submission. We can't observe the
        # ordering directly, but we can assert the first frame existed.
        assert writer.saved_frames[0]["frame_id"] == 0
        assert mission.submitted[0][1] == "scene_a__chair__1"

    def test_final_frame_carries_reachability_true(self, harness_and_fakes):
        harness, *_ = harness_and_fakes
        outcome = harness.run_one_mission(_spec())
        last = harness_and_fakes[3].saved_frames[-1]
        assert last["extras"]["reachability"] is True
        assert last["extras"]["mission_state"] == "succeeded"
        assert outcome.frames_written == len(harness_and_fakes[3].saved_frames)

    def test_env_was_reset_with_scene_name(self, harness_and_fakes):
        harness, env, *_ = harness_and_fakes
        harness.run_one_mission(_spec())
        assert env.resets == ["scene_a"]


class TestRunOneMissionFailure:
    """Mission whose executor terminal state is failed."""

    def test_outcome_marked_unreachable(self):
        env = FakeEnv()
        mission = FakeMission(status_sequence=[_terminal("failed", error_code="nav_timeout")])
        writer = FakeWriter()
        cfg = HarnessConfig(status_poll_period_s=0.0, capture_every_n_steps=10)
        harness = SimInTheLoopHarness(
            env_adapter=env,
            mission_api=mission,
            writer=writer,
            config=cfg,
            clock=ManualClock(),
            sleep=lambda _s: None,
        )
        outcome = harness.run_one_mission(_spec())
        assert outcome.reachability is False
        assert writer.saved_frames[-1]["extras"]["reachability"] is False
        assert writer.saved_frames[-1]["extras"]["mission_error_code"] == "nav_timeout"


class TestRunOneMissionTimeout:
    """Executor never reports terminal — harness must time out and cancel."""

    def test_harness_cancels_on_timeout(self):
        env = FakeEnv()
        # Status always non-terminal until cancel is observed via the
        # next status read below; FakeMission keeps returning the last
        # element of the sequence.
        mission = FakeMission(status_sequence=[_running()])
        writer = FakeWriter()
        clock = ManualClock()
        cfg = HarnessConfig(
            mission_timeout_s=1.0,
            capture_every_n_steps=100,  # don't bother capturing during the loop
            max_steps_per_mission=100,
            status_poll_period_s=0.0,
        )

        # Wrap env.step so each step also advances the manual clock past
        # the timeout deadline after enough iterations.
        original_step = env.step

        def step_and_advance() -> None:
            original_step()
            clock.advance(0.2)

        env.step = step_and_advance  # type: ignore[assignment]

        harness = SimInTheLoopHarness(
            env_adapter=env,
            mission_api=mission,
            writer=writer,
            config=cfg,
            clock=clock,
            sleep=lambda _s: None,
        )
        outcome = harness.run_one_mission(_spec())

        assert mission.cancels == 1
        assert outcome.reachability is False
        assert outcome.final_status.terminal is True
        # State is whatever the last status() returned post-cancel — in
        # this fake that is still "running", which the harness treats as
        # non-terminal and replaces with the timeout sentinel.
        assert outcome.final_status.state == "timeout"
        assert outcome.final_status.error_code == "harness_timeout"


class TestRunOneMissionStepCap:
    """Hit max_steps_per_mission before the mission finishes."""

    def test_step_cap_triggers_cancel(self):
        env = FakeEnv()
        mission = FakeMission(status_sequence=[_running()])
        writer = FakeWriter()
        cfg = HarnessConfig(
            mission_timeout_s=1000.0,  # not the limiting factor
            capture_every_n_steps=100,
            max_steps_per_mission=5,
            status_poll_period_s=0.0,
        )
        harness = SimInTheLoopHarness(
            env_adapter=env,
            mission_api=mission,
            writer=writer,
            config=cfg,
            clock=ManualClock(),
            sleep=lambda _s: None,
        )
        outcome = harness.run_one_mission(_spec())
        assert mission.cancels == 1
        assert env.step_count == cfg.max_steps_per_mission
        assert outcome.reachability is False


class TestRunMultipleMissions:
    def test_run_iterates_all_specs(self):
        env = FakeEnv()
        mission = FakeMission(
            status_sequence=[_terminal("succeeded"), _terminal("succeeded")]
        )
        # FakeMission's status() walks through the sequence, so each
        # mission needs its own terminal status call. Reset between
        # runs is handled by index clamping at the last element, which
        # is also "succeeded" — fine for this test.
        writer = FakeWriter()
        cfg = HarnessConfig(
            status_poll_period_s=0.0,
            capture_every_n_steps=100,
            max_steps_per_mission=2,
        )
        harness = SimInTheLoopHarness(
            env_adapter=env,
            mission_api=mission,
            writer=writer,
            config=cfg,
            clock=ManualClock(),
            sleep=lambda _s: None,
        )
        specs = [_spec("Chair", 1), _spec("Table", 2)]
        outcomes = harness.run(specs)
        assert len(outcomes) == 2
        assert all(o.reachability for o in outcomes)
        assert env.resets == ["scene_a", "scene_a"]
        assert writer.begins == 2
        assert writer.ends_keep == [True, True]


class TestExceptionDuringMission:
    def test_writer_episode_discarded_on_crash(self):
        class CrashingMission:
            def submit(self, *, raw_command, request_id):
                raise RuntimeError("planner offline")

            def status(self):  # pragma: no cover — never reached
                raise AssertionError

            def cancel(self):  # pragma: no cover — never reached
                raise AssertionError

        env = FakeEnv()
        writer = FakeWriter()
        harness = SimInTheLoopHarness(
            env_adapter=env,
            mission_api=CrashingMission(),
            writer=writer,
            config=HarnessConfig(),
            clock=ManualClock(),
            sleep=lambda _s: None,
        )

        with pytest.raises(RuntimeError, match="planner offline"):
            harness.run_one_mission(_spec())

        assert writer.begins == 1
        assert writer.ends_keep == [False]
