"""Tests for strafer_lab.sim_in_the_loop.harness.SimInTheLoopHarness.

Pure Python — runs in the pxr-free autonomy suite via the strafer_lab namespace stub.
The harness is exercised against fake env / mission / recorder adapters
so no Isaac Sim or rclpy is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from strafer_lab.sim_in_the_loop.harness import (
    EpisodeMeta,
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
            sim_time_s=float(self.step_count),
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
class FakeRecorder:
    """Records every begin/add_frame/end call for assertion."""

    begins: list[tuple[MissionSpec, EpisodeMeta, FrameBundle]] = field(default_factory=list)
    frames: list[FrameBundle] = field(default_factory=list)
    ends: list[tuple[MissionStatus | None, bool]] = field(default_factory=list)

    def begin_episode(self, *, spec, meta, start_bundle) -> None:
        self.begins.append((spec, meta, start_bundle))

    def add_frame(self, bundle) -> None:
        self.frames.append(bundle)

    def end_episode(self, *, status, discard: bool = False) -> None:
        self.ends.append((status, discard))


class FakeAbortSignal:
    """Scripted abort signal: returns reasons from a queue after arming."""

    def __init__(self, reasons: list[str | None]) -> None:
        self._reasons = list(reasons)
        self.armed = 0

    def arm(self) -> None:
        self.armed += 1

    def check(self) -> str | None:
        if not self._reasons:
            return None
        return self._reasons.pop(0)


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


def _harness(env, mission, recorder, cfg, *, clock=None, abort_signal=None):
    return SimInTheLoopHarness(
        env_adapter=env,
        mission_api=mission,
        recorder=recorder,
        config=cfg,
        abort_signal=abort_signal,
        clock=clock or ManualClock(),
        sleep=lambda _s: None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunOneMissionSuccess:
    """Mission that the executor reports as terminal-succeeded immediately."""

    @pytest.fixture
    def harness_and_fakes(self):
        env = FakeEnv()
        mission = FakeMission(status_sequence=[_terminal("succeeded", elapsed_s=2.0)])
        recorder = FakeRecorder()
        cfg = HarnessConfig(
            mission_timeout_s=10.0,
            capture_every_n_steps=2,
            max_steps_per_mission=10,
            status_poll_period_s=0.0,  # poll every iteration so the test is deterministic
        )
        harness = _harness(env, mission, recorder, cfg)
        return harness, env, mission, recorder

    def test_outcome_marked_reachable(self, harness_and_fakes):
        harness, *_ = harness_and_fakes
        outcome = harness.run_one_mission(_spec())
        assert outcome.reachability is True
        assert outcome.final_status.state == "succeeded"
        assert outcome.discarded is False

    def test_recorder_begins_and_keeps_episode(self, harness_and_fakes):
        harness, _, _, recorder = harness_and_fakes
        harness.run_one_mission(_spec())
        assert len(recorder.begins) == 1
        assert recorder.ends == [(_terminal("succeeded", elapsed_s=2.0), False)]

    def test_default_meta_derived_from_spec(self, harness_and_fakes):
        harness, _, mission, recorder = harness_and_fakes
        spec = _spec()
        harness.run_one_mission(spec)
        _, meta, start_bundle = recorder.begins[0]
        assert meta.mission_text == spec.raw_command
        assert meta.dispatch_command == spec.raw_command
        assert meta.target_label == spec.target_label
        # The start bundle is captured at the reset pose and is also the
        # first recorded frame.
        assert recorder.frames[0] is start_bundle
        assert mission.submitted[0] == (spec.raw_command, spec.mission_id)

    def test_meta_dispatch_command_used_for_submit(self, harness_and_fakes):
        harness, _, mission, _ = harness_and_fakes
        spec = _spec()
        meta = EpisodeMeta(
            mission_text="go to the chair",
            dispatch_command="go to the lamp",
            source_mission_source="scene-metadata",
        )
        harness.run_one_mission(spec, meta=meta)
        assert mission.submitted[0][0] == "go to the lamp"

    def test_env_was_reset_with_scene_name(self, harness_and_fakes):
        harness, env, *_ = harness_and_fakes
        harness.run_one_mission(_spec())
        assert env.resets == ["scene_a"]

    def test_frame_count_matches_outcome(self, harness_and_fakes):
        harness, _, _, recorder = harness_and_fakes
        outcome = harness.run_one_mission(_spec())
        assert outcome.frames_written == len(recorder.frames)


class TestRunOneMissionFailure:
    """Mission whose executor terminal state is failed."""

    def test_outcome_marked_unreachable_but_kept(self):
        env = FakeEnv()
        mission = FakeMission(status_sequence=[_terminal("failed", error_code="nav_timeout")])
        recorder = FakeRecorder()
        cfg = HarnessConfig(status_poll_period_s=0.0, capture_every_n_steps=10)
        harness = _harness(env, mission, recorder, cfg)
        outcome = harness.run_one_mission(_spec())
        assert outcome.reachability is False
        assert outcome.discarded is False
        status, discard = recorder.ends[0]
        assert discard is False
        assert status.error_code == "nav_timeout"


class TestRunOneMissionTimeout:
    """Executor never reports terminal — harness must time out and cancel."""

    def test_harness_cancels_on_timeout(self):
        env = FakeEnv()
        mission = FakeMission(status_sequence=[_running()])
        recorder = FakeRecorder()
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

        harness = _harness(env, mission, recorder, cfg, clock=clock)
        outcome = harness.run_one_mission(_spec())

        assert mission.cancels == 1
        assert outcome.reachability is False
        assert outcome.final_status.terminal is True
        # State is whatever the last status() returned post-cancel — in
        # this fake that is still "running", which the harness treats as
        # non-terminal and replaces with the timeout sentinel.
        assert outcome.final_status.state == "timeout"
        assert outcome.final_status.error_code == "harness_timeout"
        # Timeouts are kept (outcome=failed downstream), not discarded.
        assert recorder.ends[0][1] is False


class TestRunOneMissionStepCap:
    """Hit max_steps_per_mission before the mission finishes."""

    def test_step_cap_triggers_cancel(self):
        env = FakeEnv()
        mission = FakeMission(status_sequence=[_running()])
        recorder = FakeRecorder()
        cfg = HarnessConfig(
            mission_timeout_s=1000.0,  # not the limiting factor
            capture_every_n_steps=100,
            max_steps_per_mission=5,
            status_poll_period_s=0.0,
        )
        harness = _harness(env, mission, recorder, cfg)
        outcome = harness.run_one_mission(_spec())
        assert mission.cancels == 1
        assert env.step_count == cfg.max_steps_per_mission
        assert outcome.reachability is False


class TestAbortSignal:
    def test_abort_signal_discards_episode(self):
        env = FakeEnv()
        mission = FakeMission(status_sequence=[_running()])
        recorder = FakeRecorder()
        abort = FakeAbortSignal(reasons=[None, None, "cmd_vel_silence"])
        cfg = HarnessConfig(
            mission_timeout_s=1000.0,
            capture_every_n_steps=100,
            max_steps_per_mission=100,
            status_poll_period_s=0.0,
        )
        harness = _harness(env, mission, recorder, cfg, abort_signal=abort)
        outcome = harness.run_one_mission(_spec())
        assert abort.armed == 1
        assert mission.cancels == 1
        assert outcome.discarded is True
        assert outcome.final_status.state == "discarded"
        assert outcome.final_status.error_code == "cmd_vel_silence"
        assert recorder.ends == [(outcome.final_status, True)]
        # The abort fired on the third step.
        assert env.step_count == 3


class TestExecutorUnreachable:
    def test_consecutive_status_failures_discard(self):
        env = FakeEnv()
        no_response = MissionStatus(
            terminal=False, state="unknown", error_code="status_no_response",
        )
        mission = FakeMission(status_sequence=[no_response])
        recorder = FakeRecorder()
        cfg = HarnessConfig(
            mission_timeout_s=1000.0,
            capture_every_n_steps=100,
            max_steps_per_mission=100,
            status_poll_period_s=0.0,
            max_consecutive_status_failures=3,
        )
        harness = _harness(env, mission, recorder, cfg)
        outcome = harness.run_one_mission(_spec())
        assert outcome.discarded is True
        assert outcome.final_status.error_code == "executor_unreachable"
        assert recorder.ends[0][1] is True


class TestRunMultipleMissions:
    def test_run_iterates_spec_meta_pairs(self):
        env = FakeEnv()
        mission = FakeMission(
            status_sequence=[_terminal("succeeded"), _terminal("succeeded")]
        )
        recorder = FakeRecorder()
        cfg = HarnessConfig(
            status_poll_period_s=0.0,
            capture_every_n_steps=100,
            max_steps_per_mission=2,
        )
        harness = _harness(env, mission, recorder, cfg)
        specs = [_spec("Chair", 1), _spec("Table", 2)]
        pairs = [(s, EpisodeMeta.from_spec(s)) for s in specs]
        outcomes = harness.run(pairs)
        assert len(outcomes) == 2
        assert all(o.reachability for o in outcomes)
        assert env.resets == ["scene_a", "scene_a"]
        assert len(recorder.begins) == 2
        assert [d for _, d in recorder.ends] == [False, False]


class TestExceptionDuringMission:
    def test_recorder_episode_discarded_on_crash(self):
        class CrashingMission:
            def submit(self, *, raw_command, request_id):
                raise RuntimeError("planner offline")

            def status(self):  # pragma: no cover — never reached
                raise AssertionError

            def cancel(self):  # pragma: no cover — never reached
                raise AssertionError

        env = FakeEnv()
        recorder = FakeRecorder()
        harness = _harness(env, CrashingMission(), recorder, HarnessConfig())

        with pytest.raises(RuntimeError, match="planner offline"):
            harness.run_one_mission(_spec())

        assert len(recorder.begins) == 1
        assert recorder.ends == [(None, True)]
