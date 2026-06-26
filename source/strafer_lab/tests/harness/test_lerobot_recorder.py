"""Tests for strafer_lab.sim_in_the_loop.lerobot_recorder.

Drives BridgeLeRobotRecorder with a fake writer (no lerobot, no Isaac
Sim) and exercises the outcome mapping, injection metadata flow,
detections gating, and the /cmd_vel-silence watchdog. Pure Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from strafer_lab.sim_in_the_loop.harness import (
    EpisodeMeta,
    FrameBundle,
    MissionStatus,
)
from strafer_lab.sim_in_the_loop.lerobot_recorder import (
    BridgeLeRobotRecorder,
    CmdVelGraceWatch,
    CoverageLeRobotRecorder,
    yaw_from_quat_xyzw,
)
from strafer_lab.sim_in_the_loop.mission import MissionSpec


# ---------------------------------------------------------------------------
# Fakes + helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeWriter:
    detections_max: int | None = None
    begins: list[dict[str, Any]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    ends: list[dict[str, Any]] = field(default_factory=list)

    def begin_episode(self, **kwargs) -> int:
        self.begins.append(kwargs)
        return len(self.begins) - 1

    def add_frame(self, **kwargs) -> None:
        self.frames.append(kwargs)

    def end_episode(self, **kwargs) -> None:
        self.ends.append(kwargs)


def _spec() -> MissionSpec:
    return MissionSpec(
        mission_id="scene_a__chair__1",
        scene_name="scene_a",
        target_label="Chair",
        target_instance_id=1,
        target_position_3d=(3.0, 4.0, 0.0),
        target_room_idx=0,
        raw_command="go to the chair",
    )


def _meta(**overrides) -> EpisodeMeta:
    defaults = dict(
        mission_text="go to the chair",
        dispatch_command="go to the chair",
        source_mission_source="scene-metadata",
        target_label="Chair",
        target_object_id="1",
        target_position_3d=(3.0, 4.0, 0.0),
    )
    defaults.update(overrides)
    return EpisodeMeta(**defaults)


def _bundle(**overrides) -> FrameBundle:
    defaults = dict(
        rgb="rgb",
        depth="depth",
        robot_pos=(0.0, 0.0, 0.05),
        robot_quat=(0.0, 0.0, 0.0, 1.0),
        achieved_vel=(0.1, 0.0, 0.0),
        action=(0.5, 0.0, -0.2),
        sim_time_s=1.5,
        depth_policy="depth_policy",
    )
    defaults.update(overrides)
    return FrameBundle(**defaults)


def _terminal(state: str) -> MissionStatus:
    return MissionStatus(terminal=True, state=state)


def _recorder(writer: FakeWriter | None = None) -> tuple[BridgeLeRobotRecorder, FakeWriter]:
    writer = writer or FakeWriter()
    return BridgeLeRobotRecorder(writer=writer, scene_id="scene_a"), writer


# ---------------------------------------------------------------------------
# begin_episode mapping
# ---------------------------------------------------------------------------


class TestBeginEpisode:
    def test_maps_meta_onto_writer(self):
        recorder, writer = _recorder()
        recorder.begin_episode(spec=_spec(), meta=_meta(), start_bundle=_bundle())
        begin = writer.begins[0]
        assert begin["mission_text"] == "go to the chair"
        assert begin["scene_id"] == "scene_a"
        assert begin["target_label"] == "Chair"
        assert begin["target_object_id"] == "1"
        assert begin["target_position_3d"] == pytest.approx([3.0, 4.0, 0.0])
        assert begin["source_driver"] == "bridge"
        assert begin["source_mission_source"] == "scene-metadata"
        assert begin["injection_mode"] is None
        assert begin["generator_metadata"] is None

    def test_start_pose_is_xy_yaw_from_bundle(self):
        recorder, writer = _recorder()
        quat_90deg = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        recorder.begin_episode(
            spec=_spec(), meta=_meta(),
            start_bundle=_bundle(robot_pos=(1.0, 2.0, 0.05), robot_quat=quat_90deg),
        )
        x, y, yaw = writer.begins[0]["start_pose"]
        assert (x, y) == (1.0, 2.0)
        assert yaw == pytest.approx(math.pi / 2)

    def test_leg_initial_distance_from_start_to_target(self):
        recorder, writer = _recorder()
        recorder.begin_episode(spec=_spec(), meta=_meta(), start_bundle=_bundle())
        assert writer.begins[0]["leg_initial_distance_m"] == pytest.approx(5.0)

    def test_queue_meta_carries_paraphrases_and_generator_metadata(self):
        recorder, writer = _recorder()
        meta = _meta(
            source_mission_source="queue",
            paraphrases=("p1", "p2"),
            generator_metadata={"llm_model": "Qwen3-4B"},
        )
        recorder.begin_episode(spec=_spec(), meta=meta, start_bundle=_bundle())
        begin = writer.begins[0]
        assert begin["source_mission_source"] == "queue"
        assert begin["paraphrases"] == ["p1", "p2"]
        assert begin["generator_metadata"] == {"llm_model": "Qwen3-4B"}


# ---------------------------------------------------------------------------
# add_frame mapping
# ---------------------------------------------------------------------------


class TestAddFrame:
    def test_forwards_channels_and_state(self):
        recorder, writer = _recorder()
        recorder.begin_episode(spec=_spec(), meta=_meta(), start_bundle=_bundle())
        recorder.add_frame(_bundle())
        frame = writer.frames[0]
        assert frame["pose"] == pytest.approx([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0])
        assert frame["achieved_vel"] == pytest.approx([0.1, 0.0, 0.0])
        assert frame["action"] == pytest.approx([0.5, 0.0, -0.2])
        assert frame["rgb_perception"] == "rgb"
        assert frame["depth_m"] == "depth"
        assert frame["depth_policy_m"] == "depth_policy"
        assert frame["sim_time"] == pytest.approx(1.5)
        # Writer declared no detections → kwarg omitted entirely.
        assert "detections" not in frame

    def test_detections_forwarded_when_declared(self):
        recorder, writer = _recorder(FakeWriter(detections_max=32))
        recorder.begin_episode(spec=_spec(), meta=_meta(), start_bundle=_bundle())
        sentinel = [object()]
        recorder.add_frame(_bundle(detections=sentinel))
        assert writer.frames[0]["detections"] == sentinel

    def test_no_detections_frame_packs_empty_when_declared(self):
        recorder, writer = _recorder(FakeWriter(detections_max=32))
        recorder.begin_episode(spec=_spec(), meta=_meta(), start_bundle=_bundle())
        recorder.add_frame(_bundle(detections=None))
        assert writer.frames[0]["detections"] == []


# ---------------------------------------------------------------------------
# end_episode outcome mapping
# ---------------------------------------------------------------------------


class TestEndEpisodeOutcomes:
    def _run(self, *, meta: EpisodeMeta, status: MissionStatus | None, discard=False):
        recorder, writer = _recorder()
        recorder.begin_episode(spec=_spec(), meta=meta, start_bundle=_bundle())
        recorder.end_episode(status=status, discard=discard)
        return recorder, writer

    def test_succeeded_plain(self):
        _, writer = self._run(meta=_meta(), status=_terminal("succeeded"))
        end = writer.ends[0]
        assert end["outcome"] == "succeeded"
        assert end["outcome_category"] == "on_course"
        assert end["hard_negative_category"] is None
        assert end["injection_mode_actual"] is None

    def test_failed_kept_as_failure(self):
        _, writer = self._run(meta=_meta(), status=_terminal("failed"))
        end = writer.ends[0]
        assert end["outcome"] == "failed"
        assert end["outcome_category"] == "on_course"

    def test_timeout_kept_as_failure(self):
        _, writer = self._run(meta=_meta(), status=_terminal("timeout"))
        assert writer.ends[0]["outcome"] == "failed"

    @pytest.mark.parametrize("state", ["cancelled", "canceled", "aborted", "discarded"])
    def test_externally_killed_states_discard(self, state):
        recorder, writer = self._run(meta=_meta(), status=_terminal(state))
        assert writer.ends[0] == {"discard": True}
        assert recorder.episodes_discarded == 1
        assert recorder.episodes_kept == 0

    def test_explicit_discard(self):
        recorder, writer = self._run(meta=_meta(), status=None, discard=True)
        assert writer.ends[0] == {"discard": True}
        assert recorder.episodes_discarded == 1

    def test_injected_success_is_hard_negative(self):
        meta = _meta(
            injection_mode="wrong_instance",
            injection_mode_actual="wrong_room",
            original_target_position_3d=(9.0, 9.0, 0.0),
        )
        _, writer = self._run(meta=meta, status=_terminal("succeeded"))
        end = writer.ends[0]
        assert end["outcome"] == "wrong_room"
        assert end["outcome_category"] == "wrong_room"
        assert end["hard_negative_category"] == "wrong_room"
        assert end["injection_mode_actual"] == "wrong_room"
        assert end["original_target_position_3d"] == pytest.approx([9.0, 9.0, 0.0])

    def test_injected_failure_is_plain_failure_with_injection_metadata(self):
        meta = _meta(
            injection_mode="wrong_room",
            injection_mode_actual="wrong_room",
            original_target_position_3d=(9.0, 9.0, 0.0),
        )
        _, writer = self._run(meta=meta, status=_terminal("failed"))
        end = writer.ends[0]
        assert end["outcome"] == "failed"
        assert end["hard_negative_category"] is None
        # Injection columns still describe what was dispatched.
        assert end["injection_mode_actual"] == "wrong_room"

    def test_dropped_injection_records_nulls(self):
        meta = _meta(injection_mode="wrong_instance", injection_mode_actual=None)
        _, writer = self._run(meta=meta, status=_terminal("succeeded"))
        end = writer.ends[0]
        assert end["outcome"] == "succeeded"
        assert end["injection_mode_actual"] is None
        assert end["original_target_position_3d"] is None

    def test_kept_counter_increments(self):
        recorder, _ = self._run(meta=_meta(), status=_terminal("succeeded"))
        assert recorder.episodes_kept == 1
        assert recorder.episodes_discarded == 0


# ---------------------------------------------------------------------------
# CmdVelGraceWatch
# ---------------------------------------------------------------------------


class TestCmdVelGraceWatch:
    def _watch(self, *, grace=5.0):
        state = {"now": 100.0, "last_cmd": 0.0}
        watch = CmdVelGraceWatch(
            last_cmd_time=lambda: state["last_cmd"],
            grace_s=grace,
            now=lambda: state["now"],
        )
        return watch, state

    def test_unarmed_never_fires(self):
        watch, state = self._watch()
        state["now"] = 1e9
        assert watch.check() is None

    def test_silence_before_first_command_is_tolerated(self):
        watch, state = self._watch()
        watch.arm()
        # No /cmd_vel newer than the arm time → planning phase; the
        # mission timeout owns this window, not the watchdog.
        state["now"] += 1000.0
        assert watch.check() is None

    def test_fires_after_silence_beyond_grace(self):
        watch, state = self._watch(grace=5.0)
        watch.arm()
        state["now"] += 1.0
        state["last_cmd"] = state["now"]  # executor starts driving
        assert watch.check() is None
        state["now"] += 5.1  # silence past the grace
        assert watch.check() == "cmd_vel_silence"

    def test_fresh_commands_keep_it_quiet(self):
        watch, state = self._watch(grace=5.0)
        watch.arm()
        for _ in range(10):
            state["now"] += 2.0
            state["last_cmd"] = state["now"]
            assert watch.check() is None

    def test_rearm_ignores_previous_mission_traffic(self):
        watch, state = self._watch(grace=5.0)
        watch.arm()
        state["now"] += 1.0
        state["last_cmd"] = state["now"]  # mission 1 drove
        state["now"] += 30.0
        watch.arm()  # mission 2 starts; old traffic must not count
        state["now"] += 20.0
        assert watch.check() is None

    def test_zero_grace_disables(self):
        watch, state = self._watch(grace=0.0)
        watch.arm()
        state["now"] += 1.0
        state["last_cmd"] = state["now"]
        state["now"] += 1e6
        assert watch.check() is None


class TestYawFromQuat:
    def test_identity_quat_zero_yaw(self):
        assert yaw_from_quat_xyzw((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0)

    def test_180_deg(self):
        assert abs(yaw_from_quat_xyzw((0.0, 0.0, 1.0, 0.0))) == pytest.approx(math.pi)


# ---------------------------------------------------------------------------
# CoverageLeRobotRecorder (scripted coverage driver)
# ---------------------------------------------------------------------------


class TestCoverageRecorder:
    def _recorder(self, writer: FakeWriter | None = None):
        writer = writer or FakeWriter()
        return CoverageLeRobotRecorder(writer=writer, scene_id="scene_a"), writer

    def test_begin_sets_coverage_provenance(self):
        recorder, writer = self._recorder()
        recorder.begin_episode(start_bundle=_bundle())
        begin = writer.begins[0]
        assert begin["source_driver"] == "scripted"
        assert begin["source_mission_source"] == "coverage"
        assert begin["mission_text"] == ""
        assert begin["scene_id"] == "scene_a"
        # No mission, so no target columns are populated.
        assert "target_label" not in begin or begin.get("target_label") is None

    def test_begin_forwards_mount_quat_and_split(self):
        recorder, writer = self._recorder()
        recorder.begin_episode(
            start_bundle=_bundle(),
            episode_split="held_out_seeds",
            realized_d555_mount_quat=(1.0, 0.0, 0.0, 0.0),
        )
        begin = writer.begins[0]
        assert begin["episode_split"] == "held_out_seeds"
        assert begin["realized_d555_mount_quat"] == [1.0, 0.0, 0.0, 0.0]

    def test_end_keeps_episode_as_succeeded(self):
        recorder, writer = self._recorder()
        recorder.begin_episode(start_bundle=_bundle())
        recorder.end_episode()
        end = writer.ends[0]
        assert end["outcome"] == "succeeded"
        assert end["outcome_category"] == "on_course"
        assert end["hard_negative_category"] is None
        assert recorder.episodes_kept == 1

    def test_discard_skips_save(self):
        recorder, writer = self._recorder()
        recorder.begin_episode(start_bundle=_bundle())
        recorder.end_episode(discard=True)
        assert writer.ends[0] == {"discard": True}
        assert recorder.episodes_discarded == 1

    def test_add_frame_gates_detections_on_writer(self):
        recorder, writer = self._recorder(FakeWriter(detections_max=None))
        recorder.begin_episode(start_bundle=_bundle())
        recorder.add_frame(_bundle(detections=["box"]))
        assert "detections" not in writer.frames[0]

        recorder2, writer2 = self._recorder(FakeWriter(detections_max=32))
        recorder2.begin_episode(start_bundle=_bundle())
        recorder2.add_frame(_bundle(detections=["box"]))
        assert writer2.frames[0]["detections"] == ["box"]
