"""Tests for TransitMonitor and BackgroundMapper."""

from __future__ import annotations

import time
from threading import Event
from unittest.mock import MagicMock

import numpy as np
import pytest

from strafer_autonomy.semantic_map.background_mapper import BackgroundMapper
from strafer_autonomy.semantic_map.models import Pose2D
from strafer_autonomy.semantic_map.transit_monitor import TransitMonitor


def _random_embedding(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


class _FakeNode:
    def __init__(self, x: float, y: float):
        self.pose = Pose2D(x=x, y=y, yaw=0.0)


class _FakeMap:
    def __init__(self, top_results: list[tuple[_FakeNode, float]] | None = None):
        self._top_results = top_results or []
        self.clip_encoder = MagicMock()
        self.clip_encoder.encode_image.return_value = _random_embedding()
        self.added: list[dict] = []

    def query_by_embedding(self, embedding, n_results=3):
        return self._top_results[:n_results]

    def add_observation(self, **kwargs):
        self.added.append(kwargs)


class _FakeGoalPose:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


# ---------------------------------------------------------------------------
# TransitMonitor
# ---------------------------------------------------------------------------


class TestTransitMonitor:
    def test_inactive_by_default(self):
        monitor = TransitMonitor(_FakeMap())
        assert not monitor.is_active
        result = monitor.check(_random_embedding(), np.array([0.0, 0.0]))
        assert result["on_track"]
        assert not result["abort"]

    def test_activate_and_deactivate(self):
        monitor = TransitMonitor(_FakeMap())
        monitor.activate(_FakeGoalPose(5.0, 5.0), goal_radius_m=2.0)
        assert monitor.is_active
        monitor.deactivate()
        assert not monitor.is_active

    def test_sparse_map_does_not_abort(self):
        fake_map = _FakeMap(top_results=[(_FakeNode(0.0, 0.0), 0.1)])
        monitor = TransitMonitor(fake_map)
        monitor.activate(_FakeGoalPose(5.0, 5.0))
        result = monitor.check(_random_embedding(), np.array([0.0, 0.0]))
        assert not result["abort"]
        assert result["reason"] == "sparse_map"

    def test_on_track_when_near_goal(self):
        # All three top matches near the goal
        fake_map = _FakeMap(top_results=[
            (_FakeNode(5.0, 5.0), 0.1),
            (_FakeNode(5.5, 4.8), 0.15),
            (_FakeNode(4.9, 5.1), 0.2),
        ])
        monitor = TransitMonitor(fake_map)
        monitor.activate(_FakeGoalPose(5.0, 5.0), goal_radius_m=1.0)
        result = monitor.check(_random_embedding(), np.array([4.5, 4.5]))
        assert result["on_track"]
        assert not result["abort"]

    def test_diverges_after_three_off_course_captures(self):
        # All three top matches far from the 5,5 goal
        fake_map = _FakeMap(top_results=[
            (_FakeNode(20.0, 20.0), 0.1),
            (_FakeNode(21.0, 20.5), 0.2),
            (_FakeNode(19.5, 21.0), 0.3),
        ])
        monitor = TransitMonitor(fake_map)
        monitor.activate(_FakeGoalPose(5.0, 5.0), goal_radius_m=2.0)

        r1 = monitor.check(_random_embedding(), np.array([1.0, 1.0]))
        r2 = monitor.check(_random_embedding(), np.array([2.0, 2.0]))
        r3 = monitor.check(_random_embedding(), np.array([3.0, 3.0]))
        assert r1["on_track"] and not r1["abort"]
        assert r2["on_track"] and not r2["abort"]
        assert r3["abort"]
        assert r3["reason"] == "transit_divergence"

    def test_divergence_resets_on_good_capture(self):
        on_course_results = [
            (_FakeNode(5.0, 5.0), 0.1),
            (_FakeNode(5.2, 5.1), 0.2),
            (_FakeNode(4.9, 4.8), 0.3),
        ]
        off_course_results = [
            (_FakeNode(20.0, 20.0), 0.1),
            (_FakeNode(21.0, 20.5), 0.2),
            (_FakeNode(19.5, 21.0), 0.3),
        ]

        fake_map = _FakeMap()
        monitor = TransitMonitor(fake_map)
        monitor.activate(_FakeGoalPose(5.0, 5.0), goal_radius_m=2.0)

        fake_map._top_results = off_course_results
        monitor.check(_random_embedding(), np.array([1.0, 1.0]))
        monitor.check(_random_embedding(), np.array([2.0, 2.0]))

        # A good capture lands near the goal — should NOT abort on the next one
        fake_map._top_results = on_course_results
        result = monitor.check(_random_embedding(), np.array([4.0, 4.0]))
        assert not result["abort"]


# ---------------------------------------------------------------------------
# BackgroundMapper
# ---------------------------------------------------------------------------


def _make_ros_client(pose_sequence: list[dict]):
    ros = MagicMock()
    pose_iter = iter(pose_sequence)

    def _state():
        try:
            return {"pose": next(pose_iter), "velocity": {}, "navigation_active": False}
        except StopIteration:
            return {"pose": pose_sequence[-1], "velocity": {}, "navigation_active": False}

    ros.get_robot_state.side_effect = _state

    observation = MagicMock()
    observation.color_image_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    observation.stamp_sec = time.time()
    ros.capture_scene_observation.return_value = observation
    return ros


class TestBackgroundMapper:
    def test_tick_captures_when_moved_enough(self):
        ros = _make_ros_client([
            {"x": 0.0, "y": 0.0, "qz": 0.0, "qw": 1.0},
        ])
        fake_map = _FakeMap()
        mapper = BackgroundMapper(
            ros_client=ros,
            semantic_map=fake_map,
            min_translation_m=0.5,
        )
        mapper._tick_once()
        assert len(fake_map.added) == 1

    def test_tick_skips_when_stationary(self):
        pose = {"x": 0.0, "y": 0.0, "qz": 0.0, "qw": 1.0}
        ros = _make_ros_client([pose, pose])
        fake_map = _FakeMap()
        mapper = BackgroundMapper(
            ros_client=ros,
            semantic_map=fake_map,
            min_translation_m=0.5,
        )
        mapper._tick_once()
        mapper._tick_once()
        assert len(fake_map.added) == 1  # only first capture

    def test_tick_captures_after_rotation(self):
        import math
        yaw_start = 0.0
        yaw_after = math.radians(60)
        poses = [
            {"x": 0.0, "y": 0.0, "qz": math.sin(yaw_start / 2), "qw": math.cos(yaw_start / 2)},
            {"x": 0.0, "y": 0.0, "qz": math.sin(yaw_after / 2), "qw": math.cos(yaw_after / 2)},
        ]
        ros = _make_ros_client(poses)
        fake_map = _FakeMap()
        mapper = BackgroundMapper(
            ros_client=ros,
            semantic_map=fake_map,
            min_translation_m=1000.0,  # disable distance trigger
            min_rotation_deg=30.0,
        )
        mapper._tick_once()
        mapper._tick_once()
        assert len(fake_map.added) == 2

    def test_no_pose_does_not_crash(self):
        ros = MagicMock()
        ros.get_robot_state.return_value = {"pose": None, "velocity": {}, "navigation_active": False}
        mapper = BackgroundMapper(
            ros_client=ros, semantic_map=_FakeMap(),
        )
        mapper._tick_once()  # should just return silently

    def test_transit_monitor_integration(self):
        fake_map = _FakeMap(top_results=[
            (_FakeNode(20.0, 20.0), 0.1),
            (_FakeNode(21.0, 20.5), 0.2),
            (_FakeNode(19.5, 21.0), 0.3),
        ])
        monitor = TransitMonitor(fake_map)
        monitor.activate(_FakeGoalPose(5.0, 5.0), goal_radius_m=2.0)

        ros = _make_ros_client([
            {"x": float(i), "y": 0.0, "qz": 0.0, "qw": 1.0}
            for i in range(6)
        ])
        mapper = BackgroundMapper(
            ros_client=ros,
            semantic_map=fake_map,
            transit_monitor=monitor,
            min_translation_m=0.1,
        )
        for _ in range(4):
            mapper._tick_once()
        assert mapper.divergence_detected()
