"""Tests for validate_drive patterns, metrics, and report generation."""

import math

import numpy as np
import pytest

from strafer_bringup.validate_drive import (
    PATTERNS,
    JointSample,
    MapFrame,
    Metric,
    Pose2D,
    Segment,
    build_circle,
    build_figure8,
    build_square,
    build_strafe_square,
    compute_encoder_health,
    compute_map_quality,
    compute_odom_drift,
    compute_slam_deviation,
    format_report,
    total_duration,
)


# ═════════════════════════════════════════════════════════════════════════
# Ideal kinematics integration — all patterns should close
# ═════════════════════════════════════════════════════════════════════════

def _integrate(segments, dt=0.001):
    """Dead-reckon body-frame velocities → world-frame (x, y, θ)."""
    x, y, theta = 0.0, 0.0, 0.0
    for seg in segments:
        t = 0.0
        while t < seg.duration:
            step = min(dt, seg.duration - t)
            x += (seg.vx * math.cos(theta) -
                  seg.vy * math.sin(theta)) * step
            y += (seg.vx * math.sin(theta) +
                  seg.vy * math.cos(theta)) * step
            theta += seg.omega * step
            t += step
    return x, y, theta


class TestPatternClosure:
    """All patterns should return to the starting pose (ideal case)."""

    @pytest.mark.parametrize("name,builder", list(PATTERNS.items()))
    def test_returns_to_origin(self, name, builder):
        segs = builder()
        x, y, _ = _integrate(segs)
        d = math.hypot(x, y)
        assert d < 0.05, (
            f"Pattern '{name}' drifted {d:.3f} m from origin (ideal)")


class TestPatternProperties:
    """Basic structural checks for generated segments."""

    def test_square_has_16_segments(self):
        # 4 sides × (fwd + pause + turn + pause)
        assert len(build_square()) == 16

    def test_strafe_square_has_8_segments(self):
        # 4 sides × (move + pause)
        assert len(build_strafe_square()) == 8

    def test_circle_single_segment(self):
        assert len(build_circle()) == 1

    def test_figure8_two_segments(self):
        assert len(build_figure8()) == 2

    def test_all_durations_positive(self):
        for name, builder in PATTERNS.items():
            for seg in builder():
                assert seg.duration > 0, f"Non-positive duration in {name}"

    def test_total_duration_matches_sum(self):
        segs = build_square()
        assert total_duration(segs) == pytest.approx(
            sum(s.duration for s in segs))

    def test_is_motion_property(self):
        assert Segment(0.2, 0.0, 0.0, 1.0).is_motion
        assert Segment(0.0, 0.2, 0.0, 1.0).is_motion
        assert Segment(0.0, 0.0, 0.5, 1.0).is_motion
        assert not Segment(0.0, 0.0, 0.0, 0.5).is_motion

    def test_kwargs_accepted(self):
        """All builders accept arbitrary kwargs without error."""
        for builder in PATTERNS.values():
            builder(side=1.0, vel=0.2, omega=0.5, radius=0.5)


# ═════════════════════════════════════════════════════════════════════════
# Metrics tests
# ═════════════════════════════════════════════════════════════════════════

class TestOdomDrift:

    def test_perfect_closure(self):
        poses = [Pose2D(0, 0.0, 0.0, 0.0), Pose2D(10, 0.0, 0.0, 0.0)]
        m = compute_odom_drift(poses)
        assert m.passed
        assert "0.000" in m.value

    def test_large_drift_fails(self):
        poses = [Pose2D(0, 0.0, 0.0, 0.0), Pose2D(10, 1.0, 0.0, 0.0)]
        m = compute_odom_drift(poses)
        assert not m.passed

    def test_insufficient_data(self):
        m = compute_odom_drift([])
        assert not m.passed
        assert "N/A" in m.value


class TestSlamDeviation:

    def test_no_slam_skipped(self):
        odom = [Pose2D(0, 0.0, 0.0, 0.0)]
        m = compute_slam_deviation(odom, [])
        assert m.passed
        assert "skipped" in m.detail.lower()

    def test_identical_paths(self):
        poses = [Pose2D(float(i), i * 0.1, 0.0, 0.0) for i in range(10)]
        m = compute_slam_deviation(poses, poses)
        assert m.passed
        assert "0.000" in m.value

    def test_large_deviation_fails(self):
        odom = [Pose2D(float(i), i * 0.1, 0.0, 0.0) for i in range(10)]
        slam = [Pose2D(float(i), i * 0.1 + 0.5, 0.0, 0.0)
                for i in range(10)]
        m = compute_slam_deviation(odom, slam)
        assert not m.passed


class TestEncoderHealth:

    def _motion_seg(self):
        return [Segment(0.2, 0.0, 0.0, 2.0)]

    def test_all_active(self):
        segs = self._motion_seg()
        joints = [
            JointSample(i * 0.02, (1.0, 1.0, 1.0, 1.0))
            for i in range(100)
        ]
        m = compute_encoder_health(joints, segs, 0.0)
        assert m.passed

    def test_one_dead_encoder(self):
        segs = self._motion_seg()
        joints = [
            JointSample(i * 0.02, (1.0, 1.0, 0.0, 1.0))
            for i in range(100)
        ]
        m = compute_encoder_health(joints, segs, 0.0)
        assert not m.passed

    def test_no_data_fails(self):
        segs = self._motion_seg()
        m = compute_encoder_health([], segs, 0.0)
        assert not m.passed


class TestMapQuality:

    def test_no_maps_skipped(self):
        m = compute_map_quality([])
        assert m.passed
        assert "skipped" in m.detail.lower()

    def test_mostly_known_passes(self):
        data = np.zeros(10000, dtype=np.int8)  # all free
        mf = MapFrame(0.0, 100, 100, 0.05, 0.0, 0.0, data)
        m = compute_map_quality([mf])
        assert m.passed

    def test_all_unknown_fails(self):
        data = np.full(10000, -1, dtype=np.int8)
        mf = MapFrame(0.0, 100, 100, 0.05, 0.0, 0.0, data)
        m = compute_map_quality([mf])
        assert not m.passed


# ═════════════════════════════════════════════════════════════════════════
# Report formatting
# ═════════════════════════════════════════════════════════════════════════

class TestReport:

    def test_all_pass_verdict(self):
        metrics = [Metric("A", True, "ok", "ok"),
                   Metric("B", True, "ok", "ok")]
        txt = format_report(metrics, "square", 30.0)
        assert "ALL CHECKS PASSED" in txt

    def test_fail_verdict(self):
        metrics = [Metric("A", True, "ok", "ok"),
                   Metric("B", False, "bad", "ok")]
        txt = format_report(metrics, "square", 30.0)
        assert "SOME CHECKS FAILED" in txt

    def test_contains_pattern(self):
        txt = format_report([], "circle", 15.0)
        assert "circle" in txt
