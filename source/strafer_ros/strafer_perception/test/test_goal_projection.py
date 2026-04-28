"""Unit tests for GoalProjectionNode geometry helpers."""

import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from strafer_perception.goal_projection_node import (
    _DEPTH_MAX_M,
    _DEPTH_MIN_M,
    _ENV_DEPTH_MAX,
    _ENV_DEPTH_MIN,
    GoalProjectionNode,
    _parse_depth_env,
)


class TestMedianDepth(unittest.TestCase):
    """Tests for _median_depth (5x5 kernel, mm→m, range filter)."""

    def test_uniform_depth_returns_correct_meters(self) -> None:
        # 20x20 frame, all 2000mm = 2.0m
        depth = np.full((20, 20), 2000, dtype=np.uint16)
        result, reason = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_zero_depth_returns_all_zero_or_nan(self) -> None:
        depth = np.zeros((20, 20), dtype=np.uint16)
        result, reason = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNone(result)
        self.assertEqual(reason, "all_zero_or_nan")

    def test_out_of_range_depth_returns_out_of_range(self) -> None:
        # All pixels at 10000mm = 10m (above default _DEPTH_MAX_M=6)
        depth = np.full((20, 20), 10000, dtype=np.uint16)
        result, reason = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNone(result)
        self.assertEqual(reason, "out_of_range")

    def test_mixed_depth_takes_median_of_valid(self) -> None:
        depth = np.zeros((20, 20), dtype=np.uint16)
        # Fill 5x5 patch around (10, 10): some valid, some zero
        # 13 valid pixels at 1500mm, 12 zeros
        for r in range(8, 13):
            for c in range(8, 13):
                if (r + c) % 2 == 0:
                    depth[r, c] = 1500
        result, reason = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(result, 1.5, places=3)

    def test_clamps_to_image_bounds(self) -> None:
        # Pixel at corner (0, 0)
        depth = np.full((10, 10), 3000, dtype=np.uint16)
        result, reason = GoalProjectionNode._median_depth(depth, 0, 0)
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(result, 3.0, places=3)

    def test_float32_meters_uniform(self) -> None:
        # Isaac Sim bridge path: 32FC1 already in metres.
        depth = np.full((20, 20), 2.0, dtype=np.float32)
        result, reason = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_float32_inf_rejected(self) -> None:
        depth = np.full((20, 20), np.inf, dtype=np.float32)
        result, reason = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNone(result)
        self.assertEqual(reason, "all_zero_or_nan")

    def test_float32_mixed_inf_and_valid(self) -> None:
        depth = np.full((20, 20), np.inf, dtype=np.float32)
        for r in range(8, 13):
            for c in range(8, 13):
                if (r + c) % 2 == 0:
                    depth[r, c] = 1.5
        result, reason = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(result, 1.5, places=3)

    def test_float32_out_of_range(self) -> None:
        depth = np.full((20, 20), 10.0, dtype=np.float32)
        result, reason = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNone(result)
        self.assertEqual(reason, "out_of_range")

    def test_widened_max_admits_far_target(self) -> None:
        # 10 m sample — past default 6 m cap, inside a 15 m sim cap.
        depth = np.full((20, 20), 10.0, dtype=np.float32)
        result, reason = GoalProjectionNode._median_depth(
            depth, 10, 10, depth_min=0.3, depth_max=15.0,
        )
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(result, 10.0, places=3)

    def test_reason_split_distinguishes_failures(self) -> None:
        # Zeros → "all_zero_or_nan" (bridge / sky); finite-but-far →
        # "out_of_range" (sensor reach). Two different rejection paths,
        # two different reason codes — operator can tell them apart.
        zeros = np.zeros((20, 20), dtype=np.uint16)
        far = np.full((20, 20), 10000, dtype=np.uint16)
        _, zero_reason = GoalProjectionNode._median_depth(zeros, 10, 10)
        _, far_reason = GoalProjectionNode._median_depth(far, 10, 10)
        self.assertEqual(zero_reason, "all_zero_or_nan")
        self.assertEqual(far_reason, "out_of_range")
        self.assertNotEqual(zero_reason, far_reason)


class TestParseDepthEnv(unittest.TestCase):
    """Tests for the env-var parser used by goal-projection startup."""

    def test_unset_returns_default_silently(self) -> None:
        value, kind = _parse_depth_env(None, 6.0)
        self.assertEqual(value, 6.0)
        self.assertIsNone(kind)

    def test_empty_string_returns_default_silently(self) -> None:
        value, kind = _parse_depth_env("", 6.0)
        self.assertEqual(value, 6.0)
        self.assertIsNone(kind)

    def test_positive_override_accepted(self) -> None:
        value, kind = _parse_depth_env("15.0", 6.0)
        self.assertAlmostEqual(value, 15.0, places=4)
        self.assertEqual(kind, "override")

    def test_non_numeric_falls_back_to_default(self) -> None:
        value, kind = _parse_depth_env("not-a-number", 6.0)
        self.assertEqual(value, 6.0)
        self.assertEqual(kind, "non_numeric")

    def test_zero_falls_back_to_default(self) -> None:
        value, kind = _parse_depth_env("0", 6.0)
        self.assertEqual(value, 6.0)
        self.assertEqual(kind, "non_positive")

    def test_negative_falls_back_to_default(self) -> None:
        value, kind = _parse_depth_env("-3.0", 6.0)
        self.assertEqual(value, 6.0)
        self.assertEqual(kind, "non_positive")


class TestNodeDepthRangeResolution(unittest.TestCase):
    """Verify a fresh GoalProjectionNode picks up env overrides at __init__.

    Avoids spinning up rclpy by exercising only the resolution method on
    a stub instance. This is the same code path ``__init__`` runs.
    """

    @staticmethod
    def _make_stub():
        # Build a shim that has the bound method and a no-op logger.
        stub = SimpleNamespace()
        stub.get_logger = lambda: SimpleNamespace(
            info=lambda *_a, **_kw: None,
            warning=lambda *_a, **_kw: None,
        )
        stub._resolve_depth_env = GoalProjectionNode._resolve_depth_env.__get__(stub)
        return stub

    def test_default_preserved_when_env_unset(self) -> None:
        stub = self._make_stub()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV_DEPTH_MAX, None)
            os.environ.pop(_ENV_DEPTH_MIN, None)
            depth_max = stub._resolve_depth_env(_ENV_DEPTH_MAX, _DEPTH_MAX_M, "max")
            depth_min = stub._resolve_depth_env(_ENV_DEPTH_MIN, _DEPTH_MIN_M, "min")
        self.assertAlmostEqual(depth_max, _DEPTH_MAX_M, places=4)
        self.assertAlmostEqual(depth_min, _DEPTH_MIN_M, places=4)

    def test_override_applied(self) -> None:
        stub = self._make_stub()
        with mock.patch.dict(
            os.environ,
            {_ENV_DEPTH_MAX: "15.0", _ENV_DEPTH_MIN: "0.5"},
        ):
            depth_max = stub._resolve_depth_env(_ENV_DEPTH_MAX, _DEPTH_MAX_M, "max")
            depth_min = stub._resolve_depth_env(_ENV_DEPTH_MIN, _DEPTH_MIN_M, "min")
        self.assertAlmostEqual(depth_max, 15.0, places=4)
        self.assertAlmostEqual(depth_min, 0.5, places=4)

    def test_non_numeric_falls_back_with_warning(self) -> None:
        warnings: list[str] = []
        stub = SimpleNamespace()
        stub.get_logger = lambda: SimpleNamespace(
            info=lambda *_a, **_kw: None,
            warning=lambda msg: warnings.append(msg),
        )
        stub._resolve_depth_env = GoalProjectionNode._resolve_depth_env.__get__(stub)
        with mock.patch.dict(os.environ, {_ENV_DEPTH_MAX: "fifteen"}):
            value = stub._resolve_depth_env(_ENV_DEPTH_MAX, _DEPTH_MAX_M, "max")
        self.assertAlmostEqual(value, _DEPTH_MAX_M, places=4)
        self.assertTrue(warnings, "expected a warning on non-numeric override")
        self.assertIn("Ignoring non-numeric", warnings[0])
        self.assertIn(_ENV_DEPTH_MAX, warnings[0])

    def test_non_positive_falls_back_with_warning(self) -> None:
        warnings: list[str] = []
        stub = SimpleNamespace()
        stub.get_logger = lambda: SimpleNamespace(
            info=lambda *_a, **_kw: None,
            warning=lambda msg: warnings.append(msg),
        )
        stub._resolve_depth_env = GoalProjectionNode._resolve_depth_env.__get__(stub)
        with mock.patch.dict(os.environ, {_ENV_DEPTH_MAX: "-1.0"}):
            value = stub._resolve_depth_env(_ENV_DEPTH_MAX, _DEPTH_MAX_M, "max")
        self.assertAlmostEqual(value, _DEPTH_MAX_M, places=4)
        self.assertTrue(warnings, "expected a warning on non-positive override")
        self.assertIn("Ignoring non-positive", warnings[0])


class TestTransformPoint(unittest.TestCase):
    """Tests for _transform_point (quaternion rotation + translation)."""

    def _make_tf(self, tx=0.0, ty=0.0, tz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        return SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=tx, y=ty, z=tz),
                rotation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw),
            )
        )

    def test_identity_transform(self) -> None:
        tf = self._make_tf()
        result = GoalProjectionNode._transform_point(tf, 1.0, 2.0, 3.0)
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[1], 2.0, places=5)
        self.assertAlmostEqual(result[2], 3.0, places=5)

    def test_translation_only(self) -> None:
        tf = self._make_tf(tx=10.0, ty=20.0, tz=30.0)
        result = GoalProjectionNode._transform_point(tf, 1.0, 2.0, 3.0)
        self.assertAlmostEqual(result[0], 11.0, places=5)
        self.assertAlmostEqual(result[1], 22.0, places=5)
        self.assertAlmostEqual(result[2], 33.0, places=5)

    def test_90_degree_rotation_about_z(self) -> None:
        # 90° about Z: point (1, 0, 0) → (0, 1, 0)
        qz = math.sin(math.pi / 4)
        qw = math.cos(math.pi / 4)
        tf = self._make_tf(qz=qz, qw=qw)
        result = GoalProjectionNode._transform_point(tf, 1.0, 0.0, 0.0)
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], 1.0, places=5)
        self.assertAlmostEqual(result[2], 0.0, places=5)


class TestComputeStandoffPose(unittest.TestCase):
    """Tests for _compute_standoff_pose."""

    def _make_robot_tf(self, x=0.0, y=0.0):
        return SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=x, y=y, z=0.0),
            )
        )

    def test_standoff_along_robot_to_target(self) -> None:
        target = (5.0, 0.0, 0.0)
        robot_tf = self._make_robot_tf(x=0.0, y=0.0)
        gx, gy, yaw = GoalProjectionNode._compute_standoff_pose(target, 1.0, robot_tf)
        # Goal should be at x=4.0, y=0.0, facing +X (yaw=0)
        self.assertAlmostEqual(gx, 4.0, places=3)
        self.assertAlmostEqual(gy, 0.0, places=3)
        self.assertAlmostEqual(yaw, 0.0, places=3)

    def test_standoff_diagonal(self) -> None:
        target = (3.0, 4.0, 0.0)
        robot_tf = self._make_robot_tf(x=0.0, y=0.0)
        gx, gy, yaw = GoalProjectionNode._compute_standoff_pose(target, 1.0, robot_tf)
        # Distance robot→target = 5.0, unit vector = (0.6, 0.8)
        self.assertAlmostEqual(gx, 3.0 - 0.6, places=3)
        self.assertAlmostEqual(gy, 4.0 - 0.8, places=3)

    def test_no_robot_tf_defaults_to_x_axis(self) -> None:
        target = (2.0, 3.0, 0.0)
        gx, gy, yaw = GoalProjectionNode._compute_standoff_pose(target, 0.5, None)
        self.assertAlmostEqual(gx, 1.5, places=3)
        self.assertAlmostEqual(gy, 3.0, places=3)
        self.assertAlmostEqual(yaw, 0.0, places=3)

    def test_robot_on_target_fallback(self) -> None:
        target = (1.0, 1.0, 0.0)
        robot_tf = self._make_robot_tf(x=1.0, y=1.0)
        gx, gy, yaw = GoalProjectionNode._compute_standoff_pose(target, 0.5, robot_tf)
        # Falls back to X-axis offset
        self.assertAlmostEqual(gx, 0.5, places=3)
        self.assertAlmostEqual(gy, 1.0, places=3)
