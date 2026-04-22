"""Unit tests for GoalProjectionNode geometry helpers."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from strafer_perception.goal_projection_node import GoalProjectionNode


class TestMedianDepth(unittest.TestCase):
    """Tests for _median_depth (5x5 kernel, mm→m, range filter)."""

    def test_uniform_depth_returns_correct_meters(self) -> None:
        # 20x20 frame, all 2000mm = 2.0m
        depth = np.full((20, 20), 2000, dtype=np.uint16)
        result = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_zero_depth_returns_none(self) -> None:
        depth = np.zeros((20, 20), dtype=np.uint16)
        result = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNone(result)

    def test_out_of_range_depth_returns_none(self) -> None:
        # All pixels at 10000mm = 10m (above _DEPTH_MAX_M=6)
        depth = np.full((20, 20), 10000, dtype=np.uint16)
        result = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNone(result)

    def test_mixed_depth_takes_median_of_valid(self) -> None:
        depth = np.zeros((20, 20), dtype=np.uint16)
        # Fill 5x5 patch around (10, 10): some valid, some zero
        # 13 valid pixels at 1500mm, 12 zeros
        for r in range(8, 13):
            for c in range(8, 13):
                if (r + c) % 2 == 0:
                    depth[r, c] = 1500
        result = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1.5, places=3)

    def test_clamps_to_image_bounds(self) -> None:
        # Pixel at corner (0, 0)
        depth = np.full((10, 10), 3000, dtype=np.uint16)
        result = GoalProjectionNode._median_depth(depth, 0, 0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 3.0, places=3)

    def test_float32_meters_uniform(self) -> None:
        # Isaac Sim bridge path: 32FC1 already in metres.
        depth = np.full((20, 20), 2.0, dtype=np.float32)
        result = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 2.0, places=3)

    def test_float32_inf_rejected(self) -> None:
        depth = np.full((20, 20), np.inf, dtype=np.float32)
        result = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNone(result)

    def test_float32_mixed_inf_and_valid(self) -> None:
        depth = np.full((20, 20), np.inf, dtype=np.float32)
        for r in range(8, 13):
            for c in range(8, 13):
                if (r + c) % 2 == 0:
                    depth[r, c] = 1.5
        result = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1.5, places=3)

    def test_float32_out_of_range(self) -> None:
        depth = np.full((20, 20), 10.0, dtype=np.float32)
        result = GoalProjectionNode._median_depth(depth, 10, 10)
        self.assertIsNone(result)


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
