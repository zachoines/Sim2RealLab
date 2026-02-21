"""Tests for strafer_driver.roboclaw_node helpers.

Covers odometry integration math and yaw-to-quaternion conversion.
"""

import math

import numpy as np
import pytest


class TestOdometryMath:
    def test_stationary_no_drift(self):
        """Zero wheel speeds produce zero body velocity."""
        from strafer_shared.mecanum_kinematics import encoder_ticks_to_body_velocity
        vx, vy, omega = encoder_ticks_to_body_velocity(np.zeros(4))
        assert vx == 0.0
        assert vy == 0.0
        assert omega == 0.0

    def test_forward_integration(self):
        """Driving forward at 0.5 m/s for 1s should move x by ~0.5m."""
        from strafer_shared.mecanum_kinematics import (
            twist_to_wheel_velocities,
            wheel_vels_to_ticks_per_sec,
            encoder_ticks_to_body_velocity,
        )
        ticks = wheel_vels_to_ticks_per_sec(
            twist_to_wheel_velocities(0.5, 0.0, 0.0)
        )
        vx, vy, omega = encoder_ticks_to_body_velocity(ticks)

        dt = 1.0
        x = vx * dt
        y = vy * dt
        theta = omega * dt
        assert abs(x - 0.5) < 0.01
        assert abs(y) < 0.01
        assert abs(theta) < 0.01

    def test_rotation_integration(self):
        """Rotating at 1 rad/s for 1s should accumulate ~1 rad."""
        from strafer_shared.mecanum_kinematics import (
            twist_to_wheel_velocities,
            wheel_vels_to_ticks_per_sec,
            encoder_ticks_to_body_velocity,
        )
        ticks = wheel_vels_to_ticks_per_sec(
            twist_to_wheel_velocities(0.0, 0.0, 1.0)
        )
        vx, vy, omega = encoder_ticks_to_body_velocity(ticks)

        dt = 1.0
        theta = omega * dt
        assert abs(theta - 1.0) < 0.01

    def test_yaw_to_quaternion(self):
        """Verify yaw->quaternion conversion."""
        from strafer_driver.roboclaw_node import _yaw_to_quaternion

        # yaw = 0 -> identity rotation
        q = _yaw_to_quaternion(0.0)
        assert abs(q.w - 1.0) < 1e-6
        assert abs(q.z) < 1e-6

        # yaw = pi/2 -> 90° CCW
        q = _yaw_to_quaternion(math.pi / 2)
        assert abs(q.z - math.sin(math.pi / 4)) < 1e-6
        assert abs(q.w - math.cos(math.pi / 4)) < 1e-6

        # yaw = pi -> 180°
        q = _yaw_to_quaternion(math.pi)
        assert abs(q.z - 1.0) < 1e-6
        assert abs(q.w) < 1e-6
