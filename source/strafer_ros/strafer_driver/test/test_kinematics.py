"""Tests for strafer_shared.mecanum_kinematics and related constants.

Covers kinematic conversions, encoder unit math, matrix properties.
"""

import math

import numpy as np
import pytest


class TestKinematics:
    def test_zero_twist(self):
        from strafer_shared.mecanum_kinematics import (
            twist_to_wheel_velocities,
            wheel_vels_to_ticks_per_sec,
        )
        wheel_vels = twist_to_wheel_velocities(0.0, 0.0, 0.0)
        assert np.allclose(wheel_vels, 0.0)
        ticks = wheel_vels_to_ticks_per_sec(wheel_vels)
        assert np.allclose(ticks, 0.0)

    def test_forward_all_wheels_nonzero(self):
        from strafer_shared.mecanum_kinematics import (
            twist_to_wheel_velocities,
            wheel_vels_to_ticks_per_sec,
        )
        ticks = wheel_vels_to_ticks_per_sec(
            twist_to_wheel_velocities(0.5, 0.0, 0.0)
        )
        for i, t in enumerate(ticks):
            assert t != 0.0, f"Wheel {i} has zero ticks for forward motion"

    def test_pure_strafe_left(self):
        """Pure left strafe: front pair same sign, rear pair opposite."""
        from strafer_shared.mecanum_kinematics import twist_to_wheel_velocities
        vels = twist_to_wheel_velocities(0.0, 0.5, 0.0)
        assert np.isclose(vels[0], vels[1], atol=0.01)
        assert np.isclose(vels[2], vels[3], atol=0.01)
        assert np.sign(vels[0]) != np.sign(vels[2])

    def test_pure_rotation_ccw(self):
        """Pure CCW rotation: all magnitudes equal."""
        from strafer_shared.mecanum_kinematics import twist_to_wheel_velocities
        vels = twist_to_wheel_velocities(0.0, 0.0, 1.0)
        magnitudes = np.abs(vels)
        assert np.allclose(magnitudes, magnitudes[0], atol=0.01)

    def test_roundtrip_forward(self):
        from strafer_shared.mecanum_kinematics import (
            twist_to_wheel_velocities,
            wheel_vels_to_ticks_per_sec,
            encoder_ticks_to_body_velocity,
        )
        vx_in, vy_in, omega_in = 0.5, 0.0, 0.0
        ticks = wheel_vels_to_ticks_per_sec(
            twist_to_wheel_velocities(vx_in, vy_in, omega_in)
        )
        vx_out, vy_out, omega_out = encoder_ticks_to_body_velocity(ticks)
        assert abs(vx_out - vx_in) < 0.01
        assert abs(vy_out - vy_in) < 0.01
        assert abs(omega_out - omega_in) < 0.01

    def test_roundtrip_strafe(self):
        from strafer_shared.mecanum_kinematics import (
            twist_to_wheel_velocities,
            wheel_vels_to_ticks_per_sec,
            encoder_ticks_to_body_velocity,
        )
        vx_in, vy_in, omega_in = 0.0, 0.4, 0.0
        ticks = wheel_vels_to_ticks_per_sec(
            twist_to_wheel_velocities(vx_in, vy_in, omega_in)
        )
        vx_out, vy_out, omega_out = encoder_ticks_to_body_velocity(ticks)
        assert abs(vx_out - vx_in) < 0.01
        assert abs(vy_out - vy_in) < 0.01
        assert abs(omega_out - omega_in) < 0.01

    def test_roundtrip_combined(self):
        from strafer_shared.mecanum_kinematics import (
            twist_to_wheel_velocities,
            wheel_vels_to_ticks_per_sec,
            encoder_ticks_to_body_velocity,
        )
        vx_in, vy_in, omega_in = 0.3, 0.2, 0.5
        ticks = wheel_vels_to_ticks_per_sec(
            twist_to_wheel_velocities(vx_in, vy_in, omega_in)
        )
        vx_out, vy_out, omega_out = encoder_ticks_to_body_velocity(ticks)
        assert abs(vx_out - vx_in) < 0.01
        assert abs(vy_out - vy_in) < 0.01
        assert abs(omega_out - omega_in) < 0.01

    def test_normalized_clipped(self):
        """Values beyond [-1,1] are clipped in normalized_to_wheel_velocities."""
        from strafer_shared.mecanum_kinematics import normalized_to_wheel_velocities
        from strafer_shared.constants import MAX_WHEEL_ANGULAR_VEL
        vels = normalized_to_wheel_velocities(10.0, 10.0, 10.0)
        assert np.all(np.abs(vels) <= MAX_WHEEL_ANGULAR_VEL + 0.01)

    def test_ticks_conversion_magnitude(self):
        """1 rad/s should convert to ~85.57 ticks/s."""
        from strafer_shared.mecanum_kinematics import wheel_vels_to_ticks_per_sec
        from strafer_shared.constants import RADIANS_TO_ENCODER_TICKS
        ones = np.ones(4)
        ticks = wheel_vels_to_ticks_per_sec(ones)
        assert np.allclose(ticks, RADIANS_TO_ENCODER_TICKS, atol=0.01)


class TestEncoderConversions:
    def test_ticks_to_radians_one_revolution(self):
        """537.7 ticks = 2*pi radians."""
        from strafer_shared.constants import (
            ENCODER_PPR_OUTPUT_SHAFT,
            ENCODER_TICKS_TO_RADIANS,
        )
        rad = ENCODER_PPR_OUTPUT_SHAFT * ENCODER_TICKS_TO_RADIANS
        assert abs(rad - 2.0 * math.pi) < 1e-6

    def test_radians_to_ticks_one_revolution(self):
        """2*pi radians = 537.7 ticks."""
        from strafer_shared.constants import (
            ENCODER_PPR_OUTPUT_SHAFT,
            RADIANS_TO_ENCODER_TICKS,
        )
        ticks = 2.0 * math.pi * RADIANS_TO_ENCODER_TICKS
        assert abs(ticks - ENCODER_PPR_OUTPUT_SHAFT) < 1e-6

    def test_max_speed_ticks(self):
        """Max wheel angular velocity in ticks/s should be ~2796."""
        from strafer_shared.constants import (
            MAX_WHEEL_ANGULAR_VEL,
            RADIANS_TO_ENCODER_TICKS,
        )
        max_ticks = MAX_WHEEL_ANGULAR_VEL * RADIANS_TO_ENCODER_TICKS
        assert abs(max_ticks - 2796) < 5


class TestKinematicMatrixProperties:
    def test_forward_inverse_identity(self):
        """K_inv @ K should approximate a 3x3 identity (pseudo-inverse property)."""
        from strafer_shared.mecanum_kinematics import (
            KINEMATIC_MATRIX,
            INVERSE_KINEMATIC_MATRIX,
        )
        product = INVERSE_KINEMATIC_MATRIX @ KINEMATIC_MATRIX
        assert np.allclose(product, np.eye(3), atol=1e-10)

    def test_kinematic_matrix_shape(self):
        from strafer_shared.mecanum_kinematics import (
            KINEMATIC_MATRIX,
            INVERSE_KINEMATIC_MATRIX,
        )
        assert KINEMATIC_MATRIX.shape == (4, 3)
        assert INVERSE_KINEMATIC_MATRIX.shape == (3, 4)
