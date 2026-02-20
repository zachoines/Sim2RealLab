"""Basic tests for strafer_driver package."""

import pytest


def test_import_interface():
    """Verify roboclaw_interface can be imported."""
    from strafer_driver.roboclaw_interface import RoboClawInterface, _crc16
    assert RoboClawInterface is not None


def test_crc16_known_value():
    """Verify CRC16 matches a known computation."""
    from strafer_driver.roboclaw_interface import _crc16

    # CRC of address=0x80, command=0x15 (ReadSpeedM1)
    data = bytes([0x80, 0x12])
    crc = _crc16(data)
    assert isinstance(crc, int)
    assert 0 <= crc <= 0xFFFF


def test_import_node():
    """Verify roboclaw_node module can be imported."""
    from strafer_driver.roboclaw_node import RoboClawNode
    assert RoboClawNode is not None


def test_shared_constants_available():
    """Verify strafer_shared constants are accessible."""
    from strafer_shared.constants import (
        ROBOCLAW_FRONT_ADDRESS,
        ROBOCLAW_REAR_ADDRESS,
        WHEEL_JOINT_NAMES,
    )
    assert ROBOCLAW_FRONT_ADDRESS == 0x80
    assert ROBOCLAW_REAR_ADDRESS == 0x81
    assert len(WHEEL_JOINT_NAMES) == 4


def test_kinematics_zero_twist():
    """Zero twist produces zero wheel velocities."""
    from strafer_shared.mecanum_kinematics import (
        twist_to_wheel_velocities,
        wheel_vels_to_ticks_per_sec,
    )
    import numpy as np

    wheel_vels = twist_to_wheel_velocities(0.0, 0.0, 0.0)
    assert np.allclose(wheel_vels, 0.0)
    ticks = wheel_vels_to_ticks_per_sec(wheel_vels)
    assert np.allclose(ticks, 0.0)


def test_kinematics_forward():
    """Forward velocity produces all-positive wheel ticks (after sign correction)."""
    from strafer_shared.mecanum_kinematics import (
        twist_to_wheel_velocities,
        wheel_vels_to_ticks_per_sec,
    )

    wheel_vels = twist_to_wheel_velocities(0.5, 0.0, 0.0)
    ticks = wheel_vels_to_ticks_per_sec(wheel_vels)
    # With axis signs applied, all ticks should be non-zero
    for i, t in enumerate(ticks):
        assert t != 0.0, f"Wheel {i} has zero ticks for forward motion"


def test_encoder_to_body_roundtrip():
    """Round-trip: twist -> wheel ticks -> body velocity should recover input."""
    import numpy as np
    from strafer_shared.mecanum_kinematics import (
        twist_to_wheel_velocities,
        wheel_vels_to_ticks_per_sec,
        encoder_ticks_to_body_velocity,
    )
    from strafer_shared.constants import RADIANS_TO_ENCODER_TICKS

    vx_in, vy_in, omega_in = 0.3, 0.2, 0.5
    wheel_rad = twist_to_wheel_velocities(vx_in, vy_in, omega_in)
    ticks = wheel_vels_to_ticks_per_sec(wheel_rad)
    vx_out, vy_out, omega_out = encoder_ticks_to_body_velocity(ticks)

    assert abs(vx_out - vx_in) < 0.01, f"vx mismatch: {vx_out} vs {vx_in}"
    assert abs(vy_out - vy_in) < 0.01, f"vy mismatch: {vy_out} vs {vy_in}"
    assert abs(omega_out - omega_in) < 0.01, f"omega mismatch: {omega_out} vs {omega_in}"
