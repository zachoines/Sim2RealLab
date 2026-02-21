"""Cross-package integration tests.

Covers module imports and shared constants consistency.
"""

import pytest


class TestImports:
    def test_import_interface(self):
        from strafer_driver.roboclaw_interface import RoboClawInterface, _crc16
        assert RoboClawInterface is not None

    def test_import_node(self):
        from strafer_driver.roboclaw_node import RoboClawNode
        assert RoboClawNode is not None

    def test_import_exceptions(self):
        from strafer_driver.roboclaw_interface import (
            RoboClawError,
            RoboClawChecksumError,
            RoboClawTimeoutError,
        )
        assert issubclass(RoboClawChecksumError, RoboClawError)
        assert issubclass(RoboClawTimeoutError, RoboClawError)


class TestSharedConstants:
    def test_roboclaw_addresses(self):
        from strafer_shared.constants import (
            ROBOCLAW_FRONT_ADDRESS,
            ROBOCLAW_REAR_ADDRESS,
        )
        assert ROBOCLAW_FRONT_ADDRESS == 0x80
        assert ROBOCLAW_REAR_ADDRESS == 0x81

    def test_roboclaw_ports(self):
        from strafer_shared.constants import (
            ROBOCLAW_FRONT_PORT,
            ROBOCLAW_REAR_PORT,
        )
        assert "roboclaw" in ROBOCLAW_FRONT_PORT
        assert "roboclaw" in ROBOCLAW_REAR_PORT

    def test_wheel_joint_names_count(self):
        from strafer_shared.constants import WHEEL_JOINT_NAMES
        assert len(WHEEL_JOINT_NAMES) == 4

    def test_encoder_conversion_inverse(self):
        """RADIANS_TO_ENCODER_TICKS * ENCODER_TICKS_TO_RADIANS ≈ 1.0."""
        from strafer_shared.constants import (
            RADIANS_TO_ENCODER_TICKS,
            ENCODER_TICKS_TO_RADIANS,
        )
        assert abs(RADIANS_TO_ENCODER_TICKS * ENCODER_TICKS_TO_RADIANS - 1.0) < 1e-10

    def test_max_wheel_vel_positive(self):
        from strafer_shared.constants import MAX_WHEEL_ANGULAR_VEL
        assert MAX_WHEEL_ANGULAR_VEL > 0

    def test_wheel_axis_signs_pattern(self):
        """Signs follow [-1, 1, -1, 1] for [FL, FR, RL, RR]."""
        from strafer_shared.constants import WHEEL_AXIS_SIGNS
        assert list(WHEEL_AXIS_SIGNS) == [-1.0, 1.0, -1.0, 1.0]

    def test_baud_rate(self):
        from strafer_shared.constants import ROBOCLAW_BAUD_RATE
        assert ROBOCLAW_BAUD_RATE == 115200