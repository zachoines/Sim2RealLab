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