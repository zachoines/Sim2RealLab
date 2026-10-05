"""timestamp_fixer's default mode: stamps pass through unless a launch asks otherwise."""

import pytest


@pytest.fixture
def fixer():
    import rclpy

    from strafer_perception.timestamp_fixer import TimestampFixer

    rclpy.init()
    try:
        node = TimestampFixer()
        try:
            yield node
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


class TestTimestampFixerDefault:

    def test_default_passes_stamps_through(self, fixer):
        # A bare `ros2 run` or a launch without the pin must not restamp: restamping gives
        # an image and its camera_info different stamps, which breaks exact synchronizers.
        assert fixer.get_parameter("restamp").value is False
        assert fixer._restamp is False
