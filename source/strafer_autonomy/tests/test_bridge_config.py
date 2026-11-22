"""Tests for strafer_lab.bridge.config — pure-Python config layer.

Runs in the pxr-free autonomy suite via the ``strafer_lab`` namespace stub installed by
:mod:`conftest`. No Isaac Sim / Kit runtime is required. The OmniGraph
builder in :mod:`strafer_lab.bridge.graph` is NOT exercised here because
it imports ``omni.graph.core``; it is smoke-tested in-process by the
``run_sim_in_the_loop.py`` launch script inside the Isaac Lab env.
"""

from __future__ import annotations

import pytest

from strafer_lab.bridge.config import (
    BridgeConfig,
    CameraStreamConfig,
    build_default_bridge_config,
)
from strafer_shared.constants import (
    FRAME_BASE_LINK,
    FRAME_D555_COLOR_OPTICAL,
    FRAME_D555_LINK,
    FRAME_ODOM,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
    TOPIC_CLOCK,
    TOPIC_CMD_VEL,
    TOPIC_COLOR_CAMERA_INFO,
    TOPIC_COLOR_IMAGE,
    TOPIC_DEPTH_CAMERA_INFO,
    TOPIC_DEPTH_IMAGE,
    TOPIC_IMU_FILTERED,
    TOPIC_JOINT_STATES,
    TOPIC_ODOM,
)


@pytest.fixture
def default_cfg() -> BridgeConfig:
    return build_default_bridge_config()


class TestDefaultBridgeTopicNames:
    def test_color_image_topic(self, default_cfg):
        assert default_cfg.color_camera.image_topic == TOPIC_COLOR_IMAGE

    def test_color_camera_info_topic(self, default_cfg):
        assert default_cfg.color_camera.camera_info_topic == TOPIC_COLOR_CAMERA_INFO

    def test_depth_image_topic(self, default_cfg):
        assert default_cfg.depth_camera.image_topic == TOPIC_DEPTH_IMAGE

    def test_depth_camera_info_topic(self, default_cfg):
        assert default_cfg.depth_camera.camera_info_topic == TOPIC_DEPTH_CAMERA_INFO

    def test_odom_topic(self, default_cfg):
        assert default_cfg.odom_topic == TOPIC_ODOM

    def test_cmd_vel_topic(self, default_cfg):
        assert default_cfg.cmd_vel_topic == TOPIC_CMD_VEL

    def test_clock_topic(self, default_cfg):
        assert default_cfg.clock_topic == TOPIC_CLOCK

    def test_joint_states_topic(self, default_cfg):
        # Same topic the real driver publishes, so the inference obs pipeline
        # reconstructs wheel-FK body velocity from sim or hardware unchanged.
        assert default_cfg.joint_states_topic == TOPIC_JOINT_STATES
        assert default_cfg.joint_states_topic == "/strafer/joint_states"

    def test_imu_topic(self, default_cfg):
        # Same topic imu_filter_madgwick produces on the robot.
        assert default_cfg.imu_topic == TOPIC_IMU_FILTERED
        assert default_cfg.imu_topic == "/d555/imu/filtered"


class TestDefaultBridgeFrameIds:
    def test_odom_frame(self, default_cfg):
        assert default_cfg.odom_frame_id == FRAME_ODOM

    def test_base_frame(self, default_cfg):
        assert default_cfg.base_frame_id == FRAME_BASE_LINK

    def test_camera_mount_frame(self, default_cfg):
        assert default_cfg.camera_mount_frame_id == FRAME_D555_LINK

    def test_imu_frame(self, default_cfg):
        assert default_cfg.imu_frame_id == FRAME_D555_LINK

    def test_both_cameras_use_optical_frame(self, default_cfg):
        """Color and depth both publish in the D555 color optical frame —
        matches the real robot where the aligned depth stream is warped
        into the color optical frame by the RealSense driver."""
        assert default_cfg.color_camera.frame_id == FRAME_D555_COLOR_OPTICAL
        assert default_cfg.depth_camera.frame_id == FRAME_D555_COLOR_OPTICAL


class TestDefaultBridgePrimPaths:
    def test_chassis_prim_under_default_env(self, default_cfg):
        assert default_cfg.chassis_prim_path == (
            "/World/envs/env_0/Robot/strafer/body_link"
        )

    def test_color_camera_prim_matches_d555_perception(self, default_cfg):
        """Must match D555_PERCEPTION_CAMERA_PRIM_PATH in d555_cfg.py
        so the bridge targets the prim the TiledCameraCfg actually spawns."""
        assert default_cfg.color_camera.camera_prim_path.endswith(
            "/body_link/d555_camera_perception"
        )

    def test_depth_and_color_share_camera_prim(self, default_cfg):
        """Single USD camera prim feeds both streams — same optical center."""
        assert (
            default_cfg.color_camera.camera_prim_path
            == default_cfg.depth_camera.camera_prim_path
        )

    def test_tf_extra_targets_include_camera_prim(self, default_cfg):
        assert default_cfg.color_camera.camera_prim_path in default_cfg.tf_extra_target_prims


class TestEnvNamespaceSubstitution:
    def test_custom_env_ns_propagates_to_chassis_prim(self):
        cfg = build_default_bridge_config(env_ns="/World/envs/env_7")
        assert cfg.chassis_prim_path == "/World/envs/env_7/Robot/strafer/body_link"

    def test_custom_env_ns_propagates_to_camera_prim(self):
        cfg = build_default_bridge_config(env_ns="/World/envs/env_7")
        assert cfg.color_camera.camera_prim_path.startswith("/World/envs/env_7/")

    def test_custom_graph_path(self):
        cfg = build_default_bridge_config(graph_path="/World/AltBridge")
        assert cfg.graph_path == "/World/AltBridge"


class TestCameraStreamTypes:
    def test_color_is_rgb(self, default_cfg):
        assert default_cfg.color_camera.stream_type == "rgb"

    def test_depth_is_depth(self, default_cfg):
        assert default_cfg.depth_camera.stream_type == "depth"


class TestCameraStreamResolution:
    """Published-image resolution must match the perception-camera spec.

    ``StraferCameraAsyncPublisher`` reads the GPU render product the
    ``TiledCameraCfg`` spawns at ``PERCEPTION_WIDTH × PERCEPTION_HEIGHT``
    and uses :attr:`CameraStreamConfig.width` / :attr:`height` to size the
    published ``sensor_msgs/Image`` and to derive the ``CameraInfo`` fx /
    fy. A drift between the two would publish a wrong ``CameraInfo`` even
    though the underlying render product is correct.
    """

    def test_color_width_matches_perception_constant(self, default_cfg):
        assert default_cfg.color_camera.width == PERCEPTION_WIDTH

    def test_color_height_matches_perception_constant(self, default_cfg):
        assert default_cfg.color_camera.height == PERCEPTION_HEIGHT

    def test_depth_width_matches_perception_constant(self, default_cfg):
        assert default_cfg.depth_camera.width == PERCEPTION_WIDTH

    def test_depth_height_matches_perception_constant(self, default_cfg):
        assert default_cfg.depth_camera.height == PERCEPTION_HEIGHT

    def test_color_and_depth_share_resolution(self, default_cfg):
        """Both streams come off the same camera prim — depth-aligned-
        to-color requires identical pixel grids."""
        assert default_cfg.color_camera.width == default_cfg.depth_camera.width
        assert default_cfg.color_camera.height == default_cfg.depth_camera.height

    def test_resolution_is_640x360_d555_native(self, default_cfg):
        """Locked to the real D555 native rate. Lowering it sim-side
        would introduce a sim-to-real gap; raising it would invalidate
        VLM/RTAB-Map tunings."""
        assert default_cfg.color_camera.width == 640
        assert default_cfg.color_camera.height == 360


class TestFrozenDataclass:
    def test_bridge_config_is_frozen(self, default_cfg):
        with pytest.raises((AttributeError, TypeError)):
            default_cfg.graph_path = "/World/Mutated"  # type: ignore[misc]

    def test_camera_stream_config_is_frozen(self, default_cfg):
        with pytest.raises((AttributeError, TypeError)):
            default_cfg.color_camera.image_topic = "/hacked"  # type: ignore[misc]


class TestCmdWatchdogWindow:
    """The bridge's stop-on-silence window is the shared command-watchdog window, in sim seconds."""

    def test_defaults_from_the_shared_constant(self, default_cfg):
        import ast
        import inspect

        from strafer_lab.bridge import config
        from strafer_shared.constants import CMD_WATCHDOG_TIMEOUT_S

        assert default_cfg.cmd_watchdog_sim_s == CMD_WATCHDOG_TIMEOUT_S
        # a literal that happens to agree would pass the value: the field's default must name the constant
        fields = [n for n in ast.walk(ast.parse(inspect.getsource(config.BridgeConfig)))
                  if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "cmd_watchdog_sim_s"]
        assert len(fields) == 1 and isinstance(fields[0].value, ast.Name)
        assert fields[0].value.id == "CMD_WATCHDOG_TIMEOUT_S"

