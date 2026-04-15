"""Tests for strafer_lab.bridge.config — pure-Python config layer.

Runs in ``.venv_vlm`` via the ``strafer_lab`` namespace stub installed by
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
    TOPIC_CMD_VEL,
    TOPIC_COLOR_CAMERA_INFO,
    TOPIC_COLOR_IMAGE,
    TOPIC_DEPTH_CAMERA_INFO,
    TOPIC_DEPTH_IMAGE,
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


class TestDefaultBridgeFrameIds:
    def test_odom_frame(self, default_cfg):
        assert default_cfg.odom_frame_id == FRAME_ODOM

    def test_base_frame(self, default_cfg):
        assert default_cfg.base_frame_id == FRAME_BASE_LINK

    def test_camera_mount_frame(self, default_cfg):
        assert default_cfg.camera_mount_frame_id == FRAME_D555_LINK

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


class TestFrozenDataclass:
    def test_bridge_config_is_frozen(self, default_cfg):
        with pytest.raises((AttributeError, TypeError)):
            default_cfg.graph_path = "/World/Mutated"  # type: ignore[misc]

    def test_camera_stream_config_is_frozen(self, default_cfg):
        with pytest.raises((AttributeError, TypeError)):
            default_cfg.color_camera.image_topic = "/hacked"  # type: ignore[misc]
