# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Strafer D555 perception camera configuration.

These tests verify that ``make_d555_perception_camera_cfg()`` returns a
``TiledCameraCfg`` with the correct 640x360 resolution, RGB + depth data
types, distinct prim path from the policy camera, and the same physical
parameters (focal length, aperture, mount offset, clipping range, update
rate) as the policy camera — so a single physical D555 at deployment
serves both pipelines without recalibration.

Isaac Sim is launched by the root ``test/conftest.py``; these tests only
construct dataclasses and inspect their attributes. They do not instantiate
a full env.
"""

# --- Imports (Isaac Sim launched by root conftest.py) ---

import pytest

from strafer_shared.constants import (
    CAMERA_UPDATE_PERIOD_S,
    D555_FOCAL_LENGTH_MM,
    D555_HORIZONTAL_APERTURE_MM,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
)

from strafer_lab.tasks.navigation.d555_cfg import (
    D555_CAMERA_OFFSET,
    D555_CAMERA_PRIM_PATH,
    D555_CAMERA_ROT_ROS,
    D555_PERCEPTION_CAMERA_PRIM_PATH,
    D555_PERCEPTION_DATA_TYPES,
    make_d555_camera_cfg,
    make_d555_perception_camera_cfg,
)


# =============================================================================
# Module-level constants
# =============================================================================


class TestModuleConstants:
    """Verify the module-level constants match the design doc spec."""

    def test_perception_resolution_is_640x360(self):
        assert PERCEPTION_WIDTH == 640
        assert PERCEPTION_HEIGHT == 360

    def test_perception_data_types_include_rgb_and_depth(self):
        assert "rgb" in D555_PERCEPTION_DATA_TYPES
        assert "distance_to_image_plane" in D555_PERCEPTION_DATA_TYPES

    def test_perception_prim_path_is_distinct_from_policy(self):
        assert D555_PERCEPTION_CAMERA_PRIM_PATH != D555_CAMERA_PRIM_PATH
        assert D555_PERCEPTION_CAMERA_PRIM_PATH.endswith("/d555_camera_perception")
        assert D555_CAMERA_PRIM_PATH.endswith("/d555_camera")

    def test_perception_prim_path_shares_body_link_ancestor(self):
        """Both cameras mount on body_link so TF transforms stay aligned."""
        assert "body_link" in D555_PERCEPTION_CAMERA_PRIM_PATH
        assert "body_link" in D555_CAMERA_PRIM_PATH


# =============================================================================
# make_d555_perception_camera_cfg()
# =============================================================================


@pytest.fixture(scope="module")
def perception_cfg():
    return make_d555_perception_camera_cfg()


@pytest.fixture(scope="module")
def policy_cfg():
    return make_d555_camera_cfg(data_types=("rgb", "distance_to_image_plane"))


class TestPerceptionCameraResolution:
    def test_width(self, perception_cfg):
        assert perception_cfg.width == 640

    def test_height(self, perception_cfg):
        assert perception_cfg.height == 360

    def test_resolution_differs_from_policy(self, perception_cfg, policy_cfg):
        assert perception_cfg.width != policy_cfg.width
        assert perception_cfg.height != policy_cfg.height
        assert perception_cfg.width > policy_cfg.width
        assert perception_cfg.height > policy_cfg.height


class TestPerceptionCameraDataTypes:
    def test_rgb_present(self, perception_cfg):
        assert "rgb" in perception_cfg.data_types

    def test_depth_present(self, perception_cfg):
        assert "distance_to_image_plane" in perception_cfg.data_types

    def test_data_types_match_module_constant(self, perception_cfg):
        assert tuple(perception_cfg.data_types) == D555_PERCEPTION_DATA_TYPES


class TestPerceptionCameraPrimPath:
    def test_prim_path_matches_module_constant(self, perception_cfg):
        assert perception_cfg.prim_path == D555_PERCEPTION_CAMERA_PRIM_PATH

    def test_prim_path_distinct_from_policy(self, perception_cfg, policy_cfg):
        assert perception_cfg.prim_path != policy_cfg.prim_path


class TestPerceptionCameraPhysicalParams:
    """Both cameras simulate the same physical D555 — intrinsics must match
    so a single real-world calibration transfers to both pipelines."""

    def test_focal_length_matches_policy(self, perception_cfg, policy_cfg):
        assert perception_cfg.spawn.focal_length == policy_cfg.spawn.focal_length
        assert perception_cfg.spawn.focal_length == D555_FOCAL_LENGTH_MM

    def test_horizontal_aperture_matches_policy(self, perception_cfg, policy_cfg):
        assert (
            perception_cfg.spawn.horizontal_aperture
            == policy_cfg.spawn.horizontal_aperture
        )
        assert (
            perception_cfg.spawn.horizontal_aperture
            == D555_HORIZONTAL_APERTURE_MM
        )

    def test_clipping_range_matches_policy(self, perception_cfg, policy_cfg):
        assert perception_cfg.spawn.clipping_range == policy_cfg.spawn.clipping_range

    def test_mount_offset_position_matches_policy(self, perception_cfg, policy_cfg):
        assert perception_cfg.offset.pos == policy_cfg.offset.pos
        assert perception_cfg.offset.pos == D555_CAMERA_OFFSET

    def test_mount_offset_rotation_matches_policy(self, perception_cfg, policy_cfg):
        assert perception_cfg.offset.rot == policy_cfg.offset.rot
        assert perception_cfg.offset.rot == D555_CAMERA_ROT_ROS

    def test_ros_convention(self, perception_cfg):
        assert perception_cfg.offset.convention == "ros"

    def test_update_period_matches_policy(self, perception_cfg, policy_cfg):
        assert perception_cfg.update_period == policy_cfg.update_period
        assert perception_cfg.update_period == CAMERA_UPDATE_PERIOD_S
        assert perception_cfg.update_period == pytest.approx(1.0 / 30.0)


# =============================================================================
# Scene cfg integration — StraferSceneCfg_InfinigenPerception
# =============================================================================


class TestInfinigenPerceptionSceneCfg:
    """Verify the new Infinigen perception scene wires both cameras correctly."""

    @pytest.fixture(scope="module")
    def scene_cfg(self):
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            StraferSceneCfg_InfinigenPerception,
        )
        return StraferSceneCfg_InfinigenPerception(num_envs=1, env_spacing=0.0)

    def test_has_policy_camera(self, scene_cfg):
        """Policy camera is kept so the deployed obs shape matches training."""
        assert hasattr(scene_cfg, "d555_camera")
        assert scene_cfg.d555_camera.width < 640  # policy resolution

    def test_has_perception_camera(self, scene_cfg):
        assert hasattr(scene_cfg, "d555_camera_perception")
        assert scene_cfg.d555_camera_perception.width == 640
        assert scene_cfg.d555_camera_perception.height == 360

    def test_both_cameras_use_body_link_mount(self, scene_cfg):
        """Both cameras mount on the same body_link frame."""
        assert "body_link" in scene_cfg.d555_camera.prim_path
        assert "body_link" in scene_cfg.d555_camera_perception.prim_path

    def test_cameras_have_distinct_prim_paths(self, scene_cfg):
        assert (
            scene_cfg.d555_camera.prim_path
            != scene_cfg.d555_camera_perception.prim_path
        )

    def test_has_imu(self, scene_cfg):
        assert hasattr(scene_cfg, "d555_imu")

    def test_has_scene_geometry_placeholder(self, scene_cfg):
        """scene_geometry USD path is populated at __post_init__ time."""
        assert hasattr(scene_cfg, "scene_geometry")
        assert scene_cfg.scene_geometry.prim_path == "/World/Room"


# =============================================================================
# Gym env registration
# =============================================================================


class TestPerceptionGymRegistration:
    def test_perception_play_env_registered(self):
        """The new Infinigen perception env must be importable via gym."""
        import gymnasium as gym

        # Force the task package to register its envs.
        import strafer_lab.tasks.navigation  # noqa: F401

        env_id = "Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0"
        assert env_id in gym.envs.registry, (
            f"{env_id!r} not registered. Check "
            "source/strafer_lab/strafer_lab/tasks/navigation/__init__.py"
        )

    def test_perception_env_cfg_importable(self):
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            StraferNavEnvCfg_Real_InfinigenPerception_PLAY,
        )
        assert StraferNavEnvCfg_Real_InfinigenPerception_PLAY is not None
