# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Pure config checks for the shared D555 camera mount."""

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferSceneCfg,
    StraferSceneCfg_Infinigen,
    StraferSceneCfg_ProcRoom,
)
from test.sensors.depth_noise.scene_cfg import DepthNoiseTestSceneCfg, ENV_SPACING


def _camera_signature(scene_cfg) -> dict:
    camera = scene_cfg.d555_camera
    spawn = camera.spawn
    offset = camera.offset
    return {
        "prim_path": camera.prim_path,
        "update_period": camera.update_period,
        "height": camera.height,
        "width": camera.width,
        "focal_length": spawn.focal_length,
        "horizontal_aperture": spawn.horizontal_aperture,
        "clipping_range": spawn.clipping_range,
        "offset_pos": offset.pos,
        "offset_rot": offset.rot,
        "offset_convention": offset.convention,
    }


def _imu_signature(scene_cfg) -> dict:
    imu = scene_cfg.d555_imu
    offset = imu.offset
    return {
        "prim_path": imu.prim_path,
        "update_period": imu.update_period,
        "offset_pos": offset.pos,
        "offset_rot": offset.rot,
    }


def test_depth_noise_scene_camera_matches_navigation_scenes():
    expected = _camera_signature(StraferSceneCfg())

    assert _camera_signature(StraferSceneCfg_Infinigen()) == expected
    assert _camera_signature(StraferSceneCfg_ProcRoom()) == expected
    assert _camera_signature(DepthNoiseTestSceneCfg(num_envs=1, env_spacing=ENV_SPACING)) == expected


def test_depth_noise_scene_imu_matches_navigation_scenes():
    expected = _imu_signature(StraferSceneCfg())

    assert _imu_signature(StraferSceneCfg_Infinigen()) == expected
    assert _imu_signature(StraferSceneCfg_ProcRoom()) == expected
    assert _imu_signature(DepthNoiseTestSceneCfg(num_envs=1, env_spacing=ENV_SPACING)) == expected
