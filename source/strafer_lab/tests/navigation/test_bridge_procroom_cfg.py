# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free checks for the ProcRoom bridge capture variant.

``Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0`` is the Bridge capture stack
(full RGB + full depth + policy depth) on a procedural-room scene instead of a
loaded Infinigen USD — it isolates the scene axis of the drive-fault matrix so
the same v1 artifact and Jetson stack can be driven against a
training-distribution scene.

These tests pin that the env registers, resolves to the CNN depth runner (same
obs profile as the Infinigen bridge), composes its two cameras + contact sensor
+ ProcRoom managers, and carries NONE of the Infinigen scene-USD /
occupancy-spawn machinery. Config construction only — no Kit / GPU (env cfgs
build in ~3s with ``LD_PRELOAD`` libgomp; see the DGX env-cfg test recipe).
"""
from __future__ import annotations

import gymnasium as gym
import pytest

# Importing the package triggers the gym.register() calls.
import strafer_lab.tasks  # noqa: F401

_TASK = "Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0"


@pytest.fixture(scope="module")
def cfg():
    from isaaclab_tasks.utils import parse_env_cfg

    return parse_env_cfg(_TASK, device="cpu", num_envs=1)


# ---------------------------------------------------------------------------
# Registration resolves to the ProcRoom bridge cfg + the depth runner
# ---------------------------------------------------------------------------


def test_registered():
    assert _TASK in gym.envs.registry


def test_resolves_to_procroom_bridge_cfg_and_depth_runner():
    kwargs = gym.envs.registry[_TASK].kwargs or {}
    assert kwargs["env_cfg_entry_point"].endswith(
        ":StraferNavCfg_BridgeAutonomy_ProcRoom"
    )
    # Depth obs profile → the CNN depth runner, exactly like the Infinigen bridge.
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":STRAFER_PPO_DEPTH_RUNNER_CFG")


# ---------------------------------------------------------------------------
# Both cameras present at the deploy resolutions
# ---------------------------------------------------------------------------


def test_policy_camera_is_80x45_with_depth(cfg):
    """The 80x45 policy camera survives pruning so the gym-dump / obs shape
    matches training (the bridge stack's depth_policy token reads it)."""
    cam = cfg.scene.d555_camera
    assert cam is not None
    assert (cam.width, cam.height) == (80, 45)
    assert "distance_to_image_plane" in list(cam.data_types)


def test_perception_camera_is_640x360_rgbd(cfg):
    """The 640x360 perception camera the bridge streams as /d555/color +
    /d555/depth (Jetson downsamples the depth; RTAB-Map needs the RGB-D pair)."""
    cam = cfg.scene.d555_camera_perception
    assert cam is not None
    assert (cam.width, cam.height) == (640, 360)
    dt = list(cam.data_types)
    assert "rgb" in dt and "distance_to_image_plane" in dt


def test_perception_camera_prim_is_bridge_streamed(cfg):
    assert cfg.scene.d555_camera_perception.prim_path.endswith(
        "/d555_camera_perception"
    )


# ---------------------------------------------------------------------------
# ProcRoom source — no Infinigen scene-USD / occupancy-spawn machinery
# ---------------------------------------------------------------------------


def test_procroom_geometry_not_infinigen_usd(cfg):
    # In-env procedural geometry: a per-env primitive collection + a contact
    # sensor, and crucially NO loaded-USD scene prim (the Infinigen affordance).
    assert hasattr(cfg.scene, "room_primitives")
    assert hasattr(cfg.scene, "contact_sensor")
    assert not hasattr(cfg.scene, "scene_geometry")


def test_no_infinigen_spawn_machinery(cfg):
    # ProcRoom spawns from its own BFS free-space (yaw_range only); it never
    # gets the Infinigen occupancy-derived spawn_points_xy / spawn_z / ground
    # lift that _apply_infinigen_scene_setup binds.
    params = cfg.events.reset_robot.params
    assert "yaw_range" in params
    assert "spawn_points_xy" not in params
    assert "spawn_z" not in params
    assert getattr(cfg.events, "lift_ground", None) is None


def test_procroom_managers_selected(cfg):
    assert type(cfg.commands).__name__ == "CommandsCfg_ProcRoom"
    assert type(cfg.terminations).__name__ == "TerminationsCfg_ProcRoom"
    assert type(cfg.events).__name__ == "EventsCfg_ProcRoom_Realistic"


# ---------------------------------------------------------------------------
# The scene class mirrors InfinigenPerception but keeps ProcRoom physics
# ---------------------------------------------------------------------------


def test_scene_class_keeps_both_cameras_and_procroom_replication():
    from strafer_lab.tasks.navigation.strafer_env_cfg import (
        StraferSceneCfg_ProcRoomPerception,
    )

    scene = StraferSceneCfg_ProcRoomPerception(num_envs=1, env_spacing=10.0)
    assert scene.d555_camera.width == 80
    assert scene.d555_camera_perception.width == 640
    assert hasattr(scene, "room_primitives")
    # Per-env replicated rooms — NOT Infinigen's shared-prim replicate_physics=False.
    assert scene.replicate_physics is True
