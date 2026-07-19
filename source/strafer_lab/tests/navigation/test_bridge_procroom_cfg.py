# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free checks for the ProcRoom bridge capture variant.

``Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0`` is the Bridge capture stack
(full RGB + full depth + policy depth) on the ``procroom`` scene source — a
training-distribution scene that isolates the scene axis of the drive-fault
matrix so the same v1 artifact and Jetson stack can be driven against it.

These tests pin that the env registers, resolves to the CNN depth runner,
composes its two cameras + contact sensor + ProcRoom geometry / spawn /
managers. Config construction only — no Kit / GPU (env cfgs build in ~3s with
``LD_PRELOAD`` libgomp; see the DGX env-cfg test recipe).
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
    # Depth obs profile → the CNN depth runner.
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
# ProcRoom source — procedural geometry + its own spawn / managers
# ---------------------------------------------------------------------------


def test_procroom_geometry_present(cfg):
    # In-env procedural geometry: a per-env primitive collection + a contact
    # sensor for collision detection.
    assert hasattr(cfg.scene, "room_primitives")
    assert hasattr(cfg.scene, "contact_sensor")


def test_procroom_reset_spawns_by_yaw_range(cfg):
    # ProcRoom spawns the robot from its own BFS free-space with a randomized
    # yaw — the reset event exposes yaw_range (which the bridge pins).
    assert "yaw_range" in cfg.events.reset_robot.params


def test_procroom_managers_selected(cfg):
    assert type(cfg.commands).__name__ == "CommandsCfg_ProcRoom"
    assert type(cfg.terminations).__name__ == "TerminationsCfg_ProcRoom"
    assert type(cfg.events).__name__ == "EventsCfg_ProcRoom_Realistic"


# ---------------------------------------------------------------------------
# The perception scene class: both cameras + ProcRoom's per-env physics
# ---------------------------------------------------------------------------


def test_scene_class_keeps_both_cameras_and_procroom_replication():
    from strafer_lab.tasks.navigation.strafer_env_cfg import (
        StraferSceneCfg_ProcRoomPerception,
    )

    scene = StraferSceneCfg_ProcRoomPerception(num_envs=1, env_spacing=10.0)
    assert scene.d555_camera.width == 80
    assert scene.d555_camera_perception.width == 640
    assert hasattr(scene, "room_primitives")
    # ProcRoom rooms are per-env replicated primitives, so physics replicates.
    assert scene.replicate_physics is True


# ---------------------------------------------------------------------------
# The enriched ProcRoom bridge variant: same stack, enclosed generator
# ---------------------------------------------------------------------------

_ENRICHED_TASK = "Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0"


def test_enriched_bridge_registered_and_resolves():
    import gymnasium as gym

    assert _ENRICHED_TASK in gym.envs.registry
    kwargs = gym.envs.registry[_ENRICHED_TASK].kwargs or {}
    assert kwargs["env_cfg_entry_point"].endswith(
        ":StraferNavCfg_BridgeAutonomy_ProcRoomEnriched"
    )


def test_enriched_bridge_encloses_the_room():
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(_ENRICHED_TASK, device="cpu", num_envs=1)
    # Same two-camera capture stack + ProcRoom geometry as the open-top bridge.
    assert cfg.scene.d555_camera_perception is not None
    assert hasattr(cfg.scene, "room_primitives")
    # Plus the enclosure: a standalone ceiling entity (outside the collection)
    # and a per-env RGB fill light, with the generator fed the enrichment params.
    assert hasattr(cfg.scene, "ceiling")
    assert hasattr(cfg.scene, "ceiling_light")
    gen = cfg.events.generate_room.params
    assert gen["ceiling_entity_name"] == "ceiling"
    assert gen["wall_height"] > 1.0
    # Difficulty is un-pinned to a room-mode range (not the open-top pin at 7).
    diff = cfg.events.randomize_difficulty.params
    assert diff["min_level"] < diff["max_level"]
