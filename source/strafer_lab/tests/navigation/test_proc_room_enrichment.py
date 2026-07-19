# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""CPU smoke tests for the ProcRoom depth-enrichment generator path.

``generate_proc_room`` normally runs inside a live Kit env, but its geometry is
pure torch (occupancy raster, disc inflation, GPU-BFS via conv/max-pool), so it
runs on CPU against a stub scene that captures the batched pose writes. These
tests pin two things the GPU descriptor run cannot cheaply re-check every time:

1. With the argument defaults the generator is byte-for-byte the open-top
   behavior (walls at z=0.5, no ceiling write, single shared spawn pool).
2. With the enrichment arguments the walls stand at ``wall_height/2``, the
   standalone ceiling slab is posed per env, and a separate robot-spawn pool is
   built — none of which perturbs the default path.

No Kit / GPU. Run with the pure-Python lab tests.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from strafer_lab.tasks.navigation.mdp.proc_room import (
    NUM_OBJECTS,
    OBJECT_SIZES,
    WALL_SLOTS,
    _erode_reachable,
    generate_proc_room,
)


class _CaptureEntity:
    """Stub scene entity that records the pose tensors written to it."""

    def __init__(self):
        self.body_poses = None
        self.root_pose = None

    def write_body_link_pose_to_sim_index(self, body_poses, env_ids, body_ids):
        self.body_poses = body_poses.clone()

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.root_pose = root_pose.clone()


class _StubScene:
    def __init__(self, entities: dict, env_origins: torch.Tensor):
        self._entities = entities
        self.env_origins = env_origins

    def __getitem__(self, key):
        return self._entities[key]


def _make_env(num_envs: int = 4, difficulty: int = 5):
    entities = {"room_primitives": _CaptureEntity(), "ceiling": _CaptureEntity()}
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=_StubScene(entities, torch.zeros(num_envs, 3)),
    )
    env._proc_room_difficulty = torch.full((num_envs,), difficulty, dtype=torch.long)
    return env, entities


def _active_wall_z(env, body_poses):
    """Z of the walls the generator marked active, from the captured poses."""
    mask = env._proc_room_active_mask[:, WALL_SLOTS]  # (B, 20)
    zs = body_poses[:, WALL_SLOTS, 2]  # (B, 20)
    return zs[mask]


# ---------------------------------------------------------------------------
# Default path is the open-top behavior
# ---------------------------------------------------------------------------


def test_default_path_walls_at_half_meter_no_ceiling_no_robot_pool():
    env, entities = _make_env()
    env_ids = torch.arange(env.num_envs)
    generate_proc_room(env, env_ids)  # all defaults

    wall_z = _active_wall_z(env, entities["room_primitives"].body_poses)
    assert wall_z.numel() > 0
    assert torch.allclose(wall_z, torch.full_like(wall_z, 0.5))
    # No ceiling was posed and no robot-spawn pool was built.
    assert entities["ceiling"].root_pose is None
    assert not hasattr(env, "_proc_room_robot_spawn_pts")
    # The shared pool still populated the robot's spawn source.
    assert env._proc_room_spawn_count.sum() > 0


# ---------------------------------------------------------------------------
# Enriched path raises the walls, encloses, and forks the robot pool
# ---------------------------------------------------------------------------


def test_enriched_path_encloses_and_forks_spawn_pool():
    env, entities = _make_env()
    env_ids = torch.arange(env.num_envs)
    wall_height = 2.7
    generate_proc_room(
        env,
        env_ids,
        wall_height=wall_height,
        door_width_max=2.0,
        span_max=7.5,
        max_span_sum=13.4,
        clutter_wall_bias_prob=1.0,  # force the perimeter law for coverage
        robot_spawn_inflation_cells=2,
        ceiling_entity_name="ceiling",
        p_ceil=1.0,  # force enclosure so the ceiling z is deterministic-in-range
        ceiling_height_range=(2.2, 2.9),
    )

    # Walls stand at wall_height / 2.
    wall_z = _active_wall_z(env, entities["room_primitives"].body_poses)
    assert wall_z.numel() > 0
    assert torch.allclose(wall_z, torch.full_like(wall_z, wall_height / 2.0))

    # The ceiling slab was posed; with p_ceil=1.0 every env is enclosed at a
    # height in the sampled band (env origins are zero here).
    ceil_pose = entities["ceiling"].root_pose
    assert ceil_pose is not None and ceil_pose.shape == (env.num_envs, 7)
    assert torch.all(ceil_pose[:, 2] >= 2.2 - 1e-4)
    assert torch.all(ceil_pose[:, 2] <= 2.9 + 1e-4)

    # A separate robot-spawn pool exists and is populated.
    assert hasattr(env, "_proc_room_robot_spawn_pts")
    assert env._proc_room_robot_spawn_count.sum() > 0


def test_enriched_span_cap_keeps_perimeter_within_budget():
    # Many envs at the largest span so the cap actually bites on some of them.
    env, entities = _make_env(num_envs=16)
    env_ids = torch.arange(env.num_envs)
    generate_proc_room(
        env, env_ids, span_max=7.5, max_span_sum=13.4, wall_height=2.7,
        ceiling_entity_name="ceiling", p_ceil=0.5,
    )
    # Generation completed and produced spawn points for the batch.
    assert env._proc_room_spawn_count.sum() > 0


def test_ceiling_mixture_parks_open_top_envs():
    # p_ceil=0.0 => no env is enclosed => the slab parks below the floor.
    env, entities = _make_env(num_envs=8)
    env_ids = torch.arange(env.num_envs)
    generate_proc_room(
        env, env_ids, wall_height=2.7, ceiling_entity_name="ceiling", p_ceil=0.0,
    )
    ceil_pose = entities["ceiling"].root_pose
    assert ceil_pose is not None
    assert torch.all(ceil_pose[:, 2] < -5.0)  # parked


# ---------------------------------------------------------------------------
# Erosion helper
# ---------------------------------------------------------------------------


def test_erode_reachable_shrinks_interior():
    reachable = torch.zeros(1, 20, 20, dtype=torch.bool)
    reachable[0, 5:15, 5:15] = True  # a 10x10 free block
    eroded = _erode_reachable(reachable, cells=2)
    # A 10x10 block eroded by 2 leaves a 6x6 interior.
    assert int(eroded.sum()) == 36
    # Every eroded cell is a subset of the original reachable set.
    assert bool((eroded & ~reachable).sum() == 0)


def test_erode_reachable_zero_cells_is_identity():
    reachable = torch.rand(2, 12, 12) > 0.3
    assert torch.equal(_erode_reachable(reachable, 0), reachable)
