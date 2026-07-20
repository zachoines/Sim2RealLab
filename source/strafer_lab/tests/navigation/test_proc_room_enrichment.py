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
    TALL_OBJECT_SLOTS,
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

    # A separate robot-spawn pool exists, is populated, and is never larger than
    # the shared pool it erodes from (extra standoff => a subset of cells; the
    # per-env fallback makes them equal at most, never exceeds).
    assert hasattr(env, "_proc_room_robot_spawn_pts")
    assert env._proc_room_robot_spawn_count.sum() > 0
    assert torch.all(
        env._proc_room_robot_spawn_count <= env._proc_room_spawn_count
    )


def _room_dims_from_walls(env, body_poses):
    """Recover per-env (room_w, room_h) from the active wall segment centers.

    E/W walls sit at x=±w/2, N/S walls at y=±h/2, so the max abs coordinate over
    active wall slots is the half-span. Inactive (parked) wall slots are masked
    out. Walls are never parked by the retry ladder, so enclosed envs keep all
    their perimeter segments.
    """
    am = env._proc_room_active_mask[:, WALL_SLOTS]  # (B, 20)
    wx = torch.where(am, body_poses[:, WALL_SLOTS, 0].abs(), torch.zeros_like(am, dtype=body_poses.dtype))
    wy = torch.where(am, body_poses[:, WALL_SLOTS, 1].abs(), torch.zeros_like(am, dtype=body_poses.dtype))
    return 2.0 * wx.max(dim=1).values, 2.0 * wy.max(dim=1).values


def test_enriched_span_cap_bounds_perimeter():
    torch.manual_seed(7)
    env, entities = _make_env(num_envs=32)
    env_ids = torch.arange(env.num_envs)
    generate_proc_room(env, env_ids, span_max=7.5, max_span_sum=13.4, wall_height=2.7)
    w, h = _room_dims_from_walls(env, entities["room_primitives"].body_poses)
    # The cap holds for every room (tol: wall thickness half-extent 0.075 m).
    assert torch.all(w + h <= 13.4 + 0.2)

    # Non-vacuous: the SAME seed with the cap off exceeds the budget on some env,
    # so the cap actively shrank those rooms rather than never engaging.
    torch.manual_seed(7)
    env2, ent2 = _make_env(num_envs=32)
    generate_proc_room(env2, env_ids, span_max=7.5, max_span_sum=None, wall_height=2.7)
    w2, h2 = _room_dims_from_walls(env2, ent2["room_primitives"].body_poses)
    assert torch.any(w2 + h2 > 13.4 + 0.2)


def test_default_path_byte_identical_to_explicit_defaults():
    """The default (open-top) generation is byte-for-byte identical whether the
    enrichment kwargs are omitted or passed explicitly at their default values —
    the load-bearing safety property. A stray *unconditional* enrichment draw on
    the default path would desync the shared RNG stream and break this.

    (Exact equality holds because both runs use this branch's identical formulas;
    the one accepted deviation vs pre-PR main is the door-width multiplier's
    1-ulp float difference — `door_width_max - 0.8 != 0.4` in IEEE754 — which is
    identical across both runs here, so it does not surface.)
    """
    def _run(**kwargs):
        torch.manual_seed(20260719)
        env, ent = _make_env(num_envs=6, difficulty=6)
        generate_proc_room(env, torch.arange(env.num_envs), **kwargs)
        return ent["room_primitives"].body_poses

    implicit = _run()
    explicit = _run(
        wall_height=1.0,
        door_width_max=1.2,
        span_max=7.0,
        max_span_sum=None,
        clutter_wall_bias_prob=0.0,
        robot_spawn_inflation_cells=0,
        ceiling_entity_name=None,
        p_ceil=0.0,
        ceiling_height_range=(2.2, 2.9),
        tall_object_heights=None,
    )
    assert torch.equal(implicit, explicit)


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
# Enriched tall objects stand full-height (pose-z half of the palette lockstep)
# ---------------------------------------------------------------------------


def test_enriched_tall_objects_stand_full_height():
    """With the ``tall_object_heights`` map the shelf / cabinet / tall-cylinder
    slots are posed at ``height/2``; without it they keep the OBJECT_SIZES
    default. Both runs share a seed, so the map is the only difference."""
    heights = {"shelf": 2.0, "cabinet": 2.1, "tall_cyl": 1.8}

    def _run(tall):
        torch.manual_seed(1234)
        # Difficulty 7 places every furniture + clutter slot, so every tall slot
        # is a placement candidate; env origins are zero, so pose z is local z.
        env, ent = _make_env(num_envs=8, difficulty=7)
        generate_proc_room(
            env,
            torch.arange(env.num_envs),
            wall_height=2.7,
            tall_object_heights=tall,
        )
        return env, ent["room_primitives"].body_poses

    env_e, poses_e = _run(heights)
    env_d, poses_d = _run(None)

    placed_any = False
    for key, slots in TALL_OBJECT_SLOTS.items():
        for slot in slots:
            active = env_e._proc_room_active_mask[:, slot]
            if not active.any():
                continue
            placed_any = True
            z_e = poses_e[:, slot, 2][active]
            assert torch.allclose(z_e, torch.full_like(z_e, heights[key] / 2.0))
            z_d = poses_d[:, slot, 2][active]
            assert torch.allclose(z_d, torch.full_like(z_d, OBJECT_SIZES[slot, 2].item() / 2.0))
            assert (z_e > z_d + 1e-3).all()
    assert placed_any, "no tall slot was placed — test is vacuous"


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
