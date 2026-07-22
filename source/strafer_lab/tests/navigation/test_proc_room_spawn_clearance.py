# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn-pool clearance for the enriched generator path.

The pool the robot reset draws from must never offer a start pose already in
contact range: reset yaw is drawn independently of the spawn point, so a
candidate closer than the chassis circumscribing radius is a collision at
some yaws, and the episode opens with an error the policy could not have
avoided.

The pool's own obstacle inflation does *not* deliver that on its own — it is
a disc rasterized on a 0.1 m grid and a spawn point is a cell centre, so the
guarantee it actually gives is one grid cell short of its nominal radius.
Only the extra erosion the enriched path applies closes the gap, and the
gap it closes is measured here rather than assumed.

No Kit, no GPU. Run with the pure-Python lab tests.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from strafer_lab.tasks.navigation.mdp import proc_room as _proc_room

SEEDS = (20260101, 20268020, 20275939)
NUM_ENVS = 32
DIFFICULTY_SEED = 777


class _CaptureEntity:
    def __init__(self):
        self.body_poses = None

    def write_body_link_pose_to_sim_index(self, body_poses, env_ids, body_ids):
        self.body_poses = body_poses.clone()

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        pass


class _StubScene:
    def __init__(self, entities, env_origins):
        self._entities = entities
        self.env_origins = env_origins

    def __getitem__(self, key):
        return self._entities[key]


def _enriched_params():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    events = composed.StraferNavCfg_RLDepthEnriched_Real().events
    params = dict(events.generate_room.params)
    params.pop("collection_name", None)
    diff = events.randomize_difficulty.params
    return params, (int(diff["min_level"]), int(diff["max_level"]))


def _generate(seed, params, difficulty_range, num_envs=NUM_ENVS):
    entities = {"room_primitives": _CaptureEntity(), "ceiling": _CaptureEntity()}
    env = SimpleNamespace(
        num_envs=num_envs, device="cpu",
        scene=_StubScene(entities, torch.zeros(num_envs, 3)),
    )
    lo, hi = difficulty_range
    difficulty_rng = torch.Generator().manual_seed(DIFFICULTY_SEED)
    env._proc_room_difficulty = torch.randint(
        lo, hi + 1, (num_envs,), generator=difficulty_rng
    )
    torch.manual_seed(seed)
    _proc_room.generate_proc_room(env, torch.arange(num_envs), **params)
    pts = getattr(env, "_proc_room_robot_spawn_pts", env._proc_room_spawn_pts)
    count = getattr(env, "_proc_room_robot_spawn_count", env._proc_room_spawn_count)
    return entities["room_primitives"].body_poses, env._proc_room_active_mask, pts, count


def _distance_to_box(px, py, cx, cy, yaw, size_x, size_y):
    """Unsigned XY distance from a point to a yaw-rotated box; 0 inside."""
    dx, dy = px - cx, py - cy
    c, s = math.cos(-yaw), math.sin(-yaw)
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    return math.hypot(
        max(abs(local_x) - size_x / 2.0, 0.0),
        max(abs(local_y) - size_y / 2.0, 0.0),
    )


def spawn_clearances(poses, active_mask, spawn_pts, spawn_count, slots):
    """Distance from every pool candidate to the nearest active box in ``slots``."""
    sizes = _proc_room.OBJECT_SIZES.numpy()
    out = []
    for b in range(poses.shape[0]):
        boxes = [
            (
                float(poses[b, j, 0]), float(poses[b, j, 1]),
                2.0 * math.atan2(float(poses[b, j, 5]), float(poses[b, j, 6])),
                float(sizes[j, 0]), float(sizes[j, 1]),
            )
            for j in slots
            if bool(active_mask[b, j])
        ]
        if not boxes:
            continue
        for i in range(int(spawn_count[b])):
            px = float(spawn_pts[b, i, 0])
            py = float(spawn_pts[b, i, 1])
            out.append(min(_distance_to_box(px, py, *box) for box in boxes))
    return np.asarray(out)


def _pool_clearances(params, difficulty_range, slots):
    parts = []
    for seed in SEEDS:
        poses, active, pts, count = _generate(seed, params, difficulty_range)
        parts.append(spawn_clearances(poses, active, pts, count, slots))
    return np.concatenate(parts)


@pytest.fixture(scope="module")
def enriched():
    params, difficulty_range = _enriched_params()
    obstacles = _proc_room.FURNITURE_SLOTS + _proc_room.CLUTTER_SLOTS
    return {
        "params": params,
        "difficulty_range": difficulty_range,
        "object": _pool_clearances(params, difficulty_range, obstacles),
        "wall": _pool_clearances(params, difficulty_range, _proc_room.WALL_SLOTS),
    }


def test_enriched_spawn_pool_never_offers_a_contact_start(enriched):
    """No candidate within the chassis circumscribing radius of an object."""
    clearances = enriched["object"]
    assert len(clearances) > 1000
    worst = float(clearances.min())
    assert worst >= _proc_room.ROBOT_HALF_WIDTH, (
        f"{int((clearances < _proc_room.ROBOT_HALF_WIDTH).sum())} of "
        f"{len(clearances)} enriched spawn candidates sit inside the "
        f"{_proc_room.ROBOT_HALF_WIDTH} m chassis radius (worst {worst:.3f} m); "
        f"the robot would start in contact at some reset yaws"
    )


def test_enriched_spawn_pool_clears_walls_too(enriched):
    assert float(enriched["wall"].min()) >= _proc_room.ROBOT_HALF_WIDTH


def test_the_shared_pool_alone_does_not_deliver_the_floor(enriched):
    """The erosion is load-bearing, not decorative.

    Reading the shared pool instead of the eroded one puts candidates inside
    the chassis radius — grid quantization spends most of the nominal 0.3 m
    disc inflation before the spawn point is placed at a cell centre.
    """
    params = dict(enriched["params"])
    params["robot_spawn_inflation_cells"] = 0
    obstacles = _proc_room.FURNITURE_SLOTS + _proc_room.CLUTTER_SLOTS
    clearances = _pool_clearances(params, enriched["difficulty_range"], obstacles)
    assert float(clearances.min()) < _proc_room.ROBOT_HALF_WIDTH


def test_erosion_is_the_smallest_that_holds_the_floor(enriched):
    """One erosion cell is enough; the shipped value is not over-provisioned."""
    assert enriched["params"]["robot_spawn_inflation_cells"] == 1


def test_no_env_falls_back_to_the_uneroded_pool(enriched):
    """The floor is conditional on the erosion never emptying a room.

    ``generate_proc_room`` silently falls back per env to the shared pool when
    the extra erosion leaves nothing, and that pool is the one without the
    clearance. At the shipped erosion the fallback must never fire, so a
    future re-range cannot quietly reintroduce the hole.
    """
    for seed in SEEDS:
        entities = {"room_primitives": _CaptureEntity(), "ceiling": _CaptureEntity()}
        env = SimpleNamespace(
            num_envs=NUM_ENVS, device="cpu",
            scene=_StubScene(entities, torch.zeros(NUM_ENVS, 3)),
        )
        lo, hi = enriched["difficulty_range"]
        env._proc_room_difficulty = torch.randint(
            lo, hi + 1, (NUM_ENVS,), generator=torch.Generator().manual_seed(DIFFICULTY_SEED)
        )
        torch.manual_seed(seed)
        _proc_room.generate_proc_room(
            env, torch.arange(NUM_ENVS), **enriched["params"]
        )
        # The fallback copies the shared pool verbatim, so an env that took it
        # has point-identical pools; two independent draws never do.
        eroded = env._proc_room_robot_spawn_pts
        shared = env._proc_room_spawn_pts
        identical = torch.nonzero(
            (eroded == shared).all(dim=2).all(dim=1)
        ).flatten().tolist()
        assert not identical, (
            f"seed {seed}: envs {identical} fell back to the un-eroded shared pool"
        )


class _StubRobotData:
    def __init__(self, num_envs):
        self.default_root_pose = torch.zeros(num_envs, 7)
        self.default_root_pose[:, 6] = 1.0
        self.default_joint_pos = torch.zeros(num_envs, 4)
        self.default_joint_vel = torch.zeros(num_envs, 4)


class _StubRobot:
    def __init__(self, num_envs):
        self.data = _StubRobotData(num_envs)
        self.root_pose = None

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.root_pose = root_pose.clone()

    def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
        pass

    def write_joint_position_to_sim_index(self, position, env_ids):
        pass

    def write_joint_velocity_to_sim_index(self, velocity, env_ids):
        pass


def test_the_reset_event_reads_the_eroded_pool(enriched, monkeypatch):
    """The clearance only reaches the robot if the reset event reads the pool
    the erosion built. That branch has no config flag behind it — it is a
    ``hasattr`` on a lazily created attribute — so it is asserted here through
    the event rather than by reading the attribute directly."""
    from strafer_lab.tasks.navigation.mdp import events as _events

    monkeypatch.setattr(_events.wp, "to_torch", lambda t: t)

    entities = {"room_primitives": _CaptureEntity(), "ceiling": _CaptureEntity()}
    robot = _StubRobot(NUM_ENVS)
    entities["robot"] = robot
    env = SimpleNamespace(
        num_envs=NUM_ENVS, device="cpu",
        scene=_StubScene(entities, torch.zeros(NUM_ENVS, 3)),
    )
    lo, hi = enriched["difficulty_range"]
    env._proc_room_difficulty = torch.randint(
        lo, hi + 1, (NUM_ENVS,), generator=torch.Generator().manual_seed(DIFFICULTY_SEED)
    )
    torch.manual_seed(SEEDS[0])
    _proc_room.generate_proc_room(env, torch.arange(NUM_ENVS), **enriched["params"])
    _events.reset_robot_proc_room(
        env, torch.arange(NUM_ENVS), yaw_range=(-math.pi, math.pi)
    )

    placed = robot.root_pose[:, :2]
    eroded = env._proc_room_robot_spawn_pts
    counts = env._proc_room_robot_spawn_count
    for b in range(NUM_ENVS):
        pool = eroded[b, : int(counts[b])]
        assert bool((pool == placed[b]).all(dim=1).any()), (
            f"env {b} was placed at {placed[b].tolist()}, which is not in the "
            f"eroded pool — the reset event read the wrong pool"
        )

    poses = entities["room_primitives"].body_poses
    obstacles = _proc_room.FURNITURE_SLOTS + _proc_room.CLUTTER_SLOTS
    clearances = spawn_clearances(
        poses, env._proc_room_active_mask, placed.unsqueeze(1),
        torch.ones(NUM_ENVS, dtype=torch.long), obstacles,
    )
    assert float(clearances.min()) >= _proc_room.ROBOT_HALF_WIDTH
