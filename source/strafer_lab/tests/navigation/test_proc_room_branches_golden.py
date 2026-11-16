# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen-output guard for the generator's branches that neither sweep executes.

The vanilla sweep runs the defaults and the enriched sweep the shipped enriched arguments, both on full resets of
envs at the origin, with no health sink. Neither reaches: the compound catalogue (the seat and the phase), a call
on an env without curriculum difficulty, the enriched arguments at levels 0-3, non-zero env origins with partial
resets, the health sink, custom furniture and clutter sequences, the column phase's early exit, and the robot
pool's fallback to the shared pool. Each case below drives one of them and pins one digest per seed of the
generator's output, its health sink where one is passed, and its RNG stream position.

A diff is a change to the generator, not a test to update.

No Kit, no GPU. Run with the pure-Python lab tests.
"""
from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import torch

from strafer_lab.tasks.navigation import strafer_env_cfg as _env_cfg
from strafer_lab.tasks.navigation.mdp import proc_room as _proc_room

_ATTRS = (
    "_proc_room_spawn_pts",
    "_proc_room_spawn_count",
    "_proc_room_active_mask",
    "_proc_room_free_space",
    "_proc_room_robot_spawn_pts",
    "_proc_room_robot_spawn_count",
)


class _CaptureEntity:
    def __init__(self):
        self.body_poses = None
        self.root_pose = None

    def write_body_link_pose_to_sim_index(self, body_poses, env_ids, body_ids):
        self.body_poses = body_poses.clone()

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.root_pose = root_pose.clone()


class _StubScene:
    def __init__(self, entities, env_origins):
        self._entities = entities
        self.env_origins = env_origins

    def __getitem__(self, key):
        return self._entities[key]


def _zero_origins(num_envs):
    return torch.zeros(num_envs, 3)


def _spread_origins(num_envs):
    i = torch.arange(num_envs, dtype=torch.float32)
    return torch.stack([i * 10.0 - 35.0, i * -7.5 + 12.0, torch.full((num_envs,), 0.25)], dim=-1)


def _enriched(**overrides):
    return lambda: {**dict(_env_cfg._GENERATE_PROC_ROOM_ENRICHED.params), **overrides}


def _compound_placement():
    m = _proc_room
    rank = tuple(
        [(s,) for s in reversed(m.CLUTTER_SLOTS)]
        + [(f,) for f in reversed(m.FURNITURE_SLOTS) if f not in (22, 23, 24, 25)]
        + [(22, 23), (24, 25)]
    )
    return m.PlacementCfg(
        park_rank=rank,
        furniture_aabb_reject=True,
        compound_prob=1.0,
        relocate_blocked_bfs_seed=True,
        compounds=(
            m.CompoundCfg(members=(22, 23), offsets=((-0.4, 0.15), (0.4, 0.15))),
            m.CompoundCfg(members=(24, 25), offsets=((-0.35, 0.25), (0.35, 0.25))),
        ),
    )


def _unseatable_compound_placement():
    """One compound wider than any wall of a room at ``span_max`` 4 m: every attempt skips its wall."""
    m = _proc_room
    rank = tuple(
        [(s,) for s in reversed(m.CLUTTER_SLOTS)]
        + [(f,) for f in reversed(m.FURNITURE_SLOTS) if f not in (22, 23)]
        + [(22, 23)]
    )
    return m.PlacementCfg(
        park_rank=rank,
        compound_prob=1.0,
        compounds=(m.CompoundCfg(members=(22, 23), offsets=((-2.2, 0.15), (2.2, 0.15))),),
    )


def _custom_sequences(**columns):
    m = _proc_room
    return m.PlacementCfg(
        furniture_sequence=tuple(reversed(m.FURNITURE_SLOTS)),
        clutter_sequence=tuple(reversed(m.CLUTTER_SLOTS)),
        park_rank=tuple(reversed(m._VANILLA_PARK_RANK)),
        column_prob=1.0,
        column_count=1,
        **columns,
    )


def _column_r_min():
    """The column phase's least radius for the default column slots, as the generator resolves it."""
    m, p = _proc_room, _proc_room.PlacementCfg()
    return p.column_seed_clearance + max(
        math.hypot(m.OBJECT_SIZES[s, 0].item(), m.OBJECT_SIZES[s, 1].item()) / 2.0 for s in p.column_slots
    )


def _no_difficulty():
    return dict(max_internal_walls=1, max_furniture=6, max_clutter=10)


_ALL8 = (tuple(range(8)),)

# name: (seeds, num_envs, calls (env ids per call, in order on one env), difficulty, origins, params, health sink)
CASES = {
    "compound": (
        (20260101, 20268020, 20275939), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enriched()(), "placement": _compound_placement()}, False,
    ),
    "compound-tight": (
        (20283858, 20291777, 20299696), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enriched()(), "placement": _compound_placement(), "span_max": 4.0, "max_span_sum": 8.0}, False,
    ),
    "compound-unseatable-sink": (
        (20347210, 20260101), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enriched()(), "placement": _unseatable_compound_placement(), "span_max": 4.0,
                 "max_span_sum": 8.0}, True,
    ),
    "no-difficulty": (
        (20307615, 20315534, 20323453), 8, _ALL8, None, _zero_origins, _no_difficulty, False,
    ),
    "enriched-levels-0-1": (
        (20331372, 20339291, 20347210), 8, _ALL8, (0, 1, 0, 1, 1, 0, 1, 0), _zero_origins, _enriched(), False,
    ),
    "enriched-levels-2-3": (
        (20260101, 20268020, 20275939), 8, _ALL8, (2, 3, 2, 3, 3, 2, 3, 2), _zero_origins, _enriched(), False,
    ),
    "origins-partial-resets-sink": (
        (20283858, 20291777), 8, (tuple(range(8)), (1, 4, 6), (0, 7, 3, 2, 5)), (4, 5, 6, 7, 7, 6, 5, 4),
        _spread_origins, _enriched(), True,
    ),
    "custom-sequences": (
        (20291777, 20299696, 20307615), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enriched()(), "placement": _custom_sequences(column_radius_max=0.9)}, False,
    ),
    "column-early-exit-sink": (
        (20315534, 20323453), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enriched()(), "placement": _custom_sequences(
            column_radius_max=_proc_room.BFS_SEED_CLEARANCE + 0.05)}, True,
    ),
    # Either side of the column phase's exit, by 0.01 m: just below it the phase exits before any column; just above
    # it places columns in a 0.01 m band.
    "column-exit-just-below-sink": (
        (20355129, 20363048), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enriched()(), "placement": _custom_sequences(column_radius_max=_column_r_min() - 0.01)}, True,
    ),
    "column-exit-just-above-sink": (
        (20370967, 20378886), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enriched()(), "placement": _custom_sequences(column_radius_max=_column_r_min() + 0.01)}, True,
    ),
    "robot-pool-fallback": (
        (20315534, 20323453, 20331372), 8, _ALL8, 7, _zero_origins, _enriched(robot_spawn_inflation_cells=40),
        False,
    ),
    "sink-enriched-compound": (
        (20339291,), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enriched()(), "placement": _compound_placement(), "span_max": 4.0, "max_span_sum": 8.0}, True,
    ),
}


def _feed(h, tensor):
    t = tensor.detach().cpu().contiguous()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes())


def _run_case(module, seed, num_envs, calls, difficulty, origins, make_params, with_sink):
    """Run one case's calls on one env; returns (env, entities, sinks, tail)."""
    torch.manual_seed(seed)
    entities = {"room_primitives": _CaptureEntity(), "ceiling": _CaptureEntity()}
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=_StubScene(entities, origins(num_envs)))
    if difficulty is not None:
        env._proc_room_difficulty = torch.as_tensor(difficulty, dtype=torch.long).expand(num_envs).clone()
    sinks, outputs = [], []
    for ids in calls:
        params = make_params()
        if with_sink:
            params["health_sink"] = {}
        module.generate_proc_room(env, torch.tensor(ids), **params)
        if with_sink:
            sinks.append(params["health_sink"])
        outputs.append((entities["room_primitives"].body_poses, entities["ceiling"].root_pose))
    return env, outputs, sinks, torch.rand(4)


def _case_digest(module, seed, num_envs, calls, difficulty, origins, make_params, with_sink):
    env, outputs, sinks, tail = _run_case(
        module, seed, num_envs, calls, difficulty, origins, make_params, with_sink
    )
    h = hashlib.sha256()
    for poses, ceiling in outputs:
        _feed(h, poses)
        h.update(b"ceiling" if ceiling is not None else b"no-ceiling")
        if ceiling is not None:
            _feed(h, ceiling)
    for name in _ATTRS:
        h.update(name.encode())
        if hasattr(env, name):
            _feed(h, getattr(env, name))
    for sink in sinks:
        h.update(repr(sorted(sink.items())).encode())
    _feed(h, tail)
    return h.hexdigest()


def sweep_digests(module=_proc_room):
    """{case-key: digest} over every case and seed."""
    return {
        f"{name}-s{seed}": _case_digest(module, seed, n, calls, difficulty, origins, make_params, with_sink)
        for name, (seeds, n, calls, difficulty, origins, make_params, with_sink) in CASES.items()
        for seed in seeds
    }


_SWEEP_GOLDEN = {
    "compound-s20260101": "dbb187d8ce8e42be236debc71b28c9ded22610453cbbebba7c2fde342592a96c",
    "compound-s20268020": "6da62ff22931f1c73f88c53caa74de1aee4ef149577d5e0884b7c49b5fe50281",
    "compound-s20275939": "3241f474ffa6cfb8c8839c55f4f9438ff32672881947c09eff36b0efe0e2ec42",
    "compound-tight-s20283858": "6641673ddf33d9a04965190d24a5f7092ef5234ab626e1da1e670bc8dde66509",
    "compound-tight-s20291777": "75c2a12b80f19a097f9a066afd8a2357e2500cca37e3f339fad4a2b82debf8d5",
    "compound-tight-s20299696": "12d7c0c02788d788e5781dc52c428ac160e992349483760f1923f4eecc9aa4eb",
    "compound-unseatable-sink-s20347210": "d48ea42ce87651cd6f1dbc641090277ad4078c524f45429b084613ea6e038274",
    "compound-unseatable-sink-s20260101": "2260d93d4d19aff3161d3ae0944eb8be26268adf8ff4b0519d15a81b617d7688",
    "no-difficulty-s20307615": "944aa5d6ff3ff8397a4837b7f56a2a72fb515bb947882fb6b92e485be0690585",
    "no-difficulty-s20315534": "44b88eaeb15e89a2995e4405ae9c6173993fb7becb3eb627ce7f757d70ec78da",
    "no-difficulty-s20323453": "765b0a144c3902180085313bd75c23ebd8daa3e6c2939a13d740e20f0d81c364",
    "enriched-levels-0-1-s20331372": "10a636dee4a00ff87f0c0287ac26c4f7bfb68ef5217362135f3ceda342ea18de",
    "enriched-levels-0-1-s20339291": "f686ab3b6551689f1ec98d9816bccd8dba1b45d32ac4b47c41bf989455dd16d4",
    "enriched-levels-0-1-s20347210": "b425486a3a17d9b94f6601a74af4e2d85c40b8915c454e8a0d2562ca5591ef46",
    "enriched-levels-2-3-s20260101": "de6c38d2a7b8db849f4fbc3e7be0241eb014b5430a16709e5934c17c92ea2c00",
    "enriched-levels-2-3-s20268020": "5430b48c122bac6f4cc8937a281147b007b837b4489a949ada63880e482048e0",
    "enriched-levels-2-3-s20275939": "cd8436df7bd54e64e0d0c99d210eef88e757ab1ffeec4d8f7ff05443be44a054",
    "origins-partial-resets-sink-s20283858": "80c2bd1d7c9a1599fa90d76d11d3a5afade54fdeaac436164bebc1e26813f050",
    "origins-partial-resets-sink-s20291777": "bc551f4069005fabd607a0355265879983b5c3a8710507598195df0ccc2f1567",
    "custom-sequences-s20291777": "22dd1899b09afece6436e70038016f61070eea8eb20bfb43e2137b6ec8d8f9f0",
    "custom-sequences-s20299696": "e8eb5e592a14b50485b8d3080a3b4a8dc2bc986a1aeb7cd262caff72b11108b6",
    "custom-sequences-s20307615": "b851d5bc927e880d82ef39a4df08a9a1d5efad8c7c2470971704d69503938611",
    "column-early-exit-sink-s20315534": "7c9611487a8dd1bb67c17f26a87865f6fd8783ca8fd8acd3910b17ba6d3b6861",
    "column-early-exit-sink-s20323453": "f02d0f0dbf982522f203617e6c1eb20d77027890e166a11d7e124571eb959e1e",
    "column-exit-just-below-sink-s20355129": "a23095fd63e7ad0c5d937a877902d65ab3c4763c84aebc4ec3f6ba02445a865a",
    "column-exit-just-below-sink-s20363048": "0a588b95efb7e017fdca5938f2e698561b047b849e0f08124abf7771e1524512",
    "column-exit-just-above-sink-s20370967": "751d6801b9b3e641807192f06cd9d0503b99ea42fd2c7a6754500215896402cb",
    "column-exit-just-above-sink-s20378886": "f225b4af99194a75920770901243f07393ca095af1a2e34e81a80bbe2c45b6bd",
    "robot-pool-fallback-s20315534": "2d3bbaa71ce8cf9cc9a12071c6b5628d55c8d8aff225fba116e5d4a8deefd460",
    "robot-pool-fallback-s20323453": "127aaa629ca3efb8fe0d2939c163d5f20b2dda6574d58887fb04f42da001a469",
    "robot-pool-fallback-s20331372": "3c491c06fb133a8233a8d926a429eb1032eb4661570f9c76956f74d949273052",
    "sink-enriched-compound-s20339291": "088107271d84a287f49f23e1d9b3ec074b25cecb26a0dd47f76005877bbd9c27",
}


def _sinks(name):
    seeds, n, calls, difficulty, origins, make_params, _ = CASES[name]
    out = []
    for seed in seeds:
        out.extend(_run_case(_proc_room, seed, n, calls, difficulty, origins, make_params, True)[2])
    return out


def test_each_case_reaches_its_branch():
    """Each case drives the branch it is named for, read from the health sink, so a golden cannot pass on a case
    that silently stopped reaching its path."""
    assert sum(s["compound_phase_fired"] for s in _sinks("compound")) > 0
    assert sum(s["placed_compound_members"] for s in _sinks("compound")) > 0
    unseatable = _sinks("compound-unseatable-sink")
    assert sum(s["compound_phase_fired"] for s in unseatable) > 0
    (wide,) = _unseatable_compound_placement().compounds
    assert _proc_room._try_place_compound(wide, 4.0, 4.0, [], "cpu") is None
    tight = _sinks("compound-tight")
    assert sum(s["ladder_passes"] for s in tight) > 0
    levels_open = _sinks("enriched-levels-0-1")
    assert all(s["envs"] == 8 for s in levels_open)
    early = _sinks("column-early-exit-sink")
    assert sum(s["column_phase_fired"] for s in early) > 0
    assert sum(s["placed_columns"] for s in early) == 0
    below = _sinks("column-exit-just-below-sink")
    assert sum(s["column_phase_fired"] for s in below) > 0 and sum(s["placed_columns"] for s in below) == 0
    above = _sinks("column-exit-just-above-sink")
    assert sum(s["column_phase_fired"] for s in above) > 0 and sum(s["placed_columns"] for s in above) > 0
    assert sum(s["column_phase_fired"] for s in _sinks("custom-sequences")) > 0
    assert sum(s["placed_columns"] for s in _sinks("custom-sequences")) > 0

    seeds, n, calls, difficulty, origins, make_params, _ = CASES["no-difficulty"]
    env = _run_case(_proc_room, seeds[0], n, calls, difficulty, origins, make_params, False)[0]
    assert not hasattr(env, "_proc_room_difficulty")

    seeds, n, calls, difficulty, origins, make_params, _ = CASES["robot-pool-fallback"]
    fell_back = 0
    for seed in seeds:
        env = _run_case(_proc_room, seed, n, calls, difficulty, origins, make_params, False)[0]
        same = (env._proc_room_robot_spawn_count == env._proc_room_spawn_count) & torch.all(
            (env._proc_room_robot_spawn_pts == env._proc_room_spawn_pts).flatten(1), dim=1
        )
        fell_back += int(same.sum())
    assert fell_back > 0

    seeds, n, calls, difficulty, origins, make_params, _ = CASES["origins-partial-resets-sink"]
    assert len(calls) > 1 and any(len(c) < n for c in calls)
    assert bool((origins(n)[:, :2] != 0).all())


def test_branch_outputs_match_goldens():
    got = sweep_digests()
    assert set(got) == set(_SWEEP_GOLDEN), "sweep shape changed — goldens are stale"
    diverged = sorted(k for k in got if got[k] != _SWEEP_GOLDEN[k])
    assert not diverged, (
        f"{len(diverged)} of {len(got)} branch cases diverged from the frozen output; first: {diverged[:5]}"
    )
