# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen-output guard for the generator's branches that neither sweep executes.

The vanilla sweep runs the defaults and the enclosed-room sweep the enclosed room's arguments, both on full resets
of envs at the origin, with no health sink. Neither reaches: the compound catalogue (the seat and the phase), a
call on an env without curriculum difficulty, the enclosed room's arguments at levels 0-3, non-zero env origins
with partial resets, the health sink, custom furniture and clutter sequences, the column phase's early exit, and
the robot pool's fallback to the shared pool. Each case below drives one of them and pins one digest per seed of the
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


def _enclosed(**overrides):
    return lambda: {**dict(_env_cfg._GENERATE_PROC_ROOM_ENCLOSED.params), **overrides}


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


def _no_difficulty_negative_budgets():
    return dict(max_internal_walls=1, max_furniture=-1, max_clutter=-3)


_ALL8 = (tuple(range(8)),)

# name: (seeds, num_envs, calls (env ids per call, in order on one env), difficulty, origins, params, health sink)
CASES = {
    "compound": (
        (20260101, 20268020, 20275939), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enclosed()(), "placement": _compound_placement()}, False,
    ),
    "compound-tight": (
        (20283858, 20291777, 20299696), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enclosed()(), "placement": _compound_placement(), "span_max": 4.0, "max_span_sum": 8.0}, False,
    ),
    "compound-unseatable-sink": (
        (20347210, 20260101), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enclosed()(), "placement": _unseatable_compound_placement(), "span_max": 4.0,
                 "max_span_sum": 8.0}, True,
    ),
    "no-difficulty": (
        (20307615, 20315534, 20323453), 8, _ALL8, None, _zero_origins, _no_difficulty, False,
    ),
    "enclosed-args-levels-0-1": (
        (20331372, 20339291, 20347210), 8, _ALL8, (0, 1, 0, 1, 1, 0, 1, 0), _zero_origins, _enclosed(), False,
    ),
    "enclosed-args-levels-2-3": (
        (20260101, 20268020, 20275939), 8, _ALL8, (2, 3, 2, 3, 3, 2, 3, 2), _zero_origins, _enclosed(), False,
    ),
    "origins-partial-resets-sink": (
        (20283858, 20291777), 8, (tuple(range(8)), (1, 4, 6), (0, 7, 3, 2, 5)), (4, 5, 6, 7, 7, 6, 5, 4),
        _spread_origins, _enclosed(), True,
    ),
    "custom-sequences": (
        (20291777, 20299696, 20307615), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enclosed()(), "placement": _custom_sequences(column_radius_max=0.9)}, False,
    ),
    "column-early-exit-sink": (
        (20315534, 20323453), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enclosed()(), "placement": _custom_sequences(
            column_radius_max=_proc_room.BFS_SEED_CLEARANCE + 0.05)}, True,
    ),
    # Either side of the column phase's exit, by 0.01 m: just below it the phase exits before any column; just above
    # it places columns in a 0.01 m band.
    "column-exit-just-below-sink": (
        (20355129, 20363048), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enclosed()(), "placement": _custom_sequences(column_radius_max=_column_r_min() - 0.01)}, True,
    ),
    "column-exit-just-above-sink": (
        (20370967, 20378886), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enclosed()(), "placement": _custom_sequences(column_radius_max=_column_r_min() + 0.01)}, True,
    ),
    "robot-pool-fallback": (
        (20315534, 20323453, 20331372), 8, _ALL8, 7, _zero_origins, _enclosed(robot_spawn_inflation_cells=40),
        False,
    ),
    # An erosion that empties some envs' robot pool and not others: the fallback is per env.
    "robot-pool-fallback-mixed": (
        (20315534, 20323453, 20331372), 8, _ALL8, 7, _zero_origins, _enclosed(robot_spawn_inflation_cells=8),
        False,
    ),
    # Negative budgets on an env without curriculum difficulty take no prefix of the sequences.
    "no-difficulty-negative-budgets": (
        (20307615,), 8, _ALL8, None, _zero_origins, _no_difficulty_negative_budgets, False,
    ),
    "sink-enclosed-compound": (
        (20339291,), 8, _ALL8, 7, _zero_origins,
        lambda: {**_enclosed()(), "placement": _compound_placement(), "span_max": 4.0, "max_span_sum": 8.0}, True,
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
    "compound-s20260101": "ddf3be7ff06323882a0c3f1a347cb2e97f17a7779dab295de24b2418e6c2dfad",
    "compound-s20268020": "a8485cfcaf8764675a88a1db1ba9c900d78cdfd2f8f9166219e9327b1bb03843",
    "compound-s20275939": "da4957c8593f7b9efd8c7cbfadc550e6463b32f219667d3eb7f7030a4c7ffa8e",
    "compound-tight-s20283858": "dfa078505fe94a9494d13da4609eaf1c44ddd175a3be7f629ff3a284d6262acd",
    "compound-tight-s20291777": "66436dc08da4c47325887a2718887980ad41f1558e0ae21e29319be8841bec6f",
    "compound-tight-s20299696": "ca40ec2ede3a80cfe586306e08340cb08b5b9a7c627468065e1c0aa159c1db68",
    "compound-unseatable-sink-s20347210": "7e237790ef44c30ccd36ddd43444e9de7eeee3cfb313425486811da38700e442",
    "compound-unseatable-sink-s20260101": "ffdc2d42a9314566c7dc7b482257eaa76ff330e4d1605d5361c2b19a954b3f24",
    "no-difficulty-s20307615": "c312b341927c098e5a64a17b1fcf1dad3d30e16a77a4d7517d174993fb14d3f3",
    "no-difficulty-s20315534": "b9464aaf7aa5f45a35ad56930a63a5bbcaa2004db65bd49a11798ed759f236ac",
    "no-difficulty-s20323453": "8ef7c47b0238a75cd4a576aac9847afeaa6ba9c496822b41846981ab09788b2e",
    "enclosed-args-levels-0-1-s20331372": "045a8ec61a56f5114ca40fc1e13fc47b1f35a74fbc11df280af5b726aeb82024",
    "enclosed-args-levels-0-1-s20339291": "b8e36e1a834095e8e498901009fd6548946f5c01ed2909d03c946dad0038cdb8",
    "enclosed-args-levels-0-1-s20347210": "0e83dace014c85218c2fe4a2f30afc251b216e53db82c00a7f487441424b38e7",
    "enclosed-args-levels-2-3-s20260101": "de6c38d2a7b8db849f4fbc3e7be0241eb014b5430a16709e5934c17c92ea2c00",
    "enclosed-args-levels-2-3-s20268020": "d36d6bdb22b604e2564593b443d65367f1fc45a012485b5c81dd3e7e4e3d4555",
    "enclosed-args-levels-2-3-s20275939": "1d0c11287384210de01b3623e467fe721abd73cf45000f7f3e15fc74fdc2172a",
    "origins-partial-resets-sink-s20283858": "0d7022d760eb9bf717f20fc9150565bd715eb19cd1477f2a3d33c5b9418af2cf",
    "origins-partial-resets-sink-s20291777": "4a19aa31dda57caa485e8e9bd9ea1542a8c6d41315e1a78d2a8f93c9a3fec9a3",
    "custom-sequences-s20291777": "cccf943f8f3acb1986a43543c2b79ecccb90afa9e565d89a65f9450586b1ebe7",
    "custom-sequences-s20299696": "a5d7fa22598e1df1608bc974b28d3b5bb4fbee07ab37a925dc331f4724cde0bd",
    "custom-sequences-s20307615": "a9bb9cb2dd3346ae7750497f48be58ae88e21f57d1b897b29590f25a5153bb21",
    "column-early-exit-sink-s20315534": "f237e4eb9bc553eb64394f39f1bd2e08ed637fc0ffaa4e5ff885d77c268c3a1d",
    "column-early-exit-sink-s20323453": "1ea722b397c4631cdf9380887df98af4186fbe2df67da0e66dd9d5cb254e3651",
    "column-exit-just-below-sink-s20355129": "6048c4dcf014bbb3cef1caf6cf43bf3b39d29508385177f31252cfe43a37b91c",
    "column-exit-just-below-sink-s20363048": "f9662f3bc7719b6d92aa96166d292a9e10b9f98a6146bdad9522e2b150d55db4",
    "column-exit-just-above-sink-s20370967": "81ce84542cd75939d12058a6423a30df97e1892b5bc23eedea58229c5266cbd0",
    "column-exit-just-above-sink-s20378886": "eea67274b464b56c4468bb3747cba97dc8c83eb0178b15a36b8818169bfdb2f7",
    "robot-pool-fallback-s20315534": "eee7de65845225318290da865ed210402c48679aa6b7fd98f1145e64b95f6d83",
    "robot-pool-fallback-s20323453": "cc3b90382ffd86e3df563397d2a4d2efe00daaf35920c077c7cdee4adba6e800",
    "robot-pool-fallback-s20331372": "9460b3b54183362fd958b50b869cac97398d0b04a8c4a07f2001c38bee6ae2b7",
    "robot-pool-fallback-mixed-s20315534": "1e6b819125734f08ac82617947a61b29c187d98477e80c126259dd8b1cce5c19",
    "robot-pool-fallback-mixed-s20323453": "01a392562d5d17f5fa63f75efec4bc370fb3a168b5ae13304cfc8927564edc87",
    "robot-pool-fallback-mixed-s20331372": "86b084336e777d1edce74010f4a0c1148620cab8714aea7dad302887a7caa43d",
    "no-difficulty-negative-budgets-s20307615": "7cac94f71084b9e76963b38332779718ec6c9c127cbac738b2ca9d546bb947c1",
    "sink-enclosed-compound-s20339291": "cf43834efbbaa5a22d52ea6b95ffd6d4a0474e16c7d6c7cfb63c235baab01993",
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
    levels_open = _sinks("enclosed-args-levels-0-1")
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

    def fallbacks(name):
        seeds, n, calls, difficulty, origins, make_params, _ = CASES[name]
        out = []
        for seed in seeds:
            env = _run_case(_proc_room, seed, n, calls, difficulty, origins, make_params, False)[0]
            same = (env._proc_room_robot_spawn_count == env._proc_room_spawn_count) & torch.all(
                (env._proc_room_robot_spawn_pts == env._proc_room_spawn_pts).flatten(1), dim=1
            )
            out.append(same)
        return out

    assert sum(int(f.sum()) for f in fallbacks("robot-pool-fallback")) > 0
    for same in fallbacks("robot-pool-fallback-mixed"):  # each seed's batch holds both kinds of env
        assert bool(same.any()) and not bool(same.all())

    seeds, n, calls, difficulty, origins, make_params, _ = CASES["no-difficulty-negative-budgets"]
    env = _run_case(_proc_room, seeds[0], n, calls, difficulty, origins, make_params, False)[0]
    placed = env._proc_room_active_mask
    assert not bool(placed[:, _proc_room.FURNITURE_SLOTS].any()) and not bool(placed[:, _proc_room.CLUTTER_SLOTS].any())

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
