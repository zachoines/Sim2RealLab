# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""``randomize_friction`` on a stand-in robot: one draw per reset env sets every collision shape.

The stand-in's material table has 57 rows, the robot USD's collision-shape count. A shape that started at a rubber
friction (1.0 / 0.9) takes the draw like every other, so a material that came to resolve on the roller covers could not
hold their grip fixed.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch

NUM_ENVS, NUM_SHAPES = 8, 57


class _View:
    def __init__(self, table):
        self.table = table
        self.calls = 0

    def get_material_properties(self):
        self.calls += 1
        return self.table.clone()

    def set_material_properties(self, materials, env_ids):
        import warp as wp

        self.calls += 1
        self.written = wp.to_torch(materials).clone()
        self.ids = wp.to_torch(env_ids).clone()


def _env(table):
    view = _View(table)
    return SimpleNamespace(scene={"robot": SimpleNamespace(root_view=view)}), view


def _table(static=0.5, dynamic=0.5, restitution=0.0):
    t = torch.tensor([static, dynamic, restitution]).repeat(NUM_ENVS, NUM_SHAPES, 1)
    t[:, :, 2] = torch.linspace(0.0, 0.3, NUM_SHAPES)  # distinct restitutions, to see they are left alone
    return t


@functools.lru_cache(maxsize=None)
def _ranges():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    cfgs = {t: getattr(composed, f"StraferNavCfg_RLNoCamSubgoal_{t}")() for t in ("Real", "Robust")}
    return {t: cfg.events.randomize_friction.params["friction_range"] for t, cfg in cfgs.items()}


@pytest.mark.parametrize("tier", ["Real", "Robust"])
def test_every_shape_of_a_reset_env_takes_its_envs_draw(tier):
    from strafer_lab.tasks.navigation.mdp import events

    lo, hi = _ranges()[tier]
    table = _table()
    env, view = _env(table)
    ids = torch.tensor([1, 4, 6])
    torch.manual_seed(11)
    events.randomize_friction(env, ids, friction_range=(lo, hi))
    torch.manual_seed(11)
    want = torch.rand(len(ids)) * (hi - lo) + lo  # the draw, in the term's own expression
    out = view.written
    for k, e in enumerate(ids.tolist()):
        assert torch.equal(out[e, :, 0], want[k].expand(NUM_SHAPES))
        assert torch.equal(out[e, :, 1], want[k].expand(NUM_SHAPES) * 0.9)
    assert torch.equal(out[:, :, 2], table[:, :, 2])  # restitution untouched
    others = [e for e in range(NUM_ENVS) if e not in ids.tolist()]
    assert torch.equal(out[others], table[others])  # other envs untouched
    assert view.ids.tolist() == ids.tolist()


def test_shapes_that_start_at_rubber_still_take_the_draw():
    """The flip guard: had the covers' rubber material resolved (1.0 / 0.9), they would take the draw anyway."""
    from strafer_lab.tasks.navigation.mdp import events

    table = _table()
    table[:, :40, 0], table[:, :40, 1] = 1.0, 0.9
    env, view = _env(table)
    torch.manual_seed(3)
    events.randomize_friction(env, torch.arange(NUM_ENVS), friction_range=(0.3, 1.5))
    out = view.written
    assert torch.equal(out[:, :40, 0], out[:, 40:41, 0].expand(-1, 40))  # covers and chassis share each env's draw
    assert torch.equal(out[:, :40, 1], out[:, 40:41, 1].expand(-1, 40))  # the dynamic column too
    assert torch.equal(out[:, :, 1], out[:, :, 0] * 0.9)
    assert bool((out[:, :, 0] != 1.0).all())


def test_an_empty_reset_reads_and_writes_nothing():
    from strafer_lab.tasks.navigation.mdp import events

    env, view = _env(_table())
    events.randomize_friction(env, torch.tensor([], dtype=torch.long), friction_range=(0.6, 1.2))
    assert view.calls == 0


def test_the_term_keeps_no_state_on_the_env():
    from strafer_lab.tasks.navigation.mdp import events

    env, view = _env(_table())
    before = set(vars(env))
    events.randomize_friction(env, torch.arange(NUM_ENVS), friction_range=(0.6, 1.2))
    assert set(vars(env)) == before
