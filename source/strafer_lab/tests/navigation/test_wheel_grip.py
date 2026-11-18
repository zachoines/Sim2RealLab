# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""``randomize_wheel_grip`` on a stand-in robot: the wheel-floor grip drawn on the roller covers.

The stand-in's material table has 57 rows, the robot USD's collision-shape count: the first 40 the roller covers (one
shape on each of the 40 ``roller_cover`` links), the rest the chassis. The term draws the effective static grip mu_eff
per reset env and writes the covers 2 mu_eff - floor (PhysX averages the two materials' coefficients); by default every
shape takes that value, as the friction term before it wrote one draw to all 57.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch

NUM_ENVS, NUM_SHAPES, NUM_COVERS = 8, 57, 40
FLOOR = 0.5


class _View:
    def __init__(self, table, links=None):
        self.table = table
        self.calls = 0
        self.link_paths = [links or []]

    def get_material_properties(self):
        self.calls += 1
        return self.table.clone()

    def set_material_properties(self, materials, env_ids):
        import warp as wp

        self.calls += 1
        self.written = wp.to_torch(materials).clone()
        self.ids = wp.to_torch(env_ids).clone()


class _SimView:
    def create_rigid_body_view(self, path):
        return SimpleNamespace(max_shapes=0 if "roller_axle" in path else 1)


def _links(covers=NUM_COVERS):
    """The USD's link layout: covers and axles (no shape) per wheel, then the chassis's 17 shapes."""
    links = []
    for w in range(4):
        for r in range(10):
            links.append(f"/Robot/wheel_{w}/node_606_XXXX_0096_roller_axle__{w}__{r}")
            if len([l for l in links if "cover" in l]) < covers:
                links.append(f"/Robot/wheel_{w}/node_606_XXXX_0096_roller_cover__{w}__{r}")
    links += [f"/Robot/frame/rail_{k}" for k in range(NUM_SHAPES - covers)]
    return links


def _env(table, covers=NUM_COVERS):
    view = _View(table, _links(covers))
    robot = SimpleNamespace(root_view=view, _physics_sim_view=_SimView())
    return SimpleNamespace(scene={"robot": robot}, num_envs=NUM_ENVS, device="cpu"), view


def _table(static=0.5, dynamic=0.5):
    t = torch.tensor([static, dynamic, 0.0]).repeat(NUM_ENVS, NUM_SHAPES, 1)
    t[:, :, 2] = torch.linspace(0.0, 0.3, NUM_SHAPES)  # distinct restitutions, to see they are left alone
    return t


def _params(**kw):
    p = dict(grip_range=(0.55, 0.85), floor_friction=FLOOR, dynamic_ratio=0.9, chassis_mode="same",
             chassis_friction=0.5, chassis_friction_range=None)
    p.update(kw)
    return p


def _term(env, **kw):
    from strafer_lab.tasks.navigation.mdp import events

    params = _params(**kw)
    return events.randomize_wheel_grip(SimpleNamespace(params=params), env), params


@functools.lru_cache(maxsize=None)
def _tier_params():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    cfgs = {t: getattr(composed, f"StraferNavCfg_RLNoCamSubgoal_{t}")() for t in ("Real", "Robust")}
    return {t: dict(cfg.events.randomize_friction.params) for t, cfg in cfgs.items()}


@pytest.mark.parametrize("tier", ["Real", "Robust"])
def test_by_default_every_shape_takes_the_covers_draw_as_the_friction_term_wrote(tier):
    """The default writes what the friction term before it wrote: one uniform draw d per reset env, in its own
    expression, on all 57 shapes (static d, dynamic 0.9 d), with d's band today's (0.6, 1.2) / (0.3, 1.5)."""
    params = _tier_params()[tier]
    d_lo, d_hi = {"Real": (0.6, 1.2), "Robust": (0.3, 1.5)}[tier]
    table = _table()
    env, view = _env(table)
    term, _ = _term(env, **params)
    ids = torch.tensor([1, 4, 6])
    torch.manual_seed(11)
    term(env, ids, **params)
    torch.manual_seed(11)
    want = torch.rand(len(ids)) * (d_hi - d_lo) + d_lo
    out = view.written
    for k, e in enumerate(ids.tolist()):
        assert torch.equal(out[e, :, 0], want[k].expand(NUM_SHAPES))
        assert torch.equal(out[e, :, 1], want[k].expand(NUM_SHAPES) * 0.9)
    assert torch.equal(out[:, :, 2], table[:, :, 2])  # restitution untouched
    others = [e for e in range(NUM_ENVS) if e not in ids.tolist()]
    assert torch.equal(out[others], table[others])  # other envs untouched
    assert view.ids.tolist() == ids.tolist()


@pytest.mark.parametrize("tier", ["Real", "Robust"])
def test_the_effective_grip_is_the_band_the_cfg_states(tier):
    params = _tier_params()[tier]
    env, view = _env(_table())
    term, _ = _term(env, **params)
    torch.manual_seed(2)
    grips = []
    for _ in range(200):
        term(env, torch.arange(NUM_ENVS), **params)
        grips.append((view.written[:, :NUM_COVERS, 0] + params["floor_friction"]) / 2)  # PhysX's average
    g = torch.cat(grips)
    lo, hi = params["grip_range"]
    assert float(g.min()) >= lo - 1e-6 and float(g.max()) <= hi + 1e-6
    assert float(g.min()) < lo + 0.02 and float(g.max()) > hi - 0.02  # the draw spans the band


@pytest.mark.parametrize("mode", ["fixed", "band"])
def test_outside_same_the_covers_take_the_grip_and_the_chassis_its_own(mode):
    table = _table()
    env, view = _env(table)
    kw = dict(chassis_mode=mode, chassis_friction=0.45, chassis_friction_range=(0.3, 0.4))
    term, params = _term(env, **kw)
    ids = torch.tensor([0, 3, 5])
    torch.manual_seed(7)
    term(env, ids, **params)
    torch.manual_seed(7)
    cover = torch.rand(len(ids)) * (1.2 - 0.6) + 0.6
    chassis = torch.full((len(ids),), 0.45) if mode == "fixed" else torch.rand(len(ids)) * (0.4 - 0.3) + 0.3
    out = view.written
    for k, e in enumerate(ids.tolist()):
        assert torch.allclose(out[e, :NUM_COVERS, 0], cover[k].expand(NUM_COVERS))
        assert torch.allclose(out[e, NUM_COVERS:, 0], chassis[k].expand(NUM_SHAPES - NUM_COVERS))
        assert torch.allclose(out[e, :, 1], out[e, :, 0] * 0.9)
    assert torch.equal(out[:, :, 2], table[:, :, 2])
    others = [e for e in range(NUM_ENVS) if e not in ids.tolist()]
    assert torch.equal(out[others], table[others])


def test_the_covers_resolve_to_forty_rows_or_the_term_says_so():
    env, view = _env(_table())
    term, params = _term(env, chassis_mode="fixed")
    term(env, torch.arange(NUM_ENVS), **params)
    assert term._cover_rows.tolist() == list(range(NUM_COVERS))
    env39, _ = _env(_table(), covers=39)
    term39, params = _term(env39, chassis_mode="fixed")
    with pytest.raises(RuntimeError, match="39 roller-cover shapes"):
        term39(env39, torch.arange(NUM_ENVS), **params)


def test_a_band_below_half_the_floor_fails_when_the_term_is_built_naming_the_fallback():
    env, _ = _env(_table())
    with pytest.raises(ValueError, match="combine mode"):
        _term(env, grip_range=(0.2, 0.6))
    _term(env, grip_range=(0.25, 0.6))  # mu_cover = 0: the lowest the average represents


def test_a_band_at_a_point_draws_nothing():
    env, view = _env(_table())
    term, params = _term(env, grip_range=(0.7, 0.7))
    torch.manual_seed(5)
    term(env, torch.arange(NUM_ENVS), **params)
    after = torch.rand(4)
    torch.manual_seed(5)
    assert torch.equal(after, torch.rand(4))
    assert torch.allclose(view.written[:, :, 0], torch.full((NUM_ENVS, NUM_SHAPES), 0.9))  # 2 x 0.7 - 0.5


def test_an_empty_reset_reads_and_writes_nothing():
    env, view = _env(_table())
    term, params = _term(env)
    term(env, torch.tensor([], dtype=torch.long), **params)
    assert view.calls == 0


def test_shapes_that_start_at_rubber_still_take_the_draw():
    """The flip guard: had the covers' rubber material resolved (1.0 / 0.9), they would take the draw anyway."""
    table = _table()
    table[:, :NUM_COVERS, 0], table[:, :NUM_COVERS, 1] = 1.0, 0.9
    env, view = _env(table)
    term, params = _term(env, grip_range=(0.40, 1.00))
    torch.manual_seed(3)
    term(env, torch.arange(NUM_ENVS), **params)
    out = view.written
    assert torch.equal(out[:, :NUM_COVERS, 0], out[:, NUM_COVERS:NUM_COVERS + 1, 0].expand(-1, NUM_COVERS))
    assert torch.equal(out[:, :, 1], out[:, :, 0] * 0.9)
    assert bool((out[:, :, 0] != 1.0).all())


def test_every_scene_floor_is_the_terms_floor():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    names = [n for n in dir(composed) if n.startswith("StraferNavCfg_") and n.endswith(("_Real", "_Robust"))]
    for name in names:
        cfg = getattr(composed, name)()
        term = getattr(cfg.events, "randomize_friction", None)
        if term is None:
            continue
        floor = cfg.scene.terrain.physics_material
        assert floor.static_friction == term.params["floor_friction"] == floor.dynamic_friction, name
