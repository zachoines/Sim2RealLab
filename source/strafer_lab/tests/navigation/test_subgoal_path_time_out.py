# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The subgoal episode's path-scaled time-out (`mdp.path_time_out`, registered as `time_out`).

The term is driven on a stand-in env that holds what it reads: the command's `path_cursor.total_arc`, the action
term's cfg and `speed_scale`, `max_episode_length`, `step_dt` and `episode_length_buf`. A limit is the first counter
at which the term fires. The composition and the action-term seam are read from the composed cfgs and a built action
term.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import dataclasses
import functools
from types import SimpleNamespace

import pytest
import torch

from strafer_shared.constants import POLICY_MISSION_TIMEOUT_S

PARAMS = dict(command_name="goal_command", action_name="wheel_velocities", intercept_s=11.87, time_per_m_s=1.53,
              reference_speed_scale=0.5, ceiling_s=POLICY_MISSION_TIMEOUT_S)
ENVELOPE = (0.255, 1.0)
# (L m, s) -> limit in control steps at 30 Hz with a 600-step base
REFERENCE = {(2.0, 0.255): 879, (2.0, 0.5): 600, (5.4, 0.5): 604, (5.4, 1.0): 604, (8.0, 0.255): 1419,
             (8.0, 0.4): 905, (8.0, 1.0): 724, (10.0, 0.255): 1599, (10.0, 0.3): 1359, (12.0, 0.255): 1779,
             (12.3, 0.255): 1800, (16.0, 0.4): 1364, (20.0, 0.5): 1275, (25.55, 0.5): 1529}


def _mdp():
    from strafer_lab.tasks.navigation import mdp

    return mdp


class _ActionTerm:
    def __init__(self, s, speed_scale_range):
        self.cfg = SimpleNamespace(speed_scale_range=speed_scale_range)
        if s is not None:
            self.speed_scale = torch.as_tensor(s, dtype=torch.float32)


def _env(arc, s, speed_scale_range=ENVELOPE, base=600):
    arc = torch.as_tensor(arc, dtype=torch.float32)
    command = SimpleNamespace(path_cursor=SimpleNamespace(total_arc=arc))
    action = _ActionTerm(s, speed_scale_range)
    return SimpleNamespace(
        max_episode_length=base, step_dt=1 / 30, episode_length_buf=torch.zeros(len(arc), dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        action_manager=SimpleNamespace(get_term=lambda name: action),
    )


def _limits(env, top=1900):
    """The first counter at which the term fires, per env (-1 if it never fires up to `top`)."""
    limit = torch.full_like(env.episode_length_buf, -1)
    for counter in range(top + 1):
        env.episode_length_buf[:] = counter
        fired = _mdp().path_time_out(env, **PARAMS)
        limit = torch.where((limit < 0) & fired, torch.full_like(limit, counter), limit)
    return limit


def test_the_reference_limits_to_the_step():
    keys = list(REFERENCE)
    got = _limits(_env([k[0] for k in keys], [k[1] for k in keys])).tolist()
    assert dict(zip(keys, got)) == REFERENCE


def test_off_envelope_short_paths_equal_the_stock_time_out():
    env = _env(torch.linspace(0.0, 5.31, 200), None, speed_scale_range=None)
    for counter in range(1901):
        env.episode_length_buf[:] = counter
        assert torch.equal(_mdp().path_time_out(env, **PARAMS), _mdp().time_out(env)), counter


def test_wherever_the_rule_gives_the_base_the_term_is_the_stock_one():
    arc, s = torch.meshgrid(torch.linspace(0.0, 25.55, 60), torch.linspace(0.255, 1.0, 40), indexing="ij")
    seconds = torch.clamp(0.5 / s, min=1.0) * (11.87 + 1.53 * arc)
    at_base = seconds <= 600 / 30
    assert at_base.any() and not at_base.all()
    env = _env(arc[at_base], s[at_base])
    for counter in range(1901):
        env.episode_length_buf[:] = counter
        assert torch.equal(_mdp().path_time_out(env, **PARAMS), _mdp().time_out(env)), counter


def test_the_base_follows_the_env():
    limit = _limits(_env(torch.linspace(0.0, 25.55, 50), torch.linspace(0.255, 1.0, 50), base=1800))
    assert bool((limit == 1800).all())


def test_each_env_has_its_own_limit_and_no_state():
    env = _env([8.0, 8.0], [0.5, 0.5])
    assert _limits(env).tolist() == [724, 724]
    env.action_manager.get_term("wheel_velocities").speed_scale[1] = 0.255
    assert _limits(env).tolist() == [724, 1419]


def test_an_envelope_without_the_accessor_raises():
    with pytest.raises(AttributeError):
        _mdp().path_time_out(_env([3.0], None, speed_scale_range=ENVELOPE), **PARAMS)


def test_the_deploy_cap_pinned_either_way_gives_one_limit():
    arc = torch.linspace(0.0, 25.55, 60)
    assert torch.equal(_limits(_env(arc, [0.5] * 60)), _limits(_env(arc, [1.0] * 60)))


def test_monotone_in_the_arc_and_the_scale_and_bounded():
    arc, s = torch.meshgrid(torch.linspace(0.0, 25.55, 40), torch.linspace(0.255, 1.0, 31), indexing="ij")
    limit = _limits(_env(arc.flatten(), s.flatten())).reshape(40, 31)
    assert bool((limit[1:] >= limit[:-1]).all()) and bool((limit[:, 1:] <= limit[:, :-1]).all())
    assert bool((limit >= 600).all()) and bool((limit <= 1800).all())


@functools.lru_cache(maxsize=None)
def _composed():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    cfgs = {}
    for name in sorted(n for n in dir(composed) if n.startswith("StraferNavCfg_")):
        cls = getattr(composed, name)
        if [f.default_factory().kind for f in dataclasses.fields(cls) if f.name == "scene_source"] == ["infinigen"]:
            continue  # reads the scene corpus at construction; no Infinigen class has the subgoal objective
        cfgs[name] = cls()
    return cfgs


def test_every_subgoal_composition_carries_the_term_and_every_goal_one_the_stock():
    subgoal = goal = 0
    for name, cfg in _composed().items():
        terms = {k: v for k, v in vars(cfg.terminations).items() if hasattr(v, "time_out")}
        timed = [k for k, v in terms.items() if v is not None and v.time_out]
        assert timed == ["time_out"], (name, timed)
        term = cfg.terminations.time_out
        assert cfg.episode_length_s == 20.0, name
        if cfg.objective.kind == "subgoal":
            subgoal += 1
            assert term.func is _mdp().path_time_out and term.params == PARAMS, name
            assert term.params["ceiling_s"] == POLICY_MISSION_TIMEOUT_S
        else:
            goal += 1
            assert term.func is _mdp().time_out and not term.params, name
    assert subgoal >= 8 and goal >= 8, (subgoal, goal)


class _Asset:
    def find_joints(self, names, preserve_order=False):
        return [0, 1, 2, 3], ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]


def _action_term(name):
    cfg = _composed()[name]
    wv = cfg.actions.wheel_velocities
    env = SimpleNamespace(scene={wv.asset_name: _Asset()}, device="cpu", num_envs=16,
                          physics_dt=cfg.sim.dt, step_dt=cfg.sim.dt * cfg.decimation)
    return wv, wv.class_type(wv, env)


def test_the_action_term_exposes_the_scale_the_term_reads():
    wv, term = _action_term("StraferNavCfg_RLDepthSubgoal_Robust")
    assert wv.speed_scale_range == ENVELOPE
    assert torch.equal(term.speed_scale, term._speed_scale)
    assert bool(((term.speed_scale >= ENVELOPE[0]) & (term.speed_scale <= ENVELOPE[1])).all())
    wv, term = _action_term("StraferNavCfg_RLDepthSubgoal_Real")
    assert wv.speed_scale_range is None
    with pytest.raises(AttributeError):
        term.speed_scale
