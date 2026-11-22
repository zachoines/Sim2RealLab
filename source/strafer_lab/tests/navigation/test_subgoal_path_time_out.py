# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The subgoal episode's path-scaled time-out (`mdp.path_time_out`, registered as `time_out`).

The term is driven on a stand-in env that holds what it reads: the command's `path_arc` (the reset plan's arc) and
`path_turning`, the action term's cfg and `speed_scale`, `max_episode_length`, `step_dt` and `episode_length_buf`. A
limit is the first counter at which the term fires. The turning the command measures is read from the real
`SubgoalCommand` on a stub env with its planner calls replaced. The composition and the action-term seam are read from
the composed cfgs and a built action term.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from strafer_shared.constants import NAV_ANGULAR_VEL, NAV_VEL_SCALE, POLICY_MISSION_TIMEOUT_S

TURN_RATE = 2.0933  # rad/s, the node's angular cap at the reference scale 0.5, pinned with 11.87 and 1.53
PARAMS = dict(command_name="goal_command", action_name="wheel_velocities", intercept_s=11.87, time_per_m_s=1.53,
              turn_rate_rad_s=TURN_RATE, reference_speed_scale=0.5, ceiling_s=POLICY_MISSION_TIMEOUT_S)
ENVELOPE = (0.255, 1.0)
PI = math.pi
# (L m, s, K rad) -> limit in control steps at 30 Hz with a 600-step base
REFERENCE = {(2.0, 0.255, 0.0): 879, (2.0, 0.5, 0.0): 600, (5.4, 0.5, 0.0): 604, (5.4, 1.0, 0.0): 604,
             (8.0, 0.255, 0.0): 1419, (8.0, 0.4, 0.0): 905, (8.0, 1.0, 0.0): 724, (10.0, 0.255, 0.0): 1599,
             (10.0, 0.3, 0.0): 1359, (12.0, 0.255, 0.0): 1779, (12.3, 0.255, 0.0): 1800, (16.0, 0.4, 0.0): 1364,
             (20.0, 0.5, 0.0): 1275, (25.55, 0.5, 0.0): 1529,
             (8.0, 1.0, PI / 2): 746, (8.0, 0.255, PI / 2): 1463, (2.0, 0.5, 2 * PI): 600, (2.0, 0.255, PI): 967,
             (10.0, 0.3, PI): 1434, (16.0, 0.4, PI / 2): 1392, (20.0, 0.5, 3 * PI): 1410, (5.0, 1.0, PI / 2): 609,
             (11.5, 0.255, PI / 2): 1778, (4.0, 0.5, 2 * PI): 630, (6.0, 0.4, PI): 846}


def _mdp():
    from strafer_lab.tasks.navigation import mdp

    return mdp


class _ActionTerm:
    def __init__(self, s, speed_scale_range):
        self.cfg = SimpleNamespace(speed_scale_range=speed_scale_range)
        if s is not None:
            self.speed_scale = torch.as_tensor(s, dtype=torch.float32)


def _env(arc, s, turning=None, speed_scale_range=ENVELOPE, base=600):
    arc = torch.as_tensor(arc, dtype=torch.float32)
    turning = torch.zeros_like(arc) if turning is None else torch.as_tensor(turning, dtype=torch.float32)
    command = SimpleNamespace(path_arc=arc, path_turning=turning)
    action = _ActionTerm(s, speed_scale_range)
    return SimpleNamespace(
        max_episode_length=base, step_dt=1 / 30, episode_length_buf=torch.zeros(len(arc), dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        action_manager=SimpleNamespace(get_term=lambda name: action),
    )


def _limits(env, top=2500):
    """The first counter at which the term fires, per env (-1 if it never fires up to `top`)."""
    limit = torch.full_like(env.episode_length_buf, -1)
    for counter in range(top + 1):
        env.episode_length_buf[:] = counter
        fired = _mdp().path_time_out(env, **PARAMS)
        limit = torch.where((limit < 0) & fired, torch.full_like(limit, counter), limit)
    return limit


def test_the_reference_limits_to_the_step():
    keys = list(REFERENCE)
    env = _env([k[0] for k in keys], [k[1] for k in keys], [k[2] for k in keys])
    got = _limits(env).tolist()
    assert dict(zip(keys, got)) == REFERENCE
    assert _mdp().path_time_out_limit(env, **PARAMS).tolist() == got


def test_off_envelope_short_straight_paths_equal_the_stock_time_out():
    env = _env(torch.linspace(0.0, 5.31, 200), None, speed_scale_range=None)
    for counter in range(1901):
        env.episode_length_buf[:] = counter
        assert torch.equal(_mdp().path_time_out(env, **PARAMS), _mdp().time_out(env)), counter


def test_wherever_the_rule_gives_the_base_the_term_is_the_stock_one():
    arc, s, k = torch.meshgrid(torch.linspace(0.0, 25.55, 40), torch.linspace(0.255, 1.0, 30),
                               torch.tensor([0.0, PI / 2, 2 * PI]), indexing="ij")
    seconds = torch.clamp(0.5 / s, min=1.0) * (11.87 + 1.53 * arc + k / TURN_RATE)
    at_base = seconds <= 600 / 30
    assert at_base.any() and not at_base.all()
    env = _env(arc[at_base], s[at_base], k[at_base])
    for counter in range(1901):
        env.episode_length_buf[:] = counter
        assert torch.equal(_mdp().path_time_out(env, **PARAMS), _mdp().time_out(env)), counter


@pytest.mark.parametrize("base", [1800, 2400])
def test_the_base_follows_the_env_also_above_the_ceiling(base):
    n = 50
    limit = _limits(_env(torch.linspace(0, 25.55, n), torch.linspace(0.255, 1.0, n), torch.linspace(0, 4 * PI, n),
                         base=base))
    assert bool((limit == base).all())


def test_a_right_angle_adds_an_in_place_turn_at_the_deploy_caps_angular_speed():
    """At s >= 0.5 a 90 degree path gets about 0.75 s more than a straight one of equal length; at the floor 1.47 s."""
    for s, extra_s in ((1.0, 0.75), (0.5, 0.75), (0.255, 1.47)):
        env = _env([10.0, 10.0], [s, s], [0.0, PI / 2])
        straight, turned = _mdp().path_time_out_limit(env, **PARAMS).tolist()
        assert abs((turned - straight) / 30 - extra_s) <= 1 / 30, (s, straight, turned)
    assert (PI / 2) / TURN_RATE == pytest.approx(0.75, abs=0.001)
    assert (PI / 2) / TURN_RATE * 0.5 / 0.255 == pytest.approx(1.47, abs=0.002)


def test_the_reference_scale_and_turn_rate_are_the_nodes_cap_the_bound_was_measured_at():
    """The cfg takes both from the shared constants; 11.87 s and 1.53 s/m were measured at that cap. A retune of
    NAV_VEL_SCALE or of the chassis maximum fails here: measure the bound again at the new cap, then re-pin."""
    from strafer_lab.tasks.navigation import strafer_env_cfg

    assert (NAV_VEL_SCALE, NAV_ANGULAR_VEL) == (PARAMS["reference_speed_scale"], PARAMS["turn_rate_rad_s"])
    assert strafer_env_cfg._SUBGOAL_TIME_OUT_REFERENCE_SCALE == NAV_VEL_SCALE
    assert strafer_env_cfg._SUBGOAL_TIME_OUT_TURN_RATE_RAD_S == NAV_ANGULAR_VEL


def test_each_env_has_its_own_limit_and_no_state():
    env = _env([8.0, 8.0], [0.5, 0.5])
    assert _limits(env).tolist() == [724, 724]
    env.action_manager.get_term("wheel_velocities").speed_scale[1] = 0.255
    assert _limits(env).tolist() == [724, 1419]
    env.command_manager.get_term("goal_command").path_turning[0] = PI / 2
    assert _limits(env).tolist() == [746, 1419]


def test_an_envelope_without_the_accessor_raises():
    with pytest.raises(AttributeError):
        _mdp().path_time_out(_env([3.0], None, speed_scale_range=ENVELOPE), **PARAMS)


def test_the_deploy_cap_pinned_either_way_gives_one_limit():
    arc, k = torch.linspace(0.0, 25.55, 60), torch.linspace(0.0, 4 * PI, 60)
    assert torch.equal(_limits(_env(arc, [0.5] * 60, k)), _limits(_env(arc, [1.0] * 60, k)))


def test_monotone_in_the_arc_the_turning_and_the_scale_and_bounded():
    arc, s, k = torch.meshgrid(torch.linspace(0, 25.55, 20), torch.linspace(0.255, 1.0, 16),
                               torch.linspace(0, 3 * PI, 5), indexing="ij")
    limit = _mdp().path_time_out_limit(_env(arc.flatten(), s.flatten(), k.flatten()), **PARAMS).reshape(20, 16, 5)
    assert bool((limit[1:] >= limit[:-1]).all()) and bool((limit[:, 1:] <= limit[:, :-1]).all())
    assert bool((limit[:, :, 1:] >= limit[:, :, :-1]).all())
    assert bool((limit >= 600).all()) and bool((limit <= 1800).all())


# --- the turning the command measures -----------------------------------------------------------------------------

class _StubScene(dict):
    def __init__(self, entities, env_origins):
        super().__init__(entities)
        self.env_origins = env_origins


def _command(monkeypatch, plan, perturb=None, num_envs=1):
    """The real SubgoalCommand on a stub env, its planner replaced by `plan` (and the perturbation by `perturb`)."""
    from strafer_lab.tasks.navigation.mdp import commands

    monkeypatch.setattr(commands, "plan_path", plan)
    if perturb is not None:
        monkeypatch.setattr(commands, "perturb_waypoints", perturb)
    robot = SimpleNamespace(is_initialized=True, data=SimpleNamespace(
        root_pos_w=wp.from_torch(torch.zeros(num_envs, 3)), root_lin_vel_b=wp.from_torch(torch.zeros(num_envs, 3))))
    env = SimpleNamespace(num_envs=num_envs, device="cpu",
                          scene=_StubScene({"robot": robot}, torch.zeros(num_envs, 3)))
    env._proc_room_free_space = torch.ones(num_envs, 80, 80, dtype=torch.bool)
    pts = torch.tensor([[2.5, 0.0], [0.0, 2.5]])
    env._proc_room_spawn_pts = pts.unsqueeze(0).repeat(num_envs, 1, 1)
    env._proc_room_spawn_count = torch.full((num_envs,), len(pts), dtype=torch.long)
    term = commands.SubgoalCommand(commands.SubgoalCommandCfg(asset_name="robot", debug_vis=False), env)
    return term


def _corner(n_before, n_after, spacing=0.05):
    """A polyline along +x for `n_before` points, then along +y for `n_after`: one 90 degree turn."""
    leg1 = np.stack([np.arange(n_before) * spacing, np.zeros(n_before)], axis=1)
    corner = leg1[-1]
    leg2 = corner + np.stack([np.zeros(n_after), np.arange(1, n_after + 1) * spacing], axis=1)
    return np.concatenate([leg1, leg2]).astype(np.float32)


def test_the_turning_is_the_planned_paths_before_the_perturbation(monkeypatch):
    planned = _corner(30, 30)

    def zigzag(path, *args, **kwargs):
        out = np.array(path, dtype=np.float32)
        out[1:-1:2, 1] += 0.02
        return out

    term = _command(monkeypatch, lambda *a, **k: planned.copy(), perturb=zigzag)
    term.reset(env_ids=[0])
    from strafer_lab.tasks.navigation.path_planner import total_turning

    assert total_turning(term.path_cursor.paths[0, : int(term.path_cursor.path_len[0])].numpy()) > PI  # installed
    assert float(term.path_turning[0]) == pytest.approx(PI / 2, abs=1e-5)


def test_a_fallback_turns_zero(monkeypatch):
    from strafer_lab.tasks.navigation.path_planner import PathPlanningError

    term = _command(monkeypatch, lambda *a, **k: _corner(30, 30), perturb=lambda p, *a, **k: p)
    term.reset(env_ids=[0])
    assert float(term.path_turning[0]) == pytest.approx(PI / 2, abs=1e-5)

    def fail(*a, **k):
        raise PathPlanningError("no path")

    from strafer_lab.tasks.navigation.mdp import commands

    monkeypatch.setattr(commands, "plan_path", fail)
    term.reset(env_ids=[0])
    assert float(term.metrics["path_fallback"][0]) == 1.0 and float(term.path_turning[0]) == 0.0


@pytest.mark.parametrize("corner_at, turning", [(100, PI / 2), (550, 0.0)])
def test_only_the_installed_part_of_the_path_counts(monkeypatch, corner_at, turning):
    term = _command(monkeypatch, lambda *a, **k: _corner(corner_at, 700 - corner_at), perturb=lambda p, *a, **k: p)
    assert term.cfg.max_path_points == 512
    term.reset(env_ids=[0])
    assert int(term.path_cursor.path_len[0]) == 512
    assert float(term.path_turning[0]) == pytest.approx(turning, abs=1e-5)


def _legs(n_turns, spacing=0.05, leg=20):
    """A polyline of n_turns right angles, alternating left and right, each leg `leg` points."""
    pts, heading, p = [np.zeros(2)], 0.0, np.zeros(2)
    for i in range(n_turns + 1):
        d = np.array([math.cos(heading), math.sin(heading)])
        for _ in range(leg):
            p = p + spacing * d
            pts.append(p.copy())
        heading += (PI / 2) * (1 if i % 2 == 0 else -1)
    return np.asarray(pts, dtype=np.float32)


def test_a_subset_reset_writes_only_its_envs_turning(monkeypatch):
    queue = []
    term = _command(monkeypatch, lambda *a, **k: queue.pop(0), perturb=lambda p, *a, **k: p, num_envs=3)
    queue[:] = [_legs(0), _legs(0), _legs(0)]
    term.reset(env_ids=[0, 1, 2])
    assert term.path_turning.tolist() == [0.0, 0.0, 0.0]
    queue[:] = [_legs(1)]
    term.reset(env_ids=[2])
    assert term.path_turning.tolist() == pytest.approx([0.0, 0.0, PI / 2], abs=1e-5)
    queue[:] = [_legs(3), _legs(2)]
    term.reset(env_ids=[1, 0])
    assert term.path_turning.tolist() == pytest.approx([PI, 3 * PI / 2, PI / 2], abs=1e-5)
    assert term.path_turning.dtype == torch.float32 and term.path_turning.device == term.path_cursor.total_arc.device


@pytest.mark.parametrize("n_before, turning", [(511, PI / 2), (512, 0.0)])
def test_a_corner_at_the_last_installed_vertex_turns_nothing(monkeypatch, n_before, turning):
    """A corner at vertex 510 is inside the 512 installed points; at vertex 511, the last, it has no installed
    outgoing segment."""
    from strafer_lab.tasks.navigation.path_planner import total_turning

    term = _command(monkeypatch, lambda *a, **k: _corner(n_before, 700 - n_before), perturb=lambda p, *a, **k: p)
    term.reset(env_ids=[0])
    installed = term.path_cursor.paths[0, : int(term.path_cursor.path_len[0])].numpy()
    assert total_turning(installed) == pytest.approx(turning, abs=1e-5)
    assert float(term.path_turning[0]) == pytest.approx(turning, abs=1e-5)


def test_the_turning_sum_is_the_path_statistics_one():
    from strafer_lab.tasks.navigation.path_planner import total_turning
    from strafer_lab.tools import path_statistics

    for path in (_corner(20, 20), _corner(5, 40), np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])):
        density, _, arc, _ = path_statistics.turn_density(path)
        assert density * arc == pytest.approx(total_turning(path))


# --- the composition and the seam ---------------------------------------------------------------------------------

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
            assert term.params["turn_rate_rad_s"] == 2.0933  # pinned: a retune of the node's caps fails here
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
    assert bool(((term.speed_scale >= ENVELOPE[0]) & (term.speed_scale <= ENVELOPE[1])).all())
    before = term.speed_scale.clone()
    term.reset(torch.tensor([0, 1]))
    after = term.speed_scale
    assert torch.equal(after[2:], before[2:]) and not torch.equal(after[:2], before[:2])
    wv, term = _action_term("StraferNavCfg_RLDepthSubgoal_Real")
    assert wv.speed_scale_range is None
    with pytest.raises(AttributeError):
        term.speed_scale
