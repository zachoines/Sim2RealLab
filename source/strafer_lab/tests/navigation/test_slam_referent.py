# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The referent's age and replanning: a TF-age delay on the goal-shaped observations, a periodic replan of the subgoal
path, both built and off by default.

With the age on, the perceived goal-shaped terms project the goal through the base pose k control steps old (k per env
in its band, redrawn on an interval, clipped to the history since the reset); the critic's terms read the live pose.
With the replan on, each env's path is re-planned on its period from the robot's position to the episode's goal, with
the reset's planner; the episode's time-out keeps the reset plan's arc. Off, neither runs.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

N = 4


class _Scene(dict):
    def __init__(self, entities, origins):
        super().__init__(entities)
        self.env_origins = origins


def _pose(t, n=N):
    ang = 0.05 * t + torch.arange(n) * 0.4
    pos = torch.stack([0.5 * torch.cos(ang), 0.5 * torch.sin(ang), torch.zeros(n)], 1)
    yaw = ang + math.pi / 2
    quat = torch.stack([torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2), torch.cos(yaw / 2)], 1)
    return pos, quat


def _goal_env(n=N):
    data = SimpleNamespace()
    env = SimpleNamespace(num_envs=n, device="cpu", common_step_counter=0,
                          scene=_Scene({"robot": SimpleNamespace(data=data)}, torch.zeros(n, 3)))
    goal = torch.tensor([[2.0, 1.0, 0.5]]).repeat(n, 1)
    env.command_manager = SimpleNamespace(get_command=lambda name: goal)
    return env, data


def _set(data, pos, quat):
    data.root_pos_w = wp.from_torch(pos.contiguous())
    data.root_quat_w = wp.from_torch(quat.contiguous())


TERMS = ("goal_position_relative", "goal_distance", "goal_heading_relative", "goal_heading_to_goal")


def test_with_the_age_on_the_perceived_terms_read_the_pose_k_steps_old_and_the_critic_the_live_one():
    from strafer_lab.tasks.navigation.mdp import events
    from strafer_lab.tasks.navigation.mdp import observations as o

    env, data = _goal_env()
    live_env, live_data = _goal_env()
    torch.manual_seed(5)
    events.randomize_tf_age(env, torch.arange(N), age_steps_range=(0, 3), interval_steps=1000)
    ages = env._tf_age.age.clone()
    assert int(ages.min()) >= 0 and int(ages.max()) <= 3
    poses = []
    for t in range(1, 12):
        env.common_step_counter = t
        pos, quat = _pose(t)
        poses.append((pos, quat))
        _set(data, pos, quat)
        for name in TERMS:
            fn = getattr(o, name)
            perceived = fn(env, "goal_command", True)
            k = torch.minimum(ages, torch.tensor(t - 1))
            want = torch.empty_like(perceived)
            for e in range(N):
                old_pos, old_quat = poses[t - 1 - int(k[e])]
                _set(live_data, old_pos, old_quat)
                want[e] = fn(live_env, "goal_command", False)[e]
            assert torch.allclose(perceived, want, atol=1e-6), (name, t)
            _set(live_data, pos, quat)
            assert torch.equal(fn(env, "goal_command", False), fn(live_env, "goal_command", False)), name


def test_the_age_is_redrawn_on_its_interval_and_at_a_reset():
    from strafer_lab.tasks.navigation.mdp import events
    from strafer_lab.tasks.navigation.mdp import observations as o

    env, data = _goal_env(n=256)
    torch.manual_seed(8)
    events.randomize_tf_age(env, torch.arange(256), age_steps_range=(0, 4), interval_steps=3)
    first = env._tf_age.age.clone()
    seen = [first]
    for t in range(1, 7):
        env.common_step_counter = t
        _set(data, *_pose(t, 256))
        o.goal_position_relative(env, "goal_command", True)
        seen.append(env._tf_age.age.clone())
    assert torch.equal(seen[1], first) and torch.equal(seen[2], first)  # held within the interval
    assert not torch.equal(seen[3], first)  # redrawn on it
    before = env._tf_age.age.clone()
    events.randomize_tf_age(env, torch.tensor([7, 9]), age_steps_range=(0, 4), interval_steps=3)
    changed = (env._tf_age.age != before).nonzero().flatten().tolist()
    assert set(changed) <= {7, 9}


def test_off_the_terms_read_the_live_pose_and_no_process_exists():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    for name in ("StraferNavCfg_RLDepthSubgoal_Real", "StraferNavCfg_RLNoCamSubgoal_Robust", "StraferNavCfg_RLDepth_Real"):
        cfg = getattr(composed, name)()
        assert cfg.events.randomize_tf_age is None, name
        if cfg.objective.kind == "subgoal":
            assert cfg.commands.goal_command.replan_period_s is None, name


def _command(monkeypatch, period, plan=None, n=2):
    from strafer_lab.tasks.navigation.mdp import commands

    calls = []
    real_plan = commands.plan_path

    def recording(start, goal, *a, **k):
        calls.append((np.array(start, dtype=np.float32), np.array(goal, dtype=np.float32)))
        return (plan or real_plan)(start, goal, *a, **k)

    monkeypatch.setattr(commands, "plan_path", recording)
    data = SimpleNamespace(root_pos_w=wp.from_torch(torch.zeros(n, 3)), root_lin_vel_b=wp.from_torch(torch.zeros(n, 3)))
    robot = SimpleNamespace(is_initialized=True, data=data)
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=1.0 / 30.0, scene=_Scene({"robot": robot}, torch.zeros(n, 3)))
    env._proc_room_free_space = torch.ones(n, 80, 80, dtype=torch.bool)
    pts = torch.tensor([[2.5, 0.0], [0.0, 2.5]])
    env._proc_room_spawn_pts = pts.unsqueeze(0).repeat(n, 1, 1)
    env._proc_room_spawn_count = torch.full((n,), len(pts), dtype=torch.long)
    cfg = commands.SubgoalCommandCfg(asset_name="robot", debug_vis=False, replan_period_s=period)
    term = commands.SubgoalCommand(cfg, env)
    return term, data, calls


def test_with_the_replan_on_the_path_is_replanned_on_its_period_from_the_robots_position(monkeypatch):
    term, data, calls = _command(monkeypatch, period=0.5)
    term.reset(env_ids=torch.arange(2))
    goals = term._goal_local.copy()
    arc0, turning0 = term.path_arc.clone(), term.path_turning.clone()
    n_reset = len(calls)
    for t in range(1, 31):
        pos = torch.tensor([[0.03 * t, 0.01 * t, 0.0], [-0.02 * t, 0.02 * t, 0.0]])
        data.root_pos_w = wp.from_torch(pos.contiguous())
        term.compute(1.0 / 30.0)
        if t < 15:
            assert len(calls) == n_reset, t  # nothing before the period
    replans = calls[n_reset:]
    assert len(replans) == 4  # 2 envs x 2 periods of 15 steps in 30
    for start, goal in replans:
        assert any(np.allclose(goal, g) for g in goals)  # the episode's goal
    assert np.allclose(replans[0][0], [0.45, 0.15], atol=1e-6) and np.allclose(replans[1][0], [-0.30, 0.30], atol=1e-6)
    assert torch.equal(term.path_arc, arc0) and torch.equal(term.path_turning, turning0)  # the time-out's, the reset's


def test_off_the_command_plans_once_at_reset(monkeypatch):
    term, data, calls = _command(monkeypatch, period=None)
    term.reset(env_ids=torch.arange(2))
    n_reset = len(calls)
    for t in range(1, 61):
        data.root_pos_w = wp.from_torch(torch.tensor([[0.02 * t, 0.0, 0.0]] * 2).contiguous())
        term.compute(1.0 / 30.0)
    assert len(calls) == n_reset and term._replan_steps is None
    assert torch.equal(term.path_arc, term.path_cursor.total_arc)  # off, the time-out's arc is the cursor's


def test_a_failed_replan_keeps_the_path(monkeypatch):
    from strafer_lab.tasks.navigation.path_planner import PathPlanningError

    state = {"fail": False}

    def flaky(start, goal, *a, **k):
        if state["fail"]:
            raise PathPlanningError("blocked")
        return np.stack([start, goal]).astype(np.float32)

    term, data, calls = _command(monkeypatch, period=0.1, plan=flaky)
    term.reset(env_ids=torch.arange(2))
    before = term.path_cursor.paths.clone()
    state["fail"] = True
    for _ in range(6):
        term.compute(1.0 / 30.0)
    assert torch.equal(term.path_cursor.paths, before)
