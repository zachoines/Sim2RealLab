# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The action term's plant fields: the lag as a time constant, its per-env band, the per-wheel limit, the fall-off.

The lag's alpha = 1 - exp(-step_dt / tau), so a step reaches 63.2 % of its target after tau at any control step; tau
is drawn per env at each reset from its band, after the reset's other draws, and a band at a point draws nothing. The
per-wheel limit binds only above its value and leaves the action scale and the twist caps alone; the acceleration
limit's fall-off lowers it linearly in the wheel's speed and is off by default. Each field reaches the term from the
composed classes.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import functools
import math
from types import SimpleNamespace

import pytest
import torch

TIERS = {"realistic": "StraferNavCfg_RLDepthSubgoal_Real", "robust": "StraferNavCfg_RLDepthSubgoal_Robust"}
N = 4


@functools.lru_cache(maxsize=None)
def _cfg(tier):
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    return getattr(composed, TIERS[tier])()


class _Asset:
    def find_joints(self, names, preserve_order=False):
        return [0, 1, 2, 3], ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]


def _term(tier="realistic", num_envs=N, *, lag=True, accel=None, **overrides):
    """The tier's term with the delay, the hold and the envelope off; the lag on or off; the slew at `accel` or off."""
    cfg = _cfg(tier)
    t = cfg.actions.wheel_velocities.copy()
    t.delay_range_s = (0.0, 0.0)
    t.hold_fraction_range = (0.0, 0.0)
    t.speed_scale_range = None
    t.enable_motor_dynamics = lag
    if accel is None:
        t.max_acceleration_rad_s2, t.max_acceleration_range = math.inf, (math.inf, math.inf)
    else:
        t.max_acceleration_rad_s2, t.max_acceleration_range = accel, (accel, accel)
    for k, v in overrides.items():
        setattr(t, k, v)
    env = SimpleNamespace(scene={t.asset_name: _Asset()}, device="cpu", num_envs=num_envs,
                          physics_dt=cfg.sim.dt, step_dt=cfg.sim.dt * cfg.decimation)
    return t.class_type(t, env), env


def _forward(term, steps, n=N, command=(1.0, 0.0, 0.0)):
    """|wheel 0| of env 0 after each of `steps` control steps of a constant command."""
    action = torch.tensor([command]).repeat(n, 1)
    out = []
    for _ in range(steps):
        term.process_actions(action)
        out.append(float(term.processed_actions[0, 0].abs()))
    return out


@pytest.mark.parametrize("tau", [0.03, 0.05, 0.2])
def test_a_step_reaches_63_percent_at_tau(tau):
    term, env = _term(motor_time_constant=tau, motor_time_constant_range=None)
    y = _forward(term, 400)
    y = [v / y[-1] for v in y]
    for n, v in enumerate(y[:60], start=1):  # the exact form, at every control step
        assert v == pytest.approx(1.0 - math.exp(-n * env.step_dt / tau), abs=1e-5), (tau, n)
    k = next(i for i, v in enumerate(y) if v >= 1.0 - math.exp(-1.0))
    below = y[k - 1] if k else 0.0
    crossing = (k + (1.0 - math.exp(-1.0) - below) / (y[k] - below)) * env.step_dt
    assert abs(crossing - tau) <= env.step_dt, (tau, crossing)


@pytest.mark.parametrize("tier", sorted(TIERS))
def test_alpha_is_the_exact_form_of_the_tier_tau(tier):
    term, env = _term(tier)
    tau = _cfg(tier).actions.wheel_velocities.motor_time_constant
    alpha = term._motor_alpha.double().flatten()
    assert torch.allclose(alpha, torch.full_like(alpha, -math.expm1(-env.step_dt / tau)), atol=1e-7)
    assert torch.allclose(term._motor_keep.double().flatten(), 1.0 - alpha, atol=1e-7)


def test_tau_is_drawn_per_env_inside_its_band_at_reset_after_the_other_draws():
    lo, hi = 0.1, 0.3
    torch.manual_seed(5)
    point, env = _term(num_envs=256, accel=225.0, max_acceleration_range=(175.0, 275.0),
                       motor_time_constant_range=(0.2, 0.2))
    torch.manual_seed(5)
    band, _ = _term(num_envs=256, accel=225.0, max_acceleration_range=(175.0, 275.0),
                    motor_time_constant_range=(lo, hi))
    assert torch.equal(point._max_delta_per_step, band._max_delta_per_step)  # the build's accel draw comes first
    torch.manual_seed(6)
    point.reset(None)
    torch.manual_seed(6)
    band.reset(None)
    assert torch.equal(point._max_delta_per_step, band._max_delta_per_step)  # and the reset's
    tau = -env.step_dt / torch.log1p(-band._motor_alpha.double().flatten())
    assert float(tau.min()) >= lo - 1e-6 and float(tau.max()) <= hi + 1e-6
    assert float(tau.max() - tau.min()) > 0.5 * (hi - lo)  # drawn per env, not one value
    before = band._motor_alpha.clone()
    band.reset(torch.tensor([3, 10]))
    changed = (band._motor_alpha != before).flatten()
    assert changed[[3, 10]].all() and not changed[[0, 1, 2, 4, 5, 6, 7, 8, 9, 11]].any()


def test_a_band_at_a_point_draws_nothing():
    torch.manual_seed(9)
    term, _ = _term(num_envs=16, motor_time_constant_range=(0.2, 0.2))
    term.reset(None)
    term.reset(torch.tensor([1, 2]))
    after = torch.rand(4)
    torch.manual_seed(9)
    assert torch.equal(after, torch.rand(4))


@pytest.mark.parametrize("tier", sorted(TIERS))
def test_the_plant_fields_reach_the_term_from_the_composed_classes(tier):
    from strafer_lab.tasks.navigation import sim_real_cfg as src

    contract = {"realistic": src.REAL_ROBOT_CONTRACT, "robust": src.ROBUST_TRAINING_CONTRACT}[tier].actuator
    wv = _cfg(tier).actions.wheel_velocities
    assert wv.motor_time_constant == contract.motor_time_constant_s
    assert tuple(wv.motor_time_constant_range) == tuple(contract.motor_time_constant_range)
    assert wv.motor_time_constant_range is not None  # set on the tier, so a dropped key would read None
    assert wv.wheel_speed_limit == contract.wheel_speed_limit_rad_s is not None
    assert wv.accel_falloff_speed == contract.accel_falloff_speed_rad_s


@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_the_wheel_limit_binds_only_above_its_value_and_leaves_the_caps(scale):
    free, _ = _term(lag=False, speed_scale_range=(scale, scale))
    limited, _ = _term(lag=False, speed_scale_range=(scale, scale), wheel_speed_limit=20.0)
    for name in ("_velocity_scale", "_max_linear_vel", "_max_angular_vel", "_max_wheel_angular_vel"):
        a, b = getattr(free, name), getattr(limited, name)
        assert (torch.equal(a, b) if torch.is_tensor(a) else a == b), name
    free.reset(None)
    limited.reset(None)
    gen = torch.Generator().manual_seed(3)
    for _ in range(20):
        action = 2.4 * torch.rand(N, 3, generator=gen) - 1.2
        free.process_actions(action)
        limited.process_actions(action)
        f, lim = free.processed_actions, limited.processed_actions
        assert float(lim.abs().max()) <= 20.0 + 1e-6
        assert torch.equal(lim, f.clamp(-20.0, 20.0))  # binds only above its value


def test_the_falloff_lowers_the_limit_linearly_and_is_off_by_default():
    a0, w_nl = 225.0, 40.0
    off, env = _term(lag=False, accel=a0)
    on, _ = _term(lag=False, accel=a0, accel_falloff_speed=w_nl)
    assert math.isinf(off._accel_falloff_speed)
    command = torch.tensor([[1.0, 0.0, 0.0]]).repeat(N, 1)
    prev_on, prev_off = 0.0, 0.0
    full = {}
    for k in range(30):
        off.process_actions(command)
        on.process_actions(command)
        w_on, w_off = float(on.processed_actions[0, 0].abs()), float(off.processed_actions[0, 0].abs())
        assert w_off - prev_off <= a0 * env.step_dt + 1e-4  # off: the flat limit
        allowed = a0 * env.step_dt * max(0.0, 1.0 - prev_on / w_nl)
        assert w_on - prev_on <= allowed + 1e-4
        prev_on, prev_off = w_on, w_off
        for name, w in (("on", w_on), ("off", w_off)):
            if w >= on._max_wheel_angular_vel - 1e-3:  # a full forward command's wheel target
                full.setdefault(name, k)
    assert full["on"] > full["off"]  # the fall-off slows the approach to full speed
