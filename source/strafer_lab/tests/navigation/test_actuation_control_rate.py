# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The motor lag and the acceleration limit, stepped at the control rate, match their stated values.

The action term's `process_actions` runs once per control step (the env's `step_dt`, `decimation` physics steps). Each
test builds the realistic or the robust tier's term on a stand-in env with the composed cfg's physics and control
steps, isolates one stage (the others off), and drives it with one `process_actions` call per control step: the lag's
63 % rise must land within one control step of the stated time constant, and the time from rest to full wheel speed
within one control step of `max_wheel_angular_vel / acceleration`, at each end of the tier's band and at its nominal.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import functools
import math
from types import SimpleNamespace

import numpy as np
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


def _isolated(tier, *, lag, accel=None):
    """The tier's term with one stage on: the motor lag, or the acceleration limit pinned at `accel`."""
    cfg = _cfg(tier)
    t = cfg.actions.wheel_velocities.copy()
    t.min_delay_steps = t.max_delay_steps = 0
    t.hold_fraction_range = (0.0, 0.0)
    t.speed_scale_range = None
    t.enable_motor_dynamics = lag
    if accel is None:
        t.max_acceleration_rad_s2, t.max_acceleration_range = math.inf, (math.inf, math.inf)
    else:
        t.max_acceleration_rad_s2, t.max_acceleration_range = accel, (accel, accel)
    env = SimpleNamespace(scene={t.asset_name: _Asset()}, device="cpu", num_envs=N,
                          physics_dt=cfg.sim.dt, step_dt=cfg.sim.dt * cfg.decimation)
    return t.class_type(t, env), env


def _wheel_speeds(term, steps):
    """|wheel 0| after each of `steps` control steps of a full forward command."""
    command = torch.tensor([[1.0, 0.0, 0.0]]).repeat(N, 1)
    out = []
    for _ in range(steps):
        term.process_actions(command)
        out.append(float(term.processed_actions[0, 0].abs()))
    return np.array(out)


@pytest.mark.parametrize("tier", sorted(TIERS))
def test_the_lag_rises_63_percent_at_its_time_constant(tier):
    term, env = _isolated(tier, lag=True)
    tau = _cfg(tier).actions.wheel_velocities.motor_time_constant
    y = _wheel_speeds(term, 200)
    y = y / y[-1]
    k = int(np.argmax(y >= 1.0 - math.exp(-1.0)))  # y[k] is the response after k + 1 control steps
    below = y[k - 1] if k else 0.0
    crossing = k + (1.0 - math.exp(-1.0) - below) / (y[k] - below)  # control steps, interpolated
    assert abs(crossing * env.step_dt - tau) <= env.step_dt, (tier, crossing * env.step_dt, tau)


@pytest.mark.parametrize("tier", sorted(TIERS))
@pytest.mark.parametrize("where", ["low", "nominal", "high"])
def test_the_wheel_reaches_full_speed_at_its_acceleration(tier, where):
    wv = _cfg(tier).actions.wheel_velocities
    lo, hi = wv.max_acceleration_range
    accel = {"low": lo, "nominal": wv.max_acceleration_rad_s2, "high": hi}[where]
    term, env = _isolated(tier, lag=False, accel=accel)
    v_max = wv.max_wheel_angular_vel
    y = _wheel_speeds(term, 200)
    k = int(np.argmax(y >= v_max - 1e-4)) + 1  # control steps to full speed
    assert y[-1] == pytest.approx(v_max, abs=1e-4)
    assert abs(k * env.step_dt - v_max / accel) <= env.step_dt, (tier, where, k * env.step_dt, v_max / accel)
