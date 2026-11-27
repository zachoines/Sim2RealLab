# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The command delay and hold in seconds, the delay's redraw at reset, its part under one control step, and the IMU's
latency band.

The delay band and the hold's run length are in seconds and converted when the action term is built, so they mean the
same at any decimation: the delay as whole env steps in the buffer, the hold's runs as env steps. A redraw at each
reset, and a part under one env step held at the start of a step's physics sub-steps, are built and off by default;
their draws come after the reset's other draws. The IMU's latency band draws per env at reset, after its bias.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import functools
import math
from types import SimpleNamespace

import pytest
import torch

from strafer_shared.constants import POLICY_PERIOD_S

STEP = POLICY_PERIOD_S
N = 64


@functools.lru_cache(maxsize=None)
def _cfg(tier="Real"):
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    return getattr(composed, f"StraferNavCfg_RLDepthSubgoal_{tier}")()


class _Asset:
    def __init__(self):
        self.targets = []

    def find_joints(self, names, preserve_order=False):
        return [0, 1, 2, 3], ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]

    def set_joint_velocity_target(self, target, joint_ids=None):
        self.targets.append(target.clone())


def _term(tier="Real", num_envs=N, decimation=4, *, plant=False, **overrides):
    """The tier's action term on a stub env at the given decimation; the lag and the slew off unless `plant`."""
    cfg = _cfg(tier)
    t = cfg.actions.wheel_velocities.copy()
    if not plant:
        t.enable_motor_dynamics = False
        t.max_acceleration_rad_s2, t.max_acceleration_range = math.inf, (math.inf, math.inf)
    for k, v in overrides.items():
        setattr(t, k, v)
    asset = _Asset()
    env = SimpleNamespace(scene={t.asset_name: asset}, device="cpu", num_envs=num_envs,
                          physics_dt=cfg.sim.dt, step_dt=cfg.sim.dt * decimation)
    return t.class_type(t, env), asset, env


def test_the_redraw_keeps_the_per_env_marginal():
    torch.manual_seed(3)
    term, _, _ = _term(delay_range_s=(STEP, 3 * STEP), delay_redraw_on_reset=True, hold_fraction_range=(0.0, 0.0))
    draws = [term._delay_env_steps.clone()]
    for _ in range(157):  # 158 x 64 = 10,112 draws
        term.reset(None)
        draws.append(term._delay_env_steps.clone())
    d = torch.cat(draws)
    n = d.numel()
    assert set(d.tolist()) == {1, 2, 3}
    for v in (1, 2, 3):
        share = (d == v).double().mean().item()
        assert abs(share - 1 / 3) < 6 * math.sqrt((1 / 3) * (2 / 3) / n), (v, share)
    assert torch.equal(term._action_delay_buffer.time_lags, d[-N:].int())  # the buffer takes the drawn lag


def test_the_redraw_and_the_sub_step_draw_come_after_the_resets_other_draws():
    kw = dict(plant=True, tier="Robust", motor_time_constant_range=(0.15, 0.35), hold_fraction_range=(0.0, 0.05))
    torch.manual_seed(11)
    off, _, _ = _term(delay_range_s=(STEP, 5 * STEP), **kw)
    torch.manual_seed(11)
    on, _, _ = _term(delay_range_s=(STEP, 5 * STEP), delay_redraw_on_reset=True,
                     substep_delay_range_s=(0.0, 3 * _cfg().sim.dt), **kw)
    ids = torch.tensor([2, 9, 33])
    for _ in range(3):
        torch.manual_seed(12)
        off.reset(ids)
        torch.manual_seed(12)
        on.reset(ids)
        assert torch.equal(off._max_delta_per_step, on._max_delta_per_step)
        assert torch.equal(off._motor_alpha, on._motor_alpha)
        assert torch.equal(off._speed_scale, on._speed_scale)
        assert torch.equal(off._hold._enter_p, on._hold._enter_p)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_the_sub_step_part_holds_the_previous_target_for_k_sub_steps(k):
    tick = _cfg().sim.dt
    term, asset, _ = _term(num_envs=4, delay_range_s=(0.0, 0.0), substep_delay_range_s=(k * tick, k * tick),
                           hold_fraction_range=(0.0, 0.0), speed_scale_range=None)
    a, b = torch.tensor([[0.5, 0.0, 0.0]]).repeat(4, 1), torch.tensor([[-0.3, 0.2, 0.1]]).repeat(4, 1)
    term.process_actions(a)
    for _ in range(4):
        term.apply_actions()
    first = term.processed_actions.clone()
    term.process_actions(b)
    second = term.processed_actions.clone()
    asset.targets.clear()
    for _ in range(4):
        term.apply_actions()
    for i, target in enumerate(asset.targets):
        assert torch.equal(target, first if i < k else second), (k, i)


def test_off_by_default_the_term_applies_its_target_on_every_sub_step():
    term, asset, _ = _term(tier="Robust")
    assert not term._enable_substep_delay and not term._redraw_delay
    term.process_actions(torch.full((N, 3), 0.4))
    for _ in range(4):
        term.apply_actions()
    assert len(asset.targets) == 4 and all(torch.equal(t, term.processed_actions) for t in asset.targets)


def test_the_delay_and_the_hold_mean_the_same_seconds_at_another_decimation():
    tick = _cfg().sim.dt
    kw = dict(delay_range_s=(2 * STEP, 2 * STEP), hold_fraction_range=(0.0, 0.05), hold_run_range_s=(STEP, 1.2 * STEP))
    four, _, env4 = _term(decimation=4, **kw)
    one, _, env1 = _term(decimation=1, **kw)
    lag4 = int(four._action_delay_buffer.time_lags[0]) * env4.step_dt
    lag1 = int(one._action_delay_buffer.time_lags[0]) * env1.step_dt
    assert lag4 == pytest.approx(2 * STEP) and lag1 == pytest.approx(lag4, abs=tick / 2)
    runs4 = [r * env4.step_dt for r in four._hold._run_range]
    runs1 = [r * env1.step_dt for r in one._hold._run_range]
    assert runs4 == pytest.approx([STEP, 1.2 * STEP]) and runs1 == pytest.approx(runs4, abs=tick / 2)


def test_at_decimation_4_the_tiers_draw_their_whole_step_delays_and_runs():
    for tier, top in (("Real", 3), ("Robust", 5)):
        term, _, _ = _term(tier=tier, num_envs=512)
        assert term._delay_steps_range == (1, top), tier
        assert set(term._action_delay_buffer.time_lags.tolist()) == set(range(1, top + 1)), tier
        assert term._hold._run_range == (1.0, 1.2), tier


def test_a_delay_band_off_the_env_step_lattice_raises_naming_the_sub_step_band():
    with pytest.raises(ValueError, match="substep_delay_range_s"):
        _term(delay_range_s=(0.05, 0.1))
    with pytest.raises(ValueError, match="physics tick"):
        _term(delay_range_s=(0.0, 0.0), substep_delay_range_s=(0.0, 0.004))


def test_a_sub_step_part_of_a_whole_step_or_more_goes_into_the_buffer():
    tick = _cfg().sim.dt
    one, _, _ = _term(num_envs=4, decimation=1, delay_range_s=(STEP, STEP), substep_delay_range_s=(3 * tick, 3 * tick),
                      hold_fraction_range=(0.0, 0.0))
    assert one._action_delay_buffer.time_lags.tolist() == [4 + 3] * 4  # 4 ticks of delay + 3 carried
    four, asset, _ = _term(num_envs=4, delay_range_s=(STEP, STEP), substep_delay_range_s=(5 * tick, 5 * tick),
                           hold_fraction_range=(0.0, 0.0), speed_scale_range=None)
    assert four._action_delay_buffer.time_lags.tolist() == [1 + 1] * 4  # 1 step + the 5 ticks' whole step
    held = (four._substep_ticks % four._decimation).tolist()
    assert held == [1] * 4  # and 1 tick held at the start of each step


def test_the_imu_latency_band_draws_per_env_at_reset_after_the_bias():
    from strafer_lab.tasks.navigation.mdp.noise_models import IMUNoiseModelCfg

    base = _cfg().observations.policy.imu_linear_acceleration.noise
    fixed, band = base.copy(), base.copy()
    band.latency_steps_range = (0, 3)
    torch.manual_seed(21)
    a = fixed.class_type(fixed, 256, "cpu")
    torch.manual_seed(21)
    b = band.class_type(band, 256, "cpu")
    assert torch.equal(a._bias, b._bias)  # the band draws after the bias
    ids = torch.arange(0, 256, 2)
    torch.manual_seed(22)
    a.reset(ids)
    torch.manual_seed(22)
    b.reset(ids)
    assert torch.equal(a._bias, b._bias)
    delays = b._delay_buffer._delays
    assert int(delays.min()) >= 0 and int(delays.max()) <= 3 and len(set(delays.tolist())) == 4
    assert isinstance(IMUNoiseModelCfg().latency_steps_range, type(None))  # off by default


@pytest.mark.parametrize("tier", ["Real", "Robust"])
def test_the_contract_reaches_the_term_in_seconds(tier):
    from strafer_lab.tasks.navigation import sim_real_cfg as src

    timing = {"Real": src.REAL_ROBOT_CONTRACT, "Robust": src.ROBUST_TRAINING_CONTRACT}[tier].timing
    wv = _cfg(tier).actions.wheel_velocities
    lo, hi = timing.action_latency_range_s
    assert wv.delay_range_s == pytest.approx((timing.action_latency_s + lo, timing.action_latency_s + hi))
    assert wv.delay_redraw_on_reset == timing.action_latency_redraw_on_reset is False
    assert tuple(wv.substep_delay_range_s) == tuple(timing.action_substep_latency_range_s) == (0.0, 0.0)
    assert tuple(wv.hold_run_range_s) == tuple(timing.action_hold_run_range_s)
    imu = _cfg(tier).observations.policy.imu_angular_velocity.noise
    assert imu.latency_steps_range == timing.imu_latency_steps_range is None
