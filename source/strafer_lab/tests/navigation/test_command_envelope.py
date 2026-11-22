# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The command envelope: a per-env scale on both caps, applied by the action term's clamp.

The term is built against a stand-in env (an asset that names its four wheel joints), on the ideal tier so the
processed wheel velocities are the clamped body command through the kinematic matrix, with nothing after it. The body
command is recovered from them by the matrix's pseudo-inverse.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from strafer_shared.mecanum_kinematics import l1_clamp_twist, l1_clamp_twist_batched

N = 64


class _Asset:
    def find_joints(self, names, preserve_order=False):
        return [0, 1, 2, 3], ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]


def _term(speed_scale_range):
    from strafer_lab.tasks.navigation import strafer_env_cfg as tiers

    cfg = tiers.ActionsCfg_Ideal().wheel_velocities
    cfg.speed_scale_range = speed_scale_range
    env = SimpleNamespace(
        scene={cfg.asset_name: _Asset()}, device="cpu", num_envs=N, physics_dt=1 / 120, step_dt=1 / 30
    )
    return cfg.class_type(cfg, env)


def _body(term):
    """The clamped body command, recovered from the processed wheel velocities."""
    wheels = term.processed_actions / term._wheel_axis_signs
    return wheels @ torch.linalg.pinv(term._kinematic_matrix).T


def test_per_env_caps_clamp_each_row_as_the_scalar_form():
    rng = np.random.default_rng(0)
    v = torch.from_numpy(rng.uniform(-3, 3, (N, 3))).float()
    lin = torch.from_numpy(rng.uniform(0.2, 1.6, N)).float()
    ang = torch.from_numpy(rng.uniform(0.5, 4.2, N)).float()
    out = l1_clamp_twist_batched(v, vel_cap_linear_m_s=lin, vel_cap_angular_rad_s=ang)
    for i in range(N):
        want = l1_clamp_twist(*v[i].tolist(), vel_cap_linear_m_s=float(lin[i]), vel_cap_angular_rad_s=float(ang[i]))
        assert np.allclose(out[i].numpy(), np.asarray(want, dtype=np.float32), atol=1e-6), i


def test_scales_draw_in_range_and_reset_redraws_only_its_envs():
    term = _term((0.255, 1.0))
    s0 = term._speed_scale.clone()
    assert torch.all((s0 >= 0.255) & (s0 <= 1.0)) and s0.std() > 0.05
    ids = torch.tensor([3, 10, 41])
    term.reset(ids)
    moved = (term._speed_scale != s0).nonzero().flatten().tolist()
    assert set(moved) <= set(ids.tolist()) and moved, moved


def test_the_scale_is_uniform_on_its_range():
    """Over many resets the scales follow U[low, high]: their Kolmogorov-Smirnov distance to the uniform CDF stays
    under its critical value at alpha = 1e-6, the share under the deploy cap s = 0.5 is the uniform's within 6
    standard errors, and the extremes reach both ends. A log-uniform draw, or one from part of the range, fails."""
    lo, hi = 0.255, 1.0
    torch.manual_seed(20261118)
    term = _term((lo, hi))
    ids = torch.arange(N)
    draws = [term._speed_scale.clone()]
    for _ in range(300):
        term.reset(ids)
        draws.append(term._speed_scale.clone())
    s = torch.cat(draws).double().sort().values
    n = s.numel()
    cdf = (s - lo) / (hi - lo)
    i = torch.arange(1, n + 1, dtype=torch.float64)
    ks = torch.max(torch.maximum(i / n - cdf, cdf - (i - 1) / n)).item()
    assert ks < np.sqrt(-0.5 * np.log(1e-6 / 2)) / np.sqrt(n), ks
    p = (0.5 - lo) / (hi - lo)
    share = (s < 0.5).double().mean().item()
    assert abs(share - p) < 6 * np.sqrt(p * (1 - p) / n), share
    assert s[0].item() < lo + 0.01 and s[-1].item() > hi - 0.01, (s[0].item(), s[-1].item())


def test_the_clamp_applies_each_envs_scale():
    """Each cap alone: a full-speed command on one budget, so the per-wheel limit after the kinematics stays slack."""
    term = _term((0.255, 1.0))
    s = term._speed_scale
    term.process_actions(torch.tensor([[1.0, 0.0, 0.0]]).repeat(N, 1))
    assert torch.allclose(_body(term)[:, :2].abs().sum(-1), s * term._max_linear_vel, atol=1e-4)
    term.process_actions(torch.tensor([[0.0, 0.0, 1.0]]).repeat(N, 1))
    assert torch.allclose(_body(term)[:, 2].abs(), s * term._max_angular_vel, atol=1e-4)


def test_a_pinned_half_scale_is_the_deploy_cap():
    """s = 0.5 is the node's cap: 0.7841 m/s on |vx| + |vy| and 2.0933 rad/s on |omega|."""
    term = _term((0.5, 0.5))
    term.process_actions(torch.tensor([[1.0, 0.0, 0.0]]).repeat(N, 1))
    assert torch.allclose(_body(term)[:, :2].abs().sum(-1), torch.full((N,), 0.7841), atol=5e-4)
    term.process_actions(torch.tensor([[0.0, 0.0, 1.0]]).repeat(N, 1))
    assert torch.allclose(_body(term)[:, 2].abs(), torch.full((N,), 2.0933), atol=5e-4)


def test_the_envelope_tightens_only_the_cap():
    """Below the cap the command passes unscaled: at s = 0.5, a command of 0.3 of the full envelope on one budget runs
    at 0.3 of the full envelope, not 0.15 (which a scale on the command instead of the cap would give)."""
    term = _term((0.5, 0.5))
    term.process_actions(torch.tensor([[0.3, 0.0, 0.0]]).repeat(N, 1))
    assert torch.allclose(_body(term)[:, 0], torch.full((N,), 0.3 * term._max_linear_vel), atol=1e-4)
    term.process_actions(torch.tensor([[0.0, 0.0, 0.3]]).repeat(N, 1))
    assert torch.allclose(_body(term)[:, 2], torch.full((N,), 0.3 * term._max_angular_vel), atol=1e-4)


def test_the_envelope_off_draws_nothing():
    """Off, the term draws no scale at construction or at a reset: torch's global RNG state is unchanged, so a variant
    without the envelope keeps its random stream."""
    state = torch.random.get_rng_state()
    term = _term(None)
    term.reset(torch.tensor([0, 3, 7]))
    term.reset(None)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_off_and_unit_scale_leave_the_clamp_as_it_was():
    torch.manual_seed(0)
    a = torch.rand(N, 3) * 2 - 1
    off, unit = _term(None), _term((1.0, 1.0))
    off.process_actions(a.clone())
    unit.process_actions(a.clone())
    assert torch.allclose(off.processed_actions, unit.processed_actions, atol=1e-5)


@pytest.mark.parametrize("bad", [(0.0, 1.0), (0.6, 0.5), (0.5, 1.2)])
def test_an_impossible_range_is_refused(bad):
    with pytest.raises(ValueError):
        _term(bad)


def test_every_tier_builds_its_caps_from_the_shared_chassis_geometry():
    """The wheel radius, wheel base and track width are strafer_shared's; the wheel limit is the contract's 32.67 rad/s,
    so the full caps sit 7.9e-5 under MAX_LINEAR_VEL and MAX_ANGULAR_VEL, and the half-scale caps within 1e-4 of the
    node's NAV_LINEAR_VEL and NAV_ANGULAR_VEL (rounded to four decimals)."""
    from strafer_lab.tasks.navigation import strafer_env_cfg as tiers
    from strafer_shared.constants import (
        MAX_ANGULAR_VEL, MAX_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_LINEAR_VEL, TRACK_WIDTH, WHEEL_BASE, WHEEL_RADIUS,
    )

    for tier in (tiers.ActionsCfg_Ideal, tiers.ActionsCfg_Realistic, tiers.ActionsCfg_Robust):
        cfg = tier().wheel_velocities
        assert (cfg.wheel_radius, cfg.wheel_base, cfg.track_width) == (WHEEL_RADIUS, WHEEL_BASE, TRACK_WIDTH), tier
        env = SimpleNamespace(scene={cfg.asset_name: _Asset()}, device="cpu", num_envs=N, physics_dt=1 / 120,
                              step_dt=1 / 30)
        term = cfg.class_type(cfg, env)
        assert term._max_linear_vel == pytest.approx(MAX_LINEAR_VEL, rel=1e-4), tier
        assert term._max_angular_vel == pytest.approx(MAX_ANGULAR_VEL, rel=1e-4), tier
        assert 0.5 * term._max_linear_vel == pytest.approx(NAV_LINEAR_VEL, rel=1e-4), tier
        assert 0.5 * term._max_angular_vel == pytest.approx(NAV_ANGULAR_VEL, rel=1e-4), tier


def test_the_range_reaches_the_action_term():
    """The robust contract's range is passed into the action cfg and the built term
    draws from it (a contract field that get_action_config_params does not pass
    would be inert)."""
    from strafer_lab.tasks.navigation import sim_real_cfg
    from strafer_lab.tasks.navigation import strafer_env_cfg as tiers

    assert sim_real_cfg.ROBUST_TRAINING_CONTRACT.actuator.speed_scale_range == (0.255, 1.0)
    params = sim_real_cfg.get_action_config_params(sim_real_cfg.ROBUST_TRAINING_CONTRACT)
    assert params["speed_scale_range"] == (0.255, 1.0)
    cfg = tiers.ActionsCfg_Robust().wheel_velocities
    assert cfg.speed_scale_range == (0.255, 1.0)
    env = SimpleNamespace(
        scene={cfg.asset_name: _Asset()}, device="cpu", num_envs=N, physics_dt=1 / 120, step_dt=1 / 30
    )
    term = cfg.class_type(cfg, env)
    s = term._speed_scale
    assert torch.all((s >= 0.255) & (s <= 1.0)) and s.std() > 0.05
    for tier in (tiers.ActionsCfg_Ideal, tiers.ActionsCfg_Realistic):
        assert tier().wheel_velocities.speed_scale_range is None, tier.__name__


def test_exactly_the_robust_tier_carries_the_envelope():
    import dataclasses

    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    seen = {"robust": 0, "other": 0}
    for name in sorted(n for n in dir(composed) if n.startswith("StraferNavCfg_")):
        cls = getattr(composed, name)
        kind = [f.default_factory().kind for f in dataclasses.fields(cls) if f.name == "scene_source"]
        if kind == ["infinigen"]:
            continue  # reads the scene corpus at construction; composes the same action tiers
        cfg = cls()
        robust = cfg.realism.level == "robust"
        seen["robust" if robust else "other"] += 1
        rng = cfg.actions.wheel_velocities.speed_scale_range
        assert rng == ((0.255, 1.0) if robust else None), f"{name}: {rng}"
    assert seen["robust"] >= 6 and seen["other"] >= 6, seen
