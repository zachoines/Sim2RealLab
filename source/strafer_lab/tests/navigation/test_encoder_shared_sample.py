# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The policy's encoder and body-velocity terms carry one wheel-speed reading per control step.

On the robot ``joint_states`` and odom carry the same reported wheel speeds, so their noise is one chain. In training
the policy's ``wheel_encoder_velocities`` term keeps its ``EncoderNoiseModel``, which applies the reported speed's lag
and then the noise and keeps its output; the ``body_velocity_xy`` term's ``BodyVelocityNoiseModel``, built after it in
the same group, returns the mecanum kinematics of that output. The terms' functions and the critic stay clean, so the
policy tensor's layout is unchanged. The lag's three forms (a first-order filter, a pure delay, a moving average) take
their parameter in seconds; 0 is the identity.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import functools
import math

import pytest
import torch

N = 64
ENC, BODY = "wheel_encoder_velocities", "body_velocity_xy"


CLASSES = [f"StraferNavCfg_{obj}_{tier}{play}" for obj in ("RLDepthSubgoal", "RLNoCamSubgoal", "RLDepth")
           for tier in ("Real", "Robust") for play in ("", "_PLAY")]


@functools.lru_cache(maxsize=None)
def _composed():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    return {n: getattr(composed, n)() for n in CLASSES}


def _cfg(tier):
    return _composed()[f"StraferNavCfg_RLDepthSubgoal_{tier}"]


def _pair(tier="Real", num_envs=N, **overrides):
    """The tier's encoder model and the body-velocity model built after it, as the observation manager builds them."""
    policy = _cfg(tier).observations.policy
    enc_cfg = getattr(policy, ENC).noise.copy()
    for k, v in overrides.items():
        setattr(enc_cfg, k, v)
    body_cfg = getattr(policy, BODY).noise
    enc = enc_cfg.class_type(enc_cfg, num_envs, "cpu")
    body = body_cfg.class_type(body_cfg, num_envs, "cpu")
    return enc, body


def _fk(ticks):
    from strafer_lab.tasks.navigation.mdp import observations as o

    return o._get_inverse_kinematic_matrix_torch(ticks.device, ticks.dtype), o.ENCODER_TICKS_TO_RADIANS


@pytest.mark.parametrize("tier", ["Real", "Robust"])
def test_the_body_velocity_is_the_kinematics_of_the_same_encoder_reading(tier):
    from strafer_lab.tasks.navigation.mdp import observations as o

    enc, body = _pair(tier)
    gen = torch.Generator().manual_seed(1)
    for k in range(25):
        rad = 30.0 * torch.rand(N, 4, generator=gen) - 15.0
        ticks = rad * o.RADIANS_TO_ENCODER_TICKS
        noisy = enc(ticks)
        clean_body = (rad @ o._get_inverse_kinematic_matrix_torch(rad.device, rad.dtype).T)[:, :2]
        out = body(clean_body)
        inv, k_rad = _fk(noisy)
        assert torch.allclose(out, ((noisy * k_rad) @ inv.T)[:, :2], atol=1e-6), (tier, k)  # one chain, not two
        if k == 12:
            enc.reset(torch.tensor([3, 9]))
            body.reset(torch.tensor([3, 9]))


def test_a_body_velocity_model_needs_an_encoder_model_before_it():
    from strafer_lab.tasks.navigation.mdp import noise_models as nm

    nm._OFFERED_WHEEL_READING.clear()
    cfg = nm.BodyVelocityNoiseModelCfg()
    with pytest.raises(RuntimeError):
        cfg.class_type(cfg, N, "cpu")
    _pair()  # an encoder model then a body model: builds
    with pytest.raises(RuntimeError):  # the offer is taken once
        cfg.class_type(cfg, N, "cpu")


def test_the_noise_is_applied_once_at_the_tiers_sigma():
    torch.manual_seed(2)
    for tier, sigma in (("Real", 60.0), ("Robust", 150.0)):
        enc, _ = _pair(tier, num_envs=512, latency_steps=0, stuck_probability=0.0)
        clean = torch.full((512, 4), 800.0)
        residuals = torch.cat([enc(clean) - clean for _ in range(20)])
        assert float(residuals.std()) == pytest.approx(math.sqrt(sigma ** 2 + 1.0 / 12.0), rel=0.03), tier


def test_the_composed_groups_pair_the_two_models_and_leave_the_critic_clean():
    from strafer_lab.tasks.navigation.mdp import noise_models as nm

    for name, cfg in _composed().items():
        policy = getattr(cfg.observations, "policy", None)
        if policy is None or not hasattr(policy, BODY):
            continue
        enc, body = getattr(policy, ENC).noise, getattr(policy, BODY).noise
        assert (enc is None) == (body is None), name  # paired, or both off
        if body is not None:
            assert isinstance(enc, nm.EncoderNoiseModelCfg) and isinstance(body, nm.BodyVelocityNoiseModelCfg), name
            terms = list(vars(policy))
            assert terms.index(ENC) < terms.index(BODY), name
        critic = getattr(cfg.observations, "critic", None)
        for term in (ENC, BODY):
            if critic is not None and hasattr(critic, term):
                assert getattr(critic, term).noise is None, (name, term)


def _lagged(form, lag_s, steps=40, hz=30.0):
    from strafer_lab.tasks.navigation.mdp.noise_models import ReportedSpeedLag

    lag = ReportedSpeedLag(form, lag_s, 2, 4, 1.0 / hz, "cpu")
    out = [float(lag(torch.zeros(2, 4))[0, 0])]  # seeds at rest
    for _ in range(steps):
        out.append(float(lag(torch.ones(2, 4))[0, 0]))
    return out


def test_the_first_order_form_rises_63_percent_at_its_parameter():
    dt, tau = 1.0 / 30.0, 0.12
    y = _lagged("first_order", tau)
    for n in range(1, 30):
        assert y[n] == pytest.approx(1.0 - math.exp(-n * dt / tau), abs=1e-6)


def test_the_delay_form_shifts_the_step_by_its_parameter():
    y = _lagged("delay", 0.1)  # 3 steps at 30 Hz
    assert y[1:4] == [0.0, 0.0, 0.0] and y[4] == 1.0


def test_the_window_form_ramps_over_its_span():
    y = _lagged("window", 0.1)  # a 3-step average
    assert y[1] == pytest.approx(1 / 3) and y[2] == pytest.approx(2 / 3) and y[3] == 1.0


def test_the_lag_means_the_same_at_another_rate():
    y30, y120 = _lagged("delay", 0.1, hz=30.0), _lagged("delay", 0.1, steps=160, hz=120.0)
    # y[0] is the seed at rest and the step enters at y[1]: the delay is the index of the first 1 less one
    assert (y30.index(1.0) - 1) * (1 / 30) == pytest.approx(0.1)
    assert (y120.index(1.0) - 1) * (1 / 120) == pytest.approx(0.1)  # the same 0.1 s at four times the rate


@pytest.mark.parametrize("form", ["first_order", "delay", "window"])
def test_every_form_at_zero_is_the_identity(form):
    from strafer_lab.tasks.navigation.mdp.noise_models import ReportedSpeedLag

    lag = ReportedSpeedLag(form, 0.0, 3, 4, 1.0 / 30.0, "cpu")
    x = torch.randn(3, 4)
    assert lag(x) is x


def test_a_reset_seeds_the_lag_from_the_wheels_speed():
    enc, _ = _pair(velocity_noise_std=0.0, enable_quantization=False, latency_steps=0, stuck_probability=0.0,
                   reported_speed_lag_s=0.5)
    first = enc(torch.full((N, 4), 900.0))
    assert torch.allclose(first, torch.full((N, 4), 900.0))  # seeded at the first reading, not rising from zero
    enc.reset(torch.tensor([7]))
    after = enc(torch.zeros(N, 4))
    assert torch.allclose(after[7], torch.zeros(4))  # env 7 starts its new episode at its own speed
    assert bool((after[0] > 0).all())  # the others lag toward zero


@pytest.mark.parametrize("tier", ["Real", "Robust"])
def test_the_lag_is_off_on_the_composed_tiers(tier):
    cfg = getattr(_cfg(tier).observations.policy, ENC).noise
    assert cfg.reported_speed_lag_s == 0.0 and cfg.reported_speed_form == "first_order"
    assert cfg.control_frequency_hz == pytest.approx(1.0 / (_cfg(tier).sim.dt * _cfg(tier).decimation))
