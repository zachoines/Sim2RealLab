# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The reported speed's lag in the encoder noise model: three forms, a parameter in seconds, off at 0.

On the robot the wheel speed in ``joint_states`` and odom is the motor controller's reported speed, which trails the
wheel. The encoder noise model applies that lag before its noise, in one of three forms (a first-order filter, a pure
delay, a moving average); the parameter is in seconds, converted at the control step, so it means the same at any
decimation; 0, the default, is the identity.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import functools
import math

import pytest
import torch

N = 64
ENC = "wheel_encoder_velocities"


@functools.lru_cache(maxsize=None)
def _cfg(tier):
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    return getattr(composed, f"StraferNavCfg_RLDepthSubgoal_{tier}")()


def _encoder(tier="Real", num_envs=N, **overrides):
    """The tier's policy encoder noise model, with overrides."""
    enc_cfg = getattr(_cfg(tier).observations.policy, ENC).noise.copy()
    for k, v in overrides.items():
        setattr(enc_cfg, k, v)
    return enc_cfg.class_type(enc_cfg, num_envs, "cpu")


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
    enc = _encoder(velocity_noise_std=0.0, enable_quantization=False, latency_steps=0, stuck_probability=0.0,
                   missed_tick_prob=0.0, extra_tick_prob=0.0, reported_speed_lag_s=0.5)
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
