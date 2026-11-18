# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Per-source sensor outages and the node's stale-source stop, built and off by default.

During an outage a source repeats its previous reading; once it has been stale for its bound (the node's timeout in
control steps: 6 for the IMU's and the driver's 0.2 s, 15 for the depth's 0.5 s at 30 Hz) the env is stopped: the
commanded twist is zero and the policy's last_action holds its value from before the stop, until the source returns.
The IMU and the depth stall together on the shared USB draw; the encoders and the body velocity are one driver reading.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch

N = 8
STEP = 1.0 / 30.0
TIMEOUTS = {"imu": 0.2, "driver": 0.2, "depth": 0.5}


@pytest.fixture(autouse=True)
def _clear_outages():
    from strafer_lab.tasks.navigation.mdp import sensor_outage

    sensor_outage._ACTIVE.clear()
    yield
    sensor_outage._ACTIVE.clear()


def _process(rates=None, usb=0.0, lengths=(0.5, 0.5), n=N):
    from strafer_lab.tasks.navigation.mdp.sensor_outage import OutageProcess

    return OutageProcess(n, "cpu", STEP, rates or {}, usb, lengths, TIMEOUTS)


def test_the_bounds_are_the_nodes_timeouts_in_control_steps():
    p = _process()
    assert p.bound_steps == {"imu": 6, "driver": 6, "depth": 15}


def test_the_stop_begins_on_the_bounds_step_and_lasts_while_a_source_is_past_it():
    p = _process(rates={"driver": 30.0}, lengths=(10 * STEP, 10 * STEP), n=1)  # starts on the first step, 10 long
    stopped, stale = [], []
    for _ in range(14):
        p.advance()
        stale.append(bool(p.stale("driver")[0]))
        stopped.append(bool(p.stopped[0]))
        p._p_start["driver"] = 0.0  # one outage only
    assert stale[:10] == [True] * 10 and stale[10] is False
    assert stopped[:5] == [False] * 5  # five repeated steps before the stop
    assert stopped[5:10] == [True] * 5 and stopped[10:] == [False] * 4  # the 6th stale step stops; the return resumes


def test_the_imu_and_the_depth_stall_together_on_the_shared_draw_and_the_driver_alone():
    torch.manual_seed(4)
    p = _process(usb=30.0, n=256)  # every idle env starts a shared stall
    p.advance()
    assert torch.equal(p.stale("imu"), p.stale("depth")) and bool(p.stale("imu").all())
    assert not bool(p.stale("driver").any())
    q = _process(rates={"driver": 30.0}, n=256)
    q.advance()
    assert bool(q.stale("driver").all()) and not bool(q.stale("imu").any()) and not bool(q.stale("depth").any())


def test_a_reset_ends_the_envs_outage_and_stop():
    p = _process(rates={"imu": 30.0}, lengths=(1.0, 1.0))
    for _ in range(8):
        p.advance()
    assert bool(p.stopped.all())
    p.reset(torch.tensor([1, 5]))
    assert not bool(p.stopped[[1, 5]].any()) and bool(p.stopped[[0, 2]].all())
    assert int(p.age("imu")[1]) == 0


def _tier(tier="Real"):
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    return getattr(composed, f"StraferNavCfg_RLDepthSubgoal_{tier}")()


@pytest.mark.parametrize("name,source", [("imu_linear_acceleration", "imu"), ("wheel_encoder_velocities", "driver"),
                                         ("depth_image", "depth")])
def test_a_stale_source_repeats_its_previous_reading_bit_for_bit(name, source):
    cfg = getattr(_tier().observations.policy, name).noise
    model = cfg.class_type(cfg, N, "cpu")
    width = {"imu": 3, "driver": 4, "depth": 3600}[source]
    gen = torch.Generator().manual_seed(1)

    def reading():
        x = torch.rand(N, width, generator=gen)
        return x * 5.8 + 0.2 if source == "depth" else x * 100.0

    p = _process(rates={source: 0.0}, lengths=(4 * STEP, 4 * STEP))  # built at the first reset, before any reading
    first = model(reading()).clone()
    p._p_start[source] = 1.0
    p.advance()  # every env's source goes stale
    second = model(reading())
    assert torch.equal(second, first)  # the previous reading, bit for bit
    for _ in range(4):
        p._p_start[source] = 0.0
        p.advance()
    third = model(reading())
    assert not torch.equal(third, first)  # the source has returned


class _Asset:
    def find_joints(self, names, preserve_order=False):
        return [0, 1, 2, 3], ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]

    def set_joint_velocity_target(self, *a, **k):
        pass


def test_while_stopped_the_commanded_twist_is_zero_and_last_action_holds():
    import math

    from strafer_lab.tasks.navigation.mdp import events
    from strafer_lab.tasks.navigation.mdp.noise_models import HeldLastActionNoiseModelCfg

    cfg = _tier()
    t = cfg.actions.wheel_velocities.copy()
    t.delay_range_s, t.hold_fraction_range, t.speed_scale_range = (0.0, 0.0), (0.0, 0.0), None
    t.enable_motor_dynamics, t.max_acceleration_rad_s2, t.max_acceleration_range = False, math.inf, (math.inf, math.inf)
    env = SimpleNamespace(scene={t.asset_name: _Asset()}, device="cpu", num_envs=2, physics_dt=cfg.sim.dt, step_dt=STEP)
    term = t.class_type(t, env)
    events.randomize_sensor_outages(env, torch.arange(2), start_rate_hz={"imu": 0.0, "driver": 0.0, "depth": 0.0},
                                    usb_stall_rate_hz=0.0, length_s_range=(1.0, 1.0), timeout_s=TIMEOUTS)
    hold = HeldLastActionNoiseModelCfg()
    held = hold.class_type(hold, 2, "cpu")
    a = torch.tensor([[0.5, 0.2, 0.1], [0.3, -0.2, 0.0]])
    term.process_actions(a)
    assert held(a.clone()).tolist() == a.tolist()
    moving = term.processed_actions.clone()
    assert float(moving.abs().max()) > 0
    process = env._sensor_outages
    process._remaining["imu"][0] = 30  # env 0's IMU goes stale
    b = torch.tensor([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])
    for k in range(8):
        term.process_actions(b)
        out = held(b.clone())
        if k >= 5:  # stopped from the 6th stale step
            assert torch.equal(term.processed_actions[0], torch.zeros(4)), k
            assert out[0].tolist() == pytest.approx(b[0].tolist() if k == 0 else out[0].tolist())
            assert out[0].tolist() == pytest.approx([0.9, 0.9, 0.9])  # last_action from before the stop
        assert float(term.processed_actions[1].abs().max()) > 0  # env 1 drives on
    assert term._raw_actions.tolist() == b.tolist()  # the policy's own action is kept


def test_the_event_draws_after_the_classs_other_draws_and_off_installs_nothing():
    for tier in ("Real", "Robust"):
        cfg = _tier(tier)
        names = list(vars(cfg.events))
        at = names.index("randomize_sensor_outages")
        assert all(n.startswith("jitter_") for n in names[at + 1:]), (tier, names[at:])  # only draw-free terms after
        assert cfg.events.randomize_sensor_outages is None, tier
        assert cfg.observations.policy.last_action.noise is None, tier


def test_off_no_process_is_registered():
    from strafer_lab.tasks.navigation.mdp import sensor_outage

    _tier()
    assert sensor_outage.current_outage(N, "cpu") is None
