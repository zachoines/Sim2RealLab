# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The D555's stereo validity at render resolution, against the deploy node's reduction.

The sensor returns no depth in the left band (``u < f·B / z(u) + m``), in the occlusion shadows left of nearer surfaces,
below its minimum depth, and where nothing returns. Its invalid pixels are written as the far clamp before the 8×8
block median, as the deploy node writes Z16's zeros, so the perceived depth term followed by the shared reduction is
``obs_pipeline.downsample_depth(depth, valid_mask=valid)`` bit for bit. The critic reads the clean render.

The deployment half is ``strafer_inference.obs_pipeline``, reached by path as ``test_depth_deploy_reduction`` does.
No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from strafer_lab.tasks.navigation.mdp.d555_sensor import (
    D555SensorCfg,
    D555SensorProcess,
    sensor_frame,
    stereo_validity,
)
from strafer_lab.tasks.navigation.mdp.observations import depth_image
from strafer_shared.constants import DEPTH_MAX, PERCEPTION_HEIGHT, PERCEPTION_WIDTH

H, W = PERCEPTION_HEIGHT, PERCEPTION_WIDTH
FB = D555SensorCfg().fb_px_m


def _load_obs_pipeline():
    path = (Path(__file__).resolve().parents[3] / "strafer_ros" / "strafer_inference" / "strafer_inference"
            / "obs_pipeline.py")
    if not path.exists():
        pytest.skip(f"deploy pipeline not present at {path}")
    spec = importlib.util.spec_from_file_location("_obs_pipeline_stereo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


obs_pipeline = _load_obs_pipeline()


def _env(raw: torch.Tensor, process=None):
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(output={"distance_to_image_plane": raw}))
    env = types.SimpleNamespace(num_envs=raw.shape[0], device="cpu", scene=types.SimpleNamespace(sensors={"cam": sensor}))
    if process is not None:
        env._d555_sensor = process
    return env


def _term(env, perceived=True):
    from isaaclab.managers import SceneEntityCfg

    return depth_image(env, SceneEntityCfg("cam"), max_depth=DEPTH_MAX, perceived=perceived)


def _scene(seed: int, n: int = 1) -> torch.Tensor:
    """Walls, boxes nearer than them, a near patch and culled (inf) pixels, at 640x360."""
    g = torch.Generator().manual_seed(seed)
    z = torch.empty(n, H, W)
    for e in range(n):
        wall = 1.0 + 4.0 * float(torch.rand(1, generator=g))
        z[e] = wall + 0.4 * torch.linspace(0, 1, W)[None, :] * float(torch.rand(1, generator=g))
        for _ in range(4):
            u0 = int(torch.randint(0, W - 40, (1,), generator=g))
            v0 = int(torch.randint(0, H - 40, (1,), generator=g))
            z[e, v0:v0 + 60, u0:u0 + 80] = 0.3 + 1.5 * float(torch.rand(1, generator=g))
        z[e, 300:, 100:180] = 0.2  # below the sensor's minimum depth
        z[e, :20, 500:] = float("inf")
    return z


def _rescued(z):
    return torch.where(torch.isfinite(z), z, torch.full_like(z, DEPTH_MAX))


@pytest.mark.parametrize("seed", range(6))
def test_the_perceived_term_is_the_nodes_reduction_of_the_sensor_frame(seed):
    raw = _scene(seed)
    process = D555SensorProcess(1, "cpu", D555SensorCfg())
    got = _term(_env(raw[..., None].clone(), process))
    _, valid = sensor_frame(_rescued(raw), process)
    want = obs_pipeline.downsample_depth(raw[0].numpy(), valid_mask=valid[0].numpy())
    assert np.array_equal(got[0].numpy(), want)
    assert not bool(valid[0, :, :8].any())  # the band is there


def test_a_wall_at_2_m_gives_a_band_of_fb_over_2_plus_the_margin_and_no_core():
    valid = stereo_validity(torch.full((1, H, W), 2.0), FB, D555SensorCfg())
    width = math.ceil(FB / 2.0 + 6.5)
    assert width == 22
    assert not bool(valid[0, :, :width].any()) and bool(valid[0, :, width:].all())


def test_a_box_at_1_m_on_a_4_m_wall_shadows_its_left_and_not_its_right():
    z = torch.full((1, H, W), 4.0)
    z[0, :, 300:400] = 1.0
    valid = stereo_validity(z, FB, D555SensorCfg())
    invalid = ~valid[0, 0]
    shadow = [u for u in range(40, 300) if invalid[u]]
    assert shadow == list(range(278, 300))  # 22 px: fB (1/1 - 1/4) = 22.9 px, less the 0.5 px tolerance
    assert FB * (1.0 - 0.25) == pytest.approx(22.9, abs=0.05)
    assert not bool(invalid[300:].any())  # the box itself and the wall right of it


@pytest.mark.parametrize("wall,cell", [(0.20, DEPTH_MAX), (0.30, 0.2)])
def test_a_square_on_wall_below_the_minimum_reads_far_and_above_it_the_near_fill(wall, cell):
    process = D555SensorProcess(1, "cpu", D555SensorCfg())
    out = _term(_env(torch.full((1, H, W, 1), wall), process)).reshape(45, 80)
    assert float(out[20, 40]) == pytest.approx(cell)
    assert float(out[20, 0]) == DEPTH_MAX  # the band at 0.30 m: fB / 0.3 + 6.5 = 108 px, 13 cells


def test_an_all_culled_render_gives_the_band_only():
    raw = torch.full((1, H, W, 1), float("inf"))
    process = D555SensorProcess(1, "cpu", D555SensorCfg())
    assert bool((_term(_env(raw, process)) == DEPTH_MAX).all())
    _, valid = sensor_frame(_rescued(raw), process)
    band = math.ceil(FB / DEPTH_MAX + 6.5)
    assert band == 12
    assert not bool(valid[0, :, :band].any()) and bool(valid[0, :, band:].all())
    _, raw_valid = sensor_frame(raw, process)  # called on the cull itself: nothing returns
    assert not bool(raw_valid.any())


def test_the_process_draws_from_its_own_generator():
    cfg = D555SensorCfg(fb_scale_range=(0.98, 1.06))
    torch.manual_seed(11)
    before = torch.get_rng_state()
    a = D555SensorProcess(64, "cpu", cfg, seed=7)
    a.reset(None)
    assert torch.equal(torch.get_rng_state(), before)  # no other term's stream moves
    b = D555SensorProcess(64, "cpu", cfg, seed=7)
    b.reset(None)
    assert torch.equal(a.fb, b.fb)
    assert float(a.fb.min()) >= 0.98 * FB - 1e-4 and float(a.fb.max()) <= 1.06 * FB + 1e-4
    assert float(a.fb.std()) > 0.01 * FB
    first = a.fb.clone()
    a.reset(torch.tensor([3, 9]))
    changed = (a.fb != first).nonzero().flatten().tolist()
    assert set(changed) <= {3, 9} and changed


def test_the_event_creates_the_process_once_seeded_from_the_env():
    from strafer_lab.tasks.navigation.mdp import events

    params = dict(fb_px_m=FB, fb_scale_range=(0.98, 1.06), margin_px=6.5, core_px=0.0, min_depth_m=0.244,
                  occlusion_tol_px=0.5, invalid_depth_m=DEPTH_MAX, max_return_m=50.0)
    envs = [types.SimpleNamespace(num_envs=8, device="cpu", cfg=types.SimpleNamespace(seed=5)) for _ in range(2)]
    torch.manual_seed(1)
    events.randomize_d555_sensor(envs[0], torch.arange(8), **params)
    torch.manual_seed(2)
    events.randomize_d555_sensor(envs[1], torch.arange(8), **params)
    assert torch.equal(envs[0]._d555_sensor.fb, envs[1]._d555_sensor.fb)  # the env's seed, not the global stream
    process = envs[0]._d555_sensor
    events.randomize_d555_sensor(envs[0], torch.tensor([1]), **params)
    assert envs[0]._d555_sensor is process


def test_off_the_term_draws_nothing_and_with_a_process_the_critic_is_untouched():
    raw = _scene(3, n=2)[..., None]
    torch.manual_seed(4)
    before = torch.get_rng_state()
    off = _term(_env(raw.clone()))
    assert torch.equal(torch.get_rng_state(), before)
    process = D555SensorProcess(2, "cpu", D555SensorCfg())
    assert torch.equal(_term(_env(raw.clone(), process), perceived=False), off)
    assert not torch.equal(_term(_env(raw.clone(), process)), off)


def test_a_render_off_the_deploy_resolution_raises_with_the_sensor_on():
    process = D555SensorProcess(1, "cpu", D555SensorCfg())
    with pytest.raises(ValueError, match="stereo validity"):
        _term(_env(torch.full((1, 45, 80, 1), 2.0), process))


def test_the_contracts_fb_is_the_policy_cameras_fx_times_the_baseline_and_off_is_none():
    from strafer_lab.tasks.navigation import sim_real_cfg as src
    from strafer_lab.tasks.navigation.d555_cfg import make_d555_camera_cfg

    cam = make_d555_camera_cfg(data_types=("distance_to_image_plane",))
    fx = cam.width * cam.spawn.focal_length / cam.spawn.horizontal_aperture
    for name in ("REAL_ROBOT_CONTRACT", "ROBUST_TRAINING_CONTRACT"):
        contract = copy.deepcopy(getattr(src, name))
        contract.sensors.depth_camera.stereo_mask_enable = True
        contract.sensors.depth_camera.hole_probability = 0.0
        params = src.get_d555_sensor_params(contract)
        assert params["fb_px_m"] == pytest.approx(fx * contract.sensors.depth_camera.baseline_m, rel=1e-9)
        assert params["fb_px_m"] == pytest.approx(30.545, abs=5e-4)
        assert params["invalid_depth_m"] == contract.sensors.depth_camera.max_range_m
        noise = src.get_depth_noise(contract)
        assert noise.noise_free_at_max_range and noise.hole_probability == 0.0
        contract.sensors.depth_camera.hole_probability = 0.01
        with pytest.raises(ValueError, match="replaces the cell holes"):
            src.get_depth_noise(contract)


def test_no_noise_at_the_clamp_consumes_the_same_draws():
    from strafer_lab.tasks.navigation.mdp.noise_models import DepthNoiseModelCfg

    data = torch.rand(4, 3600) * 5.0 + 0.5
    data[:, :300] = DEPTH_MAX
    outs, states = [], []
    for flag in (False, True):
        cfg = DepthNoiseModelCfg(hole_probability=0.0, frame_drop_prob=0.0, noise_free_at_max_range=flag)
        model = cfg.class_type(cfg, 4, "cpu")
        torch.manual_seed(6)
        outs.append(model(data.clone()))
        states.append(torch.get_rng_state())
    assert torch.equal(states[0], states[1])
    assert bool((outs[1][:, :300] == DEPTH_MAX).all()) and not bool((outs[0][:, :300] == DEPTH_MAX).all())
    assert torch.equal(outs[0][:, 300:], outs[1][:, 300:])


def test_built_and_off_on_every_composition():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    for name in ("StraferNavCfg_RLDepthSubgoal_Real", "StraferNavCfg_RLDepthSubgoal_Robust",
                 "StraferNavCfg_RLNoCamSubgoal_Robust", "StraferNavCfg_RLDepth_Real"):
        cfg = getattr(composed, name)()
        assert cfg.events.randomize_d555_sensor is None, name
        if hasattr(cfg.observations.policy, "depth_image"):
            assert "perceived" not in cfg.observations.policy.depth_image.params, name  # the layout is unchanged
            assert cfg.observations.critic.depth_image.params["perceived"] is False, name
