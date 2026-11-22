# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The D555's depth noise in the stereo validity's function: the temporal and static fields, the lattice, the blobs.

On the clean render, in disparity of the 640-wide stream: a temporal field T fresh each step, a static field S held for
the episode, the disparity rounded to the 1/32 px lattice through the measured sub-pixel occupancy and the depth to
the millimetre, and invalid blobs on the floor at fixed pixels, with no memory between steps. Off, the function is the stereo
validity alone. The noise draws from its own generator, so neither the global stream nor the ``f·B`` draws move.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import types

import pytest
import torch

from strafer_lab.tasks.navigation.mdp.d555_noise import (
    LATTICE_OCCUPANCY_LUT,
    D555NoiseCfg,
    D555NoiseProcess,
    edge_distance,
    surface_geometry,
)
from strafer_lab.tasks.navigation.mdp.d555_sensor import D555SensorCfg, D555SensorProcess, sensor_frame
from strafer_shared.constants import D555_MEASURED_FX_PX, PERCEPTION_HEIGHT, PERCEPTION_WIDTH, POLICY_PERIOD_S

H, W = PERCEPTION_HEIGHT, PERCEPTION_WIDTH
FB = D555SensorCfg().fb_px_m
LENS_H = 0.279


def _wall(z=2.0, n=1):
    return torch.full((n, H, W), z)


def _floor(n=1, wall=5.0):
    rows = torch.arange(H, dtype=torch.float32)[:, None]
    floor = torch.where(rows > 185, LENS_H * D555_MEASURED_FX_PX / (rows - 180.0).clamp(min=1.0),
                        torch.full_like(rows, 1e9))
    return torch.minimum(torch.full((H, W), wall), floor.expand(H, W)).repeat(n, 1, 1)


def _process(n=1, seed=0, **noise):
    p = D555SensorProcess(n, "cpu", D555SensorCfg(), seed=seed, noise=D555NoiseCfg(**noise))
    p.reset(None)
    return p


def test_the_geometry_reads_the_rays_incidence_on_a_wall_and_the_floors_grazing_cosine():
    _, cos_wall, grad_wall, grad_v_wall = surface_geometry(_wall(), torch.tensor([FB]), D555NoiseCfg())
    u = (torch.arange(W) - 320.0) / D555_MEASURED_FX_PX
    v = (torch.arange(H) - 180.0) / D555_MEASURED_FX_PX
    ray = 1.0 / torch.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)  # a fronto-parallel wall: the ray's own angle
    assert float((cos_wall[0, 5:-5, 5:-5] - ray[5:-5, 5:-5]).abs().max()) < 1e-4
    assert float(cos_wall[0, 180, 320]) == pytest.approx(1.0, abs=1e-6) and float(grad_wall[0, 5:-5, 5:-5].max()) < 1e-4
    assert float(grad_v_wall[0, 5:-5, 5:-5].abs().max()) < 1e-4
    z = _floor()
    _, cos_floor, grad_floor, grad_v_floor = surface_geometry(z, torch.tensor([FB]), D555NoiseCfg())
    for v in (220, 260, 320):
        zf = float(z[0, v, 320])
        assert float(cos_floor[0, v, 320]) == pytest.approx(LENS_H / math.sqrt(LENS_H ** 2 + zf ** 2), rel=0.02)
        # B / h per row
        assert float(grad_floor[0, v, 320]) == pytest.approx(FB / (D555_MEASURED_FX_PX * LENS_H), rel=0.02)
        assert float(grad_v_floor[0, v, 320]) == pytest.approx(float(grad_floor[0, v, 320]), rel=1e-3)  # all of it down


def test_off_the_function_is_the_stereo_validity_and_on_neither_the_global_stream_nor_fb_moves():
    torch.manual_seed(3)
    before = torch.get_rng_state()
    sensor = D555SensorCfg(fb_scale_range=(0.98, 1.06))
    plain = D555SensorProcess(4, "cpu", sensor, seed=9)
    noisy = D555SensorProcess(4, "cpu", sensor, seed=9, noise=D555NoiseCfg())
    for p in (plain, noisy):
        p.reset(None)
    z = _floor(4)[..., None]
    sensor_frame(z, noisy)
    for p in (plain, noisy):
        p.reset(torch.tensor([1, 3]))
    assert torch.equal(plain.fb, noisy.fb)  # the f·B draws are the same with the noise on
    assert torch.equal(torch.get_rng_state(), before)


def _disparity(frame):
    return FB / frame


def test_the_static_field_is_held_within_an_episode_and_redrawn_at_a_reset():
    p = _process(t_sigma0_px=0.0, hole_p_base=0.0, hole_p_floor=0.0, hole_p_ref=0.0)  # S and the lattice only
    z = _wall(2.0)[..., None]
    a, va = sensor_frame(z, p)
    b, vb = sensor_frame(z, p)
    both = (va & vb)[..., 0]
    assert torch.equal(a[..., 0][both], b[..., 0][both])
    p.reset(None)
    c, _ = sensor_frame(z, p)
    assert not torch.equal(a[..., 0][both], c[..., 0][both])


def test_the_temporal_field_is_fresh_each_step_and_white_in_time():
    p = _process(s_sigma0_px=0.0, s_slope_gain=0.0, s_phase_bias_px=0.0, hole_p_base=0.0, hole_p_floor=0.0,
                 hole_p_ref=0.0, lattice_lut=(), round_to_mm=False)
    z = _wall(2.0)[..., None]
    frames = torch.stack([_disparity(sensor_frame(z, p)[0][0, 40:320, 60:600, 0]) for _ in range(12)])
    resid = frames - frames.mean(0)
    lag1 = (resid[1:] * resid[:-1]).mean() / resid.var()
    assert abs(float(lag1)) < 0.12  # 12 frames: the mean removes 1/12 of each, so white reads about -1/11
    assert float(resid.std()) > 0.02  # about T's 0.057 px at normal incidence


def test_the_disparity_sits_on_the_lattice_and_the_depth_on_whole_millimetres():
    p = _process(hole_p_base=0.0, hole_p_floor=0.0, hole_p_ref=0.0, round_to_mm=False)
    frame, valid = sensor_frame(_floor()[..., None], p)
    d = _disparity(frame[0, ..., 0][valid[0, ..., 0]].double())
    codes = torch.round((d - torch.floor(d + 1e-9)) * 32.0).long() % 32
    assert set(codes.unique().tolist()) <= set(LATTICE_OCCUPANCY_LUT)
    assert not ({0, 3, 6, 25, 28} & set(codes.unique().tolist()))  # the codes the unit never emits
    q = _process(hole_p_base=0.0, hole_p_floor=0.0, hole_p_ref=0.0)
    mm, v = sensor_frame(_floor()[..., None], q)
    zmm = mm[0, ..., 0][v[0, ..., 0] & (mm[0, ..., 0] < 6.0)].double() * 1000.0
    assert float((zmm - torch.round(zmm)).abs().max()) < 1e-3


def test_the_lattice_rounds_without_bias():
    p = D555NoiseProcess(1, "cpu", D555NoiseCfg(), step_dt=POLICY_PERIOD_S, seed=0, shape=(4, 4))
    d = 1.0 + 63.0 * (torch.arange(2 ** 18, dtype=torch.float64) + 0.5) / 2 ** 18
    err = p._quantise(d.float()).double() - d
    assert abs(float(err.mean())) < 1.0 / 512  # bins at or below their codes would read -1/64 px
    assert float(err.abs().max()) < 3.0 / 32  # the occupancy moves a disparity by at most two codes and a half


def test_the_blobs_take_the_floors_share_none_on_a_wall_and_recur_at_fixed_pixels_with_no_memory():
    p = _process(t_sigma0_px=0.0, s_sigma0_px=0.0, s_slope_gain=0.0, s_phase_bias_px=0.0)
    z = _floor(wall=5.0)[..., None]  # the floor, and above it a fronto-parallel wall at 5 m
    masks = []
    for _ in range(40):
        _, v = sensor_frame(z, p)
        masks.append(~v[0, ..., 0])
    masks = torch.stack(masks).float()
    floor = masks[:, 230:360, 120:600]  # floor rows at 0.5-1.8 m, columns clear of the left band
    share = float(floor.mean())
    assert share == pytest.approx(D555NoiseCfg().hole_p_floor, rel=0.35)  # one episode's static map: a wide band
    wall = masks[:, 40:150, 120:600]  # the wall: no vertical disparity gradient, no blobs
    assert float(wall.mean()) == 0.0
    # a pixel's rate is set per episode: invalid again one step later as often as twenty steps later, both well above
    # the share, which independent draws would give
    nxt = [float((floor[k:] * floor[:-k]).sum() / floor[:-k].sum()) for k in (1, 20)]
    assert nxt[0] > 5 * share and nxt[0] == pytest.approx(nxt[1], abs=0.05)


def _box_on_a_wall(n=1, box=1.5, wall=5.0):
    z = torch.full((n, H, W), wall)
    z[:, 100:250, 250:450] = box
    return z


def test_the_edge_distance_is_the_chessboard_distance_to_a_disparity_jump():
    z = torch.full((1, H, W), 4.0)
    z[:, :, 300:] = 1.0  # a jump of f·B (1/1 - 1/4) px between columns 299 and 300
    dist = edge_distance(FB / z, D555NoiseCfg().hole_edge_jump_px)
    row = dist[0, 180]
    assert row[299] == 0 and row[300] == 0
    assert [int(row[299 - k]) for k in (1, 5, 16)] == [1, 5, 16] and int(row[299 - 17]) == 17  # beyond: 17
    assert [int(row[300 + k]) for k in (1, 5, 16)] == [1, 5, 16]
    assert float(edge_distance(FB / torch.full((1, H, W), 2.0), 2.0).min()) == 17  # no edge on a flat wall


def test_the_rim_puts_its_share_on_a_depth_edge_and_none_beyond_its_reach():
    c = D555NoiseCfg()
    p = _process(n=4, t_sigma0_px=0.0, s_sigma0_px=0.0, s_slope_gain=0.0, s_phase_bias_px=0.0)
    z = _box_on_a_wall(4)[..., None]
    masks = torch.stack([~sensor_frame(z, p)[1][..., 0] for _ in range(20)]).float()
    dist = edge_distance(FB / _box_on_a_wall(1), c.hole_edge_jump_px)[0]
    clean = torch.zeros(H, W, dtype=torch.bool)
    clean[120:230, 430:475] = True  # about the box's right edge: clear of its shadow, 20 px from its top and bottom
    on_edge = clean & (dist == 0)
    assert float(masks[:, :, on_edge].mean()) == pytest.approx(c.hole_rim_p0, rel=0.3)
    assert float(masks[:, :, clean & (dist >= 12) & (dist <= 16)].mean()) < 0.01
    off = _process(n=4, t_sigma0_px=0.0, s_sigma0_px=0.0, s_slope_gain=0.0, s_phase_bias_px=0.0, hole_rim_p0=0.0)
    assert float(torch.stack([~sensor_frame(z, off)[1][..., 0] for _ in range(5)]).float()[:, :, on_edge].mean()) == 0.0


def test_the_far_clamp_gets_no_noise():
    p = _process()
    frame, valid = sensor_frame(_wall(6.0)[..., None], p)
    assert bool((frame == 6.0).all())


def test_the_per_env_spread_draws_inside_its_bands():
    cfg = D555NoiseCfg(amp_t_range=(0.5, 2.0), amp_s_range=(0.5, 2.0), clean_fraction=0.1, s_static_probability=0.5,
                       s_tau_range_s=(1.0, 10.0), patch_share_range=(0.0, 0.04), patch_zero_probability=0.25)
    n = 2000
    p = D555NoiseProcess(n, "cpu", cfg, step_dt=POLICY_PERIOD_S, seed=4, shape=(16, 32))
    p.reset(None)
    clean = (p.amp_t == 0) & (p.amp_s == 0)
    assert 0.07 < float(clean.float().mean()) < 0.13
    live = ~clean
    for amp in (p.amp_t[live], p.amp_s[live]):
        assert float(amp.min()) >= 0.5 - 1e-6 and float(amp.max()) <= 2.0 + 1e-6
        assert float(torch.log(amp).mean()) == pytest.approx(0.0, abs=0.05)  # log-uniform about 1
    decaying = p.s_rho < 1
    assert 0.45 < float(decaying.float().mean()) < 0.55
    taus = -p.step_dt / torch.log(p.s_rho[decaying])
    assert float(taus.min()) >= 1.0 - 1e-3 and float(taus.max()) <= 10.0 + 1e-3
    shares = p.patch.flatten(1).float().mean(1)
    assert 0.2 < float((shares == 0).float().mean()) < 0.32  # the point mass at 0, and draws at 0 in the band
    assert float(shares.max()) <= 0.045


def test_the_realistic_centres_draw_no_spread_and_an_env_reset_touches_only_its_state():
    p = _process(n=3)
    assert torch.equal(p.noise.amp_t, torch.ones(3)) and torch.equal(p.noise.s_rho, torch.ones(3))
    assert not bool(p.noise.patch.any())
    s0 = p.noise.s_unit.clone()
    p.reset(torch.tensor([1]))
    assert torch.equal(p.noise.s_unit[[0, 2]], s0[[0, 2]]) and not torch.equal(p.noise.s_unit[1], s0[1])


def test_the_same_seed_gives_the_same_frames():
    z = _floor(2)[..., None]
    a, b = _process(n=2, seed=5), _process(n=2, seed=5)
    for _ in range(3):
        fa, va = sensor_frame(z, a)
        fb, vb = sensor_frame(z, b)
        assert torch.equal(fa, fb) and torch.equal(va, vb)


def test_the_critic_reads_the_clean_render_with_the_noise_on():
    from isaaclab.managers import SceneEntityCfg

    from strafer_lab.tasks.navigation.mdp.observations import depth_image

    raw = _floor(2)[..., None]
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(output={"distance_to_image_plane": raw}))
    env = types.SimpleNamespace(num_envs=2, device="cpu", scene=types.SimpleNamespace(sensors={"cam": sensor}))
    clean = depth_image(env, SceneEntityCfg("cam"), perceived=False)
    env._d555_sensor = _process(n=2)
    assert torch.equal(depth_image(env, SceneEntityCfg("cam"), perceived=False), clean)
    assert not torch.equal(depth_image(env, SceneEntityCfg("cam")), clean)


def test_the_contract_refuses_a_second_disparity_noise_and_the_noise_without_the_mask():
    from strafer_lab.tasks.navigation import sim_real_cfg as src

    contract = copy.deepcopy(src.ROBUST_TRAINING_CONTRACT)
    d = contract.sensors.depth_camera
    params = src.get_d555_sensor_params(contract)
    assert set(params["noise"]) <= {f.name for f in dataclasses.fields(D555NoiseCfg)}
    assert src.get_depth_noise(contract).disparity_noise_px == 0.0
    d.disparity_noise_px = 0.08
    with pytest.raises(ValueError, match="replaces the 80x45 disparity noise"):
        src.get_depth_noise(contract)
    d.stereo_mask_enable = False
    with pytest.raises(ValueError, match="enable the mask"):
        src.get_d555_sensor_params(contract)


def test_on_at_the_centres_on_the_realistic_tier_and_with_the_spread_on_the_robust_one():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    real = composed.StraferNavCfg_RLDepthSubgoal_Real().events.randomize_d555_sensor.params["noise"]
    assert real["amp_t_range"] == real["amp_s_range"] == (1.0, 1.0) and real["clean_fraction"] == 0.0
    assert real["s_static_probability"] == 1.0 and real["patch_share_range"] == (0.0, 0.0)
    robust = composed.StraferNavCfg_RLDepthSubgoal_Robust().events.randomize_d555_sensor.params["noise"]
    assert robust["amp_t_range"] == robust["amp_s_range"] == (0.5, 2.0) and robust["clean_fraction"] == 0.1
    assert robust["s_static_probability"] == 0.5 and robust["s_tau_range_s"] == (1.0, 10.0)
    assert robust["patch_share_range"] == (0.0, 0.04) and robust["patch_zero_probability"] == 0.25
    for name in ("StraferNavCfg_RLDepthSubgoal_Real", "StraferNavCfg_RLDepthSubgoal_Robust"):
        cfg = getattr(composed, name)()
        noise = cfg.observations.policy.depth_image.noise
        # no 80x45 disparity noise
        assert noise.disparity_noise_px == 0.0 and noise.disparity_noise_px_range is None, name
        assert cfg.observations.critic.depth_image.params["perceived"] is False, name
    assert getattr(composed.StraferNavCfg_Depth_Ideal().events, "randomize_d555_sensor", None) is None
