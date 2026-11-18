# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The chassis's measured properties: the payload, the mount's translation, the IMU's per-sample noise.

The payload is added once at startup to the chassis body through Isaac Lab's own mass and centre-of-mass terms, and at
no payload no term is installed. The mount offset can draw a translation of the D555 housing after its angles; the
IMU's acceleration then gains the moved housing's lever arm and the rendered camera prim moves by the same draw, and at
a zero band nothing is drawn. The IMU's white noise can be set per sample in place of the density form.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

N = 16


def _rand_after(seed, fn):
    torch.manual_seed(seed)
    fn()
    return torch.rand(4)


def test_a_zero_translation_band_draws_nothing_and_stores_nothing():
    from strafer_lab.tasks.navigation.mdp import events

    env = SimpleNamespace(num_envs=N, device="cpu")
    ids = torch.arange(N)
    after = _rand_after(3, lambda: events.randomize_d555_mount_offset(env, ids, max_angle_deg=1.0))
    assert not hasattr(env, "_d555_mount_translation")
    env2 = SimpleNamespace(num_envs=N, device="cpu")
    after2 = _rand_after(3, lambda: events.randomize_d555_mount_offset(env2, ids, 1.0, (0.0, 0.0, 0.0)))
    assert torch.equal(after, after2) and torch.equal(env.__dict__["_d555_mount_quat"], env2._d555_mount_quat)


def test_the_translation_draws_per_env_inside_its_band_after_the_angles():
    from strafer_lab.tasks.navigation.mdp import events

    band = (0.02, 0.02, 0.01)
    plain, moved = SimpleNamespace(num_envs=N, device="cpu"), SimpleNamespace(num_envs=N, device="cpu")
    ids = torch.arange(N)
    torch.manual_seed(9)
    events.randomize_d555_mount_offset(plain, ids, 3.0)
    torch.manual_seed(9)
    events.randomize_d555_mount_offset(moved, ids, 3.0, band)
    assert torch.equal(plain._d555_mount_quat, moved._d555_mount_quat)  # the angles draw first, unchanged
    t = moved._d555_mount_translation
    assert t.shape == (N, 3)
    for axis, m in enumerate(band):
        assert float(t[:, axis].abs().max()) <= m and float(t[:, axis].std()) > 0.2 * m, axis
    before = t.clone()
    events.randomize_d555_mount_offset(moved, torch.tensor([2, 5]), 3.0, band)
    changed = (moved._d555_mount_translation != before).any(dim=1)
    assert changed[[2, 5]].all() and not changed[[0, 1, 3, 4]].any()


class _Robot:
    def __init__(self, n):
        self.data = SimpleNamespace(body_quat_w=torch.tensor([[0.0, 0.0, 0.3826834, 0.9238795]]).repeat(n, 3, 1),
                                    body_ang_acc_w=torch.tensor([[0.4, -1.2, 2.5]]).repeat(n, 3, 1))

    def find_bodies(self, names):
        return [1], ["body_link"]


def _imu_env(n, translation=None):
    import warp as wp

    sensor = SimpleNamespace(data=SimpleNamespace(lin_acc_b=wp.from_torch(torch.tensor([[0.3, -0.1, 9.81]]).repeat(n, 1)),
                                                  ang_vel_b=wp.from_torch(torch.tensor([[0.2, 0.1, -3.0]]).repeat(n, 1))))
    robot = _Robot(n)
    robot.data = SimpleNamespace(body_quat_w=wp.from_torch(robot.data.body_quat_w.contiguous()),
                                 body_ang_acc_w=wp.from_torch(robot.data.body_ang_acc_w.contiguous()))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"imu": sensor}, __getitem__=None), num_envs=n, device="cpu")
    env.scene = _Scene({"robot": robot}, {"imu": sensor})
    if translation is not None:
        env._d555_mount_translation = translation
    return env


class _Scene(dict):
    def __init__(self, entities, sensors):
        super().__init__(entities)
        self.sensors = sensors


def test_the_imu_gains_the_moved_housings_lever_arm_and_nothing_without_one():
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply_inverse

    from strafer_lab.tasks.navigation.mdp import observations as o

    n = 4
    cfg = SceneEntityCfg("imu")
    base = o.imu_linear_acceleration(_imu_env(n), cfg)
    assert torch.allclose(base, torch.tensor([[0.3, -0.1, 9.81]]).repeat(n, 1))  # no translation: the sensor's reading
    dr = torch.tensor([[0.02, -0.01, 0.005]]).repeat(n, 1)
    moved = o.imu_linear_acceleration(_imu_env(n, dr), cfg)
    omega = torch.tensor([[0.2, 0.1, -3.0]]).repeat(n, 1)
    quat = torch.tensor([[0.0, 0.0, 0.3826834, 0.9238795]]).repeat(n, 1)
    alpha = quat_apply_inverse(quat, torch.tensor([[0.4, -1.2, 2.5]]).repeat(n, 1))
    want = base + torch.linalg.cross(alpha, dr) + torch.linalg.cross(omega, torch.linalg.cross(omega, dr))
    assert torch.allclose(moved, want, atol=1e-6)


class _CamView:
    def __init__(self):
        self.calls = []

    def set_local_poses(self, translations=None, orientations=None, indices=None):
        import warp as wp

        self.calls.append((None if translations is None else wp.to_torch(translations).clone(),
                           wp.to_torch(orientations).clone(), wp.to_torch(indices).clone()))


def _cam_env(n, translation=None):
    view = _CamView()
    offset = SimpleNamespace(pos=(0.20, 0.0, 0.25), rot=(0.5, -0.5, 0.5, -0.5), convention="ros")
    camera = SimpleNamespace(cfg=SimpleNamespace(offset=offset), _view=view)
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"d555_camera": camera}), num_envs=n, device="cpu")
    env._d555_mount_quat = torch.tensor([[0.01, -0.02, 0.005, 0.99973]]).repeat(n, 1)
    if translation is not None:
        env._d555_mount_translation = translation
    return env, view


def test_the_camera_prim_moves_by_the_same_draw_and_keeps_its_orientation():
    from strafer_lab.tasks.navigation.mdp import events

    n, ids = 4, torch.tensor([0, 2])
    env, view = _cam_env(n)
    events.jitter_d555_camera_prim_pose(env, ids)
    assert view.calls[0][0] is None  # no translation: orientation only, as before
    dr = torch.tensor([[0.01, 0.02, -0.005]]) * torch.arange(1, n + 1).unsqueeze(1)
    env2, view2 = _cam_env(n, dr)
    events.jitter_d555_camera_prim_pose(env2, ids)
    translations, orientations, indices = view2.calls[0]
    assert torch.allclose(translations, torch.tensor([[0.20, 0.0, 0.25]]) + dr[ids])
    assert torch.equal(orientations, view.calls[0][1]) and indices.tolist() == ids.tolist()


def test_no_payload_installs_no_term_and_a_payload_installs_isaac_labs_two():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed
    from strafer_lab.tasks.navigation import strafer_env_cfg as sec

    for name in ("StraferNavCfg_RLDepthSubgoal_Real", "StraferNavCfg_RLNoCamSubgoal_Robust"):
        events = getattr(composed, name)().events
        assert events.add_payload_mass is None and events.shift_payload_com is None, name
    saved = sec._PAYLOAD_MASS_KG, sec._PAYLOAD_COM_SHIFT_M
    try:
        sec._PAYLOAD_MASS_KG, sec._PAYLOAD_COM_SHIFT_M = 0.959, (0.01, -0.002, 0.03)
        mass, com = sec._payload_terms()
    finally:
        sec._PAYLOAD_MASS_KG, sec._PAYLOAD_COM_SHIFT_M = saved
    assert mass.mode == com.mode == "startup"
    assert mass.func.__name__ == "randomize_rigid_body_mass" and com.func.__name__ == "randomize_rigid_body_com"
    assert mass.params["operation"] == "add" and mass.params["mass_distribution_params"] == (0.959, 0.959)
    assert mass.params["recompute_inertia"] is True
    assert com.params["com_range"] == {"x": (0.01, 0.01), "y": (-0.002, -0.002), "z": (0.03, 0.03)}
    for term in (mass, com):
        assert term.params["asset_cfg"].name == "robot" and term.params["asset_cfg"].body_names == "body_link"


@pytest.mark.parametrize("kind", ["accel", "gyro"])
def test_the_imu_per_sample_sigma_replaces_the_density_form_when_set(kind):
    import copy

    from strafer_lab.tasks.navigation import sim_real_cfg as src

    getter = {"accel": src.get_imu_accel_noise, "gyro": src.get_imu_gyro_noise}[kind]
    contract = copy.deepcopy(src.REAL_ROBOT_CONTRACT)
    imu = contract.sensors.imu
    density = imu.accel_noise_density if kind == "accel" else imu.gyro_noise_density
    today = getter(contract)
    assert today.noise_cfg.std == pytest.approx(density * math.sqrt(contract.timing.control_frequency_hz))
    setattr(imu, f"{kind}_noise_std_sample", 0.123)
    sampled = getter(contract)
    assert sampled.noise_cfg.std == pytest.approx(0.123)
    assert getattr(sampled, f"{kind}_noise_std") == pytest.approx(0.123)


class _MassView:
    def __init__(self, n, bodies=5):
        self.masses = torch.linspace(0.1, 2.0, bodies).repeat(n, 1)
        self.inertias = torch.linspace(1e-4, 9e-3, bodies * 9).reshape(bodies, 9).repeat(n, 1, 1)

    def get_masses(self):
        return self.masses.clone()

    def set_masses(self, m, ids):
        import warp as wp

        self.masses = wp.to_torch(m).clone()

    def get_inertias(self):
        return self.inertias.clone()

    def set_inertias(self, m, ids):
        import warp as wp

        self.inertias = wp.to_torch(m).clone()


def test_the_mass_factor_scales_each_bodys_inertia_with_its_mass():
    from strafer_lab.tasks.navigation.mdp import events

    view = _MassView(N)
    env = SimpleNamespace(scene={"robot": SimpleNamespace(root_view=view)}, num_envs=N, device="cpu")
    m0, i0 = view.masses.clone(), view.inertias.clone()
    for ids in (torch.arange(N), torch.tensor([3, 7]), torch.arange(N)):
        before = view.inertias.clone()
        events.randomize_mass(env, ids, mass_range=(0.85, 1.15))
        factor = view.masses / m0  # the env's factor, the same on every body
        assert torch.allclose(factor, factor[:, :1].expand_as(factor))
        assert torch.allclose(view.inertias, i0 * factor[:, :1, None], rtol=1e-6)
        others = [e for e in range(N) if e not in ids.tolist()]
        assert torch.equal(view.inertias[others], before[others])  # other envs untouched
