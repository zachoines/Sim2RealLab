# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The rendered-camera arm of the D555 mount DR, measured in the renderer.

The pure tests pin the composition sense and which variants carry the term.
What they cannot see is whether writing the camera prim's local transform
actually reaches RTX: the render consumes the scene graph, and a write that
only lands in USD could leave the image untouched. These drive the term
against a live depth render and assert the image moved, moved only on the
environments asked for, and stayed moved while the robot drove.

Isaac Sim is launched by the root ``test_sim/conftest.py``.
"""

# --- Imports (Isaac Sim launched by root conftest.py) ---

import math

import pytest
import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_from_euler_xyz

from strafer_lab.tasks.navigation.composed_env_cfg import (
    StraferNavCfg_RLDepthEnriched_Real,
)
from strafer_lab.tasks.navigation.mdp.events import jitter_d555_camera_prim_pose

NUM_ENVS = 8
SUBSET = (1, 3, 7)
JITTER_DEG = 15.0  # far above the shipped band, so a null result is unambiguous
SETTLE_STEPS = 8
DRIVE_STEPS = 20


@pytest.fixture(scope="module")
def env():
    cfg = StraferNavCfg_RLDepthEnriched_Real()
    cfg.scene.num_envs = NUM_ENVS
    instance = ManagerBasedRLEnv(cfg)
    instance.reset(seed=20260101)
    zero = torch.zeros(NUM_ENVS, instance.action_space.shape[-1], device=instance.device)
    for _ in range(SETTLE_STEPS):
        instance.step(zero)
    yield instance
    instance.close()


def _camera_local_quats(env):
    view = env.scene.sensors["d555_camera"]._view
    indices = wp.from_torch(
        torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
    )
    return wp.to_torch(view.get_local_poses(indices)[1]).clone().float()


def _depth(env):
    """Force a fresh render and return the raw depth planes (N, H, W)."""
    camera = env.scene.sensors["d555_camera"]
    env.sim.render()
    camera.update(dt=camera.cfg.update_period, force_recompute=True)
    plane = camera.data.output["distance_to_image_plane"]
    return torch.nan_to_num(plane.detach().clone().float(), nan=-1.0, posinf=-2.0)


def _per_env_max_delta(a, b):
    return (a - b).abs().flatten(1).max(dim=1).values


def _point(env, env_ids, angle_deg):
    """Put a known mount offset on ``env_ids`` and push it to the prim."""
    n = len(env_ids)
    if angle_deg == 0.0:
        offset = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).expand(n, 4)
    else:
        angle = torch.full((n,), math.radians(angle_deg), device=env.device)
        offset = quat_from_euler_xyz(torch.zeros(n, device=env.device), angle, angle)
    env._d555_mount_quat[env_ids] = offset
    jitter_d555_camera_prim_pose(env, env_ids)


def test_reset_points_every_camera_through_its_own_offset(env):
    """End-to-end: the term is wired into the enriched events, so a plain reset
    leaves each environment's camera on its own sampled misalignment."""
    quats = _camera_local_quats(env)
    # Signs are free (q and -q are one rotation), so compare on |components|.
    magnitudes = quats.abs()
    spread = magnitudes.max(dim=0).values - magnitudes.min(dim=0).values
    assert spread.max().item() > 1e-4, "every camera sits on the same orientation"
    nominal = 0.5  # |components| of the nominal mount, before any offset
    assert (magnitudes - nominal).abs().max().item() < 0.05, (
        "an offset far outside the mount band reached the prim"
    )


def test_pointing_the_prim_moves_the_render_only_where_asked(env):
    everything = torch.arange(NUM_ENVS, device=env.device)
    _point(env, everything, 0.0)
    baseline = _depth(env)
    assert _per_env_max_delta(baseline, _depth(env)).max().item() == 0.0, (
        "an untouched scene re-rendered differently — the comparison is not sound"
    )

    subset = torch.tensor(SUBSET, device=env.device)
    _point(env, subset, JITTER_DEG)
    delta = _per_env_max_delta(baseline, _depth(env))

    for i in SUBSET:
        assert delta[i].item() > 0.1, (
            f"env {i}: the render did not follow the camera prim ({delta[i].item()} m)"
        )
    others = [i for i in range(NUM_ENVS) if i not in SUBSET]
    assert delta[others].max().item() == 0.0, (
        "a per-env write reached environments it was not given"
    )

    _point(env, subset, 0.0)
    assert _per_env_max_delta(baseline, _depth(env)).max().item() == 0.0, (
        "restoring the nominal mount did not restore the image"
    )


def test_the_offset_holds_while_the_robot_drives(env):
    """A mount offset rides the chassis; a one-frame teleport would not."""
    subset = torch.tensor(SUBSET, device=env.device)
    _point(env, torch.arange(NUM_ENVS, device=env.device), 0.0)

    action = torch.zeros(NUM_ENVS, env.action_space.shape[-1], device=env.device)
    action[:, 0] = 0.4
    for _ in range(DRIVE_STEPS):
        env.step(action)

    _point(env, subset, JITTER_DEG)
    pointed = _camera_local_quats(env)[list(SUBSET)]
    for _ in range(DRIVE_STEPS):
        env.step(action)
    assert torch.allclose(
        _camera_local_quats(env)[list(SUBSET)].abs(), pointed.abs(), atol=1e-5
    ), "an episode ended mid-drive and resampled the offset — rerun"

    jittered = _depth(env)
    _point(env, subset, 0.0)
    restored = _depth(env)

    delta = _per_env_max_delta(jittered, restored)
    for i in SUBSET:
        assert delta[i].item() > 0.1, (
            f"env {i}: the offset was gone after {DRIVE_STEPS} steps of driving"
        )
    others = [i for i in range(NUM_ENVS) if i not in SUBSET]
    assert delta[others].max().item() == 0.0
