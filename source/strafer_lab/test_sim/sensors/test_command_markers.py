# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The command terms' debug visualisation, measured in the renderer the policy reads.

A debug marker made of scene geometry is rendered by every camera in the stage, the
D555 included. These put a command 1 m ahead of the camera, request debug
visualisation on both command families (the subgoal term the policy trains on and the
goal term the bridge composes), and require the policy depth to stay bit-identical.
A real sphere at the same point is the positive control: it proves the point is in
view and that the comparison sees geometry there.

Isaac Sim is launched by the root ``test_sim/conftest.py``.
"""

# --- Imports (Isaac Sim launched by root conftest.py) ---

import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from strafer_lab.tasks.navigation import mdp
from strafer_lab.tasks.navigation.composed_env_cfg import (
    StraferNavCfg_RLDepthSubgoalEnriched_Robust,
)
from strafer_shared.constants import CAMERA_OFFSET_X

SETTLE_STEPS = 8
AHEAD_OF_CAMERA_M = 1.0
MARKER_Z_M = 0.15
PROBE_PRIM = "/World/marker_probe_sphere"


@pytest.fixture(scope="module")
def env():
    cfg = StraferNavCfg_RLDepthSubgoalEnriched_Robust()
    cfg.scene.num_envs = 1
    # The bridge composes the goal objective; carry its term beside the subgoal one.
    cfg.commands.bridge_goal = mdp.GoalCommandProcRoomCfg(
        asset_name="robot", resampling_time_range=(1.0e6, 1.0e6), multi_goal=False
    )
    instance = ManagerBasedRLEnv(cfg)
    instance.reset(seed=20260921)
    zero = torch.zeros(1, instance.action_space.shape[-1], device=instance.device)
    for _ in range(SETTLE_STEPS):
        instance.step(zero)
    yield instance
    instance.close()


def _depth(env):
    """Force a fresh render; return the raw depth plane and the policy's reduced grid."""
    camera = env.scene.sensors["d555_camera"]
    env.sim.render()
    camera.update(dt=camera.cfg.update_period, force_recompute=True)
    plane = camera.data.output["distance_to_image_plane"]
    raw = torch.nan_to_num(plane.detach().clone().float(), nan=-1.0, posinf=-2.0)
    grid = mdp.depth_image(env, sensor_cfg=SceneEntityCfg("d555_camera"), max_depth=6.0)
    return raw, grid.detach().clone()


def _point_ahead_of_camera(env):
    """World point on the camera axis, ``AHEAD_OF_CAMERA_M`` in front of the lens."""
    data = env.scene["robot"].data
    root = wp.to_torch(data.root_pos_w)[0].float()
    quat = wp.to_torch(data.root_quat_w)[0:1].float()
    forward = quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]], device=env.device))[0]
    point = root + forward * (CAMERA_OFFSET_X + AHEAD_OF_CAMERA_M)
    point[2] = MARKER_Z_M
    return point


def _command_marker_prim(env):
    return env.sim.stage.GetPrimAtPath("/Visuals/Command").IsValid()


def test_requesting_command_debug_vis_leaves_the_policy_depth_unchanged(env):
    point = _point_ahead_of_camera(env)

    # Put both commands on the point, where a marker would be drawn.
    subgoal = env.command_manager.get_term("goal_command")
    goal = env.command_manager.get_term("bridge_goal")
    subgoal._subgoal[0, :2] = point[:2]
    goal._goal[0, :2] = point[:2]

    raw0, grid0 = _depth(env)
    raw, grid = _depth(env)
    assert torch.equal(raw, raw0) and torch.equal(grid, grid0), (
        "an untouched scene re-rendered differently — the comparison is not sound"
    )

    # Positive control: real geometry at the same point must reach the depth.
    probe = sim_utils.SphereCfg(radius=0.12)
    probe.func(PROBE_PRIM, probe, translation=tuple(point.tolist()))
    raw, grid = _depth(env)
    assert (raw - raw0).abs().max().item() > 0.1, "a sphere at the point did not reach the depth"
    assert not torch.equal(grid, grid0), "a sphere at the point did not reach the policy grid"
    sim_utils.delete_prim(PROBE_PRIM)
    raw, grid = _depth(env)
    assert torch.equal(raw, raw0) and torch.equal(grid, grid0), "removing the sphere did not restore the image"

    for term in (subgoal, goal):
        term.set_debug_vis(True)
        env.sim.vis_marker_registry.dispatch_callbacks()
        raw, grid = _depth(env)
        assert torch.equal(raw, raw0), f"{type(term).__name__}: debug vis reached the D555 depth"
        assert torch.equal(grid, grid0), f"{type(term).__name__}: debug vis reached the policy grid"
        assert not _command_marker_prim(env), f"{type(term).__name__}: created marker prims"
        term.set_debug_vis(False)
