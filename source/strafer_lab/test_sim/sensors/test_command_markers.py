# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Debug drawing, measured in the renderers a policy and a dataset read.

A debug marker made of scene geometry is rendered by every camera in the stage, the
D555 included. These put a command 1 m ahead of the camera, request debug
visualisation on both command families (the subgoal term the policy trains on and the
goal term the bridge composes), and require the policy depth to stay bit-identical.
Teleop draws its target through Isaac Sim's debug-draw interface instead, while it
records the perception camera. That draw does reach both cameras' RGB and neither
camera's depth, which is why teleop's marker is opt-in; this pins both halves. RGB is
not reproducible across re-renders, so the RGB half is measured by the marker's own
colour rather than by equality.
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
from strafer_lab.tasks.navigation.d555_cfg import make_d555_perception_camera_cfg
from strafer_shared.constants import CAMERA_OFFSET_X

SETTLE_STEPS = 8
AHEAD_OF_CAMERA_M = 1.0
MARKER_Z_M = 0.15
PROBE_PRIM = "/World/marker_probe_sphere"
# Teleop's target marker: a point 0.30 m over the target, bright green, size 40.
TELEOP_MARKER_LIFT_M = 0.30
TELEOP_MARKER_RGBA = (0.1, 1.0, 0.1, 1.0)
TELEOP_MARKER_SIZE = 40.0
TELEOP_MARKER_EXTENSION = "isaacsim.util.debug_draw"
GREEN_MARGIN = 60  # a pixel carries the marker's green when it exceeds red and blue by this
DEPTH_PRODUCTS = ("policy depth", "policy grid", "perception depth")


@pytest.fixture(scope="module")
def env():
    cfg = StraferNavCfg_RLDepthSubgoalEnriched_Robust()
    cfg.scene.num_envs = 1
    # The bridge composes the goal objective; carry its term beside the subgoal one.
    cfg.commands.bridge_goal = mdp.GoalCommandProcRoomCfg(
        asset_name="robot", resampling_time_range=(1.0e6, 1.0e6), multi_goal=False
    )
    # Teleop records the perception camera; carry it beside the policy one.
    cfg.scene.d555_camera_perception = make_d555_perception_camera_cfg()
    instance = ManagerBasedRLEnv(cfg)
    instance.reset(seed=7)
    zero = torch.zeros(1, instance.action_space.shape[-1], device=instance.device)
    for _ in range(SETTLE_STEPS):
        instance.step(zero)
    yield instance
    instance.close()


def _render_products(env):
    """Force a fresh render; return every image a policy or a dataset reads, by name."""
    policy = env.scene.sensors["d555_camera"]
    perception = env.scene.sensors["d555_camera_perception"]
    env.sim.render()
    for camera in (policy, perception):
        camera.update(dt=camera.cfg.update_period, force_recompute=True)

    def finite(plane):
        return torch.nan_to_num(plane.detach().clone().float(), nan=-1.0, posinf=-2.0)

    return {
        "policy rgb": policy.data.output["rgb"].detach().clone(),
        "policy depth": finite(policy.data.output["distance_to_image_plane"]),
        "policy grid": mdp.depth_image(
            env, sensor_cfg=SceneEntityCfg("d555_camera"), max_depth=6.0
        ).detach().clone(),
        "perception rgb": perception.data.output["rgb"].detach().clone(),
        "perception depth": finite(perception.data.output["distance_to_image_plane"]),
    }


def _depth(env):
    """The raw depth plane and the policy's reduced grid, freshly rendered."""
    products = _render_products(env)
    return products["policy depth"], products["policy grid"]


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


def _debug_draw_interface():
    """Teleop's debug-draw interface, with its extension enabled if the test app lacks it."""
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled(TELEOP_MARKER_EXTENSION):
        manager.set_extension_enabled_immediate(TELEOP_MARKER_EXTENSION, True)
    from isaacsim.util.debug_draw import _debug_draw

    return _debug_draw.acquire_debug_draw_interface()


def _marker_green(products):
    """Pixels carrying the marker's colour, per RGB product."""
    counts = {}
    for name in ("policy rgb", "perception rgb"):
        rgb = products[name][0, ..., :3].float()
        green = rgb[..., 1]
        counts[name] = int((
            (green - rgb[..., 0] > GREEN_MARGIN) & (green - rgb[..., 2] > GREEN_MARGIN)
        ).sum())
    return counts


def test_teleop_target_marker_reaches_rgb_and_not_depth(env):
    point = _point_ahead_of_camera(env)
    x, y = float(point[0]), float(point[1])
    # Teleop lifts the marker over its target; draw both heights, so neither can be missed.
    marker_points = [(x, y, MARKER_Z_M), (x, y, MARKER_Z_M + TELEOP_MARKER_LIFT_M)]
    draw = _debug_draw_interface()
    draw.clear_points()

    before = _render_products(env)
    assert _marker_green(before) == {"policy rgb": 0, "perception rgb": 0}, "the scene carries the marker's colour"

    draw.draw_points(marker_points, [TELEOP_MARKER_RGBA] * 2, [TELEOP_MARKER_SIZE] * 2)
    assert draw.get_num_points() == len(marker_points), "the marker never reached the interface"
    drawn = _render_products(env)
    for name, count in _marker_green(drawn).items():
        assert count > 0, f"the marker did not reach the {name} — teleop's default may be reconsidered"
    for name in DEPTH_PRODUCTS:
        assert torch.equal(drawn[name], before[name]), f"the marker reached the {name}"

    draw.clear_points()
    assert draw.get_num_points() == 0
    cleared = _render_products(env)
    assert _marker_green(cleared) == {"policy rgb": 0, "perception rgb": 0}, "clearing left the marker drawn"

    # Positive control: real geometry at the marker's own point reaches the depth it is compared in.
    probe = sim_utils.SphereCfg(radius=0.12)
    probe.func(PROBE_PRIM, probe, translation=marker_points[0])
    control = _render_products(env)
    for name in DEPTH_PRODUCTS:
        assert not torch.equal(control[name], before[name]), f"a sphere at the marker did not reach the {name}"
    sim_utils.delete_prim(PROBE_PRIM)
    restored = _render_products(env)
    for name in DEPTH_PRODUCTS:
        assert torch.equal(restored[name], before[name]), f"removing the sphere did not restore the {name}"
