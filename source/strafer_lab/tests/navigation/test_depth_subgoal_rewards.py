"""Unit tests for the DEPTH_SUBGOAL depth-sensed obstacle-proximity reward.

``depth_obstacle_proximity_penalty`` is a thin reader of the depth camera's
``distance_to_image_plane`` output, so the tests pin its contract against a
synthetic depth field mocked with a ``SimpleNamespace`` sensor — no Kit needed
(same pattern as ``test_path_tracking_mdp.py``). They cover the reward's
contract: fires on a near obstacle, is exactly zero when clear, saturates at
near-contact, ignores inf/nan pixels, reads the warp-array output the live
TiledCamera emits, and composes additively with the path-tracking readers (each
reads only its own channel — no double-counting).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import warp as wp

from strafer_lab.tasks.navigation.mdp.rewards import (
    depth_obstacle_proximity_penalty,
    path_along_track_progress,
    path_cross_track_error,
)

_SENSOR_NAME = "d555_camera"
_SENSOR_CFG = SimpleNamespace(name=_SENSOR_NAME)


def _env_with_depth(depth):
    """A minimal env whose depth camera outputs ``depth`` (num_envs,H,W,1)."""
    sensor = SimpleNamespace(
        data=SimpleNamespace(output={"distance_to_image_plane": depth})
    )
    return SimpleNamespace(scene=SimpleNamespace(sensors={_SENSOR_NAME: sensor}))


def _field(rows):
    """Shape a list of per-env HxW pixel grids into (num_envs, H, W, 1)."""
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(-1)


def _expected(min_depth, *, distance_threshold=1.0, saturation_depth=0.3, epsilon=0.1):
    floored = max(min_depth, saturation_depth)
    return max(0.0, 1.0 / (floored + epsilon) - 1.0 / (distance_threshold + epsilon))


def test_penalty_zero_when_clear():
    """Nearest surface at/beyond the threshold accrues exactly zero."""
    env = _env_with_depth(_field([[[6.0, 6.0], [6.0, 6.0]]]))
    out = depth_obstacle_proximity_penalty(env, _SENSOR_CFG)
    assert out.shape == (1,)
    assert out[0].item() == 0.0


def test_penalty_zero_exactly_at_threshold():
    """Continuity: a nearest surface exactly at the threshold is still zero."""
    env = _env_with_depth(_field([[[1.0, 2.0], [3.0, 6.0]]]))
    out = depth_obstacle_proximity_penalty(env, _SENSOR_CFG, distance_threshold=1.0)
    assert out[0].item() == 0.0


def test_penalty_fires_on_near_obstacle():
    """A pixel inside the threshold produces a positive penalty from the min."""
    env = _env_with_depth(_field([[[6.0, 0.5], [6.0, 6.0]]]))
    out = depth_obstacle_proximity_penalty(env, _SENSOR_CFG)
    assert out[0].item() > 0.0
    assert math.isclose(out[0].item(), _expected(0.5), rel_tol=1e-5)


def test_penalty_saturates_at_near_contact():
    """Below ``saturation_depth`` the penalty is constant — near-contact does
    not blow up the reward (the brief's saturation requirement)."""
    near = _env_with_depth(_field([[[0.05, 6.0], [6.0, 6.0]]]))
    at_floor = _env_with_depth(_field([[[0.3, 6.0], [6.0, 6.0]]]))
    p_near = depth_obstacle_proximity_penalty(near, _SENSOR_CFG, saturation_depth=0.3)
    p_floor = depth_obstacle_proximity_penalty(at_floor, _SENSOR_CFG, saturation_depth=0.3)
    assert math.isclose(p_near[0].item(), p_floor[0].item(), rel_tol=1e-6)
    # And it is the maximum value the term can take.
    assert math.isclose(p_near[0].item(), _expected(0.05), rel_tol=1e-5)


def test_penalty_ignores_inf_and_nan_pixels():
    """inf/nan pixels (nothing in range) are treated as the far clip, so they
    never dominate the per-env minimum."""
    inf, nan = float("inf"), float("nan")
    env = _env_with_depth(_field([[[inf, 0.4], [nan, 6.0]]]))
    out = depth_obstacle_proximity_penalty(env, _SENSOR_CFG)
    assert math.isclose(out[0].item(), _expected(0.4), rel_tol=1e-5)


def test_penalty_reads_the_warp_array_the_tiledcamera_emits():
    """The live path: ``d555_camera`` is a TiledCamera, so its
    ``distance_to_image_plane`` output is a ``wp.array`` at runtime, not a
    torch.Tensor. Feed a warp array through the non-torch branch and confirm it
    matches the torch path (the branch every training step actually takes)."""
    depth = _field([[[6.0, 0.5], [6.0, 6.0]]])
    warp_env = _env_with_depth(wp.from_torch(depth))
    torch_env = _env_with_depth(depth)
    warp_out = depth_obstacle_proximity_penalty(warp_env, _SENSOR_CFG)
    torch_out = depth_obstacle_proximity_penalty(torch_env, _SENSOR_CFG)
    torch.testing.assert_close(warp_out, torch_out)
    assert math.isclose(warp_out[0].item(), _expected(0.5), rel_tol=1e-5)


def test_penalty_is_per_env_over_the_batch():
    """The min is taken per env; a batch returns one penalty per row."""
    env = _env_with_depth(
        _field(
            [
                [[6.0, 6.0], [6.0, 6.0]],  # clear
                [[0.5, 6.0], [6.0, 6.0]],  # near
                [[0.05, 6.0], [6.0, 6.0]],  # near-contact
            ]
        )
    )
    out = depth_obstacle_proximity_penalty(env, _SENSOR_CFG)
    assert out.shape == (3,)
    assert out[0].item() == 0.0
    assert math.isclose(out[1].item(), _expected(0.5), rel_tol=1e-5)
    assert math.isclose(out[2].item(), _expected(0.05), rel_tol=1e-5)


def test_depth_and_path_rewards_read_independent_channels():
    """Additive composition, no double-counting: the depth penalty depends only
    on the depth field and the path-tracking readers depend only on the command
    term. Perturbing one channel leaves the other reward unchanged, so the two
    simply sum in the reward manager."""
    depth = _field([[[6.0, 0.5], [6.0, 6.0]]])
    sensor = SimpleNamespace(
        data=SimpleNamespace(output={"distance_to_image_plane": depth})
    )
    term = SimpleNamespace(
        cross_track_error=torch.tensor([0.12]),
        along_track_progress=torch.tensor([0.03]),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={_SENSOR_NAME: sensor}),
        command_manager=SimpleNamespace(get_term=lambda name: term),
    )

    depth_pen = depth_obstacle_proximity_penalty(env, _SENSOR_CFG)
    cross = path_cross_track_error(env, "goal_command")
    progress = path_along_track_progress(env, "goal_command")

    # Path-tracking readers see only the command term.
    torch.testing.assert_close(cross, torch.tensor([0.12]))
    torch.testing.assert_close(progress, torch.tensor([0.03]))

    # Doubling the obstacle depth (clearing it) drives the depth penalty to zero
    # but does not touch the path-tracking outputs.
    sensor.data.output["distance_to_image_plane"] = depth * 100.0
    assert depth_obstacle_proximity_penalty(env, _SENSOR_CFG)[0].item() == 0.0
    torch.testing.assert_close(path_cross_track_error(env, "goal_command"), torch.tensor([0.12]))
    assert depth_pen[0].item() > 0.0  # original still positive — independent signals
