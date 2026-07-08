"""Unit tests for the DEPTH_SUBGOAL depth-sensed obstacle-proximity reward.

``depth_obstacle_proximity_penalty`` reads the depth camera's
``distance_to_image_plane`` output plus the camera's ground-truth pose (for
floor-plane exclusion), so the tests pin its contract against synthetic depth
fields and poses mocked with ``SimpleNamespace`` — no Kit needed (same pattern
as ``test_path_tracking_mdp.py``). Two mock regimes:

- ``_env_with_depth`` mounts the camera far above the floor, taking the floor
  plane out of sensing range so the saturating-reciprocal contract is pinned
  in isolation: fires on a near obstacle, exactly zero when clear, saturates
  at near-contact, ignores inf/nan, reads the warp-array output the live
  TiledCamera emits, and composes additively with the path-tracking readers.
- ``_floor_scene_env`` mounts the camera at the robot's real height/FOV so the
  floor is always in the lower FOV — pinning the floor-plane exclusion: a bare
  floor accrues exactly zero (level or pitched), while low boxes, mid-range
  obstacles, and walls remain detected through it.
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
from strafer_shared.constants import (
    D555_FOCAL_LENGTH_MM,
    D555_HORIZONTAL_APERTURE_MM,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
)

_SENSOR_NAME = "d555_camera"
_SENSOR_CFG = SimpleNamespace(name=_SENSOR_NAME)

# Level ROS-optical->world orientation facing world +X (x_opt right -> -Y,
# y_opt down -> -Z, z_opt forward -> +X). XYZW — Isaac Lab 3.0 convention;
# matches the d555 mount's frame-alignment quaternion.
_LEVEL_QUAT_XYZW = (-0.5, 0.5, -0.5, 0.5)

# Real d555 policy-camera geometry. The 80x45 (16:9) render has square pixels,
# and at that aspect the square-pixel vertical FOV IS the real sensor's ~56 deg
# — the same derivation the reward uses. (This is the whole point of the 80x45
# resolution: RTX derives VFOV from resolution, so 16:9 gives the right FOV.)
_TAN_HALF_HFOV = D555_HORIZONTAL_APERTURE_MM / (2.0 * D555_FOCAL_LENGTH_MM)
_TAN_HALF_VFOV = _TAN_HALF_HFOV * DEPTH_HEIGHT / DEPTH_WIDTH
_CAM_HEIGHT = 0.35  # optical center above the floor, ~the robot's real mount
_FAR = 6.0  # max_depth: the "nothing in range" reading


def _make_env(depth, *, cam_pos, quat_xyzw):
    """A minimal env: depth camera output + ground-truth pose + intrinsics."""
    num_envs = depth.shape[0]  # torch.Tensor or wp.array — both expose .shape
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(
            spawn=SimpleNamespace(
                focal_length=D555_FOCAL_LENGTH_MM,
                horizontal_aperture=D555_HORIZONTAL_APERTURE_MM,
            )
        ),
        data=SimpleNamespace(
            output={"distance_to_image_plane": depth},
            pos_w=torch.tensor([cam_pos] * num_envs, dtype=torch.float32),
            quat_w_ros=torch.tensor([quat_xyzw] * num_envs, dtype=torch.float32),
        ),
    )
    return SimpleNamespace(
        scene=SimpleNamespace(
            sensors={_SENSOR_NAME: sensor},
            env_origins=torch.zeros(num_envs, 3),
        )
    )


def _env_with_depth(depth):
    """Isolation mock: camera 100 m up — the floor plane is far beyond
    ``max_depth`` at every pixel, so no reading is ever floor-classified and
    the saturating-reciprocal contract is exercised alone."""
    return _make_env(depth, cam_pos=(0.0, 0.0, 100.0), quat_xyzw=_LEVEL_QUAT_XYZW)


def _field(rows):
    """Shape a list of per-env HxW pixel grids into (num_envs, H, W, 1)."""
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(-1)


def _expected(min_depth, *, distance_threshold=1.0, saturation_depth=0.3, epsilon=0.1):
    floored = max(min_depth, saturation_depth)
    return max(0.0, 1.0 / (floored + epsilon) - 1.0 / (distance_threshold + epsilon))


# ---------------------------------------------------------------------------
# Floor-scene helpers: matrix-based ray tracing, independent of the reward's
# quaternion path — a sign/convention error in either implementation breaks
# the agreement these tests assert.
# ---------------------------------------------------------------------------


def _level_rotation():
    """Optical->world matrix for the level +X-facing camera (columns are the
    world images of the optical x/y/z axes)."""
    return torch.tensor(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64
    )


def _pitch_down(matrix, angle_rad):
    """Compose a downward pitch (rotation about world +Y) onto ``matrix``."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    pitch = torch.tensor(
        [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=torch.float64
    ).T  # Rot_Y(a): forward (1,0,0) -> (cos a, 0, -sin a)
    return pitch @ matrix


def _quat_xyzw_from_matrix(m):
    """Rotation matrix -> XYZW quaternion (w-positive branch; test-only)."""
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = (m[2, 1] - m[1, 2]) / (4.0 * w)
    y = (m[0, 2] - m[2, 0]) / (4.0 * w)
    z = (m[1, 0] - m[0, 1]) / (4.0 * w)
    return (float(x), float(y), float(z), float(w))


def _floor_image(rotation, cam_height=_CAM_HEIGHT):
    """Ray-trace a bare-floor depth image (H, W) for a camera at
    ``cam_height`` with optical->world ``rotation``; non-floor rays read
    ``_FAR``. Uses matrices, not quaternions."""
    rows = torch.arange(DEPTH_HEIGHT, dtype=torch.float64)
    cols = torch.arange(DEPTH_WIDTH, dtype=torch.float64)
    y_ndc = (2.0 * rows + 1.0 - DEPTH_HEIGHT) / DEPTH_HEIGHT
    x_ndc = (2.0 * cols + 1.0 - DEPTH_WIDTH) / DEPTH_WIDTH
    yy, xx = torch.meshgrid(y_ndc, x_ndc, indexing="ij")
    rays = torch.stack(
        [xx * _TAN_HALF_HFOV, yy * _TAN_HALF_VFOV, torch.ones_like(xx)], dim=-1
    )
    world = rays.reshape(-1, 3) @ rotation.T
    dz = world[:, 2].reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
    depth = torch.full((DEPTH_HEIGHT, DEPTH_WIDTH), _FAR, dtype=torch.float64)
    hits = dz < 0
    depth[hits] = torch.clamp(cam_height / (-dz[hits]), max=_FAR)
    return depth.to(torch.float32)


def _floor_scene_env(image, *, quat_xyzw=_LEVEL_QUAT_XYZW, cam_height=_CAM_HEIGHT):
    return _make_env(
        image.unsqueeze(0).unsqueeze(-1),
        cam_pos=(0.0, 0.0, cam_height),
        quat_xyzw=quat_xyzw,
    )


# ---------------------------------------------------------------------------
# Saturating-reciprocal contract (floor out of range)
# ---------------------------------------------------------------------------


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
    base = _env_with_depth(depth)
    term = SimpleNamespace(
        cross_track_error=torch.tensor([0.12]),
        along_track_progress=torch.tensor([0.03]),
    )
    env = SimpleNamespace(
        scene=base.scene,
        command_manager=SimpleNamespace(get_term=lambda name: term),
    )

    depth_pen = depth_obstacle_proximity_penalty(env, _SENSOR_CFG)
    cross = path_cross_track_error(env, "goal_command")
    progress = path_along_track_progress(env, "goal_command")

    # Path-tracking readers see only the command term.
    torch.testing.assert_close(cross, torch.tensor([0.12]))
    torch.testing.assert_close(progress, torch.tensor([0.03]))

    # Clearing the obstacle drives the depth penalty to zero but does not
    # touch the path-tracking outputs.
    sensor = env.scene.sensors[_SENSOR_NAME]
    sensor.data.output["distance_to_image_plane"] = torch.full_like(depth, 6.0)
    assert depth_obstacle_proximity_penalty(env, _SENSOR_CFG)[0].item() == 0.0
    torch.testing.assert_close(path_cross_track_error(env, "goal_command"), torch.tensor([0.12]))
    assert depth_pen[0].item() > 0.0  # original still positive — independent signals


# ---------------------------------------------------------------------------
# Floor-plane exclusion (real mount geometry: floor always in the lower FOV)
# ---------------------------------------------------------------------------


def test_bare_floor_reads_zero_at_real_mount_height():
    """The load-bearing case: at the robot's real camera height/FOV the bare
    floor is visible at ~0.5 m z-depth in the bottom rows — far inside the
    penalty threshold. Floor-plane exclusion must make an open scene read
    exactly zero instead of a saturated ambient penalty."""
    image = _floor_image(_level_rotation())
    # Geometry sanity: at the real 16:9 VFOV (~56 deg, 80x45) the bottom row
    # reads the floor at ~0.67 m — still well inside the penalty threshold, the
    # sub-threshold depth that made an unmasked min saturate.
    assert 0.63 < image[-1, 0].item() < 0.70
    out = depth_obstacle_proximity_penalty(_floor_scene_env(image), _SENSOR_CFG)
    assert out[0].item() == 0.0


def test_bare_floor_reads_zero_under_pitch():
    """Braking/suspension pitch moves the sensed floor closer; the exclusion
    uses the camera's true pose, so a pitched bare floor still reads zero.
    The image is ray-traced with rotation matrices while the reward rotates
    by quaternion — agreement also cross-validates the quat convention."""
    pitched = _pitch_down(_level_rotation(), math.radians(5.0))
    image = _floor_image(pitched)
    # The pitched floor reads closer than level (the false-positive bait).
    assert image[-1, 0].item() < _floor_image(_level_rotation())[-1, 0].item()
    env = _floor_scene_env(image, quat_xyzw=_quat_xyzw_from_matrix(pitched))
    out = depth_obstacle_proximity_penalty(env, _SENSOR_CFG)
    assert out[0].item() == 0.0


def test_low_box_on_floor_is_detected_through_the_exclusion():
    """A 20 cm-tall box at 0.5 m — small floor clutter dead ahead. Its face
    occludes floor pixels well above the margin, so it must be detected even
    though it lives entirely in the lower FOV the floor occupies (the case a
    row-crop heuristic would go blind to)."""
    image = _floor_image(_level_rotation())
    # Lower-FOV rows (80x45 grid) whose bare-floor depth is ~0.77-1.22 m — well
    # beyond the 0.5 m box face + margin, so the exclusion must not swallow it.
    image[34:42, :] = 0.5
    out = depth_obstacle_proximity_penalty(_floor_scene_env(image), _SENSOR_CFG)
    assert math.isclose(out[0].item(), _expected(0.5), rel_tol=1e-5)


def test_midrange_obstacle_is_visible_past_the_floor_reading():
    """An obstacle at 0.8 m sits *beyond* the ~0.5 m floor reading — with the
    floor left in the min it was invisible (min-clipped). The exclusion must
    restore the full threshold band: the penalty reads the obstacle's 0.8 m,
    not the floor's 0.5 m."""
    image = _floor_image(_level_rotation())
    image[26:34, :] = 0.8  # shallow-angle rows (80x45): floor expectation >> 0.8 m
    out = depth_obstacle_proximity_penalty(_floor_scene_env(image), _SENSOR_CFG)
    assert math.isclose(out[0].item(), _expected(0.8), rel_tol=1e-5)


def test_wall_above_horizon_is_detected_with_floor_in_view():
    """Above-horizon rays never intersect the floor plane, so a wall there is
    always obstacle-eligible; the floor below must not mask it."""
    image = _floor_image(_level_rotation())
    image[0:22, :] = 0.6  # wall face filling the above-horizon half (80x45)
    out = depth_obstacle_proximity_penalty(_floor_scene_env(image), _SENSOR_CFG)
    assert math.isclose(out[0].item(), _expected(0.6), rel_tol=1e-5)
