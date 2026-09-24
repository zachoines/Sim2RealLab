"""The video overlay that stands in for the command terms' debug markers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from strafer_lab.tasks.navigation.mdp.commands import GoalCommandCfg, SubgoalCommandCfg
from strafer_lab.tools.command_overlay import ROBOT_RGB, draw_command, draw_robot, project
from strafer_shared.constants import CHASSIS_LENGTH, CHASSIS_WIDTH

W, H = 1280, 720
FOCAL, APERTURE = 18.147562, 20.955  # the recording camera, /OmniverseKit_Persp
F_PX = W * FOCAL / APERTURE


def _straight_down(height_m):
    """A camera ``height_m`` above the origin looking straight down, +Y up in the image."""
    m = np.eye(4)
    m[3, 2] = height_m
    return m


def test_project_puts_the_origin_at_the_image_centre_and_scales_by_depth():
    uv, depth, f_px = project([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                              _straight_down(12.0), FOCAL, APERTURE, W, H)
    assert f_px == pytest.approx(F_PX)
    np.testing.assert_allclose(depth, 12.0)
    np.testing.assert_allclose(uv[0], (W / 2, H / 2))
    np.testing.assert_allclose(uv[1], (W / 2 + F_PX / 12.0, H / 2))
    np.testing.assert_allclose(uv[2], (W / 2, H / 2 - F_PX / 12.0))


def test_project_marks_points_behind_the_camera():
    _, depth, _ = project([(0.0, 0.0, 20.0)], _straight_down(12.0), FOCAL, APERTURE, W, H)
    assert depth[0] < 0


def _pixel(frame, x, y, height_m=12.0):
    (u, v), = project([(x, y, 0.0)], _straight_down(height_m), FOCAL, APERTURE, W, H)[0]
    return frame[int(round(v)), int(round(u))]


def _rgb(markers_cfg, key=None):
    proto = markers_cfg.markers[key] if key else next(iter(markers_cfg.markers.values()))
    return np.array([round(255 * c) for c in proto.visual_material.diffuse_color])


def test_the_subgoal_term_draws_its_subgoal_and_path_in_its_cfg_styles():
    cfg = SubgoalCommandCfg(asset_name="robot")
    path = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [0.0, 0.0]]])
    term = SimpleNamespace(
        cfg=cfg,
        command=torch.tensor([[-2.0, -1.0, 0.0]]),
        path_cursor=SimpleNamespace(paths=path, path_len=torch.tensor([3])),
    )
    cfg.path_spacing_m = 2.0  # one dot per stored point
    frame = draw_command(np.zeros((H, W, 3), np.uint8), term, 0, _straight_down(12.0), FOCAL, APERTURE)

    np.testing.assert_array_equal(_pixel(frame, -2.0, -1.0), _rgb(cfg.subgoal_sphere_visualizer_cfg))
    for x in (0.0, 2.0, 4.0):
        np.testing.assert_array_equal(_pixel(frame, x, 0.0), _rgb(cfg.path_points_visualizer_cfg))
    assert not _pixel(frame, 6.0, 0.0).any(), "a point past the path length was drawn"


@pytest.mark.parametrize("dist, key", [(0.5, "goal_close"), (2.0, "goal_mid"), (5.0, "goal_far")])
def test_the_goal_term_colours_its_disc_by_distance(dist, key):
    cfg = GoalCommandCfg(asset_name="robot")
    term = SimpleNamespace(
        cfg=cfg,
        command=torch.tensor([[1.0, 1.0, 0.0]]),
        metrics={"distance_to_goal": torch.tensor([dist])},
    )
    frame = draw_command(np.zeros((H, W, 3), np.uint8), term, 0, _straight_down(12.0), FOCAL, APERTURE)
    np.testing.assert_array_equal(_pixel(frame, 1.0, 1.0), _rgb(cfg.goal_sphere_visualizer_cfg, key))


def _drawn_near(frame, x, y, rgb, height_m=12.0):
    """Whether ``rgb`` is drawn within a pixel of a world point: antialiased lines blend at their rims."""
    (u, v), = project([(x, y, 0.0)], _straight_down(height_m), FOCAL, APERTURE, W, H)[0]
    u, v = int(round(u)), int(round(v))
    return (frame[v - 1 : v + 2, u - 1 : u + 2] == np.array(rgb)).all(axis=-1).any()


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2])
def test_the_robot_is_outlined_at_its_pose_with_a_line_to_its_front(yaw):
    x, y = 1.0, -0.5
    frame = draw_robot(np.zeros((H, W, 3), np.uint8), x, y, 0.0, yaw, _straight_down(12.0), FOCAL, APERTURE)
    fwd, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])

    def at(a, b):
        return (x, y) + a * fwd + b * left

    for corner in (at(CHASSIS_LENGTH / 2, CHASSIS_WIDTH / 2), at(-CHASSIS_LENGTH / 2, -CHASSIS_WIDTH / 2)):
        assert _drawn_near(frame, *corner, ROBOT_RGB)
    assert _drawn_near(frame, *at(0.12, 0.0), ROBOT_RGB), "no line from the centre toward the front"
    assert not _drawn_near(frame, *at(-0.12, 0.0), ROBOT_RGB), "the line points backward"
    assert not _drawn_near(frame, *at(0.0, 0.1), ROBOT_RGB), "the footprint is filled"


def test_a_robot_behind_the_camera_is_not_drawn():
    frame = draw_robot(np.zeros((H, W, 3), np.uint8), 0.0, 0.0, 20.0, 0.0, _straight_down(12.0), FOCAL, APERTURE)
    assert not frame.any()
