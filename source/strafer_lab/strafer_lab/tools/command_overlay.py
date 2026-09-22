"""Draw the navigation command onto recorded video frames.

The command terms create no scene geometry for debug visualisation: a prim in the stage
is rendered by every camera in it, the D555 included. Recorded video draws the goal,
the rolling subgoal and the planned path here instead, onto the finished frame, in the
styles the command cfgs carry.

The drawing is a pure function of the command state and the recording camera
(:func:`draw_command`); :class:`CommandOverlay` is the ``gym`` wrapper that applies it
to every frame ``RecordVideo`` captures, placed inside ``RecordVideo``.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

_PATH_DOT_SPACING_M = 0.2
_HEADING_TICK_M = 0.3


def project(points_w, cam_to_world, focal_length, horizontal_aperture, width, height):
    """Pixel coordinates and depth of world points seen by a USD camera.

    ``cam_to_world`` is the camera prim's 4x4 local-to-world matrix in USD's row-vector
    convention; the camera looks down its local -Z with +Y up. Pixels are square, so one
    focal length in pixels serves both axes. Returns ``(uv, depth, focal_px)``;
    ``depth <= 0`` is behind the camera.
    """
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = homo @ np.linalg.inv(np.asarray(cam_to_world, dtype=np.float64))
    depth = -cam[:, 2]
    f_px = width * focal_length / horizontal_aperture
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2.0 + f_px * cam[:, 0] / depth
        v = height / 2.0 - f_px * cam[:, 1] / depth
    return np.stack([u, v], axis=1), depth, f_px


def _style(markers_cfg, key=None):
    """(radius m, RGB 0-255) of one prototype in a marker cfg."""
    proto = markers_cfg.markers[key] if key else next(iter(markers_cfg.markers.values()))
    rgb = tuple(int(round(255 * c)) for c in proto.visual_material.diffuse_color)
    return getattr(proto, "radius", 0.0), rgb


def draw_command(frame, term, env_index, cam_to_world, focal_length, horizontal_aperture):
    """Draw one env's command onto an RGB frame and return it.

    Handles both command families by what their cfg carries: the goal term (a disc
    coloured by distance) and the subgoal term (the rolling subgoal and the path).
    """
    import cv2  # noqa: WPS433 — heavy optional dep, imported lazily

    height, width = frame.shape[:2]
    cfg = term.cfg
    x, y, heading = (float(v) for v in term.command[env_index, :3])

    def disc(points, radius_m, rgb):
        uv, depth, f_px = project(points, cam_to_world, focal_length, horizontal_aperture, width, height)
        for (u, v), d in zip(uv, depth):
            if d > 0 and 0 <= u < width and 0 <= v < height:
                r = max(2, int(round(f_px * radius_m / d)))
                cv2.circle(frame, (int(round(u)), int(round(v))), r, rgb, -1, cv2.LINE_AA)

    def tick(start_m, rgb):
        c, s = math.cos(heading), math.sin(heading)
        ends = [(x + r * c, y + r * s, 0.0) for r in (start_m, start_m + _HEADING_TICK_M)]
        uv, depth, _ = project(ends, cam_to_world, focal_length, horizontal_aperture, width, height)
        if (depth > 0).all():
            a, b = (tuple(int(round(c)) for c in p) for p in uv)
            cv2.line(frame, a, b, rgb, 2, cv2.LINE_AA)

    if hasattr(cfg, "path_points_visualizer_cfg"):
        path = term.path_cursor
        stride = max(1, int(round(_PATH_DOT_SPACING_M / max(cfg.path_spacing_m, 1e-3))))
        pts = path.paths[env_index, : int(path.path_len[env_index]) : stride].detach().cpu().numpy()
        radius, rgb = _style(cfg.path_points_visualizer_cfg)
        disc(np.column_stack([pts, np.zeros(len(pts))]), radius, rgb)
        radius, rgb = _style(cfg.subgoal_sphere_visualizer_cfg)
        disc([(x, y, 0.0)], radius, rgb)
        tick(radius, _style(cfg.subgoal_heading_visualizer_cfg)[1])
    elif hasattr(cfg, "goal_sphere_visualizer_cfg"):
        dist = float(term.metrics["distance_to_goal"][env_index])
        key = "goal_close" if dist <= 1.0 else "goal_mid" if dist <= 3.0 else "goal_far"
        radius, rgb = _style(cfg.goal_sphere_visualizer_cfg, key)
        disc([(x, y, 0.0)], radius, rgb)
        tick(radius, _style(cfg.goal_heading_visualizer_cfg)[1])
    return frame


class CommandOverlay(gym.Wrapper):
    """Draws the viewed env's ``goal_command`` onto every rendered frame.

    Reads the recording camera from the stage on each frame, so the overlay follows
    whatever pose the recording script anchored.
    """

    def render(self):
        frame = self.env.render()
        if frame is None:
            return frame
        from pxr import Usd, UsdGeom

        env = self.env.unwrapped
        prim = env.sim.stage.GetPrimAtPath(env.cfg.viewer.cam_prim_path)
        camera = UsdGeom.Camera(prim)
        cam_to_world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        term = env.command_manager.get_term("goal_command")
        return draw_command(
            np.ascontiguousarray(frame),
            term,
            env.cfg.viewer.env_index,
            cam_to_world,
            camera.GetFocalLengthAttr().Get(),
            camera.GetHorizontalApertureAttr().Get(),
        )
