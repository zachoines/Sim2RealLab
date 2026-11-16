# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The perception camera's opencvPinhole stamp takes its intrinsics off the camera's cfg.

``stamp_d555_perception_opencv_pinhole`` authors the lens attributes the ROS 2
bridge's camera-info helper reads. Its defaults are the camera cfg's own
resolution and lens, so a composition that changes the camera's focal length
stamps the intrinsics the camera renders at. Driven against an in-memory USD
stage with two env prims; no Kit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pxr import Usd, UsdGeom

from strafer_lab.tasks.navigation.d555_cfg import make_d555_perception_camera_cfg
from strafer_lab.tasks.navigation.mdp.events import stamp_d555_perception_opencv_pinhole
from strafer_shared.constants import D555_MEASURED_FOCAL_LENGTH_MM, D555_MEASURED_FX_PX

_ATTR = "d555_camera_perception"
_PREFIX = "omni:lensdistortion:opencvPinhole:"


def _env(cam_cfg, num_envs=2):
    stage = Usd.Stage.CreateInMemory()
    template = make_d555_perception_camera_cfg().prim_path
    paths = [template.replace("{ENV_REGEX_NS}", f"/World/envs/env_{i}") for i in range(num_envs)]
    for path in paths:
        UsdGeom.Camera.Define(stage, path)
    env = SimpleNamespace(
        scene=SimpleNamespace(cfg=SimpleNamespace(**{_ATTR: cam_cfg})),
        sim=SimpleNamespace(stage=stage),
    )
    return env, [stage.GetPrimAtPath(p) for p in paths]


def _stamped(prim):
    get = lambda name: prim.GetAttribute(_PREFIX + name).Get()  # noqa: E731
    size = get("imageSize")
    return {
        "fx": get("fx"), "fy": get("fy"), "cx": get("cx"), "cy": get("cy"),
        "size": None if size is None else (size[0], size[1]),
    }


def test_the_default_is_the_camera_cfgs_lens():
    """A camera whose cfg carries a focal length no constant default holds is
    stamped at that focal length, on every env's prim."""
    cam = make_d555_perception_camera_cfg()
    cam.spawn.focal_length = D555_MEASURED_FOCAL_LENGTH_MM
    env, prims = _env(cam)
    stamp_d555_perception_opencv_pinhole(env, None)
    for prim in prims:
        got = _stamped(prim)
        assert got["fx"] == pytest.approx(D555_MEASURED_FX_PX, rel=1e-6)
        assert got["fy"] == pytest.approx(D555_MEASURED_FX_PX, rel=1e-6)
        assert (got["cx"], got["cy"]) == (cam.width / 2.0, cam.height / 2.0)
        assert got["size"] == (cam.width, cam.height)


def test_the_factory_camera_is_stamped_at_the_factory_lens():
    cam = make_d555_perception_camera_cfg()
    env, prims = _env(cam)
    stamp_d555_perception_opencv_pinhole(env, None)
    want = cam.width * cam.spawn.focal_length / cam.spawn.horizontal_aperture
    for prim in prims:
        assert _stamped(prim)["fx"] == pytest.approx(want, rel=1e-6)


def test_explicit_arguments_override_the_cfg():
    cam = make_d555_perception_camera_cfg()
    env, prims = _env(cam)
    stamp_d555_perception_opencv_pinhole(
        env, None, width=320, height=180, focal_length_mm=2.0, horizontal_aperture_mm=4.0
    )
    for prim in prims:
        got = _stamped(prim)
        assert got["fx"] == pytest.approx(320 * 2.0 / 4.0, rel=1e-6)
        assert (got["cx"], got["cy"], got["size"]) == (160.0, 90.0, (320, 180))


def test_a_pruned_camera_is_skipped():
    """A stack without the perception camera leaves it ``None`` on the scene cfg;
    the stamp authors nothing."""
    env, prims = _env(None)
    stamp_d555_perception_opencv_pinhole(env, None)
    for prim in prims:
        assert not prim.GetAttribute(_PREFIX + "fx").IsValid()
