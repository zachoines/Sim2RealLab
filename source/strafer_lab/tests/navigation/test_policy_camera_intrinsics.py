# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The focal length every composed camera renders, and the measured one.

The composition goldens cannot see a camera: the contract serializer takes only
``num_envs`` and ``env_spacing`` off the scene, so a focal-length change moves
no hash. Every composed variant renders the datasheet's
``D555_FOCAL_LENGTH_MM``; the real unit's measured focal length
(fx = fy = 321.522 px at 640x360) is ``D555_MEASURED_FOCAL_LENGTH_MM``.
"""

from __future__ import annotations

import dataclasses

import pytest

from strafer_shared.constants import (
    D555_FOCAL_LENGTH_MM,
    D555_HORIZONTAL_APERTURE_MM,
    D555_MEASURED_FOCAL_LENGTH_MM,
    D555_MEASURED_FX_PX,
    PERCEPTION_WIDTH,
)


def test_the_measured_constant_is_the_measurement():
    """The shared constants hold the measured fx and its focal length over the
    datasheet aperture; the datasheet focal length is unchanged."""
    assert D555_MEASURED_FX_PX == 321.522
    assert D555_MEASURED_FOCAL_LENGTH_MM == 1.8487515
    assert PERCEPTION_WIDTH * D555_MEASURED_FOCAL_LENGTH_MM / D555_HORIZONTAL_APERTURE_MM == pytest.approx(
        D555_MEASURED_FX_PX, abs=1e-9
    )
    assert D555_FOCAL_LENGTH_MM == 1.93


def _scene_source_kind(variant_cls):
    """The variant's scene source, read off the class: constructing an
    Infinigen-source variant reads the machine's scene corpus."""
    for field in dataclasses.fields(variant_cls):
        if field.name == "scene_source":
            return field.default_factory().kind
    return None


def test_every_composed_camera_renders_the_datasheet_lens():
    """No composition moves a camera's focal length or aperture off the
    factories'. The Infinigen-source capture variants are left out (they read
    the scene corpus at construction); they compose the same two factories."""
    from isaaclab.sensors import CameraCfg

    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    seen = 0
    for name in sorted(n for n in dir(composed) if n.startswith("StraferNavCfg_")):
        cls = getattr(composed, name)
        if _scene_source_kind(cls) == "infinigen":
            continue
        scene = cls().scene
        for attr in ("d555_camera", "d555_camera_perception"):
            cam = getattr(scene, attr, None)
            if not isinstance(cam, CameraCfg):
                continue
            seen += 1
            assert cam.spawn.horizontal_aperture == D555_HORIZONTAL_APERTURE_MM, f"{name}.{attr}"
            assert cam.spawn.focal_length == D555_FOCAL_LENGTH_MM, f"{name}.{attr} renders {cam.spawn.focal_length} mm"
    assert seen > 15, "the camera sweep collapsed"
