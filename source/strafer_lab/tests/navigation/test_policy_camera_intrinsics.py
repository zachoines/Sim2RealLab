# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The policy camera's focal length, per composed variant.

The composition goldens cannot see it: the contract serializer takes only
``num_envs`` and ``env_spacing`` off the scene, so a focal-length change moves
no hash. The v4 variants render the real unit's measured focal length
(fx = fy = 321.522 px; the principal point stays at the image centre) by
overriding it in their composition; every other variant,
the enriched pair v4 is composed from included, keeps the shared constant.
"""

from __future__ import annotations

import dataclasses

import pytest

from strafer_shared.constants import (
    D555_FOCAL_LENGTH_MM,
    D555_HORIZONTAL_APERTURE_MM,
    PERCEPTION_WIDTH,
)

_V4 = ("StraferNavCfg_RLDepthSubgoalEnrichedV4_Robust", "StraferNavCfg_RLDepthSubgoalEnrichedV4_Robust_PLAY")
_V3 = ("StraferNavCfg_RLDepthSubgoalEnriched_Robust", "StraferNavCfg_RLDepthSubgoalEnriched_Robust_PLAY")


def _policy_camera(variant):
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    return getattr(composed, variant)().scene.d555_camera


@pytest.mark.parametrize("variant", _V4)
def test_v4_renders_the_measured_intrinsics(variant):
    cam = _policy_camera(variant)
    assert cam.spawn.focal_length == 1.8487515, f"{variant} renders {cam.spawn.focal_length} mm"
    fx = PERCEPTION_WIDTH * cam.spawn.focal_length / cam.spawn.horizontal_aperture
    assert fx == pytest.approx(321.522, abs=1e-6), f"{variant} renders fx = {fx} px"


@pytest.mark.parametrize("variant", _V3)
def test_v3_keeps_the_shared_constant(variant):
    cam = _policy_camera(variant)
    assert cam.spawn.focal_length == 1.93 == D555_FOCAL_LENGTH_MM, (
        f"{variant} renders {cam.spawn.focal_length} mm: the v4 override reached the IDs it is composed from"
    )


def _scene_source_kind(variant_cls):
    """The variant's scene source, read off the class: constructing an
    Infinigen-source variant reads the machine's scene corpus."""
    for field in dataclasses.fields(variant_cls):
        if field.name == "scene_source":
            return field.default_factory().kind
    return None


def test_no_other_variant_moves_the_focal_length():
    """Only the v4 variants override it; the aperture is shared by all. The
    Infinigen-source capture variants are left out (they read the scene corpus
    at construction); they compose the same two camera factories."""
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
            want = 1.8487515 if (name in _V4 and attr == "d555_camera") else D555_FOCAL_LENGTH_MM
            assert cam.spawn.focal_length == want, f"{name}.{attr} renders {cam.spawn.focal_length} mm (want {want})"
    assert seen > 15, "the camera sweep collapsed"
