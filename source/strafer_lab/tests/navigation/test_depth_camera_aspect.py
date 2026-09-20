# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Standing geometry gate: the policy and perception cameras must share an aspect ratio.

Isaac Sim derives a camera's vertical FOV from its resolution aspect ratio
(square pixels), not from vertical_aperture. Deployment block-reduces the
perception stream onto the policy grid, so train/deploy vertical-FOV parity holds
iff the two cameras' aspect ratios match — this assert is that gate.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from strafer_shared.constants import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
)


def test_policy_and_perception_cameras_share_aspect_ratio():
    """Exact-fraction aspect parity (not a float compare)."""
    policy_aspect = Fraction(DEPTH_WIDTH, DEPTH_HEIGHT)
    perception_aspect = Fraction(PERCEPTION_WIDTH, PERCEPTION_HEIGHT)
    assert policy_aspect == perception_aspect, (
        f"policy {DEPTH_WIDTH}x{DEPTH_HEIGHT} (={policy_aspect}) and perception "
        f"{PERCEPTION_WIDTH}x{PERCEPTION_HEIGHT} (={perception_aspect}) aspect ratios "
        "differ: the policy render would span a different vertical FOV than the "
        "sensor stream deployment feeds it. Keep both aspect ratios equal."
    )


# The composition rebuilds the policy camera rather than inheriting the scene
# class's, so the factory is not the seam that decides what is rendered. Every
# composed variant carrying a policy camera is checked against the resolution
# the observation term's reduction is gated on.
_POLICY_CAMERA_VARIANTS = (
    "StraferNavCfg_RLDepth_Real",
    "StraferNavCfg_RLDepth_Robust",
    "StraferNavCfg_RLDepth_Real_PLAY",
    "StraferNavCfg_RLDepth_Robust_PLAY",
    "StraferNavCfg_RLDepthSubgoal_Real",
    "StraferNavCfg_RLDepthSubgoal_Robust",
    "StraferNavCfg_RLDepthSubgoal_Real_PLAY",
    "StraferNavCfg_RLDepthSubgoal_Robust_PLAY",
    "StraferNavCfg_RLDepthEnriched_Real",
    "StraferNavCfg_RLDepthEnriched_Robust",
    "StraferNavCfg_RLDepthEnriched_Real_PLAY",
    "StraferNavCfg_RLDepthEnriched_Robust_PLAY",
    "StraferNavCfg_RLDepthSubgoalEnriched_Real",
    "StraferNavCfg_RLDepthSubgoalEnriched_Robust",
    "StraferNavCfg_RLDepthSubgoalEnriched_Real_PLAY",
    "StraferNavCfg_RLDepthSubgoalEnriched_Robust_PLAY",
)


@pytest.mark.parametrize("variant", _POLICY_CAMERA_VARIANTS)
def test_composed_variants_render_the_reductions_input(variant):
    """The rendered resolution is what the reduction is gated on.

    The composition goldens cannot reach this: the contract serializer takes
    only ``num_envs`` and ``env_spacing`` off the scene, so a camera resolution
    change moves no hash. Without this assertion the render and the reduction
    could drift apart and the term would pass the field through unreduced.
    """
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    cam = getattr(composed, variant)().scene.d555_camera
    assert (cam.height, cam.width) == (PERCEPTION_HEIGHT, PERCEPTION_WIDTH), (
        f"{variant} renders {cam.height}x{cam.width}; the reduction is gated "
        f"on {PERCEPTION_HEIGHT}x{PERCEPTION_WIDTH}"
    )
