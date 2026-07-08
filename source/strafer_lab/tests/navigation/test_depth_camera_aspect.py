# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Standing geometry gate for the depth-camera vertical-FOV fix.

Isaac Sim / RTX renders a camera's vertical FOV from its resolution-derived
square pixels and IGNORES the authored ``vertical_aperture`` (probed 2026-07-06:
setting the aperture changed the USD prim but not the rendered depth). Train↔deploy
vertical-FOV parity therefore holds **iff** the policy camera
(``DEPTH_WIDTH``×``DEPTH_HEIGHT``) and the perception camera
(``PERCEPTION_WIDTH``×``PERCEPTION_HEIGHT``) share an aspect ratio — this assert
IS the geometry gate.
"""

from __future__ import annotations

from fractions import Fraction

from strafer_shared.constants import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
)


def test_policy_and_perception_cameras_share_aspect_ratio():
    """Exact-fraction aspect parity (not float compare).

    Deployment block-averages the perception stream onto the policy grid, so a
    mismatched aspect ratio silently magnifies training depth versus deployment
    depth — the bug this fix removed (80x60 4:3 vs 640x360 16:9 was ~1.26x
    vertical). Both must stay 16:9; ``vertical_aperture`` cannot compensate
    (RTX ignores it — disproven).
    """
    policy_aspect = Fraction(DEPTH_WIDTH, DEPTH_HEIGHT)
    perception_aspect = Fraction(PERCEPTION_WIDTH, PERCEPTION_HEIGHT)
    assert policy_aspect == perception_aspect, (
        f"policy {DEPTH_WIDTH}x{DEPTH_HEIGHT} (={policy_aspect}) and perception "
        f"{PERCEPTION_WIDTH}x{PERCEPTION_HEIGHT} (={perception_aspect}) aspect ratios "
        "differ: the policy render would span a different vertical FOV than the "
        "sensor stream deployment feeds it. Keep both aspect ratios equal."
    )
