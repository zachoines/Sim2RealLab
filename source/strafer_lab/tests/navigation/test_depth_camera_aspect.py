# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Standing geometry gate: the policy and perception cameras must share an aspect ratio.

Isaac Sim derives a camera's vertical FOV from its resolution aspect ratio
(square pixels), not from vertical_aperture. Deployment block-averages the
perception stream onto the policy grid, so train/deploy vertical-FOV parity holds
iff the two cameras' aspect ratios match — this assert is that gate.
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
    """Exact-fraction aspect parity (not a float compare)."""
    policy_aspect = Fraction(DEPTH_WIDTH, DEPTH_HEIGHT)
    perception_aspect = Fraction(PERCEPTION_WIDTH, PERCEPTION_HEIGHT)
    assert policy_aspect == perception_aspect, (
        f"policy {DEPTH_WIDTH}x{DEPTH_HEIGHT} (={policy_aspect}) and perception "
        f"{PERCEPTION_WIDTH}x{PERCEPTION_HEIGHT} (={perception_aspect}) aspect ratios "
        "differ: the policy render would span a different vertical FOV than the "
        "sensor stream deployment feeds it. Keep both aspect ratios equal."
    )
