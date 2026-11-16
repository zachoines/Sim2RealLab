# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""A stack whose policy reads a camera re-renders once after each reset; every other stack keeps Isaac Lab's 0.

Without the re-render, an env's first observation after a reset is the frame rendered before it. The composition
contract does not hash ``num_rerenders_on_reset``, so this test pins it on every composed variant.
"""

from __future__ import annotations

import dataclasses


def test_camera_profiles_rerender_once_after_a_reset():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    seen = {1: [], 0: []}
    for name in sorted(n for n in dir(composed) if n.startswith("StraferNavCfg_")):
        cls = getattr(composed, name)
        kind = [f.default_factory().kind for f in dataclasses.fields(cls) if f.name == "scene_source"]
        if kind == ["infinigen"]:
            continue  # reads the scene corpus at construction; composed by the same root
        cfg = cls()
        want = 1 if cfg.sensors.obs_profile() in ("depth", "full") else 0
        assert cfg.num_rerenders_on_reset == want, f"{name}: {cfg.num_rerenders_on_reset}"
        seen[want].append(name)
    assert len(seen[1]) >= 8 and len(seen[0]) >= 6, seen
    assert "StraferNavCfg_RLDepthSubgoal_Robust" in seen[1] and "StraferNavCfg_RLNoCam" in seen[0]
