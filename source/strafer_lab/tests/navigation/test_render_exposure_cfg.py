"""Tests for the Infinigen render-exposure carb-settings fix.

Infinigen scenes bake their own physically-based HDR emitters into the USDC;
Kit's RTX renderer leaves auto-exposure off, so the recorded perception RGB
clips to white. The fix enables RTX histogram auto-exposure corpus-wide on the
sim's ``RenderCfg.carb_settings``. These tests pin the constant and the merge
behavior without launching Isaac Sim.
"""
from __future__ import annotations

from types import SimpleNamespace

from strafer_lab.tasks.navigation.strafer_env_cfg import (
    _INFINIGEN_RENDER_EXPOSURE_CARB,
    _apply_infinigen_render_exposure,
)


def _stub_cfg(carb_settings=None):
    render = SimpleNamespace(carb_settings=carb_settings)
    return SimpleNamespace(sim=SimpleNamespace(render=render))


def test_constant_enables_auto_exposure():
    assert _INFINIGEN_RENDER_EXPOSURE_CARB["rtx.post.histogram.enabled"] is True
    # whiteScale is the exposure target; lower => darker. Shipped below the
    # Kit default (10) as the operator's starting point.
    assert _INFINIGEN_RENDER_EXPOSURE_CARB["rtx.post.histogram.whiteScale"] < 10.0


def test_applies_to_empty_render_cfg():
    cfg = _stub_cfg(None)
    _apply_infinigen_render_exposure(cfg)
    assert cfg.sim.render.carb_settings["rtx.post.histogram.enabled"] is True


def test_merges_without_clobbering_existing():
    cfg = _stub_cfg({"rtx.sceneDb.ambientLightIntensity": 3.0})
    _apply_infinigen_render_exposure(cfg)
    cs = cfg.sim.render.carb_settings
    assert cs["rtx.sceneDb.ambientLightIntensity"] == 3.0  # preserved
    assert cs["rtx.post.histogram.enabled"] is True         # added
