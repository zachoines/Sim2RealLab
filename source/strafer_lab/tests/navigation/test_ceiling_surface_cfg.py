"""Tests for the enriched scenes' one-way ceiling surface.

The ceiling is a single open quad whose one normal faces down, so the robot's
camera underneath renders it and an overhead camera renders through it. Two
things carry that property and neither is visible from a cfg dump: the winding
that decides which way the face points, and the RTX back-face-culling switch
without which the face is drawn from both sides. These pin both, plus the
surface height the tall-object budget is written against, without Isaac Sim.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from strafer_lab.tasks.navigation.ceiling_surface import CeilingSurfaceCfg, quad_points
from strafer_lab.tasks.navigation.strafer_env_cfg import (
    _ENRICH_CEILING_HEIGHT_RANGE,
    _ENRICH_CEILING_SURFACE_OFFSET,
    _ENRICH_RENDER_FACE_CULLING_CARB,
    _ENRICH_TALL_OBJECT_HEIGHTS,
    _apply_enrich_render_face_culling,
)


def _stub_cfg(carb_settings=None):
    render = SimpleNamespace(carb_settings=carb_settings)
    return SimpleNamespace(sim=SimpleNamespace(render=render))


def _scene_source_kind(variant_cls):
    """Read a composed variant's scene source without constructing it."""
    for f in dataclasses.fields(variant_cls):
        if f.name == "scene_source":
            return f.default_factory().kind
    return None


def _face_normal(points):
    p = np.asarray(points, dtype=np.float64)
    n = np.cross(p[1] - p[0], p[2] - p[1])
    return n / np.linalg.norm(n)


# ---------------------------------------------------------------------------
# Geometry: the single face has to point down, at the enclosure's own height
# ---------------------------------------------------------------------------


def test_quad_winding_faces_down():
    # A flipped winding renders the ceiling to the camera above and hides it
    # from the robot below — the exact inverse of what the scene needs.
    assert _face_normal(quad_points((7.6, 7.6), -0.05)) == pytest.approx([0.0, 0.0, -1.0])


def test_quad_spans_the_requested_extent_at_one_height():
    points = np.asarray(quad_points((7.6, 4.0), -0.05))
    assert points[:, 0].min() == pytest.approx(-3.8)
    assert points[:, 0].max() == pytest.approx(3.8)
    assert points[:, 1].min() == pytest.approx(-2.0)
    assert points[:, 1].max() == pytest.approx(2.0)
    assert points[:, 2] == pytest.approx(-0.05)


def test_lowest_surface_clears_the_tallest_object():
    lowest = _ENRICH_CEILING_HEIGHT_RANGE[0] + _ENRICH_CEILING_SURFACE_OFFSET
    assert lowest == pytest.approx(2.15)
    assert max(_ENRICH_TALL_OBJECT_HEIGHTS.values()) < lowest


def test_spawn_cfg_has_no_collider():
    # Nothing ever contacts the surface, and a zero-thickness quad has no
    # collision approximation to offer; the mass is stated instead of derived.
    cfg = CeilingSurfaceCfg(size=(7.6, 7.6))
    assert cfg.collision_props is None
    assert cfg.surface_offset == 0.0


# ---------------------------------------------------------------------------
# Render: culling is what makes the single face one-way
# ---------------------------------------------------------------------------


def test_constant_enables_back_face_culling():
    assert _ENRICH_RENDER_FACE_CULLING_CARB["rtx.hydra.faceCulling.enabled"] is True


def test_applies_to_empty_render_cfg():
    cfg = _stub_cfg(None)
    _apply_enrich_render_face_culling(cfg)
    assert cfg.sim.render.carb_settings["rtx.hydra.faceCulling.enabled"] is True


def test_merges_without_clobbering_existing():
    cfg = _stub_cfg({"rtx.post.histogram.enabled": True})
    _apply_enrich_render_face_culling(cfg)
    cs = cfg.sim.render.carb_settings
    assert cs["rtx.post.histogram.enabled"] is True          # preserved
    assert cs["rtx.hydra.faceCulling.enabled"] is True       # added


# ---------------------------------------------------------------------------
# Composition: the switch follows the geometry, on every variant
# ---------------------------------------------------------------------------


def test_culling_is_set_exactly_where_the_ceiling_is():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    # ProcRoom only: the Infinigen variants bind a generated scene USD that a
    # checkout need not carry, and none of them has a ceiling entity.
    variants = [
        cls
        for cls in (getattr(composed, n) for n in dir(composed) if n.startswith("StraferNavCfg_"))
        if _scene_source_kind(cls) == "procroom"
    ]
    assert variants
    for variant in variants:
        cfg = variant()
        culled = (cfg.sim.render.carb_settings or {}).get(
            "rtx.hydra.faceCulling.enabled", False
        )
        assert culled is hasattr(cfg.scene, "ceiling"), variant.__name__
