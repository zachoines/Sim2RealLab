"""Tests for the RTX Real-Time 2.0 renderer A/B selection.

Pins the carb tokens the bridge flips (RT 2.0 = ``RealTimePathTracing`` in the
Performance preset), the safety-critical invariant that DLSS frame generation is
never enabled on the sensor path, and that the launch-arg applier mutates only
the renderer knobs — never the camera resolution / intrinsics contract. Runs
without launching Isaac Sim (SimpleNamespace stubs, same pattern as the
sim-in-the-loop termination tests).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strafer_lab.bridge import renderer_settings as rs


def _import_rsil():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_sim_in_the_loop as rsil

    return rsil


# --- pure renderer_settings ------------------------------------------------


def test_rt2_selects_realtime_path_tracing_and_performance():
    overrides = rs.renderer_kit_setting_overrides(rs.RENDERER_RT2)
    assert overrides["/rtx/rendermode"] == "RealTimePathTracing"
    # RT 2.0 must be registered at boot, not just selected.
    assert overrides["/rtx-transient/rt2Enabled"] is True
    assert overrides["/persistent/rtx/modes/rt2/enabled"] is True
    assert rs.renderer_rendering_mode(rs.RENDERER_RT2) == "performance"


def test_legacy_is_a_no_op_revert():
    assert rs.renderer_kit_setting_overrides(rs.RENDERER_LEGACY) == {}
    assert rs.renderer_rendering_mode(rs.RENDERER_LEGACY) is None


def test_frame_generation_is_never_enabled():
    """Safety invariant: no renderer choice turns DLSS frame generation on.

    DLSS-G is a present-path feature unsupported on the perception tiled camera;
    enabling it would risk the depth-policy input. Every choice must keep it off.
    """
    for renderer in rs.RENDERER_CHOICES:
        overrides = rs.renderer_kit_setting_overrides(renderer)
        assert overrides.get("/rtx-transient/dlssg/enabled", False) is False
    # The default explicitly pins it off (not merely inherited from a preset).
    assert rs.renderer_kit_setting_overrides(rs.RENDERER_RT2)["/rtx-transient/dlssg/enabled"] is False


def test_format_kit_setting_args_lowercases_bools():
    args = rs.format_kit_setting_args(
        {"/rtx/rendermode": "RealTimePathTracing", "/rtx-transient/rt2Enabled": True,
         "/rtx-transient/dlssg/enabled": False}
    )
    assert "--/rtx/rendermode=RealTimePathTracing" in args
    assert "--/rtx-transient/rt2Enabled=true" in args
    assert "--/rtx-transient/dlssg/enabled=false" in args


def test_default_renderer_is_rt2():
    assert rs.DEFAULT_RENDERER == rs.RENDERER_RT2


def test_describe_mentions_the_active_mode():
    assert "RealTimePathTracing" in rs.describe_renderer(rs.RENDERER_RT2)
    assert "RaytracedLighting" in rs.describe_renderer(rs.RENDERER_LEGACY)


def test_unknown_renderer_raises():
    with pytest.raises(ValueError):
        rs.renderer_kit_setting_overrides("interactive")


# --- launch-arg applier (run_sim_in_the_loop._apply_renderer_boot_args) ----


def _args(renderer, *, rendering_mode=None, kit_args=""):
    # ``dest="rtx_renderer"`` on the --renderer flag avoids the reserved
    # SimulationApp ``renderer`` config key.
    return SimpleNamespace(rtx_renderer=renderer, rendering_mode=rendering_mode, kit_args=kit_args)


def test_renderer_flag_dest_avoids_reserved_simapp_key(monkeypatch):
    """--renderer must NOT land on the namespace under the key ``renderer``.

    AppLauncher forwards every namespace key matching a reserved SimulationApp
    config name (``renderer`` is one) straight to Isaac Sim. A bridge value like
    'rt2' is not a valid SimApp renderer, so the flag uses ``dest=rtx_renderer``.
    """
    rsil = _import_rsil()
    monkeypatch.setattr(sys, "argv", ["run_sim_in_the_loop.py"])
    args = rsil._parse_args()
    assert args.rtx_renderer == rs.DEFAULT_RENDERER
    assert not hasattr(args, "renderer")


def test_apply_boot_args_rt2_injects_settings_and_preset():
    rsil = _import_rsil()
    args = _args(rs.RENDERER_RT2)
    rsil._apply_renderer_boot_args(args)
    assert args.rendering_mode == "performance"
    assert "--/rtx/rendermode=RealTimePathTracing" in args.kit_args
    assert "--/rtx-transient/dlssg/enabled=false" in args.kit_args


def test_apply_boot_args_legacy_leaves_args_untouched():
    rsil = _import_rsil()
    args = _args(rs.RENDERER_LEGACY, kit_args="--foo=1")
    rsil._apply_renderer_boot_args(args)
    assert args.rendering_mode is None
    assert args.kit_args == "--foo=1"


def test_apply_boot_args_respects_explicit_rendering_mode():
    rsil = _import_rsil()
    args = _args(rs.RENDERER_RT2, rendering_mode="quality")
    rsil._apply_renderer_boot_args(args)
    assert args.rendering_mode == "quality"  # operator choice wins


def test_apply_boot_args_appends_to_existing_kit_args():
    rsil = _import_rsil()
    args = _args(rs.RENDERER_RT2, kit_args="--ext-folder=/x")
    rsil._apply_renderer_boot_args(args)
    assert args.kit_args.startswith("--ext-folder=/x ")
    assert "--/rtx/rendermode=RealTimePathTracing" in args.kit_args


# --- camera-contract invariant (RT 2.0 must not change resolution) ---------


def test_bridge_publishes_native_perception_resolution():
    """The RT 2.0 flip must not touch the published resolution / intrinsics.

    The renderer selection only injects RTX launch settings; the perception
    stream resolution is fixed to the real D555 native rate. Pin it so a future
    change that couples the two trips this test.
    """
    from strafer_lab.bridge.config import build_default_bridge_config
    from strafer_shared.constants import PERCEPTION_HEIGHT, PERCEPTION_WIDTH

    cfg = build_default_bridge_config()
    for cam in (cfg.color_camera, cfg.depth_camera):
        assert (cam.width, cam.height) == (PERCEPTION_WIDTH, PERCEPTION_HEIGHT)


def test_apply_boot_args_touches_only_renderer_knobs():
    """The applier mutates rendering_mode + kit_args only — nothing scene-side."""
    rsil = _import_rsil()
    args = SimpleNamespace(
        rtx_renderer=rs.RENDERER_RT2, rendering_mode=None, kit_args="",
        scene_usd="untouched", task="untouched", decimation=1,
    )
    rsil._apply_renderer_boot_args(args)
    assert args.scene_usd == "untouched"
    assert args.task == "untouched"
    assert args.decimation == 1
