"""Strip Infinigen stage-shot authoring cameras from the scene USDC.

Infinigen's ``generate_indoors`` pipeline authors one ``camera_<room>_<index>``
per room (plus an inner same-named twin) for its own offline Cycles render
passes. Isaac Sim binds no render product to them, so removing them is a
cleanliness pass — it declutters ``usdview`` / Composer but is NOT a perf
lever (the operator confirmed disabling them in the editor produced no FPS
change). These tests lock the matcher and the strip behaviour.

Runs without Isaac Sim — needs ``pxr`` (present in env_isaaclab3).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest


pytest.importorskip("pxr")


_POSTPROCESS_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "postprocess_scene_usd.py"
)
_spec = importlib.util.spec_from_file_location(
    "postprocess_scene_usd", _POSTPROCESS_PATH,
)
postprocess_scene_usd = importlib.util.module_from_spec(_spec)
sys.modules["postprocess_scene_usd"] = postprocess_scene_usd
_spec.loader.exec_module(postprocess_scene_usd)


# The four camera prims the operator identified in seed=1's USDC (FAQ → B):
# an outer Xform-camera + inner same-named twin, one pair per room.
_OPERATOR_CAMERA_PATHS = (
    "/World/Room/camera_0_0",
    "/World/Room/camera_0_0/camera_0_0",
    "/World/Room/camera_0_1",
    "/World/Room/camera_0_1/camera_0_1",
)

# Prims that must SURVIVE the strip.
_DECOY_PATHS = (
    "/World/bedroom_0_0_wall",                 # structural mesh
    "/World/Room/d555_camera_perception",      # robot-style camera name
    "/World/some_other_camera",                # camera outside /World/Room
    "/World/Room/camera_table",                # 'camera' substring, not the pattern
)


def _make_stage_with_stage_cameras(tmp_path: Path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(tmp_path / "stage_cams.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Room")
    for path in _OPERATOR_CAMERA_PATHS:
        UsdGeom.Camera.Define(stage, path)
    UsdGeom.Mesh.Define(stage, "/World/bedroom_0_0_wall")
    UsdGeom.Camera.Define(stage, "/World/Room/d555_camera_perception")
    UsdGeom.Camera.Define(stage, "/World/some_other_camera")
    UsdGeom.Xform.Define(stage, "/World/Room/camera_table")
    return stage


def _compiled_default():
    return re.compile(postprocess_scene_usd._DEFAULT_STAGE_CAMERA_PRIM_PATTERN)


class TestStageCameraPattern:
    def test_default_matches_operator_cameras(self):
        pat = _compiled_default()
        for path in _OPERATOR_CAMERA_PATHS:
            assert pat.match(path), f"expected match: {path}"

    def test_default_rejects_decoys(self):
        pat = _compiled_default()
        for path in _DECOY_PATHS:
            assert not pat.match(path), f"expected no-match: {path}"

    def test_pattern_anchors_room_parent(self):
        """The matcher is rooted at /World/Room, not anywhere a camera lives."""
        pat = _compiled_default()
        assert not pat.match("/World/camera_0_0")
        assert not pat.match("/World/Kitchen/camera_0_0")


class TestStripStageCameras:
    def test_removes_the_four_operator_cameras(self, tmp_path):
        stage = _make_stage_with_stage_cameras(tmp_path)
        removed = postprocess_scene_usd.strip_stage_cameras(
            stage, _compiled_default(),
        )
        # Matches the acceptance bullet: removes the four cameras the operator
        # identified.
        assert removed == 4
        for path in _OPERATOR_CAMERA_PATHS:
            assert not stage.GetPrimAtPath(path).IsValid(), f"still present: {path}"

    def test_decoys_survive(self, tmp_path):
        stage = _make_stage_with_stage_cameras(tmp_path)
        postprocess_scene_usd.strip_stage_cameras(stage, _compiled_default())
        for path in _DECOY_PATHS:
            assert stage.GetPrimAtPath(path).IsValid(), f"wrongly removed: {path}"

    def test_idempotent_rerun_reports_zero(self, tmp_path):
        stage = _make_stage_with_stage_cameras(tmp_path)
        postprocess_scene_usd.strip_stage_cameras(stage, _compiled_default())
        again = postprocess_scene_usd.strip_stage_cameras(stage, _compiled_default())
        assert again == 0

    def test_empty_stage_is_zero(self, tmp_path):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateNew(str(tmp_path / "empty.usda"))
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Room")
        assert postprocess_scene_usd.strip_stage_cameras(
            stage, _compiled_default(),
        ) == 0


class TestStripStageCamerasCLI:
    def test_help_surfaces_flags_and_not_a_perf_lever_note(self, capsys):
        """--help must surface the stage-camera flags and make clear the
        strip is a cleanliness pass, not a perf lever — so a future operator
        doesn't mistake it for one (per the brief's cleanup acceptance)."""
        with pytest.raises(SystemExit):
            postprocess_scene_usd.main(["--usdc", "/nonexistent.usdc", "--help"])
        out = capsys.readouterr().out
        assert "--stage-camera-prim-pattern" in out
        assert "--keep-stage-cameras" in out
        assert "NOT A PERF LEVER" in out.upper()
