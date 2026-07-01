"""Tests for the combined-index capture-readiness gate.

``generate_scenes_metadata.py`` builds ``scenes_metadata.json``. Before this
gate it indexed any scene with floor geometry, so a scene whose embedded
``strafer_scene_metadata`` customData was absent (orphan-export defect) or empty
landed in the index with valid spawn points and read as capture-usable — then
produced episodes with no groundable targets. The gate reads the embedded
customData and skips such scenes.

Two layers:

* ``has_capture_metadata`` — the pure predicate (no ``pxr``); always runs.
* ``_process_scene`` skip behaviour — authors a tiny temp USD with a floor prim
  and absent / empty / populated customData (``pytest.importorskip("pxr")``).

The script lives under ``source/strafer_lab/scripts`` (not an importable
package), so it is loaded by file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from strafer_lab.tools import scene_metadata_reader as smr

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "generate_scenes_metadata.py"
)
_spec = importlib.util.spec_from_file_location("generate_scenes_metadata", _SCRIPT)
generate_scenes_metadata = importlib.util.module_from_spec(_spec)
sys.modules["generate_scenes_metadata"] = generate_scenes_metadata
_spec.loader.exec_module(generate_scenes_metadata)

_has_capture_metadata = generate_scenes_metadata.has_capture_metadata
_process_scene = generate_scenes_metadata._process_scene


# ---------------------------------------------------------------------------
# Pure predicate (no pxr)
# ---------------------------------------------------------------------------


class TestHasCaptureMetadata:
    def test_absent_key_is_not_capture_ready(self):
        assert _has_capture_metadata(None) is False

    def test_empty_objects_is_not_capture_ready(self):
        assert _has_capture_metadata({"objects": [], "rooms": [{}]}) is False

    def test_missing_objects_field_is_not_capture_ready(self):
        # A payload with rooms but no objects key at all (defensive).
        assert _has_capture_metadata({"rooms": [{}]}) is False

    def test_objects_field_none_is_not_capture_ready(self):
        assert _has_capture_metadata({"objects": None}) is False

    def test_populated_objects_is_capture_ready(self):
        assert _has_capture_metadata({"objects": [{"label": "chair"}]}) is True


# ---------------------------------------------------------------------------
# Non-clobbering merge (pure — no pxr). Registering a newly-generated scene
# must not drop existing high_quality_dgx entries whose heavy USDs are absent
# from this run's glob (the multi-GB corpus USDs are not in git).
# ---------------------------------------------------------------------------


_merge_scene_entries = generate_scenes_metadata._merge_scene_entries


class TestMergeSceneEntries:
    def _write_index(self, path: Path, scenes: dict) -> None:
        import json

        path.write_text(json.dumps({"scenes": scenes}))

    def test_merge_preserves_absent_corpus_entries(self, tmp_path):
        out = tmp_path / "scenes_metadata.json"
        self._write_index(out, {"scene_high_quality_dgx_000_seed2": {"floor_top_z": 0.1}})
        new = {"scene_singleroom_000_seed0": {"floor_top_z": 0.0}}
        merged = _merge_scene_entries(out, new, merge=True)
        assert set(merged) == {
            "scene_high_quality_dgx_000_seed2",
            "scene_singleroom_000_seed0",
        }
        # The prior entry survives byte-for-byte.
        assert merged["scene_high_quality_dgx_000_seed2"] == {"floor_top_z": 0.1}

    def test_merge_refreshes_reindexed_scene(self, tmp_path):
        out = tmp_path / "scenes_metadata.json"
        self._write_index(out, {"scene_x_000_seed0": {"floor_top_z": 0.1}})
        merged = _merge_scene_entries(out, {"scene_x_000_seed0": {"floor_top_z": 9.9}}, merge=True)
        assert merged["scene_x_000_seed0"] == {"floor_top_z": 9.9}

    def test_no_merge_overwrites_wholesale(self, tmp_path):
        out = tmp_path / "scenes_metadata.json"
        self._write_index(out, {"scene_old_000_seed0": {"floor_top_z": 0.1}})
        new = {"scene_new_000_seed0": {"floor_top_z": 0.0}}
        assert _merge_scene_entries(out, new, merge=False) == new

    def test_merge_with_no_existing_file_returns_new(self, tmp_path):
        out = tmp_path / "scenes_metadata.json"  # does not exist
        new = {"scene_new_000_seed0": {"floor_top_z": 0.0}}
        assert _merge_scene_entries(out, new, merge=True) == new

    def test_merge_on_corrupt_index_fails_loud(self, tmp_path):
        out = tmp_path / "scenes_metadata.json"
        out.write_text("{ this is not json")
        with pytest.raises(RuntimeError, match="Refusing to overwrite"):
            _merge_scene_entries(out, {"scene_new_000_seed0": {}}, merge=True)

    @pytest.mark.parametrize("scenes_value", ["null", "[]", "42"])
    def test_merge_on_non_dict_scenes_fails_loud(self, tmp_path, scenes_value):
        # valid JSON but 'scenes' is null / a list / a scalar -> would blow up the
        # dict merge with a raw TypeError; must fail loud with the contract message
        out = tmp_path / "scenes_metadata.json"
        out.write_text('{"scenes": %s}' % scenes_value)
        with pytest.raises(RuntimeError, match="Fix or remove the file first"):
            _merge_scene_entries(out, {"scene_x_000_seed0": {}}, merge=True)


# ---------------------------------------------------------------------------
# _process_scene skip behaviour (pxr)
# ---------------------------------------------------------------------------


def _author_scene(usd_path: Path, *, metadata: dict | None, with_floor: bool = True) -> None:
    """Author a temp scene USD with an optional floor mesh + customData.

    ``metadata=None`` leaves the customData key absent (the orphan-export
    defect). A floor mesh named to match the index's floor regex gives the
    scene the geometry the gate would otherwise accept on its own.
    """
    from pxr import Gf, UsdGeom, Usd

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    if with_floor:
        half = 2.0
        mesh = UsdGeom.Mesh.Define(stage, "/World/bedroom_0_0_floor")
        mesh.GetPointsAttr().Set(
            [
                Gf.Vec3f(-half, -half, 0.0),
                Gf.Vec3f(half, -half, 0.0),
                Gf.Vec3f(half, half, 0.0),
                Gf.Vec3f(-half, half, 0.0),
            ]
        )
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.GetExtentAttr().Set([Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, half, 0.0)])
    if metadata is not None:
        smr.write_custom_data(stage, metadata)
    stage.Save()


def _process(usd_path: Path):
    import random

    return _process_scene(
        usd_path,
        points_per_scene=20,
        wall_margin=0.3,
        robot_radius=0.35,
        obstacle_min_height=0.3,
        rng=random.Random(0),
    )


class TestProcessSceneGate:
    def test_absent_customdata_scene_is_skipped(self, tmp_path):
        pytest.importorskip("pxr")
        usd = tmp_path / "scene_absent.usdc"
        _author_scene(usd, metadata=None)
        assert _process(usd) is None

    def test_empty_objects_scene_is_skipped(self, tmp_path):
        pytest.importorskip("pxr")
        usd = tmp_path / "scene_empty.usdc"
        _author_scene(usd, metadata={"objects": [], "rooms": [{"room_type": "kitchen"}]})
        assert _process(usd) is None

    def test_populated_scene_is_indexed(self, tmp_path):
        pytest.importorskip("pxr")
        usd = tmp_path / "scene_ok.usdc"
        _author_scene(
            usd,
            metadata={
                "objects": [{"label": "chair", "instance_id": 1}],
                "rooms": [{"room_type": "kitchen"}],
            },
        )
        entry = _process(usd)
        assert entry is not None
        assert "spawn_points_xy" in entry
        assert "floor_bbox_xy" in entry

    def test_floorless_scene_is_skipped_even_with_metadata(self, tmp_path):
        # The geometry gate still applies: no floor => no spawn points => skip.
        pytest.importorskip("pxr")
        usd = tmp_path / "scene_nofloor.usdc"
        _author_scene(
            usd,
            metadata={"objects": [{"label": "chair"}], "rooms": [{"room_type": "x"}]},
            with_floor=False,
        )
        assert _process(usd) is None

    def test_malformed_customdata_scene_is_skipped_not_fatal(self, tmp_path):
        # Present-but-non-JSON customData must skip the one scene, not abort the
        # run: read_custom_data raises SceneMetadataError, which the gate catches.
        pytest.importorskip("pxr")
        from pxr import Usd, UsdGeom, Gf

        usd = tmp_path / "scene_malformed.usdc"
        stage = Usd.Stage.CreateNew(str(usd))
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        half = 2.0
        mesh = UsdGeom.Mesh.Define(stage, "/World/bedroom_0_0_floor")
        mesh.GetPointsAttr().Set(
            [
                Gf.Vec3f(-half, -half, 0.0),
                Gf.Vec3f(half, -half, 0.0),
                Gf.Vec3f(half, half, 0.0),
                Gf.Vec3f(-half, half, 0.0),
            ]
        )
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.GetExtentAttr().Set([Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, half, 0.0)])
        # Author a raw non-JSON string under the metadata key.
        smr.root_prim(stage).SetCustomDataByKey(smr.CUSTOM_DATA_KEY, "not json{{{")
        stage.Save()
        assert _process(usd) is None
