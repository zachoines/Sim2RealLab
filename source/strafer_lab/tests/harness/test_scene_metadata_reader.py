"""USD ``customData`` round-trip tests for the scene-metadata reader.

The reader is the single ``pxr`` touch-point for scene metadata. These
author a temp USD, read it straight back, and prove: the dict survives
losslessly (including the additive version key), an absent payload
hard-fails (no sidecar fallback), and the canonical hash is stable.

``pytest.importorskip("pxr")``-gated; runs under ``make test-lab-pure``
(the pure-Python lab suite carries pxr since the harness fold) and the Kit suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strafer_lab.tools import scene_metadata_reader as smr


def _author(usd_path: Path, metadata: dict, *, default_prim: bool = True) -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Xform.Define(stage, "/World")
    if default_prim:
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    smr.write_custom_data(stage, metadata)
    stage.Save()


def _metadata() -> dict:
    return {
        "objects": [
            {"label": "chair", "instance_id": 1, "position_3d": [2.5, 1.2, 0.5],
             "prim_path": "/World/Chair_1", "semantic_tags": ["seating"]},
        ],
        "rooms": [{"room_type": "kitchen", "footprint_xy": [[0, 0], [1, 0], [1, 1]]}],
        "room_adjacency": [],
    }


class TestRoundTrip:
    def test_dict_survives_losslessly(self, tmp_path):
        pytest.importorskip("pxr")
        usd = tmp_path / "scene.usdc"
        _author(usd, _metadata())
        loaded = smr.load(usd)
        assert loaded["objects"][0]["label"] == "chair"
        assert loaded["objects"][0]["position_3d"] == [2.5, 1.2, 0.5]
        assert loaded["rooms"][0]["room_type"] == "kitchen"

    def test_version_key_authored(self, tmp_path):
        pytest.importorskip("pxr")
        usd = tmp_path / "scene.usdc"
        _author(usd, _metadata())
        loaded = smr.load(usd)
        assert loaded[smr.VERSION_FIELD] == smr.SCHEMA_VERSION

    def test_falls_back_to_world_prim_without_default(self, tmp_path):
        pytest.importorskip("pxr")
        usd = tmp_path / "scene.usdc"
        _author(usd, _metadata(), default_prim=False)
        # root_prim() falls back to /World when no defaultPrim is set, so the
        # writer and reader still agree.
        loaded = smr.load(usd)
        assert loaded["objects"][0]["label"] == "chair"


class TestHardFail:
    def test_absent_customdata_raises(self, tmp_path):
        pytest.importorskip("pxr")
        from pxr import Usd, UsdGeom

        usd = tmp_path / "bare.usdc"
        stage = Usd.Stage.CreateNew(str(usd))
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        stage.Save()
        with pytest.raises(smr.SceneMetadataError):
            smr.load(usd)

    def test_missing_usd_raises(self, tmp_path):
        with pytest.raises(smr.SceneMetadataError):
            smr.load(tmp_path / "does_not_exist.usdc")


class TestHash:
    def test_hash_is_deterministic_and_key_order_independent(self):
        a = {"objects": [], "rooms": [], "room_adjacency": []}
        b = {"room_adjacency": [], "rooms": [], "objects": []}
        assert smr.metadata_hash(a) == smr.metadata_hash(b)

    def test_hash_changes_with_content(self):
        a = {"objects": [{"label": "chair"}]}
        b = {"objects": [{"label": "table"}]}
        assert smr.metadata_hash(a) != smr.metadata_hash(b)


class TestMissionGeneratorFromSceneUsd:
    def test_reads_targets_from_usd(self, tmp_path):
        pytest.importorskip("pxr")
        from strafer_lab.sim_in_the_loop.mission import MissionGenerator

        usd = tmp_path / "scene_kitchen.usdc"
        _author(usd, _metadata())
        gen = MissionGenerator.from_scene_usd(usd)
        specs = list(gen)
        assert len(specs) == 1
        # scene_name defaults to the USD stem.
        assert specs[0].scene_name == "scene_kitchen"
        assert specs[0].target_label == "chair"
