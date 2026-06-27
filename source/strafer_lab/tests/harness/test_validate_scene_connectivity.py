"""Unit tests for the connectivity step's post-save persistence guard.

The Kit-bound occupancy/connectivity computation in
``validate_scene_connectivity.py`` is exercised on real scenes under Isaac Sim.
Here we cover only the pure-python ``assert_persisted_metadata`` decision — the
read-back guard that fires after ``ctx.save_stage()``. The save's milestone log
is unconditional, so a silent disk/inotify write failure would otherwise be
reported as success; the guard reopens the file and refuses to claim success
unless the payload actually persisted.

The ``assert_persisted_metadata`` decision tests are pure-python (no Kit, no
pxr). The ``reopen_and_read_persisted`` test is ``pytest.importorskip("pxr")``-
gated: it proves the read-back reflects the *on-disk* bytes (forcing a layer
Reload past USD's shared in-memory singleton) rather than unsaved edits — the
exact silent-save-failure mode the guard exists to catch. The script lives under
``source/strafer_lab/scripts`` (not an importable package) and imports cleanly
without the Kit launcher (numpy + strafer_lab tools only), so it is loaded by
file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from strafer_lab.tools import scene_metadata_reader as smr

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "validate_scene_connectivity.py"
)
_spec = importlib.util.spec_from_file_location("validate_scene_connectivity", _SCRIPT)
validate_scene_connectivity = importlib.util.module_from_spec(_spec)
sys.modules["validate_scene_connectivity"] = validate_scene_connectivity
_spec.loader.exec_module(validate_scene_connectivity)

_assert = validate_scene_connectivity.assert_persisted_metadata


def _reread(*, objects: list, version: bool = True, connectivity: bool = True) -> dict:
    """A fake post-save read-back payload."""
    data: dict = {"objects": objects, "rooms": [{"room_type": "kitchen"}]}
    if version:
        data[smr.VERSION_FIELD] = smr.SCHEMA_VERSION
    if connectivity:
        data["connectivity"] = []
    return data


class TestPersistenceGuard:
    def test_absent_key_raises(self):
        # The whole point: a save that silently dropped the key reads back None.
        with pytest.raises(RuntimeError, match="ABSENT after save"):
            _assert(None, expect_objects=False)

    def test_missing_version_token_raises(self):
        with pytest.raises(RuntimeError, match="version token"):
            _assert(_reread(objects=[{"label": "chair"}], version=False), expect_objects=True)

    def test_missing_connectivity_raises(self):
        with pytest.raises(RuntimeError, match="connectivity"):
            _assert(_reread(objects=[{"label": "chair"}], connectivity=False), expect_objects=True)

    def test_blanked_objects_raises_when_expected(self):
        with pytest.raises(RuntimeError, match="objects"):
            _assert(_reread(objects=[]), expect_objects=True)

    def test_populated_objects_passes_when_expected(self):
        _assert(_reread(objects=[{"label": "chair"}]), expect_objects=True)

    def test_empty_objects_ok_when_not_expected(self):
        # The legitimate intermediate: key present, objects empty, connectivity
        # enriching it. Not an error when no populated embed was present going in.
        _assert(_reread(objects=[]), expect_objects=False)


_reopen = validate_scene_connectivity.reopen_and_read_persisted


class TestDiskReadBack:
    """The read-back must reflect on-disk bytes, not unsaved in-memory edits."""

    def test_reflects_on_disk_not_unsaved_edits(self, tmp_path):
        # Reproduce the silent-save-failure shape: author + save v1, mutate the
        # SAME (singleton) layer in memory WITHOUT saving, then prove the
        # read-back returns the persisted v1 — not the unsaved v2. Without the
        # Reload() in reopen_and_read_persisted this would wrongly return v2.
        pytest.importorskip("pxr")
        from pxr import Usd, UsdGeom

        usd = tmp_path / "scene.usdc"
        stage = Usd.Stage.CreateNew(str(usd))
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        smr.write_custom_data(
            stage, {"objects": [{"label": "chair"}], "rooms": [], "connectivity": []}
        )
        stage.Save()  # v1 persisted to disk

        # Author a second opinion in memory and do NOT save (the dropped write).
        smr.root_prim(stage).SetCustomDataByKey(
            smr.CUSTOM_DATA_KEY, '{"objects": [], "rooms": [], "connectivity": []}'
        )

        reread = _reopen(usd)
        assert reread is not None
        # The persisted (v1) objects survive; the unsaved blanking is not seen.
        assert len(reread["objects"]) == 1

    def test_missing_file_fails_loudly(self, tmp_path):
        # A file that vanished after save is the most extreme non-persistence:
        # the read-back must fail loudly (propagating to rc=1), not return a
        # benign None. Usd.Stage.Open raises on a missing layer.
        pytest.importorskip("pxr")
        from pxr import Tf

        with pytest.raises(Tf.ErrorException):
            _reopen(tmp_path / "does_not_exist.usdc")


class TestRefitTrimColliders:
    """``refit_trim_collider_approximations`` re-approximates only the filling
    convex-hull trim colliders, leaving everything else untouched."""

    def _mesh(self, stage, path, *, collider=True, approximation="convexHull"):
        from pxr import UsdGeom, UsdPhysics

        mesh = UsdGeom.Mesh.Define(stage, path)
        if collider:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mca = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            if approximation is not None:
                mca.GetApproximationAttr().Set(approximation)
        return mesh.GetPrim()

    def test_only_convex_hull_trim_is_refit(self):
        pytest.importorskip("pxr")
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        hulled = self._mesh(stage, "/W/skirtingboard_support", approximation="convexHull")
        faithful = self._mesh(stage, "/W/skirtingboard_ceiling", approximation="none")
        non_trim = self._mesh(stage, "/W/living_room_0_0_wall", approximation="convexHull")
        no_collider = self._mesh(stage, "/W/skirtingboard_loose", collider=False)

        count = validate_scene_connectivity.refit_trim_collider_approximations(stage)

        assert count == 1  # only the convex-hull skirting board
        assert UsdPhysics.MeshCollisionAPI(hulled).GetApproximationAttr().Get() == "none"
        # a faithful trim approximation, a non-trim hull, and a collider-less trim
        # mesh are all left exactly as they were.
        assert UsdPhysics.MeshCollisionAPI(faithful).GetApproximationAttr().Get() == "none"
        assert UsdPhysics.MeshCollisionAPI(non_trim).GetApproximationAttr().Get() == "convexHull"
        assert not no_collider.HasAPI(UsdPhysics.MeshCollisionAPI)

    def test_noop_when_no_trim(self):
        pytest.importorskip("pxr")
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        self._mesh(stage, "/W/living_room_0_0_wall", approximation="convexHull")
        assert validate_scene_connectivity.refit_trim_collider_approximations(stage) == 0
