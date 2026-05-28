"""Lock in convexHull as the default collider approximation.

The historical default was ``none`` (raw triangle mesh), which made
PhysX allocate per-prim BVH + SDF data proportional to triangle count.
On a full Infinigen multi-room house (~900 mesh prims, often 10K+
tris each) this consumed 100+ GB of unified memory before the env
finished booting and OOM-killed the box. The default flip to
``convexHull`` is the fix; this test guards against a regression that
would re-introduce the OOM.

Runs in ``.venv_harness`` — needs ``pxr``, no Isaac Sim.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# Skip cleanly when pxr isn't installed (e.g. on a fresh .venv_harness
# without USD); CI configurations that pin the harness venv to pxr
# will run the test for real.
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


def _make_stage_with_meshes(tmp_path: Path, n_meshes: int = 3):
    """Return an open in-memory Usd.Stage with N Mesh prims under /World."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(tmp_path / "tiny.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    for i in range(n_meshes):
        UsdGeom.Mesh.Define(stage, f"/World/mesh_{i}")
    return stage


def _approximation_of(prim) -> str:
    from pxr import UsdPhysics

    api = UsdPhysics.MeshCollisionAPI(prim)
    attr = api.GetApproximationAttr()
    return attr.Get() if attr.IsValid() else None


class TestDefaultApproximation:
    def test_default_is_convex_hull(self, tmp_path):
        """The default attached approximation must be convexHull, NOT none.

        Regression for the OOM-the-box bug.
        """
        from pxr import UsdPhysics

        stage = _make_stage_with_meshes(tmp_path, n_meshes=3)
        count = postprocess_scene_usd.attach_mesh_colliders(stage)
        assert count == 3
        for i in range(3):
            prim = stage.GetPrimAtPath(f"/World/mesh_{i}")
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            assert _approximation_of(prim) == "convexHull"

    def test_module_default_constant_is_convex_hull(self):
        """A reader scanning the module should see convexHull as the documented
        default, not the historical 'none'."""
        assert postprocess_scene_usd._DEFAULT_APPROXIMATION == "convexHull"


class TestMigrationFromNone:
    """Critical regression test: re-running with the new default must rewrite
    colliders that were previously baked with ``none``.

    Without this behavior, scenes baked under the old default stay broken
    even after the script's default flips — exactly the failure mode the
    operator hit on 2026-05-28.
    """

    def test_rerun_rewrites_old_none_to_convex_hull(self, tmp_path):
        stage = _make_stage_with_meshes(tmp_path, n_meshes=3)
        # Simulate the old bake.
        first = postprocess_scene_usd.attach_mesh_colliders(
            stage, approximation="none",
        )
        assert first == 3
        for i in range(3):
            assert _approximation_of(
                stage.GetPrimAtPath(f"/World/mesh_{i}"),
            ) == "none"
        # Re-run with the new default — should rewrite every prim.
        second = postprocess_scene_usd.attach_mesh_colliders(stage)
        assert second == 3, "expected re-run to migrate all 3 prims"
        for i in range(3):
            assert _approximation_of(
                stage.GetPrimAtPath(f"/World/mesh_{i}"),
            ) == "convexHull"

    def test_rerun_with_same_approx_is_zero(self, tmp_path):
        """No-op re-run reports 0 changed prims (true idempotence)."""
        stage = _make_stage_with_meshes(tmp_path, n_meshes=3)
        postprocess_scene_usd.attach_mesh_colliders(stage)  # default convexHull
        again = postprocess_scene_usd.attach_mesh_colliders(stage)
        assert again == 0


class TestApproximationOverride:
    @pytest.mark.parametrize("approx", [
        "boundingCube",
        "boundingSphere",
        "convexHull",
        "convexDecomposition",
        "meshSimplification",
        "none",
    ])
    def test_override_threads_through(self, tmp_path, approx):
        stage = _make_stage_with_meshes(tmp_path, n_meshes=2)
        postprocess_scene_usd.attach_mesh_colliders(stage, approximation=approx)
        for i in range(2):
            assert _approximation_of(
                stage.GetPrimAtPath(f"/World/mesh_{i}")
            ) == approx

    def test_invalid_approximation_raises(self, tmp_path):
        stage = _make_stage_with_meshes(tmp_path, n_meshes=1)
        with pytest.raises(ValueError, match="unknown approximation"):
            postprocess_scene_usd.attach_mesh_colliders(
                stage, approximation="not-a-real-mode",
            )


class TestCLI:
    def test_cli_help_lists_choices(self, capsys):
        """The --collider-approximation flag's help must enumerate the choices
        so an operator can discover them via --help."""
        with pytest.raises(SystemExit):
            postprocess_scene_usd.main(["--usdc", "/nonexistent.usdc", "--help"])
        out = capsys.readouterr().out
        for choice in ("convexHull", "boundingCube", "none"):
            assert choice in out
