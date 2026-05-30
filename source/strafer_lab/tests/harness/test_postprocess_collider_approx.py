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


def _make_mixed_stage(tmp_path: Path):
    """Stage with realistic Infinigen-style paths: a wall, ceiling, and chair."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(tmp_path / "mixed.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    paths = [
        "/World/bedroom_0_0_wall",       # structural
        "/World/bedroom_0_0_wall_3",     # structural (numbered variant)
        "/World/bedroom_0_0_ceiling",    # structural
        "/World/ChairFactory_123",       # furniture
        "/World/BedFactory_456",         # furniture
    ]
    for p in paths:
        UsdGeom.Mesh.Define(stage, p)
    return stage, paths


def _approximation_of(prim) -> str:
    from pxr import UsdPhysics

    api = UsdPhysics.MeshCollisionAPI(prim)
    attr = api.GetApproximationAttr()
    return attr.Get() if attr.IsValid() else None


class TestDefaultApproximation:
    def test_default_is_convex_hull(self, tmp_path):
        """The default furniture approximation must be convexHull, NOT none.

        Regression for the OOM-the-box bug.
        """
        from pxr import UsdPhysics

        stage = _make_stage_with_meshes(tmp_path, n_meshes=3)
        furniture, structural = postprocess_scene_usd.attach_mesh_colliders(stage)
        assert (furniture, structural) == (3, 0)
        for i in range(3):
            prim = stage.GetPrimAtPath(f"/World/mesh_{i}")
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            assert _approximation_of(prim) == "convexHull"

    def test_module_default_constants(self):
        """A reader scanning the module should see the documented defaults.

        Structural default is meshSimplification, NOT convexDecomposition:
        Infinigen exports one wall mesh per room with door cutouts done via
        Boolean DIFFERENCE; V-HACD with default params heals the small
        cutouts and traps the robot at the doorway.
        """
        assert postprocess_scene_usd._DEFAULT_APPROXIMATION == "convexHull"
        assert (
            postprocess_scene_usd._DEFAULT_STRUCTURAL_APPROXIMATION
            == "meshSimplification"
        )


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
        assert first == (3, 0)
        for i in range(3):
            assert _approximation_of(
                stage.GetPrimAtPath(f"/World/mesh_{i}"),
            ) == "none"
        # Re-run with the new default — should rewrite every prim.
        second = postprocess_scene_usd.attach_mesh_colliders(stage)
        assert second == (3, 0), "expected re-run to migrate all 3 furniture prims"
        for i in range(3):
            assert _approximation_of(
                stage.GetPrimAtPath(f"/World/mesh_{i}"),
            ) == "convexHull"

    def test_rerun_with_same_approx_is_zero(self, tmp_path):
        """No-op re-run reports 0 changed prims (true idempotence)."""
        stage = _make_stage_with_meshes(tmp_path, n_meshes=3)
        postprocess_scene_usd.attach_mesh_colliders(stage)  # default convexHull
        again = postprocess_scene_usd.attach_mesh_colliders(stage)
        assert again == (0, 0)


class TestStructuralDispatch:
    """Walls / ceilings / etc. must get meshSimplification, furniture convexHull.

    convexHull (or convexDecomposition with default V-HACD params) heals
    door cutouts on Infinigen wall meshes — the cutouts are volumetrically
    small relative to the room shell, so V-HACD merges them into a single
    hull that fills the doorway. meshSimplification preserves the actual
    triangle topology including cutouts. See `_DEFAULT_STRUCTURAL_PRIM_PATTERN`.
    """

    def _structural_re(self):
        import re
        return re.compile(postprocess_scene_usd._DEFAULT_STRUCTURAL_PRIM_PATTERN)

    def test_default_pattern_splits_walls_and_furniture(self, tmp_path):
        stage, paths = _make_mixed_stage(tmp_path)
        furniture, structural = postprocess_scene_usd.attach_mesh_colliders(
            stage, structural_pattern=self._structural_re(),
        )
        # 3 structural (wall, wall_3, ceiling), 2 furniture (chair, bed)
        assert (furniture, structural) == (2, 3)
        for path in paths:
            approx = _approximation_of(stage.GetPrimAtPath(path))
            if "wall" in path or "ceiling" in path:
                assert approx == "meshSimplification", (
                    f"{path} should be meshSimplification, got {approx}"
                )
            else:
                assert approx == "convexHull", (
                    f"{path} should be convexHull, got {approx}"
                )

    def test_no_pattern_means_no_structural_dispatch(self, tmp_path):
        """structural_pattern=None means every prim gets `approximation`."""
        stage, paths = _make_mixed_stage(tmp_path)
        furniture, structural = postprocess_scene_usd.attach_mesh_colliders(
            stage, structural_pattern=None,
        )
        assert (furniture, structural) == (5, 0)
        for path in paths:
            assert _approximation_of(stage.GetPrimAtPath(path)) == "convexHull"

    def test_default_pattern_matches_expected_prim_names(self):
        """Lock in the canonical Infinigen structural prim naming.

        Includes door prims (PanelDoor / LiteDoor / LouverDoor factories):
        Infinigen exports each door as TWO prims — a thin __001 leaf and
        a hollow non-001 frame. The frame's convex hull fills the entire
        doorway AABB and traps the robot; meshSimplification preserves
        the actual hollow topology.
        """
        import re
        pat = re.compile(postprocess_scene_usd._DEFAULT_STRUCTURAL_PRIM_PATTERN)
        for path in (
            "/World/bedroom_0_0_wall",
            "/World/bedroom_0_0_wall_5",
            "/World/kitchen_1_2_ceiling",
            "/World/attic_0_0_attic",
            "/World/exterior_0_0_exterior",
            "/World/something_roof",
            # nested Mesh leaves should also match
            "/World/bedroom_0_0_wall/bedroom_0_0_wall",
            # Door factory prims — frame variant (no __001)
            "/World/PanelDoorFactory_9952204__spawn_asset_3_",
            # Door factory prims — leaf variant (__001)
            "/World/PanelDoorFactory_9952204__spawn_asset_3__001",
            # Other Infinigen door factories
            "/World/LiteDoorFactory_123__spawn_asset_5_",
            "/World/LouverDoorFactory_456__spawn_asset_7__001",
            # Nested Mesh leaves under door prims
            "/World/PanelDoorFactory_9952204__spawn_asset_3_/x",
        ):
            assert pat.match(path), f"expected match: {path}"
        for path in (
            "/World/ChairFactory_123",
            "/World/BedFactory_456",
            "/World/bedroom_0_0_floor",   # floors handled separately
            "/World/Wallpaper_789",       # naming collision: 'wall' substring
            "/World/Door_misc_123",       # door substring but not a Factory prim
        ):
            assert not pat.match(path), f"expected no-match: {path}"

    def test_migration_from_old_uniform_convex_hull(self, tmp_path):
        """A USDC postprocessed BEFORE the hybrid split (everything was
        convexHull, including walls) gets walls upgraded to
        meshSimplification on re-run with the new default."""
        stage, _ = _make_mixed_stage(tmp_path)
        # Simulate the pre-hybrid state: all convexHull, no pattern.
        postprocess_scene_usd.attach_mesh_colliders(
            stage, structural_pattern=None,
        )
        # New run with hybrid default.
        furniture, structural = postprocess_scene_usd.attach_mesh_colliders(
            stage, structural_pattern=self._structural_re(),
        )
        assert structural == 3, (
            "expected 3 wall/ceiling prims to migrate convexHull → meshSimplification"
        )
        assert furniture == 0, "furniture was already convexHull; should be no-op"


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

    def test_invalid_structural_approximation_raises(self, tmp_path):
        stage = _make_stage_with_meshes(tmp_path, n_meshes=1)
        with pytest.raises(ValueError, match="unknown structural_approximation"):
            postprocess_scene_usd.attach_mesh_colliders(
                stage, structural_approximation="not-a-real-mode",
            )


class TestCLI:
    def test_cli_help_lists_choices_and_structural_flags(self, capsys):
        """--help must enumerate every approximation choice AND surface
        the structural-prim dispatch flags so an operator can discover
        them without reading the source."""
        with pytest.raises(SystemExit):
            postprocess_scene_usd.main(["--usdc", "/nonexistent.usdc", "--help"])
        out = capsys.readouterr().out
        for choice in ("convexHull", "convexDecomposition", "boundingCube", "none"):
            assert choice in out
        assert "--structural-approximation" in out
        assert "--structural-prim-pattern" in out
        assert "--ceiling-light-prim-pattern" in out


class TestCeilingLightInjection:
    """The parameterized ceiling-light pattern controls which prims get lights.

    Default keeps Infinigen's ``CeilingLightFactory_*`` behaviour; a
    foreign source overrides the pattern to author lights under its own
    fixture naming. Matched against the prim NAME, not the full path.
    """

    def _make_light_stage(self, tmp_path):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateNew(str(tmp_path / "lights.usda"))
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/CeilingLightFactory_5__spawn_asset_2_")
        UsdGeom.Xform.Define(stage, "/World/MyCeilingLight_7")
        return stage

    def _has_light_child(self, stage, prim_path: str) -> bool:
        prim = stage.GetPrimAtPath(prim_path)
        return prim.GetChild("AutoSphereLight").IsValid()

    def test_default_lights_only_infinigen_fixture(self, tmp_path):
        stage = self._make_light_stage(tmp_path)
        count = postprocess_scene_usd.inject_ceiling_light_emitters(stage, 1000.0)
        assert count == 1
        assert self._has_light_child(stage, "/World/CeilingLightFactory_5__spawn_asset_2_")
        assert not self._has_light_child(stage, "/World/MyCeilingLight_7")

    def test_override_lights_only_foreign_fixture(self, tmp_path):
        import re

        stage = self._make_light_stage(tmp_path)
        pattern = re.compile(r"^MyCeilingLight_\d+$")
        count = postprocess_scene_usd.inject_ceiling_light_emitters(
            stage, 1000.0, pattern,
        )
        assert count == 1
        assert self._has_light_child(stage, "/World/MyCeilingLight_7")
        assert not self._has_light_child(stage, "/World/CeilingLightFactory_5__spawn_asset_2_")
