"""The single-scene default binds the LIGHTEST registered scene, not the first.

The unified-memory GB10 OOM-kills while loading the heaviest scene, so the bare
``make sim-bridge`` default must resolve to the lightest scene rather than the
alphabetical ``sorted()``-first (a ~29 GB ``high_quality`` room). These tests pin
that ``_lightest_scene_usd_path`` picks by resolved ``.usdc`` file size, that it
follows the top-level symlink to measure the target (not the link), that ties
break by name, and that the single-scene consumer routes through it — while
``_get_scene_usd_paths`` keeps its name-sorted order untouched (multi-scene
consumers depend on it).

Hermetic: each test authors synthetic scenes (byte-sized stand-in ``.usdc``
files behind the real top-level-symlink layout) in a tmp dir and repoints
``SCENE_USD_DIR`` at it. Nothing reads the on-disk scene corpus, which is a
transient, regenerable artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_scene(scenes_root: Path, stem: str, size_bytes: int) -> Path:
    """Author one scene: a sized real ``.usdc`` plus its top-level symlink.

    Mirrors the corpus layout (``<scenes>/<stem>/export/<stem>.usdc`` with a
    top-level ``<scenes>/<stem>.usdc`` symlink) that ``_get_scene_usd_paths``
    discovers. The real file is made sparse via ``truncate`` — ``st_size``
    reports the logical size, which is all the picker reads.
    """
    inner = scenes_root / stem / "export" / f"{stem}.usdc"
    inner.parent.mkdir(parents=True, exist_ok=True)
    with open(inner, "wb") as fh:
        fh.truncate(size_bytes)
    link = scenes_root / f"{stem}.usdc"
    link.symlink_to(inner)
    return link


def _make_broken_scene(scenes_root: Path, stem: str) -> Path:
    """Author a scene whose top-level symlink dangles (target never created).

    Such a symlink still survives ``_get_scene_usd_paths`` discovery — suffix and
    stem read the link name, not the target — so it exercises the resolver's
    ``stat``-fails-> treat-as-heaviest fallback.
    """
    link = scenes_root / f"{stem}.usdc"
    link.symlink_to(scenes_root / stem / "export" / f"{stem}.usdc")
    return link


def _write_manifest(scenes_root: Path, stems: list[str]) -> None:
    (scenes_root / "scenes_metadata.json").write_text(
        json.dumps({"scenes": {stem: {} for stem in stems}})
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tmp scenes dir wired in as ``SCENE_USD_DIR`` + the cfg module."""
    from strafer_lab.tasks.navigation import strafer_env_cfg as cfg_mod

    root = tmp_path / "scenes"
    root.mkdir()
    monkeypatch.setattr(cfg_mod, "SCENE_USD_DIR", root)
    return SimpleNamespace(mod=cfg_mod, root=root)


def _fake_env_cfg():
    """A minimal stand-in for the bits ``_apply_infinigen_scene_setup`` writes."""
    return SimpleNamespace(
        scene=SimpleNamespace(
            scene_geometry=SimpleNamespace(spawn=SimpleNamespace(usd_path="UNSET")),
        ),
        sim=SimpleNamespace(render=SimpleNamespace(carb_settings=None)),
        events=SimpleNamespace(
            reset_robot=SimpleNamespace(params={"spawn_points_xy": [], "spawn_z": 9.9}),
            lift_ground=SimpleNamespace(params={"target_z": 9.9}),
        ),
        commands=SimpleNamespace(goal_command=SimpleNamespace(spawn_points_xy=None)),
    )


class TestLightestScenePick:
    def test_picks_smallest_resolved_usdc(self, env):
        _make_scene(env.root, "scene_a", 30_000)
        _make_scene(env.root, "scene_b", 10_000)  # lightest
        _make_scene(env.root, "scene_c", 20_000)
        _write_manifest(env.root, ["scene_a", "scene_b", "scene_c"])

        assert Path(env.mod._lightest_scene_usd_path()).stem == "scene_b"

    def test_measures_symlink_target_not_link(self, env):
        # scene_small's link path is longer than scene_big's, so if the picker
        # stat'd the symlink itself (size == target-path length) it would wrongly
        # prefer scene_big. Resolving to the target inverts that.
        small = _make_scene(env.root, "scene_small", 5_000)
        _make_scene(env.root, "scene_big", 50_000)
        _write_manifest(env.root, ["scene_small", "scene_big"])

        picked = Path(env.mod._lightest_scene_usd_path())
        assert picked == small
        assert picked.is_symlink()  # returns the top-level symlink, not the target
        assert picked.resolve().stat().st_size == 5_000

    def test_tie_break_is_name_stable(self, env):
        _make_scene(env.root, "scene_bbb", 10_000)
        _make_scene(env.root, "scene_aaa", 10_000)  # equal size
        _write_manifest(env.root, ["scene_aaa", "scene_bbb"])

        assert Path(env.mod._lightest_scene_usd_path()).stem == "scene_aaa"

    def test_missing_manifest_fallback(self, env):
        # No scenes_metadata.json -> _get_scene_usd_paths uses the scene_* glob;
        # the lightest pick must still hold on that fallback path.
        _make_scene(env.root, "scene_heavy", 40_000)
        _make_scene(env.root, "scene_light", 8_000)
        assert not (env.root / "scenes_metadata.json").exists()

        assert Path(env.mod._lightest_scene_usd_path()).stem == "scene_light"

    def test_broken_symlink_never_preferred_over_real(self, env):
        # A dangling symlink stat-fails -> treated as heaviest (never preferred),
        # so a real scene beside it still wins. Guards against a regression that
        # would rank an unresolvable target as smallest and bind an unloadable USD.
        _make_scene(env.root, "scene_real", 8_000)
        _make_broken_scene(env.root, "scene_dangling")

        assert Path(env.mod._lightest_scene_usd_path()).stem == "scene_real"

    def test_only_broken_symlink_is_last_resort(self, env):
        # If the ONLY entry is unresolvable, it is still returned (last resort) —
        # the same scene the old [0] would have handed to the loader.
        broken = _make_broken_scene(env.root, "scene_dangling")

        assert Path(env.mod._lightest_scene_usd_path()) == broken


class TestSortedOrderInvariant:
    def test_get_scene_usd_paths_stays_name_sorted(self, env):
        # Hard constraint: the multi-scene list keeps name order even though the
        # lightest (scene_c) sorts last — its ordering must not move to size.
        _make_scene(env.root, "scene_c", 10)
        _make_scene(env.root, "scene_a", 999)
        _make_scene(env.root, "scene_b", 500)
        _write_manifest(env.root, ["scene_a", "scene_b", "scene_c"])

        stems = [Path(p).stem for p in env.mod._get_scene_usd_paths()]
        assert stems == ["scene_a", "scene_b", "scene_c"]


class TestConsumerRouting:
    def test_apply_setup_binds_the_lightest(self, env):
        # The single-scene consumer routes through _lightest_scene_usd_path: with
        # no occupancy/manifest the spawn/floor branches no-op, so this isolates
        # the scene bind — it must be the lightest scene's resolved USDC.
        _make_scene(env.root, "scene_big", 200_000)
        small = _make_scene(env.root, "scene_small", 20_000)

        cfg = _fake_env_cfg()
        env.mod._apply_infinigen_scene_setup(cfg)

        assert cfg.scene.scene_geometry.spawn.usd_path == str(small.resolve())
