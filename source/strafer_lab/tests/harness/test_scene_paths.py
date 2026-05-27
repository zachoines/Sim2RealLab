"""Tests for the shared scene_metadata.json path resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from strafer_lab.tools.scene_paths import (
    find_metadata_for_usd,
    resolve_scene_metadata_path,
)


@pytest.fixture
def scene_tree(tmp_path: Path) -> Path:
    """Build a realistic ``Assets/generated/scenes/`` tree on disk.

    Layout (mirrors what prep_room_usds.py produces)::

        <tmp>/scenes/
          scene_alpha/
            scene_metadata.json
            coarse/exports/
              scene_alpha.usdc
          scene_alpha.usdc       (symlink → scene_alpha/coarse/exports/...)
          scene_beta/
            scene_metadata.json
    """
    scenes = tmp_path / "scenes"
    alpha_dir = scenes / "scene_alpha"
    (alpha_dir / "coarse" / "exports").mkdir(parents=True)
    (alpha_dir / "scene_metadata.json").write_text("{}")
    inner_usdc = alpha_dir / "coarse" / "exports" / "scene_alpha.usdc"
    inner_usdc.write_text("(stub)")
    symlink = scenes / "scene_alpha.usdc"
    symlink.symlink_to(inner_usdc)
    beta_dir = scenes / "scene_beta"
    beta_dir.mkdir()
    (beta_dir / "scene_metadata.json").write_text("{}")
    return scenes


class TestFindMetadataForUsd:
    def test_via_symlink(self, scene_tree):
        usd = scene_tree / "scene_alpha.usdc"
        result = find_metadata_for_usd(usd)
        assert result is not None
        assert result == (scene_tree / "scene_alpha" / "scene_metadata.json").resolve()

    def test_via_inner_path(self, scene_tree):
        usd = scene_tree / "scene_alpha" / "coarse" / "exports" / "scene_alpha.usdc"
        result = find_metadata_for_usd(usd)
        assert result is not None
        assert result == (scene_tree / "scene_alpha" / "scene_metadata.json").resolve()

    def test_returns_none_for_orphan_usd(self, tmp_path):
        orphan = tmp_path / "stray.usdc"
        orphan.write_text("(stub)")
        assert find_metadata_for_usd(orphan) is None


class TestResolveSceneMetadataPath:
    def test_metadata_override_wins(self, scene_tree):
        explicit = scene_tree / "scene_beta" / "scene_metadata.json"
        result = resolve_scene_metadata_path(
            scene="scene_alpha",
            metadata_override=explicit,
            usd_override=scene_tree / "scene_alpha.usdc",
        )
        assert result == explicit

    def test_usd_override_only_derives_sibling(self, scene_tree):
        usd = scene_tree / "scene_alpha.usdc"
        result = resolve_scene_metadata_path(
            scene=None,
            metadata_override=None,
            usd_override=usd,
            search_root=scene_tree,
        )
        assert result == (scene_tree / "scene_alpha" / "scene_metadata.json").resolve()

    def test_scene_name_default(self, scene_tree):
        result = resolve_scene_metadata_path(
            scene="scene_beta",
            search_root=scene_tree,
        )
        assert result == scene_tree / "scene_beta" / "scene_metadata.json"

    def test_missing_file_raises(self, scene_tree):
        with pytest.raises(FileNotFoundError):
            resolve_scene_metadata_path(
                scene="scene_nonexistent",
                search_root=scene_tree,
            )

    def test_usd_override_with_no_sibling_raises(self, tmp_path):
        orphan = tmp_path / "stray.usdc"
        orphan.write_text("(stub)")
        with pytest.raises(FileNotFoundError, match="sibling scene_metadata.json"):
            resolve_scene_metadata_path(
                scene=None,
                metadata_override=None,
                usd_override=orphan,
            )

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="at least one of"):
            resolve_scene_metadata_path()
