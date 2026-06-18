"""Tests for the shrunk scene-USD path resolver.

The sidecar resolver is gone — scene metadata now travels inside the USD
``customData``, so these tools only derive *which* USD to read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strafer_lab.tools.scene_paths import resolve_scene_usd_path, scene_usd_path


@pytest.fixture
def scenes_root(tmp_path: Path) -> Path:
    """A scenes root with one top-level ``<scene>.usdc`` symlink."""
    scenes = tmp_path / "scenes"
    alpha_dir = scenes / "scene_alpha" / "export"
    alpha_dir.mkdir(parents=True)
    inner = alpha_dir / "scene_alpha.usdc"
    inner.write_text("(stub)")
    (scenes / "scene_alpha.usdc").symlink_to(inner)
    return scenes


class TestSceneUsdPath:
    def test_derives_top_level_symlink(self):
        assert scene_usd_path("scene_alpha", search_root="Assets/generated/scenes") == (
            Path("Assets/generated/scenes/scene_alpha.usdc")
        )


class TestResolveSceneUsdPath:
    def test_usd_override_used_as_is(self, scenes_root):
        target = scenes_root / "scene_alpha.usdc"
        assert resolve_scene_usd_path(usd_override=target) == target

    def test_usd_override_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_scene_usd_path(usd_override=tmp_path / "nope.usdc")

    def test_scene_name_resolves_under_root(self, scenes_root):
        result = resolve_scene_usd_path(scene="scene_alpha", search_root=scenes_root)
        assert result == scenes_root / "scene_alpha.usdc"

    def test_scene_name_missing_raises(self, scenes_root):
        with pytest.raises(FileNotFoundError):
            resolve_scene_usd_path(scene="ghost", search_root=scenes_root)

    def test_requires_one_of_scene_or_override(self):
        with pytest.raises(ValueError):
            resolve_scene_usd_path()
