"""Tests for the Infinigen scene-prep orchestration script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def prep_mod():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "source" / "strafer_lab" / "scripts" / "prep_room_usds.py"
    spec = importlib.util.spec_from_file_location("prep_room_usds", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prep_room_usds"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class TestPresets:
    def test_high_quality_dgx_is_larger(self, prep_mod):
        dgx = prep_mod.PRESETS["high_quality_dgx"]
        windows = prep_mod.PRESETS["windows_baseline"]
        assert dgx.max_poly_count_millions > windows.max_poly_count_millions
        assert dgx.max_objects_per_room >= windows.max_objects_per_room
        assert dgx.max_rooms_per_scene >= windows.max_rooms_per_scene

    def test_preset_names_round_trip(self, prep_mod):
        for name, preset in prep_mod.PRESETS.items():
            assert preset.name == name

    def test_to_dict_serializable(self, prep_mod):
        dgx = prep_mod.PRESETS["high_quality_dgx"]
        encoded = json.dumps(dgx.to_dict())
        decoded = json.loads(encoded)
        assert decoded["name"] == "high_quality_dgx"
        assert isinstance(decoded["room_types"], list)


class TestIngestExternalUsd:
    def test_copies_scene_dirs_with_usd(self, prep_mod, tmp_path):
        source = tmp_path / "external"
        scene_a = source / "kitchen_01"
        scene_a.mkdir(parents=True)
        (scene_a / "kitchen_01.usd").write_text("USD")
        (scene_a / "kitchen_01.blend").write_bytes(b"BLEND")

        scene_empty = source / "empty_scene"
        scene_empty.mkdir()
        (scene_empty / "README.md").write_text("no usd here")

        output = tmp_path / "assets"
        imported = prep_mod.ingest_external_usd(
            source_dir=source,
            output_dir=output,
            preset_name="windows_baseline",
        )
        assert len(imported) == 1
        assert (output / "kitchen_01" / "kitchen_01.usd").exists()
        config = json.loads((output / "kitchen_01" / "scene_config.json").read_text())
        assert config["name"] == "windows_baseline"

    def test_skips_existing_destination(self, prep_mod, tmp_path):
        source = tmp_path / "external"
        scene = source / "kitchen_01"
        scene.mkdir(parents=True)
        (scene / "kitchen_01.usd").write_text("USD")

        output = tmp_path / "assets"
        (output / "kitchen_01").mkdir(parents=True)

        imported = prep_mod.ingest_external_usd(
            source_dir=source, output_dir=output,
        )
        assert imported == []


class TestCLI:
    def test_presets_list(self, prep_mod, capsys):
        rc = prep_mod.main(["presets"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "high_quality_dgx" in captured.out
        assert "windows_baseline" in captured.out

    def test_presets_json(self, prep_mod, capsys):
        rc = prep_mod.main(["presets", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        payload = json.loads(captured.out)
        assert "high_quality_dgx" in payload


class TestResolveBlenderBinary:
    """Verify STRAFER_BLENDER_BIN env-var resolver behavior.

    The resolver has a single source of truth (the process environment
    loaded by ``env_setup.sh``). These tests monkeypatch ``os.environ``
    directly and do not touch any ``.env`` file on disk.
    """

    @staticmethod
    def _real_binary() -> Path:
        """Return a path to any real executable file for existence checks.

        ``/usr/bin/true`` is present on any Linux host; fall back to the
        Python interpreter itself on macOS or unusual CI images.
        """
        candidate = Path("/usr/bin/true")
        if candidate.exists():
            return candidate
        import sys as _sys
        return Path(_sys.executable)

    def test_reads_from_env_var(self, prep_mod, monkeypatch):
        real = self._real_binary()
        monkeypatch.setenv("STRAFER_BLENDER_BIN", str(real))
        assert prep_mod._resolve_blender_binary() == str(real)

    def test_missing_env_raises(self, prep_mod, monkeypatch):
        monkeypatch.delenv("STRAFER_BLENDER_BIN", raising=False)
        with pytest.raises(RuntimeError, match="STRAFER_BLENDER_BIN is not set"):
            prep_mod._resolve_blender_binary()

    def test_nonexistent_path_raises(self, prep_mod, monkeypatch):
        monkeypatch.setenv("STRAFER_BLENDER_BIN", "/definitely/not/here/blender")
        with pytest.raises(RuntimeError, match="non-existent path"):
            prep_mod._resolve_blender_binary()
