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
