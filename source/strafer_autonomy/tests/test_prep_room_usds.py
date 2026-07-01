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
    def test_presets_have_real_gin_configs(self, prep_mod):
        for name, preset in prep_mod.PRESETS.items():
            assert preset.gin_configs, f"{name} has empty gin_configs"
            for gin_file in preset.gin_configs:
                assert not gin_file.endswith(".gin"), (
                    f"{name}: gin_configs entries must omit the .gin suffix "
                    f"(Infinigen adds it), got {gin_file!r}"
                )

    def test_presets_disable_terrain(self, prep_mod):
        # Every indoor preset should disable Infinigen's terrain generator,
        # which otherwise inflates runtime and can fail solve.
        for name, preset in prep_mod.PRESETS.items():
            assert any(
                "terrain_enabled=False" in o for o in preset.gin_overrides
            ), f"{name} does not disable terrain"

    def test_preset_names_round_trip(self, prep_mod):
        for name, preset in prep_mod.PRESETS.items():
            assert preset.name == name

    def test_texture_resolution_positive(self, prep_mod):
        for name, preset in prep_mod.PRESETS.items():
            assert preset.texture_resolution > 0, f"{name} has nonpositive texture_resolution"

    def test_max_rooms_positive(self, prep_mod):
        for name, preset in prep_mod.PRESETS.items():
            assert preset.max_rooms >= 1, f"{name} has max_rooms<1"

    def test_to_dict_serializable(self, prep_mod):
        for name, preset in prep_mod.PRESETS.items():
            encoded = json.dumps(preset.to_dict())
            decoded = json.loads(encoded)
            assert decoded["name"] == name
            assert isinstance(decoded["gin_configs"], list)
            assert isinstance(decoded["gin_overrides"], list)


class TestBuildGenerateCommand:
    def test_includes_gin_flags(self, prep_mod, tmp_path):
        cfg = prep_mod.SceneGenConfig(
            name="t",
            gin_configs=("fast_solve", "singleroom"),
            gin_overrides=("compose_indoors.terrain_enabled=False",),
            max_rooms=3,
        )
        cmd = prep_mod._build_generate_command(
            infinigen_python="/fake/python",
            output_folder=tmp_path / "out",
            seed=42,
            config=cfg,
        )
        assert cmd[0] == "/fake/python"
        assert cmd[1:3] == ["-m", "infinigen_examples.generate_indoors"]
        assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "42"
        assert "-g" in cmd
        g_idx = cmd.index("-g")
        assert cmd[g_idx + 1 : g_idx + 3] == ["fast_solve", "singleroom"]
        assert "-p" in cmd
        p_idx = cmd.index("-p")
        # Every -p payload is captured up to the next flag or end of args.
        p_values = cmd[p_idx + 1 :]
        assert "compose_indoors.terrain_enabled=False" in p_values
        assert "restrict_solving.solve_max_rooms=3" in p_values
        assert "--task" in cmd and cmd[cmd.index("--task") + 1] == "coarse"

    def test_always_appends_max_rooms_override(self, prep_mod, tmp_path):
        cfg = prep_mod.SceneGenConfig(name="t", gin_configs=(), gin_overrides=(), max_rooms=7)
        cmd = prep_mod._build_generate_command(
            infinigen_python="/fake/python",
            output_folder=tmp_path / "out",
            seed=0,
            config=cfg,
        )
        assert "-p" in cmd
        p_idx = cmd.index("-p")
        assert cmd[p_idx + 1] == "restrict_solving.solve_max_rooms=7"

    def test_omits_g_when_no_gin_configs(self, prep_mod, tmp_path):
        cfg = prep_mod.SceneGenConfig(name="t", gin_configs=(), gin_overrides=())
        cmd = prep_mod._build_generate_command(
            infinigen_python="/fake/python",
            output_folder=tmp_path / "out",
            seed=0,
            config=cfg,
        )
        assert "-g" not in cmd


class TestPredefinedFloorPlanPresets:
    """The genuine single-/two-room presets pin an EXACT room count via a
    predefined floor-plan contour (``Solver.floor_plan``), not the constraint
    solver — which can only cap *furnished* rooms, never the floor count."""

    def test_true_presets_registered(self, prep_mod):
        assert "true_singleroom" in prep_mod.PRESETS
        assert "true_tworoom" in prep_mod.PRESETS
        assert prep_mod.PRESETS["true_singleroom"].max_rooms == 1
        assert prep_mod.PRESETS["true_tworoom"].max_rooms == 2

    @pytest.mark.parametrize(
        "preset_name,expected_json",
        [
            ("true_singleroom", "true_singleroom.json"),
            ("true_tworoom", "true_tworoom.json"),
        ],
    )
    def test_injects_absolute_solver_floor_plan(
        self, prep_mod, tmp_path, preset_name, expected_json
    ):
        cfg = prep_mod.PRESETS[preset_name]
        cmd = prep_mod._build_generate_command(
            infinigen_python="/fake/python",
            output_folder=tmp_path / "out",
            seed=0,
            config=cfg,
        )
        p_values = cmd[cmd.index("-p") + 1 :]
        floor_plan = [v for v in p_values if v.startswith("Solver.floor_plan=")]
        assert len(floor_plan) == 1, (
            f"expected exactly one Solver.floor_plan binding, got {floor_plan}"
        )
        raw = floor_plan[0].split("=", 1)[1].strip('"')
        path = Path(raw)
        assert path.is_absolute(), (
            "Solver.floor_plan must be absolute: the coarse-gen subprocess runs "
            "with cwd=INFINIGEN_ROOT, so a relative path resolves against the "
            "vendored Infinigen tree, not this repo."
        )
        assert path.name == expected_json
        assert path.is_file(), f"preset points at a missing floor plan: {path}"
        # Shipped under a git-tracked source path, NOT the transient corpus dir.
        assert "floor_plans" in path.parts
        assert "Assets" not in path.parts and "generated" not in path.parts
        # Floor-plan injection must not displace the other overrides.
        assert any("terrain_enabled=False" in v for v in p_values)
        assert any("RoomConstants.n_stories=1" in v for v in p_values)
        assert f"restrict_solving.solve_max_rooms={cfg.max_rooms}" in p_values

    @pytest.mark.parametrize(
        "preset_name,expected_rooms",
        [("true_singleroom", 1), ("true_tworoom", 2)],
    )
    def test_floor_plan_json_valid(self, prep_mod, preset_name, expected_rooms):
        cfg = prep_mod.PRESETS[preset_name]
        path = prep_mod._resolve_floor_plan_path(cfg.floor_plan_file)
        plan = json.loads(path.read_text())
        rooms = plan["rooms"]
        assert len(rooms) == expected_rooms, (
            f"{preset_name} must define exactly {expected_rooms} room(s)"
        )
        # Room keys are Infinigen's "<semantics>_<level>/<n>" form; a furnishable
        # LivingRoom must be present so the scene yields groundable objects.
        room_types = {k.split("_")[0] for k in rooms}
        assert "living-room" in room_types, (
            f"{preset_name} must contain a furnishable living-room, got {room_types}"
        )
        # Every shape is an eval'd shapely literal string (as Infinigen expects).
        for room in rooms.values():
            assert isinstance(room["shape"], str)
            assert room["shape"].startswith("shapely.")

    def test_missing_floor_plan_fails_loud(self, prep_mod):
        with pytest.raises(RuntimeError, match="not found under"):
            prep_mod._resolve_floor_plan_path("does_not_exist.json")


class TestBuildExportCommand:
    def test_includes_omniverse_and_usdc(self, prep_mod, tmp_path):
        cmd = prep_mod._build_export_command(
            infinigen_python="/fake/python",
            input_folder=tmp_path / "in",
            output_folder=tmp_path / "out",
            texture_resolution=512,
        )
        assert cmd[0] == "/fake/python"
        assert cmd[1:3] == ["-m", "infinigen.tools.export"]
        assert "--omniverse" in cmd
        assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "usdc"
        assert "-r" in cmd and cmd[cmd.index("-r") + 1] == "512"


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
            preset_name="high_quality_dgx",
        )
        assert len(imported) == 1
        assert (output / "kitchen_01" / "kitchen_01.usd").exists()
        config = json.loads((output / "kitchen_01" / "scene_config.json").read_text())
        assert config["name"] == "high_quality_dgx"

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
        assert "true_singleroom" in captured.out

    def test_presets_json(self, prep_mod, capsys):
        rc = prep_mod.main(["presets", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        payload = json.loads(captured.out)
        assert "high_quality_dgx" in payload
        assert "gin_configs" in payload["high_quality_dgx"]


class TestResolveInfinigenPython:
    def test_reads_from_env_var(self, prep_mod, monkeypatch, tmp_path):
        fake = tmp_path / "python"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("STRAFER_INFINIGEN_PYTHON", str(fake))
        assert prep_mod._resolve_infinigen_python() == str(fake)

    def test_missing_env_var_raises(self, prep_mod, monkeypatch):
        monkeypatch.delenv("STRAFER_INFINIGEN_PYTHON", raising=False)
        with pytest.raises(RuntimeError, match="STRAFER_INFINIGEN_PYTHON is not set"):
            prep_mod._resolve_infinigen_python()

    def test_nonexistent_env_var_raises(self, prep_mod, monkeypatch):
        monkeypatch.setenv("STRAFER_INFINIGEN_PYTHON", "/definitely/not/here/python")
        with pytest.raises(RuntimeError, match="non-existent path"):
            prep_mod._resolve_infinigen_python()


class TestResolveIsaaclabPython:
    def test_reads_from_env_var(self, prep_mod, monkeypatch, tmp_path):
        fake = tmp_path / "python"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("STRAFER_ISAACLAB_PYTHON", str(fake))
        assert prep_mod._resolve_isaaclab_python() == str(fake)

    def test_missing_env_var_raises(self, prep_mod, monkeypatch):
        monkeypatch.delenv("STRAFER_ISAACLAB_PYTHON", raising=False)
        with pytest.raises(RuntimeError, match="STRAFER_ISAACLAB_PYTHON is not set"):
            prep_mod._resolve_isaaclab_python()

    def test_nonexistent_env_var_raises(self, prep_mod, monkeypatch):
        monkeypatch.setenv("STRAFER_ISAACLAB_PYTHON", "/definitely/not/here/python")
        with pytest.raises(RuntimeError, match="non-existent path"):
            prep_mod._resolve_isaaclab_python()


class TestResolveInfinigenRoot:
    def test_reads_from_env_var(self, prep_mod, monkeypatch, tmp_path):
        monkeypatch.setenv("INFINIGEN_ROOT", str(tmp_path))
        assert prep_mod._resolve_infinigen_root() == tmp_path

    def test_missing_env_raises(self, prep_mod, monkeypatch):
        monkeypatch.delenv("INFINIGEN_ROOT", raising=False)
        with pytest.raises(RuntimeError, match="INFINIGEN_ROOT is not set"):
            prep_mod._resolve_infinigen_root()

    def test_nonexistent_path_raises(self, prep_mod, monkeypatch):
        monkeypatch.setenv("INFINIGEN_ROOT", "/definitely/not/here/infinigen")
        with pytest.raises(RuntimeError, match="not a directory"):
            prep_mod._resolve_infinigen_root()
