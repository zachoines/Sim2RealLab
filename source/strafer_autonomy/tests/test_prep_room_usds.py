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


class TestQualityTiersAndLayout:
    def test_quality_tiers_valid(self, prep_mod):
        for name, tier in prep_mod.QUALITY_TIERS.items():
            assert tier.texture_resolution > 0, f"{name} nonpositive texture_resolution"
            assert tier.object_density in prep_mod._OBJECT_DENSITY_STEPS

    def test_build_scene_config_applies_tier_defaults(self, prep_mod):
        tier = prep_mod.QUALITY_TIERS["low"]
        cfg = prep_mod.build_scene_config(
            name=None, rooms=None, quality="low", texture_resolution=None,
            object_density=None, geometry_detail=None, max_rooms=5, seed_base=0,
        )
        assert cfg.texture_resolution == tier.texture_resolution
        assert cfg.object_density == tier.object_density
        assert cfg.geometry_detail == "standard"  # displacement is opt-in
        assert cfg.name == "organic_low"  # derived from rooms + quality
        # provenance stays JSON round-trippable
        decoded = json.loads(json.dumps(cfg.to_dict()))
        assert decoded["name"] == "organic_low"
        assert decoded["rooms"] is None

    def test_explicit_knobs_override_tier(self, prep_mod):
        cfg = prep_mod.build_scene_config(
            name="x", rooms=("living-room",), quality="high", texture_resolution=128,
            object_density="low", geometry_detail="displacement", max_rooms=5, seed_base=3,
        )
        assert cfg.texture_resolution == 128
        assert cfg.object_density == "low"
        assert cfg.geometry_detail == "displacement"
        assert cfg.rooms == ("living-room",)
        assert cfg.random_seed_base == 3
        assert cfg.name == "x"

    def test_derived_names(self, prep_mod):
        assert prep_mod._default_scene_name(None, "high") == "organic_high"
        assert prep_mod._default_scene_name(("living-room", "kitchen"), "low") == "2room_low"


class TestResolveRooms:
    def test_explicit_list_wins(self, prep_mod):
        assert prep_mod.resolve_rooms(
            rooms=["living-room", "kitchen"], num_rooms=None, room_types=None
        ) == ("living-room", "kitchen")

    def test_duplicates_allowed(self, prep_mod):
        # adversarial: multiple rooms of the same class
        assert prep_mod.resolve_rooms(
            rooms=["bedroom", "bedroom"], num_rooms=None, room_types=None
        ) == ("bedroom", "bedroom")

    def test_round_robin(self, prep_mod):
        assert prep_mod.resolve_rooms(
            rooms=None, num_rooms=3, room_types=["living-room", "bedroom"]
        ) == ("living-room", "bedroom", "living-room")

    def test_round_robin_defaults_to_living_room(self, prep_mod):
        assert prep_mod.resolve_rooms(rooms=None, num_rooms=2, room_types=None) == (
            "living-room",
            "living-room",
        )

    def test_organic_when_unset(self, prep_mod):
        assert prep_mod.resolve_rooms(rooms=None, num_rooms=None, room_types=None) is None

    def test_rejects_non_furnishable(self, prep_mod):
        with pytest.raises(ValueError, match="not furnishable"):
            prep_mod.resolve_rooms(rooms=["closet"], num_rooms=None, room_types=None)

    def test_rejects_zero_num_rooms(self, prep_mod):
        with pytest.raises(ValueError, match=">= 1"):
            prep_mod.resolve_rooms(rooms=None, num_rooms=0, room_types=None)


class TestFloorPlanTiler:
    """The parametric tiler builds a PredefinedFloorPlanSolver contour for an
    exact room list (any count, duplicates OK) — replacing hand-authored JSONs."""

    @pytest.mark.parametrize(
        "rooms,n",
        [
            (["living-room"], 1),
            (["living-room", "kitchen"], 2),
            (["bedroom", "bedroom", "living-room"], 3),
        ],
    )
    def test_room_count_types_and_keys(self, prep_mod, rooms, n):
        fp = prep_mod._build_floor_plan(rooms)
        keys = list(fp["rooms"])
        assert len(keys) == n
        # keys are "<semantics>_<level>/<n>"; duplicate types get distinct n
        assert [k.split("_")[0] for k in keys] == rooms
        assert len(set(keys)) == n
        # every shape is a shapely-expression string (Infinigen evals it)
        for section in fp.values():
            for entry in section.values():
                assert isinstance(entry["shape"], str)
                assert entry["shape"].startswith("shapely.")

    def test_two_room_connectivity(self, prep_mod):
        fp = prep_mod._build_floor_plan(["living-room", "kitchen"])
        assert len(fp["doors"]) == 1  # one connecting door
        assert fp["interiors"] == {}  # solid shared wall; only the door connects the rooms
        assert fp["entrance"] and fp["windows"]  # exterior openings

    def test_single_room_has_no_interior_walls(self, prep_mod):
        fp = prep_mod._build_floor_plan(["living-room"])
        assert fp["doors"] == {}
        assert fp["interiors"] == {}
        assert fp["entrance"] and fp["windows"]

    def test_empty_rooms_raises(self, prep_mod):
        with pytest.raises(ValueError, match="at least one room"):
            prep_mod._build_floor_plan([])


class TestBuildGenerateCommand:
    def _cmd(self, prep_mod, cfg, tmp_path, floor_plan_path=None):
        return prep_mod._build_generate_command(
            infinigen_python="/fake/python",
            output_folder=tmp_path / "out",
            seed=42,
            config=cfg,
            floor_plan_path=floor_plan_path,
        )

    def _organic(self, prep_mod, quality="high", **kw):
        return prep_mod.build_scene_config(
            name="c", rooms=None, quality=quality,
            texture_resolution=kw.get("tex"), object_density=kw.get("dens"),
            geometry_detail=kw.get("geo"), max_rooms=kw.get("mr", 5), seed_base=0,
        )

    def test_organic_command_shape(self, prep_mod, tmp_path):
        cmd = self._cmd(prep_mod, self._organic(prep_mod, mr=5), tmp_path)
        assert cmd[0] == "/fake/python"
        assert cmd[1:3] == ["-m", "infinigen_examples.generate_indoors"]
        assert cmd[cmd.index("--seed") + 1] == "42"
        assert cmd[cmd.index("--task") + 1] == "coarse"
        assert cmd[cmd.index("-g") + 1] == "base_indoors"
        p = cmd[cmd.index("-p") + 1 :]
        assert "restrict_solving.solve_max_rooms=5" in p
        assert "compose_indoors.terrain_enabled=False" in p
        # organic mode -> no predefined floor plan, no single-story pin
        assert not any(v.startswith("Solver.floor_plan") for v in p)
        assert not any("n_stories" in v for v in p)

    def test_object_density_scales_solve_steps(self, prep_mod, tmp_path):
        hi = self._cmd(prep_mod, self._organic(prep_mod, quality="high"), tmp_path)
        lo = self._cmd(prep_mod, self._organic(prep_mod, quality="low"), tmp_path)
        assert "compose_indoors.solve_steps_large=300" in hi
        assert "compose_indoors.solve_steps_small=50" in hi
        assert "compose_indoors.solve_steps_large=100" in lo
        assert "compose_indoors.solve_steps_small=5" in lo

    def test_geometry_displacement_opt_in(self, prep_mod, tmp_path):
        std = self._cmd(prep_mod, self._organic(prep_mod, geo="standard"), tmp_path)
        disp = self._cmd(prep_mod, self._organic(prep_mod, geo="displacement"), tmp_path)
        assert not any("DISPLACEMENT" in v for v in std)
        assert any("DISPLACEMENT" in v for v in disp)
        assert "compose_indoors.enable_ocmesh_room=True" in disp

    def test_omits_g_when_no_gin_configs(self, prep_mod, tmp_path):
        cfg = prep_mod.SceneGenConfig(name="t", gin_configs=(), gin_overrides=())
        assert "-g" not in self._cmd(prep_mod, cfg, tmp_path)


class TestTiledCommandBuild:
    """A tiled config (rooms set) pins the exact layout via a per-scene
    ``Solver.floor_plan`` written by the tiler, furnishing all rooms."""

    @pytest.mark.parametrize(
        "rooms",
        [("living-room",), ("living-room", "kitchen"), ("bedroom", "bedroom")],
    )
    def test_injects_absolute_floor_plan(self, prep_mod, tmp_path, rooms):
        cfg = prep_mod.build_scene_config(
            name="t", rooms=rooms, quality="low", texture_resolution=None,
            object_density=None, geometry_detail=None, max_rooms=5, seed_base=0,
        )
        # mirror generate_scenes: the tiler writes the per-scene contour
        fpp = tmp_path / "floor_plan.json"
        fpp.write_text(json.dumps(prep_mod._build_floor_plan(cfg.rooms)))
        cmd = prep_mod._build_generate_command(
            infinigen_python="/fake/python", output_folder=tmp_path / "out",
            seed=0, config=cfg, floor_plan_path=fpp,
        )
        p = cmd[cmd.index("-p") + 1 :]
        binding = [v for v in p if v.startswith("Solver.floor_plan=")]
        assert len(binding) == 1
        path = Path(binding[0].split("=", 1)[1].strip('"'))
        # absolute — the subprocess runs cwd=INFINIGEN_ROOT, so a relative path
        # would resolve against the vendored Infinigen tree, not the scene dir
        assert path.is_absolute()
        assert path == fpp.resolve()
        assert "RoomConstants.n_stories=1" in p  # predefined path needs one story
        assert f"restrict_solving.solve_max_rooms={len(rooms)}" in p  # furnish all

    def test_tiled_without_floor_plan_path_raises(self, prep_mod, tmp_path):
        cfg = prep_mod.build_scene_config(
            name="t", rooms=("living-room",), quality="low", texture_resolution=None,
            object_density=None, geometry_detail=None, max_rooms=5, seed_base=0,
        )
        with pytest.raises(ValueError, match="requires floor_plan_path"):
            prep_mod._build_generate_command(
                infinigen_python="/fake/python", output_folder=tmp_path / "out",
                seed=0, config=cfg, floor_plan_path=None,
            )


class TestGenerateScenes:
    """Drive generate_scenes end-to-end, Kit-free: stub the env resolvers + the
    subprocess so it bails right after Step 0 (the per-scene floor-plan write),
    pinning both the write branch and the ``scene_<name>_<i>_seed<s>`` derivation
    that bridge_harness_smoke.py:_DEFAULT_SCENE couples to."""

    def _stub_env(self, prep_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(prep_mod, "validate_required_env_for_generate", lambda **k: None)
        monkeypatch.setattr(prep_mod, "_resolve_infinigen_python", lambda: "/fake/py")
        monkeypatch.setattr(prep_mod, "_resolve_infinigen_root", lambda: tmp_path)
        # bail after Step 0/1 with a nonzero rc, before any real Kit/Infinigen call
        monkeypatch.setattr(prep_mod, "_run_subprocess", lambda cmd, cwd=None: (1, "stub"))

    def test_tiled_writes_floor_plan_and_derives_name(self, prep_mod, tmp_path, monkeypatch):
        self._stub_env(prep_mod, tmp_path, monkeypatch)
        cfg = prep_mod.build_scene_config(
            name="singleroom", rooms=("living-room",), quality="low",
            texture_resolution=None, object_density=None, geometry_detail=None,
            max_rooms=5, seed_base=0,
        )
        prep_mod.generate_scenes(
            config=cfg, num_scenes=1, output_dir=tmp_path,
            author_scene_metadata=False, validate_connectivity=False,
        )
        fp = tmp_path / "scene_singleroom_000_seed0" / "floor_plan.json"
        assert fp.is_file()  # tiled config writes the per-scene contour (Step 0)
        assert len(json.loads(fp.read_text())["rooms"]) == 1  # exactly one room

    def test_organic_skips_floor_plan(self, prep_mod, tmp_path, monkeypatch):
        self._stub_env(prep_mod, tmp_path, monkeypatch)
        cfg = prep_mod.build_scene_config(
            name="organic", rooms=None, quality="low",
            texture_resolution=None, object_density=None, geometry_detail=None,
            max_rooms=5, seed_base=0,
        )
        prep_mod.generate_scenes(
            config=cfg, num_scenes=1, output_dir=tmp_path,
            author_scene_metadata=False, validate_connectivity=False,
        )
        assert not (tmp_path / "scene_organic_000_seed0" / "floor_plan.json").exists()


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
        imported = prep_mod.ingest_external_usd(source_dir=source, output_dir=output)
        assert len(imported) == 1
        assert (output / "kitchen_01" / "kitchen_01.usd").exists()
        config = json.loads((output / "kitchen_01" / "scene_config.json").read_text())
        # default provenance is a generic organic config
        assert config["name"] == "ingested"
        assert config["rooms"] is None

    def test_stamps_supplied_config(self, prep_mod, tmp_path):
        source = tmp_path / "external"
        scene = source / "kitchen_01"
        scene.mkdir(parents=True)
        (scene / "kitchen_01.usd").write_text("USD")
        output = tmp_path / "assets"
        cfg = prep_mod.build_scene_config(
            name="ingested", rooms=None, quality="low", texture_resolution=None,
            object_density=None, geometry_detail=None, max_rooms=5, seed_base=0,
        )
        prep_mod.ingest_external_usd(source_dir=source, output_dir=output, config=cfg)
        stamped = json.loads((output / "kitchen_01" / "scene_config.json").read_text())
        assert stamped["texture_resolution"] == 512  # low tier

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
    def test_info_lists_dimensions(self, prep_mod, capsys):
        rc = prep_mod.main(["info"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "high" in captured.out and "low" in captured.out  # quality tiers
        assert "living-room" in captured.out  # furnishable rooms
        assert "displacement" in captured.out  # geometry detail

    def test_info_json(self, prep_mod, capsys):
        rc = prep_mod.main(["info", "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        payload = json.loads(captured.out)
        assert "quality_tiers" in payload
        assert "living-room" in payload["furnishable_rooms"]
        assert set(payload["object_density"]) == {"high", "low"}

    def test_generate_rejects_non_furnishable_room(self, prep_mod, capsys):
        # argparse choices reject it before resolve_rooms even runs
        with pytest.raises(SystemExit):
            prep_mod.main(["generate", "--rooms", "closet", "--output", "/tmp/x"])


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
