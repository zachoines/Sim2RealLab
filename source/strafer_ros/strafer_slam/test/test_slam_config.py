"""Tests for strafer_slam configuration and launch files."""

import datetime
import json
import os
import shutil
import subprocess
import types
import xml.etree.ElementTree as ET
import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from strafer_shared.constants import (
    CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z, DEPTH_MIN, DEPTH_MAX,
    FRAME_BASE_LINK, FRAME_D555_LINK, MAP_RESOLUTION,
)
from strafer_slam import database


@pytest.fixture
def pkg_dir():
    return get_package_share_directory("strafer_slam")


class TestRtabmapParams:
    """Validate rtabmap_params.yaml is well-formed and contains key settings."""

    @pytest.fixture
    def params(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "rtabmap_params.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "rtabmap_params.yaml")
        assert os.path.isfile(path)

    def test_detection_rate_is_set(self, params):
        """Detection rate should be ≤ 5 Hz for Jetson."""
        rate = params["Rtabmap/DetectionRate"]
        assert 0 < float(rate) <= 5

    def test_slam_2d_enabled(self, params):
        """Ground robot — 2D SLAM should be enforced."""
        assert params["Optimizer/Slam2D"] == "true"
        assert params["Reg/Force3DoF"] == "true"

    def test_grid_cell_size(self, params):
        cell = float(params["Grid/CellSize"])
        assert 0.01 <= cell <= 0.10

    def test_grid_range_matches_constants(self, params):
        """Grid range defaults must match constants.py."""
        assert float(params["Grid/RangeMin"]) == DEPTH_MIN
        assert float(params["Grid/RangeMax"]) == DEPTH_MAX

    def test_grid_cell_size_matches_constants(self, params):
        assert float(params["Grid/CellSize"]) == MAP_RESOLUTION

    def test_feature_type_is_orb(self, params):
        """ORB is fastest on Jetson (Type 8)."""
        ftype = params["Vis/FeatureType"]
        assert str(ftype) == "8"

    def test_ray_tracing_enabled(self, params):
        """Grid/RayTracing must be true so the depth-based occupancy
        grid marks cells along each ray as free, not just the obstacle
        endpoint. Without it, the angular spacing between depth pixels
        at 5-6m range exceeds Grid/CellSize and the resulting static
        map alternates known/unknown in stripes — Nav2's planner then
        threads a wavy path through the gaps, which the controller
        chases laterally.
        """
        assert params["Grid/RayTracing"] == "true"


class TestPointcloudToLaserscanParams:
    """Validate pointcloud_to_laserscan.yaml.

    The Z-axis ground filter is the central change vs the previous
    depthimage_to_laserscan setup — it kills the phantom 3.5 m arc that
    came from per-column min depth catching floor pixels.
    """

    @pytest.fixture
    def params(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "pointcloud_to_laserscan.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "pointcloud_to_laserscan.yaml")
        assert os.path.isfile(path)

    def test_target_frame_is_base_link(self, params):
        """Cloud must be transformed to base_link so the Z filter is in
        the chassis frame (true vertical), not the camera optical frame
        (where Z is forward).
        """
        frame = params["pointcloud_to_laserscan"]["ros__parameters"]["target_frame"]
        assert frame == "base_link"

    def test_height_filter_excludes_floor(self, params):
        """min_height must be > 0 so the floor in base_link Z=0 is filtered.
        That's the whole point of the swap.
        """
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["min_height"]) > 0.0

    def test_height_filter_within_body(self, params):
        """max_height should cap at roughly the chassis body height so
        ceiling fixtures and people heads don't end up in /scan as
        obstacles the robot won't collide with.
        """
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["max_height"]) > float(p["min_height"])
        assert float(p["max_height"]) <= 0.5

    def test_range_min_less_than_max(self, params):
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert p["range_min"] < p["range_max"]

    def test_range_matches_constants(self, params):
        """YAML defaults must stay in sync with constants.py."""
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["range_min"]) == DEPTH_MIN
        assert float(p["range_max"]) == DEPTH_MAX

    def test_angle_range_symmetric_about_zero(self, params):
        """The forward-facing D555 produces a cone of returns symmetric
        about the camera's optical axis (= base_link X after the static
        TF). The scan sweep should be symmetric around 0 rad.
        """
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["angle_min"]) == -float(p["angle_max"])

    def test_angle_increment_positive(self, params):
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["angle_increment"]) > 0.0


def _load_launch(pkg_dir):
    import importlib.util

    path = os.path.join(pkg_dir, "launch", "slam.launch.py")
    spec = importlib.util.spec_from_file_location("slam_launch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The working tree's xacro: these tests do not rebuild strafer_description, so
# its share directory may hold an older copy.
XACRO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    "strafer_description", "urdf", "strafer.urdf.xacro",
)


def _run_launch_setup(mod, monkeypatch, home, **configs):
    """Run the launch's OpaqueFunction with HOME at ``home``; return its actions.

    The context starts from the launch's declared defaults, generated under the
    current environment, and ``configs`` override them.
    """
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument
    from launch.utilities import perform_substitutions

    monkeypatch.setenv("HOME", str(home))
    ctx = LaunchContext()
    for entity in mod.generate_launch_description().entities:
        if isinstance(entity, DeclareLaunchArgument):
            ctx.launch_configurations[entity.name] = perform_substitutions(
                ctx, entity.default_value
            )
    ctx.launch_configurations.update(configs)
    return mod._launch_setup(ctx)


def _notice(actions):
    """Text of the launch's first LogInfo, the database line."""
    from launch.actions import LogInfo

    first = next(a for a in actions if isinstance(a, LogInfo))
    return "".join(s.text for s in first.msg)


def _read(path):
    """The bytes at ``path``."""
    with open(path, "rb") as f:
        return f.read()


def _write(path, data=b""):
    """Write ``data`` at ``path``; an empty file stands in for a database."""
    with open(path, "wb") as f:
        f.write(data)


class TestLaunchFile:
    """Validate slam.launch.py exists and is importable."""

    def test_launch_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "slam.launch.py")
        assert os.path.isfile(path)

    def test_launch_generates_description(self, pkg_dir):
        """Import and call generate_launch_description to catch import errors."""
        mod = _load_launch(pkg_dir)
        ld = mod.generate_launch_description()
        assert ld is not None
        # 6 DeclareLaunchArguments (+task_id, +scene_key) + 1 OpaqueFunction
        assert len(ld.entities) >= 7

    def test_scene_key_args_declared(self, pkg_dir, monkeypatch):
        from launch.actions import DeclareLaunchArgument

        monkeypatch.delenv("STRAFER_SLAM_TASK_ID", raising=False)
        monkeypatch.delenv("STRAFER_SLAM_SCENE_TOKEN", raising=False)
        ld = _load_launch(pkg_dir).generate_launch_description()
        names = {
            e.name for e in ld.entities
            if isinstance(e, DeclareLaunchArgument)
        }
        assert {"task_id", "scene_key"} <= names

    def test_scene_key_args_default_from_env(self, pkg_dir, monkeypatch):
        from launch.actions import DeclareLaunchArgument

        monkeypatch.setenv(database.TASK_ID_ENV, "enrich")
        monkeypatch.setenv(database.SCENE_TOKEN_ENV, "run4")
        ld = _load_launch(pkg_dir).generate_launch_description()
        rendered = {
            e.name: "".join(getattr(s, "text", str(s)) for s in e.default_value)
            for e in ld.entities
            if isinstance(e, DeclareLaunchArgument) and e.name in ("task_id", "scene_key")
        }
        assert rendered == {"task_id": "enrich", "scene_key": "run4"}

    def test_database_path_default_derives_from_the_key(self, pkg_dir):
        # A container boot passes no database_path; this default derives the
        # database from the key.
        from launch.actions import DeclareLaunchArgument

        ld = _load_launch(pkg_dir).generate_launch_description()
        defaults = [
            "".join(s.text for s in e.default_value)
            for e in ld.entities
            if isinstance(e, DeclareLaunchArgument) and e.name == "database_path"
        ]
        assert defaults == [database.DEFAULT_DB_ARG]


class TestSceneKeyedDatabase:
    """RTAB-Map reloads its persisted database on start. A procedurally
    regenerated sim scene is a new layout on every restart, so reloading the
    previous run's map corrupts /rtabmap/map and every Nav2 /plan derived from
    it — and because the map is also the reference trajectory analyses are
    measured against, a wrong-scene map invalidates measurements rather than
    merely degrading navigation. A map made from another camera mount places
    its features against that mount's camera pose, so a database's key carries
    the mount too, and a database whose record holds no such key is refused
    rather than adopted.
    """

    def test_no_scene_key_names_the_database_for_the_mount(self):
        # The real robot sets no scene key; its database is named for the
        # camera mount, so another mount maps fresh under another name.
        key = database.database_key("", database.CAMERA_MOUNT)
        path = os.path.expanduser(f"~/.ros/rtabmap_{key}.db")
        assert database.scene_key("", "") == ""
        assert key.startswith("@")
        assert database.resolve_database_path(database.DEFAULT_DB_ARG, key) == path
        assert database.resolve_database_path("", key) == path
        assert path != os.path.expanduser("~/.ros/rtabmap.db")

    def test_key_derives_a_distinct_default_path(self):
        assert database.scene_key("ProcRoom-v0", "run4") == "ProcRoom-v0_run4"
        key = database.database_key("ProcRoom-v0_run4", database.CAMERA_MOUNT)
        assert key.startswith("ProcRoom-v0_run4@")
        assert database.resolve_database_path(
            "~/.ros/rtabmap.db", key
        ) == os.path.expanduser(f"~/.ros/rtabmap_{key}.db")

    def test_explicit_database_path_is_honoured_verbatim(self):
        # An explicit path outranks the derived name; the guard applies to it too.
        assert database.resolve_database_path(
            "/root/.ros/rtabmap_run3.db",
            database.database_key("run4", database.CAMERA_MOUNT),
        ) == "/root/.ros/rtabmap_run3.db"

    def test_scene_token_is_slugged_for_filenames(self):
        assert "/" not in database.scene_key("a/b c", "d:e")
        assert database.scene_key("  ", "  ") == ""
        # No scene key holds '@', the mount's separator.
        assert "@" not in database.scene_key("a@b", "c@d")

    def test_reusing_a_db_under_the_same_key_is_allowed(self, tmp_path):
        # `docker restart strafer_slam` after a /clock hitch is a documented
        # recovery; it must reload the map it was building and leave its
        # record as written.
        db = str(tmp_path / "rtabmap_run4.db")
        sidecar = db + database.SIDECAR_SUFFIX
        database.claim_database(db, "run4", deleting=False)
        _write(db)
        record = _read(sidecar)
        database.claim_database(db, "run4", deleting=False)
        assert _read(sidecar) == record

    def test_foreign_scene_db_is_refused(self, tmp_path):
        db = str(tmp_path / "rtabmap.db")
        sidecar = db + database.SIDECAR_SUFFIX
        database.claim_database(db, "run4", deleting=False)
        _write(db)
        record = _read(sidecar)
        with pytest.raises(RuntimeError) as exc:
            database.claim_database(db, "run5", deleting=False)
        assert _read(sidecar) == record
        msg = str(exc.value)
        # Fail-loud means the operator is told the ways out, not just "no".
        assert "run4" in msg and "run5" in msg
        assert "STRAFER_SLAM_SCENE_TOKEN" in msg
        assert "database_path:=" in msg
        assert "rtabmap_args:=-d" in msg

    def test_delete_flag_removes_the_database_before_the_claim(self, tmp_path):
        # rtabmap_args:=-d wipes the db, so the claim removes it and its record
        # first, recorded or not: no record outlives the map it was written for.
        db = str(tmp_path / "rtabmap.db")
        sidecar = db + database.SIDECAR_SUFFIX
        database.claim_database(db, "run4", deleting=False)
        _write(db, b"map")
        database.claim_database(db, "run5", deleting=True)
        assert not os.path.exists(db)
        assert json.loads(_read(sidecar))["key"] == "run5"
        _write(db, b"map")
        os.remove(sidecar)
        database.claim_database(db, "run6", deleting=True)
        assert not os.path.exists(db)
        assert json.loads(_read(sidecar))["key"] == "run6"

    def test_a_database_that_cannot_be_removed_stops_the_claim(self, tmp_path):
        db = str(tmp_path / "rtabmap.db")
        sidecar = db + database.SIDECAR_SUFFIX
        os.mkdir(db)
        database.write_sidecar(sidecar, {"key": "run4"})
        record = _read(sidecar)
        with pytest.raises(RuntimeError, match="could not be removed"):
            database.claim_database(db, "run5", deleting=True)
        assert os.path.isdir(db) and _read(sidecar) == record

    def test_missing_db_is_claimed_not_refused(self, tmp_path):
        # A stale sidecar with no database behind it is not a wrong-scene
        # hazard — there is nothing to load.
        db = str(tmp_path / "rtabmap.db")
        database.claim_database(db, "run4", deleting=False)
        database.claim_database(db, "run9", deleting=False)
        assert json.loads(_read(db + database.SIDECAR_SUFFIX))["key"] == "run9"

    def test_a_db_with_no_record_is_refused(self, tmp_path):
        # No sidecar says which scene or camera mount a database was made
        # under, so it is refused rather than adopted.
        db = str(tmp_path / "rtabmap.db")
        _write(db)
        key = database.database_key("", database.CAMERA_MOUNT)
        with pytest.raises(RuntimeError, match="no readable key"):
            database.claim_database(db, key, deleting=False)
        assert not os.path.exists(db + database.SIDECAR_SUFFIX)

    @pytest.mark.parametrize("scene", ["", "enrich_run4"], ids=["no-scene", "scene"])
    def test_a_record_without_a_mount_is_refused(self, tmp_path, scene):
        # A record whose key holds no camera mount matches no launch, so a map
        # made from another mount never reloads, whatever its scene.
        db = str(tmp_path / (f"rtabmap_{scene}.db" if scene else "rtabmap.db"))
        sidecar = db + database.SIDECAR_SUFFIX
        _write(db)
        _write(sidecar, json.dumps({
            "key": scene,
            "task_id": "",
            "scene_token": scene,
            "database_path": db,
            "claimed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "written_by": "strafer_slam/slam.launch.py",
        }, indent=2).encode())
        record = _read(sidecar)
        key = database.database_key(scene, database.CAMERA_MOUNT)
        with pytest.raises(RuntimeError):
            database.claim_database(db, key, deleting=False)
        assert _read(sidecar) == record

    @pytest.mark.parametrize("data", [
        pytest.param(b"", id="empty"),
        pytest.param(b"{", id="truncated"),
        pytest.param(b"\xff", id="not-utf8"),
        pytest.param(b"[]", id="list"),
        pytest.param(b'"run4"', id="string"),
        pytest.param(b"null", id="null"),
        pytest.param(b'{"task_id": ""}', id="no-key"),
        pytest.param(b'{"key": 4}', id="key-not-str"),
    ])
    def test_an_unusable_record_is_refused(self, tmp_path, data):
        # A record that cannot be read says nothing about the map's scene or mount.
        db = str(tmp_path / "rtabmap_run4.db")
        sidecar = db + database.SIDECAR_SUFFIX
        _write(db)
        _write(sidecar, data)
        key = database.database_key("run4", database.CAMERA_MOUNT)
        with pytest.raises(RuntimeError):
            database.claim_database(db, key, deleting=False)
        assert _read(sidecar) == data

    @pytest.mark.parametrize("dz", [0.02, 0.0001], ids=["20mm", "0.1mm"])
    def test_a_db_from_another_mount_is_refused(self, tmp_path, dz):
        x, y, z = database.CAMERA_MOUNT
        db = str(tmp_path / "rtabmap_run4.db")
        there = database.database_key("run4", (x, y, z + dz))
        here = database.database_key("run4", database.CAMERA_MOUNT)
        database.claim_database(db, there, deleting=False)
        _write(db)
        with pytest.raises(RuntimeError) as exc:
            database.claim_database(db, here, deleting=False)
        assert there in str(exc.value) and here in str(exc.value)

    def test_a_matching_record_needs_no_write(self, tmp_path, monkeypatch):
        # A restart under the same key reloads its map even when the sidecar
        # cannot be written.
        db = str(tmp_path / "rtabmap_run4.db")
        key = database.database_key("run4", database.CAMERA_MOUNT)
        database.claim_database(db, key, deleting=False)
        _write(db)

        def unwritable(sidecar, record):
            raise OSError("no space left on device")

        monkeypatch.setattr(database, "write_sidecar", unwritable)
        database.claim_database(db, key, deleting=False)

    def test_an_unwritable_sidecar_stops_the_claim(self, tmp_path):
        # A database created without its record would be refused at its next
        # launch, so a claim that cannot be written stops before RTAB-Map starts.
        db = str(tmp_path / "rtabmap_run4.db")
        sidecar = db + database.SIDECAR_SUFFIX
        os.mkdir(sidecar)
        with pytest.raises(RuntimeError) as exc:
            database.claim_database(db, "run4", deleting=False)
        assert sidecar in str(exc.value)
        assert not os.path.exists(db)

    def test_a_claim_writes_the_whole_record(self, tmp_path):
        db = str(tmp_path / "rtabmap_run4.db")
        sidecar = db + database.SIDECAR_SUFFIX
        database.claim_database(db, "run4", deleting=False)
        record = json.loads(_read(sidecar))
        assert isinstance(record, dict)
        assert record["key"] == "run4"
        assert record["database_path"] == db
        assert record["written_by"] == "strafer_slam/slam.launch.py"
        assert not os.path.exists(sidecar + ".tmp")

    def test_a_failed_write_leaves_the_previous_record(self, tmp_path, monkeypatch):
        # The record is written beside the sidecar and renamed over it, so a
        # write that fails part way leaves the previous record whole.
        db = str(tmp_path / "rtabmap_run4.db")
        sidecar = db + database.SIDECAR_SUFFIX
        database.claim_database(db, "run4", deleting=False)
        record = _read(sidecar)

        def torn(obj, f, **kwargs):
            f.write('{"key": "ru')
            raise OSError("no space left on device")

        monkeypatch.setattr(database.json, "dump", torn)
        with pytest.raises(RuntimeError, match="Could not write the scene sidecar"):
            database.claim_database(db, "run5", deleting=False)
        assert _read(sidecar) == record


class TestCameraMount:
    """A database's key carries the camera mount that places d555_link, so its
    name and its record change whenever the description's mount does.
    """

    def test_mount_derives_a_distinct_default_path(self):
        x, y, z = database.CAMERA_MOUNT
        here = database.resolve_database_path(
            database.DEFAULT_DB_ARG, database.database_key("", database.CAMERA_MOUNT)
        )
        other = database.resolve_database_path(
            database.DEFAULT_DB_ARG, database.database_key("", (x, y, z + 0.02))
        )
        assert here != other
        assert os.path.expanduser(database.DEFAULT_DB_ARG) not in (here, other)

    def test_mount_is_written_in_millimetres_to_a_tenth(self):
        assert database.mount_tag((0.1, -0.0123, 0.3)) == "100_-12.3_300"
        assert database.database_key("run4", (0.1, -0.0123, 0.3)) == "run4@100_-12.3_300"
        # Float noise either side and negative zero round away; one quantum re-keys.
        assert database.mount_tag((0.1, -0.0, 0.3 + 1e-9)) == "100_0_300"
        assert database.mount_tag((0.1, 0.0, 0.3 - 1e-9)) == "100_0_300"
        assert database.mount_tag((0.1, 0.0, 0.3001)) == "100_0_300.1"
        assert database.mount_tag((-1e-5, 0.0, 0.0)) == "0_0_0"

    def test_mount_is_the_camera_joint_origin(self):
        # The key's mount is d555_mount's whole origin and the joint reads no
        # other value; an input the key left out would reload a map made from
        # another mount.
        assert database.CAMERA_MOUNT == (CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z)
        joints = [
            j for j in ET.parse(XACRO_PATH).getroot().iter("joint")
            if j.find("child") is not None
            and j.find("child").get("link") == FRAME_D555_LINK
        ]
        assert [j.get("name") for j in joints] == ["d555_mount"]
        joint = joints[0]
        assert joint.get("type") == "fixed"
        assert joint.find("parent").get("link") == FRAME_BASE_LINK
        origin = joint.find("origin")
        assert origin.get("xyz").split() == ["${camera_x}", "${camera_y}", "${camera_z}"]
        assert origin.get("rpy").split() == ["0", "0", "0"]


class TestSceneKeyedLaunch:
    """The launch's OpaqueFunction derives, claims and refuses from the launch's
    declared defaults, as a container boot does, overridden by the arguments a
    case passes. HOME is a temporary directory, and the scene key's environment
    is empty unless a case sets it.
    """

    @pytest.fixture(autouse=True)
    def no_scene_env(self, monkeypatch):
        monkeypatch.delenv(database.TASK_ID_ENV, raising=False)
        monkeypatch.delenv(database.SCENE_TOKEN_ENV, raising=False)

    def test_launch_keys_through_the_database_module(self, pkg_dir):
        assert _load_launch(pkg_dir).database is database

    @pytest.mark.parametrize(
        "task_id, scene",
        [("", ""), ("", "run4"), ("enrich", "run4")],
        ids=["no-scene", "token", "task-and-token"],
    )
    def test_launch_maps_fresh_beside_databases_recorded_without_a_mount(
        self, pkg_dir, monkeypatch, tmp_path, task_id, scene
    ):
        # Databases recorded without a camera mount keep every name they have,
        # so a launch derives a name none of them has and maps fresh, leaving
        # them as they are.
        ros = tmp_path / ".ros"
        ros.mkdir()
        old_dbs = ["rtabmap.db", "rtabmap_run4.db", "rtabmap_enrich_run4.db"]
        old = {
            "rtabmap.db": b"",
            "rtabmap_run4.db": b"",
            "rtabmap_run4.db" + database.SIDECAR_SUFFIX: b'{"key": "run4"}',
            "rtabmap_enrich_run4.db": b"",
            "rtabmap_enrich_run4.db" + database.SIDECAR_SUFFIX: b'{"key": "enrich_run4"}',
        }
        for name, data in old.items():
            _write(str(ros / name), data)

        monkeypatch.setenv(database.TASK_ID_ENV, task_id)
        monkeypatch.setenv(database.SCENE_TOKEN_ENV, scene)
        actions = _run_launch_setup(_load_launch(pkg_dir), monkeypatch, tmp_path)

        key = database.database_key(database.scene_key(task_id, scene), database.CAMERA_MOUNT)
        db = str(ros / f"rtabmap_{key}.db")
        assert _notice(actions) == f"[strafer_slam] database={db} key={key}"
        assert os.path.basename(db) not in old_dbs
        assert json.loads(_read(db + database.SIDECAR_SUFFIX))["key"] == key
        assert {name: _read(str(ros / name)) for name in old} == old
        assert sorted(os.listdir(ros)) == sorted(
            [*old, os.path.basename(db) + database.SIDECAR_SUFFIX]
        )

    def test_a_restart_reloads_and_another_mount_maps_fresh(
        self, pkg_dir, monkeypatch, tmp_path
    ):
        # A restart reloads its map; recreated on other mount constants, the
        # launch maps fresh under another name and refuses the first database
        # at its path.
        mod = _load_launch(pkg_dir)
        key = database.database_key("", database.CAMERA_MOUNT)
        db = str(tmp_path / ".ros" / f"rtabmap_{key}.db")
        sidecar = db + database.SIDECAR_SUFFIX
        _run_launch_setup(mod, monkeypatch, tmp_path)
        _write(db)
        record = _read(sidecar)

        _run_launch_setup(mod, monkeypatch, tmp_path)
        assert _read(sidecar) == record

        x, y, z = database.CAMERA_MOUNT
        monkeypatch.setattr(database, "CAMERA_MOUNT", (x, y, z + 0.01))
        other_key = database.database_key("", database.CAMERA_MOUNT)
        other_db = str(tmp_path / ".ros" / f"rtabmap_{other_key}.db")
        notice = _notice(_run_launch_setup(mod, monkeypatch, tmp_path))
        assert other_db != db
        assert notice == f"[strafer_slam] database={other_db} key={other_key}"
        assert _read(db) == b"" and _read(sidecar) == record

        with pytest.raises(RuntimeError):
            _run_launch_setup(mod, monkeypatch, tmp_path, database_path=db)

    @pytest.mark.parametrize(
        "arg, refused",
        [("-d", False), ("--delete_db_on_start", True), ("--udebug", True)],
        ids=["d", "delete_db_on_start", "udebug"],
    )
    def test_only_the_short_delete_flag_removes_the_database(
        self, pkg_dir, monkeypatch, tmp_path, arg, refused
    ):
        # A flag the guard does not read leaves the database and the guard in
        # place, which refuses rather than loads.
        key = database.database_key("", database.CAMERA_MOUNT)
        db = str(tmp_path / ".ros" / f"rtabmap_{key}.db")
        os.mkdir(tmp_path / ".ros")
        _write(db, b"map")
        mod = _load_launch(pkg_dir)
        if refused:
            with pytest.raises(RuntimeError):
                _run_launch_setup(mod, monkeypatch, tmp_path, rtabmap_args=arg)
            assert _read(db) == b"map"
        else:
            _run_launch_setup(mod, monkeypatch, tmp_path, rtabmap_args=arg)
            assert not os.path.exists(db)
            assert json.loads(_read(db + database.SIDECAR_SUFFIX))["key"] == key

    def test_a_delete_launch_that_stops_after_the_claim_leaves_no_map(
        self, pkg_dir, monkeypatch, tmp_path
    ):
        # The launch claims, then loads RTAB-Map's parameters. One that stops
        # there with -d has removed the other mount's map already, so the next
        # launch without -d finds no map under the record to adopt.
        x, y, z = database.CAMERA_MOUNT
        key = database.database_key("", database.CAMERA_MOUNT)
        ros = tmp_path / ".ros"
        ros.mkdir()
        db = str(ros / f"rtabmap_{key}.db")
        _write(db, b"another mount's map")
        database.write_sidecar(
            db + database.SIDECAR_SUFFIX, {"key": database.database_key("", (x, y, z + 0.02))}
        )

        def stop(stream):
            raise RuntimeError("stopped after the claim")

        mod = _load_launch(pkg_dir)
        monkeypatch.setattr(mod, "yaml", types.SimpleNamespace(safe_load=stop))
        with pytest.raises(RuntimeError, match="stopped after the claim"):
            _run_launch_setup(mod, monkeypatch, tmp_path, rtabmap_args="-d")
        assert not os.path.exists(db)

        notice = _notice(_run_launch_setup(_load_launch(pkg_dir), monkeypatch, tmp_path))
        assert notice == f"[strafer_slam] database={db} key={key}"
        assert not os.path.exists(db)
        assert all(_read(str(ros / name)) != b"another mount's map" for name in os.listdir(ros))

    @pytest.mark.parametrize("localization", ["false", "true"], ids=["mapping", "localization"])
    def test_a_db_with_no_record_is_refused_in_either_mode(
        self, pkg_dir, monkeypatch, tmp_path, localization
    ):
        # Localization loads the whole saved map, so it is guarded as mapping is.
        key = database.database_key("", database.CAMERA_MOUNT)
        db = str(tmp_path / ".ros" / f"rtabmap_{key}.db")
        os.mkdir(tmp_path / ".ros")
        _write(db)
        with pytest.raises(RuntimeError, match="no readable key"):
            _run_launch_setup(
                _load_launch(pkg_dir), monkeypatch, tmp_path, localization=localization
            )
        assert os.listdir(tmp_path / ".ros") == [os.path.basename(db)]

    @pytest.mark.parametrize(
        "database_path, rtabmap_args",
        [(None, ""), (None, "-d"), ("~/maps/x.db", "")],
        ids=["derived", "derived-d", "explicit"],
    )
    def test_rtabmap_gets_the_claimed_database(
        self, pkg_dir, monkeypatch, tmp_path, database_path, rtabmap_args
    ):
        # The rtabmap node gets the path the guard claimed and the arguments it
        # read; None leaves database_path at the launch's default.
        mod = _load_launch(pkg_dir)
        nodes = []
        real_node = mod.Node

        def record_node(**kwargs):
            nodes.append(kwargs)
            return real_node(**kwargs)

        monkeypatch.setattr(mod, "Node", record_node)
        configs = {} if database_path is None else {"database_path": database_path}
        _run_launch_setup(mod, monkeypatch, tmp_path, rtabmap_args=rtabmap_args, **configs)

        key = database.database_key("", database.CAMERA_MOUNT)
        db = (str(tmp_path / ".ros" / f"rtabmap_{key}.db") if database_path is None
              else str(tmp_path / "maps" / "x.db"))
        (rtabmap,) = [k for k in nodes if k["package"] == "rtabmap_slam"]
        assert rtabmap["executable"] == "rtabmap"
        params = {}
        for p in rtabmap["parameters"]:
            params.update(p)
        assert os.path.isfile(db + database.SIDECAR_SUFFIX)
        assert params["database_path"] == db
        assert rtabmap["arguments"] == rtabmap_args.split()


class TestDatabaseResolver:
    """``ros2 run strafer_slam database_path`` prints the database a launch
    with the same environment uses, so a harness reads the name the launch
    derives rather than rebuilding it.
    """

    @pytest.fixture
    def env(self, monkeypatch, tmp_path):
        """HOME at ``tmp_path`` and a scene key; returns the key and its database."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("STRAFER_SLAM_TASK_ID", "enrich")
        monkeypatch.setenv("STRAFER_SLAM_SCENE_TOKEN", "run4")
        key = database.database_key("enrich_run4", database.CAMERA_MOUNT)
        return key, str(tmp_path / ".ros" / f"rtabmap_{key}.db")

    def test_prints_the_database_the_key_derives(self, env, capsys):
        key, db = env
        assert database.main([]) == 0
        assert capsys.readouterr().out == f"{db}\n"

    def test_flags_outrank_the_environment(self, env, tmp_path, capsys):
        database.main(["--task-id", "", "--scene-key", "run5"])
        key = database.database_key("run5", database.CAMERA_MOUNT)
        assert capsys.readouterr().out == f"{tmp_path / '.ros' / f'rtabmap_{key}.db'}\n"

    def test_an_explicit_path_is_printed_verbatim(self, env, capsys):
        database.main(["--database-path", "/root/.ros/rtabmap_run3.db"])
        assert capsys.readouterr().out == "/root/.ros/rtabmap_run3.db\n"

    def test_key_prints_the_launch_line(self, pkg_dir, env, monkeypatch, tmp_path, capsys):
        # The launch's declared defaults read the same environment.
        key, db = env
        actions = _run_launch_setup(_load_launch(pkg_dir), monkeypatch, tmp_path)
        database.main(["--key"])
        line = capsys.readouterr().out
        assert line == f"database={db} key={key}\n"
        assert _notice(actions) == f"[strafer_slam] {line.rstrip()}"

    def test_key_prints_the_launch_line_for_tokens_that_slug(
        self, pkg_dir, env, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setenv("STRAFER_SLAM_TASK_ID", "enrich v2")
        monkeypatch.setenv("STRAFER_SLAM_SCENE_TOKEN", "07:40")
        key = database.database_key(database.scene_key("enrich v2", "07:40"), database.CAMERA_MOUNT)
        actions = _run_launch_setup(_load_launch(pkg_dir), monkeypatch, tmp_path)
        database.main(["--key"])
        line = capsys.readouterr().out
        assert key.startswith("enrich-v2_07-40@")
        assert line == f"database={tmp_path / '.ros' / f'rtabmap_{key}.db'} key={key}\n"
        assert _notice(actions) == f"[strafer_slam] {line.rstrip()}"

    def test_it_claims_writes_and_checks_nothing(self, env, capsys):
        # A database with no record, which a launch refuses.
        key, db = env
        os.makedirs(os.path.dirname(db))
        _write(db)
        database.main([])
        assert capsys.readouterr().out == f"{db}\n"
        assert os.listdir(os.path.dirname(db)) == [os.path.basename(db)]

    @pytest.mark.skipif(shutil.which("ros2") is None, reason="needs the ros2 CLI")
    def test_runs_as_an_installed_console_script(self, env):
        key, db = env
        run = subprocess.run(
            ["ros2", "run", "strafer_slam", "database_path"],
            capture_output=True, text=True, timeout=120,
        )
        assert run.returncode == 0, run.stderr
        assert run.stdout == f"{db}\n"
