"""RTAB-Map database keying for slam.launch.py.

A database's key is the scene key (``task_id`` / ``scene_key``), ``@`` and the
camera mount. Left at ``DEFAULT_DB_ARG``, ``database_path`` becomes
``~/.ros/rtabmap_<key>.db``, and a ``<db>.scene.json`` sidecar records the key.
``claim_database`` refuses an existing database recorded under another key, or
none.

``ros2 run strafer_slam database_path`` prints the database a launch with the
same environment uses, and claims, writes and checks nothing.
"""

import argparse
import datetime
import json
import os
import re

from strafer_shared.constants import CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z

# Every caller in strafer_bringup spells its database_path default exactly like
# this. Receiving this value, or an empty one, means nobody chose a path and the
# key derives one; any other value is the path. It names no database itself.
DEFAULT_DB_ARG = "~/.ros/rtabmap.db"

# The scene key's two halves default from these, in the launch and the resolver.
TASK_ID_ENV = "STRAFER_SLAM_TASK_ID"
SCENE_TOKEN_ENV = "STRAFER_SLAM_SCENE_TOKEN"

# Sits beside the database so deleting the db (or its volume) takes both.
SIDECAR_SUFFIX = ".scene.json"

# base_link -> d555_link (m): the URDF's d555_mount origin, from the same
# constants. Every database is keyed to it.
CAMERA_MOUNT = (CAMERA_OFFSET_X, CAMERA_OFFSET_Y, CAMERA_OFFSET_Z)


def slug(value: str) -> str:
    """Filename-safe form of a task id / scene token."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")


def scene_key(task_id: str, scene_token: str) -> str:
    """Combined scene key; empty when neither half is set."""
    return "_".join(p for p in (slug(task_id), slug(scene_token)) if p)


def mount_tag(mount) -> str:
    """``<x>_<y>_<z>``: a camera mount in millimetres to 0.1 mm, without ``.0``."""
    def mm(metres):
        tenths = round(metres * 10000)
        whole, tenth = divmod(abs(tenths), 10)
        return ("-" if tenths < 0 else "") + str(whole) + (f".{tenth}" if tenth else "")
    return "_".join(mm(v) for v in mount)


def database_key(scene: str, mount) -> str:
    """``<scene_key>@<mount tag>``: the key a database is recorded under.

    ``slug`` never emits ``@``, so a scene key alone never equals a database key.
    """
    return f"{scene}@{mount_tag(mount)}"


def resolve_database_path(database_path: str, key: str) -> str:
    """Expanded db path, derived from the key when the caller chose no path."""
    if database_path.strip() in ("", DEFAULT_DB_ARG):
        database_path = f"~/.ros/rtabmap_{key}.db"
    return os.path.expanduser(database_path.strip())


def recorded_key(sidecar: str):
    """The key a sidecar records; None when it is missing, unreadable or holds none."""
    try:
        with open(sidecar, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    key = record.get("key") if isinstance(record, dict) else None
    return key if isinstance(key, str) else None


def write_sidecar(sidecar: str, record: dict) -> None:
    """Replace the sidecar by one synced rename, so a crash leaves a whole record."""
    os.makedirs(os.path.dirname(sidecar) or ".", exist_ok=True)
    partial = sidecar + ".tmp"
    with open(partial, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, sidecar)


def claim_database(db_path: str, key: str, *, deleting: bool) -> None:
    """Refuse an existing database not recorded under ``key``; else claim it.

    A sidecar that is missing, unreadable or holds no key records no key, so
    the map may be of another scene or from another camera mount. ``deleting``
    is True when ``rtabmap_args`` carries ``-d`` (rtabmap wipes the db at
    start), so there is no map left to protect against. A matching record is
    left as written.
    """
    sidecar = db_path + SIDECAR_SUFFIX
    recorded = recorded_key(sidecar)
    if recorded == key:
        return

    if not deleting and os.path.isfile(db_path):
        held = f"key {recorded!r}" if recorded is not None else "no readable key"
        raise RuntimeError(
            f"RTAB-Map database {db_path!r} has {held} in {sidecar!r}, but this "
            f"launch's key is {key!r} (scene key @ camera mount in mm). Loading "
            "it could publish a map of a DIFFERENT scene, or one made from "
            "another camera mount, and every Nav2 /plan derived from it would "
            "be meaningless. Refusing to start. Pick one:\n"
            "  - other scene:   STRAFER_SLAM_SCENE_TOKEN=<token>   (db becomes "
            "~/.ros/rtabmap_<key>.db)\n"
            "  - explicit db:   database_path:=<other path>\n"
            "  - wipe & remap:  rtabmap_args:=-d   (erases this database)\n"
            "  - it really IS this scene at this mount: set the matching "
            "task_id / scene_key. A key without '@' records no camera mount "
            "and matches no launch."
        )

    # Claimed before rtabmap starts. A database left without this record is
    # refused at its next launch, so a claim that cannot be written stops here.
    record = {
        "key": key,
        "task_id": os.environ.get(TASK_ID_ENV, ""),
        "scene_token": os.environ.get(SCENE_TOKEN_ENV, ""),
        "database_path": db_path,
        "claimed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "written_by": "strafer_slam/slam.launch.py",
    }
    try:
        write_sidecar(sidecar, record)
    except OSError as exc:
        raise RuntimeError(
            f"Could not write the scene sidecar {sidecar!r} ({exc}). Without it "
            f"RTAB-Map database {db_path!r} is refused at its next launch, so "
            "this launch does not start RTAB-Map on it."
        ) from exc


def main(argv=None) -> int:
    """Print the database slam.launch.py uses with this environment."""
    parser = argparse.ArgumentParser(
        description=(
            "Print the RTAB-Map database slam.launch.py uses with the same "
            "environment and arguments. Claims, writes and checks nothing."
        ),
    )
    parser.add_argument(
        "--task-id", default=os.environ.get(TASK_ID_ENV, ""),
        help=f"scene-key half 1 (default: ${TASK_ID_ENV})",
    )
    parser.add_argument(
        "--scene-key", default=os.environ.get(SCENE_TOKEN_ENV, ""),
        help=f"scene-key half 2 (default: ${SCENE_TOKEN_ENV})",
    )
    parser.add_argument(
        "--database-path", default=DEFAULT_DB_ARG,
        help="the launch's database_path (default: %(default)s, which derives "
             "the name from the key)",
    )
    parser.add_argument(
        "--key", action="store_true",
        help="print 'database=<path> key=<key>', the launch's own line",
    )
    args = parser.parse_args(argv)

    key = database_key(scene_key(args.task_id, args.scene_key), CAMERA_MOUNT)
    path = resolve_database_path(args.database_path, key)
    print(f"database={path} key={key}" if args.key else path)
    return 0
