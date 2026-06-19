"""Jetson-free Kit smoke for the bridge harness capture path.

Boots the Capture-Bridge env headless, drives a short scripted ``/cmd_vel``
(forward + yaw) through the real :class:`SimInTheLoopHarness` orchestrator +
:class:`BridgeLeRobotRecorder` + :class:`StraferLeRobotWriter` with detections
on, then re-opens the written dataset and asserts it is well-formed. The drive
is brief by design — the smoke validates the capture path, not a room tour.

What this exercises (the surface this driver owns):
  - env reset / step with a normalized ``/cmd_vel`` action,
  - ``create() → add_frame() → save_episode() → finalize()`` writer lifecycle,
  - the perception camera's Replicator ``bbox_2d_tight`` annotator →
    ``observation.detections.*`` columns + ``meta/detection_labels.json``,
  - 16UC1 depth PNG sidecars,
  - the per-episode strafer extension columns + the discard path (a second
    mission the fake executor reports ``cancelled`` must not reach disk).

What it deliberately does NOT exercise (covered elsewhere — not a Jetson
substitute): the ROS 2 telemetry / camera publishers (``make sim-bridge``)
and the live autonomy executor driving real ``/cmd_vel`` (operator-run
multi-room acceptance). The action source here is a scripted stub, and no
``StraferAsyncPublisher`` is constructed.

Run (an Isaac Sim Kit conda env must be active — ``$ISAACLAB -p`` uses the active python):

    source env_setup.sh
    conda activate "$CONDA_ENV"
    $ISAACLAB -p source/strafer_lab/scripts/bridge_harness_smoke.py

or via the make target::

    make harness-smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_DEFAULT_SCENE = "scene_fast_singleroom_000_seed0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="Isaac-Strafer-Nav-Capture-Bridge-v0",
        help="Capture env carrying the perception + policy cameras.",
    )
    parser.add_argument(
        "--scene", default=_DEFAULT_SCENE,
        help="Scene name under Assets/generated/scenes/. Defaults to the "
             "light single-room scene so the smoke stays OOM-safe on the GB10.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Dataset root. Defaults to a timestamped path under /tmp.",
    )
    parser.add_argument(
        "--steps", type=int, default=150,
        help="Env steps for the kept mission (drives the camera sweep). The "
             "discarded mission runs a quarter of this.",
    )
    parser.add_argument(
        "--require-detections", action=argparse.BooleanOptionalAction, default=False,
        help="Treat an empty detection vocab as a hard failure. Default off "
             "(warn loudly) so a spawn that happens to face empty space does "
             "not flake the gate; turn on once the scene is trusted to be in "
             "the camera's view.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


_parser = _build_parser()
args = _parser.parse_args()
# Headless + cameras are non-negotiable for this smoke.
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Post-launch imports (need the Kit runtime active).
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import strafer_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from strafer_lab.sim_in_the_loop import (  # noqa: E402
    BridgeLeRobotRecorder,
    EpisodeMeta,
    HarnessConfig,
    MissionSpec,
    MissionStatus,
    SimInTheLoopHarness,
)
from strafer_lab.sim_in_the_loop.runtime_env import IsaacLabEnvAdapter  # noqa: E402
from strafer_lab.tools.bbox_extractor import (  # noqa: E402
    ReplicatorBboxExtractor,
    resolve_render_product_path,
)
from strafer_lab.tools.lerobot_depth import (  # noqa: E402
    PERCEPTION_DEPTH,
    POLICY_DEPTH,
    frame_path as depth_frame_path,
    read_depth_png,
)
from strafer_lab.tools.lerobot_detections import (  # noqa: E402
    DETECTIONS_BBOX,
    DETECTIONS_VALID,
    DETECTIONS_MAX_DEFAULT,
    read_detection_labels,
)
from strafer_lab.tools.lerobot_writer import (  # noqa: E402
    StraferLeRobotWriter,
    hash_scene_metadata,
    read_strafer_episodes,
)
from strafer_lab.tools import scene_metadata_reader  # noqa: E402


class _ScriptedMissionApi:
    """Fake executor: reports ``running`` until a per-mission poll budget,
    then the scripted terminal state. Lets the smoke drive both a kept
    (``succeeded``) and a discarded (``cancelled``) episode with no ROS."""

    def __init__(self, plan: list[tuple[int, str]]) -> None:
        self._plan = plan
        self._mission = -1
        self._polls = 0

    def submit(self, *, raw_command: str, request_id: str) -> str:
        self._mission += 1
        self._polls = 0
        return request_id

    def status(self) -> MissionStatus:
        self._polls += 1
        budget, state = self._plan[min(self._mission, len(self._plan) - 1)]
        if self._polls >= budget:
            return MissionStatus(terminal=True, state=state)
        return MissionStatus(terminal=False, state="running")

    def cancel(self) -> None:
        pass


def _override_scene(env_cfg, scene: str) -> dict:
    """Point the env at ``scene`` + its spawn pool; return its metadata dict.

    Reads the metadata from the scene USD's ``customData`` — hard-errors
    if the scene was not regenerated with embedded metadata (the
    clean-break behavior the detections gate depends on).
    """
    usdc = Path("Assets/generated/scenes") / f"{scene}.usdc"
    metadata = scene_metadata_reader.load(usdc)
    if usdc.is_file():
        env_cfg.scene.scene_geometry.spawn.usd_path = str(usdc.resolve())
    combined = Path("Assets/generated/scenes/scenes_metadata.json")
    if combined.is_file():
        block = json.loads(combined.read_text(encoding="utf-8")).get("scenes", {}).get(scene, {})
        pts = [list(map(float, p)) for p in block.get("spawn_points_xy", []) if len(p) >= 2]
        if pts and getattr(env_cfg.events, "reset_robot", None) is not None:
            env_cfg.events.reset_robot.params["spawn_points_xy"] = pts
    return metadata


def _fail(msg: str) -> None:
    print(f"[smoke] FAIL: {msg}", flush=True)


def main() -> int:
    output_root = Path(
        args.output or f"/tmp/bridge_harness_smoke_{time.strftime('%Y%m%dT%H%M%S')}",
    )
    print(f"[smoke] scene={args.scene}  output={output_root}", flush=True)

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    scene_metadata = _override_scene(env_cfg, args.scene)
    if getattr(env_cfg, "sensors", None) is None:
        _fail(f"{args.task} exposes no sensor stack")
        return 2
    cameras_required = tuple(env_cfg.sensors.cameras_required)
    env_cfg.terminations.time_out = None
    env_cfg.decimation = 1
    if hasattr(env_cfg.sim, "render_interval"):
        env_cfg.sim.render_interval = 1
    print(f"[smoke] cameras_required={cameras_required}", flush=True)

    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()
    unwrapped.render_enabled = False

    extractor = ReplicatorBboxExtractor(
        camera_render_product_path=resolve_render_product_path(
            unwrapped.scene["d555_camera_perception"],
        ),
    )

    # Scripted sweep: ease forward while yawing so the camera pans the room.
    cmd_vel = lambda: ((0.25, 0.0, 0.0), (0.0, 0.0, 0.6))  # noqa: E731

    adapter = IsaacLabEnvAdapter(
        env=env,
        scene_name=args.scene,
        cmd_vel_source=cmd_vel,
        cameras_required=cameras_required,
        detections_source=extractor.extract,
        on_stepped=lambda _t: simulation_app.update(),
    )

    writer = StraferLeRobotWriter(
        root=output_root,
        repo_id=f"strafer/{args.scene}",
        fps=8,
        capture_git_sha="smoke",
        scene_metadata_hash=hash_scene_metadata(scene_metadata),
        cameras_required=cameras_required,
        detections_max=DETECTIONS_MAX_DEFAULT,
        vcodec="h264",
        session_id="smoke",
    )
    recorder = BridgeLeRobotRecorder(writer=writer, scene_id=args.scene)
    mission_api = _ScriptedMissionApi(
        plan=[(args.steps, "succeeded"), (max(5, args.steps // 4), "cancelled")],
    )
    harness = SimInTheLoopHarness(
        env_adapter=adapter,
        mission_api=mission_api,
        recorder=recorder,
        config=HarnessConfig(
            mission_timeout_s=1.0e6,
            capture_every_n_steps=5,
            max_steps_per_mission=args.steps + 10,
            status_poll_period_s=0.0,
        ),
    )

    def _spec(tag: str) -> tuple[MissionSpec, EpisodeMeta]:
        spec = MissionSpec(
            mission_id=f"smoke__{tag}",
            scene_name=args.scene,
            target_label="shelf",
            target_instance_id=0,
            target_position_3d=(2.0, 0.0, 0.0),
            target_room_idx=None,
            raw_command="go to the shelf",
        )
        return spec, EpisodeMeta.from_spec(spec)

    rc = 0
    try:
        with writer:
            for spec, meta in (_spec("kept"), _spec("discarded")):
                outcome = harness.run_one_mission(spec, meta=meta)
                print(
                    f"[smoke] mission {spec.mission_id}: state={outcome.final_status.state} "
                    f"frames={outcome.frames_written}",
                    flush=True,
                )

        rc = _verify(output_root, cameras_required)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _fail(f"{type(exc).__name__}: {exc}")
        rc = 1
    finally:
        try:
            env.close()
        except Exception:
            pass
        simulation_app.close()
    return rc


def _verify(output_root: Path, cameras_required: tuple[str, ...]) -> int:
    """Re-open the dataset and assert the schema is well-formed."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"strafer/{args.scene}", root=output_root)
    n_ep = ds.meta.total_episodes
    n_fr = ds.meta.total_frames
    print(f"[smoke] reloaded: episodes={n_ep} frames={n_fr}", flush=True)

    # The discarded mission must not have reached disk.
    rows = read_strafer_episodes(output_root)
    if n_ep != 1 or len(rows) != 1:
        _fail(f"expected exactly 1 kept episode on disk, got {n_ep} / {len(rows)} rows")
        return 1
    row = rows[0]
    if row["outcome"] != "succeeded" or row["source_driver"] != "bridge":
        _fail(f"unexpected episode row: outcome={row['outcome']} driver={row['source_driver']}")
        return 1
    print(
        f"[smoke] episode row: outcome={row['outcome']} split={row['episode_split']} "
        f"leg_dist={row['leg_initial_distance_m']}",
        flush=True,
    )

    item = ds[0]
    if item["observation.state"].shape[-1] != 10 or item["action"].shape[-1] != 3:
        _fail("state/action dims wrong")
        return 1
    if DETECTIONS_BBOX not in item or DETECTIONS_VALID not in item:
        _fail("detections columns missing from the dataset")
        return 1

    # Depth sidecars.
    if "depth_full" in cameras_required:
        d = read_depth_png(depth_frame_path(output_root, 0, 0, PERCEPTION_DEPTH))
        if d.shape != (360, 640):
            _fail(f"perception depth sidecar shape {d.shape} != (360, 640)")
            return 1
    if "depth_policy" in cameras_required:
        dp = read_depth_png(depth_frame_path(output_root, 0, 0, POLICY_DEPTH))
        if dp.shape != (60, 80):
            _fail(f"policy depth sidecar shape {dp.shape} != (60, 80)")
            return 1

    # Detection vocab — the one gap a Jetson-free smoke can close on a
    # furnished scene. Soft by default (see --require-detections).
    labels = read_detection_labels(output_root)
    if labels:
        print(f"[smoke] detection vocab ({len(labels)}): {list(labels)[:12]}", flush=True)
    else:
        # The detections columns + pack path are exercised above regardless.
        # A regenerated scene carries UsdSemantics labels on its non-structural
        # prims, so an empty vocab here means either the scene predates that
        # authoring (regenerate it via prep_room_usds / extract_scene_metadata)
        # or the spawn happened to face empty space (rare on a furnished scene).
        msg = (
            "detection vocab is EMPTY — the bbox_2d_tight annotator emitted no "
            "boxes for this scene. Either the scene was not regenerated with "
            "UsdSemantics labels (re-run extract_scene_metadata --from-usd), or "
            "the spawn faced empty space."
        )
        if args.require_detections:
            _fail(msg)
            return 1
        print(f"[smoke] WARNING: {msg}", flush=True)

    print("[smoke] PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
