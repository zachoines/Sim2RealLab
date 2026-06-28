#!/usr/bin/env python3
"""Scripted coverage capture driver — diverse-perspective bulk dataset.

In-process Isaac Lab driver that runs the trained RL subgoal-follower along a
deterministic geometric coverage plan and records a LeRobot v3 dataset. The
plan visits every room repeatedly from varied approach headings; the
same-place / different-heading views it captures are the training signal a
place-recognition head, the backbone bakeoff, and the eval harness need and a
goal-reaching demo set never produces. This is the harness bulk-capture
default — reach for teleop (annotator) or the bridge (Jetson-in-loop) only for
the non-bulk paths.

It drives the env on the canonical rsl_rl inference path, the same one
``play_strafer_navigation.py`` uses: an ``OnPolicyRunner`` loads the raw
training checkpoint, ``env.get_observations()`` produces the policy
observation, and the policy's action goes straight to ``env.step``. The rolling
subgoal the policy observes comes from the env's own command term — a
:class:`~strafer_lab.tasks.navigation.mdp.capture_commands.CaptureSubgoalCommand`
swapped in under the ``goal_command`` name and fed each coverage leg's planned
path. The driver only wires these pieces together:

- the coverage plan is built by :mod:`strafer_lab.tools.coverage_plan` (pure
  geometry: every room visited, headings spread, seeded for reproducibility),
- per leg the command term rolls the subgoal along the plan's path and signals
  leg completion; the policy tracks it,
- camera / detections capture reuses
  :class:`strafer_lab.sim_in_the_loop.runtime_env.IsaacLabEnvAdapter` and the
  writer lifecycle runs through
  :class:`strafer_lab.sim_in_the_loop.lerobot_recorder.CoverageLeRobotRecorder`.

Run via the unified entry point::

    python source/strafer_lab/scripts/capture.py \\
        --driver scripted --mission-source coverage \\
        --scene <scene> --output <root> \\
        --policy-variant nocam_subgoal --checkpoint <model_step.pt>

``capture.py`` dispatches here after booting nothing itself; this script owns
its AppLauncher. The checkpoint is a raw rsl_rl training checkpoint
(``model_<step>.pt``) loaded through the runner — no export step.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Registered capture env per policy variant — the env carries the observation /
# action contract the checkpoint trained against and the capture sensor stack.
# Mirrors export_policy's variant->env mapping; --env overrides it.
_CAPTURE_ENV_BY_VARIANT = {
    "nocam_subgoal": "Isaac-Strafer-Nav-Capture-Coverage-v0",
}

_MAX_PATH_POINTS = 512

# Overhead structure prims hidden from the recording camera — Infinigen's
# ceiling / roof / attic / exterior room-structure labels.
_OVERHEAD_PRIM_RE = re.compile(
    r"(?:^|_)(ceiling|roof|attic|exterior)(?:_\d+)?$",
    re.IGNORECASE,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene", required=True, help="Scene name (scene_id).")
    parser.add_argument(
        "--scene-usd", default=None,
        help="Override the scene USD; defaults to the scene's registered USD.",
    )
    parser.add_argument("--output", required=True, help="LeRobot dataset root.")
    parser.add_argument(
        "--sensors", default="rgb_full,depth_full",
        help="Sensor stack the env renders and the writer records.",
    )
    parser.add_argument("--vcodec", default="h264", help="LeRobot v3 video codec.")
    parser.add_argument(
        "--detections", action=argparse.BooleanOptionalAction, default=True,
        help="Record per-frame Replicator detections as dataset columns "
             "(on by default for scripted capture).",
    )
    parser.add_argument(
        "--detections-max", type=int, default=None,
        help="Padded detection slots per frame; defaults to the writer's count.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Parallel env count. v1 captures single-env; >1 is rejected.",
    )
    parser.add_argument(
        "--policy-variant", default="nocam_subgoal",
        help="Selects the capture env (and thus the obs/action contract) for "
             "the trained subgoal-follower.",
    )
    parser.add_argument(
        "--env", default=None,
        help="Registered capture env id. Defaults to the env for "
             "--policy-variant.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Raw rsl_rl training checkpoint (model_<step>.pt) loaded via "
             "OnPolicyRunner — the same artifact play_strafer_navigation uses.",
    )
    parser.add_argument(
        "--coverage-visits-per-room", type=int, default=None,
        help="Minimum visits scheduled per room; defaults to the plan default.",
    )
    parser.add_argument(
        "--coverage-heading-spread-deg", type=float, default=90.0,
        help="Per-room approach-heading spread the coverage metric requires.",
    )
    parser.add_argument(
        "--held-out-scenes", default=None,
        help="Comma-separated scenes held out for home-to-home generalization. "
             "When --scene is in this set every episode is tagged "
             "episode_split=held_out_seeds.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Coverage-plan seed.")
    parser.add_argument(
        "--n-trajectories", type=int, default=None,
        help="Cap the number of visit episodes captured (default: the whole plan).",
    )
    parser.add_argument(
        "--lookahead-m", type=float, default=None,
        help="Subgoal lookahead distance; defaults to the shared deployment value.",
    )
    parser.add_argument(
        "--capture-every-n-steps", type=int, default=5,
        help="Env-step interval for frame capture; sets the writer fps.",
    )
    parser.add_argument(
        "--max-steps-per-leg", type=int, default=600,
        help="Hard cap on env steps before abandoning a viewpoint leg.",
    )
    parser.add_argument(
        "--approach-distance-m", type=float, default=0.6,
        help="Length of the final straight approach segment that realizes a "
             "visit's approach heading.",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Record a third-person overhead MP4 of the coverage sweep. "
             "Needs no live display, so it is the display-independent way to "
             "watch the traversal on a headless host.",
    )
    parser.add_argument(
        "--video-length", type=int, default=None,
        help="Cap on env steps filmed into the overhead MP4 (one frame per env "
             "step). Recording stops at the cap and the rest of the sweep is "
             "not filmed; the captured dataset is unaffected. Defaults to the "
             "whole planned sweep.",
    )
    parser.add_argument(
        "--video-dir", default="logs/coverage_capture/videos",
        help="Directory the overhead MP4 is written under.",
    )
    parser.add_argument(
        "--video-keep-ceiling", action="store_true",
        help="Keep ceiling/roof structure visible in the overhead MP4. By "
             "default --video hides overhead structure so the top-down view "
             "sees into rooms; that hide is global (the renderer has no honored "
             "per-camera path), so it also hides the ceiling from this run's "
             "recorded perception frames. Use --video for QA; run production "
             "capture without it, or pass this flag, for an untouched corpus.",
    )
    return parser


def _git_rev_parse_head(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _scene_dir_for(usd_path: Path) -> Path:
    """Resolve the per-scene directory where occupancy.npy is cached."""
    resolved = usd_path.resolve()
    for parent in resolved.parents:
        if parent.name.startswith("scene_") and parent.parent.name == "scenes":
            return parent
    return resolved.parent


def main() -> int:
    parser = _build_parser()

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    # Cameras are non-negotiable for a capture driver.
    args.enable_cameras = True

    if args.num_envs != 1:
        raise SystemExit(
            "coverage_capture v1 captures single-env; --num-envs > 1 is a "
            "named follow-up. Run with --num-envs 1.",
        )

    variant = args.policy_variant.lower()
    env_id = args.env or _CAPTURE_ENV_BY_VARIANT.get(variant)
    if env_id is None:
        raise SystemExit(
            f"no capture env registered for --policy-variant {variant!r}; "
            f"known: {sorted(_CAPTURE_ENV_BY_VARIANT)}. Pass --env explicitly.",
        )

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import importlib.metadata as _metadata

    from isaaclab.envs.common import ViewerCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    import strafer_lab.tasks  # noqa: F401  (registers envs)

    from strafer_lab.sim_in_the_loop.lerobot_recorder import (
        CoverageLeRobotRecorder,
        yaw_from_quat_xyzw,
    )
    from strafer_lab.sim_in_the_loop.runtime_env import IsaacLabEnvAdapter
    from strafer_lab.tasks.navigation import mdp
    from strafer_lab.tasks.navigation.path_planner import (
        InvalidEndpointError,
        NoPathError,
        PathPlanningError,
        plan_path,
    )
    from strafer_lab.tasks.navigation.strafer_env_cfg import (
        _get_infinigen_active_scene_floor_top_z,
    )
    from strafer_lab.tools import coverage_plan as coverage_plan_mod
    from strafer_lab.tools import scene_connectivity, scene_metadata_reader
    from strafer_lab.tools.bbox_extractor import (
        ReplicatorBboxExtractor,
        resolve_render_product_path,
    )
    from strafer_lab.tools.lerobot_detections import DETECTIONS_MAX_DEFAULT
    from strafer_lab.tools.lerobot_writer import (
        StraferLeRobotWriter,
        hash_scene_metadata,
    )
    from strafer_lab.tools.scene_paths import resolve_scene_usd_path

    def _to_torch(arr):
        if hasattr(arr, "detach"):
            return arr
        import warp as wp

        return wp.to_torch(arr)

    repo_root = Path(__file__).resolve().parents[3]
    heading_spread_rad = math.radians(args.coverage_heading_spread_deg)
    held_out = {s.strip() for s in (args.held_out_scenes or "").split(",") if s.strip()}
    episode_split = "held_out_seeds" if args.scene in held_out else None

    # --- scene metadata + cached free space -> coverage plan -----------------
    scene_usd_path = resolve_scene_usd_path(scene=args.scene, usd_override=args.scene_usd)
    scene_metadata = scene_metadata_reader.load(scene_usd_path)
    rooms = scene_metadata.get("rooms", [])
    if not rooms:
        raise SystemExit(f"scene {args.scene!r} carries no rooms[] metadata")

    occupancy = scene_connectivity.load_occupancy(_scene_dir_for(scene_usd_path))
    free_space = scene_connectivity.occupancy_to_free_space(
        occupancy.grid, grid_res=occupancy.resolution_m,
    )
    # Seal the planner's free space to the rooms: the cached grid keeps a free
    # exterior pad and porous perimeter walls (interior and exterior merge into
    # one connected free component), so without this the plan and the spawn can
    # route through / land in outside-the-house space.
    free_space = scene_connectivity.seal_free_space_to_rooms(
        free_space, rooms,
        origin_xy=occupancy.origin_xy, grid_res=occupancy.resolution_m,
    )
    plan_kwargs = {}
    if args.coverage_visits_per_room is not None:
        plan_kwargs["visits_per_room"] = args.coverage_visits_per_room
    plan = coverage_plan_mod.build_coverage_plan(
        rooms,
        free_space,
        grid_res=occupancy.resolution_m,
        grid_origin_xy=occupancy.origin_xy,
        seed=args.seed,
        heading_spread_threshold_rad=heading_spread_rad,
        room_adjacency=scene_metadata.get("room_adjacency"),
        **plan_kwargs,
    )
    metric = coverage_plan_mod.coverage_metric(plan)
    print(
        f"[coverage_capture] plan: {len(plan.waypoints)} visits over "
        f"{len(metric.rooms)} rooms; metric satisfied={metric.satisfied}",
        flush=True,
    )

    # --- env build -----------------------------------------------------------
    env_cfg = parse_env_cfg(env_id, device="cuda:0", num_envs=1)
    env_cfg.seed = args.seed
    if getattr(env_cfg, "sensors", None) is not None and args.sensors:
        from strafer_lab.tasks.navigation.composed_env_cfg import SensorStackCfg

        tokens = tuple(t.strip() for t in args.sensors.split(",") if t.strip())
        env_cfg.sensors = SensorStackCfg(cameras_required=tokens)
        env_cfg.__post_init__()
        env_cfg.scene.num_envs = 1
    cameras_required = tuple(env_cfg.sensors.cameras_required)

    # Swap the env's command term to the externally-fed capture subgoal command
    # (after any __post_init__, which would reset the commands group). The
    # goal-shaped obs terms read "goal_command", so the policy observes the
    # rolling subgoal the driver feeds it leg by leg.
    subgoal_cfg_kwargs = {}
    if args.lookahead_m is not None:
        subgoal_cfg_kwargs["lookahead_m"] = args.lookahead_m
    env_cfg.commands.goal_command = mdp.CaptureSubgoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e6, 1.0e6),
        max_path_points=_MAX_PATH_POINTS,
        debug_vis=False,
        **subgoal_cfg_kwargs,
    )
    # The driver owns episode boundaries; a training-scale timeout must not
    # teleport the robot mid-traversal. Flip / sustained-collision stay active.
    if getattr(env_cfg.terminations, "time_out", None) is not None:
        env_cfg.terminations.time_out = None
    # The goal-objective env's training-only managers read fixed-goal command
    # state (``_goal`` / ``_goal_reached_count``) the rolling subgoal command
    # does not expose. Capture is inference-only, so disable them.
    if getattr(env_cfg.events, "randomize_goal_noise", None) is not None:
        env_cfg.events.randomize_goal_noise = None
    if getattr(env_cfg.curriculum, "goal_distance", None) is not None:
        env_cfg.curriculum.goal_distance = None

    # Bind the spawned geometry to the scene whose occupancy grid + plan we
    # just built. The env cfg's __post_init__ hardcodes the first pooled
    # scene's USD for every scene; without this the sim loads the wrong
    # floorplan and the scene-frame spawn + plan land in another scene's walls.
    # scene_usd_path is already override-aware (resolve_scene_usd_path applied
    # --scene-usd), so this single unconditional bind subsumes the override.
    env_cfg.scene.scene_geometry.spawn.usd_path = str(scene_usd_path.resolve())
    # __post_init__ also pools spawn_z to the MAX floor height across all scenes
    # and lifts the ground to the first pooled scene's floor; per scene that
    # drops the robot from the wrong height into / through the floor. Pin both
    # to this scene's own floor (mirrors _apply_infinigen_scene_setup's offsets,
    # for capture only — the training env's single default is unchanged).
    active_floor_top_z = _get_infinigen_active_scene_floor_top_z(args.scene)
    if active_floor_top_z is not None:
        floor_z = float(active_floor_top_z)
        if getattr(env_cfg.events, "reset_robot", None) is not None:
            env_cfg.events.reset_robot.params["spawn_z"] = floor_z + 0.1
        if getattr(env_cfg.events, "lift_ground", None) is not None:
            env_cfg.events.lift_ground.params["target_z"] = floor_z - 0.002

    # Spawn the robot from the occupancy free-space the plan already runs on.
    # The grid, the plan targets, and spawn_points_xy share the scene-authored
    # (env-local) frame, and reset_robot_state_on_floor adds env_origins itself,
    # so the derived cell goes in as-is — no world-frame offset here. A fresh
    # list is assigned because the reset event caches its points by id().
    spawn_xy = _derive_spawn_xy(
        free_space,
        plan,
        occupancy,
        rooms=rooms,
        plan_path=plan_path,
        invalid_endpoint_errors=(NoPathError, InvalidEndpointError),
    )
    # Pre-capture gate (grid frame): the spawn must be inside a room and the
    # first real leg must be in-room and plannable, or this scene is not
    # capture-ready — fail loud here instead of silently recording garbage.
    _validate_spawn_ready(
        spawn_xy, plan, rooms, free_space, occupancy,
        plan_path=plan_path,
        invalid_endpoint_errors=(NoPathError, InvalidEndpointError),
        point_in_any_room=scene_connectivity.point_in_any_room,
    )
    env_cfg.events.reset_robot.params["spawn_points_xy"] = [spawn_xy]

    # --- runner + policy (raw rsl_rl checkpoint, canonical inference path) ----
    agent_cfg = load_cfg_from_registry(env_id, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))

    # Optional third-person overhead recording of the whole sweep — a watchable
    # MP4 that needs no live display. Mirrors play_strafer_navigation: an
    # env-anchored overhead viewer drives RecordVideo, and the capture camera is
    # placed in world frame by lifting the viewer offset by the env origin (the
    # coverage plan runs in the scene-local frame; the sim camera takes world).
    if args.video:
        env_cfg.viewer = ViewerCfg(
            eye=(0.0, 0.0, 12.0),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
            env_index=0,
            resolution=(1280, 720),
        )

    env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        base = env.unwrapped
        env_origin = base.scene.env_origins[0].detach().cpu().tolist()
        viewer = base.cfg.viewer
        world_eye = tuple(env_origin[i] + viewer.eye[i] for i in range(3))
        world_target = tuple(env_origin[i] + viewer.lookat[i] for i in range(3))
        # Absolute altitude the overhead camera follows the robot at each step.
        overhead_altitude_z = world_eye[2]
        recorder = getattr(base, "video_recorder", None)
        capture = getattr(recorder, "_capture", None) if recorder is not None else None
        # The recorder renders this camera prim; the per-step follow re-poses it.
        overhead_cam_prim_path = (
            capture.cfg.camera_prim_path if capture is not None
            else viewer.cam_prim_path
        )
        if capture is not None:
            capture.cfg.camera_position = world_eye
            capture.cfg.camera_target = world_target
        else:
            print(
                "[coverage_capture] --video: viewport capture handle "
                "unavailable; the overhead MP4 may use the default camera pose",
                flush=True,
            )
        base.sim.set_camera_view(eye=world_eye, target=world_target)
        try:
            from isaaclab_physx.renderers.kit_viewport_utils import (
                set_kit_renderer_camera_view,
            )

            set_kit_renderer_camera_view(
                eye=world_eye, target=world_target,
                camera_prim_path=viewer.cam_prim_path,
            )
        except ImportError:
            pass
        # RecordVideo writes one frame per env step and self-stops at
        # video_length, so size the default to the whole planned sweep
        # (legs x the per-leg step cap) — otherwise the MP4 truncates mid-run.
        legs = args.n_trajectories or len(plan.waypoints)
        video_length = (
            args.video_length if args.video_length is not None
            else legs * args.max_steps_per_leg
        )
        video_root = Path(args.video_dir).resolve() / (
            f"coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        video_root.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_root),
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )
        print(f"[coverage_capture] recording overhead sweep to: {video_root}", flush=True)
        if not args.video_keep_ceiling:
            _hide_overhead_structure(base)

    env = RslRlVecEnvWrapper(env)
    unwrapped = env.unwrapped
    device = unwrapped.device

    # Pre-traversal gate (runtime): confirm the sim actually loaded the scene
    # whose grid + plan drove this capture, by hashing the embedded metadata on
    # the live geometry prim. Catches a scene/grid mismatch directly, even one
    # introduced by a future cfg regression the cfg-path check would miss.
    _assert_loaded_scene_identity(
        unwrapped.scene.stage,
        geometry_prim_path=env_cfg.scene.scene_geometry.prim_path,
        cfg_usd_path=str(env_cfg.scene.scene_geometry.spawn.usd_path),
        expected_usd_path=str(scene_usd_path.resolve()),
        expected_metadata=scene_metadata,
        hash_fn=hash_scene_metadata,
        prim_metadata_reader=scene_metadata_reader.metadata_from_prim,
    )

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[coverage_capture] loading checkpoint: {args.checkpoint}", flush=True)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=device)

    subgoal_term = unwrapped.command_manager.get_term("goal_command")
    env_origin_xy = _to_torch(unwrapped.scene.env_origins)[0, :2].detach().cpu().numpy()

    step_dt = float(unwrapped.sim.get_physics_dt()) * int(unwrapped.cfg.decimation)
    writer_fps = max(1, round(1.0 / (step_dt * max(1, args.capture_every_n_steps))))

    def _policy_action():
        obs = env.get_observations()
        with torch.inference_mode():
            return policy(obs)

    def _robot_xy_local() -> np.ndarray:
        world = _to_torch(unwrapped.scene["robot"].data.root_pos_w)[0, :2]
        return world.detach().cpu().numpy() - env_origin_xy

    # Report the live policy obs width: the rolling-subgoal contract the
    # checkpoint expects (e.g. 19-dim NOCAM_SUBGOAL).
    obs0 = env.get_observations()
    policy_obs0 = obs0["policy"] if hasattr(obs0, "keys") and "policy" in obs0.keys() else obs0
    print(
        f"[coverage_capture] policy obs dim = {int(policy_obs0.shape[-1])}",
        flush=True,
    )

    # --- detections + writer + adapter + recorder ----------------------------
    detections_source = None
    if args.detections:
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path=resolve_render_product_path(
                unwrapped.scene["d555_camera_perception"],
            ),
        )
        detections_source = extractor.extract

    writer = StraferLeRobotWriter(
        root=Path(args.output),
        repo_id=f"strafer/{args.scene}",
        fps=writer_fps,
        capture_git_sha=_git_rev_parse_head(repo_root),
        scene_metadata_hash=hash_scene_metadata(scene_metadata),
        scene_metadata=scene_metadata,
        cameras_required=cameras_required,
        detections_max=(
            (args.detections_max or DETECTIONS_MAX_DEFAULT) if args.detections else None
        ),
        vcodec=args.vcodec,
    )
    adapter = IsaacLabEnvAdapter(
        env=env,
        scene_name=args.scene,
        action_source=_policy_action,
        cameras_required=cameras_required,
        detections_source=detections_source,
        step_dt=step_dt,
    )
    recorder = CoverageLeRobotRecorder(writer=writer, scene_id=args.scene)

    # --- traversal -----------------------------------------------------------
    try:
        adapter.reset(scene_name=args.scene)
        if hasattr(unwrapped, "_d555_mount_quat"):
            mount_quat = tuple(
                float(v) for v in unwrapped._d555_mount_quat[0].detach().cpu().tolist()
            )
        else:
            mount_quat = (1.0, 0.0, 0.0, 0.0)

        captured = 0
        for visit in plan.waypoints:
            if args.n_trajectories is not None and captured >= args.n_trajectories:
                break

            leg_local = _leg_path(
                _robot_xy_local(),
                np.asarray(visit.target_xy, dtype=np.float32),
                visit.approach_heading_rad,
                free_space,
                grid_res=occupancy.resolution_m,
                grid_origin_xy=occupancy.origin_xy,
                approach_distance_m=args.approach_distance_m,
                plan_path=plan_path,
                error_cls=PathPlanningError,
            )
            if leg_local is None:
                print(
                    f"[coverage_capture] skip room {visit.room_index} visit "
                    f"{visit.visit_ordinal}: no collision-free path",
                    flush=True,
                )
                continue
            # The plan / occupancy grid live in the scene-authored frame; the
            # command term tracks in world frame, so lift the leg by the env
            # origin before handing it over.
            leg_world = leg_local + env_origin_xy
            subgoal_term.set_leg(torch.as_tensor(leg_world, dtype=torch.float32, device=device))

            start_bundle = adapter.capture()
            recorder.begin_episode(
                start_bundle=start_bundle,
                episode_split=episode_split,
                realized_d555_mount_quat=mount_quat,
            )
            recorder.add_frame(start_bundle)
            for step in range(1, args.max_steps_per_leg + 1):
                if args.video:
                    _follow_overhead_camera(
                        _robot_xy_local() + env_origin_xy,
                        overhead_altitude_z,
                        overhead_cam_prim_path,
                    )
                adapter.step()
                if step % args.capture_every_n_steps == 0:
                    recorder.add_frame(adapter.capture())
                if bool(subgoal_term.path_complete[0].item()):
                    break
            recorder.add_frame(adapter.capture())
            recorder.end_episode()
            captured += 1
        print(
            f"[coverage_capture] captured {recorder.episodes_kept} episodes "
            f"({recorder.episodes_discarded} discarded)",
            flush=True,
        )
    finally:
        writer.finalize()
        env.close()
        simulation_app.close()
    return 0


def _follow_overhead_camera(robot_world_xy, altitude_z: float, cam_prim_path: str) -> None:
    """Re-pose the top-down overhead camera above the robot's world XY.

    The video recorder poses its camera prim only on the first frame, so the
    recorded sweep would otherwise stay fixed; this re-poses that same prim each
    step (the call teleop's follow uses). Eye and target share XY so the view
    stays straight down.
    """
    from isaacsim.core.utils.viewports import set_camera_view

    x, y = float(robot_world_xy[0]), float(robot_world_xy[1])
    set_camera_view(eye=[x, y, altitude_z], target=[x, y, 0.0], camera_prim_path=cam_prim_path)


def _hide_overhead_structure(base) -> None:
    """Hide ceiling / roof / attic / exterior structure for the recording.

    Sets the matched structure prims invisible in the stage's session layer so
    the top-down overhead camera sees into the rooms. USD visibility is a global
    render attribute, not per-camera (the RTX renderer does not honor the
    per-camera ``cameraVisibility`` collection in this build), so the structure
    is also hidden from the body-mounted perception camera while the recording
    runs. That camera is horizontal and rarely frames a ceiling, so the effect
    on the dataset is small, but a ``--video`` run is a QA pass: run production
    capture without ``--video`` (or with ``--video-keep-ceiling``) for an
    untouched corpus. Collision and navigation are unaffected (visibility is not
    a physics flag), and the session-layer edit leaves the scene USD on disk
    unchanged and resets each launch.
    """
    from pxr import Usd, UsdGeom  # type: ignore

    stage = base.scene.stage
    hidden: list[str] = []
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        for prim in stage.Traverse():
            if _OVERHEAD_PRIM_RE.search(prim.GetName()):
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()
                    hidden.append(str(prim.GetPath()))
    if hidden:
        print(
            f"[coverage_capture] --video: hid {len(hidden)} overhead structure "
            f"prims for the recording — also hidden from the perception camera "
            f"this run (sample: {', '.join(hidden[:5])})",
            flush=True,
        )
    else:
        print(
            "[coverage_capture] --video: no overhead structure prims matched; "
            "the roof may still occlude the top-down view for this scene",
            flush=True,
        )


def _leg_path(
    robot_xy,
    target_xy,
    approach_heading_rad: float,
    free_space,
    *,
    grid_res: float,
    grid_origin_xy,
    approach_distance_m: float,
    plan_path,
    error_cls,
):
    """Plan a collision-free path to a viewpoint with a fixed approach heading.

    Routes to a staging point one ``approach_distance_m`` behind the viewpoint
    along the requested heading, then a straight final segment whose tangent is
    that heading — so the robot arrives facing the planned direction. Falls
    back to a direct path when the staged approach is unplannable.
    """
    import numpy as np

    staging = (
        float(target_xy[0]) - approach_distance_m * math.cos(approach_heading_rad),
        float(target_xy[1]) - approach_distance_m * math.sin(approach_heading_rad),
    )
    try:
        path = plan_path(
            np.asarray(robot_xy, dtype=np.float32),
            np.asarray(staging, dtype=np.float32),
            free_space,
            grid_res=grid_res,
            grid_origin_xy=grid_origin_xy,
        )
        return np.vstack([path, np.asarray([target_xy], dtype=np.float32)])
    except error_cls:
        pass
    try:
        return plan_path(
            np.asarray(robot_xy, dtype=np.float32),
            np.asarray(target_xy, dtype=np.float32),
            free_space,
            grid_res=grid_res,
            grid_origin_xy=grid_origin_xy,
        )
    except error_cls:
        return None


def _derive_spawn_xy(
    free_space,
    plan,
    occupancy,
    *,
    rooms,
    plan_path,
    invalid_endpoint_errors,
):
    """Pick the capture start pose from the occupancy free-space.

    Returns a single ``[x, y]`` in the occupancy (scene-authored / env-local)
    frame — the same frame the coverage plan and the leg planner use, and the
    frame ``reset_robot_state_on_floor`` expects before it adds the env origin.
    The chosen cell is passable on the robot-radius-inflated ``free_space`` grid,
    inside a room footprint, and reachable by the shared planner, so legs plan
    instead of failing on a spawn the floor sampler and the grid disagree about.

    Preference order, all deterministic (no RNG, so a scene maps to one spawn):

    1. a plan viewpoint (the first is the primary anchor) that is free, in-room,
       and from which the first real leg — to the next distinct viewpoint —
       plans;
    2. a row-major scan of the inflated free grid for the first free, in-room
       cell that plans to the traversal's first free in-room viewpoint.

    The in-room containment is the second guard against the porous-exterior
    occupancy grid: a "free + reachable" cell can otherwise be the exterior
    corner that the row-major scan returns first. The reachability probe uses
    the planner's default snap radius on purpose: widening it would teleport a
    blocked start onto free space and hide a corrupted occupancy grid.
    """
    import numpy as np

    from strafer_lab.tools.scene_connectivity import (
        _cell_to_xy,
        _xy_to_cell,
        point_in_any_room,
    )

    origin_xy = occupancy.origin_xy
    grid_res = occupancy.resolution_m
    rows, cols = free_space.shape
    # Two cells clears the planner's coincident-endpoint floor, so a probe is a
    # genuine leg rather than a same-point rejection.
    min_leg_m = 2.0 * grid_res

    def _is_free(x: float, y: float) -> bool:
        r, c = _xy_to_cell((x, y), origin_xy=origin_xy, grid_res=grid_res)
        return 0 <= r < rows and 0 <= c < cols and bool(free_space[r, c])

    def _in_room(x: float, y: float) -> bool:
        return point_in_any_room(x, y, rooms)

    def _plans(x: float, y: float, goal) -> bool:
        try:
            plan_path(
                np.asarray([x, y], dtype=np.float32),
                np.asarray(goal, dtype=np.float32),
                free_space,
                grid_res=grid_res,
                grid_origin_xy=origin_xy,
            )
            return True
        except invalid_endpoint_errors:
            return False

    targets = [
        (float(wp.target_xy[0]), float(wp.target_xy[1])) for wp in plan.waypoints
    ]

    def _first_distinct(x: float, y: float):
        return next(
            (t for t in targets if math.hypot(t[0] - x, t[1] - y) > min_leg_m),
            None,
        )

    for x, y in targets:
        if not _is_free(x, y) or not _in_room(x, y):
            continue
        goal = _first_distinct(x, y)
        if goal is None or _plans(x, y, goal):
            return [x, y]

    # Fallback: the plan's viewpoints are unusable, so scan free space for an
    # in-room cell that connects to the traversal's first free in-room
    # viewpoint. Reads the inflated grid only — never raw occupancy, whose free
    # cells sit half a chassis from a wall.
    anchor = next((t for t in targets if _is_free(*t) and _in_room(*t)), None)
    for r, c in np.argwhere(free_space):
        x, y = _cell_to_xy(int(r), int(c), origin_xy=origin_xy, grid_res=grid_res)
        if not _in_room(x, y):
            continue
        if (
            anchor is None
            or math.hypot(anchor[0] - x, anchor[1] - y) <= min_leg_m
            or _plans(x, y, anchor)
        ):
            return [x, y]

    raise RuntimeError(
        "no free, in-room, plan-reachable spawn in the occupancy free-space; "
        "the scene's occupancy grid is degenerate and must be regenerated",
    )


def _validate_spawn_ready(
    spawn_xy,
    plan,
    rooms,
    free_space,
    occupancy,
    *,
    plan_path,
    invalid_endpoint_errors,
    point_in_any_room,
):
    """Pre-capture gate (grid frame): the scene must be spawn/traverse-ready.

    Raises ``SystemExit`` with a clear reason (non-zero exit) when the derived
    spawn is not inside a room footprint, or the first real leg target is not
    in-room / not plannable from the spawn. Runs on the cached grid + footprints
    before the env build, so a genuinely-bad scene fails loud instead of
    silently capturing garbage. A scene with no plannable in-room spawn already
    fails earlier in :func:`_derive_spawn_xy`; this layers the containment +
    first-leg traversability checks on top.
    """
    import numpy as np

    sx, sy = float(spawn_xy[0]), float(spawn_xy[1])
    if not point_in_any_room(sx, sy, rooms):
        raise SystemExit(
            f"capture spawn {[round(sx, 3), round(sy, 3)]} is free but lies "
            "outside every room footprint; the scene's occupancy grid is not "
            "capture-ready (regenerate it)."
        )

    grid_res = occupancy.resolution_m
    origin_xy = occupancy.origin_xy
    min_leg_m = 2.0 * grid_res
    targets = [(float(w.target_xy[0]), float(w.target_xy[1])) for w in plan.waypoints]
    goal = next(
        (t for t in targets if math.hypot(t[0] - sx, t[1] - sy) > min_leg_m),
        None,
    )
    if goal is None:
        return  # single-viewpoint plan; spawn containment is the whole gate
    if not point_in_any_room(goal[0], goal[1], rooms):
        raise SystemExit(
            f"first leg target {[round(goal[0], 3), round(goal[1], 3)]} lies "
            "outside every room footprint; the scene is not traversable."
        )
    try:
        plan_path(
            np.asarray([sx, sy], dtype=np.float32),
            np.asarray(goal, dtype=np.float32),
            free_space,
            grid_res=grid_res,
            grid_origin_xy=origin_xy,
        )
    except invalid_endpoint_errors:
        raise SystemExit(
            f"first leg {[round(sx, 3), round(sy, 3)]} -> "
            f"{[round(goal[0], 3), round(goal[1], 3)]} does not plan on the "
            "sealed free-space; the scene is not traversable."
        )
    print(
        "[coverage_capture] pre-capture gate OK: spawn in-room and first leg "
        "plannable",
        flush=True,
    )


def _assert_loaded_scene_identity(
    stage,
    *,
    geometry_prim_path: str,
    cfg_usd_path: str,
    expected_usd_path: str,
    expected_metadata,
    hash_fn,
    prim_metadata_reader,
):
    """Pre-traversal gate (runtime): the SIM must have loaded the right scene.

    Reads the embedded scene metadata off the live geometry prim (its
    ``customData`` composes through the scene-USD reference) and asserts its
    hash matches the resolved scene's. A direct catch for a scene/grid mismatch
    — the failure mode where every ``--scene`` loads one pooled scene's USD.
    Falls back to a cfg ``usd_path`` equality check, with a printed reason, only
    when the live prim exposes no embedded metadata to hash. Raises
    ``SystemExit`` (non-zero) on mismatch.
    """
    expected_hash = hash_fn(expected_metadata)
    loaded_meta = None
    root = stage.GetPrimAtPath(geometry_prim_path) if stage is not None else None
    if root is not None and root.IsValid():
        # The scene USD's default-prim payload composes onto the referencing
        # prim; check it, then its direct children if a build nests it.
        for prim in [root, *root.GetChildren()]:
            loaded_meta = prim_metadata_reader(prim)
            if loaded_meta is not None:
                break

    if loaded_meta is not None:
        loaded_hash = hash_fn(loaded_meta)
        if loaded_hash != expected_hash:
            raise SystemExit(
                f"loaded scene geometry at {geometry_prim_path} does not match "
                f"the requested scene: embedded metadata hash {loaded_hash[:12]} "
                f"!= {expected_hash[:12]} (for {expected_usd_path}). The sim "
                "loaded the wrong floorplan — spawn and plan would land in "
                "another scene's geometry."
            )
        print(
            f"[coverage_capture] scene-identity OK: {geometry_prim_path} hash "
            f"{loaded_hash[:12]} matches {Path(expected_usd_path).name}",
            flush=True,
        )
        return

    # Fallback: no embedded metadata on the live prim to hash (the reference did
    # not compose customData onto the geometry prim). Verify the cfg bound the
    # right USD and say why the hash path was skipped.
    bound = str(Path(cfg_usd_path).resolve()) if cfg_usd_path else ""
    if bound != str(Path(expected_usd_path).resolve()):
        raise SystemExit(
            f"loaded scene geometry usd_path {bound!r} != requested "
            f"{str(Path(expected_usd_path).resolve())!r}"
        )
    print(
        f"[coverage_capture] scene-identity: no embedded metadata on "
        f"{geometry_prim_path} to hash; verified via cfg usd_path bind instead",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
