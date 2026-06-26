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

Boundaries this driver only wires together:

- the coverage plan is built by :mod:`strafer_lab.tools.coverage_plan` (pure
  geometry: every room visited, headings spread, seeded for reproducibility),
- the controller is the trained RL subgoal-follower, run through the existing
  inference primitives in :mod:`strafer_shared.policy_interface` wrapped by
  :mod:`strafer_lab.tools.subgoal_controller` — no new policy loader or
  inference loop,
- env / action / obs / detections capture reuses
  :class:`strafer_lab.sim_in_the_loop.runtime_env.IsaacLabEnvAdapter` and the
  writer lifecycle is driven through
  :class:`strafer_lab.sim_in_the_loop.lerobot_recorder.CoverageLeRobotRecorder`.

Run via the unified entry point::

    python source/strafer_lab/scripts/capture.py \\
        --driver scripted --mission-source coverage \\
        --scene <scene> --output <root> \\
        --policy-variant nocam_subgoal --checkpoint <exported-policy.pt>

``capture.py`` dispatches here after booting nothing itself; this script owns
its AppLauncher. The checkpoint is an exported artifact (TorchScript .pt /
.onnx from ``export_policy.py``), not a raw rsl_rl training checkpoint.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

# Match SubgoalCommandCfg.path_complete_threshold: a leg ends when the robot is
# within this of the viewpoint, and the cursor buffer sizing follows the env's.
_PATH_COMPLETE_THRESHOLD_M = 0.3
_MAX_PATH_POINTS = 512


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", default="Isaac-Strafer-Nav-Capture-Bridge-v0",
        help="Composed capture task id (carries the perception cameras + the "
             "Infinigen scene + the mecanum action contract).",
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
        help="PolicyVariant for the trained subgoal-follower. nocam_subgoal is "
             "the only variant with a trained checkpoint today.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Exported policy artifact (TorchScript .pt / .onnx) the "
             "subgoal-follower loads via policy_interface.load_policy.",
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


def _resolve_active_spawn_points(scene: str) -> list[list[float]]:
    """Return spawn_points_xy for the active scene only (not pooled)."""
    import json

    combined = Path("Assets/generated/scenes/scenes_metadata.json")
    if not combined.is_file():
        return []
    try:
        data = json.loads(combined.read_text(encoding="utf-8"))
    except Exception:
        return []
    pool = data.get("scenes", {}).get(scene, {}).get("spawn_points_xy", [])
    return [list(map(float, pt)) for pt in pool if len(pt) >= 2]


def main() -> int:
    parser = _build_parser()

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    # Cameras are non-negotiable for a capture driver.
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import isaaclab_tasks  # noqa: F401
    import strafer_lab.tasks  # noqa: F401  (registers envs)
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.utils import parse_env_cfg

    from strafer_lab.sim_in_the_loop.lerobot_recorder import (
        CoverageLeRobotRecorder,
        yaw_from_quat_xyzw,
    )
    from strafer_lab.sim_in_the_loop.runtime_env import IsaacLabEnvAdapter
    from strafer_lab.tasks.navigation import mdp
    from strafer_lab.tasks.navigation.path_planner import (
        PathCursor,
        PathPlanningError,
        plan_path,
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
    from strafer_lab.tools.subgoal_controller import step_subgoal_controller
    from strafer_shared.constants import SUBGOAL_LOOKAHEAD_M
    from strafer_shared.policy_interface import PolicyVariant, load_policy

    if args.num_envs != 1:
        raise SystemExit(
            "coverage_capture v1 captures single-env; --num-envs > 1 is a "
            "named follow-up. Run with --num-envs 1.",
        )

    try:
        variant = PolicyVariant[args.policy_variant.upper()]
    except KeyError:
        raise SystemExit(
            f"unknown --policy-variant {args.policy_variant!r}; "
            f"valid: {[v.name.lower() for v in PolicyVariant]}",
        )

    repo_root = Path(__file__).resolve().parents[3]
    lookahead_m = args.lookahead_m if args.lookahead_m is not None else SUBGOAL_LOOKAHEAD_M
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
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    if getattr(env_cfg, "sensors", None) is not None and args.sensors:
        from strafer_lab.tasks.navigation.composed_env_cfg import SensorStackCfg

        tokens = tuple(t.strip() for t in args.sensors.split(",") if t.strip())
        env_cfg.sensors = SensorStackCfg(cameras_required=tokens)
        env_cfg.__post_init__()
        env_cfg.scene.num_envs = 1
    cameras_required = tuple(env_cfg.sensors.cameras_required)

    if args.scene_usd is not None:
        env_cfg.scene.scene_geometry.spawn.usd_path = str(Path(args.scene_usd).resolve())
    active_spawn_points = _resolve_active_spawn_points(args.scene)
    if active_spawn_points:
        env_cfg.events.reset_robot.params["spawn_points_xy"] = active_spawn_points
        if hasattr(env_cfg.commands, "goal_command"):
            env_cfg.commands.goal_command.spawn_points_xy = active_spawn_points
    # The driver owns episode boundaries; a training-scale timeout must not
    # teleport the robot mid-traversal.
    if getattr(env_cfg.terminations, "time_out", None) is not None:
        env_cfg.terminations.time_out = None

    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    step_dt = float(unwrapped.sim.get_physics_dt()) * int(unwrapped.cfg.decimation)
    writer_fps = max(1, round(1.0 / (step_dt * max(1, args.capture_every_n_steps))))

    # --- controller + detections + writer + adapter --------------------------
    policy = load_policy(args.checkpoint, variant)
    cursor = PathCursor(num_envs=1, max_points=_MAX_PATH_POINTS, device=device)
    imu_cfg = SceneEntityCfg("d555_imu")
    leg_end_distance = {"value": float("inf")}

    def _wp_to_torch(arr):
        if hasattr(arr, "detach"):
            return arr
        import warp as wp

        return wp.to_torch(arr)

    def _field(tensor) -> np.ndarray:
        return _wp_to_torch(tensor)[0].detach().cpu().numpy()

    # The coverage plan + occupancy grid are in the scene's authored frame; the
    # robot's world pose carries the env-instance origin offset. Plan and track
    # in scene-local frame (subtract the origin), matching the SubgoalCommand.
    env_origin = _wp_to_torch(unwrapped.scene.env_origins)[:, :2]

    def _robot_xy_local_all() -> "torch.Tensor":
        world = _wp_to_torch(unwrapped.scene["robot"].data.root_pos_w)[:, :2]
        return world - env_origin

    def _action_source():
        robot_xy_all = _robot_xy_local_all()
        robot_quat = _wp_to_torch(unwrapped.scene["robot"].data.root_quat_w)[0]
        robot_yaw = yaw_from_quat_xyzw(tuple(float(v) for v in robot_quat.detach().cpu().tolist()))
        state = cursor.update(robot_xy_all, lookahead_m)
        leg_end_distance["value"] = float(state.end_distance[0].item())
        subgoal_xy = state.subgoal_xy[0].detach().cpu().tolist()
        base_fields = {
            "imu_accel": _field(mdp.imu_linear_acceleration(unwrapped, imu_cfg)),
            "imu_gyro": _field(mdp.imu_angular_velocity(unwrapped, imu_cfg)),
            "encoder_vels_ticks": _field(mdp.wheel_encoder_velocities(unwrapped)),
            "body_velocity_xy": _field(mdp.body_velocity_xy(unwrapped)),
            "last_action": _field(mdp.last_action(unwrapped)),
        }
        robot_xy = robot_xy_all[0].detach().cpu().tolist()
        vx, vy, wz = step_subgoal_controller(
            policy, base_fields, robot_xy, robot_yaw, subgoal_xy, variant=variant,
        )
        return ((vx, vy, 0.0), (0.0, 0.0, wz))

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
        cmd_vel_source=_action_source,
        cameras_required=cameras_required,
        detections_source=detections_source,
        step_dt=step_dt,
    )
    recorder = CoverageLeRobotRecorder(writer=writer, scene_id=args.scene)

    # --- traversal -----------------------------------------------------------
    try:
        adapter.reset(scene_name=args.scene)
        mount_quat = None
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

            robot_xy = _robot_xy_local_all()[0].detach().cpu().numpy()
            leg = _leg_path(
                robot_xy,
                np.asarray(visit.target_xy, dtype=np.float32),
                visit.approach_heading_rad,
                free_space,
                grid_res=occupancy.resolution_m,
                grid_origin_xy=occupancy.origin_xy,
                approach_distance_m=args.approach_distance_m,
                plan_path=plan_path,
                error_cls=PathPlanningError,
            )
            if leg is None:
                print(
                    f"[coverage_capture] skip room {visit.room_index} visit "
                    f"{visit.visit_ordinal}: no collision-free path",
                    flush=True,
                )
                continue
            cursor.set_paths(
                torch.tensor([0], device=device),
                [torch.as_tensor(leg, dtype=torch.float32, device=device)],
            )
            leg_end_distance["value"] = float("inf")

            start_bundle = adapter.capture()
            recorder.begin_episode(
                start_bundle=start_bundle,
                episode_split=episode_split,
                realized_d555_mount_quat=mount_quat,
            )
            recorder.add_frame(start_bundle)
            for step in range(1, args.max_steps_per_leg + 1):
                adapter.step()
                if step % args.capture_every_n_steps == 0:
                    recorder.add_frame(adapter.capture())
                if leg_end_distance["value"] < _PATH_COMPLETE_THRESHOLD_M:
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


if __name__ == "__main__":
    sys.exit(main())
