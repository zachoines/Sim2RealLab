"""Boot the Strafer perception env with the ROS2 bridge wired up.

This is the DGX-side entry point for sim-in-the-loop runs. It launches
Isaac Sim via ``AppLauncher``, instantiates a perception-camera task at
``num_envs=1``, enables the bundled ``isaacsim.ros2.bridge`` extension,
builds the Strafer OmniGraph (cameras, odom, TF, /cmd_vel subscribe),
and then runs in one of two modes:

  - ``--mode bridge``: drive the env step loop reading /cmd_vel from the
    bridge and injecting it into the action tensor. Used for manual ops
    where the Jetson side publishes /cmd_vel directly (rqt teleop, the
    real Nav2 stack on the LAN, ``ros2 topic pub`` smoke checks).

  - ``--mode harness``: also instantiate the sim-in-the-loop harness,
    walk every mission emitted by ``MissionGenerator`` for the configured
    scene, submit each one to the Jetson autonomy executor over the
    ``execute_mission`` action, and capture reachability-labeled frames
    into a ``frames.jsonl`` dataset.

The Jetson at ``STRAFER_JETSON_HOST`` consumes the same real-robot
topics in either mode:

    /d555/color/image_raw          sensor_msgs/Image
    /d555/color/camera_info        sensor_msgs/CameraInfo
    /d555/depth/image_rect_raw     sensor_msgs/Image
    /d555/depth/camera_info        sensor_msgs/CameraInfo
    /strafer/odom                  nav_msgs/Odometry
    tf                             odom→base_link, base_link→d555_link
    /cmd_vel                       geometry_msgs/Twist  (SUBSCRIBE)

Usage:

    source env_setup.sh   # loads ROS_DOMAIN_ID, RMW_IMPLEMENTATION, LD_PRELOAD

    # Manual / Nav2 mode:
    isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py

    # Reachability-labeled dataset capture:
    isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py \\
        --mode harness \\
        --scene-metadata Assets/generated/scenes/kitchen_01/scene_metadata.json \\
        --scene-usd Assets/generated/scenes/kitchen_01/scene.usdc \\
        --output data/sim_in_the_loop/kitchen_01

Verify from another shell (Jetson or DGX):

    ros2 topic list
    ros2 topic hz /d555/color/image_raw
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.3}}'
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from isaaclab.app import AppLauncher


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("bridge", "harness"),
        default="bridge",
        help="Run mode: 'bridge' drives /cmd_vel into the env (default); "
             "'harness' walks scene_metadata.json missions through the "
             "Jetson's execute_mission action and writes a reachability dataset.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0",
        help="Isaac Lab task carrying the 640x360 perception camera. "
        "num_envs is forced to 1 for sim-in-the-loop.",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="/World/ROS2Bridge",
        help="USD path for the OmniGraph prim the bridge is built into.",
    )
    parser.add_argument(
        "--cmd-vel-timeout",
        type=float,
        default=0.5,
        help="Bridge mode only: seconds with no /cmd_vel before holding still.",
    )
    parser.add_argument(
        "--scene-usd",
        type=Path,
        default=None,
        help="Override the env's default scene USD with this path.",
    )

    # Harness-mode args
    parser.add_argument(
        "--scene-metadata",
        type=Path,
        default=None,
        help="Harness mode: path to scene_metadata.json describing the targets.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Harness mode: scene_name for frames.jsonl. Defaults to the "
             "parent dir name of --scene-metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Harness mode: output dir for the reachability dataset (one "
             "episode_NNNN/ per mission).",
    )
    parser.add_argument(
        "--max-missions",
        type=int,
        default=None,
        help="Harness mode: cap on missions to run from the generator.",
    )
    parser.add_argument(
        "--allowed-labels",
        nargs="*",
        default=None,
        help="Harness mode: only target objects whose label is in this list.",
    )
    parser.add_argument(
        "--blocked-labels",
        nargs="*",
        default=("wall", "floor", "ceiling"),
        help="Harness mode: skip targets whose label is in this list.",
    )
    parser.add_argument(
        "--mission-timeout-s",
        type=float,
        default=60.0,
        help="Harness mode: seconds before the harness cancels a stuck mission.",
    )
    parser.add_argument(
        "--capture-every-n-steps",
        type=int,
        default=5,
        help="Harness mode: env-step interval for frame capture during nav.",
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    if args.mode == "harness":
        _validate_harness_args(args)

    # AppLauncher must boot Isaac Sim before any omni.* / strafer_lab.tasks imports.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    import strafer_lab.tasks  # noqa: F401

    from isaacsim.core.utils.extensions import enable_extension
    from isaaclab_tasks.utils import parse_env_cfg

    from strafer_lab.bridge.config import build_default_bridge_config
    from strafer_lab.bridge.graph import build_bridge_graph

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Enable the bridge extension before any OmniGraph nodes are created.
    # ``isaacsim.ros2.bridge`` registers its node types with OmniGraph at
    # extension-enable time; attempting ``og.Controller.edit(...)`` with
    # ``isaacsim.ros2.bridge.*`` before the extension is hot throws a
    # "node type not registered" error.
    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    if args.scene_usd is not None:
        env_cfg.scene.scene_geometry.spawn.usd_path = str(args.scene_usd.resolve())
        print(f"[sim_in_the_loop] scene USD override → {env_cfg.scene.scene_geometry.spawn.usd_path}")

    env = gym.make(args.task, cfg=env_cfg)

    config = build_default_bridge_config(graph_path=args.graph_path)
    build_bridge_graph(config)
    print(f"[sim_in_the_loop] bridge graph built at {args.graph_path}")
    print(f"[sim_in_the_loop] chassis_prim={config.chassis_prim_path}")
    print(f"[sim_in_the_loop] color camera prim={config.color_camera.camera_prim_path}")

    if args.mode == "bridge":
        _run_bridge_mode(simulation_app, env, args)
    else:
        _run_harness_mode(simulation_app, env, args)

    env.close()
    simulation_app.close()


# ---------------------------------------------------------------------------
# Mode: bridge (drive /cmd_vel into the env)
# ---------------------------------------------------------------------------


def _run_bridge_mode(simulation_app, env, args) -> None:
    import torch

    from strafer_lab.bridge.graph import read_cmd_vel

    unwrapped = env.unwrapped
    env.reset()

    action_shape = unwrapped.action_manager.action.shape
    print(f"[sim_in_the_loop] action tensor shape = {tuple(action_shape)}")

    device = unwrapped.device
    zero_action = torch.zeros(action_shape, device=device)

    last_cmd_time = time.monotonic()
    while simulation_app.is_running():
        linear, angular = read_cmd_vel(args.graph_path)
        vx, vy, _vz = linear
        _wx, _wy, wz = angular

        now = time.monotonic()
        if any(abs(v) > 1e-6 for v in (vx, vy, wz)):
            last_cmd_time = now
            action = zero_action.clone()
            if action.shape[-1] >= 3:
                action[0, 0] = float(vx)
                action[0, 1] = float(vy)
                action[0, 2] = float(wz)
        elif now - last_cmd_time > args.cmd_vel_timeout:
            action = zero_action
        else:
            action = zero_action

        env.step(action)


# ---------------------------------------------------------------------------
# Mode: harness (walk MissionGenerator, capture reachability dataset)
# ---------------------------------------------------------------------------


def _run_harness_mode(simulation_app, env, args) -> None:
    from strafer_lab.sim_in_the_loop import (
        HarnessConfig,
        MissionGenerator,
        SimInTheLoopHarness,
    )
    from strafer_lab.sim_in_the_loop.runtime_env import IsaacLabEnvAdapter
    from strafer_lab.sim_in_the_loop.runtime_mission import Ros2MissionApi
    from strafer_lab.tools.perception_writer import PerceptionFrameWriter

    scene_name = args.scene_name or args.scene_metadata.parent.name

    generator = MissionGenerator.from_metadata_path(
        scene_metadata_path=args.scene_metadata,
        scene_name=scene_name,
        max_missions=args.max_missions,
        allowed_labels=args.allowed_labels,
        blocked_labels=tuple(args.blocked_labels or ()),
    )
    missions = list(generator)
    if not missions:
        print(f"[sim_in_the_loop] no missions to run for {scene_name}; exiting")
        return
    print(f"[sim_in_the_loop] {len(missions)} missions queued for {scene_name}")

    args.output.mkdir(parents=True, exist_ok=True)
    writer = PerceptionFrameWriter(output_root=args.output)

    env_adapter = IsaacLabEnvAdapter(
        env=env,
        graph_path=args.graph_path,
        scene_name=scene_name,
    )

    cfg = HarnessConfig(
        mission_timeout_s=args.mission_timeout_s,
        capture_every_n_steps=args.capture_every_n_steps,
        scene_type="infinigen_sim_in_the_loop",
    )

    with Ros2MissionApi() as mission_api:
        harness = SimInTheLoopHarness(
            env_adapter=env_adapter,
            mission_api=mission_api,
            writer=writer,
            config=cfg,
        )

        for spec in missions:
            if not simulation_app.is_running():
                print("[sim_in_the_loop] simulation app shut down; aborting run")
                break
            outcome = harness.run_one_mission(spec)
            print(
                f"[sim_in_the_loop] {spec.mission_id} reachable={outcome.reachability} "
                f"state={outcome.final_status.state} frames={outcome.frames_written} "
                f"elapsed={outcome.elapsed_s:.1f}s"
            )

    print(f"[sim_in_the_loop] writer stats: {writer.stats.to_dict()}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_harness_args(args: argparse.Namespace) -> None:
    missing = []
    if args.scene_metadata is None:
        missing.append("--scene-metadata")
    if args.output is None:
        missing.append("--output")
    if missing:
        raise SystemExit(
            f"--mode harness requires {' and '.join(missing)} to be set"
        )
    if not args.scene_metadata.is_file():
        raise SystemExit(f"scene metadata not found: {args.scene_metadata}")


if __name__ == "__main__":
    main()
