#!/usr/bin/env python3
"""Play a trained Strafer navigation checkpoint (inference-only rollout).

Loads an OnPolicyRunner checkpoint and steps the env in inference mode so you
can watch the policy in the Kit viewport (or record a video).

Examples:
    # Headed rollout against the play variant (8 envs):
    $ISAACLAB -p Scripts/play_strafer_navigation.py \\
        --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \\
        --checkpoint logs/rsl_rl/strafer_navigation/run_20260425_035916/model_600.pt \\
        --viz kit

    # Headless rollout that records a single MP4 of the rollout:
    $ISAACLAB -p Scripts/play_strafer_navigation.py \\
        --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \\
        --checkpoint logs/rsl_rl/strafer_navigation/run_20260425_035916/model_600.pt \\
        --headless --video --video_length 600
"""

import argparse
import os
import time
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained Strafer policy")
    parser.add_argument("--env", type=str, required=True,
                        help="Registered task ID (Play variant recommended for visualization)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model_*.pt produced by train_strafer_navigation.py")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Override scene.num_envs (default: cfg's num_envs, e.g. 8 for ProcRoom-Depth-Play)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Total inference steps before exit (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real_time", action="store_true",
                        help="Sleep to match env step_dt (only useful when watching headed)")
    parser.add_argument("--video", action="store_true",
                        help="Record a single MP4 of the rollout")
    parser.add_argument("--video_length", type=int, default=600,
                        help="Frames in the rollout MP4 (default: 600)")
    parser.add_argument("--video_dir", type=str, default="logs/rsl_rl/strafer_navigation/play_videos",
                        help="Directory for the recorded MP4 (default under logs/)")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if "NoCam" not in args.env or args.video:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab.envs.common import ViewerCfg
    import importlib.metadata as _metadata

    import strafer_lab  # noqa: F401  (registers envs)

    env_cfg = parse_env_cfg(
        args.env,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.seed = args.seed

    if args.video:
        env_cfg.viewer = ViewerCfg(
            eye=(0.0, 0.0, 12.0),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
            env_index=0,
            resolution=(1280, 720),
        )

    agent_cfg = load_cfg_from_registry(args.env, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))

    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.env, cfg=env_cfg, render_mode=render_mode)

    if args.video:
        unwrapped = env.unwrapped
        env_origin = unwrapped.scene.env_origins[0].detach().cpu().tolist()
        viewer = unwrapped.cfg.viewer
        world_eye = (
            env_origin[0] + viewer.eye[0],
            env_origin[1] + viewer.eye[1],
            env_origin[2] + viewer.eye[2],
        )
        world_target = (
            env_origin[0] + viewer.lookat[0],
            env_origin[1] + viewer.lookat[1],
            env_origin[2] + viewer.lookat[2],
        )
        recorder = getattr(unwrapped, "video_recorder", None)
        capture = getattr(recorder, "_capture", None) if recorder is not None else None
        if capture is not None:
            capture.cfg.camera_position = world_eye
            capture.cfg.camera_target = world_target
        unwrapped.sim.set_camera_view(eye=world_eye, target=world_target)
        try:
            from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
            set_kit_renderer_camera_view(
                eye=world_eye, target=world_target,
                camera_prim_path=viewer.cam_prim_path,
            )
        except ImportError:
            pass

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath(os.path.join(args.video_dir, f"play_{timestamp}"))
        os.makedirs(out_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=out_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording rollout to: {out_dir}")

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    print(f"[INFO] Rolling out {args.steps} steps (dt={dt:.4f}s, real_time={args.real_time})")

    try:
        for step in range(args.steps):
            t0 = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            if args.video and step + 1 >= args.video_length:
                break
            if args.real_time:
                sleep = dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
