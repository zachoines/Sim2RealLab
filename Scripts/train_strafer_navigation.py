#!/usr/bin/env python3
"""Training script for Strafer navigation task using RSL-RL PPO.

Environments (--env):
    Isaac-Strafer-Nav-v0          Ideal Full (no noise, debugging)
    Isaac-Strafer-Nav-Depth-v0    Ideal Depth-only
    Isaac-Strafer-Nav-NoCam-v0    Ideal Proprioceptive-only (fastest)
    Isaac-Strafer-Nav-Real-v0     Realistic Full (default, sim-to-real)
    Isaac-Strafer-Nav-Real-Depth-v0  Realistic Depth-only
    Isaac-Strafer-Nav-Robust-v0   Robust Full (stress-test training)
    Isaac-Strafer-Nav-Real-InfinigenDepth-v0   Realistic + Infinigen scenes (Phase 6)
    Isaac-Strafer-Nav-Robust-InfinigenDepth-v0 Robust + Infinigen scenes (Phase 6)
    Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0   Realistic + Procedural rooms (Phase 7)
    Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-v0 Robust + Procedural rooms (Phase 7)

Auxiliary losses (--aux):
    dapg    Demo Augmented Policy Gradient (NLL on expert demos)
    gail    Generative Adversarial Imitation Learning (WGAN-GP discriminator)

    Examples:
        --aux dapg --dapg_demos demos.h5
        --aux gail --gail_demos demos.h5
        --aux dapg --aux gail --dapg_demos demos.h5 --gail_demos demos.h5

    Legacy flags (--bc_demos etc.) are still supported for backward compat.

Video recording (overhead view):
    --video                Record periodic MP4 videos during training
    --video_length 200     Frames per clip (default: 200)
    --video_interval 2000  Steps between recordings (default: 2000)

Usage:
    .\\IsaacLab\\isaaclab.bat -p Scripts\\train_strafer_navigation.py --num_envs 512
    .\\IsaacLab\\isaaclab.bat -p Scripts\\train_strafer_navigation.py --headless --num_envs 1024
    .\\IsaacLab\\isaaclab.bat -p Scripts\\train_strafer_navigation.py --video --num_envs 64
"""

import argparse
import os
from datetime import datetime


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Strafer navigation policy")
    parser.add_argument(
        "--num_envs", type=int, default=512, help="Number of parallel environments"
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Maximum training iterations (default: from agent config)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/rsl_rl/strafer_navigation",
        help="Log directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Isaac-Strafer-Nav-Real-v0",
        help="Environment ID (default: Isaac-Strafer-Nav-Real-v0 = Realistic Full)",
    )
    # Video recording
    parser.add_argument(
        "--video", action="store_true", default=False,
        help="Record periodic overhead-view videos during training",
    )
    parser.add_argument(
        "--video_length", type=int, default=200,
        help="Length of each recorded video clip in steps (default: 200)",
    )
    parser.add_argument(
        "--video_interval", type=int, default=2000,
        help="Steps between video recordings (default: 2000)",
    )
    # Modular auxiliary losses
    parser.add_argument(
        "--aux", type=str, action="append", default=[],
        choices=["dapg", "gail"],
        help="Auxiliary loss modules to activate (can specify multiple)",
    )
    # Legacy DAPG flags (backward compatibility)
    parser.add_argument("--bc_demos", type=str, default=None,
                        help="[Legacy] Path to HDF5 demo file (use --aux dapg --dapg_demos instead)")
    parser.add_argument("--bc_weight", type=float, default=0.03, help=argparse.SUPPRESS)
    parser.add_argument("--bc_decay_steps", type=int, default=3000, help=argparse.SUPPRESS)
    parser.add_argument("--bc_batch_size", type=int, default=128, help=argparse.SUPPRESS)
    parser.add_argument("--bc_min_return_pct", type=float, default=0.0, help=argparse.SUPPRESS)

    # Import Isaac Lab app launcher and add its CLI args (--enable_cameras, etc.)
    from isaaclab.app import AppLauncher

    # Add auxiliary-specific CLI args before parsing
    from strafer_lab.tasks.navigation.agents.aux_dapg import DAPGAuxiliary
    from strafer_lab.tasks.navigation.agents.aux_gail import GAILAuxiliary
    DAPGAuxiliary.add_args(parser)
    GAILAuxiliary.add_args(parser)

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Auto-enable cameras for variants that use depth/RGB or video recording
    if "NoCam" not in args.env or args.video:
        args.enable_cameras = True

    # Launch the simulator
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now import other modules
    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.hydra import load_cfg_from_registry
    from isaaclab.envs.common import ViewerCfg

    # Import strafer_lab to register environments and inject custom network
    import strafer_lab  # noqa: F401
    from strafer_lab.tasks.navigation.agents import (
        register_strafer_network,
        install_strafer_ppo,
        register_auxiliary,
    )
    register_strafer_network()

    env_name = args.env

    # Parse environment config from registry (Isaac Lab pattern)
    env_cfg = parse_env_cfg(
        env_name,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.seed = args.seed

    # Set up overhead camera for video recording
    if args.video:
        env_cfg.viewer = ViewerCfg(
            eye=(0.0, 0.0, 12.0),
            lookat=(0.0, 0.0, 0.0),
            origin_type="env",
            env_index=0,
            resolution=(1280, 720),
        )

    # Load the agent config registered for this env variant
    agent_cfg = load_cfg_from_registry(env_name, "rsl_rl_cfg_entry_point")
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed

    print("\n" + "=" * 60)
    print("Strafer Navigation Training - RSL-RL PPO")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Max iterations: {agent_cfg.max_iterations}")
    print(f"Log directory: {args.log_dir}")
    if args.video:
        print(f"Video: every {args.video_interval} steps, {args.video_length} frames/clip")
    if args.aux:
        print(f"Auxiliary losses: {', '.join(args.aux)}")
    print("=" * 60 + "\n")

    # Create environment (with render_mode for video if needed)
    render_mode = "rgb_array" if args.video else None
    env = gym.make(env_name, cfg=env_cfg, render_mode=render_mode)

    # Wrap with video recorder before RSL-RL wrapper
    if args.video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_root = os.path.abspath(args.log_dir)
        log_dir = os.path.join(log_root, f"run_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)

        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL wrapper must be last
    env = RslRlVecEnvWrapper(env)
    print("[OK] Environment created and wrapped for RSL-RL")

    # Setup logging (use same log_dir if video already created it)
    if not args.video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_root = os.path.abspath(args.log_dir)
        log_dir = os.path.join(log_root, f"run_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    # --- Register auxiliary losses ---
    # Handle legacy --bc_demos flag (maps to DAPG)
    use_legacy_dapg = args.bc_demos is not None and "dapg" not in args.aux
    has_auxiliaries = bool(args.aux) or use_legacy_dapg

    if has_auxiliaries:
        install_strafer_ppo()

    if use_legacy_dapg:
        # Legacy path: use old-style --bc_* flags
        from strafer_lab.tasks.navigation.agents import register_dapg_loss
        register_dapg_loss(
            demo_path=args.bc_demos,
            bc_weight=args.bc_weight,
            bc_decay_steps=args.bc_decay_steps,
            bc_batch_size=args.bc_batch_size,
            min_return_percentile=args.bc_min_return_pct,
            device=agent_cfg.device,
        )
        print("[WARN] Using legacy --bc_demos flag. "
              "Prefer --aux dapg --dapg_demos for new training runs.")
    else:
        # New modular path
        if "dapg" in args.aux:
            if args.dapg_demos is None:
                raise ValueError("--aux dapg requires --dapg_demos <path>")
            register_auxiliary(DAPGAuxiliary.from_args(args, device=agent_cfg.device))

        if "gail" in args.aux:
            if args.gail_demos is None:
                raise ValueError("--aux gail requires --gail_demos <path>")
            register_auxiliary(GAILAuxiliary.from_args(args, device=agent_cfg.device))

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        runner.load(args.resume)

    # Train
    print("\n--- Starting Training ---")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print("\n[OK] Training complete!")
    print(f"Checkpoints saved to: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
