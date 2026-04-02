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

    Examples (demo obs_dim must match env variant — use Depth demos for Depth envs):
        --aux dapg --dapg_demos demos.h5
        --aux gail --gail_demos demos.h5
        --aux dapg --aux gail --dapg_demos demos.h5 --gail_demos demos.h5

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
    # Phase 1: Parse core args + AppLauncher args (before simulator launch).
    # Auxiliary-specific args (--dapg_*, --gail_*) are left as unknown here
    # and parsed in phase 2 after the aux modules can register their own args.
    parser = argparse.ArgumentParser(description="Train Strafer navigation policy")
    parser.add_argument(
        "--num_envs", type=int, default=512, help="Number of parallel environments"
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Maximum training iterations (default: from agent config)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=None,
        help="Rollout steps per env per update (default: from agent config, e.g. 48). "
             "Lower values reduce VRAM, allowing more envs for sample diversity.",
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
    # Network overrides
    parser.add_argument(
        "--depth_encoder", type=str, default=None, choices=["defm", "cnn"],
        help="Depth encoder type (default: from network config, typically 'defm'). "
             "'cnn' uses a lightweight trainable CNN instead of the frozen DeFM backbone.",
    )
    # Learning rate schedule
    parser.add_argument(
        "--lr_schedule", type=str, default=None, choices=["cosine", "linear"],
        help="LR decay schedule (default: use config's schedule). "
             "Overrides the config schedule with a smooth decay to --lr_min.",
    )
    parser.add_argument(
        "--lr_min", type=float, default=1e-5,
        help="Minimum learning rate for cosine/linear decay (default: 1e-5).",
    )
    # Modular auxiliary losses
    parser.add_argument(
        "--aux", type=str, action="append", default=[],
        choices=["dapg", "gail"],
        help="Auxiliary loss modules to activate (can specify multiple)",
    )

    # Import Isaac Lab app launcher and add its CLI args (--enable_cameras, etc.)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, remainder = parser.parse_known_args()

    # Auto-enable cameras for variants that use depth/RGB or video recording
    if "NoCam" not in args.env or args.video:
        args.enable_cameras = True

    # Launch the simulator
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now import other modules (requires simulator to be running)
    import gymnasium as gym
    import torch

    from rsl_rl.runners import OnPolicyRunner
    import importlib.metadata as _metadata
    _rsl_rl_version = _metadata.version("rsl-rl-lib")
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
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

    # Let aux modules define their own CLI args, then re-parse remainder.
    from strafer_lab.tasks.navigation.agents.aux_dapg import DAPGAuxiliary
    from strafer_lab.tasks.navigation.agents.aux_gail import GAILAuxiliary

    aux_parser = argparse.ArgumentParser()
    DAPGAuxiliary.add_args(aux_parser)
    GAILAuxiliary.add_args(aux_parser)
    aux_args = aux_parser.parse_args(remainder)

    # Merge aux args into main namespace so everything is in one place
    for key, value in vars(aux_args).items():
        setattr(args, key, value)

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
    if args.num_steps is not None:
        agent_cfg.num_steps_per_env = args.num_steps
    agent_cfg.seed = args.seed

    print("\n" + "=" * 60)
    print("Strafer Navigation Training - RSL-RL PPO")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Steps per env per update: {agent_cfg.num_steps_per_env}")
    print(f"Max iterations: {agent_cfg.max_iterations}")
    print(f"Log directory: {args.log_dir}")
    if args.video:
        print(f"Video: every {args.video_interval} steps, {args.video_length} frames/clip")
    if args.depth_encoder:
        print(f"Depth encoder: {args.depth_encoder} (CLI override)")
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
    if args.aux:
        install_strafer_ppo()

        if "dapg" in args.aux:
            if args.dapg_demos is None:
                raise ValueError("--aux dapg requires --dapg_demos <path>")
            register_auxiliary(DAPGAuxiliary.from_args(args, device=agent_cfg.device))

        if "gail" in args.aux:
            if args.gail_demos is None:
                raise ValueError("--aux gail requires --gail_demos <path>")
            register_auxiliary(GAILAuxiliary.from_args(args, device=agent_cfg.device))

    # Create runner (inject CLI overrides into the config dict)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _rsl_rl_version)
    agent_dict = agent_cfg.to_dict()
    if args.depth_encoder is not None:
        agent_dict["policy"]["depth_encoder_type"] = args.depth_encoder
        print(f"[Override] depth_encoder_type = {args.depth_encoder}")
    # Force schedule="fixed" when using CLI LR schedule to prevent
    # RSL-RL's KL-adaptive controller from fighting the decay.
    if args.lr_schedule is not None:
        agent_dict["algorithm"]["schedule"] = "fixed"
    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)

    # Attach LR decay schedule (monkey-patches alg.update to apply decay)
    if args.lr_schedule is not None:
        import math as _math

        _lr_init = agent_dict["algorithm"]["learning_rate"]
        _lr_min = args.lr_min
        _total_iters = agent_cfg.max_iterations
        _schedule_type = args.lr_schedule
        _original_update = runner.alg.update

        def _update_with_lr_schedule():
            result = _original_update()
            it = runner.current_learning_iteration
            progress = min(it / max(_total_iters, 1), 1.0)
            if _schedule_type == "cosine":
                lr = _lr_min + 0.5 * (_lr_init - _lr_min) * (1.0 + _math.cos(_math.pi * progress))
            else:  # linear
                lr = _lr_init + (_lr_min - _lr_init) * progress
            runner.alg.learning_rate = lr
            return result

        runner.alg.update = _update_with_lr_schedule
        print(f"[Override] LR schedule = {args.lr_schedule} ({_lr_init:.1e} → {_lr_min:.1e} over {_total_iters} iters)")

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
