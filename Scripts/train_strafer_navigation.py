#!/usr/bin/env python3
"""Training script for Strafer navigation task using RSL-RL PPO.

Environments (--env):
    Isaac-Strafer-Nav-v0          Ideal Full (no noise, debugging)
    Isaac-Strafer-Nav-Depth-v0    Ideal Depth-only
    Isaac-Strafer-Nav-NoCam-v0    Ideal Proprioceptive-only (fastest)
    Isaac-Strafer-Nav-Real-v0     Realistic Full (default, sim-to-real)
    Isaac-Strafer-Nav-Real-Depth-v0  Realistic Depth-only
    Isaac-Strafer-Nav-Robust-v0   Robust Full (stress-test training)
    Isaac-Strafer-Nav-Real-ProcDepth-v0   Realistic + procedural scenes (Phase 6)
    Isaac-Strafer-Nav-Robust-ProcDepth-v0 Robust + procedural scenes (Phase 6)

Usage:
    ./IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py --num_envs 512
    .\IsaacLab\isaaclab.bat -p Scripts\train_strafer_navigation.py --num_envs 512
    .\IsaacLab\isaaclab.bat -p Scripts\train_strafer_navigation.py --headless --num_envs 1024
    .\IsaacLab\isaaclab.bat -p Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-NoCam-v0 --num_envs 4096
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
        "--max_iterations", type=int, default=1000, help="Maximum training iterations"
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
    # Import Isaac Lab app launcher and add its CLI args (--enable_cameras, etc.)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Auto-enable cameras for variants that use depth/RGB
    if "NoCam" not in args.env:
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

    # Import strafer_lab to register environments and inject custom network
    import strafer_lab  # noqa: F401
    from strafer_lab.tasks.navigation.agents import register_strafer_network
    register_strafer_network()

    env_name = args.env

    print("\n" + "=" * 60)
    print("Strafer Navigation Training - RSL-RL PPO")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Log directory: {args.log_dir}")
    print("=" * 60 + "\n")

    # Parse environment config from registry (Isaac Lab pattern)
    env_cfg = parse_env_cfg(
        env_name,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.seed = args.seed

    # Load the agent config registered for this env variant
    agent_cfg = load_cfg_from_registry(env_name, "rsl_rl_cfg_entry_point")
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed

    # Create environment
    env = gym.make(env_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    print("[OK] Environment created and wrapped for RSL-RL")

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = os.path.abspath(args.log_dir)
    log_dir = os.path.join(log_root, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        runner.load(args.resume)

    # Train
    print("\n--- Starting Training ---")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print("\n[OK] Training complete!")
    print(f"Checkpoints saved to: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
