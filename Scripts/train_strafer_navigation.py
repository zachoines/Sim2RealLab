#!/usr/bin/env python3
"""Training script for Strafer navigation task using RSL-RL PPO.

Environments (--env):
    Isaac-Strafer-Nav-v0          Ideal Full (no noise, debugging)
    Isaac-Strafer-Nav-Depth-v0    Ideal Depth-only
    Isaac-Strafer-Nav-NoCam-v0    Ideal Proprioceptive-only (fastest)
    Isaac-Strafer-Nav-Real-v0     Realistic Full (default, sim-to-real)
    Isaac-Strafer-Nav-Real-Depth-v0  Realistic Depth-only
    Isaac-Strafer-Nav-Robust-v0   Robust Full (stress-test training)

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
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
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
    args = parser.parse_args()

    # Import Isaac Lab app launcher first
    from isaaclab.app import AppLauncher

    # Launch the simulator
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now import other modules
    import gymnasium as gym
    import torch

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerEnv, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    # Import strafer_lab to register environments and get agent config
    import strafer_lab  # noqa: F401
    from strafer_lab.tasks.navigation.agents.rsl_rl_ppo_cfg import StraferPPORunnerCfg

    print("\n" + "=" * 60)
    print("Strafer Navigation Training - RSL-RL PPO")
    print("=" * 60)
    print(f"Number of environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Log directory: {args.log_dir}")
    print("=" * 60 + "\n")

    # Create environment
    env_name = args.env
    print(f"Creating environment: {env_name}")
    print(
        f"Available: Isaac-Strafer-Nav-{{v0,Depth-v0,NoCam-v0,Real-v0,Real-Depth-v0,Robust-v0}}"
    )

    # Parse environment config from registry (Isaac Lab pattern)
    env_cfg = parse_env_cfg(
        env_name,
        device=args.device,
        num_envs=args.num_envs,
    )
    env = gym.make(env_name, cfg=env_cfg)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    print("[OK] Environment created and wrapped for RSL-RL")

    # Get runner config
    agent_cfg = StraferPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = os.path.abspath(args.log_dir)
    log_dir = os.path.join(log_root, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    # Create runner
    runner = RslRlOnPolicyRunnerEnv(
        env=env,
        train_cfg=agent_cfg,
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        runner.load(args.resume)

    # Train
    print("\n--- Starting Training ---")
    runner.learn(num_learning_iterations=args.max_iterations)

    print("\n[OK] Training complete!")
    print(f"Checkpoints saved to: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
