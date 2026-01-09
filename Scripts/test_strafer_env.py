#!/usr/bin/env python3
"""Test script to verify Strafer environment loads correctly in Isaac Lab.

This script must be run with Isaac Lab's Python interpreter, e.g.:
    ./IsaacLab/isaaclab.sh -p Scripts/test_strafer_env.py

Or on Windows:
    .\IsaacLab\isaaclab.bat -p Scripts\test_strafer_env.py
"""

import argparse


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Strafer environment")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    args = parser.parse_args()

    # Import Isaac Lab app launcher first (required before other isaaclab imports)
    from isaaclab.app import AppLauncher

    # Launch the simulator
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now import Isaac Lab modules
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    # Import strafer_lab to register environments
    import strafer_lab  # noqa: F401

    print("\n" + "=" * 60)
    print("Strafer Lab Environment Test")
    print("=" * 60)

    # List registered Strafer environments
    strafer_envs = [e for e in gym.envs.registry.keys() if "Strafer" in e]
    print(f"\nRegistered Strafer environments: {strafer_envs}")

    if not strafer_envs:
        print("\n[ERROR] No Strafer environments found!")
        print("Make sure strafer_lab is installed: pip install -e source/strafer_lab")
        simulation_app.close()
        return

    # Create the navigation environment
    env_name = "Isaac-Strafer-Navigation-v0"
    print(f"\nCreating environment: {env_name}")
    
    try:
        # Parse environment config from registry (Isaac Lab pattern)
        env_cfg = parse_env_cfg(
            env_name,
            device=args.device,
            num_envs=args.num_envs,
        )
        # Create environment with the parsed config
        env = gym.make(env_name, cfg=env_cfg)
        print("[OK] Environment created successfully!")
        
        # Print environment info
        print(f"\n--- Environment Info ---")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Reset and step
        print(f"\n--- Running test episode ---")
        obs, info = env.reset()
        print(f"Observation shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")
        
        # Run a few steps with random actions
        for step in range(100000):
            action = torch.zeros(args.num_envs, 3)  # [vx, vy, omega] = 0
            if step > 2:
                action[:, 0] = 0.5  # Move forward
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step + 1}: reward={reward.mean().item():.4f}")
        
        print("\n[OK] Environment test passed!")
        env.close()
        
    except Exception as e:
        print(f"\n[ERROR] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()

    simulation_app.close()


if __name__ == "__main__":
    main()
