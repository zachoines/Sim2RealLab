#!/usr/bin/env python3
"""Diagnostic script to check wheel velocity targets and actual wheel velocities."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import gymnasium as gym
    import strafer_lab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # Create environment using Isaac Lab's method
    env_cfg = parse_env_cfg("Isaac-Strafer-Navigation-v0", num_envs=1, device=args.device)
    env = gym.make("Isaac-Strafer-Navigation-v0", cfg=env_cfg)
    obs, info = env.reset()
    
    # Get action term for diagnostics
    action_term = env.unwrapped.action_manager._terms["wheel_velocities"]
    asset = action_term._asset
    joint_ids = action_term._joint_ids
    joint_names = action_term._joint_names
    
    print("\n" + "="*70)
    print("WHEEL DIAGNOSTIC")
    print("="*70)
    print(f"Joint IDs: {joint_ids}")
    print(f"Joint names: {joint_names}")
    print(f"Kinematic matrix:\n{action_term._kinematic_matrix}")
    print(f"Reorder mapping: {action_term._joint_reorder.tolist()}")
    print()
    
    # Test with pure forward action
    test_actions = [
        ("Pure Forward (vx=0.5)", [0.5, 0.0, 0.0]),
        ("Pure Strafe Left (vy=0.5)", [0.0, 0.5, 0.0]),
        ("Pure CCW Rotation (omega=0.5)", [0.0, 0.0, 0.5]),
    ]
    
    for name, action_vals in test_actions:
        print(f"\n--- {name} ---")
        action = torch.tensor([action_vals], device=args.device, dtype=torch.float32)
        
        # Manually compute what we expect
        body_vel = action * action_term._velocity_scale
        wheel_vels_kinematic = torch.matmul(body_vel, action_term._kinematic_matrix.T)
        wheel_vels_reordered = wheel_vels_kinematic[:, action_term._joint_reorder]
        
        print(f"  Input action [vx, vy, omega]: {action_vals}")
        print(f"  Body velocities [m/s, m/s, rad/s]: {body_vel.cpu().numpy()[0]}")
        print(f"  Wheel vels (kinematic order) [w1, w2, w3, w4]: {wheel_vels_kinematic.cpu().numpy()[0]}")
        print(f"  Wheel vels (USD order) [w1, w4, w2, w3]: {wheel_vels_reordered.cpu().numpy()[0]}")
        
        # Apply action and step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Read actual joint velocities
        actual_vel = asset.data.joint_vel[:, joint_ids]
        print(f"  Actual joint velocities: {actual_vel.cpu().numpy()[0]}")
    
    # Run a few more steps with forward action
    print("\n" + "="*70)
    print("Running 30 steps with forward action...")
    print("="*70)
    
    for step in range(30):
        action = torch.tensor([[0.5, 0.0, 0.0]], device=args.device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            actual_vel = asset.data.joint_vel[:, joint_ids]
            base_lin_vel = asset.data.root_lin_vel_b[:, :3]
            base_ang_vel = asset.data.root_ang_vel_b[:, :3]
            print(f"  Step {step}: wheel_vel={actual_vel.cpu().numpy()[0]}")
            print(f"            base_lin_vel={base_lin_vel.cpu().numpy()[0]}")
            print(f"            base_ang_vel={base_ang_vel.cpu().numpy()[0]}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
