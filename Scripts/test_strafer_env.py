#!/usr/bin/env python3
"""Test script to verify Strafer environment loads correctly in Isaac Lab.

This script runs predefined motion patterns to validate mecanum kinematics:
- Forward/backward
- Strafe left/right
- Rotation
- Circle (forward + rotation)
- Figure-8 pattern

Environments (--env):
    Isaac-Strafer-Nav-v0          Ideal Full (no noise)
    Isaac-Strafer-Nav-Depth-v0    Ideal Depth-only
    Isaac-Strafer-Nav-NoCam-v0    Ideal Proprioceptive-only
    Isaac-Strafer-Nav-Real-v0     Realistic Full (default)
    Isaac-Strafer-Nav-Real-Depth-v0  Realistic Depth-only
    Isaac-Strafer-Nav-Robust-v0   Robust Full (stress-test)

Usage:
    .\IsaacLab\isaaclab.bat -p Scripts\test_strafer_env.py
    .\IsaacLab\isaaclab.bat -p Scripts\test_strafer_env.py --env Isaac-Strafer-Nav-v0
    .\IsaacLab\isaaclab.bat -p Scripts\test_strafer_env.py --pattern circle --env Isaac-Strafer-Nav-Robust-v0
"""

import argparse
import math


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Strafer environment")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    parser.add_argument("--enable_cameras", action="store_true", default=True, 
                        help="Enable camera rendering (required for camera-based envs)")
    parser.add_argument("--pattern", type=str, default="all", 
                        choices=[
                            "forward", "strafe", "strafe_left", "strafe_right",
                            "rotate", "circle", "figure8", "all"
                        ],
                        help="Motion pattern to test")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Scale factor for pattern magnitudes (1.0 = default speeds)")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration per pattern in seconds")
    parser.add_argument("--env", type=str, default="Isaac-Strafer-Nav-Real-v0",
                        help="Environment ID (default: Isaac-Strafer-Nav-Real-v0 = Realistic Full)")
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
    print("Strafer Lab Environment Test - Motion Patterns")
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
    env_name = args.env
    print(f"\nCreating environment: {env_name}")
    print(f"Available: Isaac-Strafer-Nav-{{v0,Depth-v0,NoCam-v0,Real-v0,Real-Depth-v0,Robust-v0}}")
    
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
        
        # Calculate steps per pattern based on environment dt
        env_dt = env_cfg.sim.dt * env_cfg.decimation  # ~0.033s at 30Hz policy
        steps_per_pattern = int(args.duration / env_dt)
        
        print(f"\n--- Motion Test Configuration ---")
        print(f"Environment dt: {env_dt:.4f}s ({1/env_dt:.1f} Hz)")
        print(f"Pattern duration: {args.duration}s ({steps_per_pattern} steps)")
        print(f"Pattern(s) to run: {args.pattern}")
        
        # Define motion patterns
        # Each pattern is a function that returns [vx, vy, omega] given time t
        speed_scale = args.speed
        base_linear = 0.5
        base_omega = 0.5
        circle_omega = 0.3
        figure8_speed = 0.4

        def pattern_forward(t: float) -> tuple:
            """Move forward at 50% speed."""
            return (base_linear * speed_scale, 0.0, 0.0)
        
        def pattern_backward(t: float) -> tuple:
            """Move backward at 50% speed."""
            return (-base_linear * speed_scale, 0.0, 0.0)
        
        def pattern_strafe_right(t: float) -> tuple:
            """Strafe right at 50% speed."""
            return (0.0, -base_linear * speed_scale, 0.0)
        
        def pattern_strafe_left(t: float) -> tuple:
            """Strafe left at 50% speed."""
            return (0.0, base_linear * speed_scale, 0.0)
        
        def pattern_rotate_cw(t: float) -> tuple:
            """Rotate clockwise at 50% speed."""
            return (0.0, 0.0, -base_omega * speed_scale)
        
        def pattern_rotate_ccw(t: float) -> tuple:
            """Rotate counter-clockwise at 50% speed."""
            return (0.0, 0.0, base_omega * speed_scale)
        
        def pattern_circle(t: float) -> tuple:
            """Drive in a circle (forward + rotation)."""
            return (base_linear * speed_scale, 0.0, circle_omega * speed_scale)  # Forward + CCW rotation
        
        def pattern_figure8(t: float) -> tuple:
            """Drive in a figure-8 pattern."""
            # Alternate rotation direction based on time
            period = 4.0  # seconds per loop
            phase = (t % (2 * period)) / period
            omega = figure8_speed * speed_scale if phase < 1.0 else -figure8_speed * speed_scale
            return (figure8_speed * speed_scale, 0.0, omega)
        
        def pattern_strafe(t: float) -> tuple:
            """Alternate strafe left/right."""
            period = 2.0
            phase = (t % (2 * period)) / period
            vy = base_linear * speed_scale if phase < 1.0 else -base_linear * speed_scale
            return (0.0, vy, 0.0)
        
        def pattern_rotate(t: float) -> tuple:
            """Alternate rotation CW/CCW."""
            period = 2.0
            phase = (t % (2 * period)) / period
            omega = base_omega * speed_scale if phase < 1.0 else -base_omega * speed_scale
            return (0.0, 0.0, omega)
        
        # Build list of patterns to run
        pattern_map = {
            "forward": [("Forward", pattern_forward), ("Backward", pattern_backward)],
            "strafe": [("Strafe Left/Right", pattern_strafe)],
            "strafe_left": [("Strafe Left", pattern_strafe_left)],
            "strafe_right": [("Strafe Right", pattern_strafe_right)],
            "rotate": [("Rotate CW/CCW", pattern_rotate)],
            "circle": [("Circle", pattern_circle)],
            "figure8": [("Figure-8", pattern_figure8)],
            "all": [
                ("Forward", pattern_forward),
                ("Backward", pattern_backward),
                ("Strafe Left/Right", pattern_strafe),
                ("Strafe Left", pattern_strafe_left),
                ("Strafe Right", pattern_strafe_right),
                ("Rotate CW/CCW", pattern_rotate),
                ("Circle", pattern_circle),
                ("Figure-8", pattern_figure8),
            ],
        }
        
        patterns_to_run = pattern_map[args.pattern]
        
        # Reset environment
        print(f"\n--- Running Motion Patterns ---")
        obs, info = env.reset()
        
        total_steps = 0
        
        for pattern_name, pattern_fn in patterns_to_run:
            print(f"\n>>> Running pattern: {pattern_name} for {args.duration}s")
            
            # Reset for each pattern
            obs, info = env.reset()
            
            for step in range(steps_per_pattern):
                t = step * env_dt  # Current time in seconds
                
                # Get action from pattern
                vx, vy, omega = pattern_fn(t)
                action = torch.tensor([[vx, vy, omega]], device=args.device)
                
                # Expand for multiple envs if needed
                if args.num_envs > 1:
                    action = action.expand(args.num_envs, -1)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Print progress every second
                if step % int(1.0 / env_dt) == 0:
                    obs_policy = obs['policy'] if isinstance(obs, dict) else obs
                    # Extract velocities from observation (first 6 elements are lin_vel and ang_vel)
                    lin_vel = obs_policy[0, :3].cpu().numpy()
                    ang_vel = obs_policy[0, 3:6].cpu().numpy()
                    print(f"  t={t:.1f}s | action=[{vx:.2f}, {vy:.2f}, {omega:.2f}] | "
                          f"lin_vel=[{lin_vel[0]:.2f}, {lin_vel[1]:.2f}, {lin_vel[2]:.2f}] | "
                          f"ang_vel=[{ang_vel[0]:.2f}, {ang_vel[1]:.2f}, {ang_vel[2]:.2f}]")
                
                total_steps += 1
                
                # Check for termination
                if terminated.any() or truncated.any():
                    print(f"  Episode ended at step {step}")
                    obs, info = env.reset()
        
        print(f"\n[OK] Motion pattern test completed! ({total_steps} total steps)")
        env.close()
        
    except Exception as e:
        print(f"\n[ERROR] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()

    simulation_app.close()


if __name__ == "__main__":
    main()
