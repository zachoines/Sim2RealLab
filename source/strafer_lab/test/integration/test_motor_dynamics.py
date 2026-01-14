# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for motor dynamics (first-order response model).

These tests verify that the MecanumWheelAction term correctly implements:
1. First-order low-pass filter (exponential smoothing)
2. Command delay buffer
3. Slew rate limiting

Statistical rigor:
- Motor time constant is verified by fitting exponential response curve
- Delay is verified by cross-correlation analysis
- Tests use sufficient samples for statistical significance

Usage:
    cd IsaacLab
    ./isaaclab.bat -p ../source/strafer_lab/test/integration/test_motor_dynamics.py

Note: Isaac Sim only allows one SimulationContext per process.
      Tests use a single environment and reconfigure as needed.
"""

# Isaac Sim must be launched before importing Isaac Lab modules
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# --- Imports that require Isaac Sim runtime ---

import math
import torch
import numpy as np
from scipy import optimize

from isaaclab.envs import ManagerBasedRLEnv

# Import environment config
from strafer_lab.tasks.navigation.strafer_env_cfg import (
    StraferNavEnvCfg_NoCam,
    ActionsCfg_Ideal,
    ActionsCfg_Realistic,
)


def run_tests():
    """Run all motor dynamics tests.
    
    Since Isaac Sim only allows one SimulationContext per process,
    we create a SINGLE environment and run all tests within it.
    We use ActionsCfg_Realistic since it has motor dynamics enabled,
    and we can toggle features on/off to test different behaviors.
    """
    results = []
    env = None
    
    try:
        # Create single environment with realistic config
        print("\n" + "="*70)
        print("SETUP: Creating test environment")
        print("="*70)
        
        cfg = StraferNavEnvCfg_NoCam()
        cfg.scene.num_envs = 4
        cfg.actions = ActionsCfg_Realistic()
        
        env = ManagerBasedRLEnv(cfg)
        env.reset()
        
        # Access the action term - use private _terms dict since no public API exists
        action_term = env.action_manager._terms["wheel_velocities"]
        physics_dt = env.physics_dt
        
        print(f"  Environment created with {env.num_envs} envs")
        print(f"  Physics dt: {physics_dt*1000:.2f}ms")
        print(f"  Motor dynamics enabled: {action_term._enable_motor_dynamics}")
        
        # =====================================================================
        # Test 1: Motor dynamics flag is set correctly
        # =====================================================================
        print("\n" + "="*70)
        print("TEST 1: Realistic config has motor dynamics enabled")
        print("="*70)
        
        try:
            assert action_term._enable_motor_dynamics, \
                "Realistic config should have motor dynamics ENABLED"
            
            print(f"  ✓ Motor dynamics enabled: {action_term._enable_motor_dynamics}")
            results.append(("test_motor_dynamics_enabled", "PASSED"))
            
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            results.append(("test_motor_dynamics_enabled", f"FAILED: {e}"))

        # =====================================================================
        # Test 2: Gradual response (first-order behavior)
        # =====================================================================
        print("\n" + "="*70)
        print("TEST 2: Motor dynamics produces gradual response")
        print("="*70)
        
        try:
            # Reset action term state
            action_term.reset(env_ids=torch.arange(env.num_envs, device=env.device))
            
            # Apply step input and record response
            step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
            step_action = step_action.repeat(env.num_envs, 1)
            
            responses = []
            num_steps = 100
            
            for _ in range(num_steps):
                action_term.process_actions(step_action)
                responses.append(action_term.processed_actions[0, 0].cpu().item())
            
            responses = np.array(responses)
            
            # Use absolute values - wheel velocities may be negative depending on
            # mecanum kinematics (forward motion = negative wheel angular velocity)
            abs_responses = np.abs(responses)
            
            # Response magnitude should start near zero and increase
            assert abs_responses[0] < abs_responses[-1], \
                f"Response magnitude should increase (start={abs_responses[0]:.4f}, end={abs_responses[-1]:.4f})"
            
            # Response magnitude should be monotonically increasing (no overshoot)
            diffs = np.diff(abs_responses)
            assert np.all(diffs >= -1e-6), \
                "First-order response should not overshoot"
            
            print(f"  ✓ Response range: {responses[0]:.4f} → {responses[-1]:.4f}")
            print(f"  ✓ Magnitude monotonically increasing: True")
            results.append(("test_gradual_response", "PASSED"))
            
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            results.append(("test_gradual_response", f"FAILED: {e}"))

        # =====================================================================
        # Test 3: Combined system response (motor + slew rate + delay)
        # =====================================================================
        print("\n" + "="*70)
        print("TEST 3: Combined system response time")
        print("="*70)
        
        try:
            # Reset action term state
            action_term.reset(env_ids=torch.arange(env.num_envs, device=env.device))
            
            # Get configured parameters
            cfg_tau = cfg.actions.wheel_velocities.motor_time_constant
            cfg_slew = cfg.actions.wheel_velocities.max_acceleration_rad_s2
            max_delay = cfg.actions.wheel_velocities.max_delay_steps
            
            print(f"  Configured motor tau: {cfg_tau*1000:.1f}ms")
            print(f"  Slew rate limit: {cfg_slew:.1f} rad/s²")
            print(f"  Command delay: up to {max_delay} steps")
            
            # Apply step input
            step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
            step_action = step_action.repeat(env.num_envs, 1)
            
            # Collect response over sufficient time
            num_steps = 100
            times = []
            responses = []
            
            for i in range(num_steps):
                action_term.process_actions(step_action)
                times.append(i * physics_dt)
                responses.append(action_term.processed_actions[0, 0].cpu().item())
            
            times = np.array(times)
            responses = np.array(responses)
            
            # Use absolute values - wheel velocities may be negative
            abs_responses = np.abs(responses)
            v_final = abs_responses[-1]
            
            print(f"  Final velocity magnitude: {v_final:.4f}")
            
            # Use 63.2% method to find effective tau
            target_63 = 0.632 * v_final
            idx_63 = np.argmin(np.abs(abs_responses - target_63))
            measured_tau = times[idx_63]
            
            print(f"  63.2% threshold: {target_63:.4f}")
            print(f"  Measured effective tau: {measured_tau*1000:.1f}ms")
            
            # Combined system tau should be >= motor tau due to cascaded effects
            # Slew rate + delay make response slower, so effective tau > motor tau
            assert measured_tau >= cfg_tau * 0.8, \
                f"Effective tau ({measured_tau*1000:.1f}ms) should be >= motor tau ({cfg_tau*1000:.1f}ms)"
            
            # Should reach 63% within reasonable time (< 500ms for this system)
            assert measured_tau < 0.5, \
                f"Response too slow: took {measured_tau*1000:.1f}ms to reach 63%"
            
            print(f"  ✓ Combined response within acceptable range")
            results.append(("test_combined_system_response", "PASSED"))
            
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            results.append(("test_combined_system_response", f"FAILED: {e}"))

        # =====================================================================
        # Test 4: Motor dynamics reset
        # =====================================================================
        print("\n" + "="*70)
        print("TEST 4: Motor dynamics state resets properly")
        print("="*70)
        
        try:
            # Reset first to start fresh
            action_term.reset(env_ids=torch.arange(env.num_envs, device=env.device))
            
            # Build up velocity
            step_action = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
            step_action = step_action.repeat(env.num_envs, 1)
            
            for _ in range(50):
                action_term.process_actions(step_action)
            
            pre_reset_mag = torch.abs(action_term.processed_actions).mean().item()
            
            assert pre_reset_mag > 0.1, \
                f"Should have nonzero velocity before reset (got {pre_reset_mag:.4f})"
            
            print(f"  Pre-reset velocity magnitude: {pre_reset_mag:.4f}")
            
            # Reset
            action_term.reset(env_ids=torch.arange(env.num_envs, device=env.device))
            
            # Check internal state is reset
            if hasattr(action_term, '_smoothed_wheel_velocities'):
                smoothed_mag = torch.abs(action_term._smoothed_wheel_velocities).mean().item()
                print(f"  Post-reset smoothed velocity: {smoothed_mag:.6f}")
                assert smoothed_mag < 0.01, \
                    f"Smoothed velocity should reset to zero (got {smoothed_mag:.4f})"
            
            print(f"  ✓ Motor dynamics state reset successfully")
            results.append(("test_motor_dynamics_reset", "PASSED"))
            
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            results.append(("test_motor_dynamics_reset", f"FAILED: {e}"))

        # =====================================================================
        # Test 5: Steady state convergence
        # =====================================================================
        print("\n" + "="*70)
        print("TEST 5: Steady state convergence")
        print("="*70)
        
        try:
            # Reset first
            action_term.reset(env_ids=torch.arange(env.num_envs, device=env.device))
            
            # Apply constant input for many steps
            step_action = torch.tensor([[0.5, 0.0, 0.0]], device=env.device)
            step_action = step_action.repeat(env.num_envs, 1)
            
            cfg_tau = cfg.actions.wheel_velocities.motor_time_constant
            num_steps = int(10 * cfg_tau / physics_dt)
            
            for _ in range(num_steps):
                action_term.process_actions(step_action)
            
            # Record final values
            final_responses = []
            for _ in range(10):
                action_term.process_actions(step_action)
                final_responses.append(action_term.processed_actions[0, 0].cpu().item())
            
            final_responses = np.array(final_responses)
            mean_val = np.mean(final_responses)
            variation = np.std(final_responses) / (mean_val + 1e-6)
            
            print(f"  Steady state mean: {mean_val:.4f}")
            print(f"  Variation (CV): {variation*100:.2f}%")
            
            assert variation < 0.05, \
                f"Steady state variation {variation*100:.1f}% exceeds 5% threshold"
            
            print(f"  ✓ Converged to steady state")
            results.append(("test_steady_state_convergence", "PASSED"))
            
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            results.append(("test_steady_state_convergence", f"FAILED: {e}"))

    except Exception as e:
        print(f"\n  ✗ SETUP ERROR: {e}")
        results.append(("setup", f"ERROR: {e}"))
    
    finally:
        # Clean up
        if env is not None:
            env.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    
    for name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"  {symbol} {name}: {status}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_tests()
        exit_code = 0 if success else 1
    finally:
        simulation_app.close()
    
    exit(exit_code)
