# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for sim-to-real contract configuration schemas.

These tests validate the *relationships* between contract presets, not specific
hardcoded values. They ensure the contract hierarchy (ideal < real < robust)
is maintained as configs evolve.

Focus areas:
- Contract ordering: ideal has least noise/delay, robust has most
- Helper function correctness: return correct types, None for disabled
- Config hierarchy integrity: contracts follow the realism ordering

Usage:
    cd IsaacLab
    ./isaaclab.bat -p ../source/strafer_lab/test/unit/config/test_contract_schemas.py
"""

# Isaac Sim must be launched before importing Isaac Lab modules
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# --- Imports that require Isaac Sim runtime ---

from isaaclab.utils.noise import GaussianNoiseCfg

from strafer_lab.tasks.navigation.sim_real_cfg import (
    # Presets
    IDEAL_SIM_CONTRACT,
    REAL_ROBOT_CONTRACT,
    ROBUST_TRAINING_CONTRACT,
    # Helper functions
    get_imu_accel_noise,
    get_imu_gyro_noise,
    get_encoder_noise,
    get_depth_noise,
    get_rgb_noise,
    get_action_config_params,
)


def run_tests():
    """Run all contract schema tests."""
    results = []
    
    # =========================================================================
    # Test 1: Ideal has all noise disabled
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: Ideal contract has all noise disabled")
    print("="*70)
    
    try:
        contract = IDEAL_SIM_CONTRACT
        
        assert contract.sensors.imu.enable_noise is False, "IMU noise should be disabled"
        assert contract.sensors.encoders.enable_noise is False, "Encoder noise should be disabled"
        assert contract.sensors.depth_camera.enable_noise is False, "Depth noise should be disabled"
        assert contract.sensors.rgb_camera.enable_noise is False, "RGB noise should be disabled"
        
        print("  ✓ IMU noise disabled")
        print("  ✓ Encoder noise disabled")
        print("  ✓ Depth camera noise disabled")
        print("  ✓ RGB camera noise disabled")
        
        results.append(("test_ideal_noise_disabled", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_ideal_noise_disabled", f"FAILED: {e}"))

    # =========================================================================
    # Test 2: Real and robust have noise enabled
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: Real and robust contracts have noise enabled")
    print("="*70)
    
    try:
        for name, contract in [("REAL", REAL_ROBOT_CONTRACT), 
                                ("ROBUST", ROBUST_TRAINING_CONTRACT)]:
            assert contract.sensors.imu.enable_noise is True, \
                f"{name}: IMU noise should be enabled"
            assert contract.sensors.encoders.enable_noise is True, \
                f"{name}: Encoder noise should be enabled"
        
        print("  ✓ REAL: IMU and encoder noise enabled")
        print("  ✓ ROBUST: IMU and encoder noise enabled")
        
        results.append(("test_real_robust_noise_enabled", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_real_robust_noise_enabled", f"FAILED: {e}"))

    # =========================================================================
    # Test 3: Robust noise >= Real noise
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: Robust noise levels >= Real noise levels")
    print("="*70)
    
    try:
        real = REAL_ROBOT_CONTRACT
        robust = ROBUST_TRAINING_CONTRACT
        
        assert robust.sensors.imu.accel_noise_density >= real.sensors.imu.accel_noise_density, \
            "Robust IMU accel noise should be >= Real"
        assert robust.sensors.encoders.velocity_noise_std >= real.sensors.encoders.velocity_noise_std, \
            "Robust encoder noise should be >= Real"
        
        print(f"  ✓ IMU accel noise: Real={real.sensors.imu.accel_noise_density:.4f}, "
              f"Robust={robust.sensors.imu.accel_noise_density:.4f}")
        print(f"  ✓ Encoder noise: Real={real.sensors.encoders.velocity_noise_std:.4f}, "
              f"Robust={robust.sensors.encoders.velocity_noise_std:.4f}")
        
        results.append(("test_robust_noise_gte_real", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_robust_noise_gte_real", f"FAILED: {e}"))

    # =========================================================================
    # Test 4: Ideal has no motor dynamics
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 4: Ideal has no motor dynamics")
    print("="*70)
    
    try:
        assert IDEAL_SIM_CONTRACT.actuator.enable_motor_dynamics is False
        print("  ✓ Motor dynamics disabled")
        
        results.append(("test_ideal_no_motor_dynamics", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_ideal_no_motor_dynamics", f"FAILED: {e}"))

    # =========================================================================
    # Test 5: Real and robust have motor dynamics
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 5: Real and robust have motor dynamics")
    print("="*70)
    
    try:
        assert REAL_ROBOT_CONTRACT.actuator.enable_motor_dynamics is True
        assert ROBUST_TRAINING_CONTRACT.actuator.enable_motor_dynamics is True
        
        print("  ✓ REAL: Motor dynamics enabled")
        print("  ✓ ROBUST: Motor dynamics enabled")
        
        results.append(("test_real_robust_motor_dynamics", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_real_robust_motor_dynamics", f"FAILED: {e}"))

    # =========================================================================
    # Test 6: Ideal has no delays
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 6: Ideal has no delays")
    print("="*70)
    
    try:
        contract = IDEAL_SIM_CONTRACT
        
        assert contract.timing.obs_latency_steps == 0
        assert contract.timing.action_latency_steps == 0
        assert contract.actuator.min_delay_steps == 0
        assert contract.actuator.max_delay_steps == 0
        
        print("  ✓ Observation latency: 0 steps")
        print("  ✓ Action latency: 0 steps")
        print("  ✓ Actuator delay: 0-0 steps")
        
        results.append(("test_ideal_no_delays", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_ideal_no_delays", f"FAILED: {e}"))

    # =========================================================================
    # Test 7: Only robust has failures enabled
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 7: Only robust has sensor failures enabled")
    print("="*70)
    
    try:
        assert IDEAL_SIM_CONTRACT.sensors.failures.enable_failures is False
        assert REAL_ROBOT_CONTRACT.sensors.failures.enable_failures is False
        assert ROBUST_TRAINING_CONTRACT.sensors.failures.enable_failures is True
        
        print("  ✓ IDEAL: Failures disabled")
        print("  ✓ REAL: Failures disabled")
        print("  ✓ ROBUST: Failures enabled")
        
        results.append(("test_only_robust_has_failures", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_only_robust_has_failures", f"FAILED: {e}"))

    # =========================================================================
    # Test 8: Noise helper functions - ideal returns None
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 8: Noise helpers return None for ideal (disabled)")
    print("="*70)
    
    try:
        assert get_imu_accel_noise(IDEAL_SIM_CONTRACT) is None
        assert get_imu_gyro_noise(IDEAL_SIM_CONTRACT) is None
        assert get_encoder_noise(IDEAL_SIM_CONTRACT) is None
        assert get_depth_noise(IDEAL_SIM_CONTRACT) is None
        assert get_rgb_noise(IDEAL_SIM_CONTRACT) is None
        
        print("  ✓ All noise helpers return None for ideal")
        
        results.append(("test_noise_helpers_ideal_none", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_noise_helpers_ideal_none", f"FAILED: {e}"))

    # =========================================================================
    # Test 9: Noise helper functions - real returns GaussianNoiseCfg
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 9: Noise helpers return GaussianNoiseCfg for real")
    print("="*70)
    
    try:
        for func_name, func in [
            ("imu_accel", get_imu_accel_noise),
            ("imu_gyro", get_imu_gyro_noise),
            ("encoder", get_encoder_noise),
            ("depth", get_depth_noise),
            ("rgb", get_rgb_noise),
        ]:
            result = func(REAL_ROBOT_CONTRACT)
            assert isinstance(result, GaussianNoiseCfg), \
                f"{func_name} should return GaussianNoiseCfg"
            assert result.mean == 0.0, f"{func_name} should have zero mean"
            assert result.std > 0.0, f"{func_name} should have positive std"
        
        print("  ✓ All noise helpers return valid GaussianNoiseCfg")
        
        results.append(("test_noise_helpers_real_gaussian", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_noise_helpers_real_gaussian", f"FAILED: {e}"))

    # =========================================================================
    # Test 10: Action config helper - ideal
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 10: get_action_config_params for ideal")
    print("="*70)
    
    try:
        params = get_action_config_params(IDEAL_SIM_CONTRACT)
        
        assert params["enable_motor_dynamics"] is False
        assert params["min_delay_steps"] == 0
        assert params["max_delay_steps"] == 0
        
        print(f"  ✓ Motor dynamics: {params['enable_motor_dynamics']}")
        print(f"  ✓ Delay steps: {params['min_delay_steps']}-{params['max_delay_steps']}")
        
        results.append(("test_action_config_ideal", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_action_config_ideal", f"FAILED: {e}"))

    # =========================================================================
    # Test 11: Action config helper - real
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 11: get_action_config_params for real")
    print("="*70)
    
    try:
        params = get_action_config_params(REAL_ROBOT_CONTRACT)
        
        assert params["enable_motor_dynamics"] is True
        assert params["motor_time_constant"] > 0
        
        print(f"  ✓ Motor dynamics: {params['enable_motor_dynamics']}")
        print(f"  ✓ Time constant: {params['motor_time_constant']*1000:.1f}ms")
        
        results.append(("test_action_config_real", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_action_config_real", f"FAILED: {e}"))

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
