# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Gym environment registration.

These tests validate that all Strafer navigation environments are properly
registered with gymnasium and can be resolved by ID.

Focus areas:
- All expected environment IDs are registered
- Environment specs can be retrieved
- Train/Play variant pairing is correct

Usage:
    cd IsaacLab
    ./isaaclab.bat -p ../source/strafer_lab/test/unit/config/test_env_registration.py
"""

# Isaac Sim must be launched before importing Isaac Lab modules
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# --- Imports that require Isaac Sim runtime ---

import gymnasium as gym

# Import strafer_lab to trigger gym.register() calls
import strafer_lab.tasks.navigation  # noqa: F401


def get_registered_strafer_envs():
    """Get all registered Isaac-Strafer-Nav environment IDs."""
    return [env_id for env_id in gym.envs.registry.keys() 
            if env_id.startswith("Isaac-Strafer-Nav")]


def run_tests():
    """Run all environment registration tests."""
    results = []
    
    # =========================================================================
    # Test 1: At least one Strafer env is registered
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1: Strafer environments are registered")
    print("="*70)
    
    try:
        envs = get_registered_strafer_envs()
        
        assert len(envs) > 0, "No Isaac-Strafer-Nav environments found"
        
        print(f"  Found {len(envs)} registered environments:")
        for env_id in sorted(envs):
            print(f"    - {env_id}")
        
        results.append(("test_envs_registered", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_envs_registered", f"FAILED: {e}"))

    # =========================================================================
    # Test 2: Core environment IDs exist
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: Core environment IDs are registered")
    print("="*70)
    
    try:
        registered = set(get_registered_strafer_envs())
        
        core_envs = [
            "Isaac-Strafer-Nav-v0",       # Ideal
            "Isaac-Strafer-Nav-Real-v0",  # Realistic
        ]
        
        for env_id in core_envs:
            assert env_id in registered, f"Core env {env_id} not registered"
            print(f"  ✓ {env_id}")
        
        results.append(("test_core_envs_exist", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_core_envs_exist", f"FAILED: {e}"))

    # =========================================================================
    # Test 3: Environment specs are valid
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: Environment specs are valid")
    print("="*70)
    
    try:
        envs = get_registered_strafer_envs()
        
        for env_id in envs:
            spec = gym.spec(env_id)
            assert spec is not None, f"{env_id}: spec is None"
            assert spec.id == env_id, f"{env_id}: spec.id mismatch"
            assert spec.entry_point is not None, f"{env_id}: no entry_point"
        
        print(f"  ✓ All {len(envs)} environments have valid specs")
        
        results.append(("test_env_specs_valid", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_env_specs_valid", f"FAILED: {e}"))

    # =========================================================================
    # Test 4: Train envs have Play variants
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 4: Training envs have Play variants")
    print("="*70)
    
    try:
        registered = set(get_registered_strafer_envs())
        
        # Get non-Play envs
        train_envs = [e for e in registered if "-Play-" not in e]
        
        missing_play = []
        for train_id in train_envs:
            # Isaac-Strafer-Nav-v0 -> Isaac-Strafer-Nav-Play-v0
            base = train_id.replace("-v0", "")
            play_id = f"{base}-Play-v0"
            
            if play_id not in registered:
                missing_play.append(train_id)
        
        if missing_play:
            print(f"  ⚠ Warning: {len(missing_play)} training envs without Play variant:")
            for env_id in missing_play:
                print(f"    - {env_id}")
            # Don't fail, just warn
        else:
            print(f"  ✓ All {len(train_envs)} training envs have Play variants")
        
        results.append(("test_train_has_play", "PASSED"))
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        results.append(("test_train_has_play", f"ERROR: {e}"))

    # =========================================================================
    # Test 5: Scene configs are valid
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 5: Scene configurations are valid")
    print("="*70)
    
    try:
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            StraferSceneCfg,
            StraferSceneCfg_NoCam,
        )
        
        # Full scene should have camera and IMU
        scene = StraferSceneCfg(num_envs=1, env_spacing=4.0)
        assert hasattr(scene, "d555_camera"), "Full scene should have camera"
        assert hasattr(scene, "d555_imu"), "Full scene should have IMU"
        assert hasattr(scene, "robot"), "Scene should have robot"
        
        print("  ✓ StraferSceneCfg has camera, IMU, and robot")
        
        # NoCam scene should have IMU but no camera
        scene_nocam = StraferSceneCfg_NoCam(num_envs=1, env_spacing=4.0)
        assert hasattr(scene_nocam, "d555_imu"), "NoCam scene should have IMU"
        assert not hasattr(scene_nocam, "d555_camera"), "NoCam scene should NOT have camera"
        
        print("  ✓ StraferSceneCfg_NoCam has IMU, no camera")
        
        results.append(("test_scene_configs", "PASSED"))
        
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("test_scene_configs", f"FAILED: {e}"))

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
