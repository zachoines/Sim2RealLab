# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for observation structure validation.

These tests validate that:
- Observation structure matches expected dimensions

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/sensors/test_observations.py -v
"""

import torch


# =============================================================================
# Constants
# =============================================================================

# Expected observation structure for NoCam config
# imu_accel(3) + imu_gyro(3) + encoders(4) + goal(2) + action(3) = 15
EXPECTED_OBS_DIMS = [(3,), (3,), (4,), (2,), (3,)]
EXPECTED_TOTAL_DIM = 15


# =============================================================================
# Tests: Observation Pipeline Structure
# =============================================================================


def test_observation_structure(noisy_env):
    """Verify observation term structure matches expected dimensions.

    Observation structure for NoCam config (15 dims total):
    - imu_linear_acceleration: (3,) - normalized by max_accel
    - imu_angular_velocity: (3,) - normalized by max_angular_vel
    - wheel_encoder_velocities: (4,) - normalized by max_ticks_per_sec
    - goal_position: (2,) - relative [x, y] to goal (meters)
    - last_action: (3,) - previous [vx, vy, omega] command
    """
    obs_manager = noisy_env.observation_manager

    if hasattr(obs_manager, "_group_obs_term_dim"):
        group_cfg = obs_manager._group_obs_term_dim

        assert "policy" in group_cfg, "Missing 'policy' observation group"

        term_dims = group_cfg["policy"]

        print(f"\n  Observation structure validation:")
        print(f"    Number of terms: {len(term_dims)} (expected {len(EXPECTED_OBS_DIMS)})")

        assert len(term_dims) == len(EXPECTED_OBS_DIMS), (
            f"Expected {len(EXPECTED_OBS_DIMS)} terms, got {len(term_dims)}"
        )

        term_names = ["imu_accel", "imu_gyro", "encoders", "goal", "action"]
        for i, (actual, expected) in enumerate(zip(term_dims, EXPECTED_OBS_DIMS)):
            print(f"    Term {i} ({term_names[i]}): {actual} (expected {expected})")
            assert actual == expected, f"Term {i}: expected {expected}, got {actual}"

        total = sum(d[0] for d in term_dims)
        print(f"    Total dimensions: {total} (expected {EXPECTED_TOTAL_DIM})")
        assert total == EXPECTED_TOTAL_DIM, f"Expected {EXPECTED_TOTAL_DIM} total dims, got {total}"
    else:
        # Fallback: verify total dimension from actual observation
        obs_dict, _ = noisy_env.reset()
        total_dim = obs_dict["policy"].shape[1]
        print(f"\n  Observation total dimension: {total_dim} (expected {EXPECTED_TOTAL_DIM})")
        assert total_dim == EXPECTED_TOTAL_DIM, f"Expected {EXPECTED_TOTAL_DIM} dims, got {total_dim}"
