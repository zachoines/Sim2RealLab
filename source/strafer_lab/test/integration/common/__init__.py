# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Common utilities for integration tests.

This module provides shared utilities across all integration test suites:
- Statistical testing functions (chi-squared, t-tests, circular stats)
- Common constants (confidence levels, sample sizes)
- Robot control utilities (positioning, freezing)

Usage:
    from test.integration.common import CONFIDENCE_LEVEL, chi_squared_variance_test
    from test.integration.common.circular import circular_mean, circular_ci

    # Robot utilities (import AFTER Isaac Sim is launched):
    from test.integration.common.robot import (
        get_env_origins, reset_robot_pose, freeze_robot_in_place, clear_frozen_state
    )
"""

from .constants import (
    CONFIDENCE_LEVEL,
    NUM_ENVS,
    N_SETTLE_STEPS,
    N_SAMPLES_STEPS,
    DEVICE,
    # Sensor normalization constants
    IMU_ACCEL_MAX,
    IMU_GYRO_MAX,
    ENCODER_VEL_MAX,
    # Depth camera constants
    DEPTH_MAX_RANGE,
    DEPTH_MIN_RANGE,
    D555_BASELINE_M,
    D555_FOCAL_LENGTH_PX,
    D555_DISPARITY_NOISE_PX,
)

from .stats import (
    chi_squared_variance_test,
    chi_squared_variance_ci,
    variance_ratio_test,
    one_sample_t_test,
    welch_t_test,
    binomial_test,
    VarianceTestResult,
    TTestResult,
    BinomialTestResult,
)

from .circular import (
    circular_mean,
    circular_variance,
    circular_std,
    circular_confidence_interval,
    angle_in_circular_ci,
)

__all__ = [
    # Constants - general
    "CONFIDENCE_LEVEL",
    "NUM_ENVS",
    "N_SETTLE_STEPS",
    "N_SAMPLES_STEPS",
    "DEVICE",
    # Constants - sensor normalization
    "IMU_ACCEL_MAX",
    "IMU_GYRO_MAX",
    "ENCODER_VEL_MAX",
    # Constants - depth camera
    "DEPTH_MAX_RANGE",
    "DEPTH_MIN_RANGE",
    "D555_BASELINE_M",
    "D555_FOCAL_LENGTH_PX",
    "D555_DISPARITY_NOISE_PX",
    # Statistical tests
    "chi_squared_variance_test",
    "chi_squared_variance_ci",
    "variance_ratio_test",
    "one_sample_t_test",
    "welch_t_test",
    "binomial_test",
    # Result types
    "VarianceTestResult",
    "TTestResult",
    "BinomialTestResult",
    # Circular statistics
    "circular_mean",
    "circular_variance",
    "circular_std",
    "circular_confidence_interval",
    "angle_in_circular_ci",
]
