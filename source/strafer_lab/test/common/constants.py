# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared constants for integration tests.

These constants are used across all integration test suites to ensure
consistent statistical testing parameters and environment configuration.
"""

# =============================================================================
# Statistical Testing
# =============================================================================

# Confidence level for hypothesis tests
# 0.95 corresponds to α = 0.05 (Type I error rate)
CONFIDENCE_LEVEL = 0.95

# =============================================================================
# Environment Configuration
# =============================================================================

# Number of parallel environments for statistical power
# 64 envs provides sufficient samples for detecting effects with high confidence
NUM_ENVS = 64

# Steps to let physics settle after reset before collecting data
N_SETTLE_STEPS = 10

# Steps to collect observations (more steps = tighter confidence intervals)
N_SAMPLES_STEPS = 200

# Default CUDA device
DEVICE = "cuda:0"

# =============================================================================
# Sensor Constants (from REAL_ROBOT_CONTRACT)
# =============================================================================

# Depth camera parameters
DEPTH_MAX_RANGE = 6.0  # meters
DEPTH_MIN_RANGE = 0.2  # meters

# Intel RealSense D555 stereo parameters
D555_BASELINE_M = 0.095        # Stereo baseline: 95mm
D555_FOCAL_LENGTH_PX = 673.0   # Focal length at native 1280x720 resolution
D555_DISPARITY_NOISE_PX = 0.08 # Typical subpixel disparity noise

# IMU parameters
IMU_ACCEL_MAX = 156.96  # ±16g in m/s²
IMU_GYRO_MAX = 34.9     # ±2000 °/s in rad/s

# Encoder parameters
ENCODER_VEL_MAX = 5000.0  # Max ticks/sec
