# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for sensor failure modes across all noise models.

Each noise model supports a ``failure_probability`` field that causes the
sensor to return a specific "failure value" on any given step.  The IMU model
also supports ``stuck_probability`` (returns previous output).

These tests validate:
1. Failure output matches the documented value (zeros, max_range, or black).
2. Stuck mode returns the previous sample unchanged.
3. Observed failure/stuck rates match the configured probability (binomial test).

All tests run the noise models in isolation (no Isaac Sim environment needed
beyond the AppLauncher started by root conftest.py).

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_sensor_failures.py -v
"""

import torch
import numpy as np

from test.common import (
    CONFIDENCE_LEVEL,
    binomial_test,
    DEVICE,
)

from strafer_lab.tasks.navigation.mdp.noise_models import (
    IMUNoiseModel, IMUNoiseModelCfg,
    EncoderNoiseModel, EncoderNoiseModelCfg,
    DepthNoiseModel, DepthNoiseModelCfg,
    RGBNoiseModel, RGBNoiseModelCfg,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ENVS = 16
N_TRIALS = 2_000  # enough for binomial precision without being slow

# High failure prob for deterministic-like behaviour in output-value tests
HIGH_FAIL_PROB = 1.0

# Moderate failure prob for rate-matching tests
RATE_FAIL_PROB = 0.15
RATE_STUCK_PROB = 0.10


# =====================================================================
# IMU Sensor Failures
# =====================================================================

def _make_imu_with_failure(
    failure_prob: float = 0.0,
    stuck_prob: float = 0.0,
) -> IMUNoiseModel:
    cfg = IMUNoiseModelCfg(
        sensor_type="accel",
        accel_noise_std=0.01,
        accel_bias_range=(0.0, 0.0),
        accel_bias_drift_rate=0.0,
        output_size=3,
        latency_steps=0,
        failure_probability=failure_prob,
        stuck_probability=stuck_prob,
    )
    return IMUNoiseModel(cfg, N_ENVS, DEVICE)


def test_imu_failure_returns_zeros():
    """With failure_probability=1.0, every output row must be all zeros."""
    model = _make_imu_with_failure(failure_prob=HIGH_FAIL_PROB)

    data = torch.ones(N_ENVS, 3, device=DEVICE) * 5.0
    for _ in range(50):
        out = model(data.clone())
        assert torch.all(out == 0.0), (
            f"Expected all zeros on failure, got max={out.abs().max().item()}"
        )


def test_imu_stuck_returns_previous():
    """With stuck_probability=1.0, output should be identical to the previous call.

    On the very first call ``_prev_output`` is None so the model initialises it.
    From the second call onward every output must exactly equal the previous one.
    """
    model = _make_imu_with_failure(stuck_prob=HIGH_FAIL_PROB)

    data = torch.ones(N_ENVS, 3, device=DEVICE)

    # First call — initialises _prev_output
    first_out = model(data.clone())

    # Subsequent calls with *different* input should still return first_out
    for i in range(20):
        different_data = torch.randn(N_ENVS, 3, device=DEVICE) * 100
        out = model(different_data)
        assert torch.allclose(out, first_out, atol=1e-6), (
            f"Step {i}: stuck output changed. "
            f"Max diff = {(out - first_out).abs().max().item()}"
        )


def test_imu_failure_rate_matches_config():
    """Observed failure rate should match failure_probability (binomial test).

    A "failure" is detected when an entire env row is all zeros despite
    non-zero input.
    """
    model = _make_imu_with_failure(failure_prob=RATE_FAIL_PROB)

    data = torch.ones(N_ENVS, 3, device=DEVICE) * 10.0
    failures = 0
    total = 0

    for _ in range(N_TRIALS):
        out = model(data.clone())
        # Count envs where entire row is zero (failure event)
        row_is_zero = (out == 0.0).all(dim=-1)
        failures += row_is_zero.sum().item()
        total += N_ENVS

    result = binomial_test(int(failures), total, RATE_FAIL_PROB)
    print(f"\n  IMU failure rate test:")
    print(f"    Observed: {result.observed_rate:.4f}")
    print(f"    Expected: {result.expected_rate}")
    print(f"    p-value:  {result.p_value:.4f}")

    assert not result.reject_null, (
        f"IMU failure rate {result.observed_rate:.4f} differs from "
        f"configured {RATE_FAIL_PROB} (p={result.p_value:.4f})"
    )


def test_imu_stuck_rate_matches_config():
    """Observed stuck rate should match stuck_probability (binomial test).

    A "stuck" event is detected when an env's output exactly equals the
    previous output despite changing input.
    """
    model = _make_imu_with_failure(stuck_prob=RATE_STUCK_PROB)

    stuck_count = 0
    total = 0

    prev_out = model(torch.randn(N_ENVS, 3, device=DEVICE) * 10)

    for _ in range(N_TRIALS):
        data = torch.randn(N_ENVS, 3, device=DEVICE) * 10
        out = model(data)
        # A stuck env has identical output to previous step
        is_stuck = torch.all(out == prev_out, dim=-1)
        stuck_count += is_stuck.sum().item()
        total += N_ENVS
        prev_out = out.clone()

    result = binomial_test(int(stuck_count), total, RATE_STUCK_PROB)
    print(f"\n  IMU stuck rate test:")
    print(f"    Observed: {result.observed_rate:.4f}")
    print(f"    Expected: {result.expected_rate}")
    print(f"    p-value:  {result.p_value:.4f}")

    assert not result.reject_null, (
        f"IMU stuck rate {result.observed_rate:.4f} differs from "
        f"configured {RATE_STUCK_PROB} (p={result.p_value:.4f})"
    )


# =====================================================================
# Encoder Sensor Failures
# =====================================================================

def _make_encoder_with_failure(failure_prob: float = 0.0) -> EncoderNoiseModel:
    cfg = EncoderNoiseModelCfg(
        enable_quantization=False,
        velocity_noise_std=0.0,
        max_velocity=3000.0,
        missed_tick_prob=0.0,
        extra_tick_prob=0.0,
        failure_probability=failure_prob,
        latency_steps=0,
        output_size=4,
    )
    return EncoderNoiseModel(cfg, N_ENVS, DEVICE)


def test_encoder_failure_returns_zeros():
    """With failure_probability=1.0, encoder output must be all zeros."""
    model = _make_encoder_with_failure(failure_prob=HIGH_FAIL_PROB)

    data = torch.ones(N_ENVS, 4, device=DEVICE) * 1000.0
    for _ in range(50):
        out = model(data.clone())
        assert torch.all(out == 0.0), (
            f"Expected all zeros on encoder failure, got max={out.abs().max().item()}"
        )


def test_encoder_failure_rate_matches_config():
    """Encoder failure rate should match configured probability."""
    model = _make_encoder_with_failure(failure_prob=RATE_FAIL_PROB)

    data = torch.ones(N_ENVS, 4, device=DEVICE) * 500.0
    failures = 0
    total = 0

    for _ in range(N_TRIALS):
        out = model(data.clone())
        row_is_zero = (out == 0.0).all(dim=-1)
        failures += row_is_zero.sum().item()
        total += N_ENVS

    result = binomial_test(int(failures), total, RATE_FAIL_PROB)
    print(f"\n  Encoder failure rate test:")
    print(f"    Observed: {result.observed_rate:.4f}")
    print(f"    Expected: {result.expected_rate}")
    print(f"    p-value:  {result.p_value:.4f}")

    assert not result.reject_null, (
        f"Encoder failure rate {result.observed_rate:.4f} differs from "
        f"configured {RATE_FAIL_PROB} (p={result.p_value:.4f})"
    )


# =====================================================================
# Depth Camera Sensor Failures
# =====================================================================

def _make_depth_with_failure(failure_prob: float = 0.0) -> DepthNoiseModel:
    cfg = DepthNoiseModelCfg(
        baseline_m=0.095,
        focal_length_px=673.0,
        disparity_noise_px=0.0,  # disable noise for clean failure detection
        hole_probability=0.0,
        frame_drop_prob=0.0,
        min_range=0.2,
        max_range=6.0,
        failure_probability=failure_prob,
        latency_steps=0,
    )
    return DepthNoiseModel(cfg, N_ENVS, DEVICE)


def test_depth_failure_returns_max_range():
    """With failure_probability=1.0, depth output must be max_range everywhere."""
    model = _make_depth_with_failure(failure_prob=HIGH_FAIL_PROB)
    max_range = model.cfg.max_range

    # Flattened depth image: (N_ENVS, H*W)
    n_pixels = 60 * 80
    data = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 2.0  # 2 metres
    for _ in range(20):
        out = model(data.clone())
        assert torch.allclose(out, torch.full_like(out, max_range)), (
            f"Expected max_range={max_range} on depth failure, "
            f"got min={out.min().item():.4f} max={out.max().item():.4f}"
        )


def test_depth_failure_rate_matches_config():
    """Depth camera failure rate should match configured probability."""
    model = _make_depth_with_failure(failure_prob=RATE_FAIL_PROB)
    max_range = model.cfg.max_range

    n_pixels = 60 * 80
    data = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 2.0
    failures = 0
    total = 0

    for _ in range(N_TRIALS):
        out = model(data.clone())
        # A failed env has ALL pixels == max_range
        all_max = (out == max_range).all(dim=-1)
        failures += all_max.sum().item()
        total += N_ENVS

    result = binomial_test(int(failures), total, RATE_FAIL_PROB)
    print(f"\n  Depth failure rate test:")
    print(f"    Observed: {result.observed_rate:.4f}")
    print(f"    Expected: {result.expected_rate}")
    print(f"    p-value:  {result.p_value:.4f}")

    assert not result.reject_null, (
        f"Depth failure rate {result.observed_rate:.4f} differs from "
        f"configured {RATE_FAIL_PROB} (p={result.p_value:.4f})"
    )


# =====================================================================
# RGB Camera Sensor Failures
# =====================================================================

def _make_rgb_with_failure(failure_prob: float = 0.0) -> RGBNoiseModel:
    cfg = RGBNoiseModelCfg(
        pixel_noise_std=0.0,
        brightness_range=(1.0, 1.0),  # no brightness variation
        frame_drop_prob=0.0,
        failure_probability=failure_prob,
        latency_steps=0,
    )
    return RGBNoiseModel(cfg, N_ENVS, DEVICE)


def test_rgb_failure_returns_black():
    """With failure_probability=1.0, RGB output must be all zeros (black)."""
    model = _make_rgb_with_failure(failure_prob=HIGH_FAIL_PROB)

    n_pixels = 60 * 80
    data = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 0.5  # mid-grey
    for _ in range(20):
        out = model(data.clone())
        assert torch.all(out == 0.0), (
            f"Expected black (zeros) on RGB failure, got max={out.max().item():.4f}"
        )


def test_rgb_failure_rate_matches_config():
    """RGB camera failure rate should match configured probability."""
    model = _make_rgb_with_failure(failure_prob=RATE_FAIL_PROB)

    n_pixels = 60 * 80
    data = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 0.5
    failures = 0
    total = 0

    for _ in range(N_TRIALS):
        out = model(data.clone())
        all_zero = (out == 0.0).all(dim=-1)
        failures += all_zero.sum().item()
        total += N_ENVS

    result = binomial_test(int(failures), total, RATE_FAIL_PROB)
    print(f"\n  RGB failure rate test:")
    print(f"    Observed: {result.observed_rate:.4f}")
    print(f"    Expected: {result.expected_rate}")
    print(f"    p-value:  {result.p_value:.4f}")

    assert not result.reject_null, (
        f"RGB failure rate {result.observed_rate:.4f} differs from "
        f"configured {RATE_FAIL_PROB} (p={result.p_value:.4f})"
    )
