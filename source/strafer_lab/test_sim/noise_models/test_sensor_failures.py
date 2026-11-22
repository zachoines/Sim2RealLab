# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for sensor failure modes across all noise models.

A failed IMU term or encoder repeats its previous reading (``stuck_probability``), as a held source does in the
deployed node's cache (each IMU term is drawn on its own, where the robot's one IMU message holds both); on an env's
first step after a reset there is no previous reading and the current one passes. The rate cases are seeded, so a read
is reproducible, and each is tested two-sided at alpha 1e-6: at alpha 0.05 a seeded case is a fixed 1-in-20 draw on each
device's random stream, which one GPU's stream at seed 7 missed (p 0.045) with the model correct. At 80 000 samples a
rate off by 0.01 still fails (z about 8). The depth model has no failure mode of its own: a failed frame is a dropped one, which repeats the
previous frame. The RGB model returns a black frame (``failure_probability``); no registered ID observes RGB.

These tests validate:
1. A stuck output equals the previous output bit for bit, and a reset forgets it.
2. Observed stuck rates match the configured probability (binomial test).
3. No depth output is all max_range from a failure, and the RGB failure output is black.

All tests run the noise models in isolation (no Isaac Sim environment needed
beyond the AppLauncher started by root conftest.py).

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test_sim/noise_models/test_sensor_failures.py -v
"""

import torch
import numpy as np

from test_sim.common import (
    CONFIDENCE_LEVEL,
    binomial_test,
    DEVICE,
)

from strafer_shared.constants import DEPTH_HEIGHT, DEPTH_WIDTH

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
N_TRIALS = 5_000  # 5K trials × 16 envs = 80K samples for stable binomial tests

# High failure prob for deterministic-like behaviour in output-value tests
HIGH_FAIL_PROB = 1.0

# Moderate failure prob for rate-matching tests
RATE_FAIL_PROB = 0.15
RATE_STUCK_PROB = 0.10
RATE_CONFIDENCE = 1.0 - 1e-6  # the rate cases' two-sided level: a correct model does not fail by chance


# =====================================================================
# IMU Sensor Failures
# =====================================================================

def _make_imu(stuck_prob: float = 0.0) -> IMUNoiseModel:
    cfg = IMUNoiseModelCfg(
        sensor_type="accel",
        accel_noise_std=0.01,
        accel_bias_range=(0.0, 0.0),
        accel_bias_drift_rate=0.0,
        output_size=3,
        latency_steps=0,
        stuck_probability=stuck_prob,
    )
    return IMUNoiseModel(cfg, N_ENVS, DEVICE)


def test_imu_stuck_returns_previous():
    """With stuck_probability=1.0, every output after the first equals the first, bit for bit."""
    model = _make_imu(stuck_prob=HIGH_FAIL_PROB)
    first_out = model(torch.ones(N_ENVS, 3, device=DEVICE))
    for i in range(20):
        out = model(torch.randn(N_ENVS, 3, device=DEVICE) * 100)
        assert torch.equal(out, first_out), f"step {i}: max diff {(out - first_out).abs().max().item()}"


def test_imu_reset_forgets_the_previous_reading():
    """After a reset of some envs, their first stuck step passes the current reading; the rest still repeat."""
    model = _make_imu(stuck_prob=HIGH_FAIL_PROB)
    first_out = model(torch.ones(N_ENVS, 3, device=DEVICE))
    model.reset(torch.tensor([2, 5]))
    data = torch.full((N_ENVS, 3), 7.0, device=DEVICE)
    out = model(data.clone())
    kept = [e for e in range(N_ENVS) if e not in (2, 5)]
    assert torch.equal(out[kept], first_out[kept])
    assert not torch.equal(out[[2, 5]], first_out[[2, 5]])
    assert not (out[[2, 5]] == 0.0).all(), "a reset env must not read zeros"


def test_imu_stuck_rate_matches_config():
    """Observed stuck rate should match stuck_probability (binomial test)."""
    torch.manual_seed(7)
    model = _make_imu(stuck_prob=RATE_STUCK_PROB)
    stuck_count = 0
    total = 0
    prev_out = model(torch.randn(N_ENVS, 3, device=DEVICE) * 10)
    for _ in range(N_TRIALS):
        out = model(torch.randn(N_ENVS, 3, device=DEVICE) * 10)
        stuck_count += torch.all(out == prev_out, dim=-1).sum().item()
        total += N_ENVS
        prev_out = out.clone()
    result = binomial_test(int(stuck_count), total, RATE_STUCK_PROB, confidence_level=RATE_CONFIDENCE)
    print(f"\n  IMU stuck rate test: observed {result.observed_rate:.4f}, expected {result.expected_rate}, "
          f"p {result.p_value:.4f}")
    assert not result.reject_null, (
        f"IMU stuck rate {result.observed_rate:.4f} differs from configured {RATE_STUCK_PROB} (p={result.p_value:.4f})"
    )


# =====================================================================
# Encoder Sensor Failures
# =====================================================================

def _make_encoder(stuck_prob: float = 0.0) -> EncoderNoiseModel:
    cfg = EncoderNoiseModelCfg(
        enable_quantization=False,
        velocity_noise_std=0.01,
        max_velocity=3000.0,
        stuck_probability=stuck_prob,
        latency_steps=0,
        output_size=4,
    )
    return EncoderNoiseModel(cfg, N_ENVS, DEVICE)


def test_encoder_stuck_returns_previous():
    """With stuck_probability=1.0, every encoder output after the first equals the first, bit for bit."""
    model = _make_encoder(stuck_prob=HIGH_FAIL_PROB)
    first_out = model(torch.ones(N_ENVS, 4, device=DEVICE) * 1000.0)
    for i in range(20):
        out = model(torch.randn(N_ENVS, 4, device=DEVICE) * 500.0)
        assert torch.equal(out, first_out), f"step {i}: max diff {(out - first_out).abs().max().item()}"
        assert not (out == 0.0).all(), "a stuck encoder must not read zeros"


def test_encoder_stuck_rate_matches_config():
    """Encoder stuck rate should match the configured probability."""
    torch.manual_seed(7)
    model = _make_encoder(stuck_prob=RATE_STUCK_PROB)
    stuck_count = 0
    total = 0
    prev_out = model(torch.randn(N_ENVS, 4, device=DEVICE) * 500.0)
    for _ in range(N_TRIALS):
        out = model(torch.randn(N_ENVS, 4, device=DEVICE) * 500.0)
        stuck_count += torch.all(out == prev_out, dim=-1).sum().item()
        total += N_ENVS
        prev_out = out.clone()
    result = binomial_test(int(stuck_count), total, RATE_STUCK_PROB, confidence_level=RATE_CONFIDENCE)
    print(f"\n  Encoder stuck rate test: observed {result.observed_rate:.4f}, expected {result.expected_rate}, "
          f"p {result.p_value:.4f}")
    assert not result.reject_null, (
        f"Encoder stuck rate {result.observed_rate:.4f} differs from configured {RATE_STUCK_PROB} "
        f"(p={result.p_value:.4f})"
    )


# =====================================================================
# Depth Camera: no failure mode of its own
# =====================================================================

def test_depth_has_no_all_far_failure():
    """The depth model has no failure field, and a dropped frame repeats the previous one rather than reading far."""
    assert not hasattr(DepthNoiseModelCfg(), "failure_probability")
    cfg = DepthNoiseModelCfg(
        baseline_m=0.095, focal_length_px=673.0, disparity_noise_px=0.0, hole_probability=0.0,
        frame_drop_prob=HIGH_FAIL_PROB, min_range=0.2, max_range=6.0, latency_steps=0,
        height=DEPTH_HEIGHT, width=DEPTH_WIDTH,
    )
    model = DepthNoiseModel(cfg, N_ENVS, DEVICE)
    n_pixels = DEPTH_HEIGHT * DEPTH_WIDTH
    first_out = model(torch.ones(N_ENVS, n_pixels, device=DEVICE) * 2.0)
    for _ in range(20):
        out = model(torch.ones(N_ENVS, n_pixels, device=DEVICE) * 3.0)
        assert torch.equal(out, first_out)
        assert not (out == cfg.max_range).all(dim=-1).any()


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

    n_pixels = DEPTH_HEIGHT * DEPTH_WIDTH
    data = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 0.5  # mid-grey
    for _ in range(20):
        out = model(data.clone())
        assert torch.all(out == 0.0), (
            f"Expected black (zeros) on RGB failure, got max={out.max().item():.4f}"
        )


def test_rgb_failure_rate_matches_config():
    """RGB camera failure rate should match configured probability."""
    torch.manual_seed(7)
    model = _make_rgb_with_failure(failure_prob=RATE_FAIL_PROB)

    n_pixels = DEPTH_HEIGHT * DEPTH_WIDTH
    data = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 0.5
    failures = 0
    total = 0

    for _ in range(N_TRIALS):
        out = model(data.clone())
        all_zero = (out == 0.0).all(dim=-1)
        failures += all_zero.sum().item()
        total += N_ENVS

    result = binomial_test(int(failures), total, RATE_FAIL_PROB, confidence_level=RATE_CONFIDENCE)
    print(f"\n  RGB failure rate test:")
    print(f"    Observed: {result.observed_rate:.4f}")
    print(f"    Expected: {result.expected_rate}")
    print(f"    p-value:  {result.p_value:.4f}")

    assert not result.reject_null, (
        f"RGB failure rate {result.observed_rate:.4f} differs from "
        f"configured {RATE_FAIL_PROB} (p={result.p_value:.4f})"
    )
