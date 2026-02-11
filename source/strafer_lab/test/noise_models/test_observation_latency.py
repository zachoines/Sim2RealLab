# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for per-sensor observation latency.

Each noise model can be configured with a ``latency_steps`` parameter that
inserts a ``DelayBuffer`` into the observation pipeline. This test validates
end-to-end that:

1. Noise models with ``latency_steps=0`` (IMU, encoder) pass data through
   without delay — output on step N reflects the input on step N.
2. Noise models with ``latency_steps>0`` (depth, RGB cameras) produce
   stale output — e.g. a step change in input takes exactly *latency_steps*
   before it appears in the output.

These tests operate on the noise models directly (no full env needed beyond
AppLauncher) because the latency is internal to each model.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/noise_models/test_observation_latency.py -v
"""

import torch
import numpy as np

from test.common import DEVICE

from strafer_lab.tasks.navigation.mdp.noise_models import (
    IMUNoiseModel, IMUNoiseModelCfg,
    EncoderNoiseModel, EncoderNoiseModelCfg,
    DepthNoiseModel, DepthNoiseModelCfg,
    RGBNoiseModel, RGBNoiseModelCfg,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ENVS = 4


# =====================================================================
# Helpers
# =====================================================================

def _make_imu(latency_steps: int = 0) -> IMUNoiseModel:
    """Create a noiseless IMU model with configurable latency."""
    cfg = IMUNoiseModelCfg(
        sensor_type="accel",
        accel_noise_std=0.0,
        accel_bias_range=(0.0, 0.0),
        accel_bias_drift_rate=0.0,
        output_size=3,
        latency_steps=latency_steps,
    )
    return IMUNoiseModel(cfg, N_ENVS, DEVICE)


def _make_encoder(latency_steps: int = 0) -> EncoderNoiseModel:
    """Create a noiseless encoder model with configurable latency."""
    cfg = EncoderNoiseModelCfg(
        enable_quantization=False,
        velocity_noise_std=0.0,
        max_velocity=3000.0,
        missed_tick_prob=0.0,
        extra_tick_prob=0.0,
        failure_probability=0.0,
        latency_steps=latency_steps,
        output_size=4,
    )
    return EncoderNoiseModel(cfg, N_ENVS, DEVICE)


def _make_depth(latency_steps: int = 1) -> DepthNoiseModel:
    """Create a noiseless depth model with configurable latency."""
    cfg = DepthNoiseModelCfg(
        baseline_m=0.095,
        focal_length_px=673.0,
        disparity_noise_px=0.0,  # no stereo noise
        hole_probability=0.0,
        frame_drop_prob=0.0,
        min_range=0.0,
        max_range=100.0,  # very high to avoid clamping
        failure_probability=0.0,
        latency_steps=latency_steps,
    )
    return DepthNoiseModel(cfg, N_ENVS, DEVICE)


def _make_rgb(latency_steps: int = 1) -> RGBNoiseModel:
    """Create a noiseless RGB model with configurable latency."""
    cfg = RGBNoiseModelCfg(
        pixel_noise_std=0.0,
        brightness_range=(1.0, 1.0),
        frame_drop_prob=0.0,
        failure_probability=0.0,
        latency_steps=latency_steps,
    )
    return RGBNoiseModel(cfg, N_ENVS, DEVICE)


# =====================================================================
# IMU Latency Tests — expected latency: 0 steps
# =====================================================================


def test_imu_zero_latency_passthrough():
    """IMU with latency_steps=0 should return the input value immediately.

    This matches the real sensor hardware (~1-2 ms latency, negligible
    relative to the 33 ms control dt).
    """
    model = _make_imu(latency_steps=0)

    value_a = torch.ones(N_ENVS, 3, device=DEVICE) * 5.0
    value_b = torch.ones(N_ENVS, 3, device=DEVICE) * 10.0

    out_a = model(value_a.clone())
    out_b = model(value_b.clone())

    # With zero noise, output should match input on the same step
    assert torch.allclose(out_a, value_a, atol=1e-6), (
        f"IMU latency=0: output {out_a[0]} != input {value_a[0]}"
    )
    assert torch.allclose(out_b, value_b, atol=1e-6), (
        f"IMU latency=0: output {out_b[0]} != input {value_b[0]}"
    )


def test_imu_with_latency_delays_output():
    """IMU with latency_steps=2 should return the input from 2 steps ago."""
    delay = 2
    model = _make_imu(latency_steps=delay)

    sentinel = 99.0
    values = []
    outputs = []

    # Feed increasing values (0, 1, 2, ...) for easy identification
    for step in range(delay + 5):
        v = torch.ones(N_ENVS, 3, device=DEVICE) * float(step)
        values.append(step)
        out = model(v)
        outputs.append(out[0, 0].cpu().item())

    print(f"\n  IMU latency={delay} test:")
    for i, (v, o) in enumerate(zip(values, outputs)):
        print(f"    step {i}: input={v:.0f}  output={o:.1f}")

    # After the buffer is full (step >= delay), output should be input from
    # `delay` steps ago.
    for step in range(delay, len(values)):
        expected = float(step - delay)
        actual = outputs[step]
        assert abs(actual - expected) < 1e-4, (
            f"Step {step}: expected output {expected} (input from step {step-delay}), "
            f"got {actual}"
        )


# =====================================================================
# Encoder Latency Tests — expected latency: 0 steps
# =====================================================================


def test_encoder_zero_latency_passthrough():
    """Encoder with latency_steps=0 should return input immediately."""
    model = _make_encoder(latency_steps=0)

    data = torch.ones(N_ENVS, 4, device=DEVICE) * 100.0
    out = model(data.clone())

    # Encoder passes through raw data when noise/quantization is off
    assert torch.allclose(out, data, atol=1e-4), (
        f"Encoder latency=0: output diverges from input by "
        f"{(out - data).abs().max().item()}"
    )


# =====================================================================
# Depth Camera Latency Tests — expected latency: 1 step
# =====================================================================


def test_depth_one_step_latency():
    """Depth camera with latency_steps=1 delays output by exactly one step.

    A step function input (constant → different constant) should appear
    in the output one step later.
    """
    model = _make_depth(latency_steps=1)

    n_pixels = 10  # small for speed

    constant = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 3.0
    step_val = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 8.0

    # Pre-fill the buffer with the constant value
    for _ in range(3):
        model(constant.clone())

    # Now inject a step change
    out_at_step = model(step_val.clone())
    out_after = model(step_val.clone())

    # At the step itself, the output should still be the OLD value (delayed)
    assert torch.allclose(out_at_step, constant, atol=1e-4), (
        f"Depth latency=1: step change appeared immediately "
        f"(output={out_at_step[0,0].item():.2f}, expected={constant[0,0].item():.2f})"
    )

    # One step later, the step change should appear
    assert torch.allclose(out_after, step_val, atol=1e-4), (
        f"Depth latency=1: step change did not appear after 1 step "
        f"(output={out_after[0,0].item():.2f}, expected={step_val[0,0].item():.2f})"
    )


def test_depth_zero_latency_passthrough():
    """Depth camera with latency_steps=0 should return input immediately."""
    model = _make_depth(latency_steps=0)

    n_pixels = 10
    data = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 4.0
    out = model(data.clone())

    assert torch.allclose(out, data, atol=1e-4), (
        f"Depth latency=0: output diverges by {(out - data).abs().max().item()}"
    )


# =====================================================================
# RGB Camera Latency Tests — expected latency: 1 step
# =====================================================================


def test_rgb_one_step_latency():
    """RGB camera with latency_steps=1 delays output by exactly one step."""
    model = _make_rgb(latency_steps=1)

    n_pixels = 10
    constant = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 0.3
    step_val = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 0.7

    # Pre-fill
    for _ in range(3):
        model(constant.clone())

    out_at_step = model(step_val.clone())
    out_after = model(step_val.clone())

    assert torch.allclose(out_at_step, constant, atol=1e-4), (
        f"RGB latency=1: step change appeared immediately"
    )
    assert torch.allclose(out_after, step_val, atol=1e-4), (
        f"RGB latency=1: step change did not appear after 1 step"
    )


def test_rgb_reset_clears_latency_buffer():
    """After reset(), the latency buffer should be cleared.

    Injecting a value, resetting, then injecting a different value
    should not return the pre-reset value.
    """
    model = _make_rgb(latency_steps=1)

    n_pixels = 10
    pre_reset = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 0.9
    post_reset = torch.ones(N_ENVS, n_pixels, device=DEVICE) * 0.1

    # Fill buffer with pre_reset values
    for _ in range(5):
        model(pre_reset.clone())

    # Reset all envs
    model.reset()

    # After reset, feed new value. The first output should NOT be pre_reset.
    out = model(post_reset.clone())

    # The DelayBuffer fills with zeros after reset, so the first output
    # is the zero-initialized buffer value (not the pre-reset value).
    # The key assertion is that it's NOT the pre-reset value.
    assert not torch.allclose(out, pre_reset, atol=1e-3), (
        f"RGB: reset did not clear latency buffer. "
        f"Output still matches pre-reset value {pre_reset[0, 0].item()}"
    )
