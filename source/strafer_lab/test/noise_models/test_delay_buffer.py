# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for DelayBuffer observation latency.

These tests validate the DelayBuffer class which implements per-sensor
observation latency for sim-to-real domain randomization.

"""

# --- Imports (Isaac Sim launched by conftest.py) ---

import torch
import pytest

from strafer_lab.tasks.navigation.mdp.noise_models import DelayBuffer

from test.common import NUM_ENVS, DEVICE


@pytest.mark.parametrize("obs_size", [3, 4, 10])
def test_delay_buffer_zero_passthrough(obs_size):
    """Verify delay_steps=0 returns input unchanged."""
    buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=obs_size, delay_steps=0, device=DEVICE)

    # Generate random input
    data = torch.randn(NUM_ENVS, obs_size, device=DEVICE)
    output = buffer(data)

    # Should be identical (no delay)
    torch.testing.assert_close(output, data)


@pytest.mark.parametrize("delay_steps", [1, 2, 3, 5])
def test_delay_buffer_exact_delay(delay_steps):
    """Verify output is delayed by exactly delay_steps."""
    obs_size = 4
    buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=obs_size, delay_steps=delay_steps, device=DEVICE)

    # Store inputs for verification
    inputs = []
    outputs = []

    # Push data through buffer
    for i in range(delay_steps + 5):
        data = torch.full((NUM_ENVS, obs_size), float(i + 1), device=DEVICE)
        inputs.append(data.clone())
        outputs.append(buffer(data).clone())

    # First delay_steps outputs should be zeros (buffer was empty)
    for i in range(delay_steps):
        torch.testing.assert_close(
            outputs[i],
            torch.zeros_like(outputs[i]),
            msg=f"Output {i} should be zeros (buffer warming up)",
        )

    # After warming up, output should be exactly delay_steps behind input
    for i in range(delay_steps, delay_steps + 5):
        expected = inputs[i - delay_steps]
        torch.testing.assert_close(
            outputs[i],
            expected,
            msg=f"Output {i} should equal input {i - delay_steps}",
        )


def test_delay_buffer_reset_clears_history():
    """Verify reset() clears buffer history."""
    delay_steps = 2
    buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=3, delay_steps=delay_steps, device=DEVICE)

    # Fill buffer with non-zero data
    for _ in range(delay_steps + 1):
        data = torch.randn(NUM_ENVS, 3, device=DEVICE)
        buffer(data)

    # Reset buffer
    buffer.reset()

    # After reset, outputs should be zeros again
    test_data = torch.ones(NUM_ENVS, 3, device=DEVICE)
    output = buffer(test_data)
    torch.testing.assert_close(
        output,
        torch.zeros_like(output),
        msg="After reset, delayed output should be zeros",
    )


def test_delay_buffer_per_env_reset():
    """Verify reset(env_ids) only clears specified environments."""
    delay_steps = 1
    obs_size = 2
    buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=obs_size, delay_steps=delay_steps, device=DEVICE)

    # Fill buffer with identifiable data
    fill_data = torch.arange(NUM_ENVS, device=DEVICE).unsqueeze(1).expand(-1, obs_size).float()
    buffer(fill_data)

    # Push again to have fill_data in delayed position
    second_data = fill_data + 100
    output1 = buffer(second_data)

    # output1 should be fill_data
    torch.testing.assert_close(output1, fill_data, msg="First delayed output should be fill_data")

    # Reset only first 10 environments
    reset_ids = list(range(10))
    buffer.reset(reset_ids)

    # Push new data
    third_data = fill_data + 200
    output2 = buffer(third_data)

    # Reset envs should get zeros, others should get second_data
    torch.testing.assert_close(
        output2[:10],
        torch.zeros(10, obs_size, device=DEVICE),
        msg="Reset env outputs should be zeros",
    )
    torch.testing.assert_close(
        output2[10:],
        second_data[10:],
        msg="Non-reset env outputs should be previous data",
    )


@pytest.mark.parametrize("num_envs", [1, 64, 256])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_delay_buffer_device_batch_size(num_envs, device):
    """Test DelayBuffer works correctly with different batch sizes and devices."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    delay_steps = 2
    obs_size = 4
    buffer = DelayBuffer(num_envs=num_envs, obs_size=obs_size, delay_steps=delay_steps, device=device)

    # Generate input and push through buffer
    data1 = torch.ones(num_envs, obs_size, device=device)
    data2 = torch.ones(num_envs, obs_size, device=device) * 2.0
    data3 = torch.ones(num_envs, obs_size, device=device) * 3.0

    out1 = buffer(data1)
    out2 = buffer(data2)
    out3 = buffer(data3)

    # First 2 outputs should be zeros, third should be data1
    assert out1.device.type == device.split(":")[0]
    assert out1.shape == (num_envs, obs_size)
    torch.testing.assert_close(out3, data1)


def test_delay_buffer_preserves_signal_content():
    """Verify delayed signal matches original exactly (no distortion)."""
    delay_steps = 3
    obs_size = 6
    buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=obs_size, delay_steps=delay_steps, device=DEVICE)

    # Create unique identifiable patterns for each step
    patterns = []
    for i in range(10):
        # Create pattern with step number encoded in the values
        pattern = torch.arange(obs_size, device=DEVICE).float() + (i + 1) * 10
        pattern = pattern.unsqueeze(0).expand(NUM_ENVS, -1).clone()
        patterns.append(pattern)

    outputs = []
    for pattern in patterns:
        outputs.append(buffer(pattern).clone())

    # Verify outputs after warmup match inputs exactly
    for i in range(delay_steps, len(patterns)):
        torch.testing.assert_close(
            outputs[i],
            patterns[i - delay_steps],
            msg=f"Delayed output at step {i} should exactly match input at step {i - delay_steps}",
        )
