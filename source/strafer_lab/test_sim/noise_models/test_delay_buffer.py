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

from test_sim.common import NUM_ENVS, DEVICE


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

    # Until the ring has been written the buffer stands in the first frame,
    # rather than the zeros it used to return — 0.0 is a reading no sensor
    # produces, and the warm-up is the one place the policy could meet it.
    for i in range(delay_steps):
        torch.testing.assert_close(
            outputs[i],
            inputs[0],
            msg=f"Output {i} should be the first frame (buffer warming up)",
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
    """Verify reset() clears buffer history.

    The assertion is that no pre-reset frame can be read afterwards. It used to
    be that the output is zero, which is a weaker claim about a stronger
    behaviour: zero is also what an unwritten buffer returns, so it could not
    tell "the history is gone" from "nothing has arrived yet".
    """
    delay_steps = 2
    buffer = DelayBuffer(num_envs=NUM_ENVS, obs_size=3, delay_steps=delay_steps, device=DEVICE)

    # Fill the buffer with history that is identifiable and cannot recur.
    history = []
    for step in range(delay_steps + 1):
        data = torch.full((NUM_ENVS, 3), -100.0 - step, device=DEVICE)
        history.append(data.clone())
        buffer(data)

    buffer.reset()

    # After the reset the buffer stands in the incoming frame, and no step of
    # the pre-reset history can be read at any point in the new episode.
    test_data = torch.ones(NUM_ENVS, 3, device=DEVICE)
    for step in range(delay_steps + 1):
        output = buffer(test_data)
        torch.testing.assert_close(
            output,
            test_data,
            msg=f"Step {step} after reset should read the post-reset frame",
        )
        for age, stale in enumerate(history):
            assert not torch.allclose(output, stale), (
                f"Step {step} after reset read pre-reset history (age {age})"
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

    # The reset envs read their own new frame; the rest are untouched and still
    # read the frame they wrote a step ago. The second assertion is the one
    # that matters: standing in the incoming frame must not reach across into
    # the envs that kept running, or their latency silently shortens.
    torch.testing.assert_close(
        output2[:10],
        third_data[:10],
        msg="Reset env outputs should be their own post-reset frame",
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

    # The warm-up stands in data1, and the third output is data1 by delay.
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
