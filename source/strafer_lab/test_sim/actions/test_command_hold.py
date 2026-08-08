# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for command holds in action processing.

A hold is a control step on which no new command arrives, so the chassis
re-executes the last one it received. It is a different mechanism from the
delay buffer next to it: a delay shifts a command in time, a hold repeats it,
and only the hold breaks the one-command-per-step assumption a policy is
otherwise trained under.

These tests exercise the hold through the real action term, with motor
dynamics and slew limiting off so the emitted wheel velocities are the
commands themselves rather than a filtered version of them.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test_sim/actions/test_command_hold.py -v
"""

import numpy as np
import torch

from test_sim.actions.conftest import configure_action_term_dynamics


def _drive_random_commands(action_term, env, steps: int, seed: int = 0):
    """Issue a distinct command every step; return the per-step repeat mask.

    Every step asks for a different velocity, so an emitted command equal to
    the previous one can only have come from a hold.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    repeats = []
    previous = None
    for _ in range(steps):
        action = torch.rand(env.num_envs, 3, generator=generator).to(env.device) * 2.0 - 1.0
        action_term.process_actions(action)
        emitted = action_term.processed_actions.clone()
        if previous is not None:
            repeats.append(torch.all(emitted == previous, dim=1).cpu().numpy().copy())
        previous = emitted
    return np.stack(repeats)


def test_hold_repeats_the_previous_command(action_env):
    """On a held step the emitted wheel velocities are the previous step's,
    byte for byte — not a blend, not a decay toward zero."""
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=False, enable_slew=False,
        enable_hold=True,
    )
    assert action_term._hold.enabled, "the realistic tier carries a command-hold band"

    generator = torch.Generator(device="cpu").manual_seed(7)
    previous = None
    checked = 0
    for _ in range(600):
        action = torch.rand(action_env.num_envs, 3, generator=generator)
        action_term.process_actions(action.to(action_env.device) * 2.0 - 1.0)
        emitted = action_term.processed_actions.clone()
        held = action_term._hold._holding
        if previous is not None and bool(held.any()):
            assert torch.equal(emitted[held], previous[held])
            checked += int(held.sum())
        previous = emitted

    assert checked > 0, "no hold fired in 600 steps — the band is not being sampled"


def test_hold_fraction_lands_inside_the_configured_band(action_env):
    """The realized share of repeated commands has to sit inside the band the
    contract asked for, and be non-zero: a silently inert hold would leave the
    training distribution exactly where it was."""
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=False, enable_slew=False,
        enable_hold=True,
    )
    band = action_term.cfg.hold_fraction_range
    repeats = _drive_random_commands(action_term, action_env, steps=1200, seed=3)

    realized = repeats.mean()
    print(f"\n  Command hold — band {band}, realized {realized:.4f}")
    assert 0.0 < realized < band[1] + 0.02


def test_hold_is_inert_when_isolated_off(action_env):
    """The isolation switch every other action test relies on has to actually
    silence the hold, or those tests measure a delay through a repeated
    command."""
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=False, enable_slew=False,
        enable_hold=False,
    )
    assert action_term._hold.enabled is False
    repeats = _drive_random_commands(action_term, action_env, steps=400, seed=5)
    assert not repeats.any()


def test_a_first_step_hold_issues_the_at_rest_command(action_env):
    """Held on the very first step of an episode, the term emits zero.

    Nothing has been published yet, so zero is what a chassis is executing.
    Pinned because it is the deliberate asymmetry against the depth stream,
    where a first-step hold has no frame to repeat and the live one is emitted
    instead: one mechanism has a well-defined at-rest value and the other does
    not."""
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=False, enable_slew=False,
        enable_hold=True,
    )

    drive = torch.zeros(action_env.num_envs, 3, device=action_env.device)
    drive[:, 0] = 1.0

    saw_first_step_hold = False
    for trial in range(60):
        action_term.reset(env_ids=torch.arange(action_env.num_envs, device=action_env.device))
        action_term.process_actions(drive)
        held = action_term._hold._holding
        if bool(held.any()):
            saw_first_step_hold = True
            assert torch.count_nonzero(action_term.processed_actions[held]) == 0

    assert saw_first_step_hold, "no first-step hold fired in 60 resets"


def test_hold_state_is_per_environment(action_env):
    """Envs hold independently — a batch that stalled in lockstep would train
    the policy against a schedule no deployment produces."""
    action_term = configure_action_term_dynamics(
        action_env, enable_motor=False, enable_delay=False, enable_slew=False,
        enable_hold=True,
    )
    repeats = _drive_random_commands(action_term, action_env, steps=800, seed=11)
    per_env = repeats.mean(axis=0)

    assert per_env.max() > 0.0
    # Independent per-env draws leave a spread; one shared schedule would not.
    assert per_env.std() > 0.0
    assert not np.all(repeats.sum(axis=1) % action_env.num_envs == 0)
