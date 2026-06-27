"""Unit tests for IsaacLabEnvAdapter's action sourcing.

The adapter defers torch/warp imports and takes ``step_dt`` explicitly, so its
stepping logic is testable against a fake env without Isaac Sim. These cover
the scripted driver's raw-action seam (``action_source``) and the
exactly-one-source contract; the camera ``capture()`` path needs a live scene
and is exercised by the Kit smoke.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from strafer_lab.sim_in_the_loop.runtime_env import IsaacLabEnvAdapter


class _FakeEnv:
    def __init__(self) -> None:
        self.unwrapped = SimpleNamespace(
            device="cpu",
            action_manager=SimpleNamespace(action=torch.zeros(1, 3)),
        )
        self.stepped: list[torch.Tensor] = []

    def step(self, action) -> None:
        self.stepped.append(action)


def _adapter(**kwargs):
    return IsaacLabEnvAdapter(
        env=_FakeEnv(), scene_name="s", step_dt=0.1, **kwargs,
    )


class TestActionSource:
    def test_raw_action_fed_straight_to_env(self):
        action = torch.tensor([[0.5, -0.5, 0.25]])
        adapter = _adapter(action_source=lambda: action)
        adapter.step()
        assert adapter._env.stepped[0] is action  # no Twist conversion
        assert adapter._last_action == pytest.approx((0.5, -0.5, 0.25))
        assert adapter.sim_time_s == pytest.approx(0.1)


class TestSourceValidation:
    def test_requires_a_source(self):
        with pytest.raises(ValueError, match="exactly one"):
            _adapter()

    def test_rejects_both_sources(self):
        with pytest.raises(ValueError, match="exactly one"):
            _adapter(
                cmd_vel_source=lambda: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                action_source=lambda: torch.zeros(1, 3),
            )
