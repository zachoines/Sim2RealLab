# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The age of the base pose the goal-shaped observations are projected with.

On the robot the node projects the goal through TF's ``map -> base_link`` at its latest stamp, and that transform is
only as fresh as the slowest publisher in its chain, so the goal can be projected through a pose a few control ticks
old while the robot has moved. This process keeps a short ring of the robot's past base poses per env and returns, for
each env, the pose ``k`` control steps old: ``k`` drawn per env in a band at each reset and redrawn every
``interval_steps``, clipped to the history recorded since the env's reset. A band whose top is 0 leaves it disabled.

The poses are recorded once per control step, at the first goal-shaped read of the step; every goal-shaped term reads
the same aged pose, as they all read one realization of the referent's drift.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class TfAgeProcess:
    """Per-env ring of past base poses, read at a per-env age in control steps."""

    def __init__(self, num_envs: int, device, age_steps_range: tuple[int, int], interval_steps: int):
        lo, hi = sorted(int(x) for x in age_steps_range)
        self._range = (max(0, lo), max(0, hi))
        self.enabled = self._range[1] > 0
        self._num_envs = num_envs
        self._device = device
        self._length = self._range[1] + 1
        self._pos = torch.zeros(self._length, num_envs, 3, device=device)
        self._quat = torch.zeros(self._length, num_envs, 4, device=device)
        self._count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._interval = max(1, int(interval_steps))
        self._since_draw = 0
        self._head = 0
        self._last_step = None
        self._env_index = torch.arange(num_envs, device=device)

    @property
    def age(self) -> torch.Tensor:
        """Each env's drawn age in control steps (before clipping to its history)."""
        return self._age

    def _draw(self, env_ids: torch.Tensor | None) -> None:
        lo, hi = self._range
        n = self._num_envs if env_ids is None else len(env_ids)
        if n == 0:
            return
        draw = torch.randint(lo, hi + 1, (n,), device=self._device)
        if env_ids is None:
            self._age.copy_(draw)
        else:
            self._age[env_ids] = draw

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Forget the reset envs' history and redraw their age."""
        if not self.enabled:
            return
        if env_ids is None:
            self._count.zero_()
            self._draw(None)
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
        self._count[ids] = 0
        self._draw(ids)

    def observe(self, step_index: int, pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Record this step's poses (once per step) and return each env's pose at its age."""
        if step_index != self._last_step:
            self._last_step = step_index
            self._head = (self._head + 1) % self._length
            self._pos[self._head] = pos_w
            self._quat[self._head] = quat_w
            self._count = torch.clamp(self._count + 1, max=self._length)
            self._since_draw += 1
            if self._since_draw >= self._interval:
                self._since_draw = 0
                self._draw(None)
        fresh = self._count == 0
        if bool(fresh.any()):  # reset within this step: the live pose is the only one it has
            self._pos[self._head, fresh] = pos_w[fresh]
            self._quat[self._head, fresh] = quat_w[fresh]
            self._count[fresh] = 1
        k = torch.minimum(self._age, self._count - 1)
        idx = torch.remainder(self._head - k, self._length)
        return self._pos[idx, self._env_index], self._quat[idx, self._env_index]
