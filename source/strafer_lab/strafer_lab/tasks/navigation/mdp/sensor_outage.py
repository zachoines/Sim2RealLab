# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Per-source sensor outages and the deployed node's stale-source stop.

The node caches the newest message per source, and a late source repeats bit for bit until it is stale past its
timeout (the IMU and the driver's joint states and odom at 0.2 s, the depth at 0.5 s). It then publishes a zero twist
without stepping the policy, keeping ``last_action``, and resumes when the source returns. The IMU and the depth share
the D555's USB link and stall together; the encoders and the body velocity are one driver reading.

``OutageProcess`` models that per env with three sources: an outage start rate per source, plus a shared USB-stall rate
that starts the IMU and the depth together, and outage lengths drawn in a band. During an outage the source repeats.
While any source has been stale for at least its bound, ``round(timeout / step_dt)`` control steps, the env is stopped.
The action term advances the process once per control step and zeroes the commanded twist while stopped. The noise
models read the stale masks through :func:`current_outage`, and the policy's ``last_action`` is held through
``HeldLastActionNoiseModel``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

SOURCES = ("imu", "driver", "depth")

_ACTIVE: list = []


def current_outage(num_envs: int, device) -> "OutageProcess | None":
    """The outage process built for an env of this size on this device, if one is active."""
    for process in _ACTIVE:
        if process.num_envs == num_envs and str(process.device) == str(device):
            return process
    return None


class OutageProcess:
    """Per-env outage state for the three sources and the stop it drives."""

    def __init__(self, num_envs: int, device, step_dt: float, start_rate_hz: dict, usb_stall_rate_hz: float,
                 length_s_range: tuple[float, float], timeout_s: dict):
        self.num_envs = num_envs
        self.device = device
        self._p_start = {s: min(1.0, float(start_rate_hz.get(s, 0.0)) * step_dt) for s in SOURCES}
        self._p_usb = min(1.0, float(usb_stall_rate_hz) * step_dt)
        lo, hi = sorted(float(x) for x in length_s_range)
        self._length_steps = (max(1, int(round(lo / step_dt))), max(1, int(round(hi / step_dt))))
        self.bound_steps = {s: max(1, int(round(float(timeout_s[s]) / step_dt))) for s in SOURCES}
        self._remaining = {s: torch.zeros(num_envs, dtype=torch.long, device=device) for s in SOURCES}
        self._age = {s: torch.zeros(num_envs, dtype=torch.long, device=device) for s in SOURCES}
        self._stopped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stop_steps = 0  # stopped env-steps, a count for the logs
        _ACTIVE[:] = [p for p in _ACTIVE if not (p.num_envs == num_envs and str(p.device) == str(device))] + [self]

    def stale(self, source: str) -> torch.Tensor:
        """Envs whose source is in an outage this control step."""
        return self._age[source] > 0

    def age(self, source: str) -> torch.Tensor:
        """Consecutive stale control steps of the source, per env."""
        return self._age[source]

    @property
    def stopped(self) -> torch.Tensor:
        """Envs the node would stop this control step: a source stale for at least its bound."""
        return self._stopped

    def _lengths(self, n: int) -> torch.Tensor:
        lo, hi = self._length_steps
        return torch.randint(lo, hi + 1, (n,), device=self.device)

    def advance(self) -> None:
        """One control step: start outages by their rates, run them down, and update the stale ages and the stop."""
        usb = torch.rand(self.num_envs, device=self.device) < self._p_usb
        for s in SOURCES:
            idle = self._remaining[s] == 0
            start = idle & (torch.rand(self.num_envs, device=self.device) < self._p_start[s])
            if s in ("imu", "depth"):
                start = start | (idle & usb)
            if bool(start.any()):
                self._remaining[s][start] = self._lengths(int(start.sum()))
            out = self._remaining[s] > 0
            self._age[s] = torch.where(out, self._age[s] + 1, torch.zeros_like(self._age[s]))
            self._remaining[s] = torch.clamp(self._remaining[s] - 1, min=0)
        stopped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for s in SOURCES:
            stopped |= self._age[s] >= self.bound_steps[s]
        self._stopped = stopped
        self.stop_steps += int(stopped.sum())

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """End the reset envs' outages and their stop."""
        idx = slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for s in SOURCES:
            self._remaining[s][idx] = 0
            self._age[s][idx] = 0
        self._stopped[idx] = False
