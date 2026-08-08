"""Bursty hold process for temporal-texture randomization.

A deployed stream stalls in *runs*, not in isolated ticks: once a depth frame
has failed to advance, the next tick is far more likely to be stale than a
healthy tick is. A per-step Bernoulli draw cannot express that. Its runs are
geometric with a mean pinned at ``1 / (1 - p)``, so at the per-step rates a
nominally healthy stream implies, essentially every run is one tick long.

This module models a stall as a two-state process whose hold-run lengths come
from a mixture of two geometrics — a short base component and a long burst
component. That makes the stationary hold fraction and the mean run length
independent knobs, and both are sampled per environment at reset so one
training batch spans a band of temporal textures instead of a single point.

The state machine is equivalent to drawing run lengths explicitly: a geometric
run of mean ``m`` is exactly a per-step exit probability ``1 / m``, and the
mixture component is chosen once when a run begins, so a run's law never
changes mid-run.

Stationary relations, given a mean hold run ``m`` and a target hold fraction
``f``::

    exit  = 1 / m                       mean hold run  = m
    enter = f / (m * (1 - f))           mean live run  = 1 / enter
    f     = enter / (enter + exit)      reachable iff  f <= m / (m + 1)

The reachability ceiling is the statement that the mean *live* gap between
holds cannot fall below one tick.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def mixture_mean_run(mean_run: float, burst_weight: float, burst_run: float) -> float:
    """Mean hold-run length of the base/burst mixture, in control steps."""
    return (1.0 - burst_weight) * mean_run + burst_weight * burst_run


def max_hold_fraction(mean_run: float) -> float:
    """Largest stationary hold fraction a mean hold run can reach."""
    return mean_run / (mean_run + 1.0)


def entry_probability(hold_fraction: float, mean_run: float) -> float:
    """Per-step probability of starting a hold run, from the stationary pair."""
    if hold_fraction <= 0.0:
        return 0.0
    ceiling = max_hold_fraction(mean_run)
    fraction = min(hold_fraction, ceiling)
    if fraction >= 1.0:
        return 1.0
    return min(1.0, fraction / (mean_run * (1.0 - fraction)))


class HoldProcess:
    """Per-environment hold state with mixture-geometric run lengths.

    Args:
        num_envs: Number of parallel environments.
        device: Torch device for the state tensors.
        hold_fraction_range: Per-env stationary hold fraction band [min, max].
            A max of 0 leaves the process inert.
        hold_run_range: Per-env mean base-run length band [min, max], in
            control steps. Values below 1 are raised to 1 — a run that occurs
            lasts at least one tick.
        burst_weight: Share of hold runs drawn from the burst component.
        burst_run: Mean length of a burst run, in control steps.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        *,
        hold_fraction_range: tuple[float, float],
        hold_run_range: tuple[float, float],
        burst_weight: float = 0.0,
        burst_run: float = 1.0,
    ):
        self._num_envs = num_envs
        self._device = device

        f_lo, f_hi = min(hold_fraction_range), max(hold_fraction_range)
        if f_lo < 0.0 or f_hi >= 1.0:
            raise ValueError(
                f"hold_fraction_range must lie in [0, 1), got {hold_fraction_range}"
            )
        self._fraction_range = (f_lo, f_hi)
        self._run_range = (
            max(1.0, min(hold_run_range)),
            max(1.0, max(hold_run_range)),
        )
        if not 0.0 <= burst_weight <= 1.0:
            raise ValueError(f"burst_weight must lie in [0, 1], got {burst_weight}")
        self._burst_weight = burst_weight
        self._burst_run = max(1.0, burst_run)

        self.enabled = f_hi > 0.0

        self._holding = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._enter_p = torch.zeros(num_envs, device=device)
        self._base_exit_p = torch.ones(num_envs, device=device)
        self._burst_exit_p = 1.0 / self._burst_run
        self._exit_p = torch.ones(num_envs, device=device)
        # Realized clamping is reported rather than silently absorbed: the
        # shipped ranges are chosen so it never fires, and a test pins that.
        self.clamped_draws = 0

        if self.enabled:
            self.reset(None)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Redraw the per-env hold parameters and clear the run state."""
        if not self.enabled:
            return
        idx = slice(None) if env_ids is None else env_ids
        count = self._num_envs if env_ids is None else len(env_ids)
        if count == 0:
            return

        f_lo, f_hi = self._fraction_range
        m_lo, m_hi = self._run_range
        fraction = f_lo + (f_hi - f_lo) * torch.rand(count, device=self._device)
        mean_run = m_lo + (m_hi - m_lo) * torch.rand(count, device=self._device)

        mean_run_eff = (
            1.0 - self._burst_weight
        ) * mean_run + self._burst_weight * self._burst_run
        ceiling = mean_run_eff / (mean_run_eff + 1.0)
        self.clamped_draws += int(torch.count_nonzero(fraction > ceiling))
        fraction = torch.minimum(fraction, ceiling)

        enter = fraction / (mean_run_eff * (1.0 - fraction)).clamp_min(1e-6)
        self._enter_p[idx] = enter.clamp(0.0, 1.0)
        self._base_exit_p[idx] = (1.0 / mean_run).clamp(0.0, 1.0)
        self._exit_p[idx] = self._base_exit_p[idx]
        self._holding[idx] = False

    def step(self) -> torch.Tensor:
        """Advance one control step; return the per-env "held this tick" mask."""
        if not self.enabled:
            return self._holding

        draw = torch.rand(self._num_envs, device=self._device)
        was_holding = self._holding
        begins = (~was_holding) & (draw < self._enter_p)
        held = begins | (was_holding & (draw >= self._exit_p))

        # A run's component is chosen once, when the run begins, so its law does
        # not change mid-run. Written branch-free: guarding on ``begins.any()``
        # would read a device tensor on the host every control step.
        if self._burst_weight > 0.0:
            burst = torch.rand(self._num_envs, device=self._device) < self._burst_weight
            self._exit_p = torch.where(
                begins,
                torch.where(
                    burst,
                    torch.full_like(self._exit_p, self._burst_exit_p),
                    self._base_exit_p,
                ),
                self._exit_p,
            )

        self._holding = held
        return held
