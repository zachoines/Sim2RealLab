"""Referent-frame drift for goal-shaped observations.

The deployed policy reads its referent through a SLAM map frame, and that frame
moves. The error is *integrated*, not resampled: a localization offset that is
0.1 m to the left this tick is still roughly 0.1 m to the left a tick later, and
only wanders over seconds. Independent per-tick noise would model something a
recurrent policy can average away in a few ticks, which is the opposite of what
the deployed failure looks like.

Two error classes live on this axis and both are modelled here as one process:

- A smooth wander. Each of the three SE(2) components is an Ornstein-Uhlenbeck
  process, discretised exactly so its stationary standard deviation is the
  configured sigma at any step size.
- A discontinuous snap. Accepting a loop closure moves ``map->odom`` in one
  step, which no correlated random walk produces. Modelled as a Poisson jump
  added into the same state, so a jump then relaxes on the same time constant.
  Its rate and band default to zero, because no measured closure distribution
  backs them; turning them on is a config change, not a code change.

The magnitudes are per environment and drawn at reset, so a training batch
spans a band of localization qualities rather than one point. State is zeroed
at reset because a new episode is a new path off a fresh anchor, and drift
accumulates from the anchor.

The perturbation is applied to the *observed* referent only. The command the
rewards and terminations read stays true, which is what makes an arm on this
axis separate a policy that has stopped tracking from one that is tracking a
displaced frame faithfully.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def ou_coefficients(tau_s: float, step_dt: float) -> tuple[float, float]:
    """Exact discretisation of an OU process: ``(decay, innovation)``.

    The update ``x <- decay * x + innovation * sigma * n`` holds the stationary
    standard deviation at ``sigma`` for any step size, which is what lets the
    time constant be stated in seconds while the process runs on control steps.
    """
    if tau_s <= 0.0:
        raise ValueError(f"tau_s must be positive, got {tau_s}")
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    decay = math.exp(-step_dt / tau_s)
    return decay, math.sqrt(max(0.0, 1.0 - decay**2))


def axis_sigma(position_rms_m: float) -> float:
    """Per-axis sigma of a 2-D offset stated as an RMS displacement."""
    return position_rms_m / math.sqrt(2.0)


def jump_probability(rate_hz: float, step_dt: float) -> float:
    """Per-step probability of at least one Poisson arrival."""
    if rate_hz <= 0.0:
        return 0.0
    return -math.expm1(-rate_hz * step_dt)


def drift_referent(rel_xy: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Body-frame referent vector seen through a drifted frame.

    ``rel_xy`` is ``(N, 2)`` in metres and ``offsets`` is ``(N, 3)`` of
    ``(tx, ty, dtheta)`` in metres and radians. The referent is translated and
    then rotated, so every quantity derived from it — the relative offset, the
    distance, the signed bearing — comes from one vector. Perturbing those
    independently would hand the policy an observation no geometry can produce.
    """
    if rel_xy.shape[-1] != 2:
        raise ValueError(f"rel_xy must be (N, 2), got {tuple(rel_xy.shape)}")
    if offsets.shape != (rel_xy.shape[0], 3):
        raise ValueError(
            f"offsets must be ({rel_xy.shape[0]}, 3), got {tuple(offsets.shape)}"
        )
    shifted_x = rel_xy[:, 0] + offsets[:, 0]
    shifted_y = rel_xy[:, 1] + offsets[:, 1]
    cos_d = torch.cos(offsets[:, 2])
    sin_d = torch.sin(offsets[:, 2])
    return torch.stack(
        (
            cos_d * shifted_x - sin_d * shifted_y,
            sin_d * shifted_x + cos_d * shifted_y,
        ),
        dim=-1,
    )


class SubgoalDriftProcess:
    """Per-environment SE(2) drift of the referent frame.

    Args:
        num_envs: Number of parallel environments.
        device: Torch device for the state tensors.
        step_dt: Control-step period in seconds. The time constant is stated in
            seconds, so the process needs the period to discretise itself.
        position_rms_m: RMS displacement of the 2-D offset at gain 1, in metres.
        heading_sigma_deg: Stationary heading sigma at gain 1, in degrees.
        gain_range: Per-env multiplier band ``[min, max]`` on both magnitudes,
            drawn at reset. A max of 0 leaves the wander inert. The two
            magnitudes scale together because they were measured together and
            do not separate in effect.
        tau_s: Correlation time of the wander, in seconds.
        jump_rate_hz: Poisson rate of discontinuous snaps.
        jump_position_range_m: Jump displacement band ``[min, max]``, in metres,
            applied in a uniformly random direction.
        jump_heading_range_deg: Jump heading band ``[min, max]``, in degrees,
            applied with a random sign.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        *,
        step_dt: float,
        position_rms_m: float,
        heading_sigma_deg: float,
        gain_range: tuple[float, float],
        tau_s: float,
        jump_rate_hz: float = 0.0,
        jump_position_range_m: tuple[float, float] = (0.0, 0.0),
        jump_heading_range_deg: tuple[float, float] = (0.0, 0.0),
    ):
        if position_rms_m < 0.0 or heading_sigma_deg < 0.0:
            raise ValueError("drift magnitudes must be non-negative")
        g_lo, g_hi = min(gain_range), max(gain_range)
        if g_lo < 0.0:
            raise ValueError(f"gain_range must be non-negative, got {gain_range}")

        self._num_envs = num_envs
        self._device = device
        self._gain_range = (g_lo, g_hi)
        self.tau_s = float(tau_s)
        self.step_dt = float(step_dt)
        self._decay, self._innovation = ou_coefficients(tau_s, step_dt)

        # The class is stated as an RMS displacement of the 2-D offset, so each
        # axis carries 1/sqrt(2) of it.
        self._unit_sigma = torch.tensor(
            [
                axis_sigma(position_rms_m),
                axis_sigma(position_rms_m),
                math.radians(heading_sigma_deg),
            ],
            device=device,
        )
        self._wander_enabled = g_hi > 0.0 and bool(
            (self._unit_sigma > 0.0).any().item()
        )

        j_lo, j_hi = min(jump_position_range_m), max(jump_position_range_m)
        h_lo, h_hi = min(jump_heading_range_deg), max(jump_heading_range_deg)
        if j_lo < 0.0 or h_lo < 0.0:
            raise ValueError("jump bands must be non-negative")
        self._jump_position_range = (j_lo, j_hi)
        self._jump_heading_range = (math.radians(h_lo), math.radians(h_hi))
        self._jump_p = jump_probability(jump_rate_hz, step_dt)
        self._jump_enabled = self._jump_p > 0.0 and (j_hi > 0.0 or h_hi > 0.0)

        self.enabled = self._wander_enabled or self._jump_enabled

        self._state = torch.zeros(num_envs, 3, device=device)
        self._sigma = torch.zeros(num_envs, 3, device=device)
        # Counted rather than inferred: the shipped jump band is zero, so a
        # non-zero count is the signal that the dormant class was turned on.
        self.jumps = 0
        self._last_step: int | None = None

        if self.enabled:
            self.reset(None)

    @property
    def offsets(self) -> torch.Tensor:
        """Current per-env ``(tx, ty, dtheta)``, in metres and radians."""
        return self._state

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Re-anchor the named envs and redraw their magnitudes."""
        if not self.enabled:
            return
        idx = slice(None) if env_ids is None else env_ids
        count = self._num_envs if env_ids is None else len(env_ids)
        if count == 0:
            return

        self._state[idx] = 0.0
        if self._wander_enabled:
            g_lo, g_hi = self._gain_range
            gain = g_lo + (g_hi - g_lo) * torch.rand(count, 1, device=self._device)
            self._sigma[idx] = gain * self._unit_sigma

    def advance_to(self, step_index: int) -> torch.Tensor:
        """Offsets for control step ``step_index``, advancing once per step.

        Every goal-shaped observation term reads the same offsets within a
        step, and the observation group may be computed more than once per
        step, so the advance is keyed on the step index rather than performed
        per read.
        """
        if step_index != self._last_step:
            self._last_step = step_index
            self.step()
        return self._state

    def step(self) -> torch.Tensor:
        """Advance one control step; return the per-env offsets."""
        if not self.enabled:
            return self._state

        # Relaxation first, and unconditionally: it is what pulls a jump back
        # toward the anchor, so a jump-only configuration still decays. The
        # innovation is what the wander adds, and it is the only draw here.
        self._state = self._decay * self._state
        if self._wander_enabled:
            noise = torch.randn(self._num_envs, 3, device=self._device)
            self._state = self._state + self._innovation * self._sigma * noise
        if self._jump_enabled:
            self._apply_jumps()
        return self._state

    def _apply_jumps(self) -> None:
        """Add this step's Poisson snaps into the state."""
        jumped = torch.rand(self._num_envs, device=self._device) < self._jump_p
        count = int(torch.count_nonzero(jumped))
        if count == 0:
            return
        self.jumps += count

        p_lo, p_hi = self._jump_position_range
        h_lo, h_hi = self._jump_heading_range
        magnitude = p_lo + (p_hi - p_lo) * torch.rand(count, device=self._device)
        bearing = (2.0 * math.pi) * torch.rand(count, device=self._device)
        heading = h_lo + (h_hi - h_lo) * torch.rand(count, device=self._device)
        sign = torch.where(
            torch.rand(count, device=self._device) < 0.5, -1.0, 1.0
        )

        self._state[jumped, 0] += magnitude * torch.cos(bearing)
        self._state[jumped, 1] += magnitude * torch.sin(bearing)
        self._state[jumped, 2] += sign * heading
