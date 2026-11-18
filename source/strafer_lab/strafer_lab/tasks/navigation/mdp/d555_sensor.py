# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The D555's stereo validity at render resolution: which pixels the sensor returns no depth for.

The D555 matches its left imager against the right one, and the depth image is the left imager's view. A pixel has no
depth where its match would fall outside the right image, the left band ``u < f·B / z(u) + m``, and where the right
imager cannot see it, behind a nearer surface to its right: the right-image column ``r = u - f·B / z`` of some pixel to
its right lies left of its own (a pixel's shadow sits on the left of the surface that casts it). It also returns none
for a surface nearer than its matcher's disparity range reaches, nor where nothing returns at all.

The deploy node writes every invalid pixel as the far clamp before the 8×8 block median, so a block with 33 or more of
64 invalid pixels reads far. :func:`sensor_frame` does the same to a rendered frame, ahead of the shared reduction
(``observations.reduce_depth_to_policy_grid``), so the training field reads what the node's would.

The per-env state is a scale on ``f·B`` drawn at each reset from the process's own generator, seeded from the env's
seed, so no other term's random stream moves. With a noise cfg (``mdp.d555_noise``) the frame also carries the sensor's
correlated depth noise, its lattice and its invalid blobs, drawn from a second generator, so the ``f·B`` draws are the
same with the noise on or off.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from strafer_shared.constants import D555_MEASURED_FX_PX, D555_RENDER_FAR_CLIP_M, DEPTH_MAX

from .d555_noise import D555NoiseCfg, D555NoiseProcess


@dataclass(frozen=True)
class D555SensorCfg:
    """The stereo-validity mask's parameters."""

    fb_px_m: float = D555_MEASURED_FX_PX * 0.095
    """Focal length (px, at the render resolution) times the stereo baseline (m)."""

    fb_scale_range: tuple[float, float] = (1.0, 1.0)
    """Per-env scale on ``fb_px_m``, drawn uniformly at each reset."""

    margin_px: float = 6.5
    """Width of the left band beyond the disparity, in px."""

    core_px: float = 0.0
    """A fixed width the left band never falls below, in px."""

    min_depth_m: float = 0.244
    """Depth below which the sensor returns none, in m."""

    occlusion_tol_px: float = 0.5
    """Right-image overlap, in px, beyond which a pixel is occluded."""

    invalid_depth_m: float = DEPTH_MAX
    """Depth written at an invalid pixel, the deploy node's far clamp."""

    max_return_m: float = D555_RENDER_FAR_CLIP_M
    """Depth beyond which nothing returns, the render's far clip."""


def stereo_validity(depth: torch.Tensor, fb: torch.Tensor | float, cfg: D555SensorCfg) -> torch.Tensor:
    """Pixels the sensor returns depth for, on ``(N, H, W)`` depth in metres with ``fb`` per env (``(N,)``) or one value.

    Invalid: the left band, the occlusion shadows, depth below ``min_depth_m``, and no return (non-finite, not positive,
    or beyond ``max_return_m``). A pixel with no return has zero disparity, so it casts no shadow.
    """
    width = depth.shape[-1]
    fb = torch.as_tensor(fb, dtype=depth.dtype, device=depth.device).reshape(-1, 1, 1)
    returned = torch.isfinite(depth) & (depth > 0) & (depth <= cfg.max_return_m)
    disparity = torch.where(returned, fb / depth.clamp(min=1e-3), torch.zeros_like(depth))
    u = torch.arange(width, dtype=depth.dtype, device=depth.device)
    band = (u < cfg.core_px) | (u < disparity + cfg.margin_px)
    r = u - disparity
    # The minimum of r over the pixels strictly right of each pixel
    right_min = torch.flip(torch.cummin(torch.flip(r, dims=[-1]), dim=-1).values, dims=[-1])
    beyond = torch.cat([right_min[..., 1:], torch.full_like(r[..., :1], float("inf"))], dim=-1)
    occluded = beyond < r - cfg.occlusion_tol_px
    return returned & ~band & ~occluded & (depth >= cfg.min_depth_m)


class D555SensorProcess:
    """Per-env stereo-validity state: the ``f·B`` each env's mask is computed with."""

    def __init__(self, num_envs: int, device, cfg: D555SensorCfg, seed: int = 0, noise: D555NoiseCfg | None = None,
                 step_dt: float = 1.0 / 30.0):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(seed))
        self.fb = torch.full((num_envs,), float(cfg.fb_px_m), device=device)
        self.noise = None if noise is None else D555NoiseProcess(num_envs, device, noise, step_dt, seed=int(seed) + 1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Redraw the reset envs' ``f·B`` scale; a band of one value draws nothing."""
        lo, hi = (float(x) for x in self.cfg.fb_scale_range)
        if lo == hi:
            self.fb.fill_(self.cfg.fb_px_m * lo)
        else:
            ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
                   else torch.as_tensor(env_ids, dtype=torch.long, device=self.device))
            draw = torch.rand(len(ids), generator=self._generator, device=self.device)
            self.fb[ids] = self.cfg.fb_px_m * (lo + (hi - lo) * draw)
        if self.noise is not None:
            self.noise.reset(env_ids)


def sensor_frame(depth: torch.Tensor, process: D555SensorProcess) -> tuple[torch.Tensor, torch.Tensor]:
    """The frame the sensor returns: invalid pixels written ``invalid_depth_m``, and the validity mask.

    Takes ``(N, H, W)`` or ``(N, H, W, 1)`` depth in metres and returns both in that shape.
    """
    squeeze = depth.dim() == 4
    z = depth[..., 0] if squeeze else depth
    valid = stereo_validity(z, process.fb, process.cfg)
    if process.noise is not None:
        z, blobs = process.noise.step(z, process.fb, process.cfg.invalid_depth_m)
        valid = valid & ~blobs
    out = torch.where(valid, z, torch.full_like(z, process.cfg.invalid_depth_m))
    if squeeze:
        return out.unsqueeze(-1), valid.unsqueeze(-1)
    return out, valid
