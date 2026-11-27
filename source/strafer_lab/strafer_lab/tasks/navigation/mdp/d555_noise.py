# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The D555's depth noise, in disparity pixels of the 640-wide stream, on the clean rendered frame.

The real sensor's depth error is not independent per pixel. Four terms make it up, each driven by the clean render's
geometry: the incidence cosine ``cos φ`` from the back-projected surface normals, and the disparity gradient
``G = |∇d|`` with ``d = f·B / z``.

- **A temporal field T**, fresh at each step and white in time, but correlated in space: an exponential correlation
  along the rows and a box down the columns, with a white share and a lognormal scale field for its heavy tail. Its
  amplitude is ``σ_T = a_T · σ_T0 · cos φ^(−p) · (1 + r · cos 2π frac(d))``.
- **A static field S**, drawn at each reset and held for the episode in the image (or, per env, decaying with a time
  constant), streaked along the rows. Its amplitude is ``σ_S = a_S · min(σ_S0 + k · G, σ_S,max)``, plus a sub-pixel
  locking ripple ``a_S · b · sin 2π frac(d)``.
- **The disparity lattice.** The noisy disparity is rounded to 1/32 px through the measured sub-pixel occupancy (some
  sub-pixel codes never occur), each bin centred on its code so the rounding adds no bias, and the depth to the
  millimetre, as Z16 carries it.
- **Invalid blobs**, with probability ``p = p_b + p_ref · (c_ref / cos φ)^k`` clipped at ``p_max``. The base ``p_b`` is
  the surface's: ``p_base`` on a surface facing the camera or a wall, rising linearly to ``p_floor`` as the vertical
  disparity gradient ``G_v = ∂d/∂v`` reaches the floor's, plus a rim ``p_rim · exp(−e / λ)`` within 16 px of a depth
  edge (a disparity jump between neighbouring pixels), ``e`` the chessboard distance to it. They are split into
  flickering speckle (a two-scale Gaussian
  field with an AR(1) memory per step and a static share held for the episode) and static grazing dropout (a coarse
  field held for the episode), each thresholded at its own share. At the centres the speckle field is drawn fresh at
  every step around its static share: each pixel's rate is set per episode, and its draws have no memory.

A pixel at or beyond the far clamp gets no noise. The invalid blobs are added to the stereo validity's mask, and every
draw comes from the process's own generator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from strafer_shared.constants import D555_MEASURED_FX_PX, PERCEPTION_HEIGHT, PERCEPTION_WIDTH

LATTICE_OCCUPANCY_LUT = (1, 2, 2, 4, 5, 7, 8, 9, 9, 10, 11, 12, 13, 13, 14, 14,
                         16, 16, 17, 17, 18, 19, 20, 21, 23, 24, 26, 27, 29, 29, 30, 31)
"""Sub-pixel code (of 32) a disparity's 1/32 px bin maps to, the bin centred on its own code: the inverse of the unit's
measured occupancy."""


@dataclass(frozen=True)
class D555NoiseCfg:
    """The noise's parameters: the model's centres, fitted to one unit, one glossy floor and static poses, and the
    per-env spread a tier draws around them."""

    focal_px: float = D555_MEASURED_FX_PX
    """Focal length of the 640-wide stream, in px (the back-projection for the normals)."""

    principal_point_px: tuple[float, float] = (PERCEPTION_WIDTH / 2.0, PERCEPTION_HEIGHT / 2.0)
    """The render's principal point (u, v), in px."""

    cos_min: float = 0.05
    """Floor on the incidence cosine."""

    grad_max: float = 1.0
    """Ceiling on the disparity gradient, px per px."""

    t_sigma0_px: float = 0.0574
    """T's amplitude at normal incidence, in px."""

    t_angle_exponent: float = 0.269
    """T's growth with incidence: σ_T ∝ cos φ^(−p)."""

    t_phase_ripple: float = 0.237
    """T's relative swing with the disparity's sub-pixel phase."""

    t_length_x_px: float = 10.0
    """T's exponential correlation length along the rows, in px."""

    t_width_y_px: int = 7
    """T's box width down the columns, in px."""

    t_white_share: float = 0.14
    """T's share of variance that is white in space."""

    t_tail_alpha: float = 0.30
    """T's lognormal scale field's log-sigma (0 for a Gaussian T)."""

    t_tail_sigma_px: float = 16.0
    """The scale field's correlation (Gaussian sigma), in px."""

    s_sigma0_px: float = 0.022
    """S's amplitude on a surface with no disparity gradient, in px."""

    s_slope_gain: float = 0.64
    """S's growth with the disparity gradient, px per (px per px)."""

    s_sigma_max_px: float = 0.60
    """S's amplitude ceiling, in px."""

    s_length_x_px: float = 30.4
    """S's exponential correlation length along the rows, in px."""

    s_length_y_px: float = 4.25
    """S's exponential correlation length down the columns, in px."""

    s_phase_bias_px: float = 0.056
    """S's sub-pixel locking ripple, in px."""

    lattice_lut: tuple[int, ...] = LATTICE_OCCUPANCY_LUT
    """Sub-pixel code per fractional bin; an empty tuple rounds plainly to 1/32 px."""

    round_to_mm: bool = True
    """Round the depth to the millimetre, as Z16 does."""

    hole_p_base: float = 0.0
    """The invalid share off the floor: a surface facing the camera, a wall."""

    hole_p_floor: float = 0.0214
    """The invalid share on the floor, away from its edges."""

    hole_floor_grad_px: float = 0.17
    """The vertical disparity gradient, in px per row, from which a surface takes ``hole_p_floor`` in full; the share
    rises linearly from ``hole_p_base`` at 0. Half the floor's gradient on the robot (B / h: 0.095 m over the camera's
    measured 0.279 m), so a floor seen from up to twice that height takes it in full."""

    hole_p_ref: float = 0.0
    """The grazing excess at ``hole_c_ref``: none on the open floor out to 2 m."""

    hole_rim_p0: float = 0.66
    """The invalid share a depth edge adds at its own pixels, beside the occlusion shadow the stereo validity makes."""

    hole_rim_length_px: float = 1.8
    """The rim's decay: the share falls as ``exp(−e / length)`` with the chessboard distance ``e`` to the edge, in px."""

    hole_edge_jump_px: float = 2.0
    """The disparity jump between neighbouring pixels, in px, that makes a depth edge."""

    hole_k_graze: float = 3.70
    """The grazing law's exponent."""

    hole_c_ref: float = 0.125
    """The incidence cosine the grazing excess is referred to."""

    hole_p_max: float = 0.97
    """The invalid share's ceiling."""

    hole_f_speckle: float = 0.3
    """Share of the grazing excess that flickers as speckle; the rest is static dropout."""

    hole_p_speckle_max: float = 0.08
    """Ceiling on the speckle share the grazing excess adds; a base share above it (a rim's) flickers in full."""

    hole_m_short: float = 0.5
    """Speckle field: the short scale's share of variance."""

    hole_sigma_long_px: float = 12.0
    """Speckle field: the long scale's Gaussian sigma, in px."""

    hole_sigma_short_px: float = 0.8
    """Speckle field: the short scale's Gaussian sigma, in px."""

    hole_aniso: float = 2.5
    """Speckle field: the blobs' horizontal over vertical extent."""

    hole_rho: float = 0.0
    """Speckle field: the per-step AR(1) memory."""

    hole_m_static: float = 0.876
    """Speckle field: the share held for the episode, the spread of the pixels' rates."""

    hole_coarse_px: int = 4
    """The grid the long-scale fields are drawn on, in px."""

    graze_sigma_px: float = 20.0
    """Grazing dropout field's Gaussian sigma, in px."""

    graze_rho: float = 1.0
    """Grazing dropout field's per-step memory (1: held for the episode)."""

    amp_t_range: tuple[float, float] = (1.0, 1.0)
    """Per-env scale on T, drawn log-uniformly at each reset."""

    amp_s_range: tuple[float, float] = (1.0, 1.0)
    """Per-env scale on S, drawn log-uniformly at each reset."""

    clean_fraction: float = 0.0
    """Share of envs drawn with T and S at zero."""

    s_static_probability: float = 1.0
    """Share of envs whose S is held in the image for the episode; the rest decay with a time constant."""

    s_tau_range_s: tuple[float, float] = (1.0, 10.0)
    """The time constant of a decaying S, in s, drawn uniformly per env."""

    patch_share_range: tuple[float, float] = (0.0, 0.0)
    """Share of pixels in a per-episode invalid patch field, drawn uniformly per env."""

    patch_zero_probability: float = 0.0
    """Probability an env draws no patch field."""

    patch_sigma_px: float = 20.0
    """The patch field's Gaussian sigma, in px."""


def surface_geometry(depth: torch.Tensor, fb: torch.Tensor, cfg: D555NoiseCfg):
    """Disparity, incidence cosine, disparity gradient and its signed vertical part (positive where the disparity grows
    down the image, as on a floor) of ``(N, H, W)`` clean depth; ``fb`` per env (``(N,)``)."""
    height, width = depth.shape[-2:]
    cx, cy = cfg.principal_point_px
    u = (torch.arange(width, dtype=depth.dtype, device=depth.device) - cx) / cfg.focal_px
    v = (torch.arange(height, dtype=depth.dtype, device=depth.device) - cy) / cfg.focal_px
    x = depth * u
    y = depth * v[:, None]
    dx_v, dx_u = torch.gradient(x, dim=(-2, -1))
    dy_v, dy_u = torch.gradient(y, dim=(-2, -1))
    dz_v, dz_u = torch.gradient(depth, dim=(-2, -1))
    nx = dy_u * dz_v - dz_u * dy_v
    ny = dz_u * dx_v - dx_u * dz_v
    nz = dx_u * dy_v - dy_u * dx_v
    norm = torch.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    ray = torch.sqrt(x * x + y * y + depth * depth)
    cos_phi = (nx * x + ny * y + nz * depth).abs() / (norm * ray)
    disparity = fb.reshape(-1, 1, 1) / depth.clamp(min=1e-3)
    gd_v, gd_u = torch.gradient(disparity, dim=(-2, -1))
    return disparity, cos_phi, torch.sqrt(gd_u * gd_u + gd_v * gd_v), gd_v


def _conv_rows(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Valid 1-D convolution of ``(N, H, W)`` along W with ``kernel``."""
    n, h, w = x.shape
    out = F.conv1d(x.reshape(n * h, 1, w), kernel.flip(0).reshape(1, 1, -1))
    return out.reshape(n, h, -1)


def _conv_cols(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Valid 1-D convolution of ``(N, H, W)`` along H with ``kernel``."""
    return _conv_rows(x.transpose(1, 2), kernel).transpose(1, 2)


def _exp_kernel(length: float, device, dtype) -> torch.Tensor:
    """Taps of a one-sided AR(1) of correlation length ``length`` (exponential correlation), unit output variance."""
    r = math.exp(-1.0 / length)
    taps = torch.arange(int(math.ceil(5.0 * length)) + 1, device=device, dtype=dtype)
    return math.sqrt(1.0 - r * r) * r ** taps


def _gauss_kernel(sigma: float, device, dtype) -> torch.Tensor:
    radius = int(4.0 * sigma + 0.5)
    taps = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (taps / sigma) ** 2)
    return k / k.sum()


_RIM_REACH_PX = 16
"""How far from a depth edge the rim reaches, in px."""


def edge_distance(disparity: torch.Tensor, jump_px: float) -> torch.Tensor:
    """Chessboard distance, in px, from each pixel of ``(N, H, W)`` disparity to the nearest depth edge: both pixels of
    a jump above ``jump_px`` between 4-neighbours. Beyond the rim's reach it reads ``_RIM_REACH_PX + 1``."""
    edge = torch.zeros_like(disparity, dtype=torch.bool)
    jh = (disparity[..., :, 1:] - disparity[..., :, :-1]).abs() > jump_px
    jv = (disparity[..., 1:, :] - disparity[..., :-1, :]).abs() > jump_px
    edge[..., :, 1:] |= jh
    edge[..., :, :-1] |= jh
    edge[..., 1:, :] |= jv
    edge[..., :-1, :] |= jv
    dist = torch.where(edge, 0.0, float(_RIM_REACH_PX + 1)).to(disparity.dtype)
    grown = edge[:, None].to(disparity.dtype)
    for k in range(1, _RIM_REACH_PX + 1):
        grown = F.max_pool2d(grown, 3, stride=1, padding=1)
        dist = torch.where((grown[:, 0] > 0) & (dist > k), float(k), dist)
    return dist


def _unit(x: torch.Tensor) -> torch.Tensor:
    sd = x.flatten(1).std(dim=1).clamp(min=1e-12)
    return x / sd.reshape(-1, 1, 1)


class D555NoiseProcess:
    """Per-env noise state: the per-env draws, S and the invalid fields, all from the process's own generator."""

    def __init__(self, num_envs: int, device, cfg: D555NoiseCfg, step_dt: float, seed: int, shape=None):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.step_dt = float(step_dt)
        self.shape = tuple(shape or (PERCEPTION_HEIGHT, PERCEPTION_WIDTH))
        self._g = torch.Generator(device=device)
        self._g.manual_seed(int(seed))
        h, w = self.shape
        z = lambda: torch.zeros(num_envs, h, w, device=device)  # noqa: E731
        self.amp_t = torch.ones(num_envs, device=device)
        self.amp_s = torch.ones(num_envs, device=device)
        self.s_rho = torch.ones(num_envs, device=device)
        self.s_unit = z()
        self.hole_long, self.hole_short, self.hole_static, self.graze = z(), z(), z(), z()
        self.patch = torch.zeros(num_envs, h, w, dtype=torch.bool, device=device)
        self._lut = (torch.tensor(cfg.lattice_lut, dtype=torch.float32, device=device) / 32.0
                     if cfg.lattice_lut else None)

    # --- unit fields -------------------------------------------------------------------------------------------------
    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self._g, device=self.device)

    def _rand(self, n: int) -> torch.Tensor:
        return torch.rand(n, generator=self._g, device=self.device)

    def _exp_field(self, n: int, length_x: float, length_y: float) -> torch.Tensor:
        h, w = self.shape
        kx = _exp_kernel(length_x, self.device, torch.float32)
        ky = _exp_kernel(length_y, self.device, torch.float32)
        noise = self._randn(n, h + len(ky) - 1, w + len(kx) - 1)
        return _conv_cols(_conv_rows(noise, kx), ky)

    def _expbox_field(self, n: int) -> torch.Tensor:
        h, w = self.shape
        c = self.cfg
        kx = _exp_kernel(c.t_length_x_px, self.device, torch.float32)
        wy = int(c.t_width_y_px)
        box = torch.full((wy,), 1.0 / math.sqrt(wy), device=self.device)
        noise = self._randn(n, h + wy - 1, w + len(kx) - 1)
        return _conv_cols(_conv_rows(noise, kx), box)

    def _coarse(self, n: int, sigma_y: float, sigma_x: float, grid_shape) -> torch.Tensor:
        """A smooth field on the coarse grid, Gaussian-smoothed (sigmas in coarse cells), bilinearly up to the frame."""
        gh, gw = grid_shape
        ky = _gauss_kernel(max(sigma_y, 0.3), self.device, torch.float32)
        kx = _gauss_kernel(max(sigma_x, 0.3), self.device, torch.float32)
        noise = self._randn(n, gh + len(ky) - 1, gw + len(kx) - 1)
        return _conv_cols(_conv_rows(noise, kx), ky)

    def _up(self, coarse: torch.Tensor) -> torch.Tensor:
        h, w = self.shape
        cs = int(self.cfg.hole_coarse_px)
        up = F.interpolate(coarse[:, None], size=(coarse.shape[1] * cs, coarse.shape[2] * cs), mode="bilinear",
                           align_corners=True)[:, 0]
        return up[:, :h, :w]

    def _hole_long(self, n: int) -> torch.Tensor:
        c, (h, w) = self.cfg, self.shape
        cs, a = int(c.hole_coarse_px), math.sqrt(c.hole_aniso)
        coarse = self._coarse(n, c.hole_sigma_long_px / cs / a, c.hole_sigma_long_px / cs * a, (h // cs, w // cs))
        return _unit(self._up(coarse))

    def _hole_short(self, n: int) -> torch.Tensor:
        c, (h, w) = self.cfg, self.shape
        if c.hole_sigma_short_px <= 0:
            return self._randn(n, h, w)
        a = math.sqrt(c.hole_aniso)
        ky = _gauss_kernel(c.hole_sigma_short_px / a, self.device, torch.float32)
        kx = _gauss_kernel(c.hole_sigma_short_px * a, self.device, torch.float32)
        noise = self._randn(n, h + len(ky) - 1, w + len(kx) - 1)
        return _unit(_conv_cols(_conv_rows(noise, kx), ky))

    def _smooth_unit(self, n: int, sigma_px: float) -> torch.Tensor:
        """A unit field at ``sigma_px`` px, drawn on the coarse grid and normalised there."""
        h, w = self.shape
        cs = int(self.cfg.hole_coarse_px)
        coarse = self._coarse(n, sigma_px / cs, sigma_px / cs, (h // cs + 1, w // cs + 1))
        return self._up(_unit(coarse))

    # --- per-env state -----------------------------------------------------------------------------------------------
    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Redraw the reset envs' amplitudes, S's memory, S, the invalid fields and the patch field."""
        c = self.cfg
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
               else torch.as_tensor(env_ids, dtype=torch.long, device=self.device))
        n = len(ids)
        if n == 0:
            return

        def log_uniform(band):
            lo, hi = (float(x) for x in band)
            if lo == hi:
                return torch.full((n,), lo, device=self.device)
            return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * self._rand(n))

        amp_t, amp_s = log_uniform(c.amp_t_range), log_uniform(c.amp_s_range)
        if c.clean_fraction > 0:
            clean = self._rand(n) < c.clean_fraction
            amp_t, amp_s = amp_t.masked_fill(clean, 0.0), amp_s.masked_fill(clean, 0.0)
        self.amp_t[ids], self.amp_s[ids] = amp_t, amp_s
        if c.s_static_probability < 1.0:
            lo, hi = c.s_tau_range_s
            tau = lo + (hi - lo) * self._rand(n)
            decaying = self._rand(n) >= c.s_static_probability
            self.s_rho[ids] = torch.where(decaying, torch.exp(-self.step_dt / tau), torch.ones_like(tau))
        self.s_unit[ids] = self._exp_field(n, c.s_length_x_px, c.s_length_y_px)
        self.hole_long[ids] = self._hole_long(n)
        self.hole_short[ids] = self._hole_short(n)
        if c.hole_m_static > 0:
            self.hole_static[ids] = (math.sqrt(1 - c.hole_m_short) * self._hole_long(n)
                                     + math.sqrt(c.hole_m_short) * self._hole_short(n))
        self.graze[ids] = self._smooth_unit(n, c.graze_sigma_px)
        lo, hi = c.patch_share_range
        if hi > 0:
            share = lo + (hi - lo) * self._rand(n)
            share = torch.where(self._rand(n) < c.patch_zero_probability, torch.zeros_like(share), share)
            field = self._smooth_unit(n, c.patch_sigma_px)
            ordered = field.flatten(1).sort(dim=1).values
            rank = ((1.0 - share) * (ordered.shape[1] - 1)).long().clamp(0, ordered.shape[1] - 1)
            cut = ordered.gather(1, rank[:, None])
            self.patch[ids] = (field > cut.reshape(-1, 1, 1)) & (share > 0).reshape(-1, 1, 1)

    # --- one frame ---------------------------------------------------------------------------------------------------
    def _quantise(self, disparity: torch.Tensor) -> torch.Tensor:
        if self._lut is None:
            return torch.round(disparity * 32.0) / 32.0
        nearest = torch.floor(disparity * 32.0 + 0.5)  # the bin centred on each 1/32 px code
        whole = torch.floor(nearest / 32.0)
        return whole + self._lut[(nearest - 32.0 * whole).long()]

    def step(self, depth: torch.Tensor, fb: torch.Tensor, max_depth: float) -> tuple[torch.Tensor, torch.Tensor]:
        """The noisy depth and the invalid blobs of one ``(N, H, W)`` clean frame; a pixel with no return reads
        ``max_depth``."""
        c = self.cfg
        n = self.num_envs
        # no return (non-finite, not positive) reads the far clamp, which takes no noise
        depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, max_depth)
        disparity, cos_phi, grad, grad_v = surface_geometry(depth, fb, c)
        cos_c = cos_phi.clamp(c.cos_min, 1.0)
        grad_c = grad.clamp(0.0, c.grad_max)
        phase = 2.0 * math.pi * (disparity - torch.floor(disparity))
        amp_t, amp_s = self.amp_t.reshape(-1, 1, 1), self.amp_s.reshape(-1, 1, 1)

        # T, fresh each step
        sigma_t = amp_t * c.t_sigma0_px * cos_c ** (-c.t_angle_exponent) * (1.0 + c.t_phase_ripple * torch.cos(phase))
        t_unit = self._expbox_field(n)
        if c.t_white_share > 0:
            t_unit = math.sqrt(1 - c.t_white_share) * t_unit + math.sqrt(c.t_white_share) * self._randn(n, *self.shape)
        if c.t_tail_alpha > 0:
            al = c.t_tail_alpha
            t_unit = t_unit * torch.exp(al * self._smooth_unit(n, c.t_tail_sigma_px) - al * al)

        # S, held; a decaying env's S moves on by its AR(1)
        decaying = (self.s_rho < 1.0).nonzero().flatten()
        if len(decaying):
            rho = self.s_rho[decaying].reshape(-1, 1, 1)
            fresh = self._exp_field(len(decaying), c.s_length_x_px, c.s_length_y_px)
            self.s_unit[decaying] = rho * self.s_unit[decaying] + torch.sqrt(1 - rho * rho) * fresh
        sigma_s = amp_s * torch.clamp(c.s_sigma0_px + c.s_slope_gain * grad_c, max=c.s_sigma_max_px)
        static = sigma_s * self.s_unit + amp_s * c.s_phase_bias_px * torch.sin(phase)

        noisy = self._quantise(disparity + static + sigma_t * t_unit).clamp(min=1.0 / 32.0)
        z = fb.reshape(-1, 1, 1) / noisy
        if c.round_to_mm:
            z = torch.round(z * 1000.0) / 1000.0
        z = torch.where(depth >= max_depth, depth, z)  # at or beyond the far clamp the frame reads the clamp

        # invalid blobs: flickering speckle, static grazing dropout, the patch field
        floor = (grad_v / c.hole_floor_grad_px).clamp(0.0, 1.0)
        base = c.hole_p_base + (c.hole_p_floor - c.hole_p_base) * floor
        if c.hole_rim_p0 > 0:
            e = edge_distance(disparity, c.hole_edge_jump_px)
            base = base + torch.where(e <= _RIM_REACH_PX, c.hole_rim_p0 * torch.exp(-e / c.hole_rim_length_px), 0.0)
        prob = (base + c.hole_p_ref * (c.hole_c_ref / cos_c) ** c.hole_k_graze).clamp(1e-6, c.hole_p_max)
        # the base share (the surface's, the rim's) flickers in full; the ceiling holds the grazing excess's part
        p_speckle = torch.minimum(base + c.hole_f_speckle * (prob - base), base.clamp(min=c.hole_p_speckle_max))
        p_static = (prob - p_speckle).clamp(0.0, 1.0)
        inf = torch.full_like(prob, float("inf"))
        thr_speckle = torch.where(p_speckle > 1e-6, torch.special.ndtri(1.0 - p_speckle.clamp(1e-6, 1 - 1e-6)), inf)
        thr_static = torch.where(p_static > 1e-6, torch.special.ndtri(1.0 - p_static.clamp(1e-6, 1 - 1e-6)), inf)
        r = c.hole_rho
        self.hole_long = r * self.hole_long + math.sqrt(1 - r * r) * self._hole_long(n)
        self.hole_short = r * self.hole_short + math.sqrt(1 - r * r) * self._hole_short(n)
        field = math.sqrt(1 - c.hole_m_short) * self.hole_long + math.sqrt(c.hole_m_short) * self.hole_short
        if c.hole_m_static > 0:
            field = math.sqrt(1 - c.hole_m_static) * field + math.sqrt(c.hole_m_static) * self.hole_static
        if c.graze_rho < 1.0:
            g = c.graze_rho
            self.graze = g * self.graze + math.sqrt(1 - g * g) * self._smooth_unit(n, c.graze_sigma_px)
        blobs = (field > thr_speckle) | (self.graze > thr_static) | self.patch
        return z, blobs
