"""Pure depth-integrity analysis for the RTX Real-Time 2.0 renderer probe.

No Kit / torch dependency — just numpy — so the gate logic is unit-testable
without booting Isaac Sim. The Kit-side capture lives in
``scripts/probe_rt2_depth_integrity.py``; it snapshots the perception camera's
``distance_to_image_plane`` output and hands the arrays here.

Three checks back the depth-integrity gate (see the probe script's module
docstring for the full rationale):

  - :func:`check_cadence` — one depth frame per env step, uniform sim-time
    spacing (no injected / duplicated / dropped frame).
  - :func:`check_frame_diff_under_motion` — consecutive frames differ while the
    robot moves (no frozen / repeated frame).
  - :func:`depth_parity` — Real-Time 2.0 depth matches the Real-Time 1.0 render
    of the same static pose within the renderer-nondeterminism budget.
"""
from __future__ import annotations

import numpy as np

# distance_to_image_plane past the renderer far clip reads as +inf / a huge
# value; exclude those pixels from parity so a sky pixel does not dominate the
# max-abs statistic. Matches the bridge's generous render frustum far clip.
FAR_CLIP_M = 50.0


def check_cadence(sim_times, step_dt: float, *, rtol: float = 0.05) -> "tuple[bool, str]":
    """One depth frame per env step, sim time advancing uniformly by ``step_dt``.

    Fails on a degenerate frame count, non-monotonic sim time, or any
    inter-frame delta departing from ``step_dt`` by more than ``rtol`` — the
    signature of a duplicated or dropped frame.
    """
    sim_times = np.asarray(sim_times, dtype=np.float64).ravel()
    n = sim_times.size
    if n < 2:
        return False, f"need >=2 frames, got {n}"
    deltas = np.diff(sim_times)
    if np.any(deltas <= 0):
        return False, f"sim time not strictly increasing (min delta {deltas.min():.6f}s)"
    lo, hi = step_dt * (1 - rtol), step_dt * (1 + rtol)
    off = np.where((deltas < lo) | (deltas > hi))[0]
    if off.size:
        return False, (
            f"{off.size}/{deltas.size} inter-frame deltas outside "
            f"[{lo:.6f}, {hi:.6f}]s (step_dt={step_dt:.6f}s); "
            f"first bad delta {deltas[off[0]]:.6f}s at frame {off[0] + 1}"
        )
    return True, f"{n} frames, uniform {step_dt:.6f}s cadence"


def check_frame_diff_under_motion(
    depths, *, min_changed_frac: float = 0.01, eps_m: float = 1e-4
) -> "tuple[bool, str]":
    """Consecutive frames must differ while moving — no frozen / repeated frames.

    For each adjacent pair, the fraction of co-finite pixels changing by more
    than ``eps_m`` must exceed ``min_changed_frac``. A pair below the threshold
    is a repeated frame (the failure a fabricated/duplicated depth frame shows).
    """
    depths = np.asarray(depths, dtype=np.float64)
    if depths.ndim != 3 or depths.shape[0] < 2:
        return False, f"need a (N>=2, H, W) stack, got shape {depths.shape}"
    fracs = []
    for i in range(depths.shape[0] - 1):
        a, b = depths[i], depths[i + 1]
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.any():
            return False, f"frames {i},{i + 1} have no co-finite pixels"
        changed = (np.abs(a - b) > eps_m) & finite
        fracs.append(changed.sum() / finite.sum())
    fracs = np.asarray(fracs)
    worst = float(fracs.min())
    if worst < min_changed_frac:
        bad = int(fracs.argmin())
        return False, (
            f"frames {bad},{bad + 1} changed only {worst:.4f} of pixels "
            f"(< {min_changed_frac}); looks repeated/frozen"
        )
    return True, f"all {len(fracs)} consecutive pairs differ (min changed frac {worst:.4f})"


def depth_parity(
    a, b, *, max_abs_m: float = 1e-3, far_clip_m: float = FAR_CLIP_M
) -> "tuple[bool, float, str]":
    """Max abs depth difference over co-finite, in-range pixels vs ``max_abs_m``.

    Pixels that are non-finite or beyond ``far_clip_m`` in either frame are
    excluded (sky / beyond-frustum), so the statistic reflects real geometry.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False, float("inf"), f"shape mismatch {a.shape} vs {b.shape}"
    valid = np.isfinite(a) & np.isfinite(b) & (a < far_clip_m) & (b < far_clip_m)
    if not valid.any():
        return False, float("inf"), "no co-finite in-range pixels to compare"
    max_diff = float(np.abs(a[valid] - b[valid]).max())
    ok = max_diff <= max_abs_m
    return ok, max_diff, (
        f"max |delta depth| = {max_diff:.2e} m over {int(valid.sum())} pixels "
        f"({'<=' if ok else '>'} {max_abs_m:.1e} m budget)"
    )
