"""Corridor-curvature and aperture-threading statistics for planned paths.

Two statistics over the polylines the shared grid A* emits, measured
identically on any occupancy source so procedural rooms and scanned scenes
compare like for like:

* **turn density** ``sum|theta_i| / sum|d_i|`` (rad/m) with tortuosity and the
  fraction of paths that bend — how much a corridor makes the robot turn.
* **aperture threading** — the fraction of arc length whose centreline
  clearance falls below a threshold, plus the minimum clearance.

Depth descriptors cannot see either: a generator can move every depth
statistic toward a target while emitting rooms whose planned paths are
straight lines.

Three corrections make two sources comparable, and all three are applied
here rather than left to the caller:

* **resolution** — clearance is measured to an obstacle cell's *face*, not
  its centre, so a 0.05 m grid and a 0.1 m grid do not read a systematic
  half-cell apart.
* **inflation radius** — A* hugs the inflation boundary, so raw clearance
  sits near whatever radius the source inflated by. ``excess_clearance``
  subtracts it; thresholds derived on that scale map back per source.
* **arc length** — a path over 12 m and a path over 3 m are not the same
  measurement, so summaries accept a straight-line-distance band.

A fourth difference is semantic and can only be stated: a grid rasterizing
full object footprints blocks low objects that a grid sliced at robot height
does not, which biases the footprint-rasterized source toward looking more
cluttered.

The statistics are numpy; the planner is consumed unmodified. Importing
this module reaches the planner through its package, which pulls torch in
with it, so it is Kit-free but not dependency-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from strafer_lab.tasks.navigation.path_planner import (
    InvalidEndpointError,
    NoPathError,
    plan_path,
)

BENDING_TORTUOSITY = 1.05

# The planner emits float32 waypoints, so a geometrically straight polyline
# still carries ~1e-5 rad of per-vertex heading jitter. Its corners, by
# contrast, are the tens-of-degrees kind. Anything under this reads as
# straight, which is what lets a turning *fraction* mean something.
TURN_EPSILON_RAD = 1e-3


@dataclass(frozen=True)
class PathStats:
    """One planned path's curvature and clearance profile."""

    path: np.ndarray
    arc_m: float
    straight_line_m: float
    turn_density: float
    tortuosity: float
    clearance_m: np.ndarray
    inflation_radius_m: float
    # Paths from one room are not independent draws from the generator, so the
    # group is what a confidence interval must resample.
    group: str = ""

    @property
    def min_clearance_m(self) -> float:
        return float(self.clearance_m.min())

    @property
    def min_excess_clearance_m(self) -> float:
        return float(self.clearance_m.min()) - self.inflation_radius_m


def turn_density(path: np.ndarray) -> tuple[float, float, float, float]:
    """``(turn_density, tortuosity, arc_m, straight_line_m)`` for a polyline.

    Zero-length segments are dropped before the exterior angles are taken so
    a repeated waypoint cannot inject a spurious turn.
    """
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError(f"expected (N, 2) polyline with N >= 2, got {pts.shape}")
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    keep = seg_len > 0.0
    seg, seg_len = seg[keep], seg_len[keep]
    arc = float(seg_len.sum())
    straight = float(np.hypot(*(pts[-1] - pts[0])))
    if arc <= 0.0 or len(seg) < 2:
        return 0.0, 1.0 if arc <= 0.0 else arc / max(straight, 1e-12), arc, straight
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.diff(heading)
    turn = (turn + math.pi) % (2.0 * math.pi) - math.pi
    turn = np.abs(turn)
    turn[turn < TURN_EPSILON_RAD] = 0.0
    density = float(turn.sum() / arc)
    tortuosity = arc / straight if straight > 1e-9 else float("inf")
    return density, float(tortuosity), arc, straight


def waypoint_clearance(
    path: np.ndarray,
    occupancy: np.ndarray,
    *,
    grid_res: float,
    grid_origin_xy: tuple[float, float],
) -> np.ndarray:
    """Distance from each waypoint to the nearest *uninflated* obstacle face.

    Exact continuous distance to the obstacle cell's square footprint rather
    than to its centre — that half-cell is the whole resolution correction
    between a 0.05 m and a 0.1 m grid.

    Raises:
        ValueError: the grid has no obstacles. Clearance from nothing is not a
            measurement, and letting it through as an infinity poisons every
            quantile and summary downstream instead of surfacing the bad input.
    """
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    blocked = np.argwhere(np.asarray(occupancy) != 0)
    if len(blocked) == 0:
        raise ValueError("occupancy grid has no obstacles to measure against")
    centres = (
        blocked.astype(np.float64) * grid_res
        + np.asarray(grid_origin_xy, dtype=np.float64)
        + grid_res / 2.0
    )
    out = np.empty(len(pts), dtype=np.float64)
    chunk = max(1, int(4e6 // max(len(centres), 1)))
    for lo in range(0, len(pts), chunk):
        hi = min(lo + chunk, len(pts))
        d = pts[lo:hi, None, :] - centres[None, :, :]
        np.abs(d, out=d)
        np.subtract(d, grid_res / 2.0, out=d)
        np.maximum(d, 0.0, out=d)
        out[lo:hi] = np.hypot(d[:, :, 0], d[:, :, 1]).min(axis=1)
    return out


def arc_fraction_below(
    path: np.ndarray, clearance: np.ndarray, threshold: float
) -> float:
    """Fraction of arc length whose clearance is under ``threshold``.

    Each segment is weighted by its length and scored by its midpoint, so the
    answer does not depend on how densely the polyline was resampled.
    """
    pts = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(clearance, dtype=np.float64)
    seg_len = np.hypot(*np.diff(pts, axis=0).T)
    total = float(seg_len.sum())
    if total <= 0.0:
        return 0.0
    mid = 0.5 * (c[:-1] + c[1:])
    return float(seg_len[mid < threshold].sum() / total)


def path_statistics(
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    free_space: np.ndarray,
    occupancy: np.ndarray,
    *,
    grid_res: float,
    grid_origin_xy: tuple[float, float],
    inflation_radius_m: float,
    discretization_m: float = 0.05,
    group: str = "",
) -> PathStats:
    """Plan one path and measure it. Planner exceptions propagate."""
    path = plan_path(
        np.asarray(start_xy, dtype=np.float64),
        np.asarray(goal_xy, dtype=np.float64),
        free_space,
        grid_res=grid_res,
        grid_origin_xy=grid_origin_xy,
        discretization_m=discretization_m,
    )
    density, tort, arc, straight = turn_density(path)
    clearance = waypoint_clearance(
        path, occupancy, grid_res=grid_res, grid_origin_xy=grid_origin_xy
    )
    return PathStats(
        path=path,
        arc_m=arc,
        straight_line_m=straight,
        turn_density=density,
        tortuosity=tort,
        clearance_m=clearance,
        inflation_radius_m=float(inflation_radius_m),
        group=group,
    )


def sample_endpoint_pairs(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    n_pairs: int,
    min_separation_m: float,
    max_attempts_per_pair: int = 200,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random endpoint pairs at least ``min_separation_m`` apart.

    Takes an explicit ``Generator`` because the callers run beside a
    generator whose own RNG consumption is a frozen contract.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    if len(pts) < 2:
        return pairs
    for _ in range(n_pairs):
        for _ in range(max_attempts_per_pair):
            i, j = rng.integers(0, len(pts), size=2)
            if i == j:
                continue
            if float(np.hypot(*(pts[j] - pts[i]))) < min_separation_m:
                continue
            pairs.append((pts[i], pts[j]))
            break
    return pairs


def plan_over_pairs(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    free_space: np.ndarray,
    occupancy: np.ndarray,
    *,
    grid_res: float,
    grid_origin_xy: tuple[float, float],
    inflation_radius_m: float,
    discretization_m: float = 0.05,
    group: str = "",
) -> tuple[list[PathStats], dict[str, int]]:
    """Measure every pair that plans; count the ones that do not by reason.

    The failures are reported rather than dropped silently: an unplannable
    pair is exactly the hard-topology case the statistics exist to find, so a
    silent drop biases the result toward easy scenes.
    """
    stats: list[PathStats] = []
    failures = {"no_path": 0, "invalid_endpoint": 0, "no_obstacles": 0}
    pairs = list(pairs)
    if not np.any(np.asarray(occupancy) != 0):
        # Open-field difficulties really do generate these, and a clearance
        # measured against nothing is not one — count the grid out rather than
        # abort the sweep it appears in.
        failures["no_obstacles"] = len(pairs)
        return stats, failures
    for start, goal in pairs:
        try:
            stats.append(
                path_statistics(
                    start, goal, free_space, occupancy,
                    grid_res=grid_res,
                    grid_origin_xy=grid_origin_xy,
                    inflation_radius_m=inflation_radius_m,
                    discretization_m=discretization_m,
                    group=group,
                )
            )
        except NoPathError:
            failures["no_path"] += 1
        except InvalidEndpointError:
            failures["invalid_endpoint"] += 1
    return stats, failures


def excess_clearance(
    stats: Sequence[PathStats],
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled clearance on the inflation-free scale, with its arc weights.

    Every sample carries the arc length it represents, so the distribution is
    over *path length* rather than over waypoints — the two differ as soon as
    paths of unequal length are pooled.
    """
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for s in stats:
        seg_len = np.hypot(*np.diff(np.asarray(s.path, dtype=np.float64), axis=0).T)
        if seg_len.sum() <= 0.0:
            continue
        mid = 0.5 * (s.clearance_m[:-1] + s.clearance_m[1:])
        values.append(mid - s.inflation_radius_m)
        weights.append(seg_len)
    if not values:
        return np.empty(0), np.empty(0)
    return np.concatenate(values), np.concatenate(weights)


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float
) -> float:
    """Empirical weighted quantile — a step function, never interpolated.

    Interpolating between two widely separated clearance values invents a
    corridor width that no part of the path has.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(v) == 0:
        raise ValueError("no samples")
    if len(v) != len(w):
        raise ValueError(f"{len(v)} values against {len(w)} weights")
    if not w.sum() > 0.0:
        raise ValueError("weights sum to zero")
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w) / w.sum()
    return float(v[min(int(np.searchsorted(cum, q, side="left")), len(v) - 1)])


def threshold_from_body(
    reference: Sequence[PathStats], quantile: float = 0.5
) -> float:
    """A clearance threshold on the inflation-free scale, from a reference
    distribution's body.

    A threshold read off the tail (the tightest doorway a corpus happens to
    contain) reads flat across everything above it, which is useless for
    gating a lever that widens corridors rather than narrowing them.
    """
    values, weights = excess_clearance(reference)
    if len(values) == 0:
        raise ValueError("reference has no measurable arc length")
    return weighted_quantile(values, weights, quantile)


def _band_filter(
    stats: Sequence[PathStats], straight_line_band: tuple[float, float] | None
) -> list[PathStats]:
    """Keep paths whose *endpoints* lie in the band.

    Straight-line separation, not arc length: banding on arc would select
    against exactly the tortuous paths the statistics exist to find.
    """
    if straight_line_band is None:
        return list(stats)
    lo, hi = straight_line_band
    return [s for s in stats if lo <= s.straight_line_m <= hi]


def summarize(
    stats: Sequence[PathStats],
    *,
    excess_thresholds: Sequence[float] = (),
    raw_thresholds: Sequence[float] = (),
    straight_line_band: tuple[float, float] | None = None,
) -> dict:
    """Aggregate a population of measured paths.

    Turn density is reported at upper percentiles as well as the median: the
    planner shortcuts line of sight unconditionally, so any population with
    open sightlines pins its median at exactly zero and only the upper tail
    carries signal.
    """
    kept = _band_filter(stats, straight_line_band)
    out: dict = {
        "n_paths": len(kept),
        "n_paths_before_band": len(stats),
        "straight_line_band_m": list(straight_line_band) if straight_line_band else None,
    }
    if not kept:
        return out
    density = np.array([s.turn_density for s in kept])
    tort = np.array([s.tortuosity for s in kept])
    # A path whose endpoints coincide has infinite tortuosity; it would poison
    # every percentile, so it is counted out instead.
    finite_tort = tort[np.isfinite(tort)]
    arc = np.array([s.arc_m for s in kept])
    straight = np.array([s.straight_line_m for s in kept])
    min_c = np.array([s.min_clearance_m for s in kept])
    infl = {s.inflation_radius_m for s in kept}
    out.update(
        {
            "turn_density_rad_per_m": {
                "median": float(np.median(density)),
                "p75": float(np.percentile(density, 75)),
                "p90": float(np.percentile(density, 90)),
                "p99": float(np.percentile(density, 99)),
                "mean": float(density.mean()),
            },
            "tortuosity": {
                "median": float(np.median(finite_tort)) if len(finite_tort) else None,
                "p75": float(np.percentile(finite_tort, 75)) if len(finite_tort) else None,
                "p90": float(np.percentile(finite_tort, 90)) if len(finite_tort) else None,
                "n_closed_loops": int(len(tort) - len(finite_tort)),
            },
            "frac_paths_bending": float(np.mean(tort > BENDING_TORTUOSITY)),
            # A population that is mostly line-of-sight pins every percentile
            # below its turning fraction at zero, so the fraction is the only
            # reading with signal there.
            "frac_paths_turning": float(np.mean(density > 0.0)),
            "arc_m": {
                "median": float(np.median(arc)),
                "mean": float(arc.mean()),
            },
            "straight_line_m": {
                "median": float(np.median(straight)),
                "p10": float(np.percentile(straight, 10)),
                "p90": float(np.percentile(straight, 90)),
            },
            "min_clearance_m": {
                "median": float(np.median(min_c)),
                "p10": float(np.percentile(min_c, 10)),
                "min": float(min_c.min()),
            },
            "inflation_radius_m": sorted(infl),
        }
    )
    values, weights = excess_clearance(kept)
    if len(values):
        out["excess_clearance_m"] = {
            f"p{int(q * 100)}" if q != 0.5 else "median":
                weighted_quantile(values, weights, q)
            for q in (0.10, 0.25, 0.50, 0.75)
        }
    for tau in excess_thresholds:
        fracs = [
            arc_fraction_below(s.path, s.clearance_m, tau + s.inflation_radius_m)
            for s in kept
        ]
        out.setdefault("arc_below_excess", {})[_threshold_key(tau)] = float(np.mean(fracs))
    for tau in raw_thresholds:
        fracs = [arc_fraction_below(s.path, s.clearance_m, tau) for s in kept]
        out.setdefault("arc_below_raw", {})[_threshold_key(tau)] = float(np.mean(fracs))
    return out


def _threshold_key(tau: float) -> str:
    """Dict key for a threshold. Enough precision that two thresholds read off
    neighbouring quantiles of the same clearance atom do not collide."""
    return f"{tau:.6f}"


_GATE_KEYS = ("frac_paths_turning", "frac_paths_bending", "turn_density_p90",
              "turn_density_median", "min_clearance_median")


def _gate_scalars(density, tort, min_c, arc_below):
    return {
        "frac_paths_turning": float(np.mean(density > 0.0)),
        "frac_paths_bending": float(np.mean(tort > BENDING_TORTUOSITY)),
        "turn_density_p90": float(np.percentile(density, 90)),
        "turn_density_median": float(np.median(density)),
        "min_clearance_median": float(np.median(min_c)),
        **{f"arc_below_{k}": float(v.mean()) for k, v in arc_below.items()},
    }


def bootstrap_gates(
    stats: Sequence[PathStats],
    *,
    excess_thresholds: Sequence[float] = (),
    raw_thresholds: Sequence[float] = (),
    straight_line_band: tuple[float, float] | None = None,
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> dict:
    """Confidence intervals for the gate readings, resampling *groups*.

    Six paths planned inside one room share that room's layout, so resampling
    paths would report an interval several times tighter than the generator's
    real spread. Groups are the unit of replication; a population with no
    group labels falls back to one group per path.
    """
    kept = _band_filter(stats, straight_line_band)
    if not kept:
        return {"n_paths": 0, "n_groups": 0}
    density = np.array([s.turn_density for s in kept])
    tort = np.array([s.tortuosity for s in kept])
    min_c = np.array([s.min_clearance_m for s in kept])
    arc_below = {}
    for tau in excess_thresholds:
        arc_below["excess_" + _threshold_key(tau)] = np.array([
            arc_fraction_below(s.path, s.clearance_m, tau + s.inflation_radius_m)
            for s in kept
        ])
    for tau in raw_thresholds:
        arc_below["raw_" + _threshold_key(tau)] = np.array([
            arc_fraction_below(s.path, s.clearance_m, tau) for s in kept
        ])

    labels = [s.group or f"__path{i}" for i, s in enumerate(kept)]
    members: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        members.setdefault(label, []).append(i)
    groups = [np.asarray(v) for v in members.values()]

    point = _gate_scalars(density, tort, min_c, arc_below)
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(resamples) for k in point}
    for r in range(resamples):
        pick = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        sample = _gate_scalars(
            density[pick], tort[pick], min_c[pick],
            {k: v[pick] for k, v in arc_below.items()},
        )
        for k, v in sample.items():
            draws[k][r] = v
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    out = {"n_paths": len(kept), "n_groups": len(groups), "level": level}
    # One group resamples to itself every time, so the interval it would report
    # is zero-width — an artifact, not a measurement of anything.
    single = len(groups) < 2
    for k, v in point.items():
        out[k] = {
            "value": v,
            "ci_lo": None if single else float(np.quantile(draws[k], lo_q)),
            "ci_hi": None if single else float(np.quantile(draws[k], hi_q)),
        }
    return out
