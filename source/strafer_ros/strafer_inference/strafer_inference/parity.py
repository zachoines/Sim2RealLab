"""Pure train<->deploy parity comparison library (rclpy-free).

Compares the inference node's assembled observation / rolling-subgoal stream
against a reference (gym-side dump, self-check re-assembly, or the node's own
published subgoal) on a shared sim-time axis. All ROS glue -- rosbag reading,
TF composition, message deserialization -- lives in the two CLIs under
``scripts/``; this module operates on already-extracted numbers so it is
unit-testable in the pxr-free suite alongside ``obs_pipeline`` and
``generator``.

The JSONL wire contract both sides emit against is documented in
``scripts/PARITY_SCHEMA.md``; the loaders here read that format.

Bounds (all derived from shared constants, never dim literals):
  - scalar obs dims: <= 1e-5 max-abs-delta (float32 assembly noise).
  - depth obs dims:  <= 1e-3 max-abs-delta (renderer nondeterminism budget).
  - rolling subgoal:  <= MAP_RESOLUTION * 2 (0.10 m) position residual.
The join tolerance is half a policy period; ticks that do not match within
it are reported as dropped, never silently discarded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from strafer_shared.constants import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    MAP_RESOLUTION,
    POLICY_PERIOD_S,
    SUBGOAL_LOOKAHEAD_M,
)
from strafer_shared.policy_interface import (
    PolicyVariant,
    assemble_observation,
)

from .generator import RollingSubgoalGenerator
from .obs_pipeline import (
    body_frame_goal,
    build_raw_obs_dict,
    downsample_depth,
    joint_state_to_wheel_vels,
)

# ---------------------------------------------------------------------------
# Bounds / tolerances (the parity contract). Kept here as the single place a
# CLI or test reads them, all derived from strafer_shared.
# ---------------------------------------------------------------------------

OBS_SCALAR_BOUND = 1e-5
OBS_DEPTH_BOUND = 1e-3
SUBGOAL_BOUND_M = MAP_RESOLUTION * 2.0  # 0.10 m
JOIN_TOL_S = POLICY_PERIOD_S / 2.0

# Self-check re-samples the bag (not the node's exact cached inputs), so a
# correctly-wired system still shows temporal-sampling deltas far above the
# strict gym-join bounds. The self-check therefore uses looser "wiring"
# tolerances: they catch gross wiring/ordering/scale bugs (which shift many
# dims by O(0.1–2)) while tolerating re-sampling noise. The --gym-dump join
# keeps the strict OBS_SCALAR_BOUND / OBS_DEPTH_BOUND. Depth is report-only in
# self-check: frame-boundary re-sampling dominates the numeric delta, so the
# depth spatial-residual structure report — not the bound — is the signal.
SELF_CHECK_SCALAR_BOUND = 0.1
SELF_CHECK_DEPTH_BOUND = 1.0
# A parity claim over a minority of ticks is not a parity claim: below this
# matched fraction the join coverage itself fails, independent of the deltas.
MIN_MATCHED_FRACTION = 0.6

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Variant dim layout helpers (no dim literals -- everything from PolicyVariant)
# ---------------------------------------------------------------------------


def dim_names(variant: PolicyVariant) -> list[str]:
    """Per-dimension names for a variant, e.g. ``imu_accel[0]`` ... so a
    failing dimension can be reported by name, not just index."""
    names: list[str] = []
    for f in variant.fields:
        if f.dims == 1:
            names.append(f.key)
        else:
            names.extend(f"{f.key}[{i}]" for i in range(f.dims))
    return names


def split_indices(
    variant: PolicyVariant,
) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
    """Return ``(scalar_indices, depth_range)`` for a variant's obs vector.

    ``scalar_indices`` is the flat index array of every non-depth dim (the
    <=1e-5 block); ``depth_range`` is the ``(start, stop)`` half-open range of
    the depth field, or ``None`` for a camera-free variant. Computed by walking
    the field layout so it holds regardless of the depth resolution in force.
    """
    scalar_idx: list[int] = []
    depth_range: Optional[tuple[int, int]] = None
    off = 0
    for f in variant.fields:
        if f.key == "depth_image":
            depth_range = (off, off + f.dims)
        else:
            scalar_idx.extend(range(off, off + f.dims))
        off += f.dims
    return np.asarray(scalar_idx, dtype=int), depth_range


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


@dataclass
class ObsStream:
    """A parsed JSONL obs stream: aligned t_sim / obs / referent arrays."""

    t_sim: np.ndarray  # (N,) float, sim seconds
    obs: np.ndarray  # (N, D) float32
    variant: PolicyVariant
    referent_xy: np.ndarray  # (N, 2) float; NaN rows where absent
    source: str = ""

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.t_sim.shape[0])


def parse_obs_records(records: list[dict], *, source: str = "") -> ObsStream:
    """Turn a list of schema dicts into an :class:`ObsStream`.

    Every record must carry the same ``variant`` and an ``obs`` of that
    variant's ``obs_dim`` length; a mismatch is a hard error (a truncated or
    mislabeled dump must not silently pass parity).
    """
    if not records:
        raise ValueError(f"empty obs stream ({source or 'unknown source'})")

    variant_names = {r["variant"] for r in records}
    if len(variant_names) != 1:
        raise ValueError(
            f"obs stream {source!r} mixes variants: {sorted(variant_names)}"
        )
    variant = PolicyVariant[next(iter(variant_names))]
    expected = variant.obs_dim

    t_sim = np.empty(len(records), dtype=np.float64)
    obs = np.empty((len(records), expected), dtype=np.float32)
    referent = np.full((len(records), 2), np.nan, dtype=np.float64)
    for i, r in enumerate(records):
        vec = np.asarray(r["obs"], dtype=np.float32)
        if vec.shape != (expected,):
            raise ValueError(
                f"{source or 'obs stream'} record {i}: obs has {vec.shape[0]} "
                f"dims, expected {expected} for {variant.name}"
            )
        t_sim[i] = float(r["t_sim"])
        obs[i] = vec
        ref = r.get("referent")
        if ref is not None:
            referent[i] = (float(ref["x"]), float(ref["y"]))
    return ObsStream(
        t_sim=t_sim, obs=obs, variant=variant, referent_xy=referent, source=source
    )


def nonmonotonic_jumps(t_sim: np.ndarray) -> int:
    """Count backward jumps in a sim-time stream. A nonzero count is a symptom
    of two runs concatenated into one dump — the sim clock resets between runs,
    which would silently contaminate the sim-time join."""
    t = np.asarray(t_sim, dtype=np.float64)
    if t.size < 2:
        return 0
    return int(np.sum(np.diff(t) < 0.0))


def load_obs_jsonl(path: str | Path, *, source: str = "") -> ObsStream:
    """Load an obs-dump JSONL file into an :class:`ObsStream`."""
    path = Path(path)
    records: list[dict] = []
    with path.open("r") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSONL: {exc}") from exc
    return parse_obs_records(records, source=source or str(path))


# ---------------------------------------------------------------------------
# Nearest-timestamp join
# ---------------------------------------------------------------------------


@dataclass
class JoinResult:
    """Outcome of a nearest-timestamp join on two sim-time axes."""

    pairs: list[tuple[int, int, float]]  # (idx_a, idx_b, dt = t_b - t_a)
    unmatched_a: list[int]
    unmatched_b: list[int]
    n_a: int
    n_b: int

    @property
    def n_matched(self) -> int:
        return len(self.pairs)

    @property
    def matched_fraction_a(self) -> float:
        return self.n_matched / self.n_a if self.n_a else 0.0

    @property
    def worst_dt(self) -> float:
        return max((abs(dt) for _, _, dt in self.pairs), default=0.0)


def nearest_join(
    t_a: np.ndarray, t_b: np.ndarray, tol_s: float = JOIN_TOL_S
) -> JoinResult:
    """Join each ``a`` tick to its nearest ``b`` tick within ``tol_s``.

    Ticks with no counterpart inside the tolerance land in ``unmatched_a`` /
    ``unmatched_b`` -- dropped frames are accounted for, not dropped from the
    denominator. Two ``a`` ticks may map to the same ``b`` (nearest-neighbour,
    not a bijection); at the policy rate this is rare and harmless for a
    per-tick delta report.
    """
    t_a = np.asarray(t_a, dtype=np.float64)
    t_b = np.asarray(t_b, dtype=np.float64)
    order = np.argsort(t_b, kind="stable")
    tb_sorted = t_b[order]

    pairs: list[tuple[int, int, float]] = []
    matched_b: set[int] = set()
    unmatched_a: list[int] = []
    for ia, ta in enumerate(t_a):
        if tb_sorted.size == 0:
            unmatched_a.append(ia)
            continue
        pos = int(np.searchsorted(tb_sorted, ta))
        cands = []
        if pos < tb_sorted.size:
            cands.append(pos)
        if pos > 0:
            cands.append(pos - 1)
        best = min(cands, key=lambda p: abs(tb_sorted[p] - ta))
        if abs(tb_sorted[best] - ta) <= tol_s:
            ib = int(order[best])
            pairs.append((ia, ib, float(tb_sorted[best] - ta)))
            matched_b.add(ib)
        else:
            unmatched_a.append(ia)
    unmatched_b = [i for i in range(t_b.size) if i not in matched_b]
    return JoinResult(
        pairs=pairs,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        n_a=int(t_a.size),
        n_b=int(t_b.size),
    )


# ---------------------------------------------------------------------------
# Obs parity
# ---------------------------------------------------------------------------


@dataclass
class ObsParityReport:
    variant: PolicyVariant
    join: JoinResult
    per_dim_max_abs: np.ndarray  # (D,) max abs delta over matched ticks
    scalar_max_abs: float
    scalar_worst_dim: int
    scalar_worst_name: str
    scalar_bound: float
    depth_max_abs: Optional[float]
    depth_worst_dim: Optional[int]
    depth_worst_name: Optional[str]
    depth_bound: Optional[float]
    min_matched_fraction: float

    @property
    def scalar_pass(self) -> bool:
        return self.scalar_max_abs <= self.scalar_bound

    @property
    def depth_pass(self) -> Optional[bool]:
        if self.depth_max_abs is None:
            return None
        return self.depth_max_abs <= self.depth_bound

    @property
    def coverage_ok(self) -> bool:
        return self.join.matched_fraction_a >= self.min_matched_fraction

    @property
    def passed(self) -> bool:
        ok = self.coverage_ok and self.scalar_pass
        if self.depth_pass is not None:
            ok = ok and self.depth_pass
        return ok


def compute_obs_parity(
    a: ObsStream,
    b: ObsStream,
    *,
    tol_s: float = JOIN_TOL_S,
    scalar_bound: float = OBS_SCALAR_BOUND,
    depth_bound: float = OBS_DEPTH_BOUND,
    min_matched_fraction: float = MIN_MATCHED_FRACTION,
) -> ObsParityReport:
    """Join two obs streams on sim time and bound the per-dim delta.

    Splits the delta into the scalar block (bounded at ``scalar_bound``) and
    the depth block (``depth_bound``); reports each separately with the worst
    offending dimension named.
    """
    if a.variant is not b.variant:
        raise ValueError(
            f"obs streams disagree on variant: {a.variant.name} ({a.source}) "
            f"vs {b.variant.name} ({b.source})"
        )
    variant = a.variant
    join = nearest_join(a.t_sim, b.t_sim, tol_s=tol_s)

    names = dim_names(variant)
    scalar_idx, depth_range = split_indices(variant)

    if join.n_matched == 0:
        per_dim = np.full(variant.obs_dim, np.nan, dtype=np.float64)
        scalar_max = float("inf")
        scalar_worst = int(scalar_idx[0])
    else:
        da = a.obs[[ia for ia, _, _ in join.pairs]]
        db = b.obs[[ib for _, ib, _ in join.pairs]]
        per_dim = np.max(np.abs(da.astype(np.float64) - db.astype(np.float64)), axis=0)
        scalar_vals = per_dim[scalar_idx]
        local = int(np.argmax(scalar_vals))
        scalar_worst = int(scalar_idx[local])
        scalar_max = float(scalar_vals[local])

    depth_max: Optional[float] = None
    depth_worst: Optional[int] = None
    depth_worst_name: Optional[str] = None
    used_depth_bound: Optional[float] = None
    if depth_range is not None:
        start, stop = depth_range
        used_depth_bound = depth_bound
        if join.n_matched == 0:
            depth_max = float("inf")
            depth_worst = start
        else:
            depth_vals = per_dim[start:stop]
            dlocal = int(np.argmax(depth_vals))
            depth_worst = start + dlocal
            depth_max = float(depth_vals[dlocal])
        depth_worst_name = names[depth_worst]

    return ObsParityReport(
        variant=variant,
        join=join,
        per_dim_max_abs=per_dim,
        scalar_max_abs=scalar_max,
        scalar_worst_dim=scalar_worst,
        scalar_worst_name=names[scalar_worst],
        scalar_bound=scalar_bound,
        depth_max_abs=depth_max,
        depth_worst_dim=depth_worst,
        depth_worst_name=depth_worst_name,
        depth_bound=used_depth_bound,
        min_matched_fraction=min_matched_fraction,
    )


# ---------------------------------------------------------------------------
# Depth spatial-residual report (row-structured geometry vs unstructured lag)
# ---------------------------------------------------------------------------

_STRUCT_THRESH = 0.5  # per-row/col std as a fraction of the mean residual
_TIME_THRESH = 0.5  # per-tick residual std as a fraction of its mean


@dataclass
class DepthResidualReport:
    height: int
    width: int
    per_row_mean: np.ndarray  # (H,)
    per_col_mean: np.ndarray  # (W,)
    overall_mean: float
    row_structure: float  # std(row_means) / overall_mean
    col_structure: float  # std(col_means) / overall_mean
    time_variation: float  # mean over dims of std_over_ticks / overall_mean
    verdict: str


def depth_spatial_residual(
    a: ObsStream,
    b: ObsStream,
    join: JoinResult,
    *,
    height: int = DEPTH_HEIGHT,
    width: int = DEPTH_WIDTH,
) -> Optional[DepthResidualReport]:
    """Reshape the matched depth-block residual to (H, W) and score its
    structure, distinguishing a geometry mismatch from a frame-freshness lag.

    A **row-structured** residual (per-row means vary far more than the noise
    floor) is a vertical-FOV geometry-mismatch signature. An **unstructured,
    time-varying** residual (large per-tick variation, flat spatial map) is a
    frame-freshness-lag signature. Returns ``None`` for a camera-free variant.
    The verdict is a heuristic hint, not a proof; the raw per-row / per-col
    means are returned so an operator can eyeball the map.
    """
    _, depth_range = split_indices(a.variant)
    if depth_range is None:
        return None
    start, stop = depth_range
    if stop - start != height * width:
        raise ValueError(
            f"depth block is {stop - start} dims but {height}x{width}="
            f"{height * width} expected; pass matching height/width"
        )
    if join.n_matched == 0:
        raise ValueError("no matched ticks; cannot form a depth residual")

    da = a.obs[[ia for ia, _, _ in join.pairs]][:, start:stop].astype(np.float64)
    db = b.obs[[ib for _, ib, _ in join.pairs]][:, start:stop].astype(np.float64)
    resid = np.abs(da - db)  # (M, H*W)

    per_dim_mean = resid.mean(axis=0)  # (H*W,)
    grid = per_dim_mean.reshape(height, width)
    row_mean = grid.mean(axis=1)
    col_mean = grid.mean(axis=0)
    overall = float(per_dim_mean.mean())
    denom = overall + _EPS

    row_structure = float(row_mean.std() / denom)
    col_structure = float(col_mean.std() / denom)
    # Per-tick temporal variation of the whole-frame residual.
    per_tick = resid.mean(axis=1)
    time_variation = float(per_tick.std() / (per_tick.mean() + _EPS))

    if max(row_structure, col_structure) > _STRUCT_THRESH and (
        row_structure >= col_structure
    ):
        verdict = (
            "ROW-STRUCTURED residual -> vertical-FOV geometry-mismatch signature"
        )
    elif col_structure > _STRUCT_THRESH:
        verdict = "COLUMN-STRUCTURED residual -> horizontal geometry mismatch"
    elif time_variation > _TIME_THRESH:
        verdict = (
            "UNSTRUCTURED, TIME-VARYING residual -> frame-freshness-lag signature"
        )
    else:
        verdict = "diffuse residual within the noise floor -> no dominant structure"

    return DepthResidualReport(
        height=height,
        width=width,
        per_row_mean=row_mean,
        per_col_mean=col_mean,
        overall_mean=overall,
        row_structure=row_structure,
        col_structure=col_structure,
        time_variation=time_variation,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Inter-inference cadence report
# ---------------------------------------------------------------------------


@dataclass
class CadenceReport:
    n_ticks: int
    expected_period_s: float
    mode_delta_s: float
    fraction_at_expected: float  # within +/-20% of expected
    n_gaps: int  # delta > 1.5x expected
    n_bursts: int  # delta < 0.5x expected
    hist_edges: np.ndarray
    hist_counts: np.ndarray
    verdict: str


def cadence_report(
    t_sim: np.ndarray, *, expected_period_s: float = POLICY_PERIOD_S
) -> CadenceReport:
    """Histogram the inter-inference sim-time deltas.

    Training delivers exactly one fresh depth per policy step; the freshness
    gate should reproduce that as a spike of inter-inference deltas at
    ``POLICY_PERIOD_S`` in sim time regardless of wall RTF. A shifted mode,
    gaps, or bursts here is a cadence-parity bug worth surfacing.
    """
    t = np.sort(np.asarray(t_sim, dtype=np.float64))
    deltas = np.diff(t)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return CadenceReport(
            n_ticks=int(t.size),
            expected_period_s=expected_period_s,
            mode_delta_s=0.0,
            fraction_at_expected=0.0,
            n_gaps=0,
            n_bursts=0,
            hist_edges=np.asarray([]),
            hist_counts=np.asarray([]),
            verdict="too few ticks to assess cadence",
        )

    e = expected_period_s
    edges = np.array(
        [0.0, 0.5 * e, 0.8 * e, 0.9 * e, 1.1 * e, 1.2 * e, 1.5 * e, 2.0 * e, np.inf]
    )
    counts, _ = np.histogram(deltas, bins=edges)

    # Mode at ~1 ms resolution so float jitter does not fragment the peak.
    rounded = np.round(deltas / 1e-3).astype(np.int64)
    vals, freq = np.unique(rounded, return_counts=True)
    mode_delta = float(vals[int(np.argmax(freq))] * 1e-3)

    within = float(np.mean((deltas >= 0.8 * e) & (deltas <= 1.2 * e)))
    n_gaps = int(np.sum(deltas > 1.5 * e))
    n_bursts = int(np.sum(deltas < 0.5 * e))

    if within >= 0.9 and abs(mode_delta - e) <= 0.2 * e:
        verdict = f"clean: mode at {mode_delta * 1e3:.1f} ms ~= expected {e * 1e3:.1f} ms"
    elif abs(mode_delta - e) > 0.2 * e:
        verdict = (
            f"SHIFTED mode {mode_delta * 1e3:.1f} ms vs expected {e * 1e3:.1f} ms "
            "-> cadence-parity concern"
        )
    else:
        verdict = (
            f"IRREGULAR: only {within * 100:.0f}% of ticks near expected "
            f"({n_gaps} gaps, {n_bursts} bursts) -> cadence-parity concern"
        )

    return CadenceReport(
        n_ticks=int(t.size),
        expected_period_s=e,
        mode_delta_s=mode_delta,
        fraction_at_expected=within,
        n_gaps=n_gaps,
        n_bursts=n_bursts,
        hist_edges=edges,
        hist_counts=counts,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Subgoal-pick self-consistency (bag-replay)
# ---------------------------------------------------------------------------


@dataclass
class SubgoalTickResidual:
    t_sim: float
    residual_m: float
    recomputed_xy: tuple[float, float]
    published_xy: tuple[float, float]


@dataclass
class SubgoalParityReport:
    tol_m: float
    lookahead_m: float
    ticks: list[SubgoalTickResidual] = field(default_factory=list)
    n_skipped_no_path: int = 0

    @property
    def n_evaluated(self) -> int:
        return len(self.ticks)

    @property
    def max_residual_m(self) -> float:
        return max((t.residual_m for t in self.ticks), default=0.0)

    @property
    def worst_tick(self) -> Optional[SubgoalTickResidual]:
        if not self.ticks:
            return None
        return max(self.ticks, key=lambda t: t.residual_m)

    @property
    def passed(self) -> bool:
        return self.n_evaluated > 0 and self.max_residual_m <= self.tol_m


def replay_subgoal_consistency(
    events: list[tuple],
    *,
    lookahead_m: float = SUBGOAL_LOOKAHEAD_M,
    max_points: Optional[int] = None,
    tol_m: float = SUBGOAL_BOUND_M,
) -> SubgoalParityReport:
    """Replay recorded plans + robot poses through the numpy generator and
    compare each recomputed pick to the published rolling subgoal.

    ``events`` is a time-ordered list of tuples:
      - ``("plan", path_xy)``      -- path_xy is an (N, 2) array; rewinds cursor.
      - ``("tick", robot_xy, published_xy, t_sim)`` -- one generator update.

    The generator's cursor is monotonic, so replaying the events in their
    recorded order reproduces the node's cursor trajectory. Ticks arriving
    before any plan are counted as skipped. Self-consistency (not
    ground-truth): a mismatch flags a frame / ordering / lookahead wiring gap
    between the node and this reference.
    """
    gen = RollingSubgoalGenerator(lookahead_m=lookahead_m, max_points=max_points)
    report = SubgoalParityReport(tol_m=tol_m, lookahead_m=lookahead_m)
    for ev in events:
        kind = ev[0]
        if kind == "plan":
            path_xy = np.asarray(ev[1], dtype=np.float64)
            gen.set_path(path_xy)
        elif kind == "tick":
            _, robot_xy, published_xy, t_sim = ev
            if not gen.has_path:
                report.n_skipped_no_path += 1
                continue
            state = gen.update(np.asarray(robot_xy, dtype=np.float64))
            if state is None:
                report.n_skipped_no_path += 1
                continue
            rec = (float(state.subgoal_xy[0]), float(state.subgoal_xy[1]))
            pub = (float(published_xy[0]), float(published_xy[1]))
            residual = math.hypot(rec[0] - pub[0], rec[1] - pub[1])
            report.ticks.append(
                SubgoalTickResidual(
                    t_sim=float(t_sim),
                    residual_m=residual,
                    recomputed_xy=rec,
                    published_xy=pub,
                )
            )
        else:
            raise ValueError(f"unknown subgoal event kind: {kind!r}")
    return report


# ---------------------------------------------------------------------------
# Self-check re-assembly (mirrors InferenceNode._assemble_observation_or_none)
# ---------------------------------------------------------------------------


def reassemble_obs_from_extracted(
    variant: PolicyVariant,
    *,
    imu_accel: tuple[float, float, float],
    imu_gyro: tuple[float, float, float],
    joint_names: list[str],
    joint_velocities: list[float],
    body_velocity_xy: tuple[float, float],
    last_action: np.ndarray,
    referent_map_xy: tuple[float, float],
    base_in_map_xy: tuple[float, float],
    base_in_map_quat: tuple[float, float, float, float],
    depth_meters: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Re-assemble one obs vector from raw sensor values via the SAME
    ``obs_pipeline`` functions the node uses.

    This is the pure core of the obs self-check: given the values a bag carries
    (IMU, joint states, odom body velocity, the published subgoal/goal as the
    referent, the map->base_link TF, and depth for camera variants) it pins the
    node's assembly wiring/ordering/scales with no workstation dumper. ``last_action``
    is node-internal feedback, not on any topic; the self-check sources it from
    the node dump's own last_action dims (it cannot be reconstructed from the
    bag and is not independently checked).
    """
    ref_rel, ref_dist, ref_head = body_frame_goal(
        goal_map_xy=(float(referent_map_xy[0]), float(referent_map_xy[1])),
        base_in_map_xy=(float(base_in_map_xy[0]), float(base_in_map_xy[1])),
        base_in_map_quat=tuple(float(q) for q in base_in_map_quat),
    )
    wheel_vels = joint_state_to_wheel_vels(list(joint_names), list(joint_velocities))
    has_depth = any(f.key == "depth_image" for f in variant.fields)
    depth_flat = (
        downsample_depth(np.asarray(depth_meters, dtype=np.float32))
        if has_depth
        else None
    )
    raw = build_raw_obs_dict(
        variant=variant,
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        wheel_vels_rad_s=wheel_vels,
        goal_relative_xy=ref_rel,
        goal_distance=ref_dist,
        goal_heading_to_goal=ref_head,
        body_velocity_xy=body_velocity_xy,
        last_action=np.asarray(last_action, dtype=np.float32),
        depth_flat_meters=depth_flat,
    )
    return assemble_observation(raw, variant)


# ---------------------------------------------------------------------------
# Report formatting (CLI stdout)
# ---------------------------------------------------------------------------


def format_obs_report(report: ObsParityReport) -> str:
    j = report.join
    lines = [
        f"variant: {report.variant.name} (obs_dim={report.variant.obs_dim})",
        f"join: {j.n_matched}/{j.n_a} node ticks matched a reference tick "
        f"within +/-{JOIN_TOL_S * 1e3:.2f} ms "
        f"({j.matched_fraction_a * 100:.1f}%); "
        f"unmatched node={len(j.unmatched_a)} reference={len(j.unmatched_b)}; "
        f"worst |dt|={j.worst_dt * 1e3:.3f} ms",
        f"coverage: {'OK' if report.coverage_ok else 'FAIL'} "
        f"(>= {report.min_matched_fraction * 100:.0f}% required)",
        f"scalar dims: max|delta|={report.scalar_max_abs:.3e} at "
        f"{report.scalar_worst_name} (dim {report.scalar_worst_dim}) "
        f"vs bound {report.scalar_bound:.0e} -> "
        f"{'PASS' if report.scalar_pass else 'FAIL'}",
    ]
    if report.depth_max_abs is not None:
        lines.append(
            f"depth dims: max|delta|={report.depth_max_abs:.3e} at "
            f"{report.depth_worst_name} (dim {report.depth_worst_dim}) "
            f"vs bound {report.depth_bound:.0e} -> "
            f"{'PASS' if report.depth_pass else 'FAIL'}"
        )
    lines.append(f"OBS PARITY: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)


def format_depth_report(report: DepthResidualReport) -> str:
    return "\n".join(
        [
            f"depth spatial residual ({report.height}x{report.width}):",
            f"  overall mean|delta|={report.overall_mean:.3e}",
            f"  row-structure score={report.row_structure:.2f}  "
            f"col-structure score={report.col_structure:.2f}  "
            f"time-variation score={report.time_variation:.2f}",
            f"  verdict: {report.verdict}",
        ]
    )


def format_cadence_report(report: CadenceReport) -> str:
    e = report.expected_period_s
    hist = "  ".join(
        f"[{report.hist_edges[i] / e:.2g}-{report.hist_edges[i + 1] / e:.2g}x]="
        f"{int(report.hist_counts[i])}"
        for i in range(len(report.hist_counts))
    )
    return "\n".join(
        [
            f"cadence: {report.n_ticks} ticks, expected period "
            f"{e * 1e3:.2f} ms ({1.0 / e:.1f} Hz)",
            f"  mode={report.mode_delta_s * 1e3:.2f} ms  "
            f"near-expected={report.fraction_at_expected * 100:.1f}%  "
            f"gaps={report.n_gaps}  bursts={report.n_bursts}",
            f"  hist (delta/expected): {hist}",
            f"  verdict: {report.verdict}",
        ]
    )


def format_subgoal_report(report: SubgoalParityReport) -> str:
    worst = report.worst_tick
    worst_str = (
        f"worst {worst.residual_m * 100:.2f} cm at t_sim={worst.t_sim:.3f} "
        f"(recomputed={worst.recomputed_xy}, published={worst.published_xy})"
        if worst is not None
        else "no ticks evaluated"
    )
    return "\n".join(
        [
            f"subgoal pick self-consistency (lookahead={report.lookahead_m:.3f} m):",
            f"  evaluated={report.n_evaluated} ticks, "
            f"skipped(no plan)={report.n_skipped_no_path}",
            f"  {worst_str}",
            f"  bound {report.tol_m * 100:.1f} cm -> "
            f"SUBGOAL PARITY: {'PASS' if report.passed else 'FAIL'}",
        ]
    )
