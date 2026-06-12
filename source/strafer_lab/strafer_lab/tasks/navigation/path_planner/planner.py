"""Grid A* planner producing ``nav_msgs/Path``-shaped waypoint sequences.

Operates on a boolean free-space grid (True = passable) in env-local
coordinates, the same representation the procedural-room generator's
robot-radius-inflated occupancy grid uses. Output is an ``(N, 2)`` float32
array of XY waypoints at a fixed arc-length discretization, with exact start
and goal endpoints — the same waypoint semantics the inference-time hybrid
backend reads from Nav2's ``/plan`` topic, so both sides of the sim-to-real
seam consume an identical path shape.

Numpy-only on purpose: this module is imported by pure-Python unit tests and
must not pull in Isaac Lab.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

# 8-connected neighborhood: (dr, dc, step_cost_in_cells).
_SQRT2 = math.sqrt(2.0)
_NEIGHBORS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
)


class PathPlanningError(RuntimeError):
    """Base class for planning failures."""


class NoPathError(PathPlanningError):
    """No collision-free path connects start and goal."""


class InvalidEndpointError(PathPlanningError):
    """Start or goal is unusable (outside grid, inside an obstacle beyond
    snap tolerance, or coincident)."""


def _xy_to_cell(
    xy: np.ndarray, grid_origin_xy: tuple[float, float], grid_res: float
) -> tuple[int, int]:
    """Map an env-local XY point to its (row, col) grid cell."""
    row = int(math.floor((xy[0] - grid_origin_xy[0]) / grid_res))
    col = int(math.floor((xy[1] - grid_origin_xy[1]) / grid_res))
    return row, col


def _cell_to_xy(
    row: int, col: int, grid_origin_xy: tuple[float, float], grid_res: float
) -> tuple[float, float]:
    """Map a (row, col) grid cell to its center XY (env-local)."""
    x = row * grid_res + grid_origin_xy[0] + grid_res / 2.0
    y = col * grid_res + grid_origin_xy[1] + grid_res / 2.0
    return x, y


def _snap_to_free(
    free: np.ndarray, cell: tuple[int, int], max_radius_cells: int
) -> tuple[int, int] | None:
    """Return the nearest free cell within ``max_radius_cells`` (Chebyshev
    rings, deterministic scan order), or None if there is none."""
    rows, cols = free.shape
    r0, c0 = cell
    for radius in range(max_radius_cells + 1):
        best = None
        best_d2 = None
        for r in range(max(0, r0 - radius), min(rows, r0 + radius + 1)):
            for c in range(max(0, c0 - radius), min(cols, c0 + radius + 1)):
                if max(abs(r - r0), abs(c - c0)) != radius:
                    continue
                if free[r, c]:
                    d2 = (r - r0) ** 2 + (c - c0) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best, best_d2 = (r, c), d2
        if best is not None:
            return best
    return None


def _octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (_SQRT2 - 1.0) * min(dr, dc)


def _astar(
    free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    """8-connected A* on a boolean grid. Deterministic tie-breaking via a
    monotone push counter. Returns the cell path including both endpoints."""
    rows, cols = free.shape
    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score[start] = 0.0
    counter = 0
    open_heap: list[tuple[float, float, int, tuple[int, int]]] = [
        (_octile(start, goal), _octile(start, goal), counter, start)
    ]
    closed = np.zeros((rows, cols), dtype=bool)

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if closed[current]:
            continue
        closed[current] = True
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        cr, cc = current
        for dr, dc, step in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if not free[nr, nc] or closed[nr, nc]:
                continue
            # Disallow cutting a corner diagonally between two blocked cells.
            if dr != 0 and dc != 0 and not (free[cr + dr, cc] or free[cr, cc + dc]):
                continue
            tentative = g_score[current] + step
            if tentative < g_score[nr, nc]:
                g_score[nr, nc] = tentative
                came_from[(nr, nc)] = current
                h = _octile((nr, nc), goal)
                counter += 1
                heapq.heappush(open_heap, (tentative + h, h, counter, (nr, nc)))

    raise NoPathError(
        f"no collision-free path from cell {start} to cell {goal}"
    )


def _segment_is_free(
    free: np.ndarray,
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
    grid_origin_xy: tuple[float, float],
    grid_res: float,
) -> bool:
    """Line-of-sight check: sample the segment at half-cell spacing and
    require every sampled cell to be free."""
    ax, ay = a_xy
    bx, by = b_xy
    length = math.hypot(bx - ax, by - ay)
    n = max(2, int(length / (grid_res / 2.0)) + 1)
    rows, cols = free.shape
    for i in range(n + 1):
        t = i / n
        x = ax + t * (bx - ax)
        y = ay + t * (by - ay)
        r, c = _xy_to_cell(np.array([x, y]), grid_origin_xy, grid_res)
        if r < 0 or r >= rows or c < 0 or c >= cols or not free[r, c]:
            return False
    return True


def _shortcut(
    points: list[tuple[float, float]],
    free: np.ndarray,
    grid_origin_xy: tuple[float, float],
    grid_res: float,
) -> list[tuple[float, float]]:
    """Greedy line-of-sight shortcutting. Removes the grid-quantization
    zigzag from raw A* cell paths, bringing the output closer to the smooth
    gradient-following paths a deployed costmap planner emits."""
    if len(points) <= 2:
        return points
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1:
            if _segment_is_free(free, points[i], points[j], grid_origin_xy, grid_res):
                break
            j -= 1
        out.append(points[j])
        i = j
    return out


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at uniform arc-length ``spacing``.

    The first and last points are preserved exactly; interior samples sit at
    multiples of ``spacing`` along arc length. Returns float32 ``(N, 2)``
    with N >= 2 (for a degenerate zero-length input, the point is repeated).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 1:
        raise ValueError(f"expected (N, 2) polyline, got shape {pts.shape}")
    if len(pts) == 1:
        return np.repeat(pts, 2, axis=0).astype(np.float32)
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(arc[-1])
    if total <= 1e-9:
        return np.stack([pts[0], pts[-1]]).astype(np.float32)
    n_interior = int(total / spacing)
    targets = np.arange(1, n_interior + 1, dtype=np.float64) * spacing
    targets = targets[targets < total - 1e-9]
    sample_arcs = np.concatenate([[0.0], targets, [total]])
    x = np.interp(sample_arcs, arc, pts[:, 0])
    y = np.interp(sample_arcs, arc, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def perturb_waypoints(
    path: np.ndarray,
    noise_std_m: float,
    free_space: np.ndarray,
    *,
    grid_res: float,
    grid_origin_xy: tuple[float, float],
    rng: np.random.Generator,
    correlation_length_m: float = 0.5,
) -> np.ndarray:
    """Apply bounded, spatially correlated noise to a path's waypoints.

    Models the disagreement envelope between this planner and the deployed
    planner. Planner disagreement is smooth along the path (two planners
    differ in where the corridor runs, not per-waypoint), so offsets are
    sampled at control points ``correlation_length_m`` apart and linearly
    interpolated across the waypoints in between — independent per-waypoint
    noise would crinkle the polyline and inflate its arc length, corrupting
    the lookahead and along-track-progress semantics downstream.

    Each control offset is per-axis Gaussian truncated at two standard
    deviations, so any waypoint's displacement is bounded by
    ``2 * noise_std_m * sqrt(2)``. Endpoints stay exact, and a perturbed
    waypoint is kept only if it remains in free space — otherwise the
    original waypoint is retained.
    """
    out = np.asarray(path, dtype=np.float32).copy()
    n = len(out)
    if noise_std_m <= 0.0 or n <= 2:
        return out

    seg_len = np.linalg.norm(np.diff(out, axis=0), axis=1)
    mean_spacing = float(seg_len.mean())
    stride = max(1, int(round(correlation_length_m / max(mean_spacing, 1e-6))))
    control_idx = np.arange(0, n, stride)
    if control_idx[-1] != n - 1:
        control_idx = np.append(control_idx, n - 1)

    offsets = rng.normal(0.0, noise_std_m, size=(len(control_idx), 2))
    np.clip(offsets, -2.0 * noise_std_m, 2.0 * noise_std_m, out=offsets)
    offsets[0] = 0.0   # endpoints stay exact
    offsets[-1] = 0.0

    all_idx = np.arange(n, dtype=np.float64)
    noise = np.stack(
        [np.interp(all_idx, control_idx, offsets[:, k]) for k in range(2)],
        axis=1,
    ).astype(np.float32)

    candidate = out + noise
    rows, cols = free_space.shape
    for k in range(1, n - 1):
        r, c = _xy_to_cell(candidate[k], grid_origin_xy, grid_res)
        if 0 <= r < rows and 0 <= c < cols and free_space[r, c]:
            out[k] = candidate[k]
    return out


def plan_path(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    free_space: np.ndarray,
    *,
    grid_res: float,
    grid_origin_xy: tuple[float, float],
    discretization_m: float = 0.05,
    snap_radius_m: float = 0.3,
) -> np.ndarray:
    """Plan a collision-free path from ``start_xy`` to ``goal_xy``.

    Args:
        start_xy: (2,) env-local start position in meters.
        goal_xy: (2,) env-local goal position in meters.
        free_space: (rows, cols) bool grid, True = passable. Must already be
            inflated by the robot radius.
        grid_res: meters per grid cell.
        grid_origin_xy: env-local XY of the grid's (0, 0) cell corner.
        discretization_m: arc-length spacing of the output waypoints.
        snap_radius_m: endpoints falling on an occupied cell are snapped to
            the nearest free cell within this radius before planning.

    Returns:
        (N, 2) float32 waypoints (env-local), N >= 2, first == start,
        last == goal, interior points spaced ``discretization_m`` apart
        along arc length.

    Raises:
        InvalidEndpointError: start/goal outside the grid, inside an
            obstacle beyond ``snap_radius_m``, or coincident.
        NoPathError: the endpoints are in disconnected free-space regions.
    """
    start_xy = np.asarray(start_xy, dtype=np.float64).reshape(2)
    goal_xy = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    free = np.asarray(free_space, dtype=bool)
    rows, cols = free.shape

    if float(np.hypot(*(goal_xy - start_xy))) < discretization_m:
        raise InvalidEndpointError(
            f"start {start_xy.tolist()} and goal {goal_xy.tolist()} are "
            f"closer than one waypoint spacing ({discretization_m} m)"
        )

    snap_cells = max(0, int(math.ceil(snap_radius_m / grid_res)))
    cells: list[tuple[int, int]] = []
    for label, xy in (("start", start_xy), ("goal", goal_xy)):
        r, c = _xy_to_cell(xy, grid_origin_xy, grid_res)
        if r < 0 or r >= rows or c < 0 or c >= cols:
            raise InvalidEndpointError(
                f"{label} {xy.tolist()} is outside the planning grid"
            )
        if not free[r, c]:
            snapped = _snap_to_free(free, (r, c), snap_cells)
            if snapped is None:
                raise InvalidEndpointError(
                    f"{label} {xy.tolist()} is inside an obstacle and no free "
                    f"cell exists within {snap_radius_m} m"
                )
            r, c = snapped
        cells.append((r, c))

    cell_path = _astar(free, cells[0], cells[1])

    # Cell centers, with the exact endpoints substituted at both ends.
    pts = [tuple(start_xy)]
    pts += [
        _cell_to_xy(r, c, grid_origin_xy, grid_res) for r, c in cell_path[1:-1]
    ]
    pts.append(tuple(goal_xy))

    pts = _shortcut(pts, free, grid_origin_xy, grid_res)
    return resample_polyline(np.asarray(pts), discretization_m)
