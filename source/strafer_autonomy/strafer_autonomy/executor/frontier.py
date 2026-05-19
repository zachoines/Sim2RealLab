"""Wavefront frontier detection over the Nav2 global costmap.

A *frontier cell* is a free cell that has at least one unknown neighbor.
Connected components of frontier cells form *frontier clusters*; each
cluster's centroid is the candidate exploration target. The
``explore_until_visible`` skill in ``mission_runner.py`` consumes the
ranked cluster list, drives Nav2 to the best candidate, and re-scans
for the labelled target on arrival.

Pure-Python: no ROS imports, so the detector can be unit-tested against
synthetic occupancy grids without bringing up rclpy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# Nav2's default occupied threshold on the ``/global_costmap/costmap``
# topic. Cells at or above this value are treated as obstacles; below
# (and non-negative) are free.
DEFAULT_OCCUPIED_THRESHOLD = 65

# Floor on cluster size so single-cell speckle and one-pixel ribbons
# along the costmap edge do not generate spurious exploration goals.
DEFAULT_MIN_CLUSTER_CELLS = 8


@dataclass(frozen=True)
class FrontierCluster:
    """One connected group of frontier cells.

    ``cluster_key`` is a stable identity used by the explore loop to
    skip frontiers it has already visited. It is the lex-min ``(row,
    col)`` cell in the cluster — small enough to fit in a set, stable
    across detector invocations as long as the underlying frontier
    geometry doesn't shift below resolution.
    """

    cluster_key: tuple[int, int]
    cell_count: int
    centroid_xy: tuple[float, float]
    bbox_rows: tuple[int, int]
    bbox_cols: tuple[int, int]


def detect_frontier_clusters(
    *,
    data: Any,
    origin_xy: tuple[float, float],
    resolution: float,
    occupied_threshold: int = DEFAULT_OCCUPIED_THRESHOLD,
    min_cluster_cells: int = DEFAULT_MIN_CLUSTER_CELLS,
) -> list[FrontierCluster]:
    """Return frontier clusters in the occupancy grid ``data``.

    ``data`` is an int8 ndarray shape ``(height, width)`` with Nav2
    cell semantics (``-1`` unknown, ``< occupied_threshold`` free,
    ``>= occupied_threshold`` occupied). ``origin_xy`` is the
    map-frame position of the grid's ``(row=0, col=0)`` cell — the
    Nav2 ``OccupancyGrid.info.origin.position`` — and ``resolution`` is
    metres per cell.

    Clusters smaller than ``min_cluster_cells`` are dropped as noise.
    Output is unordered; the caller is responsible for ranking.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {data.shape}")
    height, width = data.shape

    is_unknown = data == -1
    is_free = (data >= 0) & (data < occupied_threshold)

    # A frontier cell is free AND adjacent (4-connected) to an unknown
    # cell. Build the mask vector-wise so detection cost stays O(N) in
    # cells rather than O(N) Python-loop iterations.
    unknown_neighbor = np.zeros_like(is_unknown)
    unknown_neighbor[:-1, :] |= is_unknown[1:, :]
    unknown_neighbor[1:, :] |= is_unknown[:-1, :]
    unknown_neighbor[:, :-1] |= is_unknown[:, 1:]
    unknown_neighbor[:, 1:] |= is_unknown[:, :-1]

    is_frontier = is_free & unknown_neighbor

    visited = np.zeros_like(is_frontier)
    clusters: list[FrontierCluster] = []

    rows, cols = np.where(is_frontier)
    for seed_idx in range(rows.size):
        sr, sc = int(rows[seed_idx]), int(cols[seed_idx])
        if visited[sr, sc]:
            continue
        # BFS over the connected frontier component.
        queue: deque[tuple[int, int]] = deque()
        queue.append((sr, sc))
        visited[sr, sc] = True
        component: list[tuple[int, int]] = []
        min_r, max_r = sr, sr
        min_c, max_c = sc, sc
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            if r < min_r:
                min_r = r
            if r > max_r:
                max_r = r
            if c < min_c:
                min_c = c
            if c > max_c:
                max_c = c
            for nr, nc in (
                (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
            ):
                if 0 <= nr < height and 0 <= nc < width:
                    if is_frontier[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        if len(component) < min_cluster_cells:
            continue

        sum_r = sum(r for r, _ in component)
        sum_c = sum(c for _, c in component)
        mean_r = sum_r / len(component)
        mean_c = sum_c / len(component)
        centroid_x = origin_xy[0] + (mean_c + 0.5) * resolution
        centroid_y = origin_xy[1] + (mean_r + 0.5) * resolution
        # Lex-min cell stabilises the cluster key across detector
        # invocations even if BFS happens to seed from a different
        # corner — sorting the component once is cheap relative to
        # the BFS itself.
        cluster_key = min(component)
        clusters.append(
            FrontierCluster(
                cluster_key=cluster_key,
                cell_count=len(component),
                centroid_xy=(centroid_x, centroid_y),
                bbox_rows=(min_r, max_r),
                bbox_cols=(min_c, max_c),
            )
        )
    return clusters


@dataclass(frozen=True)
class RankedFrontier:
    """Frontier cluster annotated with its ranking score and metadata."""

    cluster: FrontierCluster
    distance_m: float
    score: float


def rank_frontiers(
    *,
    clusters: list[FrontierCluster],
    robot_xy: tuple[float, float],
    max_distance_m: float,
    visited_keys: set[tuple[int, int]] | None = None,
    llm_prior_fn: Callable[[FrontierCluster], float] | None = None,
) -> list[RankedFrontier]:
    """Filter and rank frontier clusters by gain/cost.

    Score is ``cell_count / max(distance_m, resolution_floor)`` — a
    Euclidean-distance proxy for Nav2 plan length, sufficient for v1.
    Clusters whose centroid is farther than ``max_distance_m`` from
    the robot, or whose key is in ``visited_keys``, are dropped.
    Higher score is better; the returned list is sorted accordingly.

    ``llm_prior_fn``, when supplied, returns a non-negative
    multiplier per cluster — the seam for an LFG-style LLM-guided
    prior layered on top of the geometric score. The default
    (``None``) recovers the geometric-only ranking exactly, which is
    also the contract a feature flag like ``gain_weights.llm = 0.0``
    must preserve when the upstream consumer opts out.
    """
    rx, ry = robot_xy
    visited = visited_keys or set()
    ranked: list[RankedFrontier] = []
    for cluster in clusters:
        if cluster.cluster_key in visited:
            continue
        cx, cy = cluster.centroid_xy
        dx = cx - rx
        dy = cy - ry
        distance = float(np.sqrt(dx * dx + dy * dy))
        if distance > max_distance_m:
            continue
        # Floor the denominator so frontiers right at the robot's
        # current pose don't divide by zero and dominate the ranking
        # purely by proximity. 0.5 m matches a typical free-space
        # tolerance and keeps the score finite.
        score = cluster.cell_count / max(distance, 0.5)
        if llm_prior_fn is not None:
            score *= float(llm_prior_fn(cluster))
        ranked.append(
            RankedFrontier(cluster=cluster, distance_m=distance, score=score)
        )
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
