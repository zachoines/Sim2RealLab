"""Deterministic geometric coverage plan over a scene's rooms.

A coverage plan visits every room at least ``visits_per_room`` times and
re-approaches each room's viewpoint from headings spread around the circle.
The same-place / different-heading pairs this produces are the training
signal a place-recognition head mines as positive VPR pairs and a goal-
reaching demo set never captures.

The plan is **geometric and deterministic**, not learned: a seeded RNG fixes
the per-room base heading and per-visit rotation offsets, so the same
``(rooms, seed)`` reproduces the same plan. The trained RL subgoal-follower
that drives the robot to these viewpoints lives in the capture driver; this
module only decides *where* to look from and *which way to face*.

Pure-Python / numpy. Reuses the shared free-space + room-graph helpers in
:mod:`strafer_lab.tools.scene_connectivity` (room representative points and
the connectivity graph that orders the traversal) — it does not rasterize
free space or re-implement A*. It does not touch Kit runtime; the driver
reads the rooms + occupancy off disk and hands them in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .scene_connectivity import compute_connectivity, room_representative_xy

DEFAULT_VISITS_PER_ROOM = 2
# A room counts as heading-diverse when its widest pair of approach headings
# is more than this far apart on the circle. pi/2 (90 deg) is the documented
# acceptance threshold; exposed so the operator can tune the coverage metric.
DEFAULT_HEADING_SPREAD_RAD = math.pi / 2.0


@dataclass(frozen=True)
class VisitWaypoint:
    """One scheduled viewpoint visit.

    ``target_xy`` is the scene-local (env-local) point to occupy — the same
    frame as the occupancy grid and the rest of the plan, which the driver
    lifts by the env origin before feeding the command term;
    ``approach_heading_rad`` is the heading the robot should face on arrival
    (realized by the driver as the final approach segment's direction).
    Revisits of the same room share ``target_xy`` and differ only in
    ``approach_heading_rad`` — the VPR same-place / different-heading pair.
    """

    room_index: int
    visit_ordinal: int
    target_xy: tuple[float, float]
    approach_heading_rad: float


@dataclass(frozen=True)
class CoveragePlan:
    """An ordered traversal of viewpoint visits."""

    waypoints: tuple[VisitWaypoint, ...]
    visits_per_room: int
    heading_spread_threshold_rad: float
    seed: int


@dataclass(frozen=True)
class RoomCoverage:
    """Per-room coverage outcome for one plan."""

    room_index: int
    visit_count: int
    headings_rad: tuple[float, ...]
    max_pairwise_heading_gap_rad: float


@dataclass(frozen=True)
class CoverageMetric:
    """Checkable coverage evidence over a plan.

    ``satisfied`` is true only when every room is visited at least
    ``visits_per_room_target`` times AND its widest approach-heading pair
    exceeds ``heading_spread_threshold_rad``.
    """

    rooms: tuple[RoomCoverage, ...]
    visits_per_room_target: int
    heading_spread_threshold_rad: float

    @property
    def satisfied(self) -> bool:
        return all(
            r.visit_count >= self.visits_per_room_target
            and r.max_pairwise_heading_gap_rad > self.heading_spread_threshold_rad
            for r in self.rooms
        )


def _circular_delta(a: float, b: float) -> float:
    """Shortest unsigned angular distance between two headings (in [0, pi])."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _max_pairwise_heading_gap(headings: Sequence[float]) -> float:
    """Largest shortest-arc separation between any pair of headings."""
    if len(headings) < 2:
        return 0.0
    return max(
        _circular_delta(headings[i], headings[j])
        for i in range(len(headings))
        for j in range(i + 1, len(headings))
    )


def _room_visit_headings(rng: np.random.Generator, n_visits: int) -> list[float]:
    """Headings for one room's visits: evenly spaced + bounded jitter.

    The jitter is bounded to ``0.2 * spacing`` so the widest approach-heading
    pair stays above the ``pi/2`` spread threshold for every ``n_visits >= 2``
    — at the ``n_visits == 2`` worst case the pair is ``pi - 2*0.2*pi = 0.6*pi``
    (108 deg) apart. A larger fraction would let the two-visit default fall
    below the threshold for some seeds.
    """
    base = float(rng.uniform(0.0, 2.0 * math.pi))
    spacing = 2.0 * math.pi / n_visits
    jitter_bound = 0.2 * spacing
    headings = []
    for k in range(n_visits):
        jitter = float(rng.uniform(-jitter_bound, jitter_bound))
        headings.append(base + k * spacing + jitter)
    return headings


def _order_rooms(
    rooms: Sequence[dict[str, Any]],
    free_space: np.ndarray,
    *,
    grid_origin_xy: tuple[float, float],
    grid_res: float,
    room_adjacency: Sequence[Sequence[int]] | None,
) -> list[int]:
    """Greedy nearest-connected room order so consecutive visits are reachable.

    Builds the room connectivity graph with the shared
    :func:`scene_connectivity.compute_connectivity`, then walks it greedily by
    shortest connecting path. Rooms with no reachable neighbour are appended in
    index order so an isolated room is still scheduled.
    """
    n = len(rooms)
    if n <= 1:
        return list(range(n))

    edges = compute_connectivity(
        rooms,
        free_space,
        grid_origin_xy=grid_origin_xy,
        grid_res=grid_res,
        room_adjacency=room_adjacency,
    )
    dist: dict[tuple[int, int], float] = {}
    for e in edges:
        if e.get("reachable"):
            length = float(e.get("path_length_m", 0.0))
            dist[(e["from_idx"], e["to_idx"])] = length
            dist[(e["to_idx"], e["from_idx"])] = length

    order = [0]
    visited = {0}
    while len(order) < n:
        current = order[-1]
        nxt = min(
            (j for j in range(n) if j not in visited and (current, j) in dist),
            key=lambda j: dist[(current, j)],
            default=None,
        )
        if nxt is None:
            nxt = next(j for j in range(n) if j not in visited)
        order.append(nxt)
        visited.add(nxt)
    return order


def build_coverage_plan(
    rooms: Sequence[dict[str, Any]],
    free_space: np.ndarray,
    *,
    grid_res: float,
    grid_origin_xy: tuple[float, float],
    visits_per_room: int = DEFAULT_VISITS_PER_ROOM,
    seed: int = 0,
    heading_spread_threshold_rad: float = DEFAULT_HEADING_SPREAD_RAD,
    room_adjacency: Sequence[Sequence[int]] | None = None,
) -> CoveragePlan:
    """Build a deterministic coverage plan over ``rooms``.

    Each room contributes ``visits_per_room`` visits to the same free interior
    viewpoint (its representative point), each from a different approach
    heading. Visits are emitted pass by pass — every room once per pass — so a
    revisit only happens after the rest of the scene has been swept, and the
    traversal order within a pass follows the room connectivity graph
    (boustrophedon across passes so each pass starts near where the last
    ended).

    Args:
        rooms: scene room dicts (``footprint_xy`` / ``area_m2`` / ``room_type``).
        free_space: inflated bool free grid, True = passable.
        grid_res: meters per grid cell.
        grid_origin_xy: world XY of the grid's (0, 0) cell corner.
        visits_per_room: minimum visits scheduled per room (>= 1).
        seed: RNG seed; fixes the per-room headings so plans reproduce.
        heading_spread_threshold_rad: recorded on the plan for the metric.
        room_adjacency: optional adjacency hint forwarded to the room graph.
    """
    if visits_per_room < 1:
        raise ValueError(f"visits_per_room must be >= 1; got {visits_per_room}")
    if not rooms:
        return CoveragePlan(
            waypoints=(),
            visits_per_room=visits_per_room,
            heading_spread_threshold_rad=heading_spread_threshold_rad,
            seed=seed,
        )

    targets = [
        room_representative_xy(
            room, free_space, origin_xy=grid_origin_xy, grid_res=grid_res,
        )
        for room in rooms
    ]
    # Per-room RNG keyed off (seed, room index) so a room's headings are stable
    # regardless of how many rooms precede it or the traversal order.
    headings = [
        _room_visit_headings(np.random.default_rng([seed, i]), visits_per_room)
        for i in range(len(rooms))
    ]

    order = _order_rooms(
        rooms,
        free_space,
        grid_origin_xy=grid_origin_xy,
        grid_res=grid_res,
        room_adjacency=room_adjacency,
    )

    waypoints: list[VisitWaypoint] = []
    for visit in range(visits_per_room):
        sweep = order if visit % 2 == 0 else list(reversed(order))
        for room_index in sweep:
            waypoints.append(
                VisitWaypoint(
                    room_index=room_index,
                    visit_ordinal=visit,
                    target_xy=(
                        float(targets[room_index][0]),
                        float(targets[room_index][1]),
                    ),
                    approach_heading_rad=float(headings[room_index][visit]),
                )
            )

    return CoveragePlan(
        waypoints=tuple(waypoints),
        visits_per_room=visits_per_room,
        heading_spread_threshold_rad=heading_spread_threshold_rad,
        seed=seed,
    )


def coverage_metric(
    plan: CoveragePlan,
    *,
    visits_per_room: int | None = None,
    heading_spread_threshold_rad: float | None = None,
) -> CoverageMetric:
    """Measure per-room visit count + approach-heading spread over a plan."""
    target = (
        visits_per_room if visits_per_room is not None else plan.visits_per_room
    )
    threshold = (
        heading_spread_threshold_rad
        if heading_spread_threshold_rad is not None
        else plan.heading_spread_threshold_rad
    )
    by_room: dict[int, list[float]] = {}
    for wp in plan.waypoints:
        by_room.setdefault(wp.room_index, []).append(wp.approach_heading_rad)

    rooms = tuple(
        RoomCoverage(
            room_index=room_index,
            visit_count=len(hs),
            headings_rad=tuple(hs),
            max_pairwise_heading_gap_rad=_max_pairwise_heading_gap(hs),
        )
        for room_index, hs in sorted(by_room.items())
    )
    return CoverageMetric(
        rooms=rooms,
        visits_per_room_target=target,
        heading_spread_threshold_rad=threshold,
    )
