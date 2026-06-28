"""Verified room-to-room connectivity over a scene's cached occupancy grid.

Numpy-only (no Kit, no ``pxr``). The occupancy grid is generated Kit-side by
``validate_scene_connectivity.py`` and cached as ``<scene>/occupancy.npy``;
this module turns that grid plus the scene's room geometry into the
``connectivity[]`` graph the harness consumes.

Reachability is decided by the project's one shared grid planner
(:func:`strafer_lab.tasks.navigation.path_planner.plan_path`): a new
path-planning consumer adapts its scene representation onto the planner, it
does not grow a second A*. The occupancy grid feeds in through
:func:`occupancy_to_free_space` — the shared seam every Infinigen
path-planning consumer reuses:

    occupancy = load_occupancy(scene_dir)          # cached uint8 grid + meta
    free = occupancy_to_free_space(occupancy.grid, grid_res=occupancy.resolution_m)
    path = plan_path(start_xy, goal_xy, free, grid_res=occupancy.resolution_m,
                     grid_origin_xy=occupancy.origin_xy)

The grid convention matches the planner's: ``grid[row, col]`` with ``row``
indexing world +X and ``col`` indexing world +Y, ``grid_origin_xy`` the
world XY of cell ``(0, 0)``'s corner, ``resolution_m`` meters per cell. The
robot-radius inflation uses the same Euclidean disc structuring element as
the GPU training grid (``tasks/navigation/mdp/proc_room.py``); the radius
differs on purpose — see :data:`ROBOT_RADIUS_M`.

This module is harness/grading ground-truth only — the live autonomy stack
must never read it (its runtime counterpart is the observation-derived room
state in ``strafer_autonomy.semantic_map``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from strafer_lab.tasks.navigation.path_planner import (
    InvalidEndpointError,
    NoPathError,
    plan_path,
)

# Inflation radius for the reachability check: the chassis *inscribed* circle
# (half-width). Sourced from the shared chassis geometry. This is deliberately
# the inscribed radius, not the rotation-invariant circumscribed radius the
# training grid uses (tasks/navigation/mdp/proc_room.py): the connectivity graph
# answers "can the holonomic mecanum reach this room", and the strafer can
# rotate to present its narrow dimension through a doorway. Interior doors are
# ~0.55 m wide while the chassis is 0.36 m wide — it fits when aligned, so the
# circumscribed radius (~0.28 m, modelling the robot as a 0.56 m disc) would
# wrongly seal every standard doorway and mark whole rooms unreachable.
try:
    from strafer_shared.constants import CHASSIS_WIDTH, MAP_RESOLUTION

    ROBOT_RADIUS_M = 0.5 * CHASSIS_WIDTH
except Exception:  # pragma: no cover - strafer_shared always present in lab env
    ROBOT_RADIUS_M = 0.18
    MAP_RESOLUTION = 0.05

# Cached-occupancy sidecar layout (regenerable derived intermediate, not
# authored metadata — the connectivity graph is the authored artifact). The
# large occupancy GRID stays a sidecar (not embedded in the USD ``customData``
# the connectivity graph rides in): a multi-hundred-thousand-cell array would
# base64-bloat customData and every read of it, and the planner-side consumers
# (this module, the mission-generator adapter) are numpy-only / pxr-free, so
# moving the grid into the USD would force ``pxr`` on all of them. The
# ``occupancy.json`` USD-identity tie covers the sidecar's one downside
# (staleness). The small connectivity GRAPH, by contrast, is authored truth and
# does live in customData.
OCCUPANCY_GRID_FILENAME = "occupancy.npy"
OCCUPANCY_META_FILENAME = "occupancy.json"

# Cell size = the shared RTAB-Map / Nav2 costmap resolution, so the harness
# connectivity grid matches the grid the runtime planner sees. It is also the
# correctness floor for doorway resolution: interior doors are ~0.55 m wide and
# the inscribed inflation radius is ~0.18 m (needs > 0.36 m clear), so a ~0.55 m
# door leaves ~3 free cells at 0.05 m but only ~1 at 0.10 m — a coarser grid
# would close real doorways under inflation. Do not raise it without re-checking
# the narrowest doorway against the inflation radius.
DEFAULT_RESOLUTION_M = MAP_RESOLUTION
DEFAULT_DISCRETIZATION_M = 0.05
# Room representative points sit on open floor, so the planner only ever has
# to snap a centroid that landed under furniture back to nearby free space.
DEFAULT_SNAP_RADIUS_M = 1.0


# ---------------------------------------------------------------------------
# Cached occupancy sidecar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedOccupancy:
    """A scene's cached occupancy grid plus the metadata to map it to world XY.

    ``grid`` is ``(rows, cols)`` ``uint8`` — nonzero means blocked (a collider
    occupied or the cell was never observed free). It is the *raw* grid; run
    it through :func:`occupancy_to_free_space` before handing it to the planner.
    """

    grid: np.ndarray
    origin_xy: tuple[float, float]
    resolution_m: float
    z_slice_m: float
    meta: dict[str, Any]


def occupancy_meta(
    *,
    origin_xy: tuple[float, float],
    resolution_m: float,
    z_min_m: float,
    z_max_m: float,
    usd_path: str | None = None,
    usd_mtime_ns: int | None = None,
    usd_size: int | None = None,
) -> dict[str, Any]:
    """Build the ``occupancy.json`` payload.

    Records the grid-to-world mapping (``origin_xy`` / ``resolution_m`` /
    ``z_slice_m``) plus the source-USD identity (path + mtime + size) so a
    consumer can detect a stale grid instead of silently planning on geometry
    that no longer matches the scene.
    """
    return {
        "origin_xy": [float(origin_xy[0]), float(origin_xy[1])],
        "resolution_m": float(resolution_m),
        "z_slice_m": float(0.5 * (z_min_m + z_max_m)),
        "z_min_m": float(z_min_m),
        "z_max_m": float(z_max_m),
        "usd_path": usd_path,
        "usd_mtime_ns": usd_mtime_ns,
        "usd_size": usd_size,
    }


def save_occupancy(
    scene_dir: Path | str, grid: np.ndarray, meta: dict[str, Any]
) -> tuple[Path, Path]:
    """Write ``occupancy.npy`` + ``occupancy.json`` into ``scene_dir``."""
    scene_dir = Path(scene_dir)
    npy_path = scene_dir / OCCUPANCY_GRID_FILENAME
    json_path = scene_dir / OCCUPANCY_META_FILENAME
    np.save(npy_path, np.asarray(grid, dtype=np.uint8))
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return npy_path, json_path


def load_occupancy(scene_dir: Path | str) -> CachedOccupancy:
    """Load a scene's cached occupancy grid + metadata.

    The shared entry point for downstream Infinigen path-planning consumers
    (mission-generator oracle, grounding-negative trajectory checks): load the
    cached grid rather than re-rasterizing the scene.
    """
    scene_dir = Path(scene_dir)
    grid = np.load(scene_dir / OCCUPANCY_GRID_FILENAME)
    meta = json.loads(
        (scene_dir / OCCUPANCY_META_FILENAME).read_text(encoding="utf-8")
    )
    origin = meta["origin_xy"]
    return CachedOccupancy(
        grid=grid,
        origin_xy=(float(origin[0]), float(origin[1])),
        resolution_m=float(meta["resolution_m"]),
        z_slice_m=float(meta.get("z_slice_m", 0.0)),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Occupancy -> free-space adapter (the shared planner seam)
# ---------------------------------------------------------------------------


def _disc_dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Dilate a boolean mask by a Euclidean disc of ``radius_cells``.

    Numpy-only Minkowski sum of the blocked cells with the robot's footprint
    circle — the same configuration-space inflation the GPU training grid does
    with a conv2d disc kernel, so a doorway the policy can thread in training
    is the same width the harness reachability check sees.
    """
    if radius_cells <= 0:
        return mask.copy()
    rows, cols = mask.shape
    out = mask.copy()
    r = radius_cells
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if dr == 0 and dc == 0:
                continue
            if dr * dr + dc * dc > r * r:
                continue
            src_r0, src_r1 = max(0, -dr), rows - max(0, dr)
            src_c0, src_c1 = max(0, -dc), cols - max(0, dc)
            dst_r0, dst_r1 = max(0, dr), rows - max(0, -dr)
            dst_c0, dst_c1 = max(0, dc), cols - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def occupancy_to_free_space(
    occupancy: np.ndarray,
    *,
    grid_res: float,
    robot_radius_m: float = ROBOT_RADIUS_M,
) -> np.ndarray:
    """Invert + robot-radius-inflate a raw occupancy grid for the planner.

    Args:
        occupancy: ``(rows, cols)`` array; nonzero = blocked (occupied or
            unobserved).
        grid_res: meters per cell.
        robot_radius_m: inflation radius (default: chassis inscribed circle /
            half-width — the achievable doorway-passage radius for the
            holonomic mecanum; see :data:`ROBOT_RADIUS_M`).

    Returns:
        ``(rows, cols)`` bool grid, True = passable, blocked cells dilated by
        the robot radius — ready to hand to ``plan_path`` as ``free_space``.
    """
    occ = np.asarray(occupancy)
    blocked = occ != 0
    radius_cells = int(math.ceil(robot_radius_m / grid_res)) if grid_res > 0 else 0
    blocked = _disc_dilate(blocked, radius_cells)
    return ~blocked


# ---------------------------------------------------------------------------
# Room geometry helpers (footprint polygons / floor-mesh recovery)
# ---------------------------------------------------------------------------


def parse_room_floor_name(prim_name: str) -> tuple[str, int, int] | None:
    """Parse an Infinigen floor prim name into ``(room_type, story, index)``.

    Infinigen names per-room floor meshes ``<room_type>_<story>_<index>_floor``
    where the room type itself may contain underscores
    (``living_room_0_0_floor`` -> ``("living_room", 0, 0)``). Returns ``None``
    for a name that does not match the floor pattern.
    """
    name = prim_name
    if not name.endswith("_floor"):
        return None
    stem = name[: -len("_floor")]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        story = int(parts[-2])
        index = int(parts[-1])
    except ValueError:
        return None
    room_type = "_".join(parts[:-2])
    if not room_type:
        return None
    return room_type, story, index


def aabb_to_footprint(
    min_xy: Sequence[float], max_xy: Sequence[float]
) -> list[list[float]]:
    """Turn an axis-aligned XY bounding box into a CCW 4-corner footprint."""
    x0, y0 = float(min_xy[0]), float(min_xy[1])
    x1, y1 = float(max_xy[0]), float(max_xy[1])
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def polygon_centroid(footprint_xy: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Area-weighted centroid of a polygon; vertex mean for degenerate input."""
    pts = np.asarray(footprint_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return (0.0, 0.0)
    if pts.shape[0] < 3:
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area2 = cross.sum()
    if abs(area2) < 1e-12:
        return (float(x.mean()), float(y.mean()))
    cx = ((x + x1) * cross).sum() / (3.0 * area2)
    cy = ((y + y1) * cross).sum() / (3.0 * area2)
    return (float(cx), float(cy))


def point_in_polygon(x: float, y: float, footprint_xy: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test (numpy-free, edge-robust enough)."""
    poly = footprint_xy
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi
        ):
            inside = not inside
        j = i
    return inside


def point_in_any_room(x: float, y: float, rooms: Sequence[dict[str, Any]]) -> bool:
    """True iff ``(x, y)`` lies inside any room's footprint polygon."""
    return any(
        point_in_polygon(x, y, room.get("footprint_xy") or []) for room in rooms
    )


def seal_free_space_to_rooms(
    free_space: np.ndarray,
    rooms: Sequence[dict[str, Any]],
    *,
    origin_xy: tuple[float, float],
    grid_res: float,
) -> np.ndarray:
    """Block free cells outside the room-footprint bounding box.

    The cached occupancy grid keeps a free exterior pad ring around the house
    (the occupancy generator pads the floor-mesh AABB), and the thin-z-slab
    rasterizer leaves perimeter walls porous, so the exterior pad and the
    interior merge into one connected free component — a consumer that picks
    "any free, reachable cell" can land outside the house. Masking to the
    room-footprint AABB drops that pad without touching interior doorways
    (which lie between rooms, inside the AABB), so room-to-room routing is
    unaffected. Returns a new grid; the input is left untouched. A scene with
    no usable footprints is returned unmasked (nothing to seal against).
    """
    fps = [
        np.asarray(r.get("footprint_xy"), dtype=float)
        for r in rooms
        if r.get("footprint_xy")
    ]
    if not fps:
        return free_space.copy()
    pts = np.concatenate(fps, axis=0)
    min_x, min_y = float(pts[:, 0].min()), float(pts[:, 1].min())
    max_x, max_y = float(pts[:, 0].max()), float(pts[:, 1].max())
    rows, cols = free_space.shape
    xs = np.arange(rows) * grid_res + origin_xy[0] + grid_res / 2.0  # row -> world x
    ys = np.arange(cols) * grid_res + origin_xy[1] + grid_res / 2.0  # col -> world y
    inside = (
        ((xs >= min_x) & (xs <= max_x))[:, None]
        & ((ys >= min_y) & (ys <= max_y))[None, :]
    )
    return free_space & inside


def is_multi_story(rooms: Sequence[dict[str, Any]]) -> bool:
    """True iff any room sits on a story other than 0 (the strafer can't climb)."""
    return any(int(r.get("story", 0)) != 0 for r in rooms)


# ---------------------------------------------------------------------------
# Connectivity computation
# ---------------------------------------------------------------------------


def _xy_to_cell(
    xy: tuple[float, float], origin_xy: tuple[float, float], grid_res: float
) -> tuple[int, int]:
    row = int(math.floor((xy[0] - origin_xy[0]) / grid_res))
    col = int(math.floor((xy[1] - origin_xy[1]) / grid_res))
    return row, col


def _cell_to_xy(
    row: int, col: int, origin_xy: tuple[float, float], grid_res: float
) -> tuple[float, float]:
    x = row * grid_res + origin_xy[0] + grid_res / 2.0
    y = col * grid_res + origin_xy[1] + grid_res / 2.0
    return x, y


def room_representative_xy(
    room: dict[str, Any],
    free_space: np.ndarray,
    *,
    origin_xy: tuple[float, float],
    grid_res: float,
) -> tuple[float, float]:
    """Return a free interior point for a room — its centroid, or the nearest
    free cell *inside the room's own footprint* if the centroid landed under
    furniture. The footprint containment keeps the ring search from snapping a
    blocked centroid onto free space in a neighbouring room or the exterior."""
    footprint = room.get("footprint_xy") or []
    has_footprint = len(footprint) >= 3
    centroid = polygon_centroid(footprint)
    rows, cols = free_space.shape
    r, c = _xy_to_cell(centroid, origin_xy, grid_res)
    if 0 <= r < rows and 0 <= c < cols and free_space[r, c]:
        return centroid
    # Ring search outward for the nearest free cell (bounded by the footprint
    # diagonal so we never wander into a neighbouring room).
    if footprint:
        fp = np.asarray(footprint, dtype=float)
        span = float(np.hypot(*(fp.max(axis=0) - fp.min(axis=0))))
    else:
        span = grid_res * max(rows, cols)
    max_radius = max(1, int(math.ceil(span / grid_res)))
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and free_space[rr, cc]:
                    cx, cy = _cell_to_xy(rr, cc, origin_xy, grid_res)
                    if not has_footprint or point_in_polygon(cx, cy, footprint):
                        return (cx, cy)
    return centroid


def _path_length_m(path: np.ndarray) -> float:
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def find_room_boundary_crossing(
    path: np.ndarray,
    room_a: dict[str, Any],
    room_b: dict[str, Any],
) -> tuple[float, float]:
    """Estimate the doorway crossing point of a path between two rooms.

    Classifies each waypoint by which room footprint contains it and returns
    the midpoint between the last point inside ``room_a`` and the first point
    inside ``room_b``. Falls back to the path vertex nearest the midpoint of
    the two room centroids when the footprints leave the corridor unclassified.
    """
    pts = np.asarray(path, dtype=float)
    fa = room_a.get("footprint_xy") or []
    fb = room_b.get("footprint_xy") or []
    in_a = [point_in_polygon(p[0], p[1], fa) for p in pts]
    in_b = [point_in_polygon(p[0], p[1], fb) for p in pts]
    last_a = max((k for k, v in enumerate(in_a) if v), default=None)
    first_b = min((k for k, v in enumerate(in_b) if v), default=None)
    if last_a is not None and first_b is not None and last_a < first_b:
        mid = 0.5 * (pts[last_a] + pts[first_b])
        return (float(mid[0]), float(mid[1]))
    ca = polygon_centroid(fa)
    cb = polygon_centroid(fb)
    mid = np.array([0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1])])
    k = int(np.argmin(np.hypot(*(pts - mid).T)))
    return (float(pts[k][0]), float(pts[k][1]))


def _candidate_pairs(
    rooms: Sequence[dict[str, Any]],
    room_adjacency: Iterable[Sequence[int]] | None,
) -> list[tuple[int, int]]:
    """Undirected (i < j) candidate pairs from ``room_adjacency``, or all pairs."""
    n = len(rooms)
    pairs: set[tuple[int, int]] = set()
    if room_adjacency:
        for edge in room_adjacency:
            if len(edge) < 2:
                continue
            i, j = int(edge[0]), int(edge[1])
            if i == j or not (0 <= i < n and 0 <= j < n):
                continue
            pairs.add((min(i, j), max(i, j)))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.add((i, j))
    return sorted(pairs)


def compute_connectivity(
    rooms: Sequence[dict[str, Any]],
    free_space: np.ndarray,
    *,
    grid_origin_xy: tuple[float, float],
    grid_res: float,
    room_adjacency: Iterable[Sequence[int]] | None = None,
    snap_radius_m: float = DEFAULT_SNAP_RADIUS_M,
    discretization_m: float = DEFAULT_DISCRETIZATION_M,
) -> list[dict[str, Any]]:
    """Compute the ``connectivity[]`` graph for a scene.

    For every candidate room pair (Infinigen's ``room_adjacency`` if present,
    otherwise all unordered pairs), decide reachability by planning a
    collision-free path between the two rooms' representative points on the
    robot-radius-inflated ``free_space`` grid with the shared
    :func:`plan_path`. Cross-story pairs are dropped up front (the strafer
    cannot climb stairs).

    Returns one undirected edge dict per candidate pair (``from_idx`` <
    ``to_idx``):

    - reachable: ``{from_idx, to_idx, reachable: True, via_doorway_xy,
      path_length_m}``
    - unreachable: ``{from_idx, to_idx, reachable: False, reason}`` where
      ``reason`` is ``"stairs"`` (cross-story) or ``"blocked"`` (no path).

    ``door_state`` is layered on afterward by the door-open verification step;
    it is not set here.
    """
    edges: list[dict[str, Any]] = []
    for i, j in _candidate_pairs(rooms, room_adjacency):
        a, b = rooms[i], rooms[j]
        if int(a.get("story", 0)) != int(b.get("story", 0)):
            edges.append(
                {"from_idx": i, "to_idx": j, "reachable": False, "reason": "stairs"}
            )
            continue
        src = room_representative_xy(
            a, free_space, origin_xy=grid_origin_xy, grid_res=grid_res
        )
        dst = room_representative_xy(
            b, free_space, origin_xy=grid_origin_xy, grid_res=grid_res
        )
        try:
            path = plan_path(
                np.asarray(src),
                np.asarray(dst),
                free_space,
                grid_res=grid_res,
                grid_origin_xy=grid_origin_xy,
                discretization_m=discretization_m,
                snap_radius_m=snap_radius_m,
            )
        except (NoPathError, InvalidEndpointError):
            edges.append(
                {"from_idx": i, "to_idx": j, "reachable": False, "reason": "blocked"}
            )
            continue
        doorway = find_room_boundary_crossing(path, a, b)
        edges.append(
            {
                "from_idx": i,
                "to_idx": j,
                "reachable": True,
                "via_doorway_xy": [float(doorway[0]), float(doorway[1])],
                "path_length_m": round(_path_length_m(path), 4),
            }
        )
    return edges


def is_multi_room_incompatible(edges: Sequence[dict[str, Any]]) -> bool:
    """True iff the scene has cross-room candidate edges but none is reachable.

    Single-pair blockage is non-fatal — the mission generator can still emit
    missions over the reachable subset, so a scene is only flagged
    incompatible when *no* cross-room edge is reachable.
    """
    cross = [e for e in edges if e.get("from_idx") != e.get("to_idx")]
    if not cross:
        return False
    return not any(e.get("reachable") for e in cross)


def connectivity_matrix(
    n_rooms: int, edges: Sequence[dict[str, Any]]
) -> np.ndarray:
    """Symmetric ``(n, n)`` bool reachability matrix from the edge list.

    A reporting / debugging convenience — the authored contract is the edge
    list, not the matrix.
    """
    mat = np.zeros((n_rooms, n_rooms), dtype=bool)
    for e in edges:
        if not e.get("reachable"):
            continue
        i, j = int(e["from_idx"]), int(e["to_idx"])
        if 0 <= i < n_rooms and 0 <= j < n_rooms:
            mat[i, j] = mat[j, i] = True
    return mat
