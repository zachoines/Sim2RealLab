"""Offline free-text mission generator — the canonical ``mission_queue.yaml`` producer.

Per scene this turns the embedded ``customData`` metadata (``rooms[]`` +
``objects[]`` + the verified ``connectivity[]`` graph) plus the cached
occupancy grid into a list of mission rows shaped for
:mod:`strafer_lab.tools.mission_queue` — each round-trips through
``parse_mission_row``. The bridge dispatches ``mission_text``; the scripted
driver consumes ``planned_path``; teleop shows ``mission_text``.

Path planning routes through the project's one shared planner
(:func:`strafer_lab.tasks.navigation.path_planner.plan_path`) over the shared
cached-occupancy seam (:func:`scene_connectivity.load_occupancy` +
:func:`scene_connectivity.occupancy_to_free_space`) — no second pathfinder,
no re-rasterization. Cross-room missions are gated by ``connectivity[]``.

The pipeline is split so the heavy/optional passes are injectable callables:
the LLM waypoint planner, the paraphrase model, and the start-frame VLM
grounding pass all degrade to deterministic, model-free behaviour when their
runner is absent, so the generator's core logic is exercisable without
``transformers``, a GPU, or a rendered frame. The single ``pxr`` touch-point
(reading metadata from a USD) and the occupancy load live in
:func:`load_scene_inputs`; everything below it operates on the plain
:class:`SceneInputs` dict so unit tests build scenes in memory.

Placement: this module is importable library code; the runnable cross-product
CLI is ``scripts/build_mission_corpus.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from strafer_lab.tasks.navigation.path_planner import (
    InvalidEndpointError,
    NoPathError,
    plan_path,
)
from strafer_lab.tools import mission_text_builder, scene_classes, scene_connectivity as sc

# Bump when the row schema, validator, or fallback logic changes in a way that
# should invalidate cached queues even at an unchanged LLM seed / template.
GENERATOR_VERSION = "3"

# The mission-source tag the writer stamps for queue-sourced episodes.
SOURCE_MISSION_SOURCE = "queue"

MODES = ("endpoint", "path-shape", "mixed")

# Match the connectivity pass so the oracle snaps endpoints the same way the
# graph that gates it does.
ORACLE_SNAP_RADIUS_M = sc.DEFAULT_SNAP_RADIUS_M  # 1.0 m

# Default checkpoints. Text-only for waypoint planning + paraphrase; a VL model
# for the start-frame grounding pass. All are loaded lazily and only when a run
# opts in — the model-free fallbacks need none of them.
DEFAULT_PLANNER_MODEL = "Qwen/Qwen3-4B"
DEFAULT_PARAPHRASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_GROUNDING_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


class MissionGeneratorError(Exception):
    """Raised on an unusable scene input (missing geometry, stale occupancy)."""


class StaleOccupancyError(MissionGeneratorError):
    """Cached occupancy grid does not match the scene USD it was built from."""


# Runner protocols (all optional; absence selects the model-free fallback):
#   waypoint planner: (prompt, seed) -> raw JSON string {"waypoints", "rationale"}
#   paraphrase:       (mission_text, n, seed) -> list[str]
#   grounding:        (frame, mission_text) -> "yes" | "partial" | "no"
WaypointRunner = Callable[[str, int], str]
ParaphraseRunner = Callable[[str, int, int], list[str]]
GroundingRunner = Callable[[Any, str], str]


@dataclass(frozen=True)
class GeneratorConfig:
    """Tunables for one generation pass."""

    mode: str = "mixed"
    generator_version: str = GENERATOR_VERSION
    llm_seed: int = 42
    start_pose_seeds: int = 1
    paraphrases_per_mission: int = 3
    target_proximity_m: float = 0.5
    landmark_band_m: float = 1.5
    max_retries: int = 3
    snap_radius_m: float = ORACLE_SNAP_RADIUS_M
    discretization_m: float = sc.DEFAULT_DISCRETIZATION_M
    cross_room_default: bool = True
    # Drop targets whose only anchor is the un-groundable coordinate fallback: a
    # camera-grounded VLM cannot read a raw coordinate off an image, so a mission
    # must name an object it can actually see. Default on; flip off for A/B
    # measurement of the un-filtered pool.
    require_groundable: bool = True
    ground_start_frame: bool = False
    planner_model: str = DEFAULT_PLANNER_MODEL
    paraphrase_model: str = DEFAULT_PARAPHRASE_MODEL
    grounding_model: str = DEFAULT_GROUNDING_MODEL

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")


@dataclass
class SceneInputs:
    """Everything the per-scene generator needs, decoupled from how it loaded.

    ``free_space`` is the robot-radius-inflated boolean grid (True = passable)
    already run through :func:`scene_connectivity.occupancy_to_free_space`; it
    is ``None`` for scenes with no cached occupancy (single-room legacy set),
    in which case the generator emits same-room rows without a ``planned_path``.
    """

    scene_name: str
    metadata: dict[str, Any]
    scene_seed: int | None = None
    free_space: np.ndarray | None = None
    grid_res: float = sc.DEFAULT_RESOLUTION_M
    grid_origin_xy: tuple[float, float] = (0.0, 0.0)
    occupancy_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rooms(self) -> list[dict[str, Any]]:
        return list(self.metadata.get("rooms", []))

    @property
    def objects(self) -> list[dict[str, Any]]:
        return list(self.metadata.get("objects", []))

    @property
    def connectivity(self) -> list[dict[str, Any]]:
        # load() does not default this key — a pre-connectivity scene omits it.
        return list(self.metadata.get("connectivity", []))


@dataclass
class MissionGenStats:
    """Per-scene counters surfaced in the corpus report."""

    scene_name: str
    scene_seed: int | None
    multi_room: bool
    emitted: int = 0
    cross_room: int = 0
    same_room: int = 0
    rejected: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    llm_retries: int = 0
    path_shape_unsatisfied: int = 0
    start_frame_grounded_yes: int = 0
    start_frame_grounded_no: int = 0
    start_frame_grounded_skipped: int = 0

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1


@dataclass
class MissionQueueResult:
    rows: list[dict[str, Any]]
    stats: MissionGenStats


# ---------------------------------------------------------------------------
# Loading (the pxr + occupancy touch-points)
# ---------------------------------------------------------------------------


def check_occupancy_freshness(occ_meta: dict[str, Any], usd_path: Path | str) -> None:
    """Raise :class:`StaleOccupancyError` if the cached grid predates the USD.

    ``occupancy.json`` records the source USD's path + mtime + size; a mismatch
    means the grid was built from different geometry than the scene now on disk.
    A missing identity (older grids) is treated as fresh — there is nothing to
    compare against.
    """
    recorded_mtime = occ_meta.get("usd_mtime_ns")
    recorded_size = occ_meta.get("usd_size")
    if recorded_mtime is None and recorded_size is None:
        return
    usd_path = Path(usd_path)
    if not usd_path.exists():
        return
    stat = usd_path.stat()
    if recorded_size is not None and int(recorded_size) != stat.st_size:
        raise StaleOccupancyError(
            f"occupancy grid for {usd_path} is stale: recorded size "
            f"{recorded_size} != current {stat.st_size}. Re-run the connectivity "
            "validation to regenerate occupancy.npy."
        )
    if recorded_mtime is not None and int(recorded_mtime) != stat.st_mtime_ns:
        raise StaleOccupancyError(
            f"occupancy grid for {usd_path} is stale: recorded mtime "
            f"{recorded_mtime} != current {stat.st_mtime_ns}. Re-run the "
            "connectivity validation to regenerate occupancy.npy."
        )


def load_scene_inputs(
    *,
    scene: str | None = None,
    usd_override: Path | str | None = None,
    scenes_root: Path | str = "Assets/generated/scenes",
    scene_seed: int | None = None,
    allow_stale_occupancy: bool = False,
) -> SceneInputs:
    """Resolve a scene's USD, read its embedded metadata, and load its occupancy.

    The single ``pxr`` touch-point. The occupancy grid is loaded through the
    shared :func:`scene_connectivity.load_occupancy` seam; a scene with no
    cached grid yields ``free_space=None`` (single-room fallback). Raises
    :class:`MissionGeneratorError` when the metadata cannot be read.
    """
    from strafer_lab.tools.scene_metadata_reader import (
        SceneMetadataError,
        load as load_metadata,
    )
    from strafer_lab.tools.scene_paths import resolve_scene_usd_path

    usd_path = resolve_scene_usd_path(
        scene=scene, usd_override=usd_override, search_root=scenes_root
    )
    try:
        metadata = load_metadata(usd_path)
    except SceneMetadataError as exc:
        raise MissionGeneratorError(str(exc)) from exc

    scene_name = scene or Path(usd_path).stem
    scene_dir = Path(scenes_root) / scene_name

    free_space: np.ndarray | None = None
    grid_res = sc.DEFAULT_RESOLUTION_M
    grid_origin_xy = (0.0, 0.0)
    occ_meta: dict[str, Any] = {}
    try:
        occ = sc.load_occupancy(scene_dir)
    except FileNotFoundError:
        occ = None
    if occ is not None:
        if not allow_stale_occupancy:
            check_occupancy_freshness(occ.meta, usd_path)
        free_space = sc.occupancy_to_free_space(occ.grid, grid_res=occ.resolution_m)
        grid_res = occ.resolution_m
        grid_origin_xy = occ.origin_xy
        occ_meta = occ.meta

    return SceneInputs(
        scene_name=scene_name,
        metadata=metadata,
        scene_seed=scene_seed,
        free_space=free_space,
        grid_res=grid_res,
        grid_origin_xy=grid_origin_xy,
        occupancy_meta=occ_meta,
    )


# ---------------------------------------------------------------------------
# Scene geometry helpers (room membership, free-point sampling, scene summary)
# ---------------------------------------------------------------------------


def _valid_position(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    pos = obj.get("position_3d")
    if not isinstance(pos, (list, tuple)) or len(pos) < 3:
        return None
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if abs(x) < 1e-3 and abs(y) < 1e-3 and abs(z) < 1e-3:
        return None  # origin sentinel == no valid position
    return (x, y, z)


def room_index_of(x: float, y: float, rooms: Sequence[dict[str, Any]]) -> int | None:
    """Index of the room footprint containing ``(x, y)``, or ``None``.

    Objects in the embedded metadata carry ``room_idx=None``, so membership is
    resolved here against the room polygons rather than read off the object.
    """
    for i, room in enumerate(rooms):
        if sc.point_in_polygon(x, y, room.get("footprint_xy") or []):
            return i
    return None


def _xy_to_cell(x: float, y: float, origin_xy: tuple[float, float], res: float) -> tuple[int, int]:
    return (
        int(math.floor((x - origin_xy[0]) / res)),
        int(math.floor((y - origin_xy[1]) / res)),
    )


def free_points_in_room(
    room: dict[str, Any],
    free_space: np.ndarray,
    *,
    origin_xy: tuple[float, float],
    grid_res: float,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Sample up to ``count`` free interior points of a room.

    A point sampler over the inflated free grid (not a pathfinder): used to seed
    start poses. Returns the room representative first so a single-seed run is
    deterministic and centred, then random distinct free cells inside the
    footprint.
    """
    rep = sc.room_representative_xy(
        room, free_space, origin_xy=origin_xy, grid_res=grid_res
    )
    # The representative point can fall back to the raw centroid for a room with
    # no free interior; only seed from it when it is actually navigable, so an
    # emitted path never starts inside an obstacle.
    points: list[tuple[float, float]] = (
        [rep] if _cell_is_free(free_space, rep[0], rep[1], origin_xy, grid_res) else []
    )
    if points and count <= 1:
        return points

    footprint = room.get("footprint_xy") or []
    if not footprint:
        return points
    fp = np.asarray(footprint, dtype=float)
    rows, cols = free_space.shape
    (r0, c0) = _xy_to_cell(fp[:, 0].min(), fp[:, 1].min(), origin_xy, grid_res)
    (r1, c1) = _xy_to_cell(fp[:, 0].max(), fp[:, 1].max(), origin_xy, grid_res)
    candidates: list[tuple[float, float]] = []
    for r in range(max(0, r0), min(rows, r1 + 1)):
        for c in range(max(0, c0), min(cols, c1 + 1)):
            if not free_space[r, c]:
                continue
            x = r * grid_res + origin_xy[0] + grid_res / 2.0
            y = c * grid_res + origin_xy[1] + grid_res / 2.0
            if sc.point_in_polygon(x, y, footprint):
                candidates.append((x, y))
    if candidates:
        order = rng.permutation(len(candidates))
        for idx in order:
            pt = candidates[int(idx)]
            if pt not in points:
                points.append(pt)
            if len(points) >= count:
                break
    return points[:count]


def _min_dist_to_polyline(point: tuple[float, float], path: np.ndarray) -> float:
    """Minimum Euclidean distance from a point to a polyline (vectorized)."""
    pts = np.asarray(path, dtype=float)
    p = np.asarray(point, dtype=float)
    if len(pts) < 2:
        return float(np.hypot(*(pts[0] - p))) if len(pts) else float("inf")
    a, b = pts[:-1], pts[1:]
    ab = b - a
    ap = p - a
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    t = np.clip((ap * ab).sum(1) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.hypot(*(proj - p).T).min())


def _path_room_sequence(path: np.ndarray, rooms: Sequence[dict[str, Any]]) -> list[int]:
    """Ordered, de-duplicated room indices a path passes through (coarse sample)."""
    pts = np.asarray(path, dtype=float)
    if len(pts) == 0 or not rooms:
        return []
    stride = max(1, len(pts) // 64)
    seq: list[int] = []
    for k in range(0, len(pts), stride):
        idx = room_index_of(float(pts[k][0]), float(pts[k][1]), rooms)
        if idx is not None and (not seq or seq[-1] != idx):
            seq.append(idx)
    return seq


def _path_transit_room(
    path: np.ndarray,
    rooms: Sequence[dict[str, Any]],
    start_idx: int | None,
    target_idx: int | None,
) -> int | None:
    """Index of an intermediate room the path crosses (neither start nor target)."""
    for idx in _path_room_sequence(path, rooms):
        if idx != start_idx and idx != target_idx:
            return idx
    return None


def _nearest_path_landmark(
    path: np.ndarray,
    objects: Sequence[dict[str, Any]],
    exclude_obj: dict[str, Any] | None,
    rooms: Sequence[dict[str, Any]],
    *,
    band_m: float,
) -> str | None:
    """Label of the object the path passes closest to, with no side, or ``None``.

    A groundable proximity landmark: the nearest object within ``band_m`` of the
    planned path, returned only when its label is unambiguous along the route
    (no other object shares the label among the traversed rooms or within the
    band), so ``passing the {label}`` names exactly one thing. It never asserts
    which side the landmark is on, nor any compass bearing — neither is
    groundable for a holonomic, compass-less robot whose mission text is authored
    before the trajectory exists.
    """
    near: list[tuple[float, str]] = []
    for obj in objects:
        if obj is exclude_obj:
            continue
        pos = _valid_position(obj)
        if pos is None:
            continue
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            continue
        dist = _min_dist_to_polyline((pos[0], pos[1]), path)
        if dist <= band_m:
            near.append((dist, label))
    if not near:
        return None
    near.sort(key=lambda t: t[0])
    best_label = near[0][1]
    traversed = set(_path_room_sequence(path, rooms))
    same = 0
    for obj in objects:
        pos = _valid_position(obj)
        if pos is None or str(obj.get("label", "")).strip().lower() != best_label:
            continue
        ri = room_index_of(pos[0], pos[1], rooms)
        if ri in traversed or _min_dist_to_polyline((pos[0], pos[1]), path) <= band_m:
            same += 1
    return best_label if same == 1 else None


def build_scene_summary(inputs: SceneInputs, *, reachable_pairs: set[tuple[int, int]]) -> str:
    """Distil the scene into compact structured prose for the LLM prompt.

    Carries room types, the room-type connectivity graph, and which object sits
    in which room — but NO absolute axis system (no scene bounds, no room
    centroids). World coordinates live only in the numeric waypoint channel
    (``build_waypoint_prompt``), never in the scene prose or any instruction, so
    the model is not handed a compass to leak into mission text.
    """
    rooms = inputs.rooms
    lines: list[str] = ["Rooms:"]
    for i, room in enumerate(rooms):
        lines.append(f"  - [{i}] {room.get('room_type', 'room')}")
    if reachable_pairs:
        lines.append("Connectivity (reachable room pairs):")
        for i, j in sorted(reachable_pairs):
            if i < len(rooms) and j < len(rooms):
                lines.append(
                    f"  - {rooms[i].get('room_type', i)} [{i}] <-> "
                    f"{rooms[j].get('room_type', j)} [{j}]"
                )
    seen: set[tuple[str, str]] = set()
    obj_lines: list[str] = []
    for obj in inputs.objects:
        pos = _valid_position(obj)
        if pos is None:
            continue
        ri = room_index_of(pos[0], pos[1], rooms)
        if ri is None:
            continue
        label = str(obj.get("label", "")).strip().lower()
        room_type = str(rooms[ri].get("room_type", ri))
        key = (label, room_type)
        # Skip structural classes so the LLM-planner prose matches the target pool.
        if label and label not in scene_classes.STRUCTURAL_CLASSES and key not in seen:
            seen.add(key)
            obj_lines.append(f"  - {label} in {room_type}")
    if obj_lines:
        lines.append("Objects (by room):")
        lines.extend(obj_lines)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Connectivity gate
# ---------------------------------------------------------------------------


def reachable_room_pairs(connectivity: Sequence[dict[str, Any]]) -> set[tuple[int, int]]:
    """Order-insensitive set of reachable ``(i, j)`` room-index pairs."""
    pairs: set[tuple[int, int]] = set()
    for edge in connectivity:
        if not edge.get("reachable"):
            continue
        i, j = int(edge["from_idx"]), int(edge["to_idx"])
        pairs.add((min(i, j), max(i, j)))
    return pairs


def is_scene_multi_room(inputs: SceneInputs) -> bool:
    """A scene is multi-room when two *ground-floor* rooms are reachably linked.

    The robot operates on story 0, so an upper-floor reachable pair (two rooms
    on the same upper story, unreachable from the ground) does not make the scene
    multi-room for mission generation.
    """
    rooms = inputs.rooms
    if len(rooms) <= 1:
        return False
    if inputs.metadata.get("multi_room_incompatible"):
        return False
    return any(
        int(rooms[i].get("story", 0)) == 0 and int(rooms[j].get("story", 0)) == 0
        for (i, j) in reachable_room_pairs(inputs.connectivity)
        if i < len(rooms) and j < len(rooms)
    )


# ---------------------------------------------------------------------------
# LLM-as-planner prompt + JSON schema + validation
# ---------------------------------------------------------------------------


WAYPOINT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "waypoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["waypoints"],
}

# Few-shot template covering endpoint, single-room path-shape, and cross-room
# transit. Hashed into the cache key so a template edit invalidates cached rows.
_FEWSHOT_TEMPLATE = """You plan collision-free waypoint paths for a ground robot in an indoor scene.
Given the scene summary, a start pose, and a target, emit a JSON object
{"waypoints": [{"x": <m>, "y": <m>}, ...], "rationale": "<one sentence>"}.
The first waypoint is the start, the last is at the target. Stay on open floor,
honour any landmark or room phrasing in the instruction, and only cross between
rooms through reachable doorways. Coordinates are an opaque planning frame with
no fixed orientation; in the rationale describe routes only by the landmarks and
rooms they pass, never by absolute headings, surfaces, or sidedness.

Example (endpoint):
  Instruction: "Go to the chair."
  -> {"waypoints": [{"x": 0.5, "y": 0.5}, {"x": 2.0, "y": 1.0}, {"x": 4.2, "y": 1.8}], "rationale": "Shortest open route to the chair."}

Example (single-room landmark):
  Instruction: "Go to the chair, passing the dining table."
  -> {"waypoints": [{"x": 0.5, "y": 0.4}, {"x": 2.5, "y": 0.5}, {"x": 3.6, "y": 1.0}, {"x": 4.2, "y": 1.8}], "rationale": "Routed close past the dining table on the way to the chair."}

Example (cross-room transit):
  Instruction: "Go to the kitchen sink via the dining room."
  -> {"waypoints": [{"x": 1.0, "y": 1.0}, {"x": 3.0, "y": 1.5}, {"x": 5.5, "y": 1.5}, {"x": 7.0, "y": 2.5}], "rationale": "Routed through the dining-room doorway, then into the kitchen."}
"""


def prompt_template_hash(config: GeneratorConfig) -> str:
    """Stable hash of the few-shot template + validator-affecting config.

    Guards the cache: a template or validator-relevant config change yields a
    new hash, so a queue cached under the old template is rebuilt rather than
    silently reused.
    """
    payload = json.dumps(
        {
            "template": _FEWSHOT_TEMPLATE,
            "schema": WAYPOINT_JSON_SCHEMA,
            "target_proximity_m": config.target_proximity_m,
            "snap_radius_m": config.snap_radius_m,
            "discretization_m": config.discretization_m,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def build_waypoint_prompt(
    *,
    scene_summary: str,
    mission_text: str,
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> str:
    return (
        _FEWSHOT_TEMPLATE
        + "\nScene:\n"
        + scene_summary
        + f"\n\nStart: ({start_xy[0]:.2f}, {start_xy[1]:.2f})"
        + f"\nTarget: ({target_xy[0]:.2f}, {target_xy[1]:.2f})"
        + f'\nInstruction: "{mission_text}"\nJSON:'
    )


def _cell_is_free(free: np.ndarray, x: float, y: float, origin_xy, res: float) -> bool:
    r, c = _xy_to_cell(x, y, origin_xy, res)
    rows, cols = free.shape
    return 0 <= r < rows and 0 <= c < cols and bool(free[r, c])


def _segment_navigable(
    free: np.ndarray,
    a: tuple[float, float],
    b: tuple[float, float],
    origin_xy: tuple[float, float],
    res: float,
) -> bool:
    """Line-of-sight predicate: every half-cell sample on a->b is free.

    A geometric check on the inflated free grid, not a planner — oracle and
    fallback paths come only from :func:`plan_path`.
    """
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(length / (res / 2.0)) + 1)
    for i in range(n + 1):
        t = i / n
        if not _cell_is_free(free, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), origin_xy, res):
            return False
    return True


def parse_waypoint_json(raw: str) -> list[tuple[float, float]]:
    """Parse the LLM's JSON reply into a waypoint list; ``[]`` on any failure."""
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except (TypeError, ValueError):
        return []
    out: list[tuple[float, float]] = []
    for wp in data.get("waypoints", []):
        if isinstance(wp, dict) and "x" in wp and "y" in wp:
            out.append((float(wp["x"]), float(wp["y"])))
        elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
            out.append((float(wp[0]), float(wp[1])))
    return out


def validate_waypoints(
    waypoints: Sequence[tuple[float, float]],
    *,
    inputs: SceneInputs,
    target_xy: tuple[float, float],
    config: GeneratorConfig,
) -> tuple[bool, dict[str, bool]]:
    """Validate an LLM waypoint set against the inflated free grid.

    Checks minimum path structure (>= 2 waypoints), bounds, per-waypoint
    navigability, segment line-of-sight, and target proximity. No graph search —
    the oracle/fallback that replaces a rejected set is the one shared
    :func:`plan_path`.
    """
    checks = {"structure_ok": True, "navigable_ok": True, "segment_ok": True, "target_ok": True}
    free = inputs.free_space
    if free is None or len(waypoints) < 2:
        return False, {**checks, "structure_ok": False}

    for x, y in waypoints:
        if not _cell_is_free(free, x, y, inputs.grid_origin_xy, inputs.grid_res):
            checks["navigable_ok"] = False
            break
    for a, b in zip(waypoints, waypoints[1:]):
        if not _segment_navigable(free, a, b, inputs.grid_origin_xy, inputs.grid_res):
            checks["segment_ok"] = False
            break
    if math.hypot(waypoints[-1][0] - target_xy[0], waypoints[-1][1] - target_xy[1]) > config.target_proximity_m:
        checks["target_ok"] = False

    return all(checks.values()), checks


# ---------------------------------------------------------------------------
# Oracle (shared planner) + paraphrase + grounding fallbacks
# ---------------------------------------------------------------------------


def plan_oracle_path(
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
    inputs: SceneInputs,
    config: GeneratorConfig,
) -> np.ndarray | None:
    """Clean shortest path start->target via the one shared planner, or ``None``.

    Oracle paths are never perturbed. Returns ``None`` when the endpoints are
    unreachable / invalid (caller rejects the mission).
    """
    if inputs.free_space is None:
        return None
    try:
        return plan_path(
            np.asarray(start_xy, dtype=float),
            np.asarray(target_xy, dtype=float),
            inputs.free_space,
            grid_res=inputs.grid_res,
            grid_origin_xy=inputs.grid_origin_xy,
            discretization_m=config.discretization_m,
            snap_radius_m=config.snap_radius_m,
        )
    except (NoPathError, InvalidEndpointError):
        return None


def _path_length_m(path: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


_PARAPHRASE_TEMPLATES = (
    "Head to {ref}.",
    "Navigate to {ref}.",
    "Make your way to {ref}.",
    "Approach {ref}.",
    "Drive over to {ref}.",
)


def fallback_paraphrases(mission_text: str, ref: str, n: int) -> list[str]:
    """Deterministic templated paraphrases when no paraphrase model is available.

    Passed the precomputed anchor noun phrase ``ref`` that produced the
    ``mission_text`` arg, so the paraphrases name the same disambiguated object
    and the ``cand != mission_text`` dedupe stays meaningful.
    """
    out: list[str] = []
    for tmpl in _PARAPHRASE_TEMPLATES:
        cand = tmpl.format(ref=ref)
        if cand != mission_text and cand not in out:
            out.append(cand)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Mission text
# ---------------------------------------------------------------------------


def room_type_is_unique(rooms: Sequence[dict[str, Any]], room_type: str | None) -> bool:
    """True iff exactly one room carries ``room_type`` (so naming it is unambiguous).

    Compares on the normalized (``strip().lower()``) room_type so the consumer's
    uniqueness verdict matches the builder's ``_room_type_is_unique`` — otherwise
    a case variant could make the two disagree on whether the room scope fires.
    Idempotent: accepts an already-normalized or a raw room_type.
    """
    if not room_type:
        return False
    rt = str(room_type).strip().lower()
    return sum(1 for r in rooms if str(r.get("room_type", "")).strip().lower() == rt) == 1


def _norm_room_type(room_type: str | None) -> str | None:
    """Normalize a room_type the way the builder does (``strip().lower()``).

    Applied where the generator reads ``room_type`` off a room so the emitted
    text, the room-scope de-dup, and the builder's anchor suffix all agree on
    casing. ``None`` / empty stays ``None``.
    """
    return str(room_type).strip().lower() if room_type else None


def _anchor_names_room(ref: str, target_room: str | None) -> bool:
    """True iff the anchor ``ref`` already carries ``in the {target_room}``.

    The builder's room scope can fold the destination room into the anchor; when
    it has, the consumer must not name the room a second time (avoids
    ``... in the kitchen in the kitchen``).
    """
    return bool(target_room) and f" in the {target_room}" in ref


def endpoint_text(
    ref: str,
    target_room: str | None,
    cross_room: bool,
    *,
    room_unique: bool = False,
) -> str:
    if cross_room and target_room and room_unique and not _anchor_names_room(ref, target_room):
        return f"Go to {ref} in the {target_room}."
    return f"Go to {ref}."


def path_shape_text(
    *,
    ref: str,
    target_room: str | None,
    cross_room: bool,
    room_unique: bool = False,
    landmark: str | None = None,
    transit_room: str | None = None,
    transit_unique: bool = False,
) -> tuple[str, str]:
    """Return ``(mission_text, constraint_type_hint)`` for a path-shape mission.

    ``ref`` is the precomputed groundable anchor noun phrase. Only groundable
    spatial language is emitted: room-type / connectivity phrasing for cross-room
    missions (``via the {transit_room}`` / ``through the doorway into the
    {target_room}``) and landmark proximity for same-room missions (``passing the
    {landmark}``). No compass bearing, no surface reference, and no sidedness —
    none of which a compass-less holonomic robot can ground from mission text
    authored before the trajectory exists.
    """
    if cross_room and target_room and room_unique:
        # Name the destination room only if the anchor has not already.
        room_clause = "" if _anchor_names_room(ref, target_room) else f" in the {target_room}"
        if transit_room and transit_unique:
            return (f"Go to {ref}{room_clause} via the {transit_room}.", "room_transit")
        if room_clause:
            return (f"Go to {ref} through the doorway into the {target_room}.", "room_transit")
        return (f"Go to {ref} through the doorway.", "room_transit")
    if landmark:
        return (f"Go to {ref}, passing the {landmark}.", "landmark_relative")
    return (
        endpoint_text(ref, target_room, cross_room, room_unique=room_unique),
        "none",
    )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def mission_cache_key(
    *,
    scene_seed: int | None,
    start_pose: tuple[float, float, float],
    mission_text: str,
    llm_seed: int,
    template_hash: str,
    generator_version: str,
) -> str:
    payload = json.dumps(
        {
            "scene_seed": scene_seed,
            "start_pose": [round(v, 4) for v in start_pose],
            "mission_text": mission_text,
            "llm_seed": llm_seed,
            "template_hash": template_hash,
            "generator_version": generator_version,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def cache_header(config: GeneratorConfig) -> dict[str, Any]:
    """Identity a cached queue must match to be reused without regeneration."""
    return {
        "generator_version": config.generator_version,
        "prompt_template_hash": prompt_template_hash(config),
        "llm_seed": config.llm_seed,
        "mode": config.mode,
        "planner_model": config.planner_model,
    }


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def _round_xy_list(path: Sequence[Sequence[float]], ndigits: int = 3) -> list[dict[str, float]]:
    return [{"x": round(float(p[0]), ndigits), "y": round(float(p[1]), ndigits)} for p in path]


def _wants_path_shape(mode: str, index: int) -> bool:
    if mode == "endpoint":
        return False
    if mode == "path-shape":
        return True
    return index % 2 == 1  # mixed: alternate deterministically


def build_mission_queue(
    inputs: SceneInputs,
    config: GeneratorConfig | None = None,
    *,
    waypoint_runner: WaypointRunner | None = None,
    paraphrase_runner: ParaphraseRunner | None = None,
    grounding_runner: GroundingRunner | None = None,
    grounding_frame_provider: Callable[[dict[str, Any], tuple[float, float, float]], Any] | None = None,
) -> MissionQueueResult:
    """Generate the mission rows for one scene.

    Each returned row is a plain dict shaped for ``mission_queue.parse_mission_row``.
    The LLM / paraphrase / grounding passes use their runner when supplied and a
    deterministic, model-free fallback otherwise.
    """
    config = config or GeneratorConfig()
    rng = np.random.default_rng(config.llm_seed)
    rooms = inputs.rooms
    connectivity = inputs.connectivity
    pairs = reachable_room_pairs(connectivity)
    multi_room = is_scene_multi_room(inputs)
    summary = build_scene_summary(inputs, reachable_pairs=pairs)
    template_hash = prompt_template_hash(config)

    stats = MissionGenStats(
        scene_name=inputs.scene_name, scene_seed=inputs.scene_seed, multi_room=multi_room
    )
    rows: list[dict[str, Any]] = []
    counter = 0

    targets = [(obj, pos) for obj in inputs.objects if (pos := _valid_position(obj)) is not None]

    for obj, pos in targets:
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            stats.reject("no_label")
            continue
        # Structural surfaces are obstacles, not navigation goals (matches the
        # teleop picker's block list); a structural class is never a target.
        if label in scene_classes.STRUCTURAL_CLASSES:
            stats.reject("structural_target")
            continue

        # Resolve the disambiguating anchor once per target (was recomputed inside
        # each text helper). ``.text`` is threaded into every text call site;
        # ``.groundable`` gates the target: a coordinate-only anchor cannot be
        # grounded from a camera frame, so the mission is dropped under the
        # default ``require_groundable``.
        anchor = mission_text_builder.disambiguate(obj, inputs.objects, rooms)
        if config.require_groundable and not anchor.groundable:
            stats.reject("ungroundable_target")
            continue
        ref = anchor.text

        target_xy = (pos[0], pos[1])
        target_room_idx = room_index_of(target_xy[0], target_xy[1], rooms)
        target_room = _norm_room_type(
            rooms[target_room_idx].get("room_type") if target_room_idx is not None else None
        )
        room_unique = room_type_is_unique(rooms, target_room)

        # The strafer cannot climb stairs and always spawns on the ground floor,
        # so a target on any upper story is unreachable — drop it before planning
        # (cross-story connectivity edges are already marked unreachable, but a
        # same-room mission on an upper floor would otherwise slip through).
        if target_room_idx is not None and int(rooms[target_room_idx].get("story", 0)) != 0:
            stats.reject("target_on_unreachable_floor")
            continue

        # No cached occupancy (legacy single-room set): emit a pathless same-room
        # endpoint row — there is no grid to plan or validate against.
        if inputs.free_space is None:
            row = _emit_row(
                counter,
                inputs,
                config,
                label=label,
                obj=obj,
                ref=ref,
                pos=pos,
                target_room=target_room,
                target_room_idx=target_room_idx,
                start_room=None,
                start_room_idx=None,
                cross_room=False,
                start_pose=None,
                planned_path=[],
                mission_text=endpoint_text(ref, target_room, False, room_unique=room_unique),
                constraint_hint="none",
                paraphrase_runner=paraphrase_runner,
                stats=stats,
                template_hash=template_hash,
                validation=None,
                grounded=None,
            )
            rows.append(row)
            counter += 1
            stats.emitted += 1
            stats.same_room += 1
            continue

        if target_room_idx is None:
            stats.reject("target_outside_rooms")
            continue

        # Choose start room: a reachable other room for cross-room, else the
        # target's own room.
        start_room_idx, cross_room = _choose_start_room(
            target_room_idx, rooms, pairs, multi_room and config.cross_room_default, rng
        )
        start_room = _norm_room_type(
            rooms[start_room_idx].get("room_type") if start_room_idx is not None else None
        )
        start_candidates = (
            free_points_in_room(
                rooms[start_room_idx],
                inputs.free_space,
                origin_xy=inputs.grid_origin_xy,
                grid_res=inputs.grid_res,
                count=config.start_pose_seeds,
                rng=rng,
            )
            if inputs.free_space is not None
            else []
        )
        if not start_candidates:
            stats.reject("no_start_free_space")
            continue

        for start_xy in start_candidates:
            want_shape = _wants_path_shape(config.mode, counter)
            result = _plan_one_mission(
                inputs,
                config,
                label=label,
                target_obj=obj,
                ref=ref,
                target_xy=target_xy,
                target_room=target_room,
                target_room_idx=target_room_idx,
                room_unique=room_unique,
                start_xy=start_xy,
                start_room=start_room,
                start_room_idx=start_room_idx,
                cross_room=cross_room,
                want_path_shape=want_shape,
                scene_summary=summary,
                waypoint_runner=waypoint_runner,
                stats=stats,
            )
            if result is None:
                stats.reject("no_navigable_path")
                continue

            mission_text, planned_path, constraint_hint, validation = result
            yaw = math.atan2(target_xy[1] - start_xy[1], target_xy[0] - start_xy[0])
            start_pose = (round(start_xy[0], 4), round(start_xy[1], 4), round(yaw, 4))

            grounded = _ground_start_frame(
                config,
                obj=obj,
                mission_text=mission_text,
                start_pose=start_pose,
                cross_room=cross_room,
                grounding_runner=grounding_runner,
                grounding_frame_provider=grounding_frame_provider,
                stats=stats,
            )
            if grounded == "rejected":
                stats.reject("target_not_visible_at_start")
                continue

            row = _emit_row(
                counter,
                inputs,
                config,
                label=label,
                obj=obj,
                ref=ref,
                pos=pos,
                target_room=target_room,
                target_room_idx=target_room_idx,
                start_room=start_room,
                start_room_idx=start_room_idx,
                cross_room=cross_room,
                start_pose=start_pose,
                planned_path=planned_path,
                mission_text=mission_text,
                constraint_hint=constraint_hint,
                paraphrase_runner=paraphrase_runner,
                stats=stats,
                template_hash=template_hash,
                validation=validation,
                grounded=grounded,
            )
            rows.append(row)
            counter += 1
            stats.emitted += 1
            stats.cross_room += int(cross_room)
            stats.same_room += int(not cross_room)

    return MissionQueueResult(rows=rows, stats=stats)


def _choose_start_room(
    target_room_idx: int | None,
    rooms: Sequence[dict[str, Any]],
    pairs: set[tuple[int, int]],
    allow_cross: bool,
    rng: np.random.Generator,
) -> tuple[int | None, bool]:
    """Pick a start-room index + cross_room flag for a target in ``target_room_idx``.

    Prefers a reachable room of a *different* type so cross-room missions read
    as genuine transit (``kitchen -> bedroom``) rather than ``bedroom -> bedroom``.
    """
    if target_room_idx is None:
        return None, False
    if allow_cross:
        partners = sorted(
            {j for (i, j) in pairs if i == target_room_idx}
            | {i for (i, j) in pairs if j == target_room_idx}
        )
        partners = [p for p in partners if p < len(rooms)]
        if partners:
            target_type = rooms[target_room_idx].get("room_type")
            different = [p for p in partners if rooms[p].get("room_type") != target_type]
            pool = different or partners
            return int(pool[int(rng.integers(len(pool)))]), True
    return target_room_idx, False


def _plan_one_mission(
    inputs: SceneInputs,
    config: GeneratorConfig,
    *,
    label: str,
    target_obj: dict[str, Any],
    ref: str,
    target_xy: tuple[float, float],
    target_room: str | None,
    target_room_idx: int | None,
    room_unique: bool,
    start_xy: tuple[float, float],
    start_room: str | None,
    start_room_idx: int | None,
    cross_room: bool,
    want_path_shape: bool,
    scene_summary: str,
    waypoint_runner: WaypointRunner | None,
    stats: MissionGenStats,
) -> tuple[str, list[dict[str, float]], str, dict[str, Any]] | None:
    """Plan one (start, target) mission. Returns row payload or ``None`` (reject)."""
    oracle = plan_oracle_path(start_xy, target_xy, inputs, config)
    if oracle is None:
        return None

    if not want_path_shape:
        mission_text = endpoint_text(ref, target_room, cross_room, room_unique=room_unique)
        return mission_text, _round_xy_list(oracle), "none", {
            "structure_ok": True,
            "navigable_ok": True,
            "segment_ok": True,
            "target_ok": True,
            "retries": 0,
            "source": "oracle",
        }

    # Groundable path-shape language derived from the oracle route: a same-room
    # proximity landmark, or the cross-room transit room — never a surface/bearing.
    landmark = _nearest_path_landmark(
        oracle, inputs.objects, target_obj, inputs.rooms, band_m=config.landmark_band_m
    )
    transit_idx = _path_transit_room(oracle, inputs.rooms, start_room_idx, target_room_idx)
    transit_room = _norm_room_type(
        inputs.rooms[transit_idx].get("room_type")
        if transit_idx is not None and transit_idx < len(inputs.rooms)
        else None
    )
    mission_text, constraint_hint = path_shape_text(
        ref=ref,
        target_room=target_room,
        cross_room=cross_room,
        room_unique=room_unique,
        landmark=landmark,
        transit_room=transit_room,
        transit_unique=room_type_is_unique(inputs.rooms, transit_room),
    )

    # LLM-as-planner: try the runner, validate, retry; fall back to the clean
    # oracle path (tagged unsatisfied) on persistent failure or no runner.
    if waypoint_runner is not None and inputs.free_space is not None:
        retries = 0
        for attempt in range(config.max_retries + 1):
            prompt = build_waypoint_prompt(
                scene_summary=scene_summary,
                mission_text=mission_text,
                start_xy=start_xy,
                target_xy=target_xy,
            )
            try:
                raw = waypoint_runner(prompt, config.llm_seed + attempt)
            except Exception:
                raw = ""
            waypoints = parse_waypoint_json(raw)
            # Pin the path to the real start pose so the scripted driver tracks
            # from where the robot actually is, not where the LLM assumed (and so
            # a lone target-only waypoint becomes a usable 2-point path).
            if waypoints and (
                len(waypoints) == 1
                or math.hypot(waypoints[0][0] - start_xy[0], waypoints[0][1] - start_xy[1])
                > config.snap_radius_m
            ):
                waypoints = [start_xy, *waypoints]
            ok, checks = validate_waypoints(
                waypoints, inputs=inputs, target_xy=target_xy, config=config
            )
            if ok:
                checks["retries"] = retries
                checks["source"] = "llm"
                return mission_text, _round_xy_list(waypoints), constraint_hint, checks
            retries += 1
            stats.llm_retries += 1

    stats.path_shape_unsatisfied += 1
    return mission_text, _round_xy_list(oracle), f"{constraint_hint}_unsatisfied", {
        "structure_ok": True,
        "navigable_ok": True,
        "segment_ok": True,
        "target_ok": True,
        "retries": config.max_retries if waypoint_runner is not None else 0,
        "source": "oracle_fallback",
    }


def _ground_start_frame(
    config: GeneratorConfig,
    *,
    obj: dict[str, Any],
    mission_text: str,
    start_pose: tuple[float, float, float],
    cross_room: bool,
    grounding_runner: GroundingRunner | None,
    grounding_frame_provider: Callable[[dict[str, Any], tuple[float, float, float]], Any] | None,
    stats: MissionGenStats,
) -> Any:
    """Run the start-frame visibility check. Returns a JSON-able verdict or "rejected".

    Skips (verdict ``None``) when grounding is disabled or no frame is available
    — the render is the only Kit/GPU-dependent step, so it degrades to a no-op
    headless. A same-room "no" rejects the mission; a cross-room "no" keeps it
    with ``start_frame_grounded=False`` (target discovered via transit).
    """
    if not config.ground_start_frame or grounding_runner is None or grounding_frame_provider is None:
        stats.start_frame_grounded_skipped += 1
        return None
    frame = grounding_frame_provider(obj, start_pose)
    if frame is None:
        stats.start_frame_grounded_skipped += 1
        return None
    verdict = str(grounding_runner(frame, mission_text)).strip().lower()
    if verdict in ("yes", "partial"):
        stats.start_frame_grounded_yes += 1
        return True
    stats.start_frame_grounded_no += 1
    if cross_room:
        return False
    return "rejected"


def _emit_row(
    counter: int,
    inputs: SceneInputs,
    config: GeneratorConfig,
    *,
    label: str,
    obj: dict[str, Any],
    ref: str,
    pos: tuple[float, float, float],
    target_room: str | None,
    target_room_idx: int | None,
    start_room: str | None,
    start_room_idx: int | None,
    cross_room: bool,
    start_pose: tuple[float, float, float] | None,
    planned_path: list[dict[str, float]],
    mission_text: str,
    constraint_hint: str,
    paraphrase_runner: ParaphraseRunner | None,
    stats: MissionGenStats,
    template_hash: str,
    validation: dict[str, Any] | None,
    grounded: Any,
) -> dict[str, Any]:
    """Assemble one QueueMissionRow-shaped dict (required quartet + recognized optionals)."""
    if paraphrase_runner is not None:
        try:
            paraphrases = list(
                paraphrase_runner(mission_text, config.paraphrases_per_mission, config.llm_seed)
            )
        except Exception:
            paraphrases = fallback_paraphrases(mission_text, ref, config.paraphrases_per_mission)
    else:
        paraphrases = fallback_paraphrases(mission_text, ref, config.paraphrases_per_mission)

    mission_id = f"{inputs.scene_name}-{counter:05d}"
    key = (
        mission_cache_key(
            scene_seed=inputs.scene_seed,
            start_pose=start_pose,
            mission_text=mission_text,
            llm_seed=config.llm_seed,
            template_hash=template_hash,
            generator_version=config.generator_version,
        )
        if start_pose is not None
        else None
    )

    scene_seed = int(inputs.scene_seed) if inputs.scene_seed is not None else None
    # Fields without a writer column (mission_id, scene_seed, target_room,
    # start_room, cross_room, planned_path, grounding) are folded here so they
    # survive into the episode's generator_metadata JSON column.
    generator_metadata: dict[str, Any] = {
        "generator_version": config.generator_version,
        "mode": config.mode,
        "llm_model": config.planner_model,
        "llm_seed": config.llm_seed,
        "prompt_template_hash": template_hash,
        "source_mission_source": SOURCE_MISSION_SOURCE,
        "constraint_type_hint": constraint_hint,
        "mission_id": mission_id,
        "scene_seed": scene_seed,
        "target_room": target_room,
        "target_room_idx": int(target_room_idx) if target_room_idx is not None else None,
        "start_room": start_room,
        "start_room_idx": int(start_room_idx) if start_room_idx is not None else None,
        "cross_room": cross_room,
        "target_object_id": str(obj.get("instance_id", -1)),
        "start_frame_grounded": grounded,
        "cache_key": key,
    }
    if validation is not None:
        generator_metadata["waypoint_validation"] = validation
    if planned_path:
        generator_metadata["planned_path"] = planned_path
        generator_metadata["path_length_m"] = round(
            _path_length_m([(p["x"], p["y"]) for p in planned_path]), 4
        )
        generator_metadata["waypoint_count"] = len(planned_path)

    row: dict[str, Any] = {
        "mission_id": mission_id,
        "scene_name": inputs.scene_name,
        "target_label": label,
        "target_position_3d": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
        "mission_text": mission_text,
        "paraphrases": paraphrases,
        "cross_room": cross_room,
        "generator_metadata": generator_metadata,
    }
    if scene_seed is not None:
        row["scene_seed"] = scene_seed
    if start_pose is not None:
        row["start_pose"] = {
            "x": float(start_pose[0]), "y": float(start_pose[1]), "yaw": float(start_pose[2])
        }
    if target_room is not None:
        row["target_room"] = target_room
    if start_room is not None:
        row["start_room"] = start_room
    if planned_path:
        row["planned_path"] = planned_path
    return row


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------


def queue_to_yaml(rows: Sequence[dict[str, Any]]) -> str:
    import yaml

    return yaml.safe_dump(list(rows), sort_keys=False, default_flow_style=False, width=100)


def write_queue(path: Path | str, rows: Sequence[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(queue_to_yaml(rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Default (lazy) model runners — only loaded when a run opts in
# ---------------------------------------------------------------------------


def build_default_waypoint_runner(model_name: str = DEFAULT_PLANNER_MODEL) -> WaypointRunner:
    """Text-only waypoint planner backed by a transformers causal LM (lazy load)."""
    state: dict[str, Any] = {"model": None, "tok": None}

    def _run(prompt: str, seed: int) -> str:
        if state["model"] is None:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer

            state["tok"] = AutoTokenizer.from_pretrained(model_name)
            state["model"] = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype="auto"
            ).eval()
        import torch

        tok, model = state["tok"], state["model"]
        torch.manual_seed(seed)
        messages = [{"role": "user", "content": prompt}]
        # Qwen3 ships hybrid thinking ON by default; a <think> block would consume the
        # generation budget (slow + truncated answers). Disable it for structured output.
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok([text], return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=512, do_sample=False)
        return tok.batch_decode(out[:, ids["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    return _run


def build_default_paraphrase_runner(model_name: str = DEFAULT_PARAPHRASE_MODEL) -> ParaphraseRunner:
    """Text-only paraphrase generator adapted from the description pipeline's loader."""
    runner = build_default_waypoint_runner(model_name)

    def _run(mission_text: str, n: int, seed: int) -> list[str]:
        prompt = (
            f"Rewrite the robot instruction below {n} different ways, keeping the same "
            f"meaning and the same target object. One per line, no numbering.\n\n"
            f'Instruction: "{mission_text}"'
        )
        raw = runner(prompt, seed)
        lines = [ln.strip(" -\t") for ln in raw.splitlines() if ln.strip()]
        return [ln for ln in lines if ln][:n]

    return _run


def build_default_grounding_runner(model_name: str = DEFAULT_GROUNDING_MODEL) -> GroundingRunner:
    """VL start-frame grounding runner (lazy load). The only render-dependent pass."""
    state: dict[str, Any] = {"model": None, "proc": None}

    def _run(frame: Any, mission_text: str) -> str:
        if state["model"] is None:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            state["proc"] = AutoProcessor.from_pretrained(model_name)
            state["model"] = AutoModelForVision2Seq.from_pretrained(
                model_name, device_map="auto", torch_dtype="auto"
            ).eval()
        from PIL import Image

        model, proc = state["model"], state["proc"]
        image = frame if isinstance(frame, Image.Image) else Image.open(frame).convert("RGB")
        question = (
            f'Is the target named in this instruction visible in the image? '
            f'Instruction: "{mission_text}". Answer with one word: yes, partial, or no.'
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[chat], images=[image], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        decoded = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return decoded.strip().lower()

    return _run
