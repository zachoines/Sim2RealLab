"""Deploy-side rolling-subgoal generator: pure path -> rolling-subgoal pose.

A numpy single-robot reimplementation of the training-time arc-length
cursor (``PathCursor`` in strafer_lab's ``path_planner/cursor.py``). At
training time that torch cursor drives the policy's subgoal observation;
at deploy time the torch version is unimportable (it lives in the
sim/training lane and imports torch + isaaclab). This module reproduces the
same selection rule on the Jetson with numpy only, so the policy sees the
same subgoal pose it was trained on.

Variant-agnostic on purpose: the input is a path plus a robot pose and
the output is a rolling-subgoal pose. It carries no policy/variant
dependency, so the same generator serves any subgoal-following variant.

Anchoring itself lives one level up in the node; this module supplies the
pieces it needs: :func:`arc_length_projection` to seed a cursor by
projection instead of rewinding, and :func:`evaluate_admission` to decide
whether an anchored path may be replaced.

Kept rclpy-free for direct unit testing, mirroring ``watchdog.py`` and
``obs_pipeline.py``. All ROS glue (``/plan`` subscription, TF lookup,
subgoal publishing) lives in the node that wraps this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from strafer_shared.constants import SUBGOAL_LOOKAHEAD_M

# Matches the training cursor's epsilon. Used in three places: the
# projection denominator, the interpolation denominator, and the
# arc<=target segment search. Keep identical so the deploy pick matches.
_EPS = 1e-6


def path_arc_lengths(path: np.ndarray) -> np.ndarray:
    """Cumulative arc length at each waypoint of an ``(N, 2)`` path.

    ``arc[0]`` is always 0.0 and ``arc[-1]`` is the total length. A
    single-point path yields ``[0.0]``.
    """
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) <= 1:
        return np.array([0.0], dtype=np.float64)
    seg_norm = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
    return np.concatenate([[0.0], np.cumsum(seg_norm)])


def arc_length_projection(
    path: np.ndarray,
    robot_xy: np.ndarray,
    arc: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """``(arc_s, cross_track)`` of ``robot_xy`` projected onto ``path``.

    Non-mutating, so it also serves a candidate path the caller has not
    installed. Ties break to the first segment, matching ``torch.argmin``.
    """
    pts = np.asarray(path, dtype=np.float64)
    robot = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    if len(pts) == 0:
        raise ValueError("cannot project onto an empty path")
    if len(pts) == 1:
        return 0.0, float(np.linalg.norm(robot - pts[0]))

    if arc is None:
        arc = path_arc_lengths(pts)

    a = pts[:-1]
    d = pts[1:] - a
    seg_len = arc[1:] - arc[:-1]

    # ``t`` is the clamped [0, 1] position along each segment. The
    # denominator clamps the SQUARED length so a zero-length interior
    # segment projects to its start rather than producing NaN.
    rel = robot[None, :] - a
    t = (rel * d).sum(axis=-1) / np.clip(seg_len ** 2, _EPS, None)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * d
    dist = np.linalg.norm(robot[None, :] - proj, axis=-1)

    closest = int(np.argmin(dist))
    s = float(arc[closest] + t[closest] * seg_len[closest])
    return s, float(dist[closest])


# Stable strings: the node logs them and the counters key off them.
ADMIT_NO_ANCHOR = "no_anchor"
ADMIT_GOAL_CHANGED = "goal_changed"
ADMIT_COLLISION = "anchor_in_collision"
ADMIT_CROSS_TRACK = "cross_track_exceeded"
ADMIT_ROLLING = "rolling_mode"
REJECT_ANCHOR_HELD = "anchor_held"


@dataclass(frozen=True)
class AnchorAdmission:
    """Whether a freshly received plan may replace the anchored path."""

    admit: bool
    reason: str
    detail: str = ""


def evaluate_admission(
    *,
    has_anchor: bool,
    rolling_mode: bool = False,
    goal_changed: bool = False,
    anchor_in_collision: bool = False,
    cross_track_m: Optional[float] = None,
    cross_track_bound_m: float = 0.5,
) -> AnchorAdmission:
    """Does this candidate plan replace the anchored path?

    Usually not. Admission classes, in precedence order: ``rolling_mode``
    (the legacy re-root-on-every-plan fallback), ``no_anchor``,
    ``goal_changed``, ``anchor_in_collision`` (caller-computed -- the
    costmap lives in ROS), ``cross_track_exceeded``. Otherwise the anchor
    is held and the candidate discarded; whether its arrival still counts
    as planner liveness is the caller's business, not this predicate's.
    """
    if rolling_mode:
        return AnchorAdmission(True, ADMIT_ROLLING, "rolling anchoring configured")
    if not has_anchor:
        return AnchorAdmission(True, ADMIT_NO_ANCHOR, "no anchored path held")
    if goal_changed:
        return AnchorAdmission(True, ADMIT_GOAL_CHANGED, "active goal moved")
    if anchor_in_collision:
        return AnchorAdmission(
            True, ADMIT_COLLISION, "anchored path is in collision on the costmap"
        )
    if cross_track_m is not None and cross_track_m > cross_track_bound_m:
        return AnchorAdmission(
            True,
            ADMIT_CROSS_TRACK,
            f"cross-track {cross_track_m:.2f} m > {cross_track_bound_m:.2f} m",
        )
    return AnchorAdmission(False, REJECT_ANCHOR_HELD, "")


@dataclass
class SubgoalState:
    """One ``update`` tick's outputs (single robot).

    Attributes:
        subgoal_xy: (2,) lookahead point on the path, in the path frame.
        subgoal_heading: path tangent direction (rad) at the subgoal. For
            a single-point path this is the bearing from the robot to the
            point instead (no segment tangent exists). The trained policy
            does NOT observe this tangent; it is emitted for pose fidelity.
        cross_track: distance from the robot to its closest path point.
        along_track_progress: monotonic cursor advance since the previous
            tick (>= 0).
        end_distance: distance from the robot to the path's final point.
        cursor_arc: current monotonic arc-length cursor.
        total_arc: total path arc length.
    """

    subgoal_xy: np.ndarray
    subgoal_heading: float
    cross_track: float
    along_track_progress: float
    end_distance: float
    cursor_arc: float
    total_arc: float


class RollingSubgoalGenerator:
    """Tracks a robot's arc-length cursor along one path and emits the
    rolling subgoal a fixed lookahead ahead.

    Stateful across ticks: the cursor advances monotonically and never
    retreats, and a new path rewinds it to zero. ``update`` returns
    ``None`` until a path has been installed -- the deploy side must not
    publish a subgoal before a real plan arrives.
    """

    def __init__(
        self,
        lookahead_m: float = SUBGOAL_LOOKAHEAD_M,
        max_points: Optional[int] = None,
    ) -> None:
        """
        Args:
            lookahead_m: subgoal distance ahead of the projection along
                arc length, clamped to the path end. Defaults to the
                shared ``SUBGOAL_LOOKAHEAD_M`` train/deploy parity surface.
            max_points: optional cap on path length. Paths longer than
                this are truncated head-first (the first ``max_points``
                waypoints are kept, the tail is dropped), mirroring the
                training cursor's fixed-width buffer. ``None`` keeps the
                path as-is.
        """
        if lookahead_m <= 0.0:
            raise ValueError(f"lookahead_m must be > 0; got {lookahead_m}")
        if max_points is not None and max_points < 2:
            raise ValueError(f"max_points must be >= 2 or None; got {max_points}")
        self._lookahead_m = float(lookahead_m)
        self._max_points = max_points
        self._path: Optional[np.ndarray] = None
        self._arc: Optional[np.ndarray] = None
        self._cursor: float = 0.0

    @property
    def lookahead_m(self) -> float:
        """Lookahead distance this generator advertises."""
        return self._lookahead_m

    @property
    def has_path(self) -> bool:
        """True once a path has been installed via :meth:`set_path`."""
        return self._path is not None

    @property
    def total_arc(self) -> float:
        """Total arc length of the current path (0.0 before any path)."""
        if self._arc is None:
            return 0.0
        return float(self._arc[-1])

    @property
    def cursor_arc(self) -> float:
        """Current monotonic arc-length cursor."""
        return self._cursor

    @property
    def path(self) -> Optional[np.ndarray]:
        """The installed path as an ``(N, 2)`` array, or ``None``.

        Returned by reference; do not mutate it.
        """
        return self._path

    @property
    def arc(self) -> Optional[np.ndarray]:
        """Cumulative arc length at each installed waypoint, or ``None``."""
        return self._arc

    def set_path(
        self, path: np.ndarray, *, initial_cursor: Optional[float] = None
    ) -> None:
        """Install a new path and seed the cursor.

        Args:
            path: (N, 2) waypoints, N >= 1, in a single consistent frame.
            initial_cursor: arc-length position to start at, clamped to
                ``[0, total_arc]``. ``None`` rewinds to zero, which is the
                training cursor's behaviour at a goal resample.
        """
        pts = np.asarray(path, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[-1] != 2 or len(pts) < 1:
            raise ValueError(
                f"expected (N, 2) path with N >= 1, got shape {tuple(pts.shape)}"
            )

        # Head-first truncation: keep the first max_points waypoints. No
        # tail padding is needed (unlike the fixed-width training buffer):
        # a shorter stored path is observationally identical because padded
        # segments are zero-length and never win the closest-segment search.
        if self._max_points is not None and len(pts) > self._max_points:
            pts = pts[: self._max_points]

        self._path = pts.copy()
        self._arc = path_arc_lengths(pts)
        if initial_cursor is None:
            self._cursor = 0.0
        else:
            self._cursor = float(
                np.clip(float(initial_cursor), 0.0, float(self._arc[-1]))
            )

    def project(self, robot_xy: np.ndarray) -> Optional[tuple[float, float]]:
        """``(arc_s, cross_track)`` of ``robot_xy`` on the installed path.

        Does NOT advance the cursor. ``None`` before a path is installed.
        """
        if self._path is None:
            return None
        return arc_length_projection(self._path, robot_xy, self._arc)

    def update(
        self,
        robot_xy: np.ndarray,
        lookahead_m: Optional[float] = None,
    ) -> Optional[SubgoalState]:
        """Advance the cursor toward the robot's projection and emit the
        lookahead subgoal.

        Args:
            robot_xy: (2,) robot position in the same frame as the path.
            lookahead_m: optional per-tick override; defaults to the
                generator's configured lookahead.

        Returns:
            A :class:`SubgoalState`, or ``None`` if no path is installed.
        """
        if self._path is None or self._arc is None:
            return None

        robot = np.asarray(robot_xy, dtype=np.float64).reshape(2)
        lookahead = self._lookahead_m if lookahead_m is None else float(lookahead_m)
        path = self._path
        arc = self._arc
        n = len(path)

        # Single-point / degenerate path: the projection is the point
        # itself, the cursor cannot advance, and the heading is the bearing
        # from the robot to the point (there is no segment tangent).
        if n <= 1:
            p0 = path[0]
            cross_track = float(np.linalg.norm(robot - p0))
            new_cursor = max(self._cursor, 0.0)
            progress = new_cursor - self._cursor
            self._cursor = new_cursor
            to_p0 = p0 - robot
            return SubgoalState(
                subgoal_xy=p0.copy(),
                subgoal_heading=float(np.arctan2(to_p0[1], to_p0[0])),
                cross_track=cross_track,
                along_track_progress=progress,
                end_distance=cross_track,
                cursor_arc=new_cursor,
                total_arc=float(arc[-1]),
            )

        # Per-segment geometry: start ``a``, direction ``d``, length from
        # the arc table.
        a = path[:-1]
        d = path[1:] - a
        seg_len = arc[1:] - arc[:-1]

        # Closest segment (first index wins ties, matching torch.argmin) ->
        # cross-track error and the robot's arc-length position: arc to the
        # segment start plus the fraction into it.
        s_closest, cross_track = arc_length_projection(path, robot, arc)

        # Monotonic cursor advance: never retreats if the robot backs up or
        # re-projects nearer the path start.
        new_cursor = max(self._cursor, s_closest)
        progress = new_cursor - self._cursor
        self._cursor = new_cursor

        # Subgoal target = cursor + lookahead, clamped to the path end.
        total = float(arc[-1])
        target = min(new_cursor + lookahead, total)

        # Locate the segment containing the target arc length, then
        # interpolate the subgoal within it and read its tangent as heading.
        j = int(np.count_nonzero(arc <= target + _EPS)) - 1
        j = min(max(j, 0), n - 2)
        seg_l = seg_len[j]
        frac = np.clip((target - arc[j]) / max(seg_l, _EPS), 0.0, 1.0)
        subgoal = a[j] + frac * d[j]
        heading = float(np.arctan2(d[j, 1], d[j, 0]))

        end_distance = float(np.linalg.norm(robot - path[-1]))

        return SubgoalState(
            subgoal_xy=subgoal,
            subgoal_heading=heading,
            cross_track=cross_track,
            along_track_progress=progress,
            end_distance=end_distance,
            cursor_arc=new_cursor,
            total_arc=total,
        )
