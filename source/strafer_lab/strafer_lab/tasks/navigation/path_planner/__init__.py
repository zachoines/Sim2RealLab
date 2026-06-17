"""Sim-internal path planning for subgoal-following navigation training.

Plans collision-free waypoint paths on the per-env inflated occupancy grids
the procedural-room generator builds, and tracks a robot's progress along
those paths (closest-point projection, arc-length cursor, lookahead subgoal).

The deployed hybrid backend consumes paths from Nav2's planner server; this
package is the training-time stand-in. Rather than chasing planner parity,
the expected train/deploy planner disagreement is covered by bounded
per-waypoint perturbation (:func:`perturb_waypoints`): the policy trains
against paths noised past the planner-disagreement envelope, so any quirk
smaller than the noise bound is in-distribution at deployment.
"""

from .cursor import PathCursor, PathCursorState
from .planner import (
    InvalidEndpointError,
    NoPathError,
    PathPlanningError,
    perturb_waypoints,
    plan_path,
    resample_polyline,
)

__all__ = [
    "InvalidEndpointError",
    "NoPathError",
    "PathCursor",
    "PathCursorState",
    "PathPlanningError",
    "perturb_waypoints",
    "plan_path",
    "resample_polyline",
]
