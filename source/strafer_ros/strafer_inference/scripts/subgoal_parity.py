#!/usr/bin/env python3
"""Rolling-subgoal pick-parity CLI (bag-replay self-consistency).

Replays a recorded mission through the SAME numpy ``RollingSubgoalGenerator``
the node runs, offline: each recorded ``/plan`` installs a path (rewinding the
arc cursor), and at each recorded ``/strafer/subgoal`` the robot pose from
``/tf`` (map→base_link) is fed through the generator; the recomputed pick must
match the published subgoal within MAP_RESOLUTION*2 (0.10 m) at every tick.

This is a self-consistency check on the deployed generator, not a gym join: in
bridge mode the gym env's SubgoalCommand does not drive the mission (the
generator picks subgoals off its own planned path). See scripts/PARITY_SCHEMA.md.

For a single-goal mission every ``/plan`` ends at the one goal, so all are
installed; pass ``--goal X Y`` to filter plans whose terminal pose is far from
a known goal (mirrors the node's ``/plan`` goal-match guard).

Exit code is 0 on PASS, 1 on FAIL. Run from a sourced ROS 2 + colcon workspace.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

import numpy as np

from strafer_inference import parity as P

_TOPIC_PLAN = "/plan"
_TOPIC_SUBGOAL = "/strafer/subgoal"
# Mirrors subgoal_generator_node._PLAN_GOAL_MATCH_M: a plan whose terminal pose
# is farther than this from the target goal was computed for a different goal.
_PLAN_GOAL_MATCH_M = 0.5


def _path_xy(msg) -> Optional[np.ndarray]:
    if not msg.poses:
        return None
    return np.array(
        [(p.pose.position.x, p.pose.position.y) for p in msg.poses], dtype=np.float64
    )


def _build_events(
    bag: dict[str, list],
    *,
    map_frame: str,
    base_frame: str,
    goal_xy: Optional[tuple[float, float]],
) -> tuple[list[tuple], int, int]:
    """Merge recorded /plan installs and /strafer/subgoal ticks into one
    time-ordered event stream for the replay. Returns
    (events, n_plans_filtered, n_ticks_dropped_no_tf)."""
    from strafer_inference import bag_io

    tf_buf = bag_io.build_tf_buffer(bag.get("/tf", []), bag.get("/tf_static", []))

    # (stamp, order, event) — order 0 = plan, 1 = tick, so a plan and a tick at
    # the same stamp rewind the cursor before the tick reads it.
    keyed: list[tuple[float, int, tuple]] = []
    n_filtered = 0
    for msg in bag.get(_TOPIC_PLAN, []):
        path = _path_xy(msg)
        if path is None:
            continue
        if goal_xy is not None:
            dx = path[-1, 0] - goal_xy[0]
            dy = path[-1, 1] - goal_xy[1]
            if math.hypot(dx, dy) > _PLAN_GOAL_MATCH_M:
                n_filtered += 1
                continue
        keyed.append((bag_io.header_stamp_s(msg), 0, ("plan", path)))

    n_no_tf = 0
    for msg in bag.get(_TOPIC_SUBGOAL, []):
        t = bag_io.header_stamp_s(msg)
        base = bag_io.lookup_base_in_map(tf_buf, map_frame, base_frame, t)
        if base is None:
            n_no_tf += 1
            continue
        (bx, by), _ = base
        published = (msg.pose.position.x, msg.pose.position.y)
        keyed.append((t, 1, ("tick", (bx, by), published, t)))

    keyed.sort(key=lambda k: (k[0], k[1]))
    return [k[2] for k in keyed], n_filtered, n_no_tf


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--lookahead-m", type=float, default=P.SUBGOAL_LOOKAHEAD_M)
    ap.add_argument(
        "--max-path-points",
        type=int,
        default=0,
        help="cap on path length (>=2); 0 = unbounded, matching the node default",
    )
    ap.add_argument("--tol-m", type=float, default=P.SUBGOAL_BOUND_M)
    ap.add_argument(
        "--goal",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="filter /plan messages not ending near this goal (map frame)",
    )
    args = ap.parse_args(argv)

    from strafer_inference import bag_io

    bag = bag_io.read_bag(
        args.bag, {_TOPIC_PLAN, _TOPIC_SUBGOAL, "/tf", "/tf_static"}
    )
    goal_xy = tuple(args.goal) if args.goal else None
    events, n_filtered, n_no_tf = _build_events(
        bag, map_frame=args.map_frame, base_frame=args.base_frame, goal_xy=goal_xy
    )
    max_points = args.max_path_points if args.max_path_points >= 2 else None

    report = P.replay_subgoal_consistency(
        events,
        lookahead_m=args.lookahead_m,
        max_points=max_points,
        tol_m=args.tol_m,
    )
    print(P.format_subgoal_report(report))
    if n_filtered:
        print(f"note: {n_filtered} /plan message(s) filtered (did not end near --goal).")
    if n_no_tf:
        print(f"note: {n_no_tf} /strafer/subgoal tick(s) had no map→base TF and were skipped.")
    print("=" * 60)
    print("RESULT:", "PASS" if report.passed else "FAIL")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
