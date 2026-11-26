"""Deploy-side rolling-subgoal generator: pure path -> rolling-subgoal pose.

The selection rule lives in :mod:`strafer_shared.rolling_subgoal`: the
arc-length table, the projection, the monotonic lookahead cursor and the
anchor-admission rule, numpy only. This module re-exports them for
``strafer_inference``; all ROS glue (``/plan`` subscription, TF lookup,
subgoal publishing) lives in the node that wraps this.
"""

from strafer_shared.rolling_subgoal import (  # noqa: F401
    ADMIT_COLLISION,
    ADMIT_CROSS_TRACK,
    ADMIT_GOAL_CHANGED,
    ADMIT_NO_ANCHOR,
    ADMIT_ROLLING,
    REJECT_ANCHOR_HELD,
    AnchorAdmission,
    RollingSubgoalGenerator,
    SubgoalState,
    arc_length_projection,
    evaluate_admission,
    path_arc_lengths,
)
