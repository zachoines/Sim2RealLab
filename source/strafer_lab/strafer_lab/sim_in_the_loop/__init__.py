"""DGX-side sim-in-the-loop data capture harness.

Drives Isaac Sim through navigation missions via the bundled ROS2 bridge
(``strafer_lab.bridge``) and the Jetson autonomy stack
(``strafer_autonomy.executor``), capturing reachability-labeled frames
into a ``frames.jsonl`` dataset that the existing description and
SFT-prep pipelines can consume unchanged.

Layout:
  - :mod:`strafer_lab.sim_in_the_loop.mission`: ``MissionSpec`` +
    ``MissionGenerator`` reading scenes' ``scene_metadata.json``.
  - :mod:`strafer_lab.sim_in_the_loop.extras`: helper to attach
    reachability + mission metadata to ``PerceptionFrameWriter`` frames.
  - :mod:`strafer_lab.sim_in_the_loop.harness`: ``SimInTheLoopHarness``
    pure-Python orchestrator with injectable env / mission-API adapters
    so the run loop is fully unit-testable without Isaac Sim or rclpy.

The runtime adapters that wire the harness to a live Isaac Sim env and
to the Jetson's ``execute_mission`` ``rclpy.action.ActionClient`` are
provided separately by the launch script that owns the simulator and
ROS context — nothing in this package imports ``omni`` or ``rclpy``.
"""

from strafer_lab.sim_in_the_loop.extras import make_episode_extras
from strafer_lab.sim_in_the_loop.harness import (
    EpisodeOutcome,
    HarnessConfig,
    MissionStatus,
    SimInTheLoopHarness,
)
from strafer_lab.sim_in_the_loop.mission import MissionGenerator, MissionSpec

__all__ = [
    "EpisodeOutcome",
    "HarnessConfig",
    "MissionGenerator",
    "MissionSpec",
    "MissionStatus",
    "SimInTheLoopHarness",
    "make_episode_extras",
]
