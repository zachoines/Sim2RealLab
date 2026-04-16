"""DGX-side sim-in-the-loop data capture harness.

Drives Isaac Sim through navigation missions via the bundled ROS2 bridge
(``strafer_lab.bridge``) and the Jetson autonomy stack
(``strafer_autonomy.executor``), capturing reachability-labeled frames
into a ``frames.jsonl`` dataset that the existing description and
SFT-prep pipelines can consume unchanged.

Phase 1 (this commit) — pure-Python orchestration:
  - :mod:`strafer_lab.sim_in_the_loop.mission`: ``MissionSpec`` +
    ``MissionGenerator`` reading scenes' ``scene_metadata.json``.
  - :mod:`strafer_lab.sim_in_the_loop.extras`: helper to attach
    reachability + mission metadata to ``PerceptionFrameWriter`` frames.
  - :mod:`strafer_lab.sim_in_the_loop.harness`: ``SimInTheLoopHarness``
    pure-Python orchestrator with injectable env / mission-API callables
    so the run loop is fully unit-testable in ``.venv_vlm`` without
    Isaac Sim or rclpy.

Phase 2 (next) wires the harness's injected callables to the real
Isaac Lab env and the real ``rclpy.action.ActionClient`` for
``execute_mission``. No phase-1 file needs to change for that.
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
