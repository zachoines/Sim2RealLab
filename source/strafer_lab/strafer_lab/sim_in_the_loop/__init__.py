"""DGX-side sim-in-the-loop data capture harness.

Drives Isaac Sim through navigation missions via the bundled ROS2 bridge
(``strafer_lab.bridge``) and the Jetson autonomy stack
(``strafer_autonomy.executor``), recording each mission as one episode
of a LeRobot v3 dataset via
:class:`strafer_lab.tools.lerobot_writer.StraferLeRobotWriter`.

Layout:
  - :mod:`strafer_lab.sim_in_the_loop.mission`: ``MissionSpec`` +
    ``MissionGenerator`` reading scenes' ``scene_metadata.json``.
  - :mod:`strafer_lab.sim_in_the_loop.harness`: ``SimInTheLoopHarness``
    pure-Python orchestrator with injectable env / mission-API /
    recorder adapters so the run loop is fully unit-testable without
    Isaac Sim or rclpy.
  - :mod:`strafer_lab.sim_in_the_loop.lerobot_recorder`:
    ``BridgeLeRobotRecorder`` mapping the harness episode lifecycle onto
    the LeRobot writer, plus the /cmd_vel-silence watchdog.

The runtime adapters that wire the harness to a live Isaac Sim env and
to the Jetson's ``execute_mission`` ``rclpy.action.ActionClient`` are
provided separately by the launch script that owns the simulator and
ROS context — nothing in this package imports ``omni`` or ``rclpy``.
"""

from strafer_lab.sim_in_the_loop.harness import (
    EpisodeMeta,
    EpisodeOutcome,
    FrameBundle,
    HarnessConfig,
    MissionStatus,
    SimInTheLoopHarness,
)
from strafer_lab.sim_in_the_loop.lerobot_recorder import (
    BridgeLeRobotRecorder,
    CmdVelGraceWatch,
)
from strafer_lab.sim_in_the_loop.mission import MissionGenerator, MissionSpec

__all__ = [
    "BridgeLeRobotRecorder",
    "CmdVelGraceWatch",
    "EpisodeMeta",
    "EpisodeOutcome",
    "FrameBundle",
    "HarnessConfig",
    "MissionGenerator",
    "MissionSpec",
    "MissionStatus",
    "SimInTheLoopHarness",
]
