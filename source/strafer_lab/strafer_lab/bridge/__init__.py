"""Isaac Sim ROS2 bridge wiring for sim-in-the-loop runs.

This package configures the bundled ``isaacsim.ros2.bridge`` extension to
expose the Strafer's simulated sensors on the same topics the real robot's
Jetson stack consumes, so Nav2 / RTAB-Map / the VLM grounding client can
run unmodified against the DGX-hosted simulator.

Layout:
  - :mod:`strafer_lab.bridge.config` — pure-Python dataclass describing which
    prims, topics and frame IDs the bridge should wire. Importable without
    Isaac Sim, unit-testable.
  - :mod:`strafer_lab.bridge.graph` — OmniGraph builder. Imports Isaac Sim
    runtime modules (``omni.graph.core``) and therefore only works inside
    a live Kit process.
"""

from strafer_lab.bridge.config import BridgeConfig, CameraStreamConfig

__all__ = ["BridgeConfig", "CameraStreamConfig"]
