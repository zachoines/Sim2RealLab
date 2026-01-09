"""
Strafer Lab - Isaac Lab extension for Gobilda Strafer mecanum wheel robot.

This extension provides:
- Robot asset configuration (ArticulationCfg)
- RL training environments for navigation tasks
- ROS2 bridge integration for sim-to-real deployment
"""

# Register Gym environments
from .tasks import *

# Make assets available at package level
from .assets import STRAFER_CFG

__version__ = "0.1.0"
