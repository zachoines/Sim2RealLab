"""Agent configurations for Strafer navigation task.

This module provides RL algorithm configurations for training:
- RSL-RL (PPO)
- SKRL (PPO)
"""

from .rsl_rl_ppo_cfg import StraferPPORunnerCfg

__all__ = ["StraferPPORunnerCfg"]
