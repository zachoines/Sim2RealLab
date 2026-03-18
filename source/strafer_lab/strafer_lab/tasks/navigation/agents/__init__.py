"""Agent configurations for Strafer navigation task.

This module provides RL algorithm configurations for training:
- RSL-RL (PPO) with standard MLP or CNN-MLP hybrid network

To use the CNN-MLP hybrid (StraferActorCritic) with rsl_rl, the class
must be injected into the runner's eval namespace before training starts.
Call ``register_strafer_network()`` once at startup.
"""

from .rsl_rl_ppo_cfg import STRAFER_PPO_RUNNER_CFG, STRAFER_PPO_LSTM_RUNNER_CFG, STRAFER_PPO_DEPTH_RUNNER_CFG
from .strafer_network import StraferActorCritic
from .bc_loss import register_bc_loss, register_dapg_loss


def register_strafer_network():
    """Inject StraferActorCritic into rsl_rl so eval(class_name) resolves."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.StraferActorCritic = StraferActorCritic


__all__ = [
    "STRAFER_PPO_RUNNER_CFG",
    "STRAFER_PPO_LSTM_RUNNER_CFG",
    "STRAFER_PPO_DEPTH_RUNNER_CFG",
    "StraferActorCritic",
    "register_strafer_network",
    "register_bc_loss",
    "register_dapg_loss",
]
