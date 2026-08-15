"""Agent configurations for Strafer navigation task.

This module provides RL algorithm configurations for training:
- RSL-RL (PPO) with standard MLP, RNN, or depth-RNN network

The depth runner uses ``StraferDepthRNNModel`` (extends rsl_rl's RNNModel
with integrated depth encoding) and ``AffineBetaDistribution`` for bounded
``[-1, 1]`` actions.  Both are resolved automatically by rsl_rl's
``resolve_callable()`` from the config's ``class_name`` field — no manual
registration is needed.
"""

from .rsl_rl_ppo_cfg import STRAFER_PPO_RUNNER_CFG, STRAFER_PPO_RECURRENT_RUNNER_CFG, STRAFER_PPO_DEPTH_RUNNER_CFG
from .distributions import AffineBetaDistribution, BetaDistributionCfg
from .depth_rnn_model import StraferDepthRNNModel, StraferDepthRNNModelCfg
from .depth_encoders import SpatialSoftArgmax, DeFMDepthEncoder, DepthEncoder, create_depth_encoder
from .demo_buffer import ExpertDemoBuffer


__all__ = [
    "STRAFER_PPO_RUNNER_CFG",
    "STRAFER_PPO_RECURRENT_RUNNER_CFG",
    "STRAFER_PPO_DEPTH_RUNNER_CFG",
    "AffineBetaDistribution",
    "BetaDistributionCfg",
    "StraferDepthRNNModel",
    "StraferDepthRNNModelCfg",
    "SpatialSoftArgmax",
    "DeFMDepthEncoder",
    "DepthEncoder",
    "create_depth_encoder",
    "ExpertDemoBuffer",
]
