"""Agent configurations for Strafer navigation task.

This module provides RL algorithm configurations for training:
- RSL-RL (PPO) with standard MLP, RNN, or depth-RNN network
- Modular auxiliary loss architecture (DAPG, GAIL) via StraferPPO

The depth runner uses ``StraferDepthRNNModel`` (extends rsl_rl's RNNModel
with integrated depth encoding) and ``AffineBetaDistribution`` for bounded
``[-1, 1]`` actions.  Both are resolved automatically by rsl_rl's
``resolve_callable()`` from the config's ``class_name`` field — no manual
registration is needed.

For auxiliary losses (DAPG, GAIL), call ``install_strafer_ppo()`` once,
then register auxiliary modules via ``register_auxiliary()``.
"""

from .rsl_rl_ppo_cfg import STRAFER_PPO_RUNNER_CFG, STRAFER_PPO_LSTM_RUNNER_CFG, STRAFER_PPO_DEPTH_RUNNER_CFG
from .distributions import AffineBetaDistribution, BetaDistributionCfg
from .depth_rnn_model import StraferDepthRNNModel, StraferDepthRNNModelCfg
from .depth_encoders import SpatialSoftArgmax, DeFMDepthEncoder, DepthEncoder, create_depth_encoder
from .strafer_ppo import AuxiliaryLoss, install_strafer_ppo, register_auxiliary
from .demo_buffer import ExpertDemoBuffer
from .aux_dapg import DAPGAuxiliary
from .aux_gail import GAILAuxiliary


__all__ = [
    "STRAFER_PPO_RUNNER_CFG",
    "STRAFER_PPO_LSTM_RUNNER_CFG",
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
    "AuxiliaryLoss",
    "install_strafer_ppo",
    "register_auxiliary",
    "DAPGAuxiliary",
    "GAILAuxiliary",
]
