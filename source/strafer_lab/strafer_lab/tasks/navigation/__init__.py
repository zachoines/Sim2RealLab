"""Navigation task for Strafer mecanum wheel robot.

This module registers Gym environments organized by realism level and sensors:

IDEAL (no noise, no motor dynamics - debugging/baselines):
- ``Isaac-Strafer-Nav-v0``: Full RGB+Depth
- ``Isaac-Strafer-Nav-Depth-v0``: Depth-only
- ``Isaac-Strafer-Nav-NoCam-v0``: Proprioceptive-only

REALISTIC (motor dynamics + noise - sim-to-real target):
- ``Isaac-Strafer-Nav-Real-v0``: Full RGB+Depth with realistic dynamics
- ``Isaac-Strafer-Nav-Real-Depth-v0``: Depth-only with realistic dynamics
- ``Isaac-Strafer-Nav-Real-NoCam-v0``: Proprioceptive-only with realistic dynamics

ROBUST (aggressive noise + dynamics - stress-testing):
- ``Isaac-Strafer-Nav-Robust-v0``: Full sensors with extreme noise
- ``Isaac-Strafer-Nav-Robust-Depth-v0``: Depth-only with extreme noise
- ``Isaac-Strafer-Nav-Robust-NoCam-v0``: Proprioceptive-only with extreme noise

INFINIGEN (Phase 6 - offline Infinigen scene geometry):
- ``Isaac-Strafer-Nav-Real-InfinigenDepth-v0``: Realistic depth + Infinigen scenes
- ``Isaac-Strafer-Nav-Robust-InfinigenDepth-v0``: Robust depth + Infinigen scenes

PROCROOM (Phase 7 - procedural primitive rooms):
- ``Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0``: Realistic NoCam + proc rooms (256 envs)
- ``Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0``: Realistic Depth + proc rooms (64 envs)
- ``Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-v0``: Robust NoCam + proc rooms (256 envs)
- ``Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0``: Robust Depth + proc rooms (64 envs)

Each has a -Play variant for evaluation (fewer envs).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments - 30 total (15 configs x Train/Play)
##

_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"
_SKRL_CFG_ENTRY_POINT = f"{agents.__name__}:skrl_ppo_cfg.yaml"
_DEPTH_RUNNER_CFG = "STRAFER_PPO_DEPTH_RUNNER_CFG"
_NOCAM_RUNNER_CFG = "STRAFER_PPO_RUNNER_CFG"


def _register_nav_env(env_id: str, env_cfg_name: str, runner_cfg_name: str) -> None:
    """Register one Strafer navigation environment."""
    gym.register(
        id=env_id,
        entry_point=_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:{env_cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg_name}",
            "skrl_cfg_entry_point": _SKRL_CFG_ENTRY_POINT,
        },
    )


_ENV_REGISTRATIONS = [
    # Ideal (no noise, no motor dynamics)
    ("Isaac-Strafer-Nav-v0", "StraferNavEnvCfg", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Play-v0", "StraferNavEnvCfg_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Depth-v0", "StraferNavEnvCfg_Depth", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Depth-Play-v0", "StraferNavEnvCfg_Depth_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-NoCam-v0", "StraferNavEnvCfg_NoCam", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-NoCam-Play-v0", "StraferNavEnvCfg_NoCam_PLAY", _NOCAM_RUNNER_CFG),
    # Realistic (motor dynamics + noise)
    ("Isaac-Strafer-Nav-Real-v0", "StraferNavEnvCfg_Real", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Real-Play-v0", "StraferNavEnvCfg_Real_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Real-Depth-v0", "StraferNavEnvCfg_Real_Depth", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Real-Depth-Play-v0", "StraferNavEnvCfg_Real_Depth_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Real-NoCam-v0", "StraferNavEnvCfg_Real_NoCam", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Real-NoCam-Play-v0", "StraferNavEnvCfg_Real_NoCam_PLAY", _NOCAM_RUNNER_CFG),
    # Robust (aggressive noise + dynamics)
    ("Isaac-Strafer-Nav-Robust-v0", "StraferNavEnvCfg_Robust", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-Play-v0", "StraferNavEnvCfg_Robust_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-Depth-v0", "StraferNavEnvCfg_Robust_Depth", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-Depth-Play-v0", "StraferNavEnvCfg_Robust_Depth_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-NoCam-v0", "StraferNavEnvCfg_Robust_NoCam", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-NoCam-Play-v0", "StraferNavEnvCfg_Robust_NoCam_PLAY", _NOCAM_RUNNER_CFG),
    # Infinigen scene variants
    ("Isaac-Strafer-Nav-Real-InfinigenDepth-v0", "StraferNavEnvCfg_Real_InfinigenDepth", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Real-InfinigenDepth-Play-v0", "StraferNavEnvCfg_Real_InfinigenDepth_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-InfinigenDepth-v0", "StraferNavEnvCfg_Robust_InfinigenDepth", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-InfinigenDepth-Play-v0", "StraferNavEnvCfg_Robust_InfinigenDepth_PLAY", _DEPTH_RUNNER_CFG),
    # ProcRoom scene variants
    ("Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0", "StraferNavEnvCfg_Real_ProcRoom_NoCam", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Real-ProcRoom-NoCam-Play-v0", "StraferNavEnvCfg_Real_ProcRoom_NoCam_PLAY", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0", "StraferNavEnvCfg_Real_ProcRoom_Depth", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0", "StraferNavEnvCfg_Real_ProcRoom_Depth_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-v0", "StraferNavEnvCfg_Robust_ProcRoom_NoCam", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-Play-v0", "StraferNavEnvCfg_Robust_ProcRoom_NoCam_PLAY", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0", "StraferNavEnvCfg_Robust_ProcRoom_Depth", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Robust-ProcRoom-Depth-Play-v0", "StraferNavEnvCfg_Robust_ProcRoom_Depth_PLAY", _DEPTH_RUNNER_CFG),
]

for env_id, env_cfg_name, runner_cfg_name in _ENV_REGISTRATIONS:
    _register_nav_env(env_id, env_cfg_name, runner_cfg_name)
