"""Navigation task for Strafer mecanum wheel robot.

This module registers Gym environments organized by realism level and sensors:

IDEAL (No noise, no motor dynamics - debugging/baselines):
- ``Isaac-Strafer-Nav-v0``: Full RGB+Depth
- ``Isaac-Strafer-Nav-Depth-v0``: Depth-only
- ``Isaac-Strafer-Nav-NoCam-v0``: Proprioceptive-only

REALISTIC (Motor dynamics + noise - sim-to-real target):
- ``Isaac-Strafer-Nav-Real-v0``: Full RGB+Depth with realistic dynamics
- ``Isaac-Strafer-Nav-Real-Depth-v0``: Depth-only with realistic dynamics
- ``Isaac-Strafer-Nav-Real-NoCam-v0``: Proprioceptive-only with realistic dynamics

ROBUST (Aggressive noise + dynamics - stress-testing):
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
# Register Gym environments - 30 total (15 configs × Train/Play)
##

# =============================================================================
# IDEAL: No noise, no motor dynamics (debugging/baselines)
# =============================================================================

# Full RGB+Depth
gym.register(
    id="Isaac-Strafer-Nav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Depth-only
gym.register(
    id="Isaac-Strafer-Nav-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Depth",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Depth_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Proprioceptive-only (no camera)
gym.register(
    id="Isaac-Strafer-Nav-NoCam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_NoCam",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_NoCam_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# =============================================================================
# REALISTIC: Motor dynamics + noise (sim-to-real target)
# =============================================================================

# Full RGB+Depth with realistic dynamics
gym.register(
    id="Isaac-Strafer-Nav-Real-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Real-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Depth-only with realistic dynamics
gym.register(
    id="Isaac-Strafer-Nav-Real-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_Depth",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Real-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_Depth_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Proprioceptive-only with realistic dynamics
gym.register(
    id="Isaac-Strafer-Nav-Real-NoCam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_NoCam",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Real-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_NoCam_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# =============================================================================
# ROBUST: Aggressive noise + dynamics (stress-testing)
# =============================================================================

gym.register(
    id="Isaac-Strafer-Nav-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Robust-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Depth-only with aggressive dynamics
gym.register(
    id="Isaac-Strafer-Nav-Robust-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_Depth",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Robust-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_Depth_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Proprioceptive-only with aggressive dynamics
gym.register(
    id="Isaac-Strafer-Nav-Robust-NoCam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_NoCam",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Robust-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_NoCam_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# =============================================================================
# INFINIGEN: Infinigen scene variants (Phase 6)
# =============================================================================

# Realistic + Infinigen scenes (depth-only)
gym.register(
    id="Isaac-Strafer-Nav-Real-InfinigenDepth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_InfinigenDepth",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Real-InfinigenDepth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_InfinigenDepth_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Robust + Infinigen scenes (depth-only)
gym.register(
    id="Isaac-Strafer-Nav-Robust-InfinigenDepth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_InfinigenDepth",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Robust-InfinigenDepth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_InfinigenDepth_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# =============================================================================
# PROCROOM: Procedural primitive rooms (Phase 7)
# =============================================================================

# Realistic + ProcRoom NoCam
gym.register(
    id="Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_ProcRoom_NoCam",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Real-ProcRoom-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_ProcRoom_NoCam_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Realistic + ProcRoom Depth
gym.register(
    id="Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_ProcRoom_Depth",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_ProcRoom_Depth_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Robust + ProcRoom NoCam
gym.register(
    id="Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_ProcRoom_NoCam",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_ProcRoom_NoCam_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Robust + ProcRoom Depth
gym.register(
    id="Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_ProcRoom_Depth",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Robust-ProcRoom-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_ProcRoom_Depth_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:STRAFER_PPO_DEPTH_RUNNER_CFG",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
