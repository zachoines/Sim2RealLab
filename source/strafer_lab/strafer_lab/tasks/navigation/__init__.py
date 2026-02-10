"""Navigation task for Strafer mecanum wheel robot.

This module registers 16 Gym environments organized by realism level and sensors:

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
- ``Isaac-Strafer-Nav-Robust-NoCam-v0``: Proprioceptive-only with extreme noise

Each has a -Play variant for evaluation (fewer envs).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments - 16 total (8 configs × Train/Play)
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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Depth_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_NoCam_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Real-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Real-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_Depth_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Real-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Real_NoCam_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Robust-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Nav-Robust-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavEnvCfg_Robust_NoCam_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
