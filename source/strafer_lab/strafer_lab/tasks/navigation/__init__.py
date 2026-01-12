"""Navigation task for Strafer mecanum wheel robot.

This module registers the following Gym environments:

Full Camera (RGB + Depth):
- ``Isaac-Strafer-Navigation-v0``: Full camera with RGB+depth (19214 obs dims)
- ``Isaac-Strafer-Navigation-Play-v0``: Evaluation environment

Depth-Only Camera:
- ``Isaac-Strafer-Navigation-Depth-v0``: Depth camera only (4814 obs dims)
- ``Isaac-Strafer-Navigation-Depth-Play-v0``: Evaluation with depth

RGB-Only Camera:
- ``Isaac-Strafer-Navigation-RGB-v0``: RGB camera only (14414 obs dims)
- ``Isaac-Strafer-Navigation-RGB-Play-v0``: Evaluation with RGB

Without Camera (Proprioceptive only):
- ``Isaac-Strafer-Navigation-NoCam-v0``: Proprioceptive-only (14 obs dims)
- ``Isaac-Strafer-Navigation-NoCam-Play-v0``: Evaluation without camera

"""

import gymnasium as gym

from . import agents

##
# Register Gym environments
##

# =============================================================================
# Full Camera (RGB + Depth) - 19214 obs dims
# =============================================================================

gym.register(
    id="Isaac-Strafer-Navigation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Navigation-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavigationEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# =============================================================================
# Depth-Only Camera - 4814 obs dims
# =============================================================================

gym.register(
    id="Isaac-Strafer-Navigation-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavigationEnvCfg_DepthOnly",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Navigation-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavigationEnvCfg_DepthOnly_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# =============================================================================
# RGB-Only Camera - 14414 obs dims
# =============================================================================

gym.register(
    id="Isaac-Strafer-Navigation-RGB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavigationEnvCfg_RGBOnly",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Navigation-RGB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavigationEnvCfg_RGBOnly_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# =============================================================================
# No Camera (Proprioceptive-only) - 14 obs dims
# =============================================================================

gym.register(
    id="Isaac-Strafer-Navigation-NoCam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavigationEnvCfg_NoCam",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Strafer-Navigation-NoCam-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.strafer_env_cfg:StraferNavigationEnvCfg_NoCam_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StraferPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
