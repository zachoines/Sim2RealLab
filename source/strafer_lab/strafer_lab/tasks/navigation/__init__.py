"""Navigation task for Strafer mecanum wheel robot.

This module registers the following Gym environments:

- ``Isaac-Strafer-Navigation-v0``: Point-to-point navigation on flat ground
- ``Isaac-Strafer-Navigation-Play-v0``: Evaluation environment with fewer instances

"""

import gymnasium as gym

from . import agents

##
# Register Gym environments
##

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
