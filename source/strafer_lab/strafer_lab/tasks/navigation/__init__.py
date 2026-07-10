"""Navigation task for Strafer mecanum wheel robot.

Environments are composed over orthogonal axes — sensor stack, scene source,
realism level, objective — in ``composed_env_cfg.py`` and registered here
under a composition-legible gym-ID scheme. Two families:

RL training (fixed sensor stack — the obs contract a trained policy was fit
against):

- ``Isaac-Strafer-Nav-RLDepth-Real-v0``: depth-policy obs, ProcRoom, realistic
- ``Isaac-Strafer-Nav-RLDepth-Robust-v0``: depth-policy obs, ProcRoom, robust DR
- ``Isaac-Strafer-Nav-RLNoCam-v0``: proprioceptive obs, ProcRoom, realistic
- ``Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0``: proprioceptive obs, ProcRoom,
  realistic, rolling-subgoal path tracking
- ``Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-v0``: proprioceptive obs,
  ProcRoom, robust DR, rolling-subgoal path tracking
- ``Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-GRU-v0``: as Subgoal-Robust but a
  recurrent (GRU) policy — corner anticipation for the rolling-subgoal task
- ``Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-GRU-v0``: the GRU arm's play /
  eval + export-shape variant (realistic, fewer envs)
- ``Isaac-Strafer-Nav-RLDepth-Subgoal-Real-v0``: depth-policy obs, ProcRoom,
  realistic, rolling-subgoal path tracking — the policy sees depth and can
  leave the path to clear a sensed obstacle
- ``Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0``: depth-policy obs, ProcRoom,
  robust DR, rolling-subgoal path tracking

Capture (operator-selectable stack via ``capture.py --sensors``; the default
preset is shown):

- ``Isaac-Strafer-Nav-Capture-Teleop-v0``: full RGB only, Infinigen, realistic
- ``Isaac-Strafer-Nav-Capture-Bridge-v0``: full RGB+depth + policy depth, Infinigen
- ``Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0``: full RGB+depth + policy depth, ProcRoom
- ``Isaac-Strafer-Nav-Capture-Coverage-v0``: full RGB+depth, Infinigen

Each RL ID has a ``-Play-v0`` variant for evaluation (fewer envs).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments under the composed-variant scheme.
##

_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"
_SKRL_CFG_ENTRY_POINT = f"{agents.__name__}:skrl_ppo_cfg.yaml"
_DEPTH_RUNNER_CFG = "STRAFER_PPO_DEPTH_RUNNER_CFG"
_NOCAM_RUNNER_CFG = "STRAFER_PPO_RUNNER_CFG"
_RECURRENT_RUNNER_CFG = "STRAFER_PPO_RECURRENT_RUNNER_CFG"


def _register_nav_env(env_id: str, env_cfg_name: str, runner_cfg_name: str) -> None:
    """Register one Strafer navigation environment."""
    gym.register(
        id=env_id,
        entry_point=_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.composed_env_cfg:{env_cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg_name}",
            "skrl_cfg_entry_point": _SKRL_CFG_ENTRY_POINT,
        },
    )


# The runner cfg pairs with the observation profile: a depth-image obs needs
# the CNN depth runner; a proprioceptive (no-image) obs uses the MLP runner.
_ENV_REGISTRATIONS = [
    # RL training variants (fixed stack)
    ("Isaac-Strafer-Nav-RLDepth-Real-v0", "StraferNavCfg_RLDepth_Real", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLDepth-Real-Play-v0", "StraferNavCfg_RLDepth_Real_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLDepth-Robust-v0", "StraferNavCfg_RLDepth_Robust", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLDepth-Robust-Play-v0", "StraferNavCfg_RLDepth_Robust_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLNoCam-v0", "StraferNavCfg_RLNoCam", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLNoCam-Play-v0", "StraferNavCfg_RLNoCam_PLAY", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0", "StraferNavCfg_RLNoCamSubgoal_Real", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0", "StraferNavCfg_RLNoCamSubgoal_Real_PLAY", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-v0", "StraferNavCfg_RLNoCamSubgoal_Robust", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-Play-v0", "StraferNavCfg_RLNoCamSubgoal_Robust_PLAY", _NOCAM_RUNNER_CFG),
    # Recurrent (GRU) arm of the rolling-subgoal task — same env cfgs as the
    # MLP arm above (same obs contract), only the runner's policy differs.
    ("Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-GRU-v0", "StraferNavCfg_RLNoCamSubgoal_Robust", _RECURRENT_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-GRU-v0", "StraferNavCfg_RLNoCamSubgoal_Real_PLAY", _RECURRENT_RUNNER_CFG),
    # Depth-camera rolling-subgoal path tracking — the hybrid corner: rolling
    # subgoal obs + depth so the policy can leave the path around a sensed
    # obstacle. Depth-image obs → the CNN depth runner (already recurrent).
    ("Isaac-Strafer-Nav-RLDepth-Subgoal-Real-v0", "StraferNavCfg_RLDepthSubgoal_Real", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0", "StraferNavCfg_RLDepthSubgoal_Real_PLAY", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-v0", "StraferNavCfg_RLDepthSubgoal_Robust", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-RLDepth-Subgoal-Robust-Play-v0", "StraferNavCfg_RLDepthSubgoal_Robust_PLAY", _DEPTH_RUNNER_CFG),
    # Capture variants (operator-selectable stack)
    ("Isaac-Strafer-Nav-Capture-Teleop-v0", "StraferNavCfg_TeleopCapture", _NOCAM_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Capture-Bridge-v0", "StraferNavCfg_BridgeAutonomy", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-v0", "StraferNavCfg_BridgeAutonomy_ProcRoom", _DEPTH_RUNNER_CFG),
    ("Isaac-Strafer-Nav-Capture-Coverage-v0", "StraferNavCfg_Coverage", _NOCAM_RUNNER_CFG),
]

for env_id, env_cfg_name, runner_cfg_name in _ENV_REGISTRATIONS:
    _register_nav_env(env_id, env_cfg_name, runner_cfg_name)
