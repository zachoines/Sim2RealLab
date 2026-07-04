"""Pure (no-Kit) checks for the recurrent (GRU) rolling-subgoal runner.

``STRAFER_PPO_RECURRENT_RUNNER_CFG`` is the GRU arm of the rolling-subgoal
architecture ablation: it must be identical to the MLP baseline
``STRAFER_PPO_RUNNER_CFG`` in every PPO hyperparameter, rollout scalar, trunk
width, obs group, and experiment name — differing ONLY in the actor/critic
model class and its rnn fields. These tests pin that invariant (so a future
edit can't silently let a second hyperparameter drift and confound the
MLP-vs-GRU comparison) and verify the two GRU env ids resolve to this runner.
No Kit / GPU needed.
"""
from __future__ import annotations

import importlib

import gymnasium as gym

# Importing the package triggers the gym.register() calls.
import strafer_lab.tasks.navigation  # noqa: F401
from strafer_lab.tasks.navigation.agents.rsl_rl_ppo_cfg import (
    STRAFER_PPO_RECURRENT_RUNNER_CFG,
    STRAFER_PPO_RUNNER_CFG,
)

# The only fields that may differ between the MLP baseline and the GRU arm:
# the model-class marker plus the three recurrence fields, on actor and critic.
_ALLOWED_MODEL_DIFF = {"class_name", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"}

_GRU_ENV_IDS = (
    "Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-GRU-v0",
    "Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-GRU-v0",
)


def _model_diff_keys(baseline_model: dict, recurrent_model: dict) -> set[str]:
    keys = set(baseline_model) | set(recurrent_model)
    return {k for k in keys if baseline_model.get(k) != recurrent_model.get(k)}


def test_recurrent_runner_differs_from_mlp_only_in_policy_and_rnn_fields():
    """The GRU runner is the MLP baseline with a recurrent actor+critic — and
    nothing else. Any drift in a PPO hyperparameter, rollout scalar, trunk
    width, obs group, or experiment name would confound the ablation, and this
    test would catch it."""
    mlp = STRAFER_PPO_RUNNER_CFG.to_dict()
    rec = STRAFER_PPO_RECURRENT_RUNNER_CFG.to_dict()

    mlp_actor, rec_actor = mlp.pop("actor"), rec.pop("actor")
    mlp_critic, rec_critic = mlp.pop("critic"), rec.pop("critic")

    # Everything except the actor/critic models is identical.
    assert rec == mlp, f"non-policy runner fields drifted: {rec} != {mlp}"

    # Actor and critic differ ONLY in the model-class marker + rnn fields.
    assert _model_diff_keys(mlp_actor, rec_actor) == _ALLOWED_MODEL_DIFF
    assert _model_diff_keys(mlp_critic, rec_critic) == _ALLOWED_MODEL_DIFF


def test_recurrent_runner_is_gru_128_1_on_actor_and_critic():
    """Both actor and critic are recurrent (mirrors the DEPTH tier so the
    critic's value estimates aren't memoryless in this POMDP), GRU/128/1."""
    for model in (
        STRAFER_PPO_RECURRENT_RUNNER_CFG.actor,
        STRAFER_PPO_RECURRENT_RUNNER_CFG.critic,
    ):
        assert type(model).__name__ == "RslRlRNNModelCfg"
        assert model.rnn_type == "gru"
        assert model.rnn_hidden_dim == 128
        assert model.rnn_num_layers == 1


def test_gru_env_ids_registered_and_resolve_to_recurrent_runner():
    """Both GRU env ids are registered and their rsl_rl runner entry point
    resolves to STRAFER_PPO_RECURRENT_RUNNER_CFG — the object itself, not just
    a name-suffix match."""
    for env_id in _GRU_ENV_IDS:
        assert env_id in gym.envs.registry, f"{env_id} not registered"
        spec = gym.envs.registry[env_id]
        entry = (spec.kwargs or {}).get("rsl_rl_cfg_entry_point", "")
        module_path, _, attr = entry.partition(":")
        resolved = getattr(importlib.import_module(module_path), attr)
        assert resolved is STRAFER_PPO_RECURRENT_RUNNER_CFG, (
            f"{env_id} runner entry '{entry}' did not resolve to the "
            "recurrent runner cfg"
        )


def test_gru_arm_reuses_existing_subgoal_env_cfgs():
    """The GRU arm pairs with the SAME env cfg classes as the MLP arm (same obs
    contract) — the only variable in the ablation is the policy. The training
    id reuses the Robust training cfg; the play/eval id reuses the Real play
    cfg (the export-shape source)."""
    train = gym.envs.registry["Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-GRU-v0"]
    play = gym.envs.registry["Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-GRU-v0"]
    assert (train.kwargs or {})["env_cfg_entry_point"].endswith(
        ":StraferNavCfg_RLNoCamSubgoal_Robust"
    )
    assert (play.kwargs or {})["env_cfg_entry_point"].endswith(
        ":StraferNavCfg_RLNoCamSubgoal_Real_PLAY"
    )
