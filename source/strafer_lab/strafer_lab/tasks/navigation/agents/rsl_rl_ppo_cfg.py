"""RSL-RL PPO configuration for Strafer navigation task.

Three config tiers:
  - STRAFER_PPO_RUNNER_CFG: Standard MLP for NoCam variants (fast iteration)
  - STRAFER_PPO_RECURRENT_RUNNER_CFG: GRU actor-critic for the NoCam
    rolling-subgoal task (corner anticipation)
  - STRAFER_PPO_DEPTH_RUNNER_CFG: CNN-MLP hybrid for Depth variants

All use asymmetric actor-critic: the actor sees noisy policy observations
while the critic additionally receives privileged ground truth state.
"""

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
    RslRlPpoAlgorithmCfg,
)

from .distributions import BetaDistributionCfg
from .depth_rnn_model import StraferDepthRNNModelCfg


# =============================================================================
# NoCam: Standard MLP (19-dim obs)
# =============================================================================

STRAFER_PPO_RUNNER_CFG = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=48,
    max_iterations=3000,
    save_interval=100,
    experiment_name="strafer_navigation",
    empirical_normalization=False,
    obs_groups={"policy": ["policy"], "critic": ["critic"]},
    actor=RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.5,
        ),
    ),
    critic=RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)


# =============================================================================
# NoCam Recurrent: GRU actor-critic for the rolling-subgoal task (19-dim obs)
#
# A memoryless MLP sees a single subgoal bearing/distance sample per step and
# cannot anticipate corridor curvature — the corner only reveals itself as the
# rolling subgoal sweeps it, so tracking is reactive and the policy cuts
# corners.  A GRU infers curvature from the subgoal's motion history, letting
# the policy anticipate the turn.
#
# Both actor AND critic are recurrent (mirrors the DEPTH tier): in this POMDP a
# memoryless critic produces biased value estimates in exactly the
# history-dependent states that matter (mid-corner), which degrades advantage
# estimation.  GRU / hidden 128 / 1 layer mirrors the DEPTH memory spec.
#
# Every PPO hyperparameter, rollout scalar, trunk width, obs group, and
# experiment name is identical to STRAFER_PPO_RUNNER_CFG — the actor/critic
# model class and its rnn fields are the ONLY difference, so an MLP-vs-GRU
# comparison on the same env is a clean architecture ablation.
# =============================================================================

STRAFER_PPO_RECURRENT_RUNNER_CFG = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=48,
    max_iterations=3000,
    save_interval=100,
    experiment_name="strafer_navigation",
    empirical_normalization=False,
    obs_groups={"policy": ["policy"], "critic": ["critic"]},
    actor=RslRlRNNModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        distribution_cfg=RslRlRNNModelCfg.GaussianDistributionCfg(
            init_std=0.5,
        ),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    ),
    critic=RslRlRNNModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)


# =============================================================================
# Depth: CNN-GRU-MLP Hybrid (3619-dim obs = 19 scalar + 3600 = 80x45 depth pixels)
#
# GRU enables online system identification — the policy can infer latent
# dynamics (friction, motor delay, payload) from temporal context, improving
# sim-to-real transfer and robustness to env noise.
#
# gamma=0.99 / lam=0.97: longer effective horizon (vs 0.95/0.95) so the
# value function looks further ahead — important for navigation where
# reaching a goal 6s away requires sustained planning.
#
# init_noise_std=0.3 seeds a symmetric zero-mean Beta policy whose initial
# action std on [-1, 1] is 0.3 (concentration ≈ 5.1).
#
# entropy_coef=0.005: moderate exploration pressure for the bounded Beta
# policy.  Beta differential entropy goes negative as concentrations grow,
# so this coefficient must be high enough to resist early collapse while
# staying below the surrogate loss magnitude.
#
# schedule="fixed": adaptive LR is counterproductive with DAPG — the strong
# initial BC gradient triggers the KL controller to kill LR before RL starts.
# =============================================================================

STRAFER_PPO_DEPTH_RUNNER_CFG = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=48,
    max_iterations=10000,
    save_interval=100,
    experiment_name="strafer_navigation_depth",
    empirical_normalization=False,
    obs_groups={"actor": ["policy"], "critic": ["critic"]},
    actor=StraferDepthRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=BetaDistributionCfg(
            init_std=0.3,
            min_concentration=0.2,
        ),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        depth_encoder_type="defm",
        defm_model_name="efficientnet_b0",
        depth_embedding_dim=128,
    ),
    critic=StraferDepthRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        depth_encoder_type="defm",
        defm_model_name="efficientnet_b0",
        depth_embedding_dim=128,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.97,
        desired_kl=0.01,
        max_grad_norm=0.5,
    ),
)
