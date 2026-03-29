"""RSL-RL PPO configuration for Strafer navigation task.

Three config tiers:
  - STRAFER_PPO_RUNNER_CFG: Standard MLP for NoCam variants (fast iteration)
  - STRAFER_PPO_LSTM_RUNNER_CFG: LSTM for NoCam variants (online system ID)
  - STRAFER_PPO_DEPTH_RUNNER_CFG: CNN-MLP hybrid for Depth variants

All use asymmetric actor-critic: the actor sees noisy policy observations
while the critic additionally receives privileged ground truth state.
"""

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


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
    policy=RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
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
# NoCam LSTM: Recurrent policy for online system identification (19-dim obs)
#
# The LSTM can infer per-episode hidden variables (friction, mass, motor
# strength, D555 mount offset) from a few timesteps of interaction, enabling
# adaptive behavior that a feedforward MLP cannot achieve.
# =============================================================================

STRAFER_PPO_LSTM_RUNNER_CFG = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=48,
    max_iterations=3000,
    save_interval=100,
    experiment_name="strafer_navigation_lstm",
    empirical_normalization=False,
    obs_groups={"policy": ["policy"], "critic": ["critic"]},
    policy=RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
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
# Depth: CNN-GRU-MLP Hybrid (4819-dim obs = 19 scalar + 4800 depth pixels)
#
# GRU enables online system identification — the policy can infer latent
# dynamics (friction, motor delay, payload) from temporal context, improving
# sim-to-real transfer and robustness to env noise.
#
# gamma=0.99 / lam=0.97: longer effective horizon (vs 0.95/0.95) so the
# value function looks further ahead — important for navigation where
# reaching a goal 6s away requires sustained planning.
#
# entropy_coef=0.005: mild exploration bonus. 0.02 caused unbounded noise
# growth (mean_noise_std 1→13) that destroyed learned behavior after ~3k iters.
#
# schedule="fixed": LR decay is handled via --lr_schedule CLI arg (cosine
# or linear) which monkey-patches the runner.  "adaptive" (KL-based) is
# counterproductive with DAPG and unpredictable with cosine/linear decay.
# =============================================================================

STRAFER_PPO_DEPTH_RUNNER_CFG = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=48,
    max_iterations=10000,
    save_interval=100,
    experiment_name="strafer_navigation_depth",
    empirical_normalization=True,
    obs_groups={"policy": ["policy"], "critic": ["critic"]},
    policy=RslRlPpoActorCriticRecurrentCfg(
        class_name="StraferActorCritic",
        init_noise_std=0.3,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
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
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.97,
        desired_kl=0.01,
        max_grad_norm=0.5,
    ),
)
