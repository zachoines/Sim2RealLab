"""RSL-RL PPO configuration for Strafer navigation task."""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


STRAFER_PPO_RUNNER_CFG = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=1500,
    save_interval=50,
    experiment_name="strafer_navigation",
    empirical_normalization=False,
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)
"""PPO runner configuration for Strafer navigation.

Training parameters:
- 24 steps per environment before update
- 1500 max iterations (~36k environment steps per iteration)
- Adaptive learning rate schedule
- 3-layer MLP policy (256-256-128)
"""


class StraferPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Wrapper class for runner configuration."""

    def __init__(self):
        super().__init__(
            num_steps_per_env=24,
            max_iterations=1500,
            save_interval=50,
            experiment_name="strafer_navigation",
            empirical_normalization=False,
            policy=RslRlPpoActorCriticCfg(
                init_noise_std=1.0,
                actor_hidden_dims=[256, 256, 128],
                critic_hidden_dims=[256, 256, 128],
                activation="elu",
            ),
            algorithm=RslRlPpoAlgorithmCfg(
                value_loss_coef=1.0,
                use_clipped_value_loss=True,
                clip_param=0.2,
                entropy_coef=0.01,
                num_learning_epochs=5,
                num_mini_batches=4,
                learning_rate=1.0e-3,
                schedule="adaptive",
                gamma=0.99,
                lam=0.95,
                desired_kl=0.01,
                max_grad_norm=1.0,
            ),
        )
