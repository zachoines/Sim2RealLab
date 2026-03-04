"""CNN-MLP hybrid actor-critic for Strafer navigation with depth images.

Splits the observation into scalar features (IMU, encoders, goal, etc.) and
a 2D depth image.  The depth image is processed by a lightweight CNN encoder
and the resulting embedding is concatenated with the scalar features before
being fed through standard MLP actor/critic heads.

For NoCam variants (no depth), this falls back to a pure MLP — the CNN path
is simply skipped.

Usage:
    Inject into the rsl_rl runner namespace and set ``class_name`` in the
    PPO actor-critic config (see ``rsl_rl_ppo_cfg.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: str = "elu",
) -> nn.Sequential:
    act_cls = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# CNN Depth Encoder
# ---------------------------------------------------------------------------


class DepthEncoder(nn.Module):
    """Lightweight CNN for 60x80 single-channel depth images.

    Architecture (designed for Jetson Orin Nano inference):
        Conv2d(1, 16, 5, stride=2, pad=2)  -> 30x40x16
        Conv2d(16, 32, 3, stride=2, pad=1) -> 15x20x32
        Conv2d(32, 32, 3, stride=2, pad=1) -> 8x10x32
        AdaptiveAvgPool(4, 4)              -> 4x4x32 = 512
        Linear(512, output_dim)            -> output_dim
    """

    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 4 * 4, output_dim)

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            depth_flat: Flattened depth image, shape (B, 4800) (60*80).

        Returns:
            Embedding of shape (B, output_dim).
        """
        x = depth_flat.view(-1, 1, 60, 80)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Hybrid Actor-Critic
# ---------------------------------------------------------------------------

# Depth image dimensions in the flattened observation
_DEPTH_PIXELS = 60 * 80  # 4800


class StraferActorCritic(nn.Module):
    """CNN-MLP hybrid actor-critic for Strafer navigation.

    Handles both NoCam (pure MLP) and Depth (CNN+MLP) variants
    automatically based on observation dimensionality.

    For Depth variants the observation vector is split:
        scalar_obs = obs[:, :scalar_dim]   (19 dims for NoCam fields)
        depth_obs  = obs[:, scalar_dim:]   (4800 dims)

    The CNN encodes depth into a compact embedding, which is concatenated
    with scalar_obs before the MLP heads.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        # Custom params
        depth_embedding_dim: int = 32,
        scalar_obs_dim: int = 19,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(f"[StraferActorCritic] ignoring unknown kwargs: {list(kwargs.keys())}")

        self.obs_groups = obs_groups

        # Compute observation sizes from sample TensorDict
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])

        # Detect depth variant: if actor obs > scalar_obs_dim, depth is present
        self.has_depth = num_actor_obs > scalar_obs_dim
        self.scalar_obs_dim = scalar_obs_dim

        # Build depth encoder if needed
        if self.has_depth:
            self.depth_encoder_actor = DepthEncoder(output_dim=depth_embedding_dim)
            actor_input_dim = scalar_obs_dim + depth_embedding_dim
        else:
            self.depth_encoder_actor = None
            actor_input_dim = num_actor_obs

        # Critic may have different obs size (privileged info)
        # Check if critic also has depth
        self.critic_has_depth = num_critic_obs > scalar_obs_dim
        if self.critic_has_depth:
            self.depth_encoder_critic = DepthEncoder(output_dim=depth_embedding_dim)
            critic_input_dim = scalar_obs_dim + depth_embedding_dim
            # If critic has privileged obs beyond depth, account for it
            critic_extra = num_critic_obs - scalar_obs_dim - _DEPTH_PIXELS
            if critic_extra > 0:
                critic_input_dim += critic_extra
        else:
            self.depth_encoder_critic = None
            critic_input_dim = num_critic_obs

        # Build MLP heads
        self.actor = _build_mlp(actor_input_dim, num_actions, list(actor_hidden_dims), activation)
        self.critic = _build_mlp(critic_input_dim, 1, list(critic_hidden_dims), activation)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        # Running normalization (optional)
        self._actor_obs_norm = None
        self._critic_obs_norm = None
        if actor_obs_normalization:
            from rsl_rl.utils import EmpiricalNormalization
            self._actor_obs_norm = EmpiricalNormalization(num_actor_obs)
        if critic_obs_normalization:
            from rsl_rl.utils import EmpiricalNormalization
            self._critic_obs_norm = EmpiricalNormalization(num_critic_obs)

    # ----- obs helpers -----

    def _get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)

    def _get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups["critic"]], dim=-1)

    def _encode_actor(self, raw_obs: torch.Tensor) -> torch.Tensor:
        if self.has_depth and self.depth_encoder_actor is not None:
            scalar = raw_obs[:, :self.scalar_obs_dim]
            depth_flat = raw_obs[:, self.scalar_obs_dim:self.scalar_obs_dim + _DEPTH_PIXELS]
            depth_emb = self.depth_encoder_actor(depth_flat)
            return torch.cat([scalar, depth_emb], dim=-1)
        return raw_obs

    def _encode_critic(self, raw_obs: torch.Tensor) -> torch.Tensor:
        if self.critic_has_depth and self.depth_encoder_critic is not None:
            scalar = raw_obs[:, :self.scalar_obs_dim]
            depth_flat = raw_obs[:, self.scalar_obs_dim:self.scalar_obs_dim + _DEPTH_PIXELS]
            depth_emb = self.depth_encoder_critic(depth_flat)
            extra = raw_obs[:, self.scalar_obs_dim + _DEPTH_PIXELS:]
            parts = [scalar, depth_emb]
            if extra.shape[-1] > 0:
                parts.append(extra)
            return torch.cat(parts, dim=-1)
        return raw_obs

    # ----- interface required by rsl_rl -----

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        raw = self._get_actor_obs(obs)
        if self._actor_obs_norm is not None:
            raw = self._actor_obs_norm(raw)
        encoded = self._encode_actor(raw)
        mean = self.actor(encoded)
        self.distribution = Normal(mean, self.std.expand_as(mean))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        raw = self._get_actor_obs(obs)
        if self._actor_obs_norm is not None:
            raw = self._actor_obs_norm(raw)
        encoded = self._encode_actor(raw)
        return self.actor(encoded)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        raw = self._get_critic_obs(obs)
        if self._critic_obs_norm is not None:
            raw = self._critic_obs_norm(raw)
        encoded = self._encode_critic(raw)
        return self.critic(encoded)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        if self._actor_obs_norm is not None:
            self._actor_obs_norm.update(self._get_actor_obs(obs))
        if self._critic_obs_norm is not None:
            self._critic_obs_norm.update(self._get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True  # True = resume training

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)
