"""CNN-MLP hybrid actor-critic for Strafer navigation with depth images.

Splits the observation into scalar features (IMU, encoders, goal, etc.) and
a 2D depth image.  The depth image is processed by a CNN encoder and the
resulting embedding is concatenated with the scalar features before being fed
through standard MLP actor/critic heads.

For NoCam variants (no depth), this falls back to a pure MLP — the CNN path
is simply skipped.

DepthEncoder architecture (Option A — improved CNN):
    Conv2d(1, 32, 5, stride=2, pad=2) + BN + ELU    → 30×40×32
    Conv2d(32, 64, 3, stride=2, pad=1) + BN + ELU    → 15×20×64
    Conv2d(64, 64, 3, stride=1, pad=1) + BN + ELU    → 15×20×64  (residual add)
    Conv2d(64, 128, 3, stride=2, pad=1) + BN + ELU   → 8×10×128
    SpatialSoftArgmax                                  → 128×2 = 256
    Linear(256, output_dim)                            → output_dim (default 128)

Design rationale:
    - BatchNorm stabilizes CNN training alongside PPO gradient updates
    - Residual connection at 15×20 preserves fine spatial detail
    - SpatialSoftArgmax learns *where* features are (not just what), critical
      for navigation obstacle avoidance
    - 128-dim embedding (4× wider than original 32) retains enough spatial
      information to represent room geometry
    - ~80K params — still lightweight for Jetson Orin Nano inference

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
# Spatial Soft-Argmax
# ---------------------------------------------------------------------------


class SpatialSoftArgmax(nn.Module):
    """Differentiable spatial soft-argmax over 2D feature maps.

    For each channel, computes a softmax over all spatial positions and returns
    the expected (x, y) coordinates.  This gives the network an explicit spatial
    prior: each filter can "point at" a location in the image.

    Input:  (B, C, H, W)
    Output: (B, C*2) — (x, y) per channel, normalized to [-1, 1].
    """

    def __init__(self, temperature: float = 1.0, height: int = 8, width: int = 10):
        super().__init__()
        self.temperature = temperature
        # Register coordinate grids as buffers so they move with .to(device)
        # and are proper tensors (not inference tensors) for autograd.
        grid_y = torch.linspace(-1, 1, height).view(1, 1, height, 1)
        grid_x = torch.linspace(-1, 1, width).view(1, 1, 1, width)
        self.register_buffer("_grid_y", grid_y)
        self.register_buffer("_grid_x", grid_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Softmax over spatial dimensions
        flat = x.view(B, C, -1)  # (B, C, H*W)
        weights = torch.softmax(flat / self.temperature, dim=-1)
        weights = weights.view(B, C, H, W)

        # Expected coordinates
        exp_x = (weights * self._grid_x).sum(dim=(2, 3))  # (B, C)
        exp_y = (weights * self._grid_y).sum(dim=(2, 3))  # (B, C)

        return torch.cat([exp_x, exp_y], dim=-1)  # (B, C*2)


# ---------------------------------------------------------------------------
# CNN Depth Encoder
# ---------------------------------------------------------------------------


class DepthEncoder(nn.Module):
    """CNN encoder for 60×80 single-channel depth images.

    Architecture:
        Conv(1→32, 5×5, stride=2) + BN + ELU    → 30×40×32
        Conv(32→64, 3×3, stride=2) + BN + ELU    → 15×20×64
        Conv(64→64, 3×3, stride=1) + BN + ELU    → 15×20×64  (+ residual)
        Conv(64→128, 3×3, stride=2) + BN + ELU   → 8×10×128
        SpatialSoftArgmax                          → 256
        Linear(256 → output_dim)                   → output_dim

    The residual connection at layer 3 preserves fine spatial detail.
    SpatialSoftArgmax produces (x, y) per channel — the network learns
    to point at obstacle/wall/free-space locations, which is exactly what
    the navigation policy needs.
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        # Stage 1: 60×80 → 30×40
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        # Stage 2: 30×40 → 15×20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Stage 3: 15×20 → 15×20 (residual)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Stage 4: 15×20 → 8×10
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.act = nn.ELU()
        self.spatial_argmax = SpatialSoftArgmax(temperature=1.0, height=8, width=10)
        self.fc = nn.Linear(128 * 2, output_dim)  # 128 channels × 2 coords

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            depth_flat: Flattened depth image, shape (B, 4800) (60*80).

        Returns:
            Embedding of shape (B, output_dim).
        """
        x = depth_flat.view(-1, 1, 60, 80)

        x = self.act(self.bn1(self.conv1(x)))   # 30×40×32
        x = self.act(self.bn2(self.conv2(x)))    # 15×20×64

        # Residual block
        residual = x
        x = self.act(self.bn3(self.conv3(x)))    # 15×20×64
        x = x + residual

        x = self.act(self.bn4(self.conv4(x)))    # 8×10×128

        x = self.spatial_argmax(x)                # (B, 256)
        return self.fc(x)                         # (B, output_dim)


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
        depth_embedding_dim: int = 128,
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
