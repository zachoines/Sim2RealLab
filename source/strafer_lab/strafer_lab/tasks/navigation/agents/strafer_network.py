"""CNN-MLP hybrid actor-critic for Strafer navigation with depth images.

Splits the observation into scalar features (IMU, encoders, goal, etc.) and
a 2D depth image.  The depth image is processed by an encoder and the
resulting embedding is concatenated with the scalar features before being fed
through standard MLP actor/critic heads.

For NoCam variants (no depth), this falls back to a pure MLP — the encoder
path is simply skipped.

Depth encoder options (selected via ``depth_encoder_type``):

    "defm" (default, recommended):
        Frozen DeFM (Depth Foundation Model) backbone + trainable linear
        projection.  DeFM is a ViT/CNN pretrained on 60M depth images via
        DINOv2-style self-distillation (ETH Zurich, Jan 2026).  The frozen
        features are metric-aware and transfer across sim-to-real without
        fine-tuning.  Only the projection head (~164K params) trains.

        Variant selection via ``defm_model_name``:
            "efficientnet_b0"  — 3M params, 3ms on Jetson Orin (default)
            "efficientnet_b2"  — 7M params, 5ms on Jetson Orin
            "resnet18"         — 11M params, 12ms on Jetson Orin

    "cnn" (legacy fallback):
        Custom BN+residual CNN with SpatialSoftArgmax, trained from scratch.
        ~80K params.  Struggles to converge alongside PPO — use only if DeFM
        is unavailable.

Usage:
    Inject into the rsl_rl runner namespace and set ``class_name`` in the
    PPO actor-critic config (see ``rsl_rl_ppo_cfg.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Spatial Soft-Argmax (used by legacy CNN encoder)
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
        grid_y = torch.linspace(-1, 1, height).view(1, 1, height, 1)
        grid_x = torch.linspace(-1, 1, width).view(1, 1, 1, width)
        self.register_buffer("_grid_y", grid_y)
        self.register_buffer("_grid_x", grid_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.view(B, C, -1)
        weights = torch.softmax(flat / self.temperature, dim=-1)
        weights = weights.view(B, C, H, W)
        exp_x = (weights * self._grid_x).sum(dim=(2, 3))
        exp_y = (weights * self._grid_y).sum(dim=(2, 3))
        return torch.cat([exp_x, exp_y], dim=-1)


# ---------------------------------------------------------------------------
# DeFM Depth Encoder (frozen pretrained backbone + trainable projection)
# See https://github.com/leggedrobotics/defm
# ---------------------------------------------------------------------------

# DeFM input resolution (all variants expect 224×224)
_DEFM_INPUT_SIZE = 224


def _load_defm_backbone(model_name: str) -> tuple[nn.Module, int]:
    """Load a frozen DeFM backbone via torch.hub.

    Returns (backbone_module, feature_dim).
    """
    print(f"[DeFMDepthEncoder] Loading DeFM backbone: {model_name}")
    model = torch.hub.load(
        "leggedrobotics/defm:main",
        f"defm_{model_name}",
        pretrained=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Determine feature dim by probing with a dummy input.
    # DeFM returns a dict: {"global_backbone": (B, D), "dense_bifpn": {...}}
    # We use the global_backbone vector for RL (pooled C5 features).
    with torch.no_grad():
        dummy = torch.zeros(1, 3, _DEFM_INPUT_SIZE, _DEFM_INPUT_SIZE)
        out = model(dummy)
        global_feat = out["global_backbone"]
        feature_dim = global_feat.shape[-1]

    print(f"[DeFMDepthEncoder] Loaded: {model_name}, feature_dim={feature_dim}")
    return model, feature_dim


class DeFMDepthEncoder(nn.Module):
    """Frozen DeFM encoder + trainable projection head.

    DeFM expects 3-channel input at 224×224.  The depth image (60×80, 1ch)
    is resized and replicated to 3 channels (DeFM's log-compression handles
    the normalization internally).

    Only the projection layer trains — the backbone is frozen.
    """

    def __init__(self, output_dim: int = 128, model_name: str = "efficientnet_b0"):
        super().__init__()
        self.backbone, backbone_dim = _load_defm_backbone(model_name)
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.ELU(),
        )

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            depth_flat: Flattened depth image, shape (B, 4800) (60*80).

        Returns:
            Embedding of shape (B, output_dim).
        """
        # Reshape: (B, 4800) → (B, 1, 60, 80)
        x = depth_flat.view(-1, 1, 60, 80)
        # Resize to DeFM input resolution
        x = F.interpolate(x, size=(_DEFM_INPUT_SIZE, _DEFM_INPUT_SIZE), mode="bilinear", align_corners=False)
        # Replicate single channel to 3 channels (DeFM expects 3ch)
        x = x.expand(-1, 3, -1, -1)
        # Frozen forward pass — extract global_backbone vector from DeFM output dict
        with torch.no_grad():
            out = self.backbone(x)
            features = out["global_backbone"]
        # Trainable projection
        return self.projection(features)


# ---------------------------------------------------------------------------
# Legacy CNN Depth Encoder (fallback if DeFM unavailable)
# ---------------------------------------------------------------------------


class DepthEncoder(nn.Module):
    """CNN encoder for 60×80 single-channel depth images (legacy fallback).

    Architecture:
        Conv(1→32, 5×5, stride=2) + BN + ELU    → 30×40×32
        Conv(32→64, 3×3, stride=2) + BN + ELU    → 15×20×64
        Conv(64→64, 3×3, stride=1) + BN + ELU    → 15×20×64  (+ residual)
        Conv(64→128, 3×3, stride=2) + BN + ELU   → 8×10×128
        SpatialSoftArgmax                          → 256
        Linear(256 → output_dim)                   → output_dim
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.act = nn.ELU()
        self.spatial_argmax = SpatialSoftArgmax(temperature=1.0, height=8, width=10)
        self.fc = nn.Linear(128 * 2, output_dim)

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        x = depth_flat.view(-1, 1, 60, 80)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        residual = x
        x = self.act(self.bn3(self.conv3(x)))
        x = x + residual
        x = self.act(self.bn4(self.conv4(x)))
        x = self.spatial_argmax(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Depth encoder factory
# ---------------------------------------------------------------------------


def _create_depth_encoder(
    encoder_type: str,
    output_dim: int,
    defm_model_name: str,
) -> nn.Module:
    """Create a depth encoder by type, with automatic fallback."""
    if encoder_type == "defm":
        try:
            return DeFMDepthEncoder(output_dim=output_dim, model_name=defm_model_name)
        except Exception as e:
            print(
                f"[StraferActorCritic] WARNING: Failed to load DeFM ({e}). "
                "Falling back to legacy CNN encoder. Install DeFM for better "
                "training: pip install -e git+https://github.com/leggedrobotics/defm.git"
            )
            return DepthEncoder(output_dim=output_dim)
    elif encoder_type == "cnn":
        return DepthEncoder(output_dim=output_dim)
    else:
        raise ValueError(f"Unknown depth_encoder_type: {encoder_type!r}. Use 'defm' or 'cnn'.")


# ---------------------------------------------------------------------------
# Hybrid Actor-Critic
# ---------------------------------------------------------------------------

# Depth image dimensions in the flattened observation
_DEPTH_PIXELS = 60 * 80  # 4800


class StraferActorCritic(nn.Module):
    """CNN-MLP hybrid actor-critic for Strafer navigation.

    Handles both NoCam (pure MLP) and Depth (encoder+MLP) variants
    automatically based on observation dimensionality.

    For Depth variants the observation vector is split:
        scalar_obs = obs[:, :scalar_dim]   (auto-detected)
        depth_obs  = obs[:, scalar_dim:scalar_dim + 4800]

    The encoder compresses depth into a compact embedding, which is
    concatenated with scalar_obs before the MLP heads.
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
        depth_encoder_type: str = "defm",
        defm_model_name: str = "efficientnet_b0",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(f"[StraferActorCritic] ignoring unknown kwargs: {list(kwargs.keys())}")

        self.obs_groups = obs_groups

        # Compute observation sizes from sample TensorDict
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])

        # Dynamically compute scalar obs dim: total obs minus depth pixels
        # This avoids hardcoding 19 and breaking when obs space changes.
        if num_actor_obs > _DEPTH_PIXELS:
            self.scalar_obs_dim = num_actor_obs - _DEPTH_PIXELS
            self.has_depth = True
        else:
            self.scalar_obs_dim = num_actor_obs
            self.has_depth = False

        print(f"[StraferActorCritic] actor_obs={num_actor_obs}, scalar_dim={self.scalar_obs_dim}, "
              f"has_depth={self.has_depth}, encoder={depth_encoder_type}")

        # Build depth encoder if needed
        if self.has_depth:
            self.depth_encoder_actor = _create_depth_encoder(
                depth_encoder_type, depth_embedding_dim, defm_model_name,
            )
            actor_input_dim = self.scalar_obs_dim + depth_embedding_dim
        else:
            self.depth_encoder_actor = None
            actor_input_dim = num_actor_obs

        # Critic may have different obs size (privileged info)
        if num_critic_obs > _DEPTH_PIXELS:
            self.critic_has_depth = True
            self.critic_scalar_dim = num_critic_obs - _DEPTH_PIXELS
            # Critic gets its own encoder (separate from actor for asymmetric obs)
            self.depth_encoder_critic = _create_depth_encoder(
                depth_encoder_type, depth_embedding_dim, defm_model_name,
            )
            # Critic scalar includes privileged obs that come after depth
            critic_input_dim = self.critic_scalar_dim + depth_embedding_dim
        else:
            self.critic_has_depth = False
            self.critic_scalar_dim = num_critic_obs
            self.depth_encoder_critic = None
            critic_input_dim = num_critic_obs

        # Build MLP heads
        self.actor = _build_mlp(actor_input_dim, num_actions, list(actor_hidden_dims), activation)
        self.critic = _build_mlp(critic_input_dim, 1, list(critic_hidden_dims), activation)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        # Running normalization (optional — saved with checkpoint for deployment)
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
            # Scalar obs come first, then depth pixels, then any privileged extras
            scalar_before = raw_obs[:, :self.scalar_obs_dim]
            depth_flat = raw_obs[:, self.scalar_obs_dim:self.scalar_obs_dim + _DEPTH_PIXELS]
            depth_emb = self.depth_encoder_critic(depth_flat)
            extra = raw_obs[:, self.scalar_obs_dim + _DEPTH_PIXELS:]
            parts = [scalar_before, depth_emb]
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
