"""RNN model with integrated depth encoding for Strafer navigation.

Extends ``rsl_rl.models.RNNModel`` to add a depth-encoder stage between
observation concatenation and the RNN.  Depth pixels are split from the
flat observation, compressed through a CNN or frozen DeFM backbone, then
concatenated with the scalar features before normalization and the RNN.

When no depth is present (obs dim ≤ ``depth_obs_dim``), the model falls
through to plain ``RNNModel`` behavior.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import HiddenState

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlRNNModelCfg

from .depth_encoders import create_depth_encoder

# Depth image dimensions in the flattened observation
_DEFAULT_DEPTH_OBS_DIM = 60 * 80  # 4800


class StraferDepthRNNModel(RNNModel):
    """RNN model with a depth encoder between obs concat and the recurrent layer.

    Architecture::

        obs groups → concat flat (scalar + depth_pixels)
                         │
                   ┌─────┴──────┐
              scalar_obs    depth_pixels
                   │            │
                   │     depth_encoder
                   │            │
                   └────┬───────┘
                        │
                    concat (scalar + depth_embedding)
                        │
                    normalize
                        │
                       RNN
                        │
                       MLP → distribution (actor) or value (critic)
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        # Depth encoder params
        depth_encoder_type: str = "defm",
        defm_model_name: str = "efficientnet_b0",
        depth_embedding_dim: int = 128,
        depth_obs_dim: int = _DEFAULT_DEPTH_OBS_DIM,
    ) -> None:
        # Store depth config BEFORE super().__init__() because _get_obs_dim()
        # is called during parent construction and needs these values.
        self._depth_embedding_dim = depth_embedding_dim
        self._depth_obs_dim = depth_obs_dim
        self._depth_encoder_type = depth_encoder_type
        self._defm_model_name = defm_model_name

        # These are populated by _get_obs_dim() during super().__init__()
        self._raw_obs_dim: int = 0
        self._scalar_obs_dim: int = 0
        self._has_depth: bool = False

        # Parent chain: RNNModel → MLPModel
        # MLPModel.__init__ calls _get_obs_dim() (our override) and sets up
        # normalizer, MLP.  RNNModel.__init__ sets up RNN with obs_dim as input.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
            rnn_type,
            rnn_hidden_dim,
            rnn_num_layers,
        )

        # Create depth encoder AFTER super().__init__() so all parent state exists
        if self._has_depth:
            self.depth_encoder = create_depth_encoder(
                self._depth_encoder_type,
                self._depth_embedding_dim,
                self._defm_model_name,
            )
        else:
            self.depth_encoder = None

    # -- Override obs dim computation -----------------------------------------

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """Compute the *compressed* observation dimension.

        If the raw observation contains depth pixels (``raw_dim > depth_obs_dim``),
        returns ``scalar_dim + depth_embedding_dim`` so that the normalizer, RNN,
        and MLP all operate on the compressed representation.
        """
        active_obs_groups, raw_dim = super()._get_obs_dim(obs, obs_groups, obs_set)
        self._raw_obs_dim = raw_dim

        if raw_dim > self._depth_obs_dim:
            self._scalar_obs_dim = raw_dim - self._depth_obs_dim
            self._has_depth = True
            compressed_dim = self._scalar_obs_dim + self._depth_embedding_dim
            return active_obs_groups, compressed_dim
        else:
            self._scalar_obs_dim = raw_dim
            self._has_depth = False
            return active_obs_groups, raw_dim

    # -- Override latent computation ------------------------------------------

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Concat obs → encode depth → normalize → RNN."""
        # Step 1: Concat obs groups (replicate MLPModel.get_latent logic)
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        raw = torch.cat(obs_list, dim=-1)

        # Step 2: Encode depth if present
        if self._has_depth and self.depth_encoder is not None:
            encoded = self._encode_depth(raw)
        else:
            encoded = raw

        # Step 3: Normalize
        encoded = self.obs_normalizer(encoded)

        # Step 4: RNN
        encoded = self.rnn(encoded, masks, hidden_state).squeeze(0)

        return encoded

    # -- Override normalization update ----------------------------------------

    def update_normalization(self, obs: TensorDict) -> None:
        """Update normalizer statistics on the *compressed* observation."""
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            raw = torch.cat(obs_list, dim=-1)
            if self._has_depth and self.depth_encoder is not None:
                with torch.no_grad():
                    encoded = self._encode_depth(raw)
            else:
                encoded = raw
            self.obs_normalizer.update(encoded)  # type: ignore

    # -- Depth encoding -------------------------------------------------------

    @property
    def encoded_obs_dim(self) -> int:
        """Dimension of the encoded observation (after depth compression).

        Used by GAIL to auto-size the discriminator input layer.
        Returns the scalar_dim + depth_embedding_dim when depth is present,
        otherwise the raw obs dim.
        """
        if self._has_depth:
            return self._scalar_obs_dim + self._depth_embedding_dim
        return self._raw_obs_dim

    def encode_obs(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Encode flat observations through the depth encoder (no RNN/MLP).

        Takes a raw concatenated observation tensor and returns the encoded
        representation (scalar features + depth embedding). Used by GAIL
        to map observations into the shared latent space.

        Args:
            obs_flat: (B, raw_obs_dim) flat observations.

        Returns:
            (B, encoded_obs_dim) encoded observations.
        """
        if self._has_depth and self.depth_encoder is not None:
            return self._encode_depth(obs_flat)
        return obs_flat

    def _encode_depth(self, raw: torch.Tensor) -> torch.Tensor:
        """Split scalar/depth from flat obs, encode depth, concat back.

        Handles both 2D ``(B, D)`` and 3D ``(T, B, D)`` tensors from
        recurrent mini-batches.
        """
        leading_shape = raw.shape[:-1]
        flat = raw.reshape(-1, raw.shape[-1])

        scalar = flat[:, : self._scalar_obs_dim]
        depth_pixels = flat[:, self._scalar_obs_dim : self._scalar_obs_dim + self._depth_obs_dim]
        depth_emb = self.depth_encoder(depth_pixels)

        encoded = torch.cat([scalar, depth_emb], dim=-1)
        return encoded.reshape(*leading_shape, -1)

    # -- Export overrides (depth encoder needs custom handling) ----------------

    def as_jit(self) -> nn.Module:
        """Return a JIT-exportable version of this model."""
        if not self._has_depth:
            return super().as_jit()
        if isinstance(self.rnn.rnn, nn.GRU):
            return _DepthGRUExportModel(self)
        raise NotImplementedError(
            f"JIT export not yet implemented for StraferDepthRNNModel with {type(self.rnn.rnn).__name__}. "
            "Only GRU is supported."
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-exportable version of this model."""
        if not self._has_depth:
            return super().as_onnx(verbose)
        raise NotImplementedError(
            "ONNX export not yet implemented for StraferDepthRNNModel with depth. "
            "Use as_jit() for deployment."
        )


# ---------------------------------------------------------------------------
# JIT export wrapper for GRU + depth encoder
# ---------------------------------------------------------------------------


class _DepthGRUExportModel(nn.Module):
    """Exportable GRU model with integrated depth encoder for JIT."""

    def __init__(self, model: StraferDepthRNNModel) -> None:
        super().__init__()
        self.scalar_obs_dim = model._scalar_obs_dim
        self.depth_obs_dim = model._depth_obs_dim
        self.depth_encoder = copy.deepcopy(model.depth_encoder)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one inference step: encode depth → normalize → GRU → MLP."""
        scalar = x[:, : self.scalar_obs_dim]
        depth = x[:, self.scalar_obs_dim : self.scalar_obs_dim + self.depth_obs_dim]
        depth_emb = self.depth_encoder(depth)
        x = torch.cat([scalar, depth_emb], dim=-1)
        x = self.obs_normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset GRU hidden state."""
        self.hidden_state[:] = 0.0  # type: ignore


# ---------------------------------------------------------------------------
# Isaac Lab config
# ---------------------------------------------------------------------------


@configclass
class StraferDepthRNNModelCfg(RslRlRNNModelCfg):
    """Configuration for the RNN model with integrated depth encoding."""

    class_name: str = "strafer_lab.tasks.navigation.agents.depth_rnn_model:StraferDepthRNNModel"
    """Fully-qualified class resolved by ``rsl_rl.utils.resolve_callable()``."""

    depth_encoder_type: str = "defm"
    """Depth encoder type: ``"defm"`` (pretrained, recommended) or ``"cnn"``
    (lightweight, trained from scratch)."""

    defm_model_name: str = "efficientnet_b0"
    """DeFM backbone variant.  Only used when ``depth_encoder_type="defm"``.
    Options: ``"efficientnet_b0"``, ``"efficientnet_b2"``, ``"resnet18"``."""

    depth_embedding_dim: int = 128
    """Output dimension of the depth encoder (fed to RNN alongside scalar obs)."""

    depth_obs_dim: int = _DEFAULT_DEPTH_OBS_DIM
    """Number of depth pixels in the flattened observation (default 60×80=4800)."""
