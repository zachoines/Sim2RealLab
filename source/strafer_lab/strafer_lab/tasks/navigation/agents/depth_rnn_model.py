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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import HiddenState

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlRNNModelCfg

from .depth_encoders import create_depth_encoder, DeFMDepthEncoder

# DeFM normalization stats — must mirror ``DEFM_MEAN`` / ``DEFM_STD`` in
# ``defm.utils.utils`` so the ONNX-safe preprocessing replacement below
# stays numerically equivalent to the training-time pipeline (modulo the
# antialiasing swap).
_DEFM_MEAN = (0.248880, 0.495620, 0.492858)
_DEFM_STD = (0.139357, 0.271314, 0.297177)
_DEFM_INPUT_SIZE = 224

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
        if isinstance(self.rnn.rnn, nn.GRU):
            return _OnnxDepthGRUModel(self, verbose)
        raise NotImplementedError(
            f"ONNX export not yet implemented for StraferDepthRNNModel with "
            f"{type(self.rnn.rnn).__name__}. Only GRU is supported."
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
        # Swap DeFM's un-scriptable backbone (BiFPN's sum-generator) for a
        # pre-traced pipeline when the underlying encoder is DeFM; pure-tensor
        # CNNs script as-is. Mirrors the ONNX-side conditional below.
        if isinstance(model.depth_encoder, DeFMDepthEncoder):
            self.depth_encoder = _TorchSafeDeFMDepthEncoder(model.depth_encoder)
        else:
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
# ONNX export wrapper for GRU + depth encoder
# ---------------------------------------------------------------------------


def _onnx_safe_defm_preprocess(
    depth_flat: torch.Tensor,
    *,
    target_size: int = _DEFM_INPUT_SIZE,
    max_depth_c1: float = 100.0,
    max_depth_c2: float = 9.0,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """ONNX-traceable DeFM preprocessing.

    Functional equivalent of ``defm.utils.utils.preprocess_depth_batch`` with
    two deliberate differences for the export path:

    1. Resize uses ``F.interpolate(mode='bilinear', antialias=False)`` instead
       of ``torchvision.transforms.v2.Resize`` (which is antialiased and traces
       to ``aten::_upsample_bilinear2d_aa``, unsupported through opset 21).
    2. Operates on a flat depth tensor of shape ``(batch, 60*80)``.

    Disabling antialiasing introduces a small per-pixel delta when upscaling
    60×80 → 224×224 (~1–2% absolute on average for high-frequency content).
    Depth images on this robot are nearly piecewise-smooth, so the impact on
    the projected DeFM embedding is bounded; the alternative is no ONNX path
    at all on the current TRT/ORT op set. The training-time pipeline remains
    antialiased — match-or-document is the follow-up.
    """
    x = depth_flat.view(-1, 1, 60, 80).to(torch.float32)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_clamped = torch.clamp(x, min=0.0, max=max_depth_c1)
    log_depth = torch.log1p(x_clamped)

    c1 = log_depth / math.log1p(max_depth_c1)
    c2 = torch.clamp(log_depth / math.log1p(max_depth_c2), min=0.0, max=1.0)

    # Per-image min-max over spatial dims for the relative channel.
    flat = log_depth.view(log_depth.shape[0], -1)
    min_log = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_log = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    denom = max_log - min_log
    denom_safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
    c3 = (log_depth - min_log) / denom_safe
    c3 = torch.where(denom > 0.0, c3, torch.zeros_like(c3))

    x3 = torch.cat([c1, c2, c3], dim=1)
    x3 = F.interpolate(
        x3, size=(target_size, target_size), mode="bilinear", align_corners=False, antialias=False
    )
    # Callers can pass mean/std as pre-allocated tensors (e.g. module
    # buffers) so the trace path records buffer accesses instead of
    # creating fresh tensors. Trace-baked ``torch.tensor(...)`` pins to the
    # trace-time device and breaks ``torch.jit.load(map_location=...)``;
    # buffers move with the module. ONNX export tolerates either form
    # because ONNX Runtime EPs normalise device at session init.
    if mean is None:
        mean = torch.tensor(_DEFM_MEAN, dtype=x3.dtype, device=x3.device).view(1, 3, 1, 1)
    if std is None:
        std = torch.tensor(_DEFM_STD, dtype=x3.dtype, device=x3.device).view(1, 3, 1, 1)
    x3 = (x3 - mean) / std

    # BiFPN expects spatial dims that are multiples of 32; 224 already
    # satisfies this, so the pad is a no-op for the default target. Keep
    # the formula explicit so a future operator changing target_size sees
    # the same shape contract preprocess_depth_batch published.
    pad_h = (32 - (target_size % 32)) % 32
    pad_w = (32 - (target_size % 32)) % 32
    if pad_h or pad_w:
        x3 = F.pad(x3, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return x3


class _OnnxSafeDeFMDepthEncoder(nn.Module):
    """ONNX-exportable mirror of ``DeFMDepthEncoder`` for the export wrapper.

    Replaces the runtime ``DeFMDepthEncoder.forward()`` with a pipeline that
    feeds ``_onnx_safe_defm_preprocess`` instead of DeFM's torchvision-based
    ``preprocess_depth_batch``. Reuses the frozen backbone + projection
    weights from the trained encoder so the embedding stays anchored to the
    checkpoint.
    """

    def __init__(self, encoder: DeFMDepthEncoder) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(encoder.backbone)
        self.projection = copy.deepcopy(encoder.projection)

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        x = _onnx_safe_defm_preprocess(depth_flat)
        with torch.no_grad():
            out = self.backbone(x)
            features = out["global_backbone"]
        return self.projection(features)


class _DeFMTracePipeline(nn.Module):
    """Pure-Python ``preprocess -> backbone -> projection`` wrapper for tracing.

    Folded into a single ``forward`` so ``torch.jit.trace`` captures the full
    pipeline -- including the ``out["global_backbone"]`` dict index (un-
    scriptable as a free expression) and the keyword-only args on
    ``_onnx_safe_defm_preprocess`` -- as one static computation graph. The
    traced ``ScriptModule`` is then opaque to ``torch.jit.script`` callers
    on the outer wrapper.
    """

    def __init__(self, backbone: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        # Register the DeFM normalisation constants as buffers so the trace
        # records buffer accesses (which follow ``torch.jit.load(map_location
        # =cuda)``) rather than baked CPU constants from ``torch.tensor(...)``.
        self.register_buffer(
            "defm_mean",
            torch.tensor(_DEFM_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "defm_std",
            torch.tensor(_DEFM_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        x = _onnx_safe_defm_preprocess(
            depth_flat, mean=self.defm_mean, std=self.defm_std
        )
        with torch.no_grad():
            features = self.backbone(x)["global_backbone"]
        return self.projection(features)


class _TorchSafeDeFMDepthEncoder(nn.Module):
    """TorchScript-exportable mirror of ``DeFMDepthEncoder``.

    ``torch.jit.script(_DepthGRUExportModel)`` fails when the depth encoder
    is a ``DeFMDepthEncoder`` because DeFM's
    ``BiFPN.WeightedFusion.forward`` uses ``sum(generator)`` over a list of
    tensors, which TorchScript's static-type inference cannot resolve. This
    wrapper pre-traces the full ``preprocess -> backbone -> projection``
    pipeline at ``__init__`` so the scripter on the enclosing module
    encounters an opaque ``ScriptModule`` rather than the un-scriptable
    Python below.

    Trace is at the fixed input shape ``(1, _DEFAULT_DEPTH_OBS_DIM)`` --
    matches the batch-1 deployment-inference contract; the resulting
    artifact is not reusable at other batch sizes, which is consistent with
    the rest of the export path.

    Mirrors the ONNX-side ``_OnnxSafeDeFMDepthEncoder`` pattern. The
    backbone + projection weights are deep-copied from the trained encoder
    before tracing, so the embedding stays anchored to the checkpoint.
    """

    def __init__(self, encoder: DeFMDepthEncoder) -> None:
        super().__init__()
        # Trace on CPU. The trained encoder is typically on CUDA but the
        # exported artifact targets CPU inference (export_torchscript /
        # export_onnx do ``module.eval().cpu()``); tracing on CPU avoids
        # a device-mismatch crash inside the trace and matches the device
        # the downstream save/load round-trip operates on.
        pipeline = _DeFMTracePipeline(
            copy.deepcopy(encoder.backbone),
            copy.deepcopy(encoder.projection),
        )
        pipeline.cpu().eval()
        dummy = torch.zeros(1, _DEFAULT_DEPTH_OBS_DIM, dtype=torch.float32)
        with torch.no_grad():
            self.traced = torch.jit.trace(pipeline, dummy)

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        return self.traced(depth_flat)


class _OnnxDepthGRUModel(nn.Module):
    """Exportable GRU model with integrated depth encoder for ONNX.

    Mirrors ``_DepthGRUExportModel`` but lifts hidden state from a
    ``register_buffer`` to an explicit ``h_in`` / ``h_out`` tensor pair, since
    ONNX nodes are stateless by construction. Signature matches rsl_rl's
    stock ``_OnnxRNNModel`` (``(obs, h_in) -> (actions, h_out)``) so the
    multi-input export path in ``source/strafer_lab/scripts/export_policy.py`` consumes both
    wrappers through one code path.
    """

    is_recurrent: bool = True

    def __init__(self, model: StraferDepthRNNModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.scalar_obs_dim = model._scalar_obs_dim
        self.depth_obs_dim = model._depth_obs_dim
        # Swap DeFM's antialiased resize for an ONNX-traceable equivalent
        # when the underlying encoder uses DeFM; pure-tensor CNNs trace as-is.
        if isinstance(model.depth_encoder, DeFMDepthEncoder):
            self.depth_encoder = _OnnxSafeDeFMDepthEncoder(model.depth_encoder)
        else:
            self.depth_encoder = copy.deepcopy(model.depth_encoder)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        if not isinstance(self.rnn, nn.GRU):
            raise NotImplementedError(
                f"_OnnxDepthGRUModel only supports GRU; got {type(self.rnn).__name__}"
            )
        self.rnn.cpu()
        self.input_size = self.scalar_obs_dim + self.depth_obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One inference step with externalized GRU state.

        Args:
            obs: Flat observation, shape ``(1, scalar_obs_dim + depth_obs_dim)``.
            h_in: GRU hidden state, shape ``(num_layers, 1, hidden_size)``.

        Returns:
            ``(actions, h_out)`` — deterministic action vector and the
            updated GRU hidden state to feed back on the next tick.
        """
        scalar = obs[:, : self.scalar_obs_dim]
        depth = obs[:, self.scalar_obs_dim : self.scalar_obs_dim + self.depth_obs_dim]
        depth_emb = self.depth_encoder(depth)
        x = torch.cat([scalar, depth_emb], dim=-1)
        x = self.obs_normalizer(x)
        x, h_out = self.rnn(x.unsqueeze(0), h_in)
        x = x.squeeze(0)
        out = self.mlp(x)
        return self.deterministic_output(out), h_out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Representative inputs for ONNX tracing."""
        obs = torch.zeros(1, self.input_size, dtype=torch.float32)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size, dtype=torch.float32)
        return (obs, h_in)

    @property
    def input_names(self) -> list[str]:
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "h_out"]


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
