"""Depth encoder modules for Strafer navigation.

Provides two encoder options for compressing 60x80 single-channel depth images
into compact embeddings:

    "defm" (default, recommended):
        Frozen DeFM (Depth Foundation Model) backbone + trainable linear
        projection.  DeFM is a ViT/CNN pretrained on 60M depth images via
        DINOv2-style self-distillation (ETH Zurich, Jan 2026).  Only the
        projection head (~164K params) trains.

    "cnn" (fallback):
        Custom BN+residual CNN with SpatialSoftArgmax, trained from scratch.
        ~80K params.  Useful when DeFM is unavailable or for fast iteration.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Spatial Soft-Argmax (used by CNN encoder)
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

    with torch.no_grad():
        dummy = torch.zeros(1, 3, _DEFM_INPUT_SIZE, _DEFM_INPUT_SIZE)
        out = model(dummy)
        global_feat = out["global_backbone"]
        feature_dim = global_feat.shape[-1]

    print(f"[DeFMDepthEncoder] Loaded: {model_name}, feature_dim={feature_dim}")
    return model, feature_dim


def _get_defm_preprocess():
    """Import DeFM's vectorized depth preprocessing from the torch.hub cache."""
    import sys
    from pathlib import Path

    hub_dir = Path(torch.hub.get_dir()) / "leggedrobotics_defm_main"
    if hub_dir.is_dir() and str(hub_dir) not in sys.path:
        sys.path.insert(0, str(hub_dir))

    from defm.utils.utils import preprocess_depth_batch
    return preprocess_depth_batch


class DeFMDepthEncoder(nn.Module):
    """Frozen DeFM encoder + trainable projection head.

    Uses DeFM's native ``preprocess_depth_batch()`` to convert raw metric
    depth into the 3-channel log-normalized representation the backbone was
    pretrained on.  Only the projection layer trains (~164K params).
    """

    def __init__(self, output_dim: int = 128, model_name: str = "efficientnet_b0"):
        super().__init__()
        self.backbone, backbone_dim = _load_defm_backbone(model_name)
        self._preprocess = _get_defm_preprocess()
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.ELU(),
        )

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        x = depth_flat.view(-1, 1, 60, 80)
        x = self._preprocess(
            x,
            target_size=_DEFM_INPUT_SIZE,
            cnn_padding=True,
            device=depth_flat.device,
        )
        with torch.no_grad():
            out = self.backbone(x)
            features = out["global_backbone"]
        return self.projection(features)


# ---------------------------------------------------------------------------
# Legacy CNN Depth Encoder (fallback if DeFM unavailable)
# ---------------------------------------------------------------------------


class DepthEncoder(nn.Module):
    """CNN encoder for 60x80 single-channel depth images (legacy fallback).

    Architecture:
        Conv(1->32, 5x5, stride=2) + BN + ELU    -> 30x40x32
        Conv(32->64, 3x3, stride=2) + BN + ELU    -> 15x20x64
        Conv(64->64, 3x3, stride=1) + BN + ELU    -> 15x20x64  (+ residual)
        Conv(64->128, 3x3, stride=2) + BN + ELU   -> 8x10x128
        SpatialSoftArgmax                          -> 256
        Linear(256 -> output_dim)                   -> output_dim
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


def create_depth_encoder(
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
                f"[DepthEncoder] WARNING: Failed to load DeFM ({e}). "
                "Falling back to legacy CNN encoder. Install DeFM for better "
                "training: pip install -e git+https://github.com/leggedrobotics/defm.git"
            )
            return DepthEncoder(output_dim=output_dim)
    elif encoder_type == "cnn":
        return DepthEncoder(output_dim=output_dim)
    else:
        raise ValueError(f"Unknown depth_encoder_type: {encoder_type!r}. Use 'defm' or 'cnn'.")
