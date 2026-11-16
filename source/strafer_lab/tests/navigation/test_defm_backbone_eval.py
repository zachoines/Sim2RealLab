# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The frozen DeFM backbone stays in eval mode in training, and its weights come from one pinned revision.

The backbone here is the real EfficientNet-B0 architecture at random weights (no download), with its BatchNorm swapped
for GroupNorm as DeFM does, behind torchvision's feature extractor: it carries ``torchvision.ops.StochasticDepth``
layers, which drop residual branches at random in train mode, and a deep copy of it comes back in train mode. The
torch.hub and Hugging Face calls are replaced, so nothing reaches the network.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from strafer_lab.tasks.navigation.agents import depth_encoders
from strafer_lab.tasks.navigation.agents.depth_encoders import DeFMDepthEncoder
from strafer_shared.constants import DEPTH_HEIGHT, DEPTH_WIDTH

# The real import, captured before the ``hub`` fixture replaces it.
_REAL_GET_DEFM_PREPROCESS = depth_encoders._get_defm_preprocess


def _gn(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            setattr(module, name, nn.GroupNorm(math.gcd(32, c), c))
        else:
            _gn(child)
    return module


class _EfficientNetB0Backbone(nn.Module):
    """DeFM-shaped: torchvision's EfficientNet-B0 stages behind a feature extractor, the pooled last stage out."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import efficientnet_b0
        from torchvision.models.feature_extraction import create_feature_extractor

        torch.manual_seed(0)
        self.backbone = create_feature_extractor(_gn(efficientnet_b0(weights=None)), return_nodes={"features.6": "c5"})
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"global_backbone": self.global_pool(self.backbone(x)["c5"]).flatten(1)}


def _preprocess(x, target_size, cnn_padding, device):
    """Stand-in for DeFM's preprocess: three log channels, resized small to keep the test fast."""
    log_depth = torch.log1p(torch.clamp(x, min=0.0, max=100.0))
    return F.interpolate(torch.cat([log_depth, log_depth / 2.0, log_depth / 4.0], dim=1), size=(64, 64))


@pytest.fixture
def hub(monkeypatch):
    """torch.hub and Hugging Face replaced: records each call, returns the stand-in backbone in train mode."""
    import huggingface_hub

    calls = {"hf": [], "hub": []}

    def hf_hub_download(**kwargs):
        calls["hf"].append(kwargs)
        return f"/pinned/{kwargs['revision']}/{kwargs['filename']}"

    def hub_load(repo, entry, **kwargs):
        calls["hub"].append((repo, entry, kwargs))
        return _EfficientNetB0Backbone().train()

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(torch.hub, "load", hub_load)
    monkeypatch.setattr(depth_encoders, "_get_defm_preprocess", lambda: _preprocess)
    return calls


def _batch() -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.rand(4, DEPTH_HEIGHT * DEPTH_WIDTH, generator=g)


def test_the_backbone_drops_branches_in_train_mode():
    """The control: this backbone is stochastic in train mode, so the tests below can see a train-mode backbone."""
    backbone = _EfficientNetB0Backbone().train()
    x = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(2))
    torch.manual_seed(3)
    a = backbone(x)["global_backbone"]
    b = backbone(x)["global_backbone"]
    assert not torch.equal(a, b)
    assert any(type(m).__name__ == "StochasticDepth" for m in backbone.modules())


def test_two_training_mode_forwards_are_equal(hub):
    encoder = DeFMDepthEncoder(output_dim=16)
    encoder.train()
    assert encoder.training and encoder.projection.training
    assert _modules_in_train_mode(encoder.backbone) == []
    x = _batch()
    torch.manual_seed(4)
    assert torch.equal(encoder(x), encoder(x))


def _modules_in_train_mode(module: nn.Module) -> list[str]:
    return [name or "<root>" for name, m in module.named_modules() if m.training]


def test_the_backbone_stays_in_eval_after_a_deepcopy(hub):
    """A deep copy of an enclosing model in training mode, used as it comes (no ``train`` call): every module of the
    copy's backbone is in eval, and its forwards equal the original's. torchvision rebuilds its feature extractor in
    train mode on a deep copy, below the backbone's own flag."""
    model = nn.ModuleDict({"depth_encoder": DeFMDepthEncoder(output_dim=16)})
    model.train()
    x = _batch()
    want = model["depth_encoder"](x)
    clone = copy.deepcopy(model)
    assert _modules_in_train_mode(clone["depth_encoder"].backbone) == []
    torch.manual_seed(5)
    got = clone["depth_encoder"](x)
    assert torch.equal(got, clone["depth_encoder"](x))
    assert torch.equal(got, want)
    clone.train()
    assert _modules_in_train_mode(clone["depth_encoder"].backbone) == []


def test_eval_mode_is_unchanged(hub):
    encoder = DeFMDepthEncoder(output_dim=16)
    x = _batch()
    encoder.eval()
    a = encoder(x)
    encoder.train()
    assert torch.equal(encoder(x), a)


def test_the_loader_requests_the_pinned_weights(hub):
    backbone, dim = depth_encoders._load_defm_backbone("efficientnet_b0")
    assert hub["hf"] == [{
        "repo_id": "leggedrobotics/defm",
        "filename": "defm_efficientnet_b0.pth",
        "revision": "b6c9eb68ac833ccd93bf2571279235cb9b065251",
    }]
    (_, _, kwargs), = hub["hub"]
    assert kwargs["pretrained"] is True
    assert kwargs["pretrained_path"] == "/pinned/b6c9eb68ac833ccd93bf2571279235cb9b065251/defm_efficientnet_b0.pth"
    assert not backbone.training
    assert not any(p.requires_grad for p in backbone.parameters())
    assert dim == 192  # EfficientNet-B0's features.6 stage, measured by the loader's dummy forward


_CODE_COMMIT = "d987df5c93b5b056b8c97b59fef18fce8d1af9cd"


def test_the_loader_requests_the_pinned_code(hub, tmp_path, monkeypatch):
    """torch.hub loads the model code at the pinned commit (validation skipped, the repo trusted without a prompt),
    and the depth preprocessing is imported from that commit's cache directory, not from main's."""
    import sys

    depth_encoders._load_defm_backbone("efficientnet_b0")
    (repo, entry, kwargs), = hub["hub"]
    assert (repo, entry) == (f"leggedrobotics/defm:{_CODE_COMMIT}", "defm_efficientnet_b0")
    assert kwargs["skip_validation"] is True and kwargs["trust_repo"] is True

    for ref, tag in ((_CODE_COMMIT, "pinned"), ("main", "main")):
        utils = tmp_path / f"leggedrobotics_defm_{ref}" / "defm" / "utils"
        utils.mkdir(parents=True)
        (utils.parent / "__init__.py").write_text("")
        (utils / "__init__.py").write_text("")
        (utils / "utils.py").write_text(f"def preprocess_depth_batch(*a, **k):\n    return {tag!r}\n")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [m for m in sys.modules if m == "defm" or m.startswith("defm.")]:
        monkeypatch.delitem(sys.modules, name)
    assert _REAL_GET_DEFM_PREPROCESS()() == "pinned"
