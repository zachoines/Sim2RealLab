"""Unit tests for ``Scripts/export_policy.py`` plumbing.

These tests do not load a real rsl_rl checkpoint (that requires Isaac Sim
and a converged training run). Instead, they build a tiny stateless
``nn.Module`` that mimics the deterministic-mean contract a real export
would satisfy, run it through the export helpers, and check that:

1. The TorchScript ``.pt`` round-trips through
   ``strafer_shared.policy_interface.load_policy()`` and produces
   byte-identical actions on two consecutive calls (the determinism
   contract the Jetson-side inference node asserts at startup).
2. The ONNX ``.onnx`` round-trips through ``load_policy()`` and is
   byte-identical across two calls.
3. The ``<output>.json`` sidecar exists with all the documented fields,
   variant matches ``PolicyVariant.<variant>`` obs_dim, and recurrent
   metadata is recorded faithfully.
4. Action and observation dimensions match the variant contract
   (``obs_dim`` -> 3-dim action vector).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import export_policy
from strafer_shared.policy_interface import PolicyVariant, load_policy


class _TinyActor(nn.Module):
    """Stateless feed-forward stand-in for the rsl_rl deterministic head.

    Matches the contract the export tooling expects: ``forward(x)`` takes
    ``(1, obs_dim)`` and returns ``(1, action_dim)``. Two consecutive
    calls with the same input produce byte-identical output -- exactly
    what a deterministic-mean export must guarantee.
    """

    def __init__(self, obs_dim: int, action_dim: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(obs_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


@pytest.fixture
def nocam_actor() -> _TinyActor:
    """A reproducibly-initialised tiny actor sized for ``PolicyVariant.NOCAM``."""
    torch.manual_seed(0)
    return _TinyActor(PolicyVariant.NOCAM.obs_dim, action_dim=3).eval()


def _synthetic_obs(obs_dim: int, seed: int = 1) -> np.ndarray:
    """Return a fixed pseudo-random observation -- determinism probes need
    something nontrivial, not all-zeros."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(obs_dim).astype(np.float32)


def test_torchscript_export_round_trips_through_load_policy(
    tmp_path: Path, nocam_actor: _TinyActor
) -> None:
    """The exported .pt must load via load_policy() and produce determinism."""
    output_path = tmp_path / "tiny_nocam.pt"
    export_policy.export_torchscript(
        nocam_actor,
        output_path,
        obs_dim=PolicyVariant.NOCAM.obs_dim,
        is_recurrent=False,
    )
    assert output_path.is_file(), "TorchScript artifact was not written"

    policy = load_policy(output_path, PolicyVariant.NOCAM)
    obs = _synthetic_obs(PolicyVariant.NOCAM.obs_dim)
    action_a = policy(obs)
    action_b = policy(obs)

    assert action_a.shape == (3,), f"expected (3,) action, got {action_a.shape}"
    assert np.array_equal(action_a, action_b), (
        "load_policy() round-trip is non-deterministic for two same-obs calls -- "
        "the Jetson inference node's determinism contract would fail on robot."
    )


def test_onnx_export_round_trips_through_load_policy(
    tmp_path: Path, nocam_actor: _TinyActor
) -> None:
    """The exported .onnx must load via load_policy() and produce determinism."""
    pytest.importorskip("onnxruntime")

    output_path = tmp_path / "tiny_nocam.onnx"
    export_policy.export_onnx(
        nocam_actor,
        output_path,
        obs_dim=PolicyVariant.NOCAM.obs_dim,
    )
    assert output_path.is_file(), "ONNX artifact was not written"

    policy = load_policy(output_path, PolicyVariant.NOCAM)
    obs = _synthetic_obs(PolicyVariant.NOCAM.obs_dim)
    action_a = policy(obs)
    action_b = policy(obs)

    assert action_a.shape == (3,), f"expected (3,) action, got {action_a.shape}"
    assert np.array_equal(action_a, action_b), "ONNX inference is non-deterministic"


def test_metadata_sidecar_records_documented_fields(
    tmp_path: Path, nocam_actor: _TinyActor
) -> None:
    """All sidecar fields the strafer-inference brief consumes must be present."""
    pytest.importorskip("onnxruntime")

    output_stem = tmp_path / "tiny_nocam"
    export_policy.export_torchscript(
        nocam_actor,
        output_stem.with_suffix(".pt"),
        obs_dim=PolicyVariant.NOCAM.obs_dim,
        is_recurrent=False,
    )
    export_policy.export_onnx(
        nocam_actor,
        output_stem.with_suffix(".onnx"),
        obs_dim=PolicyVariant.NOCAM.obs_dim,
    )

    sidecar_path = export_policy.write_metadata_sidecar(
        output_stem,
        policy_variant="NOCAM",
        obs_dim=PolicyVariant.NOCAM.obs_dim,
        action_dim=3,
        env_id="Isaac-Strafer-Nav-Real-ProcRoom-NoCam-Play-v0",
        training_preset="STRAFER_PPO_RUNNER_CFG",
        source_checkpoint="logs/rsl_rl/strafer_navigation/run_test/model_999.pt",
        formats=["pt", "onnx"],
        is_recurrent=False,
    )
    assert sidecar_path.is_file(), "sidecar JSON was not written"

    payload = json.loads(sidecar_path.read_text())
    expected_keys = {
        "policy_variant",
        "obs_dim",
        "action_dim",
        "env_id",
        "training_preset",
        "source_checkpoint",
        "formats",
        "is_recurrent",
        "git_commit",
        "export_timestamp",
        "onnx_opset",
    }
    missing = expected_keys - payload.keys()
    assert not missing, f"sidecar is missing required fields: {sorted(missing)}"

    # The variant + obs_dim consistency check is a cross-brief invariant the
    # Jetson side enforces at startup -- catch divergence at the export
    # boundary too.
    assert payload["policy_variant"] == "NOCAM"
    assert payload["obs_dim"] == PolicyVariant.NOCAM.obs_dim
    assert payload["action_dim"] == 3
    assert sorted(payload["formats"]) == ["onnx", "pt"]
    assert payload["is_recurrent"] is False


def test_metadata_obs_dim_matches_variant() -> None:
    """``obs_dim`` field must mirror ``PolicyVariant.<variant>.obs_dim``."""
    # The export script validates this at runtime against the loaded
    # checkpoint; this test pins the contract value so that a future
    # change to PolicyVariant is reflected in the sidecar contract.
    assert PolicyVariant.NOCAM.obs_dim == 19
    assert PolicyVariant.DEPTH.obs_dim == 4819


def test_torchscript_export_rejects_non_pt_extension(
    tmp_path: Path, nocam_actor: _TinyActor
) -> None:
    """Catch operator typos that would emit a misnamed artifact."""
    with pytest.raises(ValueError, match=r"\.pt"):
        export_policy.export_torchscript(
            nocam_actor,
            tmp_path / "tiny_nocam.bin",
            obs_dim=PolicyVariant.NOCAM.obs_dim,
        )


def test_onnx_export_rejects_non_onnx_extension(
    tmp_path: Path, nocam_actor: _TinyActor
) -> None:
    """Catch operator typos that would emit a misnamed artifact."""
    with pytest.raises(ValueError, match=r"\.onnx"):
        export_policy.export_onnx(
            nocam_actor,
            tmp_path / "tiny_nocam.bin",
            obs_dim=PolicyVariant.NOCAM.obs_dim,
        )


def test_torchscript_determinism_check_catches_stochastic_head(
    tmp_path: Path,
) -> None:
    """A stochastic forward pass must trip the export-time determinism guard.

    Simulates the failure mode the brief warns about: shipping a model
    whose stochastic head was not frozen. The export helper should refuse
    to write such an artifact.
    """

    class _StochasticActor(nn.Module):
        """Adds gaussian noise to the output -- non-deterministic on purpose."""

        def __init__(self, obs_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(obs_dim, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mean = torch.tanh(self.linear(x))
            return mean + torch.randn_like(mean) * 0.5

    torch.manual_seed(0)
    bad_actor = _StochasticActor(PolicyVariant.NOCAM.obs_dim).eval()

    with pytest.raises(RuntimeError, match="non-deterministic"):
        export_policy.export_torchscript(
            bad_actor,
            tmp_path / "stochastic.pt",
            obs_dim=PolicyVariant.NOCAM.obs_dim,
            is_recurrent=False,
        )
