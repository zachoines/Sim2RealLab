"""Unit tests for ``strafer_shared.policy_interface.load_policy``.

Exercises the loader contract end-to-end against tiny synthetic modules
(no rsl_rl runner, no Isaac Sim). Covers:

1. Stateless ``.pt`` / ``.onnx`` round-trip: same obs -> byte-identical
   actions; ``.reset()`` is a no-op; ``is_recurrent is False``.
2. Recurrent ``.pt`` (TorchScript module with ``.reset()`` + a hidden-state
   buffer): two same-obs calls are byte-identical *with* ``reset()``
   between them, *different* without.
3. Recurrent ``.onnx`` (multi-input ``(obs, h_in) -> (actions, h_out)``,
   mirroring ``rsl_rl.models.rnn_model._OnnxRNNModel`` for GRU): same
   semantics — hidden state cached across calls, ``reset()`` zeros it.
4. Sidecar consumption: ``policy_variant`` mismatch raises a clear error
   at load time; the ``is_recurrent`` flag dispatches to the recurrent
   path even when the artifact itself would otherwise infer stateless.

The ONNX recurrent fixture builds a minimal multi-input ONNX graph by
hand via ``torch.onnx.export`` on a tiny ``nn.Module`` that mirrors
rsl_rl's port names. This avoids depending on rsl_rl at test time
(strafer_shared can't take that dep), while still exercising the exact
loader code path the rsl_rl export will hit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import export_policy
from strafer_shared.policy_interface import (
    LoadedPolicy,
    PolicyVariant,
    load_policy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TinyStatelessActor(nn.Module):
    """Stateless feed-forward stand-in (same shape as test_export_policy)."""

    def __init__(self, obs_dim: int, action_dim: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(obs_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


class _TinyRecurrentTorchActor(nn.Module):
    """Mirror of rsl_rl ``_TorchGRUModel`` shape: buffer-held hidden state.

    ``forward(obs) -> action``; the GRU consumes the in-module buffer and
    writes back the new hidden state. ``reset()`` zeros the buffer. This
    matches what ``source/strafer_lab/scripts/export_policy.py::_verify_torchscript_determinism``
    expects from a recurrent TorchScript module.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 8, action_dim: int = 3) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, num_layers=1)
        self.head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("hidden_state", torch.zeros(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, obs_dim) -> add time dim for GRU: (1, 1, obs_dim)
        out, h = self.gru(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        return torch.tanh(self.head(out.squeeze(0)))

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0


class _TinyRecurrentOnnxActor(nn.Module):
    """Multi-input ONNX-friendly GRU policy.

    ``forward(obs, h_in) -> (actions, h_out)`` matches the rsl_rl
    ``_OnnxRNNModel`` GRU port layout. The strafer-side
    ``_OnnxDepthGRUModel`` is expected to follow the same shape, so
    exercising it here is the right loader-side test for both.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 8, action_dim: int = 3) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, num_layers=1)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # obs: (1, obs_dim) -> (1, 1, obs_dim); GRU h: (num_layers, 1, hidden)
        out, h_out = self.gru(obs.unsqueeze(0), h_in)
        return torch.tanh(self.head(out.squeeze(0))), h_out


def _synthetic_obs(obs_dim: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(obs_dim).astype(np.float32)


def _write_recurrent_onnx(model: _TinyRecurrentOnnxActor, path: Path) -> None:
    obs_dim = model.gru.input_size
    hidden_dim = model.gru.hidden_size
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    dummy_h = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)
    torch.onnx.export(
        model.eval(),
        (dummy_obs, dummy_h),
        str(path),
        export_params=True,
        opset_version=18,
        input_names=["obs", "h_in"],
        output_names=["actions", "h_out"],
        dynamo=False,
    )


# ---------------------------------------------------------------------------
# Stateless paths
# ---------------------------------------------------------------------------


def test_stateless_torchscript_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    actor = _TinyStatelessActor(PolicyVariant.NOCAM.obs_dim).eval()
    out_path = tmp_path / "stateless.pt"
    export_policy.export_torchscript(
        actor, out_path, obs_dim=PolicyVariant.NOCAM.obs_dim, is_recurrent=False
    )

    policy = load_policy(out_path, PolicyVariant.NOCAM)
    assert isinstance(policy, LoadedPolicy)
    assert policy.is_recurrent is False

    obs = _synthetic_obs(PolicyVariant.NOCAM.obs_dim)
    a, b = policy(obs), policy(obs)
    assert np.array_equal(a, b)

    policy.reset()  # no-op for stateless, must not raise
    c = policy(obs)
    assert np.array_equal(a, c)


def test_stateless_onnx_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    actor = _TinyStatelessActor(PolicyVariant.NOCAM.obs_dim).eval()
    out_path = tmp_path / "stateless.onnx"
    export_policy.export_onnx(actor, out_path, obs_dim=PolicyVariant.NOCAM.obs_dim)

    policy = load_policy(out_path, PolicyVariant.NOCAM)
    assert isinstance(policy, LoadedPolicy)
    assert policy.is_recurrent is False

    obs = _synthetic_obs(PolicyVariant.NOCAM.obs_dim)
    a, b = policy(obs), policy(obs)
    assert np.array_equal(a, b)

    policy.reset()
    c = policy(obs)
    assert np.array_equal(a, c)


# ---------------------------------------------------------------------------
# Recurrent paths
# ---------------------------------------------------------------------------


def test_recurrent_torchscript_reset_required_for_determinism(tmp_path: Path) -> None:
    torch.manual_seed(0)
    actor = _TinyRecurrentTorchActor(PolicyVariant.NOCAM.obs_dim).eval()
    out_path = tmp_path / "recurrent.pt"
    export_policy.export_torchscript(
        actor, out_path, obs_dim=PolicyVariant.NOCAM.obs_dim, is_recurrent=True
    )

    policy = load_policy(out_path, PolicyVariant.NOCAM)
    assert policy.is_recurrent is True

    obs = _synthetic_obs(PolicyVariant.NOCAM.obs_dim)

    # With reset() between calls: byte-identical.
    policy.reset()
    a = policy(obs)
    policy.reset()
    b = policy(obs)
    assert np.array_equal(a, b), "reset() must zero hidden state for determinism"

    # Without reset() between calls: hidden state evolves -> different actions.
    policy.reset()
    c = policy(obs)
    d = policy(obs)
    assert not np.array_equal(c, d), (
        "recurrent policy must reflect hidden-state evolution between calls"
    )


def test_recurrent_onnx_reset_required_for_determinism(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    actor = _TinyRecurrentOnnxActor(PolicyVariant.NOCAM.obs_dim).eval()
    out_path = tmp_path / "recurrent.onnx"
    _write_recurrent_onnx(actor, out_path)

    # Sidecar tells the loader this is recurrent; without it, the loader
    # would still infer recurrence from the 'h_in' input name, but we want
    # the explicit-sidecar path covered too.
    export_policy.write_metadata_sidecar(
        out_path.with_suffix(""),
        policy_variant="NOCAM",
        obs_dim=PolicyVariant.NOCAM.obs_dim,
        action_dim=3,
        env_id="test-env",
        training_preset="test-preset",
        source_checkpoint="test-checkpoint.pt",
        formats=["onnx"],
        is_recurrent=True,
    )

    policy = load_policy(out_path, PolicyVariant.NOCAM)
    assert policy.is_recurrent is True

    obs = _synthetic_obs(PolicyVariant.NOCAM.obs_dim)

    policy.reset()
    a = policy(obs)
    policy.reset()
    b = policy(obs)
    assert np.array_equal(a, b), "ONNX reset() must zero h_in buffer"

    policy.reset()
    c = policy(obs)
    d = policy(obs)
    assert not np.array_equal(c, d), (
        "recurrent ONNX policy must thread hidden state across calls"
    )


def test_recurrent_onnx_infers_recurrence_without_sidecar(tmp_path: Path) -> None:
    """No sidecar — loader must still detect 'h_in' and pick the recurrent path."""
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    actor = _TinyRecurrentOnnxActor(PolicyVariant.NOCAM.obs_dim).eval()
    out_path = tmp_path / "recurrent_no_sidecar.onnx"
    _write_recurrent_onnx(actor, out_path)

    policy = load_policy(out_path, PolicyVariant.NOCAM)
    assert policy.is_recurrent is True

    obs = _synthetic_obs(PolicyVariant.NOCAM.obs_dim)
    policy.reset()
    a = policy(obs)
    b = policy(obs)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Sidecar consumption
# ---------------------------------------------------------------------------


def test_sidecar_variant_mismatch_raises(tmp_path: Path) -> None:
    torch.manual_seed(0)
    actor = _TinyStatelessActor(PolicyVariant.NOCAM.obs_dim).eval()
    out_path = tmp_path / "tagged.pt"
    export_policy.export_torchscript(
        actor, out_path, obs_dim=PolicyVariant.NOCAM.obs_dim, is_recurrent=False
    )
    export_policy.write_metadata_sidecar(
        out_path.with_suffix(""),
        policy_variant="DEPTH",  # deliberately wrong
        obs_dim=PolicyVariant.DEPTH.obs_dim,
        action_dim=3,
        env_id="test-env",
        training_preset="test-preset",
        source_checkpoint="test-checkpoint.pt",
        formats=["pt"],
        is_recurrent=False,
    )

    with pytest.raises(ValueError, match="policy_variant"):
        load_policy(out_path, PolicyVariant.NOCAM)


def test_sidecar_is_recurrent_flag_overrides_inference(tmp_path: Path) -> None:
    """Sidecar ``is_recurrent: false`` must keep a model with a ``reset()``
    method on the stateless path. Forces the loader to honor the sidecar
    over a heuristic on the artifact itself."""
    torch.manual_seed(0)
    actor = _TinyRecurrentTorchActor(PolicyVariant.NOCAM.obs_dim).eval()
    out_path = tmp_path / "tagged_stateless.pt"
    # Export marks it recurrent (the determinism probe calls reset()),
    # but we override the sidecar to false to exercise that code path.
    export_policy.export_torchscript(
        actor, out_path, obs_dim=PolicyVariant.NOCAM.obs_dim, is_recurrent=True
    )
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {"policy_variant": "NOCAM", "is_recurrent": False, "obs_dim": 19}
        )
    )

    policy = load_policy(out_path, PolicyVariant.NOCAM)
    assert policy.is_recurrent is False
