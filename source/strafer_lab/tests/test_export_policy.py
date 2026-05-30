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
        env_id="Isaac-Strafer-Nav-RLNoCam-Play-v0",
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


# ---------------------------------------------------------------------------
# Recurrent ONNX (depth + GRU) export
# ---------------------------------------------------------------------------


class _TinyDepthGRU(nn.Module):
    """Stand-in for ``_OnnxDepthGRUModel`` sized for unit tests.

    Mirrors the wrapper's surface -- ``forward(obs, h_in) -> (actions,
    h_out)`` plus ``get_dummy_inputs`` / ``input_names`` / ``output_names``
    -- without dragging in Isaac Lab or the real DeFM backbone. The body
    is the same shape (depth split, encoder, concat, GRU, MLP) so the
    export pipeline is exercised against representative ONNX ops.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        scalar_obs_dim: int = 8,
        depth_obs_dim: int = 16,
        depth_embedding_dim: int = 4,
        hidden_size: int = 6,
        num_layers: int = 1,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        self.scalar_obs_dim = scalar_obs_dim
        self.depth_obs_dim = depth_obs_dim
        self.input_size = scalar_obs_dim + depth_obs_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.depth_encoder = nn.Linear(depth_obs_dim, depth_embedding_dim)
        self.rnn = nn.GRU(scalar_obs_dim + depth_embedding_dim, hidden_size, num_layers)
        self.head = nn.Linear(hidden_size, action_dim)

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scalar = obs[:, : self.scalar_obs_dim]
        depth = obs[:, self.scalar_obs_dim : self.scalar_obs_dim + self.depth_obs_dim]
        depth_emb = self.depth_encoder(depth)
        x = torch.cat([scalar, depth_emb], dim=-1)
        x, h_out = self.rnn(x.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return torch.tanh(self.head(x)), h_out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        obs = torch.zeros(1, self.input_size, dtype=torch.float32)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size, dtype=torch.float32)
        return (obs, h_in)

    @property
    def input_names(self) -> list[str]:
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "h_out"]


@pytest.fixture
def tiny_depth_gru() -> _TinyDepthGRU:
    """Reproducibly-initialised tiny recurrent depth wrapper."""
    torch.manual_seed(0)
    return _TinyDepthGRU().eval()


def test_recurrent_onnx_export_writes_multi_input_artifact(
    tmp_path: Path, tiny_depth_gru: _TinyDepthGRU
) -> None:
    """Recurrent ONNX export advertises ``(obs, h_in)`` -> ``(actions, h_out)``."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    output_path = tmp_path / "tiny_depth.onnx"
    export_policy.export_onnx(
        tiny_depth_gru,
        output_path,
        obs_dim=tiny_depth_gru.input_size,
    )
    assert output_path.is_file(), "recurrent ONNX artifact was not written"

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    assert input_names == ["obs", "h_in"], (
        f"expected multi-input ONNX with names ['obs', 'h_in'], got {input_names}"
    )
    assert output_names == ["actions", "h_out"], (
        f"expected multi-output ONNX with names ['actions', 'h_out'], "
        f"got {output_names}"
    )


def test_recurrent_onnx_round_trip_is_deterministic_with_reset(
    tmp_path: Path, tiny_depth_gru: _TinyDepthGRU
) -> None:
    """Same (obs, h_in=0) twice → byte-identical actions and h_out."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    output_path = tmp_path / "tiny_depth.onnx"
    export_policy.export_onnx(
        tiny_depth_gru,
        output_path,
        obs_dim=tiny_depth_gru.input_size,
    )

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    obs = _synthetic_obs(tiny_depth_gru.input_size).reshape(1, -1)
    h_in = np.zeros(
        (tiny_depth_gru.num_layers, 1, tiny_depth_gru.hidden_size), dtype=np.float32
    )

    actions_a, h_out_a = sess.run(None, {"obs": obs, "h_in": h_in})
    actions_b, h_out_b = sess.run(None, {"obs": obs, "h_in": h_in})

    assert np.array_equal(actions_a, actions_b), (
        "recurrent ONNX is non-deterministic across two calls with h_in reset "
        "to zeros -- the inference node's reset-bounded determinism contract "
        "would fail on robot."
    )
    assert np.array_equal(h_out_a, h_out_b), (
        "recurrent ONNX h_out diverged across two calls with h_in reset -- the "
        "hidden-state propagation path is non-deterministic."
    )


def test_recurrent_onnx_state_evolves_when_h_in_carried(
    tmp_path: Path, tiny_depth_gru: _TinyDepthGRU
) -> None:
    """No-reset case: feeding the prior h_out back must change the action.

    Pins the other half of the recurrent contract -- the policy is *not*
    Markovian, so an inference node that forgets to thread h_out → h_in
    would degrade silently to a feedforward MLP.
    """
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    output_path = tmp_path / "tiny_depth.onnx"
    export_policy.export_onnx(
        tiny_depth_gru,
        output_path,
        obs_dim=tiny_depth_gru.input_size,
    )

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    obs = _synthetic_obs(tiny_depth_gru.input_size).reshape(1, -1)
    h_in = np.zeros(
        (tiny_depth_gru.num_layers, 1, tiny_depth_gru.hidden_size), dtype=np.float32
    )

    actions_a, h_out_a = sess.run(None, {"obs": obs, "h_in": h_in})
    actions_b, _ = sess.run(None, {"obs": obs, "h_in": h_out_a})

    assert not np.array_equal(actions_a, actions_b), (
        "recurrent ONNX produced identical actions despite a non-zero h_in "
        "on the second call -- the GRU is being bypassed."
    )


def test_recurrent_onnx_sidecar_marks_is_recurrent(
    tmp_path: Path, tiny_depth_gru: _TinyDepthGRU
) -> None:
    """Sidecar must record ``is_recurrent: true`` + the ONNX opset for DEPTH."""
    pytest.importorskip("onnxruntime")

    output_stem = tmp_path / "tiny_depth"
    export_policy.export_onnx(
        tiny_depth_gru,
        output_stem.with_suffix(".onnx"),
        obs_dim=tiny_depth_gru.input_size,
    )

    sidecar_path = export_policy.write_metadata_sidecar(
        output_stem,
        policy_variant="DEPTH",
        obs_dim=PolicyVariant.DEPTH.obs_dim,
        action_dim=3,
        env_id="Isaac-Strafer-Nav-RLDepth-Real-Play-v0",
        training_preset="STRAFER_PPO_DEPTH_RUNNER_CFG",
        source_checkpoint="logs/rsl_rl/strafer_navigation/run_test/model_999.pt",
        formats=["onnx"],
        is_recurrent=True,
    )
    payload = json.loads(sidecar_path.read_text())
    assert payload["policy_variant"] == "DEPTH"
    assert payload["is_recurrent"] is True
    assert payload["formats"] == ["onnx"]
    assert "onnx_opset" in payload


def test_recurrent_onnx_determinism_check_catches_stochastic_head(
    tmp_path: Path,
) -> None:
    """A stochastic recurrent wrapper must trip the determinism probe.

    Mirrors the TorchScript stochastic-head test for the multi-input ONNX
    path; the export helper must refuse to write a model whose forward
    pass is non-deterministic even with h_in reset to zeros.
    """
    pytest.importorskip("onnxruntime")

    class _StochasticDepthGRU(_TinyDepthGRU):
        def forward(
            self, obs: torch.Tensor, h_in: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            actions, h_out = super().forward(obs, h_in)
            return actions + torch.randn_like(actions) * 0.5, h_out

    torch.manual_seed(0)
    bad = _StochasticDepthGRU().eval()

    with pytest.raises(RuntimeError, match="non-deterministic"):
        export_policy.export_onnx(
            bad,
            tmp_path / "stochastic_depth.onnx",
            obs_dim=bad.input_size,
        )


# ---------------------------------------------------------------------------
# TorchScript-safe DeFM depth encoder
# ---------------------------------------------------------------------------


# The DEPTH variant's ``_DepthGRUExportModel`` is scripted via
# ``torch.jit.script``. DeFM's ``BiFPN.WeightedFusion.forward`` uses
# ``sum(generator)`` over a list of tensors, which TorchScript cannot type-
# infer. ``_TorchSafeDeFMDepthEncoder`` pre-traces the full preprocess ->
# backbone -> projection pipeline so the scripter encounters an opaque
# ScriptModule at that attribute slot rather than recursing into the
# un-scriptable Python. These tests pin the pattern against a synthetic
# DeFM-shaped stub so a future regression in the wrapper (e.g. someone
# unconditionally deep-copying the encoder again) trips here, not at the
# next real-checkpoint export.


class _StubBifpnFusion(nn.Module):
    """Mimics DeFM's ``BiFPN.WeightedFusion.forward`` un-scriptable construct."""

    def __init__(self, n_features: int = 3) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_features))

    def forward(self, features: list) -> torch.Tensor:
        import torch.nn.functional as F

        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-6)
        # The exact pattern torch.jit.script cannot infer the return type of:
        return sum(w[i] * f for i, f in enumerate(features))


class _StubDeFMBackbone(nn.Module):
    """DeFM-shaped backbone: ``(N, 3, 224, 224)`` -> ``{"global_backbone": (N, C)}``.

    Returns a dict to exercise the dict-index access pattern the real
    backbone exposes and that the trace must capture concretely.
    """

    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 3, padding=1)
        self.fusion = _StubBifpnFusion(3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        a = self.conv(x)
        b = a * 0.5
        c = a + b
        fused = self.fusion([a, b, c])
        pooled = fused.mean(dim=(2, 3))
        return {"global_backbone": pooled}


def _make_stub_defm_encoder(
    backbone_channels: int = 8, output_dim: int = 16
) -> "DeFMDepthEncoder":  # noqa: F821 — runtime import below
    """Build a ``DeFMDepthEncoder``-typed instance without hitting torch.hub.

    The real ``DeFMDepthEncoder.__init__`` downloads ~100 MB from torch.hub
    on first call; that's wrong-shaped overhead for a unit test. We skip
    the parent init and slot the stub backbone + a fresh projection so
    ``isinstance(stub, DeFMDepthEncoder)`` is still True (the conditional
    swap in ``_DepthGRUExportModel.__init__`` keys on that).
    """
    from strafer_lab.tasks.navigation.agents.depth_encoders import DeFMDepthEncoder

    stub = DeFMDepthEncoder.__new__(DeFMDepthEncoder)
    nn.Module.__init__(stub)
    stub.backbone = _StubDeFMBackbone(channels=backbone_channels)
    stub.projection = nn.Linear(backbone_channels, output_dim)
    return stub


def test_torch_safe_defm_encoder_wraps_unscriptable_backbone(
    tmp_path: Path,
) -> None:
    """The TS-safe wrapper around a DeFM-shaped stub:

    1. Builds without raising (trace captures the BiFPN sum-generator).
    2. The inner ``traced`` attribute IS a ScriptModule (opaque to outer
       scripter).
    3. The outer wrapper scripts cleanly via ``torch.jit.script``.
    4. The scripted artifact round-trips through save+load and produces
       byte-identical embeddings on two same-input calls.
    """
    from strafer_lab.tasks.navigation.agents.depth_rnn_model import (
        _TorchSafeDeFMDepthEncoder,
        _DEFAULT_DEPTH_OBS_DIM,
    )

    torch.manual_seed(0)
    stub_encoder = _make_stub_defm_encoder(output_dim=16).eval()

    wrapper = _TorchSafeDeFMDepthEncoder(stub_encoder).eval()
    assert isinstance(wrapper.traced, torch.jit.ScriptModule), (
        "trace did not produce a ScriptModule -- outer scripter would still "
        "recurse into the un-scriptable BiFPN sum-generator"
    )

    scripted = torch.jit.script(wrapper)

    out_path = tmp_path / "wrapper.pt"
    scripted.save(str(out_path))
    reloaded = torch.jit.load(str(out_path))

    dummy = torch.randn(1, _DEFAULT_DEPTH_OBS_DIM, dtype=torch.float32)
    out_a = reloaded(dummy)
    out_b = reloaded(dummy)
    assert out_a.shape == (1, 16), f"expected (1, 16), got {tuple(out_a.shape)}"
    assert torch.equal(out_a, out_b), (
        "save+reload TS DeFM-safe wrapper non-deterministic across same-input calls"
    )


def test_torch_safe_defm_encoder_matches_eager_within_trace_tolerance() -> None:
    """The traced pipeline must agree with the eager wrapper's pipeline on
    the trace shape. This guards against silent embedding drift if the
    pre-/post-processing in ``_DeFMTracePipeline`` is ever reordered."""
    from strafer_lab.tasks.navigation.agents.depth_rnn_model import (
        _TorchSafeDeFMDepthEncoder,
        _DeFMTracePipeline,
        _DEFAULT_DEPTH_OBS_DIM,
    )

    torch.manual_seed(0)
    stub_encoder = _make_stub_defm_encoder().eval()

    wrapper = _TorchSafeDeFMDepthEncoder(stub_encoder).eval()
    # Build a separate eager-Python pipeline with weight-identical copies.
    eager = _DeFMTracePipeline(
        backbone=stub_encoder.backbone,
        projection=stub_encoder.projection,
    ).eval()

    dummy = torch.randn(1, _DEFAULT_DEPTH_OBS_DIM, dtype=torch.float32)
    with torch.no_grad():
        traced_out = wrapper(dummy)
        eager_out = eager(dummy)
    # Trace records concrete graphs; on float32 the agreement is exact.
    assert torch.equal(traced_out, eager_out), (
        "traced and eager DeFM pipelines diverge -- the trace captured a "
        "different graph than the eager forward expects"
    )


def test_torch_safe_defm_encoder_only_used_for_defm_subclass() -> None:
    """The substitution in ``_DepthGRUExportModel.__init__`` keys on
    ``isinstance(model.depth_encoder, DeFMDepthEncoder)``. A non-DeFM
    encoder (e.g. the legacy CNN) must NOT be wrapped -- it scripts as
    a plain ``nn.Module`` and a needless trace would burn build time
    plus pin the artifact to a single batch size unnecessarily."""
    from strafer_lab.tasks.navigation.agents.depth_encoders import (
        DeFMDepthEncoder,
        DepthEncoder,
    )

    plain = DepthEncoder()
    assert not isinstance(plain, DeFMDepthEncoder), (
        "legacy DepthEncoder must not be a DeFMDepthEncoder subclass; "
        "the substitution conditional would over-fire and trace it"
    )
