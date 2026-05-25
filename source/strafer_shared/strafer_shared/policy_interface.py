"""Policy interface -- observation/action contract for sim and real.

This module defines how observations are assembled and actions are interpreted.
It is the single source of truth for the policy I/O contract. Both the Isaac Lab
gym environment config and the ROS2 inference node reference this module.

Observation specs must match the simulation's observation groups in:
  source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py (L249-258)
Normalization scales must match:
  source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py (L225-234)

Recurrent hidden-state contract
-------------------------------

A recurrent policy's hidden state is owned at three places along the
train -> export -> inference chain (rsl_rl trainer, the export wrapper
in ``Scripts/export_policy.py``, the inference-side loader below).
This section pins the seam-level contract so each layer can be edited
without re-deriving what the other two assume.

1. **Hidden-state tensor shape.** ``(rnn_num_layers, 1, rnn_hidden_dim)``
   for batch-1 deployment inference. The trailing-1 is a fixed batch
   axis: ONNX exports with a symbolic batch dim are collapsed to 1 by
   ``_resolve_hidden_shape`` below; TorchScript exports are scripted
   against a buffer of the same shape. ``rnn_num_layers`` and
   ``rnn_hidden_dim`` come from the rsl_rl runner config (e.g.
   ``STRAFER_PPO_DEPTH_RUNNER_CFG.actor.rnn_hidden_dim=128``,
   ``rnn_num_layers=1``). LSTM artifacts carry the pair ``(h, c)`` of
   identically-shaped tensors; GRU carries ``h`` only.

2. **Initial state.** Always zero on ``load_policy()`` return.
   TorchScript artifacts initialize via ``register_buffer(...,
   torch.zeros(...))`` at module load. ONNX artifacts initialize via
   ``np.zeros(...)`` in :class:`_RecurrentOnnxPolicy.__init__`. There
   is no warm-start codepath; if an episode benefits from carried
   state, the caller is responsible for not calling ``reset()``.

3. **Per-tick state threading.**
   - **TorchScript**: the scripted module owns the state buffer
     (``register_buffer("hidden_state", ...)``). ``forward(obs)``
     reads the buffer, runs the RNN, and writes the new state back
     in-place. The loader holds a reference to the module and never
     copies the buffer.
   - **ONNX**: the loader owns persistent ``np.ndarray`` buffers for
     ``h_in`` (and ``c_in`` on LSTM). Each ``__call__`` feeds the
     cached state into the session, reads ``h_out`` (and ``c_out``)
     from the output, and writes them back. If the loader skipped the
     write-back, the policy would reset every tick and behave like a
     feedforward MLP with no error raised — a silent-failure mode
     this contract exists to prevent.

4. **Reset trigger.** :meth:`LoadedPolicy.reset` must be called by
   the inference node on every "mission episode" boundary. A mission
   episode is the policy's view of a single contiguous decision
   problem; the trigger set the inference-side caller commits to:

   - **Action-server goal accepted** (a NEW ``/navigate_to_pose``
     goal handle, not a re-statement of the current one). The
     hidden state from the previous mission is no longer relevant.
   - **Mid-mission goal pose update** (VLM re-grounding produces a
     new goal). Controlled by an ``is_mid_mission_reset`` config flag
     in the inference node (default: True). Rationale: the hidden
     state learned to expect monotonic progress toward the *old*
     goal; carrying it into a re-grounded mission biases the policy.
   - **Watchdog trip** (any of the contracted-stale topic sources):
     the policy is paused. Hidden state stays frozen until the
     watchdog clears, then the *next* mission boundary calls
     ``reset()`` as normal. The pause itself does not reset.

   Stateless policies expose a no-op ``reset()`` so callers can
   invoke it unconditionally without an ``is_recurrent`` branch.

5. **Determinism contract.** Two consecutive calls with the same
   ``obs`` produce byte-identical actions **iff** ``reset()`` is
   called between them. Without an intervening ``reset()``, the
   hidden state has evolved by construction and the two actions
   differ — that is the recurrent-model definition, not a bug.
   Determinism probes (e.g.
   ``Scripts/export_policy.py::_verify_torchscript_determinism``)
   must condition on this; "byte-identical between consecutive calls"
   is the wrong assertion for a recurrent artifact and would force
   the scripted module into a stateless mode that defeats its purpose.

6. **Thread safety.** :class:`LoadedPolicy` and all its subclasses
   are **NOT thread-safe**. TorchScript recurrent modules mutate
   ``register_buffer("hidden_state", ...)`` in ``forward``; the ONNX
   variant mutates ``self._h_in`` (and ``self._c_in``) in
   ``__call__``. Concurrent ``policy(obs)`` calls from multiple
   threads race on those buffers. Callers must serialize all policy
   calls through a single thread, or guard with a mutex if the
   rclpy executor is a ``MultiThreadedExecutor``.

This docstring is the authoritative in-code statement of the
contract. Task briefs that touch any seam in the train -> export ->
inference chain cite it as a single load-bearing reference rather
than re-deriving the rules per brief.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from strafer_shared.constants import (
    BODY_VEL_SCALE,
    DEPTH_SCALE,
    ENCODER_VEL_SCALE,
    GOAL_DIST_SCALE,
    HEADING_SCALE,
    IMU_ACCEL_SCALE,
    IMU_GYRO_SCALE,
    MAX_ANGULAR_VEL,
    MAX_LINEAR_VEL,
)
from strafer_shared.mecanum_kinematics import (
    twist_to_wheel_velocities,
    wheel_vels_to_ticks_per_sec,
)

# ---------------------------------------------------------------------------
# Observation spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObsField:
    """A single field in the observation vector."""

    key: str  # dict key in raw sensor data
    dims: int  # number of elements
    scale: float  # normalization multiplier


_NOCAM_FIELDS: tuple[ObsField, ...] = (
    ObsField("imu_accel", 3, IMU_ACCEL_SCALE),
    ObsField("imu_gyro", 3, IMU_GYRO_SCALE),
    ObsField("encoder_vels_ticks", 4, ENCODER_VEL_SCALE),
    ObsField("goal_relative", 2, GOAL_DIST_SCALE),
    ObsField("goal_distance", 1, GOAL_DIST_SCALE),
    ObsField("goal_heading_to_goal", 1, HEADING_SCALE),
    ObsField("body_velocity_xy", 2, BODY_VEL_SCALE),
    ObsField("last_action", 3, 1.0),
)

_DEPTH_FIELDS: tuple[ObsField, ...] = _NOCAM_FIELDS + (
    ObsField("depth_image", 4800, DEPTH_SCALE),
)


class PolicyVariant(Enum):
    """Policy observation variants matching Isaac Lab environment configs."""

    NOCAM = _NOCAM_FIELDS  # 19 dims
    DEPTH = _DEPTH_FIELDS  # 4819 dims

    @property
    def obs_dim(self) -> int:
        return sum(f.dims for f in self.value)

    @property
    def fields(self) -> tuple[ObsField, ...]:
        return self.value


# ---------------------------------------------------------------------------
# Observation assembly
# ---------------------------------------------------------------------------


def assemble_observation(
    raw: dict[str, np.ndarray | list | tuple],
    variant: PolicyVariant,
) -> np.ndarray:
    """Normalize and concatenate raw sensor values into a policy observation vector.

    Args:
        raw: Dictionary mapping ObsField keys to raw sensor values.
             Each value is array-like with shape matching the field's dims.
        variant: Which observation variant to assemble.

    Returns:
        Flat float32 array of shape (variant.obs_dim,).
    """
    parts: list[np.ndarray] = []
    for field in variant.fields:
        arr = np.asarray(raw[field.key], dtype=np.float32).ravel()
        if arr.shape[0] != field.dims:
            raise ValueError(
                f"Field '{field.key}': expected {field.dims} dims, got {arr.shape[0]}"
            )
        parts.append(arr * field.scale)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Action interpretation
# ---------------------------------------------------------------------------


def interpret_action(action_normalized: np.ndarray) -> tuple[float, float, float]:
    """Denormalize [-1, 1] policy output to physical body velocities.

    Args:
        action_normalized: Array of shape (3,) with values in [-1, 1].

    Returns:
        (vx, vy, omega) in (m/s, m/s, rad/s).
    """
    return (
        float(action_normalized[0]) * MAX_LINEAR_VEL,
        float(action_normalized[1]) * MAX_LINEAR_VEL,
        float(action_normalized[2]) * MAX_ANGULAR_VEL,
    )


def action_to_wheel_ticks(action_normalized: np.ndarray) -> np.ndarray:
    """Convert normalized policy action directly to wheel ticks/sec.

    Chains interpret_action → twist_to_wheel_velocities → wheel_vels_to_ticks_per_sec.
    Convenience function for real robot motor commands.

    Args:
        action_normalized: Array of shape (3,) with values in [-1, 1].

    Returns:
        Array of shape (4,): wheel velocities in ticks/sec [FL, FR, RL, RR].
    """
    vx, vy, omega = interpret_action(action_normalized)
    wheel_vels = twist_to_wheel_velocities(vx, vy, omega)
    return wheel_vels_to_ticks_per_sec(wheel_vels)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class LoadedPolicy:
    """Callable wrapper around a loaded policy artifact.

    Exposes a uniform interface across stateless and recurrent artifacts in
    both ``.pt`` and ``.onnx`` formats. Stateless artifacts get a no-op
    ``reset()``; recurrent artifacts get a ``reset()`` that zeros hidden
    state.

    Hidden-state shape, reset semantics, determinism, and thread-safety
    rules are pinned in the "Recurrent hidden-state contract" section of
    this module's top-level docstring. Loader, exporter, and inference-
    node edits must hold that contract.
    """

    is_recurrent: bool = False

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Zero hidden state. No-op for stateless artifacts."""
        return None


class _TorchPolicy(LoadedPolicy):
    """Stateless TorchScript policy."""

    def __init__(self, model: Any, obs_dim: int) -> None:
        import torch  # local import keeps torch optional at module import time

        self._model = model
        self._obs_dim = obs_dim
        self._torch = torch

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            t = self._torch.as_tensor(obs, dtype=self._torch.float32).reshape(
                1, self._obs_dim
            )
            out = self._model(t)
            return out.squeeze(0).numpy()


class _RecurrentTorchPolicy(_TorchPolicy):
    """TorchScript recurrent policy: hidden state lives in the scripted module."""

    is_recurrent = True

    def reset(self) -> None:
        self._model.reset()


class _OnnxPolicy(LoadedPolicy):
    """Stateless ONNX policy."""

    def __init__(self, sess: Any, obs_dim: int) -> None:
        self._sess = sess
        self._obs_dim = obs_dim
        self._input_name = sess.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_f32 = obs.astype(np.float32).reshape(1, self._obs_dim)
        result = self._sess.run(None, {self._input_name: obs_f32})
        return result[0].squeeze(0)


class _RecurrentOnnxPolicy(LoadedPolicy):
    """ONNX recurrent policy: loader owns ``h_in`` / ``c_in`` numpy buffers.

    Mirrors the ``rsl_rl.models.rnn_model._OnnxRNNModel`` contract: GRU
    advertises ``(obs, h_in)`` -> ``(actions, h_out)``; LSTM advertises
    ``(obs, h_in, c_in)`` -> ``(actions, h_out, c_out)``. Per-tick the
    loader feeds the cached state and writes back the returned state.
    ``reset()`` zeros the cached buffers.
    """

    is_recurrent = True

    def __init__(self, sess: Any, obs_dim: int) -> None:
        self._sess = sess
        self._obs_dim = obs_dim

        input_specs = {inp.name: inp for inp in sess.get_inputs()}
        if "obs" not in input_specs or "h_in" not in input_specs:
            raise ValueError(
                "Recurrent ONNX policy must declare inputs 'obs' and 'h_in'; "
                f"got {sorted(input_specs)}"
            )
        self._has_c = "c_in" in input_specs

        output_names = [out.name for out in sess.get_outputs()]
        if "actions" not in output_names or "h_out" not in output_names:
            raise ValueError(
                "Recurrent ONNX policy must declare outputs 'actions' and 'h_out'; "
                f"got {output_names}"
            )
        if self._has_c and "c_out" not in output_names:
            raise ValueError(
                "Recurrent ONNX policy with 'c_in' input must declare 'c_out' output; "
                f"got {output_names}"
            )
        self._output_names = output_names

        self._h_shape = _resolve_hidden_shape(input_specs["h_in"].shape)
        self._h_in = np.zeros(self._h_shape, dtype=np.float32)
        if self._has_c:
            self._c_shape = _resolve_hidden_shape(input_specs["c_in"].shape)
            self._c_in = np.zeros(self._c_shape, dtype=np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        feed: dict[str, np.ndarray] = {
            "obs": obs.astype(np.float32).reshape(1, self._obs_dim),
            "h_in": self._h_in,
        }
        if self._has_c:
            feed["c_in"] = self._c_in
        results = self._sess.run(self._output_names, feed)
        named = dict(zip(self._output_names, results))
        # Write back hidden state so the next call sees evolved state.
        self._h_in = named["h_out"].astype(np.float32, copy=False)
        if self._has_c:
            self._c_in = named["c_out"].astype(np.float32, copy=False)
        return named["actions"].squeeze(0)

    def reset(self) -> None:
        self._h_in = np.zeros(self._h_shape, dtype=np.float32)
        if self._has_c:
            self._c_in = np.zeros(self._c_shape, dtype=np.float32)


def _resolve_hidden_shape(raw_shape: list) -> tuple[int, ...]:
    """Turn an ONNX input shape into a concrete (num_layers, 1, hidden_size).

    rsl_rl exports recurrent ports with shape ``(num_layers, batch, hidden_size)``.
    Dynamic dims (None / symbolic names from ``torch.onnx.export``) collapse
    to 1 — the deployed inference loop is single-environment, batch-1.
    """
    resolved: list[int] = []
    for dim in raw_shape:
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        else:
            resolved.append(1)
    return tuple(resolved)


def _read_sidecar(path: Path) -> dict | None:
    """Return the ``<stem>.json`` sidecar for a model path, or None if missing."""
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text())


def load_policy(path: str | Path, variant: PolicyVariant) -> LoadedPolicy:
    """Load a trained policy model and return a :class:`LoadedPolicy`.

    Supports:
        - ``.pt``  (TorchScript via ``torch.jit.load``)
        - ``.onnx`` (ONNX Runtime)

    Stateless and recurrent artifacts are dispatched at load time:
        - Recurrence is read from the ``<stem>.json`` sidecar's
          ``is_recurrent`` field when present.
        - When the sidecar is absent, recurrence is inferred from the
          loaded artifact (TorchScript: a callable ``reset`` attribute;
          ONNX: a port named ``h_in``).

    The returned object is callable — existing call sites that do
    ``policy = load_policy(...); policy(obs)`` continue to work unchanged.
    Recurrent variants expose ``.reset()`` to zero hidden state at episode
    boundaries; stateless variants expose a no-op ``.reset()`` so callers
    can fire it unconditionally.

    Args:
        path: Path to the model file.
        variant: Policy variant. Validated against the sidecar's
            ``policy_variant`` when the sidecar is present.

    Returns:
        :class:`LoadedPolicy` (callable; ``obs -> action``).

    Raises:
        ValueError: Unsupported file extension, or sidecar ``policy_variant``
            disagrees with the ``variant`` argument.
    """
    path = Path(path)
    obs_dim = variant.obs_dim

    sidecar = _read_sidecar(path)
    if sidecar is not None:
        sidecar_variant = sidecar.get("policy_variant")
        if sidecar_variant is not None and sidecar_variant != variant.name:
            raise ValueError(
                f"Sidecar at {path.with_suffix('.json')} records "
                f"policy_variant={sidecar_variant!r}, but load_policy() was "
                f"called with variant={variant.name!r}. Refusing to load a "
                f"mis-labeled artifact."
            )

    if path.suffix == ".pt":
        import torch

        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()

        if sidecar is not None and "is_recurrent" in sidecar:
            is_recurrent = bool(sidecar["is_recurrent"])
        else:
            reset_attr = getattr(model, "reset", None)
            is_recurrent = callable(reset_attr)

        if is_recurrent:
            if not callable(getattr(model, "reset", None)):
                raise ValueError(
                    f"TorchScript artifact at {path} is marked recurrent but "
                    f"exposes no callable .reset() — hidden state cannot be "
                    f"zeroed at episode boundaries."
                )
            return _RecurrentTorchPolicy(model, obs_dim)
        return _TorchPolicy(model, obs_dim)

    if path.suffix == ".onnx":
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path))
        input_names = {inp.name for inp in sess.get_inputs()}

        if sidecar is not None and "is_recurrent" in sidecar:
            is_recurrent = bool(sidecar["is_recurrent"])
        else:
            is_recurrent = "h_in" in input_names

        if is_recurrent:
            return _RecurrentOnnxPolicy(sess, obs_dim)
        return _OnnxPolicy(sess, obs_dim)

    raise ValueError(
        f"Unsupported model format: {path.suffix} (expected .pt or .onnx)"
    )


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def benchmark_policy(
    policy,
    variant: PolicyVariant,
    n_iters: int = 100,
) -> dict[str, float]:
    """Measure inference latency for a loaded policy.

    Args:
        policy: Callable returned by load_policy().
        variant: Policy variant (determines obs dimensions).
        n_iters: Number of inference iterations to time.

    Returns:
        Dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms'.
    """
    import time

    obs = np.zeros(variant.obs_dim, dtype=np.float32)

    # Warmup
    for _ in range(10):
        policy(obs)

    times: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        policy(obs)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }
