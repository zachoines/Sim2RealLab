"""Policy interface -- observation/action contract for sim and real.

This module defines how observations are assembled and actions are interpreted.
It is the single source of truth for the policy I/O contract. Both the Isaac Lab
gym environment config and the ROS2 inference node reference this module.

Observation specs must match the simulation's observation groups in:
  source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py (L249-258)
Normalization scales must match:
  source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py (L225-234)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from strafer_shared.constants import (
    DEPTH_SCALE,
    ENCODER_VEL_SCALE,
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
    ObsField("goal_relative", 2, 1.0),
    ObsField("last_action", 3, 1.0),
)

_DEPTH_FIELDS: tuple[ObsField, ...] = _NOCAM_FIELDS + (
    ObsField("depth_image", 4800, DEPTH_SCALE),
)


class PolicyVariant(Enum):
    """Policy observation variants matching Isaac Lab environment configs."""

    NOCAM = _NOCAM_FIELDS  # 15 dims
    DEPTH = _DEPTH_FIELDS  # 4815 dims

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


def load_policy(path: str | Path, variant: PolicyVariant):
    """Load a trained policy model and return a callable: obs → action.

    Supports:
        - .pt  (TorchScript via torch.jit.load)
        - .onnx (ONNX Runtime)

    Args:
        path: Path to the model file.
        variant: Policy variant (used for validation).

    Returns:
        Callable that takes a numpy observation array and returns a numpy action array.
    """
    path = Path(path)
    obs_dim = variant.obs_dim

    if path.suffix == ".pt":
        import torch

        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()

        def _infer_pt(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, obs_dim)
                out = model(t)
                return out.squeeze(0).numpy()

        return _infer_pt

    elif path.suffix == ".onnx":
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path))
        input_name = sess.get_inputs()[0].name

        def _infer_onnx(obs: np.ndarray) -> np.ndarray:
            obs_f32 = obs.astype(np.float32).reshape(1, obs_dim)
            result = sess.run(None, {input_name: obs_f32})
            return result[0].squeeze(0)

        return _infer_onnx

    else:
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
