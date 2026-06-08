"""Parity tests for the policy observation contract.

The `body_velocity_xy` obs term must mirror the real-robot signal chain
(USD wheel joint velocity -> mecanum inverse FK -> body twist), matching
what the Jetson chassis driver computes for ``/strafer/odom`` from
``/strafer/joint_states.velocity``. Historically this term read
``robot.data.root_lin_vel_b`` (sim ground truth), which silently diverged
from the deployment signal chain in three ways: it carried no encoder
noise, no wheel slip, and no wheel-radius miscalibration.

These tests pin the FK rewrite:

1. Self-consistency of the kinematic matrices in ``strafer_shared``:
   ``INVERSE_KINEMATIC_MATRIX @ KINEMATIC_MATRIX`` is identity, and
   ``body -> wheels -> body`` round-trips to machine epsilon.
2. ``body_velocity_xy`` returns ``(vx, vy)`` derived from
   ``wheel_encoder_velocities(env)`` rather than from
   ``robot.data.root_lin_vel_b``.
3. Encoder noise injected at the joint-velocity tensor propagates
   through FK into ``body_velocity_xy`` with the expected variance
   structure (Gaussian * linear-FK -> Gaussian with FK-transformed
   covariance).

These tests do not launch Isaac Sim. They stub a minimal ``env`` /
``robot`` graph that exposes the attributes the obs functions read.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from strafer_lab.tasks.navigation.mdp.observations import (
    ENCODER_TICKS_TO_RADIANS,
    WHEEL_JOINT_NAMES,
    body_velocity_xy,
    wheel_encoder_velocities,
)
from strafer_shared.mecanum_kinematics import (
    INVERSE_KINEMATIC_MATRIX,
    KINEMATIC_MATRIX,
)


# -----------------------------------------------------------------------------
# Stub env / robot graph
# -----------------------------------------------------------------------------


def _make_stub_env(wheel_vels_rad_s: torch.Tensor) -> SimpleNamespace:
    """Build a minimal ``env`` whose ``robot.data.joint_vel`` mirrors the
    given wheel angular velocities. Joint name order matches WHEEL_JOINT_NAMES.
    """
    num_envs, num_wheels = wheel_vels_rad_s.shape
    assert num_wheels == 4, "expected 4 mecanum wheels"

    joint_vel_warp = wp.from_torch(wheel_vels_rad_s.contiguous())
    # ``root_lin_vel_b`` deliberately disagrees with the FK answer so the
    # tests can prove ``body_velocity_xy`` no longer reads it.
    bogus_ground_truth = torch.full(
        (num_envs, 3), float("nan"), dtype=wheel_vels_rad_s.dtype
    )
    root_lin_vel_b_warp = wp.from_torch(bogus_ground_truth.contiguous())

    robot = SimpleNamespace(
        joint_names=list(WHEEL_JOINT_NAMES),
        data=SimpleNamespace(
            joint_vel=joint_vel_warp,
            root_lin_vel_b=root_lin_vel_b_warp,
        ),
    )
    env = SimpleNamespace(scene={"robot": robot})
    return env


# -----------------------------------------------------------------------------
# Kinematic-matrix self-consistency
# -----------------------------------------------------------------------------


def test_inverse_kinematic_matrix_inverts_kinematic_matrix() -> None:
    identity = INVERSE_KINEMATIC_MATRIX @ KINEMATIC_MATRIX
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    "body_twist",
    [
        (1.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.7),
        (0.2, -0.3, 0.1),
        (-0.4, 0.6, -0.2),
    ],
)
def test_body_to_wheels_to_body_roundtrips(body_twist: tuple[float, float, float]) -> None:
    body = np.asarray(body_twist, dtype=np.float64)
    wheels = KINEMATIC_MATRIX @ body
    recovered = INVERSE_KINEMATIC_MATRIX @ wheels
    np.testing.assert_allclose(recovered, body, atol=1e-12)


# -----------------------------------------------------------------------------
# body_velocity_xy: encoder-derived FK behaviour
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vx", "vy", "omega"),
    [
        (1.0, 0.0, 0.0),     # pure forward
        (-0.5, 0.0, 0.0),    # pure backward
        (0.0, 0.7, 0.0),     # pure left strafe
        (0.0, -0.4, 0.0),    # pure right strafe
        (0.3, 0.2, 0.0),     # diagonal
    ],
)
def test_body_velocity_xy_recovers_commanded_translation(
    vx: float, vy: float, omega: float
) -> None:
    """Given the wheel rad/s that would produce ``(vx, vy, omega)`` body
    twist, ``body_velocity_xy`` recovers ``(vx, vy)`` (the IDEAL contract
    self-consistency check; no encoder noise applied)."""
    body = np.array([vx, vy, omega], dtype=np.float64)
    wheel_rad_s = KINEMATIC_MATRIX @ body  # (4,)
    wheel_vels = torch.from_numpy(wheel_rad_s).to(dtype=torch.float64).unsqueeze(0)  # (1, 4)

    env = _make_stub_env(wheel_vels)
    out = body_velocity_xy(env)

    assert out.shape == (1, 2)
    np.testing.assert_allclose(out.numpy()[0], (vx, vy), atol=1e-10)


def test_body_velocity_xy_ignores_root_lin_vel_b() -> None:
    """``body_velocity_xy`` must not read ``robot.data.root_lin_vel_b``;
    if it did, the NaN-filled bogus ground truth set up by the stub would
    propagate into the output."""
    wheel_vels = torch.full((1, 4), 5.0, dtype=torch.float64)  # any non-zero
    env = _make_stub_env(wheel_vels)
    out = body_velocity_xy(env)
    assert not torch.isnan(out).any(), (
        "body_velocity_xy is still reading root_lin_vel_b (got NaNs from the stub)"
    )


def test_body_velocity_xy_shares_signal_chain_with_wheel_encoder_velocities() -> None:
    """Both obs functions must read the same upstream ``joint_vel`` tensor.
    Specifically, FK applied to ``wheel_encoder_velocities(env)`` (converted
    back from ticks/s to rad/s) reproduces ``body_velocity_xy(env)`` byte-
    identically."""
    body = np.array([0.4, -0.25, 0.15], dtype=np.float64)
    wheel_rad_s = KINEMATIC_MATRIX @ body
    wheel_vels = torch.from_numpy(wheel_rad_s).to(dtype=torch.float64).unsqueeze(0)

    env = _make_stub_env(wheel_vels)
    ticks = wheel_encoder_velocities(env)  # (1, 4) in ticks/s
    rad_s_from_ticks = ticks * ENCODER_TICKS_TO_RADIANS
    inv_k = torch.from_numpy(INVERSE_KINEMATIC_MATRIX).to(dtype=ticks.dtype)
    expected = (rad_s_from_ticks @ inv_k.T)[:, :2]

    out = body_velocity_xy(env)
    torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)


# -----------------------------------------------------------------------------
# Noise propagation (REAL-contract math, no Isaac Lab obs-manager required)
# -----------------------------------------------------------------------------


def test_encoder_noise_propagates_through_fk_with_expected_variance() -> None:
    """When the upstream ``joint_vel`` carries Gaussian encoder noise with
    std ``sigma`` (rad/s), ``body_velocity_xy``'s output has the FK-
    transformed Gaussian covariance:

        Var(v_body) = INVERSE_K[:2,:] @ diag(sigma^2) @ INVERSE_K[:2,:].T

    With per-wheel iid noise and INVERSE_K rows for (vx, vy) of magnitude
    ``r/4`` each, the diagonal terms reduce to ``sigma^2 * r^2 / 4`` and
    the off-diagonal ``Cov(vx, vy) = 0``.

    This mirrors what would happen if the encoder noise were applied at
    or upstream of the function. The same-sample correlation case (where
    ``wheel_encoder_velocities`` and ``body_velocity_xy`` share a single
    noise draw at the obs-manager seam) is tracked separately."""
    from strafer_shared.constants import WHEEL_RADIUS

    num_envs = 50_000
    sigma_rad_s = 0.05  # arbitrary realistic encoder noise floor in rad/s
    torch.manual_seed(0xC0FFEE)

    base = torch.zeros((num_envs, 4), dtype=torch.float64)
    noise = torch.randn_like(base) * sigma_rad_s  # iid per wheel per env
    env = _make_stub_env(base + noise)

    out = body_velocity_xy(env).numpy()  # (num_envs, 2)
    cov = np.cov(out.T)  # (2, 2)

    expected_var = (WHEEL_RADIUS ** 2) * (sigma_rad_s ** 2) / 4.0
    # Sample-cov tolerance: ~1% with 50k samples is comfortable
    np.testing.assert_allclose(cov[0, 0], expected_var, rtol=0.02)
    np.testing.assert_allclose(cov[1, 1], expected_var, rtol=0.02)
    # Off-diagonal must be near zero (iid wheels + symmetric INV_K rows)
    assert abs(cov[0, 1]) < 0.05 * expected_var, (
        f"unexpected vx-vy covariance: {cov[0, 1]:.3e}"
    )
