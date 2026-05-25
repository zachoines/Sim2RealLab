"""Mecanum wheel kinematics — single source of truth for sim and real.

The KINEMATIC_MATRIX defined here is imported by both the ROS2 hardware
nodes and the Isaac Lab action term (actions.py converts it to a PyTorch
tensor).  This eliminates any risk of sim-real kinematic drift.

Note: WHEEL_AXIS_SIGNS are NOT baked into this matrix.  Those signs
compensate for USD revolute-joint axis orientation in Isaac Sim and
are applied separately in actions.py.  The real motors don't need
that correction.
"""

import numpy as np

from strafer_shared.constants import (
    K,
    WHEEL_RADIUS,
    WHEEL_BASE,
    TRACK_WIDTH,
    MAX_LINEAR_VEL,
    MAX_ANGULAR_VEL,
    MAX_WHEEL_ANGULAR_VEL,
    RADIANS_TO_ENCODER_TICKS,
    ENCODER_TICKS_TO_RADIANS,
)

# Combined lever arm for rotation
_K = K
_r = WHEEL_RADIUS

# Forward kinematic matrix: body velocities [vx, vy, omega] -> wheel angular velocities
# Rows: [FL, FR, RL, RR]
KINEMATIC_MATRIX = np.array(
    [
        [1.0 / _r, -1.0 / _r, -_K / _r],  # wheel_1 = Front-Left
        [1.0 / _r, 1.0 / _r, _K / _r],  # wheel_2 = Front-Right
        [1.0 / _r, 1.0 / _r, -_K / _r],  # wheel_3 = Rear-Left
        [1.0 / _r, -1.0 / _r, _K / _r],  # wheel_4 = Rear-Right
    ],
    dtype=np.float64,
)

# Inverse kinematic matrix: wheel angular velocities -> body velocities [vx, vy, omega]
INVERSE_KINEMATIC_MATRIX = np.array(
    [
        [_r / 4.0, _r / 4.0, _r / 4.0, _r / 4.0],  # vx
        [-_r / 4.0, _r / 4.0, _r / 4.0, -_r / 4.0],  # vy
        [-_r / (4.0 * _K), _r / (4.0 * _K), -_r / (4.0 * _K), _r / (4.0 * _K)],  # omega
    ],
    dtype=np.float64,
)

def normalized_to_wheel_velocities(
    vx_norm: float, vy_norm: float, omega_norm: float
) -> np.ndarray:
    """Convert normalized [-1,1] commands to wheel angular velocities in rad/s.

    This is the same computation as MecanumWheelAction.process_actions()
    but without motor dynamics / delay / slew rate (those are sim-only).

    Args:
        vx_norm: Forward velocity, normalized [-1, 1].
        vy_norm: Strafe left velocity, normalized [-1, 1].
        omega_norm: CCW rotation, normalized [-1, 1].

    Returns:
        Array of shape (4,): wheel angular velocities [FL, FR, RL, RR] in rad/s.
    """
    vx_norm = np.clip(vx_norm, -1.0, 1.0)
    vy_norm = np.clip(vy_norm, -1.0, 1.0)
    omega_norm = np.clip(omega_norm, -1.0, 1.0)

    body_vel = np.array(
        [
            vx_norm * MAX_LINEAR_VEL,
            vy_norm * MAX_LINEAR_VEL,
            omega_norm * MAX_ANGULAR_VEL,
        ],
        dtype=np.float64,
    )

    wheel_vels = KINEMATIC_MATRIX @ body_vel
    wheel_vels = np.clip(wheel_vels, -MAX_WHEEL_ANGULAR_VEL, MAX_WHEEL_ANGULAR_VEL)
    return wheel_vels


def twist_to_wheel_velocities(vx: float, vy: float, omega: float) -> np.ndarray:
    """Convert physical body velocities (m/s, rad/s) to wheel angular velocities.

    Args:
        vx: Forward velocity in m/s.
        vy: Strafe left velocity in m/s.
        omega: CCW rotation in rad/s.

    Returns:
        Array of shape (4,): wheel angular velocities [FL, FR, RL, RR] in rad/s.
    """
    body_vel = np.array([vx, vy, omega], dtype=np.float64)
    wheel_vels = KINEMATIC_MATRIX @ body_vel
    wheel_vels = np.clip(wheel_vels, -MAX_WHEEL_ANGULAR_VEL, MAX_WHEEL_ANGULAR_VEL)
    return wheel_vels


def l1_clamp_twist(
    vx: float, vy: float, omega: float,
    *,
    vel_cap_linear_m_s: float,
    vel_cap_angular_rad_s: float,
) -> tuple[float, float, float]:
    """Cap (vx, vy) jointly under an L1 budget; cap omega independently.

    The chassis cannot reach max forward + max lateral simultaneously
    because each mecanum wheel has a single motor with a per-wheel cap
    (``MAX_WHEEL_ANGULAR_VEL``). At ``(vx, vy) = (MAX_LINEAR_VEL,
    MAX_LINEAR_VEL)`` the FL wheel inverse kinematics demand
    ``(vx + vy) / WHEEL_RADIUS`` rad/s — roughly 2× the cap.

    Scaling ``(vx, vy)`` by the same factor keeps the commanded
    heading; clipping each axis independently would skew it. omega
    clamps independently because it routes through a different
    per-wheel sign-correction pathway.

    Scalar form for single-tick callers (deployment-time inference
    node, unit tests). Sim's num_envs-parallel path uses
    ``l1_clamp_twist_batched`` below; the two forms must produce
    identical outputs for identical inputs (asserted in
    strafer_lab/tests/test_action_clamp.py).
    """
    l1 = abs(vx) + abs(vy)
    if l1 > vel_cap_linear_m_s and l1 > 0.0:
        scale = vel_cap_linear_m_s / l1
        vx *= scale
        vy *= scale
    if omega > vel_cap_angular_rad_s:
        omega = vel_cap_angular_rad_s
    elif omega < -vel_cap_angular_rad_s:
        omega = -vel_cap_angular_rad_s
    return float(vx), float(vy), float(omega)


def l1_clamp_twist_batched(
    body_velocities,
    *,
    vel_cap_linear_m_s: float,
    vel_cap_angular_rad_s: float,
):
    """Torch-vectorized form of :func:`l1_clamp_twist` for sim training.

    Args:
        body_velocities: Tensor of shape ``(..., 3)`` whose last axis is
            ``(vx, vy, omega)`` in (m/s, m/s, rad/s).
        vel_cap_linear_m_s: L1 budget shared by ``vx``/``vy``.
        vel_cap_angular_rad_s: Independent clamp on ``omega``.

    Returns:
        Tensor with the same shape and dtype as ``body_velocities``,
        clamped element-wise along the leading axes.

    The math is identical to the scalar form: ``(vx, vy)`` are scaled
    jointly when ``|vx| + |vy|`` exceeds the linear cap (heading
    preserved); ``omega`` is clamped per-element. Returned tensor is a
    new allocation; the input is not modified.
    """
    import torch  # local import keeps torch optional at module load

    vx = body_velocities[..., 0]
    vy = body_velocities[..., 1]
    omega = body_velocities[..., 2]

    l1 = vx.abs() + vy.abs()
    safe_l1 = l1.clamp(min=torch.finfo(l1.dtype).tiny)
    scale = torch.where(
        l1 > vel_cap_linear_m_s,
        vel_cap_linear_m_s / safe_l1,
        torch.ones_like(l1),
    )
    vx_out = vx * scale
    vy_out = vy * scale
    omega_out = omega.clamp(
        min=-vel_cap_angular_rad_s, max=vel_cap_angular_rad_s
    )
    return torch.stack([vx_out, vy_out, omega_out], dim=-1)


def wheel_vels_to_ticks_per_sec(wheel_vels_rad_s: np.ndarray) -> np.ndarray:
    """Convert wheel angular velocities (rad/s) to encoder ticks per second.

    Args:
        wheel_vels_rad_s: Array of shape (4,) in rad/s.

    Returns:
        Array of shape (4,) in ticks/sec (for RoboClaw SpeedM1/M2).
    """
    return wheel_vels_rad_s * RADIANS_TO_ENCODER_TICKS


def encoder_ticks_to_body_velocity(
    ticks_per_sec: np.ndarray,
) -> tuple[float, float, float]:
    """Convert encoder tick rates to body velocity using inverse kinematics.

    Args:
        ticks_per_sec: Array of shape (4,) [FL, FR, RL, RR] in ticks/sec.

    Returns:
        Tuple of (vx, vy, omega) in (m/s, m/s, rad/s).
    """
    wheel_vels = ticks_per_sec * ENCODER_TICKS_TO_RADIANS
    body_vel = INVERSE_KINEMATIC_MATRIX @ wheel_vels
    return float(body_vel[0]), float(body_vel[1]), float(body_vel[2])
