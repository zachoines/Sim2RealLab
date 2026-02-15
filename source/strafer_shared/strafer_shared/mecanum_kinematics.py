"""Mecanum wheel kinematics -- shared between Isaac Lab sim and ROS2 hardware.

Replicates the exact kinematic matrix from:
  source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py (lines 166-171)

This module uses only NumPy (no PyTorch) so it runs on both the training
workstation and the Jetson.
"""

import numpy as np

from strafer_shared.constants import (
    WHEEL_RADIUS,
    WHEEL_BASE,
    TRACK_WIDTH,
    MAX_LINEAR_VEL,
    MAX_ANGULAR_VEL,
    MAX_WHEEL_ANGULAR_VEL,
    WHEEL_AXIS_SIGNS,
    RADIANS_TO_ENCODER_TICKS,
    ENCODER_TICKS_TO_RADIANS,
)

# Combined lever arm for rotation
_K = (WHEEL_BASE / 2.0) + (TRACK_WIDTH / 2.0)
_r = WHEEL_RADIUS

# Forward kinematic matrix: body velocities [vx, vy, omega] -> wheel angular velocities
# Rows: [FL, FR, RL, RR]
KINEMATIC_MATRIX = np.array([
    [ 1.0 / _r, -1.0 / _r, -_K / _r],   # wheel_1 = Front-Left
    [ 1.0 / _r,  1.0 / _r,  _K / _r],   # wheel_2 = Front-Right
    [ 1.0 / _r,  1.0 / _r, -_K / _r],   # wheel_3 = Rear-Left
    [ 1.0 / _r, -1.0 / _r,  _K / _r],   # wheel_4 = Rear-Right
], dtype=np.float64)

# Inverse kinematic matrix: wheel angular velocities -> body velocities [vx, vy, omega]
INVERSE_KINEMATIC_MATRIX = np.array([
    [ _r / 4.0,  _r / 4.0,  _r / 4.0,  _r / 4.0],                          # vx
    [-_r / 4.0,  _r / 4.0,  _r / 4.0, -_r / 4.0],                          # vy
    [-_r / (4.0 * _K), _r / (4.0 * _K), -_r / (4.0 * _K), _r / (4.0 * _K)],  # omega
], dtype=np.float64)

_SIGNS = np.array(WHEEL_AXIS_SIGNS, dtype=np.float64)


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

    body_vel = np.array([
        vx_norm * MAX_LINEAR_VEL,
        vy_norm * MAX_LINEAR_VEL,
        omega_norm * MAX_ANGULAR_VEL,
    ], dtype=np.float64)

    wheel_vels = KINEMATIC_MATRIX @ body_vel
    wheel_vels *= _SIGNS
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
    wheel_vels *= _SIGNS
    wheel_vels = np.clip(wheel_vels, -MAX_WHEEL_ANGULAR_VEL, MAX_WHEEL_ANGULAR_VEL)
    return wheel_vels


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
    wheel_vels = (ticks_per_sec * ENCODER_TICKS_TO_RADIANS) / _SIGNS
    body_vel = INVERSE_KINEMATIC_MATRIX @ wheel_vels
    return float(body_vel[0]), float(body_vel[1]), float(body_vel[2])
