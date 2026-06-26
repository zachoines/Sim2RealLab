"""Subgoal-policy inference plumbing for the coverage capture driver.

Wraps the existing deployment inference primitives in
:mod:`strafer_shared.policy_interface` (``assemble_observation`` +
``interpret_action``) into one per-tick step the coverage driver calls to
turn a rolling subgoal into a body twist. The subgoal-derived observation
fields are computed here with the same geometry the env's observation terms
use (``goal_position_relative`` / ``goal_distance`` /
``goal_heading_to_goal``), so a checkpoint trained in-env sees the same
observation distribution at capture time.

Pure numpy — no torch, no Kit. The policy is the only injected dependency,
so the per-tick plumbing is unit-testable with a fake policy and a hand-built
observation, without a GPU or a checkpoint. The driver supplies the
non-subgoal base fields (imu, encoder, body velocity, last action) from the
live env and the rolling subgoal from its path cursor.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from strafer_shared.policy_interface import (
    PolicyVariant,
    assemble_observation,
    interpret_action,
)

# Observation keys this module fills in; the driver supplies the rest.
SUBGOAL_FIELD_KEYS: tuple[str, ...] = (
    "subgoal_relative",
    "subgoal_distance",
    "subgoal_heading_to_subgoal",
)


def subgoal_observation_fields(
    robot_xy: Sequence[float],
    robot_yaw: float,
    subgoal_xy: Sequence[float],
) -> dict[str, np.ndarray]:
    """Compute the subgoal-relative observation fields.

    Mirrors the env's subgoal observation terms: position in the robot's body
    frame, Euclidean distance, and the signed bearing error to the subgoal.
    The body-frame rotation is yaw-only — for a planar base the env's full
    quaternion inverse reduces to a rotation by ``-robot_yaw``.
    """
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    dx = float(subgoal_xy[0]) - rx
    dy = float(subgoal_xy[1]) - ry

    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)
    rel_x = cos_y * dx + sin_y * dy
    rel_y = -sin_y * dx + cos_y * dy

    distance = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx)
    heading_err = math.atan2(
        math.sin(bearing - robot_yaw), math.cos(bearing - robot_yaw),
    )

    return {
        "subgoal_relative": np.array([rel_x, rel_y], dtype=np.float32),
        "subgoal_distance": np.array([distance], dtype=np.float32),
        "subgoal_heading_to_subgoal": np.array([heading_err], dtype=np.float32),
    }


def assemble_subgoal_observation(
    base_fields: Mapping[str, Any],
    robot_xy: Sequence[float],
    robot_yaw: float,
    subgoal_xy: Sequence[float],
    *,
    variant: PolicyVariant = PolicyVariant.NOCAM_SUBGOAL,
) -> np.ndarray:
    """Build the flat policy observation for one tick.

    ``base_fields`` carries the non-subgoal raw observation values the driver
    reads off the env (``imu_accel`` / ``imu_gyro`` / ``encoder_vels_ticks`` /
    ``body_velocity_xy`` / ``last_action``); the subgoal fields are filled in
    here. ``assemble_observation`` validates dims and applies the variant's
    normalization scales.
    """
    raw = dict(base_fields)
    raw.update(subgoal_observation_fields(robot_xy, robot_yaw, subgoal_xy))
    return assemble_observation(raw, variant)


def step_subgoal_controller(
    policy: Callable[[np.ndarray], Any],
    base_fields: Mapping[str, Any],
    robot_xy: Sequence[float],
    robot_yaw: float,
    subgoal_xy: Sequence[float],
    *,
    variant: PolicyVariant = PolicyVariant.NOCAM_SUBGOAL,
) -> tuple[float, float, float]:
    """Run one inference step: assemble obs, run the policy, denormalize.

    Returns the physical body twist ``(vx, vy, omega_z)`` in (m/s, m/s,
    rad/s) the driver feeds to the env's mecanum action contract.
    """
    obs = assemble_subgoal_observation(
        base_fields, robot_xy, robot_yaw, subgoal_xy, variant=variant,
    )
    action_normalized = np.asarray(policy(obs), dtype=np.float32).reshape(-1)
    return interpret_action(action_normalized)
