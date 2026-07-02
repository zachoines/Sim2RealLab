"""Pure helpers for the bridge's proprioceptive telemetry.

The parity-critical logic here is the wheel-joint index resolution: the
inference obs pipeline reconstructs body velocity from wheel-FK, so the
bridge must emit wheel velocities in the canonical
:data:`~strafer_shared.constants.WHEEL_JOINT_NAMES` order. That mapping is
the one thing worth unit-testing without a simulator, so it lives here
(pure — no ``warp`` / Isaac Sim / ``sensor_msgs`` imports) while the actual
message construction stays in :mod:`strafer_lab.bridge.async_publisher`
(Kit-only). Importable in the pxr-free autonomy test suite.
"""

from __future__ import annotations

from strafer_shared.constants import WHEEL_JOINT_NAMES


def resolve_wheel_indices(joint_names) -> list[int]:
    """Indices of the wheel drive joints within an articulation's DOF order.

    Mirrors the training-side ``_get_wheel_joint_indices`` lookup so the sim
    bridge and the gym env pull the same four columns from ``joint_vel`` /
    ``joint_pos``. Returns the indices in canonical
    ``WHEEL_JOINT_NAMES`` order (``[FL, FR, RL, RR]``).
    """
    names = list(joint_names)
    indices: list[int] = []
    for name in WHEEL_JOINT_NAMES:
        try:
            indices.append(names.index(name))
        except ValueError:
            raise ValueError(
                f"Wheel joint {name!r} not found in articulation joints {names}"
            )
    return indices


def ordered_wheel_values(indices, values) -> list[float]:
    """Select per-wheel scalars in canonical ``WHEEL_JOINT_NAMES`` order.

    ``indices`` comes from :func:`resolve_wheel_indices`; ``values`` is the
    full per-joint array (any indexable of floats). Returns a plain
    ``float`` list aligned element-for-element with ``WHEEL_JOINT_NAMES``.
    """
    return [float(values[i]) for i in indices]
