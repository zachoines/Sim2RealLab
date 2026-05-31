"""Teleop-only PhysX overrides for the Strafer mecanum drivetrain.

Profiling the teleop loop showed it is PhysX-bound, and the operator
observed that the 40 free-spinning mecanum roller contacts pop / kick the
robot every 1-2 s *even at the shared config's expensive
solver_position/velocity_iteration_count = 32/16* and the default
decimation. That is the diagnostic: high iteration counts are not buying
stability, so the instability is a contact-event problem (restitution /
depenetration ejection at the roller-floor contact), not an
under-convergence problem. That opens the door to **cheaper and more
stable at once** — drop the brute-force iteration count (cheaper) and add
the contact-stabilization knobs that actually govern the pops (more
stable).

This module builds those overrides as pure data so they unit-test without
the Isaac Sim runtime, and — critically — it applies them only to a
**copy** of the shared :data:`strafer_lab.assets.STRAFER_CFG` via
``.replace()``. The shared articulation cfg and the RL-training physics
contract it encodes are never mutated; only the teleop capture variant
sees these values. The teleop driver exposes each field as a CLI knob so
the operator can sweep them live under ``--profile`` and watch the roller
behavior before any value is committed to the env config.

Two override surfaces, matching where PhysX reads each knob:

- **Articulation / rigid-body props** (on the robot spawn): solver
  iteration counts (cost) + ``max_depenetration_velocity`` (the "kick" —
  how violently an interpenetrating roller is ejected) +
  ``stabilization_threshold`` (below this body velocity, PhysX runs an
  extra per-contact stabilization pass).
- **Scene PhysX cfg** (``sim.physics``): ``enable_stabilization`` (the
  cheap global stability pass, already proven on the ProcRoom path),
  ``bounce_threshold_velocity`` (contacts above this restitute — the
  roller "bounce"), and ``friction_offset_threshold`` (contact-patch
  friction resolution; the default 4 cm is coarse for a 96 mm wheel).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeleopPhysicsOverride:
    """A teleop-only PhysX override spec. ``None`` fields keep the env default.

    Every field defaults to ``None`` so an all-``None`` override is a no-op
    (today's behavior, the shared config untouched). The teleop driver maps
    one CLI flag to each field.
    """

    # -- robot articulation / rigid-body props (cost + kick) --------------
    solver_position_iteration_count: int | None = None
    solver_velocity_iteration_count: int | None = None
    max_depenetration_velocity: float | None = None
    stabilization_threshold: float | None = None
    sleep_threshold: float | None = None

    # -- scene PhysX cfg (bounce + friction + global stabilization) -------
    enable_stabilization: bool | None = None
    bounce_threshold_velocity: float | None = None
    friction_offset_threshold: float | None = None

    def is_noop(self) -> bool:
        """True when every field is unset (the override changes nothing)."""
        return all(
            getattr(self, f) is None
            for f in (
                "solver_position_iteration_count",
                "solver_velocity_iteration_count",
                "max_depenetration_velocity",
                "stabilization_threshold",
                "sleep_threshold",
                "enable_stabilization",
                "bounce_threshold_velocity",
                "friction_offset_threshold",
            )
        )

    def touches_articulation(self) -> bool:
        """True if any robot-spawn (articulation / rigid-body) field is set."""
        return any(
            getattr(self, f) is not None
            for f in (
                "solver_position_iteration_count",
                "solver_velocity_iteration_count",
                "max_depenetration_velocity",
                "stabilization_threshold",
                "sleep_threshold",
            )
        )

    def touches_scene_physx(self) -> bool:
        """True if any scene-level PhysX field is set."""
        return any(
            getattr(self, f) is not None
            for f in (
                "enable_stabilization",
                "bounce_threshold_velocity",
                "friction_offset_threshold",
            )
        )

    def validate(self) -> None:
        """Raise ``ValueError`` on out-of-range values."""
        for name in (
            "solver_position_iteration_count",
            "solver_velocity_iteration_count",
        ):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        for name in (
            "max_depenetration_velocity",
            "stabilization_threshold",
            "sleep_threshold",
            "bounce_threshold_velocity",
            "friction_offset_threshold",
        ):
            v = getattr(self, name)
            if v is not None and v < 0.0:
                raise ValueError(f"{name} must be >= 0, got {v}")


def articulation_prop_overrides(
    override: TeleopPhysicsOverride,
) -> dict[str, object]:
    """Return the ``ArticulationRootPropertiesCfg`` kwargs the override sets.

    Only the explicitly-set fields appear, so applying them with
    ``.replace(**kwargs)`` on a copy of the shared cfg leaves every other
    property at its original value.
    """
    out: dict[str, object] = {}
    if override.solver_position_iteration_count is not None:
        out["solver_position_iteration_count"] = (
            override.solver_position_iteration_count
        )
    if override.solver_velocity_iteration_count is not None:
        out["solver_velocity_iteration_count"] = (
            override.solver_velocity_iteration_count
        )
    if override.stabilization_threshold is not None:
        out["stabilization_threshold"] = override.stabilization_threshold
    if override.sleep_threshold is not None:
        out["sleep_threshold"] = override.sleep_threshold
    return out


def rigid_prop_overrides(
    override: TeleopPhysicsOverride,
) -> dict[str, object]:
    """Return the ``RigidBodyPropertiesCfg`` kwargs the override sets.

    ``max_depenetration_velocity`` lives on the rigid-body props (the
    robot spawn's ``rigid_props``), separate from the articulation root.
    """
    out: dict[str, object] = {}
    if override.max_depenetration_velocity is not None:
        out["max_depenetration_velocity"] = override.max_depenetration_velocity
    return out


def scene_physx_overrides(
    override: TeleopPhysicsOverride,
) -> dict[str, object]:
    """Return the ``PhysxCfg`` kwargs the override sets (scene-level)."""
    out: dict[str, object] = {}
    if override.enable_stabilization is not None:
        out["enable_stabilization"] = override.enable_stabilization
    if override.bounce_threshold_velocity is not None:
        out["bounce_threshold_velocity"] = override.bounce_threshold_velocity
    if override.friction_offset_threshold is not None:
        out["friction_offset_threshold"] = override.friction_offset_threshold
    return out


# A starting preset for the "stable AND cheap" hypothesis: halve the
# solver iterations (cheaper) and turn on every contact-stabilization knob
# that targets the roller pops (more stable). Exposed as a named preset so
# the operator can apply it with one flag, then sweep individual fields
# from there. Values are deliberately conservative starting points, not a
# tuned result — the measurement run is what tunes them.
STABLE_CHEAP_PRESET = TeleopPhysicsOverride(
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=4,
    max_depenetration_velocity=1.0,
    stabilization_threshold=0.01,
    sleep_threshold=0.005,
    enable_stabilization=True,
    bounce_threshold_velocity=0.5,
    friction_offset_threshold=0.01,
)
