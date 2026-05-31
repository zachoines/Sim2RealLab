"""Unit tests for the teleop-only PhysX override spec.

Pure stdlib — runs in ``.venv_harness`` with no Isaac Sim. Verifies the
override produces exactly the cfg kwargs the set fields imply, that an
empty override is a no-op, and that the routing predicates correctly split
articulation-spawn fields from scene-PhysX fields.
"""

from __future__ import annotations

import pytest

from strafer_lab.tools.teleop_physics import (
    STABLE_CHEAP_PRESET,
    TeleopPhysicsOverride,
    articulation_prop_overrides,
    rigid_prop_overrides,
    scene_physx_overrides,
)


class TestNoop:
    def test_empty_override_is_noop(self):
        ov = TeleopPhysicsOverride()
        assert ov.is_noop()
        assert not ov.touches_articulation()
        assert not ov.touches_scene_physx()
        assert articulation_prop_overrides(ov) == {}
        assert rigid_prop_overrides(ov) == {}
        assert scene_physx_overrides(ov) == {}

    def test_any_field_set_is_not_noop(self):
        assert not TeleopPhysicsOverride(
            solver_position_iteration_count=8,
        ).is_noop()
        assert not TeleopPhysicsOverride(enable_stabilization=True).is_noop()


class TestRouting:
    def test_solver_iters_route_to_articulation_only(self):
        ov = TeleopPhysicsOverride(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        )
        assert ov.touches_articulation()
        assert not ov.touches_scene_physx()
        assert articulation_prop_overrides(ov) == {
            "solver_position_iteration_count": 8,
            "solver_velocity_iteration_count": 4,
        }
        assert scene_physx_overrides(ov) == {}

    def test_depenetration_routes_to_rigid_props(self):
        ov = TeleopPhysicsOverride(max_depenetration_velocity=1.0)
        assert ov.touches_articulation()  # rigid-body prop counts as spawn-side
        assert rigid_prop_overrides(ov) == {"max_depenetration_velocity": 1.0}
        # Not an articulation-root prop, and not a scene prop.
        assert articulation_prop_overrides(ov) == {}
        assert scene_physx_overrides(ov) == {}

    def test_scene_fields_route_to_scene_physx_only(self):
        ov = TeleopPhysicsOverride(
            enable_stabilization=True,
            bounce_threshold_velocity=0.3,
            friction_offset_threshold=0.01,
        )
        assert ov.touches_scene_physx()
        assert not ov.touches_articulation()
        assert scene_physx_overrides(ov) == {
            "enable_stabilization": True,
            "bounce_threshold_velocity": 0.3,
            "friction_offset_threshold": 0.01,
        }
        assert articulation_prop_overrides(ov) == {}
        assert rigid_prop_overrides(ov) == {}

    def test_thresholds_route_to_articulation_root(self):
        ov = TeleopPhysicsOverride(
            stabilization_threshold=0.01, sleep_threshold=0.005,
        )
        assert articulation_prop_overrides(ov) == {
            "stabilization_threshold": 0.01,
            "sleep_threshold": 0.005,
        }


class TestValidation:
    @pytest.mark.parametrize("field,bad", [
        ("solver_position_iteration_count", -1),
        ("solver_velocity_iteration_count", -2),
        ("max_depenetration_velocity", -0.5),
        ("bounce_threshold_velocity", -1.0),
        ("friction_offset_threshold", -0.01),
        ("stabilization_threshold", -0.1),
        ("sleep_threshold", -0.1),
    ])
    def test_negative_rejected(self, field, bad):
        ov = TeleopPhysicsOverride(**{field: bad})
        with pytest.raises(ValueError, match=field):
            ov.validate()

    def test_zero_iterations_allowed(self):
        # velocity iters == 0 is the PhysX default and legal.
        TeleopPhysicsOverride(
            solver_velocity_iteration_count=0,
        ).validate()


class TestStableCheapPreset:
    def test_preset_is_cheaper_and_more_stable(self):
        p = STABLE_CHEAP_PRESET
        p.validate()
        # Cheaper: fewer solver iterations than the shared 32/16.
        assert p.solver_position_iteration_count < 32
        assert p.solver_velocity_iteration_count < 16
        # More stable: the contact-stabilization knobs are engaged.
        assert p.enable_stabilization is True
        assert p.max_depenetration_velocity is not None
        assert p.stabilization_threshold is not None
        # Touches both override surfaces.
        assert p.touches_articulation()
        assert p.touches_scene_physx()
