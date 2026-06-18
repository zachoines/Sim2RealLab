"""Tests for strafer_lab.tools.grounding_injection.

Exercises the hard-negative goal-perturbation planner against a small
synthetic scene: probability gating, wrong_instance / wrong_room /
wrong_object candidate selection, the in-room → wrong_room fallbacks,
and the silent-drop case where no candidate exists. Pure Python.
"""

from __future__ import annotations

import random

import pytest

from strafer_lab.tools.grounding_injection import (
    InjectionPlan,
    plan_injection,
    resolve_target_room_idx,
)


# Two rooms: room 0 has two chairs + a table; room 1 has a lamp.
OBJECTS = [
    {"instance_id": 1, "label": "Chair", "position_3d": [1.0, 1.0, 0.0], "room_idx": 0},
    {"instance_id": 2, "label": "Chair", "position_3d": [2.0, 1.0, 0.0], "room_idx": 0},
    {"instance_id": 3, "label": "Table", "position_3d": [1.5, 2.0, 0.0], "room_idx": 0},
    {"instance_id": 4, "label": "Lamp", "position_3d": [6.0, 1.0, 0.0], "room_idx": 1},
]


class AlwaysRandom(random.Random):
    """random() returns a fixed value so the coin flip is deterministic."""

    def __init__(self, value: float) -> None:
        super().__init__(0)
        self._value = value

    def random(self) -> float:  # type: ignore[override]
        return self._value


def _plan(mode: str, *, rng=None, target=OBJECTS[0], objects=OBJECTS, prob=1.0) -> InjectionPlan:
    return plan_injection(
        mode=mode,
        probability=prob,
        rng=rng or AlwaysRandom(0.0),
        target_label=target["label"],
        target_instance_id=target["instance_id"],
        target_position_3d=target["position_3d"],
        target_room_idx=target["room_idx"],
        objects=objects,
    )


class TestProbabilityGate:
    def test_mode_off_never_injects(self):
        plan = _plan("off")
        assert plan.injection_mode is None
        assert plan.injection_mode_actual is None
        assert plan.injected is False
        assert plan.target_instance_id == 1
        assert plan.original_target_position_3d is None

    def test_coin_flip_no_records_no_request(self):
        plan = _plan("wrong_room", rng=AlwaysRandom(0.99), prob=0.3)
        assert plan.injection_mode is None
        assert plan.injection_mode_actual is None
        assert plan.target_position_3d == pytest.approx((1.0, 1.0, 0.0))

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown injection mode"):
            _plan("wrong_planet")

    def test_seeded_rng_reproduces_plan(self):
        plans = [
            _plan("wrong_room", rng=random.Random(7), prob=0.5) for _ in range(2)
        ]
        assert plans[0] == plans[1]


class TestWrongInstance:
    def test_picks_same_label_same_room_sibling(self):
        plan = _plan("wrong_instance")
        assert plan.injection_mode == "wrong_instance"
        assert plan.injection_mode_actual == "wrong_instance"
        assert plan.target_label == "Chair"
        assert plan.target_instance_id == 2  # the other chair in room 0
        assert plan.target_position_3d == pytest.approx((2.0, 1.0, 0.0))
        assert plan.original_target_position_3d == pytest.approx((1.0, 1.0, 0.0))

    def test_falls_back_to_wrong_room_without_sibling(self):
        plan = _plan("wrong_instance", target=OBJECTS[2])  # the only table
        assert plan.injection_mode == "wrong_instance"
        assert plan.injection_mode_actual == "wrong_room"
        assert plan.target_instance_id == 4  # the lamp in room 1

    def test_excludes_original_by_position_when_instance_unknown(self):
        # Queue rows carry instance_id=-1; the original chair must still
        # be excluded so the "sibling" can't be the target itself.
        plan = plan_injection(
            mode="wrong_instance",
            probability=1.0,
            rng=AlwaysRandom(0.0),
            target_label="Chair",
            target_instance_id=-1,
            target_position_3d=(1.0, 1.0, 0.0),
            target_room_idx=0,
            objects=OBJECTS,
        )
        assert plan.injection_mode_actual == "wrong_instance"
        assert plan.target_instance_id == 2


class TestWrongRoom:
    def test_picks_object_in_other_room(self):
        plan = _plan("wrong_room")
        assert plan.injection_mode_actual == "wrong_room"
        assert plan.target_instance_id == 4
        assert plan.target_position_3d == pytest.approx((6.0, 1.0, 0.0))

    def test_silent_drop_when_no_other_room(self):
        single_room = [o for o in OBJECTS if o["room_idx"] == 0]
        plan = _plan("wrong_room", objects=single_room)
        # Requested mode stays recorded; actual + original are null so the
        # drop rate is auditable downstream.
        assert plan.injection_mode == "wrong_room"
        assert plan.injection_mode_actual is None
        assert plan.original_target_position_3d is None
        assert plan.injected is False
        # The goal is untouched.
        assert plan.target_instance_id == 1
        assert plan.target_position_3d == pytest.approx((1.0, 1.0, 0.0))

    def test_drop_when_target_room_unknown(self):
        plan = plan_injection(
            mode="wrong_room",
            probability=1.0,
            rng=AlwaysRandom(0.0),
            target_label="Chair",
            target_instance_id=1,
            target_position_3d=(1.0, 1.0, 0.0),
            target_room_idx=None,
            objects=OBJECTS,
        )
        assert plan.injection_mode == "wrong_room"
        assert plan.injection_mode_actual is None


class TestWrongObject:
    def test_picks_different_label_same_room(self):
        # Target Chair(1) in room 0; the same-room different-label object is
        # the table — category confusion, the highest-value negative.
        plan = _plan("wrong_object")
        assert plan.injection_mode == "wrong_object"
        assert plan.injection_mode_actual == "wrong_object"
        assert plan.target_label == "Table"
        assert plan.target_instance_id == 3
        assert plan.target_position_3d == pytest.approx((1.5, 2.0, 0.0))
        assert plan.original_target_position_3d == pytest.approx((1.0, 1.0, 0.0))

    def test_does_not_pick_same_label_sibling(self):
        # The other chair in the room must NOT be chosen — that's
        # wrong_instance, not wrong_object.
        plan = _plan("wrong_object")
        assert plan.target_label != "Chair"

    def test_falls_back_to_wrong_room_without_other_label(self):
        # Target Lamp(4) is the only object in room 1, so there is no
        # same-room different-label candidate → fall back to wrong_room.
        plan = _plan("wrong_object", target=OBJECTS[3])
        assert plan.injection_mode == "wrong_object"
        assert plan.injection_mode_actual == "wrong_room"
        # Room 0 objects are the only other-room candidates.
        assert plan.target_instance_id in (1, 2, 3)

    def test_silent_drop_when_room_is_only_target(self):
        # A single-object scene: no same-room different-label candidate AND
        # no different-room candidate → silent drop, auditable.
        solo = [OBJECTS[0]]
        plan = _plan("wrong_object", objects=solo)
        assert plan.injection_mode == "wrong_object"
        assert plan.injection_mode_actual is None
        assert plan.original_target_position_3d is None
        assert plan.injected is False

    def test_probability_gate_applies(self):
        plan = _plan("wrong_object", rng=AlwaysRandom(0.99), prob=0.3)
        assert plan.injection_mode is None
        assert plan.injection_mode_actual is None

    def test_seeded_rng_reproduces_plan(self):
        plans = [
            _plan("wrong_object", rng=random.Random(11), prob=0.5) for _ in range(2)
        ]
        assert plans[0] == plans[1]


class TestResolveTargetRoomIdx:
    def test_matches_nearest_same_label_object(self):
        room = resolve_target_room_idx(
            target_label="chair",
            target_position_3d=(1.05, 1.0, 0.0),
            objects=OBJECTS,
        )
        assert room == 0

    def test_no_match_outside_radius(self):
        room = resolve_target_room_idx(
            target_label="chair",
            target_position_3d=(40.0, 40.0, 0.0),
            objects=OBJECTS,
        )
        assert room is None

    def test_label_mismatch_returns_none(self):
        room = resolve_target_room_idx(
            target_label="sofa",
            target_position_3d=(1.0, 1.0, 0.0),
            objects=OBJECTS,
        )
        assert room is None
