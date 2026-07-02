"""Tests for strafer_lab.bridge.proprio — the pure wheel-index logic.

Runs in the pxr-free autonomy suite via the ``strafer_lab`` namespace stub
installed by :mod:`conftest` (no Isaac Sim / Kit). The message construction
and live sim reads in :mod:`strafer_lab.bridge.async_publisher` import
``warp`` and are smoke-tested in-process by ``run_sim_in_the_loop.py`` inside
the Isaac Lab env; only the parity-critical index resolution is unit-tested
here.
"""

from __future__ import annotations

import pytest

from strafer_lab.bridge.proprio import (
    WHEEL_JOINT_NAMES,
    ordered_wheel_values,
    resolve_wheel_indices,
)


class TestResolveWheelIndices:
    def test_canonical_order_maps_to_identity(self):
        # Articulation already in canonical order → indices 0..3.
        assert resolve_wheel_indices(list(WHEEL_JOINT_NAMES)) == [0, 1, 2, 3]

    def test_indices_follow_canonical_names_not_dof_order(self):
        # A scrambled DOF layout with non-wheel joints interleaved. The
        # returned indices must select the wheels in canonical
        # WHEEL_JOINT_NAMES order regardless of the articulation's order.
        dof = [
            "some_passive_joint",
            WHEEL_JOINT_NAMES[2],  # RL at dof idx 1
            "another_joint",
            WHEEL_JOINT_NAMES[0],  # FL at dof idx 3
            WHEEL_JOINT_NAMES[3],  # RR at dof idx 4
            WHEEL_JOINT_NAMES[1],  # FR at dof idx 5
        ]
        idx = resolve_wheel_indices(dof)
        # Canonical order [FL, FR, RL, RR] → dof positions [3, 5, 1, 4].
        assert idx == [3, 5, 1, 4]
        # And selecting by those indices recovers canonical order.
        assert [dof[i] for i in idx] == list(WHEEL_JOINT_NAMES)

    def test_missing_wheel_joint_raises(self):
        dof = list(WHEEL_JOINT_NAMES[:3]) + ["not_a_wheel"]
        with pytest.raises(ValueError, match="wheel_4_drive"):
            resolve_wheel_indices(dof)

    def test_accepts_any_iterable(self):
        # tuple input (Articulation.joint_names may be a tuple) works too.
        assert resolve_wheel_indices(tuple(WHEEL_JOINT_NAMES)) == [0, 1, 2, 3]


class TestOrderedWheelValues:
    def test_selects_and_orders_values(self):
        # Full per-joint velocity array; wheels at scrambled indices.
        idx = [3, 5, 1, 4]
        full = [0.0, 9.9, 0.0, 1.0, 4.0, 2.0]  # FL=1,FR=2,RL=9.9? no — by idx
        vals = ordered_wheel_values(idx, full)
        assert vals == [full[3], full[5], full[1], full[4]]
        assert vals == [1.0, 2.0, 9.9, 4.0]

    def test_returns_plain_floats(self):
        vals = ordered_wheel_values([0, 1, 2, 3], [1, 2, 3, 4])
        assert vals == [1.0, 2.0, 3.0, 4.0]
        assert all(isinstance(v, float) for v in vals)

    def test_length_matches_wheel_count(self):
        vals = ordered_wheel_values([0, 1, 2, 3], [5.0, 6.0, 7.0, 8.0])
        assert len(vals) == len(WHEEL_JOINT_NAMES) == 4
