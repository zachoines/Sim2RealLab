"""Contract tests for ``PolicyVariant.NOCAM_SUBGOAL``.

Pins the seam the variant exists for: identical network input layout to
NOCAM (same dims, same scales, same field order) with renamed goal-referent
keys, and — critically — that the member is a *distinct* enum member rather
than a silent alias of NOCAM (Python Enums collapse members with equal
values into aliases, which would make the two variants indistinguishable at
load/dispatch time).
"""

from __future__ import annotations

import numpy as np
import pytest

from strafer_shared.policy_interface import PolicyVariant, assemble_observation


def test_nocam_subgoal_is_a_distinct_member_not_an_alias():
    assert PolicyVariant.NOCAM_SUBGOAL is not PolicyVariant.NOCAM
    assert PolicyVariant["NOCAM_SUBGOAL"].name == "NOCAM_SUBGOAL"
    # Four members: NOCAM, DEPTH, NOCAM_SUBGOAL, DEPTH_SUBGOAL (set closed here).
    assert len({v.name for v in PolicyVariant}) == 4


def test_nocam_subgoal_mirrors_nocam_shapes_and_scales():
    sub = PolicyVariant.NOCAM_SUBGOAL.fields
    base = PolicyVariant.NOCAM.fields
    assert PolicyVariant.NOCAM_SUBGOAL.obs_dim == PolicyVariant.NOCAM.obs_dim == 19
    assert len(sub) == len(base)
    for f_sub, f_base in zip(sub, base):
        assert f_sub.dims == f_base.dims
        assert f_sub.scale == f_base.scale


def test_goal_referent_keys_are_renamed():
    keys = {f.key for f in PolicyVariant.NOCAM_SUBGOAL.fields}
    assert {"subgoal_relative", "subgoal_distance", "subgoal_heading_to_subgoal"} <= keys
    assert not {"goal_relative", "goal_distance", "goal_heading_to_goal"} & keys


def test_assembly_matches_nocam_on_equivalent_raw_values():
    """Feeding the same numbers under the per-variant keys produces the same
    observation vector — the network input is variant-agnostic; only the
    data source behind the keys differs."""
    rng = np.random.default_rng(3)
    shared = {
        "imu_accel": rng.normal(size=3),
        "imu_gyro": rng.normal(size=3),
        "encoder_vels_ticks": rng.normal(size=4),
        "body_velocity_xy": rng.normal(size=2),
        "last_action": rng.normal(size=3),
    }
    goal_vals = {
        "relative": rng.normal(size=2),
        "distance": rng.normal(size=1),
        "heading": rng.normal(size=1),
    }
    raw_nocam = dict(shared)
    raw_nocam.update(
        goal_relative=goal_vals["relative"],
        goal_distance=goal_vals["distance"],
        goal_heading_to_goal=goal_vals["heading"],
    )
    raw_subgoal = dict(shared)
    raw_subgoal.update(
        subgoal_relative=goal_vals["relative"],
        subgoal_distance=goal_vals["distance"],
        subgoal_heading_to_subgoal=goal_vals["heading"],
    )
    np.testing.assert_array_equal(
        assemble_observation(raw_nocam, PolicyVariant.NOCAM),
        assemble_observation(raw_subgoal, PolicyVariant.NOCAM_SUBGOAL),
    )


def test_assembly_rejects_goal_keys_for_subgoal_variant():
    """Wiring a goal-pose pipeline into the subgoal variant fails loudly at
    assembly time instead of producing silent garbage."""
    raw = {
        "imu_accel": np.zeros(3),
        "imu_gyro": np.zeros(3),
        "encoder_vels_ticks": np.zeros(4),
        "goal_relative": np.zeros(2),
        "goal_distance": np.zeros(1),
        "goal_heading_to_goal": np.zeros(1),
        "body_velocity_xy": np.zeros(2),
        "last_action": np.zeros(3),
    }
    with pytest.raises(KeyError):
        assemble_observation(raw, PolicyVariant.NOCAM_SUBGOAL)
