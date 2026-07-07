"""Contract tests for ``PolicyVariant.DEPTH_SUBGOAL``.

Pins the seam the variant exists for: identical network input layout to DEPTH
(same 4819 dims, same scales, same field order, same depth field) with renamed
goal-referent keys, and — critically — that the member is a *distinct* enum
member rather than a silent alias of DEPTH or NOCAM_SUBGOAL (Python Enums
collapse members with equal values into aliases, which would make the variants
indistinguishable at load/dispatch time).

Mirrors ``test_policy_variant_subgoal.py`` (the NOCAM/NOCAM_SUBGOAL pair) for
the DEPTH/DEPTH_SUBGOAL pair.
"""

from __future__ import annotations

import numpy as np
import pytest

from strafer_shared.policy_interface import PolicyVariant, assemble_observation


def test_depth_subgoal_is_a_distinct_member_not_an_alias():
    assert PolicyVariant.DEPTH_SUBGOAL is not PolicyVariant.DEPTH
    assert PolicyVariant.DEPTH_SUBGOAL is not PolicyVariant.NOCAM_SUBGOAL
    assert PolicyVariant["DEPTH_SUBGOAL"].name == "DEPTH_SUBGOAL"
    # The set is closed at four: NOCAM, DEPTH, NOCAM_SUBGOAL, DEPTH_SUBGOAL.
    assert len({v.name for v in PolicyVariant}) == 4


def test_depth_subgoal_mirrors_depth_shapes_and_scales():
    """Same obs_dim (4819) and per-field shape/scale as DEPTH — the acceptance
    criterion the brief calls out explicitly."""
    sub = PolicyVariant.DEPTH_SUBGOAL.fields
    base = PolicyVariant.DEPTH.fields
    assert PolicyVariant.DEPTH_SUBGOAL.obs_dim == PolicyVariant.DEPTH.obs_dim == 4819
    assert len(sub) == len(base)
    for f_sub, f_base in zip(sub, base):
        assert f_sub.dims == f_base.dims
        assert f_sub.scale == f_base.scale


def test_depth_subgoal_reuses_the_depth_field_verbatim():
    """The 4800-dim depth field is DEPTH's, not a redefinition — the brief is
    explicit: reuse the constants, don't restate shapes/scales."""
    depth_field = PolicyVariant.DEPTH_SUBGOAL.fields[-1]
    assert depth_field.key == "depth_image"
    assert depth_field.dims == 4800
    assert depth_field is PolicyVariant.DEPTH.fields[-1]


def test_goal_referent_keys_are_renamed():
    keys = {f.key for f in PolicyVariant.DEPTH_SUBGOAL.fields}
    assert {"subgoal_relative", "subgoal_distance", "subgoal_heading_to_subgoal"} <= keys
    assert not {"goal_relative", "goal_distance", "goal_heading_to_goal"} & keys
    # The proprioceptive prefix matches NOCAM_SUBGOAL exactly; DEPTH_SUBGOAL just
    # appends the depth field.
    assert (
        PolicyVariant.DEPTH_SUBGOAL.fields[:-1]
        == PolicyVariant.NOCAM_SUBGOAL.fields
    )


def test_assembly_matches_depth_on_equivalent_raw_values():
    """Feeding the same numbers under the per-variant keys produces the same
    observation vector — the network input is variant-agnostic; only the data
    source behind the goal-shaped keys differs."""
    rng = np.random.default_rng(7)
    shared = {
        "imu_accel": rng.normal(size=3),
        "imu_gyro": rng.normal(size=3),
        "encoder_vels_ticks": rng.normal(size=4),
        "body_velocity_xy": rng.normal(size=2),
        "last_action": rng.normal(size=3),
        "depth_image": rng.random(size=4800),
    }
    goal_vals = {
        "relative": rng.normal(size=2),
        "distance": rng.normal(size=1),
        "heading": rng.normal(size=1),
    }
    raw_depth = dict(shared)
    raw_depth.update(
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
        assemble_observation(raw_depth, PolicyVariant.DEPTH),
        assemble_observation(raw_subgoal, PolicyVariant.DEPTH_SUBGOAL),
    )


def test_assembly_rejects_goal_keys_for_depth_subgoal_variant():
    """Wiring a goal-pose pipeline into the depth-subgoal variant fails loudly
    at assembly time instead of producing silent garbage."""
    raw = {
        "imu_accel": np.zeros(3),
        "imu_gyro": np.zeros(3),
        "encoder_vels_ticks": np.zeros(4),
        "goal_relative": np.zeros(2),
        "goal_distance": np.zeros(1),
        "goal_heading_to_goal": np.zeros(1),
        "body_velocity_xy": np.zeros(2),
        "last_action": np.zeros(3),
        "depth_image": np.zeros(4800),
    }
    with pytest.raises(KeyError):
        assemble_observation(raw, PolicyVariant.DEPTH_SUBGOAL)
