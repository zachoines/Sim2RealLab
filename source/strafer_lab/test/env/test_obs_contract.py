# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Verify observation contract between env config and policy_interface.

The env config (strafer_env_cfg.py) defines observation terms in a specific
order with specific scales.  The policy_interface (policy_interface.py) defines
the same fields for the ROS2 inference node.  These MUST remain in sync or
the deployed policy will receive scrambled/mis-scaled observations.

This test catches:
- Field ordering mismatches (sim assembles [A, B, C] but real expects [A, C, B])
- Scale factor drift (sim scales IMU by 1/X but real scales by 1/Y)
- Dimension mismatches (sim obs is 19 dims but PolicyVariant says 15)

No simulation needed — purely inspects config dataclass attributes.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/env/test_obs_contract.py -v
"""

import math
import pytest

from strafer_shared.policy_interface import PolicyVariant, _NOCAM_FIELDS, _DEPTH_FIELDS
from strafer_shared.constants import (
    IMU_ACCEL_SCALE,
    IMU_GYRO_SCALE,
    ENCODER_VEL_SCALE,
    GOAL_DIST_SCALE,
    HEADING_SCALE,
    BODY_VEL_SCALE,
    DEPTH_SCALE,
)

# Import env configs (triggers gym registration via __init__.py)
from strafer_lab.tasks.navigation.strafer_env_cfg import (
    ObsCfg_NoCam_Ideal,
    ObsCfg_Depth_Ideal,
)


# =====================================================================
# Canonical mapping: policy_interface field key → env config attribute name
#
# This explicit mapping is the "contract" between sim and real.
# If either side renames a field, the test fails loudly.
# =====================================================================

_FIELD_TO_ATTR = {
    "imu_accel": "imu_linear_acceleration",
    "imu_gyro": "imu_angular_velocity",
    "encoder_vels_ticks": "wheel_encoder_velocities",
    "goal_relative": "goal_position",
    "goal_distance": "goal_distance",
    "goal_heading_relative": "goal_heading_relative",
    "body_velocity_xy": "body_velocity_xy",
    "last_action": "last_action",
    "depth_image": "depth_image",
}

# Known output dimensions per observation function
# (can't call functions without sim, so we hardcode the contract here)
_FIELD_DIMS = {
    "imu_accel": 3,
    "imu_gyro": 3,
    "encoder_vels_ticks": 4,
    "goal_relative": 2,
    "goal_distance": 1,
    "goal_heading_relative": 1,
    "body_velocity_xy": 2,
    "last_action": 3,
    "depth_image": 4800,  # 60 * 80
}

# Expected scales per field (1.0 means no explicit scale on the ObsTerm)
_FIELD_SCALES = {
    "imu_accel": IMU_ACCEL_SCALE,
    "imu_gyro": IMU_GYRO_SCALE,
    "encoder_vels_ticks": ENCODER_VEL_SCALE,
    "goal_relative": 1.0,
    "goal_distance": GOAL_DIST_SCALE,
    "goal_heading_relative": HEADING_SCALE,
    "body_velocity_xy": BODY_VEL_SCALE,
    "last_action": 1.0,
    "depth_image": DEPTH_SCALE,
}


_SKIP_ATTRS = {
    "enable_corruption",
    "concatenate_terms",
    "concatenate_dim",
    "history_length",
}


def _get_policy_obs_terms(obs_cfg_class):
    """Extract ordered (attr_name, ObsTerm) pairs from a PolicyCfg instance.

    Uses the same iteration method as Isaac Lab's ObservationManager._prepare_terms():
    ``group_cfg.__dict__.items()`` which preserves Python 3.7+ insertion order.
    """
    policy_cfg = obs_cfg_class.PolicyCfg()
    terms = []
    for attr_name, val in policy_cfg.__dict__.items():
        if attr_name.startswith("_") or attr_name in _SKIP_ATTRS:
            continue
        if hasattr(val, "func"):
            terms.append((attr_name, val))
    return terms


# =====================================================================
# Tests
# =====================================================================


class TestNoCamObsContract:
    """Verify NoCam observation contract between env config and policy_interface."""

    def test_field_count_matches(self):
        """PolicyVariant.NOCAM has the same number of fields as env PolicyCfg."""
        env_terms = _get_policy_obs_terms(ObsCfg_NoCam_Ideal)
        assert len(env_terms) == len(_NOCAM_FIELDS), (
            f"NoCam field count mismatch: env has {len(env_terms)} terms, "
            f"policy_interface has {len(_NOCAM_FIELDS)} fields"
        )

    def test_field_ordering(self):
        """Fields appear in the same order in env config and policy_interface."""
        env_terms = _get_policy_obs_terms(ObsCfg_NoCam_Ideal)
        env_attr_names = [name for name, _ in env_terms]

        for i, field in enumerate(_NOCAM_FIELDS):
            expected_attr = _FIELD_TO_ATTR[field.key]
            actual_attr = env_attr_names[i]
            assert actual_attr == expected_attr, (
                f"NoCam field order mismatch at index {i}: "
                f"policy_interface expects '{expected_attr}' (key='{field.key}'), "
                f"but env config has '{actual_attr}'"
            )

    def test_total_obs_dim(self):
        """PolicyVariant.NOCAM.obs_dim matches sum of known field dims."""
        expected_dim = sum(_FIELD_DIMS[f.key] for f in _NOCAM_FIELDS)
        assert PolicyVariant.NOCAM.obs_dim == expected_dim, (
            f"NoCam obs_dim mismatch: PolicyVariant says {PolicyVariant.NOCAM.obs_dim}, "
            f"expected {expected_dim}"
        )

    def test_field_dims_match(self):
        """Each field's declared dims in policy_interface matches the known output size."""
        for field in _NOCAM_FIELDS:
            expected = _FIELD_DIMS[field.key]
            assert field.dims == expected, (
                f"NoCam field '{field.key}': policy_interface says {field.dims} dims, "
                f"expected {expected}"
            )

    def test_scale_factors_match(self):
        """Scale factors in env config ObsTerms match policy_interface field scales."""
        env_terms = _get_policy_obs_terms(ObsCfg_NoCam_Ideal)
        env_scales = {name: getattr(term, "scale", 1.0) for name, term in env_terms}

        for field in _NOCAM_FIELDS:
            attr_name = _FIELD_TO_ATTR[field.key]
            env_scale = env_scales.get(attr_name, 1.0)
            # env_scale may be None if not set explicitly
            if env_scale is None:
                env_scale = 1.0
            expected_scale = _FIELD_SCALES[field.key]
            assert math.isclose(env_scale, expected_scale, rel_tol=1e-6), (
                f"Scale mismatch for '{field.key}' (env attr '{attr_name}'): "
                f"env has {env_scale}, policy_interface expects {expected_scale}"
            )

    def test_policy_interface_scales_match_env(self):
        """policy_interface ObsField.scale matches the env config scale for each field."""
        env_terms = _get_policy_obs_terms(ObsCfg_NoCam_Ideal)
        env_scales = {name: getattr(term, "scale", 1.0) for name, term in env_terms}

        for field in _NOCAM_FIELDS:
            attr_name = _FIELD_TO_ATTR[field.key]
            env_scale = env_scales.get(attr_name, 1.0)
            if env_scale is None:
                env_scale = 1.0
            assert math.isclose(field.scale, env_scale, rel_tol=1e-6), (
                f"policy_interface scale for '{field.key}' is {field.scale}, "
                f"but env config '{attr_name}' uses {env_scale}"
            )


class TestDepthObsContract:
    """Verify Depth observation contract between env config and policy_interface."""

    def test_field_count_matches(self):
        env_terms = _get_policy_obs_terms(ObsCfg_Depth_Ideal)
        assert len(env_terms) == len(_DEPTH_FIELDS), (
            f"Depth field count mismatch: env has {len(env_terms)} terms, "
            f"policy_interface has {len(_DEPTH_FIELDS)} fields"
        )

    def test_field_ordering(self):
        env_terms = _get_policy_obs_terms(ObsCfg_Depth_Ideal)
        env_attr_names = [name for name, _ in env_terms]

        for i, field in enumerate(_DEPTH_FIELDS):
            expected_attr = _FIELD_TO_ATTR[field.key]
            actual_attr = env_attr_names[i]
            assert actual_attr == expected_attr, (
                f"Depth field order mismatch at index {i}: "
                f"policy_interface expects '{expected_attr}' (key='{field.key}'), "
                f"but env config has '{actual_attr}'"
            )

    def test_total_obs_dim(self):
        expected_dim = sum(_FIELD_DIMS[f.key] for f in _DEPTH_FIELDS)
        assert PolicyVariant.DEPTH.obs_dim == expected_dim, (
            f"Depth obs_dim mismatch: PolicyVariant says {PolicyVariant.DEPTH.obs_dim}, "
            f"expected {expected_dim}"
        )

    def test_scale_factors_match(self):
        env_terms = _get_policy_obs_terms(ObsCfg_Depth_Ideal)
        env_scales = {name: getattr(term, "scale", 1.0) for name, term in env_terms}

        for field in _DEPTH_FIELDS:
            attr_name = _FIELD_TO_ATTR[field.key]
            env_scale = env_scales.get(attr_name, 1.0)
            if env_scale is None:
                env_scale = 1.0
            expected_scale = _FIELD_SCALES[field.key]
            assert math.isclose(env_scale, expected_scale, rel_tol=1e-6), (
                f"Scale mismatch for '{field.key}': "
                f"env has {env_scale}, expected {expected_scale}"
            )


class TestCrossVariantConsistency:
    """Verify all realism levels have consistent observation structure."""

    def test_nocam_19_dims(self):
        assert PolicyVariant.NOCAM.obs_dim == 19

    def test_depth_4819_dims(self):
        assert PolicyVariant.DEPTH.obs_dim == 4819

    def test_depth_extends_nocam(self):
        """Depth fields are NoCam fields + depth_image."""
        assert _DEPTH_FIELDS[:-1] == _NOCAM_FIELDS
        assert _DEPTH_FIELDS[-1].key == "depth_image"

    def test_all_nocam_configs_same_field_count(self):
        """All NoCam observation configs (Ideal, Realistic, Robust) have same field count."""
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            ObsCfg_NoCam_Realistic,
            ObsCfg_NoCam_Robust,
        )
        ideal_count = len(_get_policy_obs_terms(ObsCfg_NoCam_Ideal))
        realistic_count = len(_get_policy_obs_terms(ObsCfg_NoCam_Realistic))
        robust_count = len(_get_policy_obs_terms(ObsCfg_NoCam_Robust))

        assert ideal_count == realistic_count == robust_count, (
            f"NoCam field count varies across realism levels: "
            f"Ideal={ideal_count}, Realistic={realistic_count}, Robust={robust_count}"
        )

    def test_all_depth_configs_same_field_count(self):
        """All Depth observation configs have same field count."""
        from strafer_lab.tasks.navigation.strafer_env_cfg import (
            ObsCfg_Depth_Realistic,
            ObsCfg_Depth_Robust,
        )
        ideal_count = len(_get_policy_obs_terms(ObsCfg_Depth_Ideal))
        realistic_count = len(_get_policy_obs_terms(ObsCfg_Depth_Realistic))
        robust_count = len(_get_policy_obs_terms(ObsCfg_Depth_Robust))

        assert ideal_count == realistic_count == robust_count, (
            f"Depth field count varies across realism levels: "
            f"Ideal={ideal_count}, Realistic={realistic_count}, Robust={robust_count}"
        )
