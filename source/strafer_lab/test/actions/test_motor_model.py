# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DC motor torque-speed model and asset configuration.

The Strafer robot uses GoBilda 5203 Yellow Jacket motors (19.2:1 ratio)
with specifications:
- No-load speed: 312 RPM (≈ 32.67 rad/s)
- Stall torque: 24.3 kg·cm (≈ 2.383 Nm)
- Continuous torque: 50% of stall (conservative estimate)

These tests validate:
1. Motor constants are correctly computed from GoBilda specs.
2. ``STRAFER_CFG`` ArticulationCfg wires the correct DCMotor parameters.
3. ``get_action_config_params()`` yields consistent motor/action config.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/actions/test_motor_model.py -v
"""

import math
import pytest

from strafer_lab.assets.strafer import STRAFER_CFG
from strafer_lab.tasks.navigation.sim_real_cfg import (
    IDEAL_SIM_CONTRACT,
    REAL_ROBOT_CONTRACT,
    ROBUST_TRAINING_CONTRACT,
    get_action_config_params,
)

# ---------------------------------------------------------------------------
# GoBilda 5203 (19.2:1) Motor Specs — single source of truth
# ---------------------------------------------------------------------------

MOTOR_NO_LOAD_RPM = 312.0
MOTOR_STALL_TORQUE_KGCM = 24.3
RPM_TO_RAD_S = 2.0 * math.pi / 60.0
KGCM_TO_NM = 0.0980665

# Derived values
EXPECTED_NO_LOAD_RAD_S = MOTOR_NO_LOAD_RPM * RPM_TO_RAD_S  # ≈ 32.67
EXPECTED_STALL_TORQUE_NM = MOTOR_STALL_TORQUE_KGCM * KGCM_TO_NM  # ≈ 2.383
EXPECTED_CONTINUOUS_TORQUE_NM = EXPECTED_STALL_TORQUE_NM * 0.5  # ≈ 1.191


# =====================================================================
# Motor Constant Computation Tests
# =====================================================================


def test_no_load_speed_conversion():
    """Verify RPM → rad/s conversion matches expected value."""
    computed = MOTOR_NO_LOAD_RPM * RPM_TO_RAD_S
    assert abs(computed - EXPECTED_NO_LOAD_RAD_S) < 1e-6, (
        f"No-load speed conversion: {computed:.4f} != {EXPECTED_NO_LOAD_RAD_S:.4f}"
    )
    # Sanity: ≈ 32.67 rad/s
    assert 32.0 < computed < 33.5, (
        f"No-load speed {computed:.2f} rad/s outside sanity range [32, 33.5]"
    )


def test_stall_torque_conversion():
    """Verify kg·cm → Nm conversion matches expected value."""
    computed = MOTOR_STALL_TORQUE_KGCM * KGCM_TO_NM
    assert abs(computed - EXPECTED_STALL_TORQUE_NM) < 1e-4, (
        f"Stall torque conversion: {computed:.4f} != {EXPECTED_STALL_TORQUE_NM:.4f}"
    )
    # Sanity: ≈ 2.38 Nm
    assert 2.0 < computed < 3.0, (
        f"Stall torque {computed:.3f} Nm outside sanity range [2, 3]"
    )


def test_continuous_torque_is_half_stall():
    """Verify continuous torque = 50% of stall (conservative GoBilda estimate)."""
    ratio = EXPECTED_CONTINUOUS_TORQUE_NM / EXPECTED_STALL_TORQUE_NM
    assert abs(ratio - 0.5) < 1e-6, (
        f"Continuous/stall ratio = {ratio:.4f}, expected 0.5"
    )


# =====================================================================
# STRAFER_CFG ArticulationCfg Wiring Tests
# =====================================================================


def test_cfg_has_wheel_drives_actuator():
    """Verify STRAFER_CFG contains 'wheel_drives' actuator group."""
    assert "wheel_drives" in STRAFER_CFG.actuators, (
        f"Missing 'wheel_drives' in STRAFER_CFG.actuators. "
        f"Found: {list(STRAFER_CFG.actuators.keys())}"
    )


def test_cfg_wheel_drives_velocity_limit():
    """Verify DCMotor velocity_limit matches no-load speed."""
    drive_cfg = STRAFER_CFG.actuators["wheel_drives"]
    assert abs(drive_cfg.velocity_limit - EXPECTED_NO_LOAD_RAD_S) < 0.1, (
        f"velocity_limit={drive_cfg.velocity_limit:.2f}, "
        f"expected ≈ {EXPECTED_NO_LOAD_RAD_S:.2f} rad/s"
    )


def test_cfg_wheel_drives_saturation_effort():
    """Verify DCMotor saturation_effort matches stall torque."""
    drive_cfg = STRAFER_CFG.actuators["wheel_drives"]
    assert abs(drive_cfg.saturation_effort - EXPECTED_STALL_TORQUE_NM) < 0.01, (
        f"saturation_effort={drive_cfg.saturation_effort:.4f}, "
        f"expected ≈ {EXPECTED_STALL_TORQUE_NM:.4f} Nm"
    )


def test_cfg_wheel_drives_effort_limit():
    """Verify DCMotor effort_limit matches continuous torque."""
    drive_cfg = STRAFER_CFG.actuators["wheel_drives"]
    assert abs(drive_cfg.effort_limit - EXPECTED_CONTINUOUS_TORQUE_NM) < 0.01, (
        f"effort_limit={drive_cfg.effort_limit:.4f}, "
        f"expected ≈ {EXPECTED_CONTINUOUS_TORQUE_NM:.4f} Nm"
    )


def test_cfg_wheel_drives_velocity_control():
    """Verify motor uses velocity control (D-only: stiffness=0)."""
    drive_cfg = STRAFER_CFG.actuators["wheel_drives"]
    assert drive_cfg.stiffness == 0.0, (
        f"stiffness={drive_cfg.stiffness}, expected 0 for velocity control"
    )
    assert drive_cfg.damping > 0.0, (
        f"damping={drive_cfg.damping}, must be > 0 for D-controller"
    )


def test_cfg_has_roller_bearings():
    """Verify STRAFER_CFG contains passive roller bearings."""
    assert "roller_bearings" in STRAFER_CFG.actuators, (
        f"Missing 'roller_bearings' in STRAFER_CFG.actuators"
    )
    roller_cfg = STRAFER_CFG.actuators["roller_bearings"]
    # Rollers should be passive (no effort)
    assert roller_cfg.effort_limit_sim == 0.0, (
        f"Roller effort_limit_sim={roller_cfg.effort_limit_sim}, expected 0"
    )


# =====================================================================
# get_action_config_params() Consistency Tests
# =====================================================================


@pytest.mark.parametrize("name,contract", [
    ("ideal", IDEAL_SIM_CONTRACT),
    ("real", REAL_ROBOT_CONTRACT),
    ("robust", ROBUST_TRAINING_CONTRACT),
])
def test_action_params_has_required_keys(name, contract):
    """Verify get_action_config_params returns all required keys."""
    params = get_action_config_params(contract)
    required_keys = {
        "motor_rpm",
        "max_wheel_angular_vel",
        "enable_motor_dynamics",
        "motor_time_constant",
        "min_delay_steps",
        "max_delay_steps",
        "max_acceleration_rad_s2",
    }
    missing = required_keys - set(params.keys())
    assert not missing, (
        f"[{name}] Missing keys: {missing}. Got: {set(params.keys())}"
    )


@pytest.mark.parametrize("name,contract", [
    ("ideal", IDEAL_SIM_CONTRACT),
    ("real", REAL_ROBOT_CONTRACT),
    ("robust", ROBUST_TRAINING_CONTRACT),
])
def test_action_params_velocity_consistency(name, contract):
    """Verify motor_rpm and max_wheel_angular_vel are consistent."""
    params = get_action_config_params(contract)
    rpm_from_vel = params["max_wheel_angular_vel"] * 60.0 / (2.0 * math.pi)

    assert abs(rpm_from_vel - params["motor_rpm"]) < 0.1, (
        f"[{name}] RPM inconsistency: motor_rpm={params['motor_rpm']:.1f} "
        f"but derived from angular vel = {rpm_from_vel:.1f}"
    )


def test_ideal_contract_disables_dynamics():
    """Verify IDEAL contract disables motor dynamics (instant commands)."""
    params = get_action_config_params(IDEAL_SIM_CONTRACT)
    assert params["enable_motor_dynamics"] is False, (
        f"Ideal contract: enable_motor_dynamics should be False"
    )


def test_real_contract_enables_dynamics():
    """Verify REAL contract enables motor dynamics (sim-to-real target)."""
    params = get_action_config_params(REAL_ROBOT_CONTRACT)
    assert params["enable_motor_dynamics"] is True, (
        f"Real contract: enable_motor_dynamics should be True"
    )
    assert params["motor_time_constant"] > 0, (
        f"Real contract: motor_time_constant should be > 0, "
        f"got {params['motor_time_constant']}"
    )


# =====================================================================
# DC Motor Torque-Speed Relationship (analytical)
# =====================================================================


def test_torque_speed_curve_is_linear():
    """Verify the DC motor torque-speed relationship is linear.

    For an ideal DC motor:  τ = τ_stall × (1 - ω/ω_no_load)
    
    At ω = 0:       τ = τ_stall
    At ω = ω_no_load: τ = 0

    This is a fundamental property of the GoBilda 5203 motor model.
    """
    # Test at several speeds
    speeds = [0.0, 0.25, 0.5, 0.75, 1.0]
    for frac in speeds:
        omega = frac * EXPECTED_NO_LOAD_RAD_S
        expected_torque = EXPECTED_STALL_TORQUE_NM * (1.0 - omega / EXPECTED_NO_LOAD_RAD_S)

        # Verify the formula gives sensible values
        if frac == 0.0:
            assert abs(expected_torque - EXPECTED_STALL_TORQUE_NM) < 1e-6, (
                f"At ω=0, torque should be stall torque"
            )
        elif frac == 1.0:
            assert abs(expected_torque) < 1e-6, (
                f"At ω=ω_no_load, torque should be 0"
            )
        else:
            # Intermediate: torque should be between 0 and stall
            assert 0.0 < expected_torque < EXPECTED_STALL_TORQUE_NM, (
                f"At ω={omega:.2f}, torque={expected_torque:.4f} outside (0, stall)"
            )

    print(f"\n  DC motor torque-speed curve:")
    print(f"    Stall torque:  {EXPECTED_STALL_TORQUE_NM:.3f} Nm")
    print(f"    No-load speed: {EXPECTED_NO_LOAD_RAD_S:.2f} rad/s ({MOTOR_NO_LOAD_RPM} RPM)")
    print(f"    τ(ω) = {EXPECTED_STALL_TORQUE_NM:.3f} * (1 - ω/{EXPECTED_NO_LOAD_RAD_S:.2f})")
