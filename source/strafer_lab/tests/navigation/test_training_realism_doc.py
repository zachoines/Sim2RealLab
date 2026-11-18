# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The realism tiers' values, pinned against `docs/context/training-realism.md`.

Reads the composed cfgs of one class per tier (`StraferNavCfg_Depth_Ideal`, `StraferNavCfg_RLDepthSubgoal_Real`,
`StraferNavCfg_RLDepthSubgoal_Robust`, and `StraferNavCfg_RLNoCamSubgoal_Real` for the camera-free stack), the tier
contracts in `sim_real_cfg.py` for the IMU noise densities, and the built action term (on a stand-in env at the
composed cfg's physics and control steps) for the trained motor lag, acceleration limit and twist caps. Each case
checks the code's value and that the module's row for that perturbation states it, so a value that changes on either
side fails until the other follows.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE = Path(__file__).resolve().parents[4] / "docs" / "context" / "training-realism.md"
CLASSES = {
    "ideal": "StraferNavCfg_Depth_Ideal",
    "realistic": "StraferNavCfg_RLDepthSubgoal_Real",
    "robust": "StraferNavCfg_RLDepthSubgoal_Robust",
    "camera-free": "StraferNavCfg_RLNoCamSubgoal_Real",
}
CONTRACTS = {"realistic": "REAL_ROBOT_CONTRACT", "robust": "ROBUST_TRAINING_CONTRACT"}


@functools.lru_cache(maxsize=None)
def _cfg(tier):
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    return getattr(composed, CLASSES[tier])()


def _contract(tier):
    from strafer_lab.tasks.navigation import sim_real_cfg

    return getattr(sim_real_cfg, CONTRACTS[tier])


class _Asset:
    def find_joints(self, names, preserve_order=False):
        return [0, 1, 2, 3], ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]


def _term(tier, **overrides):
    """The tier's action term on a stand-in env stepping at the composed cfg's physics and control periods."""
    cfg = _cfg(tier)
    term_cfg = cfg.actions.wheel_velocities.copy()
    for k, v in overrides.items():
        setattr(term_cfg, k, v)
    env = SimpleNamespace(
        scene={term_cfg.asset_name: _Asset()}, device="cpu", num_envs=4,
        physics_dt=cfg.sim.dt, step_dt=cfg.sim.dt * cfg.decimation,
    )
    return term_cfg.class_type(term_cfg, env), env


def _rise_63_s(tier):
    """Seconds for the motor lag to reach 63 % of a step, from the term's own per-control-step coefficient."""
    term, env = _term(tier)
    steps = math.log(math.exp(-1.0)) / math.log(1.0 - term._motor_alpha)
    return steps * env.step_dt


def _trained_accel(tier, accel):
    """rad/s² the acceleration limit allows per control step, for a band pinned at one value."""
    term, env = _term(tier, max_acceleration_range=(accel, accel), max_acceleration_rad_s2=accel)
    return float(term._max_delta_per_step.max()) / env.step_dt


def W(tier, *names):
    """The tier's action-term fields."""
    wv = _cfg(tier).actions.wheel_velocities
    return tuple(getattr(wv, n) for n in names) if len(names) > 1 else getattr(wv, names[0])


def NZ(tier, term, *names):
    """Fields of one observation term's noise model (None when the term has no noise)."""
    noise = getattr(_cfg(tier).observations.policy, term).noise
    if not names:
        return noise
    return tuple(getattr(noise, n) for n in names) if len(names) > 1 else getattr(noise, names[0])


def EV(tier, name, *keys):
    """Params of one event term (None when the tier has no such term)."""
    term = getattr(_cfg(tier).events, name, None)
    if term is None:
        return None
    return tuple(term.params[k] for k in keys) if len(keys) > 1 else term.params[keys[0]]


def IMU(tier, *names):
    """The tier contract's IMU fields."""
    return tuple(getattr(_contract(tier).sensors.imu, n) for n in names)


def _accel_band(tier):
    return tuple(round(_trained_accel(tier, a)) for a in W(tier, "max_acceleration_range"))


def _caps(tier):
    term = _term(tier)[0]
    return round(term._max_linear_vel, 3), round(term._max_angular_vel, 3)


def _fx(tier):
    spawn = _cfg(tier).scene.d555_camera.spawn
    return round(640 * spawn.focal_length / spawn.horizontal_aperture, 3)


ENC, DEPTH = "wheel_encoder_velocities", "depth_image"
ACC, GYR = "imu_linear_acceleration", "imu_angular_velocity"
DRIFT = ("gain_range", "position_rms_m", "heading_sigma_deg", "tau_s")

# (module row, tier, the code's value, its expected value, the text the row states)
ROWS = [
    ("Command delay", "ideal", lambda: W("ideal", "min_delay_steps", "max_delay_steps"), (0, 0), "| off |"),
    ("Command delay", "realistic", lambda: W("realistic", "min_delay_steps", "max_delay_steps"), (1, 3),
     "1–3 control steps"),
    ("Command delay", "robust", lambda: W("robust", "min_delay_steps", "max_delay_steps"), (1, 5),
     "1–5 control steps"),
    ("Command hold", "realistic", lambda: W("realistic", "hold_fraction_range", "hold_run_range"),
     ((0.0, 0.05), (1.0, 1.2)), "U(0, 0.05) of ticks per env, runs of 1–1.2"),
    ("Command hold", "robust", lambda: W("robust", "hold_fraction_range", "hold_run_range"),
     ((0.0, 0.05), (1.0, 1.2)), "| same |"),
    ("Acceleration limit", "ideal", lambda: W("ideal", "max_acceleration_rad_s2"), math.inf, "| off |"),
    ("Acceleration limit", "realistic", lambda: W("realistic", "max_acceleration_range"), (700.0, 1100.0),
     "(field: 700–1100)"),
    ("Acceleration limit", "realistic", lambda: _accel_band("realistic"), (175, 275), "175–275 rad/s²"),
    ("Acceleration limit", "robust", lambda: W("robust", "max_acceleration_range"), (500.0, 1200.0),
     "(field: 500–1200)"),
    ("Acceleration limit", "robust", lambda: _accel_band("robust"), (125, 300), "125–300 rad/s²"),
    ("Acceleration limit", "realistic",
     lambda: round(_trained_accel("realistic", W("realistic", "max_acceleration_rad_s2"))),
     225, "nominal 225 (field: 900)"),
    ("Motor lag", "ideal", lambda: W("ideal", "enable_motor_dynamics"), False, "| off |"),
    ("Motor lag", "realistic", lambda: (W("realistic", "motor_time_constant"), round(_rise_63_s("realistic"), 3)),
     (0.05, 0.216), "63 % rise in 0.216 s (field τ 0.05 s)"),
    ("Motor lag", "robust", lambda: (W("robust", "motor_time_constant"), round(_rise_63_s("robust"), 3)),
     (0.06, 0.256), "0.256 s (field τ 0.06 s)"),
    ("Command envelope", "realistic", lambda: W("realistic", "speed_scale_range"), None, "| off | off |"),
    ("Command envelope", "robust", lambda: W("robust", "speed_scale_range"), (0.255, 1.0), "s ~ U(0.255, 1.0)"),
    ("Twist clamp", "realistic", lambda: _caps("realistic"), (1.568, 4.186),
     "\\|vx\\| + \\|vy\\| ≤ 1.568 m/s, \\|ω\\| ≤ 4.186 rad/s"),
    ("Wheel speed and torque", "realistic",
     lambda: (W("realistic", "max_wheel_angular_vel"),
              round(_cfg("realistic").scene.robot.actuators["wheel_drives"].effort_limit_sim, 3)),
     (32.67, 2.383), "32.67 rad/s; 2.383 N·m"),
    ("Motor strength", "ideal", lambda: EV("ideal", "randomize_motor_strength", "strength_range"), None, "| off |"),
    ("Motor strength", "realistic", lambda: EV("realistic", "randomize_motor_strength", "strength_range"),
     (0.92, 1.08), "× U(0.92, 1.08)"),
    ("Motor strength", "robust", lambda: EV("robust", "randomize_motor_strength", "strength_range"), (0.8, 1.2),
     "× U(0.8, 1.2)"),
    ("Control period", "robust", lambda: round(_cfg("robust").sim.dt * _cfg("robust").decimation, 9),
     round(1 / 30, 9), "| 1/30 s |"),
    ("IMU noise", "ideal", lambda: NZ("ideal", ACC), None, "| off |"),
    ("IMU noise", "realistic", lambda: IMU("realistic", "accel_noise_density", "gyro_noise_density"),
     (0.0098, 0.00024), "white 0.0098 m/s²/√Hz and 0.00024 rad/s/√Hz"),
    ("IMU noise", "realistic",
     lambda: (NZ("realistic", ACC, "accel_bias_range"), NZ("realistic", GYR, "gyro_bias_range")),
     ((-0.04, 0.04), (-0.0017, 0.0017)), "±0.04 m/s², ±0.0017 rad/s"),
    ("IMU noise", "robust", lambda: IMU("robust", "accel_noise_density", "gyro_noise_density"), (0.015, 0.00036),
     "0.015, 0.00036"),
    ("IMU noise", "robust", lambda: (NZ("robust", ACC, "accel_bias_range"), NZ("robust", GYR, "gyro_bias_range")),
     ((-0.06, 0.06), (-0.0025, 0.0025)), "±0.06, ±0.0025"),
    ("IMU and encoder latency", "realistic",
     lambda: (NZ("realistic", ACC, "latency_steps"), NZ("realistic", ENC, "latency_steps")),
     (0, 0), "| none |"),
    ("IMU and encoder latency", "robust",
     lambda: (NZ("robust", GYR, "latency_steps"), NZ("robust", ENC, "latency_steps")),
     (1, 1), "1 control step"),
    ("Wheel encoder noise", "realistic",
     lambda: NZ("realistic", ENC, "velocity_noise_std", "max_velocity", "missed_tick_prob", "extra_tick_prob"),
     (0.02, 3000.0, 0.001, 0.0005), "σ 0.02 of 3000 ticks/s; a missed tick 0.001, an extra 0.0005"),
    ("Wheel encoder noise", "robust", lambda: NZ("robust", ENC, "velocity_noise_std", "missed_tick_prob"),
     (0.05, 0.005), "σ 0.05; missed 0.005"),
    ("Sensor dropouts", "realistic",
     lambda: tuple(NZ("realistic", t, "failure_probability") for t in (ACC, ENC, DEPTH)), (0.0, 0.0, 0.0), "| none |"),
    ("Sensor dropouts", "robust",
     lambda: (NZ("robust", ACC, "failure_probability"), NZ("robust", ACC, "stuck_probability"),
              NZ("robust", ENC, "failure_probability"), NZ("robust", DEPTH, "failure_probability")),
     (1e-4, 1e-4, 1e-4, 1e-3),
     "the IMU reads zero or repeats 1e-4; the encoders read zero 1e-4; the depth frame reads far 1e-3"),
    ("Depth disparity noise", "realistic",
     lambda: NZ("realistic", DEPTH, "disparity_noise_px", "disparity_noise_px_range"),
     (0.08, None), "σ_d 0.08 px"),
    ("Depth disparity noise", "robust", lambda: NZ("robust", DEPTH, "disparity_noise_px_range"), (0.002, 0.16),
     "log-uniform on [0.002, 0.16] px"),
    ("Depth holes and lost frames", "realistic", lambda: NZ("realistic", DEPTH, "hole_probability", "frame_drop_prob"),
     (0.01, 0.001), "holes 0.01 of cells; a frame lost 0.001"),
    ("Depth holes and lost frames", "robust", lambda: NZ("robust", DEPTH, "hole_probability", "frame_drop_prob"),
     (0.03, 0.01), "| 0.03; 0.01 |"),
    ("Depth stream hold", "robust", lambda: NZ("robust", DEPTH, "hold_fraction_range", "hold_run_range"),
     ((0.0, 0.35), (1.0, 1.6)), "U(0, 0.35) of ticks per env, runs of 1–1.6"),
    ("Depth latency", "realistic", lambda: NZ("realistic", DEPTH, "latency_steps_range"), (0, 2), "0–2 control steps"),
    ("Depth latency", "robust", lambda: NZ("robust", DEPTH, "latency_steps_range"), (1, 3), "| 1–3 |"),
    ("Camera lens", "robust", lambda: _fx("robust"), 321.522, "fx = fy = 321.522 px at 640×360"),
    ("Re-render after reset", "realistic",
     lambda: (_cfg("realistic").num_rerenders_on_reset, _cfg("camera-free").num_rerenders_on_reset), (1, 0),
     "1 on every stack that reads a camera, 0 on the camera-free stack"),
    ("Mount misalignment", "realistic", lambda: EV("realistic", "randomize_d555_mount", "max_angle_deg"), 1.0,
     "±1° per axis"),
    ("Mount misalignment", "robust", lambda: EV("robust", "randomize_d555_mount", "max_angle_deg"), 3.0, "| ±3° |"),
    ("Subgoal-frame drift", "robust", lambda: EV("robust", "randomize_subgoal_drift", *DRIFT),
     ((0.0, 0.5), 0.166, 6.7, 2.0), "gain U(0, 0.5) per env × 0.166 m RMS and 6.7° heading, correlation time 2 s"),
    ("Path and lookahead", "realistic",
     lambda: (_cfg("realistic").commands.goal_command.waypoint_noise_std_m,
              _cfg("realistic").commands.goal_command.lookahead_randomization_m),
     (0.025, (0.9, 1.1)), "waypoint noise 0.025 m; lookahead U(0.9, 1.1) m"),
    ("Path and lookahead", "robust", lambda: _cfg("robust").commands.goal_command.lookahead_randomization_m,
     (0.7, 1.3), "U(0.7, 1.3) m"),
    ("Friction", "realistic", lambda: EV("realistic", "randomize_friction", "friction_range"), (0.6, 1.2),
     "static U(0.6, 1.2)"),
    ("Friction", "robust", lambda: EV("robust", "randomize_friction", "friction_range"), (0.3, 1.5), "| U(0.3, 1.5) |"),
    ("Mass", "realistic", lambda: EV("realistic", "randomize_mass", "mass_range"), (0.95, 1.05), "× U(0.95, 1.05)"),
    ("Mass", "robust", lambda: EV("robust", "randomize_mass", "mass_range"), (0.85, 1.15), "× U(0.85, 1.15)"),
    ("Room", "realistic",
     lambda: (EV("realistic", "randomize_difficulty", "min_level", "max_level"),
              EV("realistic", "generate_room", "p_ceil")),
     ((4, 7), 0.7), "difficulty U{4…7}, a ceiling on 70 % of episodes"),
]


def _module_row(label):
    """The module's table row for a perturbation (its first cell names it, linked or not)."""
    rows = [line for line in MODULE.read_text(encoding="utf-8").splitlines()
            if line.startswith("| ") and (line.startswith(f"| [{label}]") or line.startswith(f"| {label} |"))]
    assert len(rows) == 1, f"{label!r}: {len(rows)} rows in {MODULE.name}"
    return rows[0]


@pytest.mark.parametrize("label, tier, value, expected, stated", ROWS,
                         ids=[f"{r[0]}-{r[1]}-{i}" for i, r in enumerate(ROWS)])
def test_the_module_states_the_tiers_values(label, tier, value, expected, stated):
    got = value()
    if isinstance(expected, float) and not math.isinf(expected):
        assert got == pytest.approx(expected), (label, tier, got)
    else:
        assert got == expected, (label, tier, got)
    assert stated in _module_row(label), f"{label} ({tier}): the module's row does not state {stated!r}"
