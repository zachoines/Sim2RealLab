# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""`docs/context/training-realism.md` against the code.

The module's three tables are parsed into cells (a cell reading "same" stands for the cell on its left plus what
follows the word). For each row, every number a cell holds is computed here from the code — the composed cfgs of one
class per tier, the built action term on a stand-in env, the noise cfgs, the event params, the robot's USD — and the
cell's numbers must equal them in order: exactly where the code's value is a whole number, and otherwise to the
precision the cell prints, with at least the significant digits the value has (up to three). An inactive perturbation
is the cell "off" (or "none"). No expected cell number is written here as a literal.

Completeness: every event term, curriculum term, action-cfg field, command-cfg field and noise-cfg field of every
composed class must map to a row, a not-applied line, or a structural field; every policy group's corruption switch
must be on exactly off the ideal tier; every noise cfg must name its documented model class (and the IMU's its sensor);
no policy term may carry a clip or a modifier, and each term's scale must be one value across the compositions; the
action term with every documented stage off must be the clamp and the kinematics alone; and the contract fields
nothing reads must be the module's "declared, not applied" list.

Not pinned (a change here passes this test): the arithmetic inside the noise models (the IMU bias clamp, the encoder's
tick handling, the depth fill), inside the observation functions (the depth term's reduction and range handling) and
inside the action term's `apply_actions`; the room generator's built-in draws (the 4 m span and 0.8 m door floors); the
Infinigen capture classes, which need the scene corpus to build; and the words of a cell beyond its numbers, its
off/none state and its basis kind.

No Kit / GPU. Run with the pure-Python lab tests.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
MODULE = REPO / "docs" / "context" / "training-realism.md"
PACKAGE = REPO / "source" / "strafer_lab" / "strafer_lab"
NAV = PACKAGE / "tasks" / "navigation"
TIERS = ("ideal", "real", "robust")
COLUMNS = ("ideal", "real", "robust", "models", "basis")
CAMERA = {"ideal": "StraferNavCfg_Depth_Ideal", "real": "StraferNavCfg_RLDepthSubgoal_Real",
          "robust": "StraferNavCfg_RLDepthSubgoal_Robust"}
CAMERA_FREE = {"ideal": "StraferNavCfg_NoCam_Ideal", "real": "StraferNavCfg_RLNoCamSubgoal_Real",
               "robust": "StraferNavCfg_RLNoCamSubgoal_Robust"}
CONTRACT = {"ideal": "IDEAL_SIM_CONTRACT", "real": "REAL_ROBOT_CONTRACT", "robust": "ROBUST_TRAINING_CONTRACT"}
LEGEND = ("measured", "datasheet", "derived", "design", "unknown")
ACC, GYR, ENC, DEPTH = "imu_linear_acceleration", "imu_angular_velocity", "wheel_encoder_velocities", "depth_image"


# --- the module ------------------------------------------------------------------------------------------------------

def _text():
    return MODULE.read_text(encoding="utf-8")


def _cells(line):
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]


@functools.lru_cache(maxsize=None)
def _rows():
    """{label: ({column: text}, the label cell)} for every row of the three tables."""
    rows = {}
    for line in _text().splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        cells = _cells(line)
        if cells[0] == "perturbation":
            assert tuple(cells[1:]) == COLUMNS, cells
            continue
        label = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cells[0])
        assert label not in rows, label
        assert len(cells) == 1 + len(COLUMNS), (label, len(cells))
        rows[label] = (dict(zip(COLUMNS, cells[1:])), cells[0])
    return rows


def _effective(label, column):
    """A cell's text, with a leading "same" replaced by the cell on its left."""
    cols = _rows()[label][0]
    text = cols[column]
    if column in TIERS and text.startswith("same"):
        left = TIERS[TIERS.index(column) - 1]
        return _effective(label, left) + text[len("same"):]
    return text


_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:e-\d+)?")


def _numbers(text):
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return _NUMBER.findall(text)


def _significant(value):
    """The fewest significant digits that state `value` to 1e-6 of itself (at most 12)."""
    for n in range(1, 13):
        if abs(float(f"{value:.{n}g}") - value) <= 1e-6 * abs(value):
            return n
    return 12


def _printed_significant(printed):
    digits = printed.partition("e")[0].replace(".", "").lstrip("0")
    return max(len(digits), 1)


def _close(printed, value):
    """The cell's number states `value`: exactly when the value is a whole number; otherwise rounded to the precision
    the cell prints, with at least the significant digits the value has, up to three."""
    value = float(value)
    if abs(value - round(value)) <= 1e-6 * max(1.0, abs(value)):
        return float(printed) == round(value)
    mantissa, _, exponent = printed.partition("e")
    decimals = len(mantissa.partition(".")[2])
    step = 10.0 ** (-decimals + (int(exponent) if exponent else 0))
    enough = _printed_significant(printed) >= min(_significant(value), 3)
    return enough and abs(float(printed) - value) <= step / 2 + 1e-12


# --- the code --------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _cfg(name):
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    return getattr(composed, name)()


@functools.lru_cache(maxsize=None)
def _composed():
    """Every composed class that builds without the scene corpus, by name."""
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    out = {}
    for name in sorted(n for n in dir(composed) if n.startswith("StraferNavCfg_")):
        cls = getattr(composed, name)
        if [f.default_factory().kind for f in dataclasses.fields(cls) if f.name == "scene_source"] == ["infinigen"]:
            continue
        out[name] = _cfg(name)
    return out


def _c(t):
    return _cfg(CAMERA[t])


def _act(t):
    return _c(t).actions.wheel_velocities


def _noise(t, term):
    return getattr(_c(t).observations.policy, term).noise


def _event(t, name, cls=None):
    term = getattr((cls or _c(t)).events, name, None)
    return None if term is None else term.params


def _contract(t):
    from strafer_lab.tasks.navigation import sim_real_cfg

    return getattr(sim_real_cfg, CONTRACT[t])


class _Asset:
    def find_joints(self, names, preserve_order=False):
        return [0, 1, 2, 3], ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]


def _term(t, num_envs=4, **overrides):
    """The tier's action term on a stand-in env at the composed cfg's physics and control steps."""
    cfg = _c(t)
    term_cfg = cfg.actions.wheel_velocities.copy()
    for k, v in overrides.items():
        setattr(term_cfg, k, v)
    env = SimpleNamespace(scene={term_cfg.asset_name: _Asset()}, device="cpu", num_envs=num_envs,
                          physics_dt=cfg.sim.dt, step_dt=cfg.sim.dt * cfg.decimation)
    return term_cfg.class_type(term_cfg, env), env


def _step_dt(t):
    return _c(t).sim.dt * _c(t).decimation


def _trained_band(t, n=20000):
    """rad/s² the acceleration limit allows, at the ends of the per-env draw, through the reset that training uses;
    the build-time draw must agree."""
    term, env = _term(t, num_envs=n)
    built = term._max_delta_per_step.flatten() / env.step_dt
    term.reset(None)
    drawn = term._max_delta_per_step.flatten() / env.step_dt
    assert abs(float(built.min()) - float(drawn.min())) < 1.0 and abs(float(built.max()) - float(drawn.max())) < 1.0
    return float(drawn.min()), float(drawn.max())


def _trained_fixed(t, accel):
    term, env = _term(t, max_acceleration_range=(accel, accel), max_acceleration_rad_s2=accel)
    return float(term._max_delta_per_step.max()) / env.step_dt


@functools.lru_cache(maxsize=None)
def _usd_shapes():
    """Per collision prim of the robot's USD: the (static, dynamic) friction of the physics material UsdShade resolves
    for it (None when none resolves), and whether it authors a binding to a prim that carries a physics material."""
    from pxr import Usd, UsdPhysics, UsdShade

    from strafer_lab.assets import strafer

    stage = Usd.Stage.Open(strafer._STRAFER_USD_PATH, load=Usd.Stage.LoadAll)
    out = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        resolved = None
        if mat and mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            api = UsdPhysics.MaterialAPI(mat.GetPrim())
            resolved = (api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get())
        rel = prim.GetRelationship("material:binding:physics")
        targets = rel.GetTargets() if rel else []
        authored = bool(targets) and stage.GetPrimAtPath(targets[0]).HasAPI(UsdPhysics.MaterialAPI)
        out.append((resolved, authored))
    return out


@functools.lru_cache(maxsize=None)
def _usd_mass():
    """kg: the masses the robot's USD authors on its rigid bodies."""
    from pxr import Usd, UsdPhysics

    from strafer_lab.assets import strafer

    stage = Usd.Stage.Open(strafer._STRAFER_USD_PATH, load=Usd.Stage.LoadAll)
    return sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0.0 for p in stage.Traverse()
               if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAPI(UsdPhysics.MassAPI))


def _friction_run(t):
    """The friction event (the wheel-grip term) on a stand-in robot whose shapes carry the physics materials their USD
    resolves (the sim's default where none resolves, as PhysX applies it); returns the dynamic / static ratio it
    writes, and the number of shapes it writes, after checking it writes every one within the covers' range."""
    import warp as wp

    from strafer_lab.tasks.navigation.mdp import events

    params = _event(t, "randomize_friction")
    default = _c(t).sim.physics_material
    shapes = [r if r is not None else (default.static_friction, default.dynamic_friction) for r, _ in _usd_shapes()]
    table = torch.tensor([[s, d, 0.0] for s, d in shapes], dtype=torch.float32).repeat(8, 1, 1)
    written = {}
    view = SimpleNamespace(get_material_properties=lambda: table.clone(),
                           set_material_properties=lambda m, ids: written.setdefault("m", wp.to_torch(m).clone()))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(root_view=view)}, num_envs=8, device="cpu")
    term = events.randomize_wheel_grip(SimpleNamespace(params=params), env)
    term(env, torch.arange(8), **params)
    out = written["m"]
    lo, hi = _cover_band(params)
    assert bool(((out[..., 0] >= lo - 1e-6) & (out[..., 0] <= hi + 1e-6)).all())  # every shape takes the draw
    ratio = out[..., 1] / out[..., 0]
    assert float(ratio.max() - ratio.min()) < 1e-6
    return float(ratio.mean()), out.shape[1]


def _cover_band(params):
    """The roller covers' static band the term writes: 2 grip - floor, PhysX averaging the two materials."""
    lo, hi = params["grip_range"]
    return 2 * lo - params["floor_friction"], 2 * hi - params["floor_friction"]


def _signature_default(func, name):
    return inspect.signature(func).parameters[name].default


OFF = "off"


def _tiers(f):
    return {t: f(t) for t in TIERS}


# --- the rows: what every cell must hold -----------------------------------------------------------------------------

ROWS = {}


def row(label, basis_words):
    def register(fn):
        ROWS[label] = (fn, frozenset(basis_words))
        return fn
    return register


@row("Command delay", {"design"})
def _():
    def cell(t):
        a = _act(t)
        assert not a.delay_redraw_on_reset and tuple(a.substep_delay_range_s) == (0.0, 0.0), t  # the row: both off
        if a.delay_range_s[1] == 0:
            return OFF
        lo, hi = a.delay_range_s
        return [1000 * lo, 1000 * hi, lo / _step_dt(t), hi / _step_dt(t)]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Command hold", {"design"})
def _():
    def cell(t):
        a = _act(t)
        return OFF if a.hold_fraction_range[1] == 0 else [*a.hold_fraction_range, *(1000 * s for s in a.hold_run_range_s)]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Acceleration limit", {"unknown", "design"})
def _():
    def cell(t):
        a = _act(t)
        if math.isinf(a.max_acceleration_rad_s2):
            return OFF
        band = _trained_band(t)
        assert all(abs(b - f) < 1e-3 * f for b, f in zip(band, a.max_acceleration_range)), (t, band)  # field = trained
        assert math.isinf(a.accel_falloff_speed), t  # flat in the wheel's speed
        return list(band)
    nominal = _act("real").max_acceleration_rad_s2
    assert _act("robust").max_acceleration_rad_s2 == nominal
    trained = _trained_fixed("real", nominal)
    assert abs(trained - nominal) < 1e-3 * nominal, (trained, nominal)
    return {**_tiers(cell), "models": [], "basis": [trained]}


def _lag(t):
    """The 63 % rise time of each env's lag, from the built term's alpha: -dt / ln(1 - alpha)."""
    alpha, dt = _term(t)[0]._motor_alpha.double().flatten(), _step_dt(t)
    return [float(-dt / math.log(1.0 - a)) for a in alpha]


@row("Motor lag", {"unknown"})
def _():
    def cell(t):
        a = _act(t)
        if not a.enable_motor_dynamics:
            return OFF
        rise = _lag(t)
        band = a.motor_time_constant_range
        assert band is None or band[0] == band[1] == a.motor_time_constant, (t, band)  # the band at the nominal
        assert all(math.isclose(r, a.motor_time_constant, rel_tol=1e-6) for r in rise), (t, rise)  # field = rise time
        lead = [100.0 * (1.0 - math.exp(-1.0))] if t == "real" else []
        return lead + [rise[0]]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Command envelope", {"design"})
def _():
    from strafer_shared import constants as k

    assert k.NAV_LINEAR_VEL == pytest.approx(k.MAX_LINEAR_VEL * k.NAV_VEL_SCALE, abs=1e-4)
    assert k.NAV_ANGULAR_VEL == pytest.approx(k.MAX_ANGULAR_VEL * k.NAV_VEL_SCALE, abs=1e-4)

    def cell(t):
        r = _act(t).speed_scale_range
        return OFF if r is None else list(r)
    return {**_tiers(cell), "models": [k.NAV_VEL_SCALE], "basis": []}


@row("Twist clamp", {"derived"})
def _():
    def cell(t):
        term = _term(t)[0]
        return [term._max_linear_vel, term._max_angular_vel]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Wheel speed and torque", {"datasheet"})
def _():
    def cell(t):
        drive = _c(t).scene.robot.actuators["wheel_drives"]
        assert type(drive).__name__ == "ImplicitActuatorCfg"  # a velocity drive with a flat effort cap
        limit = _term(t)[0]._wheel_speed_limit
        assert limit == _act(t).max_wheel_angular_vel, (t, limit)  # the per-wheel limit at the rated speed
        return [limit, drive.effort_limit_sim]
    drive = _c("ideal").scene.robot.actuators["wheel_drives"]
    rpm = _act("ideal").max_wheel_angular_vel * 60.0 / (2.0 * math.pi)
    return {**_tiers(cell), "models": [rpm, drive.damping], "basis": []}


@row("Motor strength", {"design"})
def _():
    def cell(t):
        p = _event(t, "randomize_motor_strength")
        return OFF if p is None else list(p["strength_range"])
    return {**_tiers(cell), "models": [], "basis": []}


@row("Control period", {"design"})
def _():
    from strafer_shared import constants as k

    def cell(t):
        return [1, 1.0 / _step_dt(t), 1, 1.0 / _c(t).sim.dt]
    return {**_tiers(cell), "models": [1.0 / k.POLICY_PERIOD_S, k.CAMERA_HZ], "basis": []}


@row("IMU noise", {"unknown"})
def _():
    def cell(t):
        acc, gyr = _noise(t, ACC), _noise(t, GYR)
        if acc is None:
            assert gyr is None
            return OFF
        imu = _contract(t).sensors.imu
        assert acc.accel_bias_range[0] == -acc.accel_bias_range[1] and gyr.gyro_bias_range[0] == -gyr.gyro_bias_range[1]
        sigma = [acc.accel_noise_std, gyr.gyro_noise_std, imu.accel_noise_density, imu.gyro_noise_density]
        rate = [acc.control_frequency_hz] if t == "real" else []
        bias = [acc.accel_bias_range[1], gyr.gyro_bias_range[1], acc.accel_bias_drift_rate, gyr.gyro_bias_drift_rate]
        return sigma + rate + bias
    return {**_tiers(cell), "models": [], "basis": []}


@row("IMU and encoder latency", {"design"})
def _():
    def cell(t):
        if _noise(t, ACC) is None:
            return OFF
        steps = {_noise(t, n).latency_steps for n in (ACC, GYR, ENC)}
        assert len(steps) == 1, (t, steps)
        (v,) = steps
        return "none" if v == 0 else [v]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Wheel encoder noise", {"design"})
def _():
    def cell(t):
        e = _noise(t, ENC)
        if e is None:
            return OFF
        assert e.enable_quantization
        assert e.reported_speed_lag_s == 0.0, t  # the row says the lag is off
        body = _noise(t, "body_velocity_xy")
        assert type(body).__name__ == "BodyVelocityNoiseModelCfg", t  # the body velocity is this reading's kinematics
        terms = list(vars(_c(t).observations.policy))
        assert terms.index(ENC) < terms.index("body_velocity_xy"), t  # built after the encoder model it reads
        if t == "real":
            return [e.velocity_noise_std, e.max_velocity]
        assert e.max_velocity == _noise("real", ENC).max_velocity
        return [e.velocity_noise_std]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Sensor dropouts", {"design"})
def _():
    def cell(t):
        if _noise(t, ACC) is None:
            return OFF
        stuck = {_noise(t, n).stuck_probability for n in (ACC, GYR)}
        assert len(stuck) == 1, (t, stuck)  # the row speaks of "each IMU channel" at one rate
        vals = [stuck.pop(), _noise(t, ENC).stuck_probability]
        return "none" if not any(vals) else vals
    return {**_tiers(cell), "models": [], "basis": []}


@row("Depth disparity noise", {"design"})
def _():
    def cell(t):
        d = _noise(t, DEPTH)
        if d is None:
            return OFF
        if d.disparity_noise_px_range is not None:
            return list(d.disparity_noise_px_range)
        f, b = d.focal_length_px, d.baseline_m

        def mm(z):
            return 1000.0 * z * z * d.disparity_noise_px / (f * b)
        return [d.disparity_noise_px, d.width, d.height, f, b, mm(1.0), 1, mm(2.0), 2, mm(d.max_range), d.max_range]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Depth holes and lost frames", {"design"})
def _():
    from strafer_shared import constants as k

    def cell(t):
        d = _noise(t, DEPTH)
        if d is None:
            return OFF
        assert d.hole_fill == "median"
        return [d.hole_probability, d.frame_drop_prob]
    blocks = [k.PERCEPTION_HEIGHT // k.DEPTH_HEIGHT, k.PERCEPTION_WIDTH // k.DEPTH_WIDTH]
    return {**_tiers(cell), "models": blocks, "basis": []}


@row("Depth stream hold", {"design"})
def _():
    def cell(t):
        d = _noise(t, DEPTH)
        if d is None or d.hold_fraction_range[1] == 0:
            return OFF
        assert d.hold_burst_weight == 0.0  # the row describes no burst component
        return [*d.hold_fraction_range, *d.hold_run_range]
    return {**_tiers(cell), "models": [1.0 / _step_dt("real")], "basis": []}


@row("Depth latency", {"design"})
def _():
    def cell(t):
        d = _noise(t, DEPTH)
        return OFF if d is None else list(d.latency_steps_range)
    return {**_tiers(cell), "models": [], "basis": []}


@row("Depth range and grid", {"design"})
def _():
    from strafer_shared import constants as k

    from strafer_lab.tasks.navigation.mdp import observations

    def cell(t):
        term = _c(t).observations.policy.depth_image
        params = {n: term.params.get(n, _signature_default(observations.depth_image, n))
                  for n in ("nearfield_clip", "nearfield_fill", "max_depth")}
        cam = _c(t).scene.d555_camera
        d = _noise(t, DEPTH)
        if d is not None:
            assert (d.min_range, d.max_range) == (params["nearfield_fill"], params["max_depth"])
            assert (d.width, d.height) == (k.DEPTH_WIDTH, k.DEPTH_HEIGHT)
        return [params["nearfield_clip"], params["nearfield_fill"], params["max_depth"], cam.width, cam.height,
                k.DEPTH_WIDTH, k.DEPTH_HEIGHT]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Camera lens", {"measured"})
def _():
    def cell(t):
        cam = _c(t).scene.d555_camera
        spawn = cam.spawn
        fx = cam.width * spawn.focal_length / spawn.horizontal_aperture
        vertical = getattr(spawn, "vertical_aperture", None)
        if vertical is not None:
            assert cam.height * spawn.focal_length / vertical == pytest.approx(fx)
        return [fx, cam.width, cam.height]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Re-render after reset", {"design"})
def _():
    def cell(t):
        return [_c(t).num_rerenders_on_reset, _cfg(CAMERA_FREE[t]).num_rerenders_on_reset]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Mount misalignment", {"design"})
def _():
    def cell(t):
        p = _event(t, "randomize_d555_mount")
        if p is None:
            return OFF
        assert getattr(_c(t).events, "jitter_d555_camera_prim", None) is not None  # the rendered camera follows it
        return [p["max_angle_deg"]]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Subgoal-frame drift", {"measured", "design"})
def _():
    def cell(t):
        p = _event(t, "randomize_subgoal_drift")
        if p is None:
            return OFF
        assert p["jump_rate_hz"] == 0.0  # the row describes no jumps
        return [*p["gain_range"], p["position_rms_m"], p["heading_sigma_deg"], p["tau_s"]]
    for name, cfg in _composed().items():  # "on the goal-objective IDs it moves the goal"
        if name.startswith("StraferNavCfg_RL") and cfg.objective.kind == "goal":
            assert getattr(cfg.events, "randomize_subgoal_drift", None) is not None, name
    return {**_tiers(cell), "models": [], "basis": []}


@row("Path and lookahead", {"design"})
def _():
    def cell(t):
        if t == "ideal":
            ideal = [c for c in _composed().values() if c.realism.level == "ideal"]
            assert ideal and not [c for c in ideal if c.objective.kind == "subgoal"]
            return "no subgoal ID"
        g = _c(t).commands.goal_command
        return [g.waypoint_noise_std_m, *g.lookahead_randomization_m]
    for name, cfg in _composed().items():  # "subgoal IDs only"
        if cfg.objective.kind != "subgoal":
            assert getattr(cfg.commands.goal_command, "lookahead_randomization_m", None) is None, name
    return {**_tiers(cell), "models": [], "basis": []}


@row("Friction", {"design"})
def _():
    def cell(t):
        p = _event(t, "randomize_friction")
        if p is None:
            return OFF
        assert p["chassis_mode"] == "same", t  # the row: the chassis takes the covers' value
        ratio, drawn = _friction_run(t)
        floor = _floor_static(t)
        assert p["floor_friction"] == floor, t  # the term draws against the scene's floor
        grip = list(p["grip_range"])
        cover = list(_cover_band(p))  # PhysX averages the two materials' coefficients
        if t == "real":
            covers = sum(authored for _, authored in _usd_shapes())
            return [*grip, floor, *cover, ratio, drawn, covers]
        return [*grip, *cover]
    # The covers author a rubber material that resolves as no physics material, so where no draw runs (the ideal
    # tier) every shape keeps the sim's default. A cover whose material resolved would change that grip.
    assert not [r for r, _ in _usd_shapes() if r is not None], "a collision shape now resolves a physics material"
    still = (_c("ideal").sim.physics_material.static_friction + _floor_static("ideal")) / 2.0
    return {**_tiers(cell), "models": [still], "basis": []}


def _floor_static(t):
    """The floor's static friction, with both materials left at PhysX's default (average) combine."""
    floor, robot = _c(t).scene.terrain.physics_material, _c(t).sim.physics_material
    assert floor.friction_combine_mode in (None, "average") and robot.friction_combine_mode in (None, "average")
    return floor.static_friction


@row("Mass", {"design"})
def _():
    def cell(t):
        p = _event(t, "randomize_mass")
        return OFF if p is None else list(p["mass_range"])
    return {**_tiers(cell), "models": [_usd_mass()], "basis": []}


@row("Spawn and goal", {"design"})
def _():
    from strafer_lab.tasks.navigation.mdp import proc_room

    def cell(t):
        if t == "ideal":
            pose = _event(t, "reset_robot")["pose_range"]
            g = _c(t).commands.goal_command
            assert pose["x"] == pose["y"] == (-pose["x"][1], pose["x"][1]) and pose["yaw"] == (-math.pi, math.pi)
            curriculum = _c(t).curriculum.goal_distance.params  # it rewrites goal_range at every reset
            assert curriculum["command_name"] == "goal_command"
            return [pose["x"][1], curriculum["initial_range"], curriculum["step_size"], curriculum["success_threshold"],
                    curriculum["max_range"], g.min_goal_distance, *g.resampling_time_range]
        assert _event(t, "reset_robot")["yaw_range"] == (-math.pi, math.pi)
        free = _event(t, "generate_room", _cfg(CAMERA_FREE[t])).get("robot_spawn_inflation_cells", 0)
        assert free == 0
        return [_event(t, "generate_room")["robot_spawn_inflation_cells"], proc_room.GRID_RES]
    return {**_tiers(cell), "models": [], "basis": []}


@row("Room", {"design"})
def _():
    from strafer_lab.tasks.navigation.mdp import proc_room

    def cell(t):
        p = _event(t, "generate_room")
        if p is None:
            return "a plane"
        level = _event(t, "randomize_difficulty")
        place = p["placement"]
        free_cfg = _cfg(CAMERA_FREE[t])
        free, free_level = _event(t, "generate_room", free_cfg), _event(t, "randomize_difficulty", free_cfg)
        assert free_level["min_level"] == free_level["max_level"]
        assert free.get("p_ceil", _signature_default(proc_room.generate_proc_room, "p_ceil")) == 0.0  # open top
        free_wall = free.get("wall_height", _signature_default(proc_room.generate_proc_room, "wall_height"))
        return [level["min_level"], level["max_level"], p["wall_height"], p["span_max"], p["max_span_sum"],
                p["door_width_max"], place.column_count, 100 * place.column_prob, 100 * p["clutter_wall_bias_prob"],
                100 * p["p_ceil"], *p["ceiling_height_range"], free_level["min_level"], free_wall]
    return {**_tiers(cell), "models": [], "basis": []}


# --- the tests -------------------------------------------------------------------------------------------------------

CASES = [(label, column) for label in ROWS for column in COLUMNS]


def test_every_row_of_the_module_is_checked():
    assert sorted(_rows()) == sorted(ROWS)


@functools.lru_cache(maxsize=None)
def _expected(label):
    return ROWS[label][0]()


@pytest.mark.parametrize("label, column", CASES, ids=[f"{label}-{column}" for label, column in CASES])
def test_the_cell_holds_the_codes_numbers(label, column):
    want = _expected(label)[column]
    text = _effective(label, column)
    if isinstance(want, str):
        assert text == want, f"{label} ({column}): the code reads {want!r}; the cell reads {text!r}"
        return
    got = _numbers(text)
    assert len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want)), (
        f"{label} ({column}): the cell's numbers {got} against the code's {[round(float(w), 6) for w in want]}")


@pytest.mark.parametrize("label", sorted(ROWS))
def test_the_basis_names_its_kind(label):
    basis = _rows()[label][0]["basis"]
    words = {w for w in LEGEND if re.search(rf"(?<!\bnot )\b{w}\b", basis)}
    assert words == ROWS[label][1], (label, basis)


def test_the_depth_latency_basis_names_the_centre():
    words = {1: "one", 2: "two", 3: "three"}
    centre = {t: sum(_noise(t, DEPTH).latency_steps_range) / 2 for t in ("real", "robust")}
    basis = _rows()["Depth latency"][0]["basis"]
    assert f"centred on {words[int(centre['real'])]} control step ({words[int(centre['robust'])]} on `robust`)" in basis


def test_the_pipeline_runs_in_the_stated_order():
    from strafer_lab.tasks.navigation.mdp import actions

    source = inspect.getsource(actions.MecanumWheelAction.process_actions)
    marks = ["l1_clamp_twist_batched(", "torch.clamp(wheel_vels_kinematic", "self._action_delay_buffer.compute(",
             "self._hold.step()", "self._max_delta_per_step", "self._motor_alpha *"]
    at = [source.index(m) for m in marks]
    assert at == sorted(at), dict(zip(marks, at))
    sentence = re.search(r"runs once\s+per control step: (.*?)\n\n", _text(), re.S).group(1).replace("\n", " ")
    stages = ["twist clamp", "per-wheel speed clamp", "the delay", "the hold", "the acceleration limit",
              "the motor lag"]
    at = [sentence.index(s) for s in stages]
    assert at == sorted(at), sentence


def _full_forward(term, scale):
    term._speed_scale[:] = scale
    term.process_actions(torch.tensor([[1.0, 0.0, 0.0]]).repeat(term._speed_scale.shape[0], 1))
    return term.processed_actions.abs().max()


def test_the_envelope_scales_both_caps_where_the_row_says_so():
    stages_off = dict(delay_range_s=(0.0, 0.0), hold_fraction_range=(0.0, 0.0),
                      enable_motor_dynamics=False, max_acceleration_rad_s2=math.inf,
                      max_acceleration_range=(math.inf, math.inf))
    for t in TIERS:
        scaled = "× s" in _effective("Twist clamp", t)
        assert scaled == (_act(t).speed_scale_range is not None), t
        if scaled:
            term = _term(t, **stages_off)[0]
            assert float(_full_forward(term, 0.5)) == pytest.approx(0.5 * float(_full_forward(term, 1.0)))
            assert float(term._max_angular_vel) > 0


@pytest.mark.parametrize("t", TIERS)
def test_with_every_documented_stage_off_the_term_is_the_clamp_and_the_kinematics(t):
    """A stage with no row would show here: the term with delay, hold, envelope, acceleration limit and lag off."""
    from strafer_lab.tasks.navigation.mdp.actions import l1_clamp_twist_batched

    term = _term(t, num_envs=64, delay_range_s=(0.0, 0.0), hold_fraction_range=(0.0, 0.0),
                 speed_scale_range=None, enable_motor_dynamics=False, max_acceleration_rad_s2=math.inf,
                 max_acceleration_range=(math.inf, math.inf))[0]
    gen = torch.Generator().manual_seed(7)
    for _ in range(5):
        actions = 2.4 * torch.rand(64, 3, generator=gen) - 1.2
        term.process_actions(actions)
        body = l1_clamp_twist_batched(actions.clamp(-1.0, 1.0) * term._velocity_scale,
                                      vel_cap_linear_m_s=term._max_linear_vel,
                                      vel_cap_angular_rad_s=term._max_angular_vel)
        wheels = (body @ term._kinematic_matrix.T * term._wheel_axis_signs).clamp(
            -term._wheel_speed_limit, term._wheel_speed_limit)[:, term._joint_reorder]
        assert torch.allclose(term.processed_actions, wheels, atol=1e-6), t


EVENT_ROWS = {
    "randomize_difficulty": "Room", "generate_room": "Room", "reset_robot": "Spawn and goal",
    "randomize_friction": "Friction", "randomize_mass": "Mass", "randomize_motor_strength": "Motor strength",
    "randomize_d555_mount": "Mount misalignment", "jitter_d555_camera_prim": "Mount misalignment",
    "randomize_subgoal_drift": "Subgoal-frame drift",
}


def test_every_event_term_has_a_row():
    for name, cfg in _composed().items():
        for term_name, term in vars(cfg.events).items():
            if term is None or not hasattr(term, "func"):
                continue
            assert term_name in EVENT_ROWS, f"{name}: event {term_name!r} has no row in the module"
            assert term.mode == "reset", (name, term_name, term.mode)
            label = EVENT_ROWS[term_name]
            assert label is None or label in _rows(), label


STRUCTURAL = "structural"
CHECKED = "checked by a test below"
ACTION_FIELDS = {
    "asset_name": STRUCTURAL, "class_type": CHECKED, "clip": STRUCTURAL, "debug_vis": STRUCTURAL,
    "joint_names": STRUCTURAL, "wheel_axis_signs": STRUCTURAL, "wheel_radius": "Twist clamp",
    "wheel_base": "Twist clamp", "track_width": "Twist clamp", "motor_rpm": STRUCTURAL,
    "max_wheel_angular_vel": "Wheel speed and torque", "enable_motor_dynamics": "Motor lag",
    "motor_time_constant": "Motor lag", "motor_time_constant_range": "Motor lag",
    "wheel_speed_limit": "Wheel speed and torque", "accel_falloff_speed": "Acceleration limit",
    "delay_range_s": "Command delay", "delay_redraw_on_reset": "Command delay", "substep_delay_range_s": "Command delay",
    "hold_fraction_range": "Command hold", "hold_run_range_s": "Command hold", "speed_scale_range": "Command envelope",
    "max_acceleration_rad_s2": "Acceleration limit", "max_acceleration_range": "Acceleration limit",
}


def test_every_action_field_is_documented():
    from strafer_lab.tasks.navigation.mdp.actions import MecanumWheelActionCfg

    fields = {f.name for f in dataclasses.fields(MecanumWheelActionCfg)}
    assert fields == set(ACTION_FIELDS), sorted(fields ^ set(ACTION_FIELDS))
    for name, cfg in _composed().items():
        assert type(cfg.actions.wheel_velocities) is MecanumWheelActionCfg, name
        assert [n for n, v in vars(cfg.actions).items() if hasattr(v, "class_type")] == ["wheel_velocities"], name
    assert {v for v in ACTION_FIELDS.values() if v not in (STRUCTURAL, CHECKED)} <= set(_rows())
    from strafer_lab.tasks.navigation.mdp.actions import MecanumWheelAction

    for name, cfg in _composed().items():
        assert cfg.actions.wheel_velocities.class_type is MecanumWheelAction, name


NOT_APPLIED = "declared, not applied"
NOISE_FIELDS = {
    "IMUNoiseModelCfg": {
        "class_type": CHECKED, "func": STRUCTURAL, "noise_cfg": STRUCTURAL, "sensor_type": CHECKED,
        "output_size": STRUCTURAL, "control_frequency_hz": "IMU noise", "accel_noise_std": "IMU noise",
        "accel_bias_range": "IMU noise", "accel_bias_drift_rate": "IMU noise", "gyro_noise_std": "IMU noise",
        "gyro_bias_range": "IMU noise", "gyro_bias_drift_rate": "IMU noise",
        "stuck_probability": "Sensor dropouts", "latency_steps": "IMU and encoder latency",
        "latency_steps_range": "IMU and encoder latency"},
    "EncoderNoiseModelCfg": {
        "class_type": CHECKED, "func": STRUCTURAL, "noise_cfg": STRUCTURAL, "output_size": STRUCTURAL,
        "enable_quantization": "Wheel encoder noise",
        "reported_speed_form": "Wheel encoder noise", "reported_speed_lag_s": "Wheel encoder noise",
        "velocity_noise_std": "Wheel encoder noise", "max_velocity": "Wheel encoder noise",
        "control_frequency_hz": "Wheel encoder noise",
        "stuck_probability": "Sensor dropouts", "latency_steps": "IMU and encoder latency"},
    "BodyVelocityNoiseModelCfg": {"class_type": CHECKED, "func": STRUCTURAL, "noise_cfg": STRUCTURAL},
    "DepthNoiseModelCfg": {
        "class_type": CHECKED, "func": STRUCTURAL, "noise_cfg": STRUCTURAL, "baseline_m": "Depth disparity noise",
        "focal_length_px": "Depth disparity noise", "disparity_noise_px": "Depth disparity noise",
        "disparity_noise_px_range": "Depth disparity noise", "hole_probability": "Depth holes and lost frames",
        "hole_fill": "Depth holes and lost frames", "frame_drop_prob": "Depth holes and lost frames",
        "min_range": "Depth range and grid", "max_range": "Depth range and grid", "too_close_fill": STRUCTURAL,
        "height": "Depth range and grid", "width": "Depth range and grid", "hold_fraction_range": "Depth stream hold",
        "hold_run_range": "Depth stream hold", "hold_burst_weight": "Depth stream hold",
        "hold_burst_run_steps": "Depth stream hold",
        "latency_steps": "Depth latency", "latency_steps_range": "Depth latency"},
}
NOISY_TERMS = {ACC: "IMUNoiseModelCfg", GYR: "IMUNoiseModelCfg", ENC: "EncoderNoiseModelCfg",
               "body_velocity_xy": "BodyVelocityNoiseModelCfg",
               DEPTH: "DepthNoiseModelCfg"}
NOISE_MODELS = {"IMUNoiseModelCfg": "IMUNoiseModel", "EncoderNoiseModelCfg": "EncoderNoiseModel",
                "BodyVelocityNoiseModelCfg": "BodyVelocityNoiseModel",
                "DepthNoiseModelCfg": "DepthNoiseModel"}
SENSOR_TYPES = {ACC: "accel", GYR: "gyro"}


def test_policy_corruption_is_on_exactly_off_the_ideal_tier():
    """Isaac Lab drops every policy noise model of a group whose corruption switch is off."""
    for name, cfg in _composed().items():
        assert cfg.observations.policy.enable_corruption == (cfg.realism.level != "ideal"), name


def test_no_policy_term_is_clipped_or_modified_and_each_scale_is_one_value():
    scales = {}
    for name, cfg in _composed().items():
        for term_name, term in vars(cfg.observations.policy).items():
            if not hasattr(term, "func"):
                continue
            assert term.clip is None and not term.modifiers, (name, term_name)
            scales.setdefault(term_name, set()).add(term.scale)
    assert all(len(v) == 1 for v in scales.values()), {k: v for k, v in scales.items() if len(v) > 1}


CURRICULUM_ROWS = {"goal_distance": "Spawn and goal"}


def test_every_curriculum_term_has_a_row():
    for name, cfg in _composed().items():
        terms = {k: v for k, v in vars(getattr(cfg, "curriculum", None) or SimpleNamespace()).items()
                 if v is not None and hasattr(v, "func")}
        for term_name in terms:
            assert term_name in CURRICULUM_ROWS, f"{name}: curriculum term {term_name!r} has no row in the module"
            assert cfg.realism.level == "ideal" and cfg.objective.kind == "goal", (name, term_name)  # the row's cell
    assert set(CURRICULUM_ROWS.values()) <= set(_rows())


COMMAND_FIELDS = {
    "class_type": CHECKED, "cmd_kind": STRUCTURAL, "debug_vis": STRUCTURAL, "element_names": STRUCTURAL,
    "resampling_time_range": "Spawn and goal", "asset_name": STRUCTURAL, "min_goal_distance": "Spawn and goal",
    "multi_goal": STRUCTURAL, "goal_reach_threshold": STRUCTURAL, "goal_reach_cooldown_s": STRUCTURAL,
    "spawn_points_xy": STRUCTURAL, "goal_sphere_visualizer_cfg": STRUCTURAL, "goal_heading_visualizer_cfg": STRUCTURAL,
    "goal_range": "Spawn and goal", "lookahead_m": "Path and lookahead",
    "lookahead_randomization_m": "Path and lookahead", "max_goal_attempts": STRUCTURAL, "path_spacing_m": STRUCTURAL,
    "waypoint_noise_std_m": "Path and lookahead", "dwell_radius_m": STRUCTURAL, "dwell_speed_max_m_s": STRUCTURAL,
    "dwell_steps": STRUCTURAL, "max_path_points": STRUCTURAL, "subgoal_sphere_visualizer_cfg": STRUCTURAL,
    "subgoal_heading_visualizer_cfg": STRUCTURAL, "path_points_visualizer_cfg": STRUCTURAL,
}
COMMAND_TERMS = {"GoalCommandCfg": "GoalCommand", "GoalCommandProcRoomCfg": "GoalCommandProcRoom",
                 "SubgoalCommandCfg": "SubgoalCommand"}


def test_every_command_field_is_documented():
    from strafer_lab.tasks.navigation.mdp import commands

    for cfg_name, term_name in COMMAND_TERMS.items():
        fields = {f.name for f in dataclasses.fields(getattr(commands, cfg_name))}
        assert fields <= set(COMMAND_FIELDS), (cfg_name, sorted(fields - set(COMMAND_FIELDS)))
    assert {v for v in COMMAND_FIELDS.values() if v not in (STRUCTURAL, CHECKED)} <= set(_rows())
    for name, cfg in _composed().items():
        g = cfg.commands.goal_command
        assert type(g).__name__ in COMMAND_TERMS, (name, type(g).__name__)
        assert g.class_type is getattr(commands, COMMAND_TERMS[type(g).__name__]), name
        assert [n for n, v in vars(cfg.commands).items() if hasattr(v, "class_type")] == ["goal_command"], name


def test_every_noise_field_is_documented():
    from strafer_lab.tasks.navigation.mdp import noise_models

    for cls_name, mapping in NOISE_FIELDS.items():
        fields = {f.name for f in dataclasses.fields(getattr(noise_models, cls_name))}
        assert fields == set(mapping), (cls_name, sorted(fields ^ set(mapping)))
        assert {v for v in mapping.values() if v not in (STRUCTURAL, NOT_APPLIED, CHECKED)} <= set(_rows())
    for name, cfg in _composed().items():
        for term_name, term in vars(cfg.observations.policy).items():
            noise = getattr(term, "noise", None)
            if noise is None:
                continue
            assert NOISY_TERMS.get(term_name) == type(noise).__name__, f"{name}: noisy term {term_name!r} has no row"
            assert noise.class_type is getattr(noise_models, NOISE_MODELS[type(noise).__name__]), (name, term_name)
            if term_name in SENSOR_TYPES:
                assert noise.sensor_type == SENSOR_TYPES[term_name], (name, term_name, noise.sensor_type)
            if isinstance(noise, noise_models.DepthNoiseModelCfg):
                assert noise.latency_steps_range is not None, name  # the row reads the range
                assert noise.too_close_fill == "near", name


DECLARED_NOT_APPLIED = set()
RGB_NOT_APPLIED = "RGBCameraNoiseCfg."  # "Not applied on any tier: RGB noise (no registered ID observes RGB)"


def _readers(name):
    source = "\n".join(p.read_text(encoding="utf-8") for p in PACKAGE.rglob("*.py"))
    return len(re.findall(rf"\.{name}\b|[\"']{name}[\"']", source))


def test_the_fields_nothing_reads_are_the_modules_declared_list():
    from strafer_lab.tasks.navigation import sim_real_cfg

    unread, seen = set(), set()

    def walk(cls):
        if cls in seen:
            return
        seen.add(cls)
        for f in dataclasses.fields(cls):
            if _readers(f.name) == 0:
                unread.add(f"{cls.__name__}.{f.name}")
            sub = getattr(sim_real_cfg, f.type.split("|")[0].strip(), None) if isinstance(f.type, str) else f.type
            if isinstance(sub, type) and dataclasses.is_dataclass(sub):
                walk(sub)

    walk(type(sim_real_cfg.REAL_ROBOT_CONTRACT))
    assert {u for u in unread if not u.startswith(RGB_NOT_APPLIED)} == DECLARED_NOT_APPLIED
    named = set(re.findall(r"`(\w+)`", " ".join(re.findall(r"Declared, not applied:(.*?)(?:\n\n|\Z)", _text(), re.S))))
    field_names = {d.split(".")[1] for d in DECLARED_NOT_APPLIED}
    noise_unread = {f for c, m in NOISE_FIELDS.items() for f, v in m.items() if v == NOT_APPLIED}
    assert field_names | noise_unread <= named, sorted((field_names | noise_unread) - named)
    for f in noise_unread:
        assert _readers(f) == 0, f


ANCHORS = {
    "actions.py#L220": "torch.randint", "actions.py#L231": "HoldProcess(", "actions.py#L254": "_max_delta_per_step",
    "actions.py#L395": "torch.expm1(", "actions.py#L276": "_draw_speed_scale",
    "actions.py#L451": "l1_clamp_twist_batched(", "actions.py#L572": "max_wheel_angular_vel",
    "events.py#L454": "def randomize_motor_strength", "events.py#L505": "def randomize_d555_mount_offset",
    "events.py#L407": "def randomize_subgoal_drift", "events.py#L175": "class randomize_wheel_grip(",
    "events.py#L358": "def randomize_mass", "events.py#L622": "def reset_robot_proc_room",
    "noise_models.py#L185": "class IMUNoiseModel(", "noise_models.py#L352": "class EncoderNoiseModel(",
    "noise_models.py#L478": "class DepthNoiseModel(",
    "observations.py#L45": "def reduce_depth_to_policy_grid", "d555_cfg.py#L145": "focal_length=",
    "composed_env_cfg.py#L572": "num_rerenders_on_reset = 1", "commands.py#L668": "perturb_waypoints(",
    "proc_room.py#L990": "def generate_proc_room", "sim_real_cfg.py#L713": "IDEAL_SIM_CONTRACT",
}


def test_every_line_link_lands_on_its_symbol():
    links = re.findall(r"\]\(([^)#]+)#L(\d+)\)", _text())
    seen = set()
    for path, line in links:
        target = (MODULE.parent / path).resolve()
        key = f"{target.name}#L{line}"
        assert key in ANCHORS, f"{key}: a line link with no pinned symbol"
        assert ANCHORS[key] in target.read_text(encoding="utf-8").splitlines()[int(line) - 1], key
        seen.add(key)
    assert seen == set(ANCHORS), sorted(set(ANCHORS) - seen)
