# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Contract + composition tests for the composable navigation env cfgs.

The composable env cfg (``composed_env_cfg.py``) re-expresses each RL
environment as a composition over three orthogonal axes — sensor stack,
scene source, realism level. For a trained policy to keep working, the
composed RL variant must be structurally identical to the hand-written class
it replaces: same observation terms in the same order with the same scales /
noise, same action layout, same domain-randomization sequence, same scene
size, same shared runtime.

These tests assert that byte-level identity by serializing the policy-facing
contract of the composed variant and the legacy class and comparing them in
one process. The scene's *unused* camera channels are intentionally excluded
from the comparison — a depth-only observation never reads the RGB channel,
so the composition is free to stop rendering it (a render-cost win) while the
observation tensor stays identical. A separate check confirms every sensor an
observation term reads is present with the channel it needs.

Purely inspects config dataclass attributes — no simulation stepping needed.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test/env/test_composition_contract.py -v
"""

import pytest

import strafer_lab.tasks.navigation.strafer_env_cfg as legacy
import strafer_lab.tasks.navigation.composed_env_cfg as composed


# =====================================================================
# Canonical contract serializer
# =====================================================================

# Fields that define what a trained policy experiences. The scene is checked
# separately (size + obs-referenced sensor presence) because the observation
# tensor depends on the channels an obs term reads, not on unused camera
# channels the composition may drop.
_CONTRACT_FIELDS = (
    "seed", "decimation", "episode_length_s",
    "observations", "actions", "events",
    "commands", "rewards", "terminations", "curriculum",
)
_SIM_FIELDS = ("dt", "render_interval")


def _canon(obj):
    """Order-preserving canonical form: callables -> qualname, floats -> repr,
    config objects -> ordered [name, value] pairs (term ordering is part of
    the observation contract), plain dicts -> key-sorted."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return repr(obj)
    if callable(obj) and hasattr(obj, "__qualname__"):
        return f"<fn {getattr(obj, '__module__', '?')}.{obj.__qualname__}>"
    if isinstance(obj, dict):
        return {str(k): _canon(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    d = getattr(obj, "__dict__", None)
    if d is not None:
        return ["@" + type(obj).__name__] + [
            [k, _canon(v)] for k, v in d.items() if not k.startswith("_")
        ]
    return repr(obj)


def _contract(cfg):
    snap = {f: _canon(getattr(cfg, f)) for f in _CONTRACT_FIELDS}
    sim = cfg.sim
    snap["sim"] = {f: _canon(getattr(sim, f)) for f in _SIM_FIELDS}
    return snap


def _obs_term_names(obs_cfg):
    return [
        k for k, v in obs_cfg.policy.__dict__.items()
        if not k.startswith("_") and hasattr(v, "func")
    ]


def _sensor_problems(cfg):
    """Every sensor an observation term reads must exist with its channel."""
    scene = cfg.scene
    problems = []
    if getattr(scene, "d555_imu", None) is None:
        problems.append("missing d555_imu")
    if getattr(scene, "contact_sensor", None) is None:
        problems.append("missing contact_sensor")
    names = _obs_term_names(cfg.observations)
    cam = getattr(scene, "d555_camera", None)
    if "depth_image" in names:
        if cam is None:
            problems.append("depth obs but no d555_camera")
        elif "distance_to_image_plane" not in list(cam.data_types):
            problems.append("d555_camera lacks distance_to_image_plane")
    if "rgb_image" in names:
        if cam is None or "rgb" not in list(cam.data_types):
            problems.append("rgb obs but d555_camera lacks rgb")
    return problems


# =====================================================================
# Snapshot gate: composed RL variant == legacy class it replaces
# =====================================================================

# (composed variant, legacy class it replaces)
_RL_PAIRS = [
    (composed.StraferNavCfg_RLDepth_Real, legacy.StraferNavEnvCfg_Real_ProcRoom_Depth),
    (composed.StraferNavCfg_RLDepth_Robust, legacy.StraferNavEnvCfg_Robust_ProcRoom_Depth),
    (composed.StraferNavCfg_RLNoCam, legacy.StraferNavEnvCfg_Real_ProcRoom_NoCam),
    (composed.StraferNavCfg_RLDepth_Real_PLAY, legacy.StraferNavEnvCfg_Real_ProcRoom_Depth_PLAY),
    (composed.StraferNavCfg_RLDepth_Robust_PLAY, legacy.StraferNavEnvCfg_Robust_ProcRoom_Depth_PLAY),
    (composed.StraferNavCfg_RLNoCam_PLAY, legacy.StraferNavEnvCfg_Real_ProcRoom_NoCam_PLAY),
]


@pytest.mark.parametrize("new_cls,old_cls", _RL_PAIRS, ids=lambda c: c.__name__)
def test_composed_rl_variant_matches_legacy_contract(new_cls, old_cls):
    """The composed RL variant's policy-facing contract is byte-identical to
    the hand-written class it replaces."""
    assert _contract(new_cls()) == _contract(old_cls()), (
        f"{new_cls.__name__} diverges from {old_cls.__name__} — a trained "
        "policy or the DEPTH inference package would silently break."
    )


@pytest.mark.parametrize("new_cls,old_cls", _RL_PAIRS, ids=lambda c: c.__name__)
def test_composed_rl_variant_scene_size_and_sensors(new_cls, old_cls):
    """Composed RL scene keeps the legacy env count + spacing, and every
    observation term's sensor is present with its channel."""
    new_cfg, old_cfg = new_cls(), old_cls()
    assert new_cfg.scene.num_envs == old_cfg.scene.num_envs
    assert new_cfg.scene.env_spacing == old_cfg.scene.env_spacing
    assert _sensor_problems(new_cfg) == []


def test_depth_obs_contract_invariant_across_scene_sources():
    """The depth observation a checkpoint consumes is identical whether it
    trained on plane, Infinigen, or ProcRoom — so a depth checkpoint stays
    valid no matter which dropped variant produced it."""
    ref = _canon(legacy.StraferNavEnvCfg_Real_ProcRoom_Depth().observations)
    assert _canon(legacy.StraferNavEnvCfg_Real_Depth().observations) == ref
    assert _canon(legacy.StraferNavEnvCfg_Real_InfinigenDepth().observations) == ref
    assert _canon(composed.StraferNavCfg_RLDepth_Real().observations) == ref


# =====================================================================
# Axis-cfg unit behavior
# =====================================================================


def test_sensor_stack_single_camera_selects_only_that_camera():
    """A stack of one perception RGB token requests only the perception
    camera, RGB channel only — no policy camera, no depth."""
    ss = composed.SensorStackCfg(cameras_required=("rgb_full",))
    assert ss.has_perception_camera() is True
    assert ss.has_policy_camera() is False
    assert ss.perception_data_types() == ("rgb",)
    assert ss.policy_data_types() == ()
    assert ss.obs_profile() == "nocam"


def test_sensor_stack_depth_policy_is_depth_profile():
    ss = composed.SensorStackCfg(cameras_required=("depth_policy",))
    assert ss.has_policy_camera() is True
    assert ss.policy_data_types() == ("distance_to_image_plane",)
    assert ss.obs_profile() == "depth"


def test_sensor_stack_rejects_unknown_token():
    with pytest.raises(ValueError):
        composed.SensorStackCfg(cameras_required=("lidar",)).normalized()


# =====================================================================
# Composition: a capture variant renders exactly its stack
# =====================================================================


def test_teleop_capture_reduced_stack():
    """Teleop's default reduced stack = perception RGB only: the policy camera
    is dropped and no image observation term is present."""
    cfg = composed.StraferNavCfg_TeleopCapture()
    assert getattr(cfg.scene, "d555_camera", None) is None
    per = cfg.scene.d555_camera_perception
    assert tuple(per.data_types) == ("rgb",)
    img_terms = [t for t in _obs_term_names(cfg.observations) if "image" in t]
    assert img_terms == []


def test_bridge_autonomy_expanded_stack():
    """Bridge's default expanded stack renders perception RGB+depth and the
    policy depth camera."""
    cfg = composed.StraferNavCfg_BridgeAutonomy()
    per = cfg.scene.d555_camera_perception
    assert "rgb" in list(per.data_types) and "distance_to_image_plane" in list(per.data_types)
    pol = cfg.scene.d555_camera
    assert pol is not None and "distance_to_image_plane" in list(pol.data_types)


def test_capture_variant_overrides_stack_per_session():
    """The stack is a per-session choice: overriding cameras_required on the
    axis cfg changes what the env renders."""
    cfg = composed.StraferNavCfg_TeleopCapture()
    cfg.sensors = composed.SensorStackCfg(cameras_required=("rgb_full", "depth_full"))
    # Re-run composition with the overridden stack.
    cfg.__post_init__()
    per = cfg.scene.d555_camera_perception
    assert tuple(sorted(per.data_types)) == ("distance_to_image_plane", "rgb")
