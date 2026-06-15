# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the composable navigation env cfgs.

Each composed RL variant must keep the policy-facing contract a trained policy
depends on — observation terms (order, scales, noise), action layout, DR
sequence, scene size, shared runtime. Drift flips a stored golden hash and
fails here.

A camera's rendered channels are deliberately outside the hashed contract: a
depth-only observation never reads rgb, so what a camera renders beyond the
observed channels does not change the policy. A separate check confirms every
sensor an observation term reads is present with the channel it needs.

Inspects config dataclass attributes only — no simulation stepping.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test_sim/env/test_composition_contract.py -v
"""

import hashlib
import json

import pytest

import strafer_lab.tasks.navigation.composed_env_cfg as composed


# =====================================================================
# Canonical contract serializer (mirrors the one that produced the goldens)
# =====================================================================

_CONTRACT_FIELDS = (
    "seed", "decimation", "episode_length_s",
    "observations", "actions", "events",
    "commands", "rewards", "terminations", "curriculum",
)
_SIM_FIELDS = ("dt", "render_interval")


def _canon(obj):
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
    snap["scene_num_envs"] = cfg.scene.num_envs
    snap["scene_env_spacing"] = cfg.scene.env_spacing
    return snap


def _hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# Golden contract hashes captured from the legacy RL classes
# (StraferNavEnvCfg_*_ProcRoom_*) before they were deleted. These are the
# obs/action/DR contract a trained policy + the DEPTH inference package depend
# on. Do not edit to make a test pass — a mismatch means the composition
# drifted from the contract a checkpoint was trained against.
_CONTRACT_GOLDENS = {
    "RLDepth_Real": "cd24b3ca082c169413c19ab04d48f8be920730f45da3ae3523a21b7fe90718df",
    "RLDepth_Robust": "4776f7d4cc36ccef858409400da67ed94d327a417f2aa0692fa32fd3ae710b21",
    "RLNoCam": "a14787862b3a39c7c778c189aba9d45cc217783ce2cdbd2332068f8c62805bad",
    "RLDepth_Real_PLAY": "3e340e6e46aa23aa54c7359ef204a39eaae55857b0474ebb66bf8ac648a0896d",
    "RLDepth_Robust_PLAY": "787221fd5f6ac819010311eafac0d79292f9800ad2b70c8286851ecc4fdf5659",
    "RLNoCam_PLAY": "7c60ad9bf35bcfaf3d0b5020d1e58479fefd4334035758f2400ab134e21f6345",
    # Subgoal path-tracking variants: goldens snapshotted at variant
    # introduction (no legacy class preceded them). Same do-not-edit rule —
    # a mismatch means the contract a NOCAM_SUBGOAL checkpoint trains
    # against has drifted.
    "RLNoCamSubgoal_Real": "8e9f4e81a0fecd5627805a94192cb289e2b9d91bfb04809a6b28557df0e0208d",
    "RLNoCamSubgoal_Robust": "a931dbda67d60f34a7fa575a9441c6073b7d066b3d32a83807c4178d54616b82",
    "RLNoCamSubgoal_Real_PLAY": "f72a02facf66f6e95e3c4770abd2581a913d96ef2e8c80a03ee5a681927e20f9",
    "RLNoCamSubgoal_Robust_PLAY": "63e62060edae8f774b1195a4f14df91d1b0d0309ea739575507ba36b77665f8f",
}

# The depth observation a checkpoint consumes — captured identical across the
# (now dropped) plane / Infinigen / ProcRoom depth variants, which justified
# dropping them: any depth checkpoint stays valid regardless of which produced it.
_DEPTH_OBS_GOLDEN = "179860c5765f2474043c235e0f608ba99c698e92319021bdb85eca6b2d69dce5"

_COMPOSED_RL = {
    "RLDepth_Real": composed.StraferNavCfg_RLDepth_Real,
    "RLDepth_Robust": composed.StraferNavCfg_RLDepth_Robust,
    "RLNoCam": composed.StraferNavCfg_RLNoCam,
    "RLDepth_Real_PLAY": composed.StraferNavCfg_RLDepth_Real_PLAY,
    "RLDepth_Robust_PLAY": composed.StraferNavCfg_RLDepth_Robust_PLAY,
    "RLNoCam_PLAY": composed.StraferNavCfg_RLNoCam_PLAY,
    "RLNoCamSubgoal_Real": composed.StraferNavCfg_RLNoCamSubgoal_Real,
    "RLNoCamSubgoal_Robust": composed.StraferNavCfg_RLNoCamSubgoal_Robust,
    "RLNoCamSubgoal_Real_PLAY": composed.StraferNavCfg_RLNoCamSubgoal_Real_PLAY,
    "RLNoCamSubgoal_Robust_PLAY": composed.StraferNavCfg_RLNoCamSubgoal_Robust_PLAY,
}


# =====================================================================
# Snapshot gate: composed RL variant matches the frozen legacy contract
# =====================================================================


@pytest.mark.parametrize("name", sorted(_COMPOSED_RL), ids=lambda n: n)
def test_composed_rl_variant_matches_frozen_contract(name):
    """The composed RL variant's policy-facing contract is byte-identical to
    the legacy class it replaced (frozen golden hash)."""
    got = _hash(_contract(_COMPOSED_RL[name]()))
    assert got == _CONTRACT_GOLDENS[name], (
        f"{name} diverged from the frozen contract — a trained policy or the "
        f"DEPTH inference package would silently break.\n  got    {got}\n  "
        f"golden {_CONTRACT_GOLDENS[name]}"
    )


def test_depth_obs_contract_matches_frozen_golden():
    """The depth observation a checkpoint consumes is unchanged."""
    got = _hash(_canon(composed.StraferNavCfg_RLDepth_Real().observations))
    assert got == _DEPTH_OBS_GOLDEN


def _obs_term_names(obs_cfg):
    return [
        k for k, v in obs_cfg.policy.__dict__.items()
        if not k.startswith("_") and hasattr(v, "func")
    ]


@pytest.mark.parametrize("name", sorted(_COMPOSED_RL), ids=lambda n: n)
def test_composed_rl_variant_sensors_present(name):
    """Every sensor an observation term reads is present with its channel."""
    cfg = _COMPOSED_RL[name]()
    scene = cfg.scene
    assert getattr(scene, "d555_imu", None) is not None
    assert getattr(scene, "contact_sensor", None) is not None
    names = _obs_term_names(cfg.observations)
    cam = getattr(scene, "d555_camera", None)
    if "depth_image" in names:
        assert cam is not None and "distance_to_image_plane" in list(cam.data_types)
    if "rgb_image" in names:
        assert cam is not None and "rgb" in list(cam.data_types)


@pytest.mark.parametrize(
    "name", ["RLDepth_Real", "RLDepth_Robust"], ids=lambda n: n
)
def test_depth_policy_camera_still_renders_rgb_for_viewport(name):
    """A depth-only RL variant's policy camera still renders the rgb channel.

    The observation is depth-only (guarded byte-for-byte by the frozen contract
    above), but the policy camera is also the scene's rgb render product: the
    RTX viewport / ``--video`` colour pipeline only comes up when some camera
    renders rgb. Drop it and clips are all-black and the headed viewport stalls.
    Guards against pruning the policy camera down to only the observed channel,
    which leaves the viewport / ``--video`` with no rgb render product.
    """
    cfg = _COMPOSED_RL[name]()
    cam = cfg.scene.d555_camera
    assert cam is not None, f"{name} has no policy camera"
    channels = list(cam.data_types)
    assert "rgb" in channels, (
        f"{name} policy camera dropped rgb ({channels}) — viewport/--video "
        f"will render black"
    )
    assert "distance_to_image_plane" in channels, (
        f"{name} policy camera dropped depth ({channels})"
    )


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
# Objective axis behavior
# =====================================================================


def test_subgoal_objective_swaps_task_blocks():
    """The subgoal objective selects the path-tracking command / reward /
    termination blocks while leaving the scene and obs selection alone."""
    cfg = composed.StraferNavCfg_RLNoCamSubgoal_Real()
    assert type(cfg.commands).__name__ == "CommandsCfg_ProcRoom_Subgoal"
    assert hasattr(cfg.rewards, "along_track_progress")
    assert hasattr(cfg.rewards, "cross_track_error")
    assert hasattr(cfg.terminations, "path_complete")
    assert hasattr(cfg.terminations, "off_path_divergence")


def test_subgoal_observations_match_nocam_baseline():
    """The subgoal variant's observation contract is byte-identical to the
    no-cam baseline's: same network input layout, only the command term
    behind the goal-shaped fields differs."""
    sub = composed.StraferNavCfg_RLNoCamSubgoal_Real()
    baseline = composed.StraferNavCfg_RLNoCam()
    assert _hash(_canon(sub.observations)) == _hash(_canon(baseline.observations))


def test_subgoal_robust_drops_goal_pose_noise_event():
    """Goal-pose reset noise targets the fixed-goal command's state; the
    subgoal composition disables it (waypoint noise on the command term is
    the deployment-noise analog)."""
    cfg = composed.StraferNavCfg_RLNoCamSubgoal_Robust()
    assert cfg.events.randomize_goal_noise is None


def test_subgoal_lookahead_pinned_to_shared_constant():
    """The realistic subgoal variant tracks at the shared deployment lookahead
    distance, with no randomization."""
    from strafer_shared.constants import SUBGOAL_LOOKAHEAD_M

    cmd = composed.StraferNavCfg_RLNoCamSubgoal_Real().commands.goal_command
    assert cmd.lookahead_m == SUBGOAL_LOOKAHEAD_M
    assert cmd.lookahead_randomization_m is None


def test_subgoal_robust_randomizes_lookahead_distance():
    """The robust tier widens the lookahead into a band so the policy is not
    brittle to the deployed selector's exact lookahead distance."""
    cmd = composed.StraferNavCfg_RLNoCamSubgoal_Robust().commands.goal_command
    band = cmd.lookahead_randomization_m
    assert band is not None
    lo, hi = band
    assert lo < cmd.lookahead_m < hi


def test_subgoal_objective_requires_procroom_scene():
    """The subgoal planner consumes the ProcRoom occupancy grids; other
    scene sources are rejected at composition time."""
    with pytest.raises(ValueError, match="subgoal"):
        composed._ComposedStraferNavEnvCfg(
            sensors=composed.SensorStackCfg(cameras_required=()),
            scene_source=composed.SceneSourceCfg(kind="plane"),
            realism=composed.RealismCfg(level="real"),
            objective=composed.ObjectiveCfg(kind="subgoal"),
        )


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
    cfg.__post_init__()
    per = cfg.scene.d555_camera_perception
    assert tuple(sorted(per.data_types)) == ("distance_to_image_plane", "rgb")
