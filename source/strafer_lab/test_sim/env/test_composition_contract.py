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
    "RLNoCamSubgoal_Real": "5185c4759b8da15d2b1eac5ce796e9e547eabf81819ca2599122c747d5bd5e77",
    "RLNoCamSubgoal_Robust": "1f16f07c038d3eb05c765fe99d857a2ef68e80d369dc279ad7e81c25ef760aa5",
    "RLNoCamSubgoal_Real_PLAY": "0ec2595c9efb9ea6846a4cfa9b78025babb32fe377584a0d5a2c7049176d4079",
    "RLNoCamSubgoal_Robust_PLAY": "f1fc1058d0ad698faa88986e8fd2ae0dada2c3701a7cbdf1e24e19a25f839c11",
    "RLDepthSubgoal_Real": "40a2a824377036f9cf3e7974b24b28bc0ec2ea8848706b44cc8252bbdfe1b4cb",
    "RLDepthSubgoal_Robust": "2c5969a27cd43317fd4d390e6ae4ca343b9cd6b8f654264c8e2d09543242eccc",
    "RLDepthSubgoal_Real_PLAY": "6bd733a6b1b06303dc40da19068b0f96970550994cb1ba9b69564d5e060e1865",
    "RLDepthSubgoal_Robust_PLAY": "3f0e0b5402c7279cd5ef0f6222753deca21e60c88494192e502f41962c08aa78",
    # Depth-enrichment variants — NEW IDs frozen at creation (no prior checkpoint
    # depends on them). The enrichment lives in the `events` field (un-pinned
    # difficulty + enriched generation params, incl. the tall-object heights);
    # the observation contract is byte-identical to the open-top depth variants
    # (asserted separately).
    "RLDepthEnriched_Real": "800b2cb41f3047e213d7321089b4762e2b15008cc9e280bf4da8b7a5220b9088",
    "RLDepthEnriched_Robust": "0f7bb4bc4793a8d476d50cbfe0564f20e34e71318b35f9413e37d8ae3f12e516",
    "RLDepthEnriched_Real_PLAY": "8564044db2641f23564d70902f05a7d9051b211757948925e86a1a7fa1918603",
    "RLDepthEnriched_Robust_PLAY": "d14ed49d164522a28e88793a655a3f056522c526806e770d50f7efcf33eb363b",
    "RLDepthSubgoalEnriched_Real": "bf06a62d0f569ef7b0818d1af6a84647d098b2a6a0c2dbd4d855605c87525142",
    "RLDepthSubgoalEnriched_Robust": "f86a2e37302c9e9ca6e897e00a0d04e7c04d1922fd1cb1ef17e632a7f97fe4c8",
    "RLDepthSubgoalEnriched_Real_PLAY": "95fb664644db84c2c5d3f0d11554aef9e9f3cca539cc6ffac055c5818b84f639",
    "RLDepthSubgoalEnriched_Robust_PLAY": "9c7616ac94dc12513892f9eaa47ed3d20ee4f9d75710de4a3bfb401ca4b67f3b",
}

# Frozen signature (slot name set + spawn sizes) of the pre-enrichment 44-object
# palette. The composition hash cannot see the rigid-object palette (it is on the
# scene, not a hashed manager), so this pins the NOCAM promise: NOCAM keeps this
# exact palette, and the enriched palette adds no NEW slot (only taller walls).
_PALETTE_GOLDEN = "cf3499bf212a50c571b1bda4980fbbf16c7ad743dc40c5e88b7b528de3d41832"

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
    "RLDepthSubgoal_Real": composed.StraferNavCfg_RLDepthSubgoal_Real,
    "RLDepthSubgoal_Robust": composed.StraferNavCfg_RLDepthSubgoal_Robust,
    "RLDepthSubgoal_Real_PLAY": composed.StraferNavCfg_RLDepthSubgoal_Real_PLAY,
    "RLDepthSubgoal_Robust_PLAY": composed.StraferNavCfg_RLDepthSubgoal_Robust_PLAY,
    "RLDepthEnriched_Real": composed.StraferNavCfg_RLDepthEnriched_Real,
    "RLDepthEnriched_Robust": composed.StraferNavCfg_RLDepthEnriched_Robust,
    "RLDepthEnriched_Real_PLAY": composed.StraferNavCfg_RLDepthEnriched_Real_PLAY,
    "RLDepthEnriched_Robust_PLAY": composed.StraferNavCfg_RLDepthEnriched_Robust_PLAY,
    "RLDepthSubgoalEnriched_Real": composed.StraferNavCfg_RLDepthSubgoalEnriched_Real,
    "RLDepthSubgoalEnriched_Robust": composed.StraferNavCfg_RLDepthSubgoalEnriched_Robust,
    "RLDepthSubgoalEnriched_Real_PLAY": composed.StraferNavCfg_RLDepthSubgoalEnriched_Real_PLAY,
    "RLDepthSubgoalEnriched_Robust_PLAY": composed.StraferNavCfg_RLDepthSubgoalEnriched_Robust_PLAY,
}


# =====================================================================
# Snapshot gate: composed RL variant matches the frozen legacy contract
# =====================================================================


# Every RL variant that must reach the generator's DEFAULT path — the enriched
# IDs are exactly the complement. Derived rather than listed so a new open-top
# variant is covered by construction.
_OPEN_TOP_RL_VARIANTS = tuple(
    name for name in sorted(_COMPOSED_RL) if "Enriched" not in name
)


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


# =====================================================================
# Depth-enrichment: NOCAM palette pinned, enrichment is depth-only + obs-neutral
# =====================================================================


def _palette_sig(collection_cfg):
    """Slot name → (spawn type + XYZ/radius/height) signature of a palette.

    The composition hash covers the manager cfgs + scene scalars, NOT the
    rigid-object palette, so a palette change slips past every contract golden.
    This is the explicit pin for the invariant that the open-top variants (NOCAM
    and the un-enriched depth variants) keep the exact pre-enrichment palette,
    and no enriched palette adds a new slot.
    """
    sig = {}
    for name, cfg in sorted(collection_cfg.rigid_objects.items()):
        s = cfg.spawn
        row = {"spawn": type(s).__name__}
        for a in ("size", "radius", "height"):
            if hasattr(s, a):
                v = getattr(s, a)
                row[a] = list(v) if isinstance(v, (tuple, list)) else v
        sig[name] = row
    return sig


def test_open_top_variants_keep_pre_enrichment_palette_and_no_ceiling():
    """Every open-top variant — NOCAM/NOCAM_SUBGOAL *and* the un-enriched depth
    variants — keeps the exact pre-enrichment palette and has no ceiling entity.

    Guards both directions of the scene swap: enrichment must not leak into the
    camera-free NOCAM path, and a future ``_select_scene`` regression handing the
    enriched (2.7 m wall + ceiling) scene to a *vanilla* depth variant — which
    would pass every contract golden (the hash excludes the scene) and the NOCAM
    palette check — is caught here."""
    for name in _OPEN_TOP_RL_VARIANTS:
        scene = _COMPOSED_RL[name]().scene
        got = _hash(_palette_sig(scene.room_primitives))
        assert got == _PALETTE_GOLDEN, (
            f"{name} palette diverged from the frozen pre-enrichment palette"
        )
        assert not hasattr(scene, "ceiling"), (
            f"{name} is an open-top variant but got an enriched (ceiling) scene"
        )


def test_open_top_variants_drive_the_generator_at_its_defaults():
    """The open-top variants must reach ``generate_proc_room`` with no argument
    but the collection name, and at the pinned 7/7 difficulty.

    The contract hash renders a callable as its qualified name, so it sees the
    *presence* of the event term but not which arguments it carries; this is
    what pins that the vanilla path is the generator's default path. The
    ProcRoom bridge/capture cfg is included because it consumes the same vanilla
    event term and is not part of ``_COMPOSED_RL`` — after the enriched retrain
    retires the open-top depth IDs, NOCAM and that bridge variant are the two
    remaining vanilla consumers."""
    cfgs = [(name, _COMPOSED_RL[name]()) for name in _OPEN_TOP_RL_VARIANTS]
    cfgs.append(
        ("BridgeAutonomy_ProcRoom", composed.StraferNavCfg_BridgeAutonomy_ProcRoom())
    )
    for name, cfg in cfgs:
        events = cfg.events
        assert set(events.generate_room.params) == {"collection_name"}, (
            f"{name} passes non-default generator arguments: "
            f"{sorted(set(events.generate_room.params) - {'collection_name'})}"
        )
        assert events.randomize_difficulty.params == {"min_level": 7, "max_level": 7}, (
            f"{name} no longer pins difficulty 7/7"
        )


def test_enriched_palette_adds_no_new_slot():
    """The enriched depth palette has the SAME slot-name set as pre-enrichment —
    only the walls and the tall-furniture/clutter slots grow taller. No new
    collection slot is ever added (the ceiling is a standalone entity outside the
    collection), so nothing new can be active on NOCAM."""
    from strafer_lab.tasks.navigation.mdp.proc_room import (
        build_proc_room_collection_cfg,
    )
    from strafer_lab.tasks.navigation.strafer_env_cfg import (
        _ENRICH_TALL_OBJECT_HEIGHTS,
    )

    base_names = set(build_proc_room_collection_cfg().keys())
    enriched_scene = composed.StraferNavCfg_RLDepthSubgoalEnriched_Real().scene
    enriched_names = set(enriched_scene.room_primitives.rigid_objects.keys())
    assert enriched_names == base_names
    # Only the walls and the tall furniture/clutter differ, and only in Z.
    base_sig = _palette_sig(
        type(enriched_scene.room_primitives)(
            rigid_objects=build_proc_room_collection_cfg()
        )
    )
    enr_sig = _palette_sig(enriched_scene.room_primitives)
    changed = {k for k in base_sig if base_sig[k] != enr_sig[k]}
    tall_prefixes = ("furn_shelf_", "furn_cabinet_", "clutter_tall_cyl_")
    expected = {k for k in base_names if k.startswith("wall_")} | {
        k for k in base_names if k.startswith(tall_prefixes)
    }
    assert changed == expected
    # Geometry half of the palette/pose lockstep (the generator poses these at
    # height/2, pinned in the CPU enrichment suite).
    assert enr_sig["furn_shelf_0"]["size"][2] == _ENRICH_TALL_OBJECT_HEIGHTS["shelf"]
    assert enr_sig["furn_cabinet_0"]["size"][2] == _ENRICH_TALL_OBJECT_HEIGHTS["cabinet"]
    assert enr_sig["clutter_tall_cyl_0"]["height"] == _ENRICH_TALL_OBJECT_HEIGHTS["tall_cyl"]
    for k in changed:
        z_base = base_sig[k]["size"][2] if "size" in base_sig[k] else base_sig[k]["height"]
        z_enr = enr_sig[k]["size"][2] if "size" in enr_sig[k] else enr_sig[k]["height"]
        assert z_enr > z_base
    # The ceiling is NOT part of the collection.
    assert "ceiling" not in enriched_names


def test_enriched_depth_obs_matches_open_top_twin():
    """Enrichment changes the world, not the observation: each enriched depth
    variant emits the byte-identical depth observation of its open-top twin at
    the same realism, so the deployed inference package / any depth checkpoint
    stays valid (the realistic twin's obs is itself the frozen depth golden)."""
    pairs = [
        ("RLDepthEnriched_Real", composed.StraferNavCfg_RLDepth_Real),
        ("RLDepthEnriched_Robust", composed.StraferNavCfg_RLDepth_Robust),
        ("RLDepthSubgoalEnriched_Real", composed.StraferNavCfg_RLDepthSubgoal_Real),
        ("RLDepthSubgoalEnriched_Robust", composed.StraferNavCfg_RLDepthSubgoal_Robust),
    ]
    for enr_name, twin_cls in pairs:
        enr = _hash(_canon(_COMPOSED_RL[enr_name]().observations))
        twin = _hash(_canon(twin_cls().observations))
        assert enr == twin, f"{enr_name} depth obs != its open-top twin"


def test_enriched_events_differ_from_open_top():
    """Sanity: the enriched variant's event contract is actually different from
    its open-top twin (guards against the enrichment silently no-op'ing)."""
    enriched = _hash(_canon(composed.StraferNavCfg_RLDepthSubgoalEnriched_Real().events))
    open_top = _hash(_canon(composed.StraferNavCfg_RLDepthSubgoal_Real().events))
    assert enriched != open_top


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


def test_subgoal_lookahead_band_centered_on_shared_constant():
    """The realistic subgoal variant randomizes the lookahead over a tight
    band centered on the shared deployment lookahead distance — both tiers
    range (per the DR-tier convention), realistic just tighter than robust."""
    from strafer_shared.constants import SUBGOAL_LOOKAHEAD_M

    cmd = composed.StraferNavCfg_RLNoCamSubgoal_Real().commands.goal_command
    assert cmd.lookahead_m == SUBGOAL_LOOKAHEAD_M
    band = cmd.lookahead_randomization_m
    assert band is not None
    lo, hi = band
    assert lo < SUBGOAL_LOOKAHEAD_M < hi


def test_subgoal_robust_lookahead_band_wider_than_realistic():
    """Robust widens the lookahead band relative to realistic so the policy is
    not brittle to the deployed selector's exact lookahead distance."""
    real = composed.StraferNavCfg_RLNoCamSubgoal_Real().commands.goal_command
    robust = composed.StraferNavCfg_RLNoCamSubgoal_Robust().commands.goal_command
    r_lo, r_hi = real.lookahead_randomization_m
    b_lo, b_hi = robust.lookahead_randomization_m
    assert b_lo < r_lo and b_hi > r_hi  # robust strictly wider on both sides
    assert b_lo < robust.lookahead_m < b_hi


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


def _reward_term_names(rewards_cfg):
    return {
        k for k, v in rewards_cfg.__dict__.items()
        if not k.startswith("_") and hasattr(v, "func")
    }


def test_depth_subgoal_composes_depth_obs_with_subgoal_task():
    """The depth×subgoal corner: the depth observation composed with the
    subgoal command / termination blocks. The observation is byte-identical to
    depth-direct's (only the command behind the goal-shaped fields differs)."""
    cfg = composed.StraferNavCfg_RLDepthSubgoal_Real()
    assert type(cfg.commands).__name__ == "CommandsCfg_ProcRoom_Subgoal"
    assert type(cfg.terminations).__name__ == "TerminationsCfg_ProcRoom_Subgoal"
    assert "depth_image" in _obs_term_names(cfg.observations)
    depth_direct = composed.StraferNavCfg_RLDepth_Real()
    assert _hash(_canon(cfg.observations)) == _hash(_canon(depth_direct.observations))


def test_depth_subgoal_reward_adds_depth_penalty_additively():
    """Additive, no double-counting: the depth-subgoal reward set is the
    NoCam-subgoal set plus exactly one depth-sensed term. Every base term is
    preserved; the depth term is the only addition; and the NoCam-subgoal
    reward does NOT gain it (the running two-arm ablation stays clean)."""
    depth_terms = _reward_term_names(composed.StraferNavCfg_RLDepthSubgoal_Real().rewards)
    base_terms = _reward_term_names(composed.StraferNavCfg_RLNoCamSubgoal_Real().rewards)
    assert base_terms < depth_terms  # strict superset
    assert depth_terms - base_terms == {"depth_obstacle_proximity"}
    assert "depth_obstacle_proximity" not in base_terms


def test_depth_subgoal_reads_depth_camera_for_the_penalty():
    """The depth-sensed penalty needs the policy depth camera present with its
    distance channel — the composition supplies it (the ProcRoom depth scene)."""
    cfg = composed.StraferNavCfg_RLDepthSubgoal_Real()
    cam = getattr(cfg.scene, "d555_camera", None)
    assert cam is not None
    assert "distance_to_image_plane" in list(cam.data_types)
    term = cfg.rewards.depth_obstacle_proximity
    assert term.params["sensor_cfg"].name == "d555_camera"


def test_depth_subgoal_penalty_weight_is_negative_and_params_pinned():
    """Pin the sign and numeric params independently of the contract golden.

    The term ships **inert** (weight 0.0) — the current ProcRoom env starves the
    penalty, so DEPTH_SUBGOAL delivers the validated depth-*tracking* win with
    the penalty off, re-enabled later in a hardened env. The pinned invariant is
    therefore ``weight <= 0`` (a *positive* weight would reward driving toward
    obstacles — the one thing a re-enable must never do). The params must stay
    well-ordered so the re-enable is a pure one-float flip. The golden alone
    can't catch a flipped sign — it was snapshotted from this code — so this is
    the independent guard (cf. the geometric ``obstacle_proximity.weight`` pin
    in ``test_sim/rewards``)."""
    term = composed.StraferNavCfg_RLDepthSubgoal_Real().rewards.depth_obstacle_proximity
    assert term.weight <= 0.0, "depth obstacle penalty must never reward obstacles"
    p = term.params
    assert 0.0 < p["saturation_depth"] < p["distance_threshold"] <= p["max_depth"]
    assert p["epsilon"] > 0.0
    # Floor-plane exclusion must be active with a tight (render-tolerance)
    # margin: without it the always-visible floor saturates the term into an
    # ambient per-step tax that makes early termination return-optimal.
    assert 0.0 < p["floor_margin"] <= 0.15
    # Dense-stream discipline: bound the saturated cap to the same order as
    # the along-track income (weight 10 at ~0.5 m/s ~= 0.17 raw/step), not
    # several times it. Near-contact saturation legitimately exceeds income —
    # that is what forces retreat — but a cap that dwarfs the task signal
    # recreates the ambient-tax economics where terminating early beats
    # completing the path. Cap-raw here is
    # |weight| * (1/(saturation+eps) - 1/(threshold+eps)).
    cap_raw = abs(term.weight) * (
        1.0 / (p["saturation_depth"] + p["epsilon"])
        - 1.0 / (p["distance_threshold"] + p["epsilon"])
    )
    assert cap_raw < 0.5, (
        f"saturated depth penalty {cap_raw:.3f}/step dwarfs the task signal — "
        "re-run the episode-return arithmetic before raising the weight"
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


def test_bridge_procroom_expanded_stack():
    """The ProcRoom bridge variant renders the expanded stack (perception
    RGB+depth + policy depth) over a procedural room."""
    cfg = composed.StraferNavCfg_BridgeAutonomy_ProcRoom()
    per = cfg.scene.d555_camera_perception
    assert "rgb" in list(per.data_types) and "distance_to_image_plane" in list(per.data_types)
    pol = cfg.scene.d555_camera
    assert pol is not None and "distance_to_image_plane" in list(pol.data_types)
    assert hasattr(cfg.scene, "room_primitives")


def test_capture_variant_overrides_stack_per_session():
    """The stack is a per-session choice: overriding cameras_required on the
    axis cfg changes what the env renders."""
    cfg = composed.StraferNavCfg_TeleopCapture()
    cfg.sensors = composed.SensorStackCfg(cameras_required=("rgb_full", "depth_full"))
    cfg.__post_init__()
    per = cfg.scene.d555_camera_perception
    assert tuple(sorted(per.data_types)) == ("distance_to_image_plane", "rgb")
