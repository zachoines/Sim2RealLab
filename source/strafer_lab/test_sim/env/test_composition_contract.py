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

Hermetic: the variant sweeps here construct every ``StraferNavCfg_*``, including
the Infinigen-source capture variants, which read the transient scene corpus at
construction time. A stand-in corpus is wired in below so nothing depends on the
machine having generated one.

Usage:
    cd source/strafer_lab
    isaaclab -p -m pytest test_sim/env/test_composition_contract.py -v
"""

import hashlib
import inspect
import json

import pytest

import strafer_lab.tasks.navigation.composed_env_cfg as composed
import strafer_lab.tasks.navigation.strafer_env_cfg as env_cfgs
from strafer_lab.tasks.navigation import mdp
from test_sim.common.scenes import stub_scene_corpus


# =====================================================================
# Hermeticity: a stand-in scene corpus, never the machine's
# =====================================================================


@pytest.fixture(scope="module", autouse=True)
def _hermetic_scene_corpus(tmp_path_factory):
    """Bind a stand-in scene corpus for every test in this module.

    Autouse and module-wide on purpose: the leak is structural — *any*
    construction of an Infinigen-source variant reads ``SCENE_USD_DIR``, and
    ``_mount_dr_rows`` sweeps the module for every variant there is. Opting in
    per test would re-open the hole the moment a variant is added.

    Inert for the plane / procroom sources, which never reach ``SCENE_USD_DIR``:
    the frozen goldens below recompute byte-identically under it.
    """
    with stub_scene_corpus(tmp_path_factory) as root:
        yield root


# =====================================================================
# Canonical contract serializer (mirrors the one that produced the goldens)
# =====================================================================

_CONTRACT_FIELDS = (
    "seed", "decimation", "episode_length_s",
    "observations", "actions", "events",
    "commands", "rewards", "terminations", "curriculum",
)
_SIM_FIELDS = ("dt", "render_interval")


def _canon(obj, drop=frozenset()):
    """Canonical rendering of a cfg tree. ``drop`` omits attributes and term
    parameters by name — a term parameter is as much a named field of the
    contract as a cfg attribute is, and subtracting a new field's name before
    re-freezing is how a hash movement is shown to be that field's presence
    alone rather than a reorder or a rescale."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return repr(obj)
    if callable(obj) and hasattr(obj, "__qualname__"):
        return f"<fn {getattr(obj, '__module__', '?')}.{obj.__qualname__}>"
    if isinstance(obj, dict):
        return {
            str(k): _canon(obj[k], drop)
            for k in sorted(obj, key=str)
            if k not in drop
        }
    if isinstance(obj, (list, tuple)):
        return [_canon(v, drop) for v in obj]
    d = getattr(obj, "__dict__", None)
    if d is not None:
        return ["@" + type(obj).__name__] + [
            [k, _canon(v, drop)]
            for k, v in d.items()
            if not k.startswith("_") and k not in drop
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
#
# A randomization change moves every hash here, because the snapshot walks the
# noise models, the event terms and the action term. That is correct — the
# training distribution did change — and it is why the layout golden below
# exists: it is the half a deployed checkpoint depends on, and it must not move.
_CONTRACT_GOLDENS = {
    "RLDepth_Real": "bf325fb81189d208984a0095fd782ea8122e090228a5023ff162ee4c1d3fea7a",
    "RLDepth_Robust": "3147ff16fa6863a7b410822cd3b6c8b6d5da6e4c8789e618899fc2ec056f0c6b",
    "RLNoCam": "63fe18688cb6ec0974118bc0ac4b4e8d9ba8e6a9249e5e08be9ade18aa4f82c7",
    "RLDepth_Real_PLAY": "ab290222e2b1fe08666a7ecc04f0f9aaa945bf1494f78fa8147300877a15050d",
    "RLDepth_Robust_PLAY": "13793d71f8441a9e38d4c1ad6e09c3c56afa6c819b1600c897ef0da618c6f497",
    "RLNoCam_PLAY": "6c46e340dc7cb508f59c3a122e971222afa8179219ba05cd007debc27b17fae3",
    "RLNoCamSubgoal_Real": "43cb7039c0f3600545f1c3774cea755e43acf1c5828957eb1c816a4a247fe6bf",
    "RLNoCamSubgoal_Robust": "d1bfab170702a8200316cabf667255e47c439515d8612031760fa13f154f1559",
    "RLNoCamSubgoal_Real_PLAY": "b0db740e62ed32c9f69fb2e2d7c368ab952210d14b7d446440503e81c9de7eb3",
    "RLNoCamSubgoal_Robust_PLAY": "e862b8e24dd863b3ebf6fbf607529602d76a4bd5a7aaf8c342eb8ae4d26445e1",
    "RLDepthSubgoal_Real": "bd147fe57ae82e1000bb4ecc02c31bc4aba267598438f6da3ddc4bbd1a875220",
    "RLDepthSubgoal_Robust": "51a6b578a08d9d086d0a0dd74a65f6c2e0ae8244e6144806546cc70acaf95114",
    "RLDepthSubgoal_Real_PLAY": "1b22bfef0b8f72436b598fd6eb79b7044f328b6e172bcfda998281f6d6f8fe9b",
    "RLDepthSubgoal_Robust_PLAY": "69e1b847fcba6d4cd9f5c0728100e9668295a5d80947b97cc67d406869b1c226",
    # Depth-enrichment variants — NEW IDs frozen at creation (no prior checkpoint
    # depends on them). The enrichment lives in the `events` field (un-pinned
    # difficulty, enriched generation params incl. the tall-object heights, and
    # the rendered-camera mount offset); the observation contract is
    # byte-identical to the open-top depth variants (asserted separately).
    "RLDepthEnriched_Real": "0f2e7ec333ef296443ca2a35ae7a2a5b20cb3527ebea9dfcb8ed3a0de73e773b",
    "RLDepthEnriched_Robust": "dcf440028fda1258e5694348f1c4bc464df50f1fe6c6f6a9a5dbb77d02943e83",
    "RLDepthEnriched_Real_PLAY": "14997ac72fb1721ad9ac351463ccd63b3f15746af73e6ee3c3bd36de224db22d",
    "RLDepthEnriched_Robust_PLAY": "291e1c2e71109910a0cc9a9694b04f543bae3f84f33924d4a9cfab133b5ab884",
    "RLDepthSubgoalEnriched_Real": "4e70517667e75e3b9679fa1b45ab5eea3dde0d54d6be9fcb1b7588a91766c9b0",
    "RLDepthSubgoalEnriched_Robust": "ca9de0f434fd4e82f98683c178b85a5fdc66838d57913c4cd67eec7606d84d50",
    "RLDepthSubgoalEnriched_Real_PLAY": "034b8e2081b326c8b45972cb630c69d117082fd1a71fab2c94f58ac5ef2699ba",
    "RLDepthSubgoalEnriched_Robust_PLAY": "d515524d2a643e6650a60d0033582b2caa461d032e493ec531b5cd0820e172b4",
}

# Frozen signature (slot name set + spawn sizes) of the pre-enrichment 44-object
# palette. The composition hash cannot see the rigid-object palette (it is on the
# scene, not a hashed manager), so this pins the NOCAM promise: NOCAM keeps this
# exact palette, and the enriched palette adds no NEW slot (only taller walls).
_PALETTE_GOLDEN = "cf3499bf212a50c571b1bda4980fbbf16c7ad743dc40c5e88b7b528de3d41832"

# The depth observation a checkpoint consumes — captured identical across the
# (now dropped) plane / Infinigen / ProcRoom depth variants, which justified
# dropping them: any depth checkpoint stays valid regardless of which produced it.
# This walks both observation groups, so a privileged-group change moves it; the
# policy half is pinned separately by the layout golden below.
_DEPTH_OBS_GOLDEN = "bf6ad701fa80524226ce279a54a085857c78d0d28a9d4918bb68ce3738bdb9ab"

# The policy tensor's *layout* — terms, order, params and scales, with every
# noise model dropped. This is the half of the observation contract a deployed
# checkpoint cannot survive a change to: a reordered term or a rescaled dim
# feeds the network a different quantity, while a re-ranged sensor corruption
# feeds it the same quantity drawn differently. There are exactly two, one per
# observation profile, and they are tier-invariant by construction — realism
# selects the noise, never the layout.
_POLICY_OBS_LAYOUT_GOLDENS = {
    "depth": "ccb6f617eae1dff4887191fe807e9605786c159451747295f525768ba3a9d20c",
    "nocam": "26b1ece3f2001ccee496e784ca4737d1ad1d2be7afce33264d3f9dbd64fbe20d",
}

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


# Derived rather than listed so a new open-top variant is covered by
# construction; the enriched IDs are exactly the complement.
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


def _policy_layout(cfg):
    """Hash of the policy tensor's layout, blind to every noise model."""
    return _hash(_canon(cfg.observations.policy, drop=frozenset({"noise"})))


def _obs_profile(cfg):
    return "depth" if hasattr(cfg.observations.policy, "depth_image") else "nocam"


@pytest.mark.parametrize("name", sorted(_COMPOSED_RL), ids=lambda n: n)
def test_policy_obs_layout_matches_its_profile_golden(name):
    """The tensor the policy consumes — terms, order, params, scales — is one
    of two layouts, and neither has moved.

    Separating this from the full contract hash above is what makes a
    randomization change readable. Re-ranging a sensor's noise moves every
    contract golden, correctly: the training distribution changed. It must not
    move this one, because a deployed checkpoint is fed the same quantities in
    the same order either way. A failure here is the serious kind."""
    cfg = _COMPOSED_RL[name]()
    profile = _obs_profile(cfg)
    assert _policy_layout(cfg) == _POLICY_OBS_LAYOUT_GOLDENS[profile], (
        f"{name} changed the {profile} policy observation layout — a trained "
        f"checkpoint would be fed a different tensor, not the same tensor "
        f"corrupted differently"
    )


def test_the_obs_layout_is_the_same_at_every_realism_tier():
    """Realism selects the sensor corruption, never the layout. Stated as its
    own assertion so a tier that quietly gained or dropped a term fails here
    rather than surviving as a fourth layout nobody compares."""
    layouts = {}
    for name in sorted(_COMPOSED_RL):
        cfg = _COMPOSED_RL[name]()
        layouts.setdefault(_obs_profile(cfg), set()).add(_policy_layout(cfg))
    assert set(layouts) == set(_POLICY_OBS_LAYOUT_GOLDENS)
    for profile, hashes in layouts.items():
        assert len(hashes) == 1, f"{profile} has {len(hashes)} layouts across tiers"


# =====================================================================
# Referent-frame drift: who reads the drifted referent and who reads truth
# =====================================================================

# Discovered, not listed: a tier class added without a variant to compose it
# would otherwise reach training un-gated.
_OBS_CFG_CLASSES = {
    name: getattr(env_cfgs, name)
    for name in dir(env_cfgs)
    if name.startswith("ObsCfg_")
}

def _is_drift_capable(func) -> bool:
    """Whether a term function can read the referent through the drifted frame.

    Discovered from the signature rather than from a name list: a name list
    only guards the terms that existed when it was written, and the failure a
    missed term produces — a value function fitted against a displaced referent
    — is silent.
    """
    if not callable(func):
        return False
    try:
        return "perceived" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _obs_groups(observations):
    """``(group_name, group_cfg)`` for every observation group on a cfg."""
    return [
        (name, group)
        for name, group in vars(observations).items()
        if not name.startswith("_") and hasattr(group, "__dict__")
    ]


def _obs_terms(group):
    """``(term_name, term_cfg)`` for every observation term in a group."""
    return [
        (name, term)
        for name, term in vars(group).items()
        if not name.startswith("_") and hasattr(term, "func") and hasattr(term, "params")
    ]


def _drift_switch_violations(observations, label):
    """Every group/term pair whose drift switch is set wrong.

    The rule is structural: exactly one group — ``policy`` — reads the drifted
    referent, and it does so by *not* pinning the switch, because the policy
    group's params are the layout a deployed checkpoint is hashed against.
    Every other group is privileged and must opt out explicitly, since the
    drift lives inside the term and the observation manager's
    ``enable_corruption`` switch therefore cannot strip it.
    """
    bad = []
    for group_name, group in _obs_groups(observations):
        for term_name, term in _obs_terms(group):
            if not _is_drift_capable(term.func):
                continue
            pinned = term.params.get("perceived")
            if group_name == "policy":
                if "perceived" in term.params:
                    bad.append(
                        f"{label}.policy.{term_name} pins the drift switch in "
                        f"the layout the deployed checkpoint is hashed against"
                    )
            elif pinned is not False:
                bad.append(
                    f"{label}.{group_name}.{term_name} would read the drifted "
                    f"referent (perceived={pinned!r})"
                )
    return bad


@pytest.mark.parametrize("name", sorted(_COMPOSED_RL), ids=lambda n: n)
def test_the_privileged_group_reads_the_true_referent(name):
    """Realism corrupts what the policy senses, never what the critic knows.

    Losing an opt-out would quietly fit the value function against a displaced
    referent, which is the one thing the asymmetric critic exists to avoid."""
    cfg = _COMPOSED_RL[name]()
    assert _drift_switch_violations(cfg.observations, name) == []


@pytest.mark.parametrize(
    "name", sorted(_OBS_CFG_CLASSES), ids=lambda n: n
)
def test_every_observation_tier_sets_the_drift_switch(name):
    """The same rule at the source, over every observation cfg class rather
    than only the ones a variant composes today.

    A new tier or a new goal-shaped term reaches this gate before it reaches a
    gym ID, and a term function that gains the drift without gaining an opt-out
    on the privileged groups fails here rather than in a training curve."""
    assert _drift_switch_violations(_OBS_CFG_CLASSES[name](), name) == []


def test_the_drift_switch_guard_can_see_a_violation():
    """The guard above is worth nothing if it cannot fail, and both of its
    halves are easy to break silently — a params dict that never mentions the
    switch reads as "fine" to a `.get`, and a signature check that mis-resolves
    reads every term as incapable."""
    cfg = composed.StraferNavCfg_RLNoCam()
    assert _is_drift_capable(mdp.goal_position_relative)
    assert not _is_drift_capable(mdp.body_velocity_xy)

    del cfg.observations.critic.goal_distance.params["perceived"]
    assert _drift_switch_violations(cfg.observations, "mutated") != []


def test_only_the_randomized_tiers_carry_referent_drift():
    """The drift is a realism-tier mechanism, present on every randomized RL
    variant and absent from the ideal tier — an ideal env stays structurally
    identical to a tree without this mechanism rather than carrying an inert
    term."""
    for name in sorted(_COMPOSED_RL):
        events = _COMPOSED_RL[name]().events
        assert getattr(events, "randomize_subgoal_drift", None) is not None, name
    for factory in (composed.StraferNavCfg_NoCam_Ideal, composed.StraferNavCfg_Depth_Ideal):
        assert not hasattr(factory().events, "randomize_subgoal_drift")


@pytest.mark.parametrize(
    "factory",
    [
        composed.StraferNavCfg_TeleopCapture,
        composed.StraferNavCfg_BridgeAutonomy,
        composed.StraferNavCfg_BridgeAutonomy_ProcRoom,
        composed.StraferNavCfg_BridgeAutonomy_ProcRoomEnriched,
        composed.StraferNavCfg_Coverage,
    ],
    ids=lambda f: f.__name__,
)
def test_the_capture_variants_record_the_true_referent(factory):
    """A capture or bridge variant's observation is a record, not a policy
    input. The bridge hands its scene to a deploy stack that reads its own
    drifting TF, so a sim drift on top would double-count it and would corrupt
    the sim-vs-deploy observation parity the dump exists to measure."""
    assert factory().events.randomize_subgoal_drift is None


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
    """The open-top variants reach ``generate_proc_room`` with no argument but
    the collection name, at the pinned 7/7 difficulty.

    The contract hash renders a callable as its qualified name, so it sees the
    event term's presence but not which arguments it carries. The ProcRoom
    bridge cfg is checked here too: it consumes the same vanilla event term and
    is not in ``_COMPOSED_RL``."""
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
            f"{name} does not pin difficulty 7/7"
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


# =====================================================================
# Composition: the two halves of the D555 mount DR stay paired
# =====================================================================


def _terms_calling(events_cfg, func):
    """Names of the event terms that call ``func``, so a renamed field cannot
    slip past the check."""
    return [
        name
        for name, term in vars(events_cfg).items()
        if term is not None and getattr(term, "func", None) is func
    ]


def _mount_dr_rows():
    """One row per composed variant: what it samples, jitters and renders.

    Swept from the module rather than a list of names — a new variant is only
    covered if nothing has to be added here for it.
    """
    rows = []
    for name in sorted(n for n in dir(composed) if n.startswith("StraferNavCfg_")):
        cfg = getattr(composed, name)()
        jitter = _terms_calling(cfg.events, mdp.jitter_d555_camera_prim_pose)
        assert len(jitter) <= 1, f"{name} carries the camera jitter twice: {jitter}"
        rows.append({
            "name": name,
            "cfg": cfg,
            "offset": _terms_calling(cfg.events, mdp.randomize_d555_mount_offset),
            "jitter": jitter,
        })
    return rows


def test_the_camera_jitter_never_appears_without_its_offset():
    """The jitter consumes the realization the offset leaves behind, and points
    the camera prim it names, so alone it is either inert or a lie."""
    rows = _mount_dr_rows()
    assert len(rows) > 20, "the variant sweep collapsed"
    jittered = [r for r in rows if r["jitter"]]
    assert jittered, "no variant points the rendered camera through the mount"
    for row in jittered:
        assert row["offset"], (
            f"{row['name']} jitters the camera prim with nothing sampling the mount"
        )
        term = vars(row["cfg"].events)[row["jitter"][0]]
        sensor = term.params["sensor_name"]
        assert getattr(row["cfg"].scene, sensor, None) is not None, (
            f"{row['name']} jitters {sensor}, which it does not render"
        )


def test_the_camera_jitter_rides_exactly_the_enriched_arm():
    """Among the variants that both sample the mount and render the camera the
    jitter targets, carrying it is the same question as being enriched.

    Stated as an equivalence on purpose: a new enriched tier that samples the
    mount and leaves the render at nominal fails here, and so does a jitter
    added off the enriched arm — where a nominal-mount TF tree would then
    disagree with the render.
    """
    rows = _mount_dr_rows()
    sensors = {
        vars(r["cfg"].events)[r["jitter"][0]].params["sensor_name"]
        for r in rows
        if r["jitter"]
    }
    assert sensors, "no variant points the rendered camera through the mount"
    assert len(sensors) == 1, f"the jitter targets more than one sensor: {sensors}"
    sensor = sensors.pop()

    checked = 0
    for row in rows:
        cfg = row["cfg"]
        if not row["offset"] or getattr(cfg.scene, sensor, None) is None:
            continue
        checked += 1
        assert bool(row["jitter"]) is bool(cfg.scene_source.enrich_depth), (
            f"{row['name']}: enrich_depth={cfg.scene_source.enrich_depth} but "
            f"jitter={row['jitter']}"
        )
    assert checked > 10, "the paired-arm sweep matched too few variants"
