"""Tests for strafer_navigation configuration and launch files."""

import os

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory

from unittest.mock import patch

from strafer_shared.constants import (
    CHASSIS_LENGTH,
    DEPTH_MAX,
    DEPTH_MIN,
    MAP_RESOLUTION,
    MAX_ANGULAR_VEL,
    MAX_LINEAR_VEL,
    NAV_ANGULAR_VEL,
    NAV_LINEAR_VEL,
    NAV_REVERSE_SCALE,
    NAV_REVERSE_VEL,
    TRACK_WIDTH,
)


@pytest.fixture
def pkg_dir():
    return get_package_share_directory("strafer_navigation")


@pytest.fixture
def params(pkg_dir):
    path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# YAML structure tests
# =============================================================================


class TestNav2ParamsStructure:
    """Verify nav2_params.yaml has all required top-level nodes."""

    REQUIRED_NODES = [
        "bt_navigator",
        "controller_server",
        "planner_server",
        "smoother_server",
        "behavior_server",
        "local_costmap",
        "global_costmap",
        "waypoint_follower",
        "velocity_smoother",
    ]

    def test_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        assert os.path.isfile(path)

    @pytest.mark.parametrize("node", REQUIRED_NODES)
    def test_node_present(self, params, node):
        assert node in params, f"Missing top-level key: {node}"


# =============================================================================
# Controller (MPPI) tests
# =============================================================================


class TestMPPIController:
    """Validate MPPI controller configuration."""

    @pytest.fixture
    def mppi(self, params):
        return params["controller_server"]["ros__parameters"]["FollowPath"]

    def test_plugin_is_mppi(self, mppi):
        assert mppi["plugin"] == "nav2_mppi_controller::MPPIController"

    def test_motion_model_is_omni(self, mppi):
        """Mecanum robot must use Omni motion model."""
        assert mppi["motion_model"] == "Omni"

    def test_velocity_limits_positive(self, mppi):
        assert mppi["vx_max"] > 0
        assert mppi["vy_max"] > 0
        assert mppi["wz_max"] > 0

    def test_velocity_limits_within_hardware(self, mppi):
        """YAML defaults should not exceed hardware limits."""
        assert mppi["vx_max"] <= MAX_LINEAR_VEL
        assert mppi["vy_max"] <= MAX_LINEAR_VEL
        assert mppi["wz_max"] <= MAX_ANGULAR_VEL

    def test_has_twirling_critic(self, mppi):
        """Omni robots need TwirlingCritic to penalize unnecessary spinning."""
        assert "TwirlingCritic" in mppi["critics"]

    def test_real_robot_critic_baselines(self, mppi):
        """Pin the real-robot MPPI critic baselines. PathAlign and
        PreferForward are tuned for lossy mecanum strafe — too high
        a PathAlign weight makes MPPI command lateral vy to converge
        on curved paths and the chassis strafes; PreferForward biases
        the sampling distribution toward forward motion to compensate.
        """
        assert mppi["PathAlignCritic"]["cost_weight"] == 8.0
        assert mppi["PreferForwardCritic"]["cost_weight"] == 6.0


# =============================================================================
# Costmap tests
# =============================================================================


class TestCostmaps:
    """Validate local and global costmap configuration."""

    @pytest.fixture
    def local_cm(self, params):
        return params["local_costmap"]["local_costmap"]["ros__parameters"]

    @pytest.fixture
    def global_cm(self, params):
        return params["global_costmap"]["global_costmap"]["ros__parameters"]

    def test_local_resolution_matches(self, local_cm):
        assert float(local_cm["resolution"]) == MAP_RESOLUTION

    def test_global_resolution_matches(self, global_cm):
        assert float(global_cm["resolution"]) == MAP_RESOLUTION

    def test_local_has_obstacle_and_inflation(self, local_cm):
        plugins = local_cm["plugins"]
        assert "obstacle_layer" in plugins
        assert "inflation_layer" in plugins

    def test_global_has_static_obstacle_inflation(self, global_cm):
        plugins = global_cm["plugins"]
        assert "static_layer" in plugins
        assert "obstacle_layer" in plugins
        assert "inflation_layer" in plugins

    def test_global_map_topic_is_rtabmap(self, global_cm):
        """Global static layer should subscribe to RTAB-Map's map."""
        topic = global_cm["static_layer"]["map_topic"]
        assert topic == "/rtabmap/map"

    def test_local_footprint_present(self, local_cm):
        assert "footprint" in local_cm

    def test_global_footprint_present(self, global_cm):
        assert "footprint" in global_cm

    def test_scan_range_within_depth(self, local_cm):
        scan = local_cm["obstacle_layer"]["scan"]
        assert float(scan["raytrace_min_range"]) >= DEPTH_MIN
        assert float(scan["raytrace_max_range"]) <= DEPTH_MAX


# =============================================================================
# Velocity smoother tests
# =============================================================================


class TestVelocitySmoother:
    """Validate velocity smoother configuration."""

    @pytest.fixture
    def vs(self, params):
        return params["velocity_smoother"]["ros__parameters"]

    def test_has_y_velocity(self, vs):
        """Mecanum robot must have non-zero vy limit."""
        max_vel = vs["max_velocity"]
        assert max_vel[1] > 0, "vy must be > 0 for holonomic robot"

    def test_max_vel_within_hardware(self, vs):
        max_vel = vs["max_velocity"]
        assert max_vel[0] <= MAX_LINEAR_VEL
        assert max_vel[1] <= MAX_LINEAR_VEL
        assert max_vel[2] <= MAX_ANGULAR_VEL

    def test_odom_topic(self, vs):
        assert vs["odom_topic"] == "/strafer/odom"


# =============================================================================
# Launch file tests
# =============================================================================


class TestLaunchFile:
    """Validate navigation.launch.py exists and is importable."""

    def test_launch_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
        assert os.path.isfile(path)

    def test_launch_generates_description(self, pkg_dir):
        import importlib.util

        path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
        spec = importlib.util.spec_from_file_location("nav_launch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()
        assert ld is not None
        # 1 DeclareLaunchArgument + 1 OpaqueFunction
        assert len(ld.entities) >= 2


# =============================================================================
# Constants injection test (patch function)
# =============================================================================


class TestConstantsInjection:
    """Verify _patch_params correctly overrides YAML values."""

    def test_patch_overwrites_velocities(self, pkg_dir):
        """Load YAML, patch, and verify values come from constants."""
        import importlib.util

        launch_path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
        spec = importlib.util.spec_from_file_location("nav_launch", launch_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            p = yaml.safe_load(f)

        footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)

        mod._patch_params(p, footprint, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
                          NAV_REVERSE_VEL, 1.0,
                          MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX)

        # Check MPPI controller
        ctrl = p["controller_server"]["ros__parameters"]["FollowPath"]
        assert ctrl["vx_max"] == NAV_LINEAR_VEL
        assert ctrl["vy_max"] == NAV_LINEAR_VEL
        assert ctrl["wz_max"] == NAV_ANGULAR_VEL

    def test_patch_scales_mppi_sampling_with_envelope_factor(self, pkg_dir):
        """At envelope_factor=2.0 (sim STRAFER_NAV_VEL_SCALE=1.0), MPPI
        sampling stds and prune_distance double from the YAML baseline so
        the larger velocity cap is reachable. Real-robot bringup runs with
        envelope_factor=1.0 and inherits the baseline values untouched.
        """
        import importlib.util

        launch_path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
        spec = importlib.util.spec_from_file_location("nav_launch", launch_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            baseline = yaml.safe_load(f)
        baseline_mppi = baseline["controller_server"]["ros__parameters"]["FollowPath"]
        b_vx, b_vy, b_wz = baseline_mppi["vx_std"], baseline_mppi["vy_std"], baseline_mppi["wz_std"]
        b_prune = baseline_mppi["prune_distance"]

        with open(yaml_path) as f:
            p = yaml.safe_load(f)
        footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
        mod._patch_params(p, footprint,
                          round(MAX_LINEAR_VEL, 4),
                          round(MAX_ANGULAR_VEL, 4),
                          round(MAX_LINEAR_VEL * NAV_REVERSE_SCALE, 4),
                          2.0,
                          MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX)
        ctrl = p["controller_server"]["ros__parameters"]["FollowPath"]
        assert ctrl["vx_std"] == round(b_vx * 2.0, 4)
        # vy_std is scaled by envelope_factor like the others but then
        # un-scaled back to YAML baseline inside the envelope_factor>1.0
        # block — lifted envelope is for forward+rotation, not lateral.
        assert ctrl["vy_std"] == round(b_vy, 4)
        assert ctrl["wz_std"] == round(b_wz * 2.0, 4)
        assert ctrl["prune_distance"] == round(b_prune * 2.0, 4)

        # Identity at envelope_factor=1.0 (real-robot lane).
        with open(yaml_path) as f:
            p2 = yaml.safe_load(f)
        mod._patch_params(p2, footprint, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
                          NAV_REVERSE_VEL, 1.0,
                          MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX)
        ctrl2 = p2["controller_server"]["ros__parameters"]["FollowPath"]
        assert ctrl2["vx_std"] == round(b_vx, 4)
        assert ctrl2["vy_std"] == round(b_vy, 4)
        assert ctrl2["wz_std"] == round(b_wz, 4)
        assert ctrl2["prune_distance"] == round(b_prune, 4)

        # Check costmap resolution
        lc = p["local_costmap"]["local_costmap"]["ros__parameters"]
        assert lc["resolution"] == MAP_RESOLUTION
        assert lc["footprint"] == footprint

        # Check scan ranges
        scan = lc["obstacle_layer"]["scan"]
        assert scan["raytrace_min_range"] == DEPTH_MIN
        assert scan["raytrace_max_range"] == DEPTH_MAX

    def test_patch_rebalances_mppi_critics_when_envelope_lifted(self, pkg_dir):
        """At envelope_factor>1.0 (sim), MPPI critic weights and gamma
        shift to absolute sim values; PathFollow look-ahead lifts above
        the real baseline so high-speed rollouts win on cost.

        Real-robot bringup (envelope_factor=1.0) inherits the YAML
        baselines untouched — that's the byte-identical guarantee the
        TestConstantsInjection.test_patch_overwrites_velocities check
        anchors.
        """
        import importlib.util

        launch_path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
        spec = importlib.util.spec_from_file_location("nav_launch", launch_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            baseline = yaml.safe_load(f)
        baseline_mppi = baseline["controller_server"]["ros__parameters"]["FollowPath"]
        b_path_align_w = baseline_mppi["PathAlignCritic"]["cost_weight"]
        b_prefer_fwd_w = baseline_mppi["PreferForwardCritic"]["cost_weight"]
        b_path_follow_offset = baseline_mppi["PathFollowCritic"]["offset_from_furthest"]

        # Sim lane: critic rebalance applied.
        with open(yaml_path) as f:
            p = yaml.safe_load(f)
        footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
        mod._patch_params(p, footprint,
                          round(MAX_LINEAR_VEL, 4),
                          round(MAX_ANGULAR_VEL, 4),
                          round(MAX_LINEAR_VEL * NAV_REVERSE_SCALE, 4),
                          2.0,
                          MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX)
        ctrl = p["controller_server"]["ros__parameters"]["FollowPath"]
        # Sim values are absolute (not derived from baselines).
        assert ctrl["PathAlignCritic"]["cost_weight"] == 9.0
        assert ctrl["PreferForwardCritic"]["cost_weight"] == 10.0
        assert ctrl["PathFollowCritic"]["offset_from_furthest"] == 20
        assert ctrl["gamma"] == 0.008
        # PathFollow look-ahead lifts above the baseline; the others are
        # tuned independently per lane and may sit either side of real.
        assert (
            ctrl["PathFollowCritic"]["offset_from_furthest"]
            > b_path_follow_offset
        )
        assert ctrl["gamma"] < baseline_mppi["gamma"]

        # Real-robot lane: critics untouched at envelope_factor=1.0.
        with open(yaml_path) as f:
            p2 = yaml.safe_load(f)
        mod._patch_params(p2, footprint, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
                          NAV_REVERSE_VEL, 1.0,
                          MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX)
        ctrl2 = p2["controller_server"]["ros__parameters"]["FollowPath"]
        assert ctrl2["PathAlignCritic"]["cost_weight"] == b_path_align_w
        assert ctrl2["PreferForwardCritic"]["cost_weight"] == b_prefer_fwd_w
        assert (
            ctrl2["PathFollowCritic"]["offset_from_furthest"]
            == b_path_follow_offset
        )

    def test_planner_is_smac_2d_with_soft_unknown_prefer(self, pkg_dir):
        """SmacPlanner2D with cost_travel_multiplier > 1 and
        allow_unknown True is what gives us "prefer known-free without
        refusing unknown" — NavfnPlanner's binary allow_unknown was
        too brittle to small unknown gaps in the costmap.
        """
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            baseline = yaml.safe_load(f)
        planner = baseline["planner_server"]["ros__parameters"]["GridBased"]
        assert planner["plugin"] == "nav2_smac_planner/SmacPlanner2D"
        assert planner["allow_unknown"] is True
        assert planner["cost_travel_multiplier"] > 1.0

    def test_patch_critic_overrides_skipped_just_below_unity_envelope(self, pkg_dir):
        """The critic rebalance is gated strictly on envelope_factor>1.0.
        A scale of 1.0 (real-robot) and any sub-unity factor leave the
        baselines in place. Anchors the gate so future changes don't drift
        the threshold and silently re-tune real-robot bringup.
        """
        import importlib.util

        launch_path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
        spec = importlib.util.spec_from_file_location("nav_launch", launch_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            baseline = yaml.safe_load(f)
        b_mppi = baseline["controller_server"]["ros__parameters"]["FollowPath"]
        b_path_align_w = b_mppi["PathAlignCritic"]["cost_weight"]
        b_prefer_fwd_w = b_mppi["PreferForwardCritic"]["cost_weight"]
        b_path_follow_offset = b_mppi["PathFollowCritic"]["offset_from_furthest"]

        footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
        for factor in (1.0, 0.75):
            with open(yaml_path) as f:
                p = yaml.safe_load(f)
            mod._patch_params(
                p, footprint, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
                NAV_REVERSE_VEL, factor,
                MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX,
            )
            ctrl = p["controller_server"]["ros__parameters"]["FollowPath"]
            assert ctrl["PathAlignCritic"]["cost_weight"] == b_path_align_w
            assert ctrl["PreferForwardCritic"]["cost_weight"] == b_prefer_fwd_w
            assert (
                ctrl["PathFollowCritic"]["offset_from_furthest"]
                == b_path_follow_offset
            )
            assert ctrl["gamma"] == b_mppi["gamma"]


# =============================================================================
# Promotion split invariants (_patch_params three-section structure)
# =============================================================================
#
# `_patch_params` is split into three buckets (see
# docs/tasks/context/nav2-knob-promotion.md):
#   1. universal defaults        — applied on every lane, factor-independent
#   2. velocity-envelope coupling — scale with the factor, gated by construction
#   3. behavioral overrides       — sim-only, gated on envelope_factor > 1.0
# These tests pin which knob lives in which bucket. When a knob graduates
# (bucket 3 → 1), move it from BEHAVIORAL_OVERRIDES to UNIVERSAL here and the
# tests enforce the new contract.

# Fixed sentinel matching the byte-identical golden fixtures (see
# TestPatchByteIdentical); a machine-independent stand-in for the launch-time
# ament_index BT path so the universal BT injection is exercised.
_SENTINEL_BT = "/opt/strafer/navigate_to_pose_w_smoothing_and_recovery.xml"


def _patch_at(mod, pkg_dir, factor, nav_vel, nav_omega, nav_reverse,
              *, bt=_SENTINEL_BT):
    """Load a fresh nav2_params.yaml and patch it at ``factor``."""
    yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
    with open(yaml_path) as f:
        p = yaml.safe_load(f)
    footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
    mod._patch_params(
        p, footprint, nav_vel, nav_omega, nav_reverse, factor,
        MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX, smoothing_bt_xml_path=bt,
    )
    return p


def _ctrl(p):
    return p["controller_server"]["ros__parameters"]["FollowPath"]


# Section 1 — universal defaults. Each accessor must return the SAME value at
# every envelope_factor when the resolved velocities are held fixed (i.e. the
# factor does not gate them). Velocity caps derive from the nav_* args, not the
# factor, so holding those fixed isolates the factor's (non-)effect.
_UNIVERSAL_KNOBS = {
    "FollowPath.vx_max": lambda p: _ctrl(p)["vx_max"],
    "FollowPath.vx_min": lambda p: _ctrl(p)["vx_min"],
    "FollowPath.vy_max": lambda p: _ctrl(p)["vy_max"],
    "FollowPath.wz_max": lambda p: _ctrl(p)["wz_max"],
    "behavior_server.max_rotational_vel":
        lambda p: p["behavior_server"]["ros__parameters"]["max_rotational_vel"],
    "velocity_smoother.max_velocity":
        lambda p: tuple(p["velocity_smoother"]["ros__parameters"]["max_velocity"]),
    "velocity_smoother.min_velocity":
        lambda p: tuple(p["velocity_smoother"]["ros__parameters"]["min_velocity"]),
    "local_costmap.resolution":
        lambda p: p["local_costmap"]["local_costmap"]["ros__parameters"]["resolution"],
    "local_costmap.footprint":
        lambda p: p["local_costmap"]["local_costmap"]["ros__parameters"]["footprint"],
    "local_costmap.scan.raytrace_max_range":
        lambda p: p["local_costmap"]["local_costmap"]["ros__parameters"]["obstacle_layer"]["scan"]["raytrace_max_range"],
    "global_costmap.resolution":
        lambda p: p["global_costmap"]["global_costmap"]["ros__parameters"]["resolution"],
    "global_costmap.footprint":
        lambda p: p["global_costmap"]["global_costmap"]["ros__parameters"]["footprint"],
    "bt_navigator.default_nav_to_pose_bt_xml":
        lambda p: p["bt_navigator"]["ros__parameters"]["default_nav_to_pose_bt_xml"],
}

# Section 3 — behavioral overrides. Each applies its sim value ONLY at
# envelope_factor > 1.0; at factor <= 1.0 the YAML baseline stays. `sim` is the
# absolute value _patch_params writes on the sim lane; the baseline is read with
# the same `get` accessor from the raw (unpatched) YAML — sound only because
# section 1 never touches these keys, so a knob listed here must not also become
# a universal-section knob.
_BEHAVIORAL_OVERRIDES = {
    "PathAlignCritic.cost_weight": {
        "get": lambda p: _ctrl(p)["PathAlignCritic"]["cost_weight"],
        "sim": 9.0,
    },
    "PreferForwardCritic.cost_weight": {
        "get": lambda p: _ctrl(p)["PreferForwardCritic"]["cost_weight"],
        "sim": 10.0,
    },
    "PathFollowCritic.offset_from_furthest": {
        "get": lambda p: _ctrl(p)["PathFollowCritic"]["offset_from_furthest"],
        "sim": 20,
    },
    "gamma": {
        "get": lambda p: _ctrl(p)["gamma"],
        "sim": 0.008,
    },
}


class TestPromotionSplitInvariants:
    """Pin which _patch_params section each MPPI knob lives in."""

    @pytest.fixture
    def baseline(self, pkg_dir):
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    # ── Section 1: universal defaults apply at every factor ──────────────
    @pytest.mark.parametrize("knob", list(_UNIVERSAL_KNOBS))
    def test_universal_knob_is_factor_independent(self, pkg_dir, knob):
        """A universal-section knob resolves to the same value at every
        envelope_factor when the nav_* velocities are held fixed — the factor
        does not gate it."""
        mod = _load_launch_module(pkg_dir)
        get = _UNIVERSAL_KNOBS[knob]
        values = {
            factor: get(_patch_at(
                mod, pkg_dir, factor,
                NAV_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_REVERSE_VEL))
            for factor in (0.75, 1.0, 2.0)
        }
        assert len(set(values.values())) == 1, (
            f"universal knob {knob} varies with envelope_factor: {values}"
        )

    # ── Section 3: behavioral overrides gated strictly on factor > 1.0 ───
    @pytest.mark.parametrize("knob", list(_BEHAVIORAL_OVERRIDES))
    def test_behavioral_override_applies_only_above_unity(
        self, pkg_dir, baseline, knob
    ):
        """A behavioral-override knob takes its sim value at every factor > 1.0
        (including just above unity) and the YAML baseline at factor <= 1.0 —
        the sim→real gap this brief closes lives in this bucket, and the gate is
        pinned at exactly 1.0 on both sides."""
        mod = _load_launch_module(pkg_dir)
        spec = _BEHAVIORAL_OVERRIDES[knob]
        base_val = spec["get"](baseline)

        # Must actually differ from baseline (else the gate is a no-op and the
        # knob is mislabeled).
        assert spec["sim"] != base_val, (
            f"{knob} sim value equals baseline {base_val}; gate is vacuous"
        )
        # Sim value at the production factor AND just above unity, so a gate
        # threshold drifting up into (1.0, 2.0) is caught.
        for factor in (1.0001, 2.0):
            p = _patch_at(mod, pkg_dir, factor,
                          round(MAX_LINEAR_VEL, 4), round(MAX_ANGULAR_VEL, 4),
                          round(MAX_LINEAR_VEL * NAV_REVERSE_SCALE, 4))
            assert spec["get"](p) == spec["sim"], (
                f"{knob} not at sim value at envelope_factor={factor}; the "
                "> 1.0 gate must fire for every factor above unity"
            )

        # Real-robot lane and sub-unity: baseline untouched.
        for factor in (1.0, 0.75):
            p = _patch_at(mod, pkg_dir, factor,
                          NAV_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_REVERSE_VEL)
            assert spec["get"](p) == base_val, (
                f"{knob} changed at envelope_factor={factor}; must stay baseline"
            )

    # ── Section 2: velocity-envelope coupling tracks the factor ──────────
    def test_velocity_envelope_scaling_tracks_factor(self, pkg_dir, baseline):
        """vx_std / wz_std / prune_distance scale linearly with the factor at
        every factor; vy_std scales too but is un-scaled back to baseline once
        the factor exceeds unity (lateral is not part of the lifted envelope)."""
        mod = _load_launch_module(pkg_dir)
        b = baseline["controller_server"]["ros__parameters"]["FollowPath"]
        for factor in (0.75, 1.0, 2.0):
            ctrl = _ctrl(_patch_at(mod, pkg_dir, factor,
                                   NAV_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_REVERSE_VEL))
            assert ctrl["vx_std"] == round(b["vx_std"] * factor, 4)
            assert ctrl["wz_std"] == round(b["wz_std"] * factor, 4)
            assert ctrl["prune_distance"] == round(b["prune_distance"] * factor, 4)
            expected_vy = b["vy_std"] if factor > 1.0 else round(b["vy_std"] * factor, 4)
            assert ctrl["vy_std"] == expected_vy

    def test_real_robot_lane_leaves_mppi_tuning_at_yaml_baseline(
        self, pkg_dir, baseline
    ):
        """The real-robot lane (envelope_factor == 1.0) must leave every
        section-2 and section-3 MPPI knob at its YAML baseline — this is the
        no-silent-sim-to-real-gap invariant. The byte-identical golden pin
        (TestPatchByteIdentical) is the exhaustive version; this names it."""
        mod = _load_launch_module(pkg_dir)
        b = baseline["controller_server"]["ros__parameters"]["FollowPath"]
        ctrl = _ctrl(_patch_at(mod, pkg_dir, 1.0,
                               NAV_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_REVERSE_VEL))
        for key in ("vx_std", "vy_std", "wz_std", "prune_distance", "gamma"):
            assert ctrl[key] == b[key], f"{key} drifted from baseline at factor 1.0"
        assert ctrl["PathAlignCritic"]["cost_weight"] == b["PathAlignCritic"]["cost_weight"]
        assert ctrl["PreferForwardCritic"]["cost_weight"] == b["PreferForwardCritic"]["cost_weight"]
        assert ctrl["PathFollowCritic"]["offset_from_furthest"] == b["PathFollowCritic"]["offset_from_furthest"]


class TestPatchByteIdentical:
    """Golden pin: the restructured _patch_params reproduces the pre-refactor
    output byte-for-byte at both lanes.

    The fixtures in ``test/fixtures/`` were generated from the pre-refactor
    _patch_params; comparing against them proves the three-section refactor is
    pure motion — it moved and relabeled statements without altering a single
    emitted parameter. Regenerate only on an intentional param change (a
    promotion moving a knob, or a value retune), and call that out in review.
    """

    FIX_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

    def _dump(self, p):
        return yaml.dump(p, default_flow_style=False, sort_keys=False)

    def test_factor_1_0_matches_golden(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        p = _patch_at(mod, pkg_dir, 1.0,
                      NAV_LINEAR_VEL, NAV_ANGULAR_VEL, NAV_REVERSE_VEL)
        with open(os.path.join(self.FIX_DIR, "patched_params_factor_1.0.yaml")) as f:
            golden = f.read()
        assert self._dump(p) == golden, (
            "real-robot (envelope_factor=1.0) patched params drifted from the "
            "pre-refactor golden — the refactor is no longer pure motion"
        )

    def test_factor_2_0_matches_golden(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        p = _patch_at(mod, pkg_dir, 2.0,
                      round(MAX_LINEAR_VEL, 4), round(MAX_ANGULAR_VEL, 4),
                      round(MAX_LINEAR_VEL * NAV_REVERSE_SCALE, 4))
        with open(os.path.join(self.FIX_DIR, "patched_params_factor_2.0.yaml")) as f:
            golden = f.read()
        assert self._dump(p) == golden, (
            "sim (envelope_factor=2.0) patched params drifted from the "
            "pre-refactor golden — the refactor is no longer pure motion"
        )

    def test_goldens_differ_on_gated_knobs(self):
        """Guard the pin against vacuity: the sim lane must differ from the real
        lane specifically on the four gated behavioral knobs — not merely on the
        velocity envelope — else the golden wouldn't catch a gate collapse."""
        with open(os.path.join(self.FIX_DIR, "patched_params_factor_1.0.yaml")) as f:
            c1 = _ctrl(yaml.safe_load(f))
        with open(os.path.join(self.FIX_DIR, "patched_params_factor_2.0.yaml")) as f:
            c2 = _ctrl(yaml.safe_load(f))
        assert c1["gamma"] != c2["gamma"]
        assert c1["PreferForwardCritic"]["cost_weight"] != c2["PreferForwardCritic"]["cost_weight"]
        assert c1["PathAlignCritic"]["cost_weight"] != c2["PathAlignCritic"]["cost_weight"]
        assert (
            c1["PathFollowCritic"]["offset_from_furthest"]
            != c2["PathFollowCritic"]["offset_from_furthest"]
        )


# =============================================================================
# Custom BT (event-driven replan + SmoothPath, universal)
# =============================================================================


_SMOOTHING_BT_FILENAME = "navigate_to_pose_w_smoothing_and_recovery.xml"


class TestSmoothingBT:
    """Structural invariants of the custom navigate-to-pose BT."""

    def test_bt_file_shipped(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        assert os.path.isfile(path), (
            f"Custom BT XML missing from share dir: {path}"
        )

    def test_bt_contains_smoothpath_with_simple_smoother(self, pkg_dir):
        """The BT must call SmoothPath with smoother_id="simple_smoother"
        and bind both unsmoothed and smoothed ports to the {path}
        blackboard variable so FollowPath consumes the smoothed result.
        """
        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        with open(path) as f:
            xml = f.read()
        assert "<SmoothPath" in xml, "Missing <SmoothPath> node"
        assert 'smoother_id="simple_smoother"' in xml, (
            'SmoothPath must reference smoother_id="simple_smoother" '
            "(the plugin instance configured under smoother_server)"
        )
        assert 'unsmoothed_path="{path}"' in xml
        assert 'smoothed_path="{path}"' in xml

    def test_bt_replans_only_when_invalidated(self, pkg_dir):
        """Replanning fires only via <Fallback name="ReplanIfNeeded">
        gated on <IsPathValid> + inverted <GlobalUpdatedGoal>, never
        on a time- or distance-decorator.

        The XML name is "GlobalUpdatedGoal" (single "Global"); that
        is Nav2's registered alias for the class
        GloballyUpdatedGoalCondition. "GloballyUpdatedGoal" silently
        fails to resolve and bt_navigator rejects every goal.
        """
        import xml.etree.ElementTree as ET

        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        tree = ET.parse(path)
        root = tree.getroot()
        tags = {el.tag for el in root.iter()}
        assert "DistanceController" not in tags
        assert "RateController" not in tags
        assert "GloballyUpdatedGoal" not in tags, (
            "Nav2 registers the node as 'GlobalUpdatedGoal' (single "
            "'Global'); the longer 'GloballyUpdatedGoal' silently "
            "fails to resolve, bt_navigator's loadBehaviorTree "
            "returns false, and every action goal is rejected."
        )
        replan_fallback = next(
            (
                el for el in root.iter()
                if el.tag == "Fallback" and el.get("name") == "ReplanIfNeeded"
            ),
            None,
        )
        assert replan_fallback is not None
        gate_tags = [child.tag for child in replan_fallback]
        assert gate_tags[0] == "Sequence", (
            "First child of ReplanIfNeeded must be the path-validity "
            "Sequence; the planner sits after it."
        )
        gate_seq = replan_fallback[0]
        inner_tags = {el.tag for el in gate_seq.iter()}
        assert "IsPathValid" in inner_tags
        assert "GlobalUpdatedGoal" in inner_tags, (
            "Use <GlobalUpdatedGoal/>, not <GoalUpdated/>: the latter "
            "resets on BT halt and misses goal changes between "
            "back-to-back navigate_to_pose action calls."
        )
        assert "GoalUpdated" not in inner_tags, (
            "<GoalUpdated/> in the path-validity gate is the bug — "
            "use <GlobalUpdatedGoal/> instead."
        )
        assert "Inverter" in inner_tags, (
            "Inverter must wrap GlobalUpdatedGoal; un-inverted it "
            "short-circuits the fallback into skipping the replan."
        )

    def test_bt_does_not_warmup_per_goal(self, pkg_dir):
        """The donut-coverage warmup runs once per launch from
        strafer_bringup.donut_warmup, NOT once per Nav2 goal. Nav2's
        BtActionServer halts and re-ticks the tree between goals in a
        way that doesn't reliably preserve BT decorator state — so a
        BT-internal SingleTrigger fires on every goal in practice, not
        once per session. Keep the warmup out of this BT to avoid that
        trap.
        """
        import xml.etree.ElementTree as ET

        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        tree = ET.parse(path)
        root = tree.getroot()
        tags = {el.tag for el in root.iter()}
        assert "SingleTrigger" not in tags, (
            "Per-launch warmup must live in strafer_bringup, not in the "
            "navigate-to-pose BT — BT decorator state isn't preserved "
            "across goal halts so SingleTrigger fires every goal"
        )
        # The recovery branch's <Spin spin_dist="1.57"/> is still there
        # by design (Nav2 stock recovery); only the top-of-tree warmup
        # spin is forbidden.
        spin_attrs = [
            el.get("spin_dist") for el in root.iter() if el.tag == "Spin"
        ]
        assert "6.28" not in spin_attrs, (
            "Full 6.28 rad warmup spin must not appear in this BT"
        )

    def test_bt_keeps_planner_and_follower(self, pkg_dir):
        """Sanity: the BT still issues ComputePathToPose and FollowPath
        — smoothing is additive, not a replacement.
        """
        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        with open(path) as f:
            xml = f.read()
        assert "<ComputePathToPose" in xml
        assert "<FollowPath" in xml

    def test_bt_is_well_formed_xml(self, pkg_dir):
        import xml.etree.ElementTree as ET

        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "root"
        assert root.get("main_tree_to_execute") == "MainTree"

    def test_patch_wires_smoothing_bt_on_every_lane(self, pkg_dir):
        """``_patch_params`` wires the smoothing BT into bt_navigator
        at every ``envelope_factor``.
        """
        import importlib.util

        launch_path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
        spec = importlib.util.spec_from_file_location("nav_launch", launch_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
        bt_path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        assert os.path.isfile(bt_path)
        for factor in (0.75, 1.0, 2.0):
            with open(yaml_path) as f:
                p = yaml.safe_load(f)
            mod._patch_params(
                p, footprint,
                round(MAX_LINEAR_VEL, 4),
                round(MAX_ANGULAR_VEL, 4),
                round(MAX_LINEAR_VEL * NAV_REVERSE_SCALE, 4),
                factor,
                MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX,
                smoothing_bt_xml_path=bt_path,
            )
            bt_nav = p["bt_navigator"]["ros__parameters"]
            assert bt_nav.get("default_nav_to_pose_bt_xml") == bt_path, (
                f"BT swap did not apply at envelope_factor={factor}: "
                f"{bt_nav.get('default_nav_to_pose_bt_xml')!r}"
            )

    def test_patch_skips_bt_swap_when_path_omitted(self, pkg_dir):
        """Omitting ``smoothing_bt_xml_path`` leaves
        ``default_nav_to_pose_bt_xml`` absent. Test seam only —
        ``generate_launch_description`` always passes the path.
        """
        import importlib.util

        launch_path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
        spec = importlib.util.spec_from_file_location("nav_launch", launch_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            p = yaml.safe_load(f)
        footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
        mod._patch_params(
            p, footprint, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
            NAV_REVERSE_VEL, 1.0,
            MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX,
        )
        bt_nav = p["bt_navigator"]["ros__parameters"]
        assert "default_nav_to_pose_bt_xml" not in bt_nav


# =============================================================================
# STRAFER_NAV_VEL_SCALE override
# =============================================================================


def _load_launch_module(pkg_dir):
    import importlib.util

    path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
    spec = importlib.util.spec_from_file_location("nav_launch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestNavVelScaleOverride:
    """``_resolved_nav_velocities`` honors STRAFER_NAV_VEL_SCALE.

    Real-robot bringup leaves the env var unset and gets the
    constants-derived defaults (NAV_VEL_SCALE = 0.5). Sim bringup
    exports STRAFER_NAV_VEL_SCALE=1.0 to lift the cap to the chassis
    max so MPPI matches the envelope the trained policy sees.
    """

    def test_unset_yields_constants_defaults(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        env = {k: v for k, v in os.environ.items() if k != "STRAFER_NAV_VEL_SCALE"}
        with patch.dict(os.environ, env, clear=True):
            linear, angular, reverse, _factor = mod._resolved_nav_velocities()
        assert linear == NAV_LINEAR_VEL
        assert angular == NAV_ANGULAR_VEL
        assert reverse == NAV_REVERSE_VEL

    def test_unity_scale_lifts_to_hardware_max(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        with patch.dict(os.environ, {"STRAFER_NAV_VEL_SCALE": "1.0"}, clear=False):
            linear, angular, reverse, _factor = mod._resolved_nav_velocities()
        assert linear == round(MAX_LINEAR_VEL, 4)
        assert angular == round(MAX_ANGULAR_VEL, 4)
        assert reverse == round(linear * NAV_REVERSE_SCALE, 4)

    def test_envelope_factor_doubles_at_unity_scale(self, pkg_dir):
        """At STRAFER_NAV_VEL_SCALE=1.0 the cap doubles vs the 0.5 baseline.

        ``_patch_params`` uses this factor to scale MPPI sampling noise
        (vx/vy/wz_std) and ``prune_distance`` so the larger envelope is
        actually exploitable; identity at envelope_factor=1.0 leaves
        real-robot tuning untouched.
        """
        mod = _load_launch_module(pkg_dir)
        with patch.dict(os.environ, {"STRAFER_NAV_VEL_SCALE": "1.0"}, clear=False):
            _l, _a, _r, factor = mod._resolved_nav_velocities()
        assert factor == 2.0

    def test_envelope_factor_one_when_unset(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        env = {k: v for k, v in os.environ.items() if k != "STRAFER_NAV_VEL_SCALE"}
        with patch.dict(os.environ, env, clear=True):
            _l, _a, _r, factor = mod._resolved_nav_velocities()
        assert factor == 1.0

    def test_arbitrary_scale_recomputes_all_three(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        with patch.dict(os.environ, {"STRAFER_NAV_VEL_SCALE": "0.75"}, clear=False):
            linear, angular, reverse, _factor = mod._resolved_nav_velocities()
        assert linear == round(MAX_LINEAR_VEL * 0.75, 4)
        assert angular == round(MAX_ANGULAR_VEL * 0.75, 4)
        assert reverse == round(linear * NAV_REVERSE_SCALE, 4)

    def test_non_numeric_falls_back_to_defaults(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        with patch.dict(os.environ, {"STRAFER_NAV_VEL_SCALE": "fast"}, clear=False):
            linear, angular, reverse, _factor = mod._resolved_nav_velocities()
        assert linear == NAV_LINEAR_VEL
        assert angular == NAV_ANGULAR_VEL
        assert reverse == NAV_REVERSE_VEL

    def test_non_positive_falls_back_to_defaults(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        with patch.dict(os.environ, {"STRAFER_NAV_VEL_SCALE": "0"}, clear=False):
            linear, angular, reverse, _factor = mod._resolved_nav_velocities()
        assert linear == NAV_LINEAR_VEL
        assert angular == NAV_ANGULAR_VEL
        assert reverse == NAV_REVERSE_VEL


class TestSimEnvFile:
    """Verify ``env_sim_in_the_loop.env`` exports the documented overrides."""

    @pytest.fixture
    def env_text(self):
        # Sourced by sim operators before bringup_sim_in_the_loop launch.
        bringup_dir = get_package_share_directory("strafer_bringup")
        path = os.path.join(bringup_dir, "config", "env_sim_in_the_loop.env")
        with open(path) as f:
            return f.read()

    def test_exports_unity_nav_vel_scale(self, env_text):
        assert "export STRAFER_NAV_VEL_SCALE=1.0" in env_text

    def test_navigation_timeout_dropped_below_legacy_600(self, env_text):
        # Capture the assigned float so the assertion explains regressions.
        for line in env_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("export STRAFER_NAVIGATION_TIMEOUT_S="):
                value = float(stripped.split("=", 1)[1])
                assert value < 600.0, (
                    f"STRAFER_NAVIGATION_TIMEOUT_S={value} — expected to drop "
                    "below 600 once the bridge unit-mismatch fix and "
                    "STRAFER_NAV_VEL_SCALE=1.0 let MPPI run at chassis max."
                )
                return
        pytest.fail("STRAFER_NAVIGATION_TIMEOUT_S export not found in env file")
