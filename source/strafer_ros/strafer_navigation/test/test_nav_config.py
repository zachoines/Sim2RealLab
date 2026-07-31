"""Tests for strafer_navigation configuration and launch files."""

import os

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory

from strafer_shared.constants import (
    CHASSIS_LENGTH,
    DEPTH_MAX,
    DEPTH_MIN,
    MAP_RESOLUTION,
    MAX_ANGULAR_VEL,
    MAX_LINEAR_VEL,
    NAV_ANGULAR_VEL,
    NAV_LINEAR_VEL,
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


def _load_launch_module(pkg_dir):
    import importlib.util

    path = os.path.join(pkg_dir, "launch", "navigation.launch.py")
    spec = importlib.util.spec_from_file_location("nav_launch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    def test_universalized_mppi_tuning(self, mppi):
        """Pin the four MPPI knobs that ship as universal defaults on both
        lanes. They live in nav2_params.yaml (not resolved at launch), so
        this guards against a silent revert to the pre-parity values.
        """
        assert mppi["gamma"] == 0.008
        assert mppi["PreferForwardCritic"]["cost_weight"] == 10.0
        assert mppi["PathAlignCritic"]["cost_weight"] == 9.0
        assert mppi["PathFollowCritic"]["offset_from_furthest"] == 20


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
        # 2 DeclareLaunchArgument + 1 OpaqueFunction
        assert len(ld.entities) >= 2


# =============================================================================
# Constants injection (patch function)
# =============================================================================


class TestConstantsInjection:
    """Verify _patch_params injects constants and leaves MPPI tuning at YAML.

    Both lanes share the same Nav2 config by construction — _patch_params
    takes no lane/factor argument, layers only physical constants (velocity
    caps, costmap resolution, footprint, scan ranges) plus the launch-resolved
    BT path, and reads every behavioral value from nav2_params.yaml.
    """

    def test_patch_overwrites_velocities(self, pkg_dir):
        """Load YAML, patch, and verify caps + costmap values come from
        constants."""
        mod = _load_launch_module(pkg_dir)
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            p = yaml.safe_load(f)

        footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
        mod._patch_params(p, footprint, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
                          NAV_REVERSE_VEL,
                          MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX)

        ctrl = p["controller_server"]["ros__parameters"]["FollowPath"]
        assert ctrl["vx_max"] == NAV_LINEAR_VEL
        assert ctrl["vy_max"] == NAV_LINEAR_VEL
        assert ctrl["wz_max"] == NAV_ANGULAR_VEL
        assert ctrl["vx_min"] == -NAV_REVERSE_VEL

        lc = p["local_costmap"]["local_costmap"]["ros__parameters"]
        assert lc["resolution"] == MAP_RESOLUTION
        assert lc["footprint"] == footprint
        scan = lc["obstacle_layer"]["scan"]
        assert scan["raytrace_min_range"] == DEPTH_MIN
        assert scan["raytrace_max_range"] == DEPTH_MAX

    def test_patch_leaves_mppi_tuning_at_yaml_baseline(self, pkg_dir):
        """_patch_params injects only physical constants; the MPPI sampling
        stds and critic / convergence tuning are YAML defaults it must not
        touch. This is the both-lanes-identical guarantee the golden pin
        (TestPatchByteIdentical) enforces exhaustively; this names it.
        """
        mod = _load_launch_module(pkg_dir)
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            baseline = yaml.safe_load(f)
        b = baseline["controller_server"]["ros__parameters"]["FollowPath"]

        with open(yaml_path) as f:
            p = yaml.safe_load(f)
        footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
        mod._patch_params(p, footprint, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
                          NAV_REVERSE_VEL,
                          MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX)
        ctrl = p["controller_server"]["ros__parameters"]["FollowPath"]
        for key in ("vx_std", "vy_std", "wz_std", "prune_distance", "gamma"):
            assert ctrl[key] == b[key], f"{key} drifted from YAML baseline"
        assert (
            ctrl["PathAlignCritic"]["cost_weight"]
            == b["PathAlignCritic"]["cost_weight"]
        )
        assert (
            ctrl["PreferForwardCritic"]["cost_weight"]
            == b["PreferForwardCritic"]["cost_weight"]
        )
        assert (
            ctrl["PathFollowCritic"]["offset_from_furthest"]
            == b["PathFollowCritic"]["offset_from_furthest"]
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

    def test_relaxed_planner_is_the_inflated_start_escape_hatch(self, pkg_dir):
        """SmacPlanner2D validates its start cell and refuses a robot parked
        inside the inflation halo, which starves the rolling subgoal and stops
        the robot — so the pose never changes and the refusal never clears.

        No SmacPlanner2D parameter relaxes that check, and the >=INSCRIBED band
        is sized by the footprint's inscribed radius rather than
        inflation_radius, so no costmap knob relaxes it either short of
        shrinking the footprint. NavfnPlanner clears the robot's own cell before
        propagating, which is sound because the robot occupies it, so it is
        registered as a second escape-hatch planner — not the primary, since its
        binary allow_unknown is brittle to the small unknown gaps
        cost_travel_multiplier handles softly (see the test above).
        """
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            baseline = yaml.safe_load(f)
        server = baseline["planner_server"]["ros__parameters"]

        assert server["planner_plugins"] == ["GridBased", "GridBasedRelaxed"], (
            "GridBased must stay FIRST/primary; GridBasedRelaxed is requested "
            "by name only, by the subgoal generator's fallback_planner_id"
        )
        relaxed = server["GridBasedRelaxed"]
        assert relaxed["plugin"] == "nav2_navfn_planner/NavfnPlanner"
        assert relaxed["allow_unknown"] is True
        # Same goal tolerance as the primary: the escape hatch changes WHERE
        # planning may start from, not what counts as reaching the goal.
        assert relaxed["tolerance"] == server["GridBased"]["tolerance"]

    def test_planner_plugin_packages_are_declared_dependencies(self, pkg_dir):
        """planner_server fails to configure — taking the whole nav stack
        down — if a named plugin's library is missing, so every plugin package
        named in the YAML must be an exec_depend, not a transitive accident."""
        import xml.etree.ElementTree as ET

        # package.xml lives in the source tree, not the share dir.
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        manifest = os.path.join(here, "package.xml")
        deps = {
            e.text.strip()
            for e in ET.parse(manifest).getroot().findall("exec_depend")
        }
        assert "nav2_smac_planner" in deps
        assert "nav2_navfn_planner" in deps


# =============================================================================
# Byte-identical golden pin
# =============================================================================

# Fixed sentinel matching the byte-identical golden fixture; a
# machine-independent stand-in for the launch-time ament_index BT path so the
# universal BT injection is exercised.
_SENTINEL_BT = "/opt/strafer/navigate_to_pose_w_smoothing_and_recovery.xml"


def _patch_at(mod, pkg_dir, nav_vel, nav_omega, nav_reverse, *, bt=_SENTINEL_BT):
    """Load a fresh nav2_params.yaml and patch it with the given caps."""
    yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
    with open(yaml_path) as f:
        p = yaml.safe_load(f)
    footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
    mod._patch_params(
        p, footprint, nav_vel, nav_omega, nav_reverse,
        MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX, smoothing_bt_xml_path=bt,
    )
    return p


class TestPatchByteIdentical:
    """Golden pin: _patch_params reproduces the shipped Nav2 config
    byte-for-byte.

    Both lanes share this config by construction — _patch_params has no
    per-lane branch — so a single fixture is the whole story. It is the
    configuration the operator validated on the e2e bridge nav2 mission; the
    pin means "the shipped baseline is exactly the rig-validated one."
    Regenerate only on an intentional param change (a value retune, or a
    footprint / scan-range constant change), and call that out in review.
    """

    FIX_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

    def _dump(self, p):
        return yaml.dump(p, default_flow_style=False, sort_keys=False)

    def test_matches_golden(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        p = _patch_at(mod, pkg_dir, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
                      NAV_REVERSE_VEL)
        with open(os.path.join(self.FIX_DIR, "patched_params.yaml")) as f:
            golden = f.read()
        assert self._dump(p) == golden, (
            "patched params drifted from the rig-validated golden — an "
            "unintended Nav2 config change, or regenerate the fixture if this "
            "is a deliberate retune"
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
        assert "ComputePathToPose" not in {
            el.tag for el in replan_fallback[0].iter()
        }, (
            "The path-validity gate must not contain a planner call — "
            "replanning belongs in the branch after it."
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
        — smoothing and the start-cell fallback are additive, not
        replacements. The primary planner is still named literally, so no
        external topic can take the first attempt away from it.
        """
        import xml.etree.ElementTree as ET

        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        with open(path) as f:
            xml = f.read()
        assert "<ComputePathToPose" in xml
        assert "<FollowPath" in xml

        root = ET.parse(path).getroot()
        planners = [el for el in root.iter() if el.tag == "ComputePathToPose"]
        assert [p.get("planner_id") for p in planners][0] == "GridBased", (
            "The first ComputePathToPose must name the primary planner "
            "literally; a blackboard reference there would let the selector "
            "topic replace the primary rather than follow it."
        )

    def test_bt_is_well_formed_xml(self, pkg_dir):
        import xml.etree.ElementTree as ET

        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "root"
        assert root.get("main_tree_to_execute") == "MainTree"

    def test_patch_wires_smoothing_bt(self, pkg_dir):
        """``_patch_params`` wires the smoothing BT into bt_navigator."""
        mod = _load_launch_module(pkg_dir)
        bt_path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        assert os.path.isfile(bt_path)
        p = _patch_at(mod, pkg_dir, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
                      NAV_REVERSE_VEL, bt=bt_path)
        bt_nav = p["bt_navigator"]["ros__parameters"]
        assert bt_nav.get("default_nav_to_pose_bt_xml") == bt_path, (
            f"BT swap did not apply: {bt_nav.get('default_nav_to_pose_bt_xml')!r}"
        )

    def test_relaxed_planner_reachable_only_after_the_primary_fails(
        self, pkg_dir
    ):
        """The escape-hatch planner sits behind two gates, both required.

        Structurally it is the SECOND child of a Fallback whose first child is
        the primary planner, so it is unreachable until the primary has
        actually failed. Semantically it is named by <PlannerSelector>, whose
        default is the primary — so a failure the escape hatch cannot fix
        re-runs the primary and fails again rather than widening what the lane
        will drive.
        """
        import xml.etree.ElementTree as ET

        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        root = ET.parse(path).getroot()

        fallback = next(
            (
                el for el in root.iter()
                if el.tag == "Fallback"
                and el.get("name") == "PlanWithStartCellFallback"
            ),
            None,
        )
        assert fallback is not None, "plan-step Fallback missing"
        children = list(fallback)
        assert len(children) == 2, (
            "The plan Fallback takes exactly two branches: the primary, then "
            "the start-cell retry."
        )
        assert children[0].tag == "ComputePathToPose"
        assert children[0].get("planner_id") == "GridBased"

        retry = children[1]
        assert retry.tag == "Sequence"
        selector, relaxed = list(retry)
        assert selector.tag == "PlannerSelector", (
            "The retry branch must be gated by the planner selector, not run "
            "the escape hatch unconditionally."
        )
        assert relaxed.tag == "ComputePathToPose"
        assert relaxed.get("planner_id") == "{%s}" % selector.get(
            "selected_planner"
        ).strip("{}"), (
            "The retry's planner_id must read the blackboard key the selector "
            "writes, or the gate is bypassed."
        )

    def test_bt_hardcodes_no_relaxed_planner(self, pkg_dir):
        """The escape-hatch planner's name appears nowhere in the BT: it can
        only arrive over the selector topic, which is what makes the fallback
        gated on the cause rather than on any planner failure.
        """
        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        with open(path) as f:
            xml = f.read()
        assert "GridBasedRelaxed" not in xml

    def test_bt_selector_defaults_to_the_primary_planner(self, pkg_dir):
        """With the selector node absent or silent the BT plans exactly as it
        did without it — the default_planner literal must therefore be the
        planner_server's primary.
        """
        import xml.etree.ElementTree as ET

        path = os.path.join(pkg_dir, "config", _SMOOTHING_BT_FILENAME)
        root = ET.parse(path).getroot()
        selector = next(
            el for el in root.iter() if el.tag == "PlannerSelector"
        )
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            server = yaml.safe_load(f)["planner_server"]["ros__parameters"]
        assert selector.get("default_planner") == server["planner_plugins"][0]
        assert selector.get("topic_name") == "planner_selector"

    def test_planner_selector_bt_node_is_loaded(self, pkg_dir):
        """bt_navigator only registers the BT nodes named in plugin_lib_names;
        an unregistered tag makes loadBehaviorTree return false and every goal
        is rejected.
        """
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            params = yaml.safe_load(f)
        libs = params["bt_navigator"]["ros__parameters"]["plugin_lib_names"]
        assert "nav2_planner_selector_bt_node" in libs

    def test_patch_skips_bt_swap_when_path_omitted(self, pkg_dir):
        """Omitting ``smoothing_bt_xml_path`` leaves
        ``default_nav_to_pose_bt_xml`` absent. Test seam only —
        ``generate_launch_description`` always passes the path.
        """
        mod = _load_launch_module(pkg_dir)
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            p = yaml.safe_load(f)
        footprint = mod._build_footprint(CHASSIS_LENGTH, TRACK_WIDTH)
        mod._patch_params(
            p, footprint, NAV_LINEAR_VEL, NAV_ANGULAR_VEL,
            NAV_REVERSE_VEL,
            MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX,
        )
        bt_nav = p["bt_navigator"]["ros__parameters"]
        assert "default_nav_to_pose_bt_xml" not in bt_nav


# =============================================================================
# Start-cell planner selector (the BT's cause gate)
# =============================================================================


class TestStartCellSelectorWiring:
    """The selector's parameters are derived from planner_server, never
    restated, so it can only name a registered planner and its probe matches
    the grid the primary planner actually plans on.
    """

    def test_params_come_from_planner_server(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            params = yaml.safe_load(f)
        server = params["planner_server"]["ros__parameters"]

        derived = mod._start_cell_selector_params(params)
        assert derived["primary_planner_id"] == server["planner_plugins"][0]
        assert derived["relaxed_planner_id"] == server["planner_plugins"][1]
        assert derived["relaxed_planner_id"] in server, (
            "the selector may only name a planner planner_server registered"
        )

    def test_probe_matches_the_downsampled_planning_grid(self, pkg_dir):
        """SmacPlanner2D plans on a grid whose cells carry the MAX cost of the
        block they cover, so a probe at the costmap's own resolution
        under-reports the refusal the fallback exists for.
        """
        mod = _load_launch_module(pkg_dir)
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            params = yaml.safe_load(f)
        grid = params["planner_server"]["ros__parameters"]["GridBased"]

        derived = mod._start_cell_selector_params(params)
        assert derived["downsampling_factor"] == grid["downsampling_factor"]

    def test_downsampling_ignored_when_the_planner_does_not_downsample(
        self, pkg_dir
    ):
        mod = _load_launch_module(pkg_dir)
        yaml_path = os.path.join(pkg_dir, "config", "nav2_params.yaml")
        with open(yaml_path) as f:
            params = yaml.safe_load(f)
        params["planner_server"]["ros__parameters"]["GridBased"][
            "downsample_costmap"
        ] = False
        assert mod._start_cell_selector_params(params)["downsampling_factor"] == 1

    def test_hybrid_lane_fallback_stays_independent(self, pkg_dir):
        """The hybrid lane escapes the same wedge through the subgoal
        generator's own ``fallback_planner_id``, on its own ComputePathToPose
        requests. The two paths share only the planner's registered name — the
        generator reads no selector topic — so they cannot double-engage.
        """
        gen_dir = get_package_share_directory("strafer_inference")
        gen_path = os.path.join(gen_dir, "config", "subgoal_generator.yaml")
        with open(gen_path) as f:
            gen = yaml.safe_load(f)["strafer_subgoal_generator"][
                "ros__parameters"
            ]
        assert gen["fallback_planner_id"] == "GridBasedRelaxed"
        assert not any("selector" in k for k in gen), (
            "the generator must not gain a selector-topic input; its fallback "
            "is its own"
        )
