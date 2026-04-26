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
                          NAV_REVERSE_VEL, MAP_RESOLUTION, DEPTH_MIN, DEPTH_MAX)

        # Check MPPI controller
        ctrl = p["controller_server"]["ros__parameters"]["FollowPath"]
        assert ctrl["vx_max"] == NAV_LINEAR_VEL
        assert ctrl["vy_max"] == NAV_LINEAR_VEL
        assert ctrl["wz_max"] == NAV_ANGULAR_VEL

        # Check costmap resolution
        lc = p["local_costmap"]["local_costmap"]["ros__parameters"]
        assert lc["resolution"] == MAP_RESOLUTION
        assert lc["footprint"] == footprint

        # Check scan ranges
        scan = lc["obstacle_layer"]["scan"]
        assert scan["raytrace_min_range"] == DEPTH_MIN
        assert scan["raytrace_max_range"] == DEPTH_MAX


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
            linear, angular, reverse = mod._resolved_nav_velocities()
        assert linear == NAV_LINEAR_VEL
        assert angular == NAV_ANGULAR_VEL
        assert reverse == NAV_REVERSE_VEL

    def test_unity_scale_lifts_to_hardware_max(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        with patch.dict(os.environ, {"STRAFER_NAV_VEL_SCALE": "1.0"}, clear=False):
            linear, angular, reverse = mod._resolved_nav_velocities()
        assert linear == round(MAX_LINEAR_VEL, 4)
        assert angular == round(MAX_ANGULAR_VEL, 4)
        assert reverse == round(linear * NAV_REVERSE_SCALE, 4)

    def test_arbitrary_scale_recomputes_all_three(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        with patch.dict(os.environ, {"STRAFER_NAV_VEL_SCALE": "0.75"}, clear=False):
            linear, angular, reverse = mod._resolved_nav_velocities()
        assert linear == round(MAX_LINEAR_VEL * 0.75, 4)
        assert angular == round(MAX_ANGULAR_VEL * 0.75, 4)
        assert reverse == round(linear * NAV_REVERSE_SCALE, 4)

    def test_non_numeric_falls_back_to_defaults(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        with patch.dict(os.environ, {"STRAFER_NAV_VEL_SCALE": "fast"}, clear=False):
            linear, angular, reverse = mod._resolved_nav_velocities()
        assert linear == NAV_LINEAR_VEL
        assert angular == NAV_ANGULAR_VEL
        assert reverse == NAV_REVERSE_VEL

    def test_non_positive_falls_back_to_defaults(self, pkg_dir):
        mod = _load_launch_module(pkg_dir)
        with patch.dict(os.environ, {"STRAFER_NAV_VEL_SCALE": "0"}, clear=False):
            linear, angular, reverse = mod._resolved_nav_velocities()
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
