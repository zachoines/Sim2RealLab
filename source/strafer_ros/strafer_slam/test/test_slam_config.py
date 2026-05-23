"""Tests for strafer_slam configuration and launch files."""

import os
import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from strafer_shared.constants import DEPTH_MIN, DEPTH_MAX, MAP_RESOLUTION


@pytest.fixture
def pkg_dir():
    return get_package_share_directory("strafer_slam")


class TestRtabmapParams:
    """Validate rtabmap_params.yaml is well-formed and contains key settings."""

    @pytest.fixture
    def params(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "rtabmap_params.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "rtabmap_params.yaml")
        assert os.path.isfile(path)

    def test_detection_rate_is_set(self, params):
        """Detection rate should be ≤ 5 Hz for Jetson."""
        rate = params["Rtabmap/DetectionRate"]
        assert 0 < float(rate) <= 5

    def test_slam_2d_enabled(self, params):
        """Ground robot — 2D SLAM should be enforced."""
        assert params["Optimizer/Slam2D"] == "true"
        assert params["Reg/Force3DoF"] == "true"

    def test_grid_cell_size(self, params):
        cell = float(params["Grid/CellSize"])
        assert 0.01 <= cell <= 0.10

    def test_grid_range_matches_constants(self, params):
        """Grid range defaults must match constants.py."""
        assert float(params["Grid/RangeMin"]) == DEPTH_MIN
        assert float(params["Grid/RangeMax"]) == DEPTH_MAX

    def test_grid_cell_size_matches_constants(self, params):
        assert float(params["Grid/CellSize"]) == MAP_RESOLUTION

    def test_feature_type_is_orb(self, params):
        """ORB is fastest on Jetson (Type 8)."""
        ftype = params["Vis/FeatureType"]
        assert str(ftype) == "8"

    def test_ray_tracing_enabled(self, params):
        """Grid/RayTracing must be true so the depth-based occupancy
        grid marks cells along each ray as free, not just the obstacle
        endpoint. Without it, the angular spacing between depth pixels
        at 5-6m range exceeds Grid/CellSize and the resulting static
        map alternates known/unknown in stripes — Nav2's planner then
        threads a wavy path through the gaps, which the controller
        chases laterally.
        """
        assert params["Grid/RayTracing"] == "true"


class TestPointcloudToLaserscanParams:
    """Validate pointcloud_to_laserscan.yaml.

    The Z-axis ground filter is the central change vs the previous
    depthimage_to_laserscan setup — it kills the phantom 3.5 m arc that
    came from per-column min depth catching floor pixels.
    """

    @pytest.fixture
    def params(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "pointcloud_to_laserscan.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "pointcloud_to_laserscan.yaml")
        assert os.path.isfile(path)

    def test_target_frame_is_base_link(self, params):
        """Cloud must be transformed to base_link so the Z filter is in
        the chassis frame (true vertical), not the camera optical frame
        (where Z is forward).
        """
        frame = params["pointcloud_to_laserscan"]["ros__parameters"]["target_frame"]
        assert frame == "base_link"

    def test_height_filter_excludes_floor(self, params):
        """min_height must be > 0 so the floor in base_link Z=0 is filtered.
        That's the whole point of the swap.
        """
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["min_height"]) > 0.0

    def test_height_filter_within_body(self, params):
        """max_height should cap at roughly the chassis body height so
        ceiling fixtures and people heads don't end up in /scan as
        obstacles the robot won't collide with.
        """
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["max_height"]) > float(p["min_height"])
        assert float(p["max_height"]) <= 0.5

    def test_range_min_less_than_max(self, params):
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert p["range_min"] < p["range_max"]

    def test_range_matches_constants(self, params):
        """YAML defaults must stay in sync with constants.py."""
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["range_min"]) == DEPTH_MIN
        assert float(p["range_max"]) == DEPTH_MAX

    def test_angle_range_symmetric_about_zero(self, params):
        """The forward-facing D555 produces a cone of returns symmetric
        about the camera's optical axis (= base_link X after the static
        TF). The scan sweep should be symmetric around 0 rad.
        """
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["angle_min"]) == -float(p["angle_max"])

    def test_angle_increment_positive(self, params):
        p = params["pointcloud_to_laserscan"]["ros__parameters"]
        assert float(p["angle_increment"]) > 0.0


class TestLaunchFile:
    """Validate slam.launch.py exists and is importable."""

    def test_launch_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "slam.launch.py")
        assert os.path.isfile(path)

    def test_launch_generates_description(self, pkg_dir):
        """Import and call generate_launch_description to catch import errors."""
        import importlib.util

        path = os.path.join(pkg_dir, "launch", "slam.launch.py")
        spec = importlib.util.spec_from_file_location("slam_launch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()
        assert ld is not None
        # 4 DeclareLaunchArguments + 1 OpaqueFunction
        assert len(ld.entities) >= 5
