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


class TestDepthimageParams:
    """Validate depthimage_to_laserscan.yaml."""

    @pytest.fixture
    def params(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "depthimage_to_laserscan.yaml")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "depthimage_to_laserscan.yaml")
        assert os.path.isfile(path)

    def test_scan_height_positive(self, params):
        h = params["depthimage_to_laserscan"]["ros__parameters"]["scan_height"]
        assert h > 0

    def test_range_min_less_than_max(self, params):
        p = params["depthimage_to_laserscan"]["ros__parameters"]
        assert p["range_min"] < p["range_max"]

    def test_range_matches_constants(self, params):
        """YAML defaults must stay in sync with constants.py."""
        p = params["depthimage_to_laserscan"]["ros__parameters"]
        assert float(p["range_min"]) == DEPTH_MIN
        assert float(p["range_max"]) == DEPTH_MAX

    def test_output_frame(self, params):
        frame = params["depthimage_to_laserscan"]["ros__parameters"]["output_frame"]
        assert frame == "d555_link"


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
