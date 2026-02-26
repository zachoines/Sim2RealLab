"""Tests for strafer_bringup launch files."""

import os

import pytest
from ament_index_python.packages import get_package_share_directory


@pytest.fixture
def pkg_dir():
    return get_package_share_directory("strafer_bringup")


# =============================================================================
# Launch file existence
# =============================================================================

class TestLaunchFilesExist:
    """Every bringup launch file must be installed."""

    LAUNCH_FILES = [
        "base.launch.py",
        "perception.launch.py",
        "slam.launch.py",
        "navigation.launch.py",
    ]

    @pytest.mark.parametrize("filename", LAUNCH_FILES)
    def test_launch_file_exists(self, pkg_dir, filename):
        path = os.path.join(pkg_dir, "launch", filename)
        assert os.path.isfile(path), f"Missing launch file: {filename}"


# =============================================================================
# Launch files generate valid descriptions
# =============================================================================

class TestLaunchDescriptions:
    """Each launch file must be importable and produce a LaunchDescription."""

    LAUNCH_FILES = [
        "base.launch.py",
        "perception.launch.py",
        "slam.launch.py",
        "navigation.launch.py",
    ]

    @pytest.mark.parametrize("filename", LAUNCH_FILES)
    def test_generates_description(self, pkg_dir, filename):
        import importlib.util

        path = os.path.join(pkg_dir, "launch", filename)
        spec = importlib.util.spec_from_file_location(filename, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()
        assert ld is not None
        assert len(ld.entities) >= 2, (
            f"{filename} should have at least 2 entities "
            f"(args + includes), got {len(ld.entities)}"
        )


# =============================================================================
# Layering — each level includes the one below it
# =============================================================================

class TestLayering:
    """Validate that launch files compose in the expected order."""

    def _get_include_paths(self, pkg_dir, filename):
        """Extract file paths from IncludeLaunchDescription sources."""
        import importlib.util
        from launch.actions import IncludeLaunchDescription

        path = os.path.join(pkg_dir, "launch", filename)
        spec = importlib.util.spec_from_file_location(filename, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()

        paths = []
        for entity in ld.entities:
            if isinstance(entity, IncludeLaunchDescription):
                src = entity.launch_description_source
                loc = src._LaunchDescriptionSource__location
                paths.append(loc[0].text)
        return paths

    def test_base_includes_driver_and_description(self, pkg_dir):
        paths = self._get_include_paths(pkg_dir, "base.launch.py")
        joined = " ".join(paths)
        assert "strafer_driver" in joined
        assert "strafer_description" in joined

    def test_perception_includes_base_and_perception(self, pkg_dir):
        paths = self._get_include_paths(pkg_dir, "perception.launch.py")
        joined = " ".join(paths)
        assert "strafer_bringup" in joined or "base" in joined
        assert "strafer_perception" in joined

    def test_slam_includes_perception_and_slam(self, pkg_dir):
        paths = self._get_include_paths(pkg_dir, "slam.launch.py")
        joined = " ".join(paths)
        assert "perception" in joined
        assert "strafer_slam" in joined

    def test_navigation_includes_slam_and_nav(self, pkg_dir):
        paths = self._get_include_paths(pkg_dir, "navigation.launch.py")
        joined = " ".join(paths)
        assert "slam" in joined
        assert "strafer_navigation" in joined


# =============================================================================
# Launch argument forwarding
# =============================================================================

class TestArguments:
    """Validate that key arguments are declared and forwarded."""

    def _get_arg_names(self, pkg_dir, filename):
        """Extract declared launch argument names."""
        import importlib.util
        from launch.actions import DeclareLaunchArgument

        path = os.path.join(pkg_dir, "launch", filename)
        spec = importlib.util.spec_from_file_location(filename, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()

        return [
            entity.name
            for entity in ld.entities
            if isinstance(entity, DeclareLaunchArgument)
        ]

    def test_base_has_port_args(self, pkg_dir):
        args = self._get_arg_names(pkg_dir, "base.launch.py")
        assert "front_port" in args
        assert "rear_port" in args

    def test_slam_has_localization_arg(self, pkg_dir):
        args = self._get_arg_names(pkg_dir, "slam.launch.py")
        assert "localization" in args
        assert "database_path" in args

    def test_navigation_has_all_args(self, pkg_dir):
        args = self._get_arg_names(pkg_dir, "navigation.launch.py")
        assert "front_port" in args
        assert "localization" in args
        assert "nav_log_level" in args
