"""Pins the RealSense filter and AE values perception.launch.py passes to
rs_launch.py, and checks the installed wrapper still declares each name.
No camera needed.
"""

import importlib.util
import os

import pytest
from ament_index_python.packages import get_package_share_directory

PKG_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCH_PATH = os.path.join(PKG_SRC, "launch", "perception.launch.py")

# Exactly what the deploy depth path requires of the camera node.
PINNED = {
    "decimation_filter.enable": "false",
    "spatial_filter.enable": "false",
    "temporal_filter.enable": "false",
    "hole_filling_filter.enable": "false",
    "depth_module.enable_auto_exposure": "true",
}


class _CaptureInclude:
    """Stand-in for IncludeLaunchDescription that records the raw
    ``launch_arguments`` the launch file passes, without resolving the
    included rs_launch.py.
    """

    last = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _CaptureInclude.last = self


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rs_launch_args(monkeypatch):
    # Loaded from the source tree, not share/: the test runner does not
    # rebuild strafer_perception, so the installed copy can be stale.
    mod = _load_module(LAUNCH_PATH, "perception_launch")
    monkeypatch.setattr(mod, "IncludeLaunchDescription", _CaptureInclude)
    mod.generate_launch_description()
    return dict(_CaptureInclude.last.kwargs["launch_arguments"])


class TestRealsenseFilterContract:

    @pytest.mark.parametrize("name,value", sorted(PINNED.items()))
    def test_launch_requests_pinned_value(self, rs_launch_args, name, value):
        assert rs_launch_args.get(name) == value, (
            f"perception.launch.py must pass {name}={value!r} to rs_launch.py "
            f"explicitly; got {rs_launch_args.get(name)!r}"
        )

    def test_no_params_file_or_config_file_override(self, rs_launch_args):
        # rs_launch.py applies a config_file after the launch arguments, so
        # it would override the pins above. Adopting one means replacing this
        # with a check that the file sets no PINNED key.
        assert "config_file" not in rs_launch_args
        assert "params_file" not in rs_launch_args

    @pytest.mark.parametrize("name", sorted(PINNED))
    def test_installed_wrapper_declares_pinned_name(self, name):
        # rs_launch.py builds the node's parameters from its own
        # configurable_parameters list; a name missing from it is warned
        # about ("Parameter ... is not supported") and never reaches the
        # node, so a wrapper bump that renamed one would make the pin inert.
        rs_dir = get_package_share_directory("realsense2_camera")
        rs = _load_module(
            os.path.join(rs_dir, "launch", "rs_launch.py"), "rs_launch"
        )
        declared = {p["name"] for p in rs.configurable_parameters}
        assert name in declared, (
            f"installed rs_launch.py no longer declares {name!r}; the value "
            "perception.launch.py pins for it would be silently dropped"
        )
