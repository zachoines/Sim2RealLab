"""Config, launch, and entry-point tests for the subgoal-generator node.

Mirrors test_inference_config.py. Runs under colcon (``make test-ros``)
where ament + rclpy are available; the rolling-subgoal selection itself is
covered rclpy-free in test_generator.py.
"""

import importlib
import importlib.util
import os

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory


@pytest.fixture
def pkg_dir():
    return get_package_share_directory("strafer_inference")


@pytest.fixture
def params(pkg_dir):
    path = os.path.join(pkg_dir, "config", "subgoal_generator.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def node_params(params):
    return params["strafer_subgoal_generator"]["ros__parameters"]


class TestSubgoalGeneratorParamsStructure:
    REQUIRED_KEYS = [
        "plan_topic",
        "subgoal_topic",
        "map_frame",
        "base_frame",
        "update_period_s",
        "max_path_points",
    ]

    def test_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "subgoal_generator.yaml")
        assert os.path.isfile(path)

    def test_top_level_node_key(self, params):
        assert "strafer_subgoal_generator" in params
        assert "ros__parameters" in params["strafer_subgoal_generator"]

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_key_present(self, node_params, key):
        assert key in node_params, f"Missing parameter key: {key}"

    def test_plan_and_subgoal_topics_absolute(self, node_params):
        for key in ("plan_topic", "subgoal_topic"):
            value = node_params[key]
            assert isinstance(value, str) and value.startswith("/"), (
                f"{key}={value!r} must be absolute so the node namespace "
                "does not silently remap it."
            )

    def test_subgoal_topic_is_not_the_goal_topic(self, node_params):
        # A rolling subgoal must not reuse /strafer/goal -- that would trip
        # the inference node's mid-mission hidden-state reset every tick.
        assert node_params["subgoal_topic"] != "/strafer/goal"

    def test_update_period_positive_and_sane(self, node_params):
        period = float(node_params["update_period_s"])
        assert 0.0 < period <= 1.0

    def test_lookahead_not_hardcoded_in_yaml(self, node_params):
        # Omitted on purpose: the node defaults lookahead_m to
        # SUBGOAL_LOOKAHEAD_M so the parity surface cannot drift via YAML.
        assert "lookahead_m" not in node_params


class TestLaunchFile:
    def test_launch_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "subgoal_generator.launch.py")
        assert os.path.isfile(path)

    def test_launch_generates_description(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "subgoal_generator.launch.py")
        spec = importlib.util.spec_from_file_location("subgoal_launch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()
        assert ld is not None
        assert len(ld.entities) >= 3

    def test_launch_points_at_config_default(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "subgoal_generator.launch.py")
        spec = importlib.util.spec_from_file_location("subgoal_launch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()
        config_args = [
            e for e in ld.entities if getattr(e, "name", None) == "config_file"
        ]
        assert config_args, "config_file launch argument missing"
        default = config_args[0].default_value
        rendered = "".join(getattr(s, "text", str(s)) for s in default)
        assert rendered.endswith(
            os.path.join("strafer_inference", "config", "subgoal_generator.yaml")
        )


class TestEntryPoint:
    def test_module_importable(self):
        mod = importlib.import_module("strafer_inference.subgoal_generator_node")
        assert hasattr(mod, "main") and callable(mod.main)

    def test_node_class_present(self):
        mod = importlib.import_module("strafer_inference.subgoal_generator_node")
        assert hasattr(mod, "SubgoalGeneratorNode")

    def test_console_script_installed(self):
        from ament_index_python.packages import get_package_prefix

        prefix = get_package_prefix("strafer_inference")
        script = os.path.join(
            prefix, "lib", "strafer_inference", "subgoal_generator_node"
        )
        assert os.path.isfile(script), (
            f"console_script wrapper missing at {script}; the entry_points "
            "entry in setup.py is not installed."
        )


class TestUpdatePeriodDerivation:
    """The node's default update period must come from strafer_shared so a
    training-rate change can't silently leave the generator at the old rate.
    """

    def test_default_matches_shared_constants_product(self):
        from strafer_inference.subgoal_generator_node import _default_update_period
        from strafer_shared.constants import POLICY_DECIMATION, POLICY_SIM_DT

        assert _default_update_period() == POLICY_SIM_DT * POLICY_DECIMATION
