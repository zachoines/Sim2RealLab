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
        "max_path_points",
        "replan_period_s",
        "active_goal_topic",
        "goal_telemetry_timeout_s",
        "planner_action",
        "planner_id",
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
        for key in (
            "plan_topic", "subgoal_topic", "active_goal_topic",
            "planner_action",
        ):
            value = node_params[key]
            assert isinstance(value, str) and value.startswith("/"), (
                f"{key}={value!r} must be absolute so the node namespace "
                "does not silently remap it."
            )

    def test_replan_period_below_suppression_window(self, node_params):
        # The single-node budget rule that replaced the autonomy client's
        # replan-vs-suppression warning: the plan must be refreshed faster
        # than it is allowed to go stale.
        assert (
            float(node_params["replan_period_s"])
            < float(node_params["path_timeout_s"])
        )

    def test_active_goal_topic_matches_inference_config(self, pkg_dir):
        # One telemetry channel: the generator subscribes where the
        # inference node publishes.
        path = os.path.join(pkg_dir, "config", "inference.yaml")
        with open(path) as f:
            inference = yaml.safe_load(f)
        inference_params = inference["/**"]["ros__parameters"]
        gen_path = os.path.join(pkg_dir, "config", "subgoal_generator.yaml")
        with open(gen_path) as f:
            gen = yaml.safe_load(f)
        gen_params = gen["strafer_subgoal_generator"]["ros__parameters"]
        assert (
            gen_params["active_goal_topic"]
            == inference_params["active_goal_topic"]
        )

    def test_rate_and_lookahead_not_hardcoded_in_yaml(self, node_params):
        # Both omitted on purpose: the node defaults update_period_s to
        # POLICY_SIM_DT * POLICY_DECIMATION and lookahead_m to
        # SUBGOAL_LOOKAHEAD_M, so neither the policy rate nor the parity
        # surface can drift via a config literal.
        assert "update_period_s" not in node_params
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


class _CaptureNode:
    """Records raw Node kwargs so a test can assert the parameters list."""

    last = None

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        _CaptureNode.last = self


def _load_launch(pkg_dir):
    path = os.path.join(pkg_dir, "launch", "subgoal_generator.launch.py")
    spec = importlib.util.spec_from_file_location("subgoal_launch_ovr", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSubgoalLaunchUseSimTime:
    """subgoal_generator.launch.py gains only a use_sim_time override; it
    stays policy-free (no model_path / policy_variant).
    """

    def test_use_sim_time_arg_declared(self, pkg_dir, monkeypatch):
        from launch.actions import DeclareLaunchArgument

        monkeypatch.delenv("STRAFER_USE_SIM_TIME", raising=False)
        ld = _load_launch(pkg_dir).generate_launch_description()
        names = [
            e.name for e in ld.entities
            if isinstance(e, DeclareLaunchArgument)
        ]
        assert "use_sim_time" in names

    def test_no_policy_args(self, pkg_dir, monkeypatch):
        from launch.actions import DeclareLaunchArgument

        ld = _load_launch(pkg_dir).generate_launch_description()
        names = [
            e.name for e in ld.entities
            if isinstance(e, DeclareLaunchArgument)
        ]
        assert "model_path" not in names, (
            "subgoal generator is policy-free; it must not gain model_path"
        )
        assert "policy_variant" not in names

    def test_use_sim_time_default_from_env(self, pkg_dir, monkeypatch):
        from launch.actions import DeclareLaunchArgument

        monkeypatch.delenv("STRAFER_USE_SIM_TIME", raising=False)
        ld = _load_launch(pkg_dir).generate_launch_description()
        for e in ld.entities:
            if isinstance(e, DeclareLaunchArgument) and e.name == "use_sim_time":
                rendered = "".join(
                    getattr(s, "text", str(s)) for s in e.default_value
                )
                assert rendered == "false"
                return
        raise AssertionError("use_sim_time arg not found")

    def test_use_sim_time_is_bool_coerced(self, pkg_dir, monkeypatch):
        from launch.substitutions import PythonExpression

        mod = _load_launch(pkg_dir)
        monkeypatch.setattr(mod, "Node", _CaptureNode)
        mod.generate_launch_description()
        override = _CaptureNode.last.kwargs["parameters"][-1]
        assert isinstance(override, dict)
        assert isinstance(override["use_sim_time"], PythonExpression)

    def test_node_stays_in_global_namespace(self, pkg_dir, monkeypatch):
        # The generator's /plan, /strafer/subgoal, and map->base_link TF are
        # absolute and unremapped, so it must stay in the global namespace
        # (no namespace=, no group wrap).
        from launch.actions import GroupAction

        mod = _load_launch(pkg_dir)
        monkeypatch.setattr(mod, "Node", _CaptureNode)
        ld = mod.generate_launch_description()
        assert _CaptureNode.last.kwargs["name"] == "strafer_subgoal_generator"
        assert _CaptureNode.last.kwargs.get("namespace") is None
        assert not any(isinstance(e, GroupAction) for e in ld.entities)


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
