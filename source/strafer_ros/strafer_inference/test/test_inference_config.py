"""Tests for strafer_inference configuration, launch, and entry point."""

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
    path = os.path.join(pkg_dir, "config", "inference.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def node_params(params):
    return params["strafer_inference"]["ros__parameters"]


# =============================================================================
# YAML structure tests
# =============================================================================


class TestInferenceParamsStructure:
    """Verify inference.yaml has the parameters the node declares."""

    REQUIRED_KEYS = [
        "model_path",
        "policy_variant",
        "infer_period_s",
        "cmd_vel_topic",
        "depth_topic",
        "imu_topic",
        "joint_states_topic",
        "odom_topic",
        "tf_max_age_s",
        "obs_timeout_s",
        "depth_timeout_s",
    ]

    def test_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "config", "inference.yaml")
        assert os.path.isfile(path)

    def test_top_level_node_key(self, params):
        assert "strafer_inference" in params
        assert "ros__parameters" in params["strafer_inference"]

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_key_present(self, node_params, key):
        assert key in node_params, f"Missing parameter key: {key}"

    def test_policy_variant_is_supported(self, node_params):
        from strafer_shared.policy_interface import PolicyVariant

        variant = node_params["policy_variant"]
        assert variant in {v.name for v in PolicyVariant}, (
            f"policy_variant={variant!r} is not a PolicyVariant; "
            "expected one of NOCAM / DEPTH."
        )

    def test_default_variant_is_depth(self, node_params):
        assert node_params["policy_variant"] == "DEPTH", (
            "DEPTH is the MVP variant for strafer_direct; NOCAM in "
            "strafer_direct mode has no obstacle awareness and is unsafe."
        )

    def test_infer_period_is_positive(self, node_params):
        period = float(node_params["infer_period_s"])
        assert period > 0.0
        assert period <= 1.0, (
            "Inference period > 1 s would starve the chassis of velocity "
            "commands; the policy was trained at 30 Hz."
        )

    def test_timeouts_positive(self, node_params):
        for key in (
            "tf_max_age_s",
            "obs_timeout_s",
            "depth_timeout_s",
        ):
            assert float(node_params[key]) > 0.0, (
                f"{key} must be positive; non-positive disables the watchdog"
            )

    def test_topic_names_absolute(self, node_params):
        for key in (
            "cmd_vel_topic",
            "depth_topic",
            "imu_topic",
            "joint_states_topic",
            "odom_topic",
        ):
            value = node_params[key]
            assert isinstance(value, str) and value.startswith("/"), (
                f"{key}={value!r} must be an absolute topic so the "
                "node namespace does not silently remap it."
            )

    def test_cmd_vel_topic_matches_driver_contract(self, node_params):
        assert node_params["cmd_vel_topic"] == "/strafer/cmd_vel"

    def test_depth_topic_is_bridged_perception_stream(self, node_params):
        # The 80x60 d555_camera policy camera is sim-only; the bridged
        # stream is the 640x360 d555_camera_perception at this topic.
        assert node_params["depth_topic"] == "/d555/depth/image_rect_raw"


# =============================================================================
# Launch file tests
# =============================================================================


class TestLaunchFile:
    """Validate inference.launch.py exists and is importable."""

    def test_launch_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "inference.launch.py")
        assert os.path.isfile(path)

    def test_launch_generates_description(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "inference.launch.py")
        spec = importlib.util.spec_from_file_location("inference_launch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()
        assert ld is not None
        # config_file + log_level + model_path + policy_variant +
        # use_sim_time DeclareLaunchArgument + 1 Node action
        assert len(ld.entities) >= 6

    def test_launch_points_at_config_default(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "inference.launch.py")
        spec = importlib.util.spec_from_file_location("inference_launch", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ld = mod.generate_launch_description()
        config_args = [
            e for e in ld.entities
            if getattr(e, "name", None) == "config_file"
        ]
        assert config_args, "config_file launch argument missing"
        default = config_args[0].default_value
        rendered = "".join(getattr(s, "text", str(s)) for s in default)
        assert rendered.endswith(os.path.join(
            "strafer_inference", "config", "inference.yaml"
        ))


# =============================================================================
# Launch-arg overrides — set-once knobs that override the YAML
# =============================================================================


class _CaptureNode:
    """Stand-in for launch_ros Node that records its raw kwargs, so a
    test can assert the exact ``parameters=[...]`` list the launch file
    passes (ordering + substitution types) without the ROS graph or
    launch_ros parameter normalization.
    """

    last = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        _CaptureNode.last = self


def _load_launch(pkg_dir):
    path = os.path.join(pkg_dir, "launch", "inference.launch.py")
    spec = importlib.util.spec_from_file_location("inference_launch_ovr", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestInferenceLaunchOverrides:
    """inference.launch.py exposes model_path / policy_variant /
    use_sim_time as env-defaulted launch args that override the YAML.
    """

    OVERRIDE_ARGS = ["model_path", "policy_variant", "use_sim_time"]

    def _args(self, pkg_dir, monkeypatch):
        for var in (
            "STRAFER_INFERENCE_MODEL_PATH",
            "STRAFER_POLICY_VARIANT",
            "STRAFER_USE_SIM_TIME",
        ):
            monkeypatch.delenv(var, raising=False)
        mod = _load_launch(pkg_dir)
        ld = mod.generate_launch_description()
        from launch.actions import DeclareLaunchArgument

        return {
            e.name: e
            for e in ld.entities
            if isinstance(e, DeclareLaunchArgument)
        }

    @pytest.mark.parametrize("name", OVERRIDE_ARGS)
    def test_override_arg_declared(self, pkg_dir, monkeypatch, name):
        assert name in self._args(pkg_dir, monkeypatch)

    def test_arg_defaults_read_from_env(self, pkg_dir, monkeypatch):
        args = self._args(pkg_dir, monkeypatch)

        def rendered(name):
            default = args[name].default_value
            return "".join(getattr(s, "text", str(s)) for s in default)

        # With the env unset, the launch defaults stand in.
        assert rendered("model_path") == ""
        assert rendered("policy_variant") == "DEPTH"
        assert rendered("use_sim_time") == "false"

    def test_env_overrides_arg_default(self, pkg_dir, monkeypatch):
        monkeypatch.setenv("STRAFER_INFERENCE_MODEL_PATH", "/models/depth.onnx")
        monkeypatch.setenv("STRAFER_POLICY_VARIANT", "NOCAM_SUBGOAL")
        mod = _load_launch(pkg_dir)
        ld = mod.generate_launch_description()
        from launch.actions import DeclareLaunchArgument

        args = {
            e.name: e for e in ld.entities
            if isinstance(e, DeclareLaunchArgument)
        }

        def rendered(name):
            default = args[name].default_value
            return "".join(getattr(s, "text", str(s)) for s in default)

        assert rendered("model_path") == "/models/depth.onnx"
        assert rendered("policy_variant") == "NOCAM_SUBGOAL"

    def test_override_dict_is_last_in_parameters(self, pkg_dir, monkeypatch):
        from launch.substitutions import LaunchConfiguration

        for var in (
            "STRAFER_INFERENCE_MODEL_PATH",
            "STRAFER_POLICY_VARIANT",
            "STRAFER_USE_SIM_TIME",
        ):
            monkeypatch.delenv(var, raising=False)
        mod = _load_launch(pkg_dir)
        monkeypatch.setattr(mod, "Node", _CaptureNode)
        mod.generate_launch_description()

        params = _CaptureNode.last.kwargs["parameters"]
        # YAML config_file first, override dict last (ROS applies
        # parameters in order, last wins).
        assert isinstance(params[0], LaunchConfiguration)
        assert params[0].variable_name[0].text == "config_file"
        override = params[-1]
        assert isinstance(override, dict)
        assert set(override) == {"model_path", "policy_variant", "use_sim_time"}

    def test_model_and_variant_are_raw_launch_configurations(
        self, pkg_dir, monkeypatch
    ):
        from launch.substitutions import LaunchConfiguration

        mod = _load_launch(pkg_dir)
        monkeypatch.setattr(mod, "Node", _CaptureNode)
        mod.generate_launch_description()
        override = _CaptureNode.last.kwargs["parameters"][-1]

        # Genuine strings — passed through raw, no coercion.
        assert isinstance(override["model_path"], LaunchConfiguration)
        assert isinstance(override["policy_variant"], LaunchConfiguration)

    def test_use_sim_time_is_bool_coerced(self, pkg_dir, monkeypatch):
        from launch.substitutions import PythonExpression

        mod = _load_launch(pkg_dir)
        monkeypatch.setattr(mod, "Node", _CaptureNode)
        mod.generate_launch_description()
        override = _CaptureNode.last.kwargs["parameters"][-1]

        # use_sim_time is a bool param; a raw "false" string is
        # truthy-by-presence, so it must be a PythonExpression, never a
        # bare LaunchConfiguration.
        assert isinstance(override["use_sim_time"], PythonExpression)

    def test_node_name_and_namespace_preserved(self, pkg_dir, monkeypatch):
        # Load-bearing: the hybrid executor dispatch builds its ActionClient
        # against the absolute /strafer_inference/navigate_to_pose, so the
        # node must keep this exact name + namespace.
        mod = _load_launch(pkg_dir)
        monkeypatch.setattr(mod, "Node", _CaptureNode)
        mod.generate_launch_description()
        assert _CaptureNode.last.kwargs["name"] == "strafer_inference"
        assert _CaptureNode.last.kwargs["namespace"] == "strafer_inference"

    def test_node_not_wrapped_in_namespace_group(self, pkg_dir):
        # A GroupAction / PushRosNamespace wrap would push the action
        # server off /strafer_inference and silently break the dispatch.
        from launch.actions import GroupAction

        mod = _load_launch(pkg_dir)
        ld = mod.generate_launch_description()
        assert not any(isinstance(e, GroupAction) for e in ld.entities)


# =============================================================================
# Entry point + module surface
# =============================================================================


class TestEntryPoint:
    """The console_scripts wiring lives in setup.py; verify the module
    that backs it is importable and exposes the symbols the launch
    file (and any direct ``ros2 run`` invocation) relies on.
    """

    def test_module_importable(self):
        mod = importlib.import_module("strafer_inference.inference_node")
        assert hasattr(mod, "main"), "console_script target main() missing"
        assert callable(mod.main)

    def test_node_class_present(self):
        mod = importlib.import_module("strafer_inference.inference_node")
        assert hasattr(mod, "InferenceNode")

    def test_console_script_installed(self):
        # ament_python copies the console_scripts wrapper into
        # lib/<package>/<script>; the executable name in package.xml
        # / setup.py is what the launch file references.
        from ament_index_python.packages import get_package_prefix

        prefix = get_package_prefix("strafer_inference")
        script = os.path.join(prefix, "lib", "strafer_inference", "inference_node")
        assert os.path.isfile(script), (
            f"console_script wrapper missing at {script}; "
            "the entry_points entry in setup.py is not installed."
        )


# =============================================================================
# infer_period_s derived from strafer_shared.constants
# =============================================================================


class TestInferPeriodDerivation:
    """``infer_period_s`` must come from ``strafer_shared.constants`` so
    a future training-rate experiment can't silently leave the
    deployment loop at the old rate. Anchored as a mock-patch test
    against the underlying ``POLICY_SIM_DT`` / ``POLICY_DECIMATION``
    rather than against the derived ``POLICY_PERIOD_S`` so the test
    fails the moment someone hardcodes a literal back into the node.
    """

    def test_default_matches_shared_constants_product(self):
        from strafer_inference.inference_node import _default_infer_period
        from strafer_shared.constants import (
            POLICY_DECIMATION, POLICY_SIM_DT,
        )

        assert _default_infer_period() == POLICY_SIM_DT * POLICY_DECIMATION

    def test_mock_patched_sim_dt_changes_default(self, monkeypatch):
        from strafer_inference import inference_node
        from strafer_shared import constants

        monkeypatch.setattr(constants, "POLICY_SIM_DT", 1.0 / 60.0)
        monkeypatch.setattr(constants, "POLICY_DECIMATION", 8)
        assert inference_node._default_infer_period() == (1.0 / 60.0) * 8

    def test_default_is_30_hz_today(self):
        from strafer_inference.inference_node import _default_infer_period

        # 1/30 s = 30 Hz. The product is exact in float64 but the
        # round-trip through float comparison wants a tolerance.
        assert _default_infer_period() == pytest.approx(1.0 / 30.0)


# =============================================================================
# PolicyVariant parsing
# =============================================================================


class TestPolicyVariantParsing:
    """The string param resolves to a PolicyVariant; unknown values
    must fail loudly at init time, not silently substitute a default.
    """

    def test_known_variants_resolve(self):
        from strafer_shared.policy_interface import PolicyVariant

        assert PolicyVariant["DEPTH"] is PolicyVariant.DEPTH
        assert PolicyVariant["NOCAM"] is PolicyVariant.NOCAM

    def test_unknown_variant_raises_keyerror(self):
        from strafer_shared.policy_interface import PolicyVariant

        with pytest.raises(KeyError):
            PolicyVariant["RGB"]


# =============================================================================
# Topic + frame defaults the obs pipeline depends on
# =============================================================================


class TestObsPipelineConfigDefaults:
    """Defaults in the YAML must agree with the canonical names in
    ``strafer_shared.constants`` so a rename over there flushes
    through here without operator overrides.
    """

    def test_depth_topic_matches_perception_camera(self, node_params):
        from strafer_shared.constants import TOPIC_DEPTH_IMAGE

        assert node_params["depth_topic"] == TOPIC_DEPTH_IMAGE

    def test_odom_topic_matches_strafer_shared(self, node_params):
        from strafer_shared.constants import TOPIC_ODOM

        assert node_params["odom_topic"] == TOPIC_ODOM
