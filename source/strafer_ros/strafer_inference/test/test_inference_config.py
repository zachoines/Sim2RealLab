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
        "goal_topic",
        "cmd_vel_topic",
        "depth_topic",
        "imu_topic",
        "joint_states_topic",
        "odom_topic",
        "tf_max_age_s",
        "goal_timeout_s",
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
            "goal_timeout_s",
            "obs_timeout_s",
            "depth_timeout_s",
        ):
            assert float(node_params[key]) > 0.0, (
                f"{key} must be positive; non-positive disables the watchdog"
            )

    def test_topic_names_absolute(self, node_params):
        for key in (
            "goal_topic",
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
        # 2 DeclareLaunchArgument + 1 Node action
        assert len(ld.entities) >= 3

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
