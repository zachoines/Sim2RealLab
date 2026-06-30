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


# =============================================================================
# Sim-in-the-loop bringup — donut warmup wiring
# =============================================================================


class TestSimInTheLoopBringup:
    """Sim-only bringup wires the donut_warmup node and its toggle arg.

    The warmup publishes a slow 360 deg rotation on /cmd_vel once at
    bringup so RTAB-Map gets ~20 depth frames at varied yaws at the
    spawn pose. Lives in a separate process (not in the navigate-to-pose
    BT) because once-per-session semantics are reliable only when the
    lifecycle is process-scoped — BT.CPP decorator state doesn't survive
    the halt/reset that Nav2 applies between goals.
    """

    def _load(self, pkg_dir, filename):
        import importlib.util

        path = os.path.join(pkg_dir, "launch", filename)
        spec = importlib.util.spec_from_file_location(filename, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.generate_launch_description()

    def test_sim_launch_file_exists(self, pkg_dir):
        path = os.path.join(pkg_dir, "launch", "bringup_sim_in_the_loop.launch.py")
        assert os.path.isfile(path)

    def test_sim_launch_declares_donut_warmup_arg(self, pkg_dir):
        from launch.actions import DeclareLaunchArgument

        ld = self._load(pkg_dir, "bringup_sim_in_the_loop.launch.py")
        arg_names = [
            e.name for e in ld.entities if isinstance(e, DeclareLaunchArgument)
        ]
        assert "donut_warmup" in arg_names, (
            "Sim bringup must declare a donut_warmup toggle so the "
            "operator can skip the startup spin when working off a "
            "populated map."
        )

    def test_donut_warmup_default_enabled(self, pkg_dir):
        """Default must be 'true' on the sim lane — the symptom this
        guards against is a cold-mapped first mission. Operators who
        don't want the spin pass donut_warmup:=false explicitly.
        """
        from launch.actions import DeclareLaunchArgument

        ld = self._load(pkg_dir, "bringup_sim_in_the_loop.launch.py")
        for e in ld.entities:
            if isinstance(e, DeclareLaunchArgument) and e.name == "donut_warmup":
                assert e.default_value[0].text.lower() == "true"
                return
        raise AssertionError("donut_warmup arg not found")

    def test_real_robot_navigation_declares_donut_warmup_arg(self, pkg_dir):
        """Real-robot bringup must declare the toggle so operators can
        opt in on a cold-start session.
        """
        from launch.actions import DeclareLaunchArgument

        ld = self._load(pkg_dir, "navigation.launch.py")
        arg_names = [
            e.name for e in ld.entities if isinstance(e, DeclareLaunchArgument)
        ]
        assert "donut_warmup" in arg_names

    def test_real_robot_donut_warmup_default_disabled(self, pkg_dir):
        """Default must be 'false' on the real-robot lane. The donut
        symptom is amplified by the sim's cold-mapped workflow; in
        steady-state real-robot ops the rtabmap.db persists across
        sessions and a startup spin is unnecessary motion (and a real
        safety concern on hardware).
        """
        from launch.actions import DeclareLaunchArgument

        ld = self._load(pkg_dir, "navigation.launch.py")
        for e in ld.entities:
            if isinstance(e, DeclareLaunchArgument) and e.name == "donut_warmup":
                assert e.default_value[0].text.lower() == "false", (
                    "donut_warmup must default to false on real-robot "
                    "bringup; opt-in via donut_warmup:=true per session"
                )
                return
        raise AssertionError("donut_warmup arg not found on real-robot bringup")


class TestDonutWarmupNode:
    """Smoke tests for the donut_warmup node — importable and exposes
    the parameter surface that the launch and operators rely on.
    """

    def test_module_importable(self):
        from strafer_bringup import donut_warmup  # noqa: F401

    def test_main_callable(self):
        from strafer_bringup.donut_warmup import main

        assert callable(main)

    def test_node_class_exposed(self):
        from strafer_bringup.donut_warmup import DonutWarmup

        assert DonutWarmup.__name__ == "DonutWarmup"

    def test_console_script_registered(self):
        """The package's setup.py must register donut_warmup as a
        console script so `ros2 run strafer_bringup donut_warmup` works.
        """
        import importlib.metadata as md

        eps = md.entry_points(group="console_scripts")
        names = {ep.name for ep in eps}
        assert "donut_warmup" in names


# =============================================================================
# STRAFER_NAV_BACKEND gating — auto-launch of the RL inference nodes
# =============================================================================


def _load_module(pkg_dir, filename, modname):
    import importlib.util

    path = os.path.join(pkg_dir, "launch", filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _include_basenames(entities):
    """Basenames of the launch files pulled in by IncludeLaunchDescription
    entities (or node-list items), e.g. 'inference.launch.py'.
    """
    from launch.actions import IncludeLaunchDescription

    names = []
    for entity in entities:
        if isinstance(entity, IncludeLaunchDescription):
            src = entity.launch_description_source
            loc = src._LaunchDescriptionSource__location
            names.append(os.path.basename(loc[0].text))
    return names


def _capture_launch_errors(fn):
    """Run ``fn`` while recording ERROR records on the ``launch`` logger.

    The loud empty-model check logs on the ``launch`` logger, which ROS
    configures with ``propagate=False`` and its own handler — so pytest's
    ``caplog`` (rooted at the root logger) never sees it. Attach a handler
    directly instead.
    """
    import logging

    class _ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    logger = logging.getLogger("launch")
    handler = _ListHandler()
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    return [r for r in handler.records if r.levelno >= logging.ERROR]


class TestRealBringupBackendGating:
    """autonomy.launch.py gates the inference / subgoal includes on
    STRAFER_NAV_BACKEND, read at build time (plain LaunchDescription, so
    the includes land directly in ld.entities).
    """

    def _includes(self, pkg_dir, monkeypatch, backend):
        if backend is None:
            monkeypatch.delenv("STRAFER_NAV_BACKEND", raising=False)
        else:
            monkeypatch.setenv("STRAFER_NAV_BACKEND", backend)
        # A real model path keeps the loud empty-model log out of the way.
        monkeypatch.setenv("STRAFER_INFERENCE_MODEL_PATH", "/models/depth.onnx")
        mod = _load_module(pkg_dir, "autonomy.launch.py", "autonomy_gating")
        ld = mod.generate_launch_description()
        return _include_basenames(ld.entities)

    def test_unset_launches_neither(self, pkg_dir, monkeypatch):
        names = self._includes(pkg_dir, monkeypatch, None)
        assert "inference.launch.py" not in names
        assert "subgoal_generator.launch.py" not in names

    def test_nav2_launches_neither(self, pkg_dir, monkeypatch):
        names = self._includes(pkg_dir, monkeypatch, "nav2")
        assert "inference.launch.py" not in names
        assert "subgoal_generator.launch.py" not in names

    def test_strafer_direct_launches_inference_only(self, pkg_dir, monkeypatch):
        names = self._includes(pkg_dir, monkeypatch, "strafer_direct")
        assert "inference.launch.py" in names
        assert "subgoal_generator.launch.py" not in names

    def test_hybrid_launches_both(self, pkg_dir, monkeypatch):
        names = self._includes(pkg_dir, monkeypatch, "hybrid_nav2_strafer")
        assert "inference.launch.py" in names
        assert "subgoal_generator.launch.py" in names

    def test_loud_error_on_policy_backend_empty_model(self, pkg_dir, monkeypatch):
        monkeypatch.setenv("STRAFER_NAV_BACKEND", "hybrid_nav2_strafer")
        monkeypatch.delenv("STRAFER_INFERENCE_MODEL_PATH", raising=False)
        mod = _load_module(pkg_dir, "autonomy.launch.py", "autonomy_loud")
        records = _capture_launch_errors(mod.generate_launch_description)
        assert any(
            "STRAFER_INFERENCE_MODEL_PATH" in r.getMessage() for r in records
        ), "expected a launch-time ERROR when a policy backend has no model_path"


class TestSimBringupBackendGating:
    """bringup_sim_in_the_loop.launch.py builds its node list inside the
    _launch_setup OpaqueFunction, so the includes are NOT in
    generate_launch_description().entities. Assert on the node list
    _launch_setup returns for a LaunchContext with the backend set.
    """

    # Every LaunchConfiguration _launch_setup performs; values are
    # passed through to includes and otherwise inert (viewer_port is
    # int()-parsed).
    _BASE_CONFIGS = {
        "vlm_url": "",
        "planner_url": "",
        "nav_log_level": "warn",
        "localization": "false",
        "database_path": "~/.ros/rtabmap.db",
        "rtabmap_args": "",
        "rtabmap_viz": "false",
        "viewer_port": "8765",
        "model_path": "/models/depth.onnx",
        "policy_variant": "DEPTH",
    }

    def _node_includes(self, pkg_dir, backend, model_path="/models/depth.onnx"):
        from launch import LaunchContext

        mod = _load_module(
            pkg_dir, "bringup_sim_in_the_loop.launch.py", "sim_gating"
        )
        ctx = LaunchContext()
        configs = dict(self._BASE_CONFIGS)
        configs["nav_backend"] = backend
        configs["model_path"] = model_path
        ctx.launch_configurations.update(configs)
        nodes = mod._launch_setup(ctx)
        return _include_basenames(nodes)

    def test_nav2_launches_neither(self, pkg_dir):
        names = self._node_includes(pkg_dir, "nav2")
        assert "inference.launch.py" not in names
        assert "subgoal_generator.launch.py" not in names

    def test_strafer_direct_launches_inference_only(self, pkg_dir):
        names = self._node_includes(pkg_dir, "strafer_direct")
        assert "inference.launch.py" in names
        assert "subgoal_generator.launch.py" not in names

    def test_hybrid_launches_both(self, pkg_dir):
        names = self._node_includes(pkg_dir, "hybrid_nav2_strafer")
        assert "inference.launch.py" in names
        assert "subgoal_generator.launch.py" in names

    def test_sim_declares_nav_backend_arg(self, pkg_dir):
        from launch.actions import DeclareLaunchArgument

        mod = _load_module(
            pkg_dir, "bringup_sim_in_the_loop.launch.py", "sim_args"
        )
        ld = mod.generate_launch_description()
        names = [
            e.name for e in ld.entities
            if isinstance(e, DeclareLaunchArgument)
        ]
        assert "nav_backend" in names

    def test_loud_error_on_policy_backend_empty_model(self, pkg_dir):
        records = _capture_launch_errors(
            lambda: self._node_includes(
                pkg_dir, "hybrid_nav2_strafer", model_path=""
            )
        )
        assert any(
            "STRAFER_INFERENCE_MODEL_PATH" in r.getMessage() for r in records
        ), "expected a launch-time ERROR when a policy backend has no model_path"
