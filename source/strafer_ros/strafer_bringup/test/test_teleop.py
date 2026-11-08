"""Pins the keyboard teleop service: teleop_twist_keyboard, a strafer_bringup dependency, run on
the deploy stack's /cmd_vel at walking pace. Where the package is installed, checks that the node
still declares each pinned name and nothing that would republish a held command. No robot needed.
"""

import importlib.util
import os
import re
import shlex
import xml.etree.ElementTree as ET

import pytest
import yaml

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
COMPOSE = os.path.join(TEST_DIR, "..", "..", "deploy", "docker-compose.yml")
PACKAGE_XML = os.path.join(TEST_DIR, "..", "package.xml")

EXECUTABLE = ["ros2", "run", "teleop_twist_keyboard", "teleop_twist_keyboard", "--ros-args"]
REMAPS = ["cmd_vel:=/cmd_vel"]
PINNED = {"speed": "0.3", "turn": "0.6"}
KNOWN_PARAMETERS = {"stamped", "frame_id", "speed", "turn"}


def _service():
    with open(COMPOSE) as f:
        svc = yaml.safe_load(f)["services"]["teleop"]
    command = svc["command"]
    argv = shlex.split(command) if isinstance(command, str) else [str(a) for a in command]
    return svc, argv


def _values(argv, flag):
    return [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == flag]


def _params(argv):
    return dict(p.split(":=", 1) for p in _values(argv, "-p"))


class TestTeleopService:

    def test_runs_teleop_twist_keyboard(self):
        _, argv = _service()
        assert argv[: len(EXECUTABLE)] == EXECUTABLE

    def test_publishes_on_the_stack_cmd_vel(self):
        _, argv = _service()
        assert _values(argv, "-r") == REMAPS

    @pytest.mark.parametrize("name,value", sorted(PINNED.items()))
    def test_walking_pace(self, name, value):
        _, argv = _service()
        assert _params(argv).get(name) == value

    def test_no_other_parameter(self):
        _, argv = _service()
        assert sorted(_params(argv)) == sorted(PINNED)

    def test_opt_in_and_never_restarted(self):
        svc, _ = _service()
        assert svc["profiles"] == ["teleop"]
        assert svc["restart"] == "no"

    def test_bringup_declares_the_dependency(self):
        depends = [e.text for e in ET.parse(PACKAGE_XML).getroot().findall("exec_depend")]
        assert "teleop_twist_keyboard" in depends


class TestInstalledNode:

    @pytest.fixture(scope="class")
    def source(self):
        spec = importlib.util.find_spec("teleop_twist_keyboard")
        if spec is None or not spec.origin:
            pytest.skip("teleop_twist_keyboard is not installed here; an image built before it "
                        "became a strafer_bringup dependency does not carry it")
        with open(spec.origin) as f:
            return f.read()

    def test_declares_exactly_the_known_parameters(self, source):
        declared = set(re.findall(r"declare_parameter\(\s*['\"](\w+)['\"]", source))
        assert declared == KNOWN_PARAMETERS

    def test_publishes_on_relative_cmd_vel(self, source):
        assert re.search(r"create_publisher\(\s*\w+,\s*['\"]cmd_vel['\"]", source)

    def test_nothing_republishes_a_held_command(self, source):
        assert "create_timer" not in source
