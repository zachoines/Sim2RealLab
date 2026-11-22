"""``strafer_shared.rolling_subgoal`` imports with numpy alone, and
``strafer_inference.generator`` serves its names."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import strafer_shared.rolling_subgoal as core
from strafer_inference import generator

# A fresh interpreter, as this process has already imported strafer_shared and
# rclpy. numpy is in the baseline since its compiled extensions register modules
# of their own. A finder refuses the blocked names and records every attempt, so
# a guarded import fails too; the API runs once so an import inside a function
# is reached.
_PROBE = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
import numpy
blocked = set(sys.argv[2:])
attempts = []
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.partition(".")[0] in blocked:
            attempts.append(name)
            raise ImportError(f"import of {name} refused")
        return None
sys.meta_path.insert(0, Refuse())
before = set(sys.modules)
import strafer_shared.rolling_subgoal as core
path = [[0.0, 0.0], [2.0, 0.0]]
robot = [0.5, 0.1]
rolling = core.RollingSubgoalGenerator()
rolling.set_path(path)
rolling.update(robot)
rolling.project(robot)
core.evaluate_admission(has_anchor=True, cross_track_m=9.0)
core.arc_length_projection(path, robot)
core.path_arc_lengths(path)
loaded = {name.partition(".")[0] for name in set(sys.modules) - before}
extra = loaded - set(sys.stdlib_module_names) - {"numpy", "strafer_shared"}
assert not attempts, attempts
assert not extra, sorted(extra)
print(core.__file__)
"""

_BLOCKED = ("torch", "rclpy", "isaaclab")

_OBJECTS = (
    "AnchorAdmission",
    "RollingSubgoalGenerator",
    "SubgoalState",
    "arc_length_projection",
    "evaluate_admission",
    "path_arc_lengths",
)

_CONSTANTS = (
    "ADMIT_COLLISION",
    "ADMIT_CROSS_TRACK",
    "ADMIT_GOAL_CHANGED",
    "ADMIT_NO_ANCHOR",
    "ADMIT_ROLLING",
    "REJECT_ANCHOR_HELD",
)


def test_core_imports_with_numpy_alone(tmp_path):
    shared_root = Path(core.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(shared_root), *_BLOCKED],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == Path(core.__file__).resolve()


@pytest.mark.parametrize("name", _OBJECTS)
def test_generator_serves_the_core_object(name):
    assert getattr(generator, name) is getattr(core, name)


# By value: equal identifier-like strings are interned to one object, so `is`
# would pass for a copy too.
@pytest.mark.parametrize("name", _CONSTANTS)
def test_generator_serves_the_core_value(name):
    assert getattr(generator, name) == getattr(core, name)
