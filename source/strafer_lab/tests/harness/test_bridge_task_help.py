# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""``run_sim_in_the_loop.py --task`` names every registered bridge task.

A bridge task is one whose env cfg is a bridge variant (``StraferNavCfg_BridgeAutonomy*``). The flag's default and
help are read from the script's source, so the check needs neither Kit nor the script's imports.
"""

from __future__ import annotations

import ast
from pathlib import Path

import gymnasium as gym

import strafer_lab.tasks  # noqa: F401  (registers the envs)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_sim_in_the_loop.py"


def _task_flag_text() -> str:
    """The ``--task`` argument's default and help, joined."""
    for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--task"):
            kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg in ("default", "help")}
            return f"{kw.get('default', '')} {kw.get('help', '')}"
    raise AssertionError("no --task argument in run_sim_in_the_loop.py")


def _bridge_tasks() -> list[str]:
    return sorted(task for task, spec in gym.envs.registry.items()
                  if str((spec.kwargs or {}).get("env_cfg_entry_point", "")).split(":")[-1].startswith(
                      "StraferNavCfg_BridgeAutonomy"))


def test_the_task_flag_names_every_registered_bridge_task():
    tasks = _bridge_tasks()
    assert len(tasks) >= 2, tasks  # the Infinigen and the ProcRoom bridge
    text = _task_flag_text()
    assert [t for t in tasks if t not in text] == []
