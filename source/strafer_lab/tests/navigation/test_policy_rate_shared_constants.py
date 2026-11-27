# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""The navigation env's step rate is the shared policy rate the Jetson's inference node also reads.

``_DEFAULT_NAV_SIM_DT`` and ``_DEFAULT_NAV_DECIMATION`` are bound to ``strafer_shared.constants``'s ``POLICY_SIM_DT``
and ``POLICY_DECIMATION``. The values alone cannot tell a re-export from a literal that happens to agree, so the
module's source is read too. No Kit / GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

from strafer_shared import constants

from strafer_lab.tasks.navigation import strafer_env_cfg


def _assigned(name: str) -> ast.expr:
    for node in ast.parse(Path(strafer_env_cfg.__file__).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise AssertionError(f"{name} is not assigned in strafer_env_cfg.py")


def test_the_env_step_rate_re_exports_the_shared_policy_rate():
    assert strafer_env_cfg._DEFAULT_NAV_SIM_DT is constants.POLICY_SIM_DT
    assert strafer_env_cfg._DEFAULT_NAV_DECIMATION == constants.POLICY_DECIMATION
    for name, shared in (("_DEFAULT_NAV_SIM_DT", "POLICY_SIM_DT"), ("_DEFAULT_NAV_DECIMATION", "POLICY_DECIMATION")):
        value = _assigned(name)
        assert isinstance(value, ast.Name) and value.id == shared, (name, ast.unparse(value))


def test_a_composed_env_steps_at_the_policy_period():
    from strafer_lab.tasks.navigation import composed_env_cfg as composed

    cfg = composed.StraferNavCfg_RLNoCamSubgoal_Real()
    assert cfg.sim.dt == constants.POLICY_SIM_DT and cfg.decimation == constants.POLICY_DECIMATION
    assert cfg.sim.dt * cfg.decimation == constants.POLICY_PERIOD_S
