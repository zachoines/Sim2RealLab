# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Root pytest configuration for Strafer Lab tests.

This file launches Isaac Sim once for all tests in the test/ folder.
Subfolder conftest.py files provide folder-specific fixtures but should NOT
launch their own AppLauncher.
"""

# Isaac Sim must be launched before importing Isaac Lab modules
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


def pytest_sessionfinish(session, exitstatus):
    """Clean up Isaac Sim after all tests complete."""
    simulation_app.close()
