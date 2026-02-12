# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Root pytest configuration for Strafer Lab tests.

This file launches Isaac Sim once for all tests in the test/ folder.
Subfolder conftest.py files provide folder-specific fixtures but should NOT
launch their own AppLauncher.
"""

import os
import threading

# Isaac Sim must be launched before importing Isaac Lab modules
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


def pytest_sessionfinish(session, exitstatus):
    """Clean up Isaac Sim after all tests complete.

    ``simulation_app.close()`` shuts down the Omniverse runtime.  On Windows
    the PhysX / CUDA teardown can hang indefinitely, so we set a watchdog
    timer that calls ``os._exit()`` if close does not return in time.
    """
    def _watchdog():
        """Force-kill the process if close() hangs longer than 30 s."""
        os._exit(exitstatus)

    timer = threading.Timer(30.0, _watchdog)
    timer.daemon = True
    timer.start()

    try:
        simulation_app.close()
    finally:
        timer.cancel()
        # Even after close() returns, the process sometimes stalls on
        # atexit handlers or daemon threads.  Force-exit to guarantee
        # the CI pipeline / terminal gets released immediately.
        os._exit(exitstatus)
