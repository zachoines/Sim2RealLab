"""Shared pytest fixtures and markers for strafer_autonomy tests."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _register_strafer_lab_stub() -> None:
    """Register a namespace-package stub for ``strafer_lab`` + ``strafer_lab.tools``.

    The real ``strafer_lab`` package imports ``isaaclab`` at import time via
    ``strafer_lab/__init__.py`` → ``assets/__init__.py``. That dependency is
    absent on DGX (and in most CI environments), so we cannot run
    ``import strafer_lab.tools.scene_labels`` through the normal package
    machinery.

    Instead we pre-populate ``sys.modules`` with lightweight stub packages
    that only expose ``__path__``. Python's import machinery then resolves
    ``strafer_lab.tools.<mod>`` against the real directory without ever
    executing the Isaac-Lab-dependent ``__init__.py``. The stubs are
    idempotent — if a test environment already has ``strafer_lab`` loaded
    (e.g. Isaac Sim is available), we leave it alone.
    """
    if "strafer_lab" in sys.modules:
        return

    # conftest.py lives at source/strafer_autonomy/tests/conftest.py
    # parents[0] = tests, [1] = strafer_autonomy, [2] = source, [3] = repo root
    repo_root = Path(__file__).resolve().parents[3]
    strafer_lab_path = repo_root / "source" / "strafer_lab" / "strafer_lab"
    tools_path = strafer_lab_path / "tools"
    if not tools_path.is_dir():
        return

    pkg = types.ModuleType("strafer_lab")
    pkg.__path__ = [str(strafer_lab_path)]  # type: ignore[attr-defined]
    sys.modules["strafer_lab"] = pkg

    tools_pkg = types.ModuleType("strafer_lab.tools")
    tools_pkg.__path__ = [str(tools_path)]  # type: ignore[attr-defined]
    sys.modules["strafer_lab.tools"] = tools_pkg


_register_strafer_lab_stub()


def _has_rclpy() -> bool:
    """Return True if rclpy is importable (i.e. running inside a ROS 2 environment)."""
    try:
        importlib.import_module("rclpy")
        return True
    except ImportError:
        return False


requires_ros = pytest.mark.skipif(
    not _has_rclpy(),
    reason="ROS 2 (rclpy) is not available — skipping ROS-dependent test",
)
