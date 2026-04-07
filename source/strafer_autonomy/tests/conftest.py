"""Shared pytest fixtures and markers for strafer_autonomy tests."""

from __future__ import annotations

import importlib

import pytest


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
