"""Unit tests for executor entry-point env-var plumbing."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from strafer_autonomy.executor.main import _read_bool_env, _read_float_env
from strafer_autonomy.executor.mission_runner import MissionRunnerConfig


class TestReadFloatEnv(unittest.TestCase):
    """``_read_float_env`` populates a kwargs dict from numeric env vars."""

    def test_sets_value_when_env_set(self) -> None:
        target: dict = {}
        with patch.dict(os.environ, {"FOO": "42.5"}, clear=False):
            _read_float_env("FOO", "some_key", target)
        self.assertEqual(target, {"some_key": 42.5})

    def test_skips_when_env_missing(self) -> None:
        target: dict = {}
        env = {k: v for k, v in os.environ.items() if k != "FOO_MISSING"}
        with patch.dict(os.environ, env, clear=True):
            _read_float_env("FOO_MISSING", "some_key", target)
        self.assertEqual(target, {})

    def test_skips_when_env_non_numeric(self) -> None:
        target: dict = {}
        with patch.dict(os.environ, {"FOO": "not-a-number"}, clear=False):
            _read_float_env("FOO", "some_key", target)
        self.assertEqual(target, {})


class TestNavigationTimeoutEnv(unittest.TestCase):
    """``STRAFER_NAVIGATION_TIMEOUT_S`` plumbs into ``MissionRunnerConfig``."""

    def test_default_preserved_when_unset(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "STRAFER_NAVIGATION_TIMEOUT_S"}
        with patch.dict(os.environ, env, clear=True):
            kwargs: dict = {}
            _read_float_env(
                "STRAFER_NAVIGATION_TIMEOUT_S",
                "default_navigation_timeout_s",
                kwargs,
            )
            cfg = MissionRunnerConfig(**kwargs) if kwargs else MissionRunnerConfig()
        self.assertEqual(cfg.default_navigation_timeout_s, 90.0)

    def test_env_override_reaches_config(self) -> None:
        with patch.dict(
            os.environ, {"STRAFER_NAVIGATION_TIMEOUT_S": "600.0"}, clear=False
        ):
            kwargs: dict = {}
            _read_float_env(
                "STRAFER_NAVIGATION_TIMEOUT_S",
                "default_navigation_timeout_s",
                kwargs,
            )
            cfg = MissionRunnerConfig(**kwargs)
        self.assertEqual(cfg.default_navigation_timeout_s, 600.0)


class TestReadBoolEnv(unittest.TestCase):
    """``_read_bool_env`` recognizes 0/false/no/off as False; other values as True."""

    def test_unset_skips(self) -> None:
        target: dict = {}
        env = {k: v for k, v in os.environ.items() if k != "BOOL_TEST"}
        with patch.dict(os.environ, env, clear=True):
            _read_bool_env("BOOL_TEST", "some_key", target)
        self.assertEqual(target, {})

    def test_empty_string_skips(self) -> None:
        target: dict = {}
        with patch.dict(os.environ, {"BOOL_TEST": ""}, clear=False):
            _read_bool_env("BOOL_TEST", "some_key", target)
        self.assertEqual(target, {})

    def test_falsey_values_set_false(self) -> None:
        for raw in ["0", "false", "FALSE", "no", "off", "False"]:
            target: dict = {}
            with patch.dict(os.environ, {"BOOL_TEST": raw}, clear=False):
                _read_bool_env("BOOL_TEST", "some_key", target)
            self.assertEqual(target, {"some_key": False}, f"raw={raw!r}")

    def test_truthy_values_set_true(self) -> None:
        for raw in ["1", "true", "TRUE", "yes", "on", "anything"]:
            target: dict = {}
            with patch.dict(os.environ, {"BOOL_TEST": raw}, clear=False):
                _read_bool_env("BOOL_TEST", "some_key", target)
            self.assertEqual(target, {"some_key": True}, f"raw={raw!r}")


class TestProgressAwareEnv(unittest.TestCase):
    """``STRAFER_NAV_PROGRESS_AWARE`` plumbs into ``MissionRunnerConfig``."""

    def test_default_preserved_when_unset(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "STRAFER_NAV_PROGRESS_AWARE"}
        with patch.dict(os.environ, env, clear=True):
            kwargs: dict = {}
            _read_bool_env(
                "STRAFER_NAV_PROGRESS_AWARE", "nav_progress_aware", kwargs,
            )
            cfg = MissionRunnerConfig(**kwargs) if kwargs else MissionRunnerConfig()
        self.assertTrue(cfg.nav_progress_aware)

    def test_disabled_via_env(self) -> None:
        with patch.dict(os.environ, {"STRAFER_NAV_PROGRESS_AWARE": "0"}, clear=False):
            kwargs: dict = {}
            _read_bool_env(
                "STRAFER_NAV_PROGRESS_AWARE", "nav_progress_aware", kwargs,
            )
            cfg = MissionRunnerConfig(**kwargs)
        self.assertFalse(cfg.nav_progress_aware)


if __name__ == "__main__":
    unittest.main()
