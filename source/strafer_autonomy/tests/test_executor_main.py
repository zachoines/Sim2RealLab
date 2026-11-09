"""Unit tests for executor entry-point env-var plumbing."""

from __future__ import annotations

import os
import signal
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from strafer_autonomy.clients.ros_client import RosClientConfig, _SUPPORTED_BACKENDS
from strafer_autonomy.executor import main as executor_main
from strafer_autonomy.executor.main import (
    _read_bool_env,
    _read_float_env,
    _read_str_env,
)
from strafer_autonomy.executor.mission_runner import (
    DEFAULT_AVAILABLE_SKILLS,
    MissionRunner,
    MissionRunnerConfig,
    _MissionRuntime,
)
from strafer_autonomy.schemas import Pose3D, SkillCall, SkillResult


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


class TestProgressAwareTuningEnv(unittest.TestCase):
    """The four progress-aware tuning knobs each plumb into ``MissionRunnerConfig``."""

    _ENV_TO_FIELD = {
        "STRAFER_NAV_BUDGET_SAFETY_FACTOR":   ("nav_budget_safety_factor",   2.0),
        "STRAFER_NAV_BUDGET_SETUP_OVERHEAD_S": ("nav_budget_setup_overhead_s", 5.0),
        "STRAFER_NAV_STALL_PROGRESS_M":       ("nav_stall_progress_m",       0.10),
        "STRAFER_NAV_STALL_WINDOW_S":         ("nav_stall_window_s",         20.0),
    }

    def test_dataclass_defaults_match_documentation(self) -> None:
        cfg = MissionRunnerConfig()
        for _, (field, default) in self._ENV_TO_FIELD.items():
            self.assertAlmostEqual(getattr(cfg, field), default, msg=field)

    def test_each_env_overrides_its_field(self) -> None:
        for env_name, (field, _) in self._ENV_TO_FIELD.items():
            kwargs: dict = {}
            with patch.dict(os.environ, {env_name: "42.5"}, clear=False):
                _read_float_env(env_name, field, kwargs)
            cfg = MissionRunnerConfig(**kwargs)
            self.assertEqual(getattr(cfg, field), 42.5, f"{env_name} -> {field}")


class TestClockStallBailEnv(unittest.TestCase):
    """``STRAFER_CLOCK_STALL_BAIL_WALL_S`` plumbs into ``RosClientConfig``."""

    def test_default_is_fifteen_seconds(self) -> None:
        self.assertEqual(RosClientConfig().clock_stall_bail_wall_s, 15.0)

    def test_env_overrides_field(self) -> None:
        kwargs: dict = {}
        with patch.dict(
            os.environ, {"STRAFER_CLOCK_STALL_BAIL_WALL_S": "30.0"}, clear=False
        ):
            _read_float_env(
                "STRAFER_CLOCK_STALL_BAIL_WALL_S",
                "clock_stall_bail_wall_s",
                kwargs,
            )
        cfg = RosClientConfig(**kwargs)
        self.assertEqual(cfg.clock_stall_bail_wall_s, 30.0)


class TestNavBackendEnv(unittest.TestCase):
    """``STRAFER_NAV_BACKEND`` plumbs into ``MissionRunnerConfig`` and falls
    back to ``nav2`` on an unsupported value.
    """

    def _read(self, kwargs: dict) -> None:
        _read_str_env(
            "STRAFER_NAV_BACKEND",
            "default_navigation_backend",
            kwargs,
            allowed=_SUPPORTED_BACKENDS,
        )

    def test_default_preserved_when_unset(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "STRAFER_NAV_BACKEND"}
        with patch.dict(os.environ, env, clear=True):
            kwargs: dict = {}
            self._read(kwargs)
            cfg = MissionRunnerConfig(**kwargs) if kwargs else MissionRunnerConfig()
        self.assertEqual(kwargs, {})
        self.assertEqual(cfg.default_navigation_backend, "nav2")

    def test_hybrid_reaches_config(self) -> None:
        with patch.dict(
            os.environ, {"STRAFER_NAV_BACKEND": "hybrid_nav2_strafer"}, clear=False
        ):
            kwargs: dict = {}
            self._read(kwargs)
            cfg = MissionRunnerConfig(**kwargs)
        self.assertEqual(cfg.default_navigation_backend, "hybrid_nav2_strafer")

    def test_strafer_direct_reaches_config(self) -> None:
        with patch.dict(
            os.environ, {"STRAFER_NAV_BACKEND": "strafer_direct"}, clear=False
        ):
            kwargs: dict = {}
            self._read(kwargs)
            cfg = MissionRunnerConfig(**kwargs)
        self.assertEqual(cfg.default_navigation_backend, "strafer_direct")

    def test_unknown_value_warns_and_keeps_nav2(self) -> None:
        with patch.dict(
            os.environ, {"STRAFER_NAV_BACKEND": "mppi_local_only"}, clear=False
        ):
            kwargs: dict = {}
            with self.assertLogs(
                "strafer_autonomy.executor.main", level="WARNING"
            ) as cm:
                self._read(kwargs)
            cfg = MissionRunnerConfig(**kwargs) if kwargs else MissionRunnerConfig()
        self.assertEqual(kwargs, {})  # skipped, default preserved
        self.assertEqual(cfg.default_navigation_backend, "nav2")
        # and the skip is observable, not silent
        self.assertTrue(any("mppi_local_only" in m for m in cm.output))


class TestNavBackendDispatchDefault(unittest.TestCase):
    """``_dispatch_nav_goal`` forwards ``default_navigation_backend`` to the
    ROS client when the step carries no ``execution_backend``; a per-step
    value still wins (the independent-safety invariant).
    """

    def _runner(self, backend: str):
        runner = MissionRunner(
            planner_client=MagicMock(),
            grounding_client=MagicMock(),
            ros_client=MagicMock(),
            config=MissionRunnerConfig(default_navigation_backend=backend),
        )
        ros = runner._ros_client
        ros.navigate_to_pose.return_value = SkillResult(
            step_id="n1", skill="navigate_to_pose", status="succeeded",
        )
        ros.get_map_pose.return_value = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
        }
        return runner, ros

    @staticmethod
    def _goal() -> Pose3D:
        return Pose3D(x=1.0, y=0.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)

    def test_default_backend_forwarded_when_step_omits_it(self) -> None:
        runner, ros = self._runner("hybrid_nav2_strafer")
        step = SkillCall(skill="navigate_to_pose", step_id="n1", args={})
        runner._dispatch_nav_goal(step, self._goal(), 0.0)
        self.assertEqual(
            ros.navigate_to_pose.call_args.kwargs["execution_backend"],
            "hybrid_nav2_strafer",
        )

    def test_step_arg_overrides_config_default(self) -> None:
        # Until the compiler stops hardcoding execution_backend, a per-step
        # value wins over the config default -- so the env-set hybrid default
        # stays inert (nav2) on a real mission.
        runner, ros = self._runner("hybrid_nav2_strafer")
        step = SkillCall(
            skill="navigate_to_pose", step_id="n1",
            args={"execution_backend": "nav2"},
        )
        runner._dispatch_nav_goal(step, self._goal(), 0.0)
        self.assertEqual(
            ros.navigate_to_pose.call_args.kwargs["execution_backend"], "nav2",
        )


class TestPlanBackendValidation(unittest.TestCase):
    """``_validate_plan`` accepts the dispatch backend vocabulary and rejects
    names outside it -- locking the ``_RECOGNISED_BACKENDS`` reconciliation.
    """

    @staticmethod
    def _validate(backend: str) -> list[str]:
        runner = MissionRunner.__new__(MissionRunner)
        runner._config = type(
            "C", (), {"available_skills": DEFAULT_AVAILABLE_SKILLS}
        )()
        plan = type("P", (), {"steps": [
            SkillCall(
                skill="navigate_to_pose",
                step_id="n1",
                args={"execution_backend": backend},
            ),
        ]})()
        return runner._validate_plan(plan)

    def test_dispatch_backends_validate(self) -> None:
        # Fails on the old {nav2, direct} vocabulary.
        for backend in ("nav2", "strafer_direct", "hybrid_nav2_strafer"):
            self.assertEqual(self._validate(backend), [], f"backend={backend}")

    def test_dead_direct_token_rejected(self) -> None:
        # Would have validated under the old {nav2, direct} vocabulary; the
        # reconciliation drops it.
        errors = self._validate("direct")
        self.assertTrue(any("execution_backend" in e for e in errors))

    def test_typo_rejected(self) -> None:
        errors = self._validate("hybrid")  # typo of hybrid_nav2_strafer
        self.assertTrue(any("execution_backend" in e for e in errors))


if __name__ == "__main__":
    unittest.main()


class TestShutdown(unittest.TestCase):
    """The exit path cancels the active mission before it destroys the command server."""

    def _runner_and_server(self, *, active: bool):
        ros = MagicMock()
        runner = MissionRunner(
            planner_client=MagicMock(),
            grounding_client=MagicMock(),
            ros_client=ros,
        )
        if active:
            runner._active_runtime = _MissionRuntime(
                mission_id="m1",
                request_id="m1",
                raw_command="go to the door",
                source="test",
                started_at=0.0,
            )
        server = MagicMock()
        calls: list[str] = []
        ros.cancel_active_navigation.side_effect = lambda: calls.append("cancel")
        ros.publish_zero_cmd_vel.side_effect = lambda: calls.append("zero")
        server.destroy.side_effect = lambda: calls.append("destroy")
        return runner, server, calls

    def test_an_active_mission_is_canceled_and_stopped_before_the_destroy(self) -> None:
        runner, server, calls = self._runner_and_server(active=True)
        executor_main.shutdown(server, runner)
        self.assertEqual(calls, ["cancel", "zero", "destroy"])
        self.assertEqual(runner.get_status().state, "canceling")

    def test_no_active_mission_cancels_nothing(self) -> None:
        runner, server, calls = self._runner_and_server(active=False)
        executor_main.shutdown(server, runner)
        self.assertEqual(calls, ["destroy"])

    def test_a_raising_cancel_still_destroys(self) -> None:
        runner, server, calls = self._runner_and_server(active=True)
        runner.cancel_active_mission = MagicMock(side_effect=RuntimeError("cancel failed"))
        with self.assertLogs(executor_main.logger, level="ERROR"):
            executor_main.shutdown(server, runner)
        self.assertEqual(calls, ["destroy"])


class TestMainExitPath(unittest.TestCase):
    """Every exit from ``main()`` stops the ROS client, then shuts the context down."""

    def _run_main(self, *, spin=None, build_error=None, interrupt_on_ignore=False):
        calls: list[str] = []
        interrupted: list[bool] = []
        fake_rclpy = types.ModuleType("rclpy")
        fake_rclpy.ok = lambda: True
        fake_rclpy.init = MagicMock()
        fake_rclpy.try_shutdown = lambda: calls.append("try_shutdown")
        fake_signals = types.ModuleType("rclpy.signals")
        fake_signals.SignalHandlerOptions = types.SimpleNamespace(NO=object())
        fake_rclpy.signals = fake_signals

        ros_client = MagicMock()
        ros_client.close.side_effect = lambda: calls.append("close")
        server = MagicMock()
        server.spin.side_effect = spin
        runner = MagicMock()

        def build(**_kwargs):
            calls.append("build")
            if build_error is not None:
                raise build_error
            return server, runner

        def record_signal(signum, handler):
            if handler is signal.SIG_IGN:
                calls.append(f"ignore:{signal.Signals(signum).name}")
                if interrupt_on_ignore and not interrupted:
                    interrupted.append(True)
                    raise KeyboardInterrupt
            elif handler is signal.default_int_handler:
                calls.append(f"interrupt:{signal.Signals(signum).name}")

        env = {"VLM_URL": "http://127.0.0.1:9", "PLANNER_URL": "http://127.0.0.1:9"}
        with patch.dict(sys.modules, {"rclpy": fake_rclpy, "rclpy.signals": fake_signals}), \
                patch.dict(os.environ, env, clear=False), \
                patch("signal.signal", side_effect=record_signal), \
                patch("strafer_autonomy.clients.ros_client.JetsonRosClient", return_value=ros_client), \
                patch("strafer_autonomy.executor.command_server.build_command_server", side_effect=build), \
                patch.object(executor_main, "shutdown", side_effect=lambda s, r: calls.append("shutdown")):
            try:
                executor_main.main()
            except BaseException as exc:
                return calls, exc
        return calls, None

    def test_a_handled_signal_stops_the_client_after_the_shutdown(self) -> None:
        calls, raised = self._run_main(spin=KeyboardInterrupt)
        self.assertIsNone(raised)
        self.assertEqual(
            calls,
            ["interrupt:SIGINT", "interrupt:SIGTERM", "build", "shutdown", "ignore:SIGINT", "ignore:SIGTERM", "close", "try_shutdown"],
        )

    def test_a_startup_failure_stops_the_client_and_raises_its_error(self) -> None:
        error = RuntimeError("VLM health check failed")
        calls, raised = self._run_main(build_error=error)
        self.assertIs(raised, error)
        self.assertEqual(
            calls,
            ["interrupt:SIGINT", "interrupt:SIGTERM", "build", "ignore:SIGINT", "ignore:SIGTERM", "close", "try_shutdown"],
        )

    def test_an_exception_from_spin_stops_the_client_and_raises_it(self) -> None:
        error = RuntimeError("spin failed")
        calls, raised = self._run_main(spin=error)
        self.assertIs(raised, error)
        self.assertEqual(
            calls,
            ["interrupt:SIGINT", "interrupt:SIGTERM", "build", "shutdown", "ignore:SIGINT", "ignore:SIGTERM", "close", "try_shutdown"],
        )

    def test_an_interrupt_while_ignoring_signals_still_stops_the_client(self) -> None:
        calls, raised = self._run_main(spin=KeyboardInterrupt, interrupt_on_ignore=True)
        self.assertIsInstance(raised, KeyboardInterrupt)
        self.assertEqual(calls[-4:], ["shutdown", "ignore:SIGINT", "close", "try_shutdown"])
