"""CLI validation tests for source/strafer_lab/scripts/capture.py.

Exercise argparse + the (driver, mission-source) cross-product validator
without launching Isaac Sim. The wired Tier 1 path (``teleop ×
scene-metadata``) subprocesses ``teleop_capture.py``; we inject a stub
runner so the test never spawns Isaac Sim. Other cells either raise
``NotImplementedError`` (pending tiers) or ``SystemExit`` (invalid
cross-product cells).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import capture  # noqa: E402  (post-path-mutation import)


def _base_args(driver: str, mission_source: str, **kwargs) -> list[str]:
    args = [
        "--driver", driver,
        "--mission-source", mission_source,
        "--scene", "scene_test",
        "--output", "/tmp/test_dataset_doesnotexist",
    ]
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


class TestValidCombinations:
    def test_all_seven_cells_listed(self):
        assert len(capture.VALID_COMBINATIONS) == 7

    def test_driver_choices_complete(self):
        assert set(capture.VALID_DRIVERS) == {"bridge", "teleop", "scripted"}

    def test_mission_source_choices_complete(self):
        assert set(capture.VALID_MISSION_SOURCES) == {
            "queue", "captioner", "coverage", "scene-metadata",
        }


class _StubRunner:
    """Captures the argv that would be subprocessed; returns rc=0."""

    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, argv, check: bool = False):
        self.calls.append(list(argv))
        return SimpleNamespace(returncode=self.returncode)


class TestTier1DispatchPath:
    def _run(self, runner, *args, **kwargs):
        # Re-implements capture.main with the runner override so the
        # test never spawns a real subprocess.
        parser = capture._build_parser()
        ns, extra = parser.parse_known_args(_base_args(*args, **kwargs))
        capture._validate(ns)
        return capture._dispatch(ns, extra_args=tuple(extra), runner=runner)

    def test_teleop_scene_metadata_dispatches_to_teleop_driver(self, capsys):
        runner = _StubRunner()
        rc = self._run(runner, "teleop", "scene-metadata")
        assert rc == 0
        assert len(runner.calls) == 1
        argv = runner.calls[0]
        # Subprocess targets the teleop driver script with translated args.
        assert argv[0] == sys.executable
        assert argv[1].endswith("teleop_capture.py")
        # Required forwarded args are present.
        assert "--scene" in argv and "scene_test" in argv
        assert "--output" in argv and "/tmp/test_dataset_doesnotexist" in argv
        # fps + vcodec defaults flow through.
        assert "--fps" in argv and "--vcodec" in argv
        # The resolved sensor stack is forwarded. The default
        # --capture-policy-cam=True resolves to perception + policy RGB.
        assert "--sensors" in argv
        assert argv[argv.index("--sensors") + 1] == "rgb_full,rgb_policy"

    def test_no_capture_policy_cam_resolves_to_rgb_only(self):
        runner = _StubRunner()
        # --no-capture-policy-cam falls back to the RGB-only stack.
        parser = capture._build_parser()
        argv_in = _base_args("teleop", "scene-metadata")
        argv_in.append("--no-capture-policy-cam")
        ns, extra = parser.parse_known_args(argv_in)
        capture._validate(ns)
        capture._dispatch(ns, extra_args=tuple(extra), runner=runner)
        argv = runner.calls[0]
        assert argv[argv.index("--sensors") + 1] == "rgb_full"

    def test_sensors_preset_forwarded(self):
        runner = _StubRunner()
        rc = self._run(runner, "teleop", "scene-metadata", sensors="vla")
        assert rc == 0
        argv = runner.calls[0]
        assert argv[argv.index("--sensors") + 1] == "rgb_full,depth_full"

    def test_sensors_token_list_canonicalized(self):
        runner = _StubRunner()
        # Order-insensitive: tokens are normalized to canonical order.
        self._run(runner, "teleop", "scene-metadata", sensors="depth_full,rgb_full")
        argv = runner.calls[0]
        assert argv[argv.index("--sensors") + 1] == "rgb_full,depth_full"

    def test_sensors_takes_precedence_over_bool(self):
        runner = _StubRunner()
        parser = capture._build_parser()
        argv_in = _base_args("teleop", "scene-metadata", sensors="rgb_full")
        argv_in.append("--capture-policy-cam")  # ignored when --sensors given
        ns, extra = parser.parse_known_args(argv_in)
        capture._validate(ns)
        capture._dispatch(ns, extra_args=tuple(extra), runner=runner)
        argv = runner.calls[0]
        assert argv[argv.index("--sensors") + 1] == "rgb_full"

    def test_subprocess_nonzero_propagates(self):
        runner = _StubRunner(returncode=7)
        rc = self._run(runner, "teleop", "scene-metadata")
        assert rc == 7

    def test_unknown_args_are_forwarded_to_child(self):
        # AppLauncher / pass-through flags must survive parse_known_args
        # and land on the child script's argv.
        runner = _StubRunner()
        parser = capture._build_parser()
        argv_in = _base_args("teleop", "scene-metadata") + [
            "--headless",
            "--device", "cpu",
        ]
        ns, extra = parser.parse_known_args(argv_in)
        capture._validate(ns)
        capture._dispatch(ns, extra_args=tuple(extra), runner=runner)
        child_argv = runner.calls[0]
        assert "--headless" in child_argv
        assert "--device" in child_argv and "cpu" in child_argv


class TestSensorStackResolution:
    def test_preset_resolves(self):
        assert capture.resolve_sensor_stack("bridge", capture_policy_cam=True) == (
            "rgb_full", "depth_full", "depth_policy")

    def test_token_list_resolves_and_dedups(self):
        assert capture.resolve_sensor_stack(
            "depth_policy,rgb_full,rgb_full", capture_policy_cam=False,
        ) == ("rgb_full", "depth_policy")

    def test_none_falls_back_to_bool(self):
        assert capture.resolve_sensor_stack(None, capture_policy_cam=True) == (
            "rgb_full", "rgb_policy")
        assert capture.resolve_sensor_stack(None, capture_policy_cam=False) == (
            "rgb_full",)

    def test_unknown_token_raises(self):
        with pytest.raises(SystemExit, match="unknown token"):
            capture.resolve_sensor_stack("lidar", capture_policy_cam=False)


class TestPendingCellsRaise:
    @pytest.mark.parametrize(
        "driver,mission_source,tier",
        [
            ("bridge", "scene-metadata", "Tier 2"),
            ("bridge", "queue", "Tier 2"),
            ("scripted", "captioner", "Tier 3"),
            ("scripted", "coverage", "Tier 3"),
        ],
    )
    def test_pending_cell_raises_notimplemented(self, driver, mission_source, tier):
        # Queue cells need --mission-queue to pass validation, so build
        # args accordingly.
        extra = {}
        if mission_source == "queue":
            extra["mission_queue"] = "/tmp/q.yaml"
        with pytest.raises(NotImplementedError, match=tier):
            capture.main(_base_args(driver, mission_source, **extra))


class TestInvalidCombinations:
    def test_teleop_captioner_rejected(self):
        with pytest.raises(SystemExit, match="Invalid combination"):
            capture.main(_base_args("teleop", "captioner"))

    def test_teleop_coverage_rejected(self):
        with pytest.raises(SystemExit, match="Invalid combination"):
            capture.main(_base_args("teleop", "coverage"))

    def test_bridge_captioner_rejected(self):
        with pytest.raises(SystemExit, match="Invalid combination"):
            capture.main(_base_args("bridge", "captioner"))

    def test_bridge_coverage_rejected(self):
        with pytest.raises(SystemExit, match="Invalid combination"):
            capture.main(_base_args("bridge", "coverage"))

    def test_scripted_scene_metadata_rejected(self):
        with pytest.raises(SystemExit, match="Invalid combination"):
            capture.main(_base_args("scripted", "scene-metadata"))


class TestFlagDependencies:
    def test_queue_requires_mission_queue_flag(self):
        with pytest.raises(SystemExit, match="--mission-queue"):
            capture.main(_base_args("bridge", "queue"))

    def test_num_envs_gt_one_rejected_for_teleop(self):
        with pytest.raises(SystemExit, match="--num-envs"):
            capture.main(_base_args("teleop", "scene-metadata", num_envs=4))

    def test_num_envs_gt_one_rejected_for_bridge(self):
        with pytest.raises(SystemExit, match="--num-envs"):
            capture.main(_base_args("bridge", "scene-metadata", num_envs=2))

    def test_inject_bad_grounding_rejected_for_teleop(self):
        with pytest.raises(SystemExit, match="--inject-bad-grounding"):
            capture.main(_base_args(
                "teleop", "scene-metadata",
                inject_bad_grounding="wrong_room",
            ))
