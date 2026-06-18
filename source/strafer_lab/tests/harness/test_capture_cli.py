"""CLI validation tests for source/strafer_lab/scripts/capture.py.

Exercise argparse + the (driver, mission-source) cross-product validator
without launching Isaac Sim. Wired cells subprocess their driver script
(``teleop_capture.py`` / ``run_sim_in_the_loop.py``); we inject a stub
runner so the test never spawns Isaac Sim. Other cells either raise
``NotImplementedError`` (driver not shipped) or ``SystemExit`` (invalid
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


class TestBridgeDispatchPath:
    def _run(self, runner, *args, **kwargs):
        parser = capture._build_parser()
        ns, extra = parser.parse_known_args(_base_args(*args, **kwargs))
        capture._validate(ns)
        return capture._dispatch(ns, extra_args=tuple(extra), runner=runner)

    def test_scene_metadata_cell_dispatches_to_bridge_driver(self):
        runner = _StubRunner()
        rc = self._run(runner, "bridge", "scene-metadata")
        assert rc == 0
        argv = runner.calls[0]
        assert argv[0] == sys.executable
        assert argv[1].endswith("run_sim_in_the_loop.py")
        assert argv[argv.index("--mode") + 1] == "harness"
        assert argv[argv.index("--scene-name") + 1] == "scene_test"
        assert argv[argv.index("--output") + 1] == "/tmp/test_dataset_doesnotexist"
        # Default sensor stack for the bridge driver is the bridge preset.
        assert argv[argv.index("--sensors") + 1] == "rgb_full,depth_full,depth_policy"
        # The queue flag is absent for the scene-metadata cell.
        assert "--mission-queue" not in argv
        # Detections default on → no --no-detections flag.
        assert "--no-detections" not in argv

    def test_queue_cell_forwards_mission_queue(self):
        runner = _StubRunner()
        rc = self._run(runner, "bridge", "queue", mission_queue="/tmp/q.yaml")
        assert rc == 0
        argv = runner.calls[0]
        assert argv[argv.index("--mission-queue") + 1] == "/tmp/q.yaml"

    def test_no_detections_forwarded(self):
        runner = _StubRunner()
        parser = capture._build_parser()
        argv_in = _base_args("bridge", "scene-metadata") + ["--no-detections"]
        ns, extra = parser.parse_known_args(argv_in)
        capture._validate(ns)
        capture._dispatch(ns, extra_args=tuple(extra), runner=runner)
        assert "--no-detections" in runner.calls[0]

    def test_injection_flags_forwarded(self):
        runner = _StubRunner()
        rc = self._run(
            runner, "bridge", "scene-metadata",
            inject_bad_grounding="wrong_instance",
            inject_bad_grounding_prob=0.5,
        )
        assert rc == 0
        argv = runner.calls[0]
        assert argv[argv.index("--inject-bad-grounding") + 1] == "wrong_instance"
        assert argv[argv.index("--inject-bad-grounding-prob") + 1] == "0.5"

    def test_injection_off_not_forwarded(self):
        runner = _StubRunner()
        self._run(runner, "bridge", "scene-metadata")
        assert "--inject-bad-grounding" not in runner.calls[0]

    def test_sensors_override_forwarded(self):
        runner = _StubRunner()
        self._run(runner, "bridge", "scene-metadata", sensors="rgb_full,depth_full")
        argv = runner.calls[0]
        assert argv[argv.index("--sensors") + 1] == "rgb_full,depth_full"

    def test_unknown_args_forwarded_to_child(self):
        runner = _StubRunner()
        parser = capture._build_parser()
        argv_in = _base_args("bridge", "scene-metadata") + [
            "--headless", "--max-missions", "3",
        ]
        ns, extra = parser.parse_known_args(argv_in)
        capture._validate(ns)
        capture._dispatch(ns, extra_args=tuple(extra), runner=runner)
        child_argv = runner.calls[0]
        assert "--headless" in child_argv
        assert "--max-missions" in child_argv and "3" in child_argv


class TestPendingCellsRaise:
    @pytest.mark.parametrize(
        "driver,mission_source",
        [
            ("scripted", "queue"),
            ("scripted", "captioner"),
            ("scripted", "coverage"),
        ],
    )
    def test_scripted_cells_raise_notimplemented(self, driver, mission_source):
        # Queue cells need --mission-queue to pass validation, so build
        # args accordingly.
        extra = {}
        if mission_source == "queue":
            extra["mission_queue"] = "/tmp/q.yaml"
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            capture.main(_base_args(driver, mission_source, **extra))

    def test_teleop_queue_names_its_functional_gate(self):
        # teleop x queue is pending for a specific reason: the teleop driver
        # reads scene metadata, not a queue, and the queue producer is
        # unshipped. The message must say so without naming a brief/tier.
        with pytest.raises(NotImplementedError, match="reads targets from scene"):
            capture.main(_base_args("teleop", "queue", mission_queue="/tmp/q.yaml"))


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

    def test_detections_rejected_for_teleop(self):
        with pytest.raises(SystemExit, match="--detections"):
            capture.main(
                _base_args("teleop", "scene-metadata") + ["--detections"],
            )
