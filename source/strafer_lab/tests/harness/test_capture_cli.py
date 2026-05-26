"""CLI validation tests for Scripts/capture.py.

Exercise argparse + the (driver, mission-source) cross-product validator
without launching Isaac Sim. The scaffolding commit's Tier 1 path
returns 0; all other cells raise NotImplementedError.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS_DIR = _REPO_ROOT / "Scripts"
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


class TestTier1ScaffoldingPath:
    def test_teleop_scene_metadata_returns_zero(self, capsys):
        rc = capture.main(_base_args("teleop", "scene-metadata"))
        assert rc == 0
        err = capsys.readouterr().err
        assert "Tier 1" in err
        assert "scaffolding" in err


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
