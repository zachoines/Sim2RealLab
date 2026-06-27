"""Pure-Python tests for the scripted coverage capture driver.

Exercise the driver's CLI surface, the policy-variant -> capture-env mapping,
and the per-leg path planner (``_leg_path``) without launching Isaac Sim — the
Isaac imports live inside ``main()``, so the module is importable plain. The
live env loop (the CaptureSubgoalCommand rolling the subgoal + the rsl_rl
runner stepping) is covered by the Kit smoke, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import coverage_capture as cc  # noqa: E402  (post-path-mutation import)


class _FakePlanError(Exception):
    pass


class TestCli:
    def test_checkpoint_required(self):
        parser = cc._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_known_args(["--scene", "s", "--output", "/tmp/o"])

    def test_defaults(self):
        parser = cc._build_parser()
        args, _ = parser.parse_known_args(
            ["--scene", "s", "--output", "/tmp/o", "--checkpoint", "/m.pt"],
        )
        assert args.policy_variant == "nocam_subgoal"
        assert args.env is None
        assert args.num_envs == 1
        assert args.lookahead_m is None

    def test_unknown_flags_pass_through(self):
        # AppLauncher / pass-through flags must survive parse_known_args.
        parser = cc._build_parser()
        _, extra = parser.parse_known_args(
            ["--scene", "s", "--output", "/tmp/o", "--checkpoint", "/m.pt",
             "--headless", "--device", "cpu"],
        )
        assert "--headless" in extra and "--device" in extra


class TestVariantEnvMapping:
    def test_nocam_subgoal_maps_to_coverage_env(self):
        assert (
            cc._CAPTURE_ENV_BY_VARIANT["nocam_subgoal"]
            == "Isaac-Strafer-Nav-Capture-Coverage-v0"
        )


class TestSceneDirFor:
    def test_resolves_scene_directory(self):
        usd = Path("/x/Assets/generated/scenes/scene_foo_000/scene_foo_000.usdc")
        assert cc._scene_dir_for(usd).name == "scene_foo_000"


class TestLegPath:
    """``_leg_path`` stages one approach_distance behind the viewpoint along the
    approach heading, then a straight final segment to the viewpoint."""

    def _free(self):
        return np.ones((200, 200), dtype=bool)

    def test_staged_path_ends_at_target_facing_heading(self):
        target = np.array([5.0, 5.0], dtype=np.float32)
        calls: list[tuple] = []

        def fake_plan_path(start, goal, free_space, *, grid_res, grid_origin_xy):
            calls.append((tuple(start), tuple(goal)))
            return np.asarray([start, goal], dtype=np.float32)

        leg = cc._leg_path(
            np.array([0.0, 0.0], dtype=np.float32), target, 0.0, self._free(),
            grid_res=0.05, grid_origin_xy=(0.0, 0.0), approach_distance_m=0.6,
            plan_path=fake_plan_path, error_cls=_FakePlanError,
        )
        # Last waypoint is the viewpoint; the penultimate is the staging point
        # one approach_distance behind it along the heading (heading 0 -> -x).
        assert np.allclose(leg[-1], target)
        assert np.allclose(leg[-2], [target[0] - 0.6, target[1]], atol=1e-5)

    def test_falls_back_to_direct_when_staging_unplannable(self):
        target = np.array([5.0, 5.0], dtype=np.float32)
        state = {"n": 0}

        def fake_plan_path(start, goal, free_space, *, grid_res, grid_origin_xy):
            state["n"] += 1
            if state["n"] == 1:  # staged approach fails
                raise _FakePlanError
            return np.asarray([start, goal], dtype=np.float32)

        leg = cc._leg_path(
            np.array([0.0, 0.0], dtype=np.float32), target, 1.0, self._free(),
            grid_res=0.05, grid_origin_xy=(0.0, 0.0), approach_distance_m=0.6,
            plan_path=fake_plan_path, error_cls=_FakePlanError,
        )
        assert np.allclose(leg[-1], target)

    def test_returns_none_when_unplannable(self):
        def fake_plan_path(start, goal, free_space, *, grid_res, grid_origin_xy):
            raise _FakePlanError

        leg = cc._leg_path(
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([5.0, 5.0], dtype=np.float32), 0.0, self._free(),
            grid_res=0.05, grid_origin_xy=(0.0, 0.0), approach_distance_m=0.6,
            plan_path=fake_plan_path, error_cls=_FakePlanError,
        )
        assert leg is None
