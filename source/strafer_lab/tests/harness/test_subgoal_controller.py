"""Unit tests for the subgoal-policy inference plumbing.

Pure-Python: the per-tick path subgoal -> assemble_observation -> policy ->
interpret_action is exercised with a fake policy and a hand-built base
observation, no GPU or checkpoint. Asserts the obs carries the subgoal
fields and the action maps to the mecanum body twist.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strafer_lab.tools import subgoal_controller as sc
from strafer_shared.constants import GOAL_DIST_SCALE, MAX_ANGULAR_VEL, MAX_LINEAR_VEL
from strafer_shared.policy_interface import PolicyVariant


def _base_fields():
    return {
        "imu_accel": np.zeros(3, dtype=np.float32),
        "imu_gyro": np.zeros(3, dtype=np.float32),
        "encoder_vels_ticks": np.zeros(4, dtype=np.float32),
        "body_velocity_xy": np.zeros(2, dtype=np.float32),
        "last_action": np.zeros(3, dtype=np.float32),
    }


class TestSubgoalFields:
    def test_subgoal_ahead(self):
        f = sc.subgoal_observation_fields((0.0, 0.0), 0.0, (2.0, 0.0))
        assert f["subgoal_relative"] == pytest.approx([2.0, 0.0])
        assert f["subgoal_distance"] == pytest.approx([2.0])
        assert f["subgoal_heading_to_subgoal"] == pytest.approx([0.0])

    def test_subgoal_to_the_left(self):
        f = sc.subgoal_observation_fields((0.0, 0.0), 0.0, (0.0, 1.0))
        # Body frame: a subgoal at world +y is straight ahead-left.
        assert f["subgoal_relative"] == pytest.approx([0.0, 1.0], abs=1e-6)
        assert f["subgoal_heading_to_subgoal"] == pytest.approx([math.pi / 2])

    def test_body_frame_rotation_under_yaw(self):
        # Robot yawed +90deg; a world +x subgoal sits on the robot's right.
        f = sc.subgoal_observation_fields((0.0, 0.0), math.pi / 2, (1.0, 0.0))
        assert f["subgoal_relative"] == pytest.approx([0.0, -1.0], abs=1e-6)
        assert f["subgoal_heading_to_subgoal"] == pytest.approx([-math.pi / 2])


class TestStepController:
    def test_obs_is_19_dim_with_subgoal_fields(self):
        captured = {}

        def fake_policy(obs):
            captured["obs"] = np.asarray(obs)
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        sc.step_subgoal_controller(
            fake_policy, _base_fields(), (0.0, 0.0), 0.0, (2.0, 0.0),
        )
        obs = captured["obs"]
        assert obs.shape == (PolicyVariant.NOCAM_SUBGOAL.obs_dim,)
        # subgoal_relative occupies indices 10:12, scaled by GOAL_DIST_SCALE.
        assert obs[10:12] == pytest.approx([2.0 * GOAL_DIST_SCALE, 0.0])
        assert obs[12] == pytest.approx(2.0 * GOAL_DIST_SCALE)

    def test_action_maps_to_body_twist(self):
        def fake_policy(obs):
            return np.array([0.5, -0.25, 0.1], dtype=np.float32)

        vx, vy, wz = sc.step_subgoal_controller(
            fake_policy, _base_fields(), (0.0, 0.0), 0.0, (2.0, 0.0),
        )
        assert vx == pytest.approx(0.5 * MAX_LINEAR_VEL)
        assert vy == pytest.approx(-0.25 * MAX_LINEAR_VEL)
        assert wz == pytest.approx(0.1 * MAX_ANGULAR_VEL)

    def test_missing_base_field_raises(self):
        bad = _base_fields()
        del bad["imu_gyro"]
        with pytest.raises(KeyError):
            sc.step_subgoal_controller(
                lambda obs: np.zeros(3), bad, (0.0, 0.0), 0.0, (1.0, 0.0),
            )
