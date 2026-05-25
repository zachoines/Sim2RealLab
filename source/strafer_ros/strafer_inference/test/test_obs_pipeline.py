"""Unit tests for the rclpy-free obs-pipeline helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from strafer_inference.obs_pipeline import (
    body_frame_goal,
    build_raw_obs_dict,
    downsample_depth,
    joint_state_to_wheel_vels,
    quaternion_to_yaw,
)
from strafer_shared.constants import (
    DEPTH_HEIGHT,
    DEPTH_MAX,
    DEPTH_MIN,
    DEPTH_NEARFIELD_FILL,
    DEPTH_WIDTH,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
    WHEEL_JOINT_NAMES,
)
from strafer_shared.policy_interface import (
    PolicyVariant,
    assemble_observation,
)


# =============================================================================
# downsample_depth
# =============================================================================


class TestDownsampleDepth:
    """The depth pipeline mirrors mdp/observations.py:depth_image at the
    deterministic steps (steps 1, 2, 4 in the brief). Block-averaging is
    exact-integer (640/80=8, 360/60=6) so it matches cv2.INTER_AREA to
    within float roundoff.
    """

    def test_constant_field_passes_through_scaled(self):
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH), 3.0, dtype=np.float32
        )
        out = downsample_depth(raw)
        assert out.shape == (DEPTH_HEIGHT * DEPTH_WIDTH,)
        np.testing.assert_allclose(out, 3.0 / DEPTH_MAX, atol=1e-6)

    def test_nan_and_inf_replaced_with_max_then_scaled(self):
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH), np.inf, dtype=np.float32
        )
        raw[0, 0] = np.nan
        out = downsample_depth(raw)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_nearfield_fill_applied_below_clip(self):
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH),
            DEPTH_MIN - 0.05,
            dtype=np.float32,
        )
        out = downsample_depth(raw)
        np.testing.assert_allclose(
            out, DEPTH_NEARFIELD_FILL / DEPTH_MAX, atol=1e-6
        )

    def test_block_average_matches_manual_reduction(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(
            DEPTH_MIN + 0.1, DEPTH_MAX - 0.1,
            size=(PERCEPTION_HEIGHT, PERCEPTION_WIDTH),
        ).astype(np.float32)
        out = downsample_depth(raw)

        block_h = PERCEPTION_HEIGHT // DEPTH_HEIGHT
        block_w = PERCEPTION_WIDTH // DEPTH_WIDTH
        expected = (
            raw.reshape(DEPTH_HEIGHT, block_h, DEPTH_WIDTH, block_w)
            .mean(axis=(1, 3))
            / DEPTH_MAX
        ).astype(np.float32).reshape(-1)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_clamps_above_max(self):
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH),
            DEPTH_MAX * 2,
            dtype=np.float32,
        )
        out = downsample_depth(raw)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_rejects_wrong_shape(self):
        raw = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected raw depth shape"):
            downsample_depth(raw)

    def test_output_dtype_float32(self):
        raw = np.zeros((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), dtype=np.float32)
        out = downsample_depth(raw)
        assert out.dtype == np.float32

    def test_output_dim_matches_depth_field(self):
        """The DEPTH variant's depth_image field is 4800 dims; the
        helper must produce exactly that.
        """
        depth_field = next(
            f for f in PolicyVariant.DEPTH.fields if f.key == "depth_image"
        )
        raw = np.zeros((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), dtype=np.float32)
        out = downsample_depth(raw)
        assert out.shape[0] == depth_field.dims


# =============================================================================
# body_frame_goal
# =============================================================================


class TestBodyFrameGoal:
    def test_zero_yaw_identity(self):
        rel, dist, head = body_frame_goal(
            goal_map_xy=(1.0, 0.0),
            base_in_map_xy=(0.0, 0.0),
            base_in_map_yaw=0.0,
        )
        np.testing.assert_allclose(rel, [1.0, 0.0], atol=1e-6)
        assert dist == pytest.approx(1.0)
        assert head == pytest.approx(0.0)

    def test_yaw_90_degrees_rotates_goal_to_negative_y(self):
        """Robot facing +Y (yaw=90°). A goal at (1, 0) in map sits on
        the robot's -Y body axis.
        """
        rel, dist, head = body_frame_goal(
            goal_map_xy=(1.0, 0.0),
            base_in_map_xy=(0.0, 0.0),
            base_in_map_yaw=math.pi / 2,
        )
        np.testing.assert_allclose(rel, [0.0, -1.0], atol=1e-6)
        assert dist == pytest.approx(1.0)
        assert head == pytest.approx(-math.pi / 2)

    def test_translation_subtracts_base_position(self):
        rel, dist, head = body_frame_goal(
            goal_map_xy=(3.0, 2.0),
            base_in_map_xy=(1.0, 0.0),
            base_in_map_yaw=0.0,
        )
        np.testing.assert_allclose(rel, [2.0, 2.0], atol=1e-6)
        assert dist == pytest.approx(math.hypot(2.0, 2.0))
        assert head == pytest.approx(math.atan2(2.0, 2.0))

    def test_inverse_yaw_round_trip(self):
        """Rotating the goal into body frame and back lands on the
        original map-frame offset.
        """
        for yaw in (-math.pi / 3, 0.1, 1.2, math.pi - 0.1):
            rel, _, _ = body_frame_goal(
                goal_map_xy=(2.5, -1.0),
                base_in_map_xy=(0.5, 0.5),
                base_in_map_yaw=yaw,
            )
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            back_dx = cos_y * rel[0] - sin_y * rel[1]
            back_dy = sin_y * rel[0] + cos_y * rel[1]
            assert back_dx == pytest.approx(2.0)
            assert back_dy == pytest.approx(-1.5)

    def test_output_dtype_float32(self):
        rel, _, _ = body_frame_goal(
            goal_map_xy=(1.0, 0.0),
            base_in_map_xy=(0.0, 0.0),
            base_in_map_yaw=0.0,
        )
        assert rel.dtype == np.float32


# =============================================================================
# quaternion_to_yaw
# =============================================================================


class TestQuaternionToYaw:
    def test_identity_quaternion_is_zero_yaw(self):
        assert quaternion_to_yaw(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("yaw", [-math.pi / 2, -0.5, 0.5, math.pi / 2])
    def test_round_trip_through_quaternion(self, yaw):
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)
        assert quaternion_to_yaw(0.0, 0.0, qz, qw) == pytest.approx(yaw)


# =============================================================================
# joint_state_to_wheel_vels
# =============================================================================


class TestJointStateOrdering:
    def test_picks_FL_FR_RL_RR_in_canonical_order(self):
        names = list(reversed(WHEEL_JOINT_NAMES))
        velocities = [4.0, 3.0, 2.0, 1.0]
        out = joint_state_to_wheel_vels(names, velocities)
        # reversed names: RR, RL, FR, FL -> velocities 4, 3, 2, 1
        # canonical order: FL=1, FR=2, RL=3, RR=4
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0])

    def test_extra_joints_ignored(self):
        names = ["spine_joint", *WHEEL_JOINT_NAMES, "lift"]
        velocities = [99.0, 1.0, 2.0, 3.0, 4.0, 77.0]
        out = joint_state_to_wheel_vels(names, velocities)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0])

    def test_missing_wheel_raises(self):
        names = list(WHEEL_JOINT_NAMES[:3])
        velocities = [1.0, 2.0, 3.0]
        with pytest.raises(KeyError, match="missing wheel joints"):
            joint_state_to_wheel_vels(names, velocities)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            joint_state_to_wheel_vels(list(WHEEL_JOINT_NAMES), [1.0, 2.0])


# =============================================================================
# build_raw_obs_dict + assemble_observation round-trip
# =============================================================================


class TestRawDictAssembly:
    def _make_raw(self, **overrides) -> dict:
        defaults = dict(
            imu_accel=(0.1, 0.2, 9.8),
            imu_gyro=(0.01, -0.02, 0.03),
            wheel_vels_rad_s=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            goal_relative_xy=np.array([1.5, -0.5], dtype=np.float32),
            goal_distance=math.hypot(1.5, -0.5),
            goal_heading_to_goal=math.atan2(-0.5, 1.5),
            body_velocity_xy=(0.5, 0.0),
            last_action=np.zeros(3, dtype=np.float32),
            depth_flat_normalized=np.full(
                DEPTH_HEIGHT * DEPTH_WIDTH, 0.5, dtype=np.float32
            ),
        )
        defaults.update(overrides)
        return build_raw_obs_dict(**defaults)

    def test_dict_has_every_depth_variant_field(self):
        raw = self._make_raw()
        for field in PolicyVariant.DEPTH.fields:
            assert field.key in raw, f"missing obs field {field.key}"

    def test_each_field_array_length_matches_spec(self):
        raw = self._make_raw()
        for field in PolicyVariant.DEPTH.fields:
            arr = np.asarray(raw[field.key]).ravel()
            assert arr.shape[0] == field.dims, (
                f"field {field.key}: got {arr.shape[0]}, expected {field.dims}"
            )

    def test_assemble_observation_round_trips_to_depth_obs_dim(self):
        raw = self._make_raw()
        obs = assemble_observation(raw, PolicyVariant.DEPTH)
        assert obs.shape == (PolicyVariant.DEPTH.obs_dim,)
        assert obs.dtype == np.float32

    def test_last_action_zero_on_first_tick_propagates_into_obs(self):
        """last_action sits at field offset NOCAM_FIELDS minus the trailing
        slot — assembled obs has zero in that slice on the first tick.
        """
        raw = self._make_raw(last_action=np.zeros(3, dtype=np.float32))
        obs = assemble_observation(raw, PolicyVariant.DEPTH)

        offset = 0
        for field in PolicyVariant.DEPTH.fields:
            if field.key == "last_action":
                break
            offset += field.dims
        np.testing.assert_allclose(obs[offset:offset + 3], 0.0)

    def test_last_action_raw_minus_1_to_1_not_velocity(self):
        """The brief: last_action must hold the *raw* [-1, 1]^3 policy
        output, NOT the post-interpret_action velocity. Feeding a value
        in [-1, 1] should appear unchanged (scale=1.0) in the assembled
        slice.
        """
        cached = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        raw = self._make_raw(last_action=cached)
        obs = assemble_observation(raw, PolicyVariant.DEPTH)

        offset = 0
        for field in PolicyVariant.DEPTH.fields:
            if field.key == "last_action":
                # The last_action field is intentionally unscaled in
                # policy_interface (scale=1.0); the slice must equal
                # the cached vector byte-for-byte.
                np.testing.assert_allclose(
                    obs[offset:offset + field.dims], cached
                )
                return
            offset += field.dims
        pytest.fail("last_action field not found in DEPTH variant")
