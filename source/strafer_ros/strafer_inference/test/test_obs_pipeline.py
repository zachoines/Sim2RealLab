"""Unit tests for the rclpy-free obs-pipeline helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from strafer_inference.obs_pipeline import (
    MAJORITY_INVALID_MIN,
    DepthDecodeError,
    body_frame_goal,
    build_raw_obs_dict,
    count_majority_invalid_cells,
    decode_depth_image,
    downsample_depth,
    joint_state_to_wheel_vels,
    l1_clamp_velocity,
    quat_apply_inverse_xy,
    quaternion_to_yaw,
)
from strafer_shared.constants import (
    DEPTH_HEIGHT,
    DEPTH_MAX,
    DEPTH_MIN,
    DEPTH_NEARFIELD_FILL,
    DEPTH_SCALE,
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
    """The depth pipeline mirrors mdp/observations.py:depth_image's
    deterministic steps and returns raw meters. Block-averaging is
    exact-integer (640/80=8, 360/45=8) so it matches cv2.INTER_AREA to
    within float roundoff.
    """

    def test_constant_field_passes_through_in_meters(self):
        # Raw meters, not [0, 1]: DEPTH_SCALE is applied once by
        # assemble_observation downstream.
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH), 3.0, dtype=np.float32
        )
        out = downsample_depth(raw)
        assert out.shape == (DEPTH_HEIGHT * DEPTH_WIDTH,)
        np.testing.assert_allclose(out, 3.0, atol=1e-6)

    def test_nan_and_inf_replaced_with_max_meters(self):
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH), np.inf, dtype=np.float32
        )
        raw[0, 0] = np.nan
        out = downsample_depth(raw)
        np.testing.assert_allclose(out, DEPTH_MAX, atol=1e-6)

    def test_nearfield_fill_applied_below_clip(self):
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH),
            DEPTH_MIN - 0.05,
            dtype=np.float32,
        )
        out = downsample_depth(raw)
        # Fill value is in meters (matches mdp.depth_image); scaled once later.
        np.testing.assert_allclose(out, DEPTH_NEARFIELD_FILL, atol=1e-6)

    def test_block_median_matches_manual_reduction(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(
            DEPTH_MIN + 0.1, DEPTH_MAX - 0.1,
            size=(PERCEPTION_HEIGHT, PERCEPTION_WIDTH),
        ).astype(np.float32)
        out = downsample_depth(raw)

        block_h = PERCEPTION_HEIGHT // DEPTH_HEIGHT
        block_w = PERCEPTION_WIDTH // DEPTH_WIDTH
        expected = np.median(
            raw.reshape(DEPTH_HEIGHT, block_h, DEPTH_WIDTH, block_w),
            axis=(1, 3),
        ).astype(np.float32).reshape(-1)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_block_reduction_is_not_the_mean(self):
        """Guard against a silent revert to the block mean.

        A block straddling the far-clip validity boundary is where the two
        reductions diverge.
        """
        block_h = PERCEPTION_HEIGHT // DEPTH_HEIGHT
        block_w = PERCEPTION_WIDTH // DEPTH_WIDTH
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH), 2.0, dtype=np.float32
        )
        # 3/4 wall at 2.0 m, 1/4 culled: the mean would return 3.0 m.
        raw[0:block_h // 2, 0:block_w // 2] = np.inf
        out = downsample_depth(raw).reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
        assert out[0, 0] == pytest.approx(2.0)
        mean_answer = np.mean(
            np.where(np.isfinite(raw[:block_h, :block_w]),
                     raw[:block_h, :block_w], DEPTH_MAX)
        )
        assert mean_answer == pytest.approx(3.0)

    def test_majority_sky_block_reads_as_out_of_range(self):
        """A mostly-culled block must not be pulled toward a minority near
        surface."""
        block_h = PERCEPTION_HEIGHT // DEPTH_HEIGHT
        block_w = PERCEPTION_WIDTH // DEPTH_WIDTH
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH), np.inf, dtype=np.float32
        )
        raw[0:block_h // 2, 0:block_w // 2] = 1.0
        out = downsample_depth(raw).reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
        assert out[0, 0] == pytest.approx(DEPTH_MAX)

    def test_clamps_above_max(self):
        raw = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH),
            DEPTH_MAX * 2,
            dtype=np.float32,
        )
        out = downsample_depth(raw)
        np.testing.assert_allclose(out, DEPTH_MAX, atol=1e-6)

    def test_rejects_wrong_shape(self):
        raw = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected raw depth shape"):
            downsample_depth(raw)

    def test_output_dtype_float32(self):
        raw = np.zeros((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), dtype=np.float32)
        out = downsample_depth(raw)
        assert out.dtype == np.float32

    def test_output_dim_matches_depth_field(self):
        """The DEPTH variant's depth_image field is DEPTH_WIDTH*DEPTH_HEIGHT
        (80×45 = 3600) dims; the helper must produce exactly that.
        """
        depth_field = next(
            f for f in PolicyVariant.DEPTH.fields if f.key == "depth_image"
        )
        raw = np.zeros((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), dtype=np.float32)
        out = downsample_depth(raw)
        assert out.shape[0] == depth_field.dims


# =============================================================================
# downsample_depth: explicit validity (Z16) vs the unchanged 32FC1 path
# =============================================================================


def _frozen_downsample_depth(
    depth_meters,
    *,
    max_depth=DEPTH_MAX,
    nearfield_clip=DEPTH_MIN,
    nearfield_fill=DEPTH_NEARFIELD_FILL,
):
    """downsample_depth as it stood before the validity mask, copied verbatim
    so the no-mask path is pinned byte for byte against it."""
    bh = PERCEPTION_HEIGHT // DEPTH_HEIGHT
    bw = PERCEPTION_WIDTH // DEPTH_WIDTH
    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.shape != (PERCEPTION_HEIGHT, PERCEPTION_WIDTH):
        raise ValueError(
            f"Expected raw depth shape ({PERCEPTION_HEIGHT}, "
            f"{PERCEPTION_WIDTH}); got {depth.shape}"
        )
    depth = np.where(np.isfinite(depth), depth, np.float32(max_depth))
    depth = np.median(
        depth.reshape(DEPTH_HEIGHT, bh, DEPTH_WIDTH, bw),
        axis=(1, 3),
    )
    depth = np.where(
        depth < nearfield_clip, np.float32(nearfield_fill), depth
    )
    depth = np.clip(depth, 0.0, max_depth)
    return depth.reshape(-1).astype(np.float32, copy=False)


def _set_block(frame: np.ndarray, row: int, col: int, values) -> None:
    """Write 64 values (row-major) into policy cell (row, col)'s 8x8 block."""
    bh = PERCEPTION_HEIGHT // DEPTH_HEIGHT
    bw = PERCEPTION_WIDTH // DEPTH_WIDTH
    frame[row * bh:(row + 1) * bh, col * bw:(col + 1) * bw] = np.asarray(
        values, dtype=frame.dtype
    ).reshape(bh, bw)


def _z16_frame(fill_mm: int = 2500) -> np.ndarray:
    return np.full((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), fill_mm, np.uint16)


def _z16_downsample(raw_mm: np.ndarray) -> np.ndarray:
    meters, valid = decode_depth_image(
        encoding="16UC1",
        data=raw_mm.tobytes(),
        height=raw_mm.shape[0],
        width=raw_mm.shape[1],
    )
    return downsample_depth(meters, valid_mask=valid).reshape(
        DEPTH_HEIGHT, DEPTH_WIDTH
    )


class TestDownsampleDepthNoMaskRegression:
    """Sim lane: a 32FC1 frame (no mask) is byte-identical to before."""

    @pytest.mark.parametrize("seed", range(8))
    def test_randomized_frames_are_byte_identical(self, seed):
        rng = np.random.default_rng(seed)
        shape = (PERCEPTION_HEIGHT, PERCEPTION_WIDTH)
        frame = rng.uniform(0.0, 9.0, size=shape).astype(np.float32)
        kind = rng.integers(0, 6, size=shape)
        frame[kind == 1] = np.inf
        frame[kind == 2] = np.nan
        frame[kind == 3] = 0.0
        frame[kind == 4] = rng.uniform(0.0, DEPTH_MIN, size=shape)[kind == 4]
        # Whole blocks of a single kind too, so the per-block extremes are hit.
        frame[:8, :8] = np.inf
        frame[:8, 8:16] = np.nan
        frame[:8, 16:24] = 0.0
        frame[:8, 24:32] = 0.3
        frame[:8, 32:40] = 7.5
        frame[:8, 40:48] = -np.inf
        want = _frozen_downsample_depth(frame)
        for got in (
            downsample_depth(frame),
            downsample_depth(frame, valid_mask=None),
        ):
            assert got.dtype == want.dtype
            assert got.shape == want.shape
            assert got.tobytes() == want.tobytes()


class TestDownsampleDepthValidityMask:
    """The brief's pinned vectors: Z16 invalid (0) maps to DEPTH_MAX before
    the median, the training convention; a genuine sub-0.4 m return keeps
    the nearfield fill."""

    def test_all_zero_block_reads_max_depth(self):
        raw = _z16_frame()
        _set_block(raw, 0, 0, np.zeros(64))
        out = _z16_downsample(raw)
        assert out[0, 0] == np.float32(DEPTH_MAX)
        assert out[0, 0] != np.float32(DEPTH_NEARFIELD_FILL)

    def test_majority_zero_block_reads_max_depth(self):
        # 33 invalid + 31 valid at 1.5 m: the median lands on the far clamp.
        raw = _z16_frame()
        _set_block(raw, 5, 7, [0] * MAJORITY_INVALID_MIN + [1500] * 31)
        out = _z16_downsample(raw)
        assert out[5, 7] == np.float32(DEPTH_MAX)

    def test_majority_zero_block_with_near_valid_pixels_still_reads_max(self):
        # Even when the valid minority is itself sub-0.4 m: never 0.2 m.
        raw = _z16_frame()
        _set_block(raw, 1, 1, [0] * 40 + [300] * 24)
        out = _z16_downsample(raw)
        assert out[1, 1] == np.float32(DEPTH_MAX)

    def test_genuine_sub_nearfield_block_keeps_the_fill(self):
        raw = _z16_frame()
        _set_block(raw, 2, 3, np.full(64, 300))  # 0.3 m, every pixel valid
        out = _z16_downsample(raw)
        assert out[2, 3] == np.float32(DEPTH_NEARFIELD_FILL)

    def test_exactly_half_invalid_follows_numpys_even_median(self):
        # 32 invalid (-> 6.0) + 32 valid at 2.0 m: numpy averages the two
        # middle ranks, exactly as the training reduction would.
        raw = _z16_frame()
        _set_block(raw, 3, 4, [0] * 32 + [2000] * 32)
        out = _z16_downsample(raw)
        want = np.median(
            np.array([DEPTH_MAX] * 32 + [2.0] * 32, dtype=np.float32)
        )
        np.testing.assert_allclose(out[3, 4], want, atol=1e-6)
        assert 2.0 < out[3, 4] < DEPTH_MAX

    def test_valid_pixels_decode_to_metres(self):
        out = _z16_downsample(_z16_frame(2500))
        np.testing.assert_allclose(out, 2.5, atol=1e-6)

    def test_mask_overrides_a_finite_value_and_not_the_reverse(self):
        # A masked pixel reads max_depth whatever it holds; an unmasked
        # non-finite pixel is still rescued.
        depth = np.full((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), 1.0, np.float32)
        valid = np.ones_like(depth, dtype=bool)
        depth[:8, :8] = np.inf
        valid[:8, 8:16] = False
        out = downsample_depth(depth, valid_mask=valid).reshape(
            DEPTH_HEIGHT, DEPTH_WIDTH
        )
        assert out[0, 0] == np.float32(DEPTH_MAX)
        assert out[0, 1] == np.float32(DEPTH_MAX)
        assert out[0, 2] == np.float32(1.0)

    def test_all_valid_mask_matches_the_no_mask_path(self):
        rng = np.random.default_rng(7)
        depth = rng.uniform(0.0, 8.0, (PERCEPTION_HEIGHT, PERCEPTION_WIDTH))
        depth = depth.astype(np.float32)
        valid = np.ones_like(depth, dtype=bool)
        assert (
            downsample_depth(depth, valid_mask=valid).tobytes()
            == downsample_depth(depth).tobytes()
        )

    def test_rejects_a_mask_of_the_wrong_shape(self):
        depth = np.ones((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), np.float32)
        with pytest.raises(ValueError, match="valid_mask shape"):
            downsample_depth(depth, valid_mask=np.ones((4, 4), dtype=bool))


class TestCountMajorityInvalidCells:
    def test_counts_blocks_at_or_above_33_invalid(self):
        valid = np.ones((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), dtype=bool)
        _set_block(valid, 0, 0, [False] * 64)
        _set_block(valid, 0, 1, [False] * 33 + [True] * 31)
        _set_block(valid, 0, 2, [False] * 32 + [True] * 32)  # half: not forced
        assert MAJORITY_INVALID_MIN == 33
        assert count_majority_invalid_cells(valid) == 2

    def test_matches_the_cells_the_mask_forces_to_the_far_clamp(self):
        rng = np.random.default_rng(3)
        raw = rng.integers(500, 5000, (PERCEPTION_HEIGHT, PERCEPTION_WIDTH))
        raw = raw.astype(np.uint16)
        raw[rng.random(raw.shape) < 0.5] = 0
        meters, valid = decode_depth_image(
            encoding="16UC1", data=raw.tobytes(),
            height=raw.shape[0], width=raw.shape[1],
        )
        out = downsample_depth(meters, valid_mask=valid)
        # Valid values stay below 5 m, so only a forced cell reads 6.0.
        assert count_majority_invalid_cells(valid) == int(
            np.count_nonzero(out == np.float32(DEPTH_MAX))
        )


class TestDecodeDepthImage:
    H, W = 4, 6

    def test_16uc1_is_millimetres_with_zero_invalid(self):
        raw = np.array(
            [[0, 1000, 65535, 400, 399, 1]] * self.H, dtype=np.uint16
        )
        meters, valid = decode_depth_image(
            encoding="16UC1", data=raw.tobytes(), height=self.H, width=self.W,
            step=self.W * 2,
        )
        assert meters.dtype == np.float32
        assert meters.shape == (self.H, self.W)
        np.testing.assert_allclose(
            meters[0], [0.0, 1.0, 65.535, 0.4, 0.399, 0.001], rtol=1e-6
        )
        np.testing.assert_array_equal(
            valid[0], [False, True, True, True, True, True]
        )

    def test_mono16_decodes_like_16uc1(self):
        raw = np.arange(self.H * self.W, dtype=np.uint16).reshape(self.H, self.W)
        a = decode_depth_image(
            encoding="16UC1", data=raw.tobytes(), height=self.H, width=self.W
        )
        b = decode_depth_image(
            encoding="mono16", data=raw.tobytes(), height=self.H, width=self.W
        )
        assert a[0].tobytes() == b[0].tobytes()
        np.testing.assert_array_equal(a[1], b[1])

    def test_big_endian_z16(self):
        raw = np.full((self.H, self.W), 1234, dtype=">u2")
        meters, valid = decode_depth_image(
            encoding="16UC1", data=raw.tobytes(), height=self.H, width=self.W,
            is_bigendian=True,
        )
        np.testing.assert_allclose(meters, 1.234, rtol=1e-6)
        assert valid.all()

    def test_padded_rows_honour_step(self):
        raw = np.arange(self.H * self.W, dtype=np.uint16).reshape(self.H, self.W)
        step = self.W * 2 + 4
        padded = np.zeros((self.H, step), dtype=np.uint8)
        padded[:, : self.W * 2] = raw.view(np.uint8).reshape(self.H, -1)
        padded[:, self.W * 2 :] = 0xFF  # garbage in the pad must be ignored
        meters, _ = decode_depth_image(
            encoding="16UC1", data=padded.tobytes(), height=self.H,
            width=self.W, step=step,
        )
        np.testing.assert_allclose(meters, raw * 0.001, rtol=1e-6)

    def test_32fc1_is_the_unchanged_float_path(self):
        frame = np.array(
            [[np.inf, np.nan, 0.0, 0.3, 2.5, 9.0]] * self.H, dtype=np.float32
        )
        meters, valid = decode_depth_image(
            encoding="32FC1", data=frame.tobytes(), height=self.H, width=self.W
        )
        assert valid is None
        assert meters.dtype == np.float32
        want = np.frombuffer(frame.tobytes(), dtype=np.float32).reshape(
            self.H, self.W
        )
        assert meters.tobytes() == want.tobytes()

    def test_big_endian_32fc1(self):
        frame = np.full((self.H, self.W), 2.25, dtype=">f4")
        meters, valid = decode_depth_image(
            encoding="32FC1", data=frame.tobytes(), height=self.H,
            width=self.W, is_bigendian=True,
        )
        assert meters.dtype == np.float32
        np.testing.assert_array_equal(meters, 2.25)
        assert valid is None

    @pytest.mark.parametrize("encoding", ["rgb8", "8UC1", "32FC3", "", "16SC1"])
    def test_unknown_encoding_is_an_encoding_error(self, encoding):
        with pytest.raises(DepthDecodeError) as err:
            decode_depth_image(
                encoding=encoding, data=bytes(self.H * self.W * 4),
                height=self.H, width=self.W,
            )
        assert err.value.reason == "encoding"

    @pytest.mark.parametrize("encoding,nbytes", [
        ("16UC1", 4 * 6 * 2 - 2),
        ("32FC1", 4 * 6 * 4 + 3),
        ("16UC1", 0),
    ])
    def test_bad_length_is_a_shape_error(self, encoding, nbytes):
        with pytest.raises(DepthDecodeError) as err:
            decode_depth_image(
                encoding=encoding, data=bytes(nbytes),
                height=self.H, width=self.W,
            )
        assert err.value.reason == "shape"


class TestDepthSingleScaleParity:
    """Depth must reach the network at the value sim feeds it: meters *
    DEPTH_SCALE, applied once. downsample_depth returns meters and
    assemble_observation applies the single scale, matching the sim ObsTerm.
    """

    def _assembled_depth_slice(self, meters: float) -> np.ndarray:
        raw_field = np.full(
            (PERCEPTION_HEIGHT, PERCEPTION_WIDTH), meters, dtype=np.float32
        )
        depth_meters = downsample_depth(raw_field)
        raw = build_raw_obs_dict(
            variant=PolicyVariant.DEPTH,
            imu_accel=(0.0, 0.0, 0.0),
            imu_gyro=(0.0, 0.0, 0.0),
            wheel_vels_rad_s=np.zeros(4, dtype=np.float64),
            goal_relative_xy=np.zeros(2, dtype=np.float32),
            goal_distance=0.0,
            goal_heading_to_goal=0.0,
            body_velocity_xy=(0.0, 0.0),
            last_action=np.zeros(3, dtype=np.float32),
            depth_flat_meters=depth_meters,
        )
        obs = assemble_observation(raw, PolicyVariant.DEPTH)
        return obs[-(DEPTH_HEIGHT * DEPTH_WIDTH):]

    def test_three_meter_surface_reaches_network_at_sim_value(self):
        depth_slice = self._assembled_depth_slice(3.0)
        # 3.0 m scaled once by DEPTH_SCALE -> 0.5, the sim value.
        np.testing.assert_allclose(depth_slice, 3.0 * DEPTH_SCALE, atol=1e-6)
        np.testing.assert_allclose(depth_slice, 0.5, atol=1e-6)
        # A second scale would land here; guard against it.
        assert not np.allclose(
            depth_slice, 3.0 * DEPTH_SCALE * DEPTH_SCALE
        ), "downsample_depth must return raw meters, not normalized"

    def test_scaled_depth_stays_in_unit_range(self):
        # A max-range surface saturates to exactly 1.0 after the single scale.
        np.testing.assert_allclose(
            self._assembled_depth_slice(DEPTH_MAX), 1.0, atol=1e-6
        )


# =============================================================================
# body_frame_goal
# =============================================================================


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """Yaw-only quaternion as (x, y, z, w)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple:
    """roll/pitch/yaw (ZYX) -> (x, y, z, w) unit quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def _ref_rel_xy_via_matrix(quat_xyzw, dx, dy):
    """Independent reference for the full quaternion-inverse: build R(q) from
    the quaternion and apply its transpose (world->body) to (dx, dy, 0). A
    distinct code path from the production quaternion-formula helper, so the
    parity test below is not tautological.
    """
    x, y, z, w = quat_xyzw
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    rel = R.T @ np.array([dx, dy, 0.0])
    return rel[:2]


class TestBodyFrameGoal:
    def test_zero_yaw_identity(self):
        rel, dist, head = body_frame_goal(
            goal_map_xy=(1.0, 0.0),
            base_in_map_xy=(0.0, 0.0),
            base_in_map_quat=_yaw_quat(0.0),
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
            base_in_map_quat=_yaw_quat(math.pi / 2),
        )
        np.testing.assert_allclose(rel, [0.0, -1.0], atol=1e-6)
        assert dist == pytest.approx(1.0)
        assert head == pytest.approx(-math.pi / 2)

    def test_translation_subtracts_base_position(self):
        rel, dist, head = body_frame_goal(
            goal_map_xy=(3.0, 2.0),
            base_in_map_xy=(1.0, 0.0),
            base_in_map_quat=_yaw_quat(0.0),
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
                base_in_map_quat=_yaw_quat(yaw),
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
            base_in_map_quat=_yaw_quat(0.0),
        )
        assert rel.dtype == np.float32

    def test_flat_pose_full_quat_equals_yaw_only(self):
        """At zero tilt the full-quaternion rel coincides with the yaw-only
        rel — the two transforms must agree when roll = pitch = 0.
        """
        yaw = 0.7
        goal, base = (2.5, -1.0), (0.5, 0.5)
        dx, dy = goal[0] - base[0], goal[1] - base[1]
        rel, _, _ = body_frame_goal(
            goal_map_xy=goal, base_in_map_xy=base,
            base_in_map_quat=_yaw_quat(yaw),
        )
        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
        yaw_only = np.array([cos_y * dx - sin_y * dy, sin_y * dx + cos_y * dy])
        np.testing.assert_allclose(rel, yaw_only, atol=1e-6)

    @pytest.mark.parametrize("roll, pitch, yaw", [
        (math.radians(8.0), 0.0, 0.0),                                  # roll only
        (0.0, math.radians(5.0), 0.0),                                  # pitch only
        (math.radians(4.0), math.radians(-6.0), math.radians(30.0)),    # combined
    ])
    def test_rel_xy_matches_training_full_quat_under_tilt(self, roll, pitch, yaw):
        """rel_xy must track the full quaternion-inverse (matching training's
        goal_position_relative), verified against an independent
        rotation-matrix reference under roll / pitch / combined tilt.
        """
        quat = _rpy_to_quat_xyzw(roll, pitch, yaw)
        goal, base = (2.5, -1.0), (0.5, 0.5)
        dx, dy = goal[0] - base[0], goal[1] - base[1]
        rel, _, _ = body_frame_goal(
            goal_map_xy=goal, base_in_map_xy=base, base_in_map_quat=quat,
        )
        ref = _ref_rel_xy_via_matrix(quat, dx, dy)
        np.testing.assert_allclose(rel, ref, atol=1e-5)

    @pytest.mark.parametrize("roll, pitch", [
        (math.radians(8.0), 0.0),
        (0.0, math.radians(6.0)),
    ])
    def test_yaw_only_transform_would_fail_under_tilt(self, roll, pitch):
        """Teeth: the OLD yaw-only rel disagrees with the full
        quaternion-inverse under non-zero roll/pitch — the defect this fix
        closes. If this ever passes, the parity test above is vacuous.
        """
        yaw = math.radians(25.0)
        quat = _rpy_to_quat_xyzw(roll, pitch, yaw)
        goal, base = (2.5, -1.0), (0.5, 0.5)
        dx, dy = goal[0] - base[0], goal[1] - base[1]
        ref = _ref_rel_xy_via_matrix(quat, dx, dy)
        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
        yaw_only = np.array([cos_y * dx - sin_y * dy, sin_y * dx + cos_y * dy])
        assert not np.allclose(yaw_only, ref, atol=1e-5)

    def test_distance_and_heading_stay_yaw_only_under_tilt(self):
        """distance + heading must remain the 2-D / yaw-only quantities that
        match training's goal_distance / goal_heading_to_goal, even under
        tilt (they must NOT adopt the 3-D rel's xy).
        """
        quat = _rpy_to_quat_xyzw(math.radians(7.0), math.radians(-5.0), 0.4)
        goal, base = (2.5, -1.0), (0.5, 0.5)
        dx, dy = goal[0] - base[0], goal[1] - base[1]
        _, dist, head = body_frame_goal(
            goal_map_xy=goal, base_in_map_xy=base, base_in_map_quat=quat,
        )
        yaw = quaternion_to_yaw(*quat)
        expected_head = math.atan2(dy, dx) - yaw
        expected_head = math.atan2(math.sin(expected_head), math.cos(expected_head))
        assert dist == pytest.approx(math.hypot(dx, dy))
        assert head == pytest.approx(expected_head, abs=1e-6)


def test_quat_apply_inverse_xy_matches_matrix_reference():
    quat = _rpy_to_quat_xyzw(math.radians(5.0), math.radians(-7.0), math.radians(40.0))
    out = quat_apply_inverse_xy(quat, (1.3, -0.8))
    np.testing.assert_allclose(out, _ref_rel_xy_via_matrix(quat, 1.3, -0.8), atol=1e-6)
    assert out.dtype == np.float32


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
            variant=PolicyVariant.DEPTH,
            imu_accel=(0.1, 0.2, 9.8),
            imu_gyro=(0.01, -0.02, 0.03),
            wheel_vels_rad_s=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            goal_relative_xy=np.array([1.5, -0.5], dtype=np.float32),
            goal_distance=math.hypot(1.5, -0.5),
            goal_heading_to_goal=math.atan2(-0.5, 1.5),
            body_velocity_xy=(0.5, 0.0),
            last_action=np.zeros(3, dtype=np.float32),
            depth_flat_meters=np.full(
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

    def test_depth_referent_and_depth_routing_pinned(self):
        """Regression anchor: the variant-agnostic builder still routes the
        DEPTH goal_* triplet and the depth field to the right keys with the
        right values — the routing loop is the surface this PR changed.
        """
        raw = self._make_raw()
        np.testing.assert_allclose(raw["goal_relative"], [1.5, -0.5])
        np.testing.assert_allclose(raw["goal_distance"], [math.hypot(1.5, -0.5)])
        np.testing.assert_allclose(
            raw["goal_heading_to_goal"], [math.atan2(-0.5, 1.5)]
        )
        np.testing.assert_allclose(
            raw["depth_image"], np.full(DEPTH_HEIGHT * DEPTH_WIDTH, 0.5)
        )
        assert "subgoal_relative" not in raw

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


# =============================================================================
# build_raw_obs_dict — variant-agnostic (subgoal variant, no depth)
# =============================================================================


class TestRawDictSubgoalVariant:
    """The variant-aware builder emits the subgoal_* keys (not goal_*) and
    no depth field for NOCAM_SUBGOAL, and the body-frame referent triplet
    lands in those subgoal fields.
    """

    REL = np.array([1.5, -0.5], dtype=np.float32)
    DIST = math.hypot(1.5, -0.5)
    HEAD = math.atan2(-0.5, 1.5)

    def _make_raw(self, **overrides) -> dict:
        defaults = dict(
            variant=PolicyVariant.NOCAM_SUBGOAL,
            imu_accel=(0.1, 0.2, 9.8),
            imu_gyro=(0.01, -0.02, 0.03),
            wheel_vels_rad_s=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            goal_relative_xy=self.REL,
            goal_distance=self.DIST,
            goal_heading_to_goal=self.HEAD,
            body_velocity_xy=(0.5, 0.0),
            last_action=np.zeros(3, dtype=np.float32),
        )
        defaults.update(overrides)
        return build_raw_obs_dict(**defaults)

    def test_emits_subgoal_keys_and_no_goal_or_depth_keys(self):
        raw = self._make_raw()
        assert "subgoal_relative" in raw
        assert "subgoal_distance" in raw
        assert "subgoal_heading_to_subgoal" in raw
        assert "goal_relative" not in raw
        assert "goal_distance" not in raw
        assert "depth_image" not in raw

    def test_has_every_nocam_subgoal_field(self):
        raw = self._make_raw()
        for field in PolicyVariant.NOCAM_SUBGOAL.fields:
            assert field.key in raw, f"missing {field.key}"

    def test_referent_triplet_lands_in_subgoal_fields(self):
        raw = self._make_raw()
        np.testing.assert_allclose(raw["subgoal_relative"], self.REL)
        np.testing.assert_allclose(raw["subgoal_distance"], [self.DIST])
        np.testing.assert_allclose(
            raw["subgoal_heading_to_subgoal"], [self.HEAD]
        )

    def test_assembles_to_nocam_subgoal_obs_dim(self):
        raw = self._make_raw()
        obs = assemble_observation(raw, PolicyVariant.NOCAM_SUBGOAL)
        assert obs.shape == (PolicyVariant.NOCAM_SUBGOAL.obs_dim,)
        assert obs.shape[0] == 19
        assert obs.dtype == np.float32

    def test_builds_without_depth_arg(self):
        # No depth_flat_meters supplied; the no-depth contract holds:
        # depth omitted and exactly the variant's fields are emitted.
        raw = self._make_raw()
        assert "depth_image" not in raw
        assert len(raw) == len(PolicyVariant.NOCAM_SUBGOAL.fields)

    def test_depth_variant_without_depth_arg_raises(self):
        with pytest.raises(ValueError, match="depth_image field"):
            build_raw_obs_dict(
                variant=PolicyVariant.DEPTH,
                imu_accel=(0.0, 0.0, 0.0),
                imu_gyro=(0.0, 0.0, 0.0),
                wheel_vels_rad_s=np.zeros(4, dtype=np.float64),
                goal_relative_xy=np.zeros(2, dtype=np.float32),
                goal_distance=0.0,
                goal_heading_to_goal=0.0,
                body_velocity_xy=(0.0, 0.0),
                last_action=np.zeros(3, dtype=np.float32),
            )


# =============================================================================
# build_raw_obs_dict — NOCAM (goal_* keys, no depth): the one combination
# neither the DEPTH nor the NOCAM_SUBGOAL suite exercises
# =============================================================================


class TestRawDictNocamVariant:
    def _make_raw(self, **overrides) -> dict:
        defaults = dict(
            variant=PolicyVariant.NOCAM,
            imu_accel=(0.1, 0.2, 9.8),
            imu_gyro=(0.01, -0.02, 0.03),
            wheel_vels_rad_s=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            goal_relative_xy=np.array([1.5, -0.5], dtype=np.float32),
            goal_distance=math.hypot(1.5, -0.5),
            goal_heading_to_goal=math.atan2(-0.5, 1.5),
            body_velocity_xy=(0.5, 0.0),
            last_action=np.zeros(3, dtype=np.float32),
        )
        defaults.update(overrides)
        return build_raw_obs_dict(**defaults)

    def test_emits_goal_keys_and_no_subgoal_or_depth(self):
        raw = self._make_raw()
        assert "goal_relative" in raw
        assert "goal_distance" in raw
        assert "goal_heading_to_goal" in raw
        assert "subgoal_relative" not in raw
        assert "depth_image" not in raw

    def test_has_every_nocam_field(self):
        raw = self._make_raw()
        for field in PolicyVariant.NOCAM.fields:
            assert field.key in raw, f"missing {field.key}"
        assert len(raw) == len(PolicyVariant.NOCAM.fields)

    def test_assembles_to_nocam_obs_dim(self):
        raw = self._make_raw()
        obs = assemble_observation(raw, PolicyVariant.NOCAM)
        assert obs.shape == (PolicyVariant.NOCAM.obs_dim,)
        assert obs.shape[0] == 19
        assert obs.dtype == np.float32


# =============================================================================
# build_raw_obs_dict + assemble_observation — DEPTH_SUBGOAL (subgoal_* referent
# keys AND the depth tail)
# =============================================================================


class TestRawDictDepthSubgoalVariant:
    """DEPTH_SUBGOAL is the only variant exercising both the subgoal referent
    keys and the depth field. The variant-agnostic builder must emit subgoal_*
    (not goal_*) plus depth_image, and assembly must place the depth tail in
    the same trailing position as DEPTH while routing the referent triplet to
    the subgoal fields.
    """

    REL = np.array([1.5, -0.5], dtype=np.float32)
    DIST = math.hypot(1.5, -0.5)
    HEAD = math.atan2(-0.5, 1.5)
    DEPTH_FILL = 0.5

    def _make_raw(self, **overrides) -> dict:
        defaults = dict(
            variant=PolicyVariant.DEPTH_SUBGOAL,
            imu_accel=(0.1, 0.2, 9.8),
            imu_gyro=(0.01, -0.02, 0.03),
            wheel_vels_rad_s=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            goal_relative_xy=self.REL,
            goal_distance=self.DIST,
            goal_heading_to_goal=self.HEAD,
            body_velocity_xy=(0.5, 0.0),
            last_action=np.zeros(3, dtype=np.float32),
            depth_flat_meters=np.full(
                DEPTH_HEIGHT * DEPTH_WIDTH, self.DEPTH_FILL, dtype=np.float32
            ),
        )
        defaults.update(overrides)
        return build_raw_obs_dict(**defaults)

    def test_emits_subgoal_keys_and_depth_but_no_goal_keys(self):
        raw = self._make_raw()
        assert "subgoal_relative" in raw
        assert "subgoal_distance" in raw
        assert "subgoal_heading_to_subgoal" in raw
        assert "depth_image" in raw
        assert "goal_relative" not in raw
        assert "goal_distance" not in raw

    def test_has_every_depth_subgoal_field(self):
        raw = self._make_raw()
        for field in PolicyVariant.DEPTH_SUBGOAL.fields:
            assert field.key in raw, f"missing {field.key}"
        assert len(raw) == len(PolicyVariant.DEPTH_SUBGOAL.fields)

    def test_referent_triplet_lands_in_subgoal_fields(self):
        raw = self._make_raw()
        np.testing.assert_allclose(raw["subgoal_relative"], self.REL)
        np.testing.assert_allclose(raw["subgoal_distance"], [self.DIST])
        np.testing.assert_allclose(
            raw["subgoal_heading_to_subgoal"], [self.HEAD]
        )

    def test_depth_tail_is_in_the_depth_position(self):
        """DEPTH_SUBGOAL shares DEPTH's field layout: NOCAM_SUBGOAL's 19 scalar
        dims plus the DEPTH_WIDTH*DEPTH_HEIGHT (80×45 = 3600) depth field as the
        trailing block (3619 total)."""
        assert PolicyVariant.DEPTH_SUBGOAL.obs_dim == PolicyVariant.DEPTH.obs_dim
        assert (
            PolicyVariant.DEPTH_SUBGOAL.obs_dim
            == PolicyVariant.NOCAM_SUBGOAL.obs_dim + DEPTH_WIDTH * DEPTH_HEIGHT
        )
        assert PolicyVariant.DEPTH_SUBGOAL.fields[-1].key == "depth_image"
        assert (
            PolicyVariant.DEPTH_SUBGOAL.fields[-1].dims == DEPTH_WIDTH * DEPTH_HEIGHT
        )

    def test_assembles_per_field_against_the_enum(self):
        """Assemble and assert each obs slice equals raw * scale, iterating the
        shared enum's fields — pins ordering, per-field scale, and the depth
        tail position without restating any dim/scale literal here.
        """
        raw = self._make_raw()
        obs = assemble_observation(raw, PolicyVariant.DEPTH_SUBGOAL)
        assert obs.shape == (PolicyVariant.DEPTH_SUBGOAL.obs_dim,)
        assert obs.dtype == np.float32

        offset = 0
        for field in PolicyVariant.DEPTH_SUBGOAL.fields:
            expected = np.asarray(
                raw[field.key], dtype=np.float32
            ).ravel() * field.scale
            np.testing.assert_allclose(
                obs[offset:offset + field.dims], expected,
                rtol=1e-6, atol=1e-6,
                err_msg=f"field {field.key} slice mismatch",
            )
            offset += field.dims
        assert offset == PolicyVariant.DEPTH_SUBGOAL.obs_dim

    def test_assembly_rejects_goal_keys_for_depth_subgoal(self):
        """Wiring a goal-pose pipeline into DEPTH_SUBGOAL fails loudly at
        assembly rather than producing silent garbage — same guard as
        NOCAM_SUBGOAL, extended to the depth combo."""
        raw = self._make_raw()
        raw["goal_relative"] = raw.pop("subgoal_relative")
        with pytest.raises(KeyError):
            assemble_observation(raw, PolicyVariant.DEPTH_SUBGOAL)


# =============================================================================
# l1_clamp_velocity
# =============================================================================


class TestL1ClampVelocity:
    """Brief's safety bound: per-wheel motor cap means the chassis can't
    reach max forward + max strafe simultaneously. The L1 clamp scales
    (vx, vy) jointly so the commanded heading is preserved; clipping
    each axis independently would skew it. omega clamps independently
    because it routes through a different per-wheel sign-correction
    pathway.
    """

    LIN_CAP = 1.0
    ANG_CAP = 2.0

    def _clamp(self, vx, vy, omega):
        return l1_clamp_velocity(
            vx, vy, omega,
            vel_cap_linear_m_s=self.LIN_CAP,
            vel_cap_angular_rad_s=self.ANG_CAP,
        )

    def test_within_budget_passthrough(self):
        vx, vy, omega = self._clamp(0.3, 0.4, 1.5)
        assert (vx, vy, omega) == (pytest.approx(0.3), pytest.approx(0.4), pytest.approx(1.5))

    def test_l1_at_budget_unchanged(self):
        vx, vy, omega = self._clamp(0.4, 0.6, 0.0)
        assert vx + vy == pytest.approx(self.LIN_CAP)

    def test_l1_over_budget_scales_proportionally(self):
        vx, vy, omega = self._clamp(2.0, 2.0, 0.0)
        # 2.0 + 2.0 = 4.0, scale = 1.0/4.0 = 0.25 → (0.5, 0.5)
        assert vx == pytest.approx(0.5)
        assert vy == pytest.approx(0.5)
        assert abs(vx) + abs(vy) == pytest.approx(self.LIN_CAP)

    def test_signs_preserved_under_scaling(self):
        vx, vy, _ = self._clamp(-0.9, 0.3, 0.0)
        # L1 = 1.2, scale = 1/1.2 → vx = -0.75, vy = 0.25
        assert vx == pytest.approx(-0.75)
        assert vy == pytest.approx(0.25)
        # Heading is preserved: atan2 ratio unchanged.
        assert (vy / vx) == pytest.approx(0.25 / -0.75)

    def test_heading_preserved_under_scaling(self):
        """The brief's correctness anchor: scaling (vx, vy) jointly
        keeps the commanded heading. Clip-per-axis would skew it.
        """
        original_heading = math.atan2(0.7, 0.9)
        vx, vy, _ = self._clamp(0.9, 0.7, 0.0)
        clamped_heading = math.atan2(vy, vx)
        assert clamped_heading == pytest.approx(original_heading)

    def test_brief_worst_case_0p99_triplet(self):
        """The brief's literal example: policy emits (0.99, 0.99, 0.99)
        denormalized to ~(1.55, 1.55, 4.15). With caps at
        NAV_VEL_SCALE * MAX_LINEAR_VEL / NAV_VEL_SCALE * MAX_ANGULAR_VEL,
        the L1 sum is capped and the heading at 45° survives.
        """
        from strafer_shared.constants import (
            MAX_ANGULAR_VEL, MAX_LINEAR_VEL, NAV_VEL_SCALE,
        )
        lin_cap = NAV_VEL_SCALE * MAX_LINEAR_VEL
        ang_cap = NAV_VEL_SCALE * MAX_ANGULAR_VEL
        vx_in, vy_in, omega_in = 0.99 * MAX_LINEAR_VEL, 0.99 * MAX_LINEAR_VEL, 0.99 * MAX_ANGULAR_VEL
        vx, vy, omega = l1_clamp_velocity(
            vx_in, vy_in, omega_in,
            vel_cap_linear_m_s=lin_cap,
            vel_cap_angular_rad_s=ang_cap,
        )
        assert abs(vx) + abs(vy) <= lin_cap + 1e-9
        assert abs(omega) <= ang_cap + 1e-9
        # 45° heading preserved (vx ≈ vy after clamp).
        assert vx == pytest.approx(vy)

    def test_omega_positive_clamp(self):
        _, _, omega = self._clamp(0.0, 0.0, 5.0)
        assert omega == pytest.approx(self.ANG_CAP)

    def test_omega_negative_clamp(self):
        _, _, omega = self._clamp(0.0, 0.0, -5.0)
        assert omega == pytest.approx(-self.ANG_CAP)

    def test_zero_input_zero_output(self):
        assert self._clamp(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)

    def test_l1_clamp_independent_of_omega(self):
        """omega clamping does not modify (vx, vy)."""
        vx, vy, omega = self._clamp(0.4, 0.5, 5.0)
        assert vx == pytest.approx(0.4)
        assert vy == pytest.approx(0.5)
        assert omega == pytest.approx(self.ANG_CAP)
