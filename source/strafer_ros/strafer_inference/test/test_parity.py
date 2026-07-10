"""Unit tests for the rclpy-free train↔deploy parity comparison library.

Covers the join / obs-delta / depth-spatial-residual / cadence / subgoal-replay
reports and the self-check re-assembly, all with synthetic fixtures so the
suite runs in the pxr-free colcon lane alongside test_obs_pipeline /
test_generator. No rosbag2, tf2, or rclpy import here — that glue lives in
``bag_io`` and the two CLIs, which are exercised on the rig.
"""

from __future__ import annotations

import numpy as np
import pytest

from strafer_inference import parity as P
from strafer_shared.constants import (
    BODY_VEL_SCALE,
    GOAL_DIST_SCALE,
    HEADING_SCALE,
    IMU_ACCEL_SCALE,
    IMU_GYRO_SCALE,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
    POLICY_PERIOD_S,
    WHEEL_JOINT_NAMES,
)
from strafer_shared.policy_interface import PolicyVariant


# =============================================================================
# Layout helpers (no dim literals — everything from PolicyVariant)
# =============================================================================


class TestLayout:
    def test_split_indices_scalar_only_variants(self):
        for v in (PolicyVariant.NOCAM, PolicyVariant.NOCAM_SUBGOAL):
            scalar, depth = P.split_indices(v)
            assert depth is None
            assert list(scalar) == list(range(v.obs_dim))

    def test_split_indices_depth_variants(self):
        for v in (PolicyVariant.DEPTH, PolicyVariant.DEPTH_SUBGOAL):
            scalar, depth = P.split_indices(v)
            assert depth is not None
            start, stop = depth
            # depth block is the contiguous tail
            assert stop == v.obs_dim
            assert list(scalar) == list(range(start))
            assert (stop - start) == v.fields[-1].dims

    def test_dim_names_length_and_naming(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        names = P.dim_names(v)
        assert len(names) == v.obs_dim
        assert names[0] == "imu_accel[0]"
        assert "subgoal_distance" in names  # single-dim field -> bare key


# =============================================================================
# Nearest-timestamp join
# =============================================================================


class TestNearestJoin:
    def test_exact_match(self):
        t = np.arange(5) * POLICY_PERIOD_S
        res = P.nearest_join(t, t.copy())
        assert res.n_matched == 5
        assert res.unmatched_a == []
        assert res.unmatched_b == []

    def test_skew_beyond_tolerance_unmatched(self):
        # A single reference tick far outside every a-tick's ±half-period
        # window: nothing matches, and both sides are accounted for. (Shifting
        # a whole grid by a period would alias onto later ticks — a uniform b
        # covers the line every period, so an offset copy always matches
        # something; only a lone far-off tick truly falls outside.)
        t_a = np.arange(5) * POLICY_PERIOD_S
        t_b = np.array([10.0 * POLICY_PERIOD_S])
        res = P.nearest_join(t_a, t_b, tol_s=P.JOIN_TOL_S)
        assert res.n_matched == 0
        assert res.unmatched_a == list(range(5))
        assert res.unmatched_b == [0]

    def test_tolerance_boundary(self):
        # One a, one b: pin both sides of the ±half-period edge so a regression
        # that widens the accept window is caught.
        t_a = np.array([5.0 * POLICY_PERIOD_S])
        inside = P.nearest_join(t_a, t_a + 0.99 * P.JOIN_TOL_S, tol_s=P.JOIN_TOL_S)
        outside = P.nearest_join(t_a, t_a + 1.01 * P.JOIN_TOL_S, tol_s=P.JOIN_TOL_S)
        assert inside.n_matched == 1
        assert outside.n_matched == 0

    def test_within_tolerance_matches(self):
        t_a = np.arange(5) * POLICY_PERIOD_S
        t_b = t_a + 0.4 * P.JOIN_TOL_S  # inside the window
        res = P.nearest_join(t_a, t_b, tol_s=P.JOIN_TOL_S)
        assert res.n_matched == 5


# =============================================================================
# Obs parity
# =============================================================================


def _obs_stream(t, obs, variant, referent=None):
    n = len(t)
    ref = np.full((n, 2), np.nan) if referent is None else np.asarray(referent, float)
    return P.ObsStream(
        t_sim=np.asarray(t, float),
        obs=np.asarray(obs, np.float32),
        variant=variant,
        referent_xy=ref,
        source="test",
    )


class TestObsParity:
    def test_identical_passes(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        rng = np.random.default_rng(0)
        t = np.arange(20) * POLICY_PERIOD_S
        obs = rng.standard_normal((20, v.obs_dim)).astype(np.float32)
        a = _obs_stream(t, obs, v)
        b = _obs_stream(t, obs.copy(), v)
        r = P.compute_obs_parity(a, b)
        assert r.passed
        assert r.scalar_max_abs == 0.0
        assert r.depth_max_abs is None

    def test_one_scalar_dim_perturbed_fails_named(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        t = np.arange(10) * POLICY_PERIOD_S
        obs = np.zeros((10, v.obs_dim), np.float32)
        obs_b = obs.copy()
        obs_b[3, 4] += 2e-5  # dim 4 = imu_gyro[1]
        r = P.compute_obs_parity(_obs_stream(t, obs, v), _obs_stream(t, obs_b, v))
        assert not r.passed
        assert r.scalar_worst_name == "imu_gyro[1]"
        assert r.scalar_max_abs == pytest.approx(2e-5, rel=1e-3)

    def test_depth_bound_separate_from_scalar(self):
        v = PolicyVariant.DEPTH
        t = np.arange(6) * POLICY_PERIOD_S
        obs = np.zeros((6, v.obs_dim), np.float32)
        _, (dstart, _) = P.split_indices(v)
        # 2e-4 is within the 1e-3 depth bound -> passes.
        ok = obs.copy()
        ok[2, dstart + 100] += 2e-4
        r_ok = P.compute_obs_parity(_obs_stream(t, obs, v), _obs_stream(t, ok, v))
        assert r_ok.depth_pass is True
        assert r_ok.passed
        # 2e-3 exceeds the depth bound -> fails on the depth block, named.
        bad = obs.copy()
        bad[2, dstart + 100] += 2e-3
        r_bad = P.compute_obs_parity(_obs_stream(t, obs, v), _obs_stream(t, bad, v))
        assert r_bad.depth_pass is False
        assert not r_bad.passed
        assert r_bad.depth_worst_name.startswith("depth_image[")

    def test_coverage_fails_when_few_ticks_match(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        t_a = np.arange(10) * POLICY_PERIOD_S
        obs_a = np.zeros((10, v.obs_dim), np.float32)
        # Only 3 reference ticks line up; the rest are off-grid by a full period.
        t_b = t_a[:3].copy()
        obs_b = obs_a[:3].copy()
        r = P.compute_obs_parity(_obs_stream(t_a, obs_a, v), _obs_stream(t_b, obs_b, v))
        assert r.scalar_pass  # deltas are zero where matched
        assert not r.coverage_ok  # but 3/10 < 0.6
        assert not r.passed

    def test_variant_mismatch_raises(self):
        t = np.arange(3) * POLICY_PERIOD_S
        a = _obs_stream(t, np.zeros((3, 19), np.float32), PolicyVariant.NOCAM_SUBGOAL)
        b = _obs_stream(t, np.zeros((3, 19), np.float32), PolicyVariant.NOCAM)
        with pytest.raises(ValueError):
            P.compute_obs_parity(a, b)


# =============================================================================
# NaN masking (the gym dump NaN-fills the referent-derived + last_action dims
# it cannot compute; the join masks those from the bound and reports them)
# =============================================================================


def _non_computable_dims(variant):
    """Flat indices of the referent-shaped triplet + last_action for a variant.

    These are exactly the dims the gym-side dumper NaN-fills. Predicate walks
    ``PolicyVariant.fields`` — no dim literals.
    """
    idx: list[int] = []
    off = 0
    for f in variant.fields:
        if f.key.startswith(("goal_", "subgoal_")) or f.key == "last_action":
            idx.extend(range(off, off + f.dims))
        off += f.dims
    return np.asarray(idx, dtype=int)


class TestNaNMasking:
    def test_nan_dims_masked_excluded_and_reported(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        rng = np.random.default_rng(1)
        t = np.arange(20) * POLICY_PERIOD_S
        node = rng.standard_normal((20, v.obs_dim)).astype(np.float32)
        gym = node.copy()
        nan_dims = _non_computable_dims(v)
        gym[:, nan_dims] = np.nan  # the dumper's referent + last_action fills

        r = P.compute_obs_parity(_obs_stream(t, node, v), _obs_stream(t, gym, v))

        # The NaN dims are masked, counted, and named — not silently passed.
        assert r.n_masked == nan_dims.size
        assert sorted(r.masked_dims.tolist()) == sorted(nan_dims.tolist())
        assert "last_action[0]" in r.masked_names
        assert "subgoal_relative[0]" in r.masked_names
        # The computable dims are identical, so parity passes on what remains.
        assert r.scalar_max_abs == 0.0
        assert r.scalar_pass
        assert r.passed
        # The worst scalar dim is drawn from the unmasked set, never a NaN dim.
        assert r.scalar_worst_dim not in set(nan_dims.tolist())

    def test_masking_does_not_hide_a_real_failure_in_a_computable_dim(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        t = np.arange(10) * POLICY_PERIOD_S
        node = np.zeros((10, v.obs_dim), np.float32)
        gym = node.copy()
        gym[:, _non_computable_dims(v)] = np.nan
        gym[3, 0] += 2e-5  # imu_accel[0] — a computable dim, above the 1e-5 bound

        r = P.compute_obs_parity(_obs_stream(t, node, v), _obs_stream(t, gym, v))
        assert not r.passed
        assert r.scalar_worst_name == "imu_accel[0]"
        assert r.scalar_max_abs == pytest.approx(2e-5, rel=1e-3)

    def test_depth_variant_scalars_masked_depth_still_bounded(self):
        v = PolicyVariant.DEPTH_SUBGOAL
        t = np.arange(6) * POLICY_PERIOD_S
        node = np.zeros((6, v.obs_dim), np.float32)
        gym = node.copy()
        gym[:, _non_computable_dims(v)] = np.nan  # only the scalar referent block
        _, (dstart, _) = P.split_indices(v)

        # Depth agrees within bound -> passes, with the scalar block masked.
        gym_ok = gym.copy()
        gym_ok[2, dstart + 50] += 2e-4
        r_ok = P.compute_obs_parity(_obs_stream(t, node, v), _obs_stream(t, gym_ok, v))
        assert r_ok.n_masked == _non_computable_dims(v).size
        assert r_ok.depth_pass is True
        assert r_ok.passed

        # A depth dim beyond its bound still fails — depth is never masked.
        gym_bad = gym.copy()
        gym_bad[2, dstart + 50] += 2e-3
        r_bad = P.compute_obs_parity(
            _obs_stream(t, node, v), _obs_stream(t, gym_bad, v)
        )
        assert r_bad.depth_pass is False
        assert not r_bad.passed
        assert r_bad.depth_worst_name.startswith("depth_image[")

    def test_no_nan_masks_nothing(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        t = np.arange(5) * POLICY_PERIOD_S
        obs = np.zeros((5, v.obs_dim), np.float32)
        r = P.compute_obs_parity(_obs_stream(t, obs, v), _obs_stream(t, obs.copy(), v))
        assert r.n_masked == 0
        assert r.masked_names == []
        assert r.passed


# =============================================================================
# parse_obs_records validation
# =============================================================================


class TestParseRecords:
    def test_wrong_obs_dim_rejected(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        recs = [{"t_sim": 0.0, "variant": v.name, "obs": [0.0] * (v.obs_dim - 1)}]
        with pytest.raises(ValueError, match="dims"):
            P.parse_obs_records(recs)

    def test_mixed_variants_rejected(self):
        recs = [
            {"t_sim": 0.0, "variant": "NOCAM_SUBGOAL", "obs": [0.0] * 19},
            {"t_sim": 0.1, "variant": "NOCAM", "obs": [0.0] * 19},
        ]
        with pytest.raises(ValueError, match="mixes variants"):
            P.parse_obs_records(recs)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            P.parse_obs_records([])


# =============================================================================
# Depth spatial residual (row-structured vs unstructured/time-varying)
# =============================================================================


def _depth_streams(depth_block_b, variant=PolicyVariant.DEPTH):
    """Two aligned streams: a is all-zeros, b carries the given depth block
    (shape (N, H*W)); scalar block identical so only depth differs."""
    n = depth_block_b.shape[0]
    t = np.arange(n) * POLICY_PERIOD_S
    _, (dstart, dstop) = P.split_indices(variant)
    obs_a = np.zeros((n, variant.obs_dim), np.float32)
    obs_b = obs_a.copy()
    obs_b[:, dstart:dstop] = depth_block_b
    a = _obs_stream(t, obs_a, variant)
    b = _obs_stream(t, obs_b, variant)
    join = P.nearest_join(a.t_sim, b.t_sim)
    return a, b, join


class TestDepthSpatialResidual:
    def test_row_structured_is_geometry_signature(self):
        from strafer_shared.constants import DEPTH_HEIGHT, DEPTH_WIDTH

        n = 4
        grid = np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH), np.float32)
        grid += (np.arange(DEPTH_HEIGHT)[:, None] * 1e-3).astype(np.float32)
        block = np.tile(grid.reshape(-1), (n, 1))  # same every tick
        a, b, join = _depth_streams(block)
        rep = P.depth_spatial_residual(a, b, join)
        assert rep is not None
        assert rep.row_structure > rep.col_structure
        assert "ROW-STRUCTURED" in rep.verdict

    def test_unstructured_time_varying_is_freshness_signature(self):
        from strafer_shared.constants import DEPTH_HEIGHT, DEPTH_WIDTH

        # Spatially flat, but the whole-frame offset swings tick to tick.
        offsets = np.array([1e-4, 6e-3, 1e-4, 6e-3], np.float32)
        block = np.repeat(offsets[:, None], DEPTH_HEIGHT * DEPTH_WIDTH, axis=1)
        a, b, join = _depth_streams(block)
        rep = P.depth_spatial_residual(a, b, join)
        assert rep.row_structure < P._STRUCT_THRESH
        assert rep.time_variation > P._TIME_THRESH
        assert "FRESHNESS" in rep.verdict.upper()

    def test_none_for_camera_free_variant(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        t = np.arange(3) * POLICY_PERIOD_S
        obs = np.zeros((3, v.obs_dim), np.float32)
        a, b = _obs_stream(t, obs, v), _obs_stream(t, obs.copy(), v)
        join = P.nearest_join(a.t_sim, b.t_sim)
        assert P.depth_spatial_residual(a, b, join) is None


# =============================================================================
# Cadence
# =============================================================================


class TestCadence:
    def test_uniform_period_is_clean(self):
        t = np.arange(50) * POLICY_PERIOD_S
        rep = P.cadence_report(t)
        assert rep.mode_delta_s == pytest.approx(POLICY_PERIOD_S, abs=1e-3)
        assert rep.fraction_at_expected >= 0.9
        assert "clean" in rep.verdict

    def test_gaps_flagged(self):
        # Drop every other tick for the back half -> a run of 2x-period gaps.
        head = np.arange(20) * POLICY_PERIOD_S
        tail = 20 * POLICY_PERIOD_S + np.arange(10) * 2 * POLICY_PERIOD_S
        t = np.concatenate([head, tail])
        rep = P.cadence_report(t)
        assert rep.n_gaps > 0
        assert "concern" in rep.verdict

    def test_shifted_mode_flagged(self):
        t = np.arange(30) * (POLICY_PERIOD_S * 1.5)
        rep = P.cadence_report(t)
        assert "SHIFTED" in rep.verdict


# =============================================================================
# Subgoal-pick self-consistency replay
# =============================================================================


class TestSubgoalReplay:
    def _straight_path(self):
        return np.array([(x, 0.0) for x in range(6)], dtype=np.float64)

    def test_consistent_picks_pass(self):
        path = self._straight_path()
        events = [("plan", path)]
        # Robot walks +x along the path; recompute the pick and use it as the
        # "published" value -> zero residual by construction.
        gen = P.RollingSubgoalGenerator(lookahead_m=P.SUBGOAL_LOOKAHEAD_M)
        gen.set_path(path)
        for k in range(5):
            robot = (0.5 * k, 0.0)
            state = gen.update(np.array(robot))
            events.append(("tick", robot, tuple(state.subgoal_xy), k * POLICY_PERIOD_S))
        rep = P.replay_subgoal_consistency(events)
        assert rep.passed
        assert rep.n_evaluated == 5
        assert rep.max_residual_m == pytest.approx(0.0, abs=1e-9)

    def test_outlier_tick_fails_at_that_tick(self):
        path = self._straight_path()
        gen = P.RollingSubgoalGenerator(lookahead_m=P.SUBGOAL_LOOKAHEAD_M)
        gen.set_path(path)
        events = [("plan", path)]
        picks = []
        for k in range(5):
            robot = (0.5 * k, 0.0)
            state = gen.update(np.array(robot))
            picks.append(tuple(state.subgoal_xy))
        # Re-run the same trajectory but corrupt the published subgoal at tick 2.
        events = [("plan", path)]
        for k in range(5):
            robot = (0.5 * k, 0.0)
            pub = picks[k]
            if k == 2:
                pub = (pub[0], pub[1] + 0.15)  # 15 cm off — beyond the 10 cm bound
            events.append(("tick", robot, pub, k * POLICY_PERIOD_S))
        rep = P.replay_subgoal_consistency(events)
        assert not rep.passed
        assert rep.worst_tick.t_sim == pytest.approx(2 * POLICY_PERIOD_S)
        assert rep.worst_tick.residual_m == pytest.approx(0.15, abs=1e-6)

    def test_ticks_before_plan_are_skipped(self):
        events = [
            ("tick", (0.0, 0.0), (1.0, 0.0), 0.0),
            ("plan", self._straight_path()),
            ("tick", (0.5, 0.0), (1.5, 0.0), POLICY_PERIOD_S),
        ]
        rep = P.replay_subgoal_consistency(events)
        assert rep.n_skipped_no_path == 1
        assert rep.n_evaluated == 1


# =============================================================================
# Self-check re-assembly (mirrors InferenceNode._assemble_observation_or_none)
# =============================================================================


class TestReassembly:
    def test_nocam_subgoal_matches_hand_computed_vector(self):
        v = PolicyVariant.NOCAM_SUBGOAL
        # Robot at the map origin, identity orientation; subgoal 2 m ahead (+x).
        obs = P.reassemble_obs_from_extracted(
            v,
            imu_accel=(0.0, 0.0, 9.81),
            imu_gyro=(0.0, 0.0, 0.1),
            joint_names=list(WHEEL_JOINT_NAMES),
            joint_velocities=[0.0, 0.0, 0.0, 0.0],
            body_velocity_xy=(0.5, 0.0),
            last_action=np.array([0.1, 0.2, 0.3], np.float32),
            referent_map_xy=(2.0, 0.0),
            base_in_map_xy=(0.0, 0.0),
            base_in_map_quat=(0.0, 0.0, 0.0, 1.0),
        )
        expected = np.zeros(v.obs_dim, np.float32)
        expected[0:3] = np.array([0.0, 0.0, 9.81]) * IMU_ACCEL_SCALE
        expected[3:6] = np.array([0.0, 0.0, 0.1]) * IMU_GYRO_SCALE
        expected[6:10] = 0.0  # zero wheel velocity -> zero encoder ticks
        expected[10:12] = np.array([2.0, 0.0]) * GOAL_DIST_SCALE  # subgoal_relative
        expected[12] = 2.0 * GOAL_DIST_SCALE  # subgoal_distance
        expected[13] = 0.0 * HEADING_SCALE  # heading straight ahead
        expected[14:16] = np.array([0.5, 0.0]) * BODY_VEL_SCALE
        expected[16:19] = np.array([0.1, 0.2, 0.3])  # last_action scale 1.0
        assert obs.shape == (v.obs_dim,)
        np.testing.assert_allclose(obs, expected, atol=1e-6)

    def test_depth_variant_reassembles_depth_block(self):
        v = PolicyVariant.DEPTH_SUBGOAL
        from strafer_shared.constants import DEPTH_MAX

        depth = np.full((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), 3.0, np.float32)
        obs = P.reassemble_obs_from_extracted(
            v,
            imu_accel=(0.0, 0.0, 0.0),
            imu_gyro=(0.0, 0.0, 0.0),
            joint_names=list(WHEEL_JOINT_NAMES),
            joint_velocities=[0.0, 0.0, 0.0, 0.0],
            body_velocity_xy=(0.0, 0.0),
            last_action=np.zeros(3, np.float32),
            referent_map_xy=(1.0, 0.0),
            base_in_map_xy=(0.0, 0.0),
            base_in_map_quat=(0.0, 0.0, 0.0, 1.0),
            depth_meters=depth,
        )
        _, (dstart, dstop) = P.split_indices(v)
        # constant 3 m depth -> DEPTH_SCALE (1/DEPTH_MAX) applied once downstream.
        np.testing.assert_allclose(obs[dstart:dstop], 3.0 / DEPTH_MAX, atol=1e-6)
