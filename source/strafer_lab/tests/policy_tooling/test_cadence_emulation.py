"""Unit tests for the cadence-emulation evaluation harness.

Pure numpy plus a small torch GRU — no Isaac Sim, no Kit boot, no scene assets.
The suite pins four things the harness is only trustworthy if it gets right:
the schedule sampler reproduces the requested hold/duplicate statistics, the
observation-dump loader recovers a temporal profile from the node's JSONL
schema, the signed direction offset has the sign convention the report assumes,
and restoring a batch row's recurrent hidden state after a batched forward is
equivalent to never having run that row.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import eval_cadence_emulation as ece
from strafer_shared.policy_interface import PolicyVariant


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------


class TestFieldSlices:
    def test_slices_tile_the_observation_without_gaps(self):
        for variant in PolicyVariant:
            slices = ece.field_slices(variant)
            offset = 0
            for obs_field in variant.fields:
                span = slices[obs_field.key]
                assert span.start == offset, (
                    f"{variant.name}.{obs_field.key} starts at {span.start}, "
                    f"expected {offset} — a gap or overlap means the depth block "
                    f"and the bearing dim are being read from the wrong place"
                )
                assert span.stop - span.start == obs_field.dims
                offset = span.stop
            assert offset == variant.obs_dim

    def test_depth_block_is_the_trailing_span(self):
        slices = ece.field_slices(PolicyVariant.DEPTH_SUBGOAL)
        depth = slices["depth_image"]
        assert depth.stop == PolicyVariant.DEPTH_SUBGOAL.obs_dim
        assert depth.start == PolicyVariant.NOCAM_SUBGOAL.obs_dim

    def test_referent_keys_resolve_per_variant(self):
        assert ece.bearing_key(PolicyVariant.DEPTH_SUBGOAL) == "subgoal_heading_to_subgoal"
        assert ece.relative_key(PolicyVariant.DEPTH_SUBGOAL) == "subgoal_relative"
        assert ece.bearing_key(PolicyVariant.DEPTH) == "goal_heading_to_goal"
        assert ece.relative_key(PolicyVariant.DEPTH) == "goal_relative"

    def test_bearing_field_scale_is_the_reciprocal_of_pi(self):
        # The harness divides the raw obs dim by this scale to recover radians,
        # so the scale is load-bearing: any other value silently rescales every
        # reported direction offset.
        variant = PolicyVariant.DEPTH_SUBGOAL
        key = ece.bearing_key(variant)
        scale = next(f.scale for f in variant.fields if f.key == key)
        assert scale == pytest.approx(1.0 / np.pi)
        assert np.isclose(0.75 / scale, 0.75 * np.pi)


class TestVerifyTermLayout:
    """A total-width check passes any reorder that preserves the sum."""

    def _names_and_dims(self, variant):
        names = []
        for obs_field in variant.fields:
            key = obs_field.key
            # The env's cfg spells the subgoal terms with the goal_* names.
            names.append(key.replace("subgoal_relative", "goal_position")
                         .replace("subgoal_distance", "goal_distance")
                         .replace("subgoal_heading_to_subgoal", "goal_heading_to_goal"))
        dims = [(f.dims,) for f in variant.fields]
        return names, dims

    def test_matching_layout_passes(self):
        variant = PolicyVariant.DEPTH_SUBGOAL
        names, dims = self._names_and_dims(variant)
        ece.verify_term_layout(names, dims, variant)

    def test_term_count_mismatch_is_rejected(self):
        variant = PolicyVariant.DEPTH_SUBGOAL
        names, dims = self._names_and_dims(variant)
        with pytest.raises(ValueError, match="terms but"):
            ece.verify_term_layout(names[:-1], dims[:-1], variant)

    def test_reorder_that_preserves_total_width_is_caught(self):
        variant = PolicyVariant.DEPTH_SUBGOAL
        names, dims = self._names_and_dims(variant)
        # Swap the 2-wide relative field with the 1-wide distance field: the
        # total is unchanged, the bearing and depth reads are not.
        names[3], names[4] = names[4], names[3]
        dims[3], dims[4] = dims[4], dims[3]
        assert sum(int(d[0]) for d in dims) == variant.obs_dim
        with pytest.raises(ValueError, match="field order has drifted"):
            ece.verify_term_layout(names, dims, variant)

    def test_equal_width_swap_at_the_bearing_dim_is_caught(self):
        variant = PolicyVariant.DEPTH_SUBGOAL
        names, dims = self._names_and_dims(variant)
        names[5] = "goal_distance"  # same width, wrong quantity
        with pytest.raises(ValueError, match="not a signed heading term"):
            ece.verify_term_layout(names, dims, variant)

    def test_equal_width_swap_of_relative_and_body_velocity_is_caught(self):
        variant = PolicyVariant.DEPTH_SUBGOAL
        names, dims = self._names_and_dims(variant)
        # Both are 2-wide, so every width check passes and a drift arm would
        # perturb chassis velocity instead of the referent.
        names[3], names[6] = names[6], names[3]
        assert dims[3] == dims[6]
        with pytest.raises(ValueError, match="not a relative-position term"):
            ece.verify_term_layout(names, dims, variant)

    def test_equal_width_swap_at_the_distance_dim_is_caught(self):
        variant = PolicyVariant.DEPTH_SUBGOAL
        names, dims = self._names_and_dims(variant)
        names[4] = "goal_heading_to_goal"  # same width, wrong quantity
        with pytest.raises(ValueError, match="not a distance term"):
            ece.verify_term_layout(names, dims, variant)

    def test_non_trailing_depth_block_is_caught(self):
        variant = PolicyVariant.DEPTH_SUBGOAL
        names, dims = self._names_and_dims(variant)
        names[-1] = "something_else"
        with pytest.raises(ValueError, match="not 'depth_image'"):
            ece.verify_term_layout(names, dims, variant)

    def test_multi_axis_term_dims_are_flattened(self):
        variant = PolicyVariant.DEPTH_SUBGOAL
        names, dims = self._names_and_dims(variant)
        dims[-1] = (45, 80)  # the manager may report the unflattened shape
        ece.verify_term_layout(names, dims, variant)


# ---------------------------------------------------------------------------
# Direction offset
# ---------------------------------------------------------------------------


class TestSignedDirectionOffset:
    def test_command_aligned_with_the_referent_is_zero(self):
        action = np.array([[1.0, 0.0]])
        offset, valid = ece.signed_direction_offset(action, np.array([0.0]))
        assert valid[0]
        assert np.isclose(offset[0], 0.0)

    def test_positive_offset_means_commanded_left_of_the_referent(self):
        # Referent dead ahead, command 45 degrees to the left (+y is left).
        action = np.array([[1.0, 1.0]])
        offset, _ = ece.signed_direction_offset(action, np.array([0.0]))
        assert np.isclose(np.degrees(offset[0]), 45.0)

    def test_negative_offset_means_commanded_right_of_the_referent(self):
        action = np.array([[1.0, -1.0]])
        offset, _ = ece.signed_direction_offset(action, np.array([0.0]))
        assert np.isclose(np.degrees(offset[0]), -45.0)

    def test_offset_is_relative_not_absolute(self):
        # Command and referent both 90 degrees left: the offset is still zero.
        action = np.array([[0.0, 1.0]])
        offset, _ = ece.signed_direction_offset(action, np.array([np.pi / 2.0]))
        assert np.isclose(offset[0], 0.0, atol=1e-9)

    def test_offset_wraps_into_the_principal_branch(self):
        action = np.array([[-1.0, -0.01]])
        offset, _ = ece.signed_direction_offset(action, np.array([np.pi - 0.01]))
        assert -np.pi < offset[0] <= np.pi
        assert abs(offset[0]) < 0.1, (
            "an offset either side of the +/-pi seam must stay small; a raw "
            "subtraction would report nearly a full turn"
        )

    def test_uniform_positive_scaling_leaves_the_offset_unchanged(self):
        action = np.array([[0.4, 0.3]])
        bearing = np.array([0.2])
        base, _ = ece.signed_direction_offset(action, bearing)
        scaled, _ = ece.signed_direction_offset(action * 7.5, bearing)
        assert np.isclose(base[0], scaled[0]), (
            "the L1 clamp scales vx and vy jointly, so the commanded direction "
            "must be invariant to the command magnitude"
        )

    def test_slow_commands_are_marked_invalid(self):
        action = np.array([[0.001, 0.0], [0.9, 0.0]])
        _, valid = ece.signed_direction_offset(action, np.zeros(2), min_command=0.05)
        assert not valid[0]
        assert valid[1]

    def test_summary_reports_median_and_left_share(self):
        offsets = np.radians([10.0, 20.0, -30.0, 40.0])
        summary = ece.summarize_offsets(offsets)
        assert summary["samples"] == 4
        assert np.isclose(summary["median_deg"], 15.0)
        assert np.isclose(summary["fraction_left"], 0.75)

    def test_empty_summary_is_null_not_zero(self):
        summary = ece.summarize_offsets([])
        assert summary["samples"] == 0
        assert summary["median_deg"] is None
        assert summary["fraction_left"] is None, (
            "an empty sample must not read as 'no leftward bias'; that is a "
            "different claim from 'no data'"
        )


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class TestTemporalProfile:
    def test_presets_are_ordered_by_severity(self):
        clean = ece.PRESET_PROFILES["clean"]
        band = ece.PRESET_PROFILES["band"]
        degraded = ece.PRESET_PROFILES["degraded"]
        assert clean.hold_fraction == 0.0
        assert clean.expected_inference_hz(30.0) == pytest.approx(30.0)
        assert band.hold_fraction < degraded.hold_fraction
        assert band.expected_inference_hz(30.0) == pytest.approx(23.0, abs=0.2)
        assert degraded.expected_inference_hz(30.0) == pytest.approx(11.7, abs=0.2)

    def test_hold_and_duplicate_axes_are_independent(self):
        degraded = ece.PRESET_PROFILES["degraded"]
        assert degraded.hold_fraction > 0.0 and degraded.stale_fraction > 0.0
        assert ece.PRESET_PROFILES["band"].stale_fraction == 0.0, (
            "the held fraction sets the inference rate and the duplicate "
            "fraction sets the content novelty; collapsing them into one knob "
            "would make neither measurable"
        )

    def test_unreachable_hold_fraction_is_rejected(self):
        with pytest.raises(ValueError, match="unreachable"):
            ece.TemporalProfile(name="x", hold_fraction=0.9, mean_hold_run=1.0)

    def test_unreachable_stale_fraction_is_rejected(self):
        with pytest.raises(ValueError, match="unreachable"):
            ece.TemporalProfile(name="x", stale_fraction=0.9, mean_stale_run=1.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"hold_fraction": 1.0},
            {"hold_fraction": -0.1},
            {"stale_fraction": 1.0},
            {"burst_weight": 1.5},
            {"mean_hold_run": 0.5},
        ],
    )
    def test_out_of_range_knobs_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ece.TemporalProfile(name="x", **kwargs)

    def test_resolve_applies_overrides_to_presets(self):
        resolved = ece.resolve_profiles(["band"], overrides={"hold_fraction": 0.4})
        assert resolved[0].hold_fraction == pytest.approx(0.4)
        assert ece.PRESET_PROFILES["band"].hold_fraction == pytest.approx(0.233), (
            "resolve_profiles must not mutate the shared preset table"
        )

    def test_an_overridden_arm_is_renamed_after_its_knobs(self):
        resolved = ece.resolve_profiles(
            ["degraded"], overrides={"stale_fraction": 0.0}
        )
        assert resolved[0].name == "degraded+stale0"
        assert resolved[0].stale_fraction == pytest.approx(0.0)

    def test_overrides_never_leave_an_arm_labelled_clean(self):
        # Results are keyed by arm name and the decision rule divides by the arm
        # named 'clean'. An override that kept the preset name would produce a
        # baseline that silently holds ticks.
        resolved = ece.resolve_profiles(
            ["clean", "degraded"], overrides={"hold_fraction": 0.4}
        )
        names = [profile.name for profile in resolved]
        assert "clean" not in names, (
            f"an overridden arm must not keep the baseline's name; got {names}"
        )
        assert names[0].startswith("clean+")
        assert all(profile.hold_fraction == pytest.approx(0.4) for profile in resolved)

    def test_unoverridden_arms_keep_their_preset_names(self):
        resolved = ece.resolve_profiles(["clean", "band", "degraded"])
        assert [profile.name for profile in resolved] == ["clean", "band", "degraded"]

    def test_overrides_on_a_measured_arm_are_refused(self):
        # replace() cannot reach an EmpiricalProfile, so silently ignoring the
        # flags would report a replayed distribution under a requested one.
        with pytest.raises(ValueError, match="do not apply to 'measured'"):
            ece.resolve_profiles(
                ["measured"],
                overrides={"hold_fraction": 0.4},
                dump_loader=lambda name: ece.EmpiricalProfile(
                    name=name,
                    interval_ticks=np.array([1]),
                    stale_run_lengths=np.array([], dtype=np.int64),
                    clean_run_lengths=np.array([1]),
                ),
            )

    def test_resolve_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="unknown profile"):
            ece.resolve_profiles(["nonsense"])

    def test_measured_without_a_loader_is_rejected(self):
        with pytest.raises(ValueError, match="observation dump"):
            ece.resolve_profiles(["measured"])


# ---------------------------------------------------------------------------
# Schedule sampler
# ---------------------------------------------------------------------------


def _drive(sampler: ece.ScheduleSampler, ticks: int) -> np.ndarray:
    return np.stack([sampler.next() for _ in range(ticks)])


class TestScheduleSampler:
    @pytest.mark.parametrize("name", ["clean", "band", "degraded"])
    def test_realized_statistics_track_the_request(self, name):
        profile = ece.PRESET_PROFILES[name]
        sampler = ece.ScheduleSampler(profile, 16, np.random.default_rng(7))
        _drive(sampler, 4000)
        realized = sampler.realized(30.0)
        assert realized["hold_fraction"] == pytest.approx(profile.hold_fraction, abs=0.02)
        assert realized["stale_fraction"] == pytest.approx(profile.stale_fraction, abs=0.02)
        assert realized["inference_hz"] == pytest.approx(
            profile.expected_inference_hz(30.0), abs=0.6
        )

    def test_realized_hold_runs_track_the_requested_mean(self):
        profile = ece.PRESET_PROFILES["degraded"]
        sampler = ece.ScheduleSampler(profile, 16, np.random.default_rng(11))
        _drive(sampler, 6000)
        realized = sampler.realized(30.0)
        assert realized["mean_hold_run"] == pytest.approx(
            profile.expected_hold_run(), rel=0.15
        )
        assert realized["max_hold_run"] > profile.expected_hold_run(), (
            "a burst mixture must produce hold runs well past the mean"
        )

    def test_tick_counts_partition_the_rollout(self):
        sampler = ece.ScheduleSampler(
            ece.PRESET_PROFILES["degraded"], 8, np.random.default_rng(3)
        )
        _drive(sampler, 500)
        realized = sampler.realized(30.0)
        total = realized["fresh_ticks"] + realized["stale_ticks"] + realized["held_ticks"]
        assert total == realized["ticks"] == 500 * 8

    def test_clean_profile_never_holds_or_duplicates(self):
        sampler = ece.ScheduleSampler(
            ece.PRESET_PROFILES["clean"], 4, np.random.default_rng(1)
        )
        kinds = _drive(sampler, 300)
        assert np.all(kinds == ece.TICK_FRESH)

    def test_envs_hold_independently(self):
        sampler = ece.ScheduleSampler(
            ece.PRESET_PROFILES["degraded"], 8, np.random.default_rng(5)
        )
        kinds = _drive(sampler, 400)
        held = kinds == ece.TICK_HELD
        rows_agree = np.all(held == held[:, :1], axis=1)
        assert rows_agree.mean() < 0.5, (
            "a per-env schedule must not degenerate into one global hold mask; "
            "otherwise the arm measures a synchronized stall, not a rate"
        )
        assert held.any(axis=0).all(), "every env should hold at least once"

    def test_warmup_forces_fresh_ticks_after_a_reset(self):
        sampler = ece.ScheduleSampler(
            ece.PRESET_PROFILES["degraded"],
            4,
            np.random.default_rng(2),
            warmup_ticks=3,
        )
        kinds = _drive(sampler, 3)
        assert np.all(kinds == ece.TICK_FRESH)

        _drive(sampler, 200)
        sampler.reset(np.array([True, False, False, False]))
        after = _drive(sampler, 3)
        assert np.all(after[:, 0] == ece.TICK_FRESH), (
            "a new episode is a new mission; the node cannot infer before that "
            "mission's first frame arrives"
        )

    def test_reset_of_no_envs_is_a_no_op(self):
        sampler = ece.ScheduleSampler(
            ece.PRESET_PROFILES["band"], 4, np.random.default_rng(9)
        )
        _drive(sampler, 50)
        before = sampler.realized(30.0)["ticks"]
        sampler.reset(np.zeros(4, dtype=bool))
        assert sampler.realized(30.0)["ticks"] == before

    def test_sampler_is_reproducible_under_a_fixed_seed(self):
        first = _drive(
            ece.ScheduleSampler(
                ece.PRESET_PROFILES["degraded"], 6, np.random.default_rng(42)
            ),
            200,
        )
        second = _drive(
            ece.ScheduleSampler(
                ece.PRESET_PROFILES["degraded"], 6, np.random.default_rng(42)
            ),
            200,
        )
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Observation-dump profile loader
# ---------------------------------------------------------------------------

_DEPTH_START = 4


def _write_dump(path, rows, depth_dim=3):
    """Write a dump in the node's schema: one JSON object per inference."""
    with open(path, "w", encoding="utf-8") as handle:
        for t_sim, depth_value in rows:
            obs = [0.1] * _DEPTH_START + [float(depth_value)] * depth_dim
            handle.write(
                json.dumps(
                    {
                        "t_sim": float(t_sim),
                        "variant": "DEPTH_SUBGOAL",
                        "obs": obs,
                        "referent": None,
                    }
                )
                + "\n"
            )
    return str(path)


class TestLoadDumpProfile:
    def test_back_to_back_inferences_read_as_no_holds(self, tmp_path):
        dt = 1.0 / 30.0
        rows = [(index * dt, index) for index in range(60)]
        profile = ece.load_dump_profile(
            _write_dump(tmp_path / "d.jsonl", rows), dt, depth_start=_DEPTH_START
        )
        assert profile.records == 60
        np.testing.assert_array_equal(profile.interval_ticks, np.ones(59, dtype=np.int64))
        assert profile.hold_fraction() == pytest.approx(0.0)
        assert profile.expected_inference_hz(30.0) == pytest.approx(30.0)

    def test_every_other_tick_reads_as_half_rate(self, tmp_path):
        dt = 1.0 / 30.0
        rows = [(index * 2 * dt, index) for index in range(60)]
        profile = ece.load_dump_profile(
            _write_dump(tmp_path / "d.jsonl", rows), dt, depth_start=_DEPTH_START
        )
        assert np.all(profile.interval_ticks == 2)
        assert profile.hold_fraction() == pytest.approx(0.5)
        assert profile.expected_inference_hz(30.0) == pytest.approx(15.0)

    def test_identical_depth_blocks_are_duplicate_events(self, tmp_path):
        dt = 1.0 / 30.0
        # Every content value appears twice: half the inferences see pixels
        # identical to the previous inference.
        rows = [(index * dt, index // 2) for index in range(60)]
        profile = ece.load_dump_profile(
            _write_dump(tmp_path / "d.jsonl", rows), dt, depth_start=_DEPTH_START
        )
        assert profile.stale_fraction() == pytest.approx(0.5, abs=0.02)
        assert np.all(profile.stale_run_lengths == 1)

    def test_distinct_depth_blocks_have_no_duplicates(self, tmp_path):
        dt = 1.0 / 30.0
        rows = [(index * dt, index) for index in range(40)]
        profile = ece.load_dump_profile(
            _write_dump(tmp_path / "d.jsonl", rows), dt, depth_start=_DEPTH_START
        )
        assert profile.stale_fraction() == pytest.approx(0.0)
        assert profile.stale_run_lengths.size == 0

    def test_long_gaps_are_dropped_as_capture_seams(self, tmp_path):
        dt = 1.0 / 30.0
        stamps = [index * dt for index in range(20)]
        stamps += [stamps[-1] + 60.0 + index * dt for index in range(20)]
        rows = [(t, index) for index, t in enumerate(stamps)]
        profile = ece.load_dump_profile(
            _write_dump(tmp_path / "d.jsonl", rows),
            dt,
            depth_start=_DEPTH_START,
            max_interval_ticks=10,
        )
        assert profile.interval_ticks.max() <= 10
        assert profile.interval_ticks.size == 38

    def test_replaying_a_measured_profile_reproduces_its_rate(self, tmp_path):
        dt = 1.0 / 30.0
        rng = np.random.default_rng(4)
        stamp = 0.0
        rows = []
        for index in range(4000):
            rows.append((stamp, index // 2 if index % 4 else index))
            stamp += dt * int(rng.integers(1, 4))
        profile = ece.load_dump_profile(
            _write_dump(tmp_path / "d.jsonl", rows), dt, depth_start=_DEPTH_START
        )
        sampler = ece.ScheduleSampler(profile, 8, np.random.default_rng(6))
        _drive(sampler, 4000)
        realized = sampler.realized(30.0)
        assert realized["hold_fraction"] == pytest.approx(profile.hold_fraction(), abs=0.03)
        assert realized["stale_fraction"] == pytest.approx(
            profile.stale_fraction(), abs=0.05
        )

    def test_missing_keys_are_rejected(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"t_sim": 0.0}\n{"t_sim": 0.1}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="t_sim'/'obs"):
            ece.load_dump_profile(str(path), 1.0 / 30.0, depth_start=_DEPTH_START)

    def test_malformed_json_names_the_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"t_sim": 0.0, "obs": [1,2,3,4,5]}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2 is not valid JSON"):
            ece.load_dump_profile(str(path), 1.0 / 30.0, depth_start=_DEPTH_START)

    def test_non_monotonic_stamps_are_rejected(self, tmp_path):
        dt = 1.0 / 30.0
        rows = [(0.0, 0), (2 * dt, 1), (dt, 2)]
        with pytest.raises(ValueError, match="non-monotonic"):
            ece.load_dump_profile(
                _write_dump(tmp_path / "d.jsonl", rows), dt, depth_start=_DEPTH_START
            )

    def test_short_observation_vectors_are_rejected(self, tmp_path):
        path = tmp_path / "short.jsonl"
        path.write_text(
            '{"t_sim": 0.0, "obs": [1,2]}\n{"t_sim": 0.1, "obs": [1,2]}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="too short"):
            ece.load_dump_profile(str(path), 1.0 / 30.0, depth_start=_DEPTH_START)

    def test_a_single_record_is_rejected(self, tmp_path):
        rows = [(0.0, 0)]
        with pytest.raises(ValueError, match="at least 2"):
            ece.load_dump_profile(
                _write_dump(tmp_path / "d.jsonl", rows),
                1.0 / 30.0,
                depth_start=_DEPTH_START,
            )

    def test_blank_lines_are_tolerated(self, tmp_path):
        dt = 1.0 / 30.0
        path = tmp_path / "d.jsonl"
        _write_dump(path, [(0.0, 0), (dt, 1)])
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n")
        profile = ece.load_dump_profile(str(path), dt, depth_start=_DEPTH_START)
        assert profile.records == 2


# ---------------------------------------------------------------------------
# Held-tick semantics
# ---------------------------------------------------------------------------


class TestHeldTickEquivalence:
    """The harness runs the batched forward every tick and then restores the
    held rows. That is only a faithful emulation if it is indistinguishable
    from never having run those rows at all."""

    def test_restoring_a_row_matches_never_calling_it(self):
        torch = pytest.importorskip("torch")
        torch.manual_seed(0)

        num_envs, features, hidden = 6, 5, 8
        gru = torch.nn.GRU(input_size=features, hidden_size=hidden, num_layers=1)
        gru.eval()

        held_env = 2
        schedule = [True, False, True, True, False, True]  # False == held
        inputs = [torch.randn(num_envs, features) for _ in schedule]

        # Batched forward every tick, restoring the held row's hidden column.
        batched = None
        with torch.inference_mode():
            for tick, fresh in enumerate(schedule):
                before = None if batched is None else batched.clone()
                _, batched = gru(inputs[tick].unsqueeze(0), batched)
                if not fresh and before is not None:
                    batched[:, held_env, :] = before[:, held_env, :]

        # The same row run solo, seeing only the ticks it was not held on.
        solo = None
        with torch.inference_mode():
            for tick, fresh in enumerate(schedule):
                if not fresh:
                    continue
                row = inputs[tick][held_env: held_env + 1]
                _, solo = gru(row.unsqueeze(0), solo)

        delta = (batched[:, held_env, :] - solo[:, 0, :]).abs().max().item()
        assert delta == pytest.approx(0.0, abs=1e-6), (
            f"restoring the held row's hidden column must be equivalent to "
            f"skipping the call for that row; max deviation {delta}"
        )

    def test_unheld_rows_are_untouched_by_the_restore(self):
        torch = pytest.importorskip("torch")
        torch.manual_seed(1)

        num_envs, features, hidden = 4, 3, 6
        gru = torch.nn.GRU(input_size=features, hidden_size=hidden, num_layers=1)
        gru.eval()
        held_env = 1

        with torch.inference_mode():
            state = None
            for _ in range(3):
                _, state = gru(torch.randn(num_envs, features).unsqueeze(0), state)
            before = state.clone()
            _, state = gru(torch.randn(num_envs, features).unsqueeze(0), state)
            advanced = state.clone()
            state[:, held_env, :] = before[:, held_env, :]

        for env_index in range(num_envs):
            moved = not torch.allclose(state[:, env_index, :], before[:, env_index, :])
            if env_index == held_env:
                assert not moved, "the held row's state must not have advanced"
            else:
                assert moved, "a fresh row's state must advance"
                assert torch.allclose(state[:, env_index, :], advanced[:, env_index, :])

    def test_forward_rebinds_rather_than_mutating_the_state(self):
        torch = pytest.importorskip("torch")
        torch.manual_seed(2)

        gru = torch.nn.GRU(input_size=3, hidden_size=4, num_layers=1)
        gru.eval()
        with torch.inference_mode():
            _, state = gru(torch.randn(2, 3).unsqueeze(0), None)
            snapshot = state
            copied = state.clone()
            _, state = gru(torch.randn(2, 3).unsqueeze(0), state)

        assert torch.equal(snapshot, copied), (
            "the pre-call reference must still hold the pre-call values; if the "
            "forward ever mutates in place, the restore would be a no-op"
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _episode(cause, progress=0.5, arc=2.0, steps=100):
    return {
        "env": 0,
        "steps": steps,
        "cause": cause,
        "progress_fraction": progress,
        "along_track_m": arc,
        "fresh_ticks": steps,
        "stale_ticks": 0,
        "held_ticks": 0,
    }


class TestSummaries:
    def test_completion_rate_counts_only_arrivals(self):
        episodes = [
            _episode("path_complete", progress=1.0),
            _episode("path_complete", progress=1.0),
            _episode("time_out", progress=0.4),
            _episode("off_path_divergence", progress=0.2),
        ]
        summary = ece.summarize_arm(episodes)
        assert summary["episodes"] == 4
        assert summary["completions"] == 2
        assert summary["completion_rate"] == pytest.approx(0.5)
        assert summary["cause_fractions"]["off_path_divergence"] == pytest.approx(0.25)

    def test_near_arrival_separates_the_dwell_gate_from_never_arriving(self):
        # Both fail, but one parked short of the dwell criterion at the end of
        # the path and the other never got there.
        episodes = [_episode("time_out", progress=0.99), _episode("time_out", progress=0.3)]
        summary = ece.summarize_arm(episodes)
        assert summary["completion_rate"] == pytest.approx(0.0)
        assert summary["near_arrival_rate"] == pytest.approx(0.5)

    def test_empty_arm_summary_is_explicit(self):
        assert ece.summarize_arm([]) == {"episodes": 0}

    def test_tables_render_without_episodes(self):
        arm = {
            "arm": "clean",
            "summary": {"episodes": 0},
            "realized_profile": {},
            "direction_offset": ece.summarize_offsets([]),
        }
        assert "clean" in ece.format_table([arm])
        assert "clean" in ece.format_cause_table([arm])

    def test_arm_name_survives_into_the_rendered_table(self):
        arm = {
            "arm": "degraded+stale0",
            "summary": ece.summarize_arm([_episode("path_complete", progress=1.0)]),
            "realized_profile": {
                "inference_hz": 11.7,
                "hold_fraction": 0.61,
                "stale_fraction": 0.0,
            },
            "direction_offset": ece.summarize_offsets(np.radians([5.0])),
        }
        assert "degraded+stale0" in ece.format_table([arm])

    def test_table_renders_a_null_offset_without_crashing(self):
        arm = {
            "arm": "band",
            "summary": ece.summarize_arm([_episode("path_complete", progress=1.0)]),
            "realized_profile": {
                "inference_hz": 23.0,
                "hold_fraction": 0.23,
                "stale_fraction": 0.0,
            },
            "direction_offset": ece.summarize_offsets([]),
        }
        rendered = ece.format_table([arm])
        assert "n/a" in rendered
        assert "23.00" in rendered


# ---------------------------------------------------------------------------
# The rollout loop itself
# ---------------------------------------------------------------------------

_VARIANT = PolicyVariant.DEPTH_SUBGOAL
_SLICES = ece.field_slices(_VARIANT)
_DEPTH_SPAN = _SLICES["depth_image"]
_BEARING_INDEX = _SLICES[ece.bearing_key(_VARIANT)].start
_RELATIVE_SPAN = _SLICES[ece.relative_key(_VARIANT)]
_HEADING_SCALE = next(
    f.scale for f in _VARIANT.fields if f.key == ece.bearing_key(_VARIANT)
)
_DIST_SCALE = next(
    f.scale for f in _VARIANT.fields if f.key == ece.relative_key(_VARIANT)
)
_QUARTET_INDICES = (
    _RELATIVE_SPAN.start,
    _RELATIVE_SPAN.start + 1,
    _SLICES[ece.distance_key(_VARIANT)].start,
    _BEARING_INDEX,
)


class _FakeCursor:
    def __init__(self, num_envs, torch):
        self._cursor = torch.zeros(num_envs)
        self._total = torch.full((num_envs,), 8.0)

    @property
    def total_arc(self):
        return self._total


class _FakeCommandManager:
    def __init__(self, num_envs, torch):
        self._term = type("Term", (), {"path_cursor": _FakeCursor(num_envs, torch)})()

    def get_term(self, name):
        assert name == "goal_command", f"unexpected command term {name!r}"
        return self._term


class _FakeTerminationManager:
    def __init__(self, num_envs, torch):
        self.active_terms = list(ece.TERMINATION_PRIORITY)
        self._dones = {
            name: torch.zeros(num_envs, dtype=torch.bool) for name in self.active_terms
        }

    def get_term(self, name):
        return self._dones[name]


class _FakeEnv:
    """Terminates each env on its own period, so episodes end out of phase."""

    def __init__(self, num_envs, rng, torch):
        self._torch = torch
        self._rng = rng
        self.num_envs = num_envs
        self._tick = np.zeros(num_envs, dtype=np.int64)
        self._period = rng.integers(20, 45, size=num_envs)
        self.unwrapped = type(
            "Unwrapped",
            (),
            {
                "device": torch.device("cpu"),
                "command_manager": _FakeCommandManager(num_envs, torch),
                "termination_manager": _FakeTerminationManager(num_envs, torch),
                "action_manager": type("AM", (), {"total_action_dim": 3})(),
                "step_dt": 1.0 / 30.0,
            },
        )()
        self._obs = self._make_obs()
        self.issued = []

    def _make_obs(self):
        flat = self._torch.from_numpy(
            self._rng.standard_normal((self.num_envs, _VARIANT.obs_dim)).astype(
                np.float32
            )
        )
        return {"policy": flat, "critic": flat.clone()}

    def get_observations(self):
        return self._obs

    def step(self, actions):
        assert actions.shape == (self.num_envs, 3), actions.shape
        self.issued.append(actions.clone())
        self._tick += 1
        cursor = self.unwrapped.command_manager.get_term("goal_command").path_cursor
        cursor._cursor += self._torch.from_numpy(
            np.abs(actions[:, 0].numpy()).astype(np.float32) * 0.1
        )
        done = self._tick >= self._period
        manager = self.unwrapped.termination_manager
        for name in manager.active_terms:
            manager._dones[name][:] = False
        for index in np.flatnonzero(done):
            cause = ["path_complete", "time_out", "off_path_divergence"][index % 3]
            manager._dones[cause][index] = True
            if index % 5 == 0:
                # Terms co-fire; attribution must resolve by priority.
                manager._dones["time_out"][index] = True
        self._tick[done] = 0
        self._period[done] = self._rng.integers(20, 45, size=int(done.sum()))
        cursor._cursor[self._torch.from_numpy(done)] = 0.0
        self._obs = self._make_obs()
        return (
            self._obs,
            self._torch.zeros(self.num_envs),
            self._torch.from_numpy(done.astype(np.int64)),
            {},
        )


class _FakePolicy:
    """Rebinds its hidden state like the real recurrent wrapper does."""

    training = False

    def __init__(self, num_envs, torch):
        self._torch = torch
        self.num_envs = num_envs
        self.rnn = type("RNN", (), {"hidden_state": None})()
        self.calls = 0
        self.reset_calls = 0
        self.seen_depth = []

    def __call__(self, obs):
        flat = obs["policy"]
        assert flat.shape == (self.num_envs, _VARIANT.obs_dim), flat.shape
        self.calls += 1
        self.seen_depth.append(flat[:, _DEPTH_SPAN].clone())
        if self.rnn.hidden_state is None:
            self.rnn.hidden_state = self._torch.zeros(1, self.num_envs, 8)
        self.rnn.hidden_state = self.rnn.hidden_state + 1.0
        return self._torch.tanh(flat[:, 0:3] * 0.5)

    def reset(self, dones=None):
        self.reset_calls += 1
        if dones is None:
            self.rnn.hidden_state = None
        elif self.rnn.hidden_state is not None:
            self.rnn.hidden_state[..., dones == 1, :] = 0.0


def _drive_arm(
    profile_name,
    *,
    episodes=25,
    warmup=0,
    seed=0,
    reset_hidden_on_done=True,
    drift=None,
):
    torch = pytest.importorskip("torch")
    env = _FakeEnv(6, np.random.default_rng(seed), torch)
    policy = _FakePolicy(6, torch)
    profile = ece.PRESET_PROFILES[profile_name]
    sampler = ece.ScheduleSampler(
        profile, 6, np.random.default_rng(seed + 1), warmup_ticks=warmup
    )
    kinds_log = []
    original_next = sampler.next

    def recording_next():
        kinds = original_next()
        kinds_log.append(kinds.copy())
        return kinds

    sampler.next = recording_next
    result = ece._run_arm(
        env=env, policy=policy, torch=torch, profile=profile, sampler=sampler,
        depth_span=_DEPTH_SPAN, bearing_index=_BEARING_INDEX,
        relative_span=_RELATIVE_SPAN, heading_scale=_HEADING_SCALE,
        episode_budget=episodes, step_budget=4000, min_command=0.05, tick_hz=30.0,
        quartet_indices=_QUARTET_INDICES, dist_scale=_DIST_SCALE,
        drift=drift, reset_hidden_on_done=reset_hidden_on_done,
    )
    return result, env, policy, np.stack(kinds_log)


class TestRunArm:
    """Drives the rollout loop against a mocked env and policy.

    The mocks reproduce the two behaviours the loop depends on: the recurrent
    state is rebound rather than mutated, and terminated envs are reset inside
    ``step`` before it returns.
    """

    @pytest.mark.parametrize("name", ["clean", "band", "degraded"])
    def test_arm_scores_its_episode_budget(self, name):
        result, _, _, _ = _drive_arm(name)
        summary = result["summary"]
        assert summary["episodes"] == 25
        assert not result["budget_exhausted"]
        assert set(summary["cause_counts"]) <= set(ece.TERMINATION_PRIORITY)
        assert "unattributed" not in summary["cause_counts"], (
            "every episode end must resolve to a named termination term"
        )

    def test_co_firing_terms_resolve_to_the_success_signal(self):
        # Envs at index % 5 == 0 fire time_out alongside their real cause; the
        # ones whose real cause is path_complete must not be labelled time_out.
        result, _, _, _ = _drive_arm("clean", episodes=30)
        assert result["summary"]["cause_counts"].get("path_complete", 0) > 0

    def test_held_rows_receive_the_previous_action_verbatim(self):
        _, env, _, kinds = _drive_arm("degraded")
        checked = 0
        for tick in range(1, len(env.issued)):
            for env_index in np.flatnonzero(kinds[tick] == ece.TICK_HELD):
                assert torch_equal(env.issued[tick][env_index],
                                   env.issued[tick - 1][env_index]), (
                    f"tick {tick} env {env_index}: a held tick must re-issue the "
                    f"previous command, not recompute or zero it"
                )
                checked += 1
        assert checked > 50, f"only {checked} held rows exercised"

    def test_stale_rows_see_the_block_from_their_last_fresh_tick(self):
        _, _, policy, kinds = _drive_arm("degraded")
        last_fresh = {}
        checked = 0
        for tick in range(len(policy.seen_depth)):
            for env_index in range(kinds.shape[1]):
                kind = kinds[tick][env_index]
                if kind == ece.TICK_FRESH:
                    last_fresh[env_index] = policy.seen_depth[tick][env_index].clone()
                elif kind == ece.TICK_STALE:
                    assert env_index in last_fresh
                    assert torch_equal(policy.seen_depth[tick][env_index],
                                       last_fresh[env_index]), (
                        f"tick {tick} env {env_index}: a duplicate-content tick "
                        f"must re-show the last inferred block. A held tick in "
                        f"between is irrelevant -- its row is fed live depth but "
                        f"its output is discarded."
                    )
                    checked += 1
        assert checked > 20, f"only {checked} stale rows exercised"

    def test_the_policy_runs_on_every_tick_at_full_batch_width(self):
        # Held rows are emulated by restoring state, not by shrinking the batch:
        # a narrower forward would rebind the hidden state and destroy the held
        # columns permanently.
        _, env, policy, _ = _drive_arm("degraded")
        assert policy.calls == len(env.issued)
        assert all(block.shape[0] == env.num_envs for block in policy.seen_depth)

    def test_held_rows_do_not_advance_the_hidden_state(self):
        torch = pytest.importorskip("torch")
        env = _FakeEnv(6, np.random.default_rng(9), torch)
        policy = _FakePolicy(6, torch)
        profile = ece.PRESET_PROFILES["degraded"]
        sampler = ece.ScheduleSampler(profile, 6, np.random.default_rng(10))
        seen = []
        original_next = sampler.next

        def recording_next():
            kinds = original_next()
            state = policy.rnn.hidden_state
            seen.append((kinds.copy(), None if state is None else state.clone()))
            return kinds

        sampler.next = recording_next
        ece._run_arm(
            env=env, policy=policy, torch=torch, profile=profile, sampler=sampler,
            depth_span=_DEPTH_SPAN, bearing_index=_BEARING_INDEX,
            relative_span=_RELATIVE_SPAN, heading_scale=_HEADING_SCALE,
            episode_budget=10, step_budget=1000, min_command=0.05, tick_hz=30.0,
        )
        # The mock adds 1.0 per call, so a held row's column must be unchanged
        # between consecutive pre-call snapshots (barring an episode reset).
        checked = 0
        for tick in range(1, len(seen)):
            kinds, before = seen[tick - 1]
            _, after = seen[tick]
            if before is None or after is None:
                continue
            for env_index in np.flatnonzero(kinds == ece.TICK_HELD):
                if float(after[0, env_index, 0]) == 0.0:
                    continue  # zeroed by the episode-boundary reset
                assert torch_equal(after[:, env_index, :], before[:, env_index, :]), (
                    f"tick {tick - 1} env {env_index} was held; its recurrent "
                    f"state must not have advanced"
                )
                checked += 1
        assert checked > 20, f"only {checked} held rows exercised"

    def test_fresh_rows_do_advance_the_hidden_state(self):
        _, _, policy, _ = _drive_arm("clean", episodes=5)
        assert policy.rnn.hidden_state is not None

    def test_direction_offset_is_collected_only_on_inferring_ticks(self):
        result, _, _, kinds = _drive_arm("degraded")
        offset = result["direction_offset"]
        inferring = int(np.count_nonzero(kinds != ece.TICK_HELD))
        assert offset["samples"] > 0
        assert offset["samples"] + result["direction_offset_dropped_ticks"] == inferring, (
            "every inferring tick must either yield an offset sample or be "
            "counted as dropped; held ticks must contribute neither"
        )

    def test_both_bearing_readouts_are_reported(self):
        result, _, _, _ = _drive_arm("band")
        assert result["direction_offset"]["samples"] > 0
        assert result["direction_offset_from_relative"]["samples"] > 0

    def test_progress_is_latched_from_before_the_step(self):
        result, _, _, _ = _drive_arm("clean")
        for episode in result["episodes"]:
            assert 0.0 <= episode["progress_fraction"] <= 1.0
            assert episode["along_track_m"] >= 0.0

    def test_per_episode_tick_counts_sum_to_the_episode_length(self):
        result, _, _, _ = _drive_arm("degraded")
        for episode in result["episodes"]:
            total = (episode["fresh_ticks"] + episode["stale_ticks"]
                     + episode["held_ticks"])
            assert total == episode["steps"], (
                f"episode tick kinds sum to {total} but it ran {episode['steps']} "
                f"steps; per-env accumulators are misaligned"
            )

    def test_step_ceiling_is_reported_rather_than_hidden(self):
        torch = pytest.importorskip("torch")
        env = _FakeEnv(6, np.random.default_rng(2), torch)
        policy = _FakePolicy(6, torch)
        profile = ece.PRESET_PROFILES["clean"]
        sampler = ece.ScheduleSampler(profile, 6, np.random.default_rng(3))
        result = ece._run_arm(
            env=env, policy=policy, torch=torch, profile=profile, sampler=sampler,
            depth_span=_DEPTH_SPAN, bearing_index=_BEARING_INDEX,
            relative_span=_RELATIVE_SPAN, heading_scale=_HEADING_SCALE,
            episode_budget=10_000, step_budget=40, min_command=0.05, tick_hz=30.0,
        )
        assert result["steps"] == 40
        assert result["budget_exhausted"] is True

    def test_warmup_forces_fresh_ticks_at_every_episode_start(self):
        _, _, _, kinds = _drive_arm("degraded", warmup=3, episodes=20)
        assert np.all(kinds[0] == ece.TICK_FRESH)


def torch_equal(left, right):
    import torch

    return torch.equal(left, right)


class TestInferenceTensorDiscipline:
    """Isaac Lab's observation noise models keep their own state buffers and
    write them in place. Once any rollout has run inside inference mode those
    buffers are inference tensors, and an in-place write to one from outside
    inference mode raises. Every harness call that reaches env state therefore
    has to sit inside the same context."""

    @staticmethod
    def _inference_tensor(torch):
        with torch.inference_mode():
            return torch.zeros(4)

    def test_an_in_place_write_outside_inference_mode_really_does_raise(self):
        # Guards the premise: if torch ever relaxes this, the two tests below
        # would silently stop testing anything.
        torch = pytest.importorskip("torch")
        buffer = self._inference_tensor(torch)
        with pytest.raises(RuntimeError, match="[Ii]nference tensor"):
            buffer[0] += 1.0

    def test_arm_setup_reads_observations_inside_inference_mode(self):
        torch = pytest.importorskip("torch")
        buffer = self._inference_tensor(torch)
        env = _FakeEnv(6, np.random.default_rng(0), torch)
        base_get_observations = env.get_observations

        def noisy_get_observations():
            buffer[0] += 1.0
            return base_get_observations()

        env.get_observations = noisy_get_observations
        policy = _FakePolicy(6, torch)
        profile = ece.PRESET_PROFILES["clean"]
        sampler = ece.ScheduleSampler(profile, 6, np.random.default_rng(1))
        ece._run_arm(
            env=env, policy=policy, torch=torch, profile=profile, sampler=sampler,
            depth_span=_DEPTH_SPAN, bearing_index=_BEARING_INDEX,
            relative_span=_RELATIVE_SPAN, heading_scale=_HEADING_SCALE,
            episode_budget=3, step_budget=200, min_command=0.05, tick_hz=30.0,
        )

    def test_the_arm_transition_resets_inside_inference_mode(self):
        torch = pytest.importorskip("torch")
        buffer = self._inference_tensor(torch)
        calls = {"env": 0, "policy": 0}

        class _Env:
            def reset(self):
                buffer[0] += 1.0
                calls["env"] += 1

        class _Policy:
            def reset(self, dones=None):
                assert dones is None, (
                    "the arm transition must clear the hidden state globally, "
                    "not per-env; the next arm shares nothing with this one"
                )
                calls["policy"] += 1

        ece.reset_between_arms(_Env(), _Policy(), torch)
        assert calls == {"env": 1, "policy": 1}


# ---------------------------------------------------------------------------
# Referent-frame drift
# ---------------------------------------------------------------------------


def _quartet(rel):
    """Build the scaled four-dim referent block from body-frame vectors."""
    rel = np.atleast_2d(np.asarray(rel, dtype=np.float64))
    return np.stack(
        [
            rel[:, 0] * _DIST_SCALE,
            rel[:, 1] * _DIST_SCALE,
            np.hypot(rel[:, 0], rel[:, 1]) * _DIST_SCALE,
            np.arctan2(rel[:, 1], rel[:, 0]) * _HEADING_SCALE,
        ],
        axis=1,
    )


def _drifted(quartet, offsets):
    return ece.drifted_quartet(
        quartet, offsets, dist_scale=_DIST_SCALE, heading_scale=_HEADING_SCALE
    )


class TestDriftedQuartet:
    """The four dims are near-redundant, so they move as one geometry."""

    def test_zero_offset_is_an_exact_no_op(self):
        quartet = _quartet([[1.2, -0.4]])
        out = _drifted(quartet, np.zeros((1, 3)))
        assert np.allclose(out, quartet, atol=1e-12)

    def test_pure_rotation_preserves_distance_and_shifts_the_bearing(self):
        quartet = _quartet([[2.0, 0.0]])
        angle = np.radians(15.0)
        out = _drifted(quartet, np.array([[0.0, 0.0, angle]]))
        assert out[0, 2] == pytest.approx(quartet[0, 2], rel=1e-9)
        assert out[0, 3] / _HEADING_SCALE == pytest.approx(angle, rel=1e-9)

    def test_pure_translation_moves_distance_with_the_referent(self):
        quartet = _quartet([[2.0, 0.0]])
        out = _drifted(quartet, np.array([[0.5, 0.0, 0.0]]))
        assert out[0, 2] / _DIST_SCALE == pytest.approx(2.5, rel=1e-9)
        assert out[0, 3] == pytest.approx(0.0, abs=1e-12)

    def test_lateral_translation_produces_the_expected_bearing(self):
        # 0.166 m of lateral offset at a 1.0 m lookahead: the position and
        # heading knobs are near-equal in effect, so they do not separate.
        quartet = _quartet([[1.0, 0.0]])
        out = _drifted(quartet, np.array([[0.0, 0.166, 0.0]]))
        assert np.degrees(out[0, 3] / _HEADING_SCALE) == pytest.approx(9.425, abs=1e-3)

    def test_output_is_self_consistent_for_random_inputs(self):
        rng = np.random.default_rng(11)
        rel = rng.normal(size=(128, 2)) * 3.0
        quartet = _quartet(rel)
        offsets = rng.normal(size=(128, 3)) * 0.25
        out = _drifted(quartet, offsets)
        out_x = out[:, 0] / _DIST_SCALE
        out_y = out[:, 1] / _DIST_SCALE
        assert np.allclose(out[:, 2] / _DIST_SCALE, np.hypot(out_x, out_y))
        assert np.allclose(out[:, 3] / _HEADING_SCALE, np.arctan2(out_y, out_x))

    def test_bearing_stays_wrapped(self):
        rng = np.random.default_rng(3)
        quartet = _quartet(rng.normal(size=(256, 2)) * 4.0)
        out = _drifted(quartet, rng.normal(size=(256, 3)) * 1.5)
        bearing = out[:, 3] / _HEADING_SCALE
        assert np.all(bearing > -np.pi - 1e-12)
        assert np.all(bearing <= np.pi + 1e-12)

    def test_shapes_are_validated(self):
        with pytest.raises(ValueError, match=r"quartet must be \(N, 4\)"):
            _drifted(np.zeros((2, 3)), np.zeros((2, 3)))
        with pytest.raises(ValueError, match="offsets must be"):
            _drifted(np.zeros((2, 4)), np.zeros((3, 3)))


class TestSubgoalDrift:
    """Integrated, not resampled: localization error accumulates."""

    def _drift(self, **overrides):
        params = dict(
            num_envs=64,
            position_rms_m=0.166,
            heading_sigma_rad=np.radians(6.7),
            tau_s=2.0,
            step_dt=1.0 / 30.0,
            rng=np.random.default_rng(7),
        )
        params.update(overrides)
        return ece.SubgoalDrift(**params)

    def test_axis_sigma_is_the_rms_over_root_two(self):
        drift = self._drift()
        assert drift.position_sigma_axis_m == pytest.approx(0.166 / np.sqrt(2.0))

    def test_stationary_magnitudes_match_the_request(self):
        drift = self._drift(num_envs=512)
        for _ in range(3000):
            drift.step()
        realized = drift.realized()
        assert realized["position_rms_m"] == pytest.approx(0.166, rel=0.05)
        assert realized["heading_sigma_deg"] == pytest.approx(6.7, rel=0.05)

    def test_correlation_time_sets_the_one_step_decay(self):
        drift = self._drift(num_envs=4096)
        previous = None
        for _ in range(400):
            previous = drift.step().copy()
        current = drift.step()
        slope = float(np.sum(previous * current) / np.sum(previous * previous))
        assert slope == pytest.approx(np.exp(-(1.0 / 30.0) / 2.0), rel=0.05)

    def test_a_longer_tau_holds_an_excursion_longer(self):
        slow = self._drift(tau_s=8.0, num_envs=2048)
        fast = self._drift(tau_s=0.5, num_envs=2048)
        for _ in range(600):
            slow_state = slow.step().copy()
            fast_state = fast.step().copy()
        slow_next = slow.step()
        fast_next = fast.step()
        slow_slope = float(np.sum(slow_state * slow_next) / np.sum(slow_state**2))
        fast_slope = float(np.sum(fast_state * fast_next) / np.sum(fast_state**2))
        assert slow_slope > fast_slope

    def test_reset_re_anchors_only_the_masked_envs(self):
        drift = self._drift(num_envs=6)
        for _ in range(50):
            state = drift.step()
        assert np.any(state != 0.0)
        mask = np.array([True, False, True, False, False, False])
        carried = state[~mask].copy()
        drift.reset(mask)
        assert np.all(drift._state[mask] == 0.0)
        assert np.array_equal(drift._state[~mask], carried)

    def test_disabled_when_both_magnitudes_are_zero(self):
        assert not self._drift(position_rms_m=0.0, heading_sigma_rad=0.0).enabled
        assert self._drift(position_rms_m=0.0).enabled
        assert self._drift(heading_sigma_rad=0.0).enabled

    def test_realized_reports_the_request_alongside_what_was_applied(self):
        drift = self._drift()
        drift.step()
        realized = drift.realized()
        assert realized["requested_position_rms_m"] == pytest.approx(0.166)
        assert realized["requested_heading_sigma_deg"] == pytest.approx(6.7)
        assert realized["tau_s"] == pytest.approx(2.0)
        assert realized["samples"] == 64

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tau_s": 0.0},
            {"tau_s": -1.0},
            {"step_dt": 0.0},
            {"position_rms_m": -0.1},
            {"heading_sigma_rad": -0.1},
        ],
    )
    def test_invalid_parameters_are_rejected(self, overrides):
        with pytest.raises(ValueError):
            self._drift(**overrides)


class TestArmLabel:
    """An arm still labelled `clean` must be the untouched baseline."""

    def test_untouched_profile_keeps_its_name(self):
        assert ece.arm_label("clean", hidden_reset=True, drift_gain=None) == "clean"

    def test_hidden_reset_off_is_named(self):
        assert (
            ece.arm_label("clean", hidden_reset=False, drift_gain=None)
            == "clean+nohreset"
        )

    def test_drift_gain_is_named(self):
        assert (
            ece.arm_label("clean", hidden_reset=True, drift_gain=0.5)
            == "clean+drift0.5x"
        )

    def test_composed_arm_names_every_active_knob(self):
        assert (
            ece.arm_label("band", hidden_reset=False, drift_gain=1.0)
            == "band+nohreset+drift1x"
        )

    def test_zero_gain_is_not_named(self):
        assert ece.arm_label("band", hidden_reset=True, drift_gain=0.0) == "band"


class TestDuplicateAxisCells:
    """Hold-0 duplicate-axis cells, against the harness validation ceiling."""

    @pytest.mark.parametrize(
        "stale_fraction, mean_stale_run",
        [(0.61, 2.0), (0.76, 4.0), (0.233, 1.2)],
    )
    def test_pre_registered_cells_clear_the_ceiling(
        self, stale_fraction, mean_stale_run
    ):
        profile = ece.TemporalProfile(
            name="cell",
            hold_fraction=0.0,
            stale_fraction=stale_fraction,
            mean_stale_run=mean_stale_run,
        )
        assert profile.stale_fraction == pytest.approx(stale_fraction)

    def test_the_degraded_run_length_cannot_reach_the_rate_parity_fraction(self):
        # `degraded` carries mean_stale_run 1.0, whose ceiling is 0.5.
        with pytest.raises(ValueError):
            ece.TemporalProfile(
                name="cell",
                hold_fraction=0.0,
                stale_fraction=0.61,
                mean_stale_run=1.0,
            )

    def test_burst_knobs_are_inert_once_holds_are_off(self):
        bursty = ece.TemporalProfile(
            name="cell",
            hold_fraction=0.0,
            burst_weight=0.25,
            mean_burst_run=6.0,
            stale_fraction=0.61,
            mean_stale_run=2.0,
        )
        sampler = ece.ScheduleSampler(bursty, 8, np.random.default_rng(0))
        for _ in range(400):
            kinds = sampler.next()
            assert not np.any(kinds == ece.TICK_HELD)


class TestResidualArms:
    """Arms A through C driven through the rollout loop against mocks."""

    def _drift(self, num_envs=6, **overrides):
        params = dict(
            num_envs=num_envs,
            position_rms_m=0.166,
            heading_sigma_rad=np.radians(6.7),
            tau_s=2.0,
            step_dt=1.0 / 30.0,
            rng=np.random.default_rng(5),
        )
        params.update(overrides)
        return ece.SubgoalDrift(**params)

    def test_hidden_state_is_reset_at_boundaries_by_default(self):
        result, _, policy, kinds = _drive_arm("clean", episodes=12)
        assert policy.reset_calls == kinds.shape[0]
        assert result["hidden_reset_on_done"] is True

    def test_the_hidden_state_carries_when_the_reset_is_guarded(self):
        result, _, policy, _ = _drive_arm(
            "clean", episodes=12, reset_hidden_on_done=False
        )
        assert policy.reset_calls == 0
        assert result["hidden_reset_on_done"] is False

    def test_environments_still_reset_when_the_hidden_state_carries(self):
        result, _, _, _ = _drive_arm(
            "clean", episodes=12, reset_hidden_on_done=False
        )
        # Episodes still end and are still scored; only the recurrent state
        # spans the chain.
        assert len(result["episodes"]) >= 12
        assert result["summary"]["episodes"] >= 12

    def test_episode_index_counts_chain_depth_per_env(self):
        result, _, _, _ = _drive_arm(
            "clean", episodes=24, reset_hidden_on_done=False
        )
        per_env: dict[int, list[int]] = {}
        for episode in result["episodes"]:
            per_env.setdefault(episode["env"], []).append(episode["episode_index"])
        assert per_env
        for indices in per_env.values():
            assert indices == list(range(len(indices)))

    def test_drift_off_reports_no_perceived_offset(self):
        result, _, _, _ = _drive_arm("clean", episodes=8)
        assert result["direction_offset_perceived"] is None
        assert result["subgoal_drift"] is None

    def test_drift_does_not_overwrite_the_truth_referent(self):
        # `clean` never splices depth, so the policy observation aliases the
        # env's own buffer until the drift path clones it. Without that clone
        # the truth and perceived readouts would be the same tensor.
        result, _, _, _ = _drive_arm(
            "clean", episodes=14, drift=self._drift(position_rms_m=1.5)
        )
        truth = result["direction_offset"]
        perceived = result["direction_offset_perceived"]
        assert perceived is not None
        assert truth["samples"] > 0 and perceived["samples"] > 0
        assert truth["median_deg"] != pytest.approx(perceived["median_deg"], abs=1e-6)

    def test_drift_reports_what_it_applied(self):
        result, _, _, _ = _drive_arm("clean", episodes=8, drift=self._drift())
        realized = result["subgoal_drift"]
        assert realized["samples"] > 0
        assert realized["requested_position_rms_m"] == pytest.approx(0.166)
        assert realized["position_rms_m"] > 0.0

    def test_composed_arm_runs_with_both_knobs(self):
        result, _, policy, _ = _drive_arm(
            "band",
            episodes=12,
            reset_hidden_on_done=False,
            drift=self._drift(),
        )
        assert policy.reset_calls == 0
        assert result["subgoal_drift"]["samples"] > 0
        assert result["direction_offset_perceived"] is not None
        assert result["summary"]["episodes"] >= 12


class TestResolveDriftSources:
    """A drift arm has to own its displacement, or the gain it reports is not
    the gain it names."""

    def _call(self, **kw):
        base = dict(
            env_drifts=False,
            harness_drift_requested=False,
            suppress=False,
            allow_composed=False,
            env_id="Isaac-Test-v0",
        )
        base.update(kw)
        return ece.resolve_drift_sources(**base)

    def test_a_tier_without_drift_needs_no_decision(self):
        assert self._call() == (False, False)
        assert self._call(harness_drift_requested=True) == (False, False)

    def test_an_env_that_drifts_is_left_alone_when_no_arm_asks_for_drift(self):
        assert self._call(env_drifts=True) == (False, False)

    def test_suppression_drops_the_env_term(self):
        assert self._call(env_drifts=True, harness_drift_requested=True,
                          suppress=True) == (True, False)

    def test_suppression_without_a_drift_to_suppress_is_refused(self):
        """Silently accepting it would label an arm as fixed-gain on a tier that
        never carried the term."""
        with pytest.raises(SystemExit, match="nothing to suppress"):
            self._call(suppress=True)

    def test_an_unclaimed_composition_is_refused(self):
        """The failure this guards is a composed number reported under a
        fixed-gain name, which reads as comparable to a sweep and is not."""
        with pytest.raises(SystemExit, match="under the gain's name"):
            self._call(env_drifts=True, harness_drift_requested=True)

    def test_composition_is_available_when_claimed(self):
        assert self._call(env_drifts=True, harness_drift_requested=True,
                          allow_composed=True) == (False, True)

    def test_the_two_flags_contradict(self):
        with pytest.raises(SystemExit, match="mutually exclusive"):
            self._call(env_drifts=True, harness_drift_requested=True,
                       suppress=True, allow_composed=True)
