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

    def test_bearing_dim_recovers_radians_through_its_own_scale(self):
        variant = PolicyVariant.DEPTH_SUBGOAL
        key = ece.bearing_key(variant)
        scale = next(f.scale for f in variant.fields if f.key == key)
        # The observation is the bearing times the field scale, so dividing by
        # the scale must land back on the original angle.
        for radians in (-np.pi + 1e-6, -0.5, 0.0, 1.25, np.pi - 1e-6):
            assert np.isclose(radians * scale / scale, radians)
        assert np.isclose(scale, 1.0 / np.pi)


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
