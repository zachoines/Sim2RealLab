"""Tests for the episode-end button-to-outcome translator.

The translator is the single source of truth for the brief's
"Episode-end button mapping" table. Drift here would silently relabel
hard negatives, so the table is exercised exhaustively.
"""

from __future__ import annotations

import pytest

from strafer_lab.tools.teleop_buttons import (
    EpisodeEndDecision,
    button_state_to_episode_outcome,
    describe_button_layout,
)


class TestSingleButtonChords:
    def test_y_means_succeeded(self):
        d = button_state_to_episode_outcome({"y": True})
        assert d == EpisodeEndDecision(
            outcome="succeeded",
            outcome_category="on_course",
            hard_negative_category=None,
            discard=False,
        )

    def test_b_means_failed(self):
        d = button_state_to_episode_outcome({"b": True})
        assert d.outcome == "failed"
        assert d.outcome_category == "on_course"
        assert d.hard_negative_category is None
        assert d.discard is False

    def test_select_means_trajectory_violation(self):
        d = button_state_to_episode_outcome({"select": True})
        assert d.outcome == "trajectory_violation"
        assert d.outcome_category == "trajectory_violation"
        assert d.hard_negative_category == "trajectory_violation"
        assert d.discard is False

    def test_back_means_discard(self):
        d = button_state_to_episode_outcome({"back": True})
        assert d.discard is True
        assert d.outcome == "discarded"
        assert d.hard_negative_category is None


class TestXDpadChords:
    @pytest.mark.parametrize("dpad_y", [1, -1])
    def test_x_plus_dpad_vertical_is_wrong_instance(self, dpad_y):
        d = button_state_to_episode_outcome({"x": True}, dpad_y=dpad_y)
        assert d is not None
        assert d.outcome == "wrong_instance"
        assert d.outcome_category == "wrong_instance"
        assert d.hard_negative_category == "wrong_instance"
        assert d.discard is False

    @pytest.mark.parametrize("dpad_x", [1, -1])
    def test_x_plus_dpad_horizontal_is_wrong_room(self, dpad_x):
        d = button_state_to_episode_outcome({"x": True}, dpad_x=dpad_x)
        assert d is not None
        assert d.outcome == "wrong_room"
        assert d.hard_negative_category == "wrong_room"

    def test_x_alone_returns_pending(self):
        # X without a D-pad direction is not yet committal — the
        # operator hasn't chosen the sub-mode.
        assert button_state_to_episode_outcome({"x": True}) is None

    def test_x_plus_diagonal_dpad_picks_vertical_axis(self):
        # When both axes report non-zero (diagonal hat), we deterministically
        # prefer the vertical axis (wrong_instance). Documented behavior.
        d = button_state_to_episode_outcome({"x": True}, dpad_x=1, dpad_y=1)
        assert d is not None
        assert d.outcome == "wrong_instance"


class TestPrecedence:
    def test_back_wins_over_other_chords(self):
        # Operator can scrub a bad chord by tapping Back before releasing.
        d = button_state_to_episode_outcome(
            {"back": True, "y": True, "x": True, "select": True}, dpad_x=1,
        )
        assert d is not None
        assert d.discard is True

    def test_select_wins_over_y_b_x(self):
        d = button_state_to_episode_outcome(
            {"select": True, "y": True, "b": True, "x": True}, dpad_y=1,
        )
        assert d.outcome == "trajectory_violation"

    def test_y_wins_over_b_and_x(self):
        d = button_state_to_episode_outcome(
            {"y": True, "b": True, "x": True}, dpad_y=1,
        )
        assert d.outcome == "succeeded"

    def test_b_wins_over_x(self):
        d = button_state_to_episode_outcome({"b": True, "x": True}, dpad_y=1)
        assert d.outcome == "failed"


class TestNoOpTicks:
    def test_no_buttons_returns_none(self):
        assert button_state_to_episode_outcome({}) is None
        assert button_state_to_episode_outcome({"a": True, "start": True}) is None

    def test_a_alone_does_not_close_episode(self):
        # A is reserved by the driver loop for the pause toggle; it must
        # NOT be a closing chord.
        assert button_state_to_episode_outcome({"a": True}) is None

    def test_start_alone_does_not_close_episode(self):
        # Start is reserved for save-and-quit (held).
        assert button_state_to_episode_outcome({"start": True}) is None


class TestInvalidOutcomeIsRejected:
    def test_constructing_invalid_outcome_raises(self):
        with pytest.raises(ValueError, match="outcome"):
            EpisodeEndDecision(
                outcome="not_a_real_outcome",
                outcome_category="x",
                hard_negative_category=None,
                discard=False,
            )


class TestDescribeButtonLayout:
    def test_contains_each_chord(self):
        layout = describe_button_layout()
        for keyword in (
            "Y", "B", "X + D-pad", "SELECT", "Back",
            "succeed", "fail", "wrong_instance", "wrong_room",
            "trajectory_violation", "discard",
        ):
            assert keyword in layout, f"banner missing reference to {keyword!r}"
