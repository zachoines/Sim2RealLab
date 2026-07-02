"""Unit tests for the six-source freshness watchdog."""

from __future__ import annotations

import pytest

from strafer_inference.watchdog import WatchdogTimeouts, stale_sources


@pytest.fixture
def timeouts() -> WatchdogTimeouts:
    return WatchdogTimeouts(
        goal=1.0, imu=0.2, joint_states=0.2, odom=0.2, depth=0.5, tf=0.5,
    )


def _all_fresh_rx_times(now: float) -> dict:
    return dict(
        last_goal_rx_t=now - 0.1,
        last_imu_rx_t=now - 0.05,
        last_joint_states_rx_t=now - 0.05,
        last_odom_rx_t=now - 0.05,
        last_depth_rx_t=now - 0.1,
        tf_age_s=0.1,
    )


class TestWatchdogTimeouts:
    def test_non_positive_timeout_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            WatchdogTimeouts(
                goal=0.0, imu=0.1, joint_states=0.1,
                odom=0.1, depth=0.1, tf=0.1,
            )

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            WatchdogTimeouts(
                goal=0.1, imu=-1.0, joint_states=0.1,
                odom=0.1, depth=0.1, tf=0.1,
            )


class TestStaleSources:
    def test_all_fresh_returns_empty(self, timeouts):
        now = 100.0
        assert stale_sources(
            now_monotonic_s=now,
            timeouts=timeouts,
            **_all_fresh_rx_times(now),
        ) == []

    def test_none_rx_times_treated_as_stale(self, timeouts):
        assert stale_sources(
            now_monotonic_s=100.0,
            last_goal_rx_t=None,
            last_imu_rx_t=None,
            last_joint_states_rx_t=None,
            last_odom_rx_t=None,
            last_depth_rx_t=None,
            tf_age_s=None,
            timeouts=timeouts,
        ) == ["goal", "imu", "joint_states", "odom", "depth", "tf"]

    @pytest.mark.parametrize(
        "stale_key, stale_value, expected",
        [
            ("last_goal_rx_t", 100.0 - 1.5, ["goal"]),
            ("last_imu_rx_t", 100.0 - 0.3, ["imu"]),
            ("last_joint_states_rx_t", 100.0 - 0.3, ["joint_states"]),
            ("last_odom_rx_t", 100.0 - 0.3, ["odom"]),
            ("last_depth_rx_t", 100.0 - 0.6, ["depth"]),
            ("tf_age_s", 0.6, ["tf"]),
        ],
    )
    def test_each_source_is_independent(
        self, timeouts, stale_key, stale_value, expected,
    ):
        """Brief's anchor: a stale joint_states must trip the watchdog
        even when odom is fresh — the obvious overlap that the older
        five-source draft missed.
        """
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx[stale_key] = stale_value
        assert stale_sources(
            now_monotonic_s=now, timeouts=timeouts, **rx
        ) == expected

    def test_stale_joint_states_with_fresh_odom_still_trips(self, timeouts):
        """Half-fresh obs (joint_states stale, odom fresh) is exactly
        the silent-failure mode the brief's 6-source upgrade exists to
        prevent.
        """
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_joint_states_rx_t"] = now - 0.3
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, **rx
        )
        assert "joint_states" in result
        assert "odom" not in result

    def test_order_matches_brief_enumeration(self, timeouts):
        """Log lines read consistently when multiple sources trip."""
        now = 100.0
        result = stale_sources(
            now_monotonic_s=now,
            last_goal_rx_t=None,
            last_imu_rx_t=None,
            last_joint_states_rx_t=None,
            last_odom_rx_t=None,
            last_depth_rx_t=None,
            tf_age_s=None,
            timeouts=timeouts,
        )
        assert result == [
            "goal", "imu", "joint_states", "odom", "depth", "tf",
        ]


class TestDepthEnabled:
    """No-camera variants (no depth field) drop the depth source: it must
    not trip the watchdog forever on a topic the variant never subscribes.
    """

    def test_disabled_drops_depth_even_when_absent(self, timeouts):
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_depth_rx_t"] = None  # would be stale if the source were on
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, depth_enabled=False, **rx
        )
        assert "depth" not in result
        assert result == []

    def test_disabled_drops_depth_even_when_stale(self, timeouts):
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_depth_rx_t"] = now - 5.0  # far past timeouts.depth
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, depth_enabled=False, **rx
        )
        assert "depth" not in result

    def test_disabled_still_reports_other_stale_sources(self, timeouts):
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_depth_rx_t"] = None
        rx["last_odom_rx_t"] = now - 5.0  # stale odom
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, depth_enabled=False, **rx
        )
        # Order-exact: depth dropped entirely, only the stale odom remains.
        assert result == ["odom"]

    def test_enabled_is_the_default(self, timeouts):
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_depth_rx_t"] = None
        result = stale_sources(now_monotonic_s=now, timeouts=timeouts, **rx)
        assert "depth" in result


class TestGoalActive:
    """An accepted ``navigate_to_pose`` action goal is latched for the
    whole mission — nothing restamps ``last_goal_rx_t`` while it runs, so
    ``goal_active`` models the goal source as active-goal presence. The
    rx-time path remains as the topic-driven (live-update / mid-mission
    reset) fallback.
    """

    def test_active_goal_fresh_with_no_topic_msg(self, timeouts):
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_goal_rx_t"] = None
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, goal_active=True, **rx
        )
        assert "goal" not in result
        assert result == []

    def test_active_goal_with_fresh_topic_msg_stays_fresh(self, timeouts):
        # The first goal_timeout_s of every mission lives in this cell:
        # the accept-time stamp is still fresh AND the goal is active.
        now = 100.0
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, goal_active=True,
            **_all_fresh_rx_times(now),
        )
        assert result == []

    def test_active_goal_fresh_with_stale_topic_msg(self, timeouts):
        # Missions outlive timeouts.goal by design; the accept-time stamp
        # going stale must not zero-twist a running mission.
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_goal_rx_t"] = now - 30.0
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, goal_active=True, **rx
        )
        assert "goal" not in result

    def test_idle_with_no_topic_msg_still_trips(self, timeouts):
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_goal_rx_t"] = None
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, goal_active=False, **rx
        )
        assert result == ["goal"]

    def test_idle_with_stale_topic_msg_still_trips(self, timeouts):
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_goal_rx_t"] = now - 1.5
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, goal_active=False, **rx
        )
        assert result == ["goal"]

    def test_inactive_is_the_default(self, timeouts):
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_goal_rx_t"] = None
        result = stale_sources(now_monotonic_s=now, timeouts=timeouts, **rx)
        assert "goal" in result

    def test_active_goal_does_not_mask_other_sources(self, timeouts):
        now = 100.0
        rx = _all_fresh_rx_times(now)
        rx["last_goal_rx_t"] = None
        rx["last_odom_rx_t"] = now - 5.0
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, goal_active=True, **rx
        )
        assert result == ["odom"]


class TestSubgoalSource:
    """Rolling-subgoal (hybrid) freshness: the inference-side half of the
    plan-freshness guard. Off by default; on for subgoal variants, checked
    against the longer ``timeouts.path`` budget.
    """

    def test_disabled_by_default(self, timeouts):
        now = 100.0
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, **_all_fresh_rx_times(now)
        )
        assert "subgoal" not in result

    def test_fresh_subgoal_not_stale(self, timeouts):
        now = 100.0
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, subgoal_enabled=True,
            last_subgoal_rx_t=now - 0.5, **_all_fresh_rx_times(now),
        )
        assert result == []

    def test_stale_subgoal_trips(self, timeouts):
        now = 100.0
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, subgoal_enabled=True,
            last_subgoal_rx_t=now - 5.0, **_all_fresh_rx_times(now),
        )
        assert result == ["subgoal"]

    def test_subgoal_boundary_at_one_second_budget(self, timeouts):
        # timeouts.path defaults to the ~1.0 s inference half of the split
        # stale-plan budget (the generator carries the other ~1.0 s).
        assert timeouts.path == pytest.approx(1.0)
        now = 100.0
        fresh = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, subgoal_enabled=True,
            last_subgoal_rx_t=now - 0.95, **_all_fresh_rx_times(now),
        )
        stale = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, subgoal_enabled=True,
            last_subgoal_rx_t=now - 1.05, **_all_fresh_rx_times(now),
        )
        assert "subgoal" not in fresh
        assert stale == ["subgoal"]

    def test_none_subgoal_trips(self, timeouts):
        now = 100.0
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, subgoal_enabled=True,
            last_subgoal_rx_t=None, **_all_fresh_rx_times(now),
        )
        assert "subgoal" in result

    def test_subgoal_appears_last_in_order(self, timeouts):
        now = 100.0
        rx = {k: None for k in _all_fresh_rx_times(now)}  # all stale
        result = stale_sources(
            now_monotonic_s=now, timeouts=timeouts, subgoal_enabled=True,
            last_subgoal_rx_t=None, **rx,
        )
        assert result == [
            "goal", "imu", "joint_states", "odom", "depth", "tf", "subgoal",
        ]
