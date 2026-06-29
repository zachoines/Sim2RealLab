"""Unit tests for the rclpy-free rolling-subgoal generator.

The load-bearing test is subgoal-POSITION parity: for a battery of paths
and robot poses, the generator's picked subgoal must land within 10 cm of
the hand-computed training cursor rule (at the deployed 1.0 m lookahead).
That single quantity drives all three subgoal obs fields the policy reads
(relative position / distance / bearing-to-position), so position parity
pins the observed heading too.

Expected values are computed by hand from the cursor algorithm (no torch
import) because the torch reference (strafer_lab's cursor.py) is not
importable here. An optional cross-check against that torch reference,
loaded by file path, runs only where torch is installed.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from strafer_inference.generator import RollingSubgoalGenerator, SubgoalState
from strafer_shared.constants import SUBGOAL_LOOKAHEAD_M

# Deployed lookahead per the resolved operator decision: fixed at the
# shared constant (1.0 m), not the robust-tier band.
LOOKAHEAD = SUBGOAL_LOOKAHEAD_M
GATE_TOL_M = 0.10  # the load-bearing parity bound

_SQRT2 = math.sqrt(2.0)


# =============================================================================
# Position parity battery (load-bearing): hand-computed expected subgoals
# =============================================================================

# Each case: (name, path, robot_xy, expected_subgoal, expected_heading_rad).
# All computed at LOOKAHEAD = 1.0 m. See the module docstring; the arithmetic
# for each is straightforward arc-length bookkeeping.
PARITY_BATTERY = [
    # Straight +x path, robot on the path: project at s=0.5, target s=1.5.
    (
        "straight_on_path",
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        (0.5, 0.0),
        (1.5, 0.0),
        0.0,
    ),
    # Straight +x path, robot offset in +y: closest projection is still
    # s=0.5 (cross-track 0.3 does not change the arc position).
    (
        "straight_offset",
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        (0.5, 0.3),
        (1.5, 0.0),
        0.0,
    ),
    # L-shape, robot at the origin: target s=1.0 lands mid first segment.
    (
        "l_shape_start",
        [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
        (0.0, 0.0),
        (1.0, 0.0),
        0.0,
    ),
    # L-shape, robot mid first leg: target s=2.5 crosses the corner into
    # the second (vertical) leg -> subgoal (2, 0.5), tangent +pi/2.
    (
        "l_shape_cross_corner",
        [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
        (1.5, 0.0),
        (2.0, 0.5),
        math.pi / 2,
    ),
    # L-shape, robot near the corner: target s=2.8 sits 0.8 up the second
    # leg -> subgoal (2, 0.8).
    (
        "l_shape_near_corner",
        [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)],
        (1.8, 0.0),
        (2.0, 0.8),
        math.pi / 2,
    ),
    # Diagonal zig-zag, robot at start: target s=1.0 lands one metre along
    # the first 45-degree segment -> (1/sqrt2, 1/sqrt2).
    (
        "diagonal_start",
        [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)],
        (0.0, 0.0),
        (1.0 / _SQRT2, 1.0 / _SQRT2),
        math.pi / 4,
    ),
    # Target lands EXACTLY on an interior vertex (target s=1.0 == arc[1]):
    # j selects the segment starting at that vertex, frac=0, so the subgoal
    # is the vertex itself with the NEXT segment's tangent (+pi/2 here).
    (
        "target_on_vertex",
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        (0.0, 0.0),
        (1.0, 0.0),
        math.pi / 2,
    ),
    # Negative-y segment tangent: exercises atan2 below the x-axis (the
    # straight/right-angle/45-degree fixtures only cover 0, +pi/2, +pi/4).
    (
        "downward_tangent",
        [(0.0, 0.0), (1.0, 0.0), (1.0, -1.0)],
        (0.9, 0.0),
        (1.0, -0.9),
        -math.pi / 2,
    ),
]


@pytest.mark.parametrize(
    "name, path, robot, expected_subgoal, expected_heading",
    PARITY_BATTERY,
    ids=[c[0] for c in PARITY_BATTERY],
)
class TestSubgoalPositionParity:
    """The merge gate: numpy generator reproduces the training cursor's
    subgoal POSITION within 10 cm across straight, multi-segment, and
    curved paths.
    """

    def test_position_within_gate(
        self, name, path, robot, expected_subgoal, expected_heading
    ):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array(path))
        state = gen.update(np.array(robot))
        assert state is not None
        err = float(np.linalg.norm(state.subgoal_xy - np.array(expected_subgoal)))
        assert err <= GATE_TOL_M, (
            f"{name}: subgoal {state.subgoal_xy} is {err:.4f} m from "
            f"expected {expected_subgoal} (> {GATE_TOL_M} m gate)"
        )

    def test_position_matches_analytic_exactly(
        self, name, path, robot, expected_subgoal, expected_heading
    ):
        """Hand-computed fixtures are exact, so the port should match far
        tighter than the 10 cm gate.
        """
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array(path))
        state = gen.update(np.array(robot))
        np.testing.assert_allclose(
            state.subgoal_xy, expected_subgoal, atol=1e-9
        )


# =============================================================================
# Tangent-heading fixture check (non-gating): verifies the numpy port, NOT
# what the policy observes.
# =============================================================================


@pytest.mark.parametrize(
    "name, path, robot, expected_subgoal, expected_heading",
    PARITY_BATTERY,
    ids=[c[0] for c in PARITY_BATTERY],
)
def test_tangent_heading_matches_analytic(
    name, path, robot, expected_subgoal, expected_heading
):
    gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
    gen.set_path(np.array(path))
    state = gen.update(np.array(robot))
    assert abs(state.subgoal_heading - expected_heading) <= 1e-3, (
        f"{name}: heading {state.subgoal_heading} != {expected_heading}"
    )


# =============================================================================
# Edge cases
# =============================================================================


class TestNoPathYet:
    def test_update_returns_none_before_any_path(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        assert gen.has_path is False
        assert gen.update(np.array([0.0, 0.0])) is None

    def test_total_and_cursor_zero_before_path(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        assert gen.total_arc == 0.0
        assert gen.cursor_arc == 0.0


class TestSinglePointPath:
    def test_subgoal_is_the_lone_point(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array([(2.0, 3.0)]))
        state = gen.update(np.array([0.0, 0.0]))
        assert state is not None
        np.testing.assert_allclose(state.subgoal_xy, [2.0, 3.0])
        assert state.total_arc == 0.0

    def test_heading_is_bearing_robot_to_point(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array([(2.0, 3.0)]))
        state = gen.update(np.array([0.0, 0.0]))
        # Bearing from robot to the point, NOT a segment tangent.
        assert state.subgoal_heading == pytest.approx(math.atan2(3.0, 2.0))

    def test_cross_track_and_end_distance_are_distance_to_point(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array([(2.0, 3.0)]))
        state = gen.update(np.array([0.0, 0.0]))
        assert state.cross_track == pytest.approx(math.hypot(2.0, 3.0))
        assert state.end_distance == pytest.approx(math.hypot(2.0, 3.0))

    def test_cursor_cannot_advance_on_single_point(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array([(2.0, 3.0)]))
        state = gen.update(np.array([10.0, 10.0]))
        assert state.cursor_arc == 0.0
        assert state.along_track_progress == 0.0


class TestZeroLengthSegments:
    """Duplicate waypoints make a zero-length interior segment; the
    clamped denominators must keep the projection finite (no NaN) and the
    pick identical to the dup-free path.
    """

    def test_duplicate_waypoint_no_nan_and_correct_pick(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (2.0, 0.0)]))
        state = gen.update(np.array([0.5, 0.0]))
        assert np.all(np.isfinite(state.subgoal_xy))
        np.testing.assert_allclose(state.subgoal_xy, [1.5, 0.0], atol=1e-9)

    def test_projection_onto_duplicated_point_is_finite(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (2.0, 0.0)]))
        state = gen.update(np.array([1.0, 0.5]))
        assert np.all(np.isfinite(state.subgoal_xy))
        assert math.isfinite(state.cross_track)
        assert state.cross_track == pytest.approx(0.5)
        # Robot projects onto the duplicated point at arc s=1.0.
        assert state.cursor_arc == pytest.approx(1.0)


class TestHeadFirstTruncation:
    """A path longer than max_points keeps the first max_points waypoints
    (tail dropped), mirroring the training cursor's fixed-width buffer.
    """

    def test_tail_is_dropped(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD, max_points=3)
        gen.set_path(
            np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
        )
        # Effective path is [(0,0),(1,0),(2,0)] -> total arc 2.0, not 4.0.
        assert gen.total_arc == pytest.approx(2.0)

    def test_subgoal_pins_to_last_kept_waypoint(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD, max_points=3)
        gen.set_path(
            np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
        )
        state = gen.update(np.array([1.5, 0.0]))
        np.testing.assert_allclose(state.subgoal_xy, [2.0, 0.0], atol=1e-9)
        assert state.end_distance == pytest.approx(0.5)

    def test_max_points_below_two_rejected(self):
        with pytest.raises(ValueError, match="max_points must be >= 2"):
            RollingSubgoalGenerator(lookahead_m=LOOKAHEAD, max_points=1)


class TestTargetBeyondPathEnd:
    def test_subgoal_pins_to_last_point(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array([(0.0, 0.0), (1.0, 0.0)]))
        state = gen.update(np.array([0.9, 0.0]))
        # target = 0.9 + 1.0 = 1.9, clamped to total arc 1.0 -> last point.
        np.testing.assert_allclose(state.subgoal_xy, [1.0, 0.0], atol=1e-9)
        assert state.subgoal_heading == pytest.approx(0.0)


class TestMonotonicNonRetreat:
    PATH = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]

    def test_cursor_never_retreats_when_robot_backs_up(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array(self.PATH))
        s1 = gen.update(np.array([2.0, 0.0]))
        assert s1.cursor_arc == pytest.approx(2.0)
        # Robot backs up to s=1.0: cursor holds at 2.0, progress is 0.
        s2 = gen.update(np.array([1.0, 0.0]))
        assert s2.cursor_arc == pytest.approx(2.0)
        assert s2.along_track_progress == pytest.approx(0.0)

    def test_subgoal_never_retreats(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array(self.PATH))
        s1 = gen.update(np.array([2.0, 0.0]))
        s2 = gen.update(np.array([0.5, 0.0]))  # re-projects nearer the start
        # Subgoal arc position cannot move backward along the path.
        assert s2.subgoal_xy[0] >= s1.subgoal_xy[0] - 1e-9
        assert s2.cursor_arc >= s1.cursor_arc - 1e-9

    def test_progress_is_nonnegative_each_tick(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array(self.PATH))
        prev = 0.0
        for x in (0.5, 0.2, 1.0, 0.8, 3.0, 2.0):
            s = gen.update(np.array([x, 0.0]))
            assert s.along_track_progress >= -1e-12
            assert s.cursor_arc >= prev - 1e-12
            prev = s.cursor_arc


class TestLookaheadSaturatesAtEnd:
    def test_subgoal_saturates_at_last_waypoint(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
        gen.set_path(np.array(path))
        for x in (3.2, 3.6, 4.0):
            s = gen.update(np.array([x, 0.0]))
            np.testing.assert_allclose(s.subgoal_xy, [4.0, 0.0], atol=1e-9)


class TestNewPathRewind:
    def test_new_path_resets_cursor_to_zero(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]))
        gen.update(np.array([2.0, 0.0]))
        assert gen.cursor_arc == pytest.approx(2.0)
        # A fresh plan rewinds the cursor.
        gen.set_path(np.array([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]))
        assert gen.cursor_arc == 0.0
        s = gen.update(np.array([0.0, 0.0]))
        assert s.cursor_arc == pytest.approx(0.0)
        np.testing.assert_allclose(s.subgoal_xy, [0.0, 1.0], atol=1e-9)


class TestArgminTieBreak:
    """On equidistant segments numpy argmin picks the first index, which
    matches torch.argmin. A symmetric 'V' puts the robot equidistant from
    both legs; the first leg must win.
    """

    def test_first_segment_wins_on_tie(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array([(0.0, 1.0), (1.0, 0.0), (2.0, 1.0)]))
        state = gen.update(np.array([1.0, 1.0]))
        # Projecting onto the FIRST leg gives s = 0.5 * sqrt(2); the second
        # leg would give 1.5 * sqrt(2).
        assert state.cursor_arc == pytest.approx(0.5 * _SQRT2)


class TestInputValidation:
    def test_non_positive_lookahead_rejected(self):
        with pytest.raises(ValueError, match="lookahead_m must be > 0"):
            RollingSubgoalGenerator(lookahead_m=0.0)

    def test_bad_path_shape_rejected(self):
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        with pytest.raises(ValueError, match=r"expected \(N, 2\)"):
            gen.set_path(np.array([0.0, 1.0, 2.0]))

    def test_lookahead_defaults_to_shared_constant(self):
        gen = RollingSubgoalGenerator()
        assert gen.lookahead_m == SUBGOAL_LOOKAHEAD_M


# =============================================================================
# Optional torch cross-check against the training cursor (skips without torch)
# =============================================================================


def _load_torch_cursor_class():
    """Load strafer_lab's PathCursor by file path.

    cursor.py imports only torch + dataclasses (no relative / isaaclab
    imports), so it loads standalone. Returns None if the file is absent.
    """
    repo_root = Path(__file__).resolve().parents[4]
    cursor_path = (
        repo_root
        / "source/strafer_lab/strafer_lab/tasks/navigation/path_planner/cursor.py"
    )
    if not cursor_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("cursor_torch_ref", cursor_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PathCursor


class TestTorchCrossCheck:
    """Bonus parity check on a torch-equipped machine; skips cleanly where
    torch (and thus the training cursor) is unavailable.
    """

    def _torch_pick(self, path, robot):
        torch = pytest.importorskip("torch")
        PathCursor = _load_torch_cursor_class()
        if PathCursor is None:
            pytest.skip("strafer_lab cursor.py not present")
        max_points = max(len(path), 2)
        cur = PathCursor(num_envs=1, max_points=max_points, device="cpu")
        cur.set_paths(
            torch.tensor([0]),
            [torch.tensor(path, dtype=torch.float32)],
        )
        state = cur.update(
            torch.tensor([robot], dtype=torch.float32), float(LOOKAHEAD)
        )
        return (
            state.subgoal_xy[0].numpy(),
            float(state.subgoal_heading[0]),
        )

    @pytest.mark.parametrize(
        "name, path, robot, expected_subgoal, expected_heading",
        PARITY_BATTERY,
        ids=[c[0] for c in PARITY_BATTERY],
    )
    def test_matches_torch_cursor(
        self, name, path, robot, expected_subgoal, expected_heading
    ):
        torch_subgoal, torch_heading = self._torch_pick(path, robot)
        gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
        gen.set_path(np.array(path))
        state = gen.update(np.array(robot))
        np.testing.assert_allclose(state.subgoal_xy, torch_subgoal, atol=1e-4)
        assert abs(state.subgoal_heading - torch_heading) <= 1e-4


def test_subgoalstate_is_dataclass_with_expected_fields():
    """The node consumes these fields by name; guard the public surface."""
    gen = RollingSubgoalGenerator(lookahead_m=LOOKAHEAD)
    gen.set_path(np.array([(0.0, 0.0), (1.0, 0.0)]))
    state = gen.update(np.array([0.0, 0.0]))
    assert isinstance(state, SubgoalState)
    for field in (
        "subgoal_xy",
        "subgoal_heading",
        "cross_track",
        "along_track_progress",
        "end_distance",
        "cursor_arc",
        "total_arc",
    ):
        assert hasattr(state, field)
