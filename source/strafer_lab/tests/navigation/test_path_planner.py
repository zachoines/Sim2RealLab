"""Unit tests for the grid A* path planner.

Pure numpy — no Isaac Sim, no torch. Grids mirror the procedural-room
rasterization convention: row = x index, col = y index, origin at the
bottom-left corner, cell centers at ``idx * res + origin + res / 2``.
"""

from __future__ import annotations

import numpy as np
import pytest

from strafer_lab.tasks.navigation.path_planner import (
    InvalidEndpointError,
    NoPathError,
    perturb_waypoints,
    plan_path,
    resample_polyline,
)

GRID = 80
RES = 0.1
ORIGIN = (-GRID * RES / 2.0, -GRID * RES / 2.0)
SPACING = 0.05


def _xy_to_cell(x: float, y: float) -> tuple[int, int]:
    return (
        int(np.floor((x - ORIGIN[0]) / RES)),
        int(np.floor((y - ORIGIN[1]) / RES)),
    )


def _open_grid() -> np.ndarray:
    return np.ones((GRID, GRID), dtype=bool)


def _grid_with_doorway(gap_lo: int = 38, gap_hi: int = 42) -> np.ndarray:
    """Wall across the middle row (x = 0) with a doorway gap in y."""
    free = _open_grid()
    free[40, :] = False
    free[40, gap_lo:gap_hi] = True
    return free


def _assert_collision_free(path: np.ndarray, free: np.ndarray) -> None:
    for x, y in path:
        r, c = _xy_to_cell(float(x), float(y))
        assert free[r, c], f"waypoint ({x:.3f}, {y:.3f}) lies in occupied cell ({r}, {c})"


def _arc_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def test_plan_through_doorway_is_collision_free_and_bounded():
    free = _grid_with_doorway()
    start = np.array([-2.0, -2.0])
    goal = np.array([2.0, 2.0])
    path = plan_path(
        start, goal, free,
        grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
    )
    assert path.shape[0] >= 2 and path.shape[1] == 2
    np.testing.assert_allclose(path[0], start, atol=1e-5)
    np.testing.assert_allclose(path[-1], goal, atol=1e-5)
    _assert_collision_free(path, free)
    straight = float(np.linalg.norm(goal - start))
    assert straight <= _arc_length(path) <= 3.0 * straight


def test_plan_routes_through_the_gap_not_the_wall():
    free = _grid_with_doorway(gap_lo=38, gap_hi=42)
    path = plan_path(
        np.array([-2.0, 0.0]), np.array([2.0, 0.0]), free,
        grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
    )
    # Where the path crosses x = 0 it must be inside the doorway's y-band.
    crossing_ys = path[np.abs(path[:, 0]) < 0.15][:, 1]
    assert len(crossing_ys) > 0
    y_lo = 38 * RES + ORIGIN[1]
    y_hi = 42 * RES + ORIGIN[1]
    assert ((crossing_ys >= y_lo - RES) & (crossing_ys <= y_hi + RES)).all()


def test_waypoints_are_uniformly_spaced():
    free = _open_grid()
    path = plan_path(
        np.array([-3.0, -3.0]), np.array([3.0, 2.0]), free,
        grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
    )
    deltas = np.linalg.norm(np.diff(path, axis=0), axis=1)
    # Interior spacing is exact; the final segment may be shorter.
    assert (deltas[:-1] <= SPACING + 1e-5).all()
    assert (deltas[:-1] >= SPACING - 1e-5).all()
    assert deltas[-1] <= SPACING + 1e-5


def test_no_path_raises():
    free = _open_grid()
    free[40, :] = False  # sealed wall, no doorway
    with pytest.raises(NoPathError):
        plan_path(
            np.array([-2.0, 0.0]), np.array([2.0, 0.0]), free,
            grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
        )


def test_coincident_endpoints_raise():
    with pytest.raises(InvalidEndpointError):
        plan_path(
            np.array([1.0, 1.0]), np.array([1.0, 1.0]), _open_grid(),
            grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
        )


def test_endpoint_outside_grid_raises():
    with pytest.raises(InvalidEndpointError):
        plan_path(
            np.array([-100.0, 0.0]), np.array([2.0, 0.0]), _open_grid(),
            grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
        )


def test_start_deep_in_obstacle_raises():
    free = _open_grid()
    free[10:30, 10:30] = False  # 2 m x 2 m block
    blocked_center = np.array([
        20 * RES + ORIGIN[0] + RES / 2.0,
        20 * RES + ORIGIN[1] + RES / 2.0,
    ])
    with pytest.raises(InvalidEndpointError):
        plan_path(
            blocked_center, np.array([3.0, 3.0]), free,
            grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
            snap_radius_m=0.3,
        )


def test_start_just_inside_obstacle_snaps_to_free():
    free = _open_grid()
    free[40:43, :] = False  # 0.3 m-thick slab
    # One cell deep into the slab; nearest free cell is within snap radius.
    start = np.array([40 * RES + ORIGIN[0] + RES / 2.0, 0.0])
    path = plan_path(
        start, np.array([-3.0, 0.0]), free,
        grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
        snap_radius_m=0.3,
    )
    # The path starts at the robot's exact (occupied) pose, so its head may
    # cross the occupied cell — but it must escape within the snap radius
    # and stay free from there on.
    dist_from_start = np.linalg.norm(path - start, axis=1)
    _assert_collision_free(path[dist_from_start > 0.3], free)
    np.testing.assert_allclose(path[0], start, atol=1e-5)


def test_resample_polyline_preserves_endpoints():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    out = resample_polyline(pts, 0.05)
    np.testing.assert_allclose(out[0], pts[0], atol=1e-6)
    np.testing.assert_allclose(out[-1], pts[-1], atol=1e-6)
    assert len(out) == pytest.approx(2.0 / 0.05, abs=2)


def test_perturbation_is_bounded_and_stays_free():
    free = _grid_with_doorway()
    path = plan_path(
        np.array([-2.0, -2.0]), np.array([2.0, 2.0]), free,
        grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
    )
    noise_std = 0.025
    rng = np.random.default_rng(7)
    noised = perturb_waypoints(
        path, noise_std, free, grid_res=RES, grid_origin_xy=ORIGIN, rng=rng,
    )
    # Endpoints exact.
    np.testing.assert_allclose(noised[0], path[0], atol=1e-6)
    np.testing.assert_allclose(noised[-1], path[-1], atol=1e-6)
    # Truncated at two sigma per axis -> displacement <= 2 * sigma * sqrt(2).
    disp = np.linalg.norm(noised - path, axis=1)
    assert disp.max() <= 2.0 * noise_std * np.sqrt(2.0) + 1e-6
    assert disp.max() > 0.0  # noise actually applied
    _assert_collision_free(noised, free)


def test_zero_noise_is_identity():
    free = _open_grid()
    path = plan_path(
        np.array([-2.0, 0.0]), np.array([2.0, 0.0]), free,
        grid_res=RES, grid_origin_xy=ORIGIN, discretization_m=SPACING,
    )
    out = perturb_waypoints(
        path, 0.0, free, grid_res=RES, grid_origin_xy=ORIGIN,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(out, path)
