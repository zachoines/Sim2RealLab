"""Batched arc-length cursor over per-env waypoint paths.

Tracks, for every parallel environment, where the robot is along its current
path (closest-point projection), enforces monotonic cursor advance, and
produces the rolling subgoal a fixed lookahead distance ahead along arc
length. Pure torch on purpose: importable and unit-testable without Isaac
Lab or a running simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1e-6


@dataclass
class PathCursorState:
    """Per-env tracking outputs for one update tick (all shape (B,) unless
    noted).

    Attributes:
        subgoal_xy: (B, 2) lookahead point on the path.
        subgoal_heading: path tangent direction (rad) at the subgoal.
        cross_track: distance from the robot to its closest path point.
        along_track_progress: monotonic cursor advance since the previous
            update (>= 0; zero on the first update after a path change).
        end_distance: distance from the robot to the path's final point.
        cursor_arc: current monotonic arc-length cursor.
        total_arc: total path arc length.
    """

    subgoal_xy: torch.Tensor
    subgoal_heading: torch.Tensor
    cross_track: torch.Tensor
    along_track_progress: torch.Tensor
    end_distance: torch.Tensor
    cursor_arc: torch.Tensor
    total_arc: torch.Tensor


class PathCursor:
    """Holds padded path buffers for ``num_envs`` environments.

    Paths are stored padded to ``max_points`` by repeating the final
    waypoint, so padded segments have zero length and never win the
    closest-segment search nor shift the arc-length table.
    """

    def __init__(self, num_envs: int, max_points: int, device: torch.device | str):
        if max_points < 2:
            raise ValueError("max_points must be >= 2")
        self.num_envs = num_envs
        self.max_points = max_points
        self.device = torch.device(device)
        self._paths = torch.zeros(num_envs, max_points, 2, device=self.device)
        self._path_len = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self._arc = torch.zeros(num_envs, max_points, device=self.device)
        self._cursor = torch.zeros(num_envs, device=self.device)

    @property
    def paths(self) -> torch.Tensor:
        """(B, P, 2) padded waypoint buffer (final waypoint repeated)."""
        return self._paths

    @property
    def path_len(self) -> torch.Tensor:
        """(B,) number of valid waypoints per env."""
        return self._path_len

    @property
    def total_arc(self) -> torch.Tensor:
        """(B,) total arc length per env."""
        last = (self._path_len - 1).clamp(min=0)
        return self._arc.gather(1, last.unsqueeze(1)).squeeze(1)

    def set_paths(self, env_ids: torch.Tensor, paths: list[torch.Tensor]) -> None:
        """Install new paths for ``env_ids`` and rewind their cursors.

        Args:
            env_ids: (K,) env indices.
            paths: K tensors/arrays of shape (N_k, 2), N_k >= 1. Paths longer
                than ``max_points`` are truncated head-first (the truncated
                tail is unreachable within one episode anyway).
        """
        if len(env_ids) != len(paths):
            raise ValueError(
                f"got {len(env_ids)} env ids but {len(paths)} paths"
            )
        for env_id, path in zip(env_ids.tolist(), paths):
            pts = torch.as_tensor(path, dtype=torch.float32, device=self.device)
            if pts.ndim != 2 or pts.shape[-1] != 2 or len(pts) < 1:
                raise ValueError(f"expected (N, 2) path, got shape {tuple(pts.shape)}")
            n = min(len(pts), self.max_points)
            self._paths[env_id, :n] = pts[:n]
            self._paths[env_id, n:] = pts[n - 1]
            self._path_len[env_id] = n
            seg = self._paths[env_id, 1:] - self._paths[env_id, :-1]
            self._arc[env_id, 0] = 0.0
            self._arc[env_id, 1:] = torch.cumsum(seg.norm(dim=-1), dim=0)
            self._cursor[env_id] = 0.0

    def update(
        self,
        robot_xy: torch.Tensor,
        lookahead_m: float | torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> PathCursorState:
        """Advance the cursor toward the robot's projection and emit the
        lookahead subgoal.

        Args:
            robot_xy: (B, 2) world/env-consistent robot positions for ALL
                envs (same frame the paths were stored in).
            lookahead_m: subgoal distance ahead of the projection along arc
                length (clamped to the path end). A scalar applies one
                distance to every env; a ``(num_envs,)`` tensor applies a
                per-env distance (it is indexed by ``env_ids`` internally, so
                always pass the full-width tensor).
            env_ids: optional (K,) subset to compute and mutate; other envs'
                cursors are untouched. Returned tensors are restricted to the
                subset when given.

        Returns:
            :class:`PathCursorState` for the selected envs.
        """
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = env_ids.to(self.device)

        if isinstance(lookahead_m, torch.Tensor):
            lookahead = lookahead_m.to(self.device)[ids]
        else:
            lookahead = lookahead_m

        # Slice the per-env buffers down to the selected envs.
        paths = self._paths[ids]            # (K, P, 2)
        arc = self._arc[ids]                # (K, P)
        plen = self._path_len[ids]          # (K,)
        cursor = self._cursor[ids]          # (K,)
        robot = robot_xy[ids]               # (K, 2)

        # Per-segment geometry: start point ``a``, direction ``d``, length
        # ``seg_len`` (from the arc table). ``valid`` masks the padded tail so
        # only real segments are considered (>=1 segment even on a 1-pt path).
        a = paths[:, :-1, :]                # (K, P-1, 2)
        d = paths[:, 1:, :] - a             # (K, P-1, 2)
        seg_len = arc[:, 1:] - arc[:, :-1]  # (K, P-1)
        seg_idx = torch.arange(self.max_points - 1, device=self.device)
        valid = seg_idx.unsqueeze(0) < (plen - 1).clamp(min=1).unsqueeze(1)

        # Project the robot onto every segment: ``t`` is the clamped [0,1]
        # position along each segment, ``proj`` the foot of that projection,
        # ``dist`` the robot's distance to it. Padded segments -> +inf so they
        # never win the closest-segment search below.
        rel = robot.unsqueeze(1) - a        # (K, P-1, 2)
        t = (rel * d).sum(-1) / seg_len.square().clamp(min=_EPS)
        t = t.clamp(0.0, 1.0)
        proj = a + t.unsqueeze(-1) * d
        dist = (robot.unsqueeze(1) - proj).norm(dim=-1)
        dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))

        # Closest segment -> cross-track error, and the robot's arc-length
        # position on the path: arc to the segment start + fraction into it.
        closest = dist.argmin(dim=1)        # (K,)
        gather1 = closest.unsqueeze(1)
        cross_track = dist.gather(1, gather1).squeeze(1)
        t_star = t.gather(1, gather1).squeeze(1)
        s_closest = (
            arc.gather(1, gather1).squeeze(1)
            + t_star * seg_len.gather(1, gather1).squeeze(1)
        )

        # Single-point paths: the projection is the point itself.
        single = plen <= 1
        if single.any():
            p0 = paths[:, 0, :]
            cross_track = torch.where(
                single, (robot - p0).norm(dim=-1), cross_track
            )
            s_closest = torch.where(single, torch.zeros_like(s_closest), s_closest)

        # Advance the cursor monotonically (never retreats if the robot backs
        # up); the per-step delta is along-track progress.
        new_cursor = torch.maximum(cursor, s_closest)
        progress = new_cursor - cursor
        self._cursor[ids] = new_cursor

        # Subgoal target = cursor + lookahead, clamped to the path's end.
        last = (plen - 1).clamp(min=0)
        total = arc.gather(1, last.unsqueeze(1)).squeeze(1)
        target = (new_cursor + lookahead).clamp(max=total)

        # Locate the segment containing the target arc length, then interpolate
        # the subgoal point within it and read the segment tangent as heading.
        # Padded arc entries repeat the total, so clamp into the valid range.
        j = (arc <= target.unsqueeze(1) + _EPS).sum(dim=1) - 1
        j = torch.minimum(j.clamp(min=0), (plen - 2).clamp(min=0))
        gj = j.unsqueeze(1)
        seg_l = seg_len.gather(1, gj).squeeze(1)
        frac = ((target - arc.gather(1, gj).squeeze(1)) / seg_l.clamp(min=_EPS)).clamp(0.0, 1.0)
        a_j = a.gather(1, gj.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
        d_j = d.gather(1, gj.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
        subgoal = a_j + frac.unsqueeze(-1) * d_j
        heading = torch.atan2(d_j[:, 1], d_j[:, 0])

        # Distance to the final waypoint -> path-completion signal.
        end_pt = paths.gather(1, last.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)).squeeze(1)
        end_distance = (robot - end_pt).norm(dim=-1)

        # Single-point paths: subgoal is the lone point; heading points the
        # robot at it (there is no segment tangent to read).
        if single.any():
            p0 = paths[:, 0, :]
            subgoal = torch.where(single.unsqueeze(-1), p0, subgoal)
            to_p0 = p0 - robot
            heading = torch.where(
                single, torch.atan2(to_p0[:, 1], to_p0[:, 0]), heading
            )

        return PathCursorState(
            subgoal_xy=subgoal,
            subgoal_heading=heading,
            cross_track=cross_track,
            along_track_progress=progress,
            end_distance=end_distance,
            cursor_arc=new_cursor,
            total_arc=total,
        )
