"""Externally-fed rolling-subgoal command for scene-agnostic capture.

The trained no-camera subgoal policy tracks a subgoal that rolls a fixed
lookahead ahead along a planned path; its observation reads whatever the
``goal_command`` term emits. The capture driver needs that same rolling subgoal
on a foreign capture scene, where the training-time :class:`SubgoalCommand`
cannot run: its per-episode planner reads the procedural-room free-space and
spawn buffers that only the procedural-room generator populates, so it raises on
any other scene source.

:class:`CaptureSubgoalCommand` reuses :class:`SubgoalCommand`'s rolling
machinery verbatim — the constructor, the ``PathCursor``, ``_update_command``,
and the path-tracking metrics — and replaces only the episode-reset planner:
instead of sampling and planning a path from the procedural-room buffers, it
installs an externally supplied path through :meth:`set_leg`. The driver plans
each coverage leg with the shared path planner and pushes it in; the term rolls
the subgoal and raises ``path_complete`` exactly as in training, so
``env.get_observations()`` yields the same subgoal observation the policy was
trained against.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

from isaaclab.utils import configclass

from .commands import SubgoalCommand, SubgoalCommandCfg


class CaptureSubgoalCommand(SubgoalCommand):
    """Rolling-subgoal command fed an external path, for scene-agnostic capture.

    Identical to :class:`SubgoalCommand` except the path source: there is no
    internal per-episode planner (which depends on procedural-room-only
    buffers), so the active path is supplied by the driver through
    :meth:`set_leg`. Everything downstream — the lookahead roll, the
    cross-track / along-track state, and the ``path_complete`` flag — is
    inherited unchanged, so the emitted command and the observation built from
    it match the training term.
    """

    cfg: CaptureSubgoalCommandCfg

    def set_leg(self, path: torch.Tensor | Sequence[Sequence[float]]) -> None:
        """Install ``path`` (world-frame ``(N, 2)``) as the active leg.

        Rewinds the cursor onto the new path and emits its initial subgoal so a
        subsequent ``env.get_observations()`` reflects the new leg immediately.
        Single-env capture: the path is applied to env 0.
        """
        env_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        path_t = torch.as_tensor(path, dtype=torch.float32, device=self.device)
        self._install_paths(env_ids, [path_t])

    def _resample_command(self, env_ids: Sequence[int]):
        """Park the subgoal on the robot instead of planning a fresh path.

        The training term plans here from the procedural-room free-space grid;
        on a capture scene those buffers do not exist. Resetting installs a
        one-point path at each robot so the cursor stays valid until the driver
        supplies a leg via :meth:`set_leg`.
        """
        if len(env_ids) == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        env_ids = env_ids.to(dtype=torch.long)
        robot_xy_w = wp.to_torch(self._robot.data.root_pos_w)[:, :2]
        parked = [robot_xy_w[i].reshape(1, 2).clone() for i in env_ids.tolist()]
        self._install_paths(env_ids, parked)

    def _install_paths(self, env_ids: torch.Tensor, paths: list[torch.Tensor]) -> None:
        """Set the cursor paths for ``env_ids`` and emit their initial subgoal.

        Mirrors the tail of :meth:`SubgoalCommand._resample_command` (the part
        after planning): rewind the tracking state and project the robot onto
        the freshly installed path so the command is valid before the next step.
        """
        robot_xy_w = wp.to_torch(self._robot.data.root_pos_w)[:, :2]
        self._path_cursor.set_paths(env_ids, paths)
        self.path_complete[env_ids] = False
        self.dwell_counter[env_ids] = 0
        self.along_track_progress[env_ids] = 0.0
        state = self._path_cursor.update(robot_xy_w, self._lookahead, env_ids=env_ids)
        self._subgoal[env_ids, :2] = state.subgoal_xy
        self._subgoal[env_ids, 2] = state.subgoal_heading
        self.cross_track_error[env_ids] = state.cross_track


@configclass
class CaptureSubgoalCommandCfg(SubgoalCommandCfg):
    """Configuration for the capture-only externally-fed subgoal command."""

    class_type: type = CaptureSubgoalCommand

    # Capture advances legs on *touch*, not parking: the coverage driver breaks
    # its per-leg step loop the instant ``path_complete`` fires, and the
    # non-parking capture policy drives through each leg end without stopping.
    # The dwell-based completion is a training-only shaping change for the
    # trained-policy task; keep the former instant-touch completion here by
    # firing on the first in-radius step with the speed gate disabled.
    dwell_steps: int = 1
    dwell_speed_max_m_s: float = float("inf")
