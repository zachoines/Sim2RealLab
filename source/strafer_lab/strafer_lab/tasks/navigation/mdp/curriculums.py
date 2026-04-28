"""Curriculum learning functions for Strafer navigation task.

Progressively increases task difficulty during training:
- Goal distance: Start close (2m), expand to full range (5m)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
import warp as wp


class GoalDistanceCurriculum(ManagerTermBase):
    """Progressively expands goal sampling range based on success rate.

    Tracks per-environment goal-reach successes and promotes difficulty
    when the agent consistently reaches goals. The goal_range on the
    GoalCommandCfg is updated to reflect the current difficulty level.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._success_count = torch.zeros(env.num_envs, device=env.device)
        self._difficulty = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "goal_command",
        initial_range: float = 2.0,
        max_range: float = 5.0,
        step_size: float = 0.5,
        success_threshold: int = 5,
        goal_threshold: float = 0.3,
    ) -> float:
        """Evaluate and update goal distance curriculum.

        Args:
            env: Environment instance.
            env_ids: Environments that just reset.
            command_name: Name of the goal command term.
            initial_range: Starting goal sampling radius (meters).
            max_range: Maximum goal sampling radius (meters).
            step_size: Increase in range per difficulty level.
            success_threshold: Consecutive successes needed to promote.
            goal_threshold: Distance (m) for counting a goal as reached.

        Returns:
            Mean current goal range across all environments.
        """
        if len(env_ids) == 0:
            return float(self._current_range().mean().item())

        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(
            env_ids, device=env.device
        )

        # Check if any goals were reached during the episode (not at reset instant).
        # GoalCommand resamples goals mid-episode, so checking distance at reset
        # would always fail. Instead, read the per-episode reach counter.
        command_term = env.command_manager.get_term(command_name)
        reached = command_term._goal_reached_count[env_ids_t] > 0

        # Update success counts — increment for reached, reset for not
        self._success_count[env_ids_t] = torch.where(
            reached,
            self._success_count[env_ids_t] + 1,
            torch.zeros_like(self._success_count[env_ids_t]),
        )

        # Promote if threshold met
        promote = self._success_count[env_ids_t] >= success_threshold
        if promote.any():
            promote_ids = env_ids_t[promote]
            self._difficulty[promote_ids] += 1
            self._success_count[promote_ids] = 0

        # Compute current range and update the command config
        current_range = self._current_range()
        mean_range = float(current_range.mean().item())

        # Update goal_range on the command term config
        command_term.cfg.goal_range.pos_x = (-mean_range, mean_range)
        command_term.cfg.goal_range.pos_y = (-mean_range, mean_range)

        return mean_range

    def _current_range(self) -> torch.Tensor:
        p = self.cfg.params
        initial = p.get("initial_range", 2.0)
        maximum = p.get("max_range", 5.0)
        step = p.get("step_size", 0.5)
        return torch.clamp(initial + self._difficulty * step, max=maximum)


class ObstacleCurriculum(ManagerTermBase):
    """Progressively activates more obstacles based on goal-reach success.

    Inactive obstacles are teleported far away (100m, 100m) so they don't
    interfere. Active ones are placed by the randomize_obstacles event.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        init_count = cfg.params.get("initial_count", 2)
        self._active_count = torch.full(
            (env.num_envs,), init_count, dtype=torch.long, device=env.device
        )
        self._success_count = torch.zeros(env.num_envs, device=env.device)

    @property
    def active_obstacle_count(self) -> torch.Tensor:
        """Per-environment count of active obstacles."""
        return self._active_count

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "goal_command",
        initial_count: int = 2,
        max_count: int = 8,
        step_size: int = 2,
        success_threshold: int = 10,
        goal_threshold: float = 0.3,
    ) -> float:
        """Evaluate and update obstacle curriculum.

        Args:
            env: Environment instance.
            env_ids: Environments that just reset.
            command_name: Name of the goal command term.
            initial_count: Starting number of active obstacles.
            max_count: Maximum number of obstacles.
            step_size: Obstacles added per promotion.
            success_threshold: Consecutive successes needed to promote.
            goal_threshold: Distance (m) for counting a goal as reached.

        Returns:
            Mean active obstacle count across all environments.
        """
        if len(env_ids) == 0:
            return float(self._active_count.float().mean().item())

        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(
            env_ids, device=env.device
        )

        # Check if any goals were reached during the episode
        command_term = env.command_manager.get_term(command_name)
        reached = command_term._goal_reached_count[env_ids_t] > 0

        self._success_count[env_ids_t] = torch.where(
            reached,
            self._success_count[env_ids_t] + 1,
            torch.zeros_like(self._success_count[env_ids_t]),
        )

        # Promote
        promote = self._success_count[env_ids_t] >= success_threshold
        if promote.any():
            promote_ids = env_ids_t[promote]
            self._active_count[promote_ids] = torch.clamp(
                self._active_count[promote_ids] + step_size, max=max_count
            )
            self._success_count[promote_ids] = 0

        # Deactivate excess obstacles by moving them far away
        _deactivate_excess_obstacles(env, env_ids_t, self._active_count)

        return float(self._active_count.float().mean().item())


def _deactivate_excess_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    active_counts: torch.Tensor,
) -> None:
    """Move inactive obstacles far offscreen (100, 100, -10)."""
    for i in range(8):
        obs_name = f"obstacle_{i}"
        try:
            obstacle = env.scene[obs_name]
        except KeyError:
            continue

        # Environments where this obstacle index exceeds active count
        should_hide = active_counts[env_ids] <= i
        if not should_hide.any():
            continue

        hide_ids = env_ids[should_hide]
        root_pose = wp.to_torch(obstacle.data.default_root_pose)[hide_ids].clone()
        root_pose[:, 0] = 100.0
        root_pose[:, 1] = 100.0
        root_pose[:, 2] = -10.0
        obstacle.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=hide_ids)
        obstacle.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(hide_ids), 6, device=env.device), env_ids=hide_ids
        )


class RoomComplexityCurriculum(ManagerTermBase):
    """Progressively increases procedural room complexity based on goal-reach success.

    Tracks per-environment consecutive successes and advances through
    difficulty levels that control the number of internal walls, furniture,
    and clutter objects placed by ``generate_proc_room``.

    Difficulty levels:
        0: Open field — just goals, no walls or obstacles
        1: Scattered obstacles on open ground (2 furniture, 4 clutter)
        2: Empty rectangular room
        3: Room + sparse furniture (2)
        4: Room + moderate obstacles (4 furniture, 4 clutter)
        5: Room + internal wall + clutter (1 wall, 4 furniture, 8 clutter)
        6: Room + dense clutter (1 wall, 6 furniture, 12 clutter)
        7: Full complexity (2 walls, 8 furniture, 16 clutter)
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        init_level = cfg.params.get("initial_level", 0)
        self._difficulty = torch.full(
            (env.num_envs,), init_level, dtype=torch.long, device=env.device
        )
        self._success_count = torch.zeros(env.num_envs, device=env.device)
        # Store on env so generate_proc_room can read it
        env._proc_room_difficulty = self._difficulty

    @property
    def difficulty(self) -> torch.Tensor:
        """Per-environment difficulty level."""
        return self._difficulty

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "goal_command",
        initial_level: int = 0,
        max_level: int = 7,
        success_threshold: int = 10,
        goal_threshold: float = 0.3,
    ) -> float:
        """Evaluate and update room complexity curriculum.

        Args:
            env: Environment instance.
            env_ids: Environments that just reset.
            command_name: Name of the goal command term.
            initial_level: Starting difficulty level.
            max_level: Maximum difficulty level.
            success_threshold: Consecutive successes needed to promote.
            goal_threshold: Distance (m) for counting a goal as reached.

        Returns:
            Mean difficulty level across all environments.
        """
        if len(env_ids) == 0:
            return float(self._difficulty.float().mean().item())

        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.tensor(
            env_ids, device=env.device
        )

        # Check if any goals were reached during the episode (not at reset instant).
        # GoalCommand resamples goals mid-episode, so checking distance at reset
        # would always fail. Instead, read the per-episode reach counter.
        command_term = env.command_manager.get_term(command_name)
        reached = command_term._goal_reached_count[env_ids_t] > 0

        self._success_count[env_ids_t] = torch.where(
            reached,
            self._success_count[env_ids_t] + 1,
            torch.zeros_like(self._success_count[env_ids_t]),
        )

        # Promote
        promote = self._success_count[env_ids_t] >= success_threshold
        if promote.any():
            promote_ids = env_ids_t[promote]
            self._difficulty[promote_ids] = torch.clamp(
                self._difficulty[promote_ids] + 1, max=max_level
            )
            self._success_count[promote_ids] = 0

        return float(self._difficulty.float().mean().item())
