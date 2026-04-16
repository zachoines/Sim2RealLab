"""Ranking-based transit monitor for detecting off-course navigation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .models import Pose2D

_logger = logging.getLogger(__name__)


class TransitMonitor:
    """Tracks nearest-neighbor region drift during navigation.

    Activated when navigate_to_pose starts; uses the semantic map's
    ChromaDB ANN index to classify the robot's current view. If the top-k
    matches consistently fall outside the goal region, reports divergence.
    """

    def __init__(self, semantic_map: Any) -> None:
        self._semantic_map = semantic_map
        self._goal_xy: np.ndarray | None = None
        self._goal_radius_m: float = 3.0
        self._history: list[dict] = []
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self, goal_pose: Any, goal_radius_m: float = 3.0) -> None:
        """Begin monitoring a new navigation leg."""
        self._goal_xy = np.array([goal_pose.x, goal_pose.y])
        self._goal_radius_m = goal_radius_m
        self._history = []
        self._active = True

    def deactivate(self) -> None:
        """Stop monitoring."""
        self._active = False
        self._goal_xy = None
        self._history = []

    def check(self, clip_embedding: np.ndarray, robot_xy: np.ndarray) -> dict:
        """Evaluate whether the robot appears on-track based on top-k retrieval."""
        if not self._active or self._goal_xy is None:
            return {"on_track": True, "abort": False, "reason": "inactive"}

        try:
            results = self._semantic_map.query_by_embedding(
                embedding=clip_embedding, n_results=3,
            )
        except Exception:
            _logger.debug("TransitMonitor query failed", exc_info=True)
            return {"on_track": True, "abort": False, "reason": "query_failed"}

        if len(results) < 2:
            return {"on_track": True, "abort": False, "reason": "sparse_map"}

        near_goal = sum(
            1 for node, _ in results
            if float(np.linalg.norm(
                np.array([node.pose.x, node.pose.y]) - self._goal_xy,
            )) <= self._goal_radius_m
        )
        near_robot = sum(
            1 for node, _ in results
            if float(np.linalg.norm(
                np.array([node.pose.x, node.pose.y]) - robot_xy,
            )) <= 2.0
        )

        snapshot = {
            "robot_xy": robot_xy.tolist(),
            "near_goal": near_goal,
            "near_robot": near_robot,
            "top_regions": [(n.pose.x, n.pose.y) for n, _ in results],
        }
        self._history.append(snapshot)

        if near_goal == 0 and len(self._history) >= 3:
            recent = self._history[-3:]
            all_off_course = all(h["near_goal"] == 0 for h in recent)
            if all_off_course:
                return {
                    "on_track": False,
                    "abort": True,
                    "reason": "transit_divergence",
                    "message": (
                        "Top-3 matches from wrong region for 3 consecutive captures. "
                        f"Latest top matches at: {snapshot['top_regions']}"
                    ),
                }

        return {"on_track": True, "abort": False, "reason": "ok"}
