"""Data models for the semantic spatial map."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    """2D pose in the map frame, used for semantic graph nodes."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @staticmethod
    def from_pose_map_dict(d: dict) -> Pose2D:
        qz = d.get("qz", 0.0)
        qw = d.get("qw", 1.0)
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        return Pose2D(x=d.get("x", 0.0), y=d.get("y", 0.0), yaw=yaw)


@dataclass
class DetectedObjectEntry:
    """A detected object with 3D Bayesian position tracking."""

    label: str
    position_mean: np.ndarray  # [x, y, z] in map frame
    position_cov: np.ndarray  # 3x3 covariance matrix
    bbox_2d: tuple[int, int, int, int]
    confidence: float
    observation_count: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "position_mean": self.position_mean.tolist(),
            "position_cov": self.position_cov.tolist(),
            "bbox_2d": list(self.bbox_2d),
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DetectedObjectEntry:
        return DetectedObjectEntry(
            label=d["label"],
            position_mean=np.array(d["position_mean"], dtype=np.float64),
            position_cov=np.array(d["position_cov"], dtype=np.float64),
            bbox_2d=tuple(d["bbox_2d"]),  # type: ignore[arg-type]
            confidence=d["confidence"],
            observation_count=d.get("observation_count", 1),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
        )


@dataclass
class SemanticNode:
    """A node in the semantic graph representing one observation point."""

    node_id: str
    pose: Pose2D
    timestamp: float
    clip_embedding_id: str
    text_description: str | None = None
    detected_objects: list[DetectedObjectEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "scan"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "pose_x": self.pose.x,
            "pose_y": self.pose.y,
            "pose_yaw": self.pose.yaw,
            "timestamp": self.timestamp,
            "clip_embedding_id": self.clip_embedding_id,
            "text_description": self.text_description,
            "detected_objects": [o.to_dict() for o in self.detected_objects],
            "metadata": self.metadata,
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SemanticNode:
        return SemanticNode(
            node_id=d["node_id"],
            pose=Pose2D(x=d["pose_x"], y=d["pose_y"], yaw=d["pose_yaw"]),
            timestamp=d["timestamp"],
            clip_embedding_id=d["clip_embedding_id"],
            text_description=d.get("text_description"),
            detected_objects=[
                DetectedObjectEntry.from_dict(o)
                for o in d.get("detected_objects", [])
            ],
            metadata=d.get("metadata", {}),
            source=d.get("source", "scan"),
        )


@dataclass
class SemanticEdge:
    """Edge representing traversability between two semantic nodes."""

    source: str
    target: str
    distance_m: float
    traversal_verified: bool = False
    last_traversed: float | None = None
