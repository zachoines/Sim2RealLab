"""Tests for the semantic map package (models, manager, CLIP encoder)."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from strafer_autonomy.semantic_map.models import (
    DetectedObjectEntry,
    Pose2D,
    SemanticEdge,
    SemanticNode,
)
from strafer_autonomy.semantic_map.manager import (
    MAHALANOBIS_GATE,
    SemanticMapManager,
    initial_object_covariance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_storage(tmp_path):
    """Provide a temporary storage directory for the semantic map."""
    return str(tmp_path / "semantic_map")


@pytest.fixture()
def manager(tmp_storage):
    """Create a SemanticMapManager with a mocked CLIP encoder."""
    mgr = SemanticMapManager(storage_dir=tmp_storage)
    # Mock the CLIP encoder to return random 512-dim vectors
    mock_encoder = MagicMock()
    mock_encoder.enabled = True
    mock_encoder.encode_image.side_effect = lambda img: _random_embedding()
    mock_encoder.encode_text.side_effect = lambda text: _random_embedding()
    mgr._clip_encoder = mock_encoder
    return mgr


def _random_embedding(dim: int = 512) -> np.ndarray:
    vec = np.random.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> Pose2D:
    return Pose2D(x=x, y=y, yaw=yaw)


def _make_detected_object(
    label: str = "table",
    xyz: tuple[float, float, float] = (1.0, 2.0, 0.5),
    observation_count: int = 1,
    last_seen: float = 0.0,
) -> DetectedObjectEntry:
    return DetectedObjectEntry(
        label=label,
        position_mean=np.array(xyz),
        position_cov=np.eye(3) * 0.01,
        bbox_2d=(100, 200, 300, 400),
        confidence=0.9,
        observation_count=observation_count,
        first_seen=last_seen,
        last_seen=last_seen,
    )


# ---------------------------------------------------------------------------
# Pose2D tests
# ---------------------------------------------------------------------------


class TestPose2D:
    def test_from_pose_map_dict_identity(self):
        d = {"x": 1.0, "y": 2.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
        pose = Pose2D.from_pose_map_dict(d)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert abs(pose.yaw) < 1e-6

    def test_from_pose_map_dict_90_degrees(self):
        yaw = math.pi / 2
        qz = math.sin(yaw / 2)
        qw = math.cos(yaw / 2)
        d = {"x": 0.0, "y": 0.0, "qz": qz, "qw": qw}
        pose = Pose2D.from_pose_map_dict(d)
        assert abs(pose.yaw - yaw) < 1e-6

    def test_from_empty_dict(self):
        pose = Pose2D.from_pose_map_dict({})
        assert pose.x == 0.0
        assert pose.y == 0.0


# ---------------------------------------------------------------------------
# DetectedObjectEntry tests
# ---------------------------------------------------------------------------


class TestDetectedObjectEntry:
    def test_round_trip_dict(self):
        obj = _make_detected_object()
        d = obj.to_dict()
        restored = DetectedObjectEntry.from_dict(d)
        assert restored.label == obj.label
        np.testing.assert_allclose(restored.position_mean, obj.position_mean)
        np.testing.assert_allclose(restored.position_cov, obj.position_cov)
        assert restored.bbox_2d == obj.bbox_2d


# ---------------------------------------------------------------------------
# SemanticNode tests
# ---------------------------------------------------------------------------


class TestSemanticNode:
    def test_round_trip_dict(self):
        node = SemanticNode(
            node_id="obs_0001",
            pose=_make_pose(1.0, 2.0, 0.5),
            timestamp=1000.0,
            clip_embedding_id="emb_obs_0001",
            text_description="a room with a table",
            detected_objects=[_make_detected_object()],
            source="scan",
        )
        d = node.to_dict()
        restored = SemanticNode.from_dict(d)
        assert restored.node_id == node.node_id
        assert restored.pose.x == node.pose.x
        assert len(restored.detected_objects) == 1


# ---------------------------------------------------------------------------
# initial_object_covariance tests
# ---------------------------------------------------------------------------


class TestInitialObjectCovariance:
    def test_shape(self):
        cov = initial_object_covariance(2.0, 0.0)
        assert cov.shape == (3, 3)

    def test_symmetric(self):
        cov = initial_object_covariance(3.0, math.pi / 4)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_positive_definite(self):
        cov = initial_object_covariance(2.0, 1.0, 0.3)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_depth_scaling(self):
        cov_near = initial_object_covariance(1.0, 0.0)
        cov_far = initial_object_covariance(4.0, 0.0)
        assert np.linalg.det(cov_far) > np.linalg.det(cov_near)


# ---------------------------------------------------------------------------
# SemanticMapManager tests
# ---------------------------------------------------------------------------


class TestSemanticMapManager:
    def test_add_observation(self, manager):
        node = manager.add_observation(
            pose=_make_pose(1.0, 2.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
            source="scan",
        )
        assert node.node_id == "obs_0001"
        assert manager.node_count() == 1

    def test_add_multiple_observations(self, manager):
        for i in range(5):
            manager.add_observation(
                pose=_make_pose(float(i), 0.0),
                timestamp=time.time(),
                clip_embedding=_random_embedding(),
            )
        assert manager.node_count() == 5

    def test_query_nearest(self, manager):
        manager.add_observation(
            pose=_make_pose(1.0, 1.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
        )
        manager.add_observation(
            pose=_make_pose(5.0, 5.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
        )
        result = manager.query_nearest(1.1, 1.1, max_distance_m=1.0)
        assert result is not None
        assert abs(result.pose.x - 1.0) < 0.01

    def test_query_nearest_none(self, manager):
        manager.add_observation(
            pose=_make_pose(10.0, 10.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
        )
        assert manager.query_nearest(0.0, 0.0, max_distance_m=1.0) is None

    def test_query_by_label(self, manager):
        obj = _make_detected_object(label="door", last_seen=time.time())
        manager.add_observation(
            pose=_make_pose(1.0, 1.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
            detected_objects=[obj],
        )
        result = manager.query_by_label("door")
        assert result is not None

    def test_query_by_label_case_insensitive(self, manager):
        obj = _make_detected_object(label="Door", last_seen=time.time())
        manager.add_observation(
            pose=_make_pose(1.0, 1.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
            detected_objects=[obj],
        )
        assert manager.query_by_label("door") is not None
        assert manager.query_by_label("DOOR") is not None

    def test_query_by_label_with_max_age(self, manager):
        old_time = time.time() - 7200  # 2 hours ago
        obj = _make_detected_object(label="chair", last_seen=old_time)
        manager.add_observation(
            pose=_make_pose(1.0, 1.0),
            timestamp=old_time,
            clip_embedding=_random_embedding(),
            detected_objects=[obj],
        )
        assert manager.query_by_label("chair", max_age_s=3600) is None
        assert manager.query_by_label("chair", max_age_s=10800) is not None

    def test_query_by_embedding(self, manager):
        emb = _random_embedding()
        manager.add_observation(
            pose=_make_pose(1.0, 1.0),
            timestamp=time.time(),
            clip_embedding=emb,
        )
        results = manager.query_by_embedding(emb, n_results=3)
        assert len(results) >= 1
        node, dist = results[0]
        assert node.node_id == "obs_0001"

    def test_query_by_text(self, manager):
        manager.add_observation(
            pose=_make_pose(1.0, 1.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
            text_description="a hallway with a door",
        )
        results = manager.query_by_text("door")
        assert len(results) >= 1

    def test_get_clip_embedding(self, manager):
        emb = _random_embedding()
        node = manager.add_observation(
            pose=_make_pose(),
            timestamp=time.time(),
            clip_embedding=emb,
        )
        retrieved = manager.get_clip_embedding(node.clip_embedding_id)
        np.testing.assert_allclose(retrieved, emb, atol=1e-5)

    def test_save_and_load(self, manager):
        manager.add_observation(
            pose=_make_pose(1.0, 2.0),
            timestamp=1000.0,
            clip_embedding=_random_embedding(),
            text_description="test scene",
        )
        manager.save()

        manager2 = SemanticMapManager(storage_dir=manager._storage_dir)
        manager2.load()
        assert manager2.node_count() == 1

    def test_clear(self, manager):
        manager.add_observation(
            pose=_make_pose(),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
        )
        assert manager.node_count() == 1
        manager.clear()
        assert manager.node_count() == 0

    def test_proximity_edges(self, manager):
        manager.add_observation(
            pose=_make_pose(0.0, 0.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
        )
        manager.add_observation(
            pose=_make_pose(1.0, 0.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
        )
        assert manager.graph.has_edge("obs_0001", "obs_0002")
        assert manager.graph.has_edge("obs_0002", "obs_0001")

    def test_no_edge_for_distant_nodes(self, manager):
        manager.add_observation(
            pose=_make_pose(0.0, 0.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
        )
        manager.add_observation(
            pose=_make_pose(10.0, 0.0),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
        )
        assert not manager.graph.has_edge("obs_0001", "obs_0002")


# ---------------------------------------------------------------------------
# Pruning and tiered object decay tests
# ---------------------------------------------------------------------------


class TestPruning:
    def test_prune_old_nodes(self, manager):
        old_time = time.time() - 100000  # well past 24h TTL
        manager.add_observation(
            pose=_make_pose(),
            timestamp=old_time,
            clip_embedding=_random_embedding(),
        )
        assert manager.node_count() == 1
        removed = manager.prune(max_age_s=86400)
        assert removed == 1
        assert manager.node_count() == 0

    def test_prune_keeps_recent_nodes(self, manager):
        manager.add_observation(
            pose=_make_pose(),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
        )
        removed = manager.prune(max_age_s=86400)
        assert removed == 0
        assert manager.node_count() == 1

    def test_tiered_decay_single_sighting(self, manager):
        """Single-sighting objects expire after 1 hour."""
        old_time = time.time() - 7200  # 2 hours ago
        obj = _make_detected_object(
            label="maybe_hallucinated",
            observation_count=1,
            last_seen=old_time,
        )
        manager.add_observation(
            pose=_make_pose(),
            timestamp=time.time(),  # node itself is recent
            clip_embedding=_random_embedding(),
            detected_objects=[obj],
        )
        manager.prune()
        node_data = manager.graph.nodes["obs_0001"]["data"]
        assert len(node_data["detected_objects"]) == 0

    def test_tiered_decay_few_sightings(self, manager):
        """2-4 sighting objects expire after 6 hours."""
        recent = time.time() - 3600  # 1 hour ago — within 6h window
        obj = _make_detected_object(
            label="seen_thrice",
            observation_count=3,
            last_seen=recent,
        )
        manager.add_observation(
            pose=_make_pose(),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
            detected_objects=[obj],
        )
        manager.prune()
        node_data = manager.graph.nodes["obs_0001"]["data"]
        assert len(node_data["detected_objects"]) == 1

    def test_tiered_decay_few_sightings_expired(self, manager):
        """2-4 sighting objects expire after 6 hours."""
        old = time.time() - 25000  # ~7 hours ago — past 6h window
        obj = _make_detected_object(
            label="seen_twice",
            observation_count=2,
            last_seen=old,
        )
        manager.add_observation(
            pose=_make_pose(),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
            detected_objects=[obj],
        )
        manager.prune()
        node_data = manager.graph.nodes["obs_0001"]["data"]
        assert len(node_data["detected_objects"]) == 0

    def test_tiered_decay_many_sightings(self, manager):
        """5+ sighting objects use node TTL (24h)."""
        recent = time.time() - 3600
        obj = _make_detected_object(
            label="landmark",
            observation_count=10,
            last_seen=recent,
        )
        manager.add_observation(
            pose=_make_pose(),
            timestamp=time.time(),
            clip_embedding=_random_embedding(),
            detected_objects=[obj],
        )
        manager.prune()
        node_data = manager.graph.nodes["obs_0001"]["data"]
        assert len(node_data["detected_objects"]) == 1


# ---------------------------------------------------------------------------
# Reinforcement (Bayesian update) tests
# ---------------------------------------------------------------------------


class TestReinforcement:
    def test_reinforce_updates_existing(self, manager):
        cov = np.eye(3) * 0.01
        obj1 = DetectedObjectEntry(
            label="table",
            position_mean=np.array([1.0, 2.0, 0.5]),
            position_cov=cov.copy(),
            bbox_2d=(100, 200, 300, 400),
            confidence=0.8,
            first_seen=1000.0,
            last_seen=1000.0,
        )
        manager.add_observation(
            pose=_make_pose(0.0, 0.0),
            timestamp=1000.0,
            clip_embedding=_random_embedding(),
            detected_objects=[obj1],
        )

        # Second observation of the same table nearby
        obj2 = DetectedObjectEntry(
            label="table",
            position_mean=np.array([1.05, 2.05, 0.5]),
            position_cov=cov.copy(),
            bbox_2d=(110, 210, 310, 410),
            confidence=0.9,
            first_seen=1001.0,
            last_seen=1001.0,
        )
        result = manager.reinforce_or_add_object(
            label=obj2.label,
            observed_xyz=obj2.position_mean,
            observation_cov=obj2.position_cov,
            bbox_2d=obj2.bbox_2d,
            confidence=obj2.confidence,
            timestamp=1001.0,
        )
        assert result.observation_count == 2
        assert result.confidence == 0.9

    def test_new_object_when_far(self, manager):
        cov = np.eye(3) * 0.01
        obj1 = DetectedObjectEntry(
            label="chair",
            position_mean=np.array([1.0, 2.0, 0.5]),
            position_cov=cov.copy(),
            bbox_2d=(100, 200, 300, 400),
            confidence=0.8,
            first_seen=1000.0,
            last_seen=1000.0,
        )
        manager.add_observation(
            pose=_make_pose(),
            timestamp=1000.0,
            clip_embedding=_random_embedding(),
            detected_objects=[obj1],
        )

        # Far away observation — should create new
        result = manager.reinforce_or_add_object(
            label="chair",
            observed_xyz=np.array([10.0, 20.0, 0.5]),
            observation_cov=cov.copy(),
            bbox_2d=(100, 200, 300, 400),
            confidence=0.7,
            timestamp=1001.0,
        )
        assert result.observation_count == 1

    def test_covariance_shrinks_after_reinforcement(self, manager):
        cov = np.eye(3) * 0.1
        obj = DetectedObjectEntry(
            label="cup",
            position_mean=np.array([1.0, 1.0, 0.3]),
            position_cov=cov.copy(),
            bbox_2d=(50, 50, 100, 100),
            confidence=0.7,
            first_seen=1000.0,
            last_seen=1000.0,
        )
        manager.add_observation(
            pose=_make_pose(),
            timestamp=1000.0,
            clip_embedding=_random_embedding(),
            detected_objects=[obj],
        )

        det_before = np.linalg.det(cov)
        result = manager.reinforce_or_add_object(
            label="cup",
            observed_xyz=np.array([1.02, 1.02, 0.31]),
            observation_cov=cov.copy(),
            bbox_2d=(55, 55, 105, 105),
            confidence=0.8,
            timestamp=1001.0,
        )
        det_after = np.linalg.det(result.position_cov)
        assert det_after < det_before


# ---------------------------------------------------------------------------
# CLIP encoder (mocked ONNX) tests
# ---------------------------------------------------------------------------


class TestCLIPEncoder:
    def test_disabled_when_no_models(self, tmp_path):
        from strafer_autonomy.semantic_map.clip_encoder import CLIPEncoder
        encoder = CLIPEncoder(model_dir=str(tmp_path / "nonexistent"))
        assert not encoder.enabled

    def test_disabled_returns_zeros(self, tmp_path):
        from strafer_autonomy.semantic_map.clip_encoder import CLIPEncoder
        encoder = CLIPEncoder(model_dir=str(tmp_path / "nonexistent"))
        emb = encoder.encode_image(np.zeros((224, 224, 3), dtype=np.uint8))
        assert emb.shape == (512,)
        assert np.allclose(emb, 0)
