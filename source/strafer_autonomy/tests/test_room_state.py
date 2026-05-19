"""Tests for the runtime room-state inference (room_state.py).

Covers the three helper layers: CLIP zero-shot classification, graph
clustering, and pessimistic connectivity inference with an optional
Nav2 reachability hook.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from strafer_autonomy.semantic_map.models import Pose2D, RoomEntry
from strafer_autonomy.semantic_map.room_state import (
    DEFAULT_ROOM_PROMPTS,
    RoomClassifier,
    aggregate_room_entries,
    cluster_nodes,
    infer_connectivity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    nodes: list[dict[str, Any]], edges: list[tuple[str, str]],
) -> Any:
    """Construct a DiGraph matching the SemanticMapManager's layout."""
    import networkx as nx

    g: Any = nx.DiGraph()
    for spec in nodes:
        nid = spec["node_id"]
        data = {
            "node_id": nid,
            "pose_x": float(spec.get("pose_x", 0.0)),
            "pose_y": float(spec.get("pose_y", 0.0)),
            "pose_yaw": float(spec.get("pose_yaw", 0.0)),
            "timestamp": float(spec.get("timestamp", 0.0)),
            "metadata": spec.get("metadata", {}),
            "detected_objects": spec.get("detected_objects", []),
        }
        g.add_node(nid, data=data)
    for s, t in edges:
        g.add_edge(s, t)
        g.add_edge(t, s)
    return g


def _onehot(n: int, k: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[k] = 1.0
    return v


def _fake_encoder(
    prompts: tuple[str, ...], enabled: bool = True,
) -> Any:
    """Encoder that returns orthonormal text embeddings keyed to prompts."""
    n = len(prompts)
    text_table = {p: _onehot(n, i) for i, p in enumerate(prompts)}
    encoder = MagicMock()
    encoder.enabled = enabled
    encoder.encode_text.side_effect = (
        lambda text: text_table.get(text, np.zeros(n, dtype=np.float32))
    )
    return encoder


# ---------------------------------------------------------------------------
# RoomClassifier
# ---------------------------------------------------------------------------


class TestRoomClassifier:
    def test_disabled_returns_none(self):
        encoder = MagicMock()
        encoder.enabled = False
        classifier = RoomClassifier(encoder)
        label, conf = classifier.classify(np.array([1.0, 0.0]))
        assert label is None
        assert conf == 0.0

    def test_picks_top_label(self):
        prompts = ("a kitchen", "a bedroom", "a bathroom")
        classifier = RoomClassifier(_fake_encoder(prompts), prompts=prompts)
        # Image embedding aligned with the kitchen prompt's onehot.
        image = np.array([0.9, 0.1, 0.05], dtype=np.float32)
        label, conf = classifier.classify(image)
        assert label == "kitchen"
        assert conf > 0.5

    def test_labels_strip_article(self):
        prompts = ("a kitchen", "an office", "the garage")
        classifier = RoomClassifier(_fake_encoder(prompts), prompts=prompts)
        assert classifier.labels == ("kitchen", "office", "garage")

    def test_zero_embedding_returns_none(self):
        prompts = ("a kitchen", "a bedroom")
        classifier = RoomClassifier(_fake_encoder(prompts), prompts=prompts)
        label, conf = classifier.classify(np.zeros(2, dtype=np.float32))
        assert label is None
        assert conf == 0.0

    def test_none_embedding_returns_none(self):
        classifier = RoomClassifier(_fake_encoder(DEFAULT_ROOM_PROMPTS))
        label, conf = classifier.classify(None)  # type: ignore[arg-type]
        assert label is None
        assert conf == 0.0

    def test_encode_text_failure_degrades(self):
        """If any prompt encodes to zero (e.g. CLIP partially loaded)."""
        encoder = MagicMock()
        encoder.enabled = True
        encoder.encode_text.side_effect = (
            lambda text: np.zeros(4, dtype=np.float32)
        )
        classifier = RoomClassifier(encoder)
        label, conf = classifier.classify(np.array([1.0, 0.0, 0.0, 0.0]))
        assert label is None
        assert conf == 0.0


# ---------------------------------------------------------------------------
# cluster_nodes
# ---------------------------------------------------------------------------


class TestClusterNodes:
    def test_empty_graph(self):
        g = _make_graph([], [])
        assert cluster_nodes(g) == []

    def test_chain_is_single_cluster(self):
        nodes = [
            {"node_id": f"n{i}", "pose_x": float(i), "pose_y": 0.0}
            for i in range(10)
        ]
        edges = [(f"n{i}", f"n{i + 1}") for i in range(9)]
        clusters = cluster_nodes(_make_graph(nodes, edges))
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [f"n{i}" for i in range(10)]

    def test_two_well_separated_rings(self):
        # Ring A: 5 nodes near origin
        ring_a_nodes = [
            {"node_id": f"a{i}", "pose_x": float(i), "pose_y": 0.0}
            for i in range(5)
        ]
        ring_a_edges = [(f"a{i}", f"a{(i + 1) % 5}") for i in range(5)]
        # Ring B: 5 nodes far away, no cross edges
        ring_b_nodes = [
            {"node_id": f"b{i}", "pose_x": 100.0 + i, "pose_y": 100.0}
            for i in range(5)
        ]
        ring_b_edges = [(f"b{i}", f"b{(i + 1) % 5}") for i in range(5)]

        g = _make_graph(ring_a_nodes + ring_b_nodes, ring_a_edges + ring_b_edges)
        clusters = cluster_nodes(g)
        assert len(clusters) == 2
        ids = [set(c) for c in clusters]
        a_ids = {f"a{i}" for i in range(5)}
        b_ids = {f"b{i}" for i in range(5)}
        assert a_ids in ids
        assert b_ids in ids

    def test_singletons(self):
        # Three disconnected lone nodes → three clusters
        nodes = [
            {"node_id": f"n{i}", "pose_x": float(i * 50), "pose_y": 0.0}
            for i in range(3)
        ]
        clusters = cluster_nodes(_make_graph(nodes, []))
        assert len(clusters) == 3


# ---------------------------------------------------------------------------
# aggregate_room_entries
# ---------------------------------------------------------------------------


class TestAggregateRoomEntries:
    def test_unlabeled_clusters_dropped(self):
        nodes = [
            {"node_id": "n0", "pose_x": 0.0, "pose_y": 0.0},
            {"node_id": "n1", "pose_x": 1.0, "pose_y": 0.0},
        ]
        g = _make_graph(nodes, [("n0", "n1")])
        clusters = [["n0", "n1"]]
        assert aggregate_room_entries(g, clusters) == []

    def test_single_room_centroid_and_objects(self):
        nodes = [
            {
                "node_id": "n0",
                "pose_x": 0.0,
                "pose_y": 0.0,
                "metadata": {"room_label": "kitchen", "room_conf": 0.8},
                "detected_objects": [{"label": "sink"}, {"label": "fridge"}],
            },
            {
                "node_id": "n1",
                "pose_x": 2.0,
                "pose_y": 0.0,
                "metadata": {"room_label": "kitchen", "room_conf": 0.6},
                "detected_objects": [{"label": "sink"}],
            },
        ]
        g = _make_graph(nodes, [("n0", "n1")])
        entries = aggregate_room_entries(g, [["n0", "n1"]])
        assert len(entries) == 1
        room = entries[0]
        assert room.label == "kitchen"
        assert room.centroid_xy == (1.0, 0.0)
        assert set(room.observed_objects) == {"sink", "fridge"}
        assert room.confidence == pytest.approx(0.7)
        assert set(room.member_node_ids) == {"n0", "n1"}

    def test_same_label_clusters_merged(self):
        """Two clusters both classified `bedroom` collapse to one entry."""
        nodes = [
            {
                "node_id": "a0",
                "pose_x": 0.0,
                "pose_y": 0.0,
                "metadata": {"room_label": "bedroom", "room_conf": 0.7},
            },
            {
                "node_id": "b0",
                "pose_x": 50.0,
                "pose_y": 50.0,
                "metadata": {"room_label": "bedroom", "room_conf": 0.5},
            },
        ]
        g = _make_graph(nodes, [])
        entries = aggregate_room_entries(g, [["a0"], ["b0"]])
        assert len(entries) == 1
        assert entries[0].label == "bedroom"
        assert set(entries[0].member_node_ids) == {"a0", "b0"}

    def test_majority_vote_label(self):
        """Cluster majority label wins even when minority disagrees."""
        nodes = [
            {
                "node_id": f"n{i}",
                "pose_x": float(i),
                "pose_y": 0.0,
                "metadata": {
                    "room_label": "kitchen" if i < 3 else "hallway",
                    "room_conf": 0.8,
                },
            }
            for i in range(4)
        ]
        g = _make_graph(nodes, [])
        entries = aggregate_room_entries(g, [[f"n{i}" for i in range(4)]])
        assert len(entries) == 1
        assert entries[0].label == "kitchen"


# ---------------------------------------------------------------------------
# infer_connectivity
# ---------------------------------------------------------------------------


def _entry(label: str, member_ids: list[str], cx: float, cy: float) -> RoomEntry:
    return RoomEntry(
        label=label,
        member_node_ids=tuple(member_ids),
        centroid_xy=(cx, cy),
        confidence=1.0,
        observed_objects=(),
    )


class TestInferConnectivity:
    def test_empty_or_single_room(self):
        assert infer_connectivity(_make_graph([], []), []) == []
        entry = _entry("kitchen", ["n0"], 0.0, 0.0)
        assert infer_connectivity(_make_graph([], []), [entry]) == []

    def test_traversal_connects(self):
        nodes = [
            {"node_id": "k0", "pose_x": 0.0, "pose_y": 0.0,
             "metadata": {"room_label": "kitchen", "room_conf": 1.0}},
            {"node_id": "h0", "pose_x": 1.0, "pose_y": 0.0,
             "metadata": {"room_label": "hallway", "room_conf": 1.0}},
        ]
        g = _make_graph(nodes, [("k0", "h0")])
        entries = [
            _entry("kitchen", ["k0"], 0.0, 0.0),
            _entry("hallway", ["h0"], 1.0, 0.0),
        ]
        assert infer_connectivity(g, entries) == [("hallway", "kitchen")]

    def test_disconnected_without_nav2(self):
        nodes = [
            {"node_id": "k0", "pose_x": 0.0, "pose_y": 0.0,
             "metadata": {"room_label": "kitchen", "room_conf": 1.0}},
            {"node_id": "b0", "pose_x": 100.0, "pose_y": 100.0,
             "metadata": {"room_label": "bedroom", "room_conf": 1.0}},
        ]
        g = _make_graph(nodes, [])
        entries = [
            _entry("kitchen", ["k0"], 0.0, 0.0),
            _entry("bedroom", ["b0"], 100.0, 100.0),
        ]
        assert infer_connectivity(g, entries) == []

    def test_nav2_adds_edge(self):
        nodes = [
            {"node_id": "k0", "pose_x": 0.0, "pose_y": 0.0,
             "metadata": {"room_label": "kitchen", "room_conf": 1.0}},
            {"node_id": "b0", "pose_x": 100.0, "pose_y": 100.0,
             "metadata": {"room_label": "bedroom", "room_conf": 1.0}},
        ]
        g = _make_graph(nodes, [])
        entries = [
            _entry("kitchen", ["k0"], 0.0, 0.0),
            _entry("bedroom", ["b0"], 100.0, 100.0),
        ]
        nav2 = MagicMock(return_value=True)
        edges = infer_connectivity(g, entries, nav2_reachable=nav2)
        assert edges == [("bedroom", "kitchen")]
        nav2.assert_called_once()
        a, b = nav2.call_args.args
        assert isinstance(a, Pose2D) and isinstance(b, Pose2D)

    def test_nav2_returns_false_no_edge(self):
        g = _make_graph([], [])
        entries = [
            _entry("kitchen", ["k0"], 0.0, 0.0),
            _entry("bedroom", ["b0"], 100.0, 100.0),
        ]
        nav2 = MagicMock(return_value=False)
        assert infer_connectivity(g, entries, nav2_reachable=nav2) == []

    def test_nav2_exception_swallowed(self):
        g = _make_graph([], [])
        entries = [
            _entry("kitchen", ["k0"], 0.0, 0.0),
            _entry("bedroom", ["b0"], 100.0, 100.0),
        ]
        nav2 = MagicMock(side_effect=RuntimeError("Nav2 server unavailable"))
        # Should not raise.
        assert infer_connectivity(g, entries, nav2_reachable=nav2) == []
