"""SemanticMapManager — NetworkX graph + ChromaDB vector store + CLIP encoder."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .clip_encoder import CLIPEncoder
from .models import DetectedObjectEntry, Pose2D, SemanticEdge, SemanticNode

_logger = logging.getLogger(__name__)

MAHALANOBIS_GATE = 3.0

_OBJECT_TTL_SINGLE = 3600.0  # 1 hour
_OBJECT_TTL_FEW = 21600.0  # 6 hours
_NODE_TTL_DEFAULT = 86400.0  # 24 hours

_CHROMA_COLLECTION = "semantic_map_embeddings"


def initial_object_covariance(
    depth_m: float, camera_yaw: float, camera_pitch: float = 0.0,
) -> np.ndarray:
    """Compute initial 3x3 covariance for a depth-projected object."""
    sigma_along = 0.02 * depth_m ** 2
    sigma_lateral = 0.05 + 0.01 * depth_m
    sigma_vertical = 0.03 + 0.01 * depth_m

    D = np.diag([sigma_along ** 2, sigma_lateral ** 2, sigma_vertical ** 2])

    cy, sy = math.cos(camera_yaw), math.sin(camera_yaw)
    cp, sp = math.cos(camera_pitch), math.sin(camera_pitch)

    R = np.array([
        [cy * cp, -sy, cy * sp],
        [sy * cp,  cy, sy * sp],
        [-sp,       0, cp     ],
    ])
    return R @ D @ R.T


class SemanticMapManager:
    """Semantic spatial map combining a NetworkX graph with ChromaDB vector search."""

    def __init__(self, storage_dir: str = "~/.strafer/semantic_map") -> None:
        import chromadb
        import networkx as nx

        self._storage_dir = Path(storage_dir).expanduser()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._graph_path = self._storage_dir / "graph.json"

        self._graph: nx.DiGraph = nx.DiGraph()
        self._nx = nx

        self._chroma_client = chromadb.PersistentClient(
            path=str(self._storage_dir / "chroma"),
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=_CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        self._clip_encoder = CLIPEncoder()
        self._node_counter = 0

    @property
    def clip_encoder(self) -> CLIPEncoder:
        return self._clip_encoder

    @property
    def graph(self) -> Any:
        return self._graph

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def add_observation(
        self,
        *,
        pose: Pose2D,
        timestamp: float,
        clip_embedding: np.ndarray,
        detected_objects: list[DetectedObjectEntry] | None = None,
        text_description: str | None = None,
        source: str = "scan",
        metadata: dict[str, Any] | None = None,
    ) -> SemanticNode:
        """Add an observation node to the graph and its CLIP embedding to ChromaDB."""
        self._node_counter += 1
        node_id = f"obs_{self._node_counter:04d}"
        embedding_id = f"emb_{node_id}"

        for obj in (detected_objects or []):
            self.reinforce_or_add_object(
                label=obj.label,
                observed_xyz=obj.position_mean,
                observation_cov=obj.position_cov,
                bbox_2d=obj.bbox_2d,
                confidence=obj.confidence,
                timestamp=timestamp,
                target_node_id=node_id,
            )

        node = SemanticNode(
            node_id=node_id,
            pose=pose,
            timestamp=timestamp,
            clip_embedding_id=embedding_id,
            text_description=text_description,
            detected_objects=list(detected_objects or []),
            metadata=metadata or {},
            source=source,
        )

        self._graph.add_node(node_id, data=node.to_dict())

        self._collection.add(
            ids=[embedding_id],
            embeddings=[clip_embedding.tolist()],
            metadatas=[{
                "node_id": node_id,
                "pose_x": pose.x,
                "pose_y": pose.y,
                "timestamp": timestamp,
                "source": source,
            }],
        )

        self._add_proximity_edges(node)
        self.save()
        return node

    def _add_proximity_edges(
        self, node: SemanticNode, max_distance_m: float = 2.0,
    ) -> None:
        for other_id in self._graph.nodes:
            if other_id == node.node_id:
                continue
            other_data = self._graph.nodes[other_id].get("data", {})
            ox, oy = other_data.get("pose_x", 0.0), other_data.get("pose_y", 0.0)
            dist = math.sqrt(
                (node.pose.x - ox) ** 2 + (node.pose.y - oy) ** 2,
            )
            if dist <= max_distance_m:
                edge = SemanticEdge(
                    source=node.node_id,
                    target=other_id,
                    distance_m=dist,
                )
                self._graph.add_edge(
                    node.node_id, other_id,
                    distance_m=dist,
                    traversal_verified=False,
                    last_traversed=None,
                )
                if not self._graph.has_edge(other_id, node.node_id):
                    self._graph.add_edge(
                        other_id, node.node_id,
                        distance_m=dist,
                        traversal_verified=False,
                        last_traversed=None,
                    )

    def query_nearest(
        self, x: float, y: float, max_distance_m: float = 3.0,
    ) -> SemanticNode | None:
        """Find the nearest node within max_distance_m of (x, y)."""
        best_node: SemanticNode | None = None
        best_dist = float("inf")
        for nid in self._graph.nodes:
            data = self._graph.nodes[nid].get("data", {})
            dx = data.get("pose_x", 0.0) - x
            dy = data.get("pose_y", 0.0) - y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist and dist <= max_distance_m:
                best_dist = dist
                best_node = SemanticNode.from_dict(data)
        return best_node

    def query_by_label(
        self, label: str, max_age_s: float | None = None,
    ) -> SemanticNode | None:
        """Find the most recent node containing a detected object with the given label."""
        now = time.time()
        best: SemanticNode | None = None
        best_time = 0.0
        for nid in self._graph.nodes:
            data = self._graph.nodes[nid].get("data", {})
            ts = data.get("timestamp", 0.0)
            if max_age_s is not None and (now - ts) > max_age_s:
                continue
            for obj in data.get("detected_objects", []):
                if obj.get("label", "").lower() == label.lower():
                    if ts > best_time:
                        best_time = ts
                        best = SemanticNode.from_dict(data)
                        break
        return best

    def query_by_text(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Encode query text via CLIP text encoder and search ChromaDB."""
        text_embedding = self._clip_encoder.encode_text(query_text)
        if np.allclose(text_embedding, 0):
            return []

        count = self._collection.count()
        if count == 0:
            return []
        n = min(n_results, count)

        results = self._collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=n,
        )

        output = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for i, emb_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            node_id = meta.get("node_id", "")
            node_data = self._graph.nodes.get(node_id, {}).get("data")
            entry: dict[str, Any] = {
                "embedding_id": emb_id,
                "distance": distances[i] if i < len(distances) else None,
                "node_id": node_id,
            }
            if node_data:
                entry["text_description"] = node_data.get("text_description")
                entry["pose_x"] = node_data.get("pose_x")
                entry["pose_y"] = node_data.get("pose_y")
                entry["detected_objects"] = node_data.get("detected_objects", [])
            output.append(entry)
        return output

    def query_by_embedding(
        self, embedding: np.ndarray, n_results: int = 5,
    ) -> list[tuple[SemanticNode, float]]:
        """Raw CLIP vector -> ChromaDB ANN search. Returns (node, distance) pairs."""
        count = self._collection.count()
        if count == 0:
            return []
        n = min(n_results, count)

        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n,
        )

        output = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for i, emb_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            node_id = meta.get("node_id", "")
            node_data = self._graph.nodes.get(node_id, {}).get("data")
            if node_data:
                node = SemanticNode.from_dict(node_data)
                dist = distances[i] if i < len(distances) else float("inf")
                output.append((node, dist))
        return output

    def get_clip_embedding(self, embedding_id: str) -> np.ndarray:
        """Retrieve a stored CLIP embedding by its ID."""
        result = self._collection.get(
            ids=[embedding_id],
            include=["embeddings"],
        )
        embeddings = result.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            return np.array(embeddings[0], dtype=np.float32)
        return np.zeros(512, dtype=np.float32)

    def reinforce_or_add_object(
        self,
        *,
        label: str,
        observed_xyz: np.ndarray,
        observation_cov: np.ndarray,
        bbox_2d: tuple[int, int, int, int],
        confidence: float,
        timestamp: float,
        target_node_id: str | None = None,
    ) -> DetectedObjectEntry:
        """Match via 3D Mahalanobis distance, Kalman update on match, or create new."""
        best_match: DetectedObjectEntry | None = None
        best_mahal: float = float("inf")

        for existing in self._all_detected_objects():
            if existing.label != label:
                continue
            innovation = observed_xyz - existing.position_mean
            S = existing.position_cov + observation_cov
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            mahal = float(np.sqrt(innovation @ S_inv @ innovation))
            if mahal < MAHALANOBIS_GATE and mahal < best_mahal:
                best_match = existing
                best_mahal = mahal

        if best_match is not None:
            P_prior = best_match.position_cov
            R = observation_cov
            S = P_prior + R
            K = P_prior @ np.linalg.inv(S)
            innovation = observed_xyz - best_match.position_mean
            best_match.position_mean = best_match.position_mean + K @ innovation
            best_match.position_cov = (np.eye(3) - K) @ P_prior
            best_match.observation_count += 1
            best_match.last_seen = timestamp
            best_match.confidence = max(best_match.confidence, confidence)
            best_match.bbox_2d = bbox_2d
            return best_match

        return DetectedObjectEntry(
            label=label,
            position_mean=observed_xyz.copy(),
            position_cov=observation_cov.copy(),
            bbox_2d=bbox_2d,
            confidence=confidence,
            first_seen=timestamp,
            last_seen=timestamp,
        )

    def _all_detected_objects(self) -> list[DetectedObjectEntry]:
        """Collect all DetectedObjectEntry instances across all nodes."""
        objects: list[DetectedObjectEntry] = []
        for nid in self._graph.nodes:
            data = self._graph.nodes[nid].get("data", {})
            for obj_dict in data.get("detected_objects", []):
                objects.append(DetectedObjectEntry.from_dict(obj_dict))
        return objects

    def prune(self, max_age_s: float = _NODE_TTL_DEFAULT) -> int:
        """Remove stale nodes and apply tiered decay for detected objects.

        Returns the number of nodes removed.
        """
        now = time.time()
        nodes_to_remove = []

        for nid in list(self._graph.nodes):
            data = self._graph.nodes[nid].get("data", {})
            ts = data.get("timestamp", 0.0)

            # Tiered object decay
            surviving_objects = []
            for obj_dict in data.get("detected_objects", []):
                obs_count = obj_dict.get("observation_count", 1)
                last_seen = obj_dict.get("last_seen", 0.0)
                age = now - last_seen

                if obs_count == 1 and age > _OBJECT_TTL_SINGLE:
                    continue
                if 2 <= obs_count <= 4 and age > _OBJECT_TTL_FEW:
                    continue
                if obs_count >= 5 and age > max_age_s:
                    continue
                surviving_objects.append(obj_dict)
            data["detected_objects"] = surviving_objects

            # Node-level TTL
            if (now - ts) > max_age_s:
                nodes_to_remove.append(nid)

        for nid in nodes_to_remove:
            data = self._graph.nodes[nid].get("data", {})
            emb_id = data.get("clip_embedding_id")
            self._graph.remove_node(nid)
            if emb_id:
                try:
                    self._collection.delete(ids=[emb_id])
                except Exception:
                    pass

        if nodes_to_remove:
            self.save()
        return len(nodes_to_remove)

    def save(self) -> None:
        """Persist the NetworkX graph to disk as JSON."""
        from networkx.readwrite import json_graph
        data = json_graph.node_link_data(self._graph, edges="links")
        with open(self._graph_path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)

    def load(self) -> None:
        """Load the NetworkX graph from disk if it exists."""
        from networkx.readwrite import json_graph
        if self._graph_path.exists():
            with open(self._graph_path, "r") as f:
                data = json.load(f)
            self._graph = json_graph.node_link_graph(data, directed=True, edges="links")
            max_counter = 0
            for nid in self._graph.nodes:
                if nid.startswith("obs_"):
                    try:
                        max_counter = max(max_counter, int(nid.split("_")[1]))
                    except (IndexError, ValueError):
                        pass
            self._node_counter = max_counter
        else:
            self._graph = self._nx.DiGraph()

    def clear(self) -> None:
        """Full reset of graph and vector store."""
        self._graph.clear()
        self._node_counter = 0
        try:
            self._chroma_client.delete_collection(_CHROMA_COLLECTION)
            self._collection = self._chroma_client.get_or_create_collection(
                name=_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            _logger.debug("ChromaDB collection reset failed", exc_info=True)
        if self._graph_path.exists():
            self._graph_path.unlink()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
