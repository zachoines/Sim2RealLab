"""Runtime room-state inference from the semantic map.

Three layered concerns:

1. `RoomClassifier` — CLIP zero-shot room labeling. Precomputes text
   embeddings for a fixed prompt set; classifies an image CLIP embedding
   against them by cosine similarity. No training data required.
2. `cluster_nodes` — graph community detection over the semantic-map
   proximity graph. Defaults to NetworkX greedy modularity for dense
   graphs; falls back to connected-components majority vote on
   very-sparse maps where modularity is unstable.
3. `aggregate_room_entries` / `infer_connectivity` — turn cluster
   assignments into the public `RoomEntry` list and pair-wise edge set.

All functions are pure on a NetworkX `DiGraph` snapshot of the semantic
map; the live `SemanticMapManager` delegates to them. Keeping them
ROS-free lets the connectivity Nav2 hook be injected as a plain
callable rather than dragging a ROS client into this module.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from .models import Pose2D, RoomEntry

# Default zero-shot room prompt set. Tuned to match the residential
# room types Infinigen emits + a couple of common spaces likely to show
# up on the real robot. Override via `RoomClassifier(prompts=...)`.
DEFAULT_ROOM_PROMPTS: tuple[str, ...] = (
    "a kitchen",
    "a living room",
    "a bedroom",
    "a bathroom",
    "a hallway",
    "an office",
    "a garage",
)

# Below this node count the proximity graph is too sparse for modularity
# to recover stable communities; fall back to connected components.
SPARSE_GRAPH_THRESHOLD = 30

Nav2Reachable = Callable[[Pose2D, Pose2D], bool]


class RoomClassifier:
    """CLIP zero-shot room classifier over a fixed prompt set.

    Lazily encodes the prompt set on first use so a disabled CLIP
    encoder (no ONNX models available) degrades to "no room label"
    without raising.
    """

    def __init__(
        self,
        clip_encoder: Any,
        prompts: tuple[str, ...] = DEFAULT_ROOM_PROMPTS,
    ) -> None:
        self._encoder = clip_encoder
        self._prompts = tuple(prompts)
        self._labels = tuple(_prompt_to_label(p) for p in prompts)
        self._text_embeddings: np.ndarray | None = None
        self._enabled: bool | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def prompts(self) -> tuple[str, ...]:
        return self._prompts

    def _ensure_text_embeddings(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        if not getattr(self._encoder, "enabled", False):
            self._enabled = False
            return False
        rows: list[np.ndarray] = []
        for prompt in self._prompts:
            emb = self._encoder.encode_text(prompt)
            if emb is None or np.linalg.norm(emb) == 0:
                self._enabled = False
                return False
            rows.append(emb.astype(np.float32))
        self._text_embeddings = np.stack(rows, axis=0)
        self._enabled = True
        return True

    def classify(
        self, image_embedding: np.ndarray,
    ) -> tuple[str | None, float]:
        """Return (top-1 label, cosine similarity) or (None, 0.0)."""
        if image_embedding is None:
            return (None, 0.0)
        if not self._ensure_text_embeddings():
            return (None, 0.0)
        emb = np.asarray(image_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm == 0:
            return (None, 0.0)
        emb = emb / norm
        # `_text_embeddings` rows are already L2-normalized by the
        # CLIPEncoder, so the matmul yields cosine similarities.
        sims = self._text_embeddings @ emb  # type: ignore[operator]
        top = int(np.argmax(sims))
        return (self._labels[top], float(sims[top]))


def _prompt_to_label(prompt: str) -> str:
    """Strip leading article ("a", "an", "the") to get a clean label."""
    parts = prompt.strip().split(maxsplit=1)
    if len(parts) == 2 and parts[0].lower() in {"a", "an", "the"}:
        return parts[1].strip()
    return prompt.strip()


def _node_payload(graph: Any, node_id: str) -> dict[str, Any]:
    return graph.nodes[node_id].get("data", {})


def _node_xy(graph: Any, node_id: str) -> tuple[float, float]:
    data = _node_payload(graph, node_id)
    return (float(data.get("pose_x", 0.0)), float(data.get("pose_y", 0.0)))


def cluster_nodes(graph: Any) -> list[list[str]]:
    """Partition the graph's nodes into clusters.

    Uses greedy modularity on the undirected projection for graphs of at
    least `SPARSE_GRAPH_THRESHOLD` nodes; otherwise falls back to
    connected components. Always returns a list-of-lists, never empty
    sub-lists. Returns `[]` for an empty graph.
    """
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return []

    undirected = graph.to_undirected()
    n = undirected.number_of_nodes()

    if n >= SPARSE_GRAPH_THRESHOLD:
        try:
            communities = nx.community.greedy_modularity_communities(
                undirected,
            )
            return [sorted(c) for c in communities if c]
        except Exception:
            # Fall through to components — modularity can fail on
            # disconnected graphs with empty communities in some
            # NetworkX versions.
            pass

    return [sorted(c) for c in nx.connected_components(undirected) if c]


def _cluster_label(
    graph: Any, node_ids: Iterable[str],
) -> tuple[str | None, float]:
    """Majority-vote label + mean confidence across cluster members."""
    label_counts: Counter[str] = Counter()
    confs_by_label: dict[str, list[float]] = {}
    for nid in node_ids:
        meta = _node_payload(graph, nid).get("metadata", {}) or {}
        label = meta.get("room_label")
        conf = float(meta.get("room_conf", 0.0))
        if not label:
            continue
        label_counts[label] += 1
        confs_by_label.setdefault(label, []).append(conf)
    if not label_counts:
        return (None, 0.0)
    top_label, _ = label_counts.most_common(1)[0]
    confs = confs_by_label.get(top_label, [])
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return (top_label, mean_conf)


def _cluster_observed_objects(
    graph: Any, node_ids: Iterable[str],
) -> tuple[str, ...]:
    """Deduplicated set of detected-object labels across cluster members."""
    seen: set[str] = set()
    for nid in node_ids:
        data = _node_payload(graph, nid)
        for obj in data.get("detected_objects", []) or []:
            label = obj.get("label") if isinstance(obj, dict) else None
            if label:
                seen.add(str(label))
    return tuple(sorted(seen))


def _cluster_centroid(
    graph: Any, node_ids: Iterable[str],
) -> tuple[float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for nid in node_ids:
        x, y = _node_xy(graph, nid)
        xs.append(x)
        ys.append(y)
    if not xs:
        return (0.0, 0.0)
    return (float(np.mean(xs)), float(np.mean(ys)))


def aggregate_room_entries(
    graph: Any, clusters: list[list[str]],
) -> list[RoomEntry]:
    """Turn clusters into `RoomEntry` values, merging same-label clusters.

    Same-label clusters are merged into a single entry by union: the v1
    API uses room label as the room identifier (callers index by label
    via `room_anchor`), so two clusters both classified `"bedroom"` are
    presented as one merged room. This is a known v1 limitation
    documented on the brief — multi-instance disambiguation is deferred.
    """
    by_label: dict[str, dict[str, Any]] = {}
    for member_ids in clusters:
        if not member_ids:
            continue
        label, conf = _cluster_label(graph, member_ids)
        if label is None:
            continue
        existing = by_label.get(label)
        if existing is None:
            existing = {
                "members": list(member_ids),
                "confs": [conf] if conf > 0 else [],
            }
            by_label[label] = existing
        else:
            existing["members"].extend(member_ids)
            if conf > 0:
                existing["confs"].append(conf)

    entries: list[RoomEntry] = []
    for label, payload in sorted(by_label.items()):
        member_ids = tuple(payload["members"])
        centroid = _cluster_centroid(graph, member_ids)
        objects = _cluster_observed_objects(graph, member_ids)
        confs = payload["confs"]
        confidence = float(np.mean(confs)) if confs else 0.0
        entries.append(RoomEntry(
            label=label,
            member_node_ids=member_ids,
            centroid_xy=centroid,
            confidence=confidence,
            observed_objects=objects,
        ))
    return entries


def _members_by_label(
    entries: list[RoomEntry],
) -> dict[str, set[str]]:
    return {e.label: set(e.member_node_ids) for e in entries}


def _has_inter_cluster_path(
    graph: Any, members_a: set[str], members_b: set[str],
) -> bool:
    """True iff any node in A reaches any node in B via the graph."""
    import networkx as nx

    if not members_a or not members_b:
        return False
    undirected = graph.to_undirected()
    # Walk components once and check whether each contains at least one
    # member from both sides.
    for component in nx.connected_components(undirected):
        if component & members_a and component & members_b:
            return True
    return False


def infer_connectivity(
    graph: Any,
    entries: list[RoomEntry],
    *,
    nav2_reachable: Nav2Reachable | None = None,
) -> list[tuple[str, str]]:
    """Compute the pessimistic connectivity edge set between rooms.

    Edge (A, B) is present iff (1) the graph already connects them via
    traversal OR (2) the optional Nav2 reachability callable returns
    True for their centroids. Absence of an edge means "not yet proven
    reachable" — the planner falls back to exploration rather than
    declaring a target unreachable. Labels are returned sorted within
    each pair for stability across calls.
    """
    if len(entries) < 2:
        return []
    membership = _members_by_label(entries)
    centroids = {
        e.label: Pose2D(x=e.centroid_xy[0], y=e.centroid_xy[1])
        for e in entries
    }
    labels = sorted(membership.keys())
    pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            connected = _has_inter_cluster_path(
                graph, membership[a], membership[b],
            )
            if not connected and nav2_reachable is not None:
                try:
                    connected = bool(
                        nav2_reachable(centroids[a], centroids[b]),
                    )
                except Exception:
                    connected = False
            if connected:
                pairs.add((a, b))
    return sorted(pairs)
