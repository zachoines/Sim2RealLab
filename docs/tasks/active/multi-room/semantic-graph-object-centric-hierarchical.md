# Object-centric hierarchical semantic graph

**Type:** new feature
**Owner:** DGX agent (lives in
`source/strafer_autonomy/strafer_autonomy/semantic_map/`,
DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P2 (the structural step that bridges the v1 flat
graph to the
[`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md)
map substrate; valuable on its own for fine-grained semantic
queries; the single biggest API change in the v2 series)
**Estimate:** L (~1.5–2 weeks; new node types + hierarchical
graph maintenance + backward-compat shim + tests + eval lift
measurement)
**Branch:** task/semantic-graph-object-centric-hierarchical

**Pickup gate:** Becomes pickable once
[`observation-derived-room-state`](observation-derived-room-state.md)
merges. Recommended sequencing: pick up after the L2 quality
briefs
([`room-state-temporal-smoothing`](room-state-temporal-smoothing.md),
[`room-state-uncertainty-calibration`](room-state-uncertainty-calibration.md),
[`room-label-vlm-refinement`](room-label-vlm-refinement.md))
have matured the v1 API — those briefs preserve the v1
contract; this brief extends it.

## Story

As an **autonomy stack that needs to reason about
VLM-grounded missions over fine-grained semantic landmarks
("go to the couch" not "go to the living room"; "look at the
sink" not "look at the kitchen")**, I want **the semantic
graph to expose explicit object-centric nodes connected to
place-level and room-level nodes hierarchically**, so that
**the LLM planner can ground missions at any granularity, the
v3 VLA architecture has a structured map substrate to
condition on, and v1's flat
observation-pose-only graph stops being the structural
bottleneck**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`observation-derived-room-state`](observation-derived-room-state.md)
  — v1 implementation. This brief extends its graph schema
  while preserving its API surface.
- [`autonomy-stack`](autonomy-stack.md) — primary consumer.
  This brief's hierarchical API enables object-level
  navigation that autonomy-stack currently can't express.
- [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md)
  — downstream consumer. This brief is its map-side
  prerequisite.
- [`semantic-graph-loop-closure`](semantic-graph-loop-closure.md)
  — sibling. Object-node merging across observations reuses
  loop closure's same-place detection.
- [`room-state-eval-harness`](room-state-eval-harness.md) —
  measures the object-localization lift.

## Context

### What v1 has and what's structurally missing

v1's graph is *flat*:

- Nodes are observation poses (`obs_NNNN`).
- Edges are spatial proximity (within 2 m).
- Per-node payloads: CLIP embedding, `detected_objects` list,
  per-node room label.
- Room clusters are derived *at query time* from the
  observation-pose graph.

This works for room-level questions ("which room am I in",
"are the kitchen and bedroom connected") but not for
fine-grained semantic questions ("where was the couch most
recently seen", "what's the shortest route past the kitchen
sink to the bathroom door"). Today's `query_by_label(label)`
returns an observation node that contained the object, not
the object itself — the consumer has to peek into
`detected_objects` to find the position.

State-of-art (FSR-VLN, ConceptGraphs, Hydra, Kimera-Multi)
use hierarchical scene graphs with explicit levels:

```
floor → region → room → place → object
```

We drop the floor level (mecanum, single-story) and probably
the region level (single-home, no city scale). Three
remaining levels:

```
room  ─── place ─── object
 │      ↗       ↘     │
 └── observations ────┘  (flat layer, still maintained)
```

### The three new node types

- **`ObjectNode`**: a single physical object, possibly seen
  across many observations. Wraps the existing Kalman-
  reinforced `DetectedObjectEntry` semantics (3D position
  with covariance, observation count, last-seen). Connected
  upward to its containing `PlaceNode`. Identifier:
  `obj_<label>_<seq>` (e.g. `obj_sink_001`).

  **Identity contract w.r.t. `reinforce_or_add_object`.**
  v1's `SemanticMapManager.reinforce_or_add_object` already
  merges multiple `DetectedObjectEntry`s of the same label
  into one via Mahalanobis-distance matching, but stores the
  merged entry on the **first** observation node only via
  `target_node_id`. The promotion to `ObjectNode` must
  preserve this 1-to-1 identity — one Kalman-cluster
  produces exactly one `ObjectNode`, and every
  `SemanticNode` that contributed an observation gets an
  `obs_node → object_node` edge. Otherwise the maintenance
  pass creates one `ObjectNode` per appearance and the
  hierarchy fragments. The brief acceptance below pins this.
- **`PlaceNode`**: a sub-room landmark — a small spatial
  region that contains a coherent object cluster ("the kitchen
  counter", "the couch + tv area"). Derived from spatial
  clustering of object nodes (DBSCAN over object positions).
  Connected upward to its containing `RoomNode`. Identifier:
  `plc_<seq>`.
- **`RoomNode`**: the cluster-level room, equivalent to v1's
  `RoomEntry` but promoted to a graph node. Carries the
  existing `label` / `centroid_xy` / `confidence` /
  `observed_objects` / `member_node_ids` fields. Identifier:
  `room_<label>_<seq>`.

The existing per-observation `SemanticNode` keeps its proximity
edges; the hierarchical layer is additive.

### Graph topology

```python
# Existing edges (kept, unchanged):
obs_node → obs_node          # proximity, ≤ 2 m
obs_node ─┐
          ├─ same_place      # from semantic-graph-loop-closure
obs_node ─┘

# New edges (this brief adds):
obs_node     → object_node   # "this observation saw this object"
object_node  → place_node    # "this object belongs to this place"
place_node   → room_node     # "this place is in this room"
object_node  ↔ object_node   # spatial-proximity within a place
```

The hierarchical edges are *derivable*: a maintenance pass
runs after `consolidate()` (or on cluster-cache invalidation
from observation-derived-room-state) and re-derives the
object → place → room edges from the flat layer. The brief
ships both the maintenance pass and the explicit graph API.

### API additions

```python
class SemanticMapManager:
    # Existing v1 APIs — preserved for backward compat:
    def known_rooms(self) -> list[RoomEntry]: ...
    def current_room(self, pose, *, max_distance_m=3.0) -> RoomEntry | None: ...
    def connectivity(self) -> list[tuple[str, str]]: ...
    def room_anchor(self, label: str) -> Pose2D | None: ...

    # New v2 APIs — fine-grained:
    def known_objects(self, label: str | None = None) -> list[ObjectNode]: ...
    def known_places(self, room: str | None = None) -> list[PlaceNode]: ...
    def object_anchor(self, label: str) -> Pose2D | None: ...
    def place_anchor(self, place_id: str) -> Pose2D | None: ...
    def hierarchical_graph(self) -> nx.MultiDiGraph: ...
```

`known_rooms()` etc. become thin wrappers over the new
`RoomNode` layer that produce the v1 `RoomEntry` dataclass.
Backward-compat is the explicit acceptance criterion.

### Sub-room hot-spot clustering

The user's mental model — "couches facing opposite a tv;
wall dedicated to shelving; tv surrounded by other
electronics" — is exactly what `PlaceNode` captures.
Implementation:

1. Within each `RoomNode`, gather all `ObjectNode`s whose
   `containing_room == room.label`.
2. DBSCAN over object positions (xy only, robot can't
   distinguish floors anyway) with `eps ≈ 1.5 m`.
3. Each cluster of ≥ 2 objects becomes a `PlaceNode`.
   Singleton objects link directly to the room node.
4. Place label derived heuristically from member objects
   (`["couch", "tv"]` → "media area"; `["sink", "stove"]` →
   "kitchen counter"). Heuristic ships as a configurable map
   matching the prototype map in
   [`room-label-vlm-refinement`](room-label-vlm-refinement.md);
   the LLM-derived label path is filed as a follow-up.

   **Match by Jaccard, not by frozenset equality.** The
   acceptance criterion below specifies
   `place_label_map: dict[frozenset[str], str]`. Naïvely
   looking up `place_label_map[frozenset(member_labels)]`
   misses any subset / superset match: a place with
   `{couch, tv, lamp}` won't hit the `frozenset({couch, tv})`
   key. The lookup must rank dict keys by Jaccard of
   `member_labels` against each key, then take the
   highest-scoring key above a tunable threshold (default
   `≥ 0.5`), falling back to `None` (singleton-link to
   room). This matches the
   [`room-label-vlm-refinement`](room-label-vlm-refinement.md)
   brief's scorer — same primitive, same code path.

### Tie-in to state-of-the-art

- **FSR-VLN** (arXiv:2509.13733) — 4-level floor / room /
  view / object scene graph. Direct architectural precedent.
- **ConceptGraphs** (arXiv:2309.16650) — open-vocab 3D scene
  graph with object-node merging via CLIP similarity +
  spatial proximity. Sibling of
  [`semantic-graph-loop-closure`](semantic-graph-loop-closure.md)'s
  detection pass, generalized to object level.
- **Hydra** (arXiv:2201.13360) — explicit multi-layer scene
  graph with maintenance policies between layers.
- **HOV-SG** (arXiv:2310.08864) — hierarchical open-vocab
  scene graph for navigation; the `known_objects` / `known_places`
  API surface here mirrors theirs.
- **OK-Robot** (arXiv:2401.12202) — object-grounded
  navigation. The `object_anchor` API directly enables
  OK-Robot-style mission grounding.

### Explicit vs. implicit map conditioning — the path to VLA v2

The hierarchical graph this brief builds is the *explicit*
scene-graph path (FSR-VLN / Hydra / HOV-SG / ConceptGraphs
lineage). There is also an *implicit* path: OpenScene
(CVPR 2023), VLMaps (ICRA 2023), CLIP-Fields (ICRA 2023),
LERF (ICCV 2023) and the
[OpenSceneGraph / 3D-VLA (CVPR 2024)](https://arxiv.org/pdf/2403.17846)
line, where the consumer reads a *feature field* (or a
memory bank of past CLIP embeddings via cross-attention)
without materializing explicit `ObjectNode`/`PlaceNode`/
`RoomNode` objects. The
[`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)
brief already proposes the implicit path for the cascade
validator.

These two paths *can* coexist — the hierarchy is the
**training-time annotation** the implicit path's
cross-attention learns to attend over — but the contract
needs writing. The vla-v2 conditioning brief
([`parked/experimental/vla-v2-map-conditioning`](../../parked/experimental/vla-v2-map-conditioning.md))
spells out how this brief's hierarchical layer feeds the
v2 VLA at inference. If that contract lands as
"hierarchy is sim-eval-only; VLA reads a flat memory bank"
the call to ship this brief still stands (interpretability,
planner-side reasoning) but the "biggest single brief"
framing softens — the hierarchy stops being load-bearing
for the v2 stack. Cross-reference it from the v2 VLA brief.

## Acceptance criteria

- [ ] **New node types.** `ObjectNode`, `PlaceNode`,
      `RoomNode` ship as frozen dataclasses in
      [`semantic_map/models.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/models.py)
      alongside the existing `SemanticNode` / `RoomEntry`.
- [ ] **Hierarchical graph maintained.** A maintenance pass
      in `SemanticMapManager` re-derives object → place →
      room edges from the flat layer on cluster-cache
      invalidation. The hierarchical graph is exposed via
      `hierarchical_graph()` returning a `nx.MultiDiGraph`.
- [ ] **Sub-room place clustering.** DBSCAN (or equivalent
      spatial clustering with `min_samples=2`) over object
      positions within each room. Singleton objects link
      directly to the room node, not a place node.
- [ ] **Backward compat.** All v1 APIs (`known_rooms`,
      `current_room`, `connectivity`, `room_anchor`) produce
      the same `RoomEntry` dataclass and the same behavior
      as v1 — the
      [`room-state-eval-harness`](room-state-eval-harness.md)
      v1-compat metrics must hold within the noise floor
      (V-measure delta ≤ 0.02 absolute).
- [ ] **New APIs implemented.** `known_objects(label)`,
      `known_places(room)`, `object_anchor(label)`,
      `place_anchor(place_id)` all live on
      `SemanticMapManager` and are documented in the package
      README.
- [ ] **Place-label heuristic — Jaccard ranking.** A
      configurable `place_label_map: dict[frozenset[str], str]`
      maps object-set signatures to place labels (e.g.
      `frozenset({"couch", "tv"}): "media area"`). Default
      lives in
      [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py).
      Operator can extend per-home. Lookup ranks dict keys
      by Jaccard of `member_labels` vs. each key (not
      frozenset equality); highest-score key wins above a
      tunable threshold (default `≥ 0.5`), else `None`.
      Reuses the scorer from
      [`room-label-vlm-refinement`](room-label-vlm-refinement.md).
- [ ] **`ObjectNode` identity preservation.** The
      maintenance pass produces exactly one `ObjectNode`
      per existing v1 Kalman-cluster (one
      `DetectedObjectEntry` after `reinforce_or_add_object`
      merging). Every `SemanticNode` that contributed any
      observation to the cluster gets an
      `obs_node → object_node` edge. Unit-tested with a
      seeded multi-observation Kalman cluster + asserted
      `ObjectNode` count of 1.
- [ ] **Object localization measured.** PR description
      carries object localization precision/recall on the
      [`room-state-eval-harness`](room-state-eval-harness.md)
      eval scenes — for each Infinigen object, does
      `object_anchor(label)` return a position within 1 m of
      the ground-truth instance? Reports lift over v1's
      `query_by_label`.
- [ ] **Unit tests.** Synthetic hierarchical graphs;
      maintenance-pass idempotency; DBSCAN clustering on
      seeded object distributions; place-label heuristic
      pattern-match correctness; backward-compat shim
      regression.
- [ ] **README documentation.** Expanded "Semantic-map room
      state" subsection of
      [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md)
      covering the hierarchical API, the maintenance pass,
      and the place-label heuristic.
- [ ] **No regression.** Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites pass. Existing autonomy-stack consumers (when
      they ship) must work without changes against this
      brief's backward-compat surface.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit.

## Investigation pointers

- v1 flat graph:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `_add_proximity_edges`, `add_observation`.
- Existing `DetectedObjectEntry`:
  [`semantic_map/models.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/models.py)
  — the basis for `ObjectNode`.
- Existing `RoomEntry`:
  [`semantic_map/models.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/models.py)
  — promoted to a graph node in this brief.
- DBSCAN: `sklearn.cluster.DBSCAN` (already in the dep set).
- Hierarchical scene graph references: FSR-VLN
  (arXiv:2509.13733), ConceptGraphs (arXiv:2309.16650),
  Hydra (arXiv:2201.13360), HOV-SG (arXiv:2310.08864).
- Pose-graph generalization to scene graphs: Kimera-Multi
  (arXiv:2106.14367).

## Out of scope

- **VLA-v2 integration.** This brief produces the map
  substrate; consuming it from a Vision-Language-Action model
  lives in
  [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md).
- **LLM-derived place labels.** v1.5 uses a hand-set
  object-signature → place-label map. Calling the LLM to
  generate place labels on the fly is a v2.5 follow-up if
  the heuristic plateaus.
- **Dynamic object handling.** When furniture is moved
  between sessions, the place / room re-derivation pass
  picks up the change passively (DBSCAN re-clusters). Real-
  time dynamic-scene tracking is a future brief if pursued.
- **Object-level loop closure.** The
  [`semantic-graph-loop-closure`](semantic-graph-loop-closure.md)
  brief detects same-place duplicates at the observation
  layer. Object-level merging (two observations of the same
  physical sink getting one `ObjectNode`) already happens
  via the v1 Kalman-reinforcement path; this brief preserves
  that.
- **Place connectivity** (places-as-graph-nodes for
  intra-room path planning). v1.5 ships place *labels* and
  *anchors*; routing between places within a room remains a
  Nav2 concern. Filed as a v3 follow-up if intra-room route
  planning becomes a bottleneck.
- **Multi-floor support.** Strafer is single-story; the
  brief drops the floor level intentionally.
