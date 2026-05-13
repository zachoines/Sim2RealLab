# Runtime room labeling and connectivity from observations

**Type:** new feature
**Owner:** DGX agent (`SemanticMapManager` lives in
`strafer_autonomy.semantic_map`; runs on the DGX planner/grounding
host)
**Priority:** P1 (hard prerequisite for `autonomy-stack`'s
planner room-state input; the runtime counterpart of
`scene-connectivity-validation`)
**Estimate:** M (~3–5 days; CLIP zero-shot room classifier +
graph clustering + connectivity inference + manager API +
unit tests)
**Branch:** task/observation-derived-room-state

## Story

As a **planner that must reason about cross-room missions on a
real robot with no Infinigen ground-truth available**, I want
**the `SemanticMapManager` to expose room-membership and
room-to-room connectivity derived purely from the robot's own
observations (CLIP-encoded scans + the semantic-map graph +
RTAB-Map's pose graph + Nav2's global costmap)**, so that
**`autonomy-stack`'s planner has a runtime-legal world-state
input — closing the privileged-information loophole the
multi-room audit on 2026-05-13 surfaced**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`autonomy-stack`](autonomy-stack.md) — primary consumer.
  Reads `current_room`, `known_rooms`, `connectivity` from this
  brief's manager API.
- [`scene-connectivity-validation`](scene-connectivity-validation.md)
  — sim-side ground-truth counterpart. This brief's output is
  scored against that brief's `connectivity[]` block in the
  multi-room grader.
- [`frontier-exploration-primitive`](frontier-exploration-primitive.md)
  — sibling. The frontier-exploration skill populates
  semantic-map nodes; this brief turns those nodes into rooms.

## Context

### Sim-to-real boundary (read this first)

This brief is the runtime counterpart of
[`scene-connectivity-validation`](scene-connectivity-validation.md).
The sim-side connectivity brief uses Infinigen room polygons +
the occupancy grid + ground-truth `room_adjacency`; this brief
uses only signals available on the real D555 + Jetson stack.
The two outputs have the same *shape* — `current_room`,
`known_rooms`, `connectivity` — so the
[`autonomy-stack`'s planner](autonomy-stack.md) consumes one
interface regardless of where it runs. The sim grader compares
the two as a behavioral metric for the runtime room-state agent.

### Where the signals come from

| Signal | Source | Notes |
|---|---|---|
| Visual room evidence | Current observation (RGB) → CLIP image embedding | Existing `CLIPEncoder` at [`semantic_map/clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py). |
| Room label candidates | A fixed prompt set: `"a kitchen"`, `"a living room"`, `"a bedroom"`, `"a bathroom"`, `"a hallway"`, `"an office"`, `"a garage"` | Zero-shot text encoder. No training data required. Set is configurable. |
| Pose history | RTAB-Map `/rtabmap/mapPath`, or the existing semantic-map node graph | Connectivity inference: rooms the robot has traversed between are connected. |
| Reachability of stored poses | Nav2 global planner | If `nav2_msgs/ComputePathToPose` succeeds, the start-room and goal-room are connectable from the current robot state. |

### Room-membership inference

Two layers:

**(a) Per-observation classifier.** At each `add_observation`
call, compute the CLIP image embedding (already done), then
classify against the fixed room-label prompt set's text
embeddings. Store the top-1 label + confidence on the node
metadata: `metadata["room_label"]`, `metadata["room_conf"]`.

**(b) Graph clustering.** Periodically (or on-demand from the
planner), cluster the semantic-map graph's nodes into rooms.
Two viable strategies:

- **Majority vote over connected components.** Run a connected
  components pass on the NetworkX graph filtered to recent
  nodes, then assign each component the majority room_label of
  its members. Cheap, brittle to pass-throughs (a hallway-only
  traversal lumps the kitchen and living room together).
- **Modularity / community detection.** `networkx.community
  .greedy_modularity_communities` over the proximity graph
  recovers room-shaped clusters when the doorway pinch is
  visible in edge density. More robust; assumes the graph has
  >30 nodes for stability.

Pick (b) by default; fall back to (a) for very-sparse maps
(<30 nodes). Cache cluster assignments and refresh when node
count grows by ≥10%.

### Connectivity inference

A pair (room_A, room_B) is in `connectivity` if **any** of:

1. The robot has traversed between them — there exists a
   semantic-map graph path whose endpoints are tagged with the
   two labels.
2. Nav2's global planner can compute a path from a node in A to
   a node in B *right now*. (Captures the case where the rooms
   are connected on the costmap but the robot never traveled
   between them — e.g., it explored both from a shared hallway.)
3. *Optional v2:* a frontier-exploration successor lands the
   robot in a new room AND the costmap connects it to a prior
   room. Defer this to the frontier-exploration brief.

Pessimistic by default — absence of an edge means "not yet
proven reachable," not "unreachable." This is the safe failure
mode: the planner falls back to `explore_until_visible` rather
than declaring a target unreachable.

### Manager API

```python
class SemanticMapManager:
    def current_room(self, pose: Pose2D) -> RoomEntry | None: ...
    def known_rooms(self) -> list[RoomEntry]: ...
    def connectivity(self) -> list[tuple[str, str]]: ...
    def room_anchor(self, room_label: str) -> Pose2D | None:
        """Most recent semantic-map node tagged with `room_label`,
        or None if no such node exists. Used by the
        autonomy-stack compiler as the transit destination."""
```

`RoomEntry` carries `(label: str, member_node_ids: list[str],
centroid_xy: tuple[float, float], confidence: float)`. The
centroid is computed from the cluster's node poses — observation-
derived, not metadata-derived.

### Tie-in to state-of-the-art

The pattern (CLIP zero-shot room labeling + graph clustering +
Nav2 reachability) maps onto current literature:

- **osmAG-LLM** (arXiv:2507.12753) — hierarchical topometric
  map with textual semantic objects + room attributes; LLM
  reasons over the map for object navigation. Same shape as
  this brief's output.
- **FSR-VLN** (arXiv:2509.13733) — hierarchical multi-modal
  scene graph with floor / room / view / object levels. Drop
  the floor level (single-story constraint) and the rest
  matches.
- **Region prediction for RTAB-Map** (arXiv:2303.00295) —
  CNN-based region classifier integrated with RTAB-Map's
  topological SLAM. Heavier than this brief's CLIP zero-shot
  approach; revisit if the zero-shot classifier proves
  unreliable.

## Acceptance criteria

- [ ] **CLIP zero-shot room classifier.** A small helper inside
      `strafer_autonomy.semantic_map` classifies an image
      against a fixed prompt set, returns `(label, confidence)`.
      Prompt set is configurable; default lives in a constants
      module.
- [ ] **Per-node room labels.** `SemanticMapManager
      .add_observation` stamps `metadata["room_label"]` and
      `metadata["room_conf"]` on every new node.
- [ ] **Graph clustering.** A method on
      `SemanticMapManager` clusters nodes into rooms using
      community detection (or majority-vote fallback for sparse
      maps). Result is cached and refreshed on node-count
      growth.
- [ ] **Manager API.** `current_room`, `known_rooms`,
      `connectivity`, and `room_anchor` are implemented and
      documented in the package README.
- [ ] **Connectivity uses Nav2 reachability.** The
      `connectivity` method queries the Nav2 global planner to
      add edges for rooms connectable on the costmap but not
      yet traversed. Behind a feature flag (`use_nav2_reach`)
      so it can be disabled if Nav2 is not running.
- [ ] **No `scene_metadata.json` access.** The new code reads
      only the semantic-map graph, RTAB-Map outputs, and Nav2
      costmaps / planner. A grep of the new code for
      `scene_metadata`, `scene_labels`, `room_adjacency`
      returns zero hits.
- [ ] **Smoke test — sim grader.** In the existing multi-room
      scene `scene_high_quality_dgx_000_seed0`, after a
      teleop traversal of all rooms, the manager's `known_rooms`
      contains a cluster per Infinigen room (allowing 1
      false-merge for short hallways), the connectivity graph
      matches the sim-side connectivity from
      [`scene-connectivity-validation`](scene-connectivity-validation.md)
      on the traversed subset (precision ≥ 0.9 against the
      ground-truth edges), and `current_room(robot_pose)`
      returns the correct label at 5 sample poses (one per
      room).
- [ ] **Cold-start handling.** With an empty semantic map,
      `known_rooms` is empty and `current_room` returns `None`.
      No exceptions.
- [ ] **Unit tests.** Clustering tested against synthetic
      graphs (two well-separated rings, one chain). Classifier
      tested against a small image fixture set with expected
      labels. Connectivity tested against a synthetic graph +
      mock Nav2 planner.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] **No regression** in existing semantic-map workflows.
      `query_by_label`, `query_by_text`, `query_nearest`,
      `verify_arrival` continue to behave identically on
      single-room scenes.

## Investigation pointers

- Existing semantic-map state:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `SemanticMapManager` already has the graph, CLIP encoder,
  add_observation, and proximity-edge insertion. The new APIs
  layer on top.
- CLIP encoder API:
  [`semantic_map/clip_encoder.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py).
- NetworkX community detection:
  `networkx.community.greedy_modularity_communities`. Already
  in the dependency set via `networkx`.
- Nav2 planner client: see how the
  [`nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md)
  staging loop calls Nav2 — same `ComputePathToPose` action.
- Ground-truth source for the smoke test:
  [`scene-connectivity-validation`](scene-connectivity-validation.md)'s
  emitted `connectivity[]` block on the same scene.
- Reference architectures to cross-check against:
  - osmAG-LLM (arXiv:2507.12753)
  - FSR-VLN (arXiv:2509.13733)
  - Region prediction for RTAB-Map (arXiv:2303.00295)

## Out of scope

- **Learned room classifier.** v1 uses CLIP zero-shot. Fine-
  tuning a small classifier on Infinigen renderings (a logical
  extension once the zero-shot baseline is stable) is a P2
  follow-up.
- **Multi-floor handling.** Strafer cannot climb stairs;
  `story` is not in the runtime room model.
- **Stale-cluster invalidation.** Cluster cache is refreshed by
  node-count growth, not by node *content* drift. Re-running
  the same scene twice with rearranged furniture may keep
  stale labels. Filed under map-lifecycle work in
  `STRAFER_AUTONOMY_NEXT.md`.
- **Live `scene_metadata.json` consumption.** Explicitly out
  of scope and explicitly forbidden. The harness / grader is
  the only consumer of the sim-side connectivity brief.
