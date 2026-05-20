# Map lifecycle: merge-over-prune for long-horizon retention

**Type:** new feature / refactor
**Owner:** DGX agent (lives in
`source/strafer_autonomy/strafer_autonomy/semantic_map/`,
DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P3 (no immediate consumer break; the v1
time-based TTL works fine for the current short-horizon
deployment, but fails the moment any consumer wants
multi-day persistent maps)
**Estimate:** M (~3–5 days; hierarchical layer design + merge
policy + bounded-cardinality eval on simulated long-horizon
trajectories)
**Branch:** task/semantic-map-lifecycle-merge

**Pickup gate:** Blocked-on-deps until
[`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)
ships (no merge candidates to act on without it). Un-park by
`git mv parked/multi-room/<this-brief>.md
active/multi-room/<this-brief>.md` in the PR that picks it
up.

## Story

As a **long-running deployment whose semantic map should
preserve static geometry across multi-day operation**, I want
**the map to age via merging same-place nodes (per
[`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md))
and downsampling old nodes into a long-term layer rather than
deleting them**, so that **static geometry (walls, large
furniture, room structure) is preserved while transient
observations (a cup left on a counter) decay, and the
autonomy stack can lean on a stable long-horizon map**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)
  — hard prerequisite. Supplies the same-place edges this
  brief consolidates.
- [`observation-derived-room-state`](../../active/multi-room/observation-derived-room-state.md)
  — v1 implementation. This brief replaces v1's `prune()`
  with `consolidate()`.
- [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
  — long-horizon evaluation extension lives here.

## Context

### Why pure TTL is the wrong primitive

v1's `prune()` deletes nodes after time-based TTLs:

- Single-sighting objects: 1 hour
- 2–4 sightings: 6 hours
- 5+ sightings or node-level: 24 hours

In a long-running deployment (days or weeks), static geometry
loses its observation count if the robot doesn't revisit each
captured pose every 24 hours. A 3-day-old kitchen capture
gets deleted; the next mission has to re-discover the kitchen
from scratch. This is the pathology the v1 brief's
"Stale-cluster invalidation" Out-of-scope entry points at
(deferred to "map-lifecycle work in
`STRAFER_AUTONOMY_NEXT.md`").

### Hierarchical retention design

Two-layer storage:

- **Recent layer (full resolution).** Nodes captured within
  the last N hours (configurable, default ~6h). These are
  the ones consumers query for `query_nearest` /
  `verify_arrival` / fresh CLIP-similar context.
- **Long-term layer (downsampled).** Older nodes are
  consolidated via same-place merging
  ([`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md))
  AND spatial pooling — within a configurable radius,
  multiple nodes collapse into a single "anchor" node with
  averaged CLIP embedding, union of `detected_objects`,
  preserved `room_label`. Single-sighting transient detections
  still decay; multi-sighting confirmed detections survive.

The graph layer flag lives on `SemanticNode.metadata`:
`metadata["layer"] in {"recent", "long_term"}`.

### Algorithm sketch

```python
def consolidate(
    self,
    *,
    recent_window_s: float = 6 * 3600,
    long_term_pool_radius_m: float = 1.5,
) -> ConsolidateReport:
    """Replace prune() with a lifecycle pass.

    1. Mark nodes within recent_window_s as 'recent'.
    2. For older nodes: run loop-closure detection (delegating
       to detect_loop_closures), merge same-place pairs into
       an anchor node, mark as 'long_term'.
    3. Spatial pooling pass: within long_term_pool_radius_m,
       merge clusters of long-term nodes into a single anchor.
    4. Transient-object decay: single-sighting detections on
       long-term nodes still expire per the v1 TTL.
    5. Persist via save().
    """
```

The report carries counts (recent / long-term / merged /
pruned-transient) for telemetry.

### Capacity bound

Without lifecycle, node count grows unboundedly. With this
brief:

- Recent layer cardinality is bounded by `recent_window_s /
  BackgroundMapper.poll_interval_s` × (movement frequency).
- Long-term layer cardinality is bounded by the home's
  physical area / `long_term_pool_radius_m²`.

A 100 m² home with 1.5 m pooling radius caps long-term at
~50 anchor nodes regardless of how long the deployment runs.

### Why this needs loop closure first

Spatial pooling without same-place evidence merges two
*different* spots that happen to be close (the two ends of a
narrow hallway). The loop-closure brief's CLIP-similarity
gate prevents this — only pool nodes that are *both*
spatially close AND visually similar.

### Tie-in to state-of-the-art

- **Concept-Graphs** (arXiv:2309.16650) — object-node
  consolidation via spatial + appearance similarity.
- **Hydra** (arXiv:2201.13360) — explicit multi-layer scene
  graph with consolidation policies between layers.
- **Kimera-Multi** (arXiv:2106.14367) — pose-graph
  marginalization for long-horizon multi-agent SLAM.
- **Replay buffers / experience-replay decay** in RL — the
  same shape: keep representative samples, decay redundant
  ones.

## Acceptance criteria

- [ ] **`consolidate()` method.** Replaces `prune()` on the
      manager. Configurable thresholds (recent window,
      pooling radius). Returns a `ConsolidateReport` dataclass
      with consolidation counts.
- [ ] **Two-layer metadata.** `SemanticNode.metadata["layer"]`
      stamped on every node. Existing queries
      (`query_nearest`, `query_by_label`, `query_by_text`)
      either filter by layer (recent first, fall through to
      long-term) or expose a `layer=` kwarg.
- [ ] **Same-place consolidation.** Uses the
      [`semantic-graph-loop-closure`](../../active/multi-room/semantic-graph-loop-closure.md)
      candidate pairs as the input set for merge decisions.
      Merge logic: union `detected_objects` (with the brief's
      existing Kalman semantics), mean pose, keep older
      `node_id`.
- [ ] **Spatial pooling within long-term layer.** Bounded by
      `long_term_pool_radius_m`. Pool members are merged
      into one anchor; visualization metadata preserved.
- [ ] **Transient detection decay preserved.** v1's
      single-sighting / few-sighting / many-sighting TTL
      tiers still apply to detected objects on long-term
      nodes — confirmed detections survive; unconfirmed
      decay.
- [ ] **Long-horizon eval.** PR description carries node-
      count over a simulated 7-day operation (replay the
      eval-harness trajectory 7× with `consolidate()` called
      hourly). Node count must stay bounded; cluster purity
      and label precision must hold ≥ 95 % of the single-
      session baseline.
- [ ] **Unit tests.** Synthetic graphs with seeded same-place
      pairs (loop-closure annotated) + nodes far apart with
      similar CLIP. Verify pooling respects spatial
      threshold; transient-decay applied; layer-stamp
      correctness.
- [ ] **No regression.** Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites pass.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit.

## Investigation pointers

- v1 prune logic:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `prune()`, `_OBJECT_TTL_*`, `_NODE_TTL_DEFAULT`.
- Loop-closure detection (this brief's hard dep):
  `semantic-graph-loop-closure`.
- Layered map references: Hydra (arXiv:2201.13360),
  Concept-Graphs (arXiv:2309.16650).

## Out of scope

- **Real-robot multi-day operation.** The eval is sim-only
  via repeated trajectory replay. Real-robot multi-day
  validation belongs in `next-integration-round` once the
  v2 stack matures.
- **Cross-session map persistence.** Each conda invocation
  loads `graph.json`; this brief assumes the persisted graph
  is the truth source. Cross-machine sync / multi-robot
  shared maps are a future brief.
- **Learned compression.** v1.5 uses hand-set pooling radii.
  Learning to compress the map against downstream task
  performance is a v3 follow-up.
- **Dynamic-scene handling.** When furniture is moved between
  sessions, this brief's same-place detection (CLIP +
  proximity) may merge or split incorrectly. Furniture-
  move detection is a separate brief if pursued.
