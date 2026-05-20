# Temporal smoothing of per-node room labels

**Type:** new feature
**Owner:** DGX agent (algorithm lives in
`source/strafer_autonomy/strafer_autonomy/semantic_map/`,
DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P2 (quality refinement on top of v1; no API
change to the autonomy-stack consumer)
**Estimate:** S (~1–2 days; belief-propagation pass + cluster
re-aggregation hook + tests)
**Branch:** task/room-state-temporal-smoothing

**Pickup gate:** Becomes pickable once
[`observation-derived-room-state`](observation-derived-room-state.md)
merges. Recommended ordering: file alongside or after
[`room-state-eval-harness`](room-state-eval-harness.md) so the
quality lift can be reported as a delta against the v1
baseline.

## Story

As a **planner consuming `RoomEntry.label`**, I want
**per-node room labels to be smoothed across graph neighbors
before clustering**, so that **transient misclassifications
during doorway transitions (a frame captured mid-step that
CLIP labels as `"hallway"` when it's really a kitchen view)
don't poison the cluster's majority vote and don't fragment
the room cluster into two**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`observation-derived-room-state`](observation-derived-room-state.md)
  — v1 implementation. This brief modifies its
  classifier/clustering path.
- [`room-state-eval-harness`](room-state-eval-harness.md) —
  required to measure the smoothing lift.

## Context

### The brittleness this fixes

v1 stamps `metadata["room_label"]` at `add_observation` time
from an i.i.d. CLIP zero-shot top-1 classification. The
acknowledged failure mode (the "hallway pinch" called out in
v1's clustering discussion) is that a small number of nodes
captured during doorway crossings get a transitional or
ambiguous label baked in, which then:

1. Drags the majority vote when the node lands in a room
   cluster.
2. In sparse-graph fallback mode, can split one room into
   two connected components if the bad labels coincide with
   weak edges.

Both are downstream-visible to the autonomy-stack consumer
via wrong `RoomEntry.label` or fragmented `known_rooms`.

### Why belief propagation, not HMM

Two viable smoothing primitives:

- **Belief propagation on the proximity graph.** Each node's
  effective label is a posterior over its CLIP top-k
  distribution, propagated through neighbors weighted by edge
  proximity. Static; no temporal sequence model needed.
- **HMM / particle filter over a label sequence.** Treats
  the observation chronology as a hidden-state sequence; the
  transition matrix expresses "stay in current room with high
  probability."

Belief propagation is the cheaper and more architecturally
clean choice: the semantic-map graph already encodes spatial
relationships (proximity edges), so smoothing across the
graph is a one-pass operation that doesn't need to track the
robot's path through the nodes. It also composes cleanly with
the existing clustering — the smoothed labels feed directly
into majority vote.

HMM is filed as a v2.5 alternative if belief propagation
proves insufficient (see Investigation pointers).

### Algorithm sketch

```python
def smooth_labels(
    graph: nx.DiGraph,
    *,
    n_passes: int = 2,
    neighbor_weight: float = 0.3,
) -> None:
    """Belief-propagation pass over per-node room_label/room_conf.

    Mutates `metadata["room_label"]` and `metadata["room_conf"]`
    in place. Each pass replaces each node's label distribution
    with a weighted mix of its own and its neighbors'. Two
    passes is usually enough for the hallway-pinch fix.
    """
    # Stage 1: build per-node label distributions from current
    # top-1 stamp (one-hot at `room_label`, weight `room_conf`).
    # Stage 2: for n_passes, replace each node's distribution
    # with (1 - neighbor_weight) * self + neighbor_weight *
    # mean(neighbors). Re-stamp top-1 label + new confidence.
```

The pass runs as a hook in `aggregate_room_entries` (before
the majority vote) rather than at `add_observation` time, so
late-arriving neighbors keep refining earlier labels. Cluster
cache invalidation already triggers re-smoothing via the
existing growth-fraction check.

### When NOT to smooth

- Cold-start (few nodes, no neighbors): no-op.
- Isolated nodes: no-op (no neighbors to propagate from).
- Nodes with `room_conf` already at high confidence and
  agreement with neighbors: idempotent.

### Tie-in to state-of-the-art

Belief propagation on topological maps is standard in
topological-SLAM and semantic-mapping literature; see
Kostavelis & Gasteratos 2015 survey for the foundational
treatment. Recent open-vocabulary work (Hydra, ConceptGraphs)
applies similar smoothing at the place / region level. The
two-pass `(1 - α) * self + α * neighbors` form is the
discrete-label analogue of Laplacian smoothing on a graph.

## Acceptance criteria

- [ ] **Smoothing helper.** A `smooth_labels(graph, ...)`
      function lives in
      [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py),
      pure on the graph snapshot (no manager state).
- [ ] **Wired into clustering.** Called from
      `aggregate_room_entries` (or the manager's
      `_get_known_rooms_cached`) before majority vote. The
      pre-smoothing per-node stamp is preserved on the node
      metadata; the smoothed value lives alongside as
      `metadata["room_label_smoothed"]` for inspectability.
- [ ] **No API change.** `known_rooms` / `current_room` /
      `connectivity` / `room_anchor` signatures unchanged.
      Downstream consumers see better labels, not different
      ones.
- [ ] **Quality lift measured.** v2 PR description includes
      the
      [`room-state-eval-harness`](room-state-eval-harness.md)
      numbers on cluster purity, label precision, and the
      multi-bedroom adversarial scene, vs. the v1 baseline.
      Cluster purity (V-measure) lift should be measurable
      (≥0.05 absolute) on at least one of the four eval
      scenes.
- [ ] **Unit tests.** Synthetic graphs with seeded
      misclassifications at known nodes; verify that smoothing
      flips the misclassified labels when neighbors agree and
      preserves them when neighbors disagree. Edge cases:
      cold-start no-op, isolated node no-op, idempotency on
      already-agreed clusters.
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

- Existing classifier + clustering:
  [`room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
  — `aggregate_room_entries`, `_cluster_label`.
- The hallway-pinch failure mode: documented in v1's
  "Room-membership inference" section.
- HMM alternative (v2.5 if belief propagation falls short):
  Tipaldi & Arras 2010 — Bayesian topological SLAM with
  particle filter over place identity. Heavier; only file as
  a follow-up if measured lift is insufficient.

## Out of scope

- **Cross-session smoothing.** Each session smooths its own
  graph. Multi-session smoothing requires the persistent map
  to survive map-lifecycle decay
  ([`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md))
  before it makes sense.
- **Learned smoothing weights.** v2 uses a hand-set
  `neighbor_weight=0.3`. Learning the smoothing weights
  against eval-harness metrics is a v3 hyperparameter-tuning
  follow-up.
- **HMM-over-label-sequence.** Filed in Investigation pointers
  as an alternative; not pursued in this brief.
- **Smoothing of object labels.** Object reinforcement already
  uses Kalman updates; the per-node room label is the only
  scalar this brief touches.
