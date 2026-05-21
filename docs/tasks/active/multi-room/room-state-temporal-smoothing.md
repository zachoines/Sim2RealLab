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
[`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
AND
[`room-state-uncertainty-calibration`](room-state-uncertainty-calibration.md)
merge — the latter ships the per-node
`metadata["room_label_dist"]` field this brief consumes, see
"Per-node distribution: where it comes from" below.
Recommended ordering: file alongside or after
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
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
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
    """Belief-propagation pass over per-node room labels.

    READS `metadata["room_label"]` / `metadata["room_label_dist"]`
    (raw CLIP stamps, set at `add_observation` time).
    WRITES `metadata["room_label_smoothed"]` /
    `metadata["room_conf_smoothed"]` (and the full distribution
    in `metadata["room_label_dist_smoothed"]`). The raw stamps
    are NEVER overwritten. Each pass replaces each node's
    smoothed-label distribution with a weighted mix of its own
    raw distribution and its neighbors' (previous-pass)
    smoothed distributions. Two passes is usually enough for
    the hallway-pinch fix.
    """
    # Stage 1: read per-node raw distributions from
    # `metadata["room_label_dist"]` (one-hot fallback at
    # `metadata["room_label"]` with weight `room_conf` if the
    # distribution isn't stored — see "Per-node distribution"
    # below).
    # Stage 2: for n_passes, set node.smoothed_dist =
    # (1 - neighbor_weight) * node.raw_dist + neighbor_weight *
    # mean(neighbor.smoothed_dist).
    # Stage 3: write top-1 of smoothed_dist to
    # `metadata["room_label_smoothed"]`; stamp confidence +
    # full distribution alongside.
```

`aggregate_room_entries` then reads
`metadata["room_label_smoothed"]` if present, else falls back
to `metadata["room_label"]`. Downstream consumers (the
autonomy-stack compiler, `current_room`, `room_anchor`) see
the smoothed label transparently — no API change.

**Why read-raw / write-smoothed, not in-place.** Mutating
`metadata["room_label"]` in place was the v0 sketch and is
*wrong*. Two reasons:

1. **The smoother becomes non-idempotent.** Each cache rebuild
   re-runs `smooth_labels`. If pass N reads from already-
   smoothed labels (because they overwrote the raw stamp on
   the previous rebuild), the BP recurrence stops converging
   to the local-neighborhood posterior and slides toward the
   connected-component majority. After 5–10 cache rebuilds
   (~50 new nodes on a typical home) all nodes inside the
   same connected component carry the same label — exactly
   the failure mode the brief is trying to avoid.
2. **The growth-fraction cache invalidation only fires on
   node-count drift.** v1's `_cluster_cache_stale()`
   (manager.py) triggers on growth ≥ 10%, never on label-
   content drift. An in-place mutator that runs at cache
   rebuild but *also* races against other writers (the
   [`room-label-vlm-refinement`](room-label-vlm-refinement.md)
   path mutates labels too) ends up with a cache built from
   a mix of "smoothed once" and "raw" labels. Keeping the
   raw stamp authoritative means every cache rebuild starts
   from the same fixed point.

The raw stamp also doubles as the inspectability lane the
eval harness needs: per-node "did smoothing flip this label?"
is a one-liner over `(room_label, room_label_smoothed)`
pairs.

The pass runs as a hook in `aggregate_room_entries` (before
the majority vote) rather than at `add_observation` time, so
late-arriving neighbors keep refining earlier nodes' smoothed
labels. Cluster cache invalidation via the existing growth-
fraction check is now sufficient — re-smoothing on rebuild is
idempotent because each pass starts from the raw stamps, not
from the previous rebuild's output.

### Per-node distribution: where it comes from

The "(1 - α) * self + α * neighbors" formulation is a mix
over **label distributions**, not over scalar top-1 labels.
v1 stamps `metadata["room_label"]` (a string) and
`metadata["room_conf"]` (a scalar) — that's a one-hot at
`room_label` with weight `room_conf`, the remaining mass
spread uniformly over the other prompts. Smoothing over
this degenerate distribution is **much weaker** than
smoothing over the true softmax — a single misclassified
neighbor on the hallway pinch can still flip the top-1.

Two viable shapes for the brief:

- **(Recommended) Pickup-gate on
  [`room-state-uncertainty-calibration`](room-state-uncertainty-calibration.md).**
  That brief stamps `metadata["room_label_dist"]` (the full
  softmax vector) on every node. Smoothing reads it directly
  and the BP formulation works as written. Net cost:
  smoothing ships after calibration ships. Recommended
  ordering already implies this; making it a hard pickup-gate
  removes the foot-gun.
- **(Alternative) One-hot fallback when no distribution is
  stored.** Smoothing still runs; the brief explicitly
  documents that the lift over baseline is bounded by the
  one-hot information loss. Quality lift target drops from
  "≥0.05 V-measure" to "≥0.02 V-measure" until calibration
  ships. Useful if calibration slips; not the default.

The brief picks (a) — the eval-harness measurement bar in
the acceptance criteria assumes the per-node distribution
is present.

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
      `_get_known_rooms_cached`) before majority vote.
      Aggregation reads `metadata["room_label_smoothed"]`
      when present, falling back to `metadata["room_label"]`.
- [ ] **Read-raw / write-smoothed (idempotence).** The
      smoother NEVER overwrites `metadata["room_label"]` /
      `metadata["room_conf"]` / `metadata["room_label_dist"]`
      (the raw stamps). It writes to
      `metadata["room_label_smoothed"]` /
      `metadata["room_conf_smoothed"]` /
      `metadata["room_label_dist_smoothed"]`. Each invocation
      reads only the raw stamps, so re-smoothing on cache
      rebuild is idempotent and the existing growth-fraction
      cache invalidation check is sufficient — no
      label-content-drift signal needs to be added to v1's
      `_cluster_cache_stale()`. Unit-tested: invoke
      `smooth_labels` twice on the same graph snapshot; assert
      the second invocation's outputs match the first
      byte-for-byte.
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
