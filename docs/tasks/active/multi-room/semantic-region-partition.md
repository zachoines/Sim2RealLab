# v2 region partition — feature+space clustering + open-vocab labels

**Type:** new feature / refactor
**Owner:** DGX agent (lives in
`source/strafer_autonomy/strafer_autonomy/semantic_map/`,
DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P2 (the v2 room-state quality work; replaces v1's
greedy-modularity + 7-class-argmax pipeline with a single
feature+space clustering + open-vocab labeling pass — fewer
knobs, SOTA-aligned, no training)
**Estimate:** M (~3–5 days; clustering + open-vocab labeling +
RoomEntry derivation + eval against the v1 baseline + tests)
**Branch:** task/semantic-region-partition

**Pickup gate:** Becomes pickable once
[`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
(v1) merged — done — and
[`room-state-eval-harness`](room-state-eval-harness.md) ships
(this brief reports its lift against the v1 baseline the
harness measures).

## Story

As a **planner-LLM that needs a clean room-layout graph to
reason about cross-room missions, on real homes where the
kitchen / dining / living continuum has no demarcating walls**,
I want **the `SemanticMapManager` to partition observation
nodes into regions by joint CLIP-feature + spatial clustering
and label each region open-vocab**, so that **regions emerge
from where the visual content actually changes (not from a
fixed 7-class prompt set or wall geometry that open-plan homes
don't have), the v1 brittleness at the prompt-set / threshold
boundary collapses to a single auto-selected knob, and the
planner consumes the same `RoomEntry` API it already reads —
just with regions that match how the home is actually laid
out**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md) — this brief IS v2. The version stack and the symbolic-vs-sub-symbolic split live here.
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
  — v1 substrate. This brief replaces v1's `cluster_nodes`
  (greedy modularity) + `RoomClassifier` (7-class argmax) +
  `aggregate_room_entries` (majority vote) with one
  clustering + labeling pass. The `RoomEntry` /
  `known_rooms` / `current_room` / `connectivity` /
  `room_anchor` API surface is preserved.
- [`query-room-by-text-v1`](query-room-by-text-v1.md) — the
  open-vocab text query. v1.5 aimed it at v1's clusters; this
  brief produces the region centroids it aims at. The two
  compose: v1.5 ships the query API, v2 ships better regions
  to query.
- [`room-state-eval-harness`](room-state-eval-harness.md) —
  measures the lift over the v1 baseline.

## Context

### What v1 ships and why it's brittle

v1's room-state pipeline (now in `completed/`):

```
add_observation(rgb, detected_objects):
    clip_emb = clip_encoder.encode_image(rgb)
    label, conf = RoomClassifier.classify(clip_emb)   # 7-class argmax
    metadata["room_label"], metadata["room_conf"] = label, conf

known_rooms():
    clusters = cluster_nodes(graph)                   # greedy modularity
    return aggregate_room_entries(graph, clusters)    # majority vote, same-label merge
```

Two structural failures:

1. **Greedy modularity over proximity edges only sees spatial
   adjacency.** In an open-plan home (kitchen + dining +
   living, no walls), every node is spatially adjacent to its
   neighbors, so modularity lumps the whole open space into
   one cluster. There's no wall-density pinch for modularity
   to cut on.
2. **The 7-class argmax forces every region into a fixed
   vocabulary.** A home gym, sunroom, or workshop gets
   mislabeled as the nearest of seven classes.

### What was considered and rejected

A v2 series of threshold-tuned refinements was filed and then
rejected during the PR #43 architecture review:

- Per-node belief-propagation smoothing (`neighbor_weight`,
  `n_passes` knobs).
- Temperature-scaled softmax calibration (`T` knob).
- VLM object-prototype refinement (Jaccard-threshold knob,
  hand-set `DEFAULT_ROOM_PROTOTYPES`).

Together with v1's prompt-set and cluster-cache knobs, that's
**six hand-tuned tuning points**, each brittle in some
adversarial regime. The decision: don't ship six tuning
points stacked. Feature+space clustering + open-vocab
labeling subsumes all of them with **one** auto-selected
knob, and it's what the SOTA pipelines (ConceptGraphs,
HOV-SG) actually do.

### The v2 approach — feature+space clustering

Replaces `cluster_nodes` + `RoomClassifier` +
`aggregate_room_entries`:

```python
def known_rooms():
    # Inputs per observation node (all already exist in v1):
    #   - CLIP embedding (512-d, in ChromaDB)
    #   - pose (x, y)
    #   - detected_objects (VLM, mission-driven nodes)
    #   - proximity + same_place edges

    # 1. Cluster nodes by a JOINT feature+space metric:
    #      d(a, b) = α · (1 − cos(clip_a, clip_b))
    #              + (1 − α) · normalized_spatial_dist(a, b)
    #    via HDBSCAN (auto-selects cluster count; no k to set;
    #    min_cluster_size is the only structural param).
    # 2. Each cluster → a region. Region embedding = L2-norm
    #    mean of member CLIP embeddings (the pool-and-normalize
    #    helper shared with query-room-by-text-v1).
    # 3. Region label: open-vocab. argmax CLIP-text-similarity
    #    against a candidate vocabulary (operator-supplied or a
    #    default open set), OR leave embedding-only and let the
    #    planner query via query_room_by_text. NO fixed 7-class
    #    argmax.
    # 4. Region object list = union of member detected_objects[*].label.
    # 5. confidence = top-1 label-similarity margin (top-1 minus
    #    top-2 cosine); uncertainty = normalized entropy of the
    #    label-similarity distribution. Both derived from the
    #    clustering + labeling, NOT a trained temperature.
    return [RoomEntry(...) per cluster]
```

**Why this kills five of six tuning points:**

| v1/rejected-v2 knob | Fate under v2 |
|---|---|
| Fixed 7-class prompt set | Gone — open-vocab labels |
| Temperature `T` (calibration) | Gone — confidence is a label-similarity margin, not a calibrated softmax |
| Smoothing `neighbor_weight` / `n_passes` | Gone — clustering enforces spatial+feature coherence directly |
| Jaccard prototype map | Gone — open-vocab labels; object lists are evidence, not a separate refinement |
| Cluster-cache growth fraction | Unchanged — orthogonal perf knob, not a quality knob |
| **NEW: feature/space weight `α`** | The one knob; calibrated once on the eval harness, mostly insensitive |

**Why it handles open-plan by construction:** kitchen-zone
nodes (CLIP features dominated by sink / stove / fridge) and
living-zone nodes (couch / tv) land in different clusters even
with no wall between them, because the CLIP-feature distance
is large. Greedy modularity can't do this — it only sees
spatial adjacency. This is the multi-bedroom adversarial case
solved structurally too: two physically-distinct bedrooms with
similar furniture stay separate clusters as long as the
spatial term keeps them apart (and `same_place` loop-closure
edges, when present, are respected in the joint metric).

### What we borrow, precisely

- **From ConceptGraphs** ([Gu et al., ICRA 2024](https://arxiv.org/abs/2309.16650)):
  feature-driven clustering — regions emerge from where visual
  content changes, not from walls. Object-centric.
- **From HOV-SG** ([Werby et al., RSS 2024](https://arxiv.org/abs/2403.17846)):
  open-vocab labeling against region-centroid embeddings; the
  hierarchy-from-clustering idea.
- **NOT borrowed:** dense 3D point clouds, SAM per-frame masks,
  HOV-SG's *geometric* free-space room partition (which fails
  open-plan exactly as greedy modularity does). The strafer's
  sparse pose-graph + VLM detections are coarser by design —
  appropriate for planner consumption, which needs "kitchen
  has a sink," not per-object 3D nodes.

No external repo is cloned; no weights are trained. The
"learned" content is the pretrained CLIP features — same as
ConceptGraphs / HOV-SG, which also don't train a region head.
The genuinely-trained head is the parked escape valve at
[`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md),
un-parked only if this brief's `α` knob or its sim-to-real
transfer proves insufficient on the eval harness.

### Preserved API — the planner contract doesn't move

The planner consumes `known_rooms` / `current_room` /
`connectivity` / `room_anchor` exactly as v1 shipped them.
`RoomEntry` keeps `(label, member_node_ids, centroid_xy,
confidence, observed_objects)` and gains `uncertainty` (the
field [`llm-guided-frontier-gain`](../../parked/multi-room/llm-guided-frontier-gain.md)'s
v1.5 re-visitation extension consumes). The change is entirely
internal to `SemanticMapManager`. `query_room_by_text` aims at
the new region centroids without modification.

### Tie-in to state-of-the-art

- **ConceptGraphs** (ICRA 2024) — open-vocab object-centric 3D
  scene graphs; feature-clustering, no fixed class set. The
  direct precedent for "no rooms, regions emerge."
- **HOV-SG** (RSS 2024) — hierarchical open-vocab scene graph;
  open-vocab labels via CLIP text query against region nodes.
- **OpenScene** ([CVPR 2023](https://arxiv.org/abs/2211.15654)),
  **VLMaps** (ICRA 2023) — open-vocab scene understanding;
  the implicit-feature-field cousins of this symbolic
  approach (see the implicit-mapping track in
  [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md)).

## Acceptance criteria

- [ ] **Feature+space clustering.** A `partition_regions(graph,
      *, alpha)` helper in
      [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
      clusters nodes by the joint metric via HDBSCAN (or an
      equivalent density clusterer that auto-selects cluster
      count). Pure on the graph snapshot + the ChromaDB
      embedding store.
- [ ] **Open-vocab region labels.** Region labels come from
      CLIP-text-similarity ranking against the region centroid,
      NOT a fixed 7-class argmax. The candidate vocabulary is
      operator-supplied or a default open set; the brief does
      NOT ship a hard-coded seven-class list as the only path.
- [ ] **`RoomEntry` shape preserved + `uncertainty` added.**
      `(label, member_node_ids, centroid_xy, confidence,
      observed_objects)` unchanged; `confidence` is the
      top-1-minus-top-2 label-similarity margin; `uncertainty`
      is normalized label-distribution entropy. No trained
      temperature.
- [ ] **Overlapping / same-label regions allowed.** Drop v1's
      same-label cluster merge. Two spatially-distinct
      `"bedroom"` regions stay separate `RoomEntry` values.
      (`room_anchor("bedroom")` returning one of them is the
      v1 limitation this resolves; the API may need an
      instance-aware accessor — decide and document.)
- [ ] **Single calibrated knob.** The feature/space weight
      `α` is calibrated once against the eval harness and
      shipped as a module constant with a comment pointing at
      the run report. No per-deployment tuning.
- [ ] **Quality lift measured.** PR description carries
      [`room-state-eval-harness`](room-state-eval-harness.md)
      cluster purity (V-measure), label P/R, and the
      multi-bedroom + open-plan adversarial scenes, vs. the v1
      greedy-modularity baseline. V-measure lift ≥ 0.10
      absolute on the open-plan and multi-bedroom scenes
      (where v1 structurally fails).
- [ ] **Open-plan adversarial scene.** The eval set gains an
      open-plan scene (kitchen + dining + living, no walls).
      v1 lumps it into one region; v2 must split it into
      ≥ 2 semantically-coherent regions. (Coordinate with the
      eval-harness brief on adding this scene.)
- [ ] **No new VLM calls.** Region object lists reuse
      `detected_objects` already on the nodes. Confirm by
      inspection.
- [ ] **Unit tests.** Synthetic graphs: two well-separated
      feature clusters at overlapping spatial positions
      (open-plan); two spatially-separated clusters with
      similar features (multi-bedroom); a single coherent
      cluster (single room). Verify the partition splits /
      merges as expected. Verify `confidence` / `uncertainty`
      derivation. Verify `RoomEntry` backward-compat shape.
- [ ] **No regression.** Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites pass; planner-side consumers
      ([`autonomy-stack`](autonomy-stack.md),
      [`query-room-by-text-v1`](query-room-by-text-v1.md))
      see the same API.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same commit.
      See
      [`conventions.md`'s user-facing documentation
      maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).

## Investigation pointers

- v1 substrate being replaced:
  [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
  — `RoomClassifier`, `cluster_nodes`,
  `aggregate_room_entries`.
- The pool-and-normalize helper shared with the open-vocab
  query path:
  [`query-room-by-text-v1`](query-room-by-text-v1.md).
- HDBSCAN: `hdbscan` (or `sklearn.cluster.HDBSCAN` ≥ 1.3).
  Joint metric via precomputed distance matrix over the node
  set (node counts are small — hundreds, not millions).
- Reference architectures: ConceptGraphs (arXiv:2309.16650),
  HOV-SG (arXiv:2403.17846), OpenScene (arXiv:2211.15654).

## Out of scope

- **Trained region head.** Filed parked at
  [`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md)
  as the escape valve — un-park only if this brief's `α` knob
  or sim-to-real transfer proves insufficient. The harness
  epic supplies the training corpus when that fires.
- **Task-driven dynamic granularity** (CLIO-style "Living room
  → Media area → remote" multi-level emergence). Filed parked
  at
  [`dynamic-region-granularity`](../../parked/multi-room/dynamic-region-granularity.md)
  as v3 — un-park only if v2's static regions prove too coarse
  for some mission.
- **Loop closure / same-place node dedup.** Stays at
  [`semantic-graph-loop-closure`](semantic-graph-loop-closure.md);
  this brief consumes its `same_place` edges in the joint
  metric but does not implement detection.
- **Backbone swap.** Inherited from
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md);
  this brief works on whichever CLIP backbone is loaded.
- **Implicit / sub-symbolic mapping.** The parallel track at
  [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md);
  this brief is the symbolic-layer region partition.
- **Multi-floor.** Strafer is single-story.
