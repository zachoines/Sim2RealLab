# Loop closure on the semantic-map graph

**Type:** new feature
**Owner:** DGX agent (lives in
`source/strafer_autonomy/strafer_autonomy/semantic_map/`,
DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P3 (no immediate consumer break in v1; required
infrastructure for the parked
[`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md)
brief and a quiet quality lift for long-horizon deployments)
**Estimate:** M (~3–5 days; detection pass + merge / same-place
edge type + tests + eval lift measurement)
**Branch:** task/semantic-graph-loop-closure

**Pickup gate:** Becomes pickable once
[`observation-derived-room-state`](observation-derived-room-state.md)
merges. Strongly recommended ordering:
[`room-state-eval-harness`](room-state-eval-harness.md)
ships first so the cluster-fragmentation fix can be measured
on a repeated-traversal trajectory.

## Story

As a **long-running deployment that revisits the same physical
spaces from different headings**, I want **the semantic-map
graph to detect duplicate-place nodes via CLIP-similarity +
spatial-proximity and either merge them or annotate them as
the same place**, so that **the graph does not fragment across
revisits, modularity-based clustering doesn't find spurious
multi-cluster splits of a single room, and the lifecycle work
in [`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md)
has merge candidates to act on**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`observation-derived-room-state`](observation-derived-room-state.md)
  — v1 implementation. This brief extends its proximity
  graph.
- [`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md)
  — parked downstream consumer of this brief's output.
- [`room-state-eval-harness`](room-state-eval-harness.md) —
  measures the cluster-fragmentation fix.

## Context

### The fragmentation problem

RTAB-Map handles metric-level loop closure: when the robot
returns to a known location, RTAB-Map fuses the pose
estimate. The semantic-map layer has no equivalent. Two
captures of the same physical spot from different headings
produce:

- Two separate semantic-map nodes (`obs_NNNN`).
- Different CLIP image embeddings (the view content differs).
- Different `detected_objects` sets (different objects are
  visible from different angles).
- A proximity edge between them (they're spatially close), so
  modularity will *probably* cluster them together — but only
  if the proximity-edge density is high enough to overcome
  the within-room edge density.

The pathology emerges over time: a room walked multiple times
accumulates 20–50 nodes; modularity may split one room into
several spurious clusters because some headings are
under-connected to others. v1's smoke test (single
traversal) doesn't exercise this.

### Visual loop closure on the semantic graph

Add a detection pass that runs periodically (or on demand):

```python
def detect_loop_closures(
    graph: nx.DiGraph,
    *,
    similarity_threshold: float = 0.80,
    distance_threshold_m: float = 1.0,
    candidate_pool_size: int = 20,
) -> list[tuple[str, str, float]]:
    """Return (node_a, node_b, similarity) candidates."""
    # For each node, retrieve top-K nearest-CLIP neighbors
    # via the existing ChromaDB ANN. Filter to those within
    # `distance_threshold_m` spatially. Keep pairs above
    # `similarity_threshold`.
```

The brief leaves two implementation choices open:

**(a) Merge.** Replace the two nodes with one; union the
`detected_objects`; mean the poses; preserve the older
`node_id`. Reduces node count. Loses provenance.

**(b) Same-place edge.** Add a new edge type (`same_place`,
weight = similarity) to the graph. Both nodes survive;
downstream clustering treats `same_place` edges as
high-weight. Preserves provenance. No node-count reduction.

**Recommended: (b) for v1.5.** Preserves the ability to audit
where the detection came from, and matches what RTAB-Map does
internally (loop-closure constraints, not node deletion).
The merge variant becomes the natural fit when the
[`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md)
brief picks up these candidates with explicit lifecycle
policy.

### Where the candidate pool comes from

ChromaDB already exposes ANN search via
`SemanticMapManager.query_by_embedding(emb, n_results)`. The
detection pass walks each node, encodes it (or retrieves its
stored embedding via `get_clip_embedding`), and ANN-queries
the top-K nearest neighbors. Spatial-proximity filter is a
cheap post-step (compare `pose_x`/`pose_y` against the
candidate's metadata).

ANN search is O(log N) per query; the full pass is O(N log N)
which is fine up to ~10,000 nodes — well above any expected
session size.

### When to run the detection pass

Three options, pick by cost:

- **On every add_observation**: too expensive, would slow
  capture.
- **On cluster cache invalidation**: cheap, matches existing
  growth-fraction trigger. Recommended.
- **On demand from the consumer**: an explicit
  `manager.detect_loop_closures()` API call. Useful for the
  lifecycle brief.

Ship both the periodic-on-cache-invalidation trigger AND the
explicit API call.

### Tie-in to state-of-the-art

- **DBoW** (Gálvez-López & Tardós 2012) — bag-of-words
  visual loop closure, the foundational technique RTAB-Map
  uses at the metric layer. CLIP embeddings + ANN replace
  BoW + inverted file.
- **NetVLAD** (Arandjelović et al. 2016) — learned descriptors
  for place recognition. Our use of CLIP embeddings is a
  zero-shot analogue — but NetVLAD itself is dated for 2025
  VPR; see the modern-VPR row below.
- **Modern VPR (2023–2025): CosPlace, EigenPlaces, MixVPR,
  SALAD, MegaLoc, AnyLoc.** CosPlace (CVPR 2022) and
  EigenPlaces (ICCV 2023) frame place recognition as
  classification over geocell partitions; MixVPR (WACV 2023)
  is a feature-mixer global descriptor; SALAD (CVPR 2024) is
  the current strong supervised baseline; AnyLoc (ICRA 2024)
  and MegaLoc (2024) are foundation-model-driven and lift
  raw DINOv2 / CLIP into VPR-grade descriptors without
  fine-tuning. Survey context:
  [Improving VPR with Sequence-Matching Receptiveness
  Prediction (arXiv:2503.06840)](https://arxiv.org/abs/2503.06840)
  and the
  [VPR pair-retrieval evaluation (arXiv:2603.13917)](https://arxiv.org/abs/2603.13917).
  v1.5's raw-CLIP cosine is the *floor*; if the eval-harness
  measurement (see calibration criterion below) shows the
  similarity-only signal is noise-bound on the multi-bedroom
  adversarial scene, the v3 follow-up is SALAD or AnyLoc as a
  drop-in descriptor — same ANN store, different embeddings,
  no architectural change.
- **Kimera-Multi** (arXiv:2106.14367) — explicit loop closure
  at the scene-graph layer. Same shape as this brief.
- **ConceptGraphs** (arXiv:2309.16650) — uses CLIP similarity
  + spatial proximity for object-node merging; the
  room-level case here is structurally identical.

### Threshold calibration is part of acceptance

The `similarity_threshold = 0.80` default in the algorithm
sketch is a placeholder. OpenCLIP ViT-B/32 cosine
similarities for "same physical spot, different heading"
typically sit in `[0.60, 0.85]` — and for "two bedrooms"
(distinct rooms, same furniture vocabulary) often exceed
`0.75` because CLIP's image embedding is dominated by
visible objects. **The threshold cannot be set blind**; it
must be calibrated against the
[`room-state-eval-harness`](room-state-eval-harness.md) eval
set's multi-bedroom adversarial scene, sweeping the
threshold to maximize the precision-recall trade and
documenting the operating point in the PR description. The
brief acceptance below now requires this; the spatial filter
`distance_threshold_m` is calibrated jointly.

## Acceptance criteria

- [ ] **Detection function.**
      `detect_loop_closures(graph, similarity_threshold,
      distance_threshold_m, candidate_pool_size) -> list[tuple[str, str, float]]`
      ships in
      [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
      (or a new `loop_closure.py` if the file gets crowded),
      pure on the graph + ANN store inputs.
- [ ] **Manager wiring.** `SemanticMapManager` gains
      `detect_loop_closures()` returning the candidate list,
      AND runs a detection pass on cluster-cache
      invalidation, AND annotates the detected pairs into
      the graph as `same_place` edges with weight = CLIP
      similarity.
- [ ] **Cluster downstream consumes the edges.**
      `cluster_nodes` weights `same_place` edges higher than
      proximity edges so the clusterer treats revisited
      regions as one cluster. (Modularity supports weighted
      edges natively.)
- [ ] **Fragmentation measured.** PR description carries
      cluster purity on a repeated-traversal trajectory
      (added to the
      [`room-state-eval-harness`](room-state-eval-harness.md)
      scene set) — v1 baseline vs. v1.5-with-loop-closure.
      Cluster purity should hold steady (≤ 0.05 V-measure
      drop) across 2× / 3× repeated traversals, where v1
      currently degrades.
- [ ] **No false positives on adversarial scenes.** The
      multi-bedroom eval scene must NOT have its two
      bedrooms loop-closed into one (different physical
      spaces, similar CLIP embeddings). Tune
      `distance_threshold_m` to prevent this; document the
      tuning rationale in the brief's commit message.
- [ ] **Threshold calibration on the eval set.** PR
      description carries a precision-recall sweep over
      `similarity_threshold ∈ [0.60, 0.90]` and
      `distance_threshold_m ∈ [0.5, 3.0]` on the
      [`room-state-eval-harness`](room-state-eval-harness.md)
      eval set, identifying the operating point. The
      0.80/1.0 m defaults in the algorithm sketch are
      placeholders; ship whichever pair the sweep selects.
      Documented in the brief's commit message AND surfaced
      as module-level constants with a comment pointing at
      the run report.
- [ ] **Unit tests.** Synthetic graph with seeded duplicate
      nodes; verify detection produces expected pairs;
      verify spatial-threshold filtering rejects far
      candidates with similar embeddings; verify weight on
      detected edges promotes one-cluster outcome.
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

- ANN store:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `query_by_embedding`, `get_clip_embedding`.
- Existing proximity edges:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `_add_proximity_edges`.
- DBoW reference: Gálvez-López & Tardós 2012.
- NetVLAD: Arandjelović et al. 2016.
- Kimera-Multi: arXiv:2106.14367.

## Out of scope

- **Node merging / deletion.** This brief annotates
  same-place edges; deciding when to *merge or delete* lives
  in
  [`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md).
- **Appearance-change handling.** Lighting / seasonal /
  furniture-moved variation between revisits is a v3
  concern; this brief uses static CLIP similarity.
- **Multi-session loop closure.** A node from yesterday's
  session vs. today's session can't be matched until the
  lifecycle brief defines how cross-session graphs are
  persisted; out of scope here.
- **Learned place recognition descriptors.** v1.5 uses raw
  CLIP. Modern VPR descriptors (SALAD, MegaLoc, AnyLoc,
  CosPlace, EigenPlaces, MixVPR — see Tie-in above) are a
  v3 alternative if the calibrated CLIP-cosine signal proves
  insufficient on the eval harness. Drop-in: same ANN store,
  different embedding tower; no architectural change.
