# Open-vocab room labeling on v1 — `query_room_by_text`

**Type:** new feature
**Owner:** DGX agent (lives in
`source/strafer_autonomy/strafer_autonomy/semantic_map/`,
DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P2 (replaces the fixed `DEFAULT_ROOM_PROMPTS` /
`DEFAULT_ROOM_PROTOTYPES` brittleness at the planner-query
boundary; cheap to ship; works on the existing OpenCLIP
ViT-B/32 backbone without waiting for `backbone-bakeoff`)
**Estimate:** S (~1–2 days; one new method on
`SemanticMapManager` + per-backbone eval on the room-state
harness + README documentation)
**Branch:** task/query-room-by-text-v1

**Pickup gate:** Becomes pickable once
[`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
merges. Does NOT block on
[`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
— the API surface works on whichever CLIP backbone is loaded
at runtime; the bakeoff only changes how *well* it works.

## Story

As a **planner-LLM or operator asking "is there a room with a
cooking surface and a refrigerator?" against the live
semantic map**, I want **`SemanticMapManager` to expose a
`query_room_by_text(text)` method that scores known rooms by
CLIP text→room-centroid-embedding similarity**, so that
**target-room inference stops being gated on a hand-curated
`DEFAULT_ROOM_PROTOTYPES` dict, the planner can disambiguate
without me editing a constant per home, and the architectural
brittleness called out in the multi-room audit at the
fixed-label boundary goes away on the v1 backbone — not
deferred to whichever backbone the bakeoff eventually picks**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md) — where this brief sits in the v1 / v1.5 / v2 / v2.5 / v3 / escape-valve stack, and which planner-side consumers depend on its output.
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
  — v1 implementation. This brief adds one method to the
  manager; the existing `known_rooms` / `current_room` /
  `connectivity` / `room_anchor` API surface is unchanged.
- [`semantic-region-partition`](semantic-region-partition.md)
  — v2 sibling. That brief produces the region centroids this
  query API aims at (and labels them open-vocab at
  construction). This brief is the *query-time* complement:
  planners and operators ask in free-form text against the
  region centroids rather than inspecting `RoomEntry.label`.
  The two compose — ship either order.
- [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
  — formerly housed this API as an extension. Pulled out so it
  can ship on v1's backbone without waiting. The backbone brief
  retains the per-backbone *quality* measurement of this API.
- [`room-state-eval-harness`](room-state-eval-harness.md) —
  measures the open-vocab precision@5 / MRR.

## Context

### What's brittle today (v1)

The v1 implementation in
[`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
ships a fixed
`DEFAULT_ROOM_PROMPTS = ("a kitchen", "a living room", "a
bedroom", "a bathroom", "a hallway", "an office", "a garage")`.
Every consumer of `RoomEntry.label` is locked into this seven-
class vocabulary:

- The autonomy-stack compiler reads `RoomEntry.label` to
  pick a transit destination via `room_anchor(label)`.
- The planner-LLM serializes `current_room` + `known_rooms`
  into its prompt as the seven discrete labels.

(v2's [`semantic-region-partition`](semantic-region-partition.md)
retires the fixed-prompt-set classifier outright in favor of
open-vocab region labels; this brief delivers the *query-side*
half of that on the v1 backbone immediately, independent of
when v2's partition lands.)

Two failure modes:

1. **A room that doesn't fit the seven classes** (a
   home gym, a sunroom, a dining room, a workshop) gets
   force-classified into the nearest neighbor — usually
   `"office"` or `"living room"` — and the planner reasons
   about it as the wrong type.
2. **A query the planner-LLM forms in free-form text** ("the
   room with the cooking surface and refrigerator", "the room
   with the printer") can't be answered against the discrete
   labels. Today the planner has to guess which label
   probably contains the printer; the answer is "office" but
   only because office's prototypes include "printer."

This brief replaces both with a free-form text query.

### The API

```python
class SemanticMapManager:
    # Existing v1 API — unchanged:
    def known_rooms(self) -> list[RoomEntry]: ...
    def current_room(self, pose, *, max_distance_m=3.0) -> RoomEntry | None: ...
    def connectivity(self) -> list[tuple[str, str]]: ...
    def room_anchor(self, room_label: str) -> Pose2D | None: ...

    # NEW:
    def query_room_by_text(
        self, text: str, n_results: int = 5,
    ) -> list[tuple[RoomEntry, float]]:
        """Score known rooms by CLIP text→room-centroid-embedding
        similarity. Returns the top-n rooms with similarity
        scores, sorted descending. Bypasses the fixed prompt
        set entirely.

        Implementation: for each known room, compute the mean
        CLIP embedding of its member nodes (L2-normalized after
        averaging — mean of unit vectors is not a unit vector);
        encode the query text via the same CLIP text tower used
        for `DEFAULT_ROOM_PROMPTS`; cosine-similarity against
        each centroid. Cached alongside the cluster cache and
        invalidated on the same growth-fraction trigger.
        """
```

Returns are ranked `(RoomEntry, similarity)` tuples — the
caller decides whether to threshold, take top-1, or feed the
top-K back to the planner-LLM. **No hard threshold ships in
the API**; the brittleness this brief escapes is "is 0.25
similarity good enough?" If a caller wants a threshold, it
picks one in its own code with explicit per-callsite
rationale.

### Why this can ship on v1's OpenCLIP ViT-B/32

The `query_room_by_text` API only requires:

- A text tower that produces an embedding (OpenCLIP's
  `encode_text`, already available).
- A way to pool per-room member embeddings (compute the mean,
  L2-normalize).
- Cosine similarity (one matmul).

None of those depend on the
[`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
outcome. The bakeoff measures *how well* `query_room_by_text`
works under different backbones — but the API surface and the
runtime path are backbone-agnostic. Folding this into v1
removes the dependency chain `backbone-bakeoff → query API →
planner uses it`, which was deferring an obvious quality win
behind a long-horizon investigation.

The discrete `RoomEntry.label` field is **preserved** for
backward compat. v1 consumers that read it continue to work;
new consumers can query free-form text. The eventual
`backbone-bakeoff` swap upgrades the backbone behind both
paths without code changes outside `clip_encoder.py`.

### Where the room centroid embedding comes from

Each `RoomEntry` already exposes `member_node_ids: tuple[str,
...]`. Each member node has its CLIP image embedding stored
in ChromaDB at `clip_embedding_id`. The centroid is:

```python
member_embs = [
    manager.get_clip_embedding(
        graph.nodes[nid]["data"]["clip_embedding_id"]
    )
    for nid in room.member_node_ids
]
centroid = np.mean(member_embs, axis=0)
norm = np.linalg.norm(centroid)
centroid = centroid / max(norm, 1e-8)  # L2-normalize after averaging
```

The L2-normalize-after-average step is the same one called
out in the
[`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md)
brief's pooled-anchor acceptance criterion — same primitive,
same trap. Lift the pool-and-normalize helper to
[`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
so both consumers share it.

### Cache shape

The cluster cache in v1 stores `list[RoomEntry]`. This brief
adds a sibling cache mapping `room.label → centroid embedding
(np.ndarray)`, invalidated on the same growth-fraction
trigger. On a 100 m² home with ~50–200 nodes, room centroids
recompute in <10 ms; not worth a more aggressive invalidation.

### Tie-in to state-of-the-art

- **CLIP-Fields** (ICRA 2023, arXiv:2210.05663) — dense
  CLIP feature field queried by text. The same primitive at
  the dense level; this brief does it at the room-cluster
  level, which is the right granularity for the planner's
  consumption pattern.
- **OpenScene** (CVPR 2023, arXiv:2211.15654) — 3D
  open-vocabulary scene understanding via CLIP text queries.
  Direct precedent.
- **VLMaps** (ICRA 2023, arXiv:2210.05714) — open-vocab map
  for navigation. Same query shape; their map is dense, ours
  is graph-clustered, but the consumer-facing API is
  identical.
- **HOV-SG** (RSS 2024, arXiv:2403.17846) — hierarchical
  open-vocab scene graph; their `query_by_text` against
  region-level nodes is exactly the same shape.
- **OK-Robot** (arXiv:2401.12202) — text-grounded object
  navigation via CLIP query against a memory bank. v1's API
  is the room-level analogue.

The pattern is settled enough that "fixed prompt set + argmax
cosine" is a 2021-era pattern and "open-vocab text query
against a feature bank" is the 2023-onward pattern. v1's
discrete labels were the floor; this brief lifts the runtime
to the modern standard *without* changing the backbone.

## Acceptance criteria

- [ ] **`query_room_by_text` method.** Lives on
      `SemanticMapManager` with the signature above. Returns
      `list[tuple[RoomEntry, float]]` sorted descending by
      similarity, truncated to `n_results`.
- [ ] **Centroid pooling helper.** A shared `_pool_clip(emb_list)
      -> np.ndarray` helper in
      [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
      computes the mean and L2-normalizes. Reused (or planned
      to be reused) by
      [`semantic-map-lifecycle-merge`](../../parked/multi-room/semantic-map-lifecycle-merge.md)'s
      pooled-anchor path.
- [ ] **Centroid cache.** A sibling cache to
      `_cluster_cache` mapping `room.label → centroid
      embedding`, invalidated on the same growth-fraction
      trigger. Unit-tested by adding nodes and verifying
      cache invalidation fires.
- [ ] **CLIP text encoder fallback.** If the CLIP text tower
      is disabled (no ONNX), `query_room_by_text` returns
      `[]` and logs at INFO. Same graceful-degrade pattern as
      `RoomClassifier._ensure_text_embeddings`.
- [ ] **Vision-only backbone awareness.** If a future
      [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
      pick lands a vision-only backbone (DINOv3-S) and no
      text tower is loaded, `query_room_by_text` raises
      `NotImplementedError` with a clear log message. (See
      the bakeoff brief's "Vision-only backbone fallback"
      acceptance criterion.)
- [ ] **Backward compat.** `RoomEntry.label` field unchanged.
      All existing consumers of the v1 API
      (`autonomy-stack`'s compiler, the planner-prompt
      serializer, `room_anchor`) continue to work without
      modification.
- [ ] **README documentation.** Operator-facing section in
      [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md)'s
      "Semantic-map room state" subsection covers the new
      method with one usage example.
- [ ] **Eval harness — precision@5 + MRR.** PR description
      includes the
      [`room-state-eval-harness`](room-state-eval-harness.md)
      open-vocab metrics: for each Infinigen `room_type` in
      the eval set, query the manager with a hand-curated
      free-form description ("the room with the cooking
      surface and refrigerator" for `kitchen`, etc.) and
      report precision@5 + MRR against the ground-truth room.
      Target: precision@1 ≥ 0.7 on the v1 four-scene eval
      set; precision@5 ≥ 0.9. Below those bars, file a
      follow-up to fine-tune the backbone via
      [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md).
- [ ] **Unit tests.** Synthetic graphs with seeded room
      clusters and known CLIP embeddings; verify
      `query_room_by_text` returns expected ranking; verify
      cache invalidation on node add; verify the
      `RoomEntry.label` field is unchanged by the query path.
- [ ] **No regression.** Existing
      [`test_semantic_map.py`](../../../../source/strafer_autonomy/tests/test_semantic_map.py)
      and
      [`test_room_state.py`](../../../../source/strafer_autonomy/tests/test_room_state.py)
      suites pass.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same
      commit. See
      [`conventions.md`'s user-facing documentation
      maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- v1 classifier + prompt set:
  [`semantic_map/room_state.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
  — `RoomClassifier`, `DEFAULT_ROOM_PROMPTS`. The text encoder
  this brief reuses is exactly `RoomClassifier`'s.
- v1 cache scaffolding:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `_cluster_cache`, `_cluster_cache_stale`,
  `_get_known_rooms_cached`. The centroid cache mirrors this.
- Existing text-query precedent:
  `SemanticMapManager.query_by_text(query_text, n_results)`
  already does text→ChromaDB ANN over per-node embeddings.
  Naming clash worth noting: `query_by_text` returns raw
  embedding hits; `query_room_by_text` returns aggregated
  rooms. Both surfaces co-exist; the latter is the right
  consumer for the planner.
- Reference architectures: CLIP-Fields (arXiv:2210.05663),
  OpenScene (arXiv:2211.15654), VLMaps (arXiv:2210.05714),
  HOV-SG (arXiv:2403.17846), OK-Robot (arXiv:2401.12202).

## Out of scope

- **Open-vocab labeling of per-node `room_label`.** This
  brief is query-time, not stamp-time. Per-node stamps stay
  on the v1 seven-class
  [`DEFAULT_ROOM_PROMPTS`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/room_state.py)
  set until v2's
  [`semantic-region-partition`](semantic-region-partition.md)
  replaces the classifier with open-vocab region labels.
- **Retiring `DEFAULT_ROOM_PROMPTS`.** That's v2's
  [`semantic-region-partition`](semantic-region-partition.md);
  this brief's API addition is additive and ships on the v1
  per-node stamps in the meantime.
- **Backbone choice.** Backbone selection stays at
  [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md).
  This brief works on whichever backbone is loaded.
- **Fine-tuning the CLIP text tower** for room queries.
  Filed at
  [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)
  if the precision@5 target proves insufficient.
- **Multi-floor / cross-story queries.** Strafer is single-
  story; the brief drops the floor level intentionally.
