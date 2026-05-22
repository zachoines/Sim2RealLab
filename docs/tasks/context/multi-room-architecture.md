# Multi-room architecture map

How the briefs in the multi-room epic layer on top of each other,
which orthogonal tracks they share with other epics, and which
briefs consume the room-state output downstream. The room-state
work is a multi-stage research thread; this module is the index.

For pickup-ordered, lane-filtered tables, see
[`../BOARD.md`](../BOARD.md). For the actual API surface (`RoomEntry`,
`current_room`, `known_rooms`, `connectivity`, `room_anchor`) see
[`../completed/observation-derived-room-state.md`](../completed/observation-derived-room-state.md).

---

## Version layers (multi-room epic)

The room-state stack is structured as layered improvements on top
of v1's shipped API surface. Each layer either preserves v1's
return shapes and adds new metadata, or adds new methods alongside.
Pickup-gates between layers are explicit in each brief.

The PR #43 architecture review consolidated what was a six-knob
threshold-tuned "v2 series" into a single SOTA-aligned v2: an
**unsupervised feature+space region partition with open-vocab
labels** (ConceptGraphs / HOV-SG shape). The trained-head and
VPR variants are *escape valves*, not the v2 floor — they
un-park only if the unsupervised v2 measurably falls short on
the eval harness.

| Layer | Brief | What it adds | What stays the same |
|---|---|---|---|
| **v1** (shipped) | [`observation-derived-room-state`](../completed/observation-derived-room-state.md) | The four-method `SemanticMapManager` API + CLIP zero-shot room labeling + greedy-modularity clustering + Nav2 reachability hook | — (this is the floor) |
| **v1.5** active | [`query-room-by-text-v1`](../active/multi-room/query-room-by-text-v1.md) | `query_room_by_text(text)` method against region-centroid embeddings; open-vocab query on the v1 backbone | All v1 methods + `RoomEntry.label` |
| **v1 maintenance** parked | [`room-state-runtime-ergonomics`](../parked/multi-room/room-state-runtime-ergonomics.md) | `room_anchor` per-label index + `RoomClassifier` sticky-disable retry | API surface |
| **v2 (measurement)** active | [`room-state-eval-harness`](../active/multi-room/room-state-eval-harness.md) | Measurement substrate (V-measure, P/R, time-to-converge, connectivity P/R) + the training corpus for the escape valves. Ships first | All API surface; no runtime change |
| **v2 (the quality work)** active | [`semantic-region-partition`](../active/multi-room/semantic-region-partition.md) | Feature+space HDBSCAN clustering + open-vocab labels; replaces v1's greedy-modularity + 7-class argmax. **One** `α` knob; `RoomEntry.uncertainty` from label-similarity margin | `RoomEntry` shape (+`uncertainty`); `known_rooms` / `current_room` / `connectivity` / `room_anchor` |
| **v2** active | [`semantic-graph-loop-closure`](../active/multi-room/semantic-graph-loop-closure.md) | `same_place` edge type; consumed by v2's joint clustering metric | API surface |
| **v2.5** parked | [`semantic-map-lifecycle-merge`](../parked/multi-room/semantic-map-lifecycle-merge.md) | `consolidate()` replaces v1's `prune()`; hierarchical decay (recent / long-term + spatial pooling) | Query API; `RoomEntry` shape |
| **v3 escape (granularity)** parked | [`dynamic-region-granularity`](../parked/multi-room/dynamic-region-granularity.md) | CLIO-style task-driven granularity — region expands to place / object on demand per mission. Un-park if v2 regions too coarse | All v2 + v1 methods (additive) |
| **v3 escape (partition)** parked | [`learned-region-head`](../parked/multi-room/learned-region-head.md) | Trained partition replacing v2's HDBSCAN+`α`. Un-park if the single `α` can't hold both open-plan and multi-bedroom splits, or doesn't transfer sim→real | `RoomEntry` shape; v2 clustering is the fallback under `STRAFER_REGION_HEAD_ENABLED=0` |
| **v3 escape (loop closure)** parked | [`learned-vpr-loop-closure`](../parked/multi-room/learned-vpr-loop-closure.md) | Drop-in VPR descriptor (SALAD / MegaLoc / AnyLoc) for `detect_loop_closures`. Un-park if raw-CLIP loop-closure calibration fails | All other CLIP consumers (labeling, text queries, validator cosine) |

**Escape valves are filed-on-trigger only.** Read their `## Trigger detail`
sections before un-parking. v2's unsupervised clustering may
clear the bars; pre-empting the escape valves costs v2 the
chance to prove it's sufficient.

---

## Orthogonal axes

These tracks advance independently of the version stack and feed
multiple layers transparently.

### Sub-symbolic / implicit-mapping (clip-validation + experimental epics)

The multi-room version stack above is the **symbolic** layer (LLM
planner + compiler consume discrete `RoomEntry` objects). A parallel
**sub-symbolic** layer lives in other epics:

| Brief | Epic | Role |
|---|---|---|
| [`implicit-memory-map`](../parked/clip-validation/implicit-memory-map.md) | clip-validation | **The** implicit-mapping primitive — memory bank of past embeddings + cross-attention, RAG-aware training, cold-deployment augmentation. Shared by two consumers. |
| [`cotrained-retrieval-augmented`](../parked/clip-validation/cotrained-retrieval-augmented.md) | clip-validation | Consumer #1 — co-trains the CLIP tower (Step A) + wires the cascade validator onto the implicit memory map (Step B). |
| [`vla-v2-map-conditioning`](../parked/experimental/vla-v2-map-conditioning.md) | experimental | Three-shape ablation: how the v2 VLA consumes the map (A: serialize symbolic regions / B: consume the implicit memory map as consumer #2 / C: no consumption). |

Threshold-free reasoning lives in this layer; the symbolic layer
above accepts a knob or two as the cost of producing discrete
entities the LLM planner can reason over.

### Backbone (clip-validation epic)

| Brief | Role |
|---|---|
| [`backbone-bakeoff`](../parked/clip-validation/backbone-bakeoff.md) | Replaces OpenCLIP ViT-B/32 with DINOv3 / SigLIP-2 / MobileCLIP-2 across every CLIP consumer (room labeling, text queries, validator cosine, semantic-map retrieval). Orthogonal to the version stack — upgrades v1, v2, v3 transparently. |

---

## Planner-side consumers

Briefs that consume the room-state API as input rather than
extending it. Not internal to the room-state series, but anyone
working on a room-state quality brief should know the
contract these expect.

| Brief | Reads | Role |
|---|---|---|
| [`autonomy-stack`](../active/multi-room/autonomy-stack.md) | `current_room`, `known_rooms[*].observed_objects`, `connectivity`, `room_anchor` | Lifts the multi-room runtime deferral; compiler emits transit / explore steps from room-level state |
| [`planner-far-target-staging`](../active/multi-room/planner-far-target-staging.md) | `current_room`, `room_anchor`, `target_known_poses` | Far-target multi-hop staging in the compiler |
| [`scene-connectivity-validation`](../active/multi-room/scene-connectivity-validation.md) | (sim-side only) | Harness ground-truth connectivity graph in `scene_metadata.json` — **not** runtime; scored against runtime `connectivity()` in the multi-room grader |
| [`llm-guided-frontier-gain`](../parked/multi-room/llm-guided-frontier-gain.md) | `current_room`, `connectivity`, `RoomEntry.uncertainty` (from v2) | LFG-style scalar LLM prior over frontier exploration; its re-visitation extension consumes `RoomEntry.uncertainty`, which v2's `semantic-region-partition` produces from the label-similarity margin |
| [`frontier-cognitive-fsm`](../parked/multi-room/frontier-cognitive-fsm.md) | Same as `llm-guided-frontier-gain` + per-state context | CogNav-style FSM upgrade to frontier exploration; parked-on-trigger if LFG plateaus |

Planner-architecture migration (Option C → B) lives in
[`staging-hops-shadow-mode`](../parked/multi-room/staging-hops-shadow-mode.md)
and
[`planner-scene-graph-expansion`](../parked/multi-room/planner-scene-graph-expansion.md).
Those don't extend the room-state API; they extend the
`world_state` payload the planner-LLM sees.

---

## How to pick up a brief from this map

1. **Reading this module** tells you *where* a brief fits. **Read
   the brief itself** for what to build.
2. **v1 is shipped.** Picking up an active v2 brief means it
   inherits v1's API unchanged — no v1 work needed.
3. **v2 (`semantic-region-partition`) preserves the `RoomEntry`
   return shape** (it adds `uncertainty`); it changes how
   regions are *computed* (feature+space clustering, open-vocab
   labels), not what consumers read. Downstream consumers see
   better regions, not a different API.
4. **Don't un-park an escape valve** without re-reading its
   `## Trigger detail` against the current state of v2 measurements
   in [`../active/multi-room/room-state-eval-harness.md`](../active/multi-room/room-state-eval-harness.md)'s
   most recent run. The escape valves exist *only* if v2's
   unsupervised clustering measurably falls short.

---

## Maintenance contract

This module is **the architectural index for the multi-room work**.
Same maintenance rule as [`../BOARD.md`](../BOARD.md):

- **Filing a new brief in the multi-room epic, in
  `parked/multi-room/`, or in a cross-epic brief that extends a
  room-state layer**: add a row to the relevant table in the same
  PR.
- **Un-parking an escape valve**: flip its row to active and
  note what it supersedes (e.g., `learned-region-head` becomes
  the partition and `semantic-region-partition`'s HDBSCAN
  becomes its fallback).
- **Shipping a brief**: update its status column; if it was a
  layer brief, the next layer's pickup-gate may now be open —
  reflect that in the table.

When in doubt about whether something is a multi-room concern: if
it touches `SemanticMapManager` or consumes `RoomEntry`, it goes
here.
