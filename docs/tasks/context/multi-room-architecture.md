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

| Layer | Brief | What it adds | What stays the same |
|---|---|---|---|
| **v1** (shipped) | [`observation-derived-room-state`](../completed/observation-derived-room-state.md) | The four-method `SemanticMapManager` API + CLIP zero-shot room labeling + greedy-modularity clustering + Nav2 reachability hook | — (this is the floor) |
| **v1.5** active | [`query-room-by-text-v1`](../active/multi-room/query-room-by-text-v1.md) | `query_room_by_text(text)` method against room-centroid embeddings; replaces the `DEFAULT_ROOM_PROMPTS` brittleness at the planner-query boundary | All v1 methods + `RoomEntry.label` |
| **v1 maintenance** parked | [`room-state-runtime-ergonomics`](../parked/multi-room/room-state-runtime-ergonomics.md) | `room_anchor` per-label index + `RoomClassifier` sticky-disable retry | API surface |
| **v2 L1** active | [`room-state-eval-harness`](../active/multi-room/room-state-eval-harness.md) | Measurement substrate (V-measure, P/R, time-to-converge, ECE, connectivity P/R) | All API surface; no runtime change |
| **v2 L2** active | [`room-state-temporal-smoothing`](../active/multi-room/room-state-temporal-smoothing.md) | `metadata["room_label_smoothed"]` (read-raw / write-smoothed; idempotent under cache rebuild) | `metadata["room_label"]` + all return shapes |
| **v2 L2** active | [`room-state-uncertainty-calibration`](../active/multi-room/room-state-uncertainty-calibration.md) | `RoomEntry.uncertainty` field + `metadata["room_label_dist"]` (calibrated softmax) | All v1 methods; `RoomEntry.confidence` is now calibrated probability |
| **v2 L2** active | [`room-label-vlm-refinement`](../active/multi-room/room-label-vlm-refinement.md) | VLM-refined per-node labels via Jaccard scoring against an object→room prototype map; no new VLM calls | API surface |
| **v2 L4** active | [`semantic-graph-loop-closure`](../active/multi-room/semantic-graph-loop-closure.md) | `same_place` edge type on the graph; cluster cache treats them as high-weight | API surface |
| **v2.5 L4** parked | [`semantic-map-lifecycle-merge`](../parked/multi-room/semantic-map-lifecycle-merge.md) | `consolidate()` replaces v1's `prune()`; hierarchical decay (recent / long-term + spatial pooling) | Query API; `RoomEntry` shape |
| **v3 L5** active | [`semantic-graph-object-centric-hierarchical`](../active/multi-room/semantic-graph-object-centric-hierarchical.md) | `ObjectNode` / `PlaceNode` / `RoomNode` + `known_objects` / `known_places` / `object_anchor` / `place_anchor` / `hierarchical_graph` | All v1 methods (backward-compat) |
| **v3+ escape** parked | [`learned-region-head`](../parked/multi-room/learned-region-head.md) | Learned head trained on the eval-harness corpus; one forward pass replaces v2's prompt-set + clustering + smoothing + calibration + VLM-refinement pipeline | `RoomEntry` shape; v1 path remains as fallback under `STRAFER_REGION_HEAD_ENABLED=0` |
| **v3+ escape** parked | [`learned-vpr-loop-closure`](../parked/multi-room/learned-vpr-loop-closure.md) | Drop-in VPR descriptor (SALAD / MegaLoc / AnyLoc) for `detect_loop_closures`; same ANN store + same `same_place` edge protocol | All other CLIP consumers (room labeling, text queries, validator cosine) |

**Escape valves are filed-on-trigger only.** Read their `## Trigger detail`
sections before un-parking. v2's threshold tuning may clear the
bars; pre-empting the escape valves costs the v2 review attention.

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
| [`cotrained-retrieval-augmented`](../parked/clip-validation/cotrained-retrieval-augmented.md) | clip-validation | Project's implicit-mapping primitive (memory bank + cross-attention over past CLIP embeddings). Cascade validator is consumer #1; v2 VLA is consumer #2. |
| [`vla-v2-map-conditioning`](../parked/experimental/vla-v2-map-conditioning.md) | experimental | Three-shape ablation contract for how a v2 VLA consumes the map (text-serialized vs. cross-attention memory-bank vs. no consumption). |

Threshold-free reasoning lives in this layer; the symbolic layer
above accepts thresholds as the cost of producing discrete entities
the LLM planner can reason over.

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
| [`llm-guided-frontier-gain`](../parked/multi-room/llm-guided-frontier-gain.md) | `current_room`, `connectivity`, `RoomEntry.uncertainty` (v1.5) | LFG-style scalar LLM prior over frontier exploration; v1.5 extension consumes uncertainty for known-room re-visitation |
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
   inherits v1 unchanged — no v1 work needed.
3. **A v2 quality brief that adds metadata** doesn't change the
   `RoomEntry` return shape. Downstream consumers see better
   numbers, not different ones.
4. **Don't un-park an escape valve** without re-reading its
   `## Trigger detail` against the current state of v2 measurements
   in [`../active/multi-room/room-state-eval-harness.md`](../active/multi-room/room-state-eval-harness.md)'s
   most recent run. The escape valves exist *only* if v2 measurably
   falls short.

---

## Maintenance contract

This module is **the architectural index for the multi-room work**.
Same maintenance rule as [`../BOARD.md`](../BOARD.md):

- **Filing a new brief in the multi-room epic, in
  `parked/multi-room/`, or in a cross-epic brief that extends a
  room-state layer**: add a row to the relevant table in the same
  PR.
- **Un-parking an escape valve**: move its row from "v3+ escape
  parked" to the version layer it actually replaces (typically v2
  briefs sunset into `learned-region-head`'s row).
- **Shipping a brief**: update its status column; if it was a
  layer brief, the next layer's pickup-gate may now be open —
  reflect that in the table.

When in doubt about whether something is a multi-room concern: if
it touches `SemanticMapManager` or consumes `RoomEntry`, it goes
here.
