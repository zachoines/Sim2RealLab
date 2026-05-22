# Compute room connectivity at scene-generation time and verify door-open default

**Type:** new feature / refactor
**Owner:** DGX agent (`strafer_lab` scene-generation pipeline;
no Jetson code)
**Priority:** P1 (foundational for multi-room as MVP default;
the autonomy-stack brief and the mission generator both depend
on the connectivity graph this brief produces)
**Estimate:** S (~half a day; A* over the navigable mask + JSON
schema addition + a verification pass on existing scenes)
**Branch:** task/multi-room-scene-connectivity-validation

## Story

As an **operator who has decided multi-room is the strafer MVP
default**, I want **`scene_metadata.json` to carry a *verified*
room-to-room connectivity graph (with doorway crossings and path
costs) and `prep_room_usds.py` to guarantee doors are open (or
absent between connected rooms) at scene-generation time**, so
that **the harness's training-time consumers (mission generator,
oracle driver, grader, paired training pairs) have a single
source of truth for which rooms are reachable from which, no
mission is silently unsolvable due to a closed-door scene
quirk, and the connectivity ground-truth can be used to score
the *runtime* `observation-derived-room-state` agent against a
known answer in sim**.

This brief intentionally scopes the graph to **training-time /
harness consumption only**. The live planner does not read
`scene_metadata.json` — see
[`autonomy-stack`'s Sim-to-real boundary](autonomy-stack.md#sim-to-real-boundary-read-this-first)
for the rule and
[`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
for the runtime equivalent.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)

Sibling briefs depending on this:
- [`mission-generator`](../harness/mission-generator.md)
  *(formerly path-shape-generator)* — LLM-as-planner consumes
  the graph as part of its scene prompt; mission queue filters
  cross-room missions to reachable pairs. **Primary consumer
  of this brief's output.**
- [`autonomy-stack`](autonomy-stack.md) —
  does NOT consume this graph at runtime (was previously
  stated as a consumer; corrected after the multi-room audit
  on 2026-05-13). The runtime planner reads the
  observation-derived equivalent instead. This brief's
  ground-truth IS used to *score* the runtime agent's
  connectivity inference in sim grading, but never to *drive*
  the runtime agent.
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
  — runtime counterpart of this brief. Scored against this
  brief's ground-truth.

## Context

### What `scene_metadata.json` has today

Verified against
[`extract_scene_metadata.py`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py)
and the typed reader at
[`strafer_lab/tools/scene_labels.py`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py)
on 2026-05-13:

```json
{
  "rooms": [
    {
      "room_type": "kitchen",
      "footprint_xy": [[x0,y0], [x1,y1], ...],
      "area_m2": 12.3,
      "story": 0
    }
  ],
  "objects": [
    {
      "instance_id": 42,
      "label": "chair",
      "semantic_tags": ["seating", ...],
      "prim_path": "/World/Room/...",
      "position_3d": [x, y, z],
      "bbox_3d_min": [...],
      "bbox_3d_max": [...],
      "room_idx": 0,
      "relations": [...],
      "materials": [...]
    }
  ],
  "room_adjacency": [[0, 1], [1, 0], ...]
}
```

**`room_adjacency` already exists** — it is a flat edge list of
room indices serialized from Infinigen's in-memory `room_graph`.
What it *does not* carry today: a `reachable` boolean confirming
the navigable mask actually permits the traversal, a doorway
crossing point, a path cost, or a per-edge door-state tag. The
`mission-generator` brief's reference to a `connectivity[]`
block is therefore correct in intent but not in name; this
brief produces the enriched form alongside (or replacing) the
plain `room_adjacency` edge list.

**Other schema notes the prior draft missed:**
- The room name field is `room_type` (semantic class —
  `kitchen`, `bedroom`), not `name`. Multiple rooms can share a
  `room_type`; downstream consumers must index by `(story,
  room_idx)`, not by string.
- Rooms carry a `story` (floor) integer. Cross-story connectivity
  via stairs is **out of scope for v1** — the mecanum strafer
  cannot climb stairs. The graph must explicitly drop
  cross-story edges or mark them `reachable: false` with
  `reason: "stairs"`. Scenes with multiple stories should also
  set `multi_story: true` at the top level so the mission
  generator can skip cross-story missions.
- Objects have a `room_idx` integer already pointing at their
  containing room; no point-in-polygon lookup needed for
  training-time per-object room queries (a `get_room_at_position`
  helper exists at
  [`scene_labels.py:148`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py#L148)
  for the pose-only case).

### What this brief adds

Two artifacts, both at scene-generation time:

1. **A `connectivity[]` block in `scene_metadata.json`.** For
   every (room_a, room_b) pair sharing an Infinigen
   `room_adjacency` edge (or whose footprints share a boundary
   segment), an entry with:
   - `from_idx`, `to_idx`: room indices into `rooms[]`. Indices,
     not `room_type` strings — multiple rooms can share a type.
   - `reachable`: bool — whether the connectivity check (see
     below) confirms the navigable area actually permits
     passage.
   - `via_doorway_xy`: optional `[x, y]` of the doorway crossing.
   - `path_length_m`: optional distance for difficulty grading.
   - `door_state`: optional one of `"open"`, `"absent"`,
     `"force-opened"` documenting how the open-pass was achieved.
   The plain integer-pair `room_adjacency` edge list is kept for
   backward compatibility (existing consumers in
   `mission-generator` and `scene_labels.py` are not broken).

2. **A door-state guarantee in `prep_room_usds.py` (or its
   USD-postprocess hook).** After Infinigen scene generation +
   USD postprocessing, run a check: for every
   `room_adjacency` edge, verify the navigable region allows
   passage between the two room footprints. If a doorway is
   blocked (closed door, accidental obstacle), either
   force-open the door at postprocess time (omit the closed
   door's collider in
   [`postprocess_scene_usd.py`](../../../../source/strafer_lab/scripts/postprocess_scene_usd.py)
   so the occupancy reflects the open state), or tag the scene
   `multi_room_incompatible: true` so downstream mission
   generators skip cross-room missions on it.

### Where the navigable region comes from

The original draft of this brief assumed
`prep_room_usds.py` "already touches the navigable mask for the
ground-plane lift." Verified false on 2026-05-13: that script
generates scenes via Infinigen, and
[`postprocess_scene_usd.py`](../../../../source/strafer_lab/scripts/postprocess_scene_usd.py)
attaches colliders + lights — neither emits an occupancy grid
or navigable mask.

Three viable sources, picked by cost:

- **(Recommended) NVIDIA's Isaac Sim Occupancy Map extension**
  (`omni.isaac.occupancy_map`). Reads the stage's physics
  colliders (which `postprocess_scene_usd.py` already attaches)
  and emits a 2D binary occupancy grid at a configurable
  z-slice. Documented at
  [Isaac Sim Occupancy Map Generator](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/digital_twin/ext_isaacsim_asset_generator_occupancy_map.html).
  Already vendored with Isaac Sim 5.x. Run as a one-shot
  Replicator-style script after postprocess; cache result as
  `<scene>/occupancy.npy` next to `scene_metadata.json`. This
  is the canonical primitive — using it keeps the scene-gen
  pipeline aligned with the rest of NVIDIA's tooling and
  guarantees the connectivity check operates on the same
  geometry the runtime occupancy grid (sim-side RTAB-Map and
  Nav2 costmap) will see.
- **(Fallback) Footprint-polygon rasterization.** Rasterize
  `rooms[].footprint_xy` polygons + per-object
  `bbox_3d_min/max` projections at floor height. Cheap, but
  misses geometry outside the polygon-tagged subset (e.g.,
  furniture that Infinigen places without a room tag, free
  space inside doorways defined only by collision geometry).
- **(Avoid) Custom USD mesh rasterizer.** Hand-rolling a
  triangle-rasterizer over the USD stage replicates what the
  Isaac Sim primitive already does. Don't.

The A* itself is trivial once the grid exists — a 30-line
8-connected grid search over a `(H, W)` `np.uint8` array.

### Why the brief output stays connectivity-graph-shaped
### (not just a raw occupancy grid)

The harness consumers (mission generator, oracle driver, grader)
want a small graph they can iterate, not a few-thousand-by-
few-thousand binary array. The raw occupancy is an
*intermediate* — cached so re-running connectivity is fast,
but the consumed contract is the graph.

### Computing connectivity

Pseudocode for the connectivity pass (uses the schema names
verified against the codebase):

```python
def compute_connectivity(
    scene_metadata: dict, occupancy_grid: np.ndarray,
    grid_origin_xy: tuple[float, float], grid_resolution_m: float,
) -> list[dict]:
    rooms = scene_metadata["rooms"]
    base_edges = scene_metadata.get("room_adjacency", [])
    # Iterate Infinigen's room_graph edges first; fall back to all
    # unordered pairs only if room_adjacency is empty.
    candidates = base_edges or [
        [i, j] for i in range(len(rooms)) for j in range(i + 1, len(rooms))
    ]
    edges = []
    for src_idx, dst_idx in candidates:
        if rooms[src_idx]["story"] != rooms[dst_idx]["story"]:
            edges.append({
                "from_idx": src_idx, "to_idx": dst_idx,
                "reachable": False, "reason": "stairs",
            })
            continue
        src = polygon_centroid(rooms[src_idx]["footprint_xy"])
        dst = polygon_centroid(rooms[dst_idx]["footprint_xy"])
        path = a_star_on_grid(
            occupancy_grid, src, dst, grid_origin_xy, grid_resolution_m,
        )
        if path is None:
            edges.append({
                "from_idx": src_idx, "to_idx": dst_idx,
                "reachable": False, "reason": "blocked",
            })
            continue
        doorway = find_room_boundary_crossing(path, rooms[src_idx], rooms[dst_idx])
        edges.append({
            "from_idx": src_idx, "to_idx": dst_idx,
            "reachable": True,
            "via_doorway_xy": list(doorway),
            "path_length_m": path_length_m(path, grid_resolution_m),
        })
    # Mirror to symmetric edges.
    edges.extend(_mirror(e) for e in edges)
    return edges
```

### Door-open verification

For each `room_adjacency` edge whose `reachable` came back
False, attempt remediation: search the postprocess collider set
for door-like prims (`*Door*Factory_*`) intersecting the
shared boundary, drop the matching collider, regenerate the
occupancy grid, retry. If still unreachable, tag the *edge* (not
the whole scene) with `door_state: "blocked"` and tag the scene
`multi_room_incompatible: true` only if **no** cross-room edge
is `reachable: true`. Single-pair blockage is non-fatal — the
mission generator can still produce missions on the reachable
subset.

## Acceptance criteria

- [ ] **Occupancy grid generation.** A new scene-gen step
      produces `occupancy.npy` (+ a small `occupancy.json`
      sidecar carrying `origin_xy`, `resolution_m`,
      `z_slice_m`) using
      `omni.isaac.occupancy_map`. Lives next to
      `scene_metadata.json` per existing scene-dir layout. The
      pipeline (`prep_room_usds.py ingest` or the postprocess
      hook) regenerates the occupancy after door-collider
      mutations. If the Isaac Sim primitive proves impractical,
      document the fallback rasterizer chosen and *why* in the
      script's module docstring.
- [ ] **Connectivity pass.** A new helper (sibling to
      `extract_scene_metadata.py`, not bolted onto it — keep
      the Infinigen-state-walker pure) computes `connectivity[]`
      from the occupancy grid + `rooms[].footprint_xy` +
      Infinigen's `room_adjacency` candidate set. Output is
      appended to the existing `scene_metadata.json` schema;
      backward-compatible (consumers that don't read
      `connectivity[]` still work; consumers that read
      `room_adjacency` still work).
- [ ] **Cross-story exclusion.** Cross-story edges are emitted
      as `{reachable: false, reason: "stairs"}` and the scene
      gets `multi_story: true` at the top level if any room has
      `story != 0`. Single-story scenes are unaffected.
- [ ] **Connectivity graph schema documented.** A short
      `connectivity[]` schema description added to
      `extract_scene_metadata.py`'s docstring and to
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stage 6's metadata section.
- [ ] **Door-state verification.** `prep_room_usds.py` (or
      `postprocess_scene_usd.py`, whichever is the right hook)
      runs the door-open check. Closed-door scenes either get
      the offending collider removed, OR get tagged
      `multi_room_incompatible: true` in `scene_metadata.json`
      so downstream mission generators skip cross-room missions
      on them.
- [ ] **Re-process existing scenes.** Run the connectivity pass
      against `scene_high_quality_dgx_000_seed0` (the existing
      multi-room scene) and at least 2 newly-generated multi-room
      seeds. Connectivity matrix in each is non-trivial (≥ 1
      `reachable: true` cross-room pair).
- [ ] **Smoke test (harness consumer).** The mission generator
      ([`mission-generator`](../harness/mission-generator.md))
      consumes the new connectivity block and emits cross-room
      missions only between reachable pairs. Verified by
      inspecting the generated `mission_queue.yaml`.
- [ ] **No live-autonomy regression.** A grep of
      `source/strafer_autonomy/strafer_autonomy/` for
      `room_adjacency`, `connectivity`, `via_doorway_xy`,
      `from_idx`, `to_idx` returns zero hits. This brief's
      output is harness-only.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit.
- [ ] No regression on single-room scenes. The `connectivity[]`
      block is empty (or has only same-room reflexive entries)
      for single-room scenes; consumers handle this without
      special-casing.

## Investigation pointers

- Existing scene-gen tooling:
  [`prep_room_usds.py`](../../../../source/strafer_lab/scripts/prep_room_usds.py),
  [`postprocess_scene_usd.py`](../../../../source/strafer_lab/scripts/postprocess_scene_usd.py),
  [`extract_scene_metadata.py`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py).
- Schema reader / typed accessors used by existing harness
  consumers:
  [`strafer_lab/tools/scene_labels.py`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py)
  — `RoomEntry`, `ObjectEntry`, `iter_rooms`, `iter_objects`,
  `get_room_at_position`, `get_objects_in_room`.
- Where `room_adjacency` is already written:
  [`extract_scene_metadata.py:123`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py#L123)
  and the serializer at
  [`extract_scene_metadata.py:258`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py#L258).
- Isaac Sim primitive for occupancy generation:
  [Isaac Sim Occupancy Map Generator](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/digital_twin/ext_isaacsim_asset_generator_occupancy_map.html).
  Lives in the same Kit interpreter the bridge already uses.
- A* implementation: not worth shipping a sophisticated planner;
  a 30-line grid-A* over a 2D mask is sufficient.
- The existing multi-room scene
  `scene_high_quality_dgx_000_seed0` is the natural smoke-test
  target. Verify all rooms reach each other (or document the
  closed-door exceptions found). Check that
  `multi_story` is False (or absent) — the strafer cannot
  handle stairs in v1.

## Out of scope

- **Runtime door-state changes.** Doors stay assumed-open at
  scene-gen time. Articulated openable doors at runtime is a
  separate brief if pursued.
- **Dynamic obstacles.** Connectivity is computed against the
  static navigable mask, not against runtime obstacle additions
  (e.g., a moved chair). Acceptable for v1.
- **Connectivity beyond room-level.** No sub-room reachability
  graph (e.g., "behind the couch is reachable from the door").
  Room granularity is sufficient for the planner's transit-step
  emission and the LLM mission generator's queue filtering.
- **Sim-to-real connectivity.** This brief operates on Infinigen
  metadata only. Real-robot connectivity from RTAB-Map's actual
  occupancy grid is filed at
  [`observation-derived-room-state`](../../completed/observation-derived-room-state.md).
- **Live-autonomy consumption.** Explicitly out of scope and
  explicitly forbidden. The autonomy-stack brief reads the
  observation-derived equivalent only; see
  [autonomy-stack's Sim-to-real boundary](autonomy-stack.md#sim-to-real-boundary-read-this-first).
- **Multi-story scenes.** The strafer mecanum cannot climb
  stairs. Cross-story edges are emitted as
  `{reachable: false, reason: "stairs"}`; lifts / stairs /
  multi-story routing is not pursued in v1.
