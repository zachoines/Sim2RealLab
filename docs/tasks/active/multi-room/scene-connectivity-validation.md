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
default**, I want **`scene_metadata.json` to carry a room-to-room
connectivity graph and `prep_room_usds.py` to guarantee doors are
open (or absent between connected rooms) at scene-generation
time**, so that **all downstream consumers (the autonomy-stack
brief's planner, the mission generator's LLM-as-planner, the
harness drivers' filters) have a single source of truth for
which rooms are reachable from which, and no mission is silently
unsolvable due to a closed-door scene quirk**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)

Sibling briefs depending on this:
- [`autonomy-stack`](autonomy-stack.md) —
  planner consumes the connectivity graph for transit-step
  emission.
- [`mission-generator`](../harness/mission-generator.md)
  *(formerly path-shape-generator)* — LLM-as-planner consumes
  the graph as part of its scene prompt; mission queue filters
  cross-room missions to reachable pairs.

## Context

### What `scene_metadata.json` has today

Per [`extract_scene_metadata.py`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py):

```json
{
  "scene_name": "...",
  "scene_type": "...",
  "objects": [{"label": "chair", "position": [...], "bbox": [...]}, ...],
  "rooms": [{"name": "kitchen", "polygon": [...]}, ...],
  "floor_top_z": 0.0
}
```

Per-scene metadata names rooms with polygons but **does not
record connectivity between them**. Multi-room missions today
have no way to know whether a target room is reachable from the
robot's current room — they assume it is.

### What this brief adds

Two artifacts, both at scene-generation time:

1. **A `connectivity[]` block in `scene_metadata.json`.** For
   every (room_A, room_B) pair, an entry with:
   - `from`: room name.
   - `to`: room name.
   - `reachable`: bool — whether A* over the navigable mask
     finds a path between room centroids.
   - `via_doorway`: optional `(x, y)` where the path crosses
     the room boundary (the doorway / passage point).
   - `path_length_m`: optional distance for downstream
     mission-difficulty grading.

2. **A door-state guarantee in `prep_room_usds.py`.** After
   Infinigen scene generation + USD postprocessing, run a check:
   for every doorway implied by adjacent room polygons, the
   navigable mask must allow passage. If a doorway is blocked
   (closed door, accidental obstacle), either:
   - Force-open the door at postprocess time (Infinigen exports
     doors as static geometry; postprocess can omit the closed
     door's collider so the navigable mask reflects the open
     state); or
   - Regenerate the scene with a different seed.
   Document the choice in `prep_room_usds.py`'s help text.

### Computing connectivity

Pseudocode for the connectivity pass:

```python
def compute_connectivity(scene_metadata: dict, navigable_mask: np.ndarray) -> list[dict]:
    rooms = scene_metadata["rooms"]
    edges = []
    for a, b in itertools.combinations(rooms, 2):
        a_centroid = polygon_centroid(a["polygon"])
        b_centroid = polygon_centroid(b["polygon"])
        path = a_star(navigable_mask, start=a_centroid, goal=b_centroid)
        if path is not None:
            doorway = find_room_boundary_crossing(path, a, b)
            edges.append({
                "from": a["name"], "to": b["name"],
                "reachable": True,
                "via_doorway": list(doorway),
                "path_length_m": path_length(path),
            })
            edges.append({"from": b["name"], "to": a["name"], ...})  # symmetric
        else:
            edges.append({"from": a["name"], "to": b["name"], "reachable": False})
            edges.append({"from": b["name"], "to": a["name"], "reachable": False})
    return edges
```

The navigable mask comes from Infinigen's exported scene
geometry — `prep_room_usds.py` already touches it for the
ground-plane lift. No new sim runs needed.

### Door-open verification

For each room-adjacency pair (rooms whose polygons share an
edge), check the navigable mask has at least one connected
sequence of cells crossing the shared edge. If not, log it as
a closed-door scene; either modify the postprocess step to
remove the closed-door collider, or flag the scene as
multi-room-incompatible (don't add cross-room missions for it).

## Acceptance criteria

- [ ] **Connectivity pass.**
      `extract_scene_metadata.py` (or a sibling tool — name in
      the brief work) computes the `connectivity[]` block from
      the navigable mask + room polygons. Output is appended to
      the existing `scene_metadata.json` schema; backward-compatible
      (consumers that don't read `connectivity[]` still work).
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
- [ ] **Smoke test.** A mission generator
      ([`mission-generator`](../harness/mission-generator.md))
      consumes the new connectivity block and emits cross-room
      missions only between reachable pairs. Verified by
      inspecting the generated `mission_queue.yaml`.
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
- Navigable-mask source: Infinigen exports a navigable mesh /
  occupancy in the scene's USD layer. The exact extraction
  helper may already exist for the ground-plane lift; reuse if
  so, write minimal A* if not.
- A* implementation: not worth shipping a sophisticated planner;
  a 30-line grid-A* over a 2D mask is sufficient.
- The existing multi-room scene
  `scene_high_quality_dgx_000_seed0` is the natural smoke-test
  target. Verify all rooms reach each other (or document the
  closed-door exceptions found).

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
  occupancy grid is a separate problem; future brief.
