# Task-driven dynamic region granularity (CLIO-style)

**Type:** new feature / research (filed-on-trigger)
**Owner:** DGX agent (lives in
`source/strafer_autonomy/strafer_autonomy/semantic_map/`,
DGX edit lane per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P3 — filed-on-trigger. The v3 follow-up to v2's
static region partition
([`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md)).
Un-park only when v2's static regions prove **too coarse** for
a real mission — see "Trigger detail." Do NOT pre-empt; static
regions may be enough.
**Estimate:** L (~1.5–2 wk; task-conditioned re-clustering +
multi-level hierarchy maintenance + new query API + eval)
**Branch:** task/dynamic-region-granularity

## Story

As an **autonomy stack whose mission needs finer granularity
than v2's static regions provide — "look at the remote on the
media console" when v2 only knows "living_area" — on homes
where the right level of detail depends on the mission, not on
a fixed hierarchy depth**, I want **the semantic map to
re-cluster regions at a task-conditioned granularity, exposing
sub-region (place) and object nodes on demand rather than
materializing one fixed `room → place → object` tree**, so
that **the planner can ground a mission at whatever level the
command implies ("the living room" → region; "the remote" →
object inside a media-area place), the hierarchy adapts to the
task instead of being prescribed, and open-plan homes where
"room" is ambiguous get the granularity that matters for the
current mission**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/multi-room-architecture.md`](../../context/multi-room-architecture.md) — this is v3 on the symbolic layer; the version stack lives here.
- [`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md)
  — **v2, this brief's substrate.** v2 produces static
  feature+space regions. This brief adds task-conditioned
  re-clustering and a multi-level (region → place → object)
  expansion on top, only where the task needs it.
- [`observation-derived-room-state`](../../completed/observation-derived-room-state.md)
  — v1, the original flat graph + `RoomEntry` API.
- [`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
  — primary consumer. This brief's finer granularity enables
  object-level mission grounding the compiler can't express
  against region-level state alone.
- [`learned-spatial-encoder`](learned-spatial-encoder.md) — sibling
  escape valve on the *partition* (v2 → trained head). This
  brief is orthogonal: it's about *granularity* (how deep the
  hierarchy goes per task), not about *how the partition is
  computed*. Either can land first.

## Trigger detail — when to un-park

v2's static regions are the floor. This brief is the
finer-granularity escape valve. File active only when **at
least one** of:

1. **A real mission needs object-level grounding v2 can't
   express.** "Look at the remote," "go to the left
   nightstand," "inspect the stove's left burner" — commands
   that reference an object or sub-region, where v2's
   region-level `RoomEntry` (kitchen / living_area) is too
   coarse to resolve the target. The
   [`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
   compiler hits the "I know the region but not the object"
   wall.
2. **Open-plan granularity is mission-dependent.** The same
   open-plan space needs to be "one region" for a patrol
   mission but "kitchen-zone vs. living-zone" for a
   go-to-target mission, and a single static partition can't
   serve both. CLIO's task-conditioning is exactly this case.
3. **The planner-LLM's object-disambiguation queries
   systematically fail** because the map exposes region object
   *lists* but not object *poses* / sub-region structure ("the
   chair next to the table" needs the table's location, which
   region-level state discards). This overlaps with
   [`planner-scene-graph-expansion`](planner-scene-graph-expansion.md)'s
   trigger — coordinate which brief owns the fix.

If none of these bite, the static v2 regions are sufficient
and this brief is **deleted**, not picked up.

## Context

### Why fixed `room → place → object` was the wrong shape

The original draft of this brief proposed a fixed three-level
hierarchy (`RoomNode → PlaceNode → ObjectNode`) materialized
eagerly for every home, with DBSCAN over object positions
(`eps ≈ 1.5 m`) for places and a frozenset object-signature
lookup for place labels. The PR #43 architecture review
rejected this for two reasons:

1. **"Room" is ambiguous.** Open-plan kitchen + dining +
   living, studio flats, and homes where only furniture or
   floor-type differentiates zones have no clean room
   boundaries to anchor the top level. v2 already addresses
   this by letting regions emerge from feature+space
   clustering rather than walls — but a fixed hierarchy on top
   re-imposes the rigidity.
2. **Granularity is task-dependent, not structural.** Whether
   the right unit is "living room," "media area," or "remote"
   depends on the mission. Materializing all three levels for
   every home, eagerly, is wasted structure for missions that
   only need the region — and insufficient structure when a
   mission needs a level the fixed depth didn't anticipate.

### The CLIO insight

[CLIO (Maggio et al., 2024)](https://arxiv.org/abs/2404.13696)
builds **task-driven** open-set 3D scene graphs: the right
granularity is selected per task using an information-theoretic
objective (Information Bottleneck) over the task description.
"Go to the kitchen sink" expands the kitchen region down to the
sink object; "patrol the living area" keeps the living region
coarse. The hierarchy is not a fixed tree — it's re-clustered
to the task.

Applied to the strafer's sparse pose-graph + v2 regions:

- v2's static regions are the **coarse** level (always
  present).
- When a mission references a finer target, the map
  **re-clusters within the relevant region(s)** to expose
  sub-region places and object nodes — but only there, only
  then.
- The expansion is driven by the mission text + the
  `detected_objects` already on the region's member nodes; no
  new perception.

### Multi-level, emergent — the "Living room → Media area → remote" case

Under CLIO-style task-conditioning, this expansion is on
demand:

```
Mission: "look at the remote"
  → planner identifies target "remote"
  → map expands living_area region:
       living_area
         — media_area (place: tv + remote + console cluster)
              — remote (object)
         — sitting_area (place: couch + pillow cluster)
  → planner grounds at the remote's object node
```

For a coarser mission ("patrol the living area"), no expansion
happens — `living_area` stays a single region node. Depth is a
function of the task, not a fixed schema.

### What we borrow, precisely

- **From CLIO** ([arXiv:2404.13696](https://arxiv.org/abs/2404.13696),
  code at [github.com/MIT-SPARK/Clio](https://github.com/MIT-SPARK/Clio)):
  task-driven granularity selection — re-cluster to the
  mission, don't materialize a fixed tree. The
  Information-Bottleneck objective is the reference; a simpler
  task-conditioned re-clustering may suffice at strafer scale.
- **From Hydra-2 / Khronos** ([Schmid et al., RSS 2024](https://arxiv.org/abs/2402.13817)):
  multi-layer scene graph with learned partitioning between
  layers + dynamic update. The depth-flexible nesting.
- **From ConceptGraphs** (ICRA 2024): object nodes as
  first-class, emergent from detections.
- **NOT borrowed:** a fixed `room → place → object` depth;
  eager materialization; the rejected frozenset place-label
  lookup + fixed DBSCAN `eps`.

Object identity still flows from the existing
`reinforce_or_add_object` Kalman-cluster path (one physical
object → one node), as the original brief specified — that
part survives.

## Acceptance criteria

- [ ] **Task-conditioned expansion API.** A method on
      `SemanticMapManager` that, given a region (or region set)
      + a mission/target string, re-clusters within it to
      expose sub-region places + object nodes. Coarse regions
      from v2 are the default; expansion is on demand.
- [ ] **Multi-level, depth-flexible.** The hierarchy nests to
      whatever depth the task requires (region → place →
      object), not a fixed three levels. Singleton objects link
      directly to the region; no empty intermediate places.
- [ ] **Object identity preserved.** Expansion reuses the
      existing `reinforce_or_add_object` Kalman-cluster — one
      physical object → one object node, with every
      contributing observation linked.
- [ ] **No eager materialization.** The full hierarchy is NOT
      built for every home at every `consolidate()`. It's
      computed per task, cached for the mission's duration,
      discarded after. Verify by inspection that idle homes
      carry only v2 region nodes.
- [ ] **Backward compat.** All v2 + v1 APIs (`known_rooms`,
      `current_room`, `connectivity`, `room_anchor`,
      `query_room_by_text`) unchanged. The expansion is
      additive.
- [ ] **Object-localization measured.** PR description carries
      object-localization precision/recall on the
      [`room-state-eval-harness`](../../active/multi-room/room-state-eval-harness.md)
      scenes for missions that exercise object-level grounding,
      vs. v2's region-only baseline.
- [ ] **Granularity-vs-task eval.** Demonstrate on ≥ 2
      missions that the same open-plan scene expands
      differently per task (coarse for patrol, fine for
      go-to-object).
- [ ] **Unit tests.** Synthetic graphs; task-conditioned
      expansion produces expected depth; idempotent
      re-expansion; object-identity preservation; no eager
      materialization.
- [ ] **README documentation.** The "Semantic-map room state"
      subsection of
      [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md)
      gains the task-conditioned expansion API.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same commit.

## Investigation pointers

- v2 substrate:
  [`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md)
  — `partition_regions` produces the coarse regions this brief
  expands.
- Existing object identity:
  [`semantic_map/manager.py`](../../../../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py)
  — `reinforce_or_add_object` Kalman-cluster path.
- Reference architectures:
  - **CLIO** ([arXiv:2404.13696](https://arxiv.org/abs/2404.13696),
    [code](https://github.com/MIT-SPARK/Clio)) — task-driven
    granularity; read §3 (Information Bottleneck task
    selection) and §4 (real-time construction).
  - **Hydra / Khronos** ([RSS 2024](https://arxiv.org/abs/2402.13817))
    — multi-layer scene graph, dynamic update.
  - **ConceptGraphs** (arXiv:2309.16650) — object-centric
    emergent graph.

## Out of scope

- **v2 region partition.** That's
  [`semantic-region-partition`](../../active/multi-room/semantic-region-partition.md);
  this brief consumes its regions and expands them.
- **How the partition is computed** (fixed metric vs. learned
  head). Orthogonal — see
  [`learned-spatial-encoder`](learned-spatial-encoder.md). This brief
  is about granularity, not the partition algorithm.
- **Object-pose `world_state` for the planner-LLM.** That's
  [`planner-scene-graph-expansion`](planner-scene-graph-expansion.md);
  coordinate triggers if both fire (this brief produces the
  object nodes, that brief serializes them into the planner
  prompt).
- **Sub-symbolic / VLA path.** The implicit memory map at
  [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md)
  is the parallel track.
- **Multi-floor.** Strafer is single-story; the floor level is
  dropped intentionally.
