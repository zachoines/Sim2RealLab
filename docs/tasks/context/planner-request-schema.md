# Planner request schema

Canonical reference for the `world_state` block carried on every
planner request, and for how the planner emits plans. The schema is
shared by the runtime room-state path
([`observation-derived-room-state`](../active/multi-room/observation-derived-room-state.md)),
the multi-room compiler path
([`autonomy-stack`](../active/multi-room/autonomy-stack.md)), and
the far-target staging path
([`planner-far-target-staging`](../active/multi-room/planner-far-target-staging.md)).
Whichever consuming brief lands first defines the wire shape; later
briefs reuse it.

## Status — none of this exists yet

This module documents the target contract. As of writing, **none
of the pieces are wired**:

| Piece | Current state | Brief that lands it |
|---|---|---|
| `SemanticMapManager` class | Exists, but **orphaned in production** — only constructed in tests. | [`validator-evaluation`](../active/clip-validation/validator-evaluation.md) wires `SemanticMapManager` + `BackgroundMapper` + `TransitMonitor` into `executor/main.py` on the Jetson. |
| `current_room` / `known_rooms` / `connectivity` / `room_anchor` on the manager | Not implemented. | [`observation-derived-room-state`](../active/multi-room/observation-derived-room-state.md) adds the four methods + CLIP zero-shot room classifier + graph clustering. |
| `world_state` field on `PlanRequest` wire type + Jetson populate + DGX consume | Not present. Today's request carries only `robot_state: dict \| None` and `active_mission_summary: dict \| None` (both opaque). | Either [`autonomy-stack`](../active/multi-room/autonomy-stack.md) or [`planner-far-target-staging`](../active/multi-room/planner-far-target-staging.md) — whichever ships first defines the wire shape; the second consumes it. |
| Compiler reads `world_state` and emits transit / staging steps | `_compile_single_target_steps` is intent-blind. | Compiler grows room-aware logic in [`autonomy-stack`](../active/multi-room/autonomy-stack.md); off-costmap staging helper lands in [`planner-far-target-staging`](../active/multi-room/planner-far-target-staging.md). |

Treat the rest of this module as the contract those briefs ship
against, not as a description of running code.

## Architecture

Per the decision recorded in
[`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c)
and the brief at
[`planner-architecture-alignment`](../completed/planner-architecture-alignment.md):

- The LLM remains a **thin intent classifier**. It emits one
  `MissionIntent` per request (no skill-sequence output).
- The **plan compiler** is the staging authority. It reads
  `world_state` and inflates the intent into the executable plan,
  including any transit / staging / exploration steps.
- An optional `staging_hops` advisory field on `MissionIntent` is
  reserved for the migration toward LLM-emitted hops, but is
  **not consumed** until the operator promotes it. See the
  migration steps in
  [`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c).
- When the compiler sees an unfamiliar advisory field it cannot
  validate, it **ignores it and falls back to its own logic**.
  No silent acceptance of LLM-emitted plans.

## `world_state` block

Lives on the planner request as an optional block. **Populated
entirely on the Jetson** by the executor (`mission_runner.py`) —
pose + costmap fields directly from ROS clients, room-block
fields from the Jetson-resident `SemanticMapManager`. The DGX
planner service is a pure consumer over HTTP. Absent or `None`
→ planner runs with no world knowledge (legacy behavior; still
valid).

```python
class PlannerWorldState(BaseModel):
    # Robot pose (from ROS clients in mission_runner)
    robot_pose_map: Pose2D
    last_grounding: GroundingSummary | None  # depth, stability, age_s

    # Target-label semantic-map lookup (populated per request
    # against the MissionIntent's target_label — empty list if
    # the manager has no prior sightings)
    target_known_poses: list[Pose2D]

    # Room-level (from SemanticMapManager; observation-derived only —
    # see the sim-to-real boundary in autonomy-stack)
    current_room: str | None
    known_rooms: list[RoomEntry]
    connectivity: list[tuple[str, str]]
```

`RoomEntry` matches the manager's return type defined in
[`observation-derived-room-state`](../active/multi-room/observation-derived-room-state.md),
extended with a per-room object inventory:

```python
class RoomEntry(BaseModel):
    label: str
    member_node_ids: list[str]
    centroid_xy: tuple[float, float]
    confidence: float
    observed_objects: list[str]   # deduplicated labels seen in this room
```

Component shapes (all populated Jetson-side):

| Field | Shape | Source | Notes |
|---|---|---|---|
| `robot_pose_map` | `Pose2D(x, y, yaw)` | mission_runner via ROS client | `map` frame. |
| `last_grounding` | `GroundingSummary(depth_m, stability, age_s)` or `None` | mission_runner from prior `scan_for_target` | Lets the planner reason about staleness. |
| `target_known_poses` | `list[Pose2D]` | `SemanticMapManager.query_by_label(intent.target_label)` | Empty list if no prior sighting. One entry per distinct sighting pose. The compiler picks the nearest for the warm-start transit path. |
| `current_room` | `str` or `None` | `SemanticMapManager.current_room()` | `None` at cold-start. |
| `known_rooms` | `list[RoomEntry]` (see above) | `SemanticMapManager.known_rooms()` | Empty at cold-start. `observed_objects` is the compiler's primary signal for target-room inference. |
| `connectivity` | `list[(room_a, room_b)]` | `SemanticMapManager.connectivity()` | Pessimistic — absence ≠ unreachable. |

**Note: no `global_costmap_extent`.** The compiler has no use for
costmap geometry at compile time — it doesn't have the target pose
until `project_detection_to_goal_pose` runs (step 02 of the compiled
plan). Far-target detection at *compile* time is room-shaped and
uses `target_known_poses` for distance estimation; far-target
detection at *run* time stays in the Jetson reactive staging loop
shipped by
[`nav2-far-goal-staging`](../completed/nav2-far-goal-staging.md),
which queries the actual costmap (not a bounding box).

**Why the Jetson owns the whole block.** Observations arrive at
the executor (Jetson reads the D555 + ROS topics), and the
`SemanticMapManager` + `BackgroundMapper` are wired into the
production executor by
[`validator-evaluation`](../active/clip-validation/validator-evaluation.md)
— they run alongside the executor on the Jetson, in the same
process tree. Keeping the map state Jetson-resident means the
planner request crosses the LAN with a self-contained snapshot,
and the DGX planner service stays stateless w.r.t. the map.

**Population contract.** The Jetson must populate **all** fields
it can fill. A partial `world_state` is a footgun: the compiler
on the DGX treats absent fields as "caller-asserted absent."
When a field cannot be filled (e.g., Nav2 not yet running, map
empty at cold-start), send `None` for that field rather than
omitting it.

**No `scene_metadata.json` reads.** Neither the Jetson populator
nor the DGX planner backend may read Infinigen sim-side metadata.
The sim-to-real-boundary subsection of
[`autonomy-stack`](../active/multi-room/autonomy-stack.md#sim-to-real-boundary-read-this-first)
is the binding constraint; the room block on this schema comes
from `SemanticMapManager`, not from `scene_metadata.json`.

## `MissionIntent` extensions

Carried on the planner response. The base intent surface is
defined in
[`source/strafer_autonomy/strafer_autonomy/schemas/mission.py`](../../../source/strafer_autonomy/strafer_autonomy/schemas/mission.py).

```python
@dataclass(frozen=True)
class MissionIntent:
    intent_type: str
    raw_command: str
    target_label: str | None = None
    # ... existing fields ...

    # Advisory only. Reserved for the C → B migration.
    # The compiler IGNORES this field until the migration step
    # that promotes it lands. Emitting it is harmless.
    staging_hops: tuple[str, ...] | None = None
```

`staging_hops` is a list of room labels or named landmarks the LLM
suggests the robot pass through en route to the target. Per the
migration plan in §1.10.2, it is filled and logged in step 2,
read advisorily in step 3, promoted to authoritative for
compositional intents in step 4. Until step 3 lands, the field is
inert: prompts may emit it, validators accept it, compilers
ignore it.

## Backward compatibility

- Existing planner requests without `world_state` remain valid.
  The server treats absent `world_state` as empty.
- Existing planner responses without `staging_hops` remain valid.
  The compiler treats absent `staging_hops` as "no advisory."
- The planner request payload at
  [`source/strafer_autonomy/strafer_autonomy/planner/payloads.py`](../../../source/strafer_autonomy/strafer_autonomy/planner/payloads.py)
  carries `robot_state` and `active_mission_summary` as opaque
  `dict | None`. New `world_state` is an additive field; legacy
  fields stay populated for the transition window.

## Where the wire shape lives

The Python definitions land in
[`source/strafer_autonomy/strafer_autonomy/schemas/`](../../../source/strafer_autonomy/strafer_autonomy/schemas/)
when
[`autonomy-stack`](../active/multi-room/autonomy-stack.md) or
[`planner-far-target-staging`](../active/multi-room/planner-far-target-staging.md)
ships first. Whichever brief lands second consumes the shipped
classes — no parallel `PlannerWorldState` variants. The Pydantic
wire types on the HTTP boundary live in
[`planner/payloads.py`](../../../source/strafer_autonomy/strafer_autonomy/planner/payloads.py).
