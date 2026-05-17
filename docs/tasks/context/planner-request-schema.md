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

## Architecture

Per the decision recorded in
[`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c)
and the brief at
[`planner-architecture-alignment`](../active/multi-room/planner-architecture-alignment.md):

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

Lives on the planner request as an optional block. Populated by
the executor (`mission_runner.py`) plus the DGX-side
`SemanticMapManager`. Absent or `None` → planner runs with no
world knowledge (legacy behavior; still valid).

```python
class PlannerWorldState(BaseModel):
    # Pose + costmap (populated Jetson-side from mission_runner)
    robot_pose_map: Pose2D
    global_costmap_extent: Rectangle | None  # min_x, min_y, max_x, max_y
    last_grounding: GroundingSummary | None  # depth, stability, age_s

    # Room-level (populated DGX-side from SemanticMapManager;
    # observation-derived only — see the sim-to-real boundary
    # in autonomy-stack)
    current_room: str | None
    known_rooms: list[RoomSummary]
    connectivity: list[tuple[str, str]]
```

Component shapes:

| Field | Shape | Source | Notes |
|---|---|---|---|
| `robot_pose_map` | `Pose2D(x, y, yaw)` | mission_runner | `map` frame. |
| `global_costmap_extent` | `Rectangle(min_x, min_y, max_x, max_y)` or `None` | mission_runner | `None` if Nav2 not yet running. |
| `last_grounding` | `GroundingSummary(depth_m, stability, age_s)` or `None` | mission_runner | Most recent `scan_for_target` result; lets the planner reason about staleness. |
| `current_room` | `str` or `None` | SemanticMapManager.current_room() | `None` at cold-start. |
| `known_rooms` | `list[RoomSummary(label, member_node_ids, centroid_xy, confidence)]` | SemanticMapManager.known_rooms() | Empty at cold-start. |
| `connectivity` | `list[(room_a, room_b)]` | SemanticMapManager.connectivity() | Pessimistic — absence ≠ unreachable. |

**Population contract.** Both sides must populate **all** fields they
own. A partial `world_state` is a footgun: the LLM treats absent
fields as "unknown" but the compiler treats them as
"caller-asserted absent." When a side cannot yet populate a field
(e.g., Nav2 not running), send `None` for that field rather than
omitting it.

**No `scene_metadata.json` reads.** The runtime planner backend
must never read Infinigen sim-side metadata. The
sim-to-real-boundary subsection of
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
