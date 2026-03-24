# Strafer Autonomy Interfaces

This document defines the first concrete callable interfaces between:
- `strafer_autonomy`
- `strafer_ros`
- `strafer_vlm`

The goal is to make the MVP implementable without overcommitting to the final deployment topology.

## Scope

This is the first interface pass for the autonomy-layer MVP.

It covers:
- shared schemas owned by `strafer_autonomy`
- how the Jetson-resident executor calls robot skills provided by `strafer_ros`
- how the Jetson-resident executor calls the remote planner service
- how the Jetson-resident executor calls grounding provided by `strafer_vlm`
- which parts are synchronous, asynchronous, local, or transport-specific

It does not yet define:
- cloud deployment contracts beyond the first LAN-friendly service shape
- web/mobile APIs
- full advanced orchestration behaviors like plan repair and memory

## Boundary Decisions

1. `strafer_autonomy` owns mission plans, skill calls, and execution state.
2. The first MVP executor runs on the Jetson.
3. `strafer_ros` owns sensing, depth/TF projection, navigation execution, and safety-critical robot-side control.
4. `strafer_vlm` owns semantic grounding only.
5. The planner never emits raw ROS messages.
6. The executor never calls arbitrary model prompts directly. It only invokes typed clients.
7. Planner and VLM are remote services from the executor's point of view in the chosen MVP.

## Source Of Truth

The source of truth for autonomy-layer types should live in:

```text
source/strafer_autonomy/strafer_autonomy/schemas/
```

Recommended initial schemas:
- `PlannerRequest`
- `MissionIntent`
- `MissionPlan`
- `SkillCall`
- `SkillResult`
- `SceneObservation`
- `GroundingRequest`
- `GroundingResult`
- `GoalPoseCandidate`

`strafer_ros` may later mirror some of these with ROS IDL in `source/strafer_ros/strafer_msgs`, but the initial planning contract should be owned in Python by `strafer_autonomy`.

## First End-To-End Call Path

```text
User command
  -> Jetson executor receives command
  -> planner_client.plan_mission() over LAN
  -> MissionPlan(steps=[SkillCall, ...])
  -> ros_client.capture_scene_observation() locally on Jetson
  -> vlm_client.locate_semantic_target() over LAN
  -> ros_client.project_detection_to_goal_pose() locally on Jetson
  -> ros_client.navigate_to_pose() locally on Jetson
  -> executor monitors result, cancel, retry, and timeout
```

## Shared Schemas

These are the first concrete data structures to define in `strafer_autonomy`.

### `PlannerRequest`

Purpose:
- request sent from the executor to the remote planner service

Fields:
- `request_id: str`
- `raw_command: str`
- `robot_state: dict | None`
- `active_mission_summary: dict | None`
- `available_skills: list[str]`

### `MissionIntent`

Purpose:
- planner output after interpreting user text

Fields:
- `intent_type: str`
- `target_label: str | None`
- `orientation_mode: str | None`
- `wait_mode: str | None`
- `raw_command: str`
- `requires_grounding: bool`

Example:

```json
{
  "intent_type": "wait_by_target",
  "target_label": "door",
  "orientation_mode": "face_away",
  "wait_mode": "until_next_command",
  "raw_command": "wait by the door for me",
  "requires_grounding": true
}
```

### `SkillCall`

Purpose:
- one typed executable step

Fields:
- `skill: str`
- `args: dict`
- `step_id: str`
- `timeout_s: float | None`
- `retry_limit: int`

Example:

```json
{
  "skill": "locate_semantic_target",
  "args": {"label": "door"},
  "step_id": "step_02",
  "timeout_s": 8.0,
  "retry_limit": 1
}
```

### `MissionPlan`

Purpose:
- ordered sequence of `SkillCall`s

Fields:
- `mission_id: str`
- `mission_type: str`
- `steps: list[SkillCall]`
- `raw_command: str`
- `created_at: float`

### `SkillResult`

Purpose:
- normalized executor-facing result for any skill

Fields:
- `step_id: str`
- `skill: str`
- `status: str`  (`succeeded`, `failed`, `canceled`, `timeout`)
- `outputs: dict`
- `error_code: str | None`
- `message: str | None`
- `started_at: float`
- `finished_at: float`

### `SceneObservation`

Purpose:
- synchronized robot observation used for grounding and goal projection

Fields:
- `observation_id: str`
- `stamp_sec: float`
- `color_image_bgr: Any`
- `aligned_depth_m: Any`
- `camera_frame: str`
- `camera_info: dict`
- `robot_pose_map: dict | None`
- `tf_snapshot_ready: bool`

Important note:
- this is a Jetson-local runtime object, not something to ship over the network as-is

### `GroundingRequest`

Fields:
- `request_id: str`
- `prompt: str`
- `image_rgb_u8: Any`
- `image_stamp_sec: float`
- `max_image_side: int`
- `return_debug_overlay: bool`

Important note:
- this is the executor-side logical request type
- over the network, the image payload should be compressed or serialized, for example JPEG bytes or base64

### `GroundingResult`

Fields:
- `request_id: str`
- `found: bool`
- `bbox_2d: list[int] | None`
- `label: str | None`
- `confidence: float | None`
- `raw_output: str | None`
- `latency_s: float`
- `debug_artifact_path: str | None`

### `GoalPoseCandidate`

Fields:
- `request_id: str`
- `found: bool`
- `goal_frame: str`
- `goal_pose: dict | None`
- `target_pose: dict | None`
- `standoff_m: float`
- `depth_valid: bool`
- `quality_flags: list[str]`
- `message: str | None`

## Executor To Planner Service

The chosen MVP requires an explicit planner client because the executor runs on the Jetson and the planner runs on the workstation.

### Python-side interface

```python
class PlannerClient(Protocol):
    def plan_mission(self, request: PlannerRequest) -> MissionPlan:
        ...
```

### First transport

Chosen first mode:
- LAN HTTP request-response

### `POST /plan`

Request body example:

```json
{
  "request_id": "plan_001",
  "raw_command": "wait by the door for me",
  "robot_state": null,
  "active_mission_summary": null,
  "available_skills": [
    "capture_scene_observation",
    "locate_semantic_target",
    "project_detection_to_goal_pose",
    "navigate_to_pose",
    "wait",
    "cancel_mission",
    "report_status"
  ]
}
```

Response body example:

```json
{
  "mission_id": "mission_001",
  "mission_type": "wait_by_target",
  "raw_command": "wait by the door for me",
  "created_at": 1710000000.0,
  "steps": [
    {"step_id": "step_01", "skill": "capture_scene_observation", "args": {}, "timeout_s": 1.0, "retry_limit": 0},
    {"step_id": "step_02", "skill": "locate_semantic_target", "args": {"label": "door"}, "timeout_s": 8.0, "retry_limit": 1},
    {"step_id": "step_03", "skill": "project_detection_to_goal_pose", "args": {"standoff_m": 0.7}, "timeout_s": 2.0, "retry_limit": 0},
    {"step_id": "step_04", "skill": "navigate_to_pose", "args": {"goal_source": "projected_target"}, "timeout_s": 90.0, "retry_limit": 0},
    {"step_id": "step_05", "skill": "wait", "args": {"mode": "until_next_command"}, "timeout_s": null, "retry_limit": 0}
  ]
}
```

## Executor To `strafer_ros`

The first concrete robot-side interface should be a Python `ros_client` inside the Jetson executor.

That client uses a mix of:
- ROS topic subscriptions for cached observation and state
- ROS service calls for bounded request-response work
- ROS actions for long-running execution

This is the right split because images and odometry are streaming data, while navigation is asynchronous.

## ROS Topic Inputs Used By `ros_client`

These are runtime inputs that the `ros_client` caches locally on the Jetson.

| Topic | Type | Purpose |
|------|------|---------|
| `/d555/color/image_sync` | `sensor_msgs/Image` | latest RGB frame for grounding |
| `/d555/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | aligned depth for 2D-to-3D projection |
| `/d555/color/camera_info_sync` | `sensor_msgs/CameraInfo` | intrinsics for projection |
| `/odom` | `nav_msgs/Odometry` | robot pose/velocity fallback |
| `/tf` and `/tf_static` | standard TF streams | camera-to-map transforms |

The callable interface exposed by `ros_client` should hide these subscriptions.

### `ros_client.capture_scene_observation()`

Purpose:
- return the newest synchronized observation bundle from cached ROS topics

Inputs:
- none for first implementation

Returns:
- `SceneObservation`

Behavior:
- fails if no recent color image, no recent depth image, or no camera info is available
- enforces recency threshold, for example `<= 0.5 s`
- does not copy large data more than necessary

Why this is not a ROS service:
- returning high-bandwidth images through a service is the wrong transport
- the executor already runs on the robot and can cache the local ROS streams directly

### `ros_client.get_robot_state()`

Purpose:
- return the latest robot pose and high-level runtime status

Inputs:
- none

Returns:
- dict with pose, velocity, nav state, timestamp

Implementation note:
- this can initially be built from cached `/odom` and local action state
- a dedicated ROS service can be added later only if needed

## ROS Service Interface

The first custom robot-side request-response interface should be:

### `strafer_msgs/srv/ProjectDetectionToGoalPose.srv`

Purpose:
- convert a 2D detection into a reachable goal pose using robot-local depth, intrinsics, and TF

Request fields:
- `string request_id`
- `builtin_interfaces/Time image_stamp`
- `int32[4] bbox_2d`
- `float32 standoff_m`
- `string target_label`

Response fields:
- `bool success`
- `string message`
- `geometry_msgs/PoseStamped goal_pose`
- `geometry_msgs/PoseStamped target_pose`
- `bool depth_valid`
- `string[] quality_flags`

Why this belongs in `strafer_ros`:
- it depends on aligned depth, camera calibration, TF, and robot pose
- these are robot-runtime concerns, not VLM concerns

## ROS Action Interfaces

### `ros_client.navigate_to_pose(request)`

Transport:
- wrap `nav2_msgs/action/NavigateToPose`

Inputs:
- `goal_pose`
- `behavior_tree: str | None`
- `timeout_s: float | None`

Returns:
- `SkillResult`

Feedback used by executor:
- navigation progress
- distance remaining if available
- cancel/timeout state

Important note:
- do not create a duplicate custom navigate action for the MVP if Nav2 already covers the behavior

### `ros_client.cancel_active_navigation()`

Transport:
- cancel the active `NavigateToPose` action goal

Purpose:
- used by executor for mission cancel and timeout handling

### `ros_client.orient_relative_to_target(request)`

MVP status:
- optional for first cut

Recommended transport when implemented:
- `strafer_msgs/action/OrientRelativeToTarget.action`

Goal fields:
- `geometry_msgs/PoseStamped target_pose`
- `string mode`  (`face_target`, `face_away`, `face_left`, `face_right`)
- `float32 yaw_offset_rad`
- `float32 tolerance_rad`

Result fields:
- `bool success`
- `float32 final_yaw_rad`
- `string message`

Reason to keep custom:
- this behavior is Strafer-specific and not equivalent to generic Nav2 navigation

## Executor To `strafer_vlm`

The first concrete VLM interface should be a typed client inside the Jetson executor.

### Python-side interface

```python
class GroundingClient(Protocol):
    def locate_semantic_target(self, request: GroundingRequest) -> GroundingResult:
        ...
```

### First transport

Chosen first mode:
- LAN HTTP request-response to a workstation-hosted VLM service

Important note:
- because the executor is on the Jetson and the VLM is on the workstation, this runtime path is not an in-process Python call
- in-process calls remain valid only inside the workstation VLM service implementation itself

### `vlm_client.locate_semantic_target(request)`

Inputs:
- `GroundingRequest`

Returns:
- `GroundingResult`

Error conditions:
- model unavailable
- parse failure
- image preprocessing failure
- timeout
- network error

Executor expectations:
- `found=false` is a valid result, not a transport error
- malformed output is a failed skill result
- network failure is a failed skill result with a distinct error code

## VLM HTTP Interface

### `POST /ground`

Request body:

```json
{
  "request_id": "req_123",
  "prompt": "Locate: the door",
  "image_jpeg_b64": "...",
  "image_stamp_sec": 1710000000.25,
  "max_image_side": 1024,
  "return_debug_overlay": false
}
```

Response body:

```json
{
  "request_id": "req_123",
  "found": true,
  "bbox_2d": [230, 112, 510, 620],
  "label": "door",
  "confidence": 0.92,
  "raw_output": "{...}",
  "latency_s": 2.84,
  "debug_artifact_path": null
}
```

Additional endpoint:
- `GET /health`

This lets the Jetson executor keep the same callable method while swapping the workstation service for a cloud endpoint later.

## Skill Mapping For The MVP

| Skill name | Called by | Backed by | Transport |
|-----------|-----------|-----------|-----------|
| `capture_scene_observation` | Jetson executor | `ros_client` observation cache | local ROS subscriptions |
| `locate_semantic_target` | Jetson executor | `vlm_client` | LAN HTTP first, cloud HTTP later |
| `project_detection_to_goal_pose` | Jetson executor | `strafer_ros` | local ROS service |
| `navigate_to_pose` | Jetson executor | Nav2 via `ros_client` | local ROS action |
| `orient_relative_to_target` | Jetson executor | future Strafer-specific robot behavior | local ROS action |
| `wait` | Jetson executor | local timer / mission state | local only |
| `cancel_mission` | Jetson executor | local mission cancel + ROS action cancel | local + ROS action cancel |
| `report_status` | Jetson executor | mission state + robot state | local only |

## First Implementation Recommendation

Implement in this order:

1. `strafer_autonomy` schemas
   - `PlannerRequest`, `MissionPlan`, `SkillCall`, `SkillResult`, `SceneObservation`, `GroundingRequest`, `GroundingResult`, `GoalPoseCandidate`

2. `strafer_autonomy.executor.mission_runner`
   - runs on the Jetson
   - owns timeout, retry, and cancel behavior

3. `strafer_autonomy.clients.ros_client`
   - subscribe to synchronized RGB, depth, camera info, odom, TF locally on the Jetson
   - expose `capture_scene_observation()` and `get_robot_state()`
   - wrap `NavigateToPose`

4. `strafer_msgs/srv/ProjectDetectionToGoalPose.srv`
   - robot-side projection service

5. `strafer_autonomy.clients.planner_client`
   - LAN HTTP implementation first

6. `strafer_autonomy.clients.vlm_client`
   - LAN HTTP implementation first
   - preserve adapter boundary so cloud HTTP can be added later

## Interfaces To Defer

These should not block the first MVP:
- rich mission history storage
- full clarification loop API
- cloud-specific auth protocols
- multi-robot or fleet interfaces
- advanced world memory schemas
- generalized skill discovery protocol

## Immediate Follow-On Docs

After this interface document, the next clean documents are:
1. `STRAFER_AUTONOMY_SCHEMAS.md`
2. `STRAFER_AUTONOMY_PLANNER_PROMPT.md`
3. `STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md`
