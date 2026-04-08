# Strafer Autonomy — Next Phase

This document captures the next round of improvements to the autonomy system.
It is written from the working baseline established in `phase_13`:

- Jetson executor calling DGX Spark planner (Qwen3-4B, port 8200) and VLM
  (Qwen2.5-VL-3B, port 8100) over LAN HTTP
- End-to-end mission execution tested on real hardware: user command through
  CLI, LLM planning, VLM grounding, goal projection, Nav2 navigation
- 10 implemented skills, deterministic plan compiler, mission runner with
  cancel/timeout/retry

The document is organized into six areas:
0. Safety
1. Semantic Spatial Map
2. Optimizations
3. New features
4. Deployment (Databricks integration path)
5. Synthetic data and sim-to-real (strafer_lab)

---

## 0. Safety

### 0.1 Navigation verification via semantic map

**Problem.** The robot currently trusts the projected goal pose and hands it to
Nav2 with a 90-second timeout. If the VLM returns a spurious detection (e.g.,
misidentifies a wall texture as the target), the robot drives to an incorrect
location, realizes nothing is wrong, and reports success. Simple heuristics
like bbox size or center offset are fragile — the operator may intentionally
position the robot far from the target, making size-based thresholds arbitrary.

**Proposal: semantic map verification.** After `navigate_to_pose` completes,
verify arrival by comparing the robot's current surroundings against the
semantic spatial map (Section 1):

1. Capture a new RGB frame at the arrival pose.
2. Compute a CLIP embedding of the frame.
3. Query ChromaDB for the nearest semantic map node to the robot's current pose.
4. Compare the CLIP embedding of the arrival frame against the stored
   embedding(s) at and near the target node.
5. If the visual similarity is above a configurable threshold **and** the
   spatial distance is within tolerance, the step succeeds.
6. If the environment doesn't match expectations (low embedding similarity,
   or the semantic map says this region looks nothing like where the target
   should be), fail with `arrival_verification_failed`.

**Why this is more robust than bbox-area checks:**
- Works regardless of the robot's intended standoff distance from the target.
- Catches cases where the VLM hallucinated a detection — the robot arrives at
  a location that visually doesn't match where the target was previously seen.
- Leverages persistent environmental memory rather than a single-frame snapshot.
- Graceful degradation: if the semantic map has no data for a region (first
  visit), skip verification and log a warning.

**Implementation sketch.** The `verify_arrival` skill does not exist yet. It
will be registered in `DEFAULT_AVAILABLE_SKILLS` and dispatched from
`MissionRunner._execute_step`. The `SkillCall` emitted by the planner uses
`runtime.latest_grounding.label` (not the planner-side `MissionIntent`, which
is not accessible at skill dispatch time):

```python
SkillCall(
    step_id="step_04",
    skill="verify_arrival",
    args={
        "target_label": runtime.latest_grounding.label,  # set by planner compiler
        "max_spatial_distance_m": 2.0,
        "min_visual_similarity": 0.75,        # CLIP cosine similarity threshold
        "fallback_on_empty_map": "pass",      # "pass" or "fail" when no map data
    },
    timeout_s=10.0,
    retry_limit=1,
)
```

The handler is a `MissionRunner` method with this signature:

```python
def _verify_arrival(
    self, runtime: _MissionRuntime, step: SkillCall
) -> SkillResult:
    started_at = time.time()

    # 1. Resolve the label we are verifying against.
    target_label = str(step.args.get("target_label", "")).strip()
    if not target_label and runtime.latest_grounding is not None:
        target_label = runtime.latest_grounding.label or ""
    if not target_label:
        return self._failed_result(
            step, "verify_arrival has no target_label.", "invalid_args", started_at,
        )

    # 2. Capture a fresh observation at the arrival pose.
    observation: SceneObservation = self._ros_client.capture_scene_observation()
    arrival_image_rgb = self._bgr_to_rgb(observation.color_image_bgr)

    # 3. Get the robot's current pose.
    robot_state = self._ros_client.get_robot_state()
    robot_pose: dict | None = robot_state["pose"]  # x/y/z/qx/qy/qz/qw or None
    if robot_pose is None:
        return self._failed_result(
            step, "No robot pose available for arrival verification.",
            "no_pose", started_at,
        )

    # 4. Compute CLIP embedding and query semantic map (Section 1 APIs).
    arrival_embedding = self._semantic_map.clip_encoder.encode_image(arrival_image_rgb)
    nearest_node = self._semantic_map.query_nearest(
        x=robot_pose["x"], y=robot_pose["y"],
        max_distance_m=float(step.args.get("max_spatial_distance_m", 2.0)),
    )

    # 5. Compare embeddings.
    if nearest_node is None:
        fallback = str(step.args.get("fallback_on_empty_map", "pass"))
        if fallback == "pass":
            _logger.warning("No semantic map data near (%.2f, %.2f); skipping verification.",
                            robot_pose["x"], robot_pose["y"])
            return SkillResult(
                step_id=step.step_id, skill=step.skill, status="succeeded",
                outputs={"verified": False, "reason": "no_map_data"},
                message="Arrival verification skipped (no map data).",
                started_at=started_at, finished_at=time.time(),
            )
        return self._failed_result(
            step, "No semantic map data and fallback_on_empty_map='fail'.",
            "arrival_verification_failed", started_at,
        )

    similarity = float(np.dot(arrival_embedding, nearest_node.embedding))
    min_sim = float(step.args.get("min_visual_similarity", 0.75))
    if similarity < min_sim:
        return self._failed_result(
            step,
            f"Visual similarity {similarity:.3f} < threshold {min_sim:.3f}.",
            "arrival_verification_failed", started_at,
        )

    return SkillResult(
        step_id=step.step_id, skill=step.skill, status="succeeded",
        outputs={"verified": True, "similarity": similarity},
        message=f"Arrival verified (similarity {similarity:.3f}).",
        started_at=started_at, finished_at=time.time(),
    )
```

**Where this runs.** CLIP embedding is computed locally on the Jetson
(OpenCLIP ViT-B/32, see Section 1.8). ChromaDB and NetworkX queries are local.
The VLM is not needed for verification — this is a pure embedding similarity
check, keeping the safety decision entirely on the robot.

### 0.2 Costmap collision pre-check

**Problem.** Nav2 sometimes rejects goals immediately if the projected pose
lands in an occupied or unknown costmap cell. The executor sees `goal_rejected`
but has no recovery path.

**Proposal.** Before sending a goal to Nav2, query the global costmap at the
projected pose coordinates. If the cell is occupied or unknown:
- Offset the goal along the standoff vector (move it closer to the robot)
- Or re-project with a larger standoff distance
- Log a warning so the operator knows the original projection was adjusted

This is a pre-check method on `JetsonRosClient`, not a separate skill. It
subscribes to the Nav2 global costmap and converts world coordinates to grid
cell indices.

**Implementation sketch.**

```python
# In JetsonRosClient.__init__, add a costmap subscription:
from nav_msgs.msg import OccupancyGrid

self._latest_costmap: OccupancyGrid | None = None
self._node.create_subscription(
    OccupancyGrid,
    "/global_costmap/costmap",
    self._on_costmap,
    10,
)

def _on_costmap(self, msg: OccupancyGrid) -> None:
    with self._cache_lock:
        self._latest_costmap = msg

def check_costmap_at_pose(self, x: float, y: float) -> str:
    """Return 'free', 'occupied', 'unknown', or 'no_costmap'.

    Checks the Nav2 global costmap cell at the given world (x, y).
    """
    with self._cache_lock:
        costmap = self._latest_costmap

    if costmap is None:
        return "no_costmap"

    info = costmap.info
    # Convert world coordinates to grid cell indices.
    col = int((x - info.origin.position.x) / info.resolution)
    row = int((y - info.origin.position.y) / info.resolution)

    if col < 0 or col >= info.width or row < 0 or row >= info.height:
        return "unknown"  # outside costmap bounds

    cell_value = costmap.data[row * info.width + col]

    if cell_value == -1:
        return "unknown"
    if cell_value >= 65:     # Nav2 INSCRIBED_INFLATED_OBSTACLE threshold
        return "occupied"
    return "free"
```

The pre-check is called from `JetsonRosClient.navigate_to_pose` before
constructing the `NavigateToPose.Goal`. If the result is `"occupied"` or
`"unknown"`, the caller can adjust the standoff or fail early with a
descriptive error instead of waiting for Nav2 to reject the goal.

### 0.3 VLM confidence thresholding

**Problem.** The VLM returns a confidence score with each grounding result, but
the executor currently ignores it. A low-confidence detection can still drive
navigation.

**Proposal.** Add a configurable `min_grounding_confidence` threshold (default
0.5) to `MissionRunnerConfig`. If the VLM returns a detection below this
threshold, treat it as `not_found` and continue the scan rotation. Log the
rejected detection for debugging.

The confidence value is available as `GroundingResult.confidence: float | None`.
The check should be inserted in `MissionRunner._scan_for_target`, immediately
after `self._grounding_client.locate_semantic_target(request)` returns and
before checking `grounding.found`:

```python
# Inside _scan_for_target, in the grounding attempt block (after line 519
# in mission_runner.py), insert before the `if grounding.found` check:

grounding = self._grounding_client.locate_semantic_target(
    GroundingRequest(
        request_id=f"{runtime.mission_id}:{step.step_id}:scan_{i}",
        prompt=prompt,
        image_rgb_u8=self._bgr_to_rgb(observation.color_image_bgr),
        image_stamp_sec=observation.stamp_sec,
        max_image_side=max_image_side,
    )
)
with self._lock:
    runtime.latest_grounding = grounding

# --- NEW: confidence gate ---
min_conf = self._config.min_grounding_confidence  # default 0.5
if (
    grounding.found
    and grounding.confidence is not None
    and grounding.confidence < min_conf
):
    _logger.info(
        "Scan heading %d: rejected low-confidence detection "
        "(%.3f < %.3f) for '%s'.",
        i, grounding.confidence, min_conf, label,
    )
    # Treat as not found — fall through to next heading.
    pass
elif grounding.found and grounding.bbox_2d is not None:
    return SkillResult(...)
```

The same confidence check should also be added in `_locate_semantic_target`,
after line 453 in `mission_runner.py`, before returning the successful result.

### 0.4 Odometry drift watchdog

**Problem.** If RTAB-Map loses tracking (e.g., featureless corridor), the
`map -> odom` TF becomes stale and Nav2 navigates using drifting wheel odometry
alone.

**Proposal.** Monitor the `map -> odom` TF age during navigation. If it exceeds
a threshold (default 5 seconds without update), pause navigation and attempt
recovery:
- Rotate slowly to give RTAB-Map new visual features
- If TF age recovers, resume navigation
- If not, fail the step with `slam_tracking_lost`

**Implementation sketch.** This check runs inside `JetsonRosClient` using a
`tf2_ros.Buffer` to look up the `map -> odom` transform and compare its
timestamp to the current clock:

```python
import tf2_ros
import rclpy.time

# In JetsonRosClient.__init__:
self._tf_buffer = tf2_ros.Buffer()
self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

def check_slam_tracking(self, threshold_s: float = 5.0) -> tuple[bool, float]:
    """Check if the map->odom TF is fresh.

    Returns (is_fresh, age_s). If the transform is not available at all,
    returns (False, float('inf')).
    """
    try:
        transform = self._tf_buffer.lookup_transform(
            "map", "odom", rclpy.time.Time()  # latest available
        )
        now = self._node.get_clock().now()
        age_s = (now - transform.header.stamp).nanoseconds / 1e9
        return (age_s <= threshold_s, age_s)
    except tf2_ros.LookupException:
        return (False, float("inf"))
    except tf2_ros.ExtrapolationException:
        return (False, float("inf"))
```

During active navigation, the executor can poll `check_slam_tracking()` on a
timer (e.g., every 1 second). If `is_fresh` is `False`, cancel the active Nav2
goal, attempt a slow rotation to provide RTAB-Map with new visual features, and
re-check. If the TF age does not recover after the rotation, fail the step
with `error_code="slam_tracking_lost"`.

### 0.5 Emergency stop on excessive velocity

**Problem.** A runaway controller (Nav2 MPPI or a future RL backend) could
command velocities beyond the robot's safe operating range.

**Proposal.** The driver node `RoboClawNode` (in
`strafer_driver/strafer_driver/roboclaw_node.py`) already has a `cmd_vel`
watchdog that stops motors if no `/strafer/cmd_vel` arrives within 500 ms.
Add a complementary velocity magnitude check in the same node's control loop.

The odom topic is `/strafer/odom` (type `nav_msgs/msg/Odometry`). The driver
publishes it at 50 Hz with `twist.twist.linear.{x,y}` in the body frame. The
check compares the measured planar speed against `MAX_LINEAR_VEL * 1.2` (a 20%
margin above the kinematic maximum of ~1.568 m/s, defined in
`strafer_shared.constants`).

**Implementation sketch.** In `RoboClawNode._control_loop`, after computing
`vx_body` and `vy_body` from wheel encoders and before publishing the
`Odometry` message:

```python
from math import sqrt
from strafer_shared.constants import MAX_LINEAR_VEL

VEL_LIMIT = MAX_LINEAR_VEL * 1.2  # ~1.882 m/s

# Inside RoboClawNode._control_loop, after computing vx_body, vy_body:
speed = sqrt(vx_body**2 + vy_body**2)
if speed > VEL_LIMIT:
    self.get_logger().error(
        "velocity_limit_exceeded: measured %.3f m/s > limit %.3f m/s "
        "-- zeroing motors.",
        speed, VEL_LIMIT,
    )
    self._interface.drive(0, 0, 0, 0)  # immediate motor stop
    # Publish a DiagnosticStatus for external monitoring.
    return  # skip publishing this odom cycle
```

This guards against runaway controller output. The check lives in the driver
node (closest to the hardware) so it cannot be bypassed by higher-level
software faults.

---

## 1. Semantic Spatial Map

### 1.1 Motivation

The current autonomy stack is memoryless. Every mission starts with a blank
slate -- the robot has no knowledge of where it has been, what it has seen, or
where objects were previously located. This forces a full 360-degree VLM scan on
every `go_to_target` command, even for targets the robot saw 30 seconds ago.

A persistent semantic map solves multiple problems simultaneously:

- **Arrival verification** (Section 0.1): compare arrival scene against stored
  observations instead of relying on fragile bbox heuristics.
- **Target re-acquisition**: skip the scan rotation if the semantic map already
  knows where the target was last seen.
- **Spatial reasoning**: answer queries like "what's near the door?" or "which
  room has the most chairs?" by querying the graph.
- **Operator situational awareness**: provide rich environment descriptions
  without a live video feed.
- **Mission continuity**: knowledge persists across missions and power cycles.

### 1.2 Architecture overview

The semantic map is a topological graph with vector-indexed observations:

```text
+--------------------------------------------------------------+
|  Jetson (robot-local, survives network drops)                 |
|                                                               |
|  +--------------+   +---------------+   +------------------+ |
|  | NetworkX     |   | ChromaDB      |   | OpenCLIP ViT-B/32| |
|  | spatial      |   | PersistentCl. |   | (ONNX, unified   | |
|  | graph        |   | single 512-d  |   |  image + text    | |
|  | (JSON)       |   | collection    |   |  encoder)        | |
|  +------+-------+   +------+--------+   +------+-----------+ |
|         |                  |                    |             |
|         +--------+---------+                    |             |
|                  |                              |             |
|         +--------v----------+    +--------------v-----------+ |
|         | SemanticMapManager |<--| ObservationPipeline       | |
|         | query / update     |   | + BackgroundMapper        | |
|         +--------^-----------+   | capture->embed->store     | |
|                  |               +---------------------------+ |
|         +--------+---------+                                   |
|         | MissionRunner    |                                   |
|         | (skill dispatch) |                                   |
|         +------------------+                                   |
+--------------------------------------------------------------+
                        |
                        | LAN HTTP (VLM: describe, detect_objects)
                        v
              +------------------------+
              | DGX Spark / Cloud      |
              |  VLM: describe,        |
              |       detect_objects    |
              |  Planner: plan         |
              +------------------------+
```

### 1.3 Data model

`Pose2D` is defined as a lightweight dataclass for the semantic graph. The
full `Pose3D` (from `strafer_autonomy.schemas.grounding`) uses quaternion
orientation which is unnecessarily heavy for a 2D topological map. `Pose2D`
can be derived from `SceneObservation.robot_pose_map` (a dict with
`x/y/z/qx/qy/qz/qw`) by extracting `x`, `y`, and computing yaw from the
quaternion.

```python
import math
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Pose2D:
    """2D pose in the map frame, used for semantic graph nodes."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # radians, CCW from +X

    @staticmethod
    def from_pose_map_dict(d: dict) -> "Pose2D":
        """Convert a robot_pose_map dict (x/y/z/qx/qy/qz/qw) to Pose2D."""
        qz = d.get("qz", 0.0)
        qw = d.get("qw", 1.0)
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        return Pose2D(x=d.get("x", 0.0), y=d.get("y", 0.0), yaw=yaw)
```

Each **detected object** is stored with its projected real-world position,
reinforced over repeated sightings:

```python
import numpy as np

@dataclass
class DetectedObjectEntry:
    label: str                          # e.g., "table", "door"
    position_mean: np.ndarray           # [x, y, z] in map frame — Kalman filter mean
    position_cov: np.ndarray            # 3x3 covariance matrix — spatial uncertainty
    bbox_2d: tuple[int, int, int, int]  # [x1, y1, x2, y2] pixels (latest observation)
    confidence: float                   # VLM confidence (max across sightings)
    observation_count: int = 1          # incremented on re-sighting
    first_seen: float = 0.0            # epoch seconds
    last_seen: float = 0.0             # epoch seconds
```

The position is modeled as a **3D Gaussian** (mean + covariance) rather than
a point. The depth projection from `project_detection_to_goal_pose` gives a
full `(x, y, z)` in the map frame — storing all three dimensions preserves
the ability to distinguish objects at different heights (a cup on a table vs.
a cup on the floor, a light switch vs. a door handle). Depth-projecting the
same physical object from different viewpoints yields different 3D positions
because the projection hits different surfaces of the object. The 3x3
covariance naturally captures this spread and narrows as more observations
accumulate.

**Initial covariance** is set based on depth measurement uncertainty. The
RealSense D555 stereo depth error grows roughly quadratically with range,
and the projection error is directional — elongated along the camera's
viewing ray. The vertical (z) uncertainty is smaller than the depth axis
because stereo matching is more constrained vertically.

For an object detected at depth `d` meters with camera pose
`(cx, cy, cz, cyaw, cpitch)`:

```python
def initial_object_covariance(
    depth_m: float, camera_yaw: float, camera_pitch: float = 0.0,
) -> np.ndarray:
    """Compute initial 3x3 covariance for a depth-projected object.

    The uncertainty ellipsoid is elongated along the camera's viewing
    ray because stereo depth error (proportional to d^2) dominates.
    """
    # Depth-dependent uncertainty (RealSense D555 error model)
    sigma_along = 0.02 * depth_m ** 2     # ~2% of d^2 along viewing ray
    sigma_lateral = 0.05 + 0.01 * depth_m # ~5cm base + 1cm/m lateral (horizontal)
    sigma_vertical = 0.03 + 0.01 * depth_m # slightly tighter vertical

    # Construct diagonal covariance in camera frame:
    # axis 0 = along viewing ray (depth), axis 1 = horizontal, axis 2 = vertical
    D = np.diag([sigma_along ** 2, sigma_lateral ** 2, sigma_vertical ** 2])

    # Rotate into map frame using camera yaw and pitch
    cy, sy = math.cos(camera_yaw), math.sin(camera_yaw)
    cp, sp = math.cos(camera_pitch), math.sin(camera_pitch)

    # R = Rz(yaw) @ Ry(pitch) — rotates camera-frame axes into map frame
    R = np.array([
        [cy * cp, -sy, cy * sp],
        [sy * cp,  cy, sy * sp],
        [-sp,       0, cp     ],
    ])
    return R @ D @ R.T
```

Each **node** in the semantic graph represents a distinct observation point:

```python
@dataclass
class SemanticNode:
    node_id: str                        # e.g., "obs_0042"
    pose: Pose2D                        # (x, y, yaw) in map frame
    timestamp: float                    # epoch seconds
    clip_embedding_id: str              # ChromaDB document ID (CLIP image embedding)
    text_description: str | None        # VLM-generated scene description (metadata only)
    detected_objects: list[DetectedObjectEntry]  # positioned objects with reinforcement
    metadata: dict                      # battery, nav state, mission context
```

Note: `text_description` is stored as plain metadata for operator readback
and explainability. It is NOT separately embedded — text queries go through
CLIP's text encoder into the same 512-dim space as image embeddings (see
Section 1.4).

Each **edge** represents traversability between nodes:

```python
@dataclass
class SemanticEdge:
    source: str                         # node_id
    target: str                         # node_id
    distance_m: float                   # Euclidean distance
    traversal_verified: bool            # True if robot actually drove this
    last_traversed: float | None        # epoch timestamp
```

Edges are created when the robot successfully navigates between two nodes.
Unverified edges can be inferred from spatial proximity (configurable
threshold, e.g., 2m) for approximate graph connectivity.

### 1.4 Unified CLIP embedding strategy

The map uses a **single embedding model** — OpenCLIP ViT-B/32 — for all
vector operations. CLIP's image encoder and text encoder produce vectors in
the **same 512-dimensional space**, so image-to-image, text-to-image, and
image-to-text similarity are all computed with one model and one ChromaDB
collection.

| Query direction | Encoder | Example |
|-----------------|---------|---------|
| Image → Image | CLIP image encoder | Arrival verification: "does this frame match what I saw here before?" |
| Text → Image | CLIP text encoder | "Where did I see a red cup?" → searches stored image embeddings |
| Image → Text (conceptual) | CLIP image encoder vs. stored image embeddings | "Have I been here before?" → visual place recognition |

**Why a single CLIP space instead of CLIP + MiniLM:**

The previous design stored CLIP image embeddings (512-dim) and MiniLM text
embeddings (384-dim) in separate spaces. This meant text queries ("find the
red cup") could only search against VLM-generated text descriptions — not
against the visual content directly. This is lossy because:
- The VLM may not mention all visible objects.
- Spatial relationships degrade across the text mediation.
- Two entirely separate vector collections must be maintained and queried.

With unified CLIP, a text query like "red cup" goes through CLIP's text
encoder and searches directly against stored image embeddings. The model
was trained on 400M+ image-text pairs specifically for this cross-modal
alignment. This eliminates MiniLM as a runtime dependency on the Jetson
(one fewer ONNX model, ~40 MB saved).

**Text descriptions are still stored as metadata** on each `SemanticNode`
for:
- Explainability: "The robot remembers seeing a door here because the VLM
  described 'a wooden door with a silver handle at the end of a hallway'."
- Operator readback: `query_environment` can surface these descriptions.
- Object inventory: detected object labels are aggregated across nodes.

They are just not separately embedded.

**On similarity thresholds.** There is no universal "right" threshold.
Cosine similarity values depend on the CLIP model variant, whether
embeddings are L2-normalized, and the visual diversity of the environment.
The practical approach:

- Start with `min_visual_similarity = 0.7` for query-before-scan and
  `0.75` for arrival verification (stricter — safety-critical).
- Log every comparison with its similarity score during real missions.
- Tune based on the observed distribution of true positives vs. false
  positives.
- Different query types may warrant different thresholds — this is exposed
  via `SkillCall.args` rather than a global constant.

All embeddings are stored in a single ChromaDB collection with metadata
distinguishing the source (scan, arrival, describe, background).

### 1.5 Observation pipeline

`SemanticMapManager` is a field on `MissionRunner`, initialized in `__init__`:

```python
# In MissionRunner.__init__:
class MissionRunner(MissionCommandHandler):
    def __init__(
        self,
        *,
        planner_client: PlannerClient,
        grounding_client: GroundingClient,
        ros_client: RosClient,
        config: MissionRunnerConfig | None = None,
        semantic_map: SemanticMapManager | None = None,
    ) -> None:
        self._planner_client = planner_client
        self._grounding_client = grounding_client
        self._ros_client = ros_client
        self._config = config or MissionRunnerConfig()
        self._semantic_map = semantic_map  # None = feature disabled
        # ... existing fields ...
```

When `self._semantic_map` is `None`, all semantic map logic is skipped and the
system behaves exactly as it does today. This makes the feature opt-in and
safe to merge before all components are ready.

Observations are captured during four phases of mission execution:

**0. Continuous background mapping (passive, movement-gated).**

The previous design only captured observations when skills fired. A robot
driving for 10 minutes on a patrol would capture nothing between waypoints.

`BackgroundMapper` is a lightweight thread owned by `SemanticMapManager`
that passively captures observations while the robot is moving:

```python
class BackgroundMapper:
    """Passive observation capture, decoupled from skill execution.

    Captures a CLIP embedding every time the robot moves beyond a spatial
    or angular threshold. VLM descriptions are deferred to idle periods.
    """

    def __init__(
        self,
        ros_client: RosClient,
        semantic_map: "SemanticMapManager",
        min_translation_m: float = 0.5,     # only capture if moved > 0.5m
        min_rotation_deg: float = 30.0,     # or rotated > 30 degrees
        poll_interval_s: float = 2.0,       # check position every 2s
    ) -> None:
        self._ros_client = ros_client
        self._semantic_map = semantic_map
        self._min_translation_m = min_translation_m
        self._min_rotation_rad = math.radians(min_rotation_deg)
        self._poll_interval_s = poll_interval_s
        self._last_capture_pose: Pose2D | None = None
        self._stop_event = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def _should_capture(self, current: Pose2D) -> bool:
        if self._last_capture_pose is None:
            return True
        dx = current.x - self._last_capture_pose.x
        dy = current.y - self._last_capture_pose.y
        dist = math.sqrt(dx * dx + dy * dy)
        dyaw = abs(math.atan2(
            math.sin(current.yaw - self._last_capture_pose.yaw),
            math.cos(current.yaw - self._last_capture_pose.yaw),
        ))
        return dist > self._min_translation_m or dyaw > self._min_rotation_rad

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                robot_state = self._ros_client.get_robot_state()
                pose_dict = robot_state.get("pose")
                if pose_dict is None:
                    self._stop_event.wait(self._poll_interval_s)
                    continue

                current_pose = Pose2D.from_pose_map_dict(pose_dict)
                if not self._should_capture(current_pose):
                    self._stop_event.wait(self._poll_interval_s)
                    continue

                # Capture + CLIP encode (local, no VLM call)
                observation = self._ros_client.capture_scene_observation()
                image_rgb = _bgr_to_rgb(observation.color_image_bgr)
                clip_emb = self._semantic_map.clip_encoder.encode_image(image_rgb)

                self._semantic_map.add_observation(
                    pose=current_pose,
                    timestamp=observation.stamp_sec,
                    clip_embedding=clip_emb,
                    source="background",
                )
                self._last_capture_pose = current_pose

            except Exception:
                _logger.debug("BackgroundMapper capture failed", exc_info=True)

            self._stop_event.wait(self._poll_interval_s)
```

Background-captured nodes have CLIP embeddings but no VLM descriptions or
detected objects. VLM descriptions can be backfilled during idle periods
(no active mission) by iterating over nodes with `text_description is None`
and calling `grounding_client.describe_scene(...)`.

**1. During `_scan_for_target` rotations (automatic).**

The robot already rotates and captures frames at each heading. Currently these
frames are discarded after VLM grounding. The modified loop stores each
observation in the semantic map. This code lives inside
`MissionRunner._scan_for_target` and uses only real API methods:

```python
def _scan_for_target(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
    import math

    started_at = time.time()
    label = str(step.args.get("label", "")).strip()
    if not label:
        return self._failed_result(step, "scan_for_target is missing 'label'.", "invalid_args", started_at)

    max_scan_steps = int(step.args.get("max_scan_steps", 6))
    scan_arc_deg = float(step.args.get("scan_arc_deg", 360))
    step_angle_rad = scan_arc_deg * math.pi / 180.0 / max_scan_steps
    prompt = str(step.args.get("prompt") or f"Locate: {label}")
    max_image_side = int(step.args.get("max_image_side", self._config.default_grounding_max_image_side))

    for i in range(max_scan_steps):
        # a. capture scene observation via ROS client
        try:
            observation = self._ros_client.capture_scene_observation()
            # observation: SceneObservation with .color_image_bgr (numpy BGR uint8 HxWx3),
            #   .aligned_depth_m (numpy float32 HxW), .stamp_sec, .robot_pose_map (dict|None)
            with self._lock:
                runtime.latest_observation = observation
        except Exception as exc:
            return self._failed_result(
                step, f"Scan capture failed at heading {i}: {exc}", "capture_failed", started_at,
            )

        # b. attempt grounding via grounding client
        try:
            image_rgb = self._bgr_to_rgb(observation.color_image_bgr)
            grounding = self._grounding_client.locate_semantic_target(
                GroundingRequest(
                    request_id=f"{runtime.mission_id}:{step.step_id}:scan_{i}",
                    prompt=prompt,
                    image_rgb_u8=image_rgb,
                    image_stamp_sec=observation.stamp_sec,
                    max_image_side=max_image_side,
                )
            )
            with self._lock:
                runtime.latest_grounding = grounding
        except Exception as exc:
            return self._failed_result(
                step, f"Grounding failed during scan at heading {i}: {exc}", "grounding_failed", started_at,
            )

        # c. NEW: store observation in semantic map (best-effort, never blocks scan)
        if self._semantic_map is not None:
            try:
                clip_emb = self._semantic_map.clip_encoder.encode_image(image_rgb)
                pose = Pose2D.from_pose_map_dict(observation.robot_pose_map or {})

                # Build detected object entry if grounding found a target
                detected: list[DetectedObjectEntry] = []
                if grounding.found and grounding.bbox_2d is not None:
                    # Project bbox to 3D position using depth
                    obj_xyz, obj_depth = self._project_bbox_to_map_xyz(
                        observation, grounding.bbox_2d,
                    )
                    if obj_xyz is not None:
                        obs_cov = initial_object_covariance(obj_depth, pose.yaw)
                        detected.append(DetectedObjectEntry(
                            label=grounding.label or label,
                            position_mean=obj_xyz,          # np.array([x, y, z])
                            position_cov=obs_cov,           # 3x3 depth-aware covariance
                            bbox_2d=grounding.bbox_2d,
                            confidence=grounding.confidence or 0.0,
                            first_seen=observation.stamp_sec,
                            last_seen=observation.stamp_sec,
                        ))

                self._semantic_map.add_observation(
                    pose=pose,
                    timestamp=observation.stamp_sec,
                    clip_embedding=clip_emb,
                    detected_objects=detected,
                    source="scan",
                )
            except Exception:
                _logger.debug("Semantic map observation store failed at heading %d", i, exc_info=True)

        # d. return early if target found
        if grounding.found and grounding.bbox_2d is not None:
            return SkillResult(
                step_id=step.step_id,
                skill=step.skill,
                status="succeeded",
                outputs={"heading_index": i, **self._grounding_outputs(grounding)},
                message=f"Target '{label}' found at heading {i}.",
                started_at=started_at,
                finished_at=time.time(),
            )

        # e. check cancel
        if runtime.cancel_event.is_set():
            return SkillResult(
                step_id=step.step_id, skill=step.skill, status="canceled",
                outputs={"heading_index": i}, error_code="mission_canceled",
                message="Scan canceled.", started_at=started_at, finished_at=time.time(),
            )

        # f. rotate to next heading (skip after last heading)
        if i < max_scan_steps - 1:
            try:
                rotate_result = self._ros_client.rotate_in_place(
                    step_id=f"{step.step_id}:rotate_{i}",
                    yaw_delta_rad=step_angle_rad,
                )
                if rotate_result.status != "succeeded":
                    return self._failed_result(
                        step, f"Rotation failed at heading {i}: {rotate_result.message}",
                        rotate_result.error_code or "rotation_failed", started_at,
                    )
            except Exception as exc:
                return self._failed_result(
                    step, f"Rotation failed at heading {i}: {exc}", "rotation_failed", started_at,
                )

    # Exhausted all headings -- describe last observation (best-effort)
    failure_msg = f"Target '{label}' not found after full {scan_arc_deg:.0f} deg scan."
    scan_outputs: dict[str, Any] = {"headings_checked": max_scan_steps}

    observation = runtime.latest_observation
    if observation is not None:
        try:
            desc = self._grounding_client.describe_scene(
                request_id=f"{runtime.mission_id}:{step.step_id}:describe",
                image_rgb_u8=self._bgr_to_rgb(observation.color_image_bgr),
            )
            failure_msg += f" Last observation: {desc.description}"
            scan_outputs["last_scene_description"] = desc.description
        except Exception:
            pass  # description is best-effort

    return self._failed_result(
        step, message=failure_msg, error_code="target_not_found_after_scan",
        started_at=started_at, outputs=scan_outputs,
    )
```

**2. After successful navigation (automatic).**

When the robot arrives at a target pose, capture and store an arrival
observation. This is also used for arrival verification (Section 0.1).

**3. On explicit `describe_scene` skill (operator-triggered).**

The existing `MissionRunner._describe_scene` method already calls
`self._ros_client.capture_scene_observation()` and
`self._grounding_client.describe_scene(...)`. The semantic map integration
wraps this to also store the CLIP embedding and text embedding:

```python
def _describe_scene(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
    started_at = time.time()

    # Reuse or capture a fresh observation
    observation = runtime.latest_observation
    if observation is None:
        try:
            observation = self._ros_client.capture_scene_observation()
            with self._lock:
                runtime.latest_observation = observation
        except Exception as exc:
            return self._failed_result(
                step, f"Failed to capture observation for description: {exc}",
                "capture_failed", started_at,
            )

    prompt = step.args.get("prompt")
    max_image_side = int(step.args.get("max_image_side", self._config.default_grounding_max_image_side))
    image_rgb = self._bgr_to_rgb(observation.color_image_bgr)

    try:
        desc = self._grounding_client.describe_scene(
            request_id=f"{runtime.mission_id}:{step.step_id}",
            image_rgb_u8=image_rgb,
            prompt=prompt,
            max_image_side=max_image_side,
        )
        # desc: SceneDescription with .description (str), .latency_s (float)
    except Exception as exc:
        return self._failed_result(step, f"Scene description failed: {exc}", "describe_failed", started_at)

    # NEW: store in semantic map with CLIP embedding + VLM description
    if self._semantic_map is not None:
        try:
            clip_emb = self._semantic_map.clip_encoder.encode_image(image_rgb)
            pose = Pose2D.from_pose_map_dict(observation.robot_pose_map or {})

            # Optionally detect all salient objects (see Section 1.12)
            detected: list[DetectedObjectEntry] = []
            if hasattr(self._grounding_client, "detect_objects"):
                try:
                    det_result = self._grounding_client.detect_objects(
                        request_id=f"{runtime.mission_id}:{step.step_id}:detect",
                        image_rgb_u8=image_rgb,
                    )
                    for obj in det_result.objects:
                        obj_xyz, obj_depth = self._project_bbox_to_map_xyz(
                            observation, tuple(obj.bbox_2d),
                        )
                        if obj_xyz is not None:
                            obs_cov = initial_object_covariance(obj_depth, pose.yaw)
                            detected.append(DetectedObjectEntry(
                                label=obj.label,
                                position_mean=obj_xyz,
                                position_cov=obs_cov,
                                bbox_2d=tuple(obj.bbox_2d),
                                confidence=obj.confidence,
                                first_seen=observation.stamp_sec,
                                last_seen=observation.stamp_sec,
                            ))
                except Exception:
                    _logger.debug("detect_objects failed; storing without objects", exc_info=True)

            self._semantic_map.add_observation(
                pose=pose,
                timestamp=observation.stamp_sec,
                clip_embedding=clip_emb,
                text_description=desc.description,
                detected_objects=detected,
                source="describe",
            )
        except Exception:
            _logger.debug("Semantic map store failed during describe_scene", exc_info=True)

    return SkillResult(
        step_id=step.step_id,
        skill=step.skill,
        status="succeeded",
        outputs={"description": desc.description, "latency_s": desc.latency_s},
        message="Scene described.",
        started_at=started_at,
        finished_at=time.time(),
    )
```

### 1.6 Query patterns

The semantic map supports several query patterns. The **maturity** column
indicates when each query becomes usable, since not all are available from
the initial implementation:

| Query | Method | Used by | Maturity |
|-------|--------|---------|----------|
| "Where was `<label>` last seen?" | Filter nodes by `detected_objects` entries, return nearest by timestamp or distance | `scan_for_target` (skip scan if recent) | **Step 5.** Works with basic `SemanticMapManager`. Requires `detect_objects` (Section 1.12) or `scan_for_target` grounding hits to populate object entries. |
| "Does this frame match?" | CLIP image-to-image nearest-neighbor search | `verify_arrival` | **Step 1.** Works as soon as CLIP encoder is integrated. ChromaDB handles ANN search. |
| "Find `<text query>`" | CLIP text encoder → search image embeddings (same 512-dim space) | `query_environment`, natural-language search | **Step 1.** Works out of the box with unified CLIP. Quality depends on map density. |
| "Have I been here before?" | CLIP image similarity search against all stored embeddings | Loop closure, revisit detection | **Step 1.** Works immediately. Quality improves with background mapper (denser map). |
| "What's near `<label>`?" | Find node with `<label>`, query neighboring nodes by Euclidean distance or graph edges | Planner context enrichment | **Step 5 (basic), Step 6+ (rich).** Euclidean distance works immediately. Graph edge traversal requires enough traversal-verified edges, which build up over 10+ missions. |
| "Describe the environment" | Aggregate text descriptions from recent nodes | Operator situational awareness | **Step 9+.** Requires VLM descriptions stored on nodes — only populated by `describe_scene` calls or background mapper backfill during idle. Sparse early on. |
| "What objects are in this area?" | Spatial region query across all `DetectedObjectEntry` positions | Inventory, situational awareness | **Step 5 + Section 1.12.** Requires `detect_objects` endpoint to populate positioned object entries. Richness grows with observation count and object reinforcement. |

### 1.7 Query-before-scan optimization

The most immediate optimization the semantic map enables. This block is added
at the top of `MissionRunner._scan_for_target`, before the rotation loop,
and short-circuits the full scan when the target was recently observed:

```python
def _scan_for_target(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
    import math

    started_at = time.time()
    label = str(step.args.get("label", "")).strip()
    if not label:
        return self._failed_result(step, "scan_for_target is missing 'label'.", "invalid_args", started_at)

    # --- NEW: query-before-scan short-circuit ---
    max_map_age_s = float(step.args.get("max_map_age_s", 300))

    if self._semantic_map is not None:
        known = self._semantic_map.query_by_label(label, max_age_s=max_map_age_s)
        if known is not None:
            # Target was seen recently -- verify the scene still looks similar
            try:
                observation = self._ros_client.capture_scene_observation()
                with self._lock:
                    runtime.latest_observation = observation

                image_rgb = self._bgr_to_rgb(observation.color_image_bgr)
                current_clip = self._semantic_map.clip_encoder.encode_image(image_rgb)
                stored_clip = self._semantic_map.get_clip_embedding(known.clip_embedding_id)
                similarity = float(self._semantic_map.cosine_similarity(current_clip, stored_clip))

                if similarity > 0.7:
                    _logger.info(
                        "Query-before-scan hit: label=%s node=%s sim=%.3f",
                        label, known.node_id, similarity,
                    )
                    return SkillResult(
                        step_id=step.step_id,
                        skill=step.skill,
                        status="succeeded",
                        outputs={
                            "source": "semantic_map",
                            "node_id": known.node_id,
                            "clip_similarity": similarity,
                            "pose_x": known.pose.x,
                            "pose_y": known.pose.y,
                            "pose_yaw": known.pose.yaw,
                        },
                        message=f"Target '{label}' found in semantic map (sim={similarity:.2f}), skipping scan.",
                        started_at=started_at,
                        finished_at=time.time(),
                    )
                else:
                    _logger.info(
                        "Query-before-scan miss: label=%s sim=%.3f < 0.7, falling through to scan.",
                        label, similarity,
                    )
            except Exception:
                _logger.debug("Query-before-scan failed, falling through to scan.", exc_info=True)

    # --- Existing rotation loop follows (unchanged) ---
    max_scan_steps = int(step.args.get("max_scan_steps", 6))
    # ... rest of the scan loop as shown in 1.5 ...
```

This eliminates the 20-40 second scan rotation for recently-seen targets.
When the map is empty or the label has not been seen, the code falls through
to the existing rotation loop with zero overhead beyond a dict lookup.

### 1.8 Component selection

| Component | Choice | Rationale | Jetson footprint |
|-----------|--------|-----------|-----------------|
| Spatial graph | NetworkX (in-memory, JSON persistence) | Lightweight, no server process, trivial serialization, rich graph algorithms | ~5 MB RAM for 1000 nodes |
| Vector store | ChromaDB PersistentClient | Embedded (no server), HNSW ANN index, built-in cosine similarity, survives restarts | ~50-100 MB on disk |
| Image + text encoder | OpenCLIP ViT-B/32 (ONNX) | Unified image/text embedding space (512-dim). Image encoder for observations, text encoder for queries. Eliminates need for a separate text model (MiniLM). | ~170 MB model |

**CLIP model selection for Jetson:**

| Model | Params | ONNX size | Embedding dim | Notes |
|-------|--------|-----------|---------------|-------|
| **OpenCLIP ViT-B/32** | 88M | ~170 MB | 512 | **Recommended starting point.** Established ONNX export via `open_clip`. Latency requires benchmarking on Orin Nano. |
| MobileCLIP-S2 | 35M | ~80 MB | 512 | Follow-up optimization if ViT-B/32 latency is too high. Requires manual ONNX conversion from Apple CoreNet (no off-the-shelf ONNX export). |
| OpenCLIP ViT-L/14 | 304M | ~600 MB | 768 | Too heavy for Jetson inline use. |

Recommendation: start with **OpenCLIP ViT-B/32** which has proven ONNX export
tooling (`open_clip.create_model_and_transforms` -> `torch.onnx.export`).
If benchmarking on the Orin Nano shows latency exceeding ~100ms/frame,
investigate MobileCLIP-S2 as an optimization -- but budget time for the ONNX
conversion work since Apple's CoreNet repo does not provide pre-exported ONNX
weights.

All latency figures require benchmarking on the target Orin Nano hardware
before committing to a model choice.

### 1.9 Persistence and lifecycle

```text
~/.strafer/semantic_map/
    graph.json              # NetworkX node_link_data export
    chroma/                 # ChromaDB persistent storage
        chroma.sqlite3
        ...
```

`SemanticMapManager.save()` and `load()` handle graph persistence:

```python
import json
from pathlib import Path
import networkx as nx

class SemanticMapManager:
    def __init__(self, storage_dir: str = "~/.strafer/semantic_map") -> None:
        self._storage_dir = Path(storage_dir).expanduser()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._graph_path = self._storage_dir / "graph.json"
        self._graph: nx.DiGraph = nx.DiGraph()
        # ChromaDB and CLIP encoder (image + text towers) initialized here ...

    def save(self) -> None:
        """Persist the NetworkX graph to disk as JSON."""
        data = nx.node_link_data(self._graph)
        with open(self._graph_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load the NetworkX graph from disk if it exists."""
        if self._graph_path.exists():
            with open(self._graph_path, "r") as f:
                data = json.load(f)
            self._graph = nx.node_link_graph(data, directed=True)
        else:
            self._graph = nx.DiGraph()
```

- The graph and vector store persist across power cycles.
- `save()` is called after each `add_observation()` and `add_edge()` to avoid
  data loss on unexpected shutdown. This is cheap for the expected graph sizes
  (hundreds to low thousands of nodes).
- Nodes older than a configurable TTL (default: 24 hours) are pruned on
  startup to prevent unbounded growth.
- The operator can reset the map with a `clear_semantic_map` skill or CLI
  command.
- When RTAB-Map's SLAM map is reset (`make clean-map`), the semantic map
  should also be cleared since pose references become invalid.

### 1.10 Incremental build-up strategy

The semantic map starts empty and grows during normal operation. With the
`BackgroundMapper` active, the map builds much faster than skill-only capture:

| State | Without BackgroundMapper | With BackgroundMapper | Capability |
|-------|--------------------------|----------------------|------------|
| Empty | Mission 0 | Mission 0 | No map benefits; all skills work as today |
| Sparse (4-12 nodes) | After 1 mission (scan rotations only) | After 1-2 minutes of driving | Basic target re-acquisition, CLIP text queries |
| Moderate (~50-100 nodes) | After 5-10 missions | After 10-15 minutes of driving | Arrival verification, query-before-scan, spatial queries |
| Dense (~200+ nodes) | After 20+ missions | After 30-60 minutes of driving | Full spatial reasoning, environment descriptions, patrol comparison |

The semantic map is a pure additive improvement -- the system works exactly
as it does today with an empty map, and every mission and every meter driven
makes it better.

### 1.11 Implementation plan

| Step | Description | Effort | Dependencies |
|------|-------------|--------|--------------|
| 1 | CLIP encoder wrapper — OpenCLIP ViT-B/32 ONNX, image + text towers | Small | ONNX Runtime, open_clip |
| 2 | ChromaDB integration (PersistentClient, single collection, add/query) | Small | chromadb pip |
| 3 | NetworkX graph manager (add node/edge, query, persist) | Small | networkx pip |
| 4 | `SemanticMapManager` combining graph + vector store + CLIP | Medium | Steps 1-3 |
| 5 | Observation capture in `scan_for_target` with `DetectedObjectEntry` | Small | Step 4 |
| 6 | Arrival verification skill using CLIP similarity | Medium | Steps 4-5 |
| 7 | Query-before-scan optimization | Medium | Steps 4-5 |
| 8 | `describe_and_store` integration with VLM `/describe` | Small | Step 4 |
| 9 | `BackgroundMapper` — passive movement-gated capture thread | Medium | Step 4 |
| 10 | `POST /detect_objects` VLM endpoint (Section 1.12) | Medium | strafer_vlm |
| 11 | Object reinforcement — Bayesian tracking (Mahalanobis gate + Kalman update) | Medium | Steps 5, 10 |
| 12 | Map pruning, reset, and lifecycle management | Small | Step 4 |

### 1.12 Structured object detection endpoint

**Problem.** The current VLM endpoints serve two narrow purposes:
- `POST /ground` finds ONE specific object by label → returns one bbox.
- `POST /describe` returns free-text prose → no structured data.

Neither gives the semantic map a complete inventory of what the robot sees.
An operator command like "what's in this room?" requires multiple grounding
calls (one per hypothesized label) or a description that loses spatial detail.

**Proposal: `POST /detect_objects` endpoint** on `strafer_vlm` that returns
ALL salient objects with bboxes and labels in a single call.

```python
# --- strafer_vlm/service/payloads.py (additions) ---

class DetectObjectsRequest(BaseModel):
    request_id: str
    image_jpeg_b64: str
    max_image_side: int = 1024
    max_objects: int = 20
    min_confidence: float = 0.3

class DetectedObject(BaseModel):
    label: str
    bbox_2d: list[int]            # [x1, y1, x2, y2] pixels
    confidence: float

class DetectObjectsResponse(BaseModel):
    request_id: str
    objects: list[DetectedObject]
    latency_s: float = 0.0
```

**VLM prompt.** Qwen2.5-VL supports multi-object grounding natively:

```text
User: <image>List all visible objects with their bounding boxes.
Assistant: <ref>table</ref><box>(120,340),(450,720)</box>
<ref>chair</ref><box>(500,280),(680,710)</box>
<ref>door</ref><box>(780,50),(990,980)</box>
```

The endpoint parses the `<ref>`/`<box>` tags from the VLM output and returns
structured `DetectedObject` entries. This is the same parsing logic as
`POST /ground` but applied to all objects rather than a single queried label.

**Integration with semantic map.** When `detect_objects` is available, each
observation node stores positioned `DetectedObjectEntry` instances rather
than bare label strings. Combined with depth projection (the existing
`project_detection_to_goal_pose` ROS service, called per bbox), each detected
object gets a real-world 3D position in the map frame.

**Object reinforcement via Bayesian update.** When the robot re-observes the
same object from a different viewpoint, the fixed-radius approach (e.g.,
"same label within 0.5m") is brittle — a table viewed from the front and
from the side may project 0.3-0.8m apart depending on the table's size.
Instead, `SemanticMapManager` uses a Mahalanobis distance check against
each object's spatial covariance, then applies a Kalman filter update on
match:

```python
# Matching threshold: Mahalanobis distance in 3D
MAHALANOBIS_GATE = 3.0  # ~97% confidence region for 3D Gaussian (chi2 df=3)

def reinforce_or_add_object(
    self,
    label: str,
    observed_xyz: np.ndarray,           # [x, y, z] from depth projection
    observation_cov: np.ndarray,        # 3x3 from initial_object_covariance()
    bbox_2d: tuple,
    confidence: float,
    timestamp: float,
) -> DetectedObjectEntry:
    """Match via 3D Mahalanobis distance, update via Kalman filter, or create new."""

    best_match: DetectedObjectEntry | None = None
    best_mahal: float = float("inf")

    for existing in self._all_detected_objects():
        if existing.label != label:
            continue

        # Mahalanobis distance in 3D: accounts for both the existing
        # object's uncertainty AND the new observation's uncertainty
        innovation = observed_xyz - existing.position_mean  # (3,)
        S = existing.position_cov + observation_cov         # 3x3 innovation covariance
        S_inv = np.linalg.inv(S)
        mahal = float(np.sqrt(innovation @ S_inv @ innovation))

        if mahal < MAHALANOBIS_GATE and mahal < best_mahal:
            best_match = existing
            best_mahal = mahal

    if best_match is not None:
        # 3D Kalman filter update — fuses old estimate with new observation
        P_prior = best_match.position_cov   # 3x3
        R = observation_cov                  # 3x3
        S = P_prior + R                      # 3x3 innovation covariance
        K = P_prior @ np.linalg.inv(S)       # 3x3 Kalman gain

        innovation = observed_xyz - best_match.position_mean
        best_match.position_mean = best_match.position_mean + K @ innovation
        best_match.position_cov = (np.eye(3) - K) @ P_prior

        best_match.observation_count += 1
        best_match.last_seen = timestamp
        best_match.confidence = max(best_match.confidence, confidence)
        best_match.bbox_2d = bbox_2d
        return best_match

    # No match — create new object with initial 3D uncertainty
    return DetectedObjectEntry(
        label=label,
        position_mean=observed_xyz,
        position_cov=observation_cov,
        bbox_2d=bbox_2d,
        confidence=confidence,
        first_seen=timestamp,
        last_seen=timestamp,
    )
```

**Why 3D instead of 2D:** The depth projection from
`project_detection_to_goal_pose` gives a full `(x, y, z)` in the map frame.
Collapsing to 2D throws away height, which prevents distinguishing:
- A cup on a table vs. a cup on the floor (same x/y, different z)
- A light switch vs. a door handle on the same wall
- Shelved items at different heights

The 3D covariance ellipsoid captures how uncertainty spreads in all three
axes — most elongated along the camera viewing ray (depth error), tighter
laterally, and tightest vertically (where stereo matching is most
constrained).

**Why Mahalanobis distance instead of a fixed radius:**
- A table at 3m depth has large uncertainty along the viewing ray (~0.18m
  sigma) but small lateral uncertainty (~0.08m) and small vertical
  uncertainty (~0.06m). The Mahalanobis gate naturally allows a wider match
  window along the noisy depth axis.
- A cup at 1m has tight uncertainty in all directions (~0.02m along ray,
  ~0.06m lateral, ~0.04m vertical). The gate is correspondingly tighter,
  avoiding spurious merges with nearby objects.
- The gate value (3.0) corresponds to the ~97% confidence region for a 3D
  Gaussian (chi-squared with 3 degrees of freedom).

**What this means in practice:** after 5+ sightings from different viewpoints,
the 3x3 covariance shrinks to a tight ellipsoid centered on the object's
true 3D position. The `observation_count` and covariance determinant together
give a Bayesian confidence measure — high count + small `det(position_cov)`
= confident landmark with precise 3D location.

This enables richer queries:
- "Where is the nearest table?" → 3D spatial lookup by label, prefer entries
  with small covariance (high certainty)
- "What's new since last patrol?" → filter by `first_seen` timestamp
- "Is the chair still there?" → navigate to stored mean position, re-detect
- "How confident is this object?" → `observation_count` and `det(position_cov)`
- "What's on the table?" → find objects whose z-mean is above the table's
  z-mean by a plausible offset, within the table's x/y covariance region

**Where this runs.** `detect_objects` calls the VLM on DGX (same latency as
`/ground` or `/describe`). Depth projection runs locally on the Jetson via
the existing ROS service. Object reinforcement runs locally in
`SemanticMapManager`.

**Fallback.** If the `detect_objects` endpoint is not available (VLM service
doesn't support it yet), the semantic map falls back to storing bare labels
from `scan_for_target` grounding hits — exactly as it works without this
feature. The endpoint is additive.

---

## 2. Optimizations

### 2.1 Direct planner-to-VLM orchestration (agentic planning)

**Current flow.**

```text
Jetson executor
  → POST DGX:8200/plan    (planner returns plan)
  → execute scan_for_target locally
    → capture frame locally
    → POST DGX:8100/ground (VLM returns bbox)
    → rotate if not found, repeat
  → project goal locally
  → navigate locally
```

The Jetson mediates every call. For each `scan_for_target` rotation step, a
full round-trip image upload goes from Jetson to DGX VLM and back.

**Proposed: agentic orchestration on the compute node.**

When planner and VLM are co-located (same DGX, same Databricks workspace),
the planner can call the VLM directly during plan generation to pre-validate
targets:

```text
Jetson executor
  → POST DGX:8200/plan_with_grounding
    body: { ...PlanRequest fields, image_jpeg_b64 }
    DGX internally:
      planner LLM → intent + plan steps
      if intent requires grounding and image was provided:
        planner calls POST localhost:8100/ground → GroundResponse
      planner returns PlanResponse + optional pre-grounding result
  → Jetson skips scan_for_target (target already grounded)
  → project goal locally
  → navigate locally
```

**Benefits:**
- Eliminates one LAN image upload round-trip (~2-3s saved)
- Planner has grounding context when generating the plan (can adjust if target
  not visible)
- Natural fit for Databricks model serving where both models are endpoints in
  the same workspace

**Implementation.**

The new payload models extend the existing `PlanRequest` and `PlanResponse`
in `strafer_autonomy/planner/payloads.py`:

```python
# --- strafer_autonomy/planner/payloads.py (additions) ---

class GroundResultPayload(BaseModel):
    """Subset of VLM GroundResponse fields forwarded to the caller."""
    found: bool
    bbox_2d: list[int] | None = None
    label: str | None = None
    confidence: float | None = None
    latency_s: float = 0.0

class PlanWithGroundingRequest(PlanRequest):
    """POST /plan_with_grounding — plan request with an optional camera frame."""
    image_jpeg_b64: str | None = None

class PlanWithGroundingResponse(PlanResponse):
    """Response from /plan_with_grounding — standard plan plus optional pre-grounding."""
    pre_grounding: GroundResultPayload | None = None
```

New route handler in `strafer_autonomy/planner/app.py`:

```python
# --- strafer_autonomy/planner/app.py (new route inside create_app) ---

import httpx  # async HTTP client for localhost VLM call

VLM_GROUND_URL = os.environ.get("VLM_GROUND_URL", "http://localhost:8100/ground")

@app.post(
    "/plan_with_grounding",
    response_model=PlanWithGroundingResponse,
    summary="Generate a plan and optionally pre-ground a target via the co-located VLM",
)
async def plan_with_grounding(req: PlanWithGroundingRequest) -> PlanWithGroundingResponse:
    if not _state.llm.ready:
        raise HTTPException(status_code=503, detail="Planner model is not loaded yet.")

    # Stage 1-3: reuse the existing LLM → intent → compile pipeline
    planner_request = PlannerRequest(
        request_id=req.request_id,
        raw_command=req.raw_command,
        robot_state=req.robot_state,
        active_mission_summary=req.active_mission_summary,
        available_skills=tuple(req.available_skills),
    )
    messages = build_messages(planner_request)
    loop = asyncio.get_running_loop()
    raw_output = await loop.run_in_executor(
        _state.inference_pool, lambda: _state.llm.generate(messages),
    )
    intent = parse_intent(raw_output, req.raw_command)
    mission_plan = compile_plan(intent)

    # Stage 4: optional VLM grounding on the same host
    pre_grounding: GroundResultPayload | None = None
    if req.image_jpeg_b64 and intent.requires_grounding:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                vlm_resp = await client.post(VLM_GROUND_URL, json={
                    "request_id": req.request_id,
                    "prompt": f"Locate: {intent.target_label}",
                    "image_jpeg_b64": req.image_jpeg_b64,
                    "image_stamp_sec": 0.0,
                    "max_image_side": 1024,
                })
                vlm_resp.raise_for_status()
                body = vlm_resp.json()
                pre_grounding = GroundResultPayload(
                    found=body["found"],
                    bbox_2d=body.get("bbox_2d"),
                    label=body.get("label"),
                    confidence=body.get("confidence"),
                    latency_s=body.get("latency_s", 0.0),
                )
        except Exception:
            logger.warning("[%s] VLM pre-grounding failed; plan will include scan_for_target",
                           req.request_id, exc_info=True)

    return PlanWithGroundingResponse(
        mission_id=mission_plan.mission_id,
        mission_type=mission_plan.mission_type,
        raw_command=mission_plan.raw_command,
        steps=[SkillCallPayload(
            step_id=s.step_id, skill=s.skill, args=s.args,
            timeout_s=s.timeout_s, retry_limit=s.retry_limit,
        ) for s in mission_plan.steps],
        created_at=mission_plan.created_at,
        pre_grounding=pre_grounding,
    )
```

**Fallback.** If no image is provided, or the VLM call fails, or the VLM
returns `found=false`, the response includes a standard plan with
`scan_for_target` and the Jetson handles the rotation loop as before. The
executor checks `pre_grounding.found` and skips the scan skill only when
a bbox was returned.

### 2.2 Image compression optimization

Currently every VLM call sends a JPEG base64 image over LAN HTTP. The JPEG
quality is hardcoded as `90` in the `_encode_image_to_jpeg_b64` helper inside
`strafer_autonomy/clients/vlm_client.py`:

```python
def _encode_image_to_jpeg_b64(image_rgb_u8: Any, *, quality: int = 90) -> str:
```

The helper already accepts a `quality` keyword argument, but nothing exposes
it through `HttpGroundingClientConfig`. The `max_image_side` resize is already
implemented server-side on `GroundRequest` (default 1024), so no client-side
resize is needed.

**Concrete change.** Add a `jpeg_quality` field to `HttpGroundingClientConfig`
and thread it through to the encode call:

```python
# --- strafer_autonomy/clients/vlm_client.py ---

@dataclass(frozen=True)
class HttpGroundingClientConfig:
    base_url: str
    timeout_s: float = 15.0
    ground_path: str = "/ground"
    describe_path: str = "/describe"
    health_path: str = "/health"
    headers: dict[str, str] | None = None
    max_retries: int = 2
    retry_backoff_s: float = 0.5
    jpeg_quality: int = 90              # <-- NEW: configurable JPEG quality

# In HttpGroundingClient.locate_semantic_target:
    image_jpeg_b64 = _encode_image_to_jpeg_b64(
        request.image_rgb_u8,
        quality=self._config.jpeg_quality,    # <-- pass through
    )

# In HttpGroundingClient.describe_scene:
    image_jpeg_b64 = _encode_image_to_jpeg_b64(
        image_rgb_u8,
        quality=self._config.jpeg_quality,    # <-- pass through
    )
```

This is a two-line functional change (one field, one kwarg passthrough per
call site). Callers can lower `jpeg_quality` to 75 or 60 for faster transfer
when operating over WAN or cloud endpoints. For images larger than 1 MB,
consider S3 pre-signed URL upload instead of inline base64 in a future
iteration.

### 2.3 Parallel health checks at startup

`build_command_server` in `strafer_autonomy/executor/command_server.py`
currently probes VLM health sequentially and does not probe planner health at
all. When both services are on the same host, probe them in parallel to halve
startup latency.

**Concrete change:**

```python
# --- strafer_autonomy/executor/command_server.py (replace health-check block) ---

from concurrent.futures import ThreadPoolExecutor, as_completed

def build_command_server(
    *,
    planner_client,
    grounding_client,
    ros_client,
    runner_config=None,
    server_config: CommandServerConfig | None = None,
    check_vlm_health: bool = True,
):
    # --- parallel health probes ---
    if check_vlm_health:
        futures = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            if hasattr(grounding_client, "health"):
                futures[pool.submit(grounding_client.health)] = "VLM"
            if hasattr(planner_client, "health"):
                futures[pool.submit(planner_client.health)] = "Planner"

            for future in as_completed(futures):
                name = futures[future]
                try:
                    health = future.result(timeout=10.0)
                except Exception as exc:
                    raise RuntimeError(f"{name} health check failed: {exc}") from exc
                if not health.get("model_loaded", False):
                    raise RuntimeError(
                        f"{name} service is reachable but model is not loaded: {health}"
                    )
                _logger.info("%s service healthy: %s", name, health)

    from .mission_runner import MissionRunner

    runner = MissionRunner(
        planner_client=planner_client,
        grounding_client=grounding_client,
        ros_client=ros_client,
        config=runner_config,
    )
    return AutonomyCommandServer(handler=runner, config=server_config), runner
```

Both probes run concurrently via `ThreadPoolExecutor(max_workers=2)`.
`as_completed` ensures we fail fast on the first unhealthy service instead of
waiting for both. Each future gets a 10-second timeout matching the existing
client defaults.

---

## 3. New Features

### 3.1 Rotate skills

**Current state.** `RosClient.rotate_in_place` exists with signature
`(*, step_id: str, yaw_delta_rad: float, tolerance_rad: float = 0.1, timeout_s: float | None = None) -> SkillResult`
but is only called internally by `_scan_for_target` in `MissionRunner`.
`orient_relative_to_target` is dispatched in the `_execute_step` if-chain and
has a full handler in `MissionRunner`, but `JetsonRosClient` raises
`NotImplementedError`. The skill is commented out in `DEFAULT_AVAILABLE_SKILLS`.

**Proposal: two new user-facing skills.**

| Skill | Args | Description |
|-------|------|-------------|
| `rotate_by_degrees` | `degrees: float` | Relative rotation. Positive = CCW. |
| `orient_to_direction` | `direction: str` ("north", "east", etc.) | Rotate to an absolute heading using TF. |

**Changes required across four files:**

**(a) `schemas/mission.py`** -- no change needed; `MissionIntent.orientation_mode`
already exists and can carry the direction string.

**(b) `planner/intent_parser.py`** -- add `"rotate"` to the valid set:

```python
_VALID_INTENTS = frozenset({
    "go_to_target", "wait_by_target", "cancel", "status",
    "rotate",                         # NEW
})
```

**(c) `planner/plan_compiler.py`** -- new compiler function and registry entry:

```python
import math

def _compile_rotate(intent: MissionIntent) -> list[SkillCall]:
    """Compile a rotate intent into a single rotate_by_degrees step.

    The LLM is expected to set ``orientation_mode`` to either a numeric
    degree value (relative rotation) or a cardinal direction string
    (absolute heading via orient_to_direction).
    """
    mode = (intent.orientation_mode or "").strip()

    # Attempt numeric interpretation first ("turn 90 degrees left")
    try:
        degrees = float(mode)
        return [
            SkillCall(
                step_id="step_01",
                skill="rotate_by_degrees",
                args={"degrees": degrees},
                timeout_s=30.0,
                retry_limit=0,
            ),
        ]
    except ValueError:
        pass

    # Cardinal / named direction ("face north")
    return [
        SkillCall(
            step_id="step_01",
            skill="orient_to_direction",
            args={"direction": mode or "north"},
            timeout_s=30.0,
            retry_limit=0,
        ),
    ]


_COMPILERS: dict[str, callable] = {
    "go_to_target": _compile_go_to_target,
    "wait_by_target": _compile_wait_by_target,
    "cancel": _compile_cancel,
    "status": _compile_status,
    "rotate": _compile_rotate,               # NEW
}
```

**(d) `executor/mission_runner.py`** -- new skill in the available set and
two new handlers in the `_execute_step` if-chain:

```python
DEFAULT_AVAILABLE_SKILLS = (
    ...,
    "rotate_by_degrees",                      # NEW
    "orient_to_direction",                    # NEW (when TF lookup is ready)
    ...
)
```

Handler for `rotate_by_degrees` -- thin wrapper over the existing
`RosClient.rotate_in_place`:

```python
def _rotate_by_degrees(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
    import math
    started_at = time.time()
    degrees = float(step.args.get("degrees", 0))
    try:
        return self._ros_client.rotate_in_place(
            step_id=step.step_id,
            yaw_delta_rad=math.radians(degrees),
            timeout_s=step.timeout_s,
        )
    except Exception as exc:
        return self._failed_result(
            step, f"rotate_by_degrees failed: {exc}", "rotation_failed", started_at,
        )
```

Handler for `orient_to_direction` -- uses the TF buffer to compute the
delta between the robot's current heading and the target heading:

```python
_CARDINAL_YAW_RAD: dict[str, float] = {
    "north": 0.0,
    "east": -math.pi / 2,
    "south": math.pi,
    "west": math.pi / 2,
}

def _orient_to_direction(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
    import math
    from rclpy.time import Time
    from tf2_ros import TransformException

    started_at = time.time()
    direction = str(step.args.get("direction", "north")).lower()
    target_yaw = self._CARDINAL_YAW_RAD.get(direction)
    if target_yaw is None:
        return self._failed_result(
            step, f"Unknown direction '{direction}'.", "invalid_args", started_at,
        )

    try:
        tf = self._ros_client.tf_buffer.lookup_transform(
            "map", "base_link", Time(), timeout=rclpy.duration.Duration(seconds=2.0),
        )
        q = tf.transform.rotation
        current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
    except TransformException as exc:
        return self._failed_result(
            step, f"TF lookup failed: {exc}", "tf_lookup_failed", started_at,
        )

    delta = math.atan2(math.sin(target_yaw - current_yaw),
                       math.cos(target_yaw - current_yaw))
    try:
        return self._ros_client.rotate_in_place(
            step_id=step.step_id,
            yaw_delta_rad=delta,
            tolerance_rad=float(step.args.get("tolerance_rad", 0.1)),
            timeout_s=step.timeout_s,
        )
    except Exception as exc:
        return self._failed_result(
            step, f"orient_to_direction failed: {exc}", "rotation_failed", started_at,
        )
```

Both new branches are added to the `_execute_step` if-chain:

```python
if step.skill == "rotate_by_degrees":
    return self._rotate_by_degrees(runtime, step)
if step.skill == "orient_to_direction":
    return self._orient_to_direction(runtime, step)
```

### 3.2 Multi-target mission chaining

**Current state.** `MissionIntent` has a single `target_label: str | None`
field. There is no `targets` field. Composite commands like "go to the cup,
then go to the door" cannot be represented and are rejected by `parse_intent`.

**Proposal: `go_to_targets` intent type.**

**(a) Schema change in `schemas/mission.py`** -- add one field to `MissionIntent`:

```python
@dataclass(frozen=True)
class MissionIntent:
    intent_type: str
    raw_command: str
    target_label: str | None = None
    orientation_mode: str | None = None
    wait_mode: str | None = None
    requires_grounding: bool = False
    targets: list[dict[str, Any]] | None = None   # NEW
```

`target_label` is kept for backward compatibility with single-target intents.

**(b) `planner/intent_parser.py`** -- add `"go_to_targets"` to the valid set
and extract the targets list from LLM JSON:

```python
_VALID_INTENTS = frozenset({
    "go_to_target", "wait_by_target", "cancel", "status",
    "rotate",
    "go_to_targets",                          # NEW
})
```

In `parse_intent`, after the existing `target_label` validation block:

```python
    targets = data.get("targets")
    if intent_type == "go_to_targets":
        if not isinstance(targets, list) or len(targets) < 1:
            raise IntentParseError(
                "intent_type 'go_to_targets' requires a non-empty 'targets' list."
            )
        for i, t in enumerate(targets):
            if not isinstance(t, dict) or not t.get("label"):
                raise IntentParseError(f"targets[{i}] must be a dict with a non-empty 'label'.")

    return MissionIntent(
        ...,
        targets=targets if intent_type == "go_to_targets" else None,
    )
```

Expected LLM JSON output:

```json
{
  "intent_type": "go_to_targets",
  "targets": [
    {"label": "cup", "standoff_m": 0.7},
    {"label": "door", "standoff_m": 0.5}
  ]
}
```

**(c) `planner/plan_compiler.py`** -- new compiler:

```python
def _compile_go_to_targets(intent: MissionIntent) -> list[SkillCall]:
    assert intent.targets is not None
    steps: list[SkillCall] = []
    for i, target in enumerate(intent.targets):
        base = i * 3 + 1
        label = target["label"]
        standoff = float(target.get("standoff_m", 0.7))
        steps.extend([
            SkillCall(
                step_id=f"step_{base:02d}",
                skill="scan_for_target",
                args={"label": label, "max_scan_steps": 6, "scan_arc_deg": 360},
                timeout_s=60.0,
                retry_limit=0,
            ),
            SkillCall(
                step_id=f"step_{base + 1:02d}",
                skill="project_detection_to_goal_pose",
                args={"standoff_m": standoff},
                timeout_s=2.0,
                retry_limit=0,
            ),
            SkillCall(
                step_id=f"step_{base + 2:02d}",
                skill="navigate_to_pose",
                args={"goal_source": "projected_target", "execution_backend": "nav2"},
                timeout_s=90.0,
                retry_limit=0,
            ),
        ])
    return steps


_COMPILERS: dict[str, callable] = {
    ...,
    "go_to_targets": _compile_go_to_targets,  # NEW
}
```

No changes to `_execute_step` are needed -- the compiled plan uses only
existing skills (`scan_for_target`, `project_detection_to_goal_pose`,
`navigate_to_pose`) that already have handlers.

**Semantic map interaction.** Multi-target chaining benefits directly from
query-before-scan (Section 1.7). For "go to the cup, then go to the door",
if both targets are already in the semantic map, the robot skips both scan
rotations.

### 3.3 Scene description as operator feedback

**Current state.** `describe_scene` is an implemented skill in
`DEFAULT_AVAILABLE_SKILLS` with a full handler in `MissionRunner._describe_scene`.
The VLM `POST /describe` endpoint exists. However, there is no intent type
that lets an operator invoke it directly -- it is only triggered as a
fallback when `scan_for_target` exhausts all rotations without finding
the target.

**Proposal.** Expose `describe_scene` as a first-class intent.

**(a) `planner/intent_parser.py`** -- add `"describe"` to the valid set:

```python
_VALID_INTENTS = frozenset({
    "go_to_target", "wait_by_target", "cancel", "status",
    "rotate", "go_to_targets",
    "describe",                               # NEW
})
```

No new validation logic is needed -- `describe` requires neither
`target_label` nor `targets`.

**(b) `planner/plan_compiler.py`** -- new compiler:

```python
def _compile_describe(intent: MissionIntent) -> list[SkillCall]:
    return [
        SkillCall(
            step_id="step_01",
            skill="describe_scene",
            args={"prompt": intent.raw_command},
            timeout_s=30.0,
            retry_limit=0,
        ),
    ]


_COMPILERS: dict[str, callable] = {
    ...,
    "describe": _compile_describe,            # NEW
}
```

No changes to `MissionRunner` or `DEFAULT_AVAILABLE_SKILLS` are needed --
`describe_scene` is already a recognized skill with a working handler.

The description text is returned in `SkillResult.outputs["description"]`
and surfaced in the mission result message. When the semantic map
(Section 1.5) is available, the description is also stored as an
observation node, enriching the robot's environmental memory for future
queries.

### 3.4 Environment query skill

**New skill enabled by the semantic map (Section 1.6).**

This skill queries the semantic map without moving the robot -- no VLM
call, no navigation, pure local graph and vector query.

**(a) `planner/intent_parser.py`** -- add `"query"` to `_VALID_INTENTS`.

**(b) `planner/plan_compiler.py`**:

```python
def _compile_query(intent: MissionIntent) -> list[SkillCall]:
    return [
        SkillCall(
            step_id="step_01",
            skill="query_environment",
            args={"query": intent.raw_command},
            timeout_s=5.0,
            retry_limit=0,
        ),
    ]


_COMPILERS["query"] = _compile_query
```

**(c) `executor/mission_runner.py`** -- add `"query_environment"` to
`DEFAULT_AVAILABLE_SKILLS` and a new handler in `_execute_step`:

```python
if step.skill == "query_environment":
    return self._query_environment(runtime, step)
```

```python
def _query_environment(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
    started_at = time.time()
    query = str(step.args.get("query", "")).strip()
    if not query:
        return self._failed_result(
            step, "query_environment is missing 'query'.", "invalid_args", started_at,
        )
    try:
        results = self._semantic_map.query_by_text(query)
        return SkillResult(
            step_id=step.step_id,
            skill=step.skill,
            status="succeeded",
            outputs={"query": query, "results": results},
            message=f"Found {len(results)} relevant entries.",
            started_at=started_at,
            finished_at=time.time(),
        )
    except Exception as exc:
        return self._failed_result(
            step, f"Environment query failed: {exc}", "query_failed", started_at,
        )
```

This handler returns results via `SkillResult.outputs`, which the mission
result message surfaces to the operator. The query types -- text similarity
search across stored VLM descriptions, spatial graph traversal from a target
node, or aggregation across neighboring nodes -- are determined by the
`SemanticMapManager.query_by_text` implementation (Section 1.6).

### 3.5 Patrol / waypoint sequence

A patrol is an ordered visit to a list of named locations, optionally
looping. It reuses the multi-target compiler from Section 3.2.

**(a) `planner/intent_parser.py`** -- add `"patrol"` to `_VALID_INTENTS`.
The LLM emits a `targets` list as in `go_to_targets`, plus an optional
`"loop": true` flag.

**(b) `planner/plan_compiler.py`**:

```python
def _compile_patrol(intent: MissionIntent) -> list[SkillCall]:
    """Compile a patrol by delegating to _compile_go_to_target per waypoint.

    Looping is handled at the MissionRunner level -- the compiler emits
    a single pass through all targets.
    """
    assert intent.targets is not None
    steps: list[SkillCall] = []
    for i, target in enumerate(intent.targets):
        waypoint_intent = MissionIntent(
            intent_type="go_to_target",
            raw_command=intent.raw_command,
            target_label=target["label"] if isinstance(target, dict) else str(target),
        )
        sub_steps = _compile_go_to_target(waypoint_intent)
        for j, s in enumerate(sub_steps):
            steps.append(SkillCall(
                step_id=f"step_{i * 3 + j + 1:02d}",
                skill=s.skill,
                args=s.args,
                timeout_s=s.timeout_s,
                retry_limit=s.retry_limit,
            ))
    return steps


_COMPILERS["patrol"] = _compile_patrol
```

Loop control lives in `MissionRunner._run_mission`: when
`plan.mission_type == "patrol"` and the intent's `targets` list includes a
loop flag, the runner re-executes the step sequence until the mission is
canceled. This keeps the compiler stateless and loop policy in the executor.

**Semantic map interaction.** During patrol the robot captures observations
at each waypoint, steadily building out map coverage. After a few patrol
cycles the semantic map has dense coverage of the patrol route, enabling
comparative queries ("does this location look different from last time?").

### 3.6 Plan repair on failure (future exploration)

Currently, if a skill fails, the mission fails with no automatic recovery.
Plan repair -- where the executor feeds a failure context back to the planner
and requests a revised plan -- is a natural next step but introduces
significant complexity in state management and safety bounding. This is
deferred to a future design iteration once the semantic map and multi-target
chaining are stable. Initial exploration will focus on the single highest-value
case: re-querying the semantic map for a target's last known pose after a
`scan_for_target` failure, then navigating there and re-scanning.

---

## 4. Deployment — Databricks Integration

### 4.1 Current architecture vs. Databricks target

**Current (DGX Spark on LAN):**

```text
Jetson (robot)                    DGX Spark (192.168.50.196)
  strafer_autonomy.executor         strafer_vlm service (:8100)
  strafer_ros                       planner service (:8200)
  semantic map (local)
       ←——— LAN HTTP ———→
```

**Target (Databricks):**

```text
Jetson (robot)                    Databricks Workspace
  strafer_autonomy.executor         VLM serving endpoint
  strafer_ros                       Planner serving endpoint
  semantic map (local)              (optional: agentic endpoint
       ←——— HTTPS ———→              combining planner + VLM)
```

Note: the semantic map (CLIP, ChromaDB, NetworkX) stays on the Jetson in both
architectures. Only VLM descriptions and planner calls go over the network.
This ensures the robot retains its environmental memory even when disconnected.

### 4.2 Databricks model serving fit

Both services are good candidates for Databricks custom model serving:

| Service | Payload | Latency tolerance | GPU | Fit |
|---------|---------|-------------------|-----|-----|
| Planner | ~1 KB JSON | 2-5s | Optional (Qwen3-4B runs on CPU with quantization) | Good — lightweight text-in/JSON-out |
| VLM | ~200 KB (image + prompt) | 3-5s | Required (Qwen2.5-VL-3B) | Good — within 16 MB payload limit |

Databricks constraints to design around:
- **Payload limit:** 16 MB per request (fine for 640x360 JPEG, would need S3
  for high-res images)
- **Execution timeout:** 297 seconds (fine for both services)
- **Scale-to-zero:** endpoints can scale to zero but cold start adds 30-60s
  latency. Not acceptable for real-time robot operation. Keep minimum
  instances = 1 for active deployments.
- **GPU instances:** Databricks serves GPU models via serving endpoints with
  GPU-enabled instance types

### 4.3 Recommended Databricks architecture

**Two serving endpoints:**

1. `strafer-planner` — custom Python model serving endpoint
   - Input: `PlannerRequest` (JSON)
   - Output: `MissionPlan` (JSON)
   - Model: Qwen3-4B loaded via transformers
   - Instance: CPU or small GPU, minimum 1 instance

2. `strafer-vlm` — custom Python model serving endpoint
   - Input: `GroundingRequest` (JSON with base64 image)
   - Output: `GroundingResult` (JSON)
   - Model: Qwen2.5-VL-3B-Instruct loaded via transformers
   - Instance: GPU (T4 or A10G minimum), minimum 1 instance

**Optional: agentic endpoint (planner + VLM combined):**

3. `strafer-agent` — single endpoint that wraps both models
   - Accepts `PlanWithGroundingRequest` (command + optional image)
   - Internally calls planner model, then VLM model if grounding is needed
   - Returns `PlanWithGroundingResponse` (plan + pre-grounding)
   - Reduces Jetson-to-cloud round-trips from 2+ to 1
   - Natural fit for Databricks since both models are in the same serving
     infrastructure

### 4.4 Design changes for Databricks readiness

**4.4.1 Add Databricks client implementations.**

The codebase already has a clean protocol/implementation split for both
services. No new abstraction layer is needed — Databricks support is just a
second implementation of each existing protocol.

**Existing protocol pattern (already in the codebase):**

```text
strafer_autonomy/clients/planner_client.py
  PlannerClient (Protocol)            <-- runtime-checkable interface
    plan_mission(PlannerRequest) -> MissionPlan

  HttpPlannerClientConfig             <-- frozen dataclass (base_url, timeout, retries, ...)
  HttpPlannerClient                   <-- implements PlannerClient via requests session
    __init__(config: HttpPlannerClientConfig)
    plan_mission(request) -> MissionPlan
    health() -> dict[str, Any]

strafer_autonomy/clients/vlm_client.py
  GroundingClient (Protocol)          <-- runtime-checkable interface
    locate_semantic_target(GroundingRequest) -> GroundingResult
    describe_scene(...) -> SceneDescription

  HttpGroundingClientConfig           <-- frozen dataclass (base_url, timeout, retries, ...)
  HttpGroundingClient                 <-- implements GroundingClient via requests session
    __init__(config: HttpGroundingClientConfig)
    locate_semantic_target(request) -> GroundingResult
    describe_scene(...) -> SceneDescription
    health() -> dict[str, Any]
```

`HttpPlannerClient` and `HttpGroundingClient` are the LAN transport
implementations. `main.py` reads `VLM_URL` and `PLANNER_URL` env vars,
constructs clients with default configs, and calls `build_command_server()`.

**New: Databricks implementations of the same protocols.**

Add two new classes — one per protocol — that call Databricks model serving
endpoints instead of LAN HTTP services:

```python
# strafer_autonomy/clients/databricks_planner_client.py

@dataclass(frozen=True)
class DatabricksServingPlannerClientConfig:
    endpoint_name: str                        # e.g. "strafer-planner"
    workspace_url: str                        # e.g. "https://<workspace>.databricks.net"
    token: str                                # Databricks PAT or OAuth token
    timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0

class DatabricksServingPlannerClient:
    """Implements PlannerClient via Databricks model serving endpoint."""

    def __init__(self, config: DatabricksServingPlannerClientConfig) -> None:
        ...

    def plan_mission(self, request: PlannerRequest) -> MissionPlan:
        # POST to {workspace_url}/serving-endpoints/{endpoint_name}/invocations
        # Request body: DataFrame-split format wrapping PlannerRequest fields
        # Response: MissionPlan JSON parsed from serving output
        ...

    def health(self) -> dict[str, Any]:
        # GET endpoint state via Databricks SDK or REST API
        # Maps endpoint state (READY / NOT_READY / PENDING) to health dict
        ...
```

```python
# strafer_autonomy/clients/databricks_vlm_client.py

@dataclass(frozen=True)
class DatabricksServingGroundingClientConfig:
    endpoint_name: str
    workspace_url: str
    token: str
    timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0

class DatabricksServingGroundingClient:
    """Implements GroundingClient via Databricks model serving endpoint."""

    def __init__(self, config: DatabricksServingGroundingClientConfig) -> None:
        ...

    def locate_semantic_target(self, request: GroundingRequest) -> GroundingResult:
        # POST to serving endpoint; image sent as base64 in DataFrame payload
        ...

    def describe_scene(self, *, request_id: str, image_rgb_u8: Any,
                       prompt: str | None = None,
                       max_image_side: int = 1024) -> SceneDescription:
        ...

    def health(self) -> dict[str, Any]:
        ...
```

**Update `main.py` to select client based on env vars.**

The executor entry point gains a backend selector. When `PLANNER_BACKEND` or
`VLM_BACKEND` is set to `"databricks"`, it constructs the Databricks client
instead of the HTTP client. No other code changes — the rest of the executor
depends only on the `PlannerClient` and `GroundingClient` protocols.

```python
# In main.py — client construction

if os.environ.get("PLANNER_BACKEND") == "databricks":
    planner_client = DatabricksServingPlannerClient(
        config=DatabricksServingPlannerClientConfig(
            endpoint_name=os.environ["PLANNER_ENDPOINT"],
            workspace_url=os.environ["DATABRICKS_HOST"],
            token=os.environ["DATABRICKS_TOKEN"],
        )
    )
else:
    planner_url = os.environ.get("PLANNER_URL", "http://192.168.50.196:8200")
    planner_client = HttpPlannerClient(
        config=HttpPlannerClientConfig(base_url=planner_url)
    )

if os.environ.get("VLM_BACKEND") == "databricks":
    grounding_client = DatabricksServingGroundingClient(
        config=DatabricksServingGroundingClientConfig(
            endpoint_name=os.environ["VLM_ENDPOINT"],
            workspace_url=os.environ["DATABRICKS_HOST"],
            token=os.environ["DATABRICKS_TOKEN"],
        )
    )
else:
    vlm_url = os.environ.get("VLM_URL", "http://192.168.50.196:8100")
    grounding_client = HttpGroundingClient(
        config=HttpGroundingClientConfig(base_url=vlm_url)
    )
```

Environment variables for Databricks mode:

```bash
# LAN mode (current — no changes needed)
PLANNER_URL=http://192.168.50.196:8200
VLM_URL=http://192.168.50.196:8100

# Databricks mode
PLANNER_BACKEND=databricks
VLM_BACKEND=databricks
DATABRICKS_HOST=https://<workspace>.databricks.net
DATABRICKS_TOKEN=dapi...
PLANNER_ENDPOINT=strafer-planner
VLM_ENDPOINT=strafer-vlm
```

**4.4.2 Package models for Databricks serving.**

Each model needs an MLflow `PythonModel` wrapper that replicates the full
inference pipeline from the corresponding service — not just the raw model.

**Planner model.** The planner service pipeline is:
`build_messages` -> `LLMRuntime.generate` -> `parse_intent` -> `compile_plan`
-> `PlanResponse`. `LLMRuntime` wraps `Qwen3-4B` via transformers with
tokenization, generation config, and output parsing. The MLflow wrapper must
reproduce this entire chain:

```python
# databricks/planner_model.py

class StraferPlannerModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for the full planner pipeline.

    Replicates the service-side chain:
    build_messages -> LLMRuntime.generate -> parse_intent -> compile_plan
    """

    def load_context(self, context):
        from strafer_autonomy.planner.llm_runtime import LLMRuntime
        self.runtime = LLMRuntime(model_path=context.artifacts["model"])

    def predict(self, context, model_input):
        # model_input: DataFrame with columns [request_id, raw_command, ...]
        from strafer_autonomy.planner.pipeline import (
            build_messages, parse_intent, compile_plan,
        )
        results = []
        for _, row in model_input.iterrows():
            messages = build_messages(raw_command=row["raw_command"])
            raw_output = self.runtime.generate(messages)
            intent = parse_intent(raw_output)
            plan = compile_plan(intent, request_id=row["request_id"])
            results.append(plan.model_dump())
        return results
```

**VLM model.** The VLM service wraps `Qwen2.5-VL-3B-Instruct` via
transformers with a `VLMRuntime` class. The MLflow wrapper mirrors this:

```python
# databricks/vlm_model.py

class StraferVLMModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for the VLM grounding/description pipeline."""

    def load_context(self, context):
        from strafer_autonomy.vlm.vlm_runtime import VLMRuntime
        self.runtime = VLMRuntime(model_path=context.artifacts["model"])

    def predict(self, context, model_input):
        # model_input: DataFrame with columns [request_id, image_b64, prompt, mode]
        # mode: "ground" or "describe"
        results = []
        for _, row in model_input.iterrows():
            image = decode_base64_image(row["image_b64"])
            if row["mode"] == "ground":
                result = self.runtime.ground(image=image, prompt=row["prompt"])
            else:
                result = self.runtime.describe(image=image, prompt=row.get("prompt"))
            results.append(result.model_dump())
        return results
```

**Registration and endpoint creation:**

```python
import mlflow

# Log planner model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="planner",
        python_model=StraferPlannerModel(),
        artifacts={"model": "/path/to/Qwen3-4B"},
        pip_requirements=["transformers", "torch", "strafer_autonomy"],
    )

# Log VLM model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="vlm",
        python_model=StraferVLMModel(),
        artifacts={"model": "/path/to/Qwen2.5-VL-3B-Instruct"},
        pip_requirements=["transformers", "torch", "qwen-vl-utils", "strafer_autonomy"],
    )
```

After registration, create serving endpoints via Databricks UI or SDK,
pointing at the registered MLflow model version.

**4.4.3 Health and readiness.**

Both `HttpPlannerClient.health()` and `HttpGroundingClient.health()` already
exist and return a `dict[str, Any]`. The executor's readiness checks call
these methods through the protocol-typed reference, so the Databricks clients
simply need their own `health()` implementation.

Databricks serving endpoints do not expose custom `/health` routes. Instead,
each endpoint has a built-in readiness state queryable via the Databricks REST
API or SDK. The Databricks client `health()` methods map this to the same
return format:

```python
# Inside DatabricksServingPlannerClient / DatabricksServingGroundingClient

def health(self) -> dict[str, Any]:
    # Option A: Databricks SDK
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient(host=self._workspace_url, token=self._token)
    state = w.serving_endpoints.get(self._endpoint_name).state
    return {
        "status": "ok" if state.ready == "READY" else "unavailable",
        "backend": "databricks",
        "endpoint": self._endpoint_name,
        "endpoint_state": state.ready,
    }
```

Because the executor already calls `planner_client.health()` and
`grounding_client.health()` through the protocol references, no changes to
health-check logic are needed — the Databricks implementation is transparent.

### 4.5 Migration path

| Phase | Planner | VLM | Transport |
|-------|---------|-----|-----------|
| Current | DGX Spark LAN | DGX Spark LAN | HTTP (requests) |
| Phase A | Databricks endpoint | DGX Spark LAN | Mixed (test planner on Databricks) |
| Phase B | Databricks endpoint | Databricks endpoint | HTTPS (Databricks SDK) |
| Phase C | Databricks agentic endpoint | (combined) | HTTPS (single endpoint) |

Each phase is independently testable. The `PlannerClient` and
`GroundingClient` protocol pattern means the Jetson executor code does not
change between phases — only the env-var configuration and the client
implementation selected at startup.

### 4.6 Cost and latency considerations

- **Keep minimum instances = 1** for both endpoints during active development
  and demos. Scale-to-zero cold starts (30-60s) are unacceptable for a robot
  waiting to execute a command.
- **Image size matters.** 640x360 JPEG at quality 85 is ~100 KB. Well within
  the 16 MB Databricks limit. Don't send full-resolution frames.
- **LAN vs. cloud latency.** Current LAN round-trip for VLM grounding is ~2-3s.
  Cloud adds ~100-200ms network overhead. Acceptable for the current skill
  execution model (robot is stationary during grounding).
- **Batch optimization.** For patrol/multi-target missions, consider batching
  multiple grounding requests in a single endpoint call to amortize network
  overhead.
- **Semantic map stays local.** CLIP and text encoding run on Jetson. ChromaDB
  and NetworkX are local. No cloud dependency for map operations. This means
  the robot can execute map-aided missions even during network outages (it
  just can't get new VLM descriptions or planner assistance).

---

## 5. Synthetic Data and Sim-to-Real (strafer_lab)

### 5.1 Dual purpose

`strafer_lab` began as a sim-to-real RL pipeline for training low-level,
reactive A-to-B navigation policies in Isaac Lab. That role remains -- a
trained policy serves as a hybrid Nav2 motion planner backend that handles
dynamic obstacle avoidance better than classical planners alone.

However, the same simulation infrastructure -- Isaac Sim rendering, Infinigen
room geometry, RealSense D555 camera model, domain randomization -- is also a
**synthetic data factory** capable of generating labeled training data for
every perception model in the autonomy stack.

```text
strafer_lab (Isaac Lab / Isaac Sim)
  |
  |- Role 1: RL policy training
  |     Train holonomic nav policy -> export ONNX -> Nav2 hybrid backend
  |
  +- Role 2: Synthetic data generation
        Generate (RGB, depth, pose, label) tuples
          -> fine-tune CLIP / VLM / text encoder / goal projection
```

**API surface.** Isaac Lab environments use the standard Gymnasium API:

```python
obs_dict, reward, terminated, truncated, info = env.step(action)
```

`obs_dict` is keyed by observation group name (e.g. `"policy"`), containing
concatenated float32 tensors. Raw camera data is NOT in the obs dict -- it is
accessed through the scene sensor handles:

```python
# RGB: (num_envs, 60, 80, 4) RGBA uint8
env.scene["d555_camera"].data.output["rgb"]

# Depth: (num_envs, 60, 80, 1) float32 meters
env.scene["d555_camera"].data.output["distance_to_image_plane"]
```

D555 camera config (from `d555_cfg.py`): 80x60 resolution (downsampled for
policy input), focal length 1.93 mm, horizontal aperture 3.68 mm, clip range
0.01-6.0 m (sim) with real min range 0.4 m handled by nearfield fill, update
rate 30 Hz, mounted at (0.20, 0.0, 0.25) m from `body_link`.

### 5.2 What strafer_lab already provides

The simulation environments are already configured with the infrastructure
needed for data generation:

| Capability | Current state | Data gen value |
|------------|---------------|----------------|
| **RealSense D555 camera model** (`d555_cfg.py`) | Matched FOV, resolution, depth noise model | Synthetic frames look like real robot frames |
| **Infinigen room geometry** (Phase 6 envs) | Offline USD scenes loaded into sim | Realistic indoor environments with diverse objects |
| **ProcRoom procedural rooms** (`proc_room.py`) | 44 primitive shapes, 8-level difficulty, GPU BFS | Unlimited unique room layouts |
| **Domain randomization** (`events.py`) | Friction, mass, motor strength, camera mount jitter | Distribution shift robustness |
| **Sensor noise models** (`noise_models.py`) | IMU bias drift, encoder quantization, stereo depth noise, RGB pixel noise, frame drops | Realistic sensor degradation for training robust models |
| **Gamepad demo collection** (`collect_demos.py`) | World-frame teleop, HDF5 output (`obs`, `actions`, `rewards` per episode) | Template for perception data collection |
| **30 registered environments** | 5 scene types x 3 realism levels x 2 modes (Train/Play) | Complete sim infrastructure ready to tap |

### 5.3 Infinigen as a labeled object source

Infinigen (Princeton procedural generation) creates realistic indoor scenes
with full semantic knowledge -- every object has a class label, mesh, material,
and pose **inside the Infinigen pipeline**. However, the current integration
only partially preserves this information.

**What works today:**

- Infinigen generates Blender scenes with per-object labels and categories.
- `prep_room_usds.py` converts these to USD and places them at `/World/Room`.
- `scenes_metadata.json` provides `spawn_points_xy` per scene for robot
  placement.
- The geometry renders correctly with materials, lighting, and occlusion.

**What does NOT work today:**

- `prep_room_usds.py` imports geometry as USD meshes but **does not preserve
  Infinigen's semantic labels**. The per-object class names, category tags,
  and instance IDs from Blender are dropped during USD conversion.
- The env exposes no per-object semantic labels through the `info` dict.
  There is no `info["scene_objects"]` or similar structure.
- Extracting labels at runtime would require either (a) modifying
  `prep_room_usds.py` to embed Infinigen labels as USD prim metadata, or
  (b) parsing the USD scene graph at `/World/Room` to recover prim names
  and match them against Infinigen's internal naming conventions.

**What needs to be built:**

1. **Label-preserving USD export** -- extend `prep_room_usds.py` to write
   Infinigen's object class, instance ID, and 3D bbox as custom USD prim
   attributes (e.g. `custom:semanticLabel`, `custom:instanceId`).
2. **Runtime label accessor** -- a utility that reads these attributes from
   the loaded USD stage and provides a mapping from prim path to
   `(class_name, instance_id, world_pose)`.
3. **Per-scene label manifest** -- extend `scenes_metadata.json` (or add a
   companion `scenes_labels.json`) listing all objects with their class,
   mesh bounding box, and initial pose. This avoids parsing USD at runtime.

Until step 1 is complete, Infinigen scenes provide high-quality unlabeled
geometry. ProcRoom scenes (Section 5.5) are the more practical starting point
for labeled data generation since object poses are already accessible.

### 5.4 Fine-tuning targets

| Model | Training data from sim | Fine-tune goal | Impact on autonomy stack |
|-------|----------------------|----------------|--------------------------|
| **OpenCLIP ViT-B/32** | RGB frames from Infinigen rooms, rendered from robot camera height and FOV with D555 noise model | Domain-adapted visual embeddings tuned to indoor scenes from the robot's perspective | Better semantic map retrieval, more accurate arrival verification (Section 0.1), stronger visual place recognition |
| **VLM (Qwen2.5-VL)** | RGB frames + ground-truth 2D bboxes from sim object poses | Improved grounding accuracy for objects common in the operating environment | Fewer missed detections during `scan_for_target`, fewer hallucinated bboxes |
| **Goal projection** | Depth frames with known ground-truth object distances | More accurate standoff distance estimation from depth + bbox | Robot stops closer to the intended distance from the target |
| **RL navigation policy** | Full episodes with RGB/depth observations (existing Role 1) | Reactive obstacle avoidance in cluttered environments | Smoother navigation in dynamic scenes |

### 5.5 Synthetic data collection pipeline

A new data collection script, modeled after the existing `collect_demos.py`
but for perception rather than control. The code below uses the actual
strafer_lab API surface.

```python
# scripts/collect_perception_data.py

import torch
import numpy as np
from pathlib import Path

class PerceptionDataCollector:
    """
    Run the robot through sim environments while capturing
    perception data for downstream fine-tuning.

    Uses the Isaac Lab Gymnasium API and scene sensor handles
    to extract RGB, depth, and pose data.
    """

    def __init__(self, env, output_dir: Path):
        self.env = env
        self.output_dir = output_dir
        self.camera = env.scene["d555_camera"]

    def collect_episode(self, episode_id: int, max_steps: int = 500) -> dict:
        obs_dict, info = self.env.reset()
        frames = []

        for step in range(max_steps):
            action = self.get_action(obs_dict)
            obs_dict, reward, terminated, truncated, info = self.env.step(action)

            # --- Camera data (accessed via scene sensor, NOT obs_dict) ---
            # Shape: (num_envs, 60, 80, 4) RGBA uint8
            rgba = self.camera.data.output["rgb"]
            rgb = rgba[..., :3]  # drop alpha -> (num_envs, 60, 80, 3)

            # Shape: (num_envs, 60, 80, 1) float32 meters
            depth = self.camera.data.output["distance_to_image_plane"]

            # --- Poses ---
            cam_pos = self.camera.data.pos_w           # (num_envs, 3)
            cam_quat = self.camera.data.quat_w_world   # (num_envs, 4)
            robot_pos = self.env.scene["robot"].data.root_pos_w    # (num_envs, 3)
            robot_quat = self.env.scene["robot"].data.root_quat_w  # (num_envs, 4)

            # --- ProcRoom ground truth (when available) ---
            obstacle_positions = None
            if hasattr(self.env, "_proc_room_active_mask"):
                # (num_envs, 44) bool -- which obstacle slots are populated
                active_mask = self.env._proc_room_active_mask
                # Per-slot 3D positions; prim names like "furn_table_0"
                # are the only "labels" available -- no semantic class names
                obstacle_positions = (
                    self.env.scene["obstacles"].data.object_pos_w  # (num_envs, 44, 3)
                )

            frame_data = {
                "rgb": rgb.cpu().numpy(),
                "depth": depth.cpu().numpy(),
                "cam_pos": cam_pos.cpu().numpy(),
                "cam_quat": cam_quat.cpu().numpy(),
                "robot_pos": robot_pos.cpu().numpy(),
                "robot_quat": robot_quat.cpu().numpy(),
            }
            if obstacle_positions is not None:
                frame_data["obstacle_positions"] = obstacle_positions.cpu().numpy()
                frame_data["obstacle_active_mask"] = active_mask.cpu().numpy()

            frames.append(frame_data)

            if terminated.any() or truncated.any():
                break

        return self.save_episode(episode_id, frames)
```

**Ground-truth 2D bounding boxes.** The current setup does NOT provide 2D
bboxes out of the box. Two approaches to obtain them:

1. **Isaac Sim Replicator API** -- attach a `bounding_box_2d_tight` annotator
   to the camera sensor via `omni.replicator.core`:
   ```python
   import omni.replicator.core as rep
   annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
   annotator.attach([camera_render_product_path])
   bbox_data = annotator.get_data()
   ```
   This requires wiring the annotator to the camera's render product, which
   is not yet done in `d555_cfg.py`.

2. **Manual projection** -- project known 3D object positions (from
   `env.scene["obstacles"].data.object_pos_w`) through the camera intrinsics
   (focal length 1.93 mm, aperture 3.68 mm, resolution 80x60) to pixel
   coordinates. This gives center points; full bboxes require knowing each
   object's 3D extent, which can be read from the USD prim bounds.

For ProcRoom, approach 2 is simpler since the 44 obstacle slots have known
primitive shapes (boxes, cylinders, etc.) with fixed dimensions. For Infinigen
scenes, approach 1 is necessary since object geometry is arbitrary.

**ProcRoom object inventory (44 total):**

| Category | Count | Types |
|----------|-------|-------|
| Walls | 20 | 8 long (2.0 m), 8 medium (1.0 m), 4 short (0.5 m) |
| Furniture | 8 | 2 tables, 2 shelves, 2 cabinets, 2 couches |
| Clutter | 16 | 4 boxes, 2 cylinders, 2 flat, 2 spheres, 2 cones, 2 capsules, 2 tall cylinders |

All are primitive shapes with solid randomized colors. The only "labels" are
prim names (e.g. `furn_table_0`, `clutter_box_2`). These names encode the
object category, which is sufficient for training data labeling.

**Output format.** HDF5 or WebDataset shards, compatible with standard
PyTorch dataloaders. Each shard contains:

```text
episode_0042/
  frame_0000.jpg          # RGB (80x60)
  frame_0000_depth.png    # uint16 depth in mm
  frame_0000.json         # camera pose, robot pose, obstacle positions
  ...
```

### 5.6 CLIP fine-tuning with contrastive sim data

The semantic map (Section 1) relies on CLIP embeddings for visual place
recognition. Out-of-the-box CLIP was trained on internet images -- it has
no special understanding of indoor robot perspectives at 25 cm camera height
with a D555 FOV.

**Contrastive fine-tuning approach:**

1. **Positive pairs** -- two frames of the same scene from slightly different
   robot poses (e.g., 10 cm translational offset, 5 deg rotational offset).
   The sim provides exact camera poses, so generating positives is trivial.

2. **Negative pairs** -- frames from different rooms or significantly different
   viewpoints in the same room.

3. **Hard negatives** -- frames from similar-looking but different locations
   (e.g., two doorways in different rooms). The sim's room diversity makes
   hard negative mining straightforward.

**Implementation with OpenCLIP.** Use OpenCLIP's built-in model loading and
a standard PyTorch training loop:

```python
import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Freeze text tower, fine-tune vision tower
for param in model.token_embedding.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5, weight_decay=0.01
)

# Training loop over contrastive pairs
for anchor, positive, negatives in dataloader:
    anchor_emb = model.encode_image(anchor)
    positive_emb = model.encode_image(positive)
    negative_embs = [model.encode_image(n) for n in negatives]

    loss = info_nce_loss(anchor_emb, positive_emb, negative_embs)
    loss.backward()
    optimizer.step()
```

Fine-tune with InfoNCE / NT-Xent loss on the contrastive pairs. This adapts
the CLIP embedding space to better distinguish indoor locations from the
robot's perspective without losing general visual understanding.

**Dataset scale.** Generating 10k-50k image pairs from sim is trivial -- a
few hours of rendering across the 30 registered environments, stepping the
robot through random trajectories while sampling camera jitter for positive
pairs. This scale is sufficient for contrastive fine-tuning of the vision
tower.

**Training infrastructure.** DGX Spark for fast local iteration during
development. Databricks for tracked experiments via MLflow -- each run logs
hyperparameters, contrastive loss curves, and the resulting ONNX checkpoint.
The fine-tuned model is exported to ONNX and deployed on the Jetson -- same
deployment path as the base model.

### 5.7 VLM fine-tuning with sim ground truth

The VLM (Qwen2.5-VL-3B) handles object grounding during `scan_for_target`.
Fine-tuning on domain-specific data improves detection of objects common in
the robot's operating environment.

**Training data format.** Qwen2.5-VL grounding uses a specific format with
`<ref>` and `<box>` tags. Coordinates are pixel values scaled to 0-1000:

```text
User: <image>Locate the table in this image.
Assistant: <ref>table</ref><box>(321,450),(580,890)</box>
```

For multiple objects of the same class:

```text
User: <image>Locate all boxes in this image.
Assistant: <ref>box</ref><box>(102,200),(250,410)</box><box>(600,310),(780,520)</box>
```

The corresponding JSONL training format:

```json
{
  "image": "frame_0042.jpg",
  "conversations": [
    {
      "role": "user",
      "content": "<image>Locate the table in this image."
    },
    {
      "role": "assistant",
      "content": "<ref>table</ref><box>(321,450),(580,890)</box>"
    }
  ]
}
```

Ground-truth 2D bboxes are obtained by projecting known 3D object positions
through the camera intrinsics (Section 5.5) and converting pixel coordinates
to the 0-1000 scale that Qwen2.5-VL expects.

**Negative examples are critical.** Render frames that do NOT contain the
queried object, and train the VLM to respond with a refusal rather than
hallucinating a detection:

```json
{
  "image": "frame_0099.jpg",
  "conversations": [
    {
      "role": "user",
      "content": "<image>Locate the chair in this image."
    },
    {
      "role": "assistant",
      "content": "The object is not visible in this image."
    }
  ]
}
```

Without negative examples, the VLM learns to always produce a bounding box
regardless of whether the object is present, leading to hallucinated
detections on the real robot. A ratio of roughly 1:3 negative-to-positive
examples is a practical starting point.

### 5.8 Closing the sim-to-real loop

The data pipeline creates a feedback loop between simulation and deployment:

```text
                    +------------------------------+
                    |  strafer_lab (Isaac Sim)      |
                    |                               |
  +--------------->|  1. Generate labeled data      |
  |                |  2. Train RL policy            |
  |                +-------------+----------------+
  |                              |
  |                       labeled data + policy
  |                              |
  |                +-------------v-----------------+
  |                |  DGX / Databricks              |
  |                |                                |
  |                |  3. Fine-tune CLIP, VLM,       |
  |                |     text encoder                |
  |                |  4. Export ONNX models          |
  |                +-------------+-----------------+
  |                              |
  |                       fine-tuned models
  |                              |
  |                +-------------v-----------------+
  |                |  Jetson (robot)                |
  |                |                                |
  |                |  5. Deploy models               |
  |                |  6. Run missions                |
  |                |  7. Log failures                |
  |                +-------------+-----------------+
  |                              |
  |                    failure_manifest.json
  |                              |
  +------------------------------+
       gen_failure_scenarios.py reads manifest,
       spawns targeted scenarios in sim
```

**Step 7: concrete failure feedback.** `SemanticMapManager` logs perception
failures during normal operation. These are exported as a JSON manifest:

```json
{
  "failures": [
    {
      "timestamp": "2026-04-07T14:32:01Z",
      "type": "arrival_verification_failed",
      "target_label": "red chair",
      "clip_similarity": 0.31,
      "robot_pose": [2.1, 3.4, 0.0, 0.0, 0.0, 0.12, 0.99],
      "frame_path": "failures/frame_20260407_143201.jpg",
      "depth_path": "failures/depth_20260407_143201.png",
      "scene_description": "living room with couch and bookshelf"
    },
    {
      "timestamp": "2026-04-07T15:10:44Z",
      "type": "missed_detection",
      "target_label": "water bottle",
      "vlm_response": "The object is not visible in this image.",
      "robot_pose": [1.8, 5.2, 0.0, 0.0, 0.0, -0.45, 0.89],
      "frame_path": "failures/frame_20260407_151044.jpg",
      "scene_description": "kitchen counter with small objects"
    }
  ]
}
```

A script (`gen_failure_scenarios.py`) reads this manifest and spawns similar
scenarios in simulation:

- **Arrival verification failures** (low CLIP similarity) -- generate rooms
  with visually similar distractors near the target object class, producing
  contrastive pairs that teach CLIP to distinguish them.
- **Missed detections** -- place the target object class at similar relative
  poses and distances in ProcRoom, generating positive VLM grounding examples.
- **Hallucinated detections** -- generate scenes without the hallucinated
  object class, producing negative VLM examples.

For ProcRoom, the script directly configures difficulty level and object
placement. For Infinigen, it selects scenes from the metadata catalog that
match the failure's scene description (e.g. "kitchen" scenes for kitchen
failures).

### 5.9 Sim-in-the-loop data collection (follow-up)

The data collection scripts in Section 5.5 use scripted trajectories or
random walks to drive the robot through environments. This produces
geometrically diverse data but doesn't match the real deployment distribution
— the robot doesn't scan, ground, or navigate the way it does during actual
missions.

**Proposal: run the full autonomy stack against the simulated robot** using
the Isaac Sim ROS2 Bridge (`isaacsim.ros2_bridge` / OmniGraph ROS Bridge).
Instead of a gamepad or random policy, the sim-robot is driven by the same
planner → VLM → Nav2 pipeline used on the real Jetson:

```text
+--------------------+                   +-------------------------+
| DGX Spark          |                   | DGX Spark (Isaac Sim)   |
|                    |    LAN HTTP       |                         |
|  Planner (:8200)   |<---------------->|  strafer_autonomy       |
|  VLM     (:8100)   |                  |    executor             |
|                    |                   |                         |
+--------------------+                   |  Isaac Sim ROS2 Bridge  |
                                         |    /d555/color          |
                                         |    /d555/aligned_depth  |
                                         |    /strafer/odom        |
                                         |    /cmd_vel             |
                                         |    Nav2 (sim costmap)   |
                                         +-------------------------+
```

The executor runs identically to the real robot — it reads camera topics
published by Isaac Sim's ROS2 bridge, sends them to the VLM for grounding,
projects goals via depth, and navigates via Nav2 using the sim's costmap.

**How it works:**

1. Launch Isaac Sim with a ProcRoom or Infinigen environment and the ROS2
   bridge enabled. The bridge publishes simulated sensor topics on the same
   topic names as the real D555 camera and strafer odometry.

2. Launch the autonomy executor pointing at the DGX planner and VLM services.
   The executor sees the sim's ROS topics and operates as if on the real robot.

3. A harness script generates random goal commands (e.g., "go to the table",
   "go to the door") drawn from the objects actually placed in the scene
   (known from ProcRoom's active mask or Infinigen's label manifest).

4. The executor plans, scans, grounds, navigates — and the harness captures
   every `(frame, VLM_response, ground_truth_label, ground_truth_bbox)`
   tuple for training data. Ground truth comes from the sim's scene graph;
   the VLM response is the model's actual prediction against the sim-rendered
   frame.

**Why this is valuable:**

- **Distribution match.** The collected frames come from the exact viewpoints,
  scan headings, and standoff distances the real robot uses. Random walks
  don't replicate this.
- **Natural failure cases.** When the VLM misses a target or hallucinates a
  detection, the harness captures both the VLM output and the ground truth
  — these are the hardest examples to obtain synthetically and the most
  valuable for fine-tuning.
- **End-to-end validation.** Before deploying a fine-tuned model, run the
  full stack in sim and measure mission success rate. This is a sim-based
  integration test, not just a perception benchmark.
- **No manual annotation.** Ground truth comes from the sim's scene graph.
  VLM predictions come from the actual model. The comparison is automatic.

**Dependencies:** Isaac Sim ROS2 Bridge configured for the D555 camera topics,
Nav2 running against the sim's costmap, and the same DGX services used for
real deployment. This is a follow-up feature that builds on steps 1-3 of the
perception data pipeline.

### 5.10 Implementation plan

| Step | Description | Effort | Dependencies |
|------|-------------|--------|--------------|
| 1 | Perception data collection script (`collect_perception_data.py`) | Medium | Existing Isaac Lab envs |
| 2 | 2D bbox extraction via Replicator annotator or manual projection | Medium | Isaac Sim Replicator API, camera intrinsics |
| 3 | HDF5/WebDataset export pipeline | Small | Step 1 |
| 4 | Label-preserving Infinigen USD export (`prep_room_usds.py` update) | Medium | Infinigen scene graph, USD prim metadata |
| 5 | CLIP contrastive fine-tuning (OpenCLIP + InfoNCE) | Medium | Steps 1-3, DGX/Databricks |
| 6 | VLM grounding fine-tuning (Qwen2.5-VL ref/box format) | Medium | Steps 1-3, DGX/Databricks |
| 7 | ONNX export of fine-tuned CLIP for Jetson | Small | Step 5 |
| 8 | Real-world failure logging in `SemanticMapManager` + JSON manifest | Small | Section 1 (SemanticMapManager) |
| 9 | `gen_failure_scenarios.py` -- manifest-driven scenario spawning | Large | Steps 1-8 |
| 10 | Sim-in-the-loop harness (Isaac Sim ROS2 Bridge + autonomy executor) | Large | Steps 1-3, Isaac Sim ROS2 Bridge, Nav2 sim |

---

## Summary of Proposed Changes

| Area | Item | Effort | Priority |
|------|------|--------|----------|
| Safety | Semantic map arrival verification | Medium | High |
| Safety | VLM confidence thresholding | Small | High |
| Safety | Costmap collision pre-check | Small | Medium |
| Safety | Odometry drift watchdog | Medium | Medium |
| Safety | Velocity limit watchdog | Small | Low |
| Semantic Map | CLIP encoder — unified image + text (OpenCLIP ViT-B/32 ONNX) | Small | High |
| Semantic Map | ChromaDB vector store integration | Small | High |
| Semantic Map | NetworkX spatial graph | Small | High |
| Semantic Map | SemanticMapManager | Medium | High |
| Semantic Map | Observation capture in scan_for_target | Small | High |
| Semantic Map | BackgroundMapper — passive movement-gated capture | Medium | High |
| Semantic Map | `POST /detect_objects` VLM endpoint | Medium | High |
| Semantic Map | Object reinforcement (Bayesian spatial tracking) | Medium | Medium |
| Semantic Map | Query-before-scan optimization | Medium | Medium |
| Semantic Map | Map lifecycle (pruning, reset) | Small | Medium |
| Optimization | Agentic planner+VLM endpoint | Medium | Medium |
| Optimization | Image compression tuning | Small | Low |
| Feature | `rotate_by_degrees` skill | Small | High |
| Feature | `orient_to_direction` skill | Medium | High |
| Feature | Multi-target chaining | Medium | High |
| Feature | Scene description as user intent | Small | Medium |
| Feature | Environment query skill | Small | Medium |
| Feature | Patrol / waypoint sequence | Medium | Low |
| Feature | Plan repair on failure | Large | Low |
| Deployment | Databricks client implementations | Medium | High |
| Deployment | MLflow model packaging | Medium | Medium |
| Deployment | Databricks serving endpoints | Medium | Medium |
| Deployment | Agentic combined endpoint | Medium | Low |
| Synthetic Data | Perception data collection script | Medium | High |
| Synthetic Data | Ground-truth bbox extraction from sim | Medium | High |
| Synthetic Data | HDF5/WebDataset export pipeline | Small | Medium |
| Synthetic Data | CLIP contrastive fine-tuning | Medium | High |
| Synthetic Data | VLM grounding fine-tuning | Medium | Medium |
| Synthetic Data | ONNX export of fine-tuned CLIP | Small | Medium |
| Synthetic Data | Real-world failure logging | Small | Medium |
| Synthetic Data | Failure-to-sim feedback pipeline | Large | Low |
| Synthetic Data | Sim-in-the-loop harness (ROS2 Bridge + executor) | Large | Low |

