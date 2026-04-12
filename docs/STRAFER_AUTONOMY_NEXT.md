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

**Proposal: ranking-based semantic map verification.** After `navigate_to_pose`
completes, verify arrival by asking a *ranking* question — not a threshold
question. Instead of "is cosine similarity > 0.75?" (a magic number that
varies by model, environment, and fine-tuning), ask: "does this location
look more like the goal area than like anywhere else in my map?"

**Verification flow:**

1. Capture a new RGB frame at the arrival pose.
2. Compute a CLIP embedding of the frame.
3. Query ChromaDB for the **top-k nearest neighbors** (k=5) across the
   entire map.
4. Check whether the top results are **spatially clustered near the goal
   pose** (within `goal_radius_m`, default 3.0m).
5. **Decision logic** (threshold-free):
   - If **majority** (>=3 of 5) of top-k results are near the goal pose →
     **pass**. The arrival frame looks most like goal-area observations.
   - If **majority** of top-k results are from a different region → **fail**
     with `arrival_verification_failed`. The robot is somewhere that looks
     like a different part of the map.
   - If top-k results are **mixed** or the map is too sparse to judge →
     **uncertain**. Log warning, pass with `verified: false, reason: inconclusive`.
   - If **no map data** exists near the goal → **skip** (first visit,
     graceful degradation).

**Why ranking beats thresholds:**
- Works regardless of CLIP model variant, fine-tuning, or L2 normalization.
- Transfers across environments without recalibration.
- Leverages the full map as context — the same frame that would score 0.6
  (below a fixed threshold) against one node might still be the *best match*
  for the goal area relative to all alternatives.
- Graceful degradation: with a sparse map, the system is uncertain rather
  than wrong.

**Implementation sketch.** The `verify_arrival` skill is registered in
`DEFAULT_AVAILABLE_SKILLS` and dispatched from `MissionRunner._execute_step`.

```python
SkillCall(
    step_id="step_04",
    skill="verify_arrival",
    args={
        "target_label": runtime.latest_grounding.label,  # set by planner compiler
        "goal_radius_m": 3.0,       # how close top-k results must be to goal
        "top_k": 5,                 # number of nearest neighbors to retrieve
        "majority": 3,              # how many of top_k must be near goal
        "fallback_on_empty_map": "pass",  # "pass" or "fail" when no map data
    },
    timeout_s=10.0,
    retry_limit=1,
)
```

The handler:

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

    # 3. Get the robot's current pose and goal pose.
    robot_state = self._ros_client.get_robot_state()
    robot_pose: dict | None = robot_state["pose"]
    if robot_pose is None:
        return self._failed_result(
            step, "No robot pose available for arrival verification.",
            "no_pose", started_at,
        )

    goal_pose = runtime.latest_goal_pose
    if goal_pose is None:
        return self._failed_result(
            step, "No goal pose available for arrival verification.",
            "no_goal_pose", started_at,
        )
    goal_xy = np.array([goal_pose.goal_pose.x, goal_pose.goal_pose.y])

    # 4. CLIP encode arrival frame and retrieve top-k from entire map.
    arrival_embedding = self._semantic_map.clip_encoder.encode_image(arrival_image_rgb)
    top_k = int(step.args.get("top_k", 5))
    goal_radius_m = float(step.args.get("goal_radius_m", 3.0))
    majority = int(step.args.get("majority", 3))

    results = self._semantic_map.query_by_embedding(
        embedding=arrival_embedding, n_results=top_k,
    )
    # Returns list of (node, similarity) sorted by similarity descending.

    if not results:
        fallback = str(step.args.get("fallback_on_empty_map", "pass"))
        if fallback == "pass":
            _logger.warning("No semantic map data; skipping verification.")
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

    # 5. Ranking decision: are top-k results near the goal?
    near_goal_count = 0
    for node, sim in results:
        node_xy = np.array([node.pose.x, node.pose.y])
        if np.linalg.norm(node_xy - goal_xy) <= goal_radius_m:
            near_goal_count += 1

    if near_goal_count >= majority:
        return SkillResult(
            step_id=step.step_id, skill=step.skill, status="succeeded",
            outputs={
                "verified": True,
                "near_goal_count": near_goal_count,
                "top_k": top_k,
            },
            message=f"Arrival verified ({near_goal_count}/{top_k} top matches near goal).",
            started_at=started_at, finished_at=time.time(),
        )
    else:
        # Where DO the top results think we are?
        top_regions = [
            f"({n.pose.x:.1f},{n.pose.y:.1f})" for n, _ in results[:3]
        ]
        return self._failed_result(
            step,
            f"Arrival mismatch: only {near_goal_count}/{top_k} top matches "
            f"near goal. Top-3 matches at: {', '.join(top_regions)}.",
            "arrival_verification_failed", started_at,
        )
```

**Where this runs.** CLIP embedding is computed locally on the Jetson
(OpenCLIP ViT-B/32 ONNX, see Section 1.8). ChromaDB queries are local.
The VLM is not needed for verification — this is a pure embedding retrieval
check, keeping the safety decision entirely on the robot.

**`query_by_embedding` API.** `SemanticMapManager` must expose a method
that queries ChromaDB by raw embedding vector (not text, not spatial):

```python
def query_by_embedding(
    self, embedding: np.ndarray, n_results: int = 5,
) -> list[tuple[SemanticNode, float]]:
    """Retrieve top-n nodes most similar to the given CLIP embedding."""
    results = self._collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=n_results,
    )
    # Unpack ChromaDB response into (SemanticNode, similarity) pairs
    ...
```

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

### 0.6 Transit monitoring via BackgroundMapper

**Problem.** Between `scan_for_target` and `verify_arrival`, the robot
drives blind for 30-90 seconds. If it goes off course (bad projection,
RTAB-Map drift, Nav2 replanning around obstacles), nothing detects the
error until arrival — wasting the entire transit.

**Proposal: ranking-based transit monitoring.** Piggyback on
`BackgroundMapper`'s existing 2-second capture cycle during active
navigation. At each capture, ask: "which part of my map does my current
frame look most like?" Track how the **nearest-neighbor region** shifts
over consecutive captures:

```text
Expected progression (robot driving from dining room to living room):
  t=0s:  Top-3 nearest → all dining room nodes
  t=4s:  Top-3 → 2 dining room, 1 hallway         (transitioning)
  t=8s:  Top-3 → 2 hallway, 1 living room          (on track)
  t=12s: Top-3 → 3 living room nodes near goal      (approaching ✓)

Failure case (robot takes wrong turn into bedroom):
  t=0s:  Top-3 → all dining room nodes
  t=4s:  Top-3 → 2 dining room, 1 hallway
  t=8s:  Top-3 → 2 hallway, 1 bedroom               (diverging!)
  t=12s: Top-3 → 3 bedroom nodes                     (wrong room → abort)
```

**No thresholds.** The decision is purely relative — "are my nearest
neighbors getting closer to the goal, or drifting into a different
region?" This transfers across models, environments, and fine-tuning.

**TransitMonitor integration with BackgroundMapper:**

```python
class TransitMonitor:
    """Tracks nearest-neighbor region drift during navigation.

    Activated when navigate_to_pose starts. Piggybacks on
    BackgroundMapper's 2-second capture cycle.
    """

    def __init__(self, semantic_map: SemanticMapManager):
        self._semantic_map = semantic_map
        self._goal_xy: np.ndarray | None = None
        self._goal_radius_m: float = 3.0
        self._history: list[dict] = []
        self._consecutive_divergences: int = 0
        self._active = False

    def activate(self, goal_pose, goal_radius_m: float = 3.0):
        """Called by MissionRunner when navigate_to_pose starts."""
        self._goal_xy = np.array([goal_pose.x, goal_pose.y])
        self._goal_radius_m = goal_radius_m
        self._history = []
        self._consecutive_divergences = 0
        self._active = True

    def deactivate(self):
        """Called when navigation completes or is canceled."""
        self._active = False
        self._goal_xy = None

    def check(self, clip_embedding: np.ndarray, robot_xy: np.ndarray) -> dict:
        """Called by BackgroundMapper at each capture during navigation.

        Returns status dict: {"on_track": bool, "abort": bool, ...}
        """
        if not self._active or self._goal_xy is None:
            return {"on_track": True, "abort": False, "reason": "inactive"}

        # Query top-3 nearest neighbors in the map
        results = self._semantic_map.query_by_embedding(
            embedding=clip_embedding, n_results=3,
        )

        if len(results) < 2:
            # Map too sparse to judge
            return {"on_track": True, "abort": False, "reason": "sparse_map"}

        # How many of the top-3 are near the goal area?
        near_goal = sum(
            1 for node, _ in results
            if np.linalg.norm(
                np.array([node.pose.x, node.pose.y]) - self._goal_xy
            ) <= self._goal_radius_m
        )

        # How many are near where the robot currently is?
        near_robot = sum(
            1 for node, _ in results
            if np.linalg.norm(
                np.array([node.pose.x, node.pose.y]) - robot_xy
            ) <= 2.0
        )

        snapshot = {
            "robot_xy": robot_xy.tolist(),
            "near_goal": near_goal,
            "near_robot": near_robot,
            "top_regions": [
                (n.pose.x, n.pose.y) for n, _ in results
            ],
        }
        self._history.append(snapshot)

        # Divergence detection: top matches are NOT near the goal
        # AND NOT near the robot's expected path toward the goal.
        # If all top matches are from an unrelated region for 3+
        # consecutive captures, the robot has gone off course.
        if near_goal == 0 and len(self._history) >= 3:
            # Check if the top-match region is consistent but wrong
            recent = self._history[-3:]
            all_off_course = all(h["near_goal"] == 0 for h in recent)
            if all_off_course:
                self._consecutive_divergences += 1
                if self._consecutive_divergences >= 1:
                    return {
                        "on_track": False,
                        "abort": True,
                        "reason": "transit_divergence",
                        "message": (
                            f"Top-3 matches from wrong region for "
                            f"{len(recent)} consecutive captures. "
                            f"Latest top matches at: {snapshot['top_regions']}"
                        ),
                    }
        else:
            self._consecutive_divergences = 0

        return {"on_track": True, "abort": False, "reason": "ok"}
```

**BackgroundMapper integration:**

During active navigation, `BackgroundMapper` calls
`transit_monitor.check()` at each capture and reports divergence via a
thread-safe flag that `MissionRunner` polls:

```python
# In BackgroundMapper._capture_loop (during active navigation):
clip_emb = self._semantic_map.clip_encoder.encode_image(image_rgb)
self._semantic_map.add_observation(pose=pose, clip_embedding=clip_emb, source="background")

if self._transit_monitor.is_active:
    status = self._transit_monitor.check(clip_emb, robot_xy)
    if status["abort"]:
        self._divergence_flag.set()  # threading.Event
        _logger.warning("Transit divergence: %s", status["message"])
```

```python
# In MissionRunner._navigate_to_pose (polling loop):
while not result_future.done():
    if self._background_mapper.divergence_detected():
        goal_handle.cancel_goal_async()
        return SkillResult(
            status="failed",
            error_code="transit_divergence",
            message="Navigation aborted: robot appears off course.",
        )
    time.sleep(0.5)
```

**First-visit limitation.** With an empty or sparse map, transit
monitoring gracefully degrades — it returns `"sparse_map"` and does
not abort. The system builds the map during the first traversal via
BackgroundMapper, and subsequent navigations benefit from transit
monitoring. This is inherent to any memory-based approach: you can't
validate against observations you've never made.

**Async and non-blocking.** Transit monitoring adds no latency to
navigation. CLIP encoding (~20ms on Jetson ONNX) already happens in
BackgroundMapper's thread. The check is a ChromaDB query (~5ms) on
top of work already being done.

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

**On ranking vs. thresholds.** All CLIP-based decisions in the autonomy
stack use **ranking** (retrieval + spatial clustering) rather than fixed
cosine similarity thresholds. A hardcoded threshold like `> 0.75` is:
- Dependent on the model variant, fine-tuning, and L2 normalization
- Fragile across environments (a score of 0.6 might be "clearly the same
  room" in one building and "barely related" in another)
- Not transferable — requires recalibration after every model update

Ranking-based decisions ask "which part of my map does this look like
most?" rather than "is this score above some number?" This transfers
across models, environments, and fine-tuning runs because it uses the
map itself as context.

See Section 0.1 (verify_arrival), Section 0.6 (transit monitoring), and
query-before-scan (Section 1.7) for the specific ranking approaches.

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
and short-circuits the full scan when the target was recently observed.

Uses **ranking** instead of a fixed similarity threshold: capture the
current frame, CLIP encode it, and ask "is the stored target node the
top retrieval result for my current view?" If the target's stored
observation is the best match in the entire map, the environment still
looks like it did when the target was last seen — no need to re-scan.

```python
def _scan_for_target(self, runtime: _MissionRuntime, step: SkillCall) -> SkillResult:
    import math

    started_at = time.time()
    label = str(step.args.get("label", "")).strip()
    if not label:
        return self._failed_result(step, "scan_for_target is missing 'label'.", "invalid_args", started_at)

    # --- NEW: query-before-scan short-circuit (ranking-based) ---
    max_map_age_s = float(step.args.get("max_map_age_s", 300))

    if self._semantic_map is not None:
        known = self._semantic_map.query_by_label(label, max_age_s=max_map_age_s)
        if known is not None:
            # Target was seen recently -- verify via ranking
            try:
                observation = self._ros_client.capture_scene_observation()
                with self._lock:
                    runtime.latest_observation = observation

                image_rgb = self._bgr_to_rgb(observation.color_image_bgr)
                current_clip = self._semantic_map.clip_encoder.encode_image(image_rgb)

                # Ask: is the target's stored node the top retrieval match
                # for what I'm currently seeing?
                top_results = self._semantic_map.query_by_embedding(
                    embedding=current_clip, n_results=3,
                )

                if top_results:
                    top_node, top_sim = top_results[0]
                    target_xy = np.array([known.pose.x, known.pose.y])
                    top_xy = np.array([top_node.pose.x, top_node.pose.y])
                    top_near_target = np.linalg.norm(top_xy - target_xy) < 2.0

                    if top_near_target:
                        _logger.info(
                            "Query-before-scan hit: label=%s, top match near target node=%s",
                            label, top_node.node_id,
                        )
                        return SkillResult(
                            step_id=step.step_id,
                            skill=step.skill,
                            status="succeeded",
                            outputs={
                                "source": "semantic_map",
                                "node_id": known.node_id,
                                "pose_x": known.pose.x,
                                "pose_y": known.pose.y,
                                "pose_yaw": known.pose.yaw,
                            },
                            message=f"Target '{label}' found in semantic map (top match near target), skipping scan.",
                            started_at=started_at,
                            finished_at=time.time(),
                        )
                    else:
                        _logger.info(
                            "Query-before-scan miss: label=%s, top match at (%.1f,%.1f) not near target (%.1f,%.1f), falling through.",
                            label, top_xy[0], top_xy[1], target_xy[0], target_xy[1],
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


**Platform allocation.** The synthetic data pipeline spans three machines:

| Workload | Platform | Why |
|----------|----------|-----|
| Infinigen scene generation + metadata extraction | DGX Spark (128GB unified) | High-quality scenes need VRAM; metadata extraction needs Blender Python state |
| Scene description pipeline (Stages 1-2) | DGX Spark | VLM (Qwen2.5-VL-7B) loaded standalone via transformers |
| Isaac Sim + teleop data collection | DGX Spark or Windows workstation | Whichever machine runs Isaac Sim; DGX preferred for VRAM headroom |
| Replicator bbox extraction | Runs with Isaac Sim | Same machine as data collection |
| CLIP / VLM fine-tuning | DGX Spark | GPU training |
| Model serving (planner, VLM) | DGX Spark | Already deployed |
| RL policy training | Windows workstation or DGX Spark | Existing Isaac Lab setup |

The DGX Spark (Grace Blackwell, 128GB+ unified memory) can run Infinigen
generation, Isaac Sim, and model serving concurrently. The Windows workstation
(GTX 4080, 16GB VRAM) serves as a secondary/backup or continues RL training.


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
and pose **inside the Infinigen pipeline**. The metadata is richer than
originally assumed.

**What Infinigen provides during generation (Blender Python `State` object):**

| Metadata | Type | Example |
|----------|------|---------|
| Room type | `Semantics` enum | Kitchen, Bedroom, LivingRoom, Hallway, Bathroom, Office, etc. |
| Room footprint | `shapely.Polygon` | 2D boundary geometry per room |
| Room dimensions | `(width, height, wall_height)` | Per-room area and height |
| Room connectivity | `RoomGraph` adjacency matrix | Which rooms connect to which |
| Multi-story | `GroundFloor`, `SecondFloor` tags | Floor/story assignment |
| Object semantic tags | `set[Semantics]` | `{Furniture, Seating, Chair}` (hierarchical) |
| Spatial relations | `RelationState` list | `SupportedBy`, `StableAgainst`, `CoPlanar`, `Touching` |
| Object 3D bbox | 8-corner vertex array | From `saved_mesh.json` / `instance_bbox` |
| Materials | Material slot names | Per-object material assignments |
| Instance hierarchy | Parent-child tree | `children` list per object |

**Semantic tag categories available:**

- **Room types:** Kitchen, Bedroom, LivingRoom, DiningRoom, Closet, Hallway,
  Bathroom, Garage, Balcony, Utility, StaircaseRoom, Warehouse, Office,
  MeetingRoom, OpenOffice, BreakRoom, Restroom, FactoryOffice
- **Furniture functions:** Storage, Seating, LoungeSeating, Table, Bathing,
  SideTable, Desk, Bed, Sink, CeilingLight, Lighting, KitchenCounter,
  KitchenAppliance
- **Small objects:** TableDisplayItem, OfficeShelfItem, KitchenCounterItem,
  FoodPantryItem, BathroomItem, ShelfTrinket, Dishware, Cookware, Utensils
- **Spatial:** Door, Window, Entrance, WallDecoration, FloorMat

**What does NOT survive export to USD by default:**

- Semantic tags and spatial relations live in Blender's Python `State` object
  during generation -- they are NOT written to `saved_mesh.json` or USD
- `saved_mesh.json` has object names, bboxes, materials, hierarchy -- but
  not the `Semantics` tags or `RelationState` data
- Segmentation masks encode labels via integer IDs mapped through
  `MaskTag.json`, but not in a form Replicator can use directly

**What needs to be built:**

1. **Metadata extraction during Blender generation.** Hook into Infinigen's
   generation pipeline to serialize the rich Python state before/after USD
   export. This writes room types, object tags, spatial relations, and room
   geometry to a structured JSON alongside the USD:

   ```python
   # During Infinigen scene generation (Blender Python)
   def extract_scene_metadata(state, output_path: Path):
       metadata = {
           "rooms": [],
           "objects": [],
           "room_adjacency": state.room_graph.adjacency,
       }
       for room in state.rooms:
           metadata["rooms"].append({
               "room_type": room.semantics.name,  # "Kitchen", "Hallway", etc.
               "footprint_xy": list(room.polygon.exterior.coords),
               "area_m2": room.polygon.area,
               "story": room.story,
           })
       for obj in state.objects:
           tags = [t.name for t in obj.tags if isinstance(t, Semantics)]
           relations = []
           for rel in obj.relations:
               relations.append({
                   "type": type(rel).__name__,  # "SupportedBy", "StableAgainst"
                   "target": rel.target.name if rel.target else None,
               })
           metadata["objects"].append({
               "label": infer_label(obj),  # map tags to human-readable label
               "semantic_tags": tags,
               "instance_id": obj.instance_id,
               "prim_path": f"/World/Room/{obj.name}",
               "position_3d": obj.position.tolist(),
               "bbox_3d": {"min": obj.bbox_min.tolist(), "max": obj.bbox_max.tolist()},
               "room_idx": obj.room_index,
               "relations": relations,
               "materials": [m.name for m in obj.materials],
           })
       with open(output_path / "scene_metadata.json", "w") as f:
           json.dump(metadata, f, indent=2)
   ```

2. **USD prim labeling.** Write `semanticLabel` and `instanceId` as custom
   USD prim attributes on each object so Replicator annotators
   (`bounding_box_2d_tight`, `semantic_segmentation`) can produce labeled
   output:
   ```python
   from pxr import Usd, Sdf
   stage = Usd.Stage.Open(scene_usd_path)
   for prim in stage.Traverse():
       if prim.GetTypeName() in ("Mesh", "Xform"):
           label = metadata_lookup(prim.GetName())
           if label:
               prim.CreateAttribute("semanticLabel", Sdf.ValueTypeNames.String).Set(label)
               prim.CreateAttribute("instanceId", Sdf.ValueTypeNames.Int).Set(instance_id)
   stage.Save()
   ```

3. **Runtime label accessor** -- a utility that reads `scene_metadata.json`
   and provides room types, label sets, object positions, spatial relations,
   and room footprints for data collection scripts and the description
   pipeline (Section 5.6).

**Resulting `scene_metadata.json` structure:**

```json
{
  "rooms": [
    {
      "room_type": "Kitchen",
      "footprint_xy": [[0,0], [4,0], [4,3], [0,3]],
      "area_m2": 12.0,
      "story": 0
    },
    {
      "room_type": "Hallway",
      "footprint_xy": [[4,0], [6,0], [6,3], [4,3]],
      "area_m2": 6.0,
      "story": 0
    }
  ],
  "room_adjacency": [[0, 1], [1, 0]],
  "objects": [
    {
      "label": "table",
      "semantic_tags": ["Furniture", "KitchenCounter"],
      "instance_id": 1,
      "prim_path": "/World/Room/table_001",
      "position_3d": [2.0, 1.5, 0.0],
      "bbox_3d": {"min": [1.5, 1.0, 0.0], "max": [2.5, 2.0, 0.8]},
      "room_idx": 0,
      "relations": [
        {"type": "SupportedBy", "target": null},
        {"type": "StableAgainst", "target": "wall_segment_3"}
      ],
      "materials": ["wood_oak", "wood_finish"]
    }
  ]
}
```

Note: ProcRoom scenes use solid-color primitives (boxes, cylinders, cones)
that are not useful for VLM or CLIP perception training. ProcRoom data is
valuable for RL policy training and depth/navigation data but must be
excluded from perception fine-tuning datasets. Infinigen scenes with
realistic labeled objects are the primary source for all perception data.


### 5.4 Fine-tuning targets

| Model | Training data from sim | Fine-tune goal | Impact on autonomy stack |
|-------|----------------------|----------------|--------------------------|
| **OpenCLIP ViT-B/32** | RGB frames from Infinigen rooms, rendered from robot camera height and FOV with D555 noise model | Domain-adapted visual embeddings tuned to indoor scenes from the robot's perspective | Better semantic map retrieval, more accurate arrival verification (Section 0.1), stronger visual place recognition |
| **VLM (Qwen2.5-VL)** | RGB frames + ground-truth 2D bboxes from sim object poses | Improved grounding accuracy for objects common in the operating environment | Fewer missed detections during `scan_for_target`, fewer hallucinated bboxes |
| **Goal projection** | Depth frames with known ground-truth object distances | More accurate standoff distance estimation from depth + bbox | Robot stops closer to the intended distance from the target |
| **RL navigation policy** | Full episodes with RGB/depth observations (existing Role 1) | Reactive obstacle avoidance in cluttered environments | Smoother navigation in dynamic scenes |


### 5.5 Synthetic data collection pipeline

A perception data collection script, modeled after the existing
`collect_demos.py` but capturing (RGB, depth, pose, bbox, label) tuples
instead of (obs, action, reward) tuples. The code below uses the actual
strafer_lab API surface.

#### 5.5.1 Controller strategy

The controller determines how the robot moves through environments during
data collection. The quality of perception training data depends heavily on
the viewpoint distribution matching real deployment.

**Why scripted strategies fail:**

| Strategy | Problem |
|----------|---------|
| Random walk | Collides with obstacles, gets stuck in corners, doesn't explore meaningfully |
| Goal-seeking | No obstacle avoidance -- walks straight through furniture |
| Scripted patrol | Fixed waypoints, collides along the way |
| Trained RL policy | Trained on 80x60, collision-heavy, doesn't match real navigation behavior |

None of these produce trajectories resembling real deployment, where the
robot navigates around obstacles, through doorways, along hallways.

**Two-phase approach:**

| Phase | Controller | Scale | When |
|-------|-----------|-------|------|
| **Phase A (MVP)** | Human gamepad teleop | 50-100 episodes | Now (Phase 15) |
| **Phase B (Scalable)** | ROS bridge + Nav2 autonomy | 1000+ episodes | After bridge setup (Section 5.9) |

**Phase A** uses the existing `collect_demos.py` gamepad teleop infrastructure
(world-to-body frame transform, HDF5 output pattern) adapted for perception
data capture. A human operator drives the robot through Infinigen scenes,
producing trajectories that naturally match real deployment behavior.

**Phase B** runs the full autonomy stack (Nav2 MPPI, RTAB-Map or static map)
against the simulated robot via the Isaac Sim ROS2 Bridge (Section 5.9).
The executor code is identical to real deployment -- same topics, same TF
tree, same planner/VLM calls. This produces data at the exact deployment
distribution with zero human effort.

#### 5.5.2 Camera resolution

The current sim camera in `d555_cfg.py` outputs 80x60 for RL policy
training. Perception training requires 640x360 -- the resolution used on the
real robot for VLM grounding and depth-based goal projection.

**Solution: dual camera configuration.**

```python
# d555_cfg.py -- add a second camera for perception data collection
D555_PERCEPTION_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link/d555_camera_perception",
    update_period=D555_CAMERA_UPDATE_PERIOD,  # 1/30 = 30 Hz
    height=360,
    width=640,
    data_types=("rgb", "distance_to_image_plane"),
    spawn=PinholeCameraCfg(
        focal_length=D555_FOCAL_LENGTH,       # 1.93 mm
        horizontal_aperture=D555_H_APERTURE,  # 3.68 mm
        clipping_range=(0.01, DEPTH_CLIP_FAR),
    ),
    offset=OffsetCfg(
        pos=D555_CAMERA_OFFSET_POS,   # (0.20, 0.0, 0.25)
        rot=D555_CAMERA_OFFSET_ROT,   # ROS convention quaternion
        convention="ros",
    ),
)
```

The 80x60 policy camera remains for RL training. The 640x360 perception
camera is only instantiated in perception data collection environments and
the ROS bridge. At 640x360 with a single environment, Isaac Sim renders
efficiently -- but only 1-8 environments can run simultaneously at this
resolution (vs. 64+ at 80x60).

#### 5.5.3 ProcRoom exclusion from perception pipeline

ProcRoom environments use solid-color primitive shapes (boxes, cylinders,
cones, capsules) with randomized colors. These are **not useful** for VLM
or CLIP perception training:

- A VLM fine-tuned on "locate the cyan box" won't generalize to real tables
- CLIP embeddings of solid-color shapes don't transfer to real rooms
- No `semanticLabel` USD prim attributes -- Replicator cannot produce
  labeled bboxes

ProcRoom remains valuable for RL policy training (obstacle avoidance,
navigation) and depth-based distance estimation, but **all perception
training data must come from Infinigen scenes** with realistic textures,
materials, lighting, and labeled objects.

Data collection scripts must tag frames with `"scene_type": "infinigen"` or
`"scene_type": "procroom"`. The export pipeline (Section 5.5.5) excludes
ProcRoom frames from CLIP and VLM training datasets.

#### 5.5.4 PerceptionDataCollector

```python
# scripts/collect_perception_data.py

import torch
import numpy as np
from pathlib import Path

class PerceptionDataCollector:
    """
    Human-teleoperated perception data collection through Infinigen scenes.

    Uses the 640x360 perception camera (not the 80x60 policy camera)
    and Replicator annotators for ground-truth bboxes.

    Modeled after collect_demos.py but captures (RGB, depth, pose, bbox,
    label) tuples instead of (obs, action, reward) tuples.
    """

    def __init__(self, env, output_dir: Path, scene_name: str):
        self.env = env
        self.output_dir = output_dir
        self.scene_name = scene_name
        # 640x360 perception camera (NOT the 80x60 policy camera)
        self.camera = env.scene["d555_camera_perception"]
        self.bbox_extractor = ReplicatorBboxExtractor(
            self.camera.render_product_path
        )
        self.gamepad = GamepadTeleop()  # world-to-body frame transform

    def collect_episode(self, episode_id: int, max_steps: int = 500) -> dict:
        obs_dict, info = self.env.reset()
        frames = []

        for step in range(max_steps):
            # Human drives via gamepad (same as collect_demos.py)
            action = self.gamepad.get_action()
            if action is None:
                break  # operator ended episode
            obs_dict, reward, terminated, truncated, info = self.env.step(action)

            # --- Camera data (640x360 perception camera) ---
            # Shape: (num_envs, 360, 640, 4) RGBA uint8
            rgba = self.camera.data.output["rgb"]
            rgb = rgba[..., :3]  # drop alpha -> (num_envs, 360, 640, 3)
            # Shape: (num_envs, 360, 640, 1) float32 meters
            depth = self.camera.data.output["distance_to_image_plane"]

            # --- Poses ---
            cam_pos = self.camera.data.pos_w           # (num_envs, 3)
            cam_quat = self.camera.data.quat_w_world   # (num_envs, 4)
            robot_pos = self.env.scene["robot"].data.root_pos_w
            robot_quat = self.env.scene["robot"].data.root_quat_w

            # --- Ground-truth bboxes from Replicator ---
            bboxes = self.bbox_extractor.extract()

            frame_data = {
                "rgb": rgb[0].cpu().numpy(),         # (360, 640, 3)
                "depth": depth[0].cpu().numpy(),     # (360, 640, 1)
                "cam_pos": cam_pos[0].cpu().numpy(),
                "cam_quat": cam_quat[0].cpu().numpy(),
                "robot_pos": robot_pos[0].cpu().numpy(),
                "robot_quat": robot_quat[0].cpu().numpy(),
                "bboxes": bboxes,
                "scene_type": "infinigen",
                "scene_name": self.scene_name,
            }
            frames.append(frame_data)

            if terminated.any() or truncated.any():
                break

        return self.save_episode(episode_id, frames)
```

**CLI interface** (follows `collect_demos.py` pattern):

```bash
python scripts/collect_perception_data.py \
  --task Isaac-Strafer-Nav-Real-Infinigen-Depth-Play-v0 \
  --scene scene_001 \
  --output data/perception/ \
  --max_episodes 20 \
  --max_steps 500
```

**Ground-truth 2D bounding boxes** are extracted via the Isaac Sim
Replicator API (`bounding_box_2d_tight` annotator). This requires:
1. The 640x360 perception camera wired to a render product
2. `semanticLabel` USD prim attributes on Infinigen objects (Section 5.3)

```python
import omni.replicator.core as rep

class ReplicatorBboxExtractor:
    """Extract 2D bounding boxes and semantic labels via Replicator."""

    def __init__(self, camera_render_product_path: str):
        self._bbox_annotator = rep.AnnotatorRegistry.get_annotator(
            "bounding_box_2d_tight"
        )
        self._bbox_annotator.attach([camera_render_product_path])

    def extract(self) -> list[dict]:
        bbox_data = self._bbox_annotator.get_data()
        results = []
        for bbox_entry in bbox_data["data"]:
            results.append({
                "label": bbox_entry.get("semanticLabel", "unknown"),
                "bbox_2d": [
                    int(bbox_entry["x_min"]),
                    int(bbox_entry["y_min"]),
                    int(bbox_entry["x_max"]),
                    int(bbox_entry["y_max"]),
                ],
                "instance_id": bbox_entry.get("instanceId", -1),
                "occlusion": bbox_entry.get("occlusionRatio", 0.0),
            })
        return results
```

#### 5.5.5 Output format and export

HDF5 per episode, with metadata for filtering:

```text
episode_NNNN/
  frame_0000.jpg          # RGB (640x360)
  frame_0000_depth.png    # uint16 depth in mm
  frame_0000.json         # camera pose, robot pose, bboxes, scene metadata
  ...
```

The export pipeline produces two downstream-ready datasets:

1. **VLM grounding JSONL** -- convert bboxes to 0-1000 scaled
   `<ref>label</ref><box>(x1,y1),(x2,y2)</box>` format with negative
   examples (Section 5.7). **Excludes ProcRoom frames.**

2. **CLIP contrastive CSV** -- generate (anchor, positive, negative) image
   triples for InfoNCE training (Section 5.6). **Excludes ProcRoom frames.**



### 5.6 CLIP fine-tuning with image-text contrastive data

The semantic map (Section 1) relies on CLIP embeddings for two purposes:

1. **Visual place recognition** (image-to-image) -- `BackgroundMapper`
   embeds frames and compares against stored embeddings to recognize
   previously visited locations.

2. **Text-to-image semantic queries** (text-to-image) --
   `SemanticMapManager.query_by_text("kitchen")` encodes text with CLIP's
   text tower and searches ChromaDB for matching image embeddings. This
   powers query-before-scan optimization and the `environment_query` skill.

Out-of-the-box CLIP was trained on internet images -- it has no special
understanding of indoor robot perspectives at 25 cm camera height with a
D555 FOV. Fine-tuning must adapt both the image-text alignment (use case 2)
and the image-image discrimination (use case 1).

**Why image-text alignment is the primary objective:** If we only fine-tune
with image triples (anchor/positive/negative), we adapt the vision tower to
distinguish indoor scenes but risk drifting it away from the text tower's
embedding space. After fine-tuning, `encode_image(kitchen_photo)` might land
far from `encode_text("kitchen")`, breaking text queries.

#### 5.6.1 Training approach

**Phase 1 (primary): Image-text contrastive** -- standard CLIP InfoNCE loss
on (image, text description) pairs. Both towers trained jointly:

```python
import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Standard CLIP contrastive loss (in-batch negatives)
for images, texts in dataloader:
    image_emb = model.encode_image(images)
    text_emb = model.encode_text(tokenizer(texts))

    # Symmetric cross-entropy on image-text similarity matrix
    logits = image_emb @ text_emb.T * model.logit_scale.exp()
    labels = torch.arange(len(images), device=logits.device)
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2
    loss.backward()
    optimizer.step()
```

**Phase 2 (optional follow-up): Image-image contrastive** -- if place
recognition isn't accurate enough after Phase 1, add NT-Xent loss on
(anchor, positive, negative) image triples from same/different rooms.
This preserves text alignment while sharpening visual discrimination.

#### 5.6.2 Scene description generation pipeline (5 stages)

Rich text descriptions are critical. Simple label lists ("chair, table,
bookshelf") produce shallow embeddings. Spatial and relational descriptions
("a dimly lit hallway opening into a bright kitchen, wooden table with a
cutting board on it") encode scene structure that enables queries like
"where was the chair near the window?"

The pipeline runs **after** teleop data collection, as a batch process on
the DGX Spark. It uses simulation ground truth to produce factually correct
descriptions, then refines them with models for natural language quality.

```text
Stage 1: Programmatic spatial analysis (no model, instant)
    Input:  frame bboxes + 3D positions + robot pose + scene_metadata.json
    Output: structured spatial facts JSON

    Computes: room type (point-in-polygon from room footprints),
    object distances/bearings from robot, pairwise relations
    (from Infinigen's SupportedBy/StableAgainst/CoPlanar data),
    region classification (near/midway/far from robot)

Stage 2: VLM description generation (Qwen2.5-VL-7B on DGX)
    Input:  structured spatial facts JSON + actual RGB frame
    Output: 3-5 diverse natural language descriptions per frame

    The VLM receives both the structured facts (room type,
    object positions, bearings, spatial relations) and the
    image in a single prompt. This lets it produce descriptions
    that are spatially accurate (from GT facts) AND visually
    grounded (lighting, textures, occlusion, materials).

    Produces descriptions like "a dimly lit hallway opening into
    a bright kitchen, wooden table with a cutting board on it,
    small potted plant tucked beside the counter, partially
    occluded by the doorframe"

    Critical: must be a stronger model than Qwen2.5-VL-3B (the
    model being fine-tuned). Qwen2.5-VL-7B fits on DGX Spark
    alongside other models (128GB unified memory).

    The VLM is loaded standalone via transformers
    (AutoModelForVision2Seq), completely separate from the
    strafer_vlm service package on port 8100.

Stage 3: Ground truth validation filter
    Input:  VLM descriptions + scene_metadata.json label set
    Output: validated descriptions (discard any mentioning
            objects not in scene metadata)

    Catches VLM hallucinations without human review.

Stage 4: Human spot-check (random sampling, periodic)
    Input:  50 random samples per batch of 1000 frames
    Output: quality scores (1-5 on accuracy, naturalness, detail),
            systematic error flags, prompt refinements for Stage 2

    Builds a scored validation set over time for automated
    quality evaluation without manual review.
```

#### 5.6.3 Spatial description builder (Stage 1)

Uses simulation ground truth to compute factual spatial assertions. Room
type and spatial relationships come directly from `scene_metadata.json`
(Section 5.3), not from inference or heuristics.

```python
from shapely.geometry import Point, Polygon

class SpatialDescriptionBuilder:
    """Compute spatial relations from simulation ground truth."""

    def __init__(self, scene_metadata: dict):
        self.rooms = [
            {**r, "polygon": Polygon(r["footprint_xy"])}
            for r in scene_metadata["rooms"]
        ]
        self.objects = scene_metadata["objects"]

    def build(self, frame_data: dict) -> dict:
        robot_xy = frame_data["robot_pos"][:2]
        robot_yaw = quat_to_yaw(frame_data["robot_quat"])

        # Which room is the robot in? (point-in-polygon)
        current_room = None
        for room in self.rooms:
            if room["polygon"].contains(Point(robot_xy)):
                current_room = room
                break

        described_objects = []
        for obj in self._visible_objects(frame_data["bboxes"]):
            pos = np.array(obj["position_3d"][:2])
            dist = float(np.linalg.norm(pos - robot_xy))
            bearing = self._compute_bearing(pos, robot_xy, robot_yaw)
            region = self._classify_region(dist)

            # Spatial relations from Infinigen ground truth
            relations = []
            for rel in obj.get("relations", []):
                if rel["target"]:
                    relations.append(f'{rel["type"]} {rel["target"]}')

            # Which room is this object in?
            obj_room = self.rooms[obj["room_idx"]] if "room_idx" in obj else None

            described_objects.append({
                "label": obj["label"],
                "semantic_tags": obj.get("semantic_tags", []),
                "distance_m": round(dist, 1),
                "bearing": bearing,
                "region": region,
                "room_type": obj_room["room_type"] if obj_room else None,
                "relations": relations,
                "materials": obj.get("materials", []),
            })

        return {
            "robot_room_type": current_room["room_type"] if current_room else None,
            "visible_objects": described_objects,
        }
```

#### 5.6.4 Description data format

The export pipeline produces multi-description CSV for CLIP training:

```csv
image_path,description
episode_0001/frame_0010.jpg,"a hallway with a plant and red ball at the far end"
episode_0001/frame_0010.jpg,"looking down a dimly lit hallway, table on the left, potted plant beside a red ball in the distance"
episode_0001/frame_0010.jpg,"indoor hallway with nearby table and distant plant"
episode_0001/frame_0012.jpg,"bright kitchen with wooden counter and cutting board"
```

Multiple descriptions per image (3-5) at different detail levels train CLIP
to match both broad queries ("hallway") and specific ones ("hallway with
plant beside red ball at far end").

**Dataset scale.** 50-100 teleop episodes x 500 frames x 3-5 descriptions
= 75k-250k (image, text) pairs. This scale is sufficient for CLIP
fine-tuning of both towers.

**Training infrastructure.** DGX Spark for training. Databricks for tracked
experiments via MLflow. The fine-tuned model is exported to ONNX — **both
towers separately** (`clip_visual.onnx` + `clip_text.onnx`) since the
Jetson's `clip_encoder.py` uses both `encode_image()` and `encode_text()`
for image-image verification and text-to-image semantic queries respectively.

**Evaluation gate for arrival verification.** Image-text contrastive training
directly optimizes text-to-image alignment (for `query_by_text`). Arrival
verification, however, uses image-to-image ranking — a property that is
indirectly preserved but not directly trained. After Phase 1 training,
evaluate the ranking-based verify_arrival logic on held-out sim data:
construct (arrival_frame, goal_area_observations, distractor_observations)
tuples and measure whether top-k retrieval correctly identifies the goal
region. If ranking accuracy (majority of top-k near goal) is below 85%,
proceed to Phase 2: add NT-Xent loss on (anchor, positive, negative) image
triples from same/different locations to sharpen visual discrimination.


### 5.7 VLM fine-tuning with sim ground truth

The VLM (Qwen2.5-VL-3B) handles object grounding during `scan_for_target`,
multi-object detection for semantic map population (`detect_objects`), and
scene description for operator readback (`describe_scene`). Fine-tuning on
domain-specific data improves detection of objects common in the robot's
operating environment.

**Fine-tuning method.** Use **LoRA** (rank 16-32) rather than full
fine-tuning to preserve the pretrained model's general capabilities (scene
description, OCR, reasoning) while specializing grounding performance. Full
fine-tuning risks catastrophic forgetting of the `describe_scene` capability,
which has no dedicated SFT data — it relies on the pretrained model's
zero-shot quality.

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

**Multi-object detection examples.** The `POST /detect_objects` endpoint
asks the VLM to list all visible objects with bounding boxes in a single
response (multiple `<ref>/<box>` tags). Include multi-object examples in the
training mix (~20% of examples) so the model learns this output format:

```json
{
  "image": "frame_0055.jpg",
  "conversations": [
    {
      "role": "user",
      "content": "<image>List all visible objects with their bounding boxes."
    },
    {
      "role": "assistant",
      "content": "<ref>table</ref><box>(100,300),(400,600)</box><ref>chair</ref><box>(500,350),(700,650)</box><ref>lamp</ref><box>(750,100),(850,400)</box>"
    }
  ]
}
```

**Description preservation.** Include ~10% of training examples as scene
description pairs (reusing Stage 2 output from Section 5.6) to preserve the
model's `describe_scene` capability during grounding SFT. Combined with LoRA,
this prevents catastrophic forgetting of the free-text description ability.

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


### 5.9 Sim-in-the-loop data collection via Isaac Sim ROS2 Bridge

The data collection approach in Section 5.5 uses human gamepad teleop as
the MVP controller. This section defines the scalable successor: running the
full autonomy stack against the simulated robot via the Isaac Sim ROS2
Bridge (OmniGraph ROS Bridge), producing perception data at the exact
deployment distribution with zero human effort.

```text
Windows Workstation (Isaac Sim)          DGX Spark
================================         ==========================
Isaac Sim environment                    Planner service (:8200)
  + OmniGraph ROS2 Bridge                VLM service (:8100)
    publishers:                                ^
      /d555/color/image_sync (640x360)         |
      /d555/aligned_depth/image_sync           | LAN HTTP
      /d555/color/camera_info_sync             |
      /strafer/odom                      strafer_autonomy.executor
      /d555/imu/filtered                   MissionRunner
      TF: map->odom->base_link->d555_link    JetsonRosClient (same code)
    subscriber:                              HttpPlannerClient -> DGX
      /cmd_vel                               HttpGroundingClient -> DGX
                                           Nav2 (against sim costmap)
  + depthimage_to_laserscan node             goal_projection_node
      /scan (from sim depth)
                                         Data capture harness
  + static map server (or RTAB-Map)        records (frame, VLM_response,
      /map                                  ground_truth) tuples
```

**Note:** The executor and Nav2 can run on the Jetson over the network
(same ROS2 domain via DDS discovery) while the bridge publishes from
the Windows machine. This is the recommended setup since the Jetson
already has the full ROS2 stack.

#### 5.9.1 What the bridge must provide

The real robot's ROS stack exposes a specific set of topics, message types,
and frame conventions. The bridge must replicate this interface exactly so
that `JetsonRosClient`, Nav2, and `goal_projection_node` work without
modification.

**Sensor topics (bridge publishes):**

| Topic | Type | Rate | Real source | Bridge source |
|-------|------|------|-------------|---------------|
| `/d555/color/image_sync` | sensor_msgs/Image (RGB8) | 30 Hz | D555 USB camera | Isaac Sim camera, **640x360** (not 80x60 policy resolution) |
| `/d555/aligned_depth_to_color/image_sync` | sensor_msgs/Image (16UC1, mm) | 30 Hz | D555 aligned depth | Isaac Sim depth (convert float32 m -> uint16 mm) |
| `/d555/color/camera_info_sync` | sensor_msgs/CameraInfo | 30 Hz | D555 driver | Computed from sim camera intrinsics at 640x360 |
| `/strafer/odom` | nav_msgs/Odometry | 50 Hz | RoboClaw wheel encoders | Sim wheel joint velocities -> mecanum forward kinematics |
| `/d555/imu/filtered` | sensor_msgs/Imu | 200 Hz | Madgwick filter on D555 IMU | Sim IMU sensor (already has orientation) |
| `/scan` | sensor_msgs/LaserScan | 30 Hz | depthimage_to_laserscan on depth | Same node, reading sim depth topic |

**Control topic (bridge subscribes):**

| Topic | Type | Rate | Real consumer | Bridge consumer |
|-------|------|------|---------------|-----------------|
| `/cmd_vel` | geometry_msgs/Twist | ~20 Hz | RoboClaw driver | Bridge converts `[vx, vy, wz]` to sim wheel velocities |

**TF tree (bridge broadcasts):**

```
map (static identity or from SLAM)
  +-- odom (dynamic, from sim wheel odometry)
      +-- base_link
          |-- chassis_link (static)
          |-- wheel_{1,2,3,4}_link (dynamic, from sim joint positions)
          +-- d555_link (static, +0.20m fwd, +0.25m up)
              +-- d555_color_optical_frame (static, Z-forward convention)
```

**Nav2 and SLAM:**

For sim-in-the-loop, there are two options for the map/costmap:
1. **Static map server** -- pre-render the Infinigen floor plan as an
   OccupancyGrid and serve it via `nav2_map_server`. Simplest approach.
2. **RTAB-Map on sim data** -- run RTAB-Map against the sim's camera and
   depth topics. Full fidelity but heavier.

Nav2 itself runs unmodified with the same `nav2_params.yaml` config -- MPPI
controller, omni motion model, same velocity limits.

#### 5.9.2 Bridge implementation

**New files:**

```
source/strafer_lab/strafer_lab/bridge/
    __init__.py
    ros2_bridge.py          # StraferROS2Bridge class
    depth_conversion.py     # float32 meters -> uint16 mm, nearfield clamp
    odom_integrator.py      # sim wheel velocities -> nav_msgs/Odometry
    camera_info_builder.py  # build CameraInfo from sim intrinsics
```

**Core class:**

```python
class StraferROS2Bridge:
    """Publishes Isaac Sim sensor data as ROS2 topics.

    Makes the simulated robot indistinguishable from the real robot
    from the perspective of JetsonRosClient, Nav2, and goal_projection_node.
    """

    def __init__(self, env, ros_camera_name: str = "d555_camera_perception"):
        self._env = env
        self._camera = env.scene[ros_camera_name]  # 640x360 camera
        self._node = rclpy.create_node("strafer_sim_bridge")

        # Publishers -- exact topic names matching real robot
        self._pub_color = self._node.create_publisher(
            Image, "/d555/color/image_sync", 10)
        self._pub_depth = self._node.create_publisher(
            Image, "/d555/aligned_depth_to_color/image_sync", 10)
        self._pub_cam_info = self._node.create_publisher(
            CameraInfo, "/d555/color/camera_info_sync", 10)
        self._pub_odom = self._node.create_publisher(
            Odometry, "/strafer/odom", 10)
        self._pub_imu = self._node.create_publisher(
            Imu, "/d555/imu/filtered", 10)

        self._tf_broadcaster = TransformBroadcaster(self._node)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self._node)

        # Subscriber -- receive cmd_vel and apply to sim
        self._sub_cmd_vel = self._node.create_subscription(
            Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self._latest_cmd_vel = Twist()

        # Publish static TFs once
        self._publish_static_tfs()

    def step_and_publish(self):
        """Call after each env.step(). Publishes all sensor data as ROS2."""
        now = self._node.get_clock().now().to_msg()

        # RGB: (1, 360, 640, 4) RGBA -> RGB8 sensor_msgs/Image
        rgba = self._camera.data.output["rgb"][0]
        rgb = rgba[:, :, :3].cpu().numpy()  # drop alpha
        self._pub_color.publish(self._numpy_to_image(rgb, "rgb8", now))

        # Depth: float32 meters -> uint16 millimeters
        depth_m = self._camera.data.output["distance_to_image_plane"][0, :, :, 0]
        depth_mm = self._meters_to_uint16_mm(depth_m.cpu().numpy())
        self._pub_depth.publish(self._numpy_to_image(depth_mm, "16UC1", now))

        # CameraInfo (precomputed from intrinsics)
        self._pub_cam_info.publish(self._build_camera_info(now))

        # Odometry from sim wheel velocities
        self._pub_odom.publish(self._build_odom(now))

        # IMU from sim sensor
        self._pub_imu.publish(self._build_imu(now))

        # Dynamic TF: odom -> base_link
        self._broadcast_odom_tf(now)

    def _meters_to_uint16_mm(self, depth_m: np.ndarray) -> np.ndarray:
        depth_clipped = np.clip(depth_m, DEPTH_CLIP_NEAR, DEPTH_CLIP_FAR)
        return (depth_clipped * 1000.0).astype(np.uint16)
```

#### 5.9.3 Synthetic command generation

The data capture harness needs to generate realistic navigation commands
that mimic what a human operator would issue. This is a three-stage pipeline
that avoids circular self-training:

```text
Stage 1: VLM detects objects in current frame
    -> ["table", "chair", "bookshelf", "lamp"]

Stage 2: Ground truth validates detections
    -> cross-reference against scenes_metadata.json label set
    -> discard any labels NOT confirmed by scene metadata
    -> ["table", "chair", "bookshelf"]  (lamp was hallucinated)

Stage 3: LLM generates diverse command phrasings
    -> Input: validated labels ["table", "chair", "bookshelf"]
    -> Output: "go check near the table then look around for the bookshelf"
```

**Why this avoids circular self-training:** The VLM (Qwen2.5-VL) proposes
object detections, but ground truth from `scenes_metadata.json` validates
them. A separate language model (Qwen3-4B, already on DGX) generates
diverse phrasings from the validated label set. Three models, no circular
dependency.

**Template-based fallback:** For the MVP, skip Stage 1 (VLM detection)
entirely and generate commands directly from the scene metadata label set:

```python
def generate_commands(scene_labels: list[str], n: int = 10) -> list[str]:
    templates = [
        "go to the {obj}",
        "navigate to the {obj}",
        "find the {obj}",
        "check near the {obj}",
        "go look at the {obj}",
    ]
    commands = []
    for _ in range(n):
        obj = random.choice(scene_labels)
        template = random.choice(templates)
        commands.append(template.format(obj=obj))
    return commands
```

#### 5.9.4 Unreachable object handling

Not all objects in Infinigen scenes can be reached. Doors may be closed,
clutter may block paths, objects may be in inaccessible rooms. This is a
real problem and a valuable training signal.

| Scenario | What happens | Training use |
|----------|-------------|-------------|
| **Reachable** | Nav2 plans path, robot arrives, VLM confirms | Positive sample (navigation + grounding) |
| **Unreachable (blocked)** | Nav2 fails to plan or times out | Navigation negative -- filter from goal commands OR label for feasibility predictor |
| **Visible but unreachable** | Robot sees object from afar but can't reach | Grounding positive (VLM should still detect it), navigation negative |

**MVP handling: post-hoc filtering.** During data collection (teleop or
autonomous), if the robot fails to reach a target within the Nav2 timeout,
tag that episode with `"reachable": false`:

- **CLIP training:** Still use frames along the trajectory (valid for place
  recognition regardless of whether the target was reached).
- **VLM grounding:** Still use frames where the target was visible
  (grounding is about detection, not reachability).
- **Feasibility data:** Accumulate `(command, scene, reachable)` tuples.
  This becomes training data for a future feasibility predictor -- the
  planner learns to predict whether a command is likely to succeed in a
  given scene. Not needed for Phase 15, but labeling now is free.

#### 5.9.5 Data capture harness

A harness script drives the loop: generate commands -> execute via autonomy
stack -> capture paired observations:

```python
class SimInTheLoopHarness:
    def __init__(self, env, bridge, executor, scene_labels: list[str]):
        self.env = env
        self.bridge = bridge
        self.executor = executor
        self.scene_labels = scene_labels

    def run_episode(self, max_commands: int = 10):
        for _ in range(max_commands):
            # Generate command from validated scene labels
            target = random.choice(self.scene_labels)
            command = f"go to the {target}"

            # Execute via the real autonomy stack
            result = self.executor.start_mission(
                request_id=uuid4().hex, raw_command=command, source="sim_harness",
            )

            # Wait for completion, recording every intermediate frame
            reachable = True
            while self.executor.get_status().active:
                self.env.step(self._latest_action())
                self.bridge.step_and_publish()
                self.capture_frame(target)

            # Check if Nav2 timed out (target unreachable)
            if result.status == "failed" and "timeout" in result.error:
                reachable = False

            # Log result with reachability label
            self.log_result(target, result, reachable=reachable)
```

#### 5.9.6 What this enables that teleop cannot

| Capability | Human teleop | Sim-in-the-loop |
|------------|-------------|-----------------|
| Scale | 50-100 episodes (human-limited) | 1000+ episodes (automated) |
| Scan rotation frames | Manual only | Yes -- exact 60 deg heading increments, same as real |
| VLM response capture | No | Yes -- real model predictions vs. ground truth |
| Standoff approach views | Manual only | Yes -- robot stops at 0.7m from target |
| Natural failure cases | No | Yes -- VLM misses, hallucinations, Nav2 timeouts |
| Distribution match | Good (human drives naturally) | Exact match to real deployment |
| End-to-end validation | No | Yes -- mission success rate metric |
| Unreachable detection | Manual only | Automatic (Nav2 timeout = unreachable label) |

#### 5.9.7 Effort and dependencies

This is a **Large** effort item with the following sub-tasks:

| Sub-task | Effort | Dependencies |
|----------|--------|--------------|
| Dual camera config (640x360 + 80x60) in `d555_cfg.py` | Small | None |
| `StraferROS2Bridge` class (topic publishers, TF, cmd_vel subscriber) | Medium | ROS2 on Windows or cross-machine ROS2 domain |
| Depth conversion (float32 m -> uint16 mm, D555 clipping) | Small | None |
| Odom integrator (sim wheels -> nav_msgs/Odometry) | Small | Mecanum FK from `strafer_shared` |
| CameraInfo builder (sim intrinsics -> sensor_msgs/CameraInfo) | Small | None |
| Nav2 launch with static sim map or RTAB-Map | Medium | Step above |
| Synthetic command generation (template-based + LLM paraphrase) | Small | Scene metadata labels |
| Data capture harness (command gen, frame recording, reachability labeling) | Medium | Bridge + executor working |
| Integration testing | Medium | All above |

**Total estimated effort:** 2-3 weeks for a working prototype.

**Critical dependency:** ROS2 Humble must be available on the Windows
workstation (via WSL2, Docker, or native build) or the executor/Nav2 must
run on the Jetson while the bridge publishes from the Windows machine over
the network (same ROS2 domain via DDS discovery). The latter is simpler
since the Jetson already has the full ROS2 stack.



### 5.10 Implementation plan

**Platform key:** [D] = DGX Spark, [S] = Isaac Sim host (DGX or Windows),
[J] = Jetson

| Step | Description | Platform | Effort | Dependencies |
|------|-------------|----------|--------|--------------|
| 1 | Infinigen scene generation at higher quality | [D] | Medium | 128GB VRAM |
| 2 | Scene metadata extraction (room types, object tags, spatial relations) | [D] | Medium | Infinigen Blender Python state |
| 3 | USD prim labeling (`semanticLabel`, `instanceId` attributes) | [D] | Small | Step 2 |
| 4 | Dual camera config (640x360 perception + 80x60 policy) in `d555_cfg.py` | [S] | Small | None |
| 5 | Perception data collection with gamepad teleop (`collect_perception_data.py`) | [S] | Medium | Steps 3, 4 |
| 6 | Replicator bbox extraction (`bounding_box_2d_tight` annotator) | [S] | Medium | Steps 3, 4 |
| 7 | HDF5/WebDataset export pipeline with ProcRoom filtering | [S] | Small | Steps 5, 6 |
| 8a | Stage 1: Spatial description builder (programmatic, from GT) | [D] | Medium | Step 2 metadata |
| 8b | Stage 2: VLM description generation (Qwen2.5-VL-7B, standalone) | [D] | Small | Step 8a |
| 8c | Stage 3: Ground truth validation filter | [D] | Small | Steps 8b, 2 |
| 8d | Stage 4: Human spot-check tooling | [D] | Small | Step 8c |
| 9 | CLIP image-text contrastive fine-tuning (OpenCLIP ViT-B/32) | [D] | Medium | Steps 7, 8c |
| 10 | VLM grounding fine-tuning data prep (Qwen2.5-VL ref/box format + negatives) | [D] | Medium | Steps 6, 7 |
| 11 | ONNX export of fine-tuned CLIP for Jetson | [D] | Small | Step 9 |
| 12 | Real-world failure logging in `SemanticMapManager` + JSON manifest | [J] | Small | Section 1 |
| 13 | `gen_failure_scenarios.py` -- manifest-driven scenario spawning | [D] | Large | Steps 1-12 |
| 14a | Isaac Sim ROS2 Bridge -- sensor publishers, TF, cmd_vel subscriber | [S] | Medium | ROS2 available |
| 14b | Depth conversion + odom integrator + CameraInfo builder | [S] | Small | strafer_shared constants |
| 14c | Nav2 sim launch (static map server or RTAB-Map against sim) | [S/J] | Medium | Step 14a |
| 14d | Synthetic command generation (template + LLM paraphrase) | [D] | Small | Step 2 metadata |
| 14e | Data capture harness (command gen, frame recording, reachability labeling) | [S] | Medium | Steps 14a-14d + DGX services |


---

## Summary of Proposed Changes

| Area | Item | Effort | Priority |
|------|------|--------|----------|
| Safety | Semantic map arrival verification | Medium | High |
| Safety | VLM confidence thresholding | Small | High |
| Safety | Costmap collision pre-check | Small | Medium |
| Safety | Odometry drift watchdog | Medium | Medium |
| Safety | Velocity limit watchdog | Small | Low |
| Semantic Map | CLIP encoder -- unified image + text (OpenCLIP ViT-B/32 ONNX) | Small | High |
| Semantic Map | ChromaDB vector store integration | Small | High |
| Semantic Map | NetworkX spatial graph | Small | High |
| Semantic Map | SemanticMapManager | Medium | High |
| Semantic Map | Observation capture in scan_for_target | Small | High |
| Semantic Map | BackgroundMapper -- passive movement-gated capture | Medium | High |
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
| Synthetic Data | Infinigen high-quality scene generation (DGX) | Medium | High |
| Synthetic Data | Scene metadata extraction (room types, tags, relations) | Medium | High |
| Synthetic Data | USD prim labeling for Replicator | Small | High |
| Synthetic Data | Dual camera config (640x360 perception + 80x60 policy) | Small | High |
| Synthetic Data | Perception data collection with gamepad teleop | Medium | High |
| Synthetic Data | Ground-truth bbox extraction via Replicator | Medium | High |
| Synthetic Data | HDF5/WebDataset export pipeline (ProcRoom filtering) | Small | Medium |
| Synthetic Data | Scene description pipeline (5 stages) | Medium | High |
| Synthetic Data | CLIP image-text contrastive fine-tuning | Medium | High |
| Synthetic Data | VLM grounding fine-tuning data prep | Medium | Medium |
| Synthetic Data | ONNX export of fine-tuned CLIP | Small | Medium |
| Synthetic Data | Real-world failure logging | Small | Medium |
| Synthetic Data | Failure-to-sim feedback pipeline | Large | Low |
| Synthetic Data | Isaac Sim ROS2 Bridge (StraferROS2Bridge) | Medium | High |
| Synthetic Data | Synthetic command generation (template + LLM) | Small | Medium |
| Synthetic Data | Unreachable object handling + reachability labeling | Small | Medium |
| Synthetic Data | Sim-in-the-loop data capture harness | Medium | Medium |
| Synthetic Data | Human spot-check quality tooling | Small | Medium |

