# Phase 15 — Jetson Orin Nano Workstream

**Branch:** `phase_15`
**Platform:** Jetson Orin Nano (Ubuntu, ROS2 Humble)
**Colcon workspace:** `~/strafer_ws` (symlink-install from repo)
**Design doc:** `docs/STRAFER_AUTONOMY_NEXT.md` (read-only reference)
**Integration context:** `docs/INTEGRATION_JETSON.md` (full API surface)

This file defines the tasks assigned to the Jetson agent. All work is scoped
to files that ONLY this agent touches — no overlap with the DGX or Windows
workstreams.

---

## Platform context

The Jetson owns all hardware interaction, sensing, SLAM, navigation, and
mission execution. It calls DGX Spark services over LAN HTTP for planning
and visual grounding. For full platform details, architecture diagram, ROS
topic names, TF tree, and endpoint specs, see `docs/INTEGRATION_JETSON.md`.

### Key existing classes you will interact with

**`MissionRunner`** (`executor/mission_runner.py`):
- Constructor: `__init__(self, *, planner_client, grounding_client, ros_client, config)`
- Config: `MissionRunnerConfig(available_skills, default_standoff_m=0.7, default_navigation_timeout_s=90.0, default_navigation_backend="nav2", default_grounding_max_image_side=1024)`
- Skill dispatch: hardcoded if/elif ladder in `_execute_step(self, runtime, step) -> SkillResult`
- `DEFAULT_AVAILABLE_SKILLS`: `("capture_scene_observation", "locate_semantic_target", "scan_for_target", "describe_scene", "project_detection_to_goal_pose", "navigate_to_pose", "wait", "cancel_mission", "report_status")`
- Runtime state: `_MissionRuntime` with `latest_observation`, `latest_grounding`, `latest_goal_pose`, `cancel_event`, `plan`
- Helper: `self._bgr_to_rgb(image)` converts BGR numpy to RGB

**`JetsonRosClient`** (`clients/ros_client.py`) implements `RosClient` Protocol:
- `capture_scene_observation() -> SceneObservation` (`.color_image_bgr` BGR uint8 HxWx3, `.aligned_depth_m` float32 HxW, `.stamp_sec`, `.robot_pose_map` dict|None, `.camera_info`)
- `navigate_to_pose(*, step_id, goal_pose: Pose3D, execution_backend="nav2", timeout_s) -> SkillResult`
- `project_detection_to_goal_pose(*, request_id, image_stamp_sec, bbox_2d, standoff_m, target_label) -> GoalPoseCandidate`
- `rotate_in_place(*, step_id, yaw_delta_rad, tolerance_rad=0.1, timeout_s) -> SkillResult`
- `get_robot_state() -> dict` with `"pose"` (dict x/y/z/qx/qy/qz/qw or None), `"velocity"`, `"navigation_active"`
- `orient_relative_to_target(...)` — raises `NotImplementedError`
- Topics: `/d555/color/image_sync`, `/d555/aligned_depth_to_color/image_sync`, `/d555/color/camera_info_sync`, `/strafer/odom`

**`GroundingClient`** Protocol (`clients/vlm_client.py`):
- `locate_semantic_target(request: GroundingRequest) -> GroundingResult` (`.found`, `.bbox_2d`, `.label`, `.confidence`)
- `describe_scene(*, request_id, image_rgb_u8, prompt, max_image_side) -> SceneDescription` (`.description`, `.latency_s`)

**Key schemas** (`schemas/`):
- `Pose3D(x, y, z, qx, qy, qz, qw)` — frozen dataclass
- `GroundingRequest(request_id, prompt, image_rgb_u8, image_stamp_sec, max_image_side, return_debug_overlay)`
- `GroundingResult(request_id, found, bbox_2d, label, confidence, raw_output, latency_s)`
- `GoalPoseCandidate(request_id, found, goal_frame, goal_pose, target_pose, standoff_m, depth_valid, quality_flags)`
- `SceneObservation(observation_id, stamp_sec, color_image_bgr, aligned_depth_m, camera_frame, camera_info, robot_pose_map, tf_snapshot_ready)`
- `SkillCall(skill, step_id, args, timeout_s, retry_limit)` — frozen dataclass
- `SkillResult(step_id, skill, status, outputs, error_code, message, started_at, finished_at)`
- `MissionPlan(mission_id, mission_type, raw_command, steps: tuple[SkillCall], created_at)`

---

## Owned directories (do NOT edit files outside these)

```
source/strafer_autonomy/strafer_autonomy/executor/     # mission_runner.py, command_server.py
source/strafer_autonomy/strafer_autonomy/clients/      # ros_client.py (JetsonRosClient)
source/strafer_autonomy/strafer_autonomy/schemas/       # grounding.py, observation.py, mission.py
source/strafer_ros/                                     # all ROS packages
source/strafer_shared/                                  # shared constants, policy interface
```

**Shared file protocol:** `source/strafer_autonomy/strafer_autonomy/clients/__init__.py`
and `source/strafer_autonomy/strafer_autonomy/schemas/__init__.py` may be
touched by this agent only for imports of new types it creates.

---

## Tasks (ordered by priority, then dependency)

### Task 1: Semantic Map core — SemanticMapManager (Section 1.3–1.9)

**Priority:** High | **Effort:** Medium | **Sections:** 1.3, 1.8, 1.9

Create the `SemanticMapManager` class and supporting data models. This is
the foundation that all other semantic map work depends on.

**New files to create:**

```
source/strafer_autonomy/strafer_autonomy/semantic_map/
    __init__.py
    models.py              # Pose2D, DetectedObjectEntry, SemanticNode, SemanticEdge
    clip_encoder.py        # OpenCLIP ViT-B/32 ONNX wrapper (image + text towers)
    manager.py             # SemanticMapManager (NetworkX + ChromaDB + CLIP)
```

**What to implement:**

1. `models.py` — data classes from Section 1.3:
   - `Pose2D` (frozen dataclass with `from_pose_map_dict()`)
   - `DetectedObjectEntry` (with `position_mean: np.ndarray` [x,y,z],
     `position_cov: np.ndarray` [3x3], Bayesian fields)
   - `SemanticNode`, `SemanticEdge`

2. `clip_encoder.py` — ONNX wrapper for OpenCLIP ViT-B/32:
   - `encode_image(image_rgb: np.ndarray) -> np.ndarray` (512-dim)
   - `encode_text(text: str) -> np.ndarray` (512-dim)
   - Load visual tower from `~/.strafer/models/clip_visual.onnx`
   - Load text tower from `~/.strafer/models/clip_text.onnx`
   - Fallback: single `clip_vit_b32.onnx` for base (non-fine-tuned) model
   - If ONNX files don't exist, log a warning and disable (graceful degradation)

3. `manager.py` — `SemanticMapManager` from Sections 1.8–1.9:
   - `__init__(storage_dir)` — NetworkX DiGraph + ChromaDB PersistentClient
   - `add_observation(pose, timestamp, clip_embedding, detected_objects, text_description, source)`
   - `query_nearest(x, y, max_distance_m) -> SemanticNode | None`
   - `query_by_label(label, max_age_s) -> SemanticNode | None`
   - `query_by_text(query_text) -> list[dict]` — CLIP text encode → ChromaDB search
   - `query_by_embedding(embedding, n_results) -> list[tuple]` — raw CLIP
     vector → ChromaDB ANN search (used by verify_arrival and transit monitor)
   - `get_clip_embedding(embedding_id) -> np.ndarray`
   - `reinforce_or_add_object(...)` — 3D Bayesian update (Section 1.12)
   - `initial_object_covariance(depth_m, camera_yaw, camera_pitch)` → 3x3
   - `save()`, `load()` — JSON + ChromaDB persistence
   - `clear()` — full reset
   - `prune(max_age_s)` — TTL-based node removal **with tiered decay for
     `DetectedObjectEntry`**: single-sighting objects (`observation_count == 1`)
     expire after 1 hour, 2-4 sightings after 6 hours, 5+ sightings use the
     node TTL (24h). This prevents hallucinated VLM detections from persisting
     in the map. See design doc Section 1.9.

**Dependencies:** `pip install chromadb networkx onnxruntime-gpu`

**Tests to create:**

```
source/strafer_autonomy/tests/test_semantic_map.py
```

Test add/query/reinforce/save/load/prune without requiring ROS or CLIP model
(mock the CLIP encoder with random 512-dim vectors). Include test cases for
tiered object decay (verify single-sighting objects expire before
multi-sighting ones).

---

### Task 2: Observation pipeline integration (Section 1.5)

**Priority:** High | **Effort:** Small | **Sections:** 1.5, 1.7

Integrate `SemanticMapManager` into `MissionRunner`.

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py
```

**What to implement:**

1. Add `semantic_map: SemanticMapManager | None = None` parameter to
   `MissionRunner.__init__`

2. Add `_project_bbox_to_map_xyz(observation, bbox_2d)` private helper to
   `MissionRunner`. This wraps `ros_client.project_detection_to_goal_pose`
   (with `standoff_m=0.0`) to extract the 3D map-frame position of a
   detected object from a bbox + depth observation. Returns
   `(np.ndarray | None, float)` — the xyz position and depth. See design
   doc Section 1.5 for the implementation sketch.

3. In `_scan_for_target`: after grounding, store observation in semantic map
   (best-effort, guarded by `if self._semantic_map is not None`). Use
   `_project_bbox_to_map_xyz` to get 3D positions for `DetectedObjectEntry`.

4. In `_describe_scene`: store CLIP embedding + text description

5. Query-before-scan (Section 1.7): at the top of `_scan_for_target`, check
   semantic map for recent sightings before starting the rotation loop.
   Uses **ranking** (top retrieval match near target node) not fixed thresholds.
   **When the short-circuit fires**, set `runtime.latest_goal_pose` directly
   from the stored map pose (construct a `GoalPoseCandidate` from the stored
   `Pose2D`) so downstream steps can navigate without projection. Return
   `outputs["goal_pose_set"] = True`.

6. In `_project_detection_to_goal_pose` handler: if
   `runtime.latest_goal_pose` is already set (from a semantic map
   query-before-scan hit, indicated by `outputs.get("goal_pose_set")`
   on the preceding scan result), skip projection and return success
   immediately. This avoids the data flow break where a scan short-circuit
   produces no bbox for projection.

**Depends on:** Task 1

---

### Task 3: BackgroundMapper + TransitMonitor (Sections 1.5, 0.6)

**Priority:** High | **Effort:** Medium

**New files to create:**

```
source/strafer_autonomy/strafer_autonomy/semantic_map/background_mapper.py
source/strafer_autonomy/strafer_autonomy/semantic_map/transit_monitor.py
```

**What to implement:**

1. `BackgroundMapper` thread from the design doc:
   - Movement-gated: only capture if robot moved >0.5m or rotated >30 degrees
   - Polls `ros_client.get_robot_state()` every 2 seconds
   - Captures `ros_client.capture_scene_observation()` + CLIP encode
   - Stores via `semantic_map.add_observation(source="background")`
   - `start()` / `stop()` lifecycle

2. `TransitMonitor` (Section 0.6):
   - `activate(goal_pose, goal_radius_m)` — called by MissionRunner at
     navigate_to_pose start
   - `deactivate()` — called when navigation completes
   - `check(clip_embedding, robot_xy) -> dict` — called by BackgroundMapper
     at each capture during active navigation
   - Uses **ranking-based divergence detection**: queries top-3 nearest
     neighbors in ChromaDB, tracks whether top matches drift away from
     the goal region over 3+ consecutive captures
   - Reports divergence via `threading.Event` flag that MissionRunner polls

3. Integration: BackgroundMapper calls `transit_monitor.check()` at each
   capture when transit monitor is active. MissionRunner polls
   `background_mapper.divergence_detected()` during navigate_to_pose.

**Depends on:** Task 1

---

### Task 4: Arrival verification skill (Section 0.1)

**Priority:** High | **Effort:** Medium

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py
```

**What to implement:**

1. Add `"verify_arrival"` to `DEFAULT_AVAILABLE_SKILLS`
2. Add `_verify_arrival(self, runtime, step) -> SkillResult` handler
3. Add `if step.skill == "verify_arrival"` to `_execute_step` dispatch
4. Implementation uses **ranking-based verification** (no fixed similarity
   thresholds): CLIP encode arrival frame, query ChromaDB for top-k (k=5)
   nearest neighbors across entire map, check whether the majority of
   top results are spatially near the goal pose. See design doc Section 0.1.
5. Add `query_by_embedding(embedding, n_results)` method to
   `SemanticMapManager` (wraps `chromadb_collection.query` by raw vector)
6. Wire transit monitor: call `transit_monitor.activate(goal_pose)` at
   navigate_to_pose start, poll `divergence_detected()` during navigation,
   call `transit_monitor.deactivate()` on completion

**Compiler integration note:** The DGX agent (Task 2) updates all navigation
compilers (`_compile_go_to_target`, `_compile_go_to_targets`,
`_compile_wait_by_target`) to emit a `verify_arrival` step after each
`navigate_to_pose`. The `target_label` arg is set by the compiler from the
intent, not from runtime state. This task only implements the executor-side
handler; the compiler changes are DGX-owned.

**Depends on:** Task 1

---

### Task 5: Rotate skills — executor side (Section 3.1)

**Priority:** High | **Effort:** Small

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py
```

**What to implement:**

1. Add `"rotate_by_degrees"` and `"orient_to_direction"` to `DEFAULT_AVAILABLE_SKILLS`
2. Add `_rotate_by_degrees` handler — wraps `ros_client.rotate_in_place(yaw_delta_rad=radians(degrees))`
3. Add `_orient_to_direction` handler — TF lookup for current heading, compute delta to cardinal yaw, call `rotate_in_place`
4. Add both to `_execute_step` dispatch

**Coordination:** The DGX agent creates the compiler (`_compile_rotate` in
`plan_compiler.py`) and adds `"rotate"` to `_VALID_INTENTS`. This task only
adds the executor-side handlers. The compiler emits `SkillCall(skill="rotate_by_degrees", args={"degrees": N})`
and `SkillCall(skill="orient_to_direction", args={"direction": "north"})`.

**No dependency on semantic map.**

---

### Task 6: Safety — costmap pre-check, odom watchdog, velocity watchdog (Sections 0.2, 0.4, 0.5)

**Priority:** Medium | **Effort:** Medium

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/clients/ros_client.py  # JetsonRosClient
source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py  # velocity watchdog
```

**What to implement:**

1. `JetsonRosClient.check_costmap_at_pose(x, y) -> str` — subscribe to
   `/global_costmap/costmap`, convert world coords to grid cell, check value
2. `JetsonRosClient.check_slam_tracking(threshold_s) -> (bool, float)` —
   TF lookup `map→odom`, check timestamp age
3. `RoboClawNode._control_loop` — add velocity magnitude check against
   `MAX_LINEAR_VEL * 1.2`, zero motors on exceedance

---

### Task 7: VLM confidence thresholding (Section 0.3)

**Priority:** High | **Effort:** Small

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py
```

**What to implement:**

1. Add `min_grounding_confidence: float = 0.5` to `MissionRunnerConfig`
2. In `_scan_for_target`, after `locate_semantic_target()` returns: check
   `grounding.confidence` against threshold, reject low-confidence detections

---

### Task 8: Environment query skill — executor side (Section 3.4)

**Priority:** Medium | **Effort:** Small

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py
```

**What to implement:**

1. Add `"query_environment"` to `DEFAULT_AVAILABLE_SKILLS`
2. Add `_query_environment` handler — calls `self._semantic_map.query_by_text(query)`
3. Add to `_execute_step` dispatch

**Coordination:** The DGX agent creates the compiler (`_compile_query` in
`plan_compiler.py`) and adds `"query"` to `_VALID_INTENTS`. This task only
adds the executor-side handler.

**Depends on:** Task 1

---

### Task 9: Parallel health checks at startup (Section 2.3)

**Priority:** Medium | **Effort:** Small

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/executor/command_server.py
```

**What to implement:**

Replace the sequential VLM health check in `build_command_server()` with
`ThreadPoolExecutor(max_workers=2)` probing both planner and VLM health
in parallel. Use `as_completed` to fail fast on the first unhealthy service.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

futures = {}
with ThreadPoolExecutor(max_workers=2) as pool:
    if hasattr(grounding_client, "health"):
        futures[pool.submit(grounding_client.health)] = "VLM"
    if hasattr(planner_client, "health"):
        futures[pool.submit(planner_client.health)] = "Planner"
    for future in as_completed(futures):
        name = futures[future]
        health = future.result(timeout=10.0)
        if not health.get("model_loaded", False):
            raise RuntimeError(f"{name} model not loaded: {health}")
```

---

### Task 10: Real-world failure logging (Section 5.8)

**Priority:** Medium | **Effort:** Small

**Files to modify:**

```
source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py
```

**What to implement:**

Add a `log_failure()` method to `SemanticMapManager` that records perception
failures during normal operation for downstream sim feedback:

1. `log_failure(failure_type, target_label, frame_bgr, depth, robot_pose, details)` —
   saves the frame to `~/.strafer/semantic_map/failures/` and appends an
   entry to `failure_manifest.json`
2. Failure types: `"arrival_verification_failed"`, `"missed_detection"`,
   `"hallucinated_detection"`, `"low_clip_similarity"`
3. Each entry records: timestamp, failure type, target label, CLIP similarity
   (if applicable), robot pose, frame path, depth path, scene description
4. The manifest is consumed by a future `gen_failure_scenarios.py` script
   (deferred, not assigned to Phase 15) to spawn targeted training scenarios
   in simulation

**Depends on:** Task 1

---

### Task 11: Sim-in-the-loop launch configuration

**Priority:** Medium | **Effort:** Small

**Coordination:** Unblocks the DGX agent's sim-in-the-loop harness work.
The DGX side drives the simulation with Isaac Sim's bundled
`isaacsim.ros2.bridge` extension, which publishes on the SAME real-robot
topic names this Jetson already consumes from hardware (e.g.
`/d555/color/image_sync`, `/strafer/odom`, `/scan`). This task does NOT
change `JetsonRosClient`, Nav2, or the executor Python code at all — it
creates a launch variant that brings up only the autonomy-consuming side
of the stack (no real driver, no real perception) so remote-published
topics from the DGX bridge drive the Jetson's autonomy pipeline.

**Files to create:**

```
source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py
source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
```

**Files to (possibly) modify — audit first, only change if needed:**

```
source/strafer_autonomy/strafer_autonomy/executor/command_server.py
source/strafer_autonomy/strafer_autonomy/clients/ros_client.py
```

**What to implement:**

1. **New launch file** `bringup_sim_in_the_loop.launch.py` that brings up
   ONLY the autonomy-consuming side of the stack:

   - `strafer_navigation` (Nav2) — same params file as
     `bringup_autonomy.launch.py`. Nav2 subscribes to the topics the DGX
     bridge publishes and publishes `/cmd_vel`, which the DGX bridge
     subscribes to in order to drive the simulated robot.
   - `strafer_slam` — either RTAB-Map or a static map server, whichever
     the current autonomy launch uses. For the first pass a static map
     server loading the Infinigen scene's pre-rendered occupancy grid is
     simplest; RTAB-Map on sim topics also works once we validate it.
   - `strafer_autonomy` executor + command server — unchanged.
   - `depthimage_to_laserscan` — needed to produce `/scan` from the sim
     bridge's depth topic, exactly like it does against the real D555.

   Do NOT start:
   - `strafer_driver` (RoboClaw motor driver — no motors present)
   - `strafer_perception` (D555 realsense_node — no camera present)
   - Any hardware watchdog timers that would false-positive without the
     physical USB bus.

2. **New env file** `env_sim_in_the_loop.env` that exports the
   cross-host DDS config:

   ```bash
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
   export ROS_DOMAIN_ID=42
   export HARDWARE_PRESENT=false
   ```

   The DGX side exports the same `RMW_IMPLEMENTATION` and
   `ROS_DOMAIN_ID` from its `env_setup.sh`. Both hosts must be on the
   same LAN subnet; the LAN IPs for this deployment are:

   - DGX Spark: `192.168.50.196` (hostname `dgx-spark`, or whatever the
     DGX agent's `.env` records in `STRAFER_JETSON_HOST`'s sibling
     variable — see the DGX agent's `env_setup.sh`)
   - Jetson Orin Nano: `192.168.50.24` (hostname `jetson-desktop`)

   `ROS_DOMAIN_ID=42` is arbitrary; any matching value works, the only
   rule is both hosts must agree. `rmw_cyclonedds_cpp` is chosen because
   it reliably handles cross-machine discovery, unlike FastDDS's default
   shared-memory transport.

3. **Audit** `command_server.build_command_server()` and
   `JetsonRosClient` for startup health checks that assume the real
   hardware driver is up — direct device-node polling, TF liveness
   windows shorter than network jitter (typical LAN jitter is 1-10 ms,
   so any TF liveness check under ~50 ms is at risk). If any such check
   exists, gate it behind a `HARDWARE_PRESENT` env var that defaults to
   True (preserving today's behavior on the real robot) and which the
   new `env_sim_in_the_loop.env` sets to False.

   Most likely outcome: no changes needed. The existing stack talks to
   topics, not to hardware-specific device nodes, so the network jitter
   case is the only real risk.

**Testing (manual, once DGX-side Task 4 lands):**

1. **Jetson side:**
   ```bash
   source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
   ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py
   ```
2. **DGX side:** boot the sim with the Isaac Sim ROS2 bridge enabled
   (the DGX agent ships a script for this).
3. Verify from the Jetson: `ros2 topic list` shows every expected
   topic (`/d555/color/image_sync`, `/d555/aligned_depth_to_color/image_sync`,
   `/strafer/odom`, `/d555/imu/filtered`, `/scan`, `/cmd_vel`), and
   `ros2 topic hz /d555/color/image_sync` reports ~30 Hz.
4. Submit a mission: `strafer-autonomy-cli submit "describe what you see"`.
   Executor should run the mission, planner/VLM on the DGX (LAN HTTP,
   unchanged from real-robot operation) should handle intent parsing
   and grounding, and Nav2 should publish `/cmd_vel` if the mission
   requires motion.
5. On the DGX side, verify that the simulated robot moves in response
   to `/cmd_vel`.

**Depends on:** DGX Task 4 (Isaac Sim ROS2 Bridge). That task must ship
before this one can be tested end-to-end, though the launch file and
env file can be written and partially validated (syntax, Nav2 start-up)
in isolation.

---

## Deferred tasks (not assigned to this phase)

These items from `STRAFER_AUTONOMY_NEXT.md` are explicitly deferred:

- **Multi-room navigation** (Section 1.10.1) — Medium effort, Medium priority.
  Currently, `scan_for_target` fails when the target is in a different room
  (not visible from the robot's current position). Phase 15 supports
  single-room targets and repeat visits to previously-seen targets via the
  semantic map. The short-term mitigation (navigate to stored map pose on
  scan failure) is a small follow-up after the semantic map is operational.
  See design doc Section 1.10.1 for the full mitigation path.
- **Plan repair on failure** (Section 3.6) — Large effort, Low priority. Future exploration.
- **Failure-to-sim feedback pipeline** (Section 5.8) — Large effort, Low priority. Depends on synthetic data infrastructure (DGX + Isaac Sim host).

---

## Build and test

```bash
# Build ROS workspace after changes
make build

# Run non-ROS tests
python -m pytest source/strafer_autonomy/tests/ -m "not requires_ros" -v

# Run all colcon tests
make test
```

---

## What NOT to touch

- `source/strafer_vlm/` — owned by DGX agent
- `source/strafer_autonomy/strafer_autonomy/planner/` — owned by DGX agent
- `source/strafer_autonomy/strafer_autonomy/clients/planner_client.py` — owned by DGX agent
- `source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py` — owned by DGX agent
- `source/strafer_lab/` — owned by DGX agent (batch processing) and Isaac Sim host agent (runtime)
- `Makefile` — shared, do not modify without coordination
- `docs/` — do not modify design docs during implementation
