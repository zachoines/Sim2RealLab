# Phase 15 -- Isaac Sim Host Workstream

**Branch:** `phase_15`
**Platform:** Isaac Sim host -- DGX Spark (Ubuntu, preferred) or Windows workstation
**Package:** `source/strafer_lab/` (Isaac Lab extension -- runtime components)
**Design doc:** `docs/STRAFER_AUTONOMY_NEXT.md` (read-only reference, Section 5)

This file defines the Isaac Sim runtime tasks: data collection, bbox
extraction, camera configuration, and the ROS2 Bridge. These tasks require
a running Isaac Sim instance and can execute on whichever machine has Isaac
Sim available (DGX Spark preferred for VRAM headroom, Windows workstation
as fallback).

The DGX agent owns batch processing tasks (Infinigen generation, scene
metadata extraction, description pipeline, fine-tuning) in a separate
workstream -- see `PHASE_15_DGX.md` Tasks 7-12.

---

## Platform context

`strafer_lab` is an Isaac Lab extension for a GoBilda Strafer mecanum-wheel
robot. It has two roles: RL policy training (existing) and synthetic data
generation (this phase). The existing codebase has 30 registered Gymnasium
environments across 5 scene types x 3 realism levels x 2 modes (Train/Play).

### Key existing APIs you will interact with

**Isaac Lab Gymnasium API:**
```python
obs_dict, reward, terminated, truncated, info = env.step(action)
# obs_dict["policy"] = concatenated float32 tensor (NOT raw camera data)
```

**Camera data** (accessed via scene sensor handles, NOT obs_dict):
```python
# Policy camera (80x60, existing)
env.scene["d555_camera"].data.output["rgb"]                    # (num_envs, 60, 80, 4)

# Perception camera (640x360, Task 1 of this workstream)
env.scene["d555_camera_perception"].data.output["rgb"]         # (num_envs, 360, 640, 4)
env.scene["d555_camera_perception"].data.output["distance_to_image_plane"]  # (num_envs, 360, 640, 1)
```

**D555 camera specs** (`d555_cfg.py`):
- Policy resolution: 80x60 (existing, for RL training)
- **Perception resolution: 640x360** (new, for data collection + ROS bridge)
- Focal length: 1.93 mm, horizontal aperture: 3.68 mm
- Clip range: 0.01-6.0 m (sim), real min range 0.4 m (handled by nearfield fill)
- Update rate: 30 Hz
- Mount offset from body_link: (0.20, 0.0, 0.25) m (20cm forward, 25cm up)

**Robot pose:**
```python
env.scene["robot"].data.root_pos_w    # (num_envs, 3)
env.scene["robot"].data.root_quat_w   # (num_envs, 4)
```

**Infinigen environments:**
- Scene USDs in `Assets/generated/scenes/` with `scene_metadata.json`
  (produced by DGX Task 8)
- Loaded as global prim at `/World/Room`
- Objects have `semanticLabel` USD prim attributes (set by DGX Task 8)
- **Primary source for all perception training data**

**ProcRoom environments:**
- Solid-color primitive shapes. **NOT useful for perception training.**
- Keep for RL policy training only.

**Existing `collect_demos.py`** (template for new scripts):
- Output: HDF5 with per-episode groups (`obs`, `actions`, `rewards`)
- Gamepad teleop with world-to-body frame transform
- CLI: `--task`, `--output`, `--max_episodes`

---

## Owned files (do NOT edit files outside these)

```
# Isaac Sim runtime scripts
source/strafer_lab/scripts/collect_perception_data.py     # NEW -- teleop data collection
source/strafer_lab/scripts/sim_in_the_loop.py            # NEW -- ROS bridge data harness

# Isaac Sim runtime modules
source/strafer_lab/strafer_lab/assets/strafer/d555_cfg.py  # modify -- add perception camera
source/strafer_lab/strafer_lab/tools/bbox_extractor.py     # NEW -- Replicator bbox extraction
source/strafer_lab/strafer_lab/bridge/                     # NEW -- ROS2 bridge
source/strafer_lab/test/                                   # tests for above
```

**Do NOT modify** files owned by the DGX agent:
- `source/strafer_lab/scripts/prep_room_usds.py`
- `source/strafer_lab/scripts/extract_scene_metadata.py`
- `source/strafer_lab/scripts/generate_descriptions.py`
- `source/strafer_lab/scripts/finetune_clip.py`
- `source/strafer_lab/scripts/prepare_vlm_finetune_data.py`
- `source/strafer_lab/strafer_lab/tools/scene_labels.py`
- `source/strafer_lab/strafer_lab/tools/spatial_description.py`
- `source/strafer_lab/strafer_lab/tools/dataset_export.py`

---

## Tasks (ordered by priority, then dependency)

### Task 1: Dual camera configuration (Section 5.5.2)

**Priority:** High | **Effort:** Small

**File to modify:**

```
source/strafer_lab/strafer_lab/assets/strafer/d555_cfg.py
```

**What to implement:**

Add a 640x360 perception camera config alongside the existing 80x60 policy
camera. Both share the same physical parameters (focal length, aperture,
mount offset) but differ in resolution.

```python
D555_PERCEPTION_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link/d555_camera_perception",
    update_period=D555_CAMERA_UPDATE_PERIOD,  # 1/30 = 30 Hz
    height=360,
    width=640,
    data_types=("rgb", "distance_to_image_plane"),
    spawn=PinholeCameraCfg(
        focal_length=D555_FOCAL_LENGTH,
        horizontal_aperture=D555_H_APERTURE,
        clipping_range=(0.01, DEPTH_CLIP_FAR),
    ),
    offset=OffsetCfg(
        pos=D555_CAMERA_OFFSET_POS,
        rot=D555_CAMERA_OFFSET_ROT,
        convention="ros",
    ),
)
```

Only instantiate in perception/bridge environments (not RL training). At
640x360, only 1-8 parallel envs are feasible.

You will also need to create or modify an environment config that includes
this camera in the scene definition (e.g., a new `Infinigen-Perception-Play`
env variant).

---

### Task 2: Perception data collection with gamepad teleop (Section 5.5.4)

**Priority:** High | **Effort:** Medium

**New file to create:**

```
source/strafer_lab/scripts/collect_perception_data.py
```

**What to implement:**

A data collection script modeled after `collect_demos.py` but for perception
data. **The controller is human gamepad teleop** -- not random walk, not
goal-seeking, not scripted patrol.

The operator drives the robot through Infinigen scenes via gamepad. The
script captures 640x360 RGB-D + Replicator bboxes + poses at every frame.

```python
class PerceptionDataCollector:
    def __init__(self, env, output_dir: Path, scene_name: str):
        self.env = env
        self.camera = env.scene["d555_camera_perception"]  # 640x360
        self.bbox_extractor = ReplicatorBboxExtractor(
            self.camera.render_product_path
        )
        self.gamepad = GamepadTeleop()

    def collect_episode(self, episode_id: int, max_steps: int = 500) -> dict:
        obs_dict, info = self.env.reset()
        frames = []
        for step in range(max_steps):
            action = self.gamepad.get_action()
            if action is None:
                break
            obs_dict, reward, terminated, truncated, info = self.env.step(action)

            rgba = self.camera.data.output["rgb"]
            rgb = rgba[..., :3]
            depth = self.camera.data.output["distance_to_image_plane"]
            bboxes = self.bbox_extractor.extract()

            frame_data = {
                "rgb": rgb[0].cpu().numpy(),
                "depth": depth[0].cpu().numpy(),
                "cam_pos": self.camera.data.pos_w[0].cpu().numpy(),
                "cam_quat": self.camera.data.quat_w_world[0].cpu().numpy(),
                "robot_pos": self.env.scene["robot"].data.root_pos_w[0].cpu().numpy(),
                "robot_quat": self.env.scene["robot"].data.root_quat_w[0].cpu().numpy(),
                "bboxes": bboxes,
                "scene_type": "infinigen",
                "scene_name": self.scene_name,
            }
            frames.append(frame_data)
            if terminated.any() or truncated.any():
                break
        return self.save_episode(episode_id, frames)
```

**CLI:**

```bash
python scripts/collect_perception_data.py \
  --task Isaac-Strafer-Nav-Real-Infinigen-Depth-Play-v0 \
  --scene scene_001 \
  --output data/perception/ \
  --max_episodes 20
```

**Depends on:** Task 1 (perception camera), Task 3 (bbox extractor).
Also requires labeled Infinigen USDs from DGX Task 8.

---

### Task 3: Ground-truth 2D bbox extraction via Replicator (Section 5.5.4)

**Priority:** High | **Effort:** Medium

**New files to create:**

```
source/strafer_lab/strafer_lab/tools/
    __init__.py          # This agent creates it (DGX agent adds to it later)
    bbox_extractor.py
```

**What to implement:**

```python
import omni.replicator.core as rep

class ReplicatorBboxExtractor:
    """Extract 2D bounding boxes + semantic labels via Replicator."""

    def __init__(self, camera_render_product_path: str):
        self._bbox_annotator = rep.AnnotatorRegistry.get_annotator(
            "bounding_box_2d_tight"
        )
        self._bbox_annotator.attach([camera_render_product_path])

    def extract(self) -> list[dict]:
        bbox_data = self._bbox_annotator.get_data()
        results = []
        for entry in bbox_data["data"]:
            results.append({
                "label": entry.get("semanticLabel", "unknown"),
                "bbox_2d": [int(entry["x_min"]), int(entry["y_min"]),
                            int(entry["x_max"]), int(entry["y_max"])],
                "instance_id": entry.get("instanceId", -1),
                "occlusion": entry.get("occlusionRatio", 0.0),
            })
        return results
```

Requires `semanticLabel` USD prim attributes (set by DGX Task 8).
Do NOT use manual 3D-to-2D projection.

---

### Task 4: Isaac Sim ROS2 Bridge (Section 5.9)

**Priority:** High | **Effort:** Large

**New files to create:**

```
source/strafer_lab/strafer_lab/bridge/
    __init__.py
    ros2_bridge.py
    depth_conversion.py
    odom_integrator.py
    camera_info_builder.py
```

**What to implement:**

The ROS2 Bridge publishes Isaac Sim sensor data on real robot topic names
so the full autonomy stack (JetsonRosClient, Nav2, goal_projection_node)
works unmodified against the simulated robot.

See design doc Section 5.9.1-5.9.2 for full topic table, TF tree, and
`StraferROS2Bridge` implementation sketch.

**Key topics:** `/d555/color/image_sync` (RGB8 640x360), depth (16UC1 mm),
CameraInfo, `/strafer/odom`, IMU, `/scan` (LaserScan), TF tree.

**Critical dependency:** ROS2 Humble available on the host machine (WSL2,
Docker, or native) OR the executor/Nav2 runs on the Jetson via network
(same ROS2 domain via DDS discovery).

**Depends on:** Task 1 (perception camera)

---

### Task 5: Sim-in-the-loop data capture harness (Section 5.9.5)

**Priority:** Medium | **Effort:** Medium

**New file to create:**

```
source/strafer_lab/scripts/sim_in_the_loop.py
```

**What to implement:**

A harness that drives the simulation loop: generate navigation commands
from scene labels -> execute via the autonomy stack (over ROS bridge) ->
capture paired observations with reachability labels.

When Nav2 fails to reach a target (timeout), tag the episode as
`"reachable": false`. These frames are still valid for CLIP/VLM training
but get labeled for future feasibility prediction.

See design doc Section 5.9.3-5.9.5 for command generation pipeline and
`SimInTheLoopHarness` implementation sketch.

**Depends on:** Task 4 (ROS bridge) + DGX services running + DGX Task 8
(labeled Infinigen USDs with `scene_metadata.json`)

---

## Build and test

```bash
# Run strafer_lab tests (from repo root, with Isaac Lab env active)
cd source/strafer_lab
python run_tests.py

# Or with pytest directly
python -m pytest test/ -v

# Run perception data collection (requires gamepad + Infinigen env)
python scripts/collect_perception_data.py \
  --task Isaac-Strafer-Nav-Real-Infinigen-Depth-Play-v0 \
  --scene scene_001 \
  --output data/perception/ \
  --max_episodes 10
```

---

## Deferred tasks

- **Failure-to-sim feedback pipeline** (Section 5.8) -- Depends on Jetson
  real-world failure logging and all synthetic data infrastructure.

---

## What NOT to touch

- `source/strafer_autonomy/` -- owned by Jetson + DGX agents
- `source/strafer_vlm/` -- owned by DGX agent
- `source/strafer_ros/` -- owned by Jetson agent
- `source/strafer_shared/` -- owned by Jetson agent
- DGX-owned strafer_lab files (see list above)
- `Makefile` -- shared, do not modify without coordination
- `docs/` -- do not modify design docs during implementation
