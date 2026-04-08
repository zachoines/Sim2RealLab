# Phase 15 — Windows Workstation Workstream

**Branch:** `phase_15`
**Platform:** Windows workstation (Isaac Sim / Isaac Lab, CUDA GPU)
**Package:** `source/strafer_lab/` (Isaac Lab extension)
**Design doc:** `docs/STRAFER_AUTONOMY_NEXT.md` (read-only reference, Section 5)

This file defines the tasks assigned to the Windows agent. All work is
scoped to files that ONLY this agent touches — no overlap with the Jetson
or DGX workstreams.

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
env.scene["d555_camera"].data.output["rgb"]                    # (num_envs, 60, 80, 4) RGBA uint8
env.scene["d555_camera"].data.output["distance_to_image_plane"] # (num_envs, 60, 80, 1) float32 meters
env.scene["d555_camera"].data.pos_w                             # (num_envs, 3)
env.scene["d555_camera"].data.quat_w_world                      # (num_envs, 4)
```

**D555 camera specs** (`d555_cfg.py`):
- Resolution: 80x60 (policy input, downsampled from real 640x360)
- Focal length: 1.93 mm, horizontal aperture: 3.68 mm
- Clip range: 0.01-6.0 m (sim), real min range 0.4 m (handled by nearfield fill)
- Update rate: 30 Hz
- Mount offset from body_link: (0.20, 0.0, 0.25) m (20cm forward, 25cm up)

**Robot pose:**
```python
env.scene["robot"].data.root_pos_w    # (num_envs, 3)
env.scene["robot"].data.root_quat_w   # (num_envs, 4)
```

**ProcRoom ground truth** (available at runtime when env has `_proc_room_*` attributes):
```python
env._proc_room_active_mask      # (num_envs, 44) bool — which obstacle slots are populated
env._proc_room_spawn_pts        # (num_envs, 200, 2) — BFS-verified spawn/goal XY positions
env._proc_room_difficulty       # (num_envs,) — integer difficulty level
env.scene["obstacles"].data.object_pos_w  # (num_envs, 44, 3) — per-slot 3D positions
```

**ProcRoom object inventory (44 total):**

| Slot range | Prim name prefix | Type | Dimensions (X, Y, Z) m |
|-----------|-----------------|------|------------------------|
| 0-7 | `wall_long_` | wall | 2.0 x 0.15 x 1.0 |
| 8-15 | `wall_med_` | wall | 1.0 x 0.15 x 1.0 |
| 16-19 | `wall_short_` | wall | 0.5 x 0.15 x 1.0 |
| 20-21 | `furn_table_` | furniture | 0.8 x 0.6 x 0.4 |
| 22-23 | `furn_shelf_` | furniture | 1.2 x 0.3 x 0.8 |
| 24-25 | `furn_cabinet_` | furniture | 0.5 x 0.5 x 0.6 |
| 26-27 | `furn_couch_` | furniture | 1.4 x 0.6 x 0.35 |
| 28-31 | `clutter_box_` | clutter | 0.3 x 0.3 x 0.3 |
| 32-33 | `clutter_cyl_` | clutter | r=0.15, h=0.4 |
| 34-35 | `clutter_flat_` | clutter | 0.4 x 0.4 x 0.15 |
| 36-37 | `clutter_sphere_` | clutter | r=0.15 |
| 38-39 | `clutter_cone_` | clutter | r=0.12, h=0.35 |
| 40-41 | `clutter_capsule_` | clutter | r=0.1, h=0.4 |
| 42-43 | `clutter_tall_cyl_` | clutter | r=0.1, h=0.7 |

Prim names encode category — parse the prefix (e.g., `furn_table_0` → label `"table"`).

**Infinigen environments:**
- Scene USDs in `Assets/generated/scenes/` with `scenes_metadata.json`
- Loaded as global prim at `/World/Room`
- No per-object labels currently preserved (see Task 4)

**Existing `collect_demos.py`** (template for new scripts):
- Output: HDF5 with per-episode groups (`obs`, `actions`, `rewards`)
- Gamepad teleop with world-to-body frame transform
- CLI: `--task`, `--output`, `--max_episodes`

**Registered ProcRoom environments** (use these for data collection):
- `Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0` — 8 play envs, realistic noise
- `Isaac-Strafer-Nav-Real-ProcRoom-NoCam-Play-v0` — 50 play envs, no camera
- `Isaac-Strafer-Nav-Robust-ProcRoom-Depth-Play-v0` — 8 play envs, aggressive noise

### VLM training data format (output target for DGX fine-tuning)

The DGX agent will consume the data you produce. The formats they expect:

**Qwen2.5-VL grounding SFT** (JSONL):
```json
{
  "image": "frame_0042.jpg",
  "conversations": [
    {"role": "user", "content": "<image>Locate the table in this image."},
    {"role": "assistant", "content": "<ref>table</ref><box>(321,450),(580,890)</box>"}
  ]
}
```
Coordinates are 0-1000 scaled: `x_scaled = int(x_pixel / width * 1000)`.

**CLIP contrastive pairs** (CSV manifest + image files):
```csv
anchor,positive,negative
episode_0001/frame_0010.jpg,episode_0001/frame_0012.jpg,episode_0042/frame_0005.jpg
```

---

## Owned directories (do NOT edit files outside these)

```
source/strafer_lab/                      # entire Isaac Lab extension
source/strafer_lab/scripts/              # data collection scripts
source/strafer_lab/strafer_lab/          # env configs, MDP modules, assets
source/strafer_lab/test/                 # all strafer_lab tests
```

---

## Tasks (ordered by priority, then dependency)

### Task 1: Perception data collection script (Section 5.5)

**Priority:** High | **Effort:** Medium | **Section:** 5.5

**New file to create:**

```
source/strafer_lab/scripts/collect_perception_data.py
```

**What to implement:**

A data collection script modeled after the existing `collect_demos.py` but
for perception rather than control. It drives the robot through environments
while capturing labeled perception data.

```python
class PerceptionDataCollector:
    def __init__(self, env, output_dir: Path):
        self.env = env
        self.camera = env.scene["d555_camera"]

    def collect_episode(self, episode_id: int, max_steps: int = 500) -> dict:
        obs_dict, info = self.env.reset()
        frames = []

        for step in range(max_steps):
            action = self.get_action(obs_dict)
            obs_dict, reward, terminated, truncated, info = self.env.step(action)

            # Camera data via scene sensor handles (NOT obs_dict)
            rgba = self.camera.data.output["rgb"]           # (num_envs, 60, 80, 4)
            rgb = rgba[..., :3]
            depth = self.camera.data.output["distance_to_image_plane"]  # (num_envs, 60, 80, 1)

            # Poses
            cam_pos = self.camera.data.pos_w                # (num_envs, 3)
            cam_quat = self.camera.data.quat_w_world        # (num_envs, 4)
            robot_pos = self.env.scene["robot"].data.root_pos_w
            robot_quat = self.env.scene["robot"].data.root_quat_w

            # ProcRoom ground truth
            obstacle_positions = None
            if hasattr(self.env, "_proc_room_active_mask"):
                active_mask = self.env._proc_room_active_mask
                obstacle_positions = self.env.scene["obstacles"].data.object_pos_w

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

**Key API details:**
- Isaac Lab Gymnasium API: `obs_dict, reward, terminated, truncated, info = env.step(action)`
- Camera data is on the sensor, not in `obs_dict`
- D555 camera: 80x60 resolution, 30 Hz, mounted at (0.20, 0.0, 0.25) m
- ProcRoom objects: 44 total (20 walls, 8 furniture, 16 clutter) — prim
  names encode category (e.g., `furn_table_0`, `clutter_box_2`)
- Use random policy or trained policy for driving trajectories

**Action strategies to implement:**
1. Random walk (uniform random `[vx, vy, omega]`)
2. Goal-seeking (use the env's goal command as a waypoint)
3. Scripted patrol (visit corners of the room)

**Output format:** HDF5 per episode:
```
episode_NNNN/
  frame_0000.jpg          # RGB (80x60)
  frame_0000_depth.png    # uint16 depth in mm
  frame_0000.json         # camera pose, robot pose, obstacle data
```

**CLI interface** (follow `collect_demos.py` pattern):
```bash
python scripts/collect_perception_data.py \
  --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \
  --output data/perception/ \
  --max_episodes 100 \
  --max_steps 500 \
  --action_strategy goal_seeking
```

---

### Task 2: Ground-truth 2D bbox extraction (Section 5.5)

**Priority:** High | **Effort:** Medium

**New file to create:**

```
source/strafer_lab/strafer_lab/tools/
    __init__.py
    bbox_extractor.py
```

**What to implement:**

Two approaches for extracting 2D bounding boxes from sim:

**Approach A — Isaac Sim Replicator API** (preferred for Infinigen scenes):
```python
import omni.replicator.core as rep
annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
annotator.attach([camera_render_product_path])
bbox_data = annotator.get_data()
```

**Approach B — Manual 3D-to-2D projection** (works for ProcRoom):
- Read object 3D positions from `env.scene["obstacles"].data.object_pos_w`
- Read object extents from known primitive dimensions (see ProcRoom inventory)
- Project 3D bounding box corners through camera intrinsics:
  - Focal length: 1.93 mm
  - Horizontal aperture: 3.68 mm
  - Resolution: 80x60
- Compute tight 2D bounding box from projected corners
- Filter by visibility (in front of camera, within image bounds)

**ProcRoom object inventory (for known dimensions):**

| Slot range | Type | Dimensions (X, Y, Z) m |
|-----------|------|------------------------|
| 0-7 | wall_long | 2.0 x 0.15 x 1.0 |
| 8-15 | wall_med | 1.0 x 0.15 x 1.0 |
| 16-19 | wall_short | 0.5 x 0.15 x 1.0 |
| 20-21 | furn_table | 0.8 x 0.6 x 0.4 |
| 22-23 | furn_shelf | 1.2 x 0.3 x 0.8 |
| 24-25 | furn_cabinet | 0.5 x 0.5 x 0.6 |
| 26-27 | furn_couch | 1.4 x 0.6 x 0.35 |
| 28-31 | clutter_box | 0.3 x 0.3 x 0.3 |
| 32-33 | clutter_cyl | r=0.15, h=0.4 |
| 34-35 | clutter_flat | 0.4 x 0.4 x 0.15 |
| 36-37 | clutter_sphere | r=0.15 |
| 38-39 | clutter_cone | r=0.12, h=0.35 |
| 40-41 | clutter_capsule | r=0.1, h=0.4 |
| 42-43 | clutter_tall_cyl | r=0.1, h=0.7 |

**Integration with Task 1:** `PerceptionDataCollector` calls
`bbox_extractor.extract_bboxes(env, camera)` each step, adding `bboxes`
to frame_data.

---

### Task 3: HDF5/WebDataset export pipeline (Section 5.5 output format, Section 5.10 step 3)

**Priority:** Medium | **Effort:** Small

**New file to create:**

```
source/strafer_lab/strafer_lab/tools/dataset_export.py
```

**What to implement:**

Convert collected perception data into training-ready formats:

1. **HDF5** (for local training on DGX):
   - Per-episode structure matching `collect_demos.py` pattern
   - Add `bboxes`, `labels`, `cam_pose`, `robot_pose` datasets

2. **VLM grounding JSONL** (for Qwen2.5-VL fine-tuning):
   - Convert bboxes to 0-1000 scaled `<ref>label</ref><box>(x1,y1),(x2,y2)</box>` format
   - Include negative examples (frames where queried object is not visible)
   - Output `.jsonl` files ready for SFT

3. **CLIP contrastive pairs** (for OpenCLIP fine-tuning):
   - Generate (anchor, positive, negative) image triples
   - Positive: same scene, small pose jitter
   - Negative: different room
   - Output as image file triplets with a manifest CSV

**Depends on:** Tasks 1 and 2

---

### Task 4: Label-preserving Infinigen USD export (Section 5.3)

**Priority:** Medium | **Effort:** Medium

**Files to modify:**

```
source/strafer_lab/scripts/prep_room_usds.py  (if it exists in the repo)
```

**What to implement:**

1. During Blender → USD conversion, write Infinigen's per-object class name,
   instance ID, and 3D bbox as custom USD prim attributes:
   ```
   custom:semanticLabel = "chair"
   custom:instanceId = 42
   ```

2. Extend `scenes_metadata.json` with a per-scene `objects` list:
   ```json
   {
     "scenes": {
       "scene_001": {
         "spawn_points_xy": [[1.0, 2.0], ...],
         "objects": [
           {"label": "chair", "instance_id": 42, "bbox_3d": [...], "prim_path": "/World/Room/chair_42"}
         ]
       }
     }
   }
   ```

3. Add a runtime accessor in `strafer_lab/tools/scene_labels.py`:
   ```python
   def get_scene_objects(metadata_path: Path, scene_name: str) -> list[dict]:
       """Read labeled objects from scenes_metadata.json."""
   ```

---

### Task 5: CLIP contrastive fine-tuning pipeline (Section 5.6)

**Priority:** High | **Effort:** Medium

**New file to create:**

```
source/strafer_lab/scripts/finetune_clip.py
```

**What to implement:**

- Load OpenCLIP ViT-B/32 with `open_clip.create_model_and_transforms`
- Freeze text tower, fine-tune vision tower
- InfoNCE / NT-Xent loss on contrastive pairs from Task 3
- Training loop with AdamW, lr=1e-5, weight_decay=0.01
- Export fine-tuned model to ONNX: `torch.onnx.export(model.visual, ...)`
- Log experiments with MLflow (optional, for Databricks tracking)
- CLI: `python scripts/finetune_clip.py --data data/clip_pairs/ --epochs 10 --output models/clip_finetuned.onnx`

**Dataset scale:** 10k-50k image pairs, generated by Task 3.

---

### Task 6: VLM grounding fine-tuning data prep (Section 5.7)

**Priority:** Medium | **Effort:** Medium

**New file to create:**

```
source/strafer_lab/scripts/prepare_vlm_finetune_data.py
```

**What to implement:**

Convert perception data + bboxes into Qwen2.5-VL SFT format:

```json
{
  "image": "frame_0042.jpg",
  "conversations": [
    {"role": "user", "content": "<image>Locate the table in this image."},
    {"role": "assistant", "content": "<ref>table</ref><box>(321,450),(580,890)</box>"}
  ]
}
```

- Coordinates scaled to 0-1000 range
- Generate negative examples (1:3 ratio):
  ```json
  {"role": "assistant", "content": "The object is not visible in this image."}
  ```
- Output: `.jsonl` files ready for SFT on DGX

**Depends on:** Tasks 1 and 2

---

## Build and test

```bash
# Run strafer_lab tests (from repo root, with Isaac Lab env active)
cd source/strafer_lab
python run_tests.py

# Or with pytest directly
python -m pytest test/ -v

# Run perception data collection
python scripts/collect_perception_data.py \
  --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \
  --output data/perception/ \
  --max_episodes 10
```

---

## Deferred tasks (not assigned to this phase)

These items from `STRAFER_AUTONOMY_NEXT.md` are explicitly deferred:

- **ONNX export of fine-tuned CLIP** (Section 5.10 step 7) — Small effort,
  but depends on Task 5 being validated. Covered as the final step of
  `finetune_clip.py` rather than a separate task.
- **Failure-to-sim feedback pipeline** (Section 5.8) — Large effort, Low
  priority. Depends on Jetson real-world failure logging (PHASE_15_JETSON Task 10).
- **Sim-in-the-loop harness** (Section 5.9) — Large effort, Low priority.
  Depends on Isaac Sim ROS2 Bridge configuration and full autonomy stack.

---

## What NOT to touch

- `source/strafer_autonomy/` — owned by Jetson + DGX agents
- `source/strafer_vlm/` — owned by DGX agent
- `source/strafer_ros/` — owned by Jetson agent
- `source/strafer_shared/` — owned by Jetson agent
- `Makefile` — shared, do not modify without coordination
- `docs/` — do not modify design docs during implementation
