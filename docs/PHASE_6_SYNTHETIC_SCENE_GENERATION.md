# Phase 6: Synthetic Scene Generation for Generalist Obstacle Avoidance

Build a synthetic scene pipeline using the standard NVIDIA Infinigen + Replicator
workflow so the Strafer depth policy trains against realistic, diverse indoor
obstacle layouts — improving sim-to-real obstacle avoidance.

This phase plugs into the existing repo with minimal disruption:
- Isaac Lab training remains in `Scripts/train_strafer_navigation.py`.
- Navigation task code remains in `source/strafer_lab/strafer_lab/tasks/navigation/`.
- New scene generation tooling lives under `Scripts/scenegen/`.

## Overview

Current navigation training uses 8 fixed primitive-box obstacles with randomized
positions.  Phase 6 replaces those with a scene library built from:

1. **Infinigen** (offline, runs in WSL/Linux) — procedurally generates
   photorealistic indoor room environments and exports them as USD.
2. **Omniverse Replicator** (runtime, Isaac Sim standalone script) — loads
   Infinigen rooms, spawns obstacle assets from our local packs, runs physics
   settling, applies domain randomization, and exports annotated scene data.
3. **Local OpenUSD asset packs** (already in this repo):
   - `Assets/Residential_NVD@10012` — household clutter, appliances, decor
   - `Assets/SimReady_Furniture_Misc_01_NVD@10010` — furniture, medium/large objects

This follows the standard NVIDIA pipeline described in the
[Infinigen + Replicator SDG tutorial](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_infinigen_sdg.html).

## 1. Design Goals and Constraints

### 1.1 Goals

1. Increase geometric diversity without hand-authoring obstacle layouts.
2. Produce repeatable scene sets for train/val/test split and regression testing.
3. Support both static clutter and dynamic obstacle behaviors.
4. Preserve current sim-to-real observation/action contract.
5. Keep runtime cost bounded so depth training remains practical on current hardware.

### 1.2 Non-goals (for this phase)

1. No full semantic navigation stack rewrite.
2. No dependence on VLM inference during low-level RL training.
3. No immediate requirement to modify ROS deployment packages.

### 1.3 Practical Constraints from Current Stack

| Constraint | Current Value | Planning Impact |
|-----------|---------------|-----------------|
| Depth input | 80x60 (4800 dims) | Scene complexity must prioritize geometry over texture detail. |
| Control rate | 30 Hz (decimation from 120 Hz sim) | Dynamic obstacles must move smoothly at this control rate. |
| Depth training env count | ~24-32 practical | Synthetic scenes must avoid excessive per-env memory usage. |
| NoCam training env count | 512-4096 | Keep NoCam pretraining path for rapid policy iteration. |
| Policy contract | `strafer_shared.policy_interface` | Observation ordering/scales remain unchanged. |

## 2. Pipeline Architecture (Corrected)

The pipeline follows the standard NVIDIA two-stage architecture:

### 2.1 Stage 1: Offline Environment Generation (Infinigen on WSL/Linux)

```
Infinigen (WSL / Linux)
  |- Generate indoor rooms procedurally (DiningRoom, LivingRoom, etc.)
  |- Export each as .usdc with --omniverse flag
  |- Output: outputs/omniverse/<room_name>/  (USD + textures)
         |
         v
  Copy USD rooms into Assets/generated/infinigen_rooms/
```

Infinigen generates **room shells**: walls, floors, ceiling, built-in furniture
(tables, chairs, shelves), and lighting — all as a single USD file per room.
Use `fast_solve` gin config for quick generation (~3 min/room). Infinigen rooms
provide diverse **background geometry**; all ground-level clutter and movable
obstacles are added in Stage 2 by Replicator.

### 2.2 Stage 2: Runtime Scene Composition (Isaac Sim + Replicator on Windows)

This is where all ground-level clutter and obstacle density is controlled.
Replicator spawns objects into the Infinigen room shell, randomizes their poses,
and uses physics simulation to settle them realistically onto floors and surfaces.

```
Isaac Sim standalone script (compose_scenes_replicator.py)
  |
  |- For each Infinigen room:
  |   1. Load Infinigen room USD       (infinigen_sdg_utils.load_env)
  |   2. Setup env (add colliders)      (infinigen_sdg_utils.setup_env)
  |   3. Find working area              (infinigen_sdg_utils.get_matching_prim_location)
  |   4. Spawn obstacles around working area:
  |      |- Shape distractors (20-30)   (infinigen_sdg_utils.spawn_shape_distractors)
  |      |  Primitives: capsule, cone, cylinder, sphere, cube
  |      |- Mesh distractors (10-20)    (infinigen_sdg_utils.spawn_mesh_distractors)
  |      |  From Residential + SimReady asset packs (709 assets)
  |      |- Labeled targets (5-10)      (infinigen_sdg_utils.spawn_labeled_assets)
  |      |  From asset manifest
  |   5. Randomize all poses            (infinigen_sdg_utils.randomize_poses)
  |   6. Run physics sim (settle)       (infinigen_sdg_utils.run_simulation)
  |      75% fall onto surfaces, 25% float (gravity_disabled_chance)
  |   7. Randomize lighting + cameras   (Replicator triggers)
  |   8. Capture annotated frames       (rep.orchestrator.step)
  |
  |- Output:
       |- Scene USD files (for Isaac Lab training)
       |- scene_manifest.jsonl (metadata)
       |- Annotated images (RGB, depth, segmentation) for debugging
```

Clutter density is configured entirely via `Scripts/scenegen/config/strafer_sdg.yaml`:
- `distractors.shape_distractors.num` — primitive object count
- `distractors.mesh_distractors.num` — mesh asset count
- `gravity_disabled_chance` — fraction that float vs fall onto surfaces

### 2.3 Stage 3: Isaac Lab Training Integration

```
Isaac Lab env reset
  -> sample scene_id from split manifest
  -> load scene USD prototype (Infinigen room + placed obstacles)
  -> apply per-episode randomization (positions, dynamics, noise)
  -> run RL loop unchanged (same obs/action contract)
```

## 3. Inputs and Asset Sources

### 3.1 Infinigen Environments

Generated on WSL/Linux using `infinigen_examples.generate_indoors` and exported
with `infinigen.tools.export --omniverse -f usdc`.

Room types to generate:
- `DiningRoom` — tables/chairs as obstacles
- `LivingRoom` — sofas, coffee tables, scattered objects
- `Bedroom` — beds, nightstands, narrow passages
- `Kitchen` — counters, appliances, tight corridors

Target: 10 rooms per type = 40 base environments.

### 3.2 Local OpenUSD Asset Packs (already downloaded)

| Pack | Local Path | Count | Primary Use |
|------|------------|-------|-------------|
| Residential | `Assets/Residential_NVD@10012` | 507 USD | Household clutter, appliances, decor, plants |
| SimReady Furniture | `Assets/SimReady_Furniture_Misc_01_NVD@10010` | 202 USD | Furniture and medium/large indoor obstacles |

These are used as **distractor assets** spawned into Infinigen rooms following
the standard `infinigen_sdg_utils.spawn_mesh_distractors()` pattern.

### 3.3 Asset Quality Gate

Each candidate asset must pass (via `validate_asset_pool.py`):
1. USD loads without missing references.
2. Reasonable bounding box (non-zero extents, no extreme scale).
3. Collider can be generated or approximated.
4. License metadata retained in manifest.

### 3.4 Asset Manifest

`Scripts/scenegen/build_asset_manifest.py` scans both local packs and produces
`Assets/generated/manifests/asset_manifest.jsonl` with per-asset metadata
(path, source, category, file size).

**Status: Implemented and working.** 709 assets catalogued.

## 4. Scene Specification

### 4.1 Scene Counts and Split

Generate a **200-scene library** (5 scenes per Infinigen room x 40 rooms):
- 120 train scenes
- 40 val scenes
- 40 test scenes (never used in training)

### 4.2 Per-Scene Asset Spawning (following NVIDIA tutorial pattern)

Per scene, Replicator spawns objects into the loaded Infinigen room shell around
a detected `working_area_loc` (e.g. the dining table). Objects are randomized
in pose, then physics simulation settles ~75% onto floors/surfaces while ~25%
remain floating (controlled by `gravity_disabled_chance`).

| Asset Type | Count | Source | Role |
|-----------|-------|--------|------|
| Shape distractors | 20-30 | Primitives (capsule, cone, cylinder, sphere, cube) | Ground-level clutter + visual diversity |
| Mesh distractors | 10-20 | Residential + SimReady packs (709 assets) | Realistic household clutter on floor |
| Labeled targets | 5-10 | Curated subset from asset manifest | Objects of interest for depth perception |

### 4.3 Geometric and Physical Randomization

Applied by `infinigen_sdg_utils.randomize_poses()` and physics simulation:

| Parameter | Range |
|-----------|-------|
| Uniform scale multiplier | 0.6-1.4 |
| Yaw rotation | -180 to +180 deg |
| Mass multiplier | 0.8-1.2 |
| Static friction | 0.4-1.2 |
| Dynamic friction | 0.3-1.0 |
| Restitution | 0.0-0.2 |

Objects settle via physics simulation (`run_simulation(num_frames=200)`) for
realistic placement — no manual position computation needed.

### 4.4 Dynamic Obstacle Behavior Set

Use a fixed behavior menu for reproducibility:
1. `crossing_linear`: constant velocity crossings, 0.2-0.8 m/s.
2. `stop_and_go`: alternating move/pause windows (1-3 s).
3. `patrol_loop`: waypoint loops around room perimeter.
4. `drift_noise`: low-speed random-walk perturbations.

Dynamic obstacle ratio over training curriculum:
1. Stage A: 0%
2. Stage B: 10%
3. Stage C: 20%

### 4.5 Lighting and Camera Randomization

Following the NVIDIA tutorial pattern:

| Randomizer | API | Trigger |
|-----------|-----|---------|
| Dome light intensity/color | `infinigen_sdg_utils.register_dome_light_randomizer()` | Per-frame (`OnFrameEvent(1)`) |
| Scene lights (N=3-4) | `infinigen_sdg_utils.create_scene_lights()` + `randomize_lights()` | Per-environment |
| Shape distractor colors | `infinigen_sdg_utils.register_shape_distractors_color_randomizer()` | Per-frame |
| Camera poses | `infinigen_sdg_utils.randomize_camera_poses()` | Per-capture |

### 4.6 Traversability Constraints

Reject scene samples that violate:
1. Minimum free corridor width < 0.55 m.
2. Robot spawn or goal collides with geometry.
3. No feasible path from spawn to goal in coarse grid check.

## 5. Repository Changes

### 5.1 Scripts (scene generation pipeline)

Under `Scripts/scenegen/`:

| Script | Status | Purpose |
|--------|--------|---------|
| `build_asset_manifest.py` | **Done** | Scan local packs -> asset_manifest.jsonl |
| `validate_asset_pool.py` | **Done** | Quality gate checks (filesystem + optional USD) |
| `compose_scenes_replicator.py` | **Done** | Isaac Sim standalone: Infinigen rooms + Replicator SDG |
| `split_scene_dataset.py` | **Done** | Train/val/test split from scene_manifest.jsonl |
| `export_scene_stats.py` | **Done** | Report statistics on manifests |
| `generate_infinigen_rooms.sh` | **New** | WSL/Linux script for batch Infinigen room generation |

### 5.2 Generated Data Paths

Under `Assets/generated/`:

```
Assets/generated/
├── infinigen_rooms/          # Infinigen USD exports (from WSL)
│   ├── dining_room_1/
│   ├── dining_room_2/
│   ├── living_room_1/
│   └── ...
├── scenes/                   # Composed scene USDs (from Replicator)
│   ├── train/*.usd
│   ├── val/*.usd
│   └── test/*.usd
├── manifests/
│   ├── asset_manifest.jsonl  # Local asset pack inventory (done)
│   ├── scene_manifest.jsonl  # Composed scene metadata
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
└── sdg_output/               # Annotated captures (RGB, depth, segmentation)
```

### 5.3 Isaac Lab Integration Points

1. `strafer_env_cfg.py` — ProcDepth env configs (done, events need update)
2. `mdp/events.py` — Scene-aware obstacle placement event
3. `__init__.py` — New env ID registrations (done)
4. `test/env/test_env_registration.py` — Updated for 22 env count (done)

## 6. Environment Variants

Added (registered, functional as stubs):

| Env ID | Realism | Scene Source |
|--------|---------|-------------|
| `Isaac-Strafer-Nav-Real-ProcDepth-v0` | Realistic | Proc scenes |
| `Isaac-Strafer-Nav-Real-ProcDepth-Play-v0` | Realistic | Proc scenes (eval) |
| `Isaac-Strafer-Nav-Robust-ProcDepth-v0` | Robust | Proc scenes |
| `Isaac-Strafer-Nav-Robust-ProcDepth-Play-v0` | Robust | Proc scenes (eval) |

Guidelines:
1. Observation/action contract identical to existing Depth variants.
2. Start with 24 envs for proc-scene runs.
3. Preserve NoCam variants for fast baseline iteration.

## 7. Training Curriculum

### 7.1 Four-Stage Curriculum

| Stage | Env | Iterations | Purpose |
|------|-----|------------|---------|
| S0 | Real-NoCam | 600-1000 | Fast locomotion and goal-tracking baseline |
| S1 | Real-Depth | 300-600 | Depth encoder stabilization in simple scenes |
| S2 | Real-ProcDepth (static) | 400-800 | Geometry diversity adaptation |
| S3 | Robust-ProcDepth (dynamic) | 400-800 | Dynamic obstacle robustness |

### 7.2 Evaluation Protocol

Evaluate checkpoints on:
1. In-distribution train scenes.
2. Held-out synthetic test scenes.
3. Real-world clutter courses (same metric definitions).

Track:
1. Goal success rate.
2. Collision rate per meter.
3. Time-to-goal.
4. Recovery behavior after near-collision.

## 8. Success Criteria

Phase 6 is complete when all are met:
1. 200-scene library generated and versioned with manifests.
2. New proc-scene env variants train end-to-end without manual intervention.
3. Held-out synthetic test success improves by >= 15% over box-only baseline.
4. Real-world collision rate decreases by >= 25% at matched route set.
5. Regression tests pass for env registration and observation contract.

## 9. Implementation Roadmap

### 9.1 Phase 6a: Pipeline Foundation (Week 1)

| Step | Task | Status |
|------|------|--------|
| 6a.1 | Build asset manifest from local packs + quality gate | **Done** |
| 6a.2 | Install Infinigen on WSL, generate first batch of rooms | **Done** (2 rooms verified) |
| 6a.3 | Split scene dataset tooling | **Done** |
| 6a.4 | Register ProcDepth env variants in Isaac Lab | **Done** |
| 6a.5 | Rewrite compose_scenes_replicator.py to follow NVIDIA SDG pattern | **Done** |

**Exit criterion**: 10 Infinigen rooms exported,
compose_scenes_replicator.py loads them and spawns obstacle assets.
**Current**: 2 rooms generated and verified (dining_room_1, living_room_1).
`compose_scenes_replicator.py --check-only` confirms discovery.

### 9.2 Phase 6b: Scene Library Build-Out (Week 2)

| Step | Task | Duration |
|------|------|----------|
| 6b.1 | Generate 40 Infinigen rooms (10 per room type) | 1-2 days |
| 6b.2 | Compose 200 scenes via Replicator pipeline | 1-2 days |
| 6b.3 | Add dynamic obstacle behavior library | 1-2 days |
| 6b.4 | Add traversability/path feasibility filter | 1 day |
| 6b.5 | Wire Isaac Lab ProcDepth env to load composed scene USDs | 1 day |

**Exit criterion**: 200-scene library, 100-iteration training smoke test passes.

### 9.3 Phase 6c: Sim-to-Real Validation Sprint (Week 3)

| Step | Task | Duration |
|------|------|----------|
| 6c.1 | Baseline vs proc-scene ablation training | 2-3 days |
| 6c.2 | Held-out synthetic evaluation and failure analysis | 1 day |
| 6c.3 | Real robot route set validation | 1-2 days |
| 6c.4 | Finalize Phase 6 report in `docs/` | 0.5 day |

**Exit criterion**: meet Phase 6 success metrics above.

## 10. Immediate Next Steps

### 10.1 Infinigen Setup (WSL) — Verified Steps

```bash
# 1. System prerequisites (Ubuntu 22.04 on WSL)
sudo apt-get update && sudo apt-get install -y \
    python3-pip python3-venv git cmake g++ zip \
    libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev zlib1g-dev

# 2. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 3. Create conda env and install Infinigen
conda create -n infinigen python=3.11 -y
conda activate infinigen

git clone https://github.com/princeton-vl/infinigen.git ~/infinigen
cd ~/infinigen
git submodule update --init --recursive
INFINIGEN_MINIMAL_INSTALL=True pip install -e .
pip install pyyaml coacd

# 4. Compile terrain C++ libraries (required even for indoor generation)
make terrain

# 5. Verify: generate a test dining room (~3 min)
python -m infinigen_examples.generate_indoors \
  --seed 1 --task coarse \
  --output_folder /tmp/infinigen_test/dining_room_1 \
  -g singleroom fast_solve \
  -p 'compose_indoors.room_type="DiningRoom"' \
     'compose_indoors.restrict_single_supported_roomtype=True'

# 6. Export to USDC (~2 min, requires textures baking)
python -m infinigen.tools.export \
  --input_folder /tmp/infinigen_test/dining_room_1 \
  --output_folder /tmp/infinigen_test/export/dining_room_1 \
  -f usdc -r 1024 --omniverse
# Output: /tmp/infinigen_test/export/dining_room_1/export_scene.blend/export_scene.usdc
```

**Key findings from setup:**
- Use `-g singleroom fast_solve` gin configs: `singleroom` limits to 1 room,
  `fast_solve` disables terrain and reduces solver iterations.
- Gin overrides go after `-p` flag (not bare `--` arguments).
- The export tool creates a nested `export_scene.blend/` subdirectory containing
  `export_scene.usdc` + `textures/`.
- Room sizes vary significantly: dining room ~245MB, living room ~780MB exported.
- `zip` must be installed or the export will fail at the final packaging step.

### 10.1b Batch Generation

```bash
# Use the provided batch generation script:
bash /mnt/c/Worspace/Scripts/scenegen/generate_infinigen_rooms.sh

# Or generate a single room type with custom settings:
ROOMS_PER_TYPE=3 ROOM_TYPES="DiningRoom" \
  bash /mnt/c/Worspace/Scripts/scenegen/generate_infinigen_rooms.sh
```

### 10.2 Scene Composition (Windows, Isaac Sim)

```powershell
# Compose scenes from Infinigen rooms + local asset packs
cd IsaacLab
.\isaaclab.bat -p ..\Scripts\scenegen\compose_scenes_replicator.py `
  --config ..\Scripts\scenegen\config\strafer_sdg.yaml

# Split dataset
python ..\Scripts\scenegen\split_scene_dataset.py `
  --scene_manifest ..\Assets\generated\manifests\scene_manifest.jsonl

# Train on proc-scene variant
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py `
  --env Isaac-Strafer-Nav-Real-ProcDepth-v0 `
  --num_envs 24 --headless
```

## 11. References

- NVIDIA Infinigen + Replicator SDG tutorial (Isaac Sim 4.5.0):
  https://docs.isaacsim.omniverse.nvidia.com/4.5.0/replicator_tutorials/tutorial_replicator_infinigen_sdg.html
- Infinigen GitHub repository:
  https://github.com/princeton-vl/infinigen
- Infinigen Installation:
  https://github.com/princeton-vl/infinigen/blob/main/docs/Installation.md
- Infinigen Hello Room tutorial:
  https://github.com/princeton-vl/infinigen/blob/main/docs/HelloRoom.md
- Downloadable OpenUSD asset packs:
  https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html
- Existing project context:
  `docs/SIM_TO_REAL_PLAN.md`
