# strafer_lab

Isaac Lab extension for the Strafer mecanum robot: RL navigation policy training, synthetic-data generation pipeline, and the Isaac Sim ROS 2 bridge / sim-in-the-loop harness.

`strafer_lab` is the simulation side of the sim-to-real pipeline. It
registers a family of Gym environments for PPO navigation training,
drives Infinigen for procedural indoor scene generation, extracts
semantic labels from those scenes into `semanticLabel` USD prim
attributes that Replicator's bbox annotator can consume, runs a 4-stage
description pipeline over teleop frames for VLM / CLIP fine-tune data
prep, and (in sim-in-the-loop mode) bridges simulated sensor streams
onto real robot ROS topics so the Jetson autonomy stack can drive the
sim unchanged. It runs on the DGX Spark (aarch64, preferred) or a
Windows workstation.

## Role in the system

| Surface | Host | Consumes | Produces |
|---|---|---|---|
| RL training environments | DGX Spark / Windows GPU | Gym reset → step loop | PPO checkpoints under `logs/rsl_rl/strafer_navigation/` |
| Synthetic-data pipeline | DGX Spark | Infinigen + teleop frames | Perception frames, scene metadata, VLM / CLIP SFT datasets |
| Isaac Sim ROS 2 bridge | DGX Spark | `/cmd_vel` from LAN (Nav2 / rqt) | Simulated D555 + `/strafer/odom` + TF on the wire |
| Sim-in-the-loop harness | DGX Spark | Jetson executor via `execute_mission` + mission metadata | Reachability-labelled `frames.jsonl` datasets |

Sibling packages it interacts with:

- **`strafer_shared`** — physical constants, mecanum kinematics, policy I/O contract. `strafer_lab` and `strafer_ros` agree here so trained policies transfer unchanged.
- **`strafer_autonomy`** — sim-in-the-loop harness submits missions to the Jetson executor's `execute_mission` action. Some synthetic-data tools (`scene_labels`, `spatial_description`, `dataset_export`) have unit tests under `source/strafer_autonomy/tests/` — the `strafer_autonomy` conftest installs a namespace stub so those tests can import `strafer_lab.tools.*` without loading the Isaac Sim-dependent `strafer_lab/__init__.py`.
- **`strafer_vlm`** — consumes grounding-formatted JSONL from `prepare_vlm_finetune_data.py` for LoRA fine-tuning. The description pipeline uses Qwen2.5-VL-7B loaded standalone (NOT the 3B model `strafer_vlm` serves), so the fine-tune target is never fed its own outputs.
- **`strafer_ros`** — the sim-in-the-loop bridge publishes on the same topic names the real robot's Jetson stack publishes (`/d555/color/image_raw`, `/d555/depth/image_rect_raw`, `/strafer/odom`, TF), so the Jetson autonomy stack consumes sim sensors without code changes.

## What ships today

- **31 registered Gym environments** across 3 realism levels (Ideal / Realistic / Robust) × 3 sensor modes (Full RGB+Depth / Depth-only / NoCam) + 4 Infinigen-scene depth variants + 1 Infinigen-perception Play variant + 8 ProcRoom variants, each with Train and Play flavors. See Contracts below.
- **Sim-to-real contract presets** (`tasks/navigation/sim_real_cfg.py`) — `IDEAL_SIM_CONTRACT`, `REAL_ROBOT_CONTRACT`, `ROBUST_TRAINING_CONTRACT` covering motor time constant, command delay, slew rate, IMU / encoder / depth noise, and control jitter.
- **Mecanum motor model** (`assets/strafer.py`) — `DCMotorCfg` with torque-speed curve, paired with the three-layer action pipeline (command delay → slew rate limit → first-order motor dynamics filter) in `tasks/navigation/mdp/actions.py`.
- **Synthetic-data pipeline** — Infinigen scene generation orchestrator, scene metadata extractor, 4-stage description pipeline (programmatic spatial analysis → VLM description generation → ground-truth validation → human spot-check reservoir), OpenCLIP contrastive fine-tuning with ONNX export, comprehensive VLM LoRA SFT data prep.
- **Isaac Sim ROS 2 bridge** (`bridge/`) — config + OmniGraph builder that wires simulated sensors onto the real robot's ROS topic names. Runs inside Kit via `isaacsim.ros2.bridge`.
- **Sim-in-the-loop harness** (`sim_in_the_loop/`) — pure-Python orchestrator with injectable env / mission-API adapters. Submits Jetson missions, captures reachability-labelled frames into `frames.jsonl` datasets that the existing description and SFT-prep pipelines consume unchanged.
- **Graceful Isaac-Sim fallback** (`__init__.py`) — subpackages that need Kit (`tasks`, `assets`) fail quietly with `ModuleNotFoundError`, so `strafer_lab.tools.*` stays importable from plain Python envs without `AppLauncher`.

## Contracts

### Gym environment registry (`strafer_lab.tasks.navigation`)

Registration happens in `tasks/navigation/__init__.py` via `gym.register`. Train and Play IDs share config; Play variants use fewer envs for evaluation.

| Family | Realism | Sensors | Train ID | Obs dims |
|---|---|---|---|---|
| Base | Ideal | Full (RGB+Depth) | `Isaac-Strafer-Nav-v0` | 19,219 |
| Base | Ideal | Depth-only | `Isaac-Strafer-Nav-Depth-v0` | 4,819 |
| Base | Ideal | NoCam | `Isaac-Strafer-Nav-NoCam-v0` | 19 |
| Base | Realistic | Full | `Isaac-Strafer-Nav-Real-v0` | 19,219 |
| Base | Realistic | Depth-only | `Isaac-Strafer-Nav-Real-Depth-v0` | 4,819 |
| Base | Realistic | NoCam | `Isaac-Strafer-Nav-Real-NoCam-v0` | 19 |
| Base | Robust | Full | `Isaac-Strafer-Nav-Robust-v0` | 19,219 |
| Base | Robust | Depth-only | `Isaac-Strafer-Nav-Robust-Depth-v0` | 4,819 |
| Base | Robust | NoCam | `Isaac-Strafer-Nav-Robust-NoCam-v0` | 19 |
| Infinigen depth | Realistic / Robust | Depth-only | `Isaac-Strafer-Nav-Real-InfinigenDepth-v0` / `Isaac-Strafer-Nav-Robust-InfinigenDepth-v0` | 4,819 |
| Infinigen perception (Play only) | Realistic | 640×360 RGB + depth | `Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0` | — |
| ProcRoom | Realistic / Robust | NoCam / Depth | 4 Train + 4 Play | 19 / 4,819 |

Add each `-Play-v0` suffix to get the evaluation variant.

### Observation vector (15-19D proprioceptive base, optionally + depth + RGB)

| Component | Dims | Source |
|---|---|---|
| IMU linear acceleration | 3 | D555 BMI055 accelerometer |
| IMU angular velocity | 3 | D555 BMI055 gyroscope |
| Wheel encoder velocities | 4 | GoBilda 5203 (537.7 PPR) |
| Goal position (relative) | 2 | `(x, y)` in robot frame |
| Goal distance | 1 | Euclidean to goal |
| Goal heading | 1 | Angular error |
| Body velocity | 2 | `(vx, vy)` |
| Last action | 3 | Previous `[vx, vy, omega]` |
| Depth image (optional) | 4,800 | D555 depth 80×60, normalized |
| RGB image (optional) | 14,400 | D555 RGB 80×60×3, normalized |

### Sim-to-real contract presets

| Parameter | Ideal | Realistic | Robust |
|---|---|---|---|
| Motor dynamics | Disabled | τ=50ms, range [30, 80]ms | τ=60ms, range [20, 100]ms |
| Command delay | 0 steps | 1 step, range [0, 2] | 1 step, range [0, 4] |
| Slew rate limit | ∞ | 100 rad/s² | 80 rad/s² |
| Control jitter | 0 % | ±5 % | ±10 % |
| IMU accel noise density | 0 | 0.0098 m/s²/√Hz | 0.015 m/s²/√Hz |
| IMU gyro noise density | 0 | 0.00024 rad/s/√Hz | 0.00036 rad/s/√Hz |
| Encoder velocity noise σ | 0 | 0.02 | 0.05 |
| Depth disparity noise | 0 | 0.08 px | 0.16 px |
| Sensor failure events | Disabled | Disabled | Enabled |
| Domain randomization | 0× | 1× | 1.5× |

Full actuator and sensor tuning methodology is in [`docs/SIM_TO_REAL_TUNING_GUIDE.md`](../../docs/SIM_TO_REAL_TUNING_GUIDE.md).

### Policy I/O contract

Lives in `strafer_shared.policy_interface` — the single source of truth for how observations are assembled and actions are interpreted on both the sim and real sides.

```python
PolicyVariant.NOCAM   # 19 dims, proprioceptive only
PolicyVariant.DEPTH   # 4,819 dims, + depth
# (Full 19,219-dim variant exists but is not the sim-to-real MVP target.)

assemble_observation(raw, variant)    # normalize + concatenate raw sensor values
interpret_action(action)               # [-1, 1] → physical (vx, vy, omega)
load_policy(path, variant)             # .pt / .onnx → callable
```

If the Isaac-side observation layout changes, `strafer_shared.policy_interface` is the file that changes — never local one-off logic in the Jetson runtime.

### Scripts and tools inventory

**RL training / evaluation** (require Isaac Sim + `AppLauncher`):

| Path | Purpose |
|---|---|
| `Scripts/train_strafer_navigation.py` | PPO training wrapper with video capture, DAPG / GAIL auxiliary losses, resume, headless |
| `Scripts/test_strafer_env.py` | Motion-pattern smoke test (forward / strafe / rotate / circle / figure8) |

**Synthetic-data pipeline** (mostly plain Python, no Isaac Sim):

| Path | Purpose | Depends on |
|---|---|---|
| `scripts/prep_room_usds.py` | Orchestrate Infinigen scene generation (`generate`, `ingest`, `presets` subcommands) | `INFINIGEN_ROOT`, `STRAFER_INFINIGEN_PYTHON`, `STRAFER_ISAACLAB_PYTHON` |
| `scripts/postprocess_scene_usd.py` | Bake colliders + ceiling-light emitters into an Infinigen-exported USDC (called by `prep_room_usds.py` after export, or run manually on existing scenes) | `pxr` (env_phase15) |
| `scripts/generate_scenes_metadata.py` | Walk `Assets/generated/scenes/` and author the combined `scenes_metadata.json` with per-scene spawn points + floor top Z | `pxr` (env_phase15) |
| `scripts/extract_scene_metadata.py` | Serialize Blender `State` (rooms, polygons, semantic tags, relations) into `scene_metadata.json`; label USD prims with `semanticLabel` | `bpy` (inside Blender subprocess), optional `pxr` |
| `scripts/generate_descriptions.py` | 4-stage description pipeline: programmatic spatial → Qwen2.5-VL-7B standalone → ground-truth filter → reservoir sampling for human spot-check | `scene_metadata.json`, `transformers`, Qwen2.5-VL-7B |
| `scripts/prepare_vlm_finetune_data.py` | Comprehensive VLM LoRA SFT prep: single-object grounding + 1:3 negatives + ~20% multi-object + ~10% description preservation | Perception frames, scene metadata, Stage-2 descriptions |
| `scripts/finetune_clip.py` | OpenCLIP ViT-B/32 contrastive fine-tune with MLflow tracking, exports `clip_visual.onnx` + `clip_text.onnx` for the Jetson semantic map | `open_clip_torch`, CSV from `dataset_export` |
| `scripts/collect_demos.py` | Gamepad teleop for RL demo collection (DAPG / GAIL sources) | Isaac Sim |
| `scripts/collect_perception_data.py` | Gamepad teleop for perception-data collection through the Infinigen perception env. Writes per-episode dirs matching what `generate_descriptions.py` consumes | Isaac Sim, Replicator |
| `scripts/run_sim_in_the_loop.py` | Launch Isaac Sim with the ROS 2 bridge, run in `--mode bridge` (manual Nav2 drive) or `--mode harness` (harness runs Jetson missions, captures reachability-labelled dataset) | Isaac Sim, `isaacsim.ros2.bridge`, Jetson on LAN |

**Runtime helpers** (`strafer_lab.tools.*`, plain Python, no Isaac Sim):

| Module | Exports |
|---|---|
| `scene_labels` | `get_scene_metadata`, `iter_rooms`, `iter_objects`, `get_scene_label_set`, `get_room_at_position`, `get_objects_in_room` — typed accessors over `scene_metadata.json` |
| `spatial_description` | `SpatialDescriptionBuilder`, `quat_to_yaw`, `classify_region`, `classify_bearing` — Stage-1 factual spatial relations |
| `dataset_export` | `export_clip_csv`, `export_vlm_grounding_jsonl`, `run_export`, `pixel_bbox_to_qwen`, `format_qwen_grounding_answer` — minimal CLIP CSV + VLM SFT JSONL (ProcRoom frames excluded) |
| `bbox_extractor` | `ReplicatorBboxExtractor`, `parse_bbox_data`, `DetectedBbox` — wraps Replicator's `bounding_box_2d_tight` annotator |
| `perception_writer` | `PerceptionFrameWriter`, `PerceptionWriterStats` — per-frame episode writer matching the layout `generate_descriptions.py` / `prepare_vlm_finetune_data.py` consume |
| `infinigen_label_parser` | Infinigen-specific semantic label normalization |

**Isaac Sim subpackages** (require Kit runtime):

| Subpackage | Purpose |
|---|---|
| `strafer_lab.assets` | `ArticulationCfg` for the Strafer robot + D555 camera configs |
| `strafer_lab.tasks.navigation` | Environment configs, MDP components (actions, observations, rewards, terminations, events, curriculums, commands), agent configs |
| `strafer_lab.bridge` | ROS 2 bridge config + OmniGraph builder |
| `strafer_lab.sim_in_the_loop` | Harness + runtime adapters + mission generator |

## Install

### DGX Spark (aarch64, preferred)

```bash
# 1. Source the shell wrapper (loads .env: STRAFER_BLENDER_BIN, ISAACSIM_PATH, HF_HOME...)
cd /home/zachoines/Workspace/Sim2RealLab
source env_setup.sh

# 2. Activate the conda env created via `isaaclab.sh -c env_phase15`
conda activate env_phase15

# 3. Install strafer_lab editable. --no-build-isolation is required because
#    a transitive dep (flatdict) uses legacy pkg_resources.
pip install --no-build-isolation -e source/strafer_lab
```

For the Infinigen-only conda env (`env_infinigen`) holding the aarch64 source-built `bpy==4.2.0` wheel, see the `README.md` inside the sibling `~/Workspace/blender-build/` directory — those artifacts live outside this repo because they are machine-specific.

Smoke test:

```bash
python -c "
import strafer_lab
from strafer_lab.tools.scene_labels import get_scene_label_set
from strafer_lab.tools.spatial_description import SpatialDescriptionBuilder
from strafer_lab.tools.dataset_export import run_export
print('strafer_lab tools OK')
"
```

### Windows workstation

```powershell
# From the workspace root, with the Isaac Lab venv active
& C:\Workspace\venv_isaac\Scripts\Activate.ps1
cd C:\Workspace
python -m pip install -e source/strafer_lab source/strafer_shared
```

Three environments partition the DGX stack:

| Env | Purpose | Key contents |
|---|---|---|
| `env_phase15` | Isaac Sim + Isaac Lab + `strafer_lab` | Python 3.11, Isaac Sim 5.1 (source build), Isaac Lab 2.3.2, `pxr` via `.pth` |
| `env_infinigen` | Infinigen procedural scene generation | Python 3.11, source-built `bpy==4.2.0` wheel, Infinigen 1.19.x editable `--no-deps` |
| `.venv_vlm` | VLM / planner services, batch scripts, tests | Python 3.12, PyTorch cu128, transformers, `strafer_vlm`, `strafer_autonomy` |

## Run

### List registered environments

```bash
python -c "import strafer_lab; import gymnasium as gym; \
  print([e for e in gym.envs.registry if 'Strafer' in e])"
```

### Train a PPO policy (DGX / Linux)

```bash
cd /home/zachoines/Workspace/Sim2RealLab

# NoCam Realistic (19 obs dims, fastest; recommended first run)
../IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 512 --headless

# Depth Realistic (4,819 obs dims; cameras auto-enabled)
../IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-Depth-v0 --num_envs 32 --headless

# Full Realistic
../IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-v0 --num_envs 32 --headless

# With video capture (overhead MP4 every 500 env steps)
../IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
    --num_envs 64 --max_iterations 3000 \
    --video --video_length 200 --video_interval 500 --headless

# Resume from checkpoint
../IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-NoCam-v0 \
    --resume logs/rsl_rl/strafer_navigation/run_XXXXX/model_500.pt
```

Windows equivalents use `.\IsaacLab\isaaclab.bat` and the `C:\Workspace` paths. Checkpoints land in `logs/rsl_rl/strafer_navigation/<timestamp>/`. TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/strafer_navigation
```

### Evaluate a policy

```bash
../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Strafer-Nav-Real-NoCam-Play-v0 --num_envs 50
```

Fast-path gym evaluation (no ROS overhead):

```python
from strafer_shared.policy_interface import load_policy, PolicyVariant
import gymnasium as gym, strafer_lab

env = gym.make("Isaac-Strafer-Nav-Real-NoCam-v0", num_envs=1)
policy = load_policy("logs/best_model.pt", PolicyVariant.NOCAM)
obs, info = env.reset()
done = False
while not done:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
```

### Synthetic-data pipeline

```bash
# 1. Generate Infinigen scenes
python scripts/prep_room_usds.py generate --preset dgx

# 2. Label USD prims + extract metadata
python scripts/extract_scene_metadata.py --scene Assets/generated/scenes/kitchen_01

# 3. Collect teleop perception data (requires Isaac Sim)
isaaclab -p scripts/collect_perception_data.py \
    --scene scene_001 --output data/perception/ --max-episodes 20

# 4. Run description pipeline
python scripts/generate_descriptions.py \
    --perception-root data/perception --output-root data/descriptions

# 5a. Export CLIP + basic VLM datasets
python -c "from strafer_lab.tools.dataset_export import run_export; \
  run_export('data/perception', 'data/descriptions', 'data/clip_descriptions', 'data/vlm')"

# 5b. Full VLM SFT prep
python scripts/prepare_vlm_finetune_data.py \
    --perception-root data/perception --descriptions-root data/descriptions \
    --scene-metadata-dir Assets/generated/scenes --output data/vlm_finetune

# 6. CLIP fine-tune (exports ONNX for the Jetson semantic map)
python scripts/finetune_clip.py \
    --data data/clip_descriptions/clip_descriptions.csv \
    --image-root data/perception --epochs 10 --output models/clip_finetuned/
```

### Sim-in-the-loop bridge / harness

```bash
source env_setup.sh

# Manual / Nav2 mode: DGX publishes simulated sensors, Jetson drives via Nav2
isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py

# Reachability-labelled dataset capture (harness submits missions to Jetson)
isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
    --mode harness \
    --scene-metadata Assets/generated/scenes/kitchen_01/scene_metadata.json \
    --scene-usd Assets/generated/scenes/kitchen_01/scene.usdc \
    --output data/sim_in_the_loop/kitchen_01
```

Jetson side (either mode) launches `bringup_sim_in_the_loop.launch.py` from [`strafer_ros`](../strafer_ros/README.md).

## Design

**Simulation is the source of truth for physics, noise, and observation layout.** Every constant the real robot uses (wheel radius, PID, encoder PPR, velocity limits) lives in `strafer_shared.constants` and is imported by both sides. The sim-to-real "envelope" is the `Realistic` / `Robust` preset — if the real hardware falls within that envelope, the policy transfers.

**Three-layer actuator pipeline models real-world imperfections.** Command delay (USB serial latency) → slew rate limiting (motor + gearbox inertia) → first-order motor dynamics filter (combined electrical + mechanical τ). The filter's time constant is the single most important sim-to-real parameter; target real τ in `[20, 100]` ms to stay inside the Robust envelope.

**The policy contract is hermetic.** `strafer_shared.policy_interface.assemble_observation()` and `interpret_action()` are the only places observations and actions cross the boundary between Python / NumPy and the policy's `[-1, 1]` tensor space. The gym environment and the ROS inference node both go through them, never around them.

**Fine-tune data stays separate from fine-tune target.** The 4-stage description pipeline uses Qwen2.5-VL-7B loaded standalone via `transformers.AutoModelForVision2Seq` — it does NOT call the 3B `strafer_vlm` service. Feeding the fine-tune target's own outputs back as training data causes collapse.

**ProcRoom scenes are RL-only, never perception training.** Primitive-shape rooms (solid-color boxes, cylinders) do not transfer to real indoor environments for VLM / CLIP training. Every perception export excludes `scene_type == "procroom"`.

**`strafer_lab.tools.*` stays importable without Isaac Sim.** `__init__.py` swallows `ModuleNotFoundError` from `omni` / `pxr` when the Kit runtime is absent. This is why the synthetic-data pipeline runs from plain Python envs and why some tool tests live under `source/strafer_autonomy/tests/` (using a conftest stub).

**Isaac Sim ROS 2 bridge publishes on the real robot's topic names.** Sim-in-the-loop runs the Jetson autonomy stack against simulated sensors without any code changes in `strafer_ros` or `strafer_autonomy`. That only works because the bridge preserves topic names, frame IDs, and sensor-msg shapes.

**`AppLauncher` must boot before `import strafer_lab.tasks`.** Importing the task package transitively pulls `isaaclab.managers`, which imports `omni.timeline`, which only exists after Kit boots. Scripts enforce this ordering; new scripts must follow the same rule.

## Testing

All test suites launch Isaac Sim headlessly via the root `test/conftest.py`. Use `run_tests.py` for clean output (direct pytest is drowned by Kit startup noise, and the root conftest calls `os._exit()` before pytest can print its summary).

```bash
cd /home/zachoines/Workspace/Sim2RealLab/source/strafer_lab
python run_tests.py all

# Subset for fast iteration
python run_tests.py noise_models depth_noise imu sensors
python run_tests.py terminations events commands observations curriculums
```

Available suites: `terminations`, `events`, `commands`, `observations`, `curriculums`, `rewards`, `sensors`, `actions`, `env`, `noise_models`, `depth_noise`, `imu`.

Full run takes ~30-45 min on DGX Spark (noise_models alone is ~15 min). The wrapper runs suites that need process isolation (`depth_noise`, `rewards`, `imu`) in separate subprocesses because Isaac Sim's `SimulationContext` is a singleton.

Plain-Python tool tests run from `.venv_vlm` via the namespace stub:

```bash
python -m pytest source/strafer_autonomy/tests/test_scene_labels.py \
                 source/strafer_autonomy/tests/test_spatial_description.py \
                 source/strafer_autonomy/tests/test_dataset_export.py -v
```

## Deferred / known limitations

Tracked in [`docs/DEFERRED_WORK.md`](../../docs/DEFERRED_WORK.md). Items currently open:

- **Electronics masses in the USD** — RoboClaws, Jetson, buck converter, D555 camera meshes + masses are TODO in `Scripts/setup_physics.py`. Without them, the simulated chassis inertia underestimates the real robot.
- **Policy export to `.pt` / ONNX** — `export_policy_as_jit()` wrapper and benchmark script pending; gates Jetson deployment through `strafer_inference`.
- **`strafer_inference` Jetson package** — the deployment target for trained policies does not exist yet; see [`strafer_ros`](../strafer_ros/README.md) for the interface side.
- **Goal position noise during final pre-deployment training** — for VLM-sourced goals, retrain with `goal_position_noise_std: 0.2-0.3 m` in `commands.py` to match Qwen2.5-VL-3B localization error; without this, the policy oscillates at deployment.

## References

- [`source/strafer_shared/`](../strafer_shared/) — authoritative physical constants, mecanum kinematics, policy I/O contract.
- [`source/strafer_ros/README.md`](../strafer_ros/README.md) — real-robot counterpart; consumes the same policy interface.
- [`source/strafer_autonomy/README.md`](../strafer_autonomy/README.md) — Jetson executor that sim-in-the-loop drives.
- [`source/strafer_vlm/README.md`](../strafer_vlm/README.md) — grounding service + LoRA fine-tuning tooling consuming datasets from this pipeline.
- [`docs/SIM_TO_REAL_TUNING_GUIDE.md`](../../docs/SIM_TO_REAL_TUNING_GUIDE.md) — deep-dive actuator / sensor alignment procedure pairing sim presets with hardware measurements.
- [`docs/VALIDATE_ISAAC_SIM_AND_INFINIGEN.md`](../../docs/VALIDATE_ISAAC_SIM_AND_INFINIGEN.md) — install + smoke-test runbook gating this package.
- [`docs/DEFERRED_WORK.md`](../../docs/DEFERRED_WORK.md) — open items.
