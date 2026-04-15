# Strafer Robot - Sim-to-Real

Sim-to-real development for the GoBilda Strafer mecanum robot.

This repository covers:

- Isaac Lab training on a Windows workstation **or NVIDIA DGX Spark**
- Jetson ROS2 runtime for real-robot execution
- shared sim-to-real contracts in `strafer_shared`
- VLM grounding + LLM planner services hosted off-robot (DGX Spark by default, Databricks Model Serving as an optional deployment target)
- synthetic data pipeline: Infinigen procedural scenes → scene metadata → 4-stage description generation → CLIP contrastive + VLM LoRA fine-tuning
- a single `.env` file + `env_setup.sh` for portable host configuration

<p align="center">
  <img src="docs/artifacts/strafer_top.jpeg" alt="Strafer robot, top view" width="31%"/>
  <img src="docs/artifacts/strafer_side.jpeg" alt="Strafer robot, side view" width="55%"/>
</p>

## Project Direction

The current architecture is split into four layers:

1. `strafer_lab`
   - Isaac Lab environments, training, and evaluation
2. `strafer_shared`
   - shared physical constants, mecanum kinematics, and policy I/O contract
3. `strafer_ros`
   - Jetson-side robot runtime, sensing, navigation, TF, and safety-critical execution
4. `strafer_autonomy` and `strafer_vlm`
   - planner and VLM services hosted off-robot first, with the executor staying robot-local

The current working pipeline is:

```text
user command (CLI)
  -> LLM planner service (DGX Spark, port 8200)
  -> bounded MissionPlan
  -> Jetson executor
  -> VLM grounding when needed (DGX Spark, port 8100)
  -> robot-local goal projection and Nav2 navigation
  -> real hardware
```

## Current State

### Simulation

The Isaac Lab environment is in good shape:

- 18 environment variants across realism and sensor modes
- mecanum kinematics and shared constants aligned with the robot
- sensor and actuator noise models for sim-to-real transfer
- PPO training pipeline for navigation policies

<p align="center">
  <a href="docs/artifacts/strafer_isaac_lab_test_drive.mp4">
    <img src="docs/artifacts/strafer_usd.png" alt="Click to view test drive video" width="50%"/>
  </a>
  <br/>
  <em>Robot USD model in the Isaac Sim editor - <a href="docs/artifacts/strafer_isaac_lab_test_drive.mp4">watch video</a></em>
</p>

<p align="center">
  <a href="docs/artifacts/strafer_infinitygen_scene.mp4">
    <img src="docs/artifacts/strafer_infinitygen_thumbnail.png" alt="Click to view Infinigen scene video" width="50%"/>
  </a>
  <br/>
  <em>Robot in a procedurally generated Infinigen apartment - <a href="docs/artifacts/strafer_infinitygen_scene.mp4">watch video</a></em>
</p>

### Robot Runtime

The robot-side ROS stack already has substantial coverage:

- `strafer_driver`
  - RoboClaw interface, wheel commands, odometry, joint states, watchdog
- `strafer_perception`
  - RealSense integration, depth downsampling, timestamp fixing, IMU filtering
- `strafer_description`
  - URDF and TF tree
- `strafer_slam`
  - RTAB-Map launch and tuning
- `strafer_navigation`
  - Nav2 launch and parameterization
- `strafer_bringup`
  - layered bringup launch files and validation helpers

### Autonomy and VLM

The full autonomy pipeline is operational end-to-end on real hardware:

- `strafer_autonomy`
  - Jetson-side executor with mission runner, skill dispatch, and HTTP clients
  - `strafer-executor` entry point reads `VLM_URL` / `PLANNER_URL` env vars and spins the command server
  - Operator CLI (`strafer-autonomy-cli submit/status/cancel`) for mission control
  - Alternative Databricks Model Serving clients (`DatabricksServingPlannerClient`, `DatabricksServingGroundingClient`) for when planner/VLM are hosted in a Databricks workspace instead of on LAN
- `strafer_vlm`
  - VLM grounding service (Qwen2.5-VL-3B-Instruct) on DGX Spark port 8100
  - Three endpoints: `POST /ground` (single-object grounding), `POST /describe` (free-text scene description), `POST /detect_objects` (multi-object detection, one call returns every visible `<ref>/<box>` pair — used by the semantic map)
- LLM planner service (Qwen3-4B) on DGX Spark port 8200
  - Two-stage architecture: LLM classifies commands into intent types, deterministic compiler expands each intent into a validated `MissionPlan`
  - **Nine intent types**: `go_to_target`, `wait_by_target`, `cancel`, `status`, plus `rotate` (relative degrees or cardinal direction), `go_to_targets` (ordered multi-target chaining), `describe` (operator-facing scene readback), `query` (semantic-map lookup with no navigation), `patrol` (waypoint cycle)
  - `POST /plan_with_grounding` agentic endpoint — planner pre-validates the target via a co-located VLM call before returning the plan, saving one LAN image round-trip per mission

The Jetson executor calls both services over LAN HTTP. All mission state, cancel, retry,
and safety-critical behavior stays local on the robot.

Implemented skills: `scan_for_target` (rotate-and-ground loop), `locate_semantic_target`,
`project_detection_to_goal_pose`, `navigate_to_pose`, `verify_arrival` (CLIP top-k ranking
against the semantic map, appended to every navigation-terminating plan), `describe_scene`,
`query_environment`, `rotate_by_degrees`, `orient_to_direction`, `wait`, `report_status`,
`cancel_mission`.

A Postman collection is included at `source/SImToRealLab.postman_collection.json` for
interactive testing of the DGX services.

<p align="center">
  <img src="docs/artifacts/hallway_grounding_example.png" alt="Hallway scene with yucca plant" width="35%"/>
  <img src="docs/artifacts/hallway_grounding_example_postman.png" alt="Postman grounding request finding a yucca plant" width="44%"/>
  <br/>
  <em>VLM grounding: "Green yucca plant end of hall" — Original scene (left) and Postman request showing output with bbox overlay (right)</em>
</p>

### Pipeline Snapshot

| Area | Status | Notes |
|---|---|---|
| CAD to USD | Done | Asset import, rigging, hierarchy cleanup |
| Isaac Lab environments | Done | Realism presets and sensor variants are in place |
| Shared kinematics and policy contract | Done | `strafer_shared` is the sim-to-real boundary |
| Jetson ROS driver and perception | Done | Driver, perception, URDF, SLAM, Nav2, goal projection |
| RL policy training | In progress | Navigation training is active in Isaac Lab |
| `strafer_inference` runtime | Planned | Policy runtime on the Jetson is still to be implemented |
| `strafer_autonomy` executor | Done | Mission runner, skill dispatch, CLI, HTTP clients, end-to-end on hardware |
| LLM planner service | Done | Qwen3-4B on DGX Spark:8200, two-stage intent→plan pipeline, 9 intent types, `/plan_with_grounding` agentic endpoint |
| VLM grounding service | Done | Qwen2.5-VL-3B on DGX Spark:8100, `/ground` + `/describe` + `/detect_objects` |
| Databricks backend (optional) | Done | `pyfunc` wrappers + serving clients — deploy path as an alternative to LAN HTTP |
| Synthetic data pipeline | Done (scripts) | Infinigen scene generation, metadata extraction, 4-stage description pipeline, CLIP contrastive + VLM LoRA data prep — tooling ready, full run awaits scene collection |
| End-to-end autonomy | Done | "go to the tennis ball" tested on real hardware via full pipeline |
| Full test suite | 353 passing | `strafer_autonomy` + `strafer_vlm` combined, excluding ROS-dependent tests |

## Hardware

| Component | Model | Purpose |
|---|---|---|
| Workstation | DGX Spark (or Windows PC + NVIDIA GPU) | VLM grounding, planner hosting, Isaac Lab training |
| Robot compute | Jetson Orin Nano | ROS2 runtime and robot-local execution |
| Camera | Intel RealSense D555 | RGB, depth, IMU |
| Motors | 4x GoBilda 5203 Yellow Jacket (19.2:1) | Mecanum drive |
| Motor controllers | 2x RoboClaw ST 2x45A | USB serial motor and encoder interface |
| Chassis | GoBilda Strafer v4 | 4-wheel mecanum platform |

## Repository Structure

```text
.env.example           template for per-machine configuration (copy to .env)
env_setup.sh           shell wrapper: loads .env + sets LD_PRELOAD on aarch64

source/
  strafer_lab/         Isaac Lab simulation, training, synthetic data pipeline
                         tools/       scene_labels, spatial_description, dataset_export
                         scripts/     prep_room_usds, extract_scene_metadata,
                                      generate_descriptions, finetune_clip,
                                      prepare_vlm_finetune_data
  strafer_shared/      shared constants, kinematics, policy I/O
  strafer_ros/         Jetson ROS2 packages
  strafer_autonomy/    autonomy schemas, executor, planner service, clients
                         clients/     HttpPlannerClient, HttpGroundingClient,
                                      DatabricksServingPlannerClient,
                                      DatabricksServingGroundingClient
                         planner/     FastAPI service, intent parser,
                                      plan compiler (9 intent types)
                         databricks/  MLflow pyfunc wrappers for model serving
  strafer_vlm/         VLM grounding service: /ground, /describe, /detect_objects

docs/
  SIM_TO_REAL_PLAN.md
  SIM_TO_REAL_TUNING_GUIDE.md
  STRAFER_AUTONOMY_ROADMAP.md
  STRAFER_AUTONOMY_ROS.md
  STRAFER_AUTONOMY_INTERFACES.md
  STRAFER_AUTONOMY_COMMAND_INGRESS.md
  STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md
  STRAFER_AUTONOMY_SYSTEMS_OVERVIEW.md
  STRAFER_AUTONOMY_VLM_GROUNDING.md
  STRAFER_AUTONOMY_LLM_PLANNER.md
```

## Quick Start

### Host configuration (any platform)

Every host uses the same two-file configuration mechanism:

```bash
cp .env.example .env
$EDITOR .env                 # fill in absolute paths for this machine
source env_setup.sh          # loads .env into the current shell
```

`env_setup.sh` is idempotent, loads `.env`, and applies shell-level state that
Python code cannot manage itself — on aarch64 hosts (DGX Spark, Grace) it
prepends `/lib/aarch64-linux-gnu/libgomp.so.1` to `LD_PRELOAD` so Isaac Sim's
bundled torch can find libgomp. It also creates `<INFINIGEN_ROOT>/blender`
as a symlink to the built Blender 4.2 binary when both `INFINIGEN_ROOT` and
`STRAFER_BLENDER_BIN` are set.

`.env.example` documents every variable: `STRAFER_ROOT`,
`STRAFER_BLENDER_BIN`, `INFINIGEN_ROOT`, `ISAACSIM_PATH`, `HF_HOME`. `.env` is
machine-specific and gitignored.

### Simulation on DGX Spark (preferred)

DGX Spark (Grace CPU + Blackwell GB10 GPU, aarch64 Ubuntu 24.04, CUDA 13.0) is
the primary host for Isaac Lab, Isaac Sim, the VLM/planner services, and the
synthetic data pipeline. Isaac Sim aarch64 builds are officially supported
only on DGX Spark at the moment, so this is the first-class path.

Three conda environments partition the stack:

| Env | Purpose | Key contents |
|---|---|---|
| `env_phase15` | Isaac Sim + Isaac Lab + `strafer_lab` editable | Python 3.11, Isaac Sim 5.1.x built from source, Isaac Lab v2.3.2, `pxr` USD bindings via `.pth` |
| `env_infinigen` | Infinigen procedural scene generation | Python 3.11, source-built `bpy==4.2.0` wheel, Infinigen 1.19.x editable (`--no-deps`) |
| `.venv_vlm` | VLM + planner services, batch processing scripts, test suite | Python 3.12, PyTorch cu128, transformers, `strafer_vlm`, `strafer_autonomy` |

See `docs/PHASE_15_DGX.md` for the end-to-end install script (clone Isaac Sim,
build from source, install Isaac Lab, create conda envs, wire `pxr` into
`env_phase15`, etc.). For the aarch64-only Blender 4.2 source build that
Infinigen depends on, see the `README.md` inside the sibling
`~/Workspace/blender-build/` directory — the build artifacts live outside
this repo because they are machine-specific.

### Simulation on Windows

```powershell
# Activate the Isaac Lab environment
& C:\Workspace\venv_isaac\Scripts\Activate.ps1

# Install local packages
python -m pip install -e source/strafer_lab
python -m pip install -e source/strafer_shared

# List Strafer environments
python -c "import strafer_lab; import gymnasium as gym; print([e for e in gym.envs.registry.keys() if 'Strafer' in e])"
```

#### Training

Run training from `C:\Workspace\Sim2RealLab\IsaacLab`:

```powershell
cd IsaacLab

# Recommended first run: NoCam
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 512

# Depth
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-Depth-v0 --num_envs 32

# Full RGB + depth
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-v0 --num_envs 32

# Headless
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 4096 --headless
```

#### Monitoring

```powershell
tensorboard --logdir logs\rsl_rl\strafer_navigation
```

#### Test or play

```powershell
cd IsaacLab
.\isaaclab.bat -p ..\source\strafer_lab\run_tests.py all
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Strafer-Nav-Real-NoCam-Play-v0 --num_envs 50
```

### VLM Grounding on Windows

```powershell
# Install core + inference + live dependencies
python -m pip install -e "source/strafer_vlm[qwen,live,service]"
```

```powershell
# Launch the grounding service (downloads model on first run)
uvicorn strafer_vlm.service.app:create_app --factory --host 0.0.0.0 --port 8100
```

```powershell
# Single-image grounding smoke test (CLI, no service needed)
python -m strafer_vlm.test_qwen25vl_grounding `
  --image docs\artifacts\strafer_top.jpeg `
  --prompt "the robot chassis"
```

```powershell
# Live grounding against a local camera (CLI, no service needed)
python -m strafer_vlm.live_qwen25vl_grounding --source 0 --prompt "Locate: the robot chassis"
```

See `source/strafer_vlm/README.md` for env vars, API details, and curl/PowerShell examples.
Import `source/SImToRealLab.postman_collection.json` into Postman for interactive testing.

### Robot Runtime on Jetson

```bash
# Build all ROS2 packages
make build

# Install autonomy package
pip install -e source/strafer_autonomy
```

```bash
# Navigation stack only (driver + perception + SLAM + Nav2)
make launch

# Full autonomy stack (navigation + goal projection + executor → DGX services)
VLM_URL=http://192.168.50.196:8100 PLANNER_URL=http://192.168.50.196:8200 \
    make launch-autonomy
```

```bash
# Submit a mission
strafer-autonomy-cli submit "go to the tennis ball" --detach
strafer-autonomy-cli status
strafer-autonomy-cli cancel
```

```bash
# Verify SLAM, motion, and depth
python3 source/strafer_ros/ros_test_slam.py --drive forward --duration 3 --speed 1.0

# Verify perception stack
python3 source/strafer_ros/ros_test_perception.py --record
```

For the Jetson-side package inventory and architecture, see `docs/STRAFER_AUTONOMY_ROS.md`.

## Key Design Decisions

### Monorepo with shared contracts

Sim, ROS, autonomy, and VLM code stay in one repository so the core contracts do not drift.

### `strafer_shared` is the sim-to-real boundary

The real robot and Isaac Lab must agree on:

- physical constants
- mecanum kinematics
- policy observation and action contract

### Keep execution local

Safety-critical execution remains on the robot.

Planner and VLM services may move between workstation and cloud without changing the robot-side execution boundary.

### Keep the VLM narrow

`strafer_vlm` does semantic grounding only.

Depth projection, TF transforms, reachability, and motion execution remain robot-local.

### Planner and executor stay separate

The LLM planner is a text-to-plan service.

The Jetson executor owns mission state, validation, retries, cancel, and robot-facing control.

## Environments

| Realism | Sensors | Train ID | Obs dims |
|---|---|---|---|
| Ideal | Full (RGB + depth) | `Isaac-Strafer-Nav-v0` | 19215 |
| Ideal | Depth-only | `Isaac-Strafer-Nav-Depth-v0` | 4815 |
| Ideal | NoCam | `Isaac-Strafer-Nav-NoCam-v0` | 15 |
| Realistic | Full | `Isaac-Strafer-Nav-Real-v0` | 19215 |
| Realistic | Depth-only | `Isaac-Strafer-Nav-Real-Depth-v0` | 4815 |
| Realistic | NoCam | `Isaac-Strafer-Nav-Real-NoCam-v0` | 15 |
| Robust | Full | `Isaac-Strafer-Nav-Robust-v0` | 19215 |
| Robust | Depth-only | `Isaac-Strafer-Nav-Robust-Depth-v0` | 4815 |
| Robust | NoCam | `Isaac-Strafer-Nav-Robust-NoCam-v0` | 15 |

## Documentation

Core docs:

- [Sim-to-Real Plan](docs/SIM_TO_REAL_PLAN.md)
- [Sim-to-Real Tuning Guide](docs/SIM_TO_REAL_TUNING_GUIDE.md)
- [Strafer Autonomy Roadmap](docs/STRAFER_AUTONOMY_ROADMAP.md)
- [Strafer Autonomy ROS](docs/STRAFER_AUTONOMY_ROS.md)
- [Strafer Autonomy Interfaces](docs/STRAFER_AUTONOMY_INTERFACES.md)
- [Strafer Autonomy Command Ingress](docs/STRAFER_AUTONOMY_COMMAND_INGRESS.md)
- [Strafer Autonomy MVP Runtime Decision](docs/STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md)
- [Strafer Autonomy Systems Overview](docs/STRAFER_AUTONOMY_SYSTEMS_OVERVIEW.md)
- [Strafer Autonomy VLM Grounding](docs/STRAFER_AUTONOMY_VLM_GROUNDING.md)
- [Strafer Autonomy LLM Planner](docs/STRAFER_AUTONOMY_LLM_PLANNER.md)

Supporting docs:

- [Wiring Guide](docs/WIRING_GUIDE.md)
- [D555 IMU Kernel Fix](docs/D555_IMU_KERNEL_FIX.md)
- [Isaac Lab README](IsaacLab/README.md)

## References

- [GoBilda Strafer Chassis](https://www.gobilda.com/strafer-chassis-kit-v4/)
- [GoBilda 5203 Motor](https://www.gobilda.com/5203-series-yellow-jacket-planetary-gear-motor-19-2-1-ratio-24mm-length-8mm-rex-shaft-312-rpm-3-3-5v-encoder/)
- [RoboClaw ST 2x45A](https://www.gobilda.com/roboclaw-st-2x45a-motor-controller/)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [ROS 2 Humble](https://docs.ros.org/en/humble/)

## License

MIT
