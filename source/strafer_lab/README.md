# strafer_lab

Isaac Lab extension for the Strafer mecanum robot: RL navigation policy training, synthetic-data generation pipeline, and the Isaac Sim ROS 2 bridge / sim-in-the-loop harness.

`strafer_lab` is the simulation side of the sim-to-real pipeline. It
registers a family of Gym environments for PPO navigation training,
drives Infinigen for procedural indoor scene generation, embeds each
scene's labeled metadata into the USD's `customData` and applies the
`UsdSemantics` labels Replicator's bbox annotator boxes on, runs a 4-stage
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
| Sim-in-the-loop harness | DGX Spark | Jetson executor via `execute_mission` + mission metadata | LeRobot v3 datasets (one episode per mission, outcome-labelled) |

Sibling packages it interacts with:

- **`strafer_shared`** — physical constants, mecanum kinematics, policy I/O contract. `strafer_lab` and `strafer_ros` agree here so trained policies transfer unchanged.
- **`strafer_autonomy`** — sim-in-the-loop harness submits missions to the Jetson executor's `execute_mission` action. Some synthetic-data tools (`scene_labels`, `spatial_description`, `dataset_export`) have unit tests under `source/strafer_autonomy/tests/` — the `strafer_autonomy` conftest installs a namespace stub so those tests can import `strafer_lab.tools.*` without loading the Isaac Sim-dependent `strafer_lab/__init__.py`.
- **`strafer_vlm`** — future fine-tune briefs load the harness's LeRobot v3 datasets directly (frames + detections columns + per-episode labels). Any description/paraphrase pass uses Qwen2.5-VL-7B loaded standalone (NOT the 3B model `strafer_vlm` serves), so the fine-tune target is never fed its own outputs.
- **`strafer_ros`** — the sim-in-the-loop bridge publishes on the same topic names the real robot's Jetson stack publishes (`/d555/color/image_raw`, `/d555/depth/image_rect_raw`, `/strafer/odom`, TF), so the Jetson autonomy stack consumes sim sensors without code changes.

## What ships today

- **Composed Gym environments** over three orthogonal axes — sensor stack × scene source × realism — registered as a small set of named variants (RL training: depth / no-cam, realistic / robust; capture: teleop / bridge / coverage). Capture variants take an operator-selectable sensor stack via `capture.py --sensors`. See Contracts below.
- **Sim-to-real contract presets** (`tasks/navigation/sim_real_cfg.py`) — `IDEAL_SIM_CONTRACT`, `REAL_ROBOT_CONTRACT`, `ROBUST_TRAINING_CONTRACT` covering motor time constant, command delay, slew rate, IMU / encoder / depth noise, the temporal texture of the depth and command streams (holds and per-env observation age), and the drift of the SLAM frame the goal-shaped observations are read through.
- **Mecanum motor model** (`assets/strafer.py`) — `DCMotorCfg` with torque-speed curve, paired with the three-layer action pipeline (command delay → slew rate limit → first-order motor dynamics filter) in `tasks/navigation/mdp/actions.py`.
- **Synthetic-data pipeline** — Infinigen scene generation orchestrator, scene metadata extractor, and the harness capture drivers writing the canonical LeRobot v3 training corpus (the legacy description / SFT-prep / CLIP-CSV scripts are retired under `scripts/retired/`).
- **Isaac Sim ROS 2 bridge** (`bridge/`) — config + OmniGraph builder that wires simulated sensors onto the real robot's ROS topic names. Runs inside Kit via `isaacsim.ros2.bridge`.
- **Sim-in-the-loop harness** (`sim_in_the_loop/`) — pure-Python orchestrator with injectable env / mission-API / recorder adapters. Submits Jetson missions and records one LeRobot v3 episode per mission (RGB video, depth sidecars, normalized `/cmd_vel` actions, Replicator detections columns, outcome + hard-negative metadata) via `BridgeLeRobotRecorder` → `StraferLeRobotWriter`.
- **Graceful Isaac-Sim fallback** (`__init__.py`) — subpackages that need Kit (`tasks`, `assets`) fail quietly with `ModuleNotFoundError`, so `strafer_lab.tools.*` stays importable from plain Python envs without `AppLauncher`.

## Contracts

### Gym environment registry (`strafer_lab.tasks.navigation`)

Registration happens in `tasks/navigation/__init__.py` via `gym.register`. Each
variant is composed over three orthogonal axes — sensor stack × scene source ×
realism — in `tasks/navigation/composed_env_cfg.py`. RL variants have a fixed
sensor stack (the obs contract a trained policy was fit against); capture
variants take an operator-selectable stack via `capture.py --sensors`.

| Family | Gym ID | sensors | scene | realism | Obs dims |
|---|---|---|---|---|---|
| RL | `Isaac-Strafer-Nav-RLDepth-Real-v0` | depth policy | ProcRoom | Realistic | 4,819 |
| RL | `Isaac-Strafer-Nav-RLDepth-Robust-v0` | depth policy | ProcRoom | Robust | 4,819 |
| RL | `Isaac-Strafer-Nav-RLNoCam-v0` | none | ProcRoom | Realistic | 19 |
| Capture | `Isaac-Strafer-Nav-Capture-Teleop-v0` | rgb_full (default) | Infinigen | Realistic | — |
| Capture | `Isaac-Strafer-Nav-Capture-Bridge-v0` | rgb_full+depth_full+depth_policy (default) | Infinigen | Realistic | — |
| Capture | `Isaac-Strafer-Nav-Capture-Coverage-v0` | rgb_full+depth_full (default) | Infinigen | Realistic | — |

Add the `-Play-v0` suffix to an RL ID to get its evaluation variant (fewer
envs): `Isaac-Strafer-Nav-RLDepth-Real-Play-v0`,
`Isaac-Strafer-Nav-RLDepth-Robust-Play-v0`, `Isaac-Strafer-Nav-RLNoCam-Play-v0`.

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

Grouped by the placement rule (see
[`conventions.md` → Script and tool placement](../../docs/tasks/context/conventions.md#script-and-tool-placement)):
runnable scripts under `scripts/`, importable modules under
`strafer_lab.tools.*`, deprecated code under a sibling `retired/`.

**Runnable entry points** (`source/strafer_lab/scripts/`, launched via `$ISAACLAB -p <path>`):

| Path | Purpose | Runtime |
|---|---|---|
| `scripts/train_strafer_navigation.py` | PPO training wrapper with video capture, resume, headless | IsaacLab |
| `scripts/play_strafer_navigation.py` | Inference rollout from a checkpoint *or* an exported `.pt` (via `--policy`) — headed in the Kit viewport or headless with MP4 capture | IsaacLab |
| `scripts/eval_cadence_emulation.py` | Closed-loop evaluation of a checkpoint under an emulated inference cadence — per-env schedules of held ticks (no inference, previous action re-issued, recurrent state frozen) and duplicate-depth ticks, scored on completion, along-track progress, and signed direction offset. Several profiles per session; `--profile measured` replays a temporal profile out of an inference-node observation dump | IsaacLab |
| `scripts/test_strafer_env.py` | Motion-pattern smoke test (forward / strafe / rotate / circle / figure8) | IsaacLab |
| `scripts/export_policy.py` | Convert an rsl_rl checkpoint to a deployable TorchScript `.pt` + (where supported) ONNX `.onnx`. Round-trips the artifact through `strafer_shared.policy_interface.load_policy()` and refuses to write if the deterministic-mean head wasn't frozen. Emits a JSON sidecar with variant, dimensions, source checkpoint, repo SHA, ONNX opset, `is_recurrent`, and `trained_period_s` | IsaacLab |
| `scripts/benchmark_policy.py` | Median / p95 / p99 inference-latency stats on an exported artifact. Accepts an ONNX execution-provider preference list so the Jetson lane can record TRT-EP latency after rsync | pure-Python |
| `scripts/capture.py` | Unified harness data-capture entry point (`--driver × --mission-source` cross-product per `harness-architecture.md`). Wires `(teleop, scene-metadata)` via `scripts/teleop_capture.py`, `(bridge, scene-metadata)` / `(bridge, queue)` via `scripts/run_sim_in_the_loop.py --mode harness`, and `(scripted, coverage)` via `scripts/coverage_capture.py`; the scripted `queue` / `captioner` cells raise `NotImplementedError` until those mission sources ship | Isaac Sim (drivers); pure-Python for argv validation |
| `scripts/teleop_capture.py` | In-process Isaac Lab teleop driver. Boots `AppLauncher`, loads `Isaac-Strafer-Nav-Capture-Teleop-v0`, reads gamepad, drives `env.step()`, and writes a LeRobot v3 dataset via `StraferLeRobotWriter` | Isaac Sim, pygame, lerobot |
| `scripts/coverage_capture.py` | In-process scripted coverage driver — the bulk-capture default. Boots `AppLauncher`, runs the trained RL subgoal-follower (`policy_interface.load_policy`) along a deterministic geometric coverage plan (`tools/coverage_plan.py`: every room visited from spread approach headings), and records a LeRobot v3 dataset with detections + the realized-mount-quat column + the embedded `scene_metadata` sidecar. The diverse same-place / different-heading views are the training signal, not goal-reaching demos | Isaac Sim, lerobot, torch |
| `scripts/run_sim_in_the_loop.py` | Launch Isaac Sim with the ROS 2 bridge in `--mode bridge` (manual Nav2 drive) or `--mode harness` (walks scene-metadata targets or a `mission_queue.yaml` through the Jetson autonomy stack and records a LeRobot v3 dataset with detections + optional `--inject-bad-grounding` hard negatives) | Isaac Sim, `isaacsim.ros2.bridge`, lerobot, Jetson on LAN |
| `scripts/bridge_harness_smoke.py` | Jetson-free Kit smoke of the bridge harness capture path: scripted-`/cmd_vel` sweep through the real harness + recorder + `StraferLeRobotWriter` (detections on), then asserts the round-tripped dataset (episodes/frames, depth sidecars, detections vocab, discard path). `make harness-smoke`. Does **not** exercise the ROS publishers or the live executor | Isaac Sim, lerobot |
| `scripts/collect_demos.py` | Gamepad teleop recorder — writes one HDF5 group per episode (`obs` / `actions` / `rewards`). No training entry point reads this format today; demo-driven training consumes the teleop driver's LeRobot datasets instead. Shares the family-aware reader at `strafer_lab.tools.gamepad_reader` with the harness teleop driver | Isaac Sim |
| `scripts/roller_bounce_probe.py` | Headless mecanum-roller contact diagnostic: spins in place across wheel speeds and logs chassis-z to quantify the high-yaw bounce. Exposes `--sim-hz` + PhysX solver knobs | IsaacLab |
| `scripts/ceiling_sidedness_probe.py` | Headless one-way-ceiling check on the enriched rooms: records the policy camera's depth and RGB from inside the enclosure plus an overhead camera's ceiling pixel share, with `--ceiling` / `--face-culling` arms for the geometry and the RTX culling switch separately. `--compare` reads two recordings back with no Kit boot. The answer is renderer-build-specific — re-run it on an RTX / Isaac Sim upgrade | IsaacLab |
| `scripts/prep_room_usds.py` | Orchestrate Infinigen scene generation (`generate`, `ingest`, `presets` subcommands) | IsaacLab + `INFINIGEN_ROOT`, `STRAFER_INFINIGEN_PYTHON` |
| `scripts/postprocess_scene_usd.py` | Bake colliders + ceiling-light emitters into an Infinigen-exported USDC (called by `prep_room_usds.py`, or run manually) | IsaacLab (`pxr`) |
| `scripts/generate_scenes_metadata.py` | Walk `Assets/generated/scenes/` and author the combined `scenes_metadata.json` with per-scene spawn points + floor top Z | IsaacLab (`pxr`) |
| `scripts/extract_scene_metadata.py` | Build per-scene metadata (rooms, polygons, semantic tags, relations) and embed it in the scene USD's root-prim `customData`; apply `UsdSemantics` detection labels (non-structural classes) + `semanticLabel` provenance attrs. `--from-usd` (prim-name parse) runs under `$ISAACLAB -p` (Kit, for the semantics schema); `--blend` builds the dict in Blender | `pxr` + Kit (`$ISAACLAB`); `--blend` path: `bpy` |
| `scripts/validate_scene_connectivity.py` | Generate the scene's occupancy grid (occupancy-map extension → cached `<scene>/occupancy.npy`), verify room-to-room reachability with the shared grid planner, force doors open, and merge the verified `connectivity[]` graph + `multi_story` flag into the USD `customData` (called by `prep_room_usds.py`, or run manually). `--rasterize-fallback` if the omap extension is unavailable | Kit (`$ISAACLAB`, `isaacsim.asset.gen.omap`) |
| `scripts/measure_path_statistics.py` | Baseline `tools/path_statistics.py` over procedural rooms and the scanned-scene corpus in one run: turn density / tortuosity / bending fraction, aperture-threading fraction with the threshold derived from the reference distribution's body, group-resampled intervals. `--arm` re-ranges a generator kwarg to compare arms | numpy + torch (no Kit) |
| `scripts/build_mission_corpus.py` | Run the offline mission generator across scenes + seeds into a `mission_queue.yaml` corpus: per-scene `data/mission_queues/<scene>/queue.yaml` + unioned `corpus.yaml`, cached + round-trip-checked. `--mode {endpoint,path-shape,mixed}`; the LLM waypoint / paraphrase / VLM start-frame-grounding passes are opt-in (`--use-planner-llm` / `--use-paraphrase-llm` / `--ground-start-frame`), model-free otherwise | `pxr` + numpy (no Kit) |

**One-shot asset authoring** (`source/strafer_lab/scripts/asset_authoring/`, run by hand to (re)build or inspect the robot/asset USD):

| Path | Purpose |
|---|---|
| `scripts/asset_authoring/setup_physics.py` | Author PhysX collision + articulation roots + masses onto the robot USD |
| `scripts/asset_authoring/collapse_redundant_xforms.py` | Collapse redundant single-child Xform/Scope prims while preserving world transforms |
| `scripts/asset_authoring/inspect_robot_prim_layout.py` | Dump a tree view of the robot prim hierarchy + basic physics flags |
| `scripts/asset_authoring/run_empty_lab.py` | Launch an empty Isaac Lab world (asset staging / smoke) |

**Runtime helpers** (`strafer_lab.tools.*`, plain Python, no Isaac Sim):

| Module | Exports |
|---|---|
| `scene_metadata_reader` | `load(scene_usd)` — the single `pxr` touch-point that reads a scene's embedded `customData` metadata (hard-fails when absent); `write_custom_data`, `metadata_hash` |
| `scene_labels` | `iter_rooms`, `iter_objects`, `get_scene_label_set_from_data`, `get_room_at_position`, `get_objects_in_room` (pure-data) + `get_scene_metadata`/`get_scene_label_set` (read a scene USD) — typed accessors over a scene's embedded metadata |
| `scene_connectivity` | `occupancy_to_free_space` (the shared cached-occupancy → planner adapter: invert + robot-radius disc inflation), `compute_connectivity` (verified room graph via `plan_path`), `load_occupancy`/`save_occupancy`, `is_multi_story`, `is_multi_room_incompatible` — numpy-only, consumed by `validate_scene_connectivity.py` and the other Infinigen path-planning consumers |
| `path_statistics` | `turn_density`, `waypoint_clearance`, `arc_fraction_below`, `path_statistics`, `sample_endpoint_pairs`, `plan_over_pairs`, `summarize`, `bootstrap_gates`, `threshold_from_body` — corridor-curvature and aperture-threading statistics over planned paths, with the resolution / inflation-radius / arc-length corrections that make two occupancy sources comparable. Numpy-only, consumes `plan_path` unmodified. Run via `scripts/measure_path_statistics.py` |
| `spatial_description` | `SpatialDescriptionBuilder`, `quat_to_yaw`, `classify_region`, `classify_bearing` — Stage-1 factual spatial relations |
| `bbox_extractor` | `ReplicatorBboxExtractor`, `parse_bbox_data`, `DetectedBbox` — wraps Replicator's `bounding_box_2d_tight` annotator |
| `mission_queue` | `load_mission_queue`, `QueueMissionRow`, `queue_row_to_mission_spec` — `mission_queue.yaml` parser for the `queue` mission source |
| `build_mission_queue` | `build_mission_queue`, `load_scene_inputs`, `GeneratorConfig`, `SceneInputs` — the canonical `mission_queue.yaml` producer: connectivity-gated free-text missions with oracle waypoints (via the shared `plan_path`), pluggable LLM/VLM passes, cache keys. Run via `scripts/build_mission_corpus.py` |
| `grounding_injection` | `plan_injection`, `InjectionPlan`, `resolve_target_room_idx` — hard-negative goal perturbation for `--inject-bad-grounding` |
| `infinigen_label_parser` | Infinigen-specific semantic label normalization |

**Deprecated** (`scripts/retired/`, `tools/retired/`) — retained for reference / mining, **not imported or invoked by any live entry point**; the harness brief deletes each as it supersedes it:

| Path | Superseded by |
|---|---|
| `scripts/retired/generate_descriptions.py` | captioner mission source in `harness-architecture.md` |
| `scripts/retired/prepare_vlm_finetune_data.py` | direct LeRobot v3 loading |
| `scripts/retired/finetune_clip.py` | the `cotrained-retrieval-augmented` multi-task recipe |
| `tools/retired/dataset_export.py` | direct LeRobot v3 loading |

**Isaac Sim subpackages** (require Kit runtime):

| Subpackage | Purpose |
|---|---|
| `strafer_lab.assets` | `ArticulationCfg` for the Strafer robot + D555 camera configs |
| `strafer_lab.tasks.navigation` | Environment configs, MDP components (actions, observations, rewards, terminations, events, curriculums, commands), agent configs |
| `strafer_lab.bridge` | ROS 2 bridge config + OmniGraph builder |
| `strafer_lab.sim_in_the_loop` | Harness + runtime adapters + mission generator |

## Install

### Linux (DGX Spark, aarch64 — preferred)

`strafer_lab` runs in the `env_isaaclab3beta2` conda env (Isaac Sim 6.0.1.0 +
Isaac Lab `v3.0.0-beta2.patch1` on Python 3.12), paired with a clone of Isaac
Lab at that tag. This subsection is the canonical recreate recipe for that pair
— [`repo-topology.md`](../../docs/tasks/context/repo-topology.md#python-environments-dgx)
links here.

An earlier pair (`env_isaaclab3` + an Isaac Lab `develop` clone, Isaac Sim
6.0.0.0, torch 2.10) still exists on the DGX and is what `.env` and the
Makefile defaults currently select. It is kept as a rollback artifact, is never
written to, and is not rebuilt — this recipe does not describe it.

**Prereqs** (audited on `gx10-d1d8`):

| Item | Value |
|---|---|
| OS | Ubuntu 24.04 LTS (aarch64) |
| GPU / driver | NVIDIA GB10 (Blackwell) / 580.82.09 (NVIDIA recommends ≥ 580.95.05 for Spark) |
| CUDA | 13.0 (`/usr/local/cuda-13.0`) |
| Python | 3.12 (Isaac Sim 6 requires `==3.12.*`) |

Three packages build from source on aarch64 — `nlopt` (needs `cmake` +
`swig`), `quadprog` (needs the Python dev headers), and `imgui-bundle` (needs
GL / X11). `isaaclab.sh --install` installs most of these itself via `apt`,
which needs sudo; installing them first keeps the build non-interactive.
`imgui-bundle` now ships a prebuilt aarch64 wheel, so the old "compiles from
source, ~2 min" note no longer applies — it is listed here only because the
installer's own apt list still covers it.

```bash
sudo apt-get install -y cmake swig build-essential python3.12-dev \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
  libxkbcommon-dev libgl1-mesa-dev libopengl-dev libglx-dev
```

**Build the pair from scratch.** Every step below is transcribed from the pip
logs of the build that produced the env in use; the ordering is not
interchangeable, and the reasons are given where a step looks redundant.

The sequence has been run end to end into a throwaway env and the result
compared against the working env's `pip freeze`: **288 of 300 distributions
match exactly**, including every pinned package, all 25 `isaacsim` wheels and
all 15 `isaaclab_*` editables. The 12 that differ are two editable lines
carrying the Sim2RealLab working-tree revision, and ten unpinned transitives
that had moved on since the original build — `usd-exchange` 2.3.0 → 3.0.0,
`viser` 1.0.30 → 1.1.0, `rerun-sdk` 0.36.0 → 0.36.3, `protobuf` 7.35.1 →
7.36.0, plus GitPython, Pygments, python-dotenv, xxhash, jupyterlab_widgets and
widgetsnbextension at patch level. Only ~15 of the 300 are version-controlled
by the commands below; the rest are whatever the resolver returns on the day.
A rebuild that must match exactly needs a constraints file, which does not
exist yet — `usd-exchange` crossing a major version is the one to check first.

```bash
# 1. The Isaac Lab clone, at the tag. Detached HEAD is expected.
git clone git@github.com:isaac-sim/IsaacLab.git ~/Documents/repos/IsaacLab-3beta2
cd ~/Documents/repos/IsaacLab-3beta2
git checkout v3.0.0-beta2.patch1

# 2. The env.
conda create -n env_isaaclab3beta2 python=3.12 -y
conda activate env_isaaclab3beta2

# 3. CUDA torch, first, so the Isaac Sim install resolves against it.
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# 4. Isaac Sim, pinned exactly. Install it here rather than letting
#    `isaaclab.sh --install isaacsim` do it: that path asks for
#    `isaacsim[all]>=6.0.0`, which pins neither the version nor the
#    `extscache` extra, and `extscache` decides the Kit extension payload.
pip install "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com

# 5. A defensive re-pin. The recorded build's Isaac Sim step left torch alone
#    (its log reports the pins already satisfied and uninstalls nothing), but
#    isaacsim[all] is free to move the stack and step 6 depends on it being
#    right. --no-deps, or pip re-resolves torch's whole closure off cu130.
pip install --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# 6. Isaac Lab's own editables and extras. `--install rsl_rl` no longer
#    exists; an unknown token only WARNS and is skipped, so a stale command
#    appears to succeed and silently installs nothing. This token set is the
#    13 core submodules + the teleop/mimic submodules + the newton and
#    visualizer extras — i.e. `all` without the `rl` extra (`contrib` and `ov`
#    are excluded from `all` upstream and need naming to get them). The `rl`
#    extra is skipped deliberately: it hard-pins
#    rsl-rl-lib==5.0.1, and this stack needs 5.4.2 (step 7).
./isaaclab.sh --install mimic,newton,visualizer

# 7. rsl-rl at the version this stack needs, installed directly.
pip install rsl-rl-lib==5.4.2

# 8. onnxscript floor. rsl-rl only floors it at >=0.5.4, and 0.6.2 is an
#    export hazard on aarch64, so assert the floor that matters.
pip install "onnxscript>=0.7.1"

# 9. The strafer packages. strafer_lab needs --no-build-isolation: a
#    transitive dep (flatdict) uses legacy pkg_resources.
cd ~/Workspace/Sim2RealLab
source env_setup.sh          # exports LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1,
                             # which Isaac Lab's aarch64 build requires at process
                             # start; it cannot be set from inside Python. Note that
                             # .env still names the earlier pair, so this exports the
                             # earlier $ISAACLAB — it is sourced here for LD_PRELOAD.
pip install -e source/strafer_shared
pip install --no-build-isolation -e source/strafer_lab

# 10. lerobot for the harness capture writer, --no-deps so its strict
#     torch / rerun-sdk bounds cannot drag the pinned stack backwards, then
#     only the runtime deps the writer actually uses.
pip install --no-deps lerobot==0.5.1
pip install "datasets>=4.0.0,<5.0.0" "av>=15.0.0,<16.0.0" "jsonlines>=4.0.0,<5.0.0"

# 11. THE TORCH PIN GOES LAST. `isaaclab.sh --install` downgraded torch to
#     2.10.0 and uninstalled torchaudio without reinstalling it (see the
#     hazard note below). --no-deps here, which is why step 12 is needed.
pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# 12. The torch 2.10 leg pulled cuDNN back to 9.15.1.9 and re-pinning torch
#     with --no-deps does not lift it. Restore the build torch 2.11 wants.
pip install --index-url https://download.pytorch.org/whl/cu130 \
  nvidia-cudnn-cu13==9.19.0.56

# 13. Remaining pins.
pip install onnxruntime==1.25.1   # without it 15 tests skip silently,
                                  # including the ONNX export gate
pip install packaging==26.0       # isaaclab_rl's editable pins it down to 23.2
pip install torchcodec==0.16.0    # closes the lerobot video-decode break;
                                  # ABI-coupled to torch, declares no torch
                                  # dependency — re-check on any torch move
pip install py-spy==0.4.2         # the sampling profiler the Kit boot-hang
                                  # forensics used; part of the validated set
```

**Pinning the rest.** The commands above name about fifteen packages. The other
~267 are whatever the resolver returns on the day, and they move:
[`constraints-env_isaaclab3beta2.txt`](constraints-env_isaaclab3beta2.txt) holds
the 282 resolved versions from the environment Stage 3 validated, so a rebuild
can reproduce that set rather than a contemporary one. Two things make it not a
drop-in `-c`, both checked by dry-run:

```bash
export PIP_CONSTRAINT=$PWD/source/strafer_lab/constraints-env_isaaclab3beta2.txt

# Every torch step needs PyPI alongside the cu130 index. Constrained, torch's
# own dependencies must resolve to the pinned versions, and the cu130 index does
# not carry them: with cu130 alone pip reports
#   torch 2.11.0+cu130 depends on jinja2 / the user requested (constraint) jinja2==3.1.5
# and fails. Add --extra-index-url to steps 3, 5 and 11:
#   --index-url https://download.pytorch.org/whl/cu130 \
#   --extra-index-url https://pypi.org/simple
# Constrained, those steps also land nvidia-cudnn-cu13 9.19.0.56 directly, so
# step 12's repair becomes a no-op rather than a fix.

# Step 6 must run OUTSIDE the constraint:
env -u PIP_CONSTRAINT ./isaaclab.sh --install mimic,newton,visualizer
# _ensure_cuda_torch uninstalls torch and installs torch==2.10.0, which the
# constraint forbids — pip reports "The user requested torch==2.10.0 / the user
# requested (constraint) torch==2.11.0+cu130" and the install dies part-way.
# Because that step ran unconstrained, follow it with a convergence pass before
# continuing, which re-pins whatever it moved:
pip install --extra-index-url https://pypi.org/simple \
    -r <(grep -E '^[A-Za-z0-9._-]+==' source/strafer_lab/constraints-env_isaaclab3beta2.txt)
```

Honestly labelled: the file parses, and the three resolutions above were
verified by dry-run. **A full rebuild under these constraints has not been run
end to end.** The throwaway environment from the rebuild is the place to prove
it.

**`isaaclab.sh --install` is not idempotent, and re-running it breaks the env.**
`_ensure_cuda_torch` compares the full version string against `2.10.0+cu130`;
against anything else it uninstalls `torch`, `torchvision` *and* `torchaudio`
and reinstalls only the first two. It runs twice per invocation, so the second
pass reports `PyTorch 2.10.0+cu130 already installed.` and the log looks clean.
Any later `--install` — even one just to add an extra — therefore silently
downgrades torch and leaves torchaudio missing. Re-run steps 11–12 after every
invocation, or do not invoke it again.

**EULA.** The marker is
`$CONDA_PREFIX/lib/python3.12/site-packages/isaacsim/kit/EULA_ACCEPTED` (three
bytes, `yes`), written by an interactive first boot. Accept it that way. Do not
set `OMNI_KIT_ACCEPT_EULA`.

**Kit-app modifications on the clone.** Two local edits existed on the earlier
clone. One is obsolete; the other is a step to apply to a new clone, which is
pristine until it is.

- `omni.kit.pip_archive` shim — **obsolete**. A pristine clone boots. Do not
  reapply it.
- `omni.kit.telemetry` removal — **kept, as a privacy decision, not a fix.** A
  pristine boot leaves an `omni.telemetry.transmitter` process resident with
  `enableAnonymousData=true`, reporting to an external endpoint. Removing the
  extension was measured inert for the intermittent boot deadlock (10/40 hangs
  with it removed vs 9/40 with it present). It is not inert for everything: on
  a pristine clone 2 of 16 boots instead died with SIGSEGV, in a stack naming
  `libomni.kit.telemetry.plugin.so`, and the same arm with the extension
  removed crashed 0 of 16. Those counts are too small to settle it on their own,
  so treat the crash as a second reason to keep the removal rather than as an
  established one. The edit is one line dropped from each of the two
  Kit app files in the clone, and it is machine-local — the clone is not a
  tracked part of this repository:

  ```bash
  # in the Isaac Lab clone
  sed -i '/^"omni\.kit\.telemetry" = {}$/d' \
    apps/isaaclab.python.kit apps/isaaclab.python.headless.kit
  ```

**Verify the env** — `pip check` is not a usable gate here: it reports 19
findings on this pair and 25 on the earlier one, most of them the ordinary
consequence of the `--no-deps` and pinned-version steps above rather than
anything wrong. Gate on these instead:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.11.0+cu130 True
python -c "import torch, lerobot; print(lerobot.__version__)"   # 0.5.1

# No duplicate dist-info — two versions of one package is how the earlier env
# ended up reporting the wrong isaaclab version.
ls $CONDA_PREFIX/lib/python3.12/site-packages | grep dist-info \
  | sed 's/-[0-9].*//' | sort | uniq -d          # expect no output

# Every editable resolves to the tree it should
python -c "import isaaclab, strafer_lab; print(isaaclab.__file__); print(strafer_lab.__file__)"
```

`packaging` is a genuine three-way conflict with no solution: `isaacsim-core`
pins `==26.0`, `isaaclab_rl` pins `<24`, `lerobot` wants `>=24.2,<26.0`. 26.0 is
the deliberate choice; pip will keep printing the other two as conflicts. Do not
"fix" it.

**DGX Spark limitations** (Isaac Sim aarch64 build): no SkillGen, no
OpenXR, no JAX-GPU, no Livestream — none affect `strafer_lab`. `nvidia-smi`
reports VRAM as `[N/A]` because the GB10 shares unified memory with the
CPU; watch `free -h` for memory pressure at high `--num_envs` instead.

For the Infinigen-only conda env (`env_infinigen`, aarch64 source-built
`bpy==4.2.0`), see the `README.md` in the sibling `~/Workspace/blender-build/`
directory — those artifacts are machine-specific and live outside this repo.


### Windows workstation

```powershell
# From the workspace root, with the Isaac Lab venv active
& C:\Workspace\venv_isaac\Scripts\Activate.ps1
cd C:\Workspace
python -m pip install -e source/strafer_lab source/strafer_shared
```

Three environments partition the DGX stack — the Isaac Sim / Isaac Lab conda
env (this package), `.venv_vlm` (the VLM + planner services), and
`env_infinigen` (scene-gen, Python 3.11). `isaacsim-core` pins torch to an exact
version, so the Isaac Lab env's torch is not free to move at all; `.venv_vlm`
tracks its own. On the candidate pair the two happen to share a torch minor and
differ only in the CUDA build; the currently selected pair is a minor behind.
The full table, the why-separate rationale, and each recreate recipe live in
[`repo-topology.md` → Python environments (DGX)](../../docs/tasks/context/repo-topology.md#python-environments-dgx).

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
../IsaacLab/isaaclab.sh -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLNoCam-v0 --num_envs 512 --headless

# Depth Realistic (4,819 obs dims; cameras auto-enabled)
../IsaacLab/isaaclab.sh -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-v0 --num_envs 32 --headless

# Full Realistic
../IsaacLab/isaaclab.sh -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-v0 --num_envs 32 --headless

# With video capture (overhead MP4 every 500 env steps)
../IsaacLab/isaaclab.sh -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-v0 \
    --num_envs 64 --max_iterations 3000 \
    --video --video_length 200 --video_interval 500 --headless

# Resume from checkpoint
../IsaacLab/isaaclab.sh -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLNoCam-v0 \
    --resume logs/rsl_rl/strafer_navigation/run_XXXXX/model_500.pt
```

Windows equivalents use `.\IsaacLab\isaaclab.bat` and the `C:\Workspace` paths. Checkpoints land in `logs/rsl_rl/strafer_navigation/<timestamp>/`. TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/strafer_navigation
```

### Evaluate a policy

```bash
../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Strafer-Nav-RLNoCam-Play-v0 --num_envs 50
```

Fast-path gym evaluation (no ROS overhead):

```python
from strafer_shared.policy_interface import load_policy, PolicyVariant
import gymnasium as gym, strafer_lab

env = gym.make("Isaac-Strafer-Nav-RLNoCam-v0", num_envs=1)
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
# 1. Generate Infinigen scenes — embeds metadata + UsdSemantics labels into
#    each USD (so the scene is capture-ready); needs $ISAACLAB set
python scripts/prep_room_usds.py generate --quality high \
    --num-scenes 1 --output Assets/generated/scenes

# 2. (only for a pre-existing/bare USD) re-author the embedded metadata
$ISAACLAB -p scripts/extract_scene_metadata.py \
    --from-usd --usd Assets/generated/scenes/<scene>.usdc

# 3. Collect teleop perception data via the harness entry point (requires Isaac Sim)
isaaclab -p source/strafer_lab/scripts/capture.py \
    --driver teleop --mission-source scene-metadata \
    --scene scene_001 \
    --output data/sim_in_the_loop/scene_001 \
    --fps 8

# 4. Capture through the autonomy stack as a side effect of
#    integration testing (requires the Jetson stack on the LAN)
isaaclab -p source/strafer_lab/scripts/capture.py \
    --driver bridge --mission-source scene-metadata \
    --scene scene_001 \
    --output data/sim_in_the_loop/scene_001 \
    --headless --enable_cameras

# 5. THE BULK-CAPTURE DEFAULT — diverse-perspective coverage sweep driven by
#    the trained RL subgoal-follower (every room visited from spread headings).
#    Teleop (step 3, annotator) and bridge (step 4, Jetson-in-loop) are the
#    non-bulk paths. Checkpoint is an exported artifact from export_policy.py.
isaaclab -p source/strafer_lab/scripts/capture.py \
    --driver scripted --mission-source coverage \
    --scene scene_001 \
    --output data/sim_in_the_loop/scene_001 \
    --policy-variant nocam_subgoal \
    --checkpoint models/strafer_nocam_subgoal.pt \
    --headless --enable_cameras
```

Downstream fine-tunes load the captured LeRobot v3 dataset directly
(HF `LeRobotDataset` + the `lerobot_depth` / `lerobot_detections` /
`read_strafer_episodes` helpers). The legacy description → CSV → SFT
prep chain lives under `scripts/retired/` for reference only.

### Sim-in-the-loop bridge / harness

```bash
source env_setup.sh

# Manual / Nav2 mode: DGX publishes simulated sensors, Jetson drives via Nav2
isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py

# LeRobot v3 dataset capture (harness submits missions to the Jetson;
# preferred front door is capture.py --driver bridge, which dispatches here)
isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
    --mode harness --headless --enable_cameras \
    --scene-name kitchen_01 \
    --output data/sim_in_the_loop/kitchen_01
```

Jetson side (either mode) launches `bringup_sim_in_the_loop.launch.py` from [`strafer_ros`](../strafer_ros/README.md).

## Design

**Simulation is the source of truth for physics, noise, and observation layout.** Every constant the real robot uses (wheel radius, PID, encoder PPR, velocity limits) lives in `strafer_shared.constants` and is imported by both sides. The sim-to-real "envelope" is the `Realistic` / `Robust` preset — if the real hardware falls within that envelope, the policy transfers.

**Four-layer actuator pipeline models real-world imperfections.** Command delay (USB serial latency) → command hold (a control step that publishes nothing, so the chassis re-executes its last command) → slew rate limiting (motor + gearbox inertia) → first-order motor dynamics filter (combined electrical + mechanical τ). The filter's time constant is the single most important sim-to-real parameter; target real τ in `[20, 100]` ms to stay inside the Robust envelope. Delay and hold are different failure modes: a delay shifts a command in time, a hold repeats it.

**The policy contract is hermetic.** `strafer_shared.policy_interface.assemble_observation()` and `interpret_action()` are the only places observations and actions cross the boundary between Python / NumPy and the policy's `[-1, 1]` tensor space. The gym environment and the ROS inference node both go through them, never around them.

**Fine-tune data stays separate from fine-tune target.** The 4-stage description pipeline uses Qwen2.5-VL-7B loaded standalone via `transformers.AutoModelForVision2Seq` — it does NOT call the 3B `strafer_vlm` service. Feeding the fine-tune target's own outputs back as training data causes collapse.

**ProcRoom scenes are RL-only, never perception training.** Primitive-shape rooms (solid-color boxes, cylinders) do not transfer to real indoor environments for VLM / CLIP training. Every perception export excludes `scene_type == "procroom"`.

**`strafer_lab.tools.*` stays importable without Isaac Sim.** `__init__.py` swallows `ModuleNotFoundError` from `omni` / `pxr` when the Kit runtime is absent. This is why the synthetic-data pipeline runs from plain Python envs and why some tool tests live under `source/strafer_autonomy/tests/` (using a conftest stub).

**Isaac Sim ROS 2 bridge publishes on the real robot's topic names.** Sim-in-the-loop runs the Jetson autonomy stack against simulated sensors without any code changes in `strafer_ros` or `strafer_autonomy`. That only works because the bridge preserves topic names, frame IDs, and sensor-msg shapes.

**`AppLauncher` must boot before `import strafer_lab.tasks`.** Importing the task package transitively pulls `isaaclab.managers`, which imports `omni.timeline`, which only exists after Kit boots. Scripts enforce this ordering; new scripts must follow the same rule.

## Testing

Both test trees run in the Isaac Lab conda env; the split is whether they boot Kit:

- **`test_sim/`** — env / sensor / reward / observation suites that need Isaac Sim. The root `test_sim/conftest.py` boots Kit headlessly once per session; use `run_tests.py` for clean output (direct pytest is drowned by Kit startup noise, and the root conftest calls `os._exit()` before pytest can print its summary).
- **`tests/`** — suites that run without booting Kit, foldered by intent:
  - **`tests/harness/`** — capture / writer / mission-picker / profiler tooling (lerobot + USD/`pxr`).
  - **`tests/policy_tooling/`** — `test_export_policy.py`, `test_load_policy.py`: the `.pt`/`.onnx` export and `strafer_shared.policy_interface` loader round-trips.
  - **`tests/contracts/`** — `test_action_clamp.py`, `test_obs_contract_parity.py`, `test_recurrent_contract_e2e.py`: the sim↔real boundary guards (action clamp, encoder-FK observation parity, recurrent hidden-state contract).

`tests/` imports `lerobot` / `pxr` / `warp` / `isaaclab_tasks` / `onnx`, all of which the Isaac Lab env already carries — so the whole pure-Python tree runs there (no separate `.venv_harness`).

```bash
# Everything strafer_lab (Kit suites + pure-Python), from the repo root
make test-lab

# Fast iteration — pure-Python tests/ only, no Kit boot (~seconds)
make test-lab-pure

# Kit suites directly, or a subset for fast iteration
$ISAACLAB -p source/strafer_lab/run_tests.py all
$ISAACLAB -p source/strafer_lab/run_tests.py noise_models depth_noise imu sensors
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

- **Electronics masses in the USD** — RoboClaws, Jetson, buck converter, D555 camera meshes + masses are TODO in `source/strafer_lab/scripts/asset_authoring/setup_physics.py`. Without them, the simulated chassis inertia underestimates the real robot.
- **`strafer_inference` Jetson package** — the deployment target for trained policies does not exist yet; see [`strafer_ros`](../strafer_ros/README.md) for the interface side.
- **Goal position noise during final pre-deployment training** — for VLM-sourced goals, retrain with `goal_position_noise_std: 0.2-0.3 m` in `commands.py` to match Qwen2.5-VL-3B localization error; without this, the policy oscillates at deployment.

## References

- [`source/strafer_shared/`](../strafer_shared/) — authoritative physical constants, mecanum kinematics, policy I/O contract.
- [`source/strafer_ros/README.md`](../strafer_ros/README.md) — real-robot counterpart; consumes the same policy interface.
- [`source/strafer_autonomy/README.md`](../strafer_autonomy/README.md) — Jetson executor that sim-in-the-loop drives.
- [`source/strafer_vlm/README.md`](../strafer_vlm/README.md) — grounding service + LoRA fine-tuning tooling consuming datasets from this pipeline.
- [`docs/SIM_TO_REAL_TUNING_GUIDE.md`](../../docs/SIM_TO_REAL_TUNING_GUIDE.md) — deep-dive actuator / sensor alignment procedure pairing sim presets with hardware measurements.
- [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../docs/INTEGRATION_SIM_IN_THE_LOOP.md) — cross-host bridge runbook (Stage 2 covers `make sim-bridge` smoke-test gating this package).
- [`docs/tasks/DEFERRED_WORK.md`](../../docs/tasks/DEFERRED_WORK.md) — open items.
