# Sim2RealLab system flow diagrams

Runtime reference for the six end-to-end information-flow paths that
make up the Sim2RealLab pipeline. Each diagram shows the operator
command that triggers the flow, every process / file / topic the data
passes through, cross-host boundaries, and the artifact the flow
produces. Nodes link into the source so GitHub renders them clickable.

This document describes **what the system does today**, not the design
rationale (see [`STRAFER_AUTONOMY_NEXT.md`](STRAFER_AUTONOMY_NEXT.md))
and not the test plan (a future `INTEGRATION_*.md` on the cycle).

## Index

- [Diagram legend](#diagram-legend)
- [Flow 1 — VLM data gathering and training](#flow-1--vlm-data-gathering-and-training)
- [Flow 2 — CLIP data gathering and training](#flow-2--clip-data-gathering-and-training)
- [Flow 3 — Strafer RL training (Isaac Lab)](#flow-3--strafer-rl-training-isaac-lab)
- [Flow 4 — Perception data gathering (teleop)](#flow-4--perception-data-gathering-teleop)
- [Flow 5 — Perception data gathering (sim-in-the-loop)](#flow-5--perception-data-gathering-sim-in-the-loop)
- [Flow 6 — End-to-end real-robot execution](#flow-6--end-to-end-real-robot-execution)
- [Cross-flow data dependency map](#cross-flow-data-dependency-map)

## Diagram legend

Consistent conventions across all six flows:

| Shape | Mermaid syntax | Meaning |
|---|---|---|
| Rectangle | `[Text]` | Process, service, script, node |
| Stadium | `([Text])` | ROS action / service call |
| Hexagon | <code>{{Text}}</code> | ROS topic |
| Rounded rect | `(Text)` | Data file / directory / JSON artifact |
| Cylinder | `[(Text)]` | Data store (ChromaDB, checkpoints directory) |
| Circle | `((Text))` | Human operator / external actor |
| Dashed border | `---` in style or `-.->` arrow | Optional branch / deferred path |

Host subgraphs are labelled `DGX Spark`, `Jetson Orin Nano`, `Real robot
hardware`. Arrows are labelled with transport (`ROS`, `HTTP`, `file`,
`subprocess`) when the medium matters.

Every node with a clear source-code home has a `click` link using a
relative path from this document (`../source/...`). GitHub renders those
as clickable when the Mermaid is viewed online.

---

## Flow 1 — VLM data gathering and training

**When**: batch pipeline, run manually after a teleop session has
produced enough episode frames (typically hundreds to thousands of
frames across several Infinigen scenes). Re-runs on demand when the
dataset grows or the description prompt changes.

**Triggered by**: operator on the DGX, inside `.venv_vlm` for the
batch-processing steps and `env_infinigen` (only if the 7B description
model is run in a separate env; default is `.venv_vlm`). Not automated
by CI.

**Produces**: a fine-tuned Qwen2.5-VL-3B-Instruct LoRA adapter in
`outputs/qwen25vl_lora_run*/`. **Consumes**: perception episodes from
Flow 4 or Flow 5, plus scene metadata produced by
`extract_scene_metadata.py`.

```mermaid
flowchart TB
    Operator((Operator<br/>DGX Spark))

    subgraph DGX["DGX Spark"]
        direction TB

        PerceptionDir[("data/perception/<br/>episode_NNNN/")]
        SceneMeta[("Assets/generated/scenes/<br/>scene_metadata.json")]

        GenDesc["generate_descriptions.py<br/>4-stage pipeline"]
        Qwen7B["Qwen2.5-VL-7B<br/>standalone (transformers)"]
        Spatial["SpatialDescriptionBuilder<br/>Stage 1 (pure Python)"]
        DescOut[("data/descriptions/<br/>descriptions.jsonl<br/>spotcheck.jsonl<br/>batch_stats.json")]

        PrepVLM["prepare_vlm_finetune_data.py<br/>single-obj + negatives +<br/>multi-obj + description-preserve"]
        SFTData[("data/vlm_finetune/<br/>train.jsonl + val.jsonl")]

        DatasetIO["dataset_io.py<br/>flat / chat JSONL loader"]
        TrainLoRA["train_qwen25vl_lora.py<br/>LoRA SFT"]
        EvalLoRA["eval_qwen25vl_grounding.py<br/>offline eval with IoU"]

        Adapter[("outputs/qwen25vl_lora_run*/<br/>adapter_config.json<br/>adapter_model.safetensors")]
    end

    Operator -->|bash $ python scripts/generate_descriptions.py<br/>python scripts/prepare_vlm_finetune_data.py<br/>python -m strafer_vlm.training.train_qwen25vl_lora| GenDesc
    PerceptionDir --> GenDesc
    SceneMeta --> GenDesc
    GenDesc --> Spatial
    Spatial --> Qwen7B
    Qwen7B --> DescOut

    PerceptionDir --> PrepVLM
    SceneMeta --> PrepVLM
    DescOut --> PrepVLM
    PrepVLM --> SFTData

    SFTData --> DatasetIO
    DatasetIO --> TrainLoRA
    TrainLoRA --> Adapter
    Adapter --> EvalLoRA
    SFTData -. val split .-> EvalLoRA

    click GenDesc "../source/strafer_lab/scripts/generate_descriptions.py" "4-stage description pipeline"
    click Spatial "../source/strafer_lab/strafer_lab/tools/spatial_description.py" "Stage-1 factual relations"
    click PrepVLM "../source/strafer_lab/scripts/prepare_vlm_finetune_data.py" "Comprehensive VLM SFT prep"
    click DatasetIO "../source/strafer_vlm/strafer_vlm/training/dataset_io.py" "Grounding dataset loader"
    click TrainLoRA "../source/strafer_vlm/strafer_vlm/training/train_qwen25vl_lora.py" "LoRA fine-tune"
    click EvalLoRA "../source/strafer_vlm/strafer_vlm/training/eval_qwen25vl_grounding.py" "Offline eval"
```

**Notes.** The 7B model loaded by `generate_descriptions.py` is
intentionally distinct from the 3B model that
[`strafer_vlm`](../source/strafer_vlm/README.md) serves on port 8100;
feeding the fine-tune target's own outputs back as training data causes
collapse. ProcRoom frames are excluded by `prepare_vlm_finetune_data.py`
(they do not transfer to real rooms). The adapter produced here is
loaded into the running VLM service by setting `GROUNDING_MODEL` to the
adapter-merged model path or wiring PEFT loading.

---

## Flow 2 — CLIP data gathering and training

**When**: batch pipeline, typically run once the description pipeline
has produced enough `(image, description)` pairs. Re-runs when the
semantic-map CLIP encoder needs to be re-trained (e.g., after a scene
distribution change).

**Triggered by**: operator on the DGX, inside `.venv_vlm`.

**Produces**: two ONNX files — `clip_visual.onnx` and `clip_text.onnx` —
loaded by the Jetson-side `CLIPEncoder` in
[`strafer_autonomy.semantic_map`](../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py)
at executor startup. Same upstream perception data as Flow 1, different
export format.

```mermaid
flowchart TB
    Operator((Operator<br/>DGX Spark))

    subgraph DGX["DGX Spark"]
        direction TB

        PerceptionDir[("data/perception/<br/>episode_NNNN/")]
        DescOut[("data/descriptions/<br/>descriptions.jsonl")]

        DatasetExport["dataset_export.py<br/>run_export()"]
        ClipCSV[("data/clip_descriptions/<br/>clip_descriptions.csv")]

        FinetuneClip["finetune_clip.py<br/>OpenCLIP ViT-B/32<br/>symmetric InfoNCE"]
        CLIPSrc["open_clip ViT-B/32<br/>base weights"]
        MLflow[("MLflow tracking<br/>(optional)")]
        Checkpoint[("models/clip_finetuned/<br/>pytorch_model.bin")]

        ExportONNX["ONNX export<br/>(inside finetune_clip.py)"]
        CLIPVis[("models/clip_finetuned/<br/>clip_visual.onnx")]
        CLIPTxt[("models/clip_finetuned/<br/>clip_text.onnx")]
    end

    subgraph Jetson["Jetson Orin Nano"]
        CLIPEnc["CLIPEncoder<br/>encode_image / encode_text"]
        SemMap["SemanticMapManager<br/>uses encoder for verify_arrival +<br/>query_environment"]
    end

    Operator -->|bash $ python -c 'from strafer_lab.tools.dataset_export import run_export; run_export ...'<br/>bash $ python scripts/finetune_clip.py| DatasetExport

    PerceptionDir --> DatasetExport
    DescOut --> DatasetExport
    DatasetExport --> ClipCSV

    ClipCSV --> FinetuneClip
    CLIPSrc --> FinetuneClip
    FinetuneClip -.-> MLflow
    FinetuneClip --> Checkpoint
    Checkpoint --> ExportONNX
    ExportONNX --> CLIPVis
    ExportONNX --> CLIPTxt

    CLIPVis -.->|file copy to Jetson| CLIPEnc
    CLIPTxt -.->|file copy to Jetson| CLIPEnc
    CLIPEnc --> SemMap

    click DatasetExport "../source/strafer_lab/strafer_lab/tools/dataset_export.py" "CLIP CSV + basic VLM JSONL exporter"
    click FinetuneClip "../source/strafer_lab/scripts/finetune_clip.py" "OpenCLIP contrastive fine-tune"
    click CLIPEnc "../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py" "Jetson-side ONNX encoder"
    click SemMap "../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py" "Semantic map manager"
```

**Notes.** Both towers (visual + text) are trained jointly because the
Jetson uses `encode_image()` for place recognition AND `encode_text()`
for text queries. ProcRoom frames are excluded just like Flow 1. MLflow
tracking is optional — `finetune_clip.py` runs without it if
`--mlflow-experiment` is omitted.

---

## Flow 3 — Strafer RL training (Isaac Lab)

**When**: RL training runs, typically long-lived (hundreds to thousands
of PPO iterations). Manually triggered; not automated. Shorter
validation runs (10-100 iterations) are part of the install runbook in
[`VALIDATE_ISAAC_SIM_AND_INFINIGEN.md`](VALIDATE_ISAAC_SIM_AND_INFINIGEN.md).

**Triggered by**: operator on the DGX in `env_phase15`. The launcher
wrapper is Isaac Lab's `isaaclab.sh`.

**Produces**: PPO checkpoints under `logs/rsl_rl/strafer_navigation/<timestamp>/`.
Optional MP4 clips under `videos/train/` when `--video` is passed.

```mermaid
flowchart TB
    Operator((Operator<br/>DGX Spark))

    subgraph DGX["DGX Spark — env_phase15"]
        direction TB

        Launcher["isaaclab.sh -p<br/>Scripts/train_strafer_navigation.py"]
        AppLauncher["AppLauncher<br/>Isaac Sim Kit boot"]

        TasksInit["strafer_lab.tasks.navigation<br/>__init__.py<br/>gym.register × 30"]
        EnvCfg["strafer_env_cfg.py<br/>StraferNavEnvCfg_Real_ProcRoom_NoCam"]
        SimRealCfg["sim_real_cfg.py<br/>REAL_ROBOT_CONTRACT"]
        MDP["tasks/navigation/mdp/<br/>actions / observations / rewards /<br/>terminations / events / curriculums / commands"]

        GymEnv["gym.make(...)<br/>ManagerBasedRLEnv"]
        PolicyInterface["strafer_shared.policy_interface<br/>PolicyVariant.NOCAM"]

        PPO["RSL-RL PPO runner<br/>OnPolicyRunner"]
        RunnerCfg["rsl_rl_ppo_cfg.py<br/>STRAFER_PPO_RUNNER_CFG"]

        VideoRec["gym.wrappers.RecordVideo<br/>(--video)"]

        Tensorboard[("logs/rsl_rl/<br/>strafer_navigation/&lt;ts&gt;/<br/>events.tfevents.*")]
        Checkpoint[("logs/rsl_rl/<br/>strafer_navigation/&lt;ts&gt;/<br/>model_N.pt")]
        Videos[("logs/rsl_rl/<br/>strafer_navigation/&lt;ts&gt;/videos/train/<br/>*.mp4")]
    end

    Operator -->|bash $ ../IsaacLab/isaaclab.sh -p Scripts/train_strafer_navigation.py<br/>--env Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0<br/>--num_envs 512 --max_iterations 3000 --headless| Launcher

    Launcher --> AppLauncher
    AppLauncher --> TasksInit
    TasksInit --> EnvCfg
    EnvCfg --> SimRealCfg
    EnvCfg --> MDP
    TasksInit --> GymEnv
    GymEnv --> PolicyInterface

    RunnerCfg --> PPO
    GymEnv --> PPO
    PPO --> Tensorboard
    PPO --> Checkpoint

    PPO -.->|--video| VideoRec
    VideoRec -.-> Videos

    click Launcher "../Scripts/train_strafer_navigation.py" "Training wrapper (RSL-RL / aux losses / video)"
    click TasksInit "../source/strafer_lab/strafer_lab/tasks/navigation" "30 registered gym environments"
    click EnvCfg "../source/strafer_lab/strafer_lab/tasks/navigation" "Env configs per realism × sensor"
    click SimRealCfg "../source/strafer_lab/strafer_lab/tasks/navigation" "Ideal / Realistic / Robust presets"
    click MDP "../source/strafer_lab/strafer_lab/tasks/navigation" "MDP components"
    click PolicyInterface "../source/strafer_shared/strafer_shared/policy_interface.py" "Shared obs / action contract"
    click RunnerCfg "../source/strafer_lab/strafer_lab/tasks/navigation/agents" "PPO runner cfgs"
```

**Notes.** `train_strafer_navigation.py` (NOT Isaac Lab's stock
`train.py`) is the correct wrapper — the stock script does not import
`strafer_lab.tasks`, so the Strafer envs never get registered. The
`--video` branch records periodic MP4 clips at the configured interval;
the writer captures env-step frames via `gym.wrappers.RecordVideo`. Obs
dimensionality is 19 for `-NoCam-` envs, 4,819 for `-Depth-`, 19,219
for full RGB+Depth.

---

## Flow 4 — Perception data gathering (teleop)

**When**: manual teleop sessions, typically 10-50 episodes per Infinigen
scene before moving on. The operator drives through a procedurally
generated room with a gamepad while the env renders from the 640×360
perception camera and Replicator stamps semantic bboxes on every frame.

**Triggered by**: operator on the DGX in `env_phase15`, with a USB
gamepad connected. Isaac Sim runs at `num_envs=1` (the 640×360 render
caps throughput).

**Produces**: per-episode directories under `data/perception/episode_NNNN/`
containing `frames.jsonl` + `frame_NNNN.jpg` + optional
`frame_NNNN.depth.npy`. Layout exactly matches what
[`generate_descriptions.py`](../source/strafer_lab/scripts/generate_descriptions.py)
and
[`prepare_vlm_finetune_data.py`](../source/strafer_lab/scripts/prepare_vlm_finetune_data.py)
consume — no translation step.

```mermaid
flowchart TB
    Operator((Operator<br/>+ gamepad))

    subgraph DGX["DGX Spark — env_phase15"]
        direction TB

        Collect["collect_perception_data.py<br/>(scripts/)"]
        AppLauncher["AppLauncher<br/>Isaac Sim Kit"]
        Env["Isaac-Strafer-Nav-Real-<br/>InfinigenPerception-Play-v0"]
        SceneUSD[("Assets/generated/scenes/<br/>kitchen_NN/scene.usdc")]
        Robot["StraferRobot<br/>ArticulationCfg"]
        Camera["d555_camera_perception<br/>640×360 RGB + depth"]

        Replicator["Replicator annotator<br/>bounding_box_2d_tight"]
        BBoxExtractor["ReplicatorBboxExtractor<br/>parse rows + labels"]

        Writer["PerceptionFrameWriter<br/>per-episode dirs"]

        EpisodeDir[("data/perception/<br/>episode_NNNN/<br/>frames.jsonl + frame_*.jpg +<br/>frame_*.depth.npy")]
        Stats[("data/perception/<br/>writer_stats.json")]
    end

    Operator -->|bash $ isaaclab -p scripts/collect_perception_data.py<br/>--scene scene_001 --output data/perception/<br/>--max-episodes 20| Collect

    Collect --> AppLauncher
    AppLauncher --> Env
    SceneUSD --> Env
    Env --> Robot
    Env --> Camera

    Operator -->|gamepad<br/>twist| Collect
    Collect -->|action tensor| Env

    Camera --> Replicator
    Replicator --> BBoxExtractor
    Camera -->|RGB JPEG| Writer
    Camera -->|depth npy| Writer
    BBoxExtractor --> Writer

    Writer --> EpisodeDir
    Writer --> Stats

    click Collect "../source/strafer_lab/scripts/collect_perception_data.py" "Gamepad teleop perception collection"
    click Env "../source/strafer_lab/strafer_lab/tasks/navigation" "Infinigen perception env"
    click Replicator "../source/strafer_lab/strafer_lab/tools/bbox_extractor.py" "Replicator bbox annotator wrapper"
    click BBoxExtractor "../source/strafer_lab/strafer_lab/tools/bbox_extractor.py" "DetectedBbox parser"
    click Writer "../source/strafer_lab/strafer_lab/tools/perception_writer.py" "Per-episode writer"
```

**Notes.** `A` = keep episode, `B` = discard, `Start` = save & quit. The
env always advances; after keep/discard the script calls `env.reset()`
and starts the next episode. Replicator's `bounding_box_2d_tight`
annotator needs `semanticLabel` USD prim attributes to exist on scene
objects — those are written by
[`extract_scene_metadata.py`](../source/strafer_lab/scripts/extract_scene_metadata.py)
before any collection run.

---

## Flow 5 — Perception data gathering (sim-in-the-loop)

**When**: autonomous dataset-capture runs, driven by the real Jetson
autonomy stack against the simulated robot. Used to build
reachability-labelled datasets where ground truth (did the robot
actually reach the goal? did Nav2 decide it was unreachable?) is
available per frame.

**Triggered by**: operator on the DGX (harness) + operator on the
Jetson (bringup launch). Both sides must be up; the harness submits
missions via ROS action to the Jetson and captures frames as they are
executed.

**Produces**: `data/sim_in_the_loop/<scene>/episode_NNNN/frames.jsonl`
with mission / reachability / episode-outcome labels attached to each
frame's metadata. Consumed by the same description and SFT-prep
pipelines as Flow 4.

```mermaid
flowchart TB
    DGXOp((Operator<br/>DGX))
    JetsonOp((Operator<br/>Jetson))

    subgraph DGX["DGX Spark — env_phase15"]
        direction TB

        RunSIL["run_sim_in_the_loop.py<br/>--mode harness"]
        AppLauncher["AppLauncher<br/>Isaac Sim Kit"]
        Env["Isaac-Strafer-Nav-Real-<br/>InfinigenPerception-Play-v0"]
        SceneUSD[("Assets/generated/scenes/<br/>scene.usdc + scene_metadata.json")]

        BridgeGraph["bridge/graph.py<br/>OmniGraph builder"]
        BridgeCfg["bridge/config.py<br/>BridgeConfig / CameraStreamConfig"]
        Ros2Ext["isaacsim.ros2.bridge<br/>extension"]

        Harness["SimInTheLoopHarness<br/>harness.py"]
        MissionGen["MissionGenerator<br/>mission.py"]
        RuntimeEnv["runtime_env.py<br/>env adapter"]
        RuntimeMission["runtime_mission.py<br/>rclpy.ActionClient wrapper"]
        Extras["extras.py<br/>reachability + mission metadata"]
        Writer["PerceptionFrameWriter"]

        Dataset[("data/sim_in_the_loop/<br/>&lt;scene&gt;/episode_NNNN/<br/>frames.jsonl + frame_*.jpg")]
    end

    subgraph Jetson["Jetson Orin Nano"]
        direction TB

        Bringup["bringup_sim_in_the_loop.launch.py<br/>perception + SLAM + Nav2 +<br/>goal_projection_node"]
        Executor["strafer-executor<br/>AutonomyCommandServer"]
        Runner["MissionRunner"]
        RosClient["JetsonRosClient"]
        GroundClient["HttpGroundingClient"]
    end

    subgraph Bus["ROS 2 DDS on LAN (ROS_DOMAIN_ID=42)"]
        direction LR
        TopicCam{{/d555/color/image_raw<br/>/d555/depth/image_rect_raw<br/>camera_info sync topics}}
        TopicOdom{{/strafer/odom + /tf}}
        TopicCmd{{/cmd_vel}}
        ActionEM([execute_mission action])
    end

    subgraph VLMSvc["DGX:8100 (HTTP)"]
        VLM["strafer_vlm service<br/>POST /ground /describe /detect_objects"]
    end

    DGXOp -->|bash $ isaaclab -p scripts/run_sim_in_the_loop.py<br/>--mode harness --scene-metadata ... --scene-usd ...<br/>--output data/sim_in_the_loop/kitchen_01| RunSIL
    JetsonOp -->|bash $ ros2 launch strafer_bringup<br/>bringup_sim_in_the_loop.launch.py| Bringup

    RunSIL --> AppLauncher
    AppLauncher --> Env
    SceneUSD --> Env
    BridgeCfg --> BridgeGraph
    Env --> BridgeGraph
    BridgeGraph --> Ros2Ext

    Ros2Ext -->|publish| TopicCam
    Ros2Ext -->|publish| TopicOdom
    TopicCam --> Bringup
    TopicOdom --> Bringup

    RunSIL --> Harness
    Harness --> MissionGen
    MissionGen --> RuntimeMission
    RuntimeMission -->|submit goal| ActionEM
    ActionEM --> Executor
    Executor --> Runner
    Runner --> RosClient
    RosClient --> Bringup
    Runner --> GroundClient
    GroundClient -->|HTTP| VLM

    Bringup -->|cmd_vel| TopicCmd
    TopicCmd -->|subscribe| Ros2Ext
    Ros2Ext -.->|drive sim action| Env

    Harness --> RuntimeEnv
    RuntimeEnv -.->|per-frame RGB / depth / pose| Extras
    RuntimeMission -.->|mission outcome| Extras
    Extras --> Writer
    Writer --> Dataset

    click RunSIL "../source/strafer_lab/scripts/run_sim_in_the_loop.py" "SIL launcher"
    click BridgeGraph "../source/strafer_lab/strafer_lab/bridge/graph.py" "OmniGraph builder"
    click BridgeCfg "../source/strafer_lab/strafer_lab/bridge/config.py" "Bridge config"
    click Harness "../source/strafer_lab/strafer_lab/sim_in_the_loop/harness.py" "SIL harness"
    click MissionGen "../source/strafer_lab/strafer_lab/sim_in_the_loop/mission.py" "Mission generator"
    click RuntimeEnv "../source/strafer_lab/strafer_lab/sim_in_the_loop/runtime_env.py" "Env adapter"
    click RuntimeMission "../source/strafer_lab/strafer_lab/sim_in_the_loop/runtime_mission.py" "ROS action adapter"
    click Extras "../source/strafer_lab/strafer_lab/sim_in_the_loop/extras.py" "Reachability + mission metadata"
    click Writer "../source/strafer_lab/strafer_lab/tools/perception_writer.py" "Frame writer (shared with Flow 4)"
    click Bringup "../source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py" "Jetson-side SIL bringup"
    click Executor "../source/strafer_autonomy/strafer_autonomy/executor/main.py" "Autonomy executor entry"
    click Runner "../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py" "Mission runner"
    click RosClient "../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py" "Jetson ROS client"
    click GroundClient "../source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py" "HTTP grounding client"
    click VLM "../source/strafer_vlm/strafer_vlm/service/app.py" "VLM service"
```

**Notes.** This is the most complex flow in the system: it spans both
hosts, has a feedback loop (`/cmd_vel` from the Jetson drives the sim,
whose sensors come back to the Jetson), and crosses both ROS and HTTP
boundaries. The Jetson autonomy stack runs unchanged — the Isaac Sim
ROS 2 bridge publishes on the same topic names the real robot's
hardware normally produces (`/d555/color/image_raw`, aligned depth,
`/strafer/odom`, TF). **IMU is not published** by the bridge (Isaac Sim
has no `ROS2PublishImu` node), so RTAB-Map runs visual-only in this
mode. Reachability labels come from monitoring Nav2's success/failure
result and any executor-side failure codes.

---

## Flow 6 — End-to-end real-robot execution

**When**: every real-robot mission submission — the "production" path
all other flows feed into. Continuously available when the robot and
DGX services are both up.

**Triggered by**: operator via `strafer-autonomy-cli` over SSH (or any
ROS action client on the Jetson's domain).

**Produces**: physical robot motion + mission action feedback + final
result JSON. Also updates the Jetson-local semantic map with any new
detections observed during the mission.

```mermaid
flowchart TB
    Operator((Operator<br/>SSH terminal))

    subgraph Jetson["Jetson Orin Nano"]
        direction TB

        CLI["strafer-autonomy-cli submit<br/>'go to the tennis ball'"]
        Action([execute_mission action])
        CmdSrv["AutonomyCommandServer<br/>command_server.py"]
        Runner["MissionRunner<br/>13 skills"]

        PlannerCli["HttpPlannerClient"]
        GroundCli["HttpGroundingClient"]
        RosClient["JetsonRosClient<br/>sensor cache + Nav2 + rotate"]
        SemMap["SemanticMapManager<br/>verify_arrival + query_environment"]

        TopicSync{{/d555/color/image_sync<br/>aligned_depth_sync<br/>camera_info_sync<br/>/strafer/odom + /tf}}
        ProjSrv([/strafer/project_detection_to_goal_pose srv])
        ProjNode["goal_projection_node"]
        Nav2Action([/navigate_to_pose action])
        Nav2["Nav2 MPPI holonomic<br/>strafer_navigation"]
        RTAB["RTAB-Map SLAM<br/>strafer_slam"]
        TopicCmd{{/cmd_vel}}
        Driver["roboclaw_node<br/>strafer_driver"]
    end

    subgraph DGX["DGX Spark"]
        direction TB
        Planner["strafer_autonomy.planner<br/>POST /plan<br/>Qwen3-4B → intent → compile"]
        VLM["strafer_vlm service<br/>POST /ground<br/>Qwen2.5-VL-3B"]
    end

    subgraph Robot["Real robot hardware"]
        direction LR
        D555["RealSense D555<br/>RGB + depth + IMU"]
        RoboClaws["2× RoboClaw ST 2x45A<br/>(0x80 + 0x81)"]
        Motors["4× GoBilda 5203<br/>mecanum wheels"]
    end

    Operator -->|bash $ strafer-autonomy-cli submit 'go to the tennis ball'| CLI
    CLI --> Action
    Action --> CmdSrv
    CmdSrv --> Runner

    Runner -->|PlannerRequest| PlannerCli
    PlannerCli -->|HTTP /plan| Planner
    Planner -->|MissionPlan: scan→project→nav→verify| PlannerCli

    Runner -->|GroundingRequest| GroundCli
    GroundCli -->|HTTP /ground| VLM
    VLM -->|bbox normalized 0..1000| GroundCli

    Runner --> RosClient
    RosClient -->|subscribe| TopicSync
    D555 --> TopicSync

    RosClient -->|project srv| ProjSrv
    ProjSrv --> ProjNode
    ProjNode -->|map-frame PoseStamped| RosClient

    RosClient -->|navigate_to_pose goal| Nav2Action
    Nav2Action --> Nav2
    Nav2 --> RTAB
    Nav2 --> TopicCmd
    TopicCmd --> Driver
    Driver -->|serial| RoboClaws
    RoboClaws --> Motors

    Runner --> SemMap
    SemMap -.->|top-k query| Runner

    Runner -.->|feedback| CmdSrv
    CmdSrv -.->|action feedback| Action
    Action -.->|state / step_id / skill| CLI
    CLI -.->|JSON| Operator

    click CLI "../source/strafer_autonomy/strafer_autonomy/cli.py" "Operator CLI"
    click CmdSrv "../source/strafer_autonomy/strafer_autonomy/executor/command_server.py" "AutonomyCommandServer"
    click Runner "../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py" "MissionRunner (13 skills)"
    click PlannerCli "../source/strafer_autonomy/strafer_autonomy/clients/planner_client.py" "HttpPlannerClient"
    click GroundCli "../source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py" "HttpGroundingClient"
    click RosClient "../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py" "JetsonRosClient"
    click SemMap "../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py" "SemanticMapManager"
    click ProjNode "../source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py" "Goal projection node"
    click Nav2 "../source/strafer_ros/strafer_navigation" "Nav2 MPPI holonomic"
    click RTAB "../source/strafer_ros/strafer_slam" "RTAB-Map SLAM"
    click Driver "../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py" "RoboClaw driver"
    click Planner "../source/strafer_autonomy/strafer_autonomy/planner/app.py" "Planner service"
    click VLM "../source/strafer_vlm/strafer_vlm/service/app.py" "VLM service"
```

**Notes.** This is the flow all the upstream training and data
pipelines feed into. The planner runs on DGX port 8200, the VLM on DGX
port 8100, both reached via LAN HTTP. Every mission that ends in
`navigate_to_pose` appends a `verify_arrival` step — a CLIP top-k
ranking against the semantic map — so the executor can catch cases
where the robot "arrived" somewhere that does not look like the goal
area. The flow also powers the `/plan_with_grounding` optimization,
which lets the planner pre-ground the target via a co-located VLM call
and save one LAN image round-trip per mission.

---

## Cross-flow data dependency map

How the six flows feed each other. Solid arrows are hard dependencies
(downstream cannot run without upstream output); dashed arrows are soft
(the consumer augments or validates its own state).

```mermaid
flowchart LR
    subgraph Collect["Data collection"]
        F4["Flow 4<br/>Perception teleop"]
        F5["Flow 5<br/>Sim-in-the-loop"]
    end

    subgraph Train["Training"]
        F1["Flow 1<br/>VLM fine-tune"]
        F2["Flow 2<br/>CLIP fine-tune"]
        F3["Flow 3<br/>RL PPO training"]
    end

    subgraph Serve["Serving / production"]
        F6["Flow 6<br/>Real-robot execution"]
    end

    F4 -->|episodes| F1
    F4 -->|episodes| F2
    F5 -->|reachability-labelled episodes| F1
    F5 -->|reachability-labelled episodes| F2

    F1 -->|LoRA adapter| F6
    F2 -->|clip_visual.onnx<br/>clip_text.onnx| F6
    F3 -->|PPO checkpoint<br/>— deferred: strafer_inference| F6

    F6 -.->|mission failures<br/>— deferred: failure-to-sim pipeline| F5
    F6 -.->|runtime semantic map entries| F6
```

**Notes.**

- Flow 4 and Flow 5 are interchangeable upstream sources for both
  training flows; the layout convention is identical. Flow 5 adds
  reachability + mission outcome metadata to each frame's JSONL record.
- Flow 3's checkpoint feeds Flow 6 through the deferred
  `strafer_inference` path. Until that ships, Flow 6 uses Nav2 as its
  only `navigate_to_pose` backend.
- The `Flow 6 → Flow 5` feedback edge is the deferred "failure-to-sim"
  pipeline (see [`DEFERRED_WORK.md`](DEFERRED_WORK.md)) — it closes the
  loop by turning real-world mission failures into targeted sim
  regression tests.
- Flow 6 feeding back into itself is the semantic map: every mission
  enriches the map (via detections + `detect_objects` calls), which
  later missions query through `verify_arrival` and `query_environment`.
