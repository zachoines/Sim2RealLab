# Setup environment
```bash
source env_setup.sh
conda activate env_isaaclab3
```

# Run test cases
```bash
# One command per host — auto-dispatches (DGX -> test-dgx, Jetson -> test-jetson)
make test

# DGX e2e: autonomy + vlm + lab. SKIP_KIT=1 swaps the ~40-min Kit suite for
# the fast pure-Python lab half.
make test-dgx
SKIP_KIT=1 make test-dgx

# Jetson e2e: autonomy + ros + driver
make test-jetson

# Individual suites
make test-autonomy   # planner/executor unit tests (host-agnostic)
make test-vlm        # VLM service tests (.venv_vlm)
make test-lab        # all strafer_lab: Kit (run_tests.py) + pure-Python, env_isaaclab3
make test-lab-pure   # fast: strafer_lab pure-Python only, no Kit boot
make test-ros        # ROS 2 packages via colcon (Jetson)
make test-driver     # strafer_driver unit tests (Jetson)
```

# Generate an Infinigen scene corpus (prerequisite for Infinigen training + harness capture)
`generate` now **chains the metadata authoring**: it embeds the per-scene
`objects[]` / `rooms[]` into the USD's `customData` and applies the
`UsdSemantics` detection labels, so one command yields a capture-ready *and*
detections-ready scene. Only the combined manifest (step 2, spawn-point
discovery) remains separate. Full contract: `docs/SCENE_PROVIDER_CONTRACT.md`.
```bash
source env_setup.sh
# 1) Geometry + embedded metadata + detection labels, in one command.
#    Two dimensions: --rooms <types> pins an EXACT tiled layout (duplicates OK);
#    omit for an organic house. --quality {high,low} sets texture + object
#    density (GB10 memory). `prep_room_usds.py info` lists the room types/knobs.
#    Needs $ISAACLAB set (the metadata pass applies the Kit-only UsdSemantics
#    schema); --no-scene-metadata skips it for a geometry-only build.
python source/strafer_lab/scripts/prep_room_usds.py generate \
    --rooms living-room --quality low --name singleroom \
    --output Assets/generated/scenes
# (organic corpus: --quality high --num-scenes 10; two-room: --rooms living-room kitchen)
# note the printed <scene> id, e.g. scene_singleroom_000_seed0

# 2) Combined scenes_metadata.json (spawn points; makes the scene discoverable)
#    --merge preserves existing entries whose heavy USDs aren't on disk this
#    run (the multi-GB high_quality_dgx corpus is not in git); omit to rebuild.
$ISAACLAB -p source/strafer_lab/scripts/generate_scenes_metadata.py --merge
```
Re-author embedded metadata on an existing USD (USD-only, no Blender):
```bash
$ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \
    --from-usd --usd Assets/generated/scenes/<scene>.usdc
```
After both steps: usable by `make sim-bridge`, Infinigen-variant training, and
`capture.py --scene <scene>`.

# Training fresh PPO policy
## (a) Fast, no video
```bash
$ISAACLAB -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-v0 \
    --num_envs 64 \
    --max_iterations 10 --headless
```

## (b) Longer, with video
```bash
$ISAACLAB -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-v0 \
    --num_envs 64 \
    --max_iterations 1000 \
    --headless --video --video_length 200 \
    --video_interval 2000
```

## Open tensorboard
```bash
tensorboard --logdir ~/Workspace/Sim2RealLab/logs/rsl_rl/strafer_navigation
```

## Evaluate Policy
```bash
$ISAACLAB -p source/strafer_lab/scripts/play_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-Play-v0 \
    --checkpoint logs/rsl_rl/strafer_navigation/run_20260425_035916/model_999.pt \
    --viz kit --real_time --steps 2000
```

# Env smoke tests 
## Quick test of the perception env (what the bridge uses)
```bash
$ISAACLAB -p source/strafer_lab/scripts/test_strafer_env.py --env Isaac-Strafer-Nav-Capture-Teleop-v0 --num_envs 1 --duration 5 --headless
```

## ProcRoom-Depth smoke (the variant you want full training on)
```bash
$ISAACLAB -p source/strafer_lab/scripts/test_strafer_env.py --env Isaac-Strafer-Nav-RLDepth-Real-v0 --num_envs 2 --duration 5 --headless
```

## NoCam smoke (fastest, guaranteed to run)
```bash
$ISAACLAB -p source/strafer_lab/scripts/test_strafer_env.py --env Isaac-Strafer-Nav-RLNoCam-v0 --num_envs 8 --duration 10 --headless
```

## (b) Fast, video recorded but camera sits at world origin (frames multiple envs)
```bash
$ISAACLAB -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-v0 \
    --num_envs 64 --max_iterations 50 \
    --headless --video --video_length 200 --video_interval 2000
```

# Collect ~100 episodes (gamepad, headed):
```bash
source env_setup.sh
$ISAACLAB -p source/strafer_lab/scripts/collect_demos.py \
    --task Isaac-Strafer-Nav-RLDepth-Real-Play-v0 \
    --output demos/ --max_episodes 100 --viz kit
```

# DAPG smoke (50 iters, fresh policy):
```bash
$ISAACLAB -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-v0 \
    --num_envs 64 --max_iterations 50 \
    --aux dapg --dapg_demos demos/ \
    --dapg_weight 0.03 --dapg_decay 30 --dapg_batch_size 128
```

# GAIL smoke (50 iters, fresh policy):
```bash
$ISAACLAB -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-v0 \
    --num_envs 64 --max_iterations 50 \
    --aux gail --gail_demos demos/ \
    --gail_reward_weight 1.0 --gail_disc_lr 3e-4 --gail_disc_batch_size 256
```
Watch TB for `dapg_nll`, `dapg_weight`, `gail_reward`, `gail_disc_loss`, `gail_disc_expert`, `gail_disc_policy` — those scalars come from the aux loop and prove it executed.


# Fine-tune a live checkpoint with demos:
```bash
$ISAACLAB -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-v0 \
    --num_envs 128 --max_iterations 6000 \
    --resume logs/rsl_rl/strafer_navigation/run_20260425_035916/model_999.pt \
    --seed 1337 \
    --aux dapg --dapg_demos demos/ \
        --dapg_weight 0.05 \
        --dapg_decay 4000 \
        --dapg_batch_size 64 \
        --dapg_min_return_pct 0.25 \
        --dapg_action_noise 0.03 \
    --headless --video --video_length 300 --video_interval 30000
```

# Headed inference rollout from a trained checkpoint
Loads model_*.pt and steps the env in inference mode so you can watch the
policy in the Kit viewport. Use the Play variant (8 envs by default).

## (a) Headed, watch in the viewport, real-time pacing
```bash
$ISAACLAB -p source/strafer_lab/scripts/play_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-Play-v0 \
    --checkpoint logs/rsl_rl/strafer_navigation/run_20260425_035916/model_600.pt \
    --viz kit --real_time --steps 2000
```

## (b) Headless rollout that records a single MP4 over env_0
```bash
$ISAACLAB -p source/strafer_lab/scripts/play_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Real-Play-v0 \
    --checkpoint logs/rsl_rl/strafer_navigation/run_20260425_035916/model_600.pt \
    --headless --video --video_length 600
```
MP4 lands in `logs/rsl_rl/strafer_navigation/play_videos/play_<timestamp>/`.

# Export a checkpoint to a deployable artifact
Converts an `rsl_rl` PPO checkpoint into a TorchScript `.pt` (and
optionally an ONNX `.onnx`) loadable by
`strafer_shared.policy_interface.load_policy()` on the Jetson. Writes a
`<output>.json` sidecar with variant, dimensions, source checkpoint,
git commit, and ONNX opset.

## NoCam — TorchScript only
```bash
$ISAACLAB -p source/strafer_lab/scripts/export_policy.py \
    --checkpoint logs/rsl_rl/strafer_navigation/run_<timestamp>/model_<step>.pt \
    --output models/strafer_nocam_v0 \
    --variant NOCAM
```

## NoCam — TorchScript + ONNX (TRT-EP path on Jetson)
```bash
$ISAACLAB -p source/strafer_lab/scripts/export_policy.py \
    --checkpoint logs/rsl_rl/strafer_navigation/run_<timestamp>/model_<step>.pt \
    --output models/strafer_nocam_v0 \
    --variant NOCAM \
    --formats pt,onnx
```

## Depth — both formats (recurrent multi-input)
Depth export uses the recurrent `(obs, h_in) -> (actions, h_out)`
signature; the loader threads hidden state across ticks. Sidecar records
`is_recurrent: true` so the inference node knows to call `.reset()` on
episode boundaries. DeFM's `BiFPN.WeightedFusion` uses `sum(generator)`
which `torch.jit.script` cannot type-infer; the TorchScript path
substitutes a pre-traced pipeline (`_TorchSafeDeFMDepthEncoder`) so
the same export call emits both artifacts.
```bash
$ISAACLAB -p source/strafer_lab/scripts/export_policy.py \
    --checkpoint logs/rsl_rl/strafer_navigation/run_<timestamp>/model_<step>.pt \
    --output models/strafer_depth_v0 \
    --variant DEPTH \
    --formats pt,onnx
```

## Smoke-test the exported artifact in sim
Re-runs `play_strafer_navigation.py` against the exported `.pt` instead
of the rsl_rl checkpoint; verifies the export didn't break the policy.
Single-env only (the export is deployment-shape).
```bash
$ISAACLAB -p source/strafer_lab/scripts/play_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLNoCam-Play-v0 \
    --policy models/strafer_nocam_v0.pt \
    --num_envs 1 --viz kit --real_time --steps 2000
```

## Bench inference latency on an exported artifact
Reports median / p95 / p99 over 1000 iterations on a synthetic obs.
```bash
# DGX (CPU EP) -- regression check on the export toolchain.
python source/strafer_lab/scripts/benchmark_policy.py --model models/strafer_nocam_v0.pt --iters 1000

# Jetson (TRT EP preferred, then CUDA, then CPU fallback) -- run after rsync.
# Depth variants: keep TRT first (TRT ~4.7 ms, CUDA ~7.5 ms, CPU ~84 ms).
# NoCam variants: --providers CPUExecutionProvider is fastest (~0.056 ms;
# GPU kernel-launch overhead dominates the 19-dim MLP).
python3 source/strafer_lab/scripts/benchmark_policy.py \
    --model models/strafer_depth_v0.onnx \
    --providers TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider \
    --iters 1000
```

# Full-stack sim-in-the-loop autonomy

End-to-end orchestration: DGX runs the VLM + planner + Isaac Sim
bridge, Jetson runs the autonomy stack, the operator
workstation connects Foxglove Studio over an SSH tunnel for live
visual debugging.

## DGX shell A — VLM grounding service
```bash
source env_setup.sh
make serve-vlm
```

## DGX shell B — LLM planner service
```bash
source env_setup.sh
make serve-planner
```

## DGX shell C — Isaac Sim bridge
```bash
source env_setup.sh
make sim-bridge          # headless (daily-driver, ~85 ms/loop faster)
# or:
make sim-bridge-gui      # editor viewport open (visual debug, slower)
# Pin the bridge to a chosen scene (else the env default loads):
SCENE_USD=Assets/generated/scenes/scene_singleroom_000_seed0.usdc make sim-bridge
# or by discoverable name: SCENE_NAME=scene_singleroom_000_seed0 make sim-bridge
```

## Jetson shell 1 — full bringup (perception + SLAM + Nav2 + executor + foxglove_bridge)
```bash
make launch-sim
# Equivalent to:
#   source ~/strafer_ws/install/setup.bash
#   source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
#   ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py \
#       vlm_url:=http://192.168.50.196:8100 planner_url:=http://192.168.50.196:8200
#
# Config lives in the compose env file (container-primary), NOT shell vars:
#   edit source/strafer_ros/deploy/compose/sim.env  (VLM_URL/PLANNER_URL,
#   STRAFER_NAV_BACKEND, timeouts), then: make launch-sim
# Different launch args (donut_warmup, viewer, rtabmap_viz) or command: edit the
#   `command:` in deploy/docker-compose.sim.yml, or iterate live via the dev overlay
#   (docker compose -f docker-compose.yml -f docker-compose.dev.yml up).
# Wipe the RTAB-Map database before launching (fresh map):
#   make clean-map && make launch-sim
```

## Jetson shell 2 — submit a mission
```bash
# Container-primary: `make submit` execs the CLI inside the running sim container
# (the CLI is a ROS action client -> ExecuteMission over DDS). Target must match a
# groundable object the loaded scene actually contains (its customData objects[];
# Infinigen doesn't guarantee a specific one).
make submit CMD="go to the <scene target object>"
# status / cancel (exec directly; no make wrapper):
SIM='docker compose -f source/strafer_ros/deploy/docker-compose.sim.yml'
$SIM exec strafer-sim strafer-autonomy-cli status
$SIM exec strafer-sim strafer-autonomy-cli cancel
# bare-metal (advanced): source .../env_sim_in_the_loop.env; strafer-autonomy-cli submit "..."
```

## Obs / subgoal parity (diagnostic — trained-policy deployment lane)
```bash
# 1) Enable the node's obs dump (diagnostic; adds one JSONL line per tick,
#    written after the cmd_vel publish so it never delays control). Launch the
#    inference node with a parking artifact loaded + a mission running — an
#    empty model_path assembles no subgoal obs and dumps nothing.
ros2 run strafer_inference inference_node --ros-args \
    -p policy_variant:=NOCAM_SUBGOAL \
    -p model_path:=/home/jetson/workspaces/Sim2RealLab/models/strafer_nocam_subgoal_v0.onnx \
    -p obs_dump_path:=/tmp/node_obs.jsonl -p use_sim_time:=true
# 2) Record a >=30 s bag with the robot moving.
ros2 bag record -o /tmp/parity_bag /d555/imu/filtered /strafer/joint_states \
    /strafer/odom /strafer/subgoal /plan /tf /tf_static /clock

# 3a) Obs parity vs a workstation gym dump (strict gate: scalar <=1e-5, depth <=1e-3).
python3 source/strafer_ros/strafer_inference/scripts/obs_parity.py \
    --node-dump /tmp/node_obs.jsonl --gym-dump /tmp/gym_obs.jsonl
# 3b) Obs self-check (no workstation): re-assemble the reference from the bag's raw topics.
python3 source/strafer_ros/strafer_inference/scripts/obs_parity.py \
    --node-dump /tmp/node_obs.jsonl --self-check --bag /tmp/parity_bag
# 4) Rolling-subgoal pick parity (bag-replay self-consistency, <=10 cm).
python3 source/strafer_ros/strafer_inference/scripts/subgoal_parity.py --bag /tmp/parity_bag
# JSONL contract both sides emit against: scripts/PARITY_SCHEMA.md
```

## Operator workstation — Foxglove Studio over SSH (live debug visualizer)
```bash
# operator workstation, terminal 1: keep this open while debugging
ssh -L 8765:localhost:8765 jetson-desktop
```
Then open Foxglove Studio (desktop app or
<https://app.foxglove.dev/>) → **Open connection** → **Foxglove
WebSocket** → `ws://localhost:8765`. First-time setup: **Layout** →
**Import from file** → `source/strafer_ros/strafer_bringup/foxglove/strafer_layout.json`.

## Reset between runs
```bash
make kill           # Jetson — clear stale ros2 / nav2 / executor / foxglove_bridge
```

# Harness data capture (`source/strafer_lab/scripts/capture.py`)

Moved to its own guide — see
[`docs/HARNESS_DATA_CAPTURE.md`](HARNESS_DATA_CAPTURE.md). The guide
covers `env_isaaclab3` lerobot setup, the Infinigen scene-corpus
clean-slate regen procedure, mission-queue generation, validation +
production capture commands, the operator button mapping, optional flags,
round-trip verification, and a troubleshooting table.

```bash
# Generate free-text mission queues from scene metadata (per-scene queue.yaml
# + unioned corpus.yaml; headless, model-free by default). See the guide.
$STRAFER_ISAACLAB_PYTHON source/strafer_lab/scripts/build_mission_corpus.py --mode mixed

# Bulk-capture default — diverse-perspective coverage sweep (trained RL
# subgoal-follower over a geometric coverage plan). Teleop / bridge are the
# non-bulk paths. Checkpoint is an exported export_policy.py artifact.
$ISAACLAB -p source/strafer_lab/scripts/capture.py \
    --driver scripted --mission-source coverage \
    --scene <scene> --output data/sim_in_the_loop/<scene> \
    --policy-variant nocam_subgoal --checkpoint models/strafer_nocam_subgoal.pt \
    --headless --enable_cameras
```

# Sim-in-the-loop bridge + DDS bench
## Shell 1 — start the bridge with viewport
```bash
source env_setup.sh
make sim-bridge-gui
# optionally pin a scene: SCENE_USD=<scene>.usdc make sim-bridge-gui
```

## Shell 2 — subscribe and measure publish rates from the env_infinigen 3.11 env
```bash
source env_setup.sh
PYTHONPATH="$STRAFER_ROS2_HUMBLE_PY311_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH="$STRAFER_ROS2_HUMBLE_PY311_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
$STRAFER_INFINIGEN_PYTHON /tmp/sim_bridge_bench/bench.py
```