# Setup environment
```bash
source env_setup.sh
conda activate env_isaaclab3
```

# Run test cases
```bash
$ISAACLAB -p source/strafer_lab/run_tests.py all
make test-dgx
```

# Training fresh PPO policy
## (a) Fast, no video
```bash
$ISAACLAB -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
    --num_envs 64 \
    --max_iterations 10 --headless
```

## (b) Longer, with video
```bash
$ISAACLAB -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
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
$ISAACLAB -p Scripts/play_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \
    --checkpoint logs/rsl_rl/strafer_navigation/run_20260425_035916/model_999.pt \
    --viz kit --real_time --steps 2000
```

# Env smoke tests 
## Quick test of the perception env (what the bridge uses)
```bash
$ISAACLAB -p Scripts/test_strafer_env.py --env Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0 --num_envs 1 --duration 5 --headless
```

## ProcRoom-Depth smoke (the variant you want full training on)
```bash
$ISAACLAB -p Scripts/test_strafer_env.py --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 --num_envs 2 --duration 5 --headless
```

## NoCam smoke (fastest, guaranteed to run)
```bash
$ISAACLAB -p Scripts/test_strafer_env.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 8 --duration 10 --headless
```

## (b) Fast, video recorded but camera sits at world origin (frames multiple envs)
```bash
$ISAACLAB -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
    --num_envs 64 --max_iterations 50 \
    --headless --video --video_length 200 --video_interval 2000
```

# Collect ~100 episodes (gamepad, headed):
```bash
source env_setup.sh
$ISAACLAB -p source/strafer_lab/scripts/collect_demos.py \
    --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \
    --output demos/ --max_episodes 100 --viz kit
```

# DAPG smoke (50 iters, fresh policy):
```bash
$ISAACLAB -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
    --num_envs 64 --max_iterations 50 \
    --aux dapg --dapg_demos demos/ \
    --dapg_weight 0.03 --dapg_decay 30 --dapg_batch_size 128
```

# GAIL smoke (50 iters, fresh policy):
```bash
$ISAACLAB -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
    --num_envs 64 --max_iterations 50 \
    --aux gail --gail_demos demos/ \
    --gail_reward_weight 1.0 --gail_disc_lr 3e-4 --gail_disc_batch_size 256
```
Watch TB for `dapg_nll`, `dapg_weight`, `gail_reward`, `gail_disc_loss`, `gail_disc_expert`, `gail_disc_policy` — those scalars come from the aux loop and prove it executed.


# Fine-tune a live checkpoint with demos:
```bash
$ISAACLAB -p Scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
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
$ISAACLAB -p Scripts/play_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \
    --checkpoint logs/rsl_rl/strafer_navigation/run_20260425_035916/model_600.pt \
    --viz kit --real_time --steps 2000
```

## (b) Headless rollout that records a single MP4 over env_0
```bash
$ISAACLAB -p Scripts/play_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \
    --checkpoint logs/rsl_rl/strafer_navigation/run_20260425_035916/model_600.pt \
    --headless --video --video_length 600
```
MP4 lands in `logs/rsl_rl/strafer_navigation/play_videos/play_<timestamp>/`.

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
# Override service URLs if the DGX moves:
#   VLM_URL=http://other:8100 PLANNER_URL=http://other:8200 make launch-sim
# Disable the headless visualizer:
#   ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py viewer:=false
# Wipe the RTAB-Map database before launching (fresh map):
#   make clean-map && make launch-sim
```

## Jetson shell 2 — submit a mission
```bash
strafer-autonomy-cli submit "go to the couch"
strafer-autonomy-cli status
strafer-autonomy-cli cancel
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

# Sim-in-the-loop bridge + DDS bench
## Shell 1 — start the bridge with viewport
```bash
source env_setup.sh
make sim-bridge-gui
```

## Shell 2 — subscribe and measure publish rates from the env_infinigen 3.11 env
```bash
source env_setup.sh
PYTHONPATH="$STRAFER_ROS2_HUMBLE_PY311_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH="$STRAFER_ROS2_HUMBLE_PY311_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
$STRAFER_INFINIGEN_PYTHON /tmp/sim_bridge_bench/bench.py
```