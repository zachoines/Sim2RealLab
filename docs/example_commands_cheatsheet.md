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

# Full-stack autonomy
## Shell 1: VLM service
```bash
source env_setup.sh
make serve-vlm
```

## Shell 2: LLM planner
```bash
source env_setup.sh
make serve-planner
```

## Shell 3: sim bridge with viewport
```bash
source env_setup.sh
make sim-bridge-gui
```

## Shell 1: full bringup (RTAB-Map + Nav2 + executor)
```bash
source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py \
    vlm_url:=http://192.168.50.196:8100 \
    planner_url:=http://192.168.50.196:8200 \
    rtabmap_args:='-d'
```

## Shell 2: send a real mission
```bash
ros2 action send_goal /execute_mission strafer_autonomy_msgs/action/ExecuteMission \
    '{prompt: "go to the couch"}' --feedback
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