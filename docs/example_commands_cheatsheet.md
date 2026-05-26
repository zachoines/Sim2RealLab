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

# Export a checkpoint to a deployable artifact
Converts an `rsl_rl` PPO checkpoint into a TorchScript `.pt` (and
optionally an ONNX `.onnx`) loadable by
`strafer_shared.policy_interface.load_policy()` on the Jetson. Writes a
`<output>.json` sidecar with variant, dimensions, source checkpoint,
git commit, and ONNX opset.

## NoCam — TorchScript only
```bash
$ISAACLAB -p Scripts/export_policy.py \
    --checkpoint logs/rsl_rl/strafer_navigation/run_<timestamp>/model_<step>.pt \
    --output models/strafer_nocam_v0 \
    --variant NOCAM
```

## NoCam — TorchScript + ONNX (TRT-EP path on Jetson)
```bash
$ISAACLAB -p Scripts/export_policy.py \
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
$ISAACLAB -p Scripts/export_policy.py \
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
$ISAACLAB -p Scripts/play_strafer_navigation.py \
    --env Isaac-Strafer-Nav-Real-ProcRoom-NoCam-Play-v0 \
    --policy models/strafer_nocam_v0.pt \
    --num_envs 1 --viz kit --real_time --steps 2000
```

## Bench inference latency on an exported artifact
Reports median / p95 / p99 over 1000 iterations on a synthetic obs.
```bash
# DGX (CPU EP) -- regression check on the export toolchain.
python Scripts/benchmark_policy.py --model models/strafer_nocam_v0.pt --iters 1000

# Jetson (TRT EP preferred, then CUDA, then CPU fallback) -- run after rsync.
python3 Scripts/benchmark_policy.py \
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
# Skip the ~one-rotation startup warmup spin (iteration sessions):
#   DONUT_WARMUP=false make launch-sim
# Pass arbitrary extra launch args (e.g. open RTAB-Map viz, wipe DB):
#   LAUNCH_ARGS="rtabmap_viz:=true rtabmap_args:=-d" make launch-sim
# Disable the headless visualizer:
#   ros2 launch strafer_bringup bringup_sim_in_the_loop.launch.py viewer:=false
# Wipe the RTAB-Map database before launching (fresh map):
#   make clean-map && make launch-sim
```

## Jetson shell 2 — submit a mission
```bash
source source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env
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

# Harness data capture (`Scripts/capture.py`)

The unified harness data-capture entry point per
[`harness-architecture.md`](tasks/active/harness/harness-architecture.md).
One CLI, two flags (`--driver` × `--mission-source`), one LeRobot v3
dataset per scene under `data/sim_in_the_loop/<scene_name>/`.

Tier 1 wires `(teleop, scene-metadata)` end-to-end. Other cells raise
`NotImplementedError` with a pointer to the tier that ships them.

## One-time env setup

`env_isaaclab3` ships with `torch 2.10.0+cu130` + `numpy 2.3.1` +
`huggingface-hub 0.36`. `lerobot 0.5.1` pins are mostly compatible
except for `numpy`, `huggingface-hub`, and `rerun-sdk` — a normal
`pip install lerobot` would downgrade numpy (risks breaking
Isaac Sim) and major-bump huggingface-hub (risks breaking
transformers). Install `--no-deps` and layer only the runtime deps the
writer actually uses:

```bash
conda activate env_isaaclab3

# Note: $ISAACLAB is an Isaac Lab wrapper that only forwards args after
# its own flags — it can't run `-m pip` directly. Use the conda env's
# Python (`python -m pip` works once you're inside the env).

# 1. Install lerobot core without dragging its strict pins in
python -m pip install --no-deps "lerobot==0.5.1"

# 2. Install only the runtime deps StraferLeRobotWriter uses, refusing
#    to upgrade anything that's already installed and satisfies the new pin
python -m pip install --upgrade-strategy only-if-needed \
    "datasets>=4.0.0,<5.0.0" \
    "av>=15.0.0,<16.0.0" \
    "jsonlines>=4.0.0,<5.0.0"

# 3. Verify lerobot imports + Isaac Sim's torch still has CUDA
python -c "import torch, lerobot; print('torch', torch.__version__, 'lerobot', lerobot.__version__, 'cuda', torch.cuda.is_available())"
# Expected: torch 2.10.0+cu130 lerobot 0.5.1 cuda True
```

Pip will print warnings that lerobot's strict pins on `numpy`, `huggingface-hub`,
`rerun-sdk`, `setuptools`, `packaging`, and a few `wandb` / `pynput` /
`pyserial` / `termcolor` deps aren't satisfied. **Those warnings are
expected and safe to ignore** — `--no-deps` deliberately skipped them
to keep Isaac Sim's stack intact. The narrow LeRobot v3 writer surface
the harness actually uses (`LeRobotDataset.create / add_frame /
save_episode / finalize`) was end-to-end smoke-tested against this
install and works (verified 2026-05-26 on the DGX).

Pure-Python unit tests (writer / depth / mission picker / button
translator / CLI dispatch) run in `.venv_harness`, NOT `env_isaaclab3`,
so they stay isolated from the runtime stack:

```bash
make test-harness   # 116 tests, ~2 s
```

## Extract scene_metadata.json (one-time per scene)

The mission picker reads `Assets/generated/scenes/<scene>/scene_metadata.json`.
If you have only the `.usdc` (no Blender / in-process Infinigen
`State`), parse it from prim names:

```bash
SCENE=scene_high_quality_dgx_000_seed0

$ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \
    --from-usd \
    --usd    Assets/generated/scenes/${SCENE}.usdc \
    --output Assets/generated/scenes/${SCENE}

# Sanity check — should be non-empty
python -c "
import json
d = json.load(open('Assets/generated/scenes/${SCENE}/scene_metadata.json'))
print(f'rooms={len(d.get(\"rooms\",[]))}  objects={len(d.get(\"objects\",[]))}')
for o in d['objects'][:10]:
    print(' -', o.get('label'), o.get('instance_id'))
"
```

**Known limitation:** `--from-usd` cannot recover room polygons, so
the picker shows `rooms=0` for these scenes. Room semantics (which
hard-negative button chord maps to "wrong_room") still work; the
operator commits to the failure mode at capture time. For full room
geometry, run from a Blender stage or extract from the in-process
Infinigen `State`. Tracked in
[`docs/tasks/active/harness/infinigen-scene-corpus.md`](tasks/active/harness/infinigen-scene-corpus.md).

## Validation capture (small batch — driver wiring works)

```bash
SCENE=scene_high_quality_dgx_000_seed0
RUN_ID=$(date +%Y%m%dT%H%M%S)
OUT=data/sim_in_the_loop/${SCENE}_validation_${RUN_ID}

$ISAACLAB -p Scripts/capture.py \
    --driver teleop --mission-source scene-metadata \
    --scene  ${SCENE} \
    --output ${OUT} \
    --fps 8 \
    --max-episodes 5
```

### Button mapping (per
[`harness-architecture.md` §Episode-end button mapping](tasks/active/harness/harness-architecture.md#episode-end-button-mapping-teleop-only))

| Button | `outcome` | Kept? |
|---|---|---|
| `Y` (triangle / north) | `succeeded` | yes |
| `B` (circle / east) | `failed` | yes |
| `X` + D-pad ↑/↓ | `wrong_instance` hard negative | yes |
| `X` + D-pad ←/→ | `wrong_room` hard negative | yes |
| `SELECT` (share / minus) | `trajectory_violation` | yes |
| `Back` (view) | discard | **no** |
| `A` (tap) | toggle `REC` ↔ `PAUSED` | — |
| `Start` (hold ≥ 1 s) | save + quit cleanly | — |

`X` alone (no D-pad direction) is not committal — push the D-pad to
commit. Between episodes the driver re-prompts via the console mission
picker (numeric index; Ctrl-D quits cleanly).

### Optional flags

| Flag | Default | Use |
|---|---|---|
| `--no-pip-window` | (PIP on) | Suppress the cv2 first-person preview window |
| `--no-capture-policy-cam` | (policy cam on) | Drop the 80×60 policy camera (smaller dataset) |
| `--operator-handle <name>` | none | Stamped on every episode for multi-operator runs |
| `--target-label-filter chair table` | none | Narrow the picker list |
| `--max-steps-per-episode 1500` | 1500 | Auto-close cap (logs `outcome=failed`) |
| `--headless` | (headed) | AppLauncher pass-through; cv2 PIP still works |
| `--device cpu` | gpu | AppLauncher pass-through |

## Round-trip verification

```bash
# Re-open the dataset via stock LeRobotDataset + read the strafer sidecar
$ISAACLAB -p -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from strafer_lab.tools.lerobot_writer import read_strafer_episodes
from pathlib import Path
root = Path('${OUT}')
d = LeRobotDataset(repo_id='strafer/${SCENE}', root=root)
print(f'episodes={d.num_episodes}  frames={len(d)}')
for row in read_strafer_episodes(root):
    print(f'  ep{row[\"episode_index\"]}  outcome={row[\"outcome\"]:22s}  '
          f'target={row[\"target_label\"]:20s}  git_sha={row[\"capture_git_sha\"][:8]}')
"

# Spot-check a saved frame for PIP-overlay contamination (CRITICAL: per the brief)
ffmpeg -y -i ${OUT}/videos/chunk-000/observation.images.perception/file-000000.mp4 \
       -frames:v 1 /tmp/capture_smoke_frame.png
xdg-open /tmp/capture_smoke_frame.png   # must NOT show [REC] / step / distance overlay
```

## Production capture (≥ 30 episodes, multi-scene)

Per the brief's acceptance bar — run after the
[Infinigen scene-corpus brief](tasks/active/harness/infinigen-scene-corpus.md)
generates richer scenes. Same command pattern; raise `--max-episodes`
and `--max-steps-per-episode`, and commit a summary under
`docs/artifacts/teleop_acceptance/<run_id>/`.

## Troubleshooting

| Symptom | Most likely cause | What to check |
|---|---|---|
| `ModuleNotFoundError: No module named 'lerobot'` | env_isaaclab3 doesn't have lerobot | Run the one-time env setup above |
| `scene_metadata.json not found` | Scene hasn't been extracted | Run the extraction step above |
| `--output already exists` | LeRobotDataset.create refuses to overwrite | Pick a fresh path per session (timestamp helps) |
| `No gamepad detected` | pygame can't find the joystick | `jstest /dev/input/js0` to confirm the kernel sees it |
| Wrong button does the wrong thing | family auto-detect picked wrong | Add `--family-override ps5` (or `xbox` / `switch`) — TODO if needed |
| `cv2.error: ... The function is not implemented. Rebuild the library with ... GTK+ ...` on `cv2.namedWindow` | `opencv-python-headless` (env_isaaclab3's variant) has no GUI backend by design | The driver now degrades to "PIP off" automatically — capture continues; use the Isaac Sim editor viewport as your live view. Pass `--no-pip-window` to silence the warning |
| PIP HUD overlay leaks into saved frames | cv2 putText is rendering into the perception render product | **Hard acceptance fail per the brief.** File a bug; the cv2 window must be a separate top-level surface |
| Round-trip via HF `LeRobotDataset` fails with codec error | torchcodec missing on aarch64 | The wheel marker excludes aarch64; LeRobot falls back to PyAV which works. If you've manually installed torchcodec, uninstall it |

---

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