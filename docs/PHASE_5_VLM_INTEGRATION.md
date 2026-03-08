# Phase 5: VLM-Based Goal Setting with Qwen2.5-VL-3B

Natural language command interpretation and visual grounding for autonomous navigation goal generation.

## Overview

Phase 5 introduces a high-level Vision-Language Model (VLM) that interprets natural language commands alongside RGB-D imagery from the Intel RealSense D555 to produce navigation goals for the low-level RL policy. The user says "go to the chair next to the couch," the VLM identifies the chair in the camera frame, the depth map yields a 3D position, and the RL policy navigates there.

This phase does **not** replace the trained RL controller from Phases 1–4. It adds a semantic reasoning layer on top of it, completing the full autonomy stack: language → perception → planning → control.

## 0. Workstation-First Development (No ROS Yet)

Before ROS integration, we develop and validate the VLM stack offline on the Windows training workstation. This keeps iteration fast while we finalize prompting, dataset format, training, and metrics.

### 0.1 Local scripts in this repo

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `source/strafer_vlm/strafer_vlm/test_qwen25vl_grounding.py` | Base-model smoke test on one image | `--image`, `--prompt` | Raw model text, parsed JSON, optional bbox overlay image |
| `source/strafer_vlm/strafer_vlm/train_qwen25vl_lora.py` | LoRA fine-tuning routine | Train/eval JSON or JSONL dataset | Adapter checkpoint dir + `training_summary.json`; optional merged model |
| `source/strafer_vlm/strafer_vlm/eval_qwen25vl_grounding.py` | Offline grounding evaluation | Eval JSON or JSONL dataset + model(+adapter) | `predictions.jsonl` + `evaluation_summary.json` (IoU/Acc metrics) |
| `source/strafer_vlm/strafer_vlm/qwen_vl_common.py` | Shared parser/prompt/metrics utilities | Used by all 3 scripts | Canonical dataset parsing and JSON extraction |

### 0.2 Windows setup and smoke test

```powershell
# From repo root (Windows)
python -m venv .venv_vlm
.\.venv_vlm\Scripts\Activate.ps1

# Install local package
pip install -e source/strafer_vlm

# Core dependencies for local test/train/eval
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate peft trl datasets pillow
```

```powershell
# Base-model single-image check (zero-shot)
python -m strafer_vlm.test_qwen25vl_grounding `
  --image data\vlm\images\frame_0001.jpg `
  --prompt "the chair next to the couch" `
  --output-image outputs\vlm_smoke\frame_0001_bbox.jpg `
  --output-json outputs\vlm_smoke\frame_0001_output.json
```

Notes:
- Start with zero-shot evaluation first; only fine-tune if domain accuracy is insufficient.
- On Windows, start with standard LoRA (`--load-in-4bit` disabled) unless your environment has a working bitsandbytes setup.

## 1. Two-Model Architecture

### 1.1 Design Rationale

The system uses two models operating at fundamentally different timescales:

| Layer | Model | Rate | Responsibility |
|-------|-------|------|----------------|
| **High-level** | Qwen2.5-VL-3B (VLM) | 0.5–2 Hz | Parse NL command, identify target object in RGB frame, output bounding box |
| **Low-level** | RL navigation policy (.pt/.onnx) | 20–50 Hz | Reactive obstacle avoidance, mecanum wheel control, goal-reaching |

This hierarchical separation is the dominant paradigm in 2024–2025 robotics. NVIDIA's GR00T N1 pairs a VLM ("System 2") with a diffusion-based action generator ("System 1") at 120 Hz. NaVILA (RSS 2025) combines a high-level VLA producing mid-level commands with a low-level RL locomotion policy trained in Isaac Lab, achieving 88% real-world navigation success. Google DeepMind's Mobility VLA (CoRL 2024) uses a long-context VLM for goal identification with a topological graph + MPC controller for wheeled navigation.

The key advantages over an end-to-end VLA approach for our mecanum-wheel platform:

- **Temporal decoupling**: Reactive obstacle avoidance at 20–50 Hz cannot wait for VLM inference at 1–5 seconds per frame. End-to-end VLAs like OpenVLA operate at ~5 Hz — too slow for safe mobile navigation with depth-based collision avoidance.
- **Sim-to-real isolation**: The RL policy handles the sim-to-real gap through domain randomization in Isaac Lab. The VLM leverages internet-scale pretraining for visual grounding. Each component's gap is addressed with the appropriate technique — neither contaminates the other.
- **Safety through layering**: The RL policy enforces hard constraints (collision avoidance, velocity limits) regardless of VLM output. If the VLM hallucinates a goal behind a wall, the RL policy navigates safely toward it without driving through obstacles.
- **Independent testing**: Validate the RL policy in simulation with scripted goals, test the VLM with static images and text, then integrate.

### 1.2 Inter-Model Communication

The two models communicate through a **waypoint interface** — the VLM produces a goal pose `(x, y)` in the `map` frame, and the RL policy consumes it as a relative goal observation.

```
User command (text)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  VLM Goal-Setting Pipeline (0.5–2 Hz)                    │
│                                                          │
│  1. Receive RGB frame from /d555/color/image_sync        │
│  2. Receive aligned depth from                           │
│     /d555/aligned_depth_to_color/image_raw               │
│  3. Run Qwen2.5-VL-3B inference:                         │
│     Input: RGB + prompt ("Locate: the chair next         │
│            to the couch")                                │
│     Output: bounding box [x1, y1, x2, y2]               │
│  4. Compute bbox center pixel (cx, cy)                   │
│  5. Sample depth at (cx, cy) from aligned depth frame    │
│  6. Deproject (cx, cy, depth) → 3D point in camera frame │
│     using intrinsics from /d555/color/camera_info_sync   │
│  7. Transform camera_link → map via TF2                  │
│  8. Publish goal_pose on /strafer/goal                   │
└──────────────────┬───────────────────────────────────────┘
                   │  geometry_msgs/PoseStamped (frame: map)
                   ▼
┌──────────────────────────────────────────────────────────┐
│  RL Inference Node (20–50 Hz)                            │
│                                                          │
│  1. Subscribe /strafer/goal → store latest goal          │
│  2. Transform goal to robot-relative (x, y) each cycle   │
│  3. Assemble obs via policy_interface.assemble_observation│
│     with goal_relative = (rel_x, rel_y)                  │
│  4. Run policy forward pass → action [vx, vy, omega]     │
│  5. Publish /strafer/cmd_vel                             │
│  6. Repeat until goal reached or new goal received       │
└──────────────────────────────────────────────────────────┘
```

### 1.3 Goal Lifecycle and Multi-Step Commands

For compound commands ("go to the kitchen, wait 5 seconds, then come back"), a lightweight **command sequencer** node parses the command into discrete steps and manages the state machine:

```
State: IDLE → NAVIGATING → WAITING → NAVIGATING → IDLE
```

The sequencer:
1. Sends the full natural language command to the VLM once to extract a structured plan (JSON list of actions)
2. Issues one goal at a time to the RL inference node
3. Monitors `/strafer/odom` for goal-reached events (distance < threshold)
4. Advances to the next step upon completion

For MVP, the sequencer handles only `navigate_to(object)` and `wait(seconds)`. More complex behaviors (follow, patrol, search) are deferred to Phase 6.

### 1.4 ROS2 Node Graph

```
/d555/color/image_sync ─────────────────┐
/d555/aligned_depth_to_color/image_raw ─┤
                                        ▼
                              ┌─────────────────┐
  User text ─────────────────►│  vlm_goal_node  │──► /strafer/goal (PoseStamped, map frame)
  (service call)              └─────────────────┘
                                                             │
                                                             ▼
                   ┌────────────────────┐   ┌──────────────┐
  /d555/imu/filtered ──►│ inference_node  │──►│ roboclaw_node│──► Motors
  /strafer/joint_states ►│ (RL policy)     │   │ (driver)     │
                   └────────────────────┘   └──────────────┘
                          │
                          ▼
                   /strafer/cmd_vel
```

### 1.5 New Topics and Services

The VLM node publishes goals to the existing `/strafer/goal` topic — the same topic used by manual control and Nav2. The inference node requires no changes; the VLM is just another goal source.

| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| `/strafer/goal` | `geometry_msgs/PoseStamped` | VLM → RL | Navigation goal in `map` frame (shared with Nav2 and manual control) |
| `/strafer/vlm_status` | `std_msgs/String` | VLM → UI | Status: `idle`, `detecting`, `navigating`, `goal_reached`, `object_not_found` |
| `/strafer/set_command` | `strafer_msgs/srv/SetCommand` | UI → VLM | Natural language command input (custom service: `string command` req, `bool success, string status` resp) |
| `/strafer/vlm_detections` | `vision_msgs/Detection2DArray` | VLM → debug | Bounding boxes for RViz2 visualization |

## 2. Qwen2.5-VL-3B Model

### 2.1 Why Qwen2.5-VL-3B

Qwen2.5-VL-3B (Alibaba, Apache 2.0, January 2025) is a 3-billion parameter vision-language model purpose-built for visual grounding tasks. It is the smallest model in the Qwen2.5-VL family (3B, 7B, 72B) and is designed for on-device deployment.

**Native grounding output**: Qwen2.5-VL outputs bounding box coordinates and keypoints directly in the original image frame. Coordinates use a relative [0, 1000] scale, so the model returns `bbox_2d: [x1, y1, x2, y2]` normalized to a 1000×1000 grid regardless of input resolution. This eliminates the need for a separate object detection model.

**Spatial reasoning**: Unlike simpler detection models, Qwen2.5-VL handles complex referring expressions with spatial qualifiers — "the red chair next to the table" or "the door on the left" — because its language decoder reasons over the visual features jointly with the text prompt.

**Edge-deployable**: At ~3B parameters, the model fits on the Jetson Orin Nano (8 GB shared memory) with INT4 quantization (~2–3 GB). The Qwen2.5 text-only 7B model runs at usable speeds on Orin Nano with INT4 via MLC, and the 3B VL variant is well within the memory envelope.

**Fine-tuning ecosystem**: Qwen2.5-VL has first-class support in HuggingFace Transformers (≥4.45), LLaMA-Factory, MS-Swift, and Unsloth. LoRA fine-tuning of the 3B model requires only ~8–14 GB VRAM, fitting comfortably on a single RTX 4090.

### 2.2 How We Use It for Navigation Goal Setting

The VLM operates as a **grounding oracle**: given an RGB image and a text description of the target, it returns the bounding box of the referenced object.

**Inference prompt format** (chat template):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a robot navigation assistant. Given an image from the robot's camera and an object description, locate the described object and return its bounding box as JSON. If the object is not visible, return {\"found\": false}."
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "<base64 or path>"},
        {"type": "text", "text": "Locate: the chair next to the couch"}
      ]
    }
  ]
}
```

**Expected output**:

```json
{"found": true, "bbox_2d": [234, 156, 456, 378], "label": "chair"}
```

**Pixel-to-goal conversion pipeline**:

```python
# 1. Denormalize bbox from [0,1000] to pixel coordinates
img_h, img_w = depth_image.shape[:2]  # actual image dimensions
cx_pixel = int((x1 + x2) / 2 * img_w / 1000)
cy_pixel = int((y1 + y2) / 2 * img_h / 1000)

# 2. Sample depth at bbox center using a 5×5 median patch (aligned depth frame)
# Use /d555/aligned_depth_to_color/image_raw — depth registered to color frame
half = 2
y0, y1p = max(0, cy_pixel - half), min(img_h, cy_pixel + half + 1)
x0, x1p = max(0, cx_pixel - half), min(img_w, cx_pixel + half + 1)
patch = depth_image[y0:y1p, x0:x1p]
depth_m = float(np.median(patch[patch > 0])) / 1000.0  # mm → m

# 3. Deproject to 3D point in camera frame using intrinsics from camera_info_sync
# Subscribe to /d555/color/camera_info_sync and cache K matrix on startup
# fx = K[0], fy = K[4], cx_intr = K[2], cy_intr = K[5]
X_cam = (cx_pixel - cx_intr) * depth_m / fx
Y_cam = (cy_pixel - cy_intr) * depth_m / fy
Z_cam = depth_m

# 4. Transform to map frame via TF2 (camera_link → map using full TF tree)
# Use map frame so the goal persists across robot motion
goal_map = tf_buffer.transform(point_cam_stamped, "map")

# 5. Offset goal to approach position (don't drive INTO the object)
dist = math.hypot(goal_map.x - robot_x, goal_map.y - robot_y)
approach_distance = 0.5  # meters
goal_map.x -= approach_distance * (goal_map.x - robot_x) / dist
goal_map.y -= approach_distance * (goal_map.y - robot_y) / dist
```

**Robustness strategies**:

- **Depth sampling**: Use median of a 5×5 patch around bbox center instead of single pixel to reject depth noise/holes
- **Confidence filtering**: Prompt the VLM to also return a confidence score; reject detections below threshold
- **Temporal persistence**: Require object detection in 2 of 3 consecutive frames before committing to a goal
- **Out-of-frame handling**: If the target object is not visible, rotate in place (publish angular-only `cmd_vel`) and re-query

### 2.3 Training Data

**Phase 1: Zero-shot evaluation (no fine-tuning)**

Qwen2.5-VL-3B has strong zero-shot grounding capability from pretraining. The first step is to evaluate whether zero-shot performance is sufficient for common indoor objects (chairs, tables, doors, appliances). If zero-shot accuracy exceeds 80% on household objects in your operating environment, skip fine-tuning entirely for MVP.

**Phase 2: Auto-labeled bootstrap (if fine-tuning needed)**

If zero-shot falls short on domain-specific objects or lighting conditions, build a fine-tuning dataset using an auto-labeling pipeline:

1. **Capture**: Record 500–2,000 RGB frames from the RealSense D555 in target operating environments (home rooms, office spaces). Vary lighting, viewpoints, and object arrangements.

2. **Auto-label with Grounding DINO**: Run Grounding DINO (zero-shot, text-prompted) on collected frames to automatically generate bounding boxes for target object categories. Grounding DINO achieves 52.5 AP zero-shot on COCO — strong enough for bootstrap labels.

   ```bash
   # Using Autodistill (Roboflow) for automated labeling
   pip install autodistill autodistill-grounding-dino
   ```

3. **Human review**: Manually review and correct 20–40% of auto-labels using Label Studio or CVAT. Focus on edge cases: partially occluded objects, ambiguous referring expressions, unusual viewpoints.

4. **Augment with referring expressions**: For each bounding box, write 2–3 natural language descriptions of varying complexity:
   - Simple: "the chair"
   - Spatial: "the chair next to the window"
   - Descriptive: "the blue office chair on the left side"

**Existing datasets for warm-up fine-tuning**:

| Dataset | Size | Description | Use |
|---------|------|-------------|-----|
| RefCOCO | 142K expressions / 20K images | Short referring expressions (~3.6 words) | General grounding ability |
| RefCOCOg | 104K expressions / 27K images | Longer expressions (~8.4 words) | Complex spatial reasoning |
| Visual Genome | 5.4M region descriptions / 108K images | Scene graph annotations | Broad vocabulary |

**Target dataset size**: 500–2,000 annotated images with 2–3 referring expressions each = 1,000–6,000 training examples. Research shows LoRA fine-tuning of VLMs works with datasets this small.

**Data format** (Qwen2.5-VL chat format with normalized coordinates):

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "images/frame_0042.jpg"},
          {"type": "text", "text": "Locate: the red chair next to the bookshelf"}
        ]
      },
      {
        "role": "assistant",
        "content": "{\"found\": true, \"bbox_2d\": [345, 220, 567, 489], \"label\": \"red chair\"}"
      }
    ]
  }
]
```

**Equivalent flat format** (also supported by local scripts):

```json
[
  {
    "image": "images/frame_0042.jpg",
    "prompt": "Locate: the red chair next to the bookshelf",
    "target": {
      "found": true,
      "bbox_2d": [345, 220, 567, 489],
      "label": "red chair"
    }
  }
]
```

Bounding box coordinates are normalized to [0, 1000] scale relative to the full image. The model handles internal image resizing automatically — no double conversion is needed.

**Isaac Sim synthetic data** (optional, high-impact):

NVIDIA Isaac Sim Replicator can generate synthetic training data from reconstructed environments with domain randomization (lighting, textures, object poses, camera positions) and automatic 2D/3D bounding box annotations. This provides volume and variety without manual labeling. GR00T N1's training showed a 40% performance boost from synthetic data augmentation.

### 2.4 Fine-Tuning Procedure

**Method**: LoRA (Low-Rank Adaptation) applied to the language decoder only. The vision encoder is frozen to prevent catastrophic forgetting and reduce memory.

**Framework options** (ranked by ease of use for this task):

| Framework | Pros | Cons |
|-----------|------|------|
| **LLaMA-Factory** | Code-free YAML config, GUI (LlamaBoard), native Qwen2-VL/2.5-VL support | Less flexibility for custom data loaders |
| **HuggingFace PEFT + TRL** | Most documented, SFTTrainer handles chat templates | More boilerplate code |
| **Unsloth** | 2× faster, 70% less VRAM via optimized kernels | Newer, less community support for VLMs |
| **MS-Swift** | Official Qwen fine-tuning tool, best Qwen2.5-VL support | Documentation primarily in Chinese |

**LoRA configuration**:

```yaml
# LLaMA-Factory config example
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: q_proj,v_proj,k_proj,o_proj  # attention projections only
freeze_vision_tower: true

# Training hyperparameters
learning_rate: 2.0e-4
num_train_epochs: 5
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
gradient_checkpointing: true
bf16: true
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0

# Evaluation
eval_strategy: epoch
metric_for_best_model: eval_loss
```

**VRAM requirements**:

| Method | VRAM | Training Time (1K examples) |
|--------|------|-----------------------------|
| LoRA (FP16) | ~8–14 GB | ~30–60 min on RTX 4090 |
| QLoRA (INT4 + LoRA) | ~4–8 GB | ~45–90 min on RTX 4090 |
| Full fine-tuning | ~28–48 GB | Not recommended for 3B |

**Evaluation metric**: Accuracy@0.5 — percentage of predictions with IoU ≥ 0.5 against ground truth bounding box. This is the standard metric for referring expression comprehension on RefCOCO benchmarks. Target: ≥ 85% on held-out test set.

**Training hardware**:

- **Minimum**: RTX 3090 (24 GB) with QLoRA — ~$900–1,200 used
- **Recommended**: RTX 4090 (24 GB) with LoRA — ~$1,600, completes in under 1 hour
- **Budget**: Google Colab free tier (T4 16 GB) with QLoRA batch_size=1, or Vast.ai RTX 4090 at $0.30–0.50/hr (~$2–5 per training run)

**Post-training**:

```bash
# Merge LoRA adapter into base model
python -m peft.merge_and_unload \
  --base_model Qwen/Qwen2.5-VL-3B-Instruct \
  --lora_adapter ./outputs/checkpoint-final \
  --output_dir ./merged_model

# Quantize for edge deployment (AWQ INT4)
python -m autoawq.quantize \
  --model_path ./merged_model \
  --quant_path ./merged_model_awq_int4 \
  --bits 4 --group_size 128
```

### 2.5 Local Training and Evaluation Routine (Implemented)

The repository now includes a workstation-first loop for Qwen2.5-VL-3B:
1. Smoke test base model on single images
2. Fine-tune with LoRA on local grounding data
3. Evaluate on held-out data with IoU-based metrics

**Train command**:

```powershell
python -m strafer_vlm.train_qwen25vl_lora `
  --train-dataset data\vlm\train.jsonl `
  --eval-dataset data\vlm\val.jsonl `
  --output-dir outputs\qwen25vl_lora_run1 `
  --num-train-epochs 5 `
  --learning-rate 2e-4 `
  --per-device-train-batch-size 2 `
  --gradient-accumulation-steps 8 `
  --torch-dtype bfloat16 `
  --freeze-vision `
  --merge-adapter-after-training
```

**Eval command**:

```powershell
python -m strafer_vlm.eval_qwen25vl_grounding `
  --dataset data\vlm\test.jsonl `
  --model Qwen/Qwen2.5-VL-3B-Instruct `
  --adapter outputs\qwen25vl_lora_run1 `
  --output-dir outputs\qwen25vl_eval_run1 `
  --iou-threshold 0.5
```

**Training inputs**:
- JSON/JSONL dataset in chat format (`messages`) or flat format (`image`, `prompt`, `target`)
- `target.found` boolean and `target.bbox_2d` normalized to [0, 1000] when found is true
- Optional `target.label` and `target.confidence`
- Images are auto-downscaled by default (`--max-image-side 1024`) in test/train/eval scripts to avoid GPU OOM on 8-12 GB cards

**Training outputs**:
- LoRA adapter checkpoint(s) in `--output-dir`
- `training_summary.json` (dataset sizes, hyperparameters, train/eval losses)
- Optional merged model directory when `--merge-adapter-after-training` is enabled

**Evaluation outputs**:
- `predictions.jsonl`: per-sample prompt, ground truth, parsed prediction, raw model output, IoU, latency
- `evaluation_summary.json`: parse success rate, found/not-found accuracy, precision/recall/F1, mean IoU, Acc@IoU threshold

**Canonical prediction schema** (for later ROS integration):

```json
{
  "found": true,
  "bbox_2d": [234, 156, 456, 378],
  "label": "chair",
  "confidence": 0.87
}
```

For ROS phase integration, the contract is:
- Input to VLM: one RGB frame + one referring-expression prompt (`Locate: ...`)
- Output from VLM: normalized 2D bbox JSON (above), then depth projection converts it to a `map`-frame goal pose.

## 3. Deployment on Jetson Orin Nano

### 3.1 Hardware Constraints

| Resource | Budget | Notes |
|----------|--------|-------|
| Memory | 8 GB shared (CPU+GPU) | Must fit VLM + RL policy + ROS2 + drivers |
| Compute | 1024 CUDA cores, 40 TOPS (INT8) | Ampere architecture (SM87) |
| Power | 7–25W (Super Mode) | Enable Super Mode for 1.7–2× perf boost |
| Storage | NVMe SSD required | SD card too slow for model loading |

**Memory budget estimate** (with INT4 Qwen2.5-VL-3B):

| Component | Memory |
|-----------|--------|
| Qwen2.5-VL-3B (INT4/AWQ) | ~2.0–2.5 GB |
| RL policy (ONNX/TensorRT) | ~10–50 MB |
| ROS2 stack + driver nodes | ~300–500 MB |
| RealSense pipeline | ~200–400 MB |
| Linux OS + system | ~800 MB–1 GB |
| **Total** | **~3.5–4.5 GB** |
| **Remaining for KV cache + inference** | **~3.5–4.5 GB** |

This is tight but viable. Key: the VLM runs at low frequency (0.5–2 Hz) and can process one image at a time — no batching needed, keeping KV cache small.

### 3.2 ROS2 Package: `strafer_vlm`

```
source/strafer_ros/strafer_vlm/
├── package.xml
├── setup.py
├── config/
│   └── vlm_params.yaml
├── launch/
│   └── vlm.launch.py
├── strafer_vlm/
│   ├── __init__.py
│   ├── vlm_goal_node.py        # Main node: subscribes to images, serves commands
│   ├── vlm_inference.py        # Model loading, tokenization, inference wrapper
│   ├── depth_projection.py     # Bbox → 3D point via aligned depth + camera_info
│   └── command_sequencer.py    # Multi-step command parsing and state machine
├── models/
│   └── .gitkeep                # Quantized model weights (not in git, downloaded)
└── test/
    ├── test_grounding.py       # Offline VLM accuracy on test images
    └── test_projection.py      # Depth projection unit tests
```

**`strafer_msgs` dependency**: Add `SetCommand.srv` to the `strafer_msgs` package:
```
# strafer_msgs/srv/SetCommand.srv
string command
---
bool success
string status
```
The VLM node serves this as `/strafer/set_command`. This replaces the non-existent `std_srvs/SetString`.

**Node parameters** (`vlm_params.yaml`):

```yaml
vlm_goal_node:
  ros__parameters:
    model_path: "/home/jetson/models/qwen2.5-vl-3b-awq-int4"
    inference_rate: 1.0          # Hz, how often to re-detect if navigating
    confidence_threshold: 0.7    # Minimum detection confidence
    approach_distance: 0.5       # Meters offset from detected object
    goal_reached_threshold: 0.3  # Meters, when to declare goal reached
    depth_patch_size: 5          # Median filter patch for depth sampling
    max_detection_depth: 5.0     # Meters, reject detections beyond this
    enable_rotation_search: true # Rotate to find off-screen objects
    quantization: "awq_int4"     # Quantization method
```

### 3.3 Inference Optimization Strategies

**Strategy A: Direct HuggingFace Transformers (simplest, MVP)**

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "path/to/merged_model_awq_int4",
    torch_dtype=torch.float16,
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
```

Expected latency: ~3–8 seconds per inference on Orin Nano (INT4). Acceptable at 0.5 Hz goal-setting rate.

**Strategy B: MLC-LLM (optimized for Jetson)**

NVIDIA's Jetson benchmarks use MLC for VLM inference with INT4 quantization. MLC compiles the model to optimized GPU kernels, typically achieving 1.5–3× speedup over HuggingFace on Jetson.

```bash
# Compile model for Jetson (on host machine with matching architecture)
mlc_llm compile Qwen2.5-VL-3B-Instruct --quantization q4f16_1 \
  --device cuda --target jetson-orin-nano
```

Expected latency: ~2–5 seconds per inference.

**Strategy C: TensorRT-LLM (highest performance, most complex)**

TensorRT-LLM now has Qwen2.5-VL support. Build the engine on a host machine (Orin Nano lacks memory for engine building) and deploy the serialized engine to the Jetson.

```bash
# On host machine with sufficient memory
trtllm-build --model_dir ./merged_model \
  --output_dir ./trt_engine \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int4 \
  --max_batch_size 1 \
  --max_input_len 2048
```

Expected latency: ~1–3 seconds per inference. Most complex to set up but best performance.

**Strategy D: Cloud API fallback (offload to workstation or cloud)**

For complex multi-turn reasoning or when the Jetson is compute-bound, offload VLM inference to the Windows training workstation or a cloud API:

```python
# ROS2 node can switch between local and remote inference
class VLMInference:
    def __init__(self, mode="local"):
        if mode == "local":
            self.backend = LocalQwenBackend(model_path)
        elif mode == "remote":
            self.backend = RemoteBackend(endpoint="http://workstation:8000/v1/chat/completions")
        elif mode == "cloud":
            self.backend = GeminiFlashBackend(api_key)  # Gemini has native bbox output
```

The Windows workstation (with training GPU) can run vLLM or TGI serving the full-precision model at ~0.5–1s latency over local WiFi. This is useful during development and for handling complex commands that exceed the INT4 model's capability.

### 3.4 Latency Budget

For a navigation command "go to the chair":

| Step | Time | Where |
|------|------|-------|
| Image capture (D555 RGB + depth) | ~33 ms | RealSense pipeline |
| VLM inference (INT4, MLC) | ~2–5 s | Jetson GPU |
| Depth sampling + 3D projection | ~1 ms | CPU |
| TF transform (camera → base_link) | ~1 ms | CPU |
| Goal publication | ~1 ms | ROS2 |
| **Total (initial goal)** | **~2–5 s** | |
| RL policy loop (continuous) | ~20–50 ms/step | Jetson GPU |

The 2–5 second initial detection latency is acceptable because:
1. The user issues a command and expects a brief pause before the robot begins moving
2. Once the goal is set, the RL policy runs at high frequency for smooth, reactive control
3. Re-detection (to update the goal if the robot's view changes) happens at 0.5–2 Hz in parallel with RL control

### 3.5 Integration with Existing Stack

Phase 5 integrates with the existing ROS2 nodes from earlier phases:

```
Phase 2 (strafer_driver)     ← receives /strafer/cmd_vel from RL inference node
Phase 3 (strafer_perception) ← provides /d555/* topics to VLM node
                               (requires align_depth.enable: true in D555 launch)
Phase 3 (strafer_slam)       ← provides /map and TF tree (map → odom → base_link)
Phase 4 (strafer_inference)  ← RL policy node, subscribes to /strafer/goal (unchanged)
Phase 5 (strafer_vlm)        ← NEW: VLM grounding + goal generation → /strafer/goal
```

**Perception prerequisite**: Enable aligned depth in `strafer_perception/launch/d555.launch.py` by setting `align_depth.enable: true`. This publishes `/d555/aligned_depth_to_color/image_raw` — depth registered to the color camera frame, required for pixel-accurate RGB-to-3D projection.

**Training prerequisite**: The RL policy must be trained with goal position noise (`goal_position_noise_std: 0.2–0.3 m`) to be robust to the ±0.2–0.5m localization error typical of VLM visual grounding. Without this, the policy may oscillate or overshoot at deployment. Add this to `Isaac-Strafer-Nav-Real-NoCam-v0` training before exporting the Phase 4 checkpoint.

**Launch composition** (`strafer_bringup/launch/full_stack.launch.py`):

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    return LaunchDescription([
        # Hardware
        IncludeLaunchDescription('strafer_driver/launch/driver.launch.py'),
        IncludeLaunchDescription('strafer_perception/launch/d555.launch.py'),

        # SLAM + Navigation
        IncludeLaunchDescription('strafer_slam/launch/rtabmap.launch.py'),

        # RL Inference
        IncludeLaunchDescription('strafer_inference/launch/inference.launch.py'),

        # VLM Goal Setting (Phase 5)
        IncludeLaunchDescription('strafer_vlm/launch/vlm.launch.py'),
    ])
```

### 3.6 JetPack Configuration

Ensure the Jetson Orin Nano is configured for maximum VLM performance:

```bash
# Enable Super Mode (25W, up to 2× performance)
sudo nvpmodel -m 0
sudo jetson_clocks

# Verify power mode
nvpmodel -q

# Check available memory
free -h
nvidia-smi  # or tegrastats
```

**Required software**:
- JetPack 6.2+ (for Super Mode support)
- CUDA 12.x (included with JetPack)
- Python 3.10+
- PyTorch 2.x (aarch64 wheel from `pypi.jetson-ai-lab.io`)
- Transformers ≥ 4.45 (for Qwen2.5-VL support)
- NVMe SSD for model storage (not SD card)

## 4. Implementation Roadmap

### 4.1 MVP (Phase 5a): Single-Object Navigation

| Step | Task | Duration |
|------|------|----------|
| 5a.1 | Evaluate zero-shot Qwen2.5-VL-3B grounding on test images | 1–2 days |
| 5a.2 | Build `strafer_vlm` package: node, inference wrapper, depth projection | 3–5 days |
| 5a.3 | Test end-to-end: text command → bbox → 3D goal → RL navigation | 2–3 days |
| 5a.4 | Optimize inference (MLC or TensorRT-LLM quantization) | 2–3 days |

**MVP success criterion**: The robot receives "go to the [object]" commands for 5+ common household objects and navigates within 0.5m of the target with ≥ 80% success rate.

### 4.2 Fine-Tuning Sprint (Phase 5b): Domain Adaptation

| Step | Task | Duration |
|------|------|----------|
| 5b.1 | Capture 500+ RGB frames from operating environments | 1–2 days |
| 5b.2 | Auto-label with Grounding DINO, human review | 2–3 days |
| 5b.3 | Fine-tune Qwen2.5-VL-3B with LoRA on RTX 4090 | 1 day |
| 5b.4 | Evaluate on held-out test set (target: ≥ 85% Acc@0.5) | 1 day |
| 5b.5 | Quantize merged model, deploy to Jetson | 1 day |

### 4.3 Multi-Step Commands (Phase 5c): Command Sequencer

| Step | Task | Duration |
|------|------|----------|
| 5c.1 | Implement command sequencer with state machine | 2–3 days |
| 5c.2 | Add "wait" and "return to start" actions | 1 day |
| 5c.3 | Integration test: compound commands end-to-end | 2–3 days |

## 5. Future Directions (Phase 6+)

- **Nav2 integration**: VLM generates semantic goals → Nav2 plans global path → RL policy executes local segments
- **Qwen3-VL upgrade**: Qwen3-VL-4B (released October 2025) offers improved grounding; evaluate as drop-in replacement
- **Object search behavior**: When target is not in frame, use VLM to reason about likely locations ("the kitchen is usually past the living room") and plan exploration
- **Continuous VLM monitoring**: While navigating, periodically re-detect the target to correct for drift in long-range navigation
- **Multi-modal feedback**: Report status back to user via VLM-generated natural language ("I can see the chair, navigating now")

## References

- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) — Architecture and benchmark details
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) — Model card and usage
- [NaVILA](https://arxiv.org/abs/2412.04453) — Hierarchical VLA for robot navigation (RSS 2025)
- [Mobility VLA](https://openreview.net/forum?id=JScswMfEQ0) — VLM + topological navigation (CoRL 2024)
- [NVIDIA GR00T N1](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/) — Dual-system VLA architecture
- [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) — Unified fine-tuning for VLMs
- [Grounding DINO](https://arxiv.org/abs/2303.05499) — Open-set object detection for auto-labeling
- [JetPack 6.2 Super Mode](https://www.edge-ai-vision.com/2025/01/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) — Performance benchmarks
- [TensorRT-LLM Qwen2.5-VL Support](https://nvidia.github.io/TensorRT-LLM/release-notes.html) — Engine building and deployment
