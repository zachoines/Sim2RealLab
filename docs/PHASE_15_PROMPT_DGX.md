# DGX Spark Agent -- Initial Prompt

Copy the text below (between the `---` markers) and paste it as the first message to Claude Code on the DGX Spark.

---

I'm working on a robotics project called Sim2RealLab. This DGX Spark (Grace Blackwell, 128GB unified memory) hosts two FastAPI services: a VLM (Qwen2.5-VL-3B on port 8100) and a planner (Qwen3-4B on port 8200). A Jetson Orin Nano robot calls these services over LAN HTTP for visual grounding and mission planning.

You are the **DGX agent** for phase_15. You have two workstreams:

**Workstream A (planner/VLM services):** Add new VLM endpoints, expand the planner's intent types and compilers, create Databricks client implementations. (Tasks 1-6)

**Workstream B (synthetic data batch processing):** Infinigen high-quality scene generation, scene metadata extraction (room types, object tags, spatial relations), 4-stage scene description pipeline (programmatic spatial analysis -> VLM description generation -> ground truth validation -> human spot-check), CLIP image-text contrastive fine-tuning, and VLM grounding data prep. (Tasks 7-12)

Two other agents are working in parallel -- one on the Jetson (executor/semantic map) and one on the Isaac Sim host (data collection, Replicator bboxes, ROS bridge). You MUST stay within your assigned file boundaries to avoid merge conflicts.

## Setup

```bash
cd ~/Documents/repos/Sim2RealLab
git checkout phase_15
git pull origin phase_15
source .venv_vlm/bin/activate
```

## Context files to read FIRST (in this order)

1. `docs/PHASE_15_DGX.md` -- **your task list, file ownership rules, and platform context**. This is your primary instruction set. Read it completely before writing any code. It covers both workstreams.
2. `docs/INTEGRATION_DGX_SPARK.md` -- full API surface: VLM endpoints, planner pipeline, client protocols, env vars, NVRTC fix.
3. `docs/STRAFER_AUTONOMY_NEXT.md` -- the design document. Reference:
   - Sections 1.12, 2.1-2.2, 3.1-3.5, 4.4 for Workstream A
   - Section 5 (especially 5.3, 5.6, 5.7) for Workstream B

## Rules

- **Only modify files listed in your "Owned directories" section.** Do NOT touch executor/mission_runner.py, ros_client.py, strafer_ros/, strafer_shared/, or Isaac Sim host files (collect_perception_data.py, bbox_extractor.py, bridge/, d555_cfg.py).
- For Workstream B, you own the batch processing scripts and tools in `source/strafer_lab/` (see exact file list in PHASE_15_DGX.md). The Isaac Sim host agent owns the runtime scripts.
- The scene description pipeline uses Qwen2.5-VL-7B (NOT the 3B model being fine-tuned) loaded standalone via `transformers.AutoModelForVision2Seq` in `generate_descriptions.py`. Do NOT import from the `strafer_vlm` package. The 7B model fits easily alongside other workloads on 128GB.
- Use conventional commits (feat:, fix:, test:, etc.). Do NOT add Co-Authored-By lines.
- Run `python -m pytest source/strafer_autonomy/tests/ source/strafer_vlm/tests/ -m "not requires_ros" -v` after each significant change.

## Start

Read `docs/PHASE_15_DGX.md` now. For Workstream A, start with Task 1 (`POST /detect_objects`) or Task 2 (new intent types) -- they're independent. For Workstream B, start with Task 8 (scene metadata extraction) since Tasks 9-12 all depend on it.

---
