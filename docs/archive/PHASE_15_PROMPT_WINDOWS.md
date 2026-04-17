# Isaac Sim Host Agent -- Initial Prompt

Copy the text below (between the `---` markers) and paste it as the first message to Claude Code on the machine running Isaac Sim (DGX Spark or Windows workstation).

---

I'm working on a robotics project called Sim2RealLab. This machine runs Isaac Sim / Isaac Lab for simulation. The project includes `strafer_lab`, an Isaac Lab extension for training and data generation with a GoBilda Strafer mecanum-wheel robot.

You are the **Isaac Sim host agent** for phase_15. Your job is to build the simulation runtime components: dual camera configuration, gamepad-teleoperated data collection through Infinigen scenes, ground-truth bbox extraction via Replicator, and the Isaac Sim ROS2 Bridge for scalable autonomous data collection.

A separate DGX agent handles batch processing (Infinigen generation, scene metadata extraction, description pipeline, model fine-tuning). A Jetson agent handles the executor/semantic map. You MUST stay within your assigned file boundaries to avoid merge conflicts.

## Setup

```bash
cd <path-to>/Sim2RealLab
git checkout phase_15
git pull origin phase_15
```

## Context files to read FIRST (in this order)

1. `docs/PHASE_15_WINDOWS.md` -- **your task list, file ownership rules, and platform context**. This is your primary instruction set. Read it completely before writing any code. It includes the Isaac Lab API surface, camera specs, and what files are owned by the DGX agent (do not touch those).
2. `docs/STRAFER_AUTONOMY_NEXT.md` -- the design document. Reference Section 5 (Synthetic Data and Sim-to-Real) for architecture, dual camera config, data collection approach, ROS2 Bridge spec, and command generation pipeline.

## Rules

- **Only modify files listed in your "Owned files" section of PHASE_15_WINDOWS.md.** Do NOT touch DGX-owned files (prep_room_usds.py, extract_scene_metadata.py, generate_descriptions.py, finetune_clip.py, scene_labels.py, spatial_description.py, dataset_export.py), `source/strafer_autonomy/`, `source/strafer_vlm/`, `source/strafer_ros/`, `source/strafer_shared/`, or the `Makefile`.
- Use the **640x360 perception camera** (`d555_camera_perception`), not the 80x60 policy camera, for all perception data collection.
- **Human gamepad teleop is the controller** for MVP data collection. Model after the existing `collect_demos.py`.
- **ProcRoom scenes are NOT useful for perception training.** Only use Infinigen scenes.
- Use Isaac Sim Replicator API for bbox extraction, NOT manual 3D-to-2D projection.
- Use conventional commits (feat:, fix:, test:, etc.). Do NOT add Co-Authored-By lines.
- Start with Task 1 (dual camera config) -- all other tasks depend on it.

## Start

Read `docs/PHASE_15_WINDOWS.md` now, then begin with Task 1: add the 640x360 `D555_PERCEPTION_CAMERA_CFG` to `source/strafer_lab/strafer_lab/assets/strafer/d555_cfg.py`.

---
