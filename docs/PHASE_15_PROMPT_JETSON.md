# Jetson Agent — Initial Prompt

Copy the text below (between the `---` markers) and paste it as the first message to Claude Code on the Jetson.

---

I'm working on a robotics project called Sim2RealLab — a Jetson Orin Nano robot with autonomous navigation powered by a remote DGX Spark running VLM and planner services over LAN HTTP.

You are the **Jetson agent** for phase_15. Your job is to implement the semantic spatial map system and integrate it into the executor and ROS client. Two other agents are working in parallel on the same branch — one on the DGX Spark (planner/VLM changes) and one on a Windows workstation (Isaac Lab simulation). You MUST stay within your assigned file boundaries to avoid merge conflicts.

## Setup

```bash
cd ~/workspaces/Sim2RealLab
git checkout phase_15
git pull origin phase_15
```

## Context files to read FIRST (in this order)

1. `docs/PHASE_15_JETSON.md` — **your task list, file ownership rules, and platform context**. This is your primary instruction set. Read it completely before writing any code.
2. `docs/INTEGRATION_JETSON.md` — full API surface of the real robot: ROS topics, TF tree, message types, client classes, schemas, data flow.
3. `docs/STRAFER_AUTONOMY_NEXT.md` — the design document with architecture details, code sketches, and data models. Reference Sections 0 (Safety), 1 (Semantic Map), 2.3 (Parallel health checks), 3.1 (Rotate skills), 3.4 (Environment query).

## Rules

- **Only modify files listed in your "Owned directories" section.** Do NOT touch `source/strafer_vlm/`, `source/strafer_autonomy/strafer_autonomy/planner/`, `source/strafer_autonomy/strafer_autonomy/clients/planner_client.py`, `source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py`, `source/strafer_lab/` (owned by DGX agent for batch processing and Isaac Sim host agent for runtime), or the `Makefile`.
- Use conventional commits (feat:, fix:, test:, etc.). Do NOT add Co-Authored-By lines.
- Run `python -m pytest source/strafer_autonomy/tests/ -m "not requires_ros" -v` after each significant change.
- Start with Task 1 (SemanticMapManager) — everything else depends on it.

## Start

Read `docs/PHASE_15_JETSON.md` now, then begin with Task 1: create the `source/strafer_autonomy/strafer_autonomy/semantic_map/` package with `models.py`, `clip_encoder.py`, and `manager.py`.

---
