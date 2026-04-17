# Prompt: Documentation audit + master end-to-end flow diagrams

You are a software-documentation agent working on `Sim2RealLab` (branch
`phase_15`). Your task has two parts:

1. **Audit the existing `docs/` tree** and propose a cleanup plan
   (which docs are outdated, which overlap, which to keep).
2. **Author a master "system flows" reference** under
   `docs/SYSTEM_FLOW_DIAGRAMS.md` containing six Mermaid diagrams, one
   per named information-flow path, with hyperlinks from diagram nodes
   to the actual classes / scripts in `source/`.

Both parts must be **non-destructive**: do NOT delete or move any
existing file without an explicit user approval round. Propose, wait,
then act.

---

## Project context (no prior conversation, here is what you need)

**Repo:** `/home/zachoines/Workspace/Sim2RealLab`. Branch `phase_15`.
Three parallel agent workstreams own distinct file trees:

- **Jetson Orin Nano agent** owns the executor and ROS bringup —
  `source/strafer_autonomy/strafer_autonomy/executor/`,
  `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`,
  all of `source/strafer_ros/`.
- **DGX Spark agent** owns VLM + planner services + Isaac Sim host
  scope — `source/strafer_vlm/`, `source/strafer_autonomy/strafer_autonomy/planner/`,
  `source/strafer_autonomy/strafer_autonomy/clients/{planner,vlm}_client.py`,
  all of `source/strafer_lab/`, plus
  `source/strafer_autonomy/strafer_autonomy/semantic_map/` and the
  agent runtime in `source/strafer_autonomy/strafer_autonomy/cli.py`.
- **Hardware sits across two physical machines:**
  - DGX Spark (aarch64) — 128 GB unified memory, hosts Isaac Sim + VLM
    + planner services.
  - Jetson Orin Nano — onboard the robot, runs the executor, Nav2,
    RTAB-Map SLAM, the RealSense D555 driver, RoboClaw motor drivers.

**Cross-host transport:** `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` +
`ROS_DOMAIN_ID=42` over LAN (DGX `192.168.50.196`,
Jetson `192.168.50.24`). HTTP for VLM / planner service calls
(ports 8100 and 8200 on the DGX).

**Real robot vs sim split:** the same Jetson autonomy stack consumes
the same ROS2 topics whether they come from the real D555 + RoboClaws
or from the Isaac Sim ROS2 bridge running on the DGX. The bridge is
implemented in
[strafer_lab.bridge](../source/strafer_lab/strafer_lab/bridge/__init__.py)
and the sim-in-the-loop harness is at
[strafer_lab.sim_in_the_loop](../source/strafer_lab/strafer_lab/sim_in_the_loop/__init__.py).

---

## Existing `docs/` tree (run `ls docs/` to verify current state)

Living design docs (definitely keep, possibly trim):

- `STRAFER_AUTONOMY_SYSTEMS_OVERVIEW.md` — has Mermaid diagrams of the
  static system roles. Your new diagrams complement this; consider
  whether the overview's diagrams should link out to your flow diagrams.
- `STRAFER_AUTONOMY_NEXT.md` — large (4225 lines) design master.
  Ground truth for current intent. Do NOT modify; reference only.
- `STRAFER_AUTONOMY_ROADMAP.md`, `STRAFER_AUTONOMY_INTERFACES.md`,
  `STRAFER_AUTONOMY_COMMAND_INGRESS.md`,
  `STRAFER_AUTONOMY_LLM_PLANNER.md`,
  `STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md`,
  `STRAFER_AUTONOMY_ROS.md`, `STRAFER_AUTONOMY_VLM_GROUNDING.md` —
  per-subsystem design docs.
- `SIM_TO_REAL_PLAN.md`, `SIM_TO_REAL_TUNING_GUIDE.md` — RL training
  tuning + sim-to-real strategy.
- `INTEGRATION_DGX_SPARK.md`, `INTEGRATION_JETSON.md` — host setup.
- `WIRING_GUIDE.md`, `D555_IMU_KERNEL_FIX.md` — hardware-side notes.
- `VALIDATE_ISAAC_SIM_AND_INFINIGEN.md` — install validation runbook
  (recently authored, current).

Likely outdated / candidates for archival:

- `PHASE_15_DGX.md`, `PHASE_15_JETSON.md`, `PHASE_15_WINDOWS.md` —
  per-agent task lists for the in-progress phase. Most of the listed
  tasks have shipped; surviving content is mostly historical.
- `PHASE_15_PROMPT_DGX.md`, `PHASE_15_PROMPT_JETSON.md`,
  `PHASE_15_PROMPT_WINDOWS.md` — the agent-launch prompts that
  produced the work in the per-agent task lists. Already-delivered
  briefs.

For the audit, **propose** which of those eight `PHASE_15_*` files
should be deleted vs. moved to a `docs/archive/` subdirectory vs.
kept in place. Do not act on the proposal until the user approves.

---

## Source-tree pointers for hyperlinks (verify each path exists before linking)

If a path below has changed since this prompt was written, locate the
new path with `Glob` / `Grep` rather than fabricating one.

**VLM service + training**
- Service: [source/strafer_vlm/strafer_vlm/service/app.py](../source/strafer_vlm/strafer_vlm/service/app.py)
- Inference runtime: [source/strafer_vlm/strafer_vlm/inference/](../source/strafer_vlm/strafer_vlm/inference/)
- Training: [source/strafer_vlm/strafer_vlm/training/](../source/strafer_vlm/strafer_vlm/training/)
- VLM SFT data prep: [source/strafer_lab/scripts/prepare_vlm_finetune_data.py](../source/strafer_lab/scripts/prepare_vlm_finetune_data.py)
- Description pipeline: [source/strafer_lab/scripts/generate_descriptions.py](../source/strafer_lab/scripts/generate_descriptions.py)

**Planner service + agent**
- Planner service: [source/strafer_autonomy/strafer_autonomy/planner/app.py](../source/strafer_autonomy/strafer_autonomy/planner/app.py)
- Plan compiler: [source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py](../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py)
- Intent parser: [source/strafer_autonomy/strafer_autonomy/planner/intent_parser.py](../source/strafer_autonomy/strafer_autonomy/planner/intent_parser.py)
- Operator CLI: [source/strafer_autonomy/strafer_autonomy/cli.py](../source/strafer_autonomy/strafer_autonomy/cli.py)

**Executor (Jetson)**
- Mission runner: [source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
- Command server: [source/strafer_autonomy/strafer_autonomy/executor/command_server.py](../source/strafer_autonomy/strafer_autonomy/executor/command_server.py)
- ROS client: [source/strafer_autonomy/strafer_autonomy/clients/ros_client.py](../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
- Semantic map: [source/strafer_autonomy/strafer_autonomy/semantic_map/](../source/strafer_autonomy/strafer_autonomy/semantic_map/)

**Strafer Lab (Isaac Sim training + data + bridge)**
- Tasks / envs: [source/strafer_lab/strafer_lab/tasks/](../source/strafer_lab/strafer_lab/tasks/)
- Training script: [Scripts/train_strafer_navigation.py](../Scripts/train_strafer_navigation.py)
- Demo collection: [source/strafer_lab/scripts/collect_demos.py](../source/strafer_lab/scripts/collect_demos.py)
- Perception teleop collection: [source/strafer_lab/scripts/collect_perception_data.py](../source/strafer_lab/scripts/collect_perception_data.py)
- Replicator bbox extractor: [source/strafer_lab/strafer_lab/tools/bbox_extractor.py](../source/strafer_lab/strafer_lab/tools/bbox_extractor.py)
- Perception writer: [source/strafer_lab/strafer_lab/tools/perception_writer.py](../source/strafer_lab/strafer_lab/tools/perception_writer.py)
- ROS2 bridge config: [source/strafer_lab/strafer_lab/bridge/config.py](../source/strafer_lab/strafer_lab/bridge/config.py)
- ROS2 bridge OmniGraph builder: [source/strafer_lab/strafer_lab/bridge/graph.py](../source/strafer_lab/strafer_lab/bridge/graph.py)
- Sim-in-the-loop harness: [source/strafer_lab/strafer_lab/sim_in_the_loop/harness.py](../source/strafer_lab/strafer_lab/sim_in_the_loop/harness.py)
- Sim-in-the-loop env adapter: [source/strafer_lab/strafer_lab/sim_in_the_loop/runtime_env.py](../source/strafer_lab/strafer_lab/sim_in_the_loop/runtime_env.py)
- Sim-in-the-loop ROS adapter: [source/strafer_lab/strafer_lab/sim_in_the_loop/runtime_mission.py](../source/strafer_lab/strafer_lab/sim_in_the_loop/runtime_mission.py)
- Mission generator: [source/strafer_lab/strafer_lab/sim_in_the_loop/mission.py](../source/strafer_lab/strafer_lab/sim_in_the_loop/mission.py)
- Sim-in-the-loop launcher: [source/strafer_lab/scripts/run_sim_in_the_loop.py](../source/strafer_lab/scripts/run_sim_in_the_loop.py)
- Scene metadata extractor: [source/strafer_lab/scripts/extract_scene_metadata.py](../source/strafer_lab/scripts/extract_scene_metadata.py)
- Infinigen label parser: [source/strafer_lab/strafer_lab/tools/infinigen_label_parser.py](../source/strafer_lab/strafer_lab/tools/infinigen_label_parser.py)

**Strafer ROS (Jetson runtime)**
- Bringup launches: [source/strafer_ros/strafer_bringup/launch/](../source/strafer_ros/strafer_bringup/launch/)
- Sim-in-the-loop launch: [source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py](../source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py)
- Driver (RoboClaw): [source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py](../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py)
- Perception: [source/strafer_ros/strafer_perception/](../source/strafer_ros/strafer_perception/)
- SLAM: [source/strafer_ros/strafer_slam/](../source/strafer_ros/strafer_slam/)
- Navigation: [source/strafer_ros/strafer_navigation/](../source/strafer_ros/strafer_navigation/)

**Shared constants (single source of truth)**
- [source/strafer_shared/strafer_shared/constants.py](../source/strafer_shared/strafer_shared/constants.py)

---

## Six flows to diagram

For each flow below, produce ONE Mermaid diagram in
`docs/SYSTEM_FLOW_DIAGRAMS.md` with:

- **Top of diagram:** the operator command (`bash $ ...`) that kicks
  the flow off, including the host the command runs on.
- **Vertical or horizontal information flow** — pick whichever reads
  best per diagram. Bias toward `flowchart TB` or `flowchart LR`.
- **Cross-host boundaries clearly marked** — use Mermaid `subgraph`
  blocks labelled `DGX Spark`, `Jetson Orin Nano`, `Real robot
  hardware`, etc.
- **Process / file / topic distinction** — use distinct node shapes
  (rectangles for processes, parallelograms or rounded rects for
  data files, hexagons for ROS topics — pick a convention and stick
  to it consistently across all six diagrams; document the legend at
  the top of the file).
- **Hyperlinks** on every node that maps to a real file/class. Mermaid
  syntax: `click NodeId "../source/.../file.py" "Tooltip text"`. GitHub
  renders these as clickable when viewed online.
- **End state** — what artifact the flow produces (e.g. "model
  checkpoint at `models/clip_v3.pt`").

The six flows the user named:

1. **VLM data gathering and training flow** — from teleop perception
   data through `generate_descriptions.py` (multi-stage description
   pipeline using the VLM service for grounding), then
   `prepare_vlm_finetune_data.py`, then a training run under
   `source/strafer_vlm/strafer_vlm/training/`. End: fine-tuned
   Qwen2.5-VL checkpoint.

2. **CLIP data gathering and training flow** — same upstream perception
   data, but flowing through `dataset_export.py` (CLIP CSV variant)
   and `finetune_clip.py`. End: CLIP checkpoint used by the executor's
   semantic map for image-text retrieval.

3. **Strafer model training flow (Isaac Lab RL)** — `Scripts/train_strafer_navigation.py`
   running PPO under `strafer_lab.tasks.navigation` against ProcRoom /
   Infinigen envs. End: PPO checkpoint under `logs/rsl_rl/strafer_navigation/`.
   Show the optional `--video` recording branch.

4. **Perception data gathering — teleop** — operator on the DGX runs
   `collect_perception_data.py` with a gamepad, drives the robot
   through an Infinigen scene, frames land in `data/perception/episode_NNNN/`.
   Show the `ReplicatorBboxExtractor` + `PerceptionFrameWriter` data
   flow.

5. **Perception data gathering — sim-in-the-loop bridge** — the
   harness path: `run_sim_in_the_loop.py --mode harness` on DGX
   submits missions over LAN to the Jetson executor, which runs Nav2
   against the simulated D555 streams the DGX bridge publishes,
   `/cmd_vel` flows back to drive the sim, frames land in
   `data/sim_in_the_loop/episode_NNNN/` with reachability labels.
   This is the most complex diagram — it spans both hosts, has a
   feedback loop, and crosses both ROS and HTTP boundaries.

6. **End-to-end real-robot execution** — operator submits a mission
   via `strafer-autonomy-cli submit "..."`, the executor runs the
   plan against the live D555 + RoboClaw + Nav2 stack on the Jetson,
   the planner / VLM services on the DGX provide intent parsing and
   grounding over LAN HTTP. End: physical robot motion. This is the
   "production" flow that all the other flows feed into.

For each flow, also write 2-4 sentences ABOVE the diagram describing
**when this flow runs** (manual? batch? continuous?), **who triggers
it** (operator? CI? cron?), and **what it produces or consumes**.

---

## Output file structure

`docs/SYSTEM_FLOW_DIAGRAMS.md` — one file, with this structure:

```markdown
# Sim2RealLab system flow diagrams

[Top-of-page index linking to each flow section]

## Diagram legend
[Node shape conventions, host subgraph colors, link types]

## Flow 1 — VLM data gathering and training
[Description paragraph]
[Mermaid diagram]
[Notes / caveats]

## Flow 2 — CLIP data gathering and training
...

[etc. for all 6 flows]

## Cross-flow data dependency map
[Optional: a small Mermaid graph showing which flows produce inputs
to which other flows — e.g. teleop perception → both VLM and CLIP
training flows; sim-in-the-loop → both VLM training data and
reachability-labelled CLIP data.]
```

---

## Doc-audit deliverable

Create `docs/_audit_proposal.md` (gitignored — not a permanent doc,
just a working artifact for the user to review). Format:

```markdown
# Doc audit proposal

## Files to keep as-is
- `docs/STRAFER_AUTONOMY_SYSTEMS_OVERVIEW.md` — reason
- ...

## Files to retitle / consolidate
- `docs/A.md` + `docs/B.md` → propose merging into `docs/X.md` because ...

## Files to archive (move to `docs/archive/`)
- `docs/PHASE_15_*.md` — reason: tasks complete, content is historical
- ...

## Files to delete
- `docs/X.md` — reason: superseded by Y, no incoming links

## Proposed new docs
- `docs/SYSTEM_FLOW_DIAGRAMS.md` — the master flow reference
- (anything else you think is missing)
```

After the user approves the proposal, do the moves / deletes in a
**separate commit** from the new diagrams doc so each is independently
revertable.

---

## Constraints

- **No transient documentation references in code or new docs.** Banned
  phrases: `Task N`, `DGX Task N`, `Phase 15`, `phase_15`,
  `Phase A1` / `Phase B4` / etc., `Section 5.5.4`, specific commit
  hashes (`commit b529e69`), branch names. The diagrams + audit
  document **what the system does**, not which task tracked the work
  or which design-doc section it cites.
- Use the documented file ownership boundaries when proposing edits.
  In particular: the DGX agent has APPEND-ONLY ownership of
  `source/strafer_shared/strafer_shared/constants.py` and zero ability
  to modify any other `strafer_shared/` file. The Jetson agent owns
  the executor + ROS bringup. Do not propose edits across these lines
  without flagging them as cross-team requests.
- Diagrams should reflect **the system as it actually exists today** —
  before drawing a flow, verify the listed scripts and modules are
  present and that their imports / function names match. Do not
  diagram aspirational flows.
- For Mermaid `click` links, use **relative paths** from the doc's
  location (i.e. `../source/...`). They render as clickable links on
  GitHub.
- Keep diagrams readable. If a flow has more than ~25 nodes, split
  it: top-level overview diagram + drill-down sub-diagrams.

---

## Suggested approach

1. Read every doc in `docs/` (skim long ones — 4000+ lines is OK to
   spot-check) to build the audit picture.
2. Read `__init__.py` of each `source/strafer_*/` package + every
   script entry point to confirm what flows actually exist.
3. Write `_audit_proposal.md` and stop. Wait for user approval.
4. Once audit is approved, write `SYSTEM_FLOW_DIAGRAMS.md` in one
   sitting and verify all `click` links resolve to real files.
5. Render check: install or ask the user to confirm the GitHub render
   looks correct (Mermaid in a Markdown preview, or push a draft
   branch and check the GitHub UI).

Report back when each step is complete; do not chain steps without a
checkpoint.
