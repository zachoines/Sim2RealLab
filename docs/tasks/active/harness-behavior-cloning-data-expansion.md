# Expand the harness data path for behavior-cloning-grade output

**Type:** refactor / new feature
**Owner:** DGX agent (`strafer_lab` harness + `perception_writer`
extensions; no Jetson-side code changes — the bridge already
publishes everything we need to capture)
**Priority:** P2 (foundational data-pipeline upgrade; unblocks
the v2 VLA research path and sharpens the v1 learned-validator
brief)
**Estimate:** M–L (~half-week to a week; bridge-tick-rate writer +
new fields + post-processing + docs sweep)
**Branch:** task/harness-behavior-cloning-data-expansion

## Story

As an **operator who needs the Infinigen harness to emit
behavior-cloning-grade trajectories from any of three driver
modes (bridge / teleop / oracle)**, I want **per-tick capture of
`(frame, depth, pose, commanded velocity, mission text, mission
id)` with shared timestamps, plus per-step labels (`stop_target`,
`progress`) and language augmentation, all written to a
**driver-agnostic schema**, so that **downstream training
pipelines (the v2 VLA brief, the learned-validator brief, the
clip-eval brief, the future MVP-as-teacher distillation brief)
all consume the same dataset shape regardless of whether it came
from the autonomy stack, a human teleoperator, or an in-process
oracle policy**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../context/branching-and-prs.md)
- [`context/conventions.md`](../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md)
  — the bridge mainloop is the per-tick boundary this brief hooks
  into; treat its perf table as the budget envelope.

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../MISSION_VALIDATION_ARCHITECTURE.md),
specifically §3.2 (learned validator's training-data needs) and
§3.6 (MVP-as-teacher distillation).

Sibling briefs covering the other driver modes:
- [`harness-teleop-driver`](harness-teleop-driver.md) — gamepad
  driver running in-process Isaac Lab. Reuses this brief's
  schema; ships independently.
- [`harness-oracle-driver`](harness-oracle-driver.md) — future
  in-process scripted-policy driver, blocked on teleop shipping
  + scale becoming the bottleneck.
- [`harness-mission-generator`](harness-mission-generator.md) —
  the canonical mission queue source feeding all three drivers.

Sibling brief covering multi-room defaulting:
- [`multi-room-scene-connectivity-validation`](multi-room-scene-connectivity-validation.md) —
  produces the room-connectivity graph that the bridge driver's
  multi-room missions rely on for transit-step planning.

Downstream briefs that depend on this:
- [`strafer-vla-v2-architecture`](strafer-vla-v2-architecture.md) —
  hard prerequisite (no behavior cloning without an action stream).
- [`learned-mid-mission-validator`](learned-mid-mission-validator.md) —
  benefits significantly from per-step `progress` and
  language-augmented `mission_text`, but does not strictly require
  this brief to ship first.
- A future `mvp-teacher-vla-distillation` brief.

## Context

### Three driver modes, one schema

The harness has (or will have) three driver modes, each suited to
a different data-collection regime:

| Driver | Action source | Throughput | Distribution match | Ship status |
|---|---|---|---|---|
| **`bridge`** | Jetson autonomy stack via ROS 2 (planner + executor + Nav2 + RL controller) | ~6–15 FPS, single mission at a time | Matches deployment | Current direction; this brief upgrades it |
| **`teleop`** | Human gamepad via in-process Isaac Lab, no ROS | ~30–60 FPS, single mission at a time, ~60 episodes / hr operator-paced | Matches *human-driven* deployment, which is what published wheeled VLAs train on | Filed as [`harness-teleop-driver`](harness-teleop-driver.md) |
| **`oracle`** *(future)* | Scripted policy in-process (A* on navigable mask + heuristic stop) | ~Hundreds of FPS, parallel envs | Synthetic; intended for scale supplements | Filed as [`harness-oracle-driver`](harness-oracle-driver.md) (sketch); not picked up until teleop throughput is the bottleneck |

**This brief ships the schema + the bridge driver upgrades.** The
teleop and oracle drivers are sibling briefs that emit the same
schema. Downstream consumers (clip-eval, learned-validator, v2
VLA) are agnostic to which driver produced the data.

### What the bridge harness emits today

Per [`run_sim_in_the_loop.py --mode harness`](../../../source/strafer_lab/scripts/run_sim_in_the_loop.py)
+ [`tools/perception_writer.py`](../../../source/strafer_lab/strafer_lab/tools/perception_writer.py),
each mission produces:

```
data/sim_in_the_loop/<scene_name>/episode_NNNN/
├── frames.jsonl       # one record per skill-call capture
├── frame_*.jpg        # RGB JPEGs aligned to frames.jsonl rows
└── (no depth)
```

`frames.jsonl` schema today:

```
{frame_id, image_path, scene_name, scene_type, robot_pos,
 robot_quat, bboxes, mission_id, target_label,
 target_position_3d, reachability, mission_state}
```

The capture rate is **per-skill-call**, not per-bridge-tick — when
`scan_for_target` rotates and grounds, it produces a few frames;
when `navigate_to_pose` runs Nav2 for 30 s, it produces zero
intermediate frames. This is sufficient for VLM SFT (Stage 6 of
[`INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md))
but fundamentally unsuitable for behavior cloning.

### What behavior cloning needs

A VLA / progress-monitor / temporal-head validator wants
`(observation_t, action_t)` pairs at a regular cadence over the
**entire** mission, with shared timestamps and per-step labels
sufficient to score the model offline. Six gaps to close:

1. **Action stream.** Capture `/cmd_vel` at the bridge tick rate.
   Distinguish *commanded* (the v1 stack's intent — what the VLA
   learns to emit) from *achieved* (`/odom`-derived — what the
   chassis actually did). Most published VLAs train on commanded;
   match that convention but emit both for analysis.
2. **Time alignment.** Use `/clock` (sim time) as the canonical
   timestamp on every record. The bridge already publishes
   `/clock` at 50 Hz. Frames, depth, pose, and `cmd_vel` records
   all carry the same timestamp source so downstream training
   doesn't have to interpolate.
3. **Per-tick capture + depth.** Capture every bridge tick
   (~125 ms headless w/ cameras → ~8 Hz). Capture aligned depth
   (`/d555/depth/image_rect_raw`, 16UC1 mm) as PNG alongside the
   RGB JPEG.
4. **Mission-text augmentation.** Use the existing 7B Qwen2.5-VL
   in [`generate_descriptions.py`](../../../source/strafer_lab/scripts/generate_descriptions.py)
   to paraphrase each harness mission in N styles. Keep the
   original; emit augmentations alongside.
5. **Per-step labels.** Two new fields per per-tick record:
   - `stop_target`: True for the last K ticks of a
     `mission_state=succeeded` mission, False elsewhere.
   - `progress`: `[0, 1]` scalar = `1 − geodesic(robot_pos,
     target_position_3d) / leg_initial_distance`. Geodesic
     distance from A* on the costmap — computable post-hoc on the
     DGX, doesn't need to be online.
6. **Hindsight relabeling pass (optional, post-processing).** A
   trajectory that ended near object B (when object A was the
   goal) is a valid demonstration of "go to B" if you relabel the
   mission text. Compute the nearest scene-metadata object to the
   final pose; emit a parallel `frames.jsonl.hindsight` with the
   relabeled mission.

Items 1–3 are **strict prerequisites** for any behavior-cloning
work. Items 4–6 are recommended; the brief ships all six but
gates the strict ones in acceptance. The brief also adds a
**hard-negative injection flag**
(`--inject-bad-grounding {wrong_room, wrong_instance, off}`)
under the recommended tier — both the CLIP-eval brief and the
learned-validator brief consume it, so it lives here rather than
in either downstream brief.

### Why custom over Isaac Lab's native BC frameworks

Isaac Lab and Isaac Sim have opinionated behavior-cloning support
— `omni.replicator` for synthetic-data rendering,
`isaaclab.envs.ManagerBasedRLEnv` for in-process RL trajectories,
**Isaac Lab Mimic** for HDF5 demo capture (released 2024,
manipulation-targeted, used by GR00T training pipelines), and
`omni.isaac.core.utils.data_logger` as the low-level write loop.
Each one assumes the canonical pattern: **single process,
in-memory, env stepping at policy rate, action source as a Python
object inside the same process as the env.**

Strafer breaks every assumption because the bridge harness is
intentionally cross-host: the Jetson autonomy stack on
`192.168.50.24` drives Isaac Sim on `192.168.50.196` over ROS 2
DDS, and `/clock` is the only authority both sides see. This is
how strafer preserves sim-real-distribution-match — the same
Jetson stack that runs against real sensors also runs against
simulated ones — but it puts the action source on a different
machine than the env, which Mimic's recorder API cannot represent.

What we reuse: `omni.replicator` for depth + bbox annotation
(already used indirectly via the perception camera); MLflow
tracking; Mimic's HDF5 *output schema* as a downstream-pipeline
target (see the recommended-tier export below). What we do not
reuse: Mimic's recorder API itself, since it cannot be hooked to
a ROS-driven action source on another host without writing more
glue than re-implementing the recorder.

Adjacent prior art (Habitat-Lab BC, AI2-THOR + ALFRED, gym-gazebo)
either uses a different simulator or doesn't carry the cross-host
ROS pattern. The combination "Isaac Sim + cross-host ROS +
autonomy-stack-as-demonstrator + mecanum holonomic +
real-robot-mirroring topic names" is novel-ish at the
system-assembly level even though no individual piece is hard —
expect to write the recorder, not import it.

The road not taken (named so a future reader can see it was
considered): collapse the harness into a parallel in-process
Isaac Lab env, with the autonomy stack imported as Python and
no ROS. That unlocks Mimic + 100× parallel rollout throughput,
but it requires re-implementing the planner / executor / Nav2 /
RTAB as in-process Python and produces demos with a sim-real
distribution mismatch (no LAN latency, no DDS jitter, no
message-time interpolation). The trade goes the other way for
strafer; document, don't pursue.

### Output shape

Replace the current `frames.jsonl` with a richer split. Backward
compatibility for existing consumers (the description pipeline,
VLM SFT prep) is preserved by keeping a `frames_skill.jsonl`
alongside the new per-tick records:

```
data/sim_in_the_loop/<scene_name>/episode_NNNN/
├── frames_skill.jsonl       # legacy schema, per-skill-call (existing consumers)
├── frames_tick.jsonl        # NEW: per-bridge-tick records
├── actions.jsonl            # NEW: per-tick (timestamp, vx, vy, omega_z, source)
├── mission.json             # NEW: mission_text + paraphrases + final outcome
├── frame_*.jpg              # RGB JPEGs (now denser)
├── depth_*.png              # NEW: 16UC1 PNGs
└── progress.jsonl           # NEW (post-processing pass): per-tick progress + stop_target
```

`frames_tick.jsonl` schema:

```
{tick_id, sim_time, image_path, depth_path, robot_pos, robot_quat,
 mission_id, last_cmd_vel: {vx, vy, omega_z}, last_skill_id}
```

`actions.jsonl` schema:

```
{sim_time, vx_cmd, vy_cmd, omega_z_cmd, vx_achieved, vy_achieved, omega_z_achieved}
```

`mission.json` schema:

```
{mission_id, target_label, target_position_3d,
 mission_text_original,
 mission_text_paraphrases: [...],
 reachability, mission_state, leg_initial_distance, scene_seed}
```

### Data volume sanity

Multi-room missions are longer than single-room (transit + scan +
navigate per target). Realistic episode length: ~30 s for
single-room, ~60–90 s for cross-room. Volume math at the
high-end (90 s × 8 Hz = ~720 ticks per mission):

| Stream | Per frame | Per mission (90 s × 8 Hz = ~720 frames) |
|---|---|---|
| RGB JPEG @ q=85 | ~40 KB | ~28 MB |
| Depth PNG (16UC1) | ~150 KB | ~108 MB |
| `frames_tick.jsonl` row | ~0.5 KB | ~0.4 MB |
| `actions.jsonl` row | ~0.1 KB | ~0.08 MB |
| `progress.jsonl` row (post-proc) | ~0.1 KB | ~0.08 MB |
| **Total** | | **~140 MB / mission** (cross-room worst case) |

30 missions × 3 scenes × 140 MB ≈ 12 GB. Tractable on the DGX.
Single-room missions remain at ~46 MB / mission. The existing
`data/` tree is gitignored, so commit footprint is unchanged.

### Why this lives outside `next-integration-round`

[`next-integration-round`](next-integration-round.md) validates
that the v1 stack composes end-to-end against
[`INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md).
That brief's job is to *exercise* the pipeline at its current
fidelity, not to upgrade it. This brief upgrades the pipeline so
that future runs of `next-integration-round` and downstream
training briefs all benefit from the richer output. Both can
land in either order; this brief is more useful first if the
operator's near-term work is VLA-flavored, and `next-integration-round`
is more useful first if the priority is shaking out integration
bugs.

### Why the teleop driver is filed separately

The bridge-driver upgrades and the teleop driver share the
schema but live in different code paths
(`run_sim_in_the_loop.py`'s bridge mainloop vs. a new in-process
Isaac Lab entry point). Filing them separately keeps each brief
scoped — the bridge changes are mostly writer plumbing inside
the existing harness; the teleop driver is a new entry point
with its own UX surface (gamepad mapping, mission queue,
episode buttons). Both can ship in either order; if teleop ships
first, the schema this brief defines is the *contract* that
teleop emits against; if this brief ships first, teleop is a
straightforward second consumer.

## Acceptance criteria

Strict prerequisites (items 1–3) are gated; items 4–6 are
gated separately so the brief can ship the foundation even if the
post-processing passes need iteration.

### Strict (must ship)

- [ ] **Per-tick capture loop (bridge driver).**
      `run_sim_in_the_loop.py --mode harness` writes one
      `frames_tick.jsonl` row per bridge tick while a mission is
      active, with the bridge's `/clock` timestamp as `sim_time`.
      RGB JPEG and depth PNG written alongside. (The teleop
      driver's per-tick capture is in-process Isaac Lab; lives
      in [`harness-teleop-driver`](harness-teleop-driver.md);
      this brief covers the bridge path only.)
- [ ] **Action stream.** `actions.jsonl` written with
      `(sim_time, vx_cmd, vy_cmd, omega_z_cmd, vx_achieved,
      vy_achieved, omega_z_achieved)` rows at the bridge tick
      rate. Source for `_cmd` fields is the executor-published
      `/cmd_vel`; source for `_achieved` is `/odom`-derived.
- [ ] **Time alignment.** Every `frames_tick.jsonl` row's
      `sim_time` matches the corresponding `actions.jsonl` row
      within one bridge tick. Smoke-tested with a unit / golden
      test that loads both files and checks alignment.
- [ ] **Backward compatibility.** Existing consumers
      (`generate_descriptions.py`,
      `prepare_vlm_finetune_data.py`,
      `dataset_export.py`) consume `frames_skill.jsonl`
      unchanged. The legacy file emits the same schema as
      today's `frames.jsonl`.
- [ ] **`mission.json`** written per episode with the original
      mission text, the paraphrase list (empty if item 4 not
      shipped), final outcome, and `leg_initial_distance` for
      progress computation.
- [ ] **Smoke test.** A single-mission run on
      `scene_fast_singleroom_000_seed0` produces a populated
      `episode_NNNN/` directory with all strict files non-empty.
      Captured as a one-line `ls -la` excerpt in the PR
      description.

### Recommended (should ship; gated separately)

- [ ] **Mission-text paraphrase generator.** A new harness flag
      `--paraphrase-missions N` runs the 7B Qwen2.5-VL on each
      mission's `(target_label, scene_name)` and emits N
      paraphrases into `mission.json`. Default N=0 (off).
- [ ] **Per-step labels post-processor.**
      `source/strafer_lab/strafer_lab/tools/compute_progress.py`
      consumes a finished episode and emits `progress.jsonl` with
      `(sim_time, progress, stop_target)` rows. Geodesic distance
      via A* on the local costmap; A* implementation can reuse
      Nav2's planner output if available, or use a simple grid
      A* on the scene's navigable mask.
- [ ] **Hindsight relabel pass.**
      `source/strafer_lab/strafer_lab/tools/relabel_hindsight.py`
      consumes a finished episode and emits a parallel
      `frames_tick.jsonl.hindsight` with the mission text
      relabeled to the actual final-pose nearest object (drawn
      from `scene_metadata.json`). Conservative: only relabel
      when the final pose is within 1.0 m of a uniquely-named
      scene object; otherwise mark `hindsight=ambiguous` and
      skip.
- [ ] **`--inject-bad-grounding` flag for hard-negative
      generation.**
      [`run_sim_in_the_loop.py`](../../../source/strafer_lab/scripts/run_sim_in_the_loop.py)
      accepts `--inject-bad-grounding {wrong_room,wrong_instance,off}`
      (default `off`). When enabled, each harness mission has a
      configurable probability (`--inject-bad-grounding-prob`,
      default 0.3) of having its `target_position_3d` perturbed
      *after* the executor projects the goal:
  - `wrong_room` swaps the goal to a randomly-selected object
    in a different room polygon (drawn from `scene_metadata.json`'s
    `rooms[]`).
  - `wrong_instance` swaps the goal to another scene-metadata
    object with the same `target_label` if one exists in the
    same room; falls back to `wrong_room` if no same-label
    sibling exists.
  Perturbed missions are tagged in `mission.json` with
  `injection_mode` and `original_target_position_3d` so
  downstream consumers (the learned-validator brief, the
  CLIP-eval brief's `--root-cause-pass`) can label them as
  hard negatives without re-deriving the perturbation. **This
  flag is consumed by both
  [`clip-mid-mission-validator-evaluation`](clip-mid-mission-validator-evaluation.md)
  and
  [`learned-mid-mission-validator`](learned-mid-mission-validator.md);
  this brief owns its definition.**
- [ ] **HDF5 / RLDS export converter for downstream VLA pipelines.**
      `source/strafer_lab/strafer_lab/tools/export_bc_dataset.py`
      consumes a finished episode (or a directory of episodes)
      and emits one of two formats, selected by `--format`:
  - `--format mimic-hdf5`: per-mission HDF5 conforming to the
    Isaac Lab Mimic schema (one HDF5 per scene, one episode-group
    per mission, with `obs/`, `actions/`, and `meta/` subgroups).
    Lets downstream consumers reuse Mimic's dataset inspectors,
    replay tools, and trajectory visualizers.
  - `--format rlds`: TFDS-style RLDS shards consumable by the
    Octo training pipeline directly.
  The strict-tier JSONL output stays the source of truth — the
  HDF5 / RLDS files are derived artifacts, regenerable from
  JSONL, not committed. Document which downstream consumer each
  format targets in the export script's help text.

### Doc surfaces

- [ ] [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stage 5 / Stage 6 sections updated to describe the new
      output tree.
- [ ] [`source/strafer_lab/README.md`](../../../source/strafer_lab/README.md)
      "Scripts and tools inventory" gains entries for
      `compute_progress.py`, `relabel_hindsight.py`, and
      `export_bc_dataset.py`; the "Contracts" section names the
      new file layout.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in
      [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../INTEGRATION_SIM_IN_THE_LOOP.md)
      Stage 5 + Stage 6 — the existing description pipeline,
      VLM SFT prep, and CLIP CSV export must all continue to
      work against `frames_skill.jsonl` without code changes.

## Investigation pointers

- The bridge mainloop's tick boundary is in
  [`run_sim_in_the_loop.py`](../../../source/strafer_lab/scripts/run_sim_in_the_loop.py);
  the perception writer is in
  [`source/strafer_lab/strafer_lab/tools/perception_writer.py`](../../../source/strafer_lab/strafer_lab/tools/perception_writer.py).
  These are the two files most of the work touches.
- The bridge already publishes `/clock` at 50 Hz; the harness
  process can read it via the existing rclpy spin path. Confirm
  the publisher rate against
  [`bridge-runtime-invariants.md`](../context/bridge-runtime-invariants.md#telemetry-vs-cameras-split-publishers).
- Camera resolution: 640×360 perception cam,
  [`test_d555_perception_cfg.py:50`](../../../source/strafer_lab/test/sensors/test_d555_perception_cfg.py#L50).
  Depth is 16UC1 millimeters per
  [`depth_downsampler.py:3-7`](../../../source/strafer_ros/strafer_perception/strafer_perception/depth_downsampler.py#L3-L7).
- The 7B Qwen2.5-VL paraphrase generator already exists in
  [`generate_descriptions.py`](../../../source/strafer_lab/scripts/generate_descriptions.py)
  Stage 2 — reuse the prompt-template + model-loading scaffold;
  the paraphrase prompt is the differing piece.
- Scene metadata for hindsight relabeling lives in
  [`scene_metadata.json`](../../../Assets/generated/scenes/) per
  scene; `objects[]` carries label + position. Use the same
  loader the harness already uses.
- For geodesic distance: Nav2's global costmap publishes the
  occupancy that A* runs on. A* fallback can use the scene's
  navigable mask from Infinigen metadata.

## Out of scope

- **Real-robot data capture.** Sim-side only. A future brief may
  layer real-robot output once the sim pipeline is calibrated.
- **Multi-camera capture.** Strafer has one D555 forward-facing
  camera; multi-cam is a future hardware change, not a harness
  change.
- **Domain randomization sweeps** (per-episode lighting / texture
  / start-pose perturbation). Filed separately under sim-to-real
  tuning work; does not block VLA sim-eval.
- **Path-shape mission language** ("hug the wall," "via the
  dining room"). The paraphrase generator emits endpoint-shaped
  variations only — Family-3 / case-3 work is a separate brief
  if pursued.
- **Replacing the existing per-skill capture path.** The legacy
  `frames_skill.jsonl` ships unchanged for backward compatibility
  with the description / VLM SFT pipelines.
- **Compressing or sharding the output for cloud-scale training.**
  Local DGX storage is sufficient for the data volumes this
  brief produces. WebDataset / cloud-bucket packing is a future
  brief if needed. (Single-host RLDS / Mimic-HDF5 conversion
  *is* in scope as a recommended-tier item — see acceptance.)
- **Action-tokenization decisions.** Belongs to the v2 VLA brief
  ([`strafer-vla-v2-architecture`](strafer-vla-v2-architecture.md)),
  not here. This brief produces continuous-valued action records;
  whether the VLA quantizes them into 256 bins or feeds a
  diffusion head is a downstream choice.
