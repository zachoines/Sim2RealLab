# Harness data-capture architecture

**Type:** architecture / new feature (consolidates 5 prior briefs)
**Owner:** DGX agent (in-process Isaac Lab + writer + bridge integration; no Jetson code)
**Priority:** P1 (foundational — every downstream training brief consumes this)
**Estimate:** XL — split across implementation PRs (see [Implementation tiers](#implementation-tiers))
**Branch:** see [Implementation tiers](#implementation-tiers); this brief is docs-only

## Status

This brief is the architectural spec for the harness data-capture system. It supersedes and replaces:

- [`completed/behavior-cloning-data-expansion.md`](../../completed/behavior-cloning-data-expansion.md) — schema-expansion plan; retired in favor of LeRobot v3 from the first capture
- [`completed/teleop-driver.md`](../../completed/teleop-driver.md) — folded in as [Driver: teleop](#driver-teleop)
- [`completed/trajectory-first-captioning.md`](../../completed/trajectory-first-captioning.md) — folded in as [Mission source: captioner](#mission-source-captioner)
- [`completed/oracle-driver.md`](../../completed/oracle-driver.md) — folded in as the scripted driver under [Driver: scripted](#driver-scripted)
- [`completed/output-format-alignment.md`](../../completed/output-format-alignment.md) — format decision (LeRobot v3) folded in as [Output format](#output-format--lerobot-v3)

The four-PR audit that motivated this consolidation lives in conversation history (2026-05-24); the short version is in [Why this consolidation](#why-this-consolidation).

## Story

As a **DGX operator who needs the harness to produce one canonical training corpus consumable by every downstream consumer** (CLIP fine-tuning, VLM SFT, behavior cloning for VLAs, retrieval-augmented validators, room-state eval, future GR00T / π0 / OpenVLA fine-tunes), I want **one entry point (`Scripts/capture.py`), one on-disk schema (LeRobot v3), and a clear cross-product of action sources × mission sources**, so that **the project's data-collection surface is reasonable for a single operator to operate, the format survives the 2026 wheeled-VLA training ecosystem, and the harness is not a code-archaeology exercise the next time we need to reason about it**.

## Context bundle

Read these before starting any implementation PR:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md) — the bridge mainloop is one of the three drivers; treat its perf table as the budget envelope

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md) §3.6 — the data-path options for VLA training; this brief is their unification.

## Why this consolidation

Before this brief: 4 active + 1 parked driver briefs (`behavior-cloning-data-expansion`, `teleop-driver`, `trajectory-first-captioning`, `oracle-driver` parked, `output-format-alignment` parked) each described one slice of the same system. The shapes overlapped. Three problems surfaced in the 2026-05-24 audit:

1. **Theoretical back-compat.** None of the current data-capture entry points (`run_sim_in_the_loop.py --mode harness`, `collect_perception_data.py`) have ever been run against real data. The "preserve `frames.jsonl` for legacy consumers" plan in the original BC-expansion brief was protecting code that has never produced production data. The downstream consumers (`generate_descriptions.py`, `prepare_vlm_finetune_data.py`, `dataset_export.py`, `finetune_clip.py`) have also never run against real data. The migration cost of changing format is genuinely zero today; later it compounds.

2. **Format alignment was deferred-on-trigger** even though every downstream consumer in 2026 (GR00T, π0/openpi, OpenVLA, Octo) consumes either LeRobot v2/v3 or RLDS natively, and 1k-trajectory re-export is multi-hour. The trigger ("until first downstream brief commits") is firing right now — the four named exploration briefs (`vla-v2-architecture`, `vla-v2-map-conditioning`, `cotrained-retrieval-augmented`, `implicit-memory-map`) plus `backbone-bakeoff` plus `room-state-eval-harness` are all the trigger.

3. **Four driver entry points was a design smell.** Bridge / teleop / oracle / trajectory-first each had a separate script with its own argv surface. The cross-product (driver × mission source) was implicit and under-specified — for example, "coverage capture for room-state eval" had no home in the driver taxonomy. Operator's mental model was a 2-D matrix the briefs flattened to a 1-D list.

This brief's contribution: lock the format (LeRobot v3), specify one CLI entry point with two flags (`--driver` × `--mission-source`), and define each cell of the cross-product in one place.

## Architecture overview

```
Scripts/capture.py
  --driver        {bridge, teleop, scripted}      ← who provides the action
  --mission-source {queue, captioner, coverage,    ← where missions/labels come from
                    scene-metadata}
  --output <lerobot_dataset_root>
  --scene <scene_name>
  [--num-envs N]                                   ← scripted only
  [--mission-queue <yaml>]                         ← queue mission-source only
  [--n-trajectories N]                             ← captioner / coverage modes
  [--inject-bad-grounding {off, wrong_room, wrong_instance}]
  [--paraphrase-missions N]                        ← apply Qwen2.5-VL paraphrase pass
  [--capture-policy-cam / --no-capture-policy-cam]
```

The `(driver, mission_source)` cells that exist:

| `--driver` | `--mission-source` | Purpose | Throughput | Where it came from |
|---|---|---|---|---|
| `bridge` | `scene-metadata` | Autonomy-stack validation runs; emit training data as a side-effect of integration testing | ~6–15 FPS, 1 env | Replaces `run_sim_in_the_loop.py --mode harness` |
| `bridge` | `queue` | Autonomy stack walking a curated mission queue | ~6–15 FPS, 1 env | New combination |
| `teleop` | `queue` | Gamepad-driven demos against a curated queue (primary VLA training source) | ~30–60 FPS, ~30–40 episodes/hr operator-paced | Replaces `collect_perception_data.py`; subsumes `teleop-driver.md` |
| `teleop` | `scene-metadata` | Gamepad-driven ad-hoc; operator picks targets from scene metadata | Same | New combination |
| `scripted` | `queue` | Oracle driver (A* + NoCam RL controller) following `mission_queue.yaml`'s `planned_path` | Parallel-env-bound (see [`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md)) | Subsumes `oracle-driver.md` |
| `scripted` | `captioner` | Random reachable A→B + post-hoc speaker-model captioning (trajectory-first) | Same | Subsumes `trajectory-first-captioning.md` |
| `scripted` | `coverage` | "Visit every room + repeat 2–3× from different headings" — for room-state eval and VPR training data | Same | New combination; moved from `room-state-eval-harness.md` |

`collect_demos.py` (HDF5 for the RL DAPG continuous-control bootstrap path) is **orthogonal** to this system and stays separate. Its consumers are RL training scripts; ours are VLM / CLIP / VLA training scripts. Cross-reference but don't fold in.

## Output format — LeRobot v3

The canonical on-disk format is the [LeRobot dataset format](https://huggingface.co/docs/lerobot/), current major version at PR-pickup time (v3 as of 2026-05). The HF `lerobot` library is the canonical writer + loader; we author via `LeRobotDataset` append rather than writing a custom file recorder. The format is what GR00T, π0/openpi consume natively; OpenVLA / Octo consume RLDS, which is a one-pass export from LeRobot v3.

### Repository layout per scene

```
data/sim_in_the_loop/<scene_name>/                  ← one LeRobot dataset per scene
├── meta/
│   ├── info.json                                   # LeRobot dataset metadata (version, fps, etc.)
│   ├── modality.json                               # strafer embodiment spec — see below
│   ├── tasks.jsonl                                 # unique task strings (mission_text de-duplicated)
│   ├── episodes.jsonl                              # per-episode metadata + harness extensions
│   └── scenes/<scene_id>/scene_metadata.json       # per-scene static GT (room polygons, connectivity, objects)
├── data/chunk-NNN/
│   └── episode_NNNN.parquet                        # per-tick records
└── videos/chunk-NNN/episode_NNNN/
    ├── observation.images.perception.mp4           # 640×360 RGB
    ├── observation.images.policy.mp4               # 80×60 RGB (optional, --capture-policy-cam)
    └── observation.depth.perception/               # 16UC1 PNG sequence (per-frame depth)
        ├── 0000.png
        └── ...
```

### Per-tick parquet schema

| Column | Type | Notes |
|---|---|---|
| `timestamp` | float64 | Sim time from `/clock` (bridge) or `env.episode_length_buf` × dt (in-process) |
| `frame_index` | int64 | LeRobot-required monotonic index |
| `episode_index` | int64 | LeRobot-required |
| `task_index` | int64 | Index into `meta/tasks.jsonl` |
| `observation.state` | float32[N] | `(x, y, z, qx, qy, qz, qw, vx_achieved, vy_achieved, omega_z_achieved)` |
| `action` | float32[3] | `(vx_cmd, vy_cmd, omega_z_cmd)` |
| `observation.images.perception` | reference | Resolved by LeRobot loader to MP4 frame |
| `observation.images.policy` | reference (optional) | Same |
| `observation.depth.perception` | reference | Resolved to 16UC1 PNG; LeRobot supports image-sequence references |

### `meta/modality.json` for strafer

```json
{
  "state": {
    "pose": {"absolute": true, "rotation_type": "quaternion", "indices": [0, 1, 2, 3, 4, 5, 6]},
    "achieved_vel": {"absolute": false, "indices": [7, 8, 9], "components": ["vx", "vy", "omega_z"]}
  },
  "action": {
    "cmd_vel": {"absolute": false, "indices": [0, 1, 2], "components": ["vx", "vy", "omega_z"]}
  },
  "video": {
    "perception": {"original_key": "observation.images.perception", "fps": 8, "resolution": [640, 360]},
    "policy":     {"original_key": "observation.images.policy",     "fps": 8, "resolution": [80, 60]}
  },
  "depth": {
    "perception": {"original_key": "observation.depth.perception", "format": "png16uc1_mm", "scale": 0.001}
  },
  "annotation": {
    "mission_text": "annotation.tasks.mission_text"
  }
}
```

### Harness extensions to `meta/episodes.jsonl`

Every episode row carries the standard LeRobot fields (`episode_index`, `tasks`, `length`) plus strafer-specific extensions:

```jsonl
{"episode_index": 17, "tasks": ["Go to the chair by the south wall."], "length": 312,
 "scene_id": "scene_high_quality_dgx_000_seed0",
 "target_label": "chair", "target_object_id": "chair_3", "target_position_3d": [4.2, 1.8, 0.0],
 "start_pose": {"x": 0.5, "y": 0.5, "yaw": 0.0},
 "outcome": "succeeded",
 "outcome_category": "on_course",
 "paraphrases": ["Approach the chair near the south wall.", "..."],
 "source_driver": "teleop",
 "source_mission_source": "queue",
 "hard_negative_category": null,
 "injection_mode": null, "injection_mode_actual": null, "original_target_position_3d": null,
 "operator_handle": "z", "session_id": "20260524_153000",
 "generator_metadata": {"llm_model": "Qwen3-4B", "llm_seed": 42, "speaker_model": null, "speaker_seed": null},
 "leg_initial_distance_m": 3.42}
```

Field-by-field semantics:

- `scene_id` → resolves to `meta/scenes/<scene_id>/scene_metadata.json` for per-scene static GT (room polygons, connectivity, object inventories). Per-frame GT room_idx is **not stored** — eval scripts compute it on demand via [`scene_labels.get_room_at_position(pose)`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py#L148) reading scene_metadata. Caching at eval time is a consumer concern.
- `outcome ∈ {succeeded, failed, wrong_instance, wrong_room, trajectory_violation, discarded}` — what the operator / scripted driver / bridge reported at episode end.
- `outcome_category ∈ {on_course, wrong_instance, wrong_room, trajectory_violation}` — the cascade-validator-facing label (drops the success/failure distinction; collapses `succeeded` and `failed` cleanly-finished episodes into `on_course` and surfaces the deliberate-failure modes as hard negatives).
- `paraphrases` — populated by an optional `--paraphrase-missions N` Qwen2.5-VL pass (subsumes the BC-expansion brief's paraphrase generator). Empty array if disabled.
- `source_driver` / `source_mission_source` — which `(--driver, --mission-source)` cell produced this episode. Lets downstream training filter or weight by source.
- `hard_negative_category` — non-null when the episode is a deliberate hard negative (operator `X` / `SELECT` button, scripted `--inject-bad-grounding`, or captioner-synthesized negative).
- `injection_*` fields — non-null when `--inject-bad-grounding` perturbed the goal. `injection_mode_actual` records the resolved mode (so a `wrong_instance` request that fell back to `wrong_room` is visible in the data, not silent).
- `generator_metadata` — non-null when an LLM was involved (mission-generator's planner LLM or the captioner's speaker model). Carries model name + seed for reproducibility.

### Multi-camera capture

`StraferSceneCfg_InfinigenPerception` ships two cameras (`d555_camera` at 80×60 policy-cam resolution, `d555_camera_perception` at 640×360). Default is `--capture-policy-cam` ON for v1 training corpora (~5% extra wall-time per step, ~3 KB/frame extra storage, lets training scripts choose resolution). Operator can disable for storage-constrained sessions.

### Depth representation

16UC1 PNG in millimeters per frame, stored as an image sequence under `videos/.../observation.depth.perception/`. Chosen over float32 `.npy` (~5× smaller, 1mm precision, matches `depth_downsampler.py` 16UC1 convention used by the real-robot perception stack — sim-real format match). Loader-side: LeRobot's image-sequence reference resolves these; the `meta/modality.json` `depth.perception` block declares the format + scale.

### Action chunk encoding

The strict schema stores per-tick actions. Action chunks (π0/openpi expects 50-step chunks at 50 Hz; OpenVLA single-step; Octo variable) are a **loader-side concern**. LeRobot v3's loader builds chunks at load time from the per-tick parquet. We do not pre-chunk in the dataset.

Default policy command rate is 30 Hz (the deployed RL policy rate), so a 1-second action chunk is 30 ticks. This goes in `meta/info.json` as the dataset's `fps`.

## Driver: bridge

Action source: the Jetson autonomy stack (planner + executor + Nav2 + RL controller) via ROS 2 publishing `/cmd_vel`. Bridge mode brings up Isaac Sim headless on the DGX, instantiates the env, and reads `/cmd_vel` over CycloneDDS each tick. Same code path as today's `run_sim_in_the_loop.py --mode bridge`, plus per-tick writer hooks.

**Why bridge is structurally different from teleop and scripted:** the action source is on a different machine, and `/clock` is the only authority both sides see. Isaac Lab's in-process recorder APIs (`RecorderManager`, Isaac Lab Mimic) cannot represent this. The bridge driver writes via a custom recorder that pulls `/cmd_vel` from the ROS graph each tick and hands it to `LeRobotDataset.append`. The in-process drivers (teleop, scripted) can use `LeRobotDataset.append` directly inside the env loop.

**Throughput:** ~6–15 FPS headless; drops to ~6 FPS when planner + VLM are in the loop. A 30s mission produces ~180 frames at best. This is fine for end-to-end validation runs but is **not** the bulk training-data path; use teleop or scripted for that.

**Distribution:** matches deployment exactly — same Jetson stack, same Nav2 / RTAB-Map / executor that runs against the real D555. Bridge data is the canonical sim-real-comparable corpus.

### Bridge × scene-metadata (the existing `--mode harness`)

The bridge driver walks `scene_metadata.json` targets directly: enumerate `objects[]`, filter by `--label-include` / `--label-exclude`, dispatch each as a mission. No `mission_queue.yaml` needed. This is today's `run_sim_in_the_loop.py --mode harness` behavior.

### Bridge × queue

The bridge driver consumes `mission_queue.yaml` rows produced by [`mission-generator`](mission-generator.md): pop a row, dispatch the mission, capture the episode, advance. Multi-room missions, paraphrases, planned_path all available; the bridge ignores `planned_path` (its planner emits its own).

## Driver: teleop

Action source: gamepad via in-process Isaac Lab. No ROS, no bridge. Reads gamepad input through the existing mapping in [`collect_demos.py`](../../../../source/strafer_lab/scripts/collect_demos.py): left stick → `(vx, vy)`, right stick → `ωz`, triggers as speed modulators.

**This driver subsumes `collect_perception_data.py`** (which is the v0 of teleop today). The teleop driver's implementation rewrites or refactors that script; there is **not** a separate `teleop_collect.py` shipped alongside.

### Why teleop, not the bridge, for bulk training data

| Cost | Bridge | Teleop |
|---|---|---|
| Throughput | ~6–15 FPS headless; ~6 FPS with planner+VLM | ~30–60 FPS, operator-paced ~30–40 episodes/hr |
| Reliability | Mission outcome depends on Nav2 / MPPI / planner; many missions fail for stack reasons | Operator drives directly; mission succeeds iff operator decides it did |
| Distribution match | Matches v1 stack's decision distribution | Matches **human-teleop** distribution — what every published wheeled VLA (RT-2, OpenVLA, π0, NaVid) is actually trained on |

Bridge stays useful as the validation loop (it's how we verify the v1 stack composes end-to-end). It's the wrong tool for capturing 10k trajectories.

### Episode-end button mapping (teleop only)

| Button | `outcome` | `outcome_category` | `hard_negative_category` | Episode kept? |
|---|---|---|---|---|
| `Y` (triangle) | `succeeded` | `on_course` | null | Yes |
| `B` (circle) | `failed` | `on_course` | null | Yes |
| `X` + D-pad | `wrong_instance` | `wrong_instance` | `wrong_instance` | Yes (hard negative) |
| `X` + D-pad | `wrong_room` | `wrong_room` | `wrong_room` | Yes (hard negative) |
| `SELECT` (share) | `trajectory_violation` | `trajectory_violation` | `trajectory_violation` | Yes (path-shape hard negative) |
| `Back` | — | — | — | No (discarded; index does not advance) |

The `X` and `SELECT` buttons are the cleanest hard-negative source in the project — the operator commits to the specific failure mode at capture time, no post-hoc inference. Scripted-side hard negatives use `--inject-bad-grounding`.

### Operator UX

- **Live third-person view** in the Isaac Sim editor viewport (headed, not headless).
- **First-person PIP** sourced from `d555_camera_perception` (640×360). Catches "target visible to operator but blocked from D555 FoV" failures before they pollute the corpus.
- **Suggested-path overlay** in the editor viewport only (mission_queue.yaml's `planned_path` rendered as a polyline via Isaac Sim debug-draw). **Critical: must not appear in captured frames** — debug-draw renders to the editor viewport, not into the perception camera's render product. Acceptance for the teleop PR explicitly verifies this separation.
- **HUD mission text + active paraphrase** in the editor overlay; operator can cycle paraphrases via D-pad.
- **REC indicator** (red `REC` / gray `PAUSED`) shows capture state.
- **Console-only** episode timer + distance-to-target + queue position (once per second).

Deferred to follow-up briefs if real teleop sessions surface the need: constraint visualization ("highlight south wall"), prev-trajectory ghosts, quick-skip / queue manipulation tools, voice mission entry, AR/VR teleop, multi-camera live view, multi-operator collaborative teleop.

### Realistic data-volume budget (single operator, single session)

Audit-calibrated against DROID-style measurements (~30–40 demos/hr per operator with comparable UX overhead). The acceptance run is the recalibration measurement; treat the table as a soft upper bound.

| Episode type | Time per episode | Episodes / hour |
|---|---|---|
| Queue endpoint mission, success | ~1.5–2 min | ~30–40 |
| Operator-typed path-shape mission | ~2.5–3 min | ~20–25 |
| `X`-tagged hard negative (wrong-instance/room) | ~2 min | ~25–30 |
| `SELECT`-tagged path-shape negative | ~2.5–3 min | ~20–25 |

| Training target | Volume | Estimated operator time |
|---|---|---|
| CLIP fine-tune, 1 scene | ~1k frames (~30 episodes) | ~1 hour |
| VLM SFT, 1 scene | ~5k frames (~50 episodes) | ~1.5–2 hours |
| CLIP cascade hard negatives, 1 scene | ~30 success + 30 X-tagged | ~2 hours |
| v2 VLA endpoint missions, 3 scenes | ~2k demos × 5× replay-perturb = ~10k effective | ~60–80 hours |
| v2 VLA path-shape missions per constraint | ~1k path-shape demos | ~50 hours per constraint — **bottleneck; motivates the scripted driver** |

## Driver: scripted

Action source: scripted policy in-process. Two controller options:

- **`--controller rl`** (default once available): the NoCam waypoint-following RL policy from [`subgoal-env`](../trained-policy/subgoal-env.md), loaded via `strafer_shared.policy_interface.load_policy()` from `~/.strafer/models/`. Recorded `action` rows are the controller's `(vx, vy, ωz)` output — what a deployed VLA would learn to emit at the controller-output level.
- **`--controller proportional`**: a debug proportional `(vx, vy)` controller toward the next waypoint. No policy load. Useful for sanity-checking the driver before the RL checkpoint is available.

**The scripted driver is parked for execution** until [`subgoal-env`](../trained-policy/subgoal-env.md) ships the NoCam waypoint-follower. The proportional fallback exists for debug but produces low-quality demos that don't justify the scale-out it would enable. Don't pick up Phase 3 (see [Implementation tiers](#implementation-tiers)) until subgoal-env is shipped or the operator decides demo quality is acceptable.

### Parallel envs

Scripted is the only driver that benefits from `--num-envs > 1`. The cap is set by [`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md)'s measured ceiling. The perception scene config currently asserts ~1–8 envs in [`strafer_env_cfg.py:335-337`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py#L335-L337); the throughput brief is what measures the real number. Until that brief ships, default `--num-envs=1`.

### Scripted × queue (oracle path)

The driver consumes `mission_queue.yaml` rows; for each row it reads `planned_path` (LLM-emitted by the mission generator), and the controller path-tracks each waypoint to within R=0.5m, then advances. Stop heuristic: within R=0.5m of `target_position_3d`, emit `stop=True` and end the episode with `outcome=succeeded`.

Path fallback: when `--mode endpoint` queues have no `planned_path`, the driver computes its own A* shortest path on the navigable mask + connectivity graph from [`scene-connectivity-validation`](../multi-room/scene-connectivity-validation.md).

### Scripted × captioner (trajectory-first path)

The driver runs in "no-mission, random reachable A→B" mode. Targets sampled from the scene's navigable mask filtered by:
- reachable per the connectivity graph
- minimum geodesic distance > R (default 2m)
- on multi-room scenes, prefer trajectories that pass ≥ 1 room boundary

After capture, a post-hoc captioner pass runs the 7B Qwen2.5-VL with an **instructive-voice prompt** (imperative, second-person, future/present tense) over each trajectory's frames and emits:
- 1 positive caption + N=3 paraphrases (assigned to `tasks` / `paraphrases`)
- N=2 negative captions sampled from `wrong_instance` (same-room same-label) and `wrong_room` (different-room) targets — emitted as **additional episode rows referencing the same parquet trajectory data** but with different `episode_index` + `tasks` + `hard_negative_category` fields.

**Speaker-instructive eval rubric (gate before any captioned corpus is shipped):**

Sample 50 captions uniformly at random from ≥ 2 scenes (≥ 25 per scene). Score each against four binary checks:
1. **Voice.** Imperative ("Go to the chair") not declarative ("The robot went to...").
2. **Tense.** Future / present, not past.
3. **Perspective.** Second-person addressing the robot, not third-person.
4. **Groundedness.** The named target is visible in at least one frame.

Two independent inspectors; disagreements adjudicated to the conservative ("fail") call. Threshold: fewer than 5 of 50 fail (10%). Iterate on the captioner prompt until threshold met; freeze + document the final prompt in the captioner script's docstring. Re-evaluation trigger: captioner prompt or model version change.

**Captioner VRAM budget:** 7B Qwen2.5-VL FP16 = ~14 GB weights; with 32-frame trajectory inputs and a few hundred output tokens of kv-cache, ~22 GB per call. Implications: captioner runs as a **batch job, offline**, not interleaved with collection. Smaller model (Qwen2.5-VL-3B) is an option if VRAM is contested; document the trade in eval-rubric output.

### Scripted × coverage (new — for room-state eval and VPR training)

The driver runs a coverage-biased target sampler: ensure every room in the scene gets visited ≥ N=2 times across the dataset (configurable via `--coverage-visits-per-room`). After base coverage, **repeated traversals** sample previously-visited locations and approach from different headings (random rotation offset) — these produce the same-place / different-heading pairs that [`learned-spatial-encoder`](../../parked/multi-room/learned-spatial-encoder.md)'s place-recognition head mines as positive VPR pairs.

Episodes from coverage mode get `source_mission_source = "coverage"`, `outcome = "succeeded"` (no mission to fail), and no `tasks` (empty mission text). Downstream consumers that need labels (backbone-bakeoff, room-state-eval) compute per-frame GT room_idx on demand from `(pose, scene_metadata)`.

## Mission source: queue

Reads `mission_queue.yaml` produced offline by [`mission-generator`](mission-generator.md). Each row carries `mission_text + paraphrases[] + planned_path` (LLM-emitted waypoints) + per-row metadata. Consumers:
- `bridge` driver: dispatches the mission via the autonomy stack; ignores `planned_path`.
- `teleop` driver: displays `mission_text` to operator; ignores `planned_path` (operator drives directly).
- `scripted` driver: consumes `planned_path` as the controller's waypoint sequence.

Queue is the canonical mission source for production training-data capture. `mission-generator` owns its production.

## Mission source: scene-metadata

The driver walks `scene_metadata.json` targets directly: enumerate `objects[]`, filter by label, dispatch each as a one-target mission. Mission text is a templated "go to the {label}" string (no paraphrases). Used by:
- `bridge` driver (today's `--mode harness` behavior) — bulk single-target validation runs
- `teleop` driver (ad-hoc operator sessions when the operator wants to pick targets manually rather than follow a queue)

Not used by `scripted` driver (scripted always consumes either a queue or runs in captioner/coverage mode).

## Mission source: captioner

Driver-side: scripted runs random reachable A→B. Post-capture: speaker-model emits `tasks` + `paraphrases` per the [Scripted × captioner](#scripted--captioner-trajectory-first-path) section above. Only valid with `--driver scripted`.

## Mission source: coverage

Driver-side: scripted runs coverage-biased target sampling. No `tasks` field; `outcome = "succeeded"` by convention. Per [Scripted × coverage](#scripted--coverage-new--for-room-state-eval-and-vpr-training). Only valid with `--driver scripted`.

## Consumer table

How each downstream brief reads from a unified LeRobot v3 dataset:

| Consumer | Fields read | `(driver, mission_source)` pairs that produce useful data | Per-consumer notes |
|---|---|---|---|
| [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md) | `observation.images.*`, `scene_id` → `meta/scenes/<id>/scene_metadata.json` for per-frame GT room_idx | Any | Re-encodes images under each candidate backbone; scores against per-frame GT room_idx computed on demand |
| [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md) Step A | `observation.images.perception`, `tasks`, `episodes.jsonl.hard_negative_category`, `episodes.jsonl.scene_id` (for same-region positives) | bridge/teleop × queue, scripted × captioner | Multi-task fine-tune: image-vs-caption InfoNCE + same-region contrastive + hard-negative target-text |
| [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md) Step B | Above + `episodes.jsonl.episode_index` for held-out-trajectory protocol | Same | RAG-aware training against the SemanticMapManager-built retrieval index |
| [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md) | `observation.images.perception`, `observation.state.pose`, `episodes.jsonl.scene_id`, `episodes.jsonl.episode_index` | Any | Memory-bank training with `K_train ∈ {0,1,2,4,8}` augmentation; cross-attention training |
| [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md) | `observation.images.*`, `observation.depth.perception`, `observation.state.pose`, `action`, `tasks`, `episodes.jsonl.paraphrases` | teleop × queue (primary per §3.6.a), bridge supplements, scripted × queue at scale | LeRobot v3 is GR00T/π0's native loader format — direct consumption |
| [`vla-v2-map-conditioning`](../../parked/experimental/vla-v2-map-conditioning.md) Option A | Above + `episodes.jsonl.scene_id` → `meta/scenes/<id>/scene_metadata.json` (text-serialized into prompt) | Same | Symbolic regions serialized into the prompt |
| [`vla-v2-map-conditioning`](../../parked/experimental/vla-v2-map-conditioning.md) Option B | Above + the implicit memory bank (built post-capture from the same dataset) | Same | Consumer #2 of `implicit-memory-map` |
| [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md) | `observation.images.perception`, `observation.state.pose`, `episodes.jsonl.scene_id`, `episodes.jsonl.episode_index` | scripted × coverage (canonical), bridge/teleop × any | Replays through `SemanticMapManager.add_observation`; scores against per-scene GT |

Field divergence: every consumer reads from one dataset. Differences are which fields each loader projects, not which dataset they consume.

## Cross-cutting concerns

### Hard-negative injection (`--inject-bad-grounding`)

Bridge and scripted drivers accept `--inject-bad-grounding {off, wrong_room, wrong_instance}` (default `off`) with `--inject-bad-grounding-prob 0.3`. When enabled, each mission has the configured probability of having its `target_position_3d` perturbed **after** the executor projects the goal:
- `wrong_room`: swap the goal to a randomly-selected object in a different room polygon.
- `wrong_instance`: swap to another same-label object in the same room if one exists; fall back to `wrong_room` if no same-label sibling.

Perturbed episodes record `injection_mode + injection_mode_actual + original_target_position_3d` in `episodes.jsonl` so downstream consumers see the actual mode, not just the requested one (silent fallback was a 2026-05-15 audit finding). The cascade validator's `--root-cause-pass` and the co-trained validator's hard-negative set both consume these fields.

Teleop has no `--inject-bad-grounding`; teleop hard negatives come from the `X` and `SELECT` buttons.

### Mission-text paraphrasing (`--paraphrase-missions N`)

Optional post-capture pass that runs the 7B Qwen2.5-VL on each episode's `(target_label, scene_name)` and emits N paraphrases into `episodes.jsonl.paraphrases`. Default `N=0` (off). Reuses the model-loading scaffold from [`generate_descriptions.py`](../../../../source/strafer_lab/scripts/generate_descriptions.py) Stage 2 (or its successor — see [Retired downstream scripts](#retired-downstream-scripts)).

### Throughput

`scripted --num-envs N` and any "thousands of trajectories per hour" claim is gated on [`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md). The perception scene config caps at ~1–8 envs per its own self-doc; the throughput brief's measurement is what unblocks scale-out claims in downstream briefs.

### LeRobot ecosystem alignment

Direct consumers (no conversion):
- GR00T (NVIDIA's wheeled / humanoid foundation model) — LeRobot v3 with `meta/modality.json` is the canonical input format.
- π0 / openpi (Physical Intelligence's flow-matching VLA) — LeRobot v3 native.
- HF `lerobot` library — for loading and any LeRobot-compatible policy fine-tune.

One-pass exports:
- OpenVLA / Octo (RLDS / TFDS tfrecord) — LeRobot v3 → RLDS is a documented one-pass conversion.
- Isaac Lab Mimic / robomimic (HDF5) — LeRobot v3 → robomimic HDF5 is a one-pass conversion if we ever want to train an Isaac Lab Mimic baseline.

Conversion scripts live in `source/strafer_lab/strafer_lab/tools/export_*.py` as derived files (regenerable; not committed to the dataset).

### Cosmos replay-perturbation

Filed-on-trigger at [`cosmos-replay-perturbation`](../../parked/harness/cosmos-replay-perturbation.md); not in scope for the initial implementation. When picked up, Cosmos outputs land as additional MP4s under `videos/.../observation.images.perception_cosmos_{seed}.mp4` referencing the same parquet rows + `episodes.jsonl` entries with `generator_metadata.cosmos_seed` set.

## Retired downstream scripts

The following scripts have never been run against production data (verified 2026-05-24: `~/.strafer/models/` does not exist on the DGX, the CLIP fine-tune has never been performed, no `data/sim_in_the_loop/` corpus has been produced). They will be **retired** in the implementation PRs:

- [`scripts/generate_descriptions.py`](../../../../source/strafer_lab/scripts/generate_descriptions.py) — VLM SFT scene-description pipeline. Function moves into the captioner mission source (Qwen2.5-VL is the same model, the per-frame description prompt is recoverable from this script's Stage 2).
- [`scripts/prepare_vlm_finetune_data.py`](../../../../source/strafer_lab/scripts/prepare_vlm_finetune_data.py) — VLM SFT prep. Replaced by direct LeRobot v3 loading.
- [`scripts/finetune_clip.py`](../../../../source/strafer_lab/scripts/finetune_clip.py) — OpenCLIP fine-tune. Replaced by the multi-task recipe in [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md) Step A.
- [`source/strafer_lab/strafer_lab/tools/dataset_export.py`](../../../../source/strafer_lab/strafer_lab/tools/dataset_export.py) — CLIP CSV exporter. Replaced by direct LeRobot v3 loading.

The retirement happens in the implementation PRs that supersede each script's function, not in this docs-only PR.

## Implementation tiers

Each tier ships as a separate PR with its own branch. This brief stays open (in flight) until the last tier ships.

### Tier 1 — Writer + teleop driver (PR B)

Branch: `task/harness-writer-teleop`. Estimate: M.

- Implement `Scripts/capture.py --driver teleop --mission-source scene-metadata`.
- Wire `LeRobotDataset.append` writer for the in-process path.
- Subsume `collect_perception_data.py` (refactor or rewrite, no parallel scripts).
- Episode-end button mapping (`Y/B/X/SELECT/Back`).
- Operator UX baseline: PIP, HUD mission text, REC indicator.
- Acceptance run: capture ≥ 30 episodes on ≥ 2 scenes; verify schema by loading via HF `LeRobotDataset` round-trip.

Suggested-path overlay + paraphrase pass + queue support deferred to Tier 1.5 if needed (the queue mission source doesn't ship until `mission-generator` ships; sequence Tier 1.5 after mission-generator).

### Tier 2 — Bridge driver migration (PR C)

Branch: `task/harness-bridge-driver`. Estimate: M.

- Rewire `run_sim_in_the_loop.py --mode harness` to write LeRobot v3 via the same `LeRobotDataset.append` path.
- Cross-host action source: custom `RecorderTerm` that pulls `/cmd_vel` from the ROS graph each tick.
- `--inject-bad-grounding` flag wired here (bridge is the primary consumer; scripted gets it in Tier 3).
- Acceptance: bridge mode captures a multi-room mission end-to-end; LeRobot dataset round-trips; smoke test confirms `episodes.jsonl` extensions populated.

### Tier 3 — Scripted driver + queue/captioner/coverage mission sources (PR D)

Branch: `task/harness-scripted-driver`. Estimate: L.

**Gated on [`subgoal-env`](../trained-policy/subgoal-env.md) shipping** for the RL controller. Proportional fallback can land first as a debug build.

- Implement `--driver scripted --controller {rl, proportional}`.
- Parallel-env orchestration (target `num_envs` set by [`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md)).
- Mission sources: `queue`, `captioner`, `coverage`.
- Captioner: instructive-voice prompt + 4-check eval rubric + failure-pair synthesis.
- Coverage: target sampler + repeated-traversal sampler.
- Acceptance: oracle path completes 1 episode per scene via RL; captioner produces ≥ 100 positive + ≥ 200 negative rows from ≥ 2 multi-room scenes; coverage produces a dataset with every room visited ≥ 2× including repeated approaches.

### Tier 4 — Eval harness consumer

Lives in [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md) (its own brief). Not part of this consolidation; cross-referenced for completeness.

## Acceptance for this brief

This brief is the architectural spec; it does not ship code. It is "complete" (move-to-completed-stamped) only when **all** of Tiers 1, 2, 3 have shipped. Until then it sits as the in-flight architecture doc that each implementation PR references.

- [ ] Tier 1 shipped (PR B)
- [ ] Tier 2 shipped (PR C)
- [ ] Tier 3 shipped (PR D)
- [ ] Retired downstream scripts (`generate_descriptions.py` etc.) deleted by the PR that supersedes each script's function (not necessarily in this brief's PRs).
- [ ] Cross-references in all consumer briefs ([`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md), [`vla-v2-map-conditioning`](../../parked/experimental/vla-v2-map-conditioning.md), [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md), [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md), [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md), [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md)) updated to point at this brief and the LeRobot v3 schema. This is checked in the docs-only PR (PR A) and re-verified at each tier's ship.
- [ ] [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md) updated to describe the unified `Scripts/capture.py` entry point + LeRobot v3 output tree.
- [ ] [`source/strafer_lab/README.md`](../../../../source/strafer_lab/README.md) "Scripts and tools inventory" reflects the new entry point + retired scripts.

## Investigation pointers

Lifted from the source briefs:

- Bridge mainloop tick boundary: [`run_sim_in_the_loop.py`](../../../../source/strafer_lab/scripts/run_sim_in_the_loop.py); phase-profiler scaffold at lines 258–350.
- Current writer (to be replaced): [`source/strafer_lab/strafer_lab/tools/perception_writer.py`](../../../../source/strafer_lab/strafer_lab/tools/perception_writer.py).
- Gamepad mapping: [`source/strafer_lab/scripts/collect_demos.py`](../../../../source/strafer_lab/scripts/collect_demos.py). Reused verbatim for teleop.
- Camera resolutions: 640×360 perception, 80×60 policy; [`test_d555_perception_cfg.py:50`](../../../../source/strafer_lab/test/sensors/test_d555_perception_cfg.py#L50).
- Depth format reference (sim-real convention): [`depth_downsampler.py:3-7`](../../../../source/strafer_ros/strafer_perception/strafer_perception/depth_downsampler.py#L3-L7) — 16UC1 millimeters.
- LeRobot v3 spec: [https://huggingface.co/docs/lerobot/](https://huggingface.co/docs/lerobot/) (current major at PR-pickup time).
- GR00T modality.json examples: [https://github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) under `getting_started/`.
- π0 / openpi loader: [https://github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).
- Scene metadata + GT room lookup: [`scene_labels.get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py#L148).
- Speaker-Follower (instructive-prompt origin): Fried et al., NeurIPS 2018.
- Isaac Lab `RecorderManager`: [`source/isaaclab/isaaclab/managers/recorder_manager.py`](../../../../../IsaacLab/source/isaaclab/isaaclab/managers/recorder_manager.py) — useful for in-process drivers but cannot represent the cross-host bridge action source.

## Out of scope

- **`collect_demos.py` (HDF5 path for RL DAPG continuous-control bootstrap).** Different downstream consumer, different format. Cross-referenced but not folded in.
- **Real-robot capture.** Sim-only. The LeRobot v3 format choice is informed by what works on the real robot but not blocked by it.
- **Online (training-time) Cosmos augmentation.** Cosmos is the [`cosmos-replay-perturbation`](../../parked/harness/cosmos-replay-perturbation.md) brief's scope; offline batch only.
- **Re-architecting the perception scene to support more parallel envs.** Filed at [`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md); the env-count cap is a measurement result, not a fix.
- **Action-tokenization decisions.** Belongs to [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md). This brief produces continuous-valued action records; whether the VLA quantizes them is a downstream choice.
- **Real-robot mission generation.** Sim-only; [`mission-generator`](mission-generator.md) depends on Infinigen scene metadata.
- **Networked teleop** (operator on one machine, sim on another). Out of scope for teleop driver. The bridge already covers cross-host operation.
- **Multi-operator collaborative teleop, voice mission entry, AR/VR teleop.** Future briefs if needed.
