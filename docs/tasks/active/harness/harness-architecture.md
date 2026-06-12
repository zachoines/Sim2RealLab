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

As a **DGX operator who needs the harness to produce one canonical training corpus consumable by every downstream consumer** (CLIP fine-tuning, VLM SFT, behavior cloning for VLAs, retrieval-augmented validators, room-state eval, future GR00T / π0 / OpenVLA fine-tunes), I want **one entry point (`source/strafer_lab/scripts/capture.py`), one on-disk schema (LeRobot v3), and a clear cross-product of action sources × mission sources**, so that **the project's data-collection surface is reasonable for a single operator to operate, the format survives the 2026 wheeled-VLA training ecosystem, and the harness is not a code-archaeology exercise the next time we need to reason about it**.

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

2. **Format alignment was deferred-on-trigger** even though every downstream consumer in 2026 consumes some variant of the LeRobot family (GR00T currently v2, π0 / openpi following, others tracking the upstream) or RLDS (OpenVLA / Octo, exportable from LeRobot in one pass), and 1k-trajectory re-export is multi-hour. The trigger ("until first downstream brief commits") is firing right now — the four named exploration briefs (`vla-v2-architecture`, `vla-v2-map-conditioning`, `cotrained-retrieval-augmented`, `implicit-memory-map`) plus `backbone-bakeoff` plus `room-state-eval-harness` are all the trigger. Picking LeRobot v3 today targets the upstream-current shape; ecosystem-specific converters get filed-on-trigger when each fine-tune actually gets picked up.

3. **Four driver entry points was a design smell.** Bridge / teleop / oracle / trajectory-first each had a separate script with its own argv surface. The cross-product (driver × mission source) was implicit and under-specified — for example, "coverage capture for room-state eval" had no home in the driver taxonomy. Operator's mental model was a 2-D matrix the briefs flattened to a 1-D list.

This brief's contribution: lock the format (LeRobot v3), specify one CLI entry point with two flags (`--driver` × `--mission-source`), and define each cell of the cross-product in one place.

## Architecture overview

```
source/strafer_lab/scripts/capture.py
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

The canonical on-disk format is the [LeRobot dataset format v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), the current upstream major. The HF `lerobot` library is the canonical writer (via `LeRobotDataset.create()` → `add_frame()` → `save_episode()` → `finalize()`) and loader.

### Why v3, given the ecosystem state

LeRobot v3 is the **modern reference format** for wheeled / mobile-manipulation BC corpora — it's where the upstream ecosystem is moving and what new pipelines target. The downstream training stacks the project draws inspiration from (NVIDIA GR00T, Physical Intelligence π0, OpenVLA, Octo, Isaac Lab Mimic / robomimic) are **reference architectures we design *after*, not models currently consumed in-repo** (verified 2026-05-26: `grep -ri gr00t source/` returns zero matches; same for π0 / OpenVLA / Octo / Mimic).

| Consumer | Currently needs | Trigger to add an adapter |
|---|---|---|
| HF `lerobot` library + any LeRobot-compatible training script | LeRobot v3 | None — stock, ships day one. |
| NVIDIA GR00T (aspirational) | LeRobot v2 + GR00T-style `meta/modality.json` | File a v3→v2 converter brief if/when the project commits to fine-tuning GR00T. LeRobot ships a documented v3→v2 conversion script; this is a small follow-up, not foundational. |
| π0 / openpi (aspirational) | Tracks GR00T's v2 requirement today; should converge to v3 as upstream catches up | Same converter as GR00T. |
| OpenVLA / Octo (aspirational) | RLDS (TFDS tfrecord) | File a LeRobot v3 → RLDS export brief at pickup time. |
| Isaac Lab Mimic / robomimic (aspirational) | HDF5 (robomimic schema) | File a converter brief at pickup time. |

Converters get filed on the trigger that the project actually commits to fine-tuning a specific model — **not preemptively**. This keeps PR B's writer simple (one stock LeRobot v3 pattern) and defers ecosystem-specific complexity to the brief that picks up each fine-tune.

### Repository layout per scene

```
data/sim_in_the_loop/<scene_name>/                  ← one LeRobot v3 dataset per scene
├── meta/
│   ├── info.json                                   # LeRobot v3 dataset metadata (codebase_version, fps, features dict, strafer cameras block)
│   ├── tasks.jsonl                                 # unique task strings (mission_text deduplicated)
│   ├── episodes/                                   # LeRobot v3 chunked Parquet per-episode metadata + strafer extensions
│   │   ├── chunk-000/
│   │   │   └── episodes-0000.parquet
│   │   └── ...
│   ├── splits.jsonl                                # optional sidecar — named splits beyond defaults (held-out trajectories, adversarial sets)
│   ├── detection_labels.json                       # strafer custom — id↔string vocab for observation.detections.label_id (present when detections captured)
│   └── scenes/<scene_id>/scene_metadata.json       # strafer custom — per-scene static GT (room polygons, connectivity, objects)
├── data/chunk-000/
│   ├── file-0000.parquet                           # LeRobot v3 per-shard concatenated frame data (many episodes per shard)
│   └── ...
└── videos/chunk-000/
    ├── observation.images.perception/
    │   ├── file-0000.mp4                           # MP4 H.264, multiple episodes per file
    │   └── ...
    ├── observation.images.policy/                  # optional (--capture-policy-cam)
    │   └── file-0000.mp4
    └── observation.depth.perception/               # strafer custom — 16UC1 PNG per-episode sequences (see Depth representation)
        ├── episode-0000/
        │   ├── 0000.png
        │   └── ...
        └── ...
```

Per-episode metadata lives in chunked Parquet under `meta/episodes/`, **not** in a flat `episodes.jsonl` — this is the v3 shape. Frame data is concatenated per shard (`file-0000.parquet`), **not** per-episode files. Videos are also per-shard MP4s with many episodes concatenated. Depth is the one strafer custom outside the LeRobot schema: 16UC1 PNG sequences per episode at deterministic sidecar paths (see [Depth representation](#depth-representation--sidecar-png-sequence)).

### Per-frame `features` schema (declared in `meta/info.json`)

LeRobot v3's `info.json` declares each column via the `features` dict, per the v3 spec:

| Column | dtype | shape | Notes |
|---|---|---|---|
| `timestamp` | float64 | () | Sim time from `/clock` (bridge) or `env.episode_length_buf` × dt (in-process). LeRobot v3 required. |
| `frame_index` | int64 | () | LeRobot v3 required. |
| `episode_index` | int64 | () | LeRobot v3 required. |
| `task_index` | int64 | () | Index into `meta/tasks.jsonl`. |
| `observation.state.pose` | float32 | (7,) | `(x, y, z, qx, qy, qz, qw)` |
| `observation.state.achieved_vel` | float32 | (3,) | `(vx, vy, omega_z)` from `/odom`-derived encoder FK |
| `action` | float32 | (3,) | `(vx_cmd, vy_cmd, omega_z_cmd)` |
| `observation.images.perception` | video | (360, 640, 3) | MP4 H.264; LeRobot v3 native video feature |
| `observation.images.policy` | video (optional) | (60, 80, 3) | MP4 H.264 |
| `observation.depth.perception` | strafer sidecar | (360, 640) | 16UC1 PNG sequence outside the LeRobot schema — see [Depth representation](#depth-representation--sidecar-png-sequence) |
| `observation.detections.bbox` | float32 (optional) | (detections_max, 4) | Per-frame 2D boxes, pixel `(x_min, y_min, x_max, y_max)`, zero-padded — see [Detections](#detections--first-class-padded-columns) |
| `observation.detections.label_id` | int64 (optional) | (detections_max,) | Index into `meta/detection_labels.json`; `-1` in padding rows |
| `observation.detections.occlusion` | float32 (optional) | (detections_max,) | Replicator occlusion ratio, `0.0` visible → `1.0` occluded |
| `observation.detections.valid` | bool (optional) | (detections_max,) | Padding mask — the only authority on which rows are real detections |

### Camera intrinsics + preprocessing block (strafer extension to `info.json`)

Strafer-specific extension to `info.json`, lives under a `cameras` key alongside LeRobot v3 native fields. Consumers that need preprocessing or geometric reasoning read this block directly; LeRobot's stock loader ignores it.

```json
"cameras": {
  "perception": {
    "raw_resolution": [640, 360],
    "fov_h_deg": 87.0, "fov_v_deg": 58.0,
    "fx": null, "fy": null, "cx": null, "cy": null,
    "preprocessing_hint": "letterbox-to-square for ViT inputs; sensor intrinsics are post-undistortion"
  },
  "policy": {
    "raw_resolution": [80, 60],
    "fov_h_deg": 87.0, "fov_v_deg": 58.0,
    "fx": null, "fy": null, "cx": null, "cy": null,
    "preprocessing_hint": "policy cam consumed at native resolution"
  }
}
```

`backbone-bakeoff` reads `raw_resolution` + `preprocessing_hint` to decide resize-vs-letterbox per candidate; `vla-v2-map-conditioning` Option B reads `fx/fy/cx/cy` for any depth-back-projection work. Intrinsics are populated by the bridge / scripted driver from Isaac Sim's render-product calibration at capture time; `null` means not-yet-populated (acceptable for early teleop captures before the intrinsics-extraction pass lands).

### Harness extensions to `meta/episodes/` per-episode rows

Every per-episode row carries the standard LeRobot v3 columns (`episode_index`, `tasks`, `length`, `dataset_from_index`, `dataset_to_index`, `data/chunk_index`, `data/file_index`, `videos/.../chunk_index`, `videos/.../file_index`) plus strafer-specific extension columns:

| Column | Type | Notes |
|---|---|---|
| `scene_id` | string | Resolves to `meta/scenes/<scene_id>/scene_metadata.json`. **Strafer custom extension** — LeRobot loaders see this as an opaque string; consumers that need scene context look it up themselves. |
| `target_label` | string | Semantic class of the mission's primary target (e.g. `"chair"`). |
| `target_object_id` | string | Specific instance ID into `scene_metadata.objects[]`. |
| `target_position_3d` | float32[3] | `(x, y, z)` in scene frame. |
| `start_pose` | float32[3] | `(x, y, yaw)` of the episode's start. |
| `outcome` | string | `{succeeded, failed, wrong_instance, wrong_room, trajectory_violation, discarded}`. |
| `outcome_category` | string | `{on_course, wrong_instance, wrong_room, trajectory_violation}` — collapses success/failure into `on_course`; surfaces deliberate-failure modes as hard negatives. |
| `paraphrases` | list[string] | Populated by `--paraphrase-missions N`; empty array when disabled. |
| `source_driver` | string | `{bridge, teleop, scripted}`. |
| `source_mission_source` | string | `{queue, captioner, coverage, scene-metadata}`. |
| `hard_negative_category` | string \| null | `{wrong_instance, wrong_room, trajectory_violation}` or null. |
| `injection_mode` | string \| null | Requested mode for `--inject-bad-grounding`; null if no injection requested. |
| `injection_mode_actual` | string \| null | Resolved mode after fallback. **Downstream training MUST filter / weight on this column, not `injection_mode`** — see [Hard-negative injection](#hard-negative-injection---inject-bad-grounding). Null means "no injection" *or* "perturbation requested but fallback dropped it" (`injection_mode` non-null disambiguates the two). |
| `original_target_position_3d` | float32[3] \| null | Pre-injection goal; null when `injection_mode_actual` is null. |
| `operator_handle` | string \| null | Per-teleop session. |
| `session_id` | string | Capture wall-clock start; identifies a teleop session or scripted batch. |
| `generator_metadata` | json | `{llm_model, llm_seed, speaker_model, speaker_seed, cosmos_seed}` — non-null when an LLM was involved. |
| `leg_initial_distance_m` | float32 | For progress / time-to-arrival computation. |
| `episode_split` | string | `{train, val, held_out_seeds, multi_bedroom_adversarial, open_plan_adversarial}` — see [Train / val / held-out splits](#train--val--held-out-splits). |
| `capture_git_sha` | string | `git rev-parse HEAD` of the repo at capture time. **Reproducibility anchor.** |
| `scene_metadata_hash` | string | sha256 of `scene_metadata.json` at capture time. Detects scene mutations across captures. |

Per-frame GT room_idx is **not stored** — eval scripts compute it on demand via [`scene_labels.get_room_at_position(pose)`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py#L148) reading `scene_metadata.json`. Caching at eval time is a consumer concern.

### Train / val / held-out splits

Three named consumers need hold-out semantics: [`validator-evaluation`](../clip-validation/validator-evaluation.md)'s `held_out_seeds`, [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md)'s held-out-trajectory protocol, [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md)'s per-trajectory exclusion. The schema defines two layers:

1. **`episode_split` column** in per-episode metadata. Default values populated at capture time:
   - `train` — most episodes (default).
   - `val` — 10% randomly sampled by `episode_index % 10 == 0` (deterministic given a stable index sequence).
   - `multi_bedroom_adversarial`, `open_plan_adversarial` — named sets owned by [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md); set explicitly per scene.
   - `held_out_seeds` — for `validator-evaluation`; set explicitly per scene seed.

2. **`meta/splits.jsonl`** sidecar. Optional. Each row: `{name, episode_indices, scope, description}`. Consumers needing a hold-out protocol beyond the defaults (e.g., `implicit-memory-map`'s per-trajectory exclusion at training-script setup) append a row and rewrite the sidecar; the per-episode `episode_split` column remains source of truth for the original capture-time assignment, and the sidecar carries derived/named splits.

Consumers MUST pin the split they use and record the name in their training-run report so eval numbers are comparable across briefs.

### Multi-camera capture

`StraferSceneCfg_InfinigenPerception` ships two cameras (`d555_camera` at 80×60 policy-cam resolution, `d555_camera_perception` at 640×360). Default is `--capture-policy-cam` ON for v1 training corpora (~5% extra wall-time per step, ~3 KB/frame extra storage, lets training scripts choose resolution). Operator can disable for storage-constrained sessions. Both cameras are LeRobot v3 native `video` features (MP4 H.264).

### Depth representation — sidecar PNG sequence

Depth is captured as 16UC1 PNG sequences (1mm precision; the D555's noise floor at 3m is ~10–20mm, so 1mm quantization sits below the sensor floor — `harness-throughput-measurement` confirmed this is sim-real-comparable). PNG matches `depth_downsampler.py`'s 16UC1 convention used by the real-robot perception stack, preserving sim-real format match.

**LeRobot v3's video pipeline is MP4-only**; standard codecs don't support 16-bit depth video. Implementation in PR B (Tier 1) chose a **sidecar-PNG layout** over the originally-sketched `StraferDepthSequenceFeature` LeRobot subclass: the HF datasets feature-extension API (`register_feature`) is marked experimental, and 16UC1 depth doesn't fit any native dtype cleanly. PNG frames live under `videos/observation.depth.perception/episode-NNNNNN/NNNNNN.png` at deterministic paths keyed off `(episode_index, frame_index)` from the parquet rows; pure-Python read/write helpers ship in [`strafer_lab.tools.lerobot_depth`](../../../../source/strafer_lab/strafer_lab/tools/lerobot_depth.py). Stock LeRobot v3 consumers load every other column normally and ignore the sidecar tree; sim-side consumers import the helpers directly.

This is a less-elegant-looking layout than registering a feature subclass, but it has zero upstream coupling, ships without monkey-patching LeRobot, and is round-trip-tested against synthetic + real arrays (`test_lerobot_depth.py`). If a future ecosystem consumer demands a registered feature, that's a small adapter built around the same on-disk format — the format is the contract, not the loader API.

### Detections — first-class padded columns

Per-frame 2D object detections (Replicator ground truth in sim) are **first-class parquet columns, not a sidecar**. The size asymmetry against depth drives the split decision: a frame's detections are a few hundred bytes of numbers — exactly what a parquet column stores well — so first-classing them costs nothing, lets consumers read straight from the per-shard parquet rows (no sidecar walk), and avoids compounding the depth sidecar's small-files cost. Primary consumers: [`vlm-grounding-finetune`](../../parked/clip-validation/vlm-grounding-finetune.md) (bbox-grounding LoRA training pairs) and the validator's text-alignment contrast pool (sibling-detection labels).

The columns are **operator-selectable like the camera stack**: a writer constructed with `detections_max=N` declares all four columns at `N` padded slots; one constructed without it declares none (the schema shrinks — never zero-filled columns). Default slot count is 32.

| Column | dtype | shape | Semantics |
|---|---|---|---|
| `observation.detections.bbox` | float32 | (detections_max, 4) | Pixel `(x_min, y_min, x_max, y_max)` in the perception camera's render-product resolution, top-left origin — same convention as [`bbox_extractor.DetectedBbox`](../../../../source/strafer_lab/strafer_lab/tools/bbox_extractor.py). Zero in padding rows. |
| `observation.detections.label_id` | int64 | (detections_max,) | Index into `meta/detection_labels.json`'s `labels[]`. `-1` in padding rows so an accidental vocab lookup fails loudly. |
| `observation.detections.occlusion` | float32 | (detections_max,) | Replicator `occlusionRatio`: `0.0` fully visible → `1.0` fully occluded. `0.0` in padding rows. |
| `observation.detections.valid` | bool | (detections_max,) | Padding mask. **The only authority on which rows are real** — consumers must mask on it, never on zero-bbox or label sentinels. |

**Padding / truncation.** Degenerate boxes (zero-or-negative area) are dropped at pack time. When a frame's detection count exceeds `detections_max`, the largest-pixel-area boxes are kept (deterministic tie-break on label then bounds); at 32 slots truncation should be rare in our indoor scenes.

**Label vocab.** `meta/detection_labels.json` holds `{"labels": [...]}`; `label_id` indexes that list. Ids accumulate in first-seen order at capture time and are **dataset-local** — stable within one dataset, not across datasets. Consumers merging corpora must join through the label strings.

**Producer.** Replicator's `bbox_2d_tight` annotator on the perception camera → [`bbox_extractor.parse_bbox_data`](../../../../source/strafer_lab/strafer_lab/tools/bbox_extractor.py) → `DetectedBbox` → `add_frame(detections=...)`. The writer/schema support is in place from Tier 1's writer; the per-driver annotator wiring lands with the bridge (Tier 2) and scripted (Tier 3) drivers, which capture with detections **on by default**. Teleop sessions may run with detections off — the operator-perceived frame rate wins there, and the grounding consumers feed on bridge/scripted output.

### Action chunk encoding

Per-tick action records live in the parquet data; action chunks (π0/openpi expects 50-step chunks at 50 Hz; OpenVLA single-step; Octo variable) are a **loader-side concern**. LeRobot v3's loader builds chunks at load time from the per-tick rows. We do not pre-chunk in the dataset.

Default policy command rate is 30 Hz (the deployed RL policy rate), so a 1-second action chunk is 30 ticks. This goes in `meta/info.json` as the dataset's `fps`.

## Driver: bridge

Action source: the Jetson autonomy stack (planner + executor + Nav2 + RL controller) via ROS 2 publishing `/cmd_vel`. Bridge mode brings up Isaac Sim headless on the DGX, instantiates the env, and reads `/cmd_vel` over CycloneDDS each tick. Same code path as today's `run_sim_in_the_loop.py --mode bridge`, plus per-tick writer hooks.

**Why bridge is structurally different from teleop and scripted:** the action source is on a different machine, and `/clock` is the only authority both sides see. Isaac Lab's in-process recorder APIs (`RecorderManager`, Isaac Lab Mimic) cannot represent this. The bridge driver writes via a custom recorder that pulls `/cmd_vel` from the ROS graph each tick and hands it to `LeRobotDataset.add_frame()`. The in-process drivers (teleop, scripted) call `add_frame()` directly inside the env loop. All three drivers share the same writer lifecycle: `LeRobotDataset.create()` at start → `add_frame()` per tick → `save_episode()` at each episode boundary → `finalize()` at process exit.

**Throughput:** ~6–15 FPS headless; drops to ~6 FPS when planner + VLM are in the loop. A 30s mission produces ~180 frames at best. This is fine for end-to-end validation runs but is **not** the bulk training-data path; use teleop or scripted for that.

**Distribution:** matches deployment exactly — same Jetson stack, same Nav2 / RTAB-Map / executor that runs against the real D555. Bridge data is the canonical sim-real-comparable corpus.

### Bridge × scene-metadata (the existing `--mode harness`)

The bridge driver walks `scene_metadata.json` targets directly: enumerate `objects[]`, filter by `--label-include` / `--label-exclude`, dispatch each as a mission. No `mission_queue.yaml` needed. This is today's `run_sim_in_the_loop.py --mode harness` behavior.

### Bridge × queue

The bridge driver consumes `mission_queue.yaml` rows produced by [`mission-generator`](mission-generator.md): pop a row, dispatch the mission, capture the episode, advance. Multi-room missions, paraphrases, planned_path all available; the bridge ignores `planned_path` (its planner emits its own).

### Discard semantics (bridge)

When `/cmd_vel` goes silent mid-drive for longer than `--cmd-vel-grace`, the executor becomes unreachable (consecutive status-poll failures — a planner / executor crash mid-episode), the mission ends in an externally-killed terminal state (`cancelled` / `aborted`), or the harness crashes mid-mission, the current episode is discarded. The episode is **not** saved to the dataset (`save_episode()` is skipped); per the Tier 1 writer's discard contract the episode-index slot is **reused** by the next mission, so kept episodes stay contiguous. Discards are logged but not analyzed by the consumer table — they're operational noise, not training signal.

Missions that run to a terminal `failed` / `timeout` through the full stack are **kept** with `outcome = failed`: the executor's status surface doesn't distinguish "Nav2 non-recoverable" from "genuinely couldn't reach the goal" cross-host, and a failed-but-real attempt is filterable signal where a half-captured episode is not.

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

**Speaker-instructive eval rubric (project policy; gate before any captioned corpus is shipped):**

This rubric is **project policy** — Speaker-Follower (Fried et al., NeurIPS 2018) introduced the instructive-vs-descriptive distinction but did not publish a quantitative quality threshold. The 50-caption / <10% failure bar is the project's empirical anchor; revisit if real captioner output systematically clears or fails the bar by a wide margin.

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

### Discard semantics (scripted)

When Isaac Sim env reset fails for one of the parallel envs, a per-env crash terminates a trajectory mid-flight, the RL controller (or proportional fallback) fails to converge within `--controller-timeout` seconds, or the path-tracker exceeds `--max-replan-attempts` without making progress, the affected episode is marked `outcome = discarded` and is **not** saved to the dataset. Other parallel envs continue independently. `episode_index` advances cleanly for the surviving envs; the discarded slot is not reused.

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
| [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md) Step A | `observation.images.perception`, `tasks`, per-episode `hard_negative_category`, per-episode `scene_id` (for same-region positives) | bridge/teleop × queue, scripted × captioner | Multi-task fine-tune: image-vs-caption InfoNCE + same-region contrastive + hard-negative target-text |
| [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md) Step B | Above + `episode_index` for held-out-trajectory protocol (via `episode_split` or `meta/splits.jsonl`) | Same | RAG-aware training against the SemanticMapManager-built retrieval index |
| [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md) | `observation.images.perception`, `observation.state.pose`, per-episode `scene_id`, `episode_index` (for per-trajectory exclusion) | Any | Memory-bank training with `K_train ∈ {0,1,2,4,8}` augmentation; cross-attention training |
| [`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md) | `observation.images.*`, `observation.depth.perception` (via strafer custom feature), `observation.state.pose`, `action`, `tasks`, per-episode `paraphrases` | teleop × queue (primary per §3.6.a), bridge supplements, scripted × queue at scale | LeRobot v3 native columns + the strafer depth feature class |
| [`vla-v2-map-conditioning`](../../parked/experimental/vla-v2-map-conditioning.md) Option A | Above + per-episode `scene_id` → `meta/scenes/<id>/scene_metadata.json` (text-serialized into prompt) | Same | Symbolic regions serialized into the prompt |
| [`vla-v2-map-conditioning`](../../parked/experimental/vla-v2-map-conditioning.md) Option B | Above + the implicit memory bank (built post-capture from the same dataset) | Same | Consumer #2 of `implicit-memory-map` |
| [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md) | `observation.images.perception`, `observation.state.pose`, per-episode `scene_id`, `episode_index` | scripted × coverage (canonical), bridge/teleop × any | Replays through `SemanticMapManager.add_observation`; scores against per-scene GT |

Field divergence: every consumer reads from one dataset. Differences are which fields each loader projects, not which dataset they consume.

## Cross-cutting concerns

### Hard-negative injection (`--inject-bad-grounding`)

Bridge and scripted drivers accept `--inject-bad-grounding {off, wrong_room, wrong_instance}` (default `off`) with `--inject-bad-grounding-prob 0.3`. When enabled, each mission has the configured probability of having its `target_position_3d` perturbed **after** the executor projects the goal:
- `wrong_room`: swap the goal to a randomly-selected object in a different room polygon.
- `wrong_instance`: swap to another same-label object in the same room if one exists; fall back to `wrong_room` if no same-label sibling.

Perturbed episodes record `injection_mode + injection_mode_actual + original_target_position_3d` in per-episode metadata so downstream consumers see the actual mode, not just the requested one (silent fallback was a 2026-05-15 audit finding). The cascade validator's `--root-cause-pass` and the co-trained validator's hard-negative set both consume these fields.

**Downstream training filters and weights MUST key off `injection_mode_actual`, not `injection_mode`.** When `wrong_instance` is requested at p=0.3 on a scene with few duplicate labels, the actual `wrong_instance` rate may be much lower because the fallback to `wrong_room` fires often. Filtering by the requested mode would mis-weight the corpus; filtering by the actual mode produces an honest split.

**When no fallback candidate exists** (single-room scene where the requested target is the only same-label instance AND no different-room candidate exists): the perturbation silently drops. Recorded as `injection_mode_actual = null` and `original_target_position_3d = null` while `injection_mode` remains set to the requested mode. This lets a consumer audit how often the drop happens per scene without flooding the dataset with mislabeled negatives.

Teleop has no `--inject-bad-grounding`; teleop hard negatives come from the `X` and `SELECT` buttons.

### Mission-text paraphrasing (`--paraphrase-missions N`)

Optional post-capture pass that runs the 7B Qwen2.5-VL on each episode's `(target_label, scene_name)` and writes N paraphrases into the per-episode `paraphrases` column. Default `N=0` (off). Reuses the model-loading scaffold from [`generate_descriptions.py`](../../../../source/strafer_lab/scripts/retired/generate_descriptions.py) Stage 2 (or its successor — see [Retired downstream scripts](#retired-downstream-scripts)).

### Throughput

`scripted --num-envs N` and any "thousands of trajectories per hour" claim is gated on [`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md). The perception scene config caps at ~1–8 envs per its own self-doc; the throughput brief's measurement is what unblocks scale-out claims in downstream briefs.

### LeRobot ecosystem alignment

**What's stock-loadable from a LeRobot v3 dataset** (no strafer-side adapter required): the `(observation.images.*, observation.state.*, action, tasks, episode_index, frame_index, timestamp)` columns via HF `LeRobotDataset` directly, plus the `observation.detections.*` padded columns (plain parquet features — though interpreting `label_id` needs `meta/detection_labels.json`, which is strafer plumbing below). LeRobot's own training scripts and any LeRobot-v3-compatible policy fine-tune get this for free.

**What requires strafer-side custom plumbing** (ships with the harness package, so any in-repo consumer sees it automatically):

- `observation.depth.perception` — sidecar PNG sequence outside the LeRobot schema; read via `strafer_lab.tools.lerobot_depth` (see [Depth representation](#depth-representation--sidecar-png-sequence)).
- `meta/episodes/`'s strafer extension columns (`scene_id`, `outcome`, `outcome_category`, `paraphrases`, `hard_negative_category`, `injection_mode_actual`, `episode_split`, `capture_git_sha`, `scene_metadata_hash`, etc.) — LeRobot's stock loader exposes these as extra parquet columns but doesn't interpret them. Consumers read them directly.
- `meta/scenes/<scene_id>/scene_metadata.json` — strafer custom, not a LeRobot v3 concept. Consumers resolve `scene_id` → file path themselves.
- `cameras` block in `info.json` — strafer custom for intrinsics + preprocessing hints.

**What requires a follow-up converter brief, filed-on-trigger when the project commits to fine-tuning a specific ecosystem model**:

- **GR00T** (aspirational): LeRobot v3 → v2 conversion (LeRobot ships the script) + GR00T-style `meta/modality.json` adapter. File at the point we actually pick up GR00T fine-tuning.
- **π0 / openpi** (aspirational): tracks GR00T's v2 requirement today; should converge to v3 as upstream catches up. Same converter as GR00T.
- **OpenVLA / Octo** (aspirational): LeRobot v3 → RLDS export.
- **Isaac Lab Mimic / robomimic** (aspirational): LeRobot v3 → robomimic HDF5 export.

**None of these are currently consumed in-repo** (verified 2026-05-26 by grep). They are reference architectures the project designs *after*; the brief lists them so a future agent doesn't have to re-derive the conversion landscape. Converters live at `source/strafer_lab/strafer_lab/tools/export_*.py` when they ship, as derived files (regenerable; not committed to the dataset).

### Cosmos replay-perturbation

Filed-on-trigger at [`cosmos-replay-perturbation`](../../parked/harness/cosmos-replay-perturbation.md); not in scope for the initial implementation. When picked up, Cosmos outputs land as additional MP4s under `videos/.../observation.images.perception_cosmos_{seed}.mp4` referencing the same parquet rows + per-episode metadata entries with `generator_metadata.cosmos_seed` set.

## Retired downstream scripts

The following scripts have never been run against production data (verified 2026-05-24: `~/.strafer/models/` does not exist on the DGX, the CLIP fine-tune has never been performed, no `data/sim_in_the_loop/` corpus has been produced). They will be **retired** in the implementation PRs:

- [`scripts/retired/generate_descriptions.py`](../../../../source/strafer_lab/scripts/retired/generate_descriptions.py) — VLM SFT scene-description pipeline. Function moves into the captioner mission source (Qwen2.5-VL is the same model, the per-frame description prompt is recoverable from this script's Stage 2).
- [`scripts/retired/prepare_vlm_finetune_data.py`](../../../../source/strafer_lab/scripts/retired/prepare_vlm_finetune_data.py) — VLM SFT prep. Replaced by direct LeRobot v3 loading.
- [`scripts/retired/finetune_clip.py`](../../../../source/strafer_lab/scripts/retired/finetune_clip.py) — OpenCLIP fine-tune. Replaced by the multi-task recipe in [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md) Step A.
- [`source/strafer_lab/strafer_lab/tools/retired/dataset_export.py`](../../../../source/strafer_lab/strafer_lab/tools/retired/dataset_export.py) — CLIP CSV exporter. Replaced by direct LeRobot v3 loading.

The retirement happens in the implementation PRs that supersede each script's function, not in this docs-only PR.

### Per-tick labels carried forward as loader-side computation

The retired BC-expansion brief specified per-tick `progress` and `stop_target` columns in the parquet schema. These **do not** appear in the per-frame `features` schema above; they are intentionally **loader-side computable** from existing fields rather than baked into the dataset:

- `progress(t) = 1 - geodesic(state.pose(t), target_position_3d) / leg_initial_distance_m` — where `geodesic` uses the shared geodesic-A* helper called out in [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md)'s "Geodesic-A* utility" section. `target_position_3d` and `leg_initial_distance_m` live in per-episode metadata.
- `stop_target(t) = True` for the last K ticks of an `outcome=succeeded` episode (default K=5; configurable per loader), False elsewhere.

A consumer that wants these columns at training time computes them in the dataset loader rather than reading them from the parquet. This keeps the strict schema minimal; consumers that don't need them don't pay storage cost. The shared loader-side helper can live alongside the geodesic-A* utility in `source/strafer_lab/strafer_lab/tools/loader_helpers.py` when first needed.

## Implementation tiers

Each tier ships as a separate PR with its own branch. This brief stays open (in flight) until the last tier ships.

### Tier 1 — Writer + teleop driver (PR B)

Branch: `task/harness-writer-teleop`. Estimate: M.

- Implement `source/strafer_lab/scripts/capture.py --driver teleop --mission-source scene-metadata`.
- Wire the LeRobot v3 writer using the documented API: `LeRobotDataset.create()` at startup → `add_frame()` per tick within an episode → `save_episode()` at episode boundaries → **`finalize()` at process exit** to consolidate shard parquet and write the per-shard concatenated layout. Forgetting `finalize()` corrupts the dataset (per LeRobot v3 docs); the writer must guarantee `finalize()` runs even on exceptional exit (atexit handler or context manager).
- 16UC1 PNG depth (writer + loader symmetric). Shipped as the sidecar-PNG layout rather than the originally-sketched `StraferDepthSequenceFeature` registered feature class — see [Depth representation](#depth-representation--sidecar-png-sequence) for the recorded decision.
- Subsume `collect_perception_data.py` (refactor or rewrite, no parallel scripts).
- Episode-end button mapping (`Y/B/X/SELECT/Back`); set `outcome`, `outcome_category`, `hard_negative_category` per the [Episode-end button mapping](#episode-end-button-mapping-teleop-only) table.
- Populate `capture_git_sha` + `scene_metadata_hash` + `episode_split` (defaults to `train` for ordinary captures; `val` for `episode_index % 10 == 0`) on every saved episode.
- Operator UX baseline: PIP, HUD mission text, REC indicator.
- Acceptance run: capture ≥ 30 episodes on ≥ 2 scenes; verify schema by loading via HF `LeRobotDataset` round-trip; confirm `meta/info.json` declares all features correctly; confirm the depth custom feature loads + decodes 16UC1 → float32 meters correctly.

Suggested-path overlay + paraphrase pass + queue support deferred to Tier 1.5 if needed (the queue mission source doesn't ship until `mission-generator` ships; sequence Tier 1.5 after mission-generator).

### Tier 2 — Bridge driver migration (PR C)

Branch: `task/harness-bridge-driver`. Estimate: M.

- Rewire `run_sim_in_the_loop.py --mode harness` to write LeRobot v3 via the same `create()` / `add_frame()` / `save_episode()` / `finalize()` lifecycle as Tier 1.
- Cross-host action source: custom recorder that pulls `/cmd_vel` from the ROS graph each tick. (Shipped as the `IsaacLabEnvAdapter` sampling the bridge's rclpy `/cmd_vel` subscription per step + `BridgeLeRobotRecorder` mapping the harness episode lifecycle onto the writer — a plain recorder class, not an Isaac Lab `RecorderTerm`, per the constraint noted under [Driver: bridge](#driver-bridge).)
- Wire the perception camera's Replicator `bbox_2d_tight` annotator → `bbox_extractor.parse_bbox_data` → `add_frame(detections=...)` (detections on by default for bridge captures — see [Detections](#detections--first-class-padded-columns)).
- `--inject-bad-grounding` flag wired here (bridge is the primary consumer; scripted gets it in Tier 3).
- Acceptance: bridge mode captures a multi-room mission end-to-end; LeRobot dataset round-trips; smoke test confirms per-episode metadata columns (under `meta/episodes/`) populate correctly, including the strafer extensions (`outcome`, `outcome_category`, `injection_mode_actual` when injection ran, `capture_git_sha`, `scene_metadata_hash`, `episode_split`).

### Tier 3 — Scripted driver + queue/captioner/coverage mission sources (PR D)

Branch: `task/harness-scripted-driver`. Estimate: L.

**Gated on [`subgoal-env`](../trained-policy/subgoal-env.md) shipping** for the RL controller. Proportional fallback can land first as a debug build.

- Implement `--driver scripted --controller {rl, proportional}`.
- Parallel-env orchestration (target `num_envs` set by [`harness-throughput-measurement`](../../parked/harness/harness-throughput-measurement.md)).
- Mission sources: `queue`, `captioner`, `coverage`.
- Detections annotator wiring, same as Tier 2 (on by default for scripted captures).
- Captioner: instructive-voice prompt + 4-check eval rubric + failure-pair synthesis.
- Coverage: target sampler + repeated-traversal sampler.
- Acceptance: oracle path completes 1 episode per scene via RL; captioner produces ≥ 100 positive + ≥ 200 negative rows from ≥ 2 multi-room scenes; coverage produces a dataset with every room visited ≥ 2× including repeated approaches.

### Tier 4 — Eval harness consumer

Lives in [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md) (its own brief). Not part of this consolidation; cross-referenced for completeness.

## Acceptance for this brief

This brief is the architectural spec; it does not ship code. It is "complete" (move-to-completed-stamped) only when **all** of Tiers 1, 2, 3 have shipped. Until then it sits as the in-flight architecture doc that each implementation PR references.

- [ ] Tier 1 shipped (PR B)
- [x] Tier 2 shipped (PR C)
- [ ] Tier 3 shipped (PR D)
- [ ] Retired downstream scripts (`generate_descriptions.py` etc.) deleted by the PR that supersedes each script's function (not necessarily in this brief's PRs).
- [ ] Cross-references in all consumer briefs ([`vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md), [`vla-v2-map-conditioning`](../../parked/experimental/vla-v2-map-conditioning.md), [`cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md), [`implicit-memory-map`](../../parked/clip-validation/implicit-memory-map.md), [`backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md), [`room-state-eval-harness`](../multi-room/room-state-eval-harness.md)) updated to point at this brief and the LeRobot v3 schema. This is checked in the docs-only PR (PR A) and re-verified at each tier's ship.
- [ ] [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md) updated to describe the unified `source/strafer_lab/scripts/capture.py` entry point + LeRobot v3 output tree.
- [ ] [`source/strafer_lab/README.md`](../../../../source/strafer_lab/README.md) "Scripts and tools inventory" reflects the new entry point + retired scripts.

## Investigation pointers

Lifted from the source briefs:

- Bridge mainloop tick boundary: [`run_sim_in_the_loop.py`](../../../../source/strafer_lab/scripts/run_sim_in_the_loop.py); phase-profiler scaffold at lines 258–350.
- Legacy `frames.jsonl` writer (`tools/perception_writer.py`): deleted by Tier 2's PR when the bridge driver moved onto the LeRobot writer; recoverable from git history.
- Gamepad mapping: [`source/strafer_lab/scripts/collect_demos.py`](../../../../source/strafer_lab/scripts/collect_demos.py). Reused verbatim for teleop.
- Camera resolutions: 640×360 perception, 80×60 policy; [`test_d555_perception_cfg.py:50`](../../../../source/strafer_lab/test_sim/sensors/test_d555_perception_cfg.py#L50).
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
