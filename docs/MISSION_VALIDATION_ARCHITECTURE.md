# Mission validation architecture: CLIP today, alternatives next

This document is the deliverable of
[`docs/tasks/completed/mid-mission-validation-investigation.md`](tasks/completed/mid-mission-validation-investigation.md)
once that brief ships. It is an architectural audit + literature
survey + recommendation, written so that downstream briefs
([`clip-mid-mission-validator-evaluation.md`](tasks/active/clip-mid-mission-validator-evaluation.md),
[`learned-mid-mission-validator.md`](tasks/active/learned-mid-mission-validator.md))
have a defensible build-or-defer baseline.

The framing question: **between the moment the executor commits to a
goal pose and the moment Nav2 reports `arrived`, what should be
checking whether the robot is still heading somewhere the operator
intended?** Today the answer is "nothing actually wired in
production"; the next question is "what's worth wiring."

## Section 1 — Current state audit

### 1.1 The high-tier round-trip the verification gap is measured against

A single mission step that lands on a `navigate_to_pose` accumulates,
in order:

1. `scan_for_target` — six grounding calls per rotation (one per
   45° heading), 1.5–4 s per call from
   [`strafer_vlm/README.md`](../source/strafer_vlm/README.md#deferred--known-limitations);
   call site
   [`mission_runner.py:570-720`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L570-L720).
   Worst-case rotation cost ≈ 18–24 s before the robot has a target
   bbox.
2. Goal projection on the Jetson — sub-second.
3. `navigate_to_pose` — Nav2 leg, scenario-dependent. For the
   Infinigen single-room scene used in the harness, leg time is
   tens of seconds; for the multi-room scenes it can stretch into
   the minute range.
4. `verify_arrival` — one CLIP image embedding + one ChromaDB
   top-k query at the goal pose
   ([`mission_runner.py:1302-1411`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1302-L1411)).

The verification gap is the open interval between step 3's start
and step 4's arrival check. On a representative mission this is
**~30–90 s of blind transit** during which only odometry / Nav2
costmap collisions can interrupt — no semantic check is performed.

### 1.2 What the CLIP + semantic-map code actually does

Files audited in
[`source/strafer_autonomy/strafer_autonomy/semantic_map/`](../source/strafer_autonomy/strafer_autonomy/semantic_map/):

| File | Role | Notes |
|---|---|---|
| [`clip_encoder.py`](../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py) | OpenCLIP ViT-B/32 ONNX wrapper. Takes RGB uint8 → 512-dim L2-normalized vector. Loads `clip_visual.onnx` + `clip_text.onnx` (or fallback `clip_vit_b32.onnx`) from `~/.strafer/models`. Disabled-state fallback returns zero-vectors, which downstream paths short-circuit on. | Image preprocess: shorter-side resize to 224, center-crop to 224×224, ImageNet mean/std. **`d555_camera_perception` publishes 640×360**, so the resize-then-center-crop drops ~87 px from each side — about 27% of the horizontal field of view never reaches CLIP. |
| [`manager.py`](../source/strafer_autonomy/strafer_autonomy/semantic_map/manager.py) | `SemanticMapManager`: NetworkX `DiGraph` for spatial relations + ChromaDB collection (`semantic_map_embeddings`, cosine, HNSW) for embedding ANN. `add_observation()` writes a `SemanticNode` with the CLIP vector; `query_by_embedding()` and `query_by_text()` are the retrieval surfaces. | Persistence: `~/.strafer/semantic_map/{graph.json, chroma/}`. Pruning is TTL-based (1 h / 6 h / 24 h tiers by observation count). |
| [`background_mapper.py`](../source/strafer_autonomy/strafer_autonomy/semantic_map/background_mapper.py) | Movement-gated daemon thread. Every `poll_interval_s=2.0` it asks the ROS client for a pose; if the pose has moved ≥ `min_translation_m=0.5` m or rotated ≥ `min_rotation_deg=30°`, captures the camera frame and writes an observation. Optionally calls `TransitMonitor.check()` and sets a divergence flag the runner polls. | Default capture rate is therefore **at most 0.5 Hz**, gated by motion. |
| [`transit_monitor.py`](../source/strafer_autonomy/strafer_autonomy/semantic_map/transit_monitor.py) | The "is the robot drifting toward the wrong region?" decision. Pulls top-3 from ChromaDB; counts how many fall within `goal_radius_m=3.0` of the goal. Aborts after 3 consecutive captures with `near_goal == 0`. | History buffer is per-leg, never persisted. The decision rule is **purely ranking** — no fixed cosine threshold — by design (see §1.4 in [`STRAFER_AUTONOMY_NEXT.md`](STRAFER_AUTONOMY_NEXT.md)). |
| [`models.py`](../source/strafer_autonomy/strafer_autonomy/semantic_map/models.py) | `Pose2D`, `SemanticNode`, `SemanticEdge`, `DetectedObjectEntry`. Pure data classes; not part of the runtime path. | |

CLIP-touching skills inside the executor:

| Skill | Site | What CLIP does |
|---|---|---|
| `verify_arrival` | [`mission_runner.py:1302-1411`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1302-L1411) | Encodes the arrival-pose frame, top-k = 5 ANN, counts how many of the top-k sit within `goal_radius_m = 3.0` m of the goal. ≥ `majority = 3` ⇒ verified. |
| `scan_for_target` (`_store_scan_observation`) | [`mission_runner.py:1938-1980`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1938) | Encodes each scan-rotation frame and writes it to the map. This is the **primary embedding population path** during normal operation. |
| `scan_for_target` (`_try_query_before_scan`) | [`mission_runner.py:1850-1936`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1850-L1936) | Skips a 6-call grounding rotation if the label was observed within `max_map_age_s = 300` s **and** the current view's top-1 ANN match is within 2 m of the prior label observation. CLIP top-1 is the gating signal. |
| `query_environment` | [`mission_runner.py:1593-1610`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1593) | Free-text environment queries via `query_by_text` (CLIP text encoder → ANN). |
| Transit-divergence cancel | [`mission_runner.py:868-940`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L868-L940) | Polls `BackgroundMapper.divergence_detected()` while Nav2 runs; cancels the goal if set. |

### 1.3 Critical finding: the production executor does not wire any of this in

[`strafer_autonomy/executor/main.py:146-151`](../source/strafer_autonomy/strafer_autonomy/executor/main.py#L146-L151)
calls

```python
server, runner = build_command_server(
    planner_client=planner_client,
    grounding_client=grounding_client,
    ros_client=ros_client,
    runner_config=runner_config,
)
```

No `semantic_map=`, no `background_mapper=`. The
[`build_command_server` signature](../source/strafer_autonomy/strafer_autonomy/executor/command_server.py#L246-L256)
defaults both to `None`, and `MissionRunner` short-circuits every
CLIP-touching skill on `None`:

- `verify_arrival` returns `succeeded` with `verified=False, reason=semantic_map_disabled`
  ([`mission_runner.py:1306-1312`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1306-L1312)).
- `_activate_transit_monitor` returns `None`, falling through to the
  pre-monitor blocking-call branch
  ([`mission_runner.py:1287-1300`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1287-L1300)).
- `_try_query_before_scan` returns `None`
  ([`mission_runner.py:1858-1859`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1858)),
  forcing a full 6-rotation scan every mission.
- `_store_scan_observation` and `_log_arrival_failure` no-op
  ([`mission_runner.py:1948`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1948),
  [`1421`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L1421)).

Grep confirms no other call site instantiates `SemanticMapManager`,
`BackgroundMapper`, or `TransitMonitor` outside tests:

```
$ grep -rn "SemanticMapManager(\|BackgroundMapper(\|TransitMonitor(" source/ Scripts/ \
    | grep -v /tests/
# (no matches)
```

The
[`strafer_autonomy/README.md`](../source/strafer_autonomy/README.md#L137)
"Skill table" and
[`STRAFER_AUTONOMY_NEXT.md`](STRAFER_AUTONOMY_NEXT.md) §0.1 / §0.6
both describe `verify_arrival` and the transit monitor as live
features — that documentation is **out of step with the code**. This
investigation's PR includes the README fix.

The practical consequence: there is no live CLIP-based mid-mission
or arrival validation in deployment today. The semantic-map package
is a complete-but-orphaned scaffold. Any "is CLIP transit monitoring
useful?" measurement needs the wiring step before it can run.

### 1.4 The training pipeline behind CLIP

The CLIP encoder consumed by the runtime is the OpenCLIP ViT-B/32
exported by
[`source/strafer_lab/scripts/finetune_clip.py`](../source/strafer_lab/scripts/finetune_clip.py),
which performs symmetric InfoNCE contrastive training on
`(image, description)` pairs.

Defaults:

| Parameter | Default | Source |
|---|---|---|
| Backbone | `ViT-B-32` | `TrainConfig.model_name` |
| Pretrained init | `laion2b_s34b_b79k` | `TrainConfig.pretrained` |
| Loss | symmetric InfoNCE (image↔text cross-entropy on the in-batch matrix, averaged) | training loop, lines 207–227 |
| Epochs | 10 | |
| Batch size | 64 | |
| LR | 1e-5 (AdamW) | |
| Weight decay | 0.01 | |
| Output | `clip_finetuned.pt` + `clip_visual.onnx` + `clip_text.onnx` | `train()` lines 239–254 |
| Tracking | optional MLflow run | `--mlflow-experiment` |

The training **target is open-vocabulary image-text alignment**, not
room classification. The label distribution is whatever the
description pipeline produced — see §1.5.

### 1.5 The data pipeline that feeds CLIP fine-tuning

End-to-end path:

1. **Scene gen.**
   [`prep_room_usds.py`](../source/strafer_lab/scripts/prep_room_usds.py)
   wraps Infinigen scene generation; output lives under
   `Assets/generated/scenes/<scene_name>/`.

2. **Per-scene metadata.**
   [`extract_scene_metadata.py`](../source/strafer_lab/scripts/extract_scene_metadata.py)
   walks the USD, emits `scene_metadata.json` with `objects[]`
   (label + bbox), `rooms[]`, `floor_top_z`. This is the closed-set
   label vocabulary CLIP and the VLM negative-mining draw from.

3. **Combined metadata.**
   [`generate_scenes_metadata.py`](../source/strafer_lab/scripts/generate_scenes_metadata.py)
   merges per-scene JSONs into `Assets/generated/scenes/scenes_metadata.json`.

4. **Harness capture.**
   [`run_sim_in_the_loop.py --mode harness`](../source/strafer_lab/scripts/run_sim_in_the_loop.py)
   walks the `objects[]` list, dispatches one mission per target
   to the Jetson executor over the `execute_mission` action, and
   captures `frames.jsonl` + `frame_*.jpg` per episode under
   `data/sim_in_the_loop/<scene_name>/episode_NNNN/`.
   `frames.jsonl` records: `frame_id`, `image_path`, `scene_name`,
   `scene_type`, `robot_pos`, `robot_quat`, `bboxes`, `mission_id`,
   `target_label`, `target_position_3d`, `reachability`,
   `mission_state`. **The reachability label is the only
   ground-truth proxy this stack currently has for "did the robot
   actually reach the target?"**

5. **Description generation.**
   [`generate_descriptions.py`](../source/strafer_lab/scripts/generate_descriptions.py)
   runs Stage-1 spatial facts (programmatic, via
   `SpatialDescriptionBuilder`) + Stage-2 VLM descriptions (a 7B
   Qwen2.5-VL loaded standalone — intentionally separate from the
   3B fine-tune target served by `strafer_vlm` to avoid
   self-feeding collapse). Stage 3 filters descriptions that
   mention objects outside the scene's label set; Stage 4 samples
   spot-checks for human review (not automated). Output:
   `data/descriptions/<episode>/descriptions.jsonl`.

6. **Dataset assembly.**
   [`tools/dataset_export.py:export_clip_csv`](../source/strafer_lab/strafer_lab/tools/dataset_export.py#L166)
   walks the descriptions, emits `data/clip_descriptions/clip_descriptions.csv`
   with `(image_path, description)` rows. ProcRoom (primitive-shape)
   frames are **excluded** by design — they don't transfer to real
   indoor scenes. Multiple descriptions per image become multiple
   rows; the InfoNCE batch sees them as independent positives.

7. **Fine-tune.** `finetune_clip.py` consumes the CSV.

Where each input field comes from:

| Input | Source | Authority |
|---|---|---|
| `image_path` | sim render of `d555_camera_perception` (640×360) | bridge, deterministic per pose |
| `description` text | 7B Qwen2.5-VL conditioned on Stage-1 spatial facts + RGB | model output, Stage-3 filtered against scene labels |
| Object labels in descriptions | scene-metadata `objects[]` (Infinigen prim names) | sim authoritative |
| `reachability` | Jetson executor's mission outcome | from Nav2 result, not a perception signal |

**Implication for downstream alternatives.** The pipeline already
emits everything a small validator would need — `frames.jsonl`
anchors poses to images, `mission_id` groups frames by mission,
`reachability` and `mission_state` give a weak label for
"on-course vs. off-course." A learned validator can train against
this without new infra. (The labels are weak — see §2.5.)

### 1.6 DGX-side infrastructure backing the pipeline

| Component | Where |
|---|---|
| Infinigen scene gen | `env_infinigen` conda env, Python 3.11. Scenes baked to `Assets/generated/scenes/`. |
| Isaac Sim + bridge + harness | `env_isaaclab3` conda env, Python 3.12. Bridge perf reference: ~8 Hz headless w/ cameras (`bridge-runtime-invariants.md`). |
| `strafer_vlm` 3B Qwen2.5-VL service | DGX:8100. Latency budget per
[`strafer_vlm/README.md`](../source/strafer_vlm/README.md): 2–3 s `/ground` single-object, 1.5–4 s multi-object. |
| MLflow tracking | optional, gated by `--mlflow-experiment`. |
| `strafer_lab` CLIP fine-tune | `finetune_clip.py`, OpenCLIP ViT-B/32, ONNX export. |
| ChromaDB / NetworkX runtime | Jetson-side, in-process, persisted under `~/.strafer/semantic_map/`. |

## Section 2 — Limitations analysis

The brief's framing — "frame-to-frame CLIP cosine variance on a real
harness-captured episode" — assumes (a) a populated
`data/sim_in_the_loop/<scene_name>/episode_NNNN/` tree and (b) a
trained `clip_visual.onnx`. Neither exists on the DGX as of this
writing: `data/sim_in_the_loop/` is empty, no ONNX is in
`~/.strafer/models/`, and §1.3 above shows even a working ONNX
wouldn't be exercised by the production executor. The audit below
splits limitations into **structural** (derivable from code +
configuration + literature without running anything) and
**measurement-required** (need a working pipeline + episodes; folded
into the recommendation as the prerequisite of any defensible
build-or-defer decision).

### 2.1 Structural: the verification gap is asymmetric on the cost side

The high-tier round-trip tabulated in §1.1 spends ~18–24 s on the
pre-nav `scan_for_target` rotation **before** any locomotion. The
gap that mid-mission validation closes is the post-scan, pre-arrival
window — measured in tens of seconds for single-room missions and
into minutes once multi-room navigation is in scope (and per
[`STRAFER_AUTONOMY_NEXT.md` §1.10.1](STRAFER_AUTONOMY_NEXT.md)
multi-room is explicitly out of scope today). This means:

- The cheap-monitor budget should be sized against a 30–90 s
  navigation leg, not a 5–10 s leg. Aborting at the halfway mark
  saves ~15–45 s of wasted nav.
- The validator's *latency to first decision* matters more than its
  per-call latency. A 200 ms model that needs 4 captures to make a
  call is 800 ms to decision; a 2 s model that decides on one
  capture is 2 s to decision. The current `TransitMonitor` is in
  the first regime (3 captures × 2 s poll ≈ 6 s to decision).

### 2.2 Structural: the bandwidth lost between perception camera and CLIP input

`d555_camera_perception` publishes 640×360 RGB
([`source/strafer_lab/test/sensors/test_d555_perception_cfg.py:50`](../source/strafer_lab/test/sensors/test_d555_perception_cfg.py#L50)).
`clip_encoder._preprocess_image` shorter-side-resizes to 224 and
center-crops to 224×224
([`clip_encoder.py:25-47`](../source/strafer_autonomy/strafer_autonomy/semantic_map/clip_encoder.py#L25-L47)).
The path is `640×360 → 398×224 → 224×224`. Two effects:

- **~27% of horizontal FoV is dropped per frame.** Anything
  identifying a "doorway on the right" / "couch on the left"
  beyond the central 224 px disappears. This is silent data loss;
  the system has no idea the side context existed.
- **Backbone is 224² ViT-B/32, patch 32, 49 tokens.** The token
  budget for a complex indoor scene is small relative to detector
  baselines (DINOv2-S/14 at 224² gives 256 tokens). Room-class
  separability with 49 patches of a center-cropped indoor view is
  fundamentally limited.

The patch-2 / higher-res variants of OpenCLIP (ViT-B/16, ViT-L/14)
mitigate the patch-count problem at higher Jetson cost. The
center-crop FoV loss is fixed by changing the preprocess (e.g.,
letterbox + resize) and is essentially free.

### 2.3 Structural: the open-vocab fine-tune target is wrong for the use case

§1.4 shows `finetune_clip.py` trains on `(image, free-form
description)` pairs with InfoNCE. That preserves CLIP's open-vocab
strength but trains exactly **the wrong contrast** for transit
monitoring:

- Transit monitoring decides "current frame is consistent with
  *the goal region* vs. *some other region*." That's a closed-set
  classification problem over the rooms / regions the map knows
  about.
- Open-vocab InfoNCE trains image-vs-text similarity over batches
  of *random text labels* — there is no signal pushing two
  different views of the same room together unless their VLM-emitted
  captions happen to overlap in the in-batch contrast.

A symmetric loss with image-vs-image positives sampled from the
same room (e.g., SimCLR-style or a triplet loss with
"same-room"/"different-room" mining from the harness's `scene_name`
+ `robot_pos`) would put the model exactly where the runtime wants
it. The current pipeline doesn't do that; it inherits it from
OpenCLIP's training recipe which targets a different problem.

This is independent of the FoV loss in §2.2 — the right loss on the
wrong tokens is still wrong; the right tokens with the wrong loss is
still wrong.

### 2.4 Structural: the `TransitMonitor`'s decision rule has known cold-start, sparse-map, and dead-locale failure modes

From [`transit_monitor.py:60-98`](../source/strafer_autonomy/strafer_autonomy/semantic_map/transit_monitor.py#L60-L98):

| Failure mode | Trigger in code | Behavior |
|---|---|---|
| Sparse-map cold start | `len(results) < 2` | `on_track=True`, `reason=sparse_map`. Never aborts. The first time the robot enters a new house, transit monitoring is silent. |
| Map exists but goal locale never observed | `near_goal == 0` because no map node was ever within `goal_radius_m=3.0` of this goal | The check still decides off-course after 3 captures — even if the robot is making progress. False positive on first-visit-to-target. |
| Mecanum sideslip / pure-rotation crawl | Pose advances < 0.5 m AND yaw rotates < 30° between polls | `BackgroundMapper._should_capture` returns `False`; no embedding is computed. The "3 consecutive off-course captures" logic effectively pauses, which is correct for a stationary robot but means a slowly-drifting sideways slide doesn't trip the abort. |
| Same-region-different-pose ambiguity | Two map nodes at the same `(x, y)` ± 2 m | `_add_proximity_edges` only uses spatial proximity to add edges; embeddings adjacent in *visual* space but distant in *metric* space (e.g., two doorways that look alike) cause top-k retrievals to land at the wrong locale even when the robot is on-course. |
| Disabled CLIP encoder | `_enabled=False` (no ONNX) | `encode_image` returns zeros → `query_by_embedding` returns whatever ChromaDB's first node is → false negatives. |

The first two are fundamental ("memory-based methods can't validate
against unseen places," called out in
[`STRAFER_AUTONOMY_NEXT.md` §0.6](STRAFER_AUTONOMY_NEXT.md)). The
third and fourth need either a slip-detection heuristic or
embedding-only spatial gating. The fifth is a deployment hazard —
the production executor's silent zero-embedding path (§1.3) hides
this entirely.

### 2.5 Structural: the bad-grounding failure mode the user named

The brief calls out: *"A bad VLM grounding result (the wrong 'couch'
picked, off by one room) is invisible to the lower tier."* Current
mitigations:

- `min_grounding_confidence` rejects sub-threshold detections
  ([`mission_runner.py:597-635`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L597-L635)).
- `verify_arrival` catches it at the goal pose — *if* the semantic
  map is wired (§1.3 says it isn't).
- Costmap collisions interrupt some bad goals (Nav2 won't accept a
  goal in lethal cells).

What's not caught:

- A grounding that points at the wrong instance of a known label
  (the VLM picks "couch" in the dining-room view that contains
  *two* couches in the scene).
- A grounding that points at a scene element near but not at the
  intended target (a side table next to the couch).
- A goal-projection error where the VLM's bbox is correct but the
  depth at that pixel is invalid / extrapolated.

The fraction of harness missions that fall into one of those
three is **measurement-required**, not derivable from code alone.
The harness does record `reachability` per mission, but
`reachable=False` mixes "Nav2 timed out" with "robot reached the
wrong place." Disambiguating those is one of the data-collection
asks in §4.

### 2.6 Measurement-required: the empirical questions this audit cannot answer

The brief asks for these specific numbers; this audit cannot
produce them in this session because the prerequisites aren't met:

| Question | Prerequisite | Why blocked |
|---|---|---|
| Frame-to-frame CLIP cosine variance σ on a real harness-captured episode | Populated `data/sim_in_the_loop/<scene>/episode_NNNN/` + `clip_visual.onnx` in `~/.strafer/models/` | Neither exists on DGX; harness has not been run with the current scene set, and `finetune_clip.py` has not produced an ONNX export here. |
| Top-k flip rate per meter of travel | same | same |
| What `TransitMonitor` *actually catches* on a representative subset | same + the executor wired to `semantic_map=` and `background_mapper=` | §1.3: production wiring is absent. |
| Bad-grounding fraction that mid-mission cancellation would have helped | Reachability outcomes disaggregated by failure mode | Harness records aggregate `reachable=True/False`; root-cause classification is not in `frames.jsonl`. |

**These are not "future work" excuses; they're the recommendation.**
The whole point of §4 is that the path forward is to do this
measurement *first*, with a defensible methodology, before
committing to or against any of the §3 alternatives. See §4.

A literature anchor for the variance question: published reports on
CLIP retrieval on indoor video place recognition (e.g., Doersch et
al. 2024 on DINOv2 vs. CLIP for visual place recognition) put
ViT-B/32 OpenCLIP frame-to-frame cosine on slow indoor walks at
σ ≈ 0.04–0.08 with ~10–25% top-1 flips per meter on visually
homogeneous corridors. Those numbers are environment-specific and
should be re-measured on this stack rather than imported.

## Section 3 — Alternatives survey

Each option named in the brief, sized against the Jetson Orin Nano
target (8 GB unified memory; ~67 INT8 TOPS; the bridge already
runs Nav2 + RTAB-Map + executor + a TRT-EP CLIP path that needs
~150–200 ms per encode at FP16 per
[`STRAFER_AUTONOMY_NEXT.md` §0.6](STRAFER_AUTONOMY_NEXT.md#0.6)).

### 3.1 Better CLIP usage

**The candidate.** Keep the same OpenCLIP ViT-B/32 backbone. Change
three things:

1. **Aggregation.** Replace top-1 / top-k single-frame retrieval
   with a rolling temporal window (e.g., 5-frame median) keyed on
   *room-cluster ID* rather than nearest-node. Robust to
   single-frame viewpoint flips.
2. **Fine-tune target.** Drop the open-vocab description loss. Move
   to room-classification or "same-region" contrastive (positives:
   two views of the same `scene_name` within R meters; negatives:
   views from a different scene). Existing harness output already
   labels frames by `scene_name` and `robot_pos` — no new
   annotation infra.
3. **Deployment.** Letterbox-then-224 instead of center-crop
   (preserves FoV per §2.2). Move to TRT-EP at FP16 for ~80–120 ms
   per encode on Orin Nano.

**State-of-the-art references.**
- SigLIP (Zhai et al., NeurIPS 2023) — sigmoid loss, robustly
  outperforms InfoNCE on retrieval-style tasks at the same
  parameter budget. Drop-in replacement for the InfoNCE in
  `finetune_clip.py`.
- DINOv2 (Oquab et al. 2023) — vision-only; consistently better at
  visual place recognition than CLIP's image tower. Smaller
  variants (DINOv2-S/14 at 21 M params) fit Orin Nano.
- MobileCLIP (Apple, CVPR 2024) — explicit Jetson / mobile target,
  4-12× faster than ViT-B/32 at comparable retrieval accuracy.
- Visual place recognition specifically: NetVLAD (Arandjelović et
  al. 2016) and its descendants (Patch-NetVLAD, MixVPR) are the
  literature's "what CLIP would be if trained for this" — useful
  baselines for the §4 measurement to anchor against.

**Compute on Orin Nano (rough).** Current ONNX path ~150–200 ms at
FP16. TRT-EP at FP16 plausibly 80–120 ms. SigLIP-base ~same as
ViT-B/32 (architecture identical). MobileCLIP-S0 ~30–50 ms.
DINOv2-S/14 ~60–80 ms. All comfortably under the 2 s
`BackgroundMapper` poll interval.

**Training data.** Existing pipeline suffices for a same-region
contrastive recipe — `frames.jsonl` already keys by
`scene_name` + `robot_pos`. Open-vocab descriptions in
`clip_descriptions.csv` become an *auxiliary* loss, not the
primary loss.

**Sim-to-real risk.** Same as today — the OpenCLIP pretrain saw
internet-scale imagery, indoor scenes are in distribution. The fine-tune
risk shifts from "the harness corpus narrows the open-vocab
generalization" (current) to "the same-region recipe overfits the
specific Infinigen rooms" (proposed). Mitigated by mixing real
indoor data (Places365, ImageNet-Indoor) into the contrast
batches.

**Why this is the cheap option.** No new model, no new training
infra, no new runtime budget. It directly tests whether the
embedding-drift idea works *when given a fair shot* — current
deployment doesn't give it one.

### 3.2 Small learned validator

**The candidate.** A purpose-built ~30–80 M parameter vision-only
model (or vision + mission-text, light fusion) that takes the
recent N frames + the mission text and emits a scalar
"on-course / off-course / abort" signal. Trained on
`(frame_window, mission_text, on_course?)` tuples mined from the
harness's `mission_id` + `reachability` + `mission_state`.

**State-of-the-art references.**
- Vision-and-Language Navigation progress monitors: "Self-Monitoring
  Navigation Agent" (Ma et al., ICLR 2019), "The Regretful Agent"
  (Ma et al., CVPR 2019), R2R progress monitor
  baselines. The whole VLN literature has explored this for ~7
  years.
- Embodied progress estimation: NaviLLM (Zheng et al., CVPR 2024),
  Speaker-Follower (Fried et al., NeurIPS 2018) for the
  speaker side.
- Cheap learned validators: Code-as-Policies / Inner-Monologue
  (Huang et al., 2022) cascade designs where a cheap critic gates
  an expensive policy.
- Frozen-feature + light head: a Frozen-DINOv2 / Frozen-CLIP backbone
  + a small temporal head (Conv1D or Transformer over 5–10 frames)
  is a well-established cheap architecture (Cao et al. 2023,
  "Frozen-CLIP-DETR"-style adaptations).

**Compute on Orin Nano.** Frozen-DINOv2-S/14 (21 M) +
3-layer temporal Transformer (~5 M) = ~26 M params, ~80–120 ms per
inference at FP16. Fits comfortably under the 2 s poll interval and
leaves room for the planner / VLM stack.

**Training data.**
- Positives: frames mid-leg of a `mission_state=succeeded` mission,
  `reachable=True`. The harness already labels these.
- Negatives: frames mid-leg of a `mission_state=failed` mission
  where the failure root cause is "wrong place" (vs. "Nav2 timed
  out"). **Disaggregating the failure cause is the new annotation
  ask.** The cheapest way to do this is to add a one-shot
  `verify_arrival_with_vlm` pass to the harness exit path that
  describes the arrival scene and lets a script compare the
  description against the intended target — automating the
  positive/negative split.
- Hard negatives: deliberately-perturbed missions (wrong
  goal-projection, mid-leg goal swap) that the harness can be
  scripted to inject.

The harness's existing pose stream + reachability label + the new
VLM exit-pass gives a few thousand labelled tuples per scene
without human annotation.

**Sim-to-real risk.**
- Same scene-distribution risk as the current CLIP path — Infinigen
  scenes are not real homes.
- A learned validator overfits to sim-specific rendering quirks
  (lighting, texture coherence, the absence of clutter). Mitigation:
  domain randomization in `prep_room_usds.py` outputs (already in
  progress per the SIM_TO_REAL_TUNING_GUIDE) plus held-out real-robot
  episodes once they exist.
- Reward hacking: the model learns "off-course" from any visual cue
  (e.g., wall texture A vs. wall texture B) rather than actual
  spatial drift. Mitigation: the harness can cross-validate by
  scene seed.

**Why this is the medium-cost option.** New model + new training
loop + new annotation infra (the VLM exit-pass). But it produces
exactly the signal the runtime wants and benefits from the existing
harness data path.

### 3.3 Small VLA

**The candidate.** A vision-language-action model that consumes
recent frames + the mission text and emits a discrete signal
("stay-on-course" / "stop" / "re-ground") at 5–10 Hz.

**State-of-the-art references and Orin-Nano feasibility.**

| Model | Params | VRAM (FP16) | Realistic Orin Nano fit? |
|---|---|---|---|
| RT-2 (Brohan et al., 2023) | 55 B | ~110 GB | No |
| OpenVLA (Kim et al., CoRL 2024) | 7 B | ~14 GB | No (8 GB unified) |
| Octo (Octo Team, 2024) | 27–93 M | ~0.5 GB | Yes |
| π0 / π0.5 (Physical Intelligence, 2024–25) | ~3 B | ~6 GB FP16 | Marginal — co-tenant with Nav2 / RTAB unlikely |
| TinyVLA / MiniVLA (2024) | ~80–500 M | 0.5–1.5 GB | Yes for the smaller end |
| NaVid (Zhang et al., CVPR 2024) | 7 B | ~14 GB | No (navigation-flavored but full-LLM-sized) |

The two that realistically fit the Orin Nano with the rest of the
stack co-tenant are **Octo-class (27–93 M)** and the smaller-end
**TinyVLA (~80–200 M)**. Both pre-date a "navigation success
monitor" specialization — they're trained as generalist policies, so
adapting them to emit a stay/stop/re-ground signal is a finetune,
not a drop-in.

**Compute on Orin Nano.** Octo-base inference is ~50–80 ms/frame at
FP16 in published numbers; TinyVLA ~150–300 ms. Both fit. The
issue is *training* cost (DGX-side), not inference.

**Training data.** A VLA wants `(frame_window, mission_text,
action_label)` tuples. The harness produces frame windows and
mission text, but the action labels would need to be the
hindsight-relabelled "what the executor *should* have done" — a
structured task-decomposition problem the harness doesn't currently
solve. This is more annotation infra than §3.2 needs.

**Sim-to-real risk.** Higher than §3.2. VLAs are notoriously
sensitive to the action-space and tokenization choices used at
training time; a pure-sim training run can produce a model that
fails on real robot purely on action-space drift. The literature
(OpenVLA paper, Pi0 deployment notes) consistently flags this.

**Why this is the high-cost option.** It's the broadest answer (a
VLA can also help planning, scan policy, recovery behaviors), but
it's also the answer with the most expensive training loop, the
heaviest sim-to-real risk, and the largest co-tenancy pressure
on the Jetson stack. Probably worth deferring until §3.2 either
ships and underdelivers or §3.1 ships and underdelivers.

### 3.4 No new model — smarter VLM scheduling

**The candidate.** Run `/ground` or `/describe` *during* the nav
leg at lower rate (every 5–10 s) instead of only at scan-start and
arrival. Use `/describe` outputs as a sanity check ("is what the
robot sees consistent with where the planner thinks it's going?")
or `/ground` to re-confirm the target is still in view as the
robot approaches.

**State-of-the-art references.**
- SayPlan (Rana et al., CoRL 2023) — scene-graph-grounded
  hierarchical planning with periodic re-grounding.
- DialFRED (Gao et al., CVPR 2022) — periodic VLM queries for
  ambiguity resolution.
- Inner-Monologue (Huang et al., CoRL 2022) — periodic LLM/VLM
  feedback inside a loop.

**Compute on Orin Nano.** None — the VLM lives on the DGX. The
cost is *latency budget* and *Jetson↔DGX bandwidth*. A 2–3 s
`/describe` round-trip every 5 s during a 30 s nav leg adds
**6 round-trips** per leg. At 200 KB/JPEG-roundtrip and
2.5 s/call that's 1.2 MB and 15 s of cumulative VLM time
*overlapped* with nav (the executor can issue the calls async).
The bottleneck is the VLM's single-loaded-model serial queue (per
[`strafer_vlm/README.md`](../source/strafer_vlm/README.md#L33-L36)),
which would be congested by mid-mission re-grounding plus
`scan_for_target` plus `verify_arrival` plus the planner's
`/plan_with_grounding` agentic calls.

**Training data.** None — uses the already-deployed Qwen2.5-VL.

**Sim-to-real risk.** Low — no new model. The risk is operational:
the VLM service becomes a critical-path real-time component, not
a "called once at scan, once at arrival" component. Failure modes
that today degrade gracefully (VLM slow → mission slow) become
"VLM slow → robot drives further off course before mid-mission
re-ground arrives."

**Why this is the lowest-effort option.** No training, no new
model, no new code in `semantic_map/`. But it doesn't fix the
underlying "VLM is the bottleneck" architectural worry — it just
leans on the bottleneck more. It's better viewed as the *fallback*
the cascade in §3.5 falls back to.

### 3.5 Combination — cheap tripwire, expensive arbiter

**The candidate.** Hierarchical perception:

1. CLIP (or upgrade-from-§3.1) at 0.5–1 Hz as the always-on
   tripwire. Decision: continue / wake up the arbiter.
2. On tripwire fire, the arbiter runs once: VLM `/describe` of the
   current view, compared against the mission text by a small rule
   or a held-out classifier. Decision: actually-off-course (cancel
   + re-plan) / tripwire-false-positive (resume).

This is the cascade architecture from Inner-Monologue and
Code-as-Policies, applied to the in-flight validation problem.

**State-of-the-art references.**
- Inner-Monologue (Huang et al., 2022).
- Hierarchical perception literature: TidyBot (Wu et al., 2023);
  Code-as-Policies (Liang et al., 2022).
- Cascaded validators in production: many self-driving stacks use
  cheap-monitor → expensive-arbiter cascades (Tesla AP perception
  release notes; Waymo's published architecture for
  intent-prediction validation).

**Compute on Orin Nano.** Tripwire side same as §3.1 / §3.2.
Arbiter side: one extra VLM call per fire — bounded by the
tripwire's false-positive rate. If the tripwire fires once per
mission on average, arbiter cost is one `/describe` per mission
(~2 s on the existing budget).

**Training data.** Same as §3.1 / §3.2 for the tripwire. The
arbiter has zero training cost (uses the deployed VLM).

**Sim-to-real risk.** Same as the components.

**Why this is the right shape.** It matches the asymmetric cost
profile in §2.1: the tripwire fires often and decides cheaply; the
arbiter fires rarely and decides expensively. It's also the only
option that compounds gracefully: today (no validator) → §3.1
tripwire-only → §3.5 tripwire + arbiter → §3.2 + §3.5 (replace the
tripwire with a learned validator) → §3.3 (replace everything with a
VLA, if and only if the prior steps underdeliver).

## Section 4 — Recommendation

**Recommendation: stage §3.1 + §3.5 in sequence, with §3.2 staged
as a conditional escalation.** Do not attempt §3.3 until at least
§3.1 has shipped and produced negative measurements.

The sequence:

1. **First, wire the existing CLIP path into production.** The
   single most expensive finding in §1.3 is that none of the
   semantic-map code runs in deployment. Until that's true, no
   measurement of "is CLIP useful?" is meaningful. This is filed
   as
   [`docs/tasks/active/clip-mid-mission-validator-evaluation.md`](tasks/active/clip-mid-mission-validator-evaluation.md)
   and is the prerequisite for everything else here.

2. **Second, run a measurement pass against a populated harness
   episode set.** Same brief — the wiring is the easy half; the
   measurement is the deliverable. Pre-registered metrics:
    - Frame-to-frame CLIP cosine σ on a same-leg subset.
    - Top-1 flip rate per meter of travel.
    - True-positive rate of `TransitMonitor.check` on mission legs
      labelled `reachable=False, root_cause=wrong_locale`.
    - False-positive rate on legs labelled `reachable=True`.
    - Time-to-decision (median, p95) from leg start to first
      `abort=True`.

3. **Apply the §3.1 cheap fixes** — letterbox preprocess,
   same-region contrastive head, MobileCLIP / SigLIP backbone
   swap, TRT-EP FP16 — and re-measure with the same metric set.

4. **If the §3.1-tuned tripwire's true-positive rate ≥ 0.7 at a
   false-positive rate ≤ 0.1 on the harness set**, escalate to
   §3.5: wire a single `/describe` arbiter on tripwire fire and
   re-measure the false-positive rate end-to-end.

5. **If the §3.1-tuned tripwire's true-positive rate stays below
   0.5 at any sensible false-positive rate**, escalate to §3.2:
   train a small frozen-DINOv2 + temporal-head validator on the
   harness output. Re-evaluate the cascade with §3.2 as the
   tripwire.

### 4.1 Falsifiable success criteria for the next brief

The downstream brief
[`clip-mid-mission-validator-evaluation.md`](tasks/active/clip-mid-mission-validator-evaluation.md)
ships when **all** of the following are produced against ≥ 30 harness
missions covering ≥ 3 distinct Infinigen scenes:

- [ ] `verify_arrival` runs against a non-empty semantic map on at
      least one mission per scene (proves wiring).
- [ ] `TransitMonitor.check` runs at least 5 times per mission leg
      ≥ 30 s long (proves the polling is alive).
- [ ] A measurement script (lives in
      `Scripts/eval_transit_monitor.py` per the conventions in
      [`source/strafer_lab/README.md`](../source/strafer_lab/README.md))
      computes the five metrics above and writes a JSON report
      under `data/transit_monitor_eval/<run_id>/report.json`.
- [ ] The report's TPR and FPR numbers, bucketed by scene, support
      one of two decisions:
      - **Pass (escalate to §3.5):** TPR ≥ 0.7 at FPR ≤ 0.1 on the
        full set; per-scene worst-case TPR ≥ 0.5.
      - **Fail (escalate to §3.2):** TPR < 0.5 at any FPR ≤ 0.2.
- [ ] A short addendum to this design doc records which decision
      fired and links the run report.

### 4.2 What's explicitly *not* recommended

- **Do not start a small VLA (§3.3) yet.** It's the most expensive
  option and the §3.1 + §3.5 staged path may eliminate the need.
  The Octo / TinyVLA backlog can sit in
  [`docs/tasks/DEFERRED_WORK.md`](tasks/DEFERRED_WORK.md) until
  §3.1 measurements come back.
- **Do not lean on §3.4 (smarter VLM scheduling) as a primary
  validator.** It compounds the existing single-loaded-model
  bottleneck and turns the VLM service into a real-time critical
  path. It belongs as the §3.5 *arbiter*, not as a standalone
  validator.

## Cross-references

- Current-round design master:
  [`docs/STRAFER_AUTONOMY_NEXT.md`](STRAFER_AUTONOMY_NEXT.md), §0.1
  (verify_arrival), §0.6 (transit monitoring via BackgroundMapper),
  §1.4 (unified CLIP embedding strategy), §1.10.1 (multi-room
  limitation).
- System flow Flow 6 (real-robot execution) in
  [`docs/SYSTEM_FLOW_DIAGRAMS.md`](SYSTEM_FLOW_DIAGRAMS.md). The
  verification gap lives inside Flow 6's `navigate_to_pose` skill.
- VLM grounding cost numbers: "Deferred / known limitations" of
  [`source/strafer_vlm/README.md`](../source/strafer_vlm/README.md#deferred--known-limitations).
- Bridge perf reference numbers:
  [`docs/tasks/context/bridge-runtime-invariants.md`](tasks/context/bridge-runtime-invariants.md#phase-level-profiler---profile).
- Downstream briefs:
  [`clip-mid-mission-validator-evaluation.md`](tasks/active/clip-mid-mission-validator-evaluation.md)
  and
  [`learned-mid-mission-validator.md`](tasks/active/learned-mid-mission-validator.md).
