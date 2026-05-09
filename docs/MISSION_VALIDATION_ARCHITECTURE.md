# Mission validation architecture: CLIP today, alternatives next

This document is the deliverable of
[`docs/tasks/completed/mid-mission-validation-investigation.md`](tasks/completed/mid-mission-validation-investigation.md)
once that brief ships. It is an architectural audit + literature
survey + recommendation, written so that downstream briefs
([`clip-mid-mission-validator-evaluation.md`](tasks/active/clip-mid-mission-validator-evaluation.md),
[`clip-cotrained-retrieval-augmented.md`](tasks/active/clip-cotrained-retrieval-augmented.md))
have a defensible build-or-defer baseline. (A previously-named
`learned-mid-mission-validator` brief was retired — see
[`completed/learned-mid-mission-validator.md`](tasks/completed/learned-mid-mission-validator.md).)

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
into minutes for multi-room missions, which are now the MVP
default per
[`multi-room-autonomy-stack`](tasks/active/multi-room-autonomy-stack.md)
(which lifts §1.10.1's deferral) and
[`multi-room-scene-connectivity-validation`](tasks/active/multi-room-scene-connectivity-validation.md).
The v1 *measurement* in §4 below is calibrated against
single-room data first; multi-room re-test is a follow-up after
the v1 stack's multi-room work ships. This means:

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

### 2.3 Structural: the validation problem has a granularity hierarchy the current setup ignores

The brief frames mid-mission validation as "kitchen-trending vs.
living-room-trending" — a room-grain decision. Real missions aren't
that uniform. There are three failure granularities, and the right
validator architecture differs at each level:

| Granularity | Example | Validator question | What the existing setup catches |
|---|---|---|---|
| **Wrong room** | The mission says "go to the kitchen" but the robot drifts into the bedroom. | Is the current view consistent with the goal *region*? | Endpoint-CLIP retrieval can plausibly catch this once wired and trained. |
| **Wrong instance, same room** | The dining room has two windows; the planner pointed at the wrong one. The robot navigates to within 3 m of the wrong window. | Is the current view consistent with the goal *target description*? | Endpoint-CLIP retrieval **cannot** catch this — `near_goal_count ≥ majority` is true at the wrong window because the goal pose is stamped on a map node within the radius. The signal needed is image-vs-target-text alignment, not image-vs-region retrieval. |
| **Wrong trajectory shape** | The mission says "move to the other end of the room by hugging the wall." The robot ends up at the right endpoint via a center-of-room path. | Is the path-so-far consistent with the trajectory *constraint* in the mission text? | Nothing. Nav2 plans on costmap occupancy, not "hug the wall," so the constraint never reached the lower stack. |

The current `verify_arrival` and `TransitMonitor` are case-1 tools.
They're underdesigned for case 2 and structurally incompatible with
case 3.

**The fine-tune target is wrong for case 1 in a specific way, and
wrong for case 2 in a different way:**

- For **case 1 (room-grain)**: the open-vocab InfoNCE in
  [`finetune_clip.py`](../source/strafer_lab/scripts/finetune_clip.py)
  trains image-vs-random-text-label similarity. There is no signal
  pushing two different views of the same room together unless their
  VLM-emitted captions happen to overlap in the in-batch contrast. A
  symmetric loss with image-vs-image positives sampled from the same
  `scene_name` + `robot_pos` neighborhood (SimCLR-style; triplet with
  same-room / different-room mining) would put the model exactly
  where the runtime wants it.
- For **case 2 (instance-grain)**: same-region contrastive is
  *actively wrong*. It pushes the embeddings of the dining room's
  two windows *together* (they're both same-room positives), which
  is the opposite of what instance-level discrimination needs. The
  right signal is **image-vs-target-text**: the mission's target
  phrase ("the window on the south wall") encoded by the CLIP text
  tower, cosine'd against the live frame. This is closer to CLIP's
  *original* design point and doesn't need a map at all.
- For **case 3 (trajectory-grain)**: no CLIP recipe helps. The
  planner has to decompose "hug the wall" into a checkable trajectory
  constraint (a sequence of waypoints + distance-to-wall threshold,
  or a costmap potential the executor can monitor). Without that
  decomposition the validator has no spec to check against. Case 3
  is a **planner-side prerequisite**, not a perception problem, and
  is filed accordingly in §4.

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

**These failure modes don't make CLIP dead-on-arrival, but they do
narrow CLIP's role.** Splitting them by whether they're inherent or
fixable (and which case in §2.3 they apply to):

| Failure mode | Verdict | Reason |
|---|---|---|
| Cold-start | Inherent to any memory-based method | Mitigated by an offline mapping pass before deployment, or by accepting that the first mission in a new house is unguarded. Only relevant for case-1 image-vs-image retrieval. |
| Sparse-map | Inherent to memory-based | Degrades gracefully (returns `sparse_map`); the system gets better with use. Same scope as cold-start. |
| Goal locale never observed | **Fixable** | Switch the validator's primary signal from image-vs-image retrieval (needs prior map) to **image-vs-target-text** alignment (needs only the mission text + current frame). This is what `verify_arrival` *should* be doing for case-2 instance-level checks. Image-vs-image retrieval moves to a secondary "have I been here before?" role. |
| Mecanum sideslip / pure-rotation crawl | Tractable | Slip-detection heuristic on `BackgroundMapper._should_capture` or capture-rate gating off `/odom` velocity. |
| Same-region-different-pose visual aliasing | Tractable | Embedding-only spatial gating: filter top-k results by metric distance before majority-counting. |
| Disabled CLIP encoder | Deployment hygiene | Loud-fail at startup instead of silent zero-vectors. Covered in the §4 eval brief. |

The bigger reframing: **CLIP image-vs-image retrieval is a
"place-recognition" signal, not a "validation" signal.**
Image-vs-text alignment is the more durable validator use, because
it's map-free and works on first visit. Both should be inputs to
the §3.5 cascade, not the cascade's decision.

### 2.5 Structural: the bad-grounding failure mode the user named

The brief calls out: *"A bad VLM grounding result (the wrong 'couch'
picked, off by one room) is invisible to the lower tier."* Current
mitigations:

- `min_grounding_confidence` rejects sub-threshold detections
  ([`mission_runner.py:597-635`](../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L597-L635)).
- `verify_arrival` catches it at the goal pose — *if* the semantic
  map is wired (§1.3 says it isn't), AND if the failure is
  case-1 (wrong-room) per §2.3. Case-2 (wrong-instance,
  same-room) defeats the goal-radius majority rule.
- Costmap collisions interrupt some bad goals (Nav2 won't accept a
  goal in lethal cells).

What's not caught:

- **Case 2.** A grounding that points at the wrong instance of a
  known label (the VLM picks "couch" in the dining-room view that
  contains *two* couches in the scene, or "the window" when there
  are windows on the north and south walls).
- **Case 2 (variant).** A grounding that points at a scene element
  near but not at the intended target (a side table next to the
  couch).
- A goal-projection error where the VLM's bbox is correct but the
  depth at that pixel is invalid / extrapolated.
- **Case 3.** Trajectory-shape violations ("hug the wall," "approach
  from the east"). The planner doesn't decompose them, Nav2 doesn't
  honor them, and no validator has a spec to check against.

The fraction of harness missions that fall into each is
**measurement-required**, not derivable from code alone. The
harness does record `reachability` per mission, but
`reachable=False` mixes "Nav2 timed out" with "robot reached the
wrong room" and "robot reached the wrong instance"; `reachable=True`
hides any trajectory-shape violation. Disambiguating these is the
key data-collection ask in §4.

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

**The candidate.** Keep the OpenCLIP ViT-B/32 backbone. Change four
things, split by which §2.3 case each addresses:

1. **Deployment (both cases).** Letterbox-then-224 instead of
   center-crop (preserves FoV per §2.2). Move to TRT-EP at FP16
   for ~80–120 ms per encode on Orin Nano. Loud-fail at startup if
   ONNX is missing (instead of silent zero-vectors).
2. **Aggregation (both cases).** Replace top-1 / top-k single-frame
   retrieval with a rolling temporal window (e.g., 5-frame median).
   Robust to single-frame viewpoint flips.
3. **Case-1 fine-tune target (room-grain).** Drop the open-vocab
   description loss. Move to "same-region" contrastive (positives:
   two views of the same `scene_name` within R meters; negatives:
   views from a different scene). Harness output already keys
   frames by `scene_name` + `robot_pos` — no new annotation infra.
4. **Case-2 signal (instance-grain).** Add **image-vs-target-text
   alignment** as a parallel signal: encode the mission's target
   phrase via the CLIP text tower, cosine against the live frame,
   threshold or rank against an alternate-instance pool. Map-free,
   works on first visit, exactly the use case CLIP was originally
   trained for. Crucially, do **not** use same-region contrastive
   here — it pushes alternate-instance embeddings together, which
   destroys instance discrimination (see §2.3).

The two fine-tune targets are in tension. If both heads are needed,
the recipe is a multi-task loss (image-vs-image contrastive head +
image-vs-text alignment head sharing the visual tower) — or two
separate fine-tunes serving the two cases.

**State-of-the-art references.**
- SigLIP (Zhai et al., NeurIPS 2023) — sigmoid loss, robustly
  outperforms InfoNCE on retrieval-style tasks at the same
  parameter budget. Drop-in replacement for the InfoNCE in
  `finetune_clip.py`. Better at *both* signals than vanilla CLIP.
- DINOv2 (Oquab et al. 2023) — vision-only; consistently better at
  visual place recognition than CLIP's image tower. Smaller
  variants (DINOv2-S/14 at 21 M params) fit Orin Nano. **Best for
  case 1; cannot do case 2 alone (no text tower).**
- MobileCLIP (Apple, CVPR 2024) — explicit Jetson / mobile target,
  4–12× faster than ViT-B/32. Has both towers, can do case 1 + 2.
- DFN-CLIP / OpenCLIP-DFN — improved-data CLIP variants with better
  zero-shot text-image alignment than vanilla CLIP. Case-2 lift.
- Visual place recognition (case-1 anchor): NetVLAD (Arandjelović
  et al. 2016) and its descendants (Patch-NetVLAD, MixVPR) — useful
  baselines for the §4 measurement.

**Compute on Orin Nano (rough).** Current ONNX path ~150–200 ms at
FP16. TRT-EP at FP16 plausibly 80–120 ms. SigLIP-base ~same as
ViT-B/32 (architecture identical). MobileCLIP-S0 ~30–50 ms.
DINOv2-S/14 ~60–80 ms. Image-vs-text adds one text-tower call per
mission (text encoded once; cached for the leg) — negligible.

**Training data.** Case 1 uses harness output's
`scene_name` + `robot_pos` keying directly. Case 2 uses the
`target_label` already in `frames.jsonl` plus a synthetic
"alternate target description" pool (other labels in the same scene
metadata) for negative mining. Open-vocab descriptions in
`clip_descriptions.csv` become an auxiliary loss for the text-tower
head.

**Sim-to-real risk.** Same as today for case 1 (OpenCLIP pretrain
saw internet-scale indoor imagery). Higher for case 2 because
target phrases like "the window on the south wall" carry pose
information that doesn't transfer if real-room layouts differ from
Infinigen's. Mitigation: keep target phrasing in distribution
(reuse Qwen-described phrasings from the harness rather than
hand-authored).

**Why this is the cheap option.** No new model, no new training
infra, no new runtime budget. The case-1 + case-2 split is what
makes this serious — current deployment doesn't even attempt
case 2.

### 3.2 Small learned validator (retired)

**Retired 2026-05-09.** The proposed architecture (frozen
DINOv2-S/14 + frozen sentence encoder + ~5 M trainable fusion
head) is structurally underpowered against the §3.5 cascade
alternative this project is shipping. The cascade includes
Qwen2.5-VL-3B as the arbiter plus the planner LLM as judge —
~75× more pretrained capacity and 3 orders of magnitude more
pretraining data than a small frozen-head fine-tune. Empirical
literature (Cao et al. 2023 "Vision-Language Models are Strong
Generalists"; OpenVLA 2024; Speaker-Follower analysis 2018) is
consistent: at 5 M trainable parameters trained on thousands of
examples, you don't outperform billions of pretrained
parameters even when those are zero-shot.

The original brief is preserved at
[`completed/learned-mid-mission-validator.md`](tasks/completed/learned-mid-mission-validator.md)
with its retirement stamp and rationale. The case-2 / case-3
coverage that motivated this option is now addressed by:

- **CLIP cascade improvements** —
  [`clip-cotrained-retrieval-augmented`](tasks/active/clip-cotrained-retrieval-augmented.md)
  files the path for improving the cascade itself
  (co-trained CLIP with the trajectory-first speaker model;
  retrieval-augmented inference using the
  SemanticMapManager's existing memory primitive).
- **End-to-end VLA research arm** —
  [`strafer-vla-v2-architecture`](tasks/active/strafer-vla-v2-architecture.md)
  for an alternative that has billions of pretrained
  parameters as starting capital.

The historical content of §3.2 is preserved below for context.
None of it is recommended going forward.

**(Historical) The candidate.** A purpose-built ~30–80 M parameter
vision + mission-text fusion model that takes the recent N frames
and the mission text and emits a categorical signal over the §2.3
hierarchy: `{on_course, wrong_room, wrong_instance, trajectory_violation, ambiguous}`.
Trained on `(frame_window, mission_text, category_label)` tuples
mined from harness output. The mission-text fusion is **not
optional** — it's what gives the model a chance at case 2 (and at
case 3, once the planner emits the constraint). A vision-only
variant collapses to the case-1 signal that §3.1 already provides.

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

**Training data.** Five-way categorical labels mined from harness
output:

- `on_course`: frames mid-leg of `mission_state=succeeded`,
  `reachable=True` missions where the final pose is within R m of
  `target_position_3d`. The harness already labels these.
- `wrong_room`: failed missions where the final pose is in a
  different room polygon than `target_position_3d` (room polygons
  come from `scene_metadata.json`'s `rooms[]`).
- `wrong_instance`: failed missions where the final pose is in
  the *correct* room but >R m from `target_position_3d` AND the
  scene metadata shows ≥ 2 objects with the same `target_label`.
  Needs a one-shot VLM exit-pass (`/describe` + LLM-as-judge
  comparison against the mission target phrase) to confirm the
  robot reached an *alternate* instance and not random clutter.
- `trajectory_violation`: missions tagged with a trajectory
  constraint by the planner (case 3) where the path-so-far
  violated the constraint. **Empty until the planner emits these
  constraints** — see §4 for the planner-side prerequisite.
- `ambiguous`: anything the automated labeling can't classify;
  excluded from train/eval.
- Hard negatives: deliberately-perturbed missions (wrong
  goal-projection, mid-leg goal swap, wrong-instance forced via
  metadata override) that the harness can be scripted to inject.

The harness's existing pose stream + scene metadata + the
VLM exit-pass gives a few thousand labelled tuples per scene
without human annotation. The five-way label is what lets the
validator emit a *useful* abort signal — not just "stop" but "stop,
re-plan with the correct instance" or "stop, ask the operator to
re-state the path constraint."

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
recent frames + the mission text and either (a) emits a validation
signal at 5–10 Hz or (b) replaces a mid-level skill (planner /
executor) outright. The two options have very different cost
profiles, so this section first sizes the integration depth, then
the model choice, then the data path.

#### 3.3.a Integration depth — where would a VLA actually plug in?

Three insertion depths exist for any VLA, ranked by how invasive
each is:

| Depth | What gets replaced | What stays | Action output | Realistic for strafer? |
|---|---|---|---|---|
| **Low-level controller** (RT-2 / OpenVLA / π0 style) | RL policy + Nav2 | RealSense + chassis + executor's safety wrapper | Tokenized continuous velocity or wheel torques at 5–10 Hz; an action-diffusion or autoregressive head un-tokenizes to continuous control. | Big regression risk — the existing RL policy already handles mecanum dynamics. Teaching a VLA mecanum + sim-to-real costs more than it saves. Don't go here first. |
| **Mid-level skill provider** (NaVid / NaviLLM style) | `strafer_autonomy.planner` + parts of the executor | Nav2, RL policy, scan / grounding skills | Skill calls (`navigate_to_pose(x,y,θ)`, `rotate(δθ)`, `scan_for_target(label)`) | **The natural insertion point.** The skill abstraction is already designed to be replaceable; Nav2's costmap-based obstacle handling stays; the RL policy's dynamics expertise stays. |
| **Planner-only replacement** | The LLM in `strafer_autonomy.planner` | Everything else | Plan JSON (skill list) | Low risk, low ceiling — a VLA used as a planner is mostly an expensive LLM that happens to see images. |

On the action-output question specifically: modern VLAs **don't**
emit raw velocity commands directly from text. They emit action
*tokens* (256-bin per axis is common) or use a diffusion head; a
runtime decoder un-tokenizes to continuous control. The "thinking"
happens autoregressively over visual + language + (optionally)
prior-action tokens at 5–10 Hz. Direct natural-language → low-level
torques is theoretically possible, but practically nobody builds it
that way for embedded — the action space is too rich for
autoregressive decoding to keep up.

On real-time semantic mapping + path planning + control inside one
model: the long-term VLA vision, but Orin Nano can't run a 7 B
model at 10 Hz alongside Nav2 + RTAB. Even Octo's ~50–80 ms/frame
numbers leave thin margins. Practically, the map + path planner
stay external; the VLA emits goals or skills.

For this stack, **insertion at the mid-level skill provider** is
the only depth worth scoping seriously. The other two are filed as
"don't do this without a separate brief."

#### 3.3.b Model choices and Orin Nano feasibility

| Model | Params | VRAM (FP16) | Wheeled / mobile focus? | Orin Nano fit? |
|---|---|---|---|---|
| RT-2 (Brohan et al., 2023) | 55 B | ~110 GB | Manipulation | No |
| OpenVLA (Kim et al., CoRL 2024) | 7 B | ~14 GB | Generalist | No (8 GB unified) |
| Octo (Octo Team, 2024) | 27–93 M | ~0.5 GB | Generalist; navigation extensions exist | Yes |
| π0 / π0.5 (Physical Intelligence, 2024–25) | ~3 B | ~6 GB FP16 | Mobile manipulation | Marginal — co-tenant with Nav2 / RTAB unlikely |
| TinyVLA / MiniVLA (2024) | ~80–500 M | 0.5–1.5 GB | Generalist; small targets explicitly | Yes for the smaller end |
| NaVid (Zhang et al., CVPR / RSS 2024) | 7 B (Vicuna-7B-based) | ~14 GB | Pure VLN | No |
| NaVILA (Cheng et al., 2024) | 8 B + smaller | Varies | Legged + wheeled | Marginal at the smallest variant |
| MobileVLA (Sermanet et al.) | 70 M – 500 M | 0.5–1.5 GB | Mobile platform target | Yes |

The two that realistically fit the Orin Nano with the rest of the
stack co-tenant are **Octo-class (27–93 M)** and the smaller-end
**TinyVLA / MobileVLA (~80–200 M)**. Octo + Qwen2.5-VL-3B
(already deployed) approximate a GR00T-style dual-system
architecture: Qwen as the slow VLM emitting action-latent or skill
tokens, Octo as the fast policy consuming them. The "wiring" — what
GR00T calls the action-latent vocabulary — is the design surface
you'd own.

#### 3.3.c Compute and training cost

**Inference (Orin Nano).** Octo-base ~50–80 ms/frame at FP16;
TinyVLA / MobileVLA ~150–300 ms. Both fit the existing
`BackgroundMapper` poll budget. The bottleneck is **training cost
on the DGX**, not inference.

**Training data.** A VLA wants `(frame_window, mission_text,
action_label)` tuples. The action labels need to be the
hindsight-relabelled "what the executor *should* have done." This
is more annotation infra than §3.2 needs, but it's exactly what
§3.6 (MVP-as-teacher distillation) makes available for free once
the integration round ships.

**Sim-to-real risk.** Higher than §3.2. VLAs are notoriously
sensitive to action-space and tokenization choices at training
time; a pure-sim training run can fail on real robot purely on
action-space drift. The OpenVLA paper and π0 deployment notes
consistently flag this. Mitigation: distill from the MVP teacher
(§3.6), which already operates in the deployment action space.

#### 3.3.d Cross-reference: GR00T

**Nvidia GR00T-N1** (March 2025) is a humanoid VLA framework with a
dual-system architecture: a slow VLM that emits action latents and
a fast diffusion policy that consumes them at 50–100 Hz. The
*architecture* generalizes beyond humanoids; Isaac GR00T added
wheeled / legged base support in 2025 per Nvidia's release notes.
For strafer specifically, GR00T-N1 is too large to run on the
Jetson Orin Nano end-to-end, but the dual-system *pattern* is
useful: that's what an Octo + Qwen-3B mid-level deployment would
approximate at strafer's compute budget. If staying inside the
Nvidia stack matters operationally (Isaac Sim integration, Nvidia
support), GR00T is the closest shipping example of the pattern.

#### 3.3.e Why this is the high-cost option

It's the broadest answer (a VLA at the mid-level can drive
planning, scan policy, recovery behaviors, and instance-grain
validation), but it's also the answer with the most expensive
training loop, the heaviest sim-to-real risk, and the largest
co-tenancy pressure on the Jetson stack. **Worth deferring until
either §3.2 underdelivers or the MVP-as-teacher path (§3.6) makes
the data corpus available cheaply.** Don't burn DGX cycles training
a VLA from scratch when the integration round will produce the
right corpus as a byproduct.

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
   tripwire. Two parallel signals: image-vs-image place recognition
   (case-1 wrong-room) + image-vs-target-text alignment (case-2
   wrong-instance). Either falling below threshold wakes up the
   arbiter.
2. On tripwire fire, the arbiter runs once: VLM `/describe` of the
   current view, plus a structured comparison against the
   **mission text** (LLM-as-judge: "does the current scene
   description match what the operator asked for?"). The arbiter
   does not generate an operator-facing description — it emits a
   classification (`actually_off_course` / `tripwire_false_positive`
   / `instance_mismatch_re_ground`) that the executor consumes.

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
arbiter fires rarely and decides expensively. It also handles
case 2 cleanly (the image-vs-text tripwire fires on
wrong-instance; the arbiter confirms via mission-text comparison).
It's the only option that compounds gracefully: today (no
validator) → §3.1 tripwire-only → §3.5 tripwire + arbiter → §3.2 +
§3.5 (replace the tripwire with a learned validator) → §3.3
(replace mid-level skills with a VLA, if and only if the prior
steps underdeliver).

### 3.6 Data-path options for VLA training (the ones that make §3.3 affordable)

§3.3 deferred a from-scratch VLA because training cost + data
infra were prohibitive in isolation. Four complementary data
regimes change that calculus, each with different cost /
quality / scale trade-offs. **The §3.3 framing is
data-source-agnostic; pick the regimes that fit and combine
them.**

The four regimes:

| Regime | Direction | Strength | Brief |
|---|---|---|---|
| **§3.6.a Teleop demos** | Forward (operator intent → trajectory) | Quality + operator hard negatives | [`harness-teleop-driver`](tasks/active/harness-teleop-driver.md) |
| **§3.6.b MVP-as-teacher** | Forward (autonomy stack → trajectory) | Deployment-distribution match | bridge driver in [`harness-behavior-cloning-data-expansion`](tasks/active/harness-behavior-cloning-data-expansion.md) |
| **§3.6.c In-process oracle** | Forward (mission queue → trajectory) | Parallel-env scale | [`harness-oracle-driver`](tasks/active/harness-oracle-driver.md) |
| **§3.6.d Trajectory-first captioning** | Post-hoc (trajectory → mission text) | FoV-honest labels + bulk scale + synthesized hard negatives | [`harness-trajectory-first-captioning`](tasks/active/harness-trajectory-first-captioning.md) |

The first three are *forward-generation* regimes: a mission
exists first, the driver executes it, the trajectory is the
output. The fourth is *post-hoc captioning*: the trajectory
exists first (random-but-reachable A→B), and a speaker model
generates the mission text after-the-fact. Both directions are
needed; neither replaces the other.

#### 3.6.a Teleop demos (primary; canonical)

**The candidate.** Run the harness's
[`harness-teleop-driver`](tasks/active/harness-teleop-driver.md)
mode. Operator drives the robot via gamepad through Infinigen
scenes; episodes are tagged at capture time as
`succeeded` / `failed` / `wrong_instance` / `wrong_room` /
`trajectory_violation` via dedicated buttons. Output is the
canonical schema from
[`harness-behavior-cloning-data-expansion`](tasks/active/harness-behavior-cloning-data-expansion.md).

**Why this is the recommended primary source.** Every published
wheeled-VLA (RT-2 navigation derivatives, NaVid, VLN-CE models,
NaVILA) trains on human teleop demos. This is the *canonical*
paradigm; "MVP-as-teacher distillation" is a specialty case.

**Throughput.** Single-operator, ~60 success episodes / hour;
~30 path-shape episodes / hour. Honest budgets per training
target are tabulated in
[`harness-teleop-driver`](tasks/active/harness-teleop-driver.md);
~30–40 hours of operator time gets to a v2 VLA endpoint corpus
(with hindsight-relabel + replay-perturbation multipliers, both
recommended-tier in the harness brief).

**Limitations.** Operator-paced; reflects operator play style;
path-shape data is operator-bottlenecked. The procedural
path-shape generator
([`harness-procedural-path-shape-generator`](tasks/active/harness-procedural-path-shape-generator.md))
is filed against this exact bottleneck.

#### 3.6.b MVP-as-teacher distillation (secondary; conditional on v1 stability)

**The candidate.** Treat the existing two-tier architecture (RL
controller + Nav2 + `strafer_autonomy` + `strafer_vlm`) as the
*demonstrator* once the integration round ships. Harvest its
deployment trajectories — bridge-driver mode of
[`harness-behavior-cloning-data-expansion`](tasks/active/harness-behavior-cloning-data-expansion.md)
— as `(frame_sequence, mission_text, action_sequence, outcome)`
tuples and use them to fine-tune a small VLA.

**Why secondary, not primary.** The bridge harness was designed
for end-to-end validation, not bulk training-data capture: ~6–15
FPS throughput, fragile on the current MPPI / Nav2 path quirks,
and produces a distribution that reflects *the v1 stack's*
decisions rather than what a generalist demonstrator would do.
Worth using as a *supplement* to teleop once the v1 stack is
reliable, not as the primary corpus.

**Free auxiliary signals (when the bridge driver runs).** The
harness's `mission.json` records `target_position_3d`,
`reachability`, `mission_state`. Infinigen scene metadata gives
ground-truth grounding labels — predicted-target-position vs.
true-position becomes a free auxiliary loss. `reachability`
outcomes are a free reward signal for RLHF / DPO after SFT.

**Sequencing (when this path becomes viable).**

1. Ship [`next-integration-round`](tasks/active/next-integration-round.md).
2. Ship the bridge-driver upgrades in
   [`harness-behavior-cloning-data-expansion`](tasks/active/harness-behavior-cloning-data-expansion.md)
   so the action stream is captured.
3. File `mvp-teacher-vla-distillation.md` to build the dataset
   assembly tooling and run the supplementary SFT pass against an
   existing teleop-trained checkpoint.

#### 3.6.c In-process oracle (future; for scale supplements only)

**The candidate.**
[`harness-oracle-driver`](tasks/active/harness-oracle-driver.md)
— a scripted policy in-process Isaac Lab that uses A* on the
navigable mask plus heuristics for "stop near target." Crude but
viable; trades demo quality for massive parallel throughput
(1000s of envs simultaneously).

**Why this is filed-on-trigger, not now.** Teleop + bridge cover
v1 measurement and v2's first training pass. Oracle exists for
the regime where teleop throughput is the binding constraint —
typically when you're trying to scale beyond ~10k trajectories
or ablate across many scene seeds. Don't build it preemptively.

#### 3.6.d Trajectory-first captioning (complementary regime)

**The candidate.** Drivers (typically the oracle in a "no
mission, just navigate" mode) traverse random-but-reachable
A→B paths. After-the-fact, a speaker model — Qwen2.5-VL-7B
following an instructive-voice prompt — generates
`mission_text` + paraphrases + synthesized failure-pair
negatives from the captured frames. Filed in
[`harness-trajectory-first-captioning`](tasks/active/harness-trajectory-first-captioning.md).

**Why this is the right complement to §3.6.a–c.** The
forward-generation regimes assume a mission text exists, then
ask "did the trajectory satisfy it?" Trajectory-first reverses
the question — "what mission text describes this trajectory?" —
and gets two structural advantages:

- **FoV-honest labels by construction.** The captioner sees
  only the frames the camera actually saw. It cannot annotate
  landmarks the camera missed because it has no signal that
  they exist. The egocentric-FoV problem that breaks naive
  forward-generation landmark annotation goes away.
- **No LLM-waypoint hallucination.** Paths are real; the
  speaker's hallucinations are bounded to language and
  recoverable via paraphrase ensembling. Forward-generation's
  LLM-as-planner pass has to invent waypoints that are then
  validated post-hoc; trajectory-first never invents
  waypoints.
- **Scale.** Random-A→B sampling produces trajectories at
  parallel-env throughput; the captioner runs in offline VLM
  batches. Realistic single-day output: ~10k trajectories ×
  ~15 mission-row variants each = ~150k training tuples.

**Literature precedent.** Speaker-Follower (Fried et al.,
NeurIPS 2018) is the canonical reference; HER (Andrychowicz et
al. 2017), R2R-EnvDrop, NaVid's data path, RT-2's hindsight
relabeling, and OpenVLA's caption pipeline all variants of the
same pattern. Strafer is in well-trodden territory here.

**The critical engineering constraint.** Speaker models must
generate *instructive* text (imperative voice, second-person,
future / present tense), not *descriptive* text (past tense,
third-person). A VLA trained on descriptive captions learns to
*describe* trajectories rather than *execute* operator intent
— it fails at runtime because operator inputs ("go to the
chair") look out-of-distribution. The brief gates on a held-out
instructive-quality eval before scaling.

**What it doesn't replace.** Operator-intent demos (teleop's
strength) and path-shape demonstrations (operator-typed teleop
or path-shape-biased oracle) — random A*-shortest doesn't
naturally produce wall-hugging or "via the dining room"
trajectories. Trajectory-first scales bulk and produces hard
negatives; teleop carries the operator-intent and path-shape
distributions that bulk captioning can't reach.

#### 3.6.e Caveats specific to all VLA distillation paths

- **Distillation inherits demonstrator failures.** Teleop
  inherits operator habits; bridge-driver inherits v1 stack
  failures; oracle inherits scripted-policy idiosyncrasies;
  trajectory-first inherits speaker-model phrasing biases.
  Plan an explicit "RLHF-on-failures" phase regardless of
  source.
- **Action-space tokenization is sticky.** Pick discrete bins
  (256/axis is conventional) vs. action diffusion early.
  Switching later is a from-scratch retrain.
- **Data volume.** Even hundreds of missions per scene is small
  relative to OpenVLA's training set (~970 k trajectories). Plan
  to *adapt* an existing VLA checkpoint, not train from scratch.
  Trajectory-first captioning multiplies effective volume by ~10×
  via positive + negative + paraphrase variants per trajectory.
- **Sim-to-real gap.** Distilled-from-sim VLAs notoriously fail
  on real-robot deployment without a transfer step. Applies
  equally to all four sources.

**Why this is the right shape for strafer specifically.** The
existing planner + executor + RL policy is *already* a coarse
dual-system architecture (slow planning at planner-call rate, fast
control at 30 Hz). The gap to GR00T-style VLAs is that GR00T's two
systems are jointly gradient-trained; strafer's share a *skill
schema* (the plan JSON) instead. Closing that gap with teleop
demos as the primary source, trajectory-first as the bulk
multiplier, and bridge / oracle as supplements is cheaper than
rewriting the architecture from scratch.

## Section 4 — Recommendation

**Recommendation: stage §3.1 + §3.5 in sequence, with §3.2 staged
as a conditional escalation, and §3.6 (MVP-as-teacher distillation)
opened in parallel once the integration round ships.** Do not
attempt a from-scratch VLA (§3.3 in isolation) — only enter the VLA
path via §3.6's distillation route.

Two metric axes to measure against, not one:

- **Per case** (room-grain, instance-grain, trajectory-shape) per
  §2.3. The cheap CLIP path passes case 1 plausibly, must be
  augmented with image-vs-text for case 2, and cannot do case 3
  alone.
- **Per stack layer** (tripwire, arbiter, end-to-end cascade). A
  noisy tripwire is fine if the arbiter cleans it up; an isolated
  tripwire metric without the arbiter pass is misleading.

The sequence:

1. **Wire the existing CLIP path into production.** The single
   most expensive finding in §1.3 is that none of the semantic-map
   code runs in deployment. Until that's true, no measurement is
   meaningful. Filed as
   [`docs/tasks/active/clip-mid-mission-validator-evaluation.md`](tasks/active/clip-mid-mission-validator-evaluation.md);
   prerequisite for everything else here.

2. **Run a measurement pass against a populated harness episode
   set, disaggregated by case.** Same brief. Pre-registered
   metrics, **per case**:
   - Frame-to-frame CLIP cosine σ on a same-leg subset (case-1
     diagnostic).
   - Top-1 flip rate per meter of travel (case-1 diagnostic).
   - **Case-1 TPR / FPR**: `TransitMonitor.check`'s
     decision on legs labelled `wrong_room` vs. `on_course`.
   - **Case-2 TPR / FPR**: same metric on legs labelled
     `wrong_instance` vs. `on_course`. Image-vs-image retrieval is
     expected to perform near-random here; that's the point of
     measuring it before adding image-vs-text.
   - Time-to-decision (median, p95) from leg start to first
     `abort=True`, per case.
   - Cascade-end-to-end TPR / FPR (after the §3.5 arbiter pass)
     for both cases.

3. **Score the cascade against industry-standard
   binary-classifier statistics.** Per the eval brief's
   simplified §4.1 framework: ROC-AUC + 95% bootstrap CI per
   case + signal; PR-AUC + CI; confusion matrix at production
   threshold; Brier score for calibration; McNemar's test
   comparing image-vs-image and image-vs-text tripwires; time-
   to-decision CDF; cascade end-to-end (post-arbiter) numbers.

4. **If the cascade meets ≥ 0.85 ROC-AUC lower-bound per case**,
   ship §3.5 in production. The arbiter wraps the abort with a
   single VLM `/describe` + LLM-as-judge call against mission
   text. Case 3 stays unaddressed (see prerequisite below).

5. **If the cascade fails the AUC bar**, file
   [`clip-cotrained-retrieval-augmented`](tasks/active/clip-cotrained-retrieval-augmented.md)
   to push improvements: a co-training step that fine-tunes
   CLIP with the trajectory-first speaker model, plus a
   retrieval-augmented inference step that uses the
   SemanticMapManager's memory primitive. Both steps re-run
   the same industry-standard statistics for clean ablation.
   **No "small learned validator" escalation** — that path was
   retired (§3.2 above for rationale).

6. **In parallel with steps 1–5**, once
   [`next-integration-round`](tasks/active/next-integration-round.md)
   ships, open §3.6.b (MVP-as-teacher distillation). The
   MVP's deployment traces become the VLA training corpus
   essentially for free; the resulting VLA in
   [`strafer-vla-v2-architecture`](tasks/active/strafer-vla-v2-architecture.md)
   is evaluated against the same per-case statistics as an
   end-to-end alternative to the cascade.

**Case-3 prerequisite.** Trajectory-shape constraints ("hug the
wall," "approach from the east") cannot be validated until the
*planner* decomposes them into a checkable spec. This is a
planner-side change, not a perception change. Filed in §4.3 as a
separate prerequisite brief; case-3 metrics are excluded from the
§4.1 success criteria above until that brief ships.

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
      labels each mission as `on_course` / `wrong_room` /
      `wrong_instance` / `ambiguous` (the `--root-cause-pass`
      mode), computes the per-case metrics above, and writes a
      JSON report under `data/transit_monitor_eval/<run_id>/report.json`.
- [ ] **The report uses industry-standard binary-classifier
      statistics**: ROC-AUC + 95% bootstrap CI per case + signal,
      PR-AUC + CI, confusion matrix at production threshold,
      Brier score for calibration, McNemar's test comparing
      image-vs-image and image-vs-text tripwires, time-to-decision
      CDF, cascade end-to-end (post-arbiter) numbers. The
      acceptance bar is **ROC-AUC ≥ 0.85 (lower 95% CI bound)
      per case** (≥ 0.90 for cascade end-to-end).
- [ ] One of two outcomes, recorded in a §4.4 addendum to this
      design doc:
      - **Cascade meets the bar.** Cascade ships to production
        behind `STRAFER_SEMANTIC_MAP_ENABLED`; arbiter wrapper
        is implemented inside the same brief.
      - **Cascade fails the bar.**
        [`clip-cotrained-retrieval-augmented`](tasks/active/clip-cotrained-retrieval-augmented.md)
        is filed as a research follow-up for improvements;
        no "small learned validator" escalation.

### 4.2 What's explicitly *not* recommended

- **Do not train a small frozen-head learned validator (the
  retired §3.2).** Empirically dominated by the §3.5 cascade's
  pretrained-VLM arbiter. Filed in
  [`completed/learned-mid-mission-validator`](tasks/completed/learned-mid-mission-validator.md)
  with retirement rationale. Improvements to the cascade live
  in
  [`clip-cotrained-retrieval-augmented`](tasks/active/clip-cotrained-retrieval-augmented.md);
  end-to-end alternatives live in
  [`strafer-vla-v2-architecture`](tasks/active/strafer-vla-v2-architecture.md).
- **Do not start a from-scratch small VLA (§3.3 in isolation).**
  Only enter the VLA path via §3.6's distillation route — the
  training cost is otherwise prohibitive and the data corpus
  doesn't exist outside the integration round's harness output.
- **Do not lean on §3.4 (smarter VLM scheduling) as a primary
  validator.** It compounds the existing single-loaded-model
  bottleneck and turns the VLM service into a real-time critical
  path. It belongs as the §3.5 *arbiter*, not as a standalone
  validator.
- **Do not train the §3.2 learned validator with vision-only
  inputs.** The mission-text fusion is what gives the model a
  chance at case 2. A vision-only validator is just a more
  expensive case-1 tripwire.

### 4.3 Prerequisite briefs filed alongside this recommendation

Three follow-up briefs exist in the same PR; one more is named
here as a prerequisite for the case-3 framing in §2.3:

| Brief | Status | Prerequisite for |
|---|---|---|
| [`clip-mid-mission-validator-evaluation.md`](tasks/active/clip-mid-mission-validator-evaluation.md) | Filed | All case-1 / case-2 measurements; gates §3.5 ship vs. cascade-improvements follow-up |
| [`completed/learned-mid-mission-validator.md`](tasks/completed/learned-mid-mission-validator.md) | **Retired** | Was scoped as the case-2 escalation; retired because the small-frozen-head design is structurally underpowered against the §3.5 cascade. See file for full rationale. |
| [`clip-cotrained-retrieval-augmented.md`](tasks/active/clip-cotrained-retrieval-augmented.md) | Filed (research) | Cascade improvement path. A co-training step (CLIP fine-tune with the trajectory-first speaker) plus a retrieval-augmented inference step (cross-attention over the SemanticMapManager memory). Replaces the retired learned-validator as the "what to do if the cascade underdelivers" path. |
| [`multi-room-autonomy-stack.md`](tasks/active/multi-room-autonomy-stack.md) | Filed (P1) | Lifts §1.10.1's multi-room deferral. Stored-map fallback in `scan_for_target` + planner transit-step emission. |
| [`multi-room-scene-connectivity-validation.md`](tasks/active/multi-room-scene-connectivity-validation.md) | Filed (P1) | Connectivity graph in `scene_metadata.json` + door-open guarantee at scene-gen time. Hard prerequisite for the runtime brief and the mission generator. |
| **`clip-multi-room-validator-remeasure.md`** | **To be filed when [`multi-room-autonomy-stack`](tasks/active/multi-room-autonomy-stack.md) ships** | Re-runs the v1 clip-eval metrics on multi-room data; recalibrates per-case ROC-AUC bars. |
| **`planner-trajectory-constraint-decomposition.md`** | **To be filed when case 3 becomes live** | Case-3 (trajectory-shape) validation. Without the planner emitting decomposed trajectory constraints, no validator architecture can score case 3 — there's no spec to check against. Out of scope for the current PR; filed only when a real mission requires it. |
| **`mvp-teacher-vla-distillation.md`** | **To be filed when [`next-integration-round`](tasks/active/next-integration-round.md) ships** | §3.6.b distillation path. Depends on the integration round producing the action-labeled corpus; can run in parallel with cascade improvements. |

## Cross-references

- Current-round design master:
  [`docs/STRAFER_AUTONOMY_NEXT.md`](STRAFER_AUTONOMY_NEXT.md), §0.1
  (verify_arrival), §0.6 (transit monitoring via BackgroundMapper),
  §1.4 (unified CLIP embedding strategy), §1.10.1 (formerly
  multi-room deferral; lifted by
  [`multi-room-autonomy-stack`](tasks/active/multi-room-autonomy-stack.md);
  multi-room is the MVP default going forward) for the original
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
  [`clip-cotrained-retrieval-augmented.md`](tasks/active/clip-cotrained-retrieval-augmented.md).
  (`learned-mid-mission-validator` retired; see
  [`completed/learned-mid-mission-validator.md`](tasks/completed/learned-mid-mission-validator.md).)
