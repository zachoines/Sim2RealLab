# Harness data capture

End-to-end setup + run guide for `source/strafer_lab/scripts/capture.py` — the unified
harness data-capture entry point per
[`harness-architecture.md`](tasks/active/harness/harness-architecture.md).
One CLI, two flags (`--driver` × `--mission-source`), one LeRobot v3
dataset per scene under `data/sim_in_the_loop/<scene_name>/`.

`(teleop, scene-metadata)`, `(bridge, scene-metadata)`, `(bridge, queue)`,
and `(scripted, coverage)` are wired end-to-end; `(scripted, coverage)` is
the **bulk-capture default**. The remaining scripted `queue` / `captioner`
cells raise `NotImplementedError` until those mission sources ship.

This guide is the source of truth for harness operator workflows; the
[cheatsheet](example_commands_cheatsheet.md) keeps a short pointer
to it under "Harness data capture" but no longer carries the full
procedure.

---

## One-time env setup

Bring up the `env_isaaclab3` conda env per
[`source/strafer_lab/README.md` → Install (Linux / DGX Spark)](../source/strafer_lab/README.md#install)
— that recipe installs Isaac Sim 6, Isaac Lab, **and** the `--no-deps`
`lerobot` layering the harness writer needs (with the rationale for why
`lerobot` is installed `--no-deps`). Confirm the env is capture-ready:

```bash
conda activate env_isaaclab3
python -c "import torch, lerobot; print('torch', torch.__version__, 'lerobot', lerobot.__version__, 'cuda', torch.cuda.is_available())"
# Expected: torch 2.10.0+cu130 lerobot 0.5.1 cuda True
```

Pure-Python unit tests (writer / depth / mission picker / button
translator / CLI dispatch / scene-path resolver) run in `env_isaaclab3`
without booting Kit:

```bash
make test-lab-pure
```

---

## Infinigen scene corpus

The mission picker reads the per-scene metadata **embedded in the scene
USD's `customData`** (key `strafer_scene_metadata`) — there is no sidecar
file. `prep_room_usds.py generate` authors it into the USD at
generation time, so the metadata can never be paired with a stale or
missing file; the reader hard-errors on a USD that carries none.

**Infinigen is one provider, not the only one.** The teleop harness
consumes the per-scene metadata embedded in `<scene>.usdc` plus the
combined `scenes_metadata.json` manifest, and never imports Infinigen at
runtime. See [`SCENE_PROVIDER_CONTRACT.md`](SCENE_PROVIDER_CONTRACT.md)
for the general interface — the field-by-field schemas, the postprocess
CLI override surface, and the adapter-writer's checklist for bringing in
a second source (downloaded packs, hand-authored maps, ProcTHOR /
Habitat / Cosmos exports). The contract accommodates the next consumer,
[`mission-text-enrichment`](tasks/parked/harness/mission-text-enrichment.md)
(reserves the `objects[].descriptors` namespace + a populated `rooms[]`
block), without re-shipping; the
[`scene-metadata-in-usd`](tasks/completed/scene-metadata-in-usd.md) move
to `customData` is the same contract with a different storage backend.

### Clean-slate scene regeneration

When the picker offers objects that aren't actually present in the
loaded scene (e.g. selecting a "bed" places the target marker into
mid-air) — which now requires an un-regenerated scene, since fresh
scenes embed their metadata at generation — regenerate from a clean
slate. `generate` chains the metadata + detection-label authoring **and**
the room-connectivity step (occupancy grid + verified `connectivity[]`
graph + door-open guarantee — see
[`SCENE_PROVIDER_CONTRACT.md`](SCENE_PROVIDER_CONTRACT.md) §b-conn), so
the only separate step is the combined manifest:

```bash
cd ~/Workspace/Sim2RealLab

# 1. Archive the existing scenes/ tree (don't delete — easy rollback)
mv Assets/generated/scenes Assets/generated/scenes.old.$(date +%Y%m%d)

# 2. Regenerate one high-quality scene from a fresh seed. This embeds the
#    per-scene metadata + UsdSemantics detection labels into the USD.
#    Needs $ISAACLAB set (the metadata pass uses the Kit-only schema).
python source/strafer_lab/scripts/prep_room_usds.py generate \
    --config high_quality_dgx \
    --num-scenes 1 \
    --output Assets/generated/scenes
# ~hours; Infinigen scene synthesis is mostly single-threaded (Blender
# spawns parallel workers for mesh ops, but the constraint solver loop
# is GIL-bound). Run two seeds in parallel on multi-core hosts.

# 3. Author the combined scenes_metadata.json (spawn_points_xy + floor_top_z)
SCENE=$(ls Assets/generated/scenes/scene_*.usdc | head -1 | xargs basename | sed 's/\.usdc$//')
python source/strafer_lab/scripts/generate_scenes_metadata.py \
    --scenes-dir Assets/generated/scenes

# 4. Sanity check (reads the metadata back from the USD customData)
$ISAACLAB -p -c "
from strafer_lab.tools.scene_metadata_reader import load
d = load('Assets/generated/scenes/${SCENE}.usdc')
print(f'rooms={len(d.get(\"rooms\",[]))}  objects={len(d.get(\"objects\",[]))}')
import json
c = json.load(open('Assets/generated/scenes/scenes_metadata.json'))
print(f'scenes_metadata entries: {sorted(c[\"scenes\"].keys())}')"
```

Use the new `$SCENE` for your subsequent capture sessions. The
extractor drops degenerate (0,0,0) entries (Infinigen creature pre-
placement prims + USD bbox-fallback rows) so the picker only ever sees
spatially-valid targets.

If the picker still offers "missing" objects after this clean-slate,
the issue is in `extract_scene_metadata.py --from-usd`'s prim-name
parser, not the teleop driver.

### Re-author embedded metadata (one-time per existing scene)

A scene generated before metadata-in-USD landed (or any bare `.usdc`)
carries no `customData`. Re-author it from the USD's prim names — this
embeds the metadata + applies the `UsdSemantics` detection labels:

```bash
SCENE=scene_high_quality_dgx_000_seed1

$ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \
    --from-usd \
    --usd Assets/generated/scenes/${SCENE}.usdc

# Sanity check — should be non-empty (reads back from customData)
$ISAACLAB -p -c "
from strafer_lab.tools.scene_metadata_reader import load
d = load('Assets/generated/scenes/${SCENE}.usdc')
print(f'rooms={len(d.get(\"rooms\",[]))}  objects={len(d.get(\"objects\",[]))}')
for o in d['objects'][:10]:
    print(' -', o.get('label'), o.get('instance_id'))
"
```

**Known limitation:** `--from-usd` cannot recover room *polygons*, so
the picker shows `rooms=0` for these scenes right after this step. Room
semantics (which hard-negative button chord maps to "wrong_room") still
work; the operator commits to the failure mode at capture time. For full
room polygons, run from a Blender stage or extract from the in-process
Infinigen `State`. Tracked in
[`docs/tasks/active/harness/infinigen-scene-corpus.md`](tasks/active/harness/infinigen-scene-corpus.md).

The connectivity step below back-fills a *rectangular* `rooms[]` (one
axis-aligned footprint per floor mesh, with `room_type` + `story`) when
the metadata has none — enough for room indexing and the connectivity
graph, though coarser than the true constraint-solver polygons.

### Re-author the connectivity graph (one-time per existing scene)

After metadata is embedded, generate the occupancy grid + verified
`connectivity[]` graph and force doors open (idempotent; re-run after any
geometry change):

```bash
$ISAACLAB -p source/strafer_lab/scripts/validate_scene_connectivity.py \
    --usd Assets/generated/scenes/${SCENE}.usdc

# Inspect without authoring (prints the connectivity matrix, leaves the USD
# untouched): add --no-write.
```

This caches `<scene>/occupancy.npy` (+ `occupancy.json`) next to the scene
and merges `connectivity[]` + `multi_story` into the USD `customData`. If
the occupancy-map extension is unavailable, add `--rasterize-fallback`.

---

## Generate a mission queue (free-text targets at scale)

The capture drivers consume a `mission_queue.yaml` via `--mission-source
queue`. Hand-authoring one is fine for a handful of missions; for scale,
`build_mission_corpus.py` generates them from the embedded scene metadata.
Per scene it reads `rooms[]` + `objects[]` + the verified `connectivity[]`
graph, loads the cached occupancy grid, and emits one row per reachable
target: free-text `mission_text`, paraphrases, and an oracle `planned_path`
routed through the one shared planner (`path_planner.plan_path`). Cross-room
missions are the default on multi-room scenes; same-room only where no
cross-room pair is reachable. Each row round-trips through
`mission_queue.load_mission_queue`.

```bash
# All discoverable scenes -> per-scene data/mission_queues/<scene>/queue.yaml
# + a unioned data/mission_queues/corpus.yaml. Pure-Python (numpy + pxr to
# read the USD); no Kit boot.
$STRAFER_ISAACLAB_PYTHON source/strafer_lab/scripts/build_mission_corpus.py \
    --mode mixed

# One scene, endpoint missions only (no path-shape language, no LLM):
$STRAFER_ISAACLAB_PYTHON source/strafer_lab/scripts/build_mission_corpus.py \
    --scenes scene_high_quality_dgx_000_seed2 --mode endpoint
```

`--mode {endpoint, path-shape, mixed}` selects the language mix; `mixed` is
the default. The LLM-as-planner waypoint pass (`--use-planner-llm`), the LLM
paraphrase pass (`--use-paraphrase-llm`), and the start-frame VLM grounding
pass (`--ground-start-frame`) are opt-in and need a GPU; with them off the
generator falls back to the clean oracle path + templated paraphrases +
skipped grounding, so it runs headless. Generated queues are cached under
`data/mission_queue_cache/<scene>/<scene_seed>.json` keyed by the generator
version + few-shot-template hash + LLM seed, so a re-run under an unchanged
template is free and a template change invalidates rather than reuses.

The occupancy grid must match the scene USD it was built from — the tool
hard-errors on a stale grid (re-run the connectivity validation above to
regenerate it). Pass `--allow-stale-occupancy` only to knowingly proceed
against a grid the latest postprocess pass has not yet been re-baked into.

Then capture against the generated queue:

```bash
$ISAACLAB -p source/strafer_lab/scripts/capture.py \
    --driver bridge --mission-source queue \
    --mission-queue data/mission_queues/${SCENE}/queue.yaml \
    --scene ${SCENE} --output data/sim_in_the_loop/${SCENE}_${RUN_ID}
```

---

## Validation capture (small batch — driver wiring works)

**Important:** the `--output` path must NOT exist (LeRobot v3
`LeRobotDataset.create()` refuses to overwrite). Use a fresh timestamp
on every invocation, OR rely on the driver's auto-suffix fallback
(if the path exists, the driver appends `_YYYYMMDDTHHMMSS` and prints
the resolved path it actually used).

```bash
SCENE=scene_high_quality_dgx_000_seed1

# Re-stamp $RUN_ID + $OUT on every invocation — bash variables persist
# across runs in the same shell, so a stale $OUT is the #1 trip-up.
RUN_ID=$(date +%Y%m%dT%H%M%S)
OUT=data/sim_in_the_loop/${SCENE}_validation_${RUN_ID}

# World-arcade twin-stick (default), overhead structure hidden, target marker on
$ISAACLAB -p source/strafer_lab/scripts/capture.py \
    --driver teleop --mission-source scene-metadata \
    --scene  ${SCENE} \
    --output ${OUT} \
    --fps 8 \
    --max-episodes 5 \
    --hide-overhead
```

### Egocentric mode (first-person classic controls)

```bash
$ISAACLAB -p source/strafer_lab/scripts/capture.py \
    --driver teleop --mission-source scene-metadata \
    --scene  ${SCENE} \
    --output ${OUT} \
    --fps 8 \
    --max-episodes 5 \
    --control-mode egocentric
```

The Kit viewport possesses the perception camera's prim path in
egocentric mode — no separate follow-cam render pass. Stick-up moves
the robot forward in whatever direction it's facing, regardless of
world frame.

### Button mapping (per
[`harness-architecture.md` §Episode-end button mapping](tasks/active/harness/harness-architecture.md#episode-end-button-mapping-teleop-only))

| Button | `outcome` | Kept? |
|---|---|---|
| `Y` (triangle / north) | `succeeded` | yes |
| `B` (circle / east) | `failed` | yes |
| `X` + D-pad ↑/↓ | `wrong_instance` hard negative | yes |
| `X` + D-pad ←/→ | `wrong_room` hard negative | yes |
| `SELECT` (share / minus) | `trajectory_violation` | yes |
| `Back` (view) | discard | **no** |
| `A` (tap) | toggle `REC` ↔ `PAUSED` | — |
| `Start` (hold ≥ 1 s) | save + quit cleanly | — |

`X` alone (no D-pad direction) is not committal — push the D-pad to
commit. Between episodes the driver re-prompts via the console mission
picker (numeric index; Ctrl-D quits cleanly).

### Optional flags

| Flag | Default | Use |
|---|---|---|
| `--control-mode world_arcade` | `world_arcade` | Top-down editor viewport, stick = world-frame velocity (today's default; arcade twin-stick) |
| `--control-mode egocentric` | — | Kit viewport possesses the perception camera prim (no follow-cam render pass); stick = body-frame velocity (classic first-person controls). Useful for CLIP-style coverage where the operator needs to see what the robot sees |
| `--hide-overhead` | off | At startup, set ceiling / roof / attic / exterior prims (Infinigen's overhead structure naming) to invisible in the editor viewport. Use with `--control-mode world_arcade` — these prims otherwise occlude the top-down view. Does NOT modify the scene USDC on disk; reset on next launch. `--overhead-regex 'pattern'` overrides the matcher for stubborn scenes |
| `--no-target-marker` | (marker on) | Suppress the green debug-draw sphere at the active target's position. Marker is operator-only (debug-draw is outside Replicator's render product) so it never enters captured frames; this flag is just for visual quiet |
| `--capture-rate-hz 8` | matches `--fps` | Writer sample rate, decoupled from the env step rate. Env still steps every sim tick; writer only calls `add_frame` every `round(env_step_hz / capture_rate_hz)` ticks. Raise env_step_hz for smoother viewport without inflating dataset sample count |
| `--no-pip-window` | (PIP on) | Suppress the cv2 first-person preview window |
| `--no-capture-policy-cam` | (policy cam on) | Drop the 80×60 policy camera (RGB + depth sidecar). Smaller dataset, but downstream policies that mirror the RL observation pipeline need it |
| `--operator-handle <name>` | none | Stamped on every episode for multi-operator runs |
| `--target-label-filter chair table` | none | Narrow the picker list |
| `--max-steps-per-episode 1500` | 1500 | Auto-close cap (logs `outcome=failed`) |
| `--scene-usd <path>` | (env default) | Override the scene USD. The driver reads the mission targets from that USD's embedded `customData`, so target labels always match the geometry |
| `--viz kit,rerun` | `kit` (forced) | AppLauncher pass-through. The driver forces `--viz kit` if you don't pass one — the editor viewport is required for teleop |
| `--device cpu` | `cuda:0` | AppLauncher pass-through |

---

## Bridge driver (autonomy stack in the loop)

The bridge driver records the same LeRobot v3 schema while the
**Jetson autonomy stack** drives over `/cmd_vel` — same Nav2 /
RTAB-Map / executor that runs against the real D555. Prerequisites:
the sim-in-the-loop bringup is healthy end-to-end
([`INTEGRATION_SIM_IN_THE_LOOP.md`](INTEGRATION_SIM_IN_THE_LOOP.md)
Stages 1–4: DDS discovery, Jetson `bringup_sim_in_the_loop.launch.py`,
the executor, and the VLM/planner services).

```bash
RUN_ID=$(date +%Y%m%dT%H%M%S)
OUT=data/sim_in_the_loop/${SCENE}_bridge_${RUN_ID}

# Walk every embedded scene target (one mission = one episode)
$ISAACLAB -p source/strafer_lab/scripts/capture.py \
    --driver bridge --mission-source scene-metadata \
    --scene  ${SCENE} \
    --output ${OUT} \
    --headless --enable_cameras

# Or walk a curated mission queue (bridge ignores planned_path)
$ISAACLAB -p source/strafer_lab/scripts/capture.py \
    --driver bridge --mission-source queue \
    --mission-queue data/mission_queues/${SCENE}/queue.yaml \
    --scene  ${SCENE} \
    --output ${OUT} \
    --headless --enable_cameras
```

Bridge-specific behavior and flags:

| Flag | Default | Use |
|---|---|---|
| `--sensors` | `bridge` preset (`rgb_full,depth_full,depth_policy`) | Per-session sensor stack; `rgb_full` + `depth_full` are mandatory (the Jetson navigates on the bridged camera streams) |
| `--detections` / `--no-detections` | on | Replicator `bbox_2d_tight` detections as first-class `observation.detections.*` columns + `meta/detection_labels.json` |
| `--inject-bad-grounding {off,wrong_room,wrong_instance,wrong_object}` | `off` | Hard-negative goal perturbation: the dispatched goal is swapped while the recorded mission text keeps naming the original target. `wrong_room` = different room; `wrong_instance` = same label, same room; `wrong_object` = different label, same room (category confusion). The in-room modes fall back to `wrong_room` when their candidate is absent. Pair with `--inject-bad-grounding-prob` (default 0.3). Downstream filters must key off the per-episode `injection_mode_actual`, not `injection_mode` |
| `--max-missions N` (pass-through) | all | Cap the mission stream — use `--max-missions 1` for a single-mission smoke |
| `--cmd-vel-grace S` (pass-through) | 30 | Mid-drive `/cmd_vel` silence beyond this discards the episode (`outcome` never reaches disk; the index slot is reused) |
| `--mission-timeout-s S` (pass-through) | 60 | Per-mission ceiling before the harness cancels (kept as `outcome=failed`) |

Episodes are **discarded** (never reach disk) on `/cmd_vel` silence
past the grace, executor unreachability (5 consecutive status-poll
failures), externally-killed terminal states (`cancelled` /
`aborted`), or a mid-mission crash; the run aborts after 3
consecutive crashed missions. `succeeded` / `failed` / `timeout`
missions are kept and labelled.

### Jetson-free smoke (`make harness-smoke`)

Before standing up the Jetson, you can exercise the bridge capture
**code path** — the writer lifecycle, depth sidecars, the detections
annotator + columns, and the discard path — with a scripted `/cmd_vel`
sweep and a fake executor, no ROS:

```bash
make harness-smoke                          # defaults: scene_true_singleroom_000_seed0
SCENE=<scene> REQUIRE_DETECTIONS=1 make harness-smoke
```

It drives one `succeeded` mission (kept) and one `cancelled` mission
(must not reach disk), then re-opens the dataset and asserts the
episode count, the strafer extension columns, the depth sidecars, and
that the `observation.detections.*` columns are present. It does **not**
bring up the ROS publishers or a live executor — that's the operator
acceptance run above.

---

## Scripted coverage driver (bulk-capture default)

The coverage driver is the **default bulk-capture path**: the trained RL
subgoal-follower drives a deterministic geometric coverage traversal that
visits every room repeatedly from spread approach headings. The diverse
same-place / different-heading views are the training signal for the
place-recognition head, the backbone bakeoff, and the eval harness — a
goal-reaching teleop demo set under-samples them irrecoverably. Teleop
(annotator) and the bridge (Jetson-in-loop) are the non-bulk paths.

```bash
RUN_ID=$(date +%Y%m%dT%H%M%S)
OUT=data/sim_in_the_loop/${SCENE}_coverage_${RUN_ID}

# Export the trained rsl_rl checkpoint to a load_policy-consumable artifact
# (one-time per checkpoint; coverage loads the EXPORTED .pt, not model_*.pt):
$ISAACLAB -p source/strafer_lab/scripts/export_policy.py \
    --checkpoint logs/rsl_rl/strafer_navigation/<run>/model_<step>.pt \
    --output models/strafer_nocam_subgoal --variant NOCAM_SUBGOAL

$ISAACLAB -p source/strafer_lab/scripts/capture.py \
    --driver scripted --mission-source coverage \
    --scene  ${SCENE} \
    --output ${OUT} \
    --policy-variant nocam_subgoal \
    --checkpoint models/strafer_nocam_subgoal.pt \
    --headless --enable_cameras
```

Coverage-specific flags:

| Flag | Default | Use |
|---|---|---|
| `--policy-variant` | `nocam_subgoal` | `PolicyVariant` selecting the trained subgoal-follower's observation contract. `nocam_subgoal` is the only variant with a trained checkpoint today; the depth subgoal variant is planned and swaps in through the same `load_policy` path |
| `--checkpoint` | (required) | Exported policy artifact (TorchScript `.pt` / `.onnx` from `export_policy.py`), not a raw `model_*.pt` training checkpoint |
| `--coverage-visits-per-room` | 2 | Minimum visits scheduled per room, each from a different approach heading |
| `--held-out-scenes a,b` | none | When `--scene` is in the set, every episode is tagged `episode_split=held_out_seeds` and a whole-scene row is written to `meta/splits.jsonl` (home-to-home generalization split) |
| `--detections` / `--no-detections` | on | Same Replicator detections columns as the bridge driver |
| `--seed` | 0 | Seeds the coverage plan so a run reproduces |

Artifacts: the LeRobot v3 dataset, `meta/strafer_episodes.parquet` (with
the per-episode `realized_d555_mount_quat` column), the embedded
`meta/scenes/<scene>/scene_metadata.json` sidecar, and — for held-out
scenes — `meta/splits.jsonl`. The live capture runs on the GPU; the
`make harness-smoke` Kit gate exercises the shared writer code path.

**Detection vocab:** the smoke reports the detection vocab and, with
`--require-detections` / `REQUIRE_DETECTIONS=1`, treats an empty one as a
hard failure. A scene regenerated through `prep_room_usds generate` (or
re-authored via `extract_scene_metadata --from-usd`) carries the
`UsdSemantics.LabelsAPI` (`"class"`) labels the Replicator
`bounding_box_2d_tight` annotator boxes on, so the vocab is non-empty.
Run `make harness-smoke REQUIRE_DETECTIONS=1` on a regenerated scene as
the detections gate. An empty vocab now means the scene predates the
metadata-in-USD authoring (re-author it) or the spawn faced empty space;
the capture path — annotator wiring, `parse_bbox_data`, the padded
`observation.detections.*` columns, and the dataset round-trip — is
exercised regardless.

---

## Round-trip verification

```bash
# Re-open the dataset via stock LeRobotDataset + read the strafer sidecar
$ISAACLAB -p -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from strafer_lab.tools.lerobot_writer import read_strafer_episodes
from pathlib import Path
root = Path('${OUT}')
d = LeRobotDataset(repo_id='strafer/${SCENE}', root=root)
print(f'episodes={d.num_episodes}  frames={len(d)}')
for row in read_strafer_episodes(root):
    print(f'  ep{row[\"episode_index\"]}  outcome={row[\"outcome\"]:22s}  '
          f'target={row[\"target_label\"]:20s}  git_sha={row[\"capture_git_sha\"][:8]}')
"

# Spot-check a saved frame for PIP-overlay contamination (hard acceptance bar)
ffmpeg -y -i ${OUT}/videos/chunk-000/observation.images.perception/file-000000.mp4 \
       -frames:v 1 /tmp/capture_smoke_frame.png
xdg-open /tmp/capture_smoke_frame.png   # must NOT show [REC] / step / distance overlay
```

---

## Production capture (≥ 30 episodes, multi-scene)

Per the
[harness architecture brief's](tasks/active/harness/harness-architecture.md)
acceptance bar — run after the
[Infinigen scene-corpus brief](tasks/active/harness/infinigen-scene-corpus.md)
generates richer scenes. Same command pattern; raise `--max-episodes`
and `--max-steps-per-episode`, and commit a summary under
`docs/artifacts/teleop_acceptance/<run_id>/`.

---

## Troubleshooting

| Symptom | Most likely cause | What to check |
|---|---|---|
| `ModuleNotFoundError: No module named 'lerobot'` | The Isaac Lab env doesn't have lerobot yet | Run the one-time env setup above |
| `SceneMetadataError: ... carries no embedded scene metadata` | The scene USD predates metadata-in-USD (no `customData`) | Regenerate via `prep_room_usds generate`, or re-author with `extract_scene_metadata --from-usd` (above) |
| `--output already exists` | `LeRobotDataset.create` refuses to overwrite; bash variables persist across runs in the same shell, so a stale `$OUT` is the usual culprit | Re-stamp `$RUN_ID`/`$OUT` per run, or rely on the driver's auto-suffix fallback |
| `No gamepad detected` | pygame can't find the joystick | `jstest /dev/input/js0` to confirm the kernel sees it |
| Wrong button does the wrong thing | Controller-family auto-detect picked wrong | Add `--family-override ps5` (or `xbox` / `switch`) |
| `cv2.error: ... The function is not implemented. Rebuild the library with ... GTK+ ...` on `cv2.namedWindow` | The installed opencv has no GUI backend (the `-headless` variant) | The driver degrades to "PIP off" automatically — capture continues; use the Isaac Sim editor viewport as your live view. Pass `--no-pip-window` to silence the warning |
| Top-down view shows roof / can't see the robot | Infinigen exports ceiling + exterior-hull prims | Add `--hide-overhead`. If geometry still occludes, override the matcher with `--overhead-regex 'your_pattern'` |
| Round-trip via HF `LeRobotDataset` fails with a codec error | `torchcodec` missing on aarch64 | The wheel marker excludes aarch64; LeRobot falls back to PyAV which works. If you've manually installed `torchcodec`, uninstall it |
