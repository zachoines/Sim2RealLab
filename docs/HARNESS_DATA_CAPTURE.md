# Harness data capture

End-to-end setup + run guide for `Scripts/capture.py` — the unified
harness data-capture entry point per
[`harness-architecture.md`](tasks/active/harness/harness-architecture.md).
One CLI, two flags (`--driver` × `--mission-source`), one LeRobot v3
dataset per scene under `data/sim_in_the_loop/<scene_name>/`.

Tier 1 wires `(teleop, scene-metadata)` end-to-end. Other cells raise
`NotImplementedError` with a pointer to the tier that ships them.

This guide is the source of truth for harness operator workflows; the
[cheatsheet](example_commands_cheatsheet.md) keeps a short pointer
to it under "Harness data capture" but no longer carries the full
procedure.

---

## One-time env setup

First bring up the Isaac Lab Python env per
[`docs/DGX_SPARK_SETUP.md`](DGX_SPARK_SETUP.md) — that runbook covers
Miniconda, Isaac Sim 5.1, the aarch64 PyTorch wheel, and Isaac Lab
itself. The harness layers `lerobot` into that env.

Why `--no-deps`: the Isaac Lab env ships with `torch 2.10.0+cu130` +
`numpy 2.3.1` + `huggingface-hub 0.36`. `lerobot 0.5.1` pins are mostly
compatible except for `numpy`, `huggingface-hub`, and `rerun-sdk` — a
normal `pip install lerobot` would downgrade numpy (risks breaking
Isaac Sim) and major-bump huggingface-hub (risks breaking
transformers). Install `--no-deps` and layer only the runtime deps the
writer actually uses:

```bash
conda activate <your-isaac-lab-env>   # the one created in DGX_SPARK_SETUP.md

# Note: $ISAACLAB is an Isaac Lab wrapper that only forwards args after
# its own flags — it can't run `-m pip` directly. Use the env's
# Python (`python -m pip` works once you're inside the env).

# 1. Install lerobot core without dragging its strict pins in
python -m pip install --no-deps "lerobot==0.5.1"

# 2. Install only the runtime deps StraferLeRobotWriter uses, refusing
#    to upgrade anything that's already installed and satisfies the new pin
python -m pip install --upgrade-strategy only-if-needed \
    "datasets>=4.0.0,<5.0.0" \
    "av>=15.0.0,<16.0.0" \
    "jsonlines>=4.0.0,<5.0.0"

# 3. Verify lerobot imports + Isaac Sim's torch still has CUDA
python -c "import torch, lerobot; print('torch', torch.__version__, 'lerobot', lerobot.__version__, 'cuda', torch.cuda.is_available())"
# Expected: torch 2.10.0+cu130 lerobot 0.5.1 cuda True
```

Pip will print warnings that lerobot's strict pins on `numpy`, `huggingface-hub`,
`rerun-sdk`, `setuptools`, `packaging`, and a few `wandb` / `pynput` /
`pyserial` / `termcolor` deps aren't satisfied. **Those warnings are
expected and safe to ignore** — `--no-deps` deliberately skipped them
to keep Isaac Sim's stack intact. The narrow LeRobot v3 writer surface
the harness actually uses (`LeRobotDataset.create / add_frame /
save_episode / finalize`) was end-to-end smoke-tested against this
install and works.

Pure-Python unit tests (writer / depth / mission picker / button
translator / CLI dispatch / scene-path resolver) run in `.venv_harness`,
isolated from the Isaac Lab runtime stack:

```bash
make test-harness
```

---

## Infinigen scene corpus

The mission picker reads `Assets/generated/scenes/<scene>/scene_metadata.json`.
A scene's USDC export and its metadata sidecar are siblings inside the
scene directory; `prep_room_usds.py` authors both.

**Infinigen is one provider, not the only one.** The teleop harness
consumes three artifacts (`scene_metadata.json`, a `<scene>.usdc`, the
combined `scenes_metadata.json`) and never imports Infinigen at runtime.
See [`SCENE_PROVIDER_CONTRACT.md`](SCENE_PROVIDER_CONTRACT.md) for the
general interface — the field-by-field schemas, the postprocess CLI
override surface, and the adapter-writer's checklist for bringing in a
second source (downloaded packs, hand-authored maps, ProcTHOR /
Habitat / Cosmos exports). The contract was written to accommodate two
in-flight consumers without re-shipping:
[`mission-text-enrichment`](tasks/parked/harness/mission-text-enrichment.md)
(reserves the `objects[].descriptors` namespace + a populated `rooms[]`
block) and
[`scene-metadata-in-usd`](tasks/active/harness/scene-metadata-in-usd.md)
(the same contract with a USD `customData` storage backend).

### Clean-slate scene regeneration

When the picker offers objects that aren't actually present in the
loaded scene (e.g. selecting a "bed" places the target marker into
mid-air), the per-scene `scene_metadata.json` is stale relative to the
USDC. Regenerate from a clean slate:

```bash
cd ~/Workspace/Sim2RealLab

# 1. Archive the existing scenes/ tree (don't delete — easy rollback)
mv Assets/generated/scenes Assets/generated/scenes.old.$(date +%Y%m%d)

# 2. Regenerate one high-quality scene from a fresh seed
python source/strafer_lab/scripts/prep_room_usds.py generate \
    --config high_quality_dgx \
    --num-scenes 1 \
    --output Assets/generated/scenes
# ~hours; Infinigen scene synthesis is mostly single-threaded (Blender
# spawns parallel workers for mesh ops, but the constraint solver loop
# is GIL-bound). Run two seeds in parallel on multi-core hosts.

# 3. Extract per-scene metadata (objects[])
SCENE=$(ls -d Assets/generated/scenes/scene_*/ | head -1 | xargs basename)
$ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \
    --from-usd \
    --usd    Assets/generated/scenes/${SCENE}.usdc \
    --output Assets/generated/scenes/${SCENE}

# 4. Author the combined scenes_metadata.json (spawn_points_xy + floor_top_z)
python source/strafer_lab/scripts/generate_scenes_metadata.py \
    --scenes-dir Assets/generated/scenes

# 5. Sanity check
python -c "
import json
d = json.load(open(f'Assets/generated/scenes/${SCENE}/scene_metadata.json'))
print(f'rooms={len(d.get(\"rooms\",[]))}  objects={len(d.get(\"objects\",[]))}')
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

### Extract scene_metadata.json (one-time per existing scene)

If you have only the `.usdc` (no Blender / in-process Infinigen
`State`), parse it from prim names:

```bash
SCENE=scene_high_quality_dgx_000_seed1

$ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \
    --from-usd \
    --usd    Assets/generated/scenes/${SCENE}.usdc \
    --output Assets/generated/scenes/${SCENE}

# Sanity check — should be non-empty
python -c "
import json
d = json.load(open('Assets/generated/scenes/${SCENE}/scene_metadata.json'))
print(f'rooms={len(d.get(\"rooms\",[]))}  objects={len(d.get(\"objects\",[]))}')
for o in d['objects'][:10]:
    print(' -', o.get('label'), o.get('instance_id'))
"
```

**Known limitation:** `--from-usd` cannot recover room polygons, so
the picker shows `rooms=0` for these scenes. Room semantics (which
hard-negative button chord maps to "wrong_room") still work; the
operator commits to the failure mode at capture time. For full room
geometry, run from a Blender stage or extract from the in-process
Infinigen `State`. Tracked in
[`docs/tasks/active/harness/infinigen-scene-corpus.md`](tasks/active/harness/infinigen-scene-corpus.md).

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
$ISAACLAB -p Scripts/capture.py \
    --driver teleop --mission-source scene-metadata \
    --scene  ${SCENE} \
    --output ${OUT} \
    --fps 8 \
    --max-episodes 5 \
    --hide-overhead
```

### Egocentric mode (first-person classic controls)

```bash
$ISAACLAB -p Scripts/capture.py \
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
| `--scene-usd <path>` | (env default) | Override the scene USD. The driver re-derives `--scene-metadata` from the USD's sibling `scene_metadata.json` so target labels match the geometry |
| `--viz kit,rerun` | `kit` (forced) | AppLauncher pass-through. The driver forces `--viz kit` if you don't pass one — the editor viewport is required for teleop |
| `--device cpu` | `cuda:0` | AppLauncher pass-through |

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
| `scene_metadata.json not found` | Scene hasn't been extracted | Run the extraction step above |
| `--output already exists` | `LeRobotDataset.create` refuses to overwrite; bash variables persist across runs in the same shell, so a stale `$OUT` is the usual culprit | Re-stamp `$RUN_ID`/`$OUT` per run, or rely on the driver's auto-suffix fallback |
| `No gamepad detected` | pygame can't find the joystick | `jstest /dev/input/js0` to confirm the kernel sees it |
| Wrong button does the wrong thing | Controller-family auto-detect picked wrong | Add `--family-override ps5` (or `xbox` / `switch`) |
| `cv2.error: ... The function is not implemented. Rebuild the library with ... GTK+ ...` on `cv2.namedWindow` | The installed opencv has no GUI backend (the `-headless` variant) | The driver degrades to "PIP off" automatically — capture continues; use the Isaac Sim editor viewport as your live view. Pass `--no-pip-window` to silence the warning |
| Top-down view shows roof / can't see the robot | Infinigen exports ceiling + exterior-hull prims | Add `--hide-overhead`. If geometry still occludes, override the matcher with `--overhead-regex 'your_pattern'` |
| Round-trip via HF `LeRobotDataset` fails with a codec error | `torchcodec` missing on aarch64 | The wheel marker excludes aarch64; LeRobot falls back to PyAV which works. If you've manually installed `torchcodec`, uninstall it |
