# Richer Infinigen scene corpus for harness data capture

**Type:** new feature (scene-generation pipeline)
**Owner:** DGX agent
**Priority:** P1 — gates the [`harness-architecture`](harness-architecture.md) Tier 1 acceptance run (≥ 30 episodes on ≥ 2 scenes) and every later corpus capture.
**Estimate:** M (generate N scenes via existing `prep_room_usds.py` + `extract_scene_metadata.py` + verify; no new code expected).
**Branch:** `task/harness-infinigen-scene-corpus`

## Story

As a **DGX operator who needs to run real teleop capture sessions against the harness** I want **a corpus of ≥ 4 Infinigen scenes with full `scene_metadata.json` (rooms + objects + room polygons) under `Assets/generated/scenes/`** so that **the harness teleop driver's mission picker has interesting targets across multiple scene seeds, and the brief's "≥ 30 episodes on ≥ 2 scenes" acceptance bar is actually achievable.**

## Context bundle

- [`harness-architecture.md` §Driver: teleop](harness-architecture.md#driver-teleop) — consumes the scenes via the `scene-metadata` mission source.
- [`harness-architecture.md` §Implementation tiers → Tier 1](harness-architecture.md#tier-1--writer--teleop-driver-pr-b) — names the ≥ 30 ep × ≥ 2 scene acceptance.
- [`context/conventions.md`](../../context/conventions.md) — commit / branching norms.

Existing tools already in tree:

- [`scripts/prep_room_usds.py`](../../../../source/strafer_lab/scripts/prep_room_usds.py) — wraps Infinigen `generate_indoors` (`fast_solve.gin singleroom.gin` / `dgx` preset) and emits the per-scene `.usdc` + companion dir under `Assets/generated/scenes/<scene_name>/`.
- [`scripts/extract_scene_metadata.py`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py) — three modes:
  - `--blend` reads the Blender stage (full rooms + polygons + relations).
  - `--from-usd` parses prim names from the `.usdc` (objects + labels; **no room polygons**, no relations).
  - `--metadata + --label-usd-only` labels prims using an existing JSON.
- [`scripts/generate_scenes_metadata.py`](../../../../source/strafer_lab/scripts/generate_scenes_metadata.py) — authors the combined top-level `scenes_metadata.json` with per-scene spawn points + floor top Z.

## Current state (2026-05-26)

`Assets/generated/scenes/`:

```
scene_fast_singleroom_000_seed0/      ← coarse/ export/ scene_config.json   (no scene_metadata.json)
scene_high_quality_dgx_000_seed0/     ← coarse/ export/ scene_config.json   (no scene_metadata.json; 492 objects via --from-usd; rooms=0)
scenes_metadata.json                  ← spawn points only
```

Issues:

1. **Only two scenes exist**, both seed 0. The brief's bar is ≥ 2 scenes; meeting it with seed 0 + seed 0 doesn't satisfy the spirit (no real distribution coverage). Need seeds 1..N as well.
2. **`scene_metadata.json` is missing per-scene**. `extract_scene_metadata.py --from-usd` produces a usable one but with `rooms=0` (cannot recover room polygons from USD prim names alone). The hard-negative `wrong_room` chord still works at capture time (operator commits to the failure mode) but consumers that need `get_room_at_position()` (e.g. room-state-eval, per-frame GT room_idx) can't function.
3. **`scene_fast_singleroom`'s object count is sparse** — the operator pushback noted this. `fast_solve.gin singleroom.gin` is the throughput-tuned preset; the real corpus should be the `dgx` / `high_quality_dgx` preset where furniture density is realistic.

## Acceptance

Ship a corpus that meets all of:

- [ ] ≥ 4 Infinigen scenes generated under `Assets/generated/scenes/`, covering at least two seeds and (where presets differ) at least the `high_quality_dgx` profile. Naming: `scene_high_quality_dgx_<NNN>_seed<S>` (zero-padded NNN, distinct seeds).
- [ ] Each scene has a complete `scene_metadata.json` next to its `.usdc`, ideally from the Blender path so `rooms[]` is populated. If only the `.usdc` survives, fall back to `--from-usd` and document the `rooms=0` limitation in the per-scene README — downstream consumers (`room-state-eval-harness`, `cotrained-retrieval-augmented`) must surface a clean error against rooms=0 rather than silently mislabeling.
- [ ] `scenes_metadata.json` (combined) refreshed via `generate_scenes_metadata.py` so spawn points + floor top Z are present for every new scene.
- [ ] Verification per scene: `python -c "import json; d=json.load(open('Assets/generated/scenes/<name>/scene_metadata.json')); assert len(d['objects']) > 50, len(d['objects'])"` passes.
- [ ] A capture smoke against each new scene: `Scripts/capture.py --driver teleop --mission-source scene-metadata --scene <name> --output /tmp/smoke_<name> --max-episodes 1` runs end-to-end (1 episode is enough; we're verifying the scene + picker work).
- [ ] `docs/example_commands_cheatsheet.md` "Extract scene_metadata.json" subsection refreshed to point at the new scene names + drop the "Known limitation: rooms=0" warning if the corpus is rooms-populated.

## Out of scope

- **Procedurally generating during capture sessions.** Scenes are pre-generated; the harness consumes the files.
- **A new scene-generation script.** `prep_room_usds.py` is sufficient; the work is operating it correctly and capturing the outputs.
- **Real-robot scenes.** Sim only.
- **Auto-curating which scenes to capture against.** The operator picks per-session; this brief just makes more available.

## Implementation outline

1. Run `prep_room_usds.py --preset dgx --seed N` for `N ∈ {1, 2, 3}` (or whatever yields ≥ 4 distinct scenes including the existing seed 0). Each invocation takes ~tens of minutes per seed on the DGX (Infinigen is single-threaded scene synthesis).
2. For each generated scene, extract `scene_metadata.json` via the Blender path if the stage is still in memory; otherwise `--from-usd`. Document which path was used in the per-scene dir.
3. Refresh the combined `scenes_metadata.json` via `generate_scenes_metadata.py`.
4. Smoke each scene with a 1-episode capture (see acceptance).
5. Refresh the cheatsheet's scene names + drop the `--from-usd` rooms=0 warning if all scenes have populated rooms.

## Investigation pointers

- Existing scene generator: [`scripts/prep_room_usds.py`](../../../../source/strafer_lab/scripts/prep_room_usds.py) — read `--help` for the preset list; `--preset dgx` is the high-quality target.
- Scene metadata typed accessors: [`strafer_lab.tools.scene_labels`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py) — what consumers read from `scene_metadata.json`.
- Picker filtering: [`strafer_lab.tools.teleop_mission_picker`](../../../../source/strafer_lab/strafer_lab/tools/teleop_mission_picker.py) — confirms which object labels survive the default block list (`wall`/`floor`/`ceiling`).
- Mission generator (shared with bridge harness mode): [`strafer_lab.sim_in_the_loop.mission`](../../../../source/strafer_lab/strafer_lab/sim_in_the_loop/mission.py) — sorts by `(label, instance_id)` for stable ordering.
