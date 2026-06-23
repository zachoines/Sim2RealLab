# Richer Infinigen scene corpus for harness data capture

**Type:** scene-generation pipeline (corpus growth + silent-failure guards)
**Owner:** DGX agent
**Priority:** P1 — gates the [`harness-architecture`](harness-architecture.md) Tier 1 acceptance run (≥ 30 episodes on ≥ 2 scenes), the real-Qwen corpus run, and every later corpus capture.
**Estimate:** M (operate `prep_room_usds.py` to add scenes + re-embed the defective one; plus two index/persistence guards so a metadata-less scene can never silently ship or be counted).
**Branch:** `task/harness-infinigen-scene-corpus`

## Story

As a **DGX operator who needs to run real teleop capture sessions against the harness** I want **a corpus of ≥ 4 fully-usable `high_quality_dgx` Infinigen scenes — each with embedded `strafer_scene_metadata` customData (objects + rooms + connectivity) and a cached occupancy grid under `Assets/generated/scenes/`** so that **the teleop mission picker has groundable targets across multiple seeds and the "≥ 30 episodes on ≥ 2 scenes" acceptance bar is achievable on metadata that is actually present rather than silently empty.**

## Context bundle

- [`harness-architecture.md` §Driver: teleop](harness-architecture.md#driver-teleop) — consumes scenes via the `scene-metadata` mission source.
- [`harness-architecture.md` §Implementation tiers → Tier 1](harness-architecture.md#tier-1--writer--teleop-driver-pr-b) — names the ≥ 30 ep × ≥ 2 scene acceptance.
- [`../../../SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) — the producer/consumer contract; what a "usable scene" owes its consumers.
- [`../../context/conventions.md`](../../context/conventions.md) — commit / branching norms.

Producer tools in tree (all four MAY import `infinigen` — they are the scene-source PRODUCER side; the consumer-side no-infinigen litmus does not bind them):

- [`scripts/prep_room_usds.py`](../../../../source/strafer_lab/scripts/prep_room_usds.py) — `generate` wraps Infinigen `generate_indoors` + `infinigen.tools.export` to USDC, then chains the metadata embed + connectivity internally.
- [`scripts/extract_scene_metadata.py`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py) — `--from-usd` parses prim names from the `.usdc` into `objects[]` and applies the `UsdSemantics.LabelsAPI` detection labels (Kit-bound); leaves `rooms == []` (prim names can't recover room polygons). `--blend` reads the Blender stage for full rooms + polygons.
- [`scripts/generate_scenes_metadata.py`](../../../../source/strafer_lab/scripts/generate_scenes_metadata.py) — authors the combined `scenes_metadata.json` (per-scene spawn points + floor top Z). **Now gated** on embedded customData (see Guards).
- [`scripts/validate_scene_connectivity.py`](../../../../source/strafer_lab/scripts/validate_scene_connectivity.py) — Kit-bound; generates the occupancy grid (`occupancy.npy`) and back-fills `rooms[]` + appends `connectivity[]` into the customData. **Now guarded** against authoring an empty payload and against a non-persisting save.

The per-scene `scene_metadata.json` sidecar (singular) is **RETIRED** — written only by retired code and stale (`rooms = 0`). The live sources of truth are the **USD customData** (read via `strafer_lab.tools.scene_metadata_reader`) and the combined **`scenes_metadata.json`** (plural, the spawn-points index).

## Current state (2026-06-22)

`Assets/generated/scenes/` holds three scene dirs; `scenes_metadata.json` indexes all three:

| Scene | Preset | customData | Usable? |
|---|---|---|---|
| `scene_fast_singleroom_000_seed0` | `fast_singleroom` (max_rooms=1) | key **ABSENT** | No — throughput preset, **not counted** toward the corpus regardless |
| `scene_high_quality_dgx_000_seed1` | `high_quality_dgx` | key **PRESENT but `objects=0`** (rooms=11) | **No — DEFECTIVE.** Silent-failure member of the index |
| `scene_high_quality_dgx_000_seed2` | `high_quality_dgx` | `objects=691`, `rooms=9`, `connectivity` present | **Yes** — fully usable |

So **only seed2 is fully usable today.** The 2026-05-26 inventory in the prior version of this brief ("492 objects seed0 high_quality") is OBSOLETE — there is no `high_quality` seed0 on disk now.

### The seed1 defect — root cause

seed1's USD customData key is **present but its `objects[]` is empty** (rooms=11 was back-filled from floor meshes). The base metadata embed (`extract_scene_metadata.py --from-usd`) never successfully ran: seed1's `scene_config.json` records that its export ran as an orphan subprocess after the wrapper crashed ("postprocess/symlink/metadata run by hand"), and that by-hand pipeline skipped the embed. The connectivity step later ran on seed1 and — reading an absent payload, then `setdefault`-ing `objects=[]` and back-filling `rooms[]` — authored a present-but-**empty** customData key (`objects=0, rooms=11, version=1`), which is exactly why it slipped into `scenes_metadata.json`.

> **Forensic correction to earlier notes:** the key is **not** absent on seed1 — `grep -a -c strafer_scene_metadata` returns 0 only because the connectivity re-save compressed the crate token table. The grep token count is an unreliable proxy for key presence; the **authoritative** signal is the reader (`objects == 0`). The fix is to populate `objects[]`, not to author a missing key. seed1's USD is fully intact (11.4 GB, all object/floor/door prims); its retired 867-object sidecar (`source=usd_prim_names`) is trustworthy provenance for what re-embedding yields.

Fix: **re-embed on the EXISTING USD** (do not regenerate from scratch). See the runbook.

## Guards (Phase-1 code — shipped with this brief)

Two guards so a scene can never silently ship or be counted with empty metadata:

- **GUARD 1 — index gate** in `generate_scenes_metadata.py` (`has_capture_metadata` + the `_process_scene` check). Reuses the already-open `Usd.Stage` to read embedded customData and **skips with a loud warning** when the key is absent *or* `objects == 0`, instead of emitting a spawn entry. This is the load-bearing guard that stops a seed1-style scene from being counted. New dependency: `generate_scenes_metadata.py → scene_metadata_reader` (same pxr-only env, no Kit). Agent-testable.
- **GUARD 2 — fail-hard + read-back** in `validate_scene_connectivity.py` (`assert_persisted_metadata` + an up-front check). Fails hard up front **only** when the base customData key is entirely absent (so connectivity can never be the step that first authors an empty payload); a present-but-empty payload is a legitimate intermediate it enriches. After `save`, it reopens the file and asserts the key persisted (version stamped, `connectivity[]` present, `objects[]` non-empty when an embed was expected). Kit-bound: agent-codable but operator-verified end-to-end.

> Transient when GUARD 1 first runs (it is the guard working, not a regression): a fresh index regen **drops both seed0 (key absent) and seed1 (objects=0)**, leaving only **seed2** indexed. seed1 re-enters after the operator re-embeds it. (This corrects an earlier note that predicted seed0 would survive — seed0's customData is absent, so the gate drops it too.)

## Definition of a usable scene

The "≥ 4" bar counts `high_quality_dgx` (or richer) scenes only; `fast_singleroom_seed0` does **not** count (throughput preset, max_rooms=1). The four target scenes are **seed1 (after re-embed), seed2, seed3, seed4**. A scene counts only if ALL hold, verified against the USD customData (not the retired sidecar):

- [ ] customData key `strafer_scene_metadata` is PRESENT with `objects > 50` (high-quality acceptance bar).
- [ ] `rooms > 0`, and every room carries `footprint_xy`.
- [ ] The scene appears in `scenes_metadata.json` with `spawn_points_xy`.
- [ ] Occupancy cache present (`<scene>/occupancy.npy`).

A scene missing the key, or with `objects == 0`, is a silent failure and MUST NOT be counted.

## Acceptance

- [ ] ≥ 4 fully-usable `high_quality_dgx` scenes per the checklist above (seed1 re-embedded; seed2; seed3; seed4).
- [ ] `scenes_metadata.json` refreshed via `generate_scenes_metadata.py`; the GUARD-1 gate keeps every metadata-less scene out of it.
- [ ] Per-scene verification via embedded customData (`scene_metadata_reader.load(<usdc>)`): `objects > 50`, `rooms > 0`, every room has `footprint_xy`. Do **not** verify via the retired per-scene `scene_metadata.json` sidecar.
- [ ] A 1-episode capture smoke against ≥ 2 new/re-embedded scenes: `capture.py --driver teleop --mission-source scene-metadata --scene <name> --output /tmp/smoke_<name> --max-episodes 1` runs end-to-end (Kit-bound, operator).

## Out of scope

- Procedurally generating during capture sessions (scenes are pre-generated).
- A new scene-generation script (`prep_room_usds.py` is sufficient).
- Real-robot scenes (sim only).
- Auto-curating which scenes to capture against (operator picks per session).

## Two-phase ship

- **Phase 1 (AGENT, this PR):** GUARD 1 + GUARD 2 code + tests + this refreshed brief + the operator runbook. Ship the agent-code portion per the docs lifecycle.
- **Phase 2 (OPERATOR-gated):** the actual scene work (seed1 re-embed + seed3/seed4 generate) is operator-run (GPU/Blender + Kit). **This brief stays ACTIVE until the operator confirms ≥ 4 usable scenes** against the checklist. Do not stamp the corpus bar closed on Phase-1 code alone.

## Operator runbook (Phase 2)

> Tier legend: **[OP-GPU]** Blender/Infinigen (`STRAFER_INFINIGEN_PYTHON`); **[OP-KIT]** Isaac Sim launcher (`$ISAACLAB -p`); **[AGENT]** pure pxr/python under `STRAFER_ISAACLAB_PYTHON` (no Kit).

**Pre-flight.** `source env_setup.sh`, then confirm `ISAACLAB`, `STRAFER_INFINIGEN_PYTHON`, `STRAFER_ISAACLAB_PYTHON`, `INFINIGEN_ROOT` are all set (`prep_room_usds.py generate` enforces `$ISAACLAB` up front so a multi-hour generate isn't wasted).

Ordering matters — seed1 re-embed precedes the final index regen:

1. **[OP-KIT] Re-embed seed1** on its existing USD (do NOT regenerate; do NOT change the embed builder). The `--usd` MUST be absolute: `omni.usd`'s `ctx.open_stage()` resolves a *relative* path against the USD layer's own directory, producing a doubled path (`…/export_scene.blend/Assets/…/export_scene.usdc`) that fails to open/save. `$(realpath …)` from the repo root supplies the absolute path (this is why the `prep_room_usds.py` wrapper, which resolves `output_dir`, works while a hand-run relative path does not):
   ```
   $ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py --from-usd \
     --usd "$(realpath Assets/generated/scenes/scene_high_quality_dgx_000_seed1/export/export_scene.blend/export_scene.usdc)"
   $ISAACLAB -p source/strafer_lab/scripts/validate_scene_connectivity.py \
     --usd "$(realpath Assets/generated/scenes/scene_high_quality_dgx_000_seed1/export/export_scene.blend/export_scene.usdc)"
   ```
   **Confirm the re-embed landed — do NOT trust rc=0 alone** (the orphan-export disk/inotify error can truncate a save silently). The reliable check is the reader, not grep (grep token count is compression-dependent):
   ```
   LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 $STRAFER_ISAACLAB_PYTHON -c \
     "from strafer_lab.tools import scene_metadata_reader as smr; \
      m = smr.load('Assets/generated/scenes/scene_high_quality_dgx_000_seed1/export/export_scene.blend/export_scene.usdc'); \
      print('objects', len(m['objects']), 'rooms', len(m['rooms']))"
   ```
   Expect `objects ≈ 867`, `rooms > 0`. (GUARD 2 now also fails the connectivity run hard if the base embed didn't run, and verifies the save persisted.)

2. **[OP-GPU + OP-KIT] Generate seed3 + seed4** — the `high_quality_dgx` preset (max_rooms=5, texture_resolution=1024); this single command chains the embed + connectivity internally:
   ```
   $ISAACLAB -p source/strafer_lab/scripts/prep_room_usds.py generate \
     --config high_quality_dgx --num-scenes 2 --seed-base 3 \
     --output Assets/generated/scenes
   ```
   (`seed = seed-base + i`, so this yields `scene_high_quality_dgx_000_seed3` and `..._001_seed4`. Two single-scene runs `--seed-base 3 --num-scenes 1` then `--seed-base 4 --num-scenes 1` yield `_000_seed3` / `_000_seed4`.) The real flags are `--config / --num-scenes / --seed-base` — there is no `--preset dgx --seed N` on `generate` (`--preset` exists only on `ingest`).

3. **[AGENT] Regenerate the combined index** (pxr-only; owned here so it's not forgotten):
   ```
   LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
     $STRAFER_ISAACLAB_PYTHON source/strafer_lab/scripts/generate_scenes_metadata.py \
     --scenes-dir Assets/generated/scenes
   ```
   Run this AFTER step 1 so seed1 re-enters the index. (Running it before step 1 correctly drops seed1 and seed0 — that is GUARD 1, not a regression.)

4. **[AGENT] Verify each new/re-embedded scene** via the `scene_metadata_reader.load()` one-liner from step 1 against the scene's `.usdc`: assert `objects > 50`, `rooms > 0`, every room has `footprint_xy`. Verify against the USD customData, NOT the retired `scene_metadata.json` sidecar.

**Known `rooms == 0` risk.** `--from-usd` itself leaves `rooms = []`; seed2 has `rooms = 9` only because `validate_scene_connectivity.py` back-fills `rooms[]` from floor meshes. If a scene comes back `rooms == 0`, the operator must run the Blender path (`extract_scene_metadata.py --blend ... ` under `$STRAFER_BLENDER_BIN`) to populate room polygons.

**Occupancy freshness** is checked by the connectivity tooling, not here. After a re-embed, `occupancy.npy` mtime may lag the rewritten USDC — treat occupancy *presence* as the agent-side check and leave freshness re-validation to the operator's connectivity re-run.
