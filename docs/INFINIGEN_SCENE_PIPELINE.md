# Infinigen scene pipeline

This is the **Infinigen-specific** build pipeline — the concrete scripts,
environment, and commands that turn Infinigen into a conformant scene
provider. The platform-agnostic scene-provider contract (the on-disk layout,
the USD-`customData` metadata schema, the combined manifest, and the consumer
obligations) is [`SCENE_PROVIDER_CONTRACT.md`](SCENE_PROVIDER_CONTRACT.md);
Infinigen is one provider that produces artifacts in the shapes it documents.
Any other source (a downloaded scene pack, a hand-authored map, a ProcTHOR /
Habitat / Cosmos export) needs its own authoring step but reuses everything
downstream unchanged.

Section references (`§a`–`§h`) below point at the
[scene-provider contract](SCENE_PROVIDER_CONTRACT.md). The USD-prim
expectations (§e) and the postprocess CLI surface (§f) stay in that contract —
they document Infinigen's prim patterns *as the overridable defaults of an
agnostic tool*, so a second source overrides them there.

---

## Producing the artifacts — the Infinigen pipeline

Infinigen produces a conformant bundle in **two steps**. `prep_room_usds.py
generate` now **chains** the metadata authoring, so one command yields a
capture-ready *and* detections-ready scene; only the combined manifest
(spawn-point discovery) remains a separate pass.

| # | Script | Produces | Runtime / env |
|---|---|---|---|
| 1 | `prep_room_usds.py generate` | room geometry: `<scene>.usdc` symlink + `scene_config.json` + the Blender export tree; bakes colliders via `postprocess_scene_usd.py`; **then chains `extract_scene_metadata.py --from-usd`** to embed the per-scene metadata (`objects[]` + `rooms[]`, §b) in the USD's `customData` + apply the `UsdSemantics` detection labels | orchestrator; spawns `STRAFER_INFINIGEN_PYTHON` (Infinigen/`bpy`) + `STRAFER_ISAACLAB_PYTHON` (`pxr`, postprocess) + `$ISAACLAB` (Kit, for the `UsdSemantics` authoring) |
| 2 | `generate_scenes_metadata.py` | combined `scenes_metadata.json` (spawn points + floor-Z, §c) — required for the runtime to discover the scene | `$ISAACLAB -p` (`pxr`) |

```bash
source env_setup.sh
# 1) geometry + embedded metadata + detection labels, in one command.
#    --rooms <types> pins an EXACT tiled layout (duplicates OK), else organic;
#    --quality {high,low} sets texture + object density. `... info` lists knobs.
python source/strafer_lab/scripts/prep_room_usds.py generate \
    --rooms living-room --quality low --name singleroom --output Assets/generated/scenes
# 2) combined manifest (discoverability); --merge keeps existing entries whose
#    heavy USDs aren't on disk this run (the high-quality corpus is ungit'd)
$ISAACLAB -p source/strafer_lab/scripts/generate_scenes_metadata.py --merge
```

`<scene>` is the id printed by step 1 (e.g. `scene_singleroom_000_seed0`).

**Re-authoring metadata on an existing USD** (USD-only / no Blender —
best-effort prim-name labels, `rooms=[]`) runs the same embedder
standalone, under the Kit launcher because the `UsdSemantics` schema is
Kit-provided:

```bash
$ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \
    --from-usd --usd Assets/generated/scenes/<scene>.usdc
```

For richer metadata (rooms + semantic tags) the `.blend` builder runs in
Blender and hands its dict to a Kit authoring pass via a transient JSON
(not a discovered sidecar):

```bash
$STRAFER_BLENDER_BIN --background --python source/strafer_lab/scripts/extract_scene_metadata.py -- \
    --blend Assets/generated/scenes/<scene>/coarse/scene.blend \
    --metadata-out /tmp/<scene>_metadata.json
$ISAACLAB -p source/strafer_lab/scripts/extract_scene_metadata.py \
    --author-from-json /tmp/<scene>_metadata.json \
    --usd Assets/generated/scenes/<scene>.usdc
```

`extract_scene_metadata` also exposes an **in-process hook** —
`extract_from_state()` builds the dict from Infinigen's live generation
state — for a producer that wants to author metadata without a post-hoc
USD parse.

---

## Infinigen prim-naming conventions (convenience reference)

These are the Infinigen prim patterns the postprocess + metadata passes key
on. They are the **defaults** of an agnostic tool; the authoritative,
overridable-knob table is §f (`postprocess_scene_usd.py` CLI surface) in the
[contract](SCENE_PROVIDER_CONTRACT.md) — a second source overrides them there,
it does not edit these.

| Prim kind | Pattern (default) | Matched on |
|---|---|---|
| Floor | `^/World/[^/]+_floor(?:/[^/]+_floor)?$` | full prim path |
| Structural (walls/ceilings/…) | `^/World/[^/]+_(?:wall\|ceiling\|roof\|attic\|exterior)(?:_\d+)?(?:/.+)?$` | full prim path |
| Door arm | `^/World/[A-Za-z]*Door[A-Za-z]*Factory_\d+__spawn_asset_\d+.*$` | full prim path |
| Ceiling-light fixture | `^CeilingLightFactory_\d+__spawn_asset_\d+_$` | **prim name** (leaf), not path |
| Object instance de-dup token | `__spawn_asset_<N>_` inside `prim_path` | substring |

---

## Related

- [`SCENE_PROVIDER_CONTRACT.md`](SCENE_PROVIDER_CONTRACT.md) — the
  platform-agnostic artifact contract this pipeline conforms to (schemas,
  on-disk layout, consumer obligations, adapter checklist).
- [`HARNESS_DATA_CAPTURE.md`](HARNESS_DATA_CAPTURE.md) — operator workflow for
  capturing against a conformant scene.
