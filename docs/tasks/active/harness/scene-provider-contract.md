# Scene-provider contract documentation + extensibility hooks

**Type:** documentation + small refactor (un-baking the last Infinigen-specific seams)
**Owner:** DGX agent
**Priority:** P2 — not gating any current acceptance bar; soft-blocks operator's stated intent to add other USD sources for data variety.
**Estimate:** M (~one focused day: write the contract doc, audit the three Infinigen-coupled scripts, parameterize the one not-yet-CLI-overridable knob, add an ingest worked example).
**Branch:** `task/scene-provider-contract`

## Story

As a **harness operator who wants to bring in additional USD scenes (downloaded scene packs, hand-authored maps, future ProcTHOR / Habitat / Cosmos exports) for better dataset variety** I want **an explicit, documented contract that says "if your scene produces these artifacts in this shape, the teleop harness consumes them with no code changes"** so that **adding a second scene source is a weekend of glue code against a stable interface — not a code-archaeology exercise to discover what the runtime actually requires.**

## Motivation

Audited 2026-05-29 during PR #63: the runtime teleop driver (`teleop_capture.py`) is already scene-source-agnostic — it consumes three artifacts (`scene_metadata.json`, a USDC file, the combined `scenes_metadata.json`) and never imports Infinigen code. The Infinigen coupling is entirely at **scene-build time**: three scripts (`prep_room_usds.py`, `extract_scene_metadata.py --from-usd`, `postprocess_scene_usd.py`) plus a label-parser helper.

The build-time coupling is mostly fine — a second source naturally wants its own `prep_<src>_usds.py` and metadata extractor. But two specific debts make the boundary fuzzier than it needs to be:

1. **The contract is implicit.** What `teleop_capture.py` reads is "what works today" rather than a documented schema. A second-source author would have to reverse-engineer the field names + JSON shape + USD prim expectations from the picker tests + the writer tests + the env config.
2. **One Infinigen-specific knob has no CLI escape hatch.** `postprocess_scene_usd.py`'s `_CEILING_LIGHT_NAME_RE` regex hardcodes `CeilingLightFactory_\d+__spawn_asset_\d+_` — works only for Infinigen-named light prims. Floor + structural patterns are already CLI-overridable (`--floor-prim-pattern`, `--structural-prim-pattern`).

Existing prior art for external-USD ingest: [`prep_room_usds.py:ingest_external_usd`](../../../../source/strafer_lab/scripts/prep_room_usds.py) (the `ingest` subcommand) already copies external USD directories into the scenes tree and stamps `scene_config.json`. It does NOT run extract or postprocess — that's the operator's job today, and the contract doc should walk them through it.

## Acceptance

Ship the following:

- [ ] **`docs/SCENE_PROVIDER_CONTRACT.md`** — a single source of truth for what artifacts a scene provider must produce for the harness to consume it. Sections:
  - On-disk layout (which file lands where, with the `<scene_name>` convention)
  - `scene_metadata.json` schema — field-by-field, with required vs optional, types, units, semantic guarantees (e.g. "`position_3d` is XYZ in world frame, meters; (0,0,0) means *no valid position* and the entry should be dropped — the extractor's `_drop_origin_records` helper handles this")
  - `scenes_metadata.json` schema — combined-manifest fields (`spawn_points_xy`, `floor_top_z`, plus anything else the env config reads)
  - USD prim expectations — what the runtime + postprocess scripts traverse (collision-API-applied meshes, ceiling-light prims, the floor strip pattern, structural prims for hybrid collider dispatch)
  - postprocess CLI surface — every regex flag with the "if your source names things differently, override with this flag" mapping
  - Adapter-writer's checklist: a numbered "to add a new source X, ship these N things"
  - Worked examples: (a) a hand-authored single-USD ingest (operator drops one USDC in, hand-writes metadata, postprocess + symlink, capture) and (b) a programmatic adapter sketch (the `prep_<src>_usds.py` parallel)
- [ ] **`docs/HARNESS_DATA_CAPTURE.md`** updated to reference the contract doc in the "Infinigen scene corpus" section ("Infinigen is one provider; see SCENE_PROVIDER_CONTRACT.md for the general interface").
- [ ] **Parameterize `_CEILING_LIGHT_NAME_RE`** in [postprocess_scene_usd.py](../../../../source/strafer_lab/scripts/postprocess_scene_usd.py). Add a `--ceiling-light-prim-pattern` CLI flag mirroring `--floor-prim-pattern`. Default keeps Infinigen behaviour (`CeilingLightFactory_\d+__spawn_asset_\d+_`). One new test asserting the override works.
- [ ] **Optional but recommended**: split `inject_ceiling_light_emitters` into its own importable function with a documented signature, so a foreign-source adapter that needs to author lights differently (e.g. inject `UsdLux.DiskLight` at floor-up positions) can bypass it entirely without forking `postprocess_scene_usd.py`. No new CLI surface, just a contract on the function.
- [ ] **A test under `tests/harness/test_scene_provider_contract.py`** that round-trips a minimal hand-authored scene_metadata.json + dummy USDC through the picker + writer to lock in the contract — so future refactors that accidentally tighten the field requirements break a test, not a downstream user.
- [ ] Brief is shipped to `completed/` per the conventions (see `docs/tasks/context/conventions.md`).

## Approach — what the contract should actually say

The boundary is **artifact-based, not interface-based**. We do NOT want a Python `SceneSourceAdapter` ABC that every provider implements — that forces foreign sources to be Python-importable and to fit our class shape. Instead the contract is:

> *Produce these files in this shape. The harness will pick them up.*

The runtime is enforced by tests (the round-trip test above), not by static typing. This matches how the current Infinigen scripts work — they're three independent scripts that produce conformant artifacts, not subclasses of a shared base.

Worked example for a one-off USD ingest (what the contract doc should walk through):

```bash
# 1. Drop the USDC + symlink it at the scenes root
SCENE=my_scene_alpha
mkdir -p Assets/generated/scenes/$SCENE/export/scene_alpha.blend
cp ~/Downloads/scene_alpha.usdc Assets/generated/scenes/$SCENE/export/scene_alpha.blend/scene_alpha.usdc
ln -sf $(realpath Assets/generated/scenes/$SCENE/export/scene_alpha.blend/scene_alpha.usdc) \
    Assets/generated/scenes/$SCENE.usdc

# 2. Hand-author (or programmatically derive) scene_metadata.json
#    — minimum required fields per the contract doc
cat > Assets/generated/scenes/$SCENE/scene_metadata.json <<'EOF'
{
  "objects": [
    {"label": "chair", "instance_id": 1, "position_3d": [2.5, 1.2, 0.5],
     "prim_path": "/World/Chair_1"},
    ...
  ],
  "rooms": []
}
EOF

# 3. Postprocess with CLI overrides matching your USD's prim naming
$ISAACLAB -p source/strafer_lab/scripts/postprocess_scene_usd.py \
    --usdc Assets/generated/scenes/$SCENE.usdc \
    --floor-prim-pattern '^/World/floor.*$' \
    --structural-prim-pattern '^/World/(?:wall|ceiling).*$' \
    --ceiling-light-prim-pattern '^MyCeilingLight_\d+$'   # the NEW flag

# 4. Author the combined manifest entry
python source/strafer_lab/scripts/generate_scenes_metadata.py \
    --scenes-dir Assets/generated/scenes

# 5. Capture
$ISAACLAB -p Scripts/capture.py \
    --driver teleop --mission-source scene-metadata \
    --scene $SCENE --output data/sim_in_the_loop/$SCENE_$(date +%s)
```

Steps 1, 3, 4, 5 are scene-source-agnostic and stay constant. Only step 2 (metadata authoring) is source-specific — and the contract doc explains exactly what it has to produce.

## Out of scope

- A `SceneSourceAdapter` ABC / plugin system. The artifact contract is enough; we don't need Python-level extensibility.
- A scene-provider browser / GUI for picking which source to use. Operators run the prep scripts directly.
- Refactoring `infinigen_label_parser.py` — it's correctly Infinigen-specific; the right adapter for a second source is a new sibling module, not a generalization of this one.
- The `scene-metadata-in-usd` brief ([`scene-metadata-in-usd.md`](scene-metadata-in-usd.md)) is the OTHER half of the long-term picture — store metadata inside the USD `customData` so it travels with the geometry. That brief and this one are complementary but independently shippable; either one can land first. When both land, the contract doc updates to describe both storage locations (sidecar JSON + USD customData) and the reader fallback order.
- Documenting RL-training-side scene consumption (`StraferNavEnvCfg_Real_InfinigenDepth` et al). The contract here is teleop-harness-specific; an RL-side contract is a separate brief if needed.

## Triggered by

PR #63 architecture conversation 2026-05-29. Operator: *"What if in the future I wanted to replace or add data collection with another API other than Infinigen. Is our current system becoming too baked into Infinigen-specific? Or are we forming a generalized contract that USDs must follow to be used by the teleop harness?"*

The audit answered "closer to a generalized contract than baked-in, with two seams to clean up." This brief is those two cleanups + writing the contract down.
