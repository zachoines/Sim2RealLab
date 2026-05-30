# Move scene metadata inside the USD (drop the sidecar)

**Type:** refactor (scene-metadata authoring + consumers)
**Owner:** TBD
**Priority:** P2 — quality-of-life + correctness improvement; not gating any current acceptance bar.
**Estimate:** M (touches authoring + ~3 consumer paths; bounded test surface).
**Branch:** `task/scene-metadata-in-usd`
**Blocked on:** _Resolved_ — [`scene-provider-contract`](../../completed/scene-provider-contract.md) shipped (PR #66). That brief documents the storage-agnostic artifact interface — [`SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) — that this brief implements with a USD `customData` backend.

## Story

As a **harness operator who moves scene USDs around** I want **scene metadata to travel with the USD** so that **I never end up with a USD whose sidecar is stale, missing, or pointing at a different scene — the operator-facing failure mode that motivated `scene_paths.py` and its path-resolution gymnastics in the first place.**

## Motivation

Today, each scene is a pair: `<X>.usdc` (the geometry) and a sibling
`<X>/scene_metadata.json` (the labeled `objects[]`, room polygons,
spatial relations). The two files MUST be kept in sync, but the only
mechanism enforcing that is layout convention. Real failure modes
already seen during PR #63 review:

- The picker offers a "bed" that isn't in the loaded USD (sidecar
  was for a different scene revision).
- `--scene-usd` override required us to build
  [`scene_paths.py`](../../../../source/strafer_lab/strafer_lab/tools/scene_paths.py)
  to walk the filesystem looking for the right sidecar.
- The bridge harness and the teleop picker each independently parse
  `scene_metadata.json` and trust the layout convention; a future
  consumer added without that knowledge would silently drift.

USD has a first-class mechanism for this:
[`customData`](https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/properties/set-custom-metadata.html).
We can store the metadata dict on the root prim of each scene USD —
portable, invariant (the bytes ship together), no path resolution.

## Acceptance

Ship a refactor that meets all of:

- [ ] `extract_scene_metadata.py` is modified to write the metadata
  payload to the scene USD's root-prim `customData` (key:
  `strafer_scene_metadata`) IN ADDITION to the existing sidecar (the
  sidecar is kept during the transition window for backwards
  compatibility — see migration plan below).
- [ ] A new helper `strafer_lab.tools.scene_metadata_reader.load(scene_usd_path)`
  prefers the embedded `customData` and falls back to the sibling
  sidecar with a deprecation log line.
- [ ] All in-tree consumers route through that helper:
  - `strafer_lab.tools.teleop_mission_picker.load_candidates`
  - `strafer_lab.sim_in_the_loop.MissionGenerator` (bridge harness)
  - `strafer_lab/scripts/generate_scenes_metadata.py` (when aggregating
    per-scene metadata into the combined file)
- [ ] `scene_paths.py` shrinks: `find_metadata_for_usd` is no longer
  needed because the reader takes the USD path directly. The
  resolver keeps `resolve_scene_metadata_path` only for the
  explicit-sidecar-override case.
- [ ] Tests:
  - Round-trip test: write metadata into a USD's `customData`, re-open
    via the reader, confirm the dict round-trips.
  - Fallback test: USD with no `customData` + sibling sidecar → reader
    falls back, logs deprecation once per process.
  - Schema parity test: for every existing scene under
    `Assets/generated/scenes/`, the embedded payload matches the
    sidecar bit-for-bit after re-running the extractor.
- [ ] Documentation:
  - [`docs/HARNESS_DATA_CAPTURE.md`](../../../HARNESS_DATA_CAPTURE.md)
    "Extract scene_metadata.json" section is updated to describe the
    embedded form and note the sidecar is deprecated.
  - A short note in the [`harness-architecture`](harness-architecture.md)
    brief mentioning that "scene metadata" now refers to USD
    customData rather than a sidecar file.

## Migration plan

Two-phase rollout to avoid breaking in-flight datasets:

**Phase 1 (this brief)** — Write to both. Reader prefers embedded,
falls back to sidecar. Old datasets keep working unchanged.

**Phase 2 (follow-up brief, filed after Phase 1 lands)** — Drop the
sidecar from `extract_scene_metadata.py`. Reader emits a hard error if
neither source is present. Regenerate the corpus.

## Risks

- **USD file size**: the embedded dict adds tens of KB per scene
  (mostly the `objects[]` array). Negligible vs the geometry payload.
- **USD round-trip cost**: reading customData requires `pxr` (USD
  runtime). The current sidecar is a 1-line JSON load. The reader
  helper centralizes this so it's one cost site, not N. For the
  `.venv_harness` test environment that doesn't have `pxr`, the
  reader falls back to the sidecar — keeps unit tests fast.
- **Schema versioning**: today's sidecar has no version field; if
  we're rewriting the authoring path, this is the moment to add
  `"strafer_scene_metadata_version": 1`. Phase 2 can require the
  field; Phase 1 tolerates its absence.
- **Tooling unaware of `customData`**: USD viewers (usdview, Omniverse
  Composer) will show the embedded dict in the property panel; that's
  benign. Replicator's annotators don't read `customData` so the
  RL/perception pipelines are unaffected.

## Out of scope

- Migrating the per-frame `strafer_episodes.parquet` sidecar (lerobot
  dataset extension) into the LeRobot dataset proper. That's a
  separate `lerobot-extension-columns-as-features` brief.
- Changing the schema of the metadata payload itself. The fields stay
  identical to today's sidecar; only the storage location moves.

## Triggered by

PR #63 review thread on
[`scene_paths.py:45`](../../../../source/strafer_lab/strafer_lab/tools/scene_paths.py)
— "I question if we should be crawling around for metadata like this
— as opposed to enforcing a caller pass correct path. Lastly, why
are we even maintaining a separate metadata sidecar anyways? Why not
store within USD itself (portable; invariant; generate new at USD
creation)."

The tight resolver in `scene_paths.py` removes the "crawling" objection
for the operator-ergonomic path-derivation case, but the more
fundamental "why have a sidecar at all" question remains. This brief
addresses it.
