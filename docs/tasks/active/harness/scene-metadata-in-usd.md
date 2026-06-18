# Move scene metadata inside the USD (drop the sidecar)

**Type:** refactor (scene-metadata authoring + consumers)
**Owner:** DGX (harness epic)
**Priority:** P2 — quality-of-life + correctness improvement; not gating any current acceptance bar.
**Estimate:** M–L (clean-break cutover: authoring + ~6 live consumers + a cross-lane test surface + the contract doc; bounded, but wider than a 3-consumer swap).
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
- **Generation doesn't yield a capture-ready scene.** `prep_room_usds.py
  generate` writes the room USD but **not** `scene_metadata.json` — the
  extractor (`extract_scene_metadata.py`) is a separate, un-chained step,
  and its in-process hook (`extract_from_state`) isn't wired in. So a
  freshly generated scene loads in the bridge yet fails teleop capture
  with `scene_metadata.json not found` (observed 2026-06-07 on a
  `fast_singleroom` scene). Authoring the metadata at USD-creation time
  fixes this **and** the staleness modes above in one move.
- **Generated scenes are not detections-ready either.** A sibling authoring
  gap in the same producer: `extract_scene_metadata.py` stamps a custom
  `semanticLabel` string attr on the object prims (what it reads back to
  build the metadata) but never applies the USD `Semantics` schema
  (`semanticType=class` / `semanticData`) that Replicator's
  `bounding_box_2d_tight` annotator boxes on. So the harness detections
  producer emits **zero** boxes and `observation.detections.*` is empty on
  bridge/scripted capture even with objects plainly in frame — verified
  2026-06-18 on `scene_fast_singleroom_000_seed0` (156 prims carry
  `semanticLabel`, zero carry `Semantics`; confirmed against the recorded
  perception frames). Authoring `Semantics` alongside the metadata closes
  this in the same `extract_scene_metadata` pass.

USD has a first-class mechanism for this:
[`customData`](https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/properties/set-custom-metadata.html).
We can store the metadata dict on the root prim of each scene USD —
portable, invariant (the bytes ship together), no path resolution.

## Acceptance

Ship a **clean-break cutover** — no dual-write, no deprecation window, no
sidecar fallback, per the project's clean-break convention (cf.
[`env-cfg-composition`](../../completed/env-cfg-composition.md): "no
deprecation window and no shim"). Meets all of:

- [ ] **Authoring writes `customData` only.** `extract_scene_metadata.py`
  writes the metadata payload to the scene USD's root-prim `customData`
  (key `strafer_scene_metadata`, including `"strafer_scene_metadata_version": 1`)
  and **no longer writes the `scene_metadata.json` sidecar**.
- [ ] **Authored at USD-creation time.** `prep_room_usds.py generate`
  runs/embeds the extractor (wire the in-process `extract_from_state` hook,
  or chain the post-hoc extractor at export) so a single `generate` yields a
  capture-ready scene — closes the ergonomics gap in Motivation.
- [ ] **Applies UsdSemantics labels for detections.** `extract_scene_metadata.py`
  applies the `UsdSemantics.LabelsAPI` (`instance_name="class"`) onto each
  labeled object prim — in addition to the custom `semanticLabel` provenance
  attr it writes today — via `isaacsim.core.utils.semantics.add_labels(prim,
  [label], instance_name="class")`. The harness detections producer
  (`bbox_extractor`'s Replicator `bounding_box_2d_tight` annotator) boxes prims
  by the applied semantics schema, **not** by `semanticLabel` or `customData`;
  without this, `observation.detections.*` is empty on every capture (see the
  detections-readiness item in Motivation). Note: Isaac Sim 6 dropped the
  legacy `Semantics` / `SemanticsAPI` schema (no `from pxr import Semantics`) in
  favor of `UsdSemantics.LabelsAPI`; `isaacsim.core.utils.semantics` also
  exposes `upgrade_prim_semantics_to_labels` for any legacy assets. **Verified
  2026-06-18** that applying `add_labels(..., "class")` from the existing
  `semanticLabel` values makes the annotator emit boxes that flow through
  `parse_bbox_data` unchanged. Acceptance proof: after regenerating the corpus,
  `make harness-smoke REQUIRE_DETECTIONS=1` (the bridge driver's Jetson-free
  smoke) passes with a non-empty detection vocab on a regenerated scene.
- [ ] **Apply labels to non-structural classes only — filter at authoring, not
  downstream.** `add_labels` is applied **only** to object prims whose class is
  not in the structural set `{floor, ceiling, wall, exterior, staircase}` (the
  same classes [`generate_scenes_metadata._ROOM_STRUCT_RE`](../../../../source/strafer_lab/scripts/generate_scenes_metadata.py#L61)
  matches — factor the literal class set into one shared definition so the regex
  and this denylist can't drift). **Why at the producer, not in the consumer:**
  `pack_detections` truncates to the `detections_max` **largest-pixel-area**
  boxes ([`lerobot_detections.py`](../../../../source/strafer_lab/strafer_lab/tools/lerobot_detections.py#L166)),
  and walls / floor / ceiling are the biggest things in every frame — so
  "label everything + filter downstream" would let structure win all 32 slots
  and evict the furniture the column exists for. door / window stay labelled
  (not structural here; they're groundable transit landmarks and not
  area-dominant). The class denylist is a **configurable flag**, not hardcoded,
  so a future consumer can flip it and regenerate. **`objects[]` itself stays
  complete** — the structural rows are load-bearing for the shared path planner,
  which rasterizes walls / floor into its occupancy grid; only what gets
  `add_labels` for the annotator is filtered. Document this producer-side
  class-filter policy in [`harness-architecture`](harness-architecture.md)'s
  Detections section.
- [ ] **One reader, hard-fail.** New helper
  `strafer_lab.tools.scene_metadata_reader.load(scene_usd_path)` reads the
  embedded `customData` via `pxr` and **raises if it is absent** (no sidecar
  fallback). It is the single `pxr` touch-point for scene metadata.
- [ ] **Every live consumer cut over to the reader** — this is an *audit*, not
  just the list below:
  - `strafer_lab.tools.teleop_mission_picker.load_candidates`
  - `strafer_lab.sim_in_the_loop.MissionGenerator` (bridge harness)
  - **`strafer_lab.tools.scene_labels.get_scene_metadata`** — the typed-accessor
    layer; **imported by `strafer_autonomy/tests/` (cross-lane)** — coordinate.
  - `strafer_lab/scripts/generate_scenes_metadata.py` (reads each scene's
    `customData` when aggregating the combined `scenes_metadata.json` manifest)
  - `strafer_lab.tools.lerobot_writer` — `scene_metadata_hash` switches from
    sha256 of the **sidecar file bytes** to sha256 of the canonical embedded
    dict (`json.dumps(..., sort_keys=True)`).
  - `--scene-metadata` CLI on `teleop_capture.py` / `run_sim_in_the_loop.py`
    — derive from the USD path; drop the sidecar-path default.
  - the retired `scripts/retired/*` readers are **not** updated (deprecated).
- [ ] **`scene_paths.py` shrinks to USD-path derivation only.** The sidecar
  resolver (`resolve_scene_metadata_path` / `find_metadata_for_usd`) is
  deleted — the reader takes the USD path directly; nothing crawls for a
  sibling JSON.
- [ ] **Test strategy:**
  - Pure-data seams for the fast suites — `teleop_mission_picker` already has
    `load_candidates_from_data`; add the equivalent for `scene_labels`. The
    `.venv_vlm` (`make test-dgx`, autonomy) suite tests against in-memory
    dicts — it has no `pxr`.
  - The reader's USD round-trip + `test_scene_provider_contract.py`'s parity
    check are `pytest.importorskip("pxr")`-gated. Since the harness fold,
    `env_isaaclab3` carries `pxr`, so they now run under `make test-lab-pure`
    (the strafer_lab pure-Python suite) as well as the Kit suite — no
    `usd-core` side-install needed.
  - `make test-dgx`, `make test-lab`, and the Kit suite all green; **no
    test resolves a sidecar path** — the clean-break proof at the test layer.
- [ ] **Regenerate the scene corpus** so every scene under
  `Assets/generated/scenes/` carries `customData` and no sidecar (re-run the
  extractor; the corpus is a small, regenerable build artifact).
- [ ] **Docs updated in the same PR:** `SCENE_PROVIDER_CONTRACT.md` (§a/§b —
  the per-scene metadata is USD `customData`; the sidecar is *gone*, not a
  REQUIRED file) + its "Producing the artifacts" pipeline section, the
  cheatsheet scene-gen recipe, [`docs/HARNESS_DATA_CAPTURE.md`](../../../HARNESS_DATA_CAPTURE.md),
  and a one-line note in [`harness-architecture`](harness-architecture.md)
  that "scene metadata" now means USD `customData`.

## Cutover (clean break — no transition window)

One PR: no dual-write, no reader fallback — matching the project's
clean-break convention ([`env-cfg-composition`](../../completed/env-cfg-composition.md):
"no deprecation window and no shim"). The only "migration" is **regenerating
the scene corpus** so every USD carries `customData`. There are no production
datasets to preserve — per [`harness-architecture`](harness-architecture.md)
nothing has run against real data, and scenes are regenerable build
artifacts. The reader **hard-errors** on a USD that lacks the embedded
payload, so any un-regenerated scene surfaces immediately instead of silently
falling back to a stale sidecar (the exact failure mode this brief kills).

## Risks

- **USD file size**: the embedded dict adds tens of KB per scene (mostly
  `objects[]`). Negligible vs the geometry payload.
- **`pxr` outside `env_isaaclab3`**: reading `customData` needs `pxr`.
  `env_isaaclab3` (which now hosts the strafer_lab pure-Python suite,
  `make test-lab-pure`) has it, so the reader test runs there; only the
  `.venv_vlm` autonomy suite lacks `pxr`, and with the sidecar gone there is
  no JSON fallback for it. Mitigated by the pure-data seams: the reader is
  the *only* `pxr`-bound path and the `.venv_vlm` suite never hits it.
- **Tooling unaware of `customData`**: usdview / Omniverse Composer show the
  embedded dict in the property panel (benign). Replicator's annotators don't
  read `customData` — they read the applied `Semantics` schema, which is why
  the detections-readiness acceptance bullet authors `Semantics` onto prims
  directly rather than relying on the metadata payload. The metadata storage
  move (sidecar → `customData`) is itself perception-neutral.

## Coordination

This brief is **harness-epic-owned** and touches shared metadata surfaces —
coordinate before landing:

- **Producers + consumers are harness territory.**
  `extract_scene_metadata.py` / `generate_scenes_metadata.py` (producers)
  and `teleop_mission_picker` / `MissionGenerator` (consumers) — sequence
  with [`harness-architecture`](harness-architecture.md).
- **The schema is load-bearing downstream.** multi-room's room-graph
  consumers read `rooms[]` / `room_adjacency`, and clip-validation's
  [`mission-text-enrichment`](../../parked/harness/mission-text-enrichment.md) reserves
  `objects[].descriptors` — both per
  [`SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md). This
  brief moves **storage + authoring timing only; it does NOT change the
  schema** (kept out of scope below), so those consumers stay unaffected —
  any schema change must go through the contract + its round-trip test.
- **Cross-lane test surface — sequence early, not late.** The reader cutover
  reaches `scene_labels.get_scene_metadata`, which `strafer_autonomy/tests/`
  imports, so the sidecar→`customData` cutover crosses into the Jetson lane's
  test gate. Coordinate that surface at the **start** of the work (confirm the
  pure-data seam + how the `.venv_vlm` autonomy suite gets its dict without
  `pxr`) rather than discovering it at the end.
- **Land the detections slice as part of "shipped."** Because the semantics
  fix rides here (one corpus regen, not two), "scene-metadata-in-usd shipped"
  must also mean "detections proven non-empty end-to-end": the semantics
  authoring + class filter + corpus regen + `make harness-smoke
  REQUIRE_DETECTIONS=1` (Jetson-free) all pass in the same land. The smoke is
  the perfect tripwire — it needs no Jetson, so the detections proof is part of
  the PR's own gate, not a deferred operator run.

## Out of scope

- Migrating the per-frame `strafer_episodes.parquet` sidecar (lerobot
  dataset extension) into the LeRobot dataset proper. That's a
  separate `lerobot-extension-columns-as-features` brief.
- Changing the `objects[]` / `rooms[]` payload schema. Those fields stay
  identical to today's sidecar; only the storage location moves (plus an
  additive top-level `strafer_scene_metadata_version` key, allowed by the
  contract's additive-fields policy).

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
