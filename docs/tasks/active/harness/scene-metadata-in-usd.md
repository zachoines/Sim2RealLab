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
- [ ] **Test strategy (keeps the fast venvs `pxr`-free):**
  - Pure-data seams for the fast suites — `teleop_mission_picker` already has
    `load_candidates_from_data`; add the equivalent for `scene_labels`. The
    `.venv_harness` (`make test-harness`) and `.venv_vlm` (`make test-dgx`,
    autonomy) suites test against in-memory dicts — no `pxr`.
  - The reader's USD round-trip + `test_scene_provider_contract.py`'s parity
    check are `pytest.importorskip("pxr")`-gated and run in the Kit suite
    (`run_tests.py`). *(Alternative for fuller fast-suite coverage:
    `pip install usd-core` into both venvs and run the reader there.)*
  - `make test-dgx`, `make test-harness`, and the Kit suite all green; **no
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
- **`pxr` in the fast test venvs**: reading `customData` needs `pxr`, which
  `.venv_harness` and `.venv_vlm` lack — and with the sidecar gone there is no
  JSON fallback for them. Mitigated by the pure-data seams + the
  `pxr`-gated reader test (Acceptance): the reader is the *only* `pxr`-bound
  path and the fast suites never hit it. (Or add `usd-core` to both venvs.)
- **Tooling unaware of `customData`**: usdview / Omniverse Composer show the
  embedded dict in the property panel (benign). Replicator's annotators don't
  read `customData`, so RL / perception pipelines are unaffected.

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
