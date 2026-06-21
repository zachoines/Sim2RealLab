# Source-agnostic per-physical-instance discriminator in the scene contract

**Type:** contract refinement (removes a consumer→Infinigen-convention soft coupling)
**Owner:** DGX agent
**Priority:** P3 — the boundary holds today via a working fallback; this removes a soft coupling, it does not fix a break.
**Estimate:** S (one contract field + the Infinigen adapter populates it + retrofit the one consumer that reaches for the prim token).
**Branch:** `task/scene-contract-instance-discriminator`

**Blocked on / trigger (filed-on-trigger):** un-park when **either** (a) `mission-text-enrichment` is picked up and wants a stable per-instance handle, or (b) a **second (non-Infinigen) scene source** is brought up and we want consumers to stop relying on Infinigen prim conventions even as a fast path. Until then the existing fallback is sufficient — this is a cleanup, not a blocker.

## Story

As a **consumer of `SCENE_PROVIDER_CONTRACT.md`** I want **a source-agnostic per-physical-instance discriminator on each `objects[]` entry** so that **the picker (and any future consumer) stops reaching for the Infinigen `__spawn_asset_<N>_` prim token to tell two same-label objects apart, and a non-Infinigen source de-dups on a clean contract field rather than relying on the fallback tuple.**

## Context bundle

- [`SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) — the `objects[]` schema (`instance_id` / `prim_path` notes) + the new "Consumer obligations" section this brief closes the one named soft coupling in.
- [`mission-text-enrichment`](../harness/mission-text-enrichment.md) — the disambiguator that needs per-instance identity; the primary beneficiary.

## Context

The contract is honest about the gap today (the `objects[]` table): `instance_id` is **not unique per object** — for Infinigen it is the *factory-class* id, so all 23 `BowlFactory` bowls share one id. To tell instances apart, the picker's **primary** de-dup key is the Infinigen `__spawn_asset_<N>_` token in `prim_path` ([`teleop_mission_picker.py`](../../../../source/strafer_lab/strafer_lab/tools/teleop_mission_picker.py), `_SPAWN_ASSET_RE`), with a `(label, instance_id, prim_path)` fallback.

This is a **soft coupling, not a break**:
- It does not break a non-Infinigen source — that source has no `__spawn_asset_` token, so the picker falls back to `(label, instance_id, prim_path)`, which is unique for any source that gives each object a distinct `prim_path` (a USD invariant). So consumers **degrade gracefully** today.
- But it is a consumer (the picker) knowing about an *Infinigen prim convention* as its fast path — the exact thing the scene-source-agnostic boundary wants to avoid. And it leans on `prim_path` being present + unique, which the contract only marks *recommended*.

## Approach

Add a source-agnostic per-physical-instance discriminator the adapter mints and the consumer de-dups on:

- **Contract:** add `objects[].uid` (or similar) — **required**, a string unique per physical object **within a scene** (not across seeds). The contract states it is the canonical instance handle; `instance_id` stays as-is (the factory/class stamp); `prim_path` stays recommended for prim labelling but is no longer the de-dup mechanism.
- **Infinigen adapter:** `extract_scene_metadata` mints `uid` per object (it already has the `__spawn_asset_<N>_` token + the prim path to derive a stable per-instance value).
- **distractor-asset-injection** + any future source: mint `uid` at author time (it already mints non-colliding `instance_id`s — extend to a unique `uid`).
- **Consumer retrofit:** `teleop_mission_picker` de-dups on `uid`, deleting `_SPAWN_ASSET_RE`. After this, a grep of the consumer for `spawn_asset` / `prim_path` returns zero — the litmus in the contract's Consumer-obligations section passes for the picker.

## Acceptance

- [ ] `objects[].uid` specified in `SCENE_PROVIDER_CONTRACT.md` (required, per-scene-unique, source-agnostic); the "Known soft coupling" note updated to "resolved".
- [ ] Infinigen adapter (`extract_scene_metadata`) populates `uid`; the corpus regenerated so existing scenes carry it.
- [ ] `teleop_mission_picker` de-dups on `uid`; `_SPAWN_ASSET_RE` removed; the consumer litmus grep (`infinigen|Factory|spawn_asset|prim_path|bpy`) returns zero hits.
- [ ] Round-trip + picker tests green; `make test-lab-pure` green. Brief shipped to `completed/`.

## Out of scope

- **Cross-scene / cross-seed identity** — `uid` is per-scene only (the contract already says cross-scene identity is not well-defined in Infinigen's model).
- **Disambiguation language** — that's `mission-text-enrichment` (which disambiguates by *position / relations*, not by `uid`; `uid` is for de-dup + provenance, not for the human-readable string).
