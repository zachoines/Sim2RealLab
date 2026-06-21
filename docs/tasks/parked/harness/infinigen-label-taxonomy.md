# Comprehensive Infinigen object→label taxonomy for scene metadata

**Type:** quality refactor (label-inference coverage)
**Owner:** DGX agent
**Priority:** P3 — does not block any acceptance bar; the current
seed list + longest-tag fallback produces usable labels today. This
brief upgrades label *quality* (which feeds detection vocab + grounding).
**Estimate:** M (~1–2 days — enumerate Infinigen factory classes, build
the map, decide canonical-noun vs Infinigen-tag policy, add a pxr-free
mapping test, re-author + spot-check a scene's detection labels).
**Branch:** `task/infinigen-label-taxonomy`

**Blocked on / trigger:** Filed-on-trigger. Un-park when a detection-vocab
or grounding-quality measurement (e.g. the harness `observation.detections.*`
column review, or a VLM-grounding eval) shows the longest-tag fallback in
[`_infer_label`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py)
producing noisy / non-canonical labels that hurt detection recall or
referring-expression grounding. Until then the hand-picked priority list
([`LABEL_CATEGORY_PRIORITY`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py))
+ longest-tag fallback is sufficient.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`docs/SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) §b — the `objects[].label` field contract this improves.

## Motivation

`extract_scene_metadata._infer_label` turns an Infinigen object's tag set
into a single `objects[].label` string. Today it:

1. prefers a tag in `LABEL_CATEGORY_PRIORITY` — a small, hand-picked seed
   list of 8 concrete-noun categories (`Table`, `Chair`, `Bed`, `Door`,
   `Window`, `Lamp`, `Sink`, `Couch`), externalized from the function body
   in `scene-connectivity-validation` (PR #96) so the taxonomy lives in one
   place;
2. otherwise falls back to the **longest tag string**, then to the prim
   name stem.

The longest-tag fallback is a heuristic, not a taxonomy: an object whose
most-specific tag is `Semantics(food-pantry)` or a factory class like
`NatureShelfTrinketsFactory` gets a long, non-canonical label. That label
flows into two quality-sensitive places:

- **Detection vocab** — `objects[].label` becomes the `UsdSemantics` class
  Replicator's `bounding_box_2d_tight` annotator boxes on (the
  `observation.detections.*` column). Non-canonical labels fragment the
  vocab and weaken detection training.
- **Grounding** — the VLM grounds referring expressions against the label
  set; noisy labels reduce grounding precision.

This brief builds a comprehensive, canonical Infinigen factory-class →
label map (or a principled policy) so the inferred labels are stable,
human-named nouns across the full Infinigen object catalog, not just the
8 seeded categories.

## Acceptance (sketch — refine on pickup)

- [ ] Enumerate the Infinigen factory classes that appear in the generated
      corpus (walk `Assets/generated/scenes/*` object labels + the Infinigen
      factory registry) and map each to a canonical lowercase noun.
- [ ] Replace / extend `LABEL_CATEGORY_PRIORITY` + the longest-tag fallback
      with the map (keep a sane fallback for unmapped classes).
- [ ] pxr-free unit test: a representative tag/factory set maps to the
      expected canonical labels.
- [ ] Re-author one scene's metadata and spot-check the `objects[].label`
      distribution (fewer non-canonical / over-long labels).
- [ ] Update `SCENE_PROVIDER_CONTRACT.md` §b `label` wording if the
      inference policy changes.

## Out of scope

- Per-instance disambiguation of many-of-a-kind clutter (263 shelves all
  labeled `shelf`) — that is
  [`mission-text-enrichment`](mission-text-enrichment.md)'s `descriptors`
  namespace, which composes with (does not replace) a clean base label.
- Changing the metadata storage backend or the detections column shape.
