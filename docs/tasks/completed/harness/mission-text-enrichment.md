# Disambiguating mission_text against many-of-a-kind scene clutter

**Type:** new feature (mission-text authoring) + Infinigen extension
**Owner:** DGX agent
**Priority:** P2 — quality lift on every queue-sourced / teleop mission;
the naive template silently trains VLAs on under-specified language.
**Estimate:** L — split into two phases of roughly equal weight:
(v1) the allocentric + conjunctive disambiguator over metadata already
embedded in the scene USD, and (v2) the vendored Infinigen extension
that adds color / material / size descriptors so the disambiguator has
more groundable axes to bind to.
**Branch:** `task/mission-text-enrichment`

## Status

| Phase | State |
|---|---|
| **v1 — allocentric + conjunctive disambiguator (incl. live `room_scope`)** | **shipped** (this PR; stamp below) |
| **filter-vs-emit decision** | **decided + shipped — FILTER** (the generator drops non-groundable targets; see [The measurement & the decision](#the-measurement--the-decision)) |
| **v2 — Infinigen color/material/size descriptors** | remaining — separate future PR |
| further filter tuning (lift groundable yield beyond room_scope) | remaining — folds into v2 |
| parity follow-up (picker + old `MissionGenerator` route through the builder) | remaining — fast-follow |

## Story

As a **VLA training operator who needs unambiguous language labels for
every captured episode** I want **mission_text strings that actually
pick out one object even when a scene holds 233 shelves or 91 bottles**
so that **the policy learns to ground referring expressions rather than
learning that "go to the shelf" is a wildcard satisfied by any shelf
(loss-zero on wrong-shelf predictions, no grounding pressure)**.

## Motivation

Before this brief, every target reference in the offline mission
generator (`build_mission_queue`) and in the teleop picker / the old
bridge `MissionGenerator` emitted a bare `the {label}`.
On `scene_high_quality_dgx_000_seed2` the post-structural-filter target
set is **646 objects across 48 labels**, dominated by a few
high-cardinality classes:

```
233 shelf      22 jar         16 food_box
 91 bottle     21 bowl        14 food_bag
 54 book_stack 17 simple_bookcase
 23 window     17 lamp
```

The naive template produces the same string for all 233 shelves and all
91 bottles. Two failure modes follow: (1) **operator UX** — the picker
shows hundreds of indistinguishable rows; (2) **VLA training
corruption** — one string paired with hundreds of trajectories teaches
the model that the noun phrase is a wildcard. v1 closes this by emitting
one globally-unique anchor per target.

## Acceptance — v1 (shipped)

### Module

`strafer_lab.tools.mission_text_builder` — a **scene-source-agnostic,
pure-Python** builder. No `pxr`, no renderer, no source-specific name
token. It reads only the documented `objects[]` fields (`label`,
`position_3d`, the **flat** `bbox_3d_min` / `bbox_3d_max` keys) and
treats `label` as an opaque category noun.

```python
@dataclass(frozen=True)
class AnchorResult:
    text: str          # the anchor noun phrase
    groundable: bool    # True for every tier except the coordinate escape valve
    tier: str           # which qualifier resolved it

def disambiguate(target_obj: dict, all_objects: Sequence[dict],
                 rooms: Sequence[dict]) -> AnchorResult: ...
```

`groundable` is `False` **iff** the only way to make the string unique
was a raw coordinate — a camera-grounded model cannot read a coordinate
off an image. `tier ∈ {singleton, room_scope, z_extremum, size_extremum,
nearest_neighbor, conjunction, coordinate_fallback}`.

### Qualifier waterfall (least-effort, cheapest-first)

Build the same-normalized-label competitor set **after** excluding
`scene_classes.STRUCTURAL_CLASSES` and invalid/origin positions (the
exact post-filter target set the generator already uses). Accumulate the
minimal qualifiers, breaking the instant the competitor set is reduced
to one; a qualifier is composed **only** if it strictly reduces the set.

1. **room_scope** — **LIVE.** Resolve the target's containing room
   point-in-polygon against `rooms[].footprint_xy` (objects carry
   `room_idx = null`, so membership is recomputed; the ray-cast is a local
   copy of `scene_connectivity.point_in_polygon`, so the builder keeps its
   single project dependency and stays scene-source-agnostic). When that
   room's `room_type` is **globally unique** in the scene, restrict the
   competitor set to same-room same-label objects and carry an
   `in the {room_type}` suffix. Applied **first, as a competitor-set
   SCOPE** — the extrema/neighbour tiers then run within it, so the grammar
   composes `the largest {label} in the {room_type}`. Adopted only if it
   strictly shrinks the competitor set. **Room-type uniqueness gate:** seed2
   has 2 bedrooms / 2 bathrooms / 2 closets, so `the bedroom` is itself
   ungroundable — room_scope does **not** fire on a non-unique room_type and
   the global tiers run unchanged. `room_type` is treated as an opaque
   `strip().lower()` string; a bearing-tainted room_type is guarded the same
   way neighbour labels are.
2. **region_scope** — reserved no-op. Any within-room region split could
   only be phrased with banned bearing words. Authors nothing.
3. **z_extremum** — the lone lowest / highest competitor by vertical
   position: `the lowest {label}` / `the highest {label}`.
4. **size_extremum** — the lone largest / smallest competitor by bbox
   volume (from the flat keys): `the largest {label}` /
   `the smallest {label}`. Skips cleanly when bbox is absent / degenerate
   (so an unknown-size competitor can never let the target falsely claim
   to be the largest).
5. **nearest_distinct_neighbor** — `the {label} next to the {neighbour}`,
   where the neighbour is the nearest **uniquely-named** object within a
   `NEIGHBOR_PROXIMITY_M = 2.0 m` adjacency band (so "next to" stays
   groundable) and the phrase partitions the competitors. Euclidean on
   `position_3d`; distinct-label anti-stutter guard.
6. **conjunction** — up to three of the above
   (`the lowest {label} next to the {neighbour}`).
7. **coordinate_fallback** — terminal escape valve keyed on the
   **full-precision** `position_3d` (round-trippable, so distinct
   positions yield distinct strings). The sole string-uniqueness
   guarantee; `groundable = False`.

### Hard groundability constraint

A holonomic mecanum robot decouples heading from travel and the anchor
is authored before any trajectory exists, so **no** emitted text carries
a sidedness / cardinal / surface-bearing word
(`north|south|east|west|left|right|wall`). The only surface forms a v1
anchor may take are the tiers listed above plus the allocentric
`in the {room_type}` room scope — all groundable. A bearing word can only enter
through the opaque object label; a (data-defective) bearing-tainted
label degrades to a label-free, un-groundable coordinate anchor rather
than emit the banned word, so the constraint holds on the **output**
regardless of input cleanliness. This matches the ban the generator
already enforces
(`test_build_mission_queue.test_no_cardinal_or_wall_language_in_any_text`).

### Consumer wiring

- **`build_mission_queue.py` — PRIMARY required consumer.** The target
  loop computes the `AnchorResult` **once per target**
  (`mission_text_builder.disambiguate(obj, all_objects, rooms)`) and
  threads `.text` as `ref` into the text helpers (`endpoint_text`,
  `path_shape_text`, `fallback_paraphrases` — all now take the precomputed
  `ref`, not the target + competitor set). This replaced the previous
  ~3×-per-target recomputation inside each helper, so the dead
  `target_noun_phrase` seam was removed.
- **Groundable filter (the shipped decision).** With the result in hand the
  loop gates on `.groundable`: `if require_groundable and not
  result.groundable: stats.reject("ungroundable_target"); continue`. A
  `require_groundable: bool = True` `GeneratorConfig` field (CLI
  `--require-groundable` / `--no-require-groundable` on
  `build_mission_corpus.py`) lets an A/B run emit the un-filtered pool.
- **Room-suffix de-dup.** When the anchor already carries `in the
  {room_type}` (room_scope fired) the consumer's cross-room room naming is
  suppressed so the text never reads `... in the kitchen in the kitchen`.
- **Byte-identical:** a target with no same-label competitor still emits
  `the {label}` byte-for-byte (`AnchorResult(text, groundable=True,
  tier="singleton")`), so the singleton case is unchanged.
- **Parity follow-up (remaining):** `teleop_mission_picker
  .load_candidates_from_data` (`go to the {normalized}`) and the old
  `sim_in_the_loop.mission.MissionGenerator._build_raw_command`
  (`Go to the {label}.`) are live producers but are a parity
  fast-follow, not v1 blast radius. `build_mission_queue.py` is a
  parallel producer, not a replacement for `MissionGenerator`
  (`run_sim_in_the_loop.py` selects via `--mission-queue`).

### Tests

`source/strafer_lab/tests/harness/test_mission_text_builder.py`
(+ frozen fixture `fixtures/scene_high_quality_dgx_000_seed2.json`,
a trimmed capture of the scene USD's `customData`):

- **Global string-uniqueness snapshot** over seed2's full 646-target
  set — every `.text` distinct and deterministic; the coordinate escape
  valve is terminal and its strings never collide (the scene has zero
  exact-position collisions, asserted).
- **The measurement** — per-tier histogram + coordinate-fallthrough rate
  + **groundable target yield** + per-label worst-offender breakdown
  (pinned to the room_scope-live numbers).
- **Determinism** — same scene + target → identical `AnchorResult`.
- **Per-qualifier unit scenes** — each live tier individually necessary
  and exercised (z_extremum, size_extremum, nearest_distinct_neighbor,
  conjunction, coordinate_fallback).
- **room_scope unit scenes** — unique room_type resolving alone
  (`the {label} in the {room_type}`); composing with an extremum
  (`the largest {label} in the {room_type}`); a non-unique room_type
  (2 bedrooms) that does **not** fire and never emits `in the bedroom`;
  and a singleton-in-a-unique-room that stays byte-identical `the {label}`.
- **No-bearing-word** over seed2 (incl. coordinate strings) +
  allow-list-grammar check (now permits the trailing `in the {room_type}`)
  + the bearing-tainted-label guard.
- **Byte-identical** singleton case.
- **Second-scene robustness** — an independent synthetic dense scene
  (seed1's USD currently embeds zero objects, so a synthetic dense scene
  stands in for the brief's original "repeat on seed1" check — see the
  drift note below).
- **Generator filter** (`test_build_mission_queue.py`) — a
  forced-coordinate-fallback target is rejected with reason
  `ungroundable_target` under the default and emitted under
  `--no-require-groundable`; plus the room-suffix de-dup helpers.

`make test-lab-pure`: **505 passed, 1 skipped** (the existing generator
tests change only for the filter + the once-per-target threading refactor).

### The measurement & the decision

Over seed2 (646 post-filter targets), with the **live room_scope**,
resolution lands:

| tier | count | share |
|---|---|---|
| singleton | 9 | 1.4% |
| room_scope | 12 | 1.9% |
| z_extremum | 54 | 8.4% |
| size_extremum | 27 | 4.2% |
| nearest_neighbor | 5 | 0.8% |
| conjunction | 82 | 12.7% |
| **coordinate_fallback** | **457** | **70.7%** |

**GROUNDABLE TARGET YIELD = 189 / 646** — the headline: the corpus
target-pool size once the generator filters the coordinate fallthrough
(up from 156 before room_scope).

**Coordinate-fallthrough rate = 457 / 646 = 70.7%** (down from 75.9%
allocentric-only). The room scope is not a silver bullet — it shifts work
into the composing `conjunction` tier (18 → 82) by letting an extremum
resolve *within* a uniquely-typed room, lifting groundable yield by 33,
but the long tail of same-label clutter inside a single room still falls
to the coordinate valve. Worst offenders: shelf 218 / 233 → coordinates
(15 groundable, was 9), bottle 83 / 91, book_stack 44 / 54, window
13 / 23.

**Decision — FILTER (shipped).** A coordinate anchor
(`...approximately at (x, y, z)`) is ungroundable by a camera-grounded
VLM, so a mission must name an object the model can see. The generator now
drops `groundable == False` targets by default (`require_groundable`,
reject reason `ungroundable_target`); `--no-require-groundable` keeps the
old emit-everything behaviour for A/B measurement. v2's color/material
axes (and any further filter tuning) are the lever to lift the 189 yield
further.

**This unblocks the real-Qwen corpus run** with a groundable-only target
pool of **189 missions/scene** on seed2.

## Acceptance — v2 (remaining): Infinigen extension for color / material / size

Adds physically-motivated material/color/size descriptors to every
spawned asset. Strictly additive: v1 works without v2; v2 multiplies the
groundable axes (11 colors × ~8 materials × 3 size buckets) so the
disambiguator escapes far fewer of the 457 still-coordinate targets to
coordinates.

- **Per-asset descriptor capture hook.** A vendored mixin under
  `source/strafer_lab/strafer_lab/infinigen_extensions/`
  (`DescriptorMixin` / `patch_descriptors`) hooks `finalize_assets` to
  record `dominant_color_hsv`, `dominant_material_kind`, `size_bucket`,
  and the most-specific subclass name. Registered via a one-liner
  monkey-patch in the prep entry script; upstream Infinigen stays
  untouched (vendored, not forked).
- **Plumbing.** Extractor writes a `descriptors` sub-dict per `objects[]`
  entry (`color_name`, `color_hsv`, `material`, `material_subclass`,
  `size_bucket`); rides inside the USD `customData` for free. HSV → 11
  basic-color bucketing + the size bucketer live in
  `strafer_lab.tools.color_buckets`.
- **Disambiguator promotion.** The waterfall gains color + material
  qualifiers **ahead of** `nearest_distinct_neighbor`
  (`the {color} {label}` / `the {material} {label}`), composing into the
  conjunction. Opt-in behind a `--with-descriptors` flag; existing
  scenes without descriptors keep working (the builder skips qualifiers
  whose fields are missing).
- **Tests.** HSV bucketer correctness; the same global-uniqueness
  snapshot plus an assertion that the average anchor length drops vs v1.

The v2 Infinigen-extension audit (where each asset plugs in, the tag
system, what survives to metadata today, the vendored-mixin pattern, and
the cost/benefit) is retained in the [Operator FAQ](#operator-faq)
below — it is the v2 design record, not v1 scope.

## Prerequisites & corrected anchors

The brief's earlier revision drifted; corrected against the repo as it
exists:

- **Metadata lives in the scene USD `customData`**
  (`strafer_scene_metadata`, read by `scene_metadata_reader.load`); the
  on-disk `scene_metadata.json` sidecars are **retired** and may be
  stale — do not read them. Cite by symbol, not line.
- **`objects[].uid` does not exist** (parked under
  `scene-contract-instance-discriminator`); `instance_id` is the
  factory-class id (shared by siblings, `-1` for structural) and is not
  a uniqueness key. The builder keys uniqueness on `position_3d`, never
  on an id or a name token.
- **`rooms[]` is populated on the high-quality scenes** (seed2 has 9
  rooms, each with `room_type` + `footprint_xy`). Objects carry
  `room_idx = null`, so membership is recomputed point-in-polygon — this is
  what makes `room_scope` LIVE in v1. `relations` / `materials` remain
  empty on every current object, so region/relational/color/material tiers
  stay reserved no-ops.
- **`scene_high_quality_dgx_000_seed1`'s USD currently embeds zero
  objects** (only its retired sidecar holds them), so it cannot serve as
  the second real measurement scene the original brief assumed; seed2 is
  the sole real scene, with a synthetic dense scene covering the
  second-scene robustness invariant.
- Reuses only agnostic contract seams (`scene_classes.STRUCTURAL_CLASSES`
  and a locally-defined `str.strip().lower()` normalization). Does **not**
  reuse `spatial_description.classify_bearing` / `classify_region` (both
  robot-egocentric, returning the exact banned tokens) nor
  `scene_labels.ObjectEntry.bbox_3d_min/max` (reads a nonexistent nested
  `bbox_3d.min` → zeros; the builder reads the flat keys directly).

## Operator FAQ

_The v2 design record — three questions the operator raised in PR #63
review. Retained for the v2 phase; not v1 scope._

### Q1. Would this involve Infinigen extensions? Maybe it should.

**Verdict: yes, in scope as v2.** The extension surface is a
finalize-assets hook, not a factory rewrite — every
`finalize_assets(assets)` call already has the per-instance color /
material / scale state in memory; recording it costs one dict assignment
per spawned asset, deterministic given the factory seed. The mixin is
vendored under `source/strafer_lab/strafer_lab/infinigen_extensions/`
and registered via a one-liner monkey-patch at prep time, so upstream
Infinigen pulls cleanly. Cost: ~1 day + the HSV→color-name bucketer.
Benefit: two high-cardinality groundable axes that combine
multiplicatively with the allocentric ones.

### Q2. Is there a standard format for extending Infinigen?

**Yes — the `AssetFactory` subclass pattern + the `Semantics` tag enum.**
The constraint solver already attaches semantic tags to every object
state during a generation pass, and `extract_scene_metadata` already
serializes `semantic_tags` / `relations` / `materials` / per-room
geometry. Color is the conspicuous gap (baked into the shader graph,
never read back); material class is opaque (we know which family was
sampled but discard it after finalize). v2's `descriptors` sub-dict fills
that gap. No community extension carries color/material out to a sidecar;
the closest precedent records material slot names per render frame, not
per scene.

### Q3. The naive "go to the {label} {spatial} in the {room} near the {neighbor}" isn't powerful enough.

**Agreed — that's the failure mode, and the measurement quantifies it.**
A single neighbor anchor devolves to the same shared landmark for most of
a 233-shelf cluster. Room scope is now **live but composing**, not a
standalone fix: it only helps the targets in a uniquely-typed room and
only by letting an extremum resolve *within* that room — which is why it
lifts groundable yield from 156 to **189 / 646 (29.3%)** rather than
solving the long tail. v1 leans on **conjunctive** constraints
(room_scope + extrema + neighbour, accumulated until the competitor set is
one); the remaining 70.7% coordinate fallthrough is the evidence for the
shipped FILTER and for v2's color/material axes.

## Out of scope

- v2 color / material / size descriptors (separate future PR) and any
  further filter-yield tuning beyond room_scope.
- Region / relational qualifiers (reserved no-ops until `relations`
  populate and a groundable within-room grammar that avoids bearing words
  is designed). `room_scope` itself is now LIVE — no longer out of scope.
- Hand-authored / free-form LLM mission_text — the paraphrase pass and
  `mission-generator` own stylistic variation; this brief produces the
  canonical programmatic anchor that rides underneath.
- Runtime re-disambiguation against a moved object; cross-scene canonical
  IDs; forking Infinigen; visual (texture/decal) descriptors.

## Triggered by

PR #63 testing feedback — the picker still presented many
indistinguishable rows for the heavy-cardinality labels. The prior audit
filed this brief with bearing-word spatial qualifiers and punted color /
material as "would require Infinigen extensions"; the operator pushed
back on three fronts (the [Operator FAQ](#operator-faq)). This is the
honest version: v1 ships a groundable allocentric disambiguator + the
measurement that decides what comes next.

---

_Shipped (v1): allocentric + conjunctive disambiguator at work commit
`c7429bb`; live `room_scope` + the groundable FILTER at work commit
`4d465df`; branch `task/mission-text-enrichment` (PR pending
open). Groundable yield 189 / 646 on seed2 (70.7% coordinate
fallthrough). Remaining phases: v2 color/material descriptors and any
further filter-yield tuning._
