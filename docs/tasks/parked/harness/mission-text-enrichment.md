# Disambiguating mission_text against many-of-a-kind scene clutter

**Type:** new feature (mission-text authoring) + Infinigen extension
**Owner:** DGX agent
**Priority:** P2 — quality lift on every teleop / queue-source mission;
not gating a current acceptance bar but the naive template silently
trains VLAs on under-specified language.
**Estimate:** L — split into two sub-deliverables of roughly equal
weight: (v1) the spatial + conjunctive disambiguator that uses
metadata already in `scene_metadata.json`, and (v2) the Infinigen
extension that adds color / material / size descriptors per object
so the disambiguator's conjunctive language has more axes to bind to.
**Branch:** `task/mission-text-enrichment`
**Blocked on:** _Resolved_ — [`scene-provider-contract`](../../completed/scene-provider-contract.md) shipped (PR #66). This brief's disambiguator consumes contract-conformant metadata and the reserved `objects[].descriptors` namespace per [`SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md); the contract reserved the namespace and the `rooms[]` absence-handling rules so this brief can plug in without forcing a contract re-ship.

## Story

As a **VLA training operator who needs unambiguous language labels
for every captured episode** I want **mission_text strings that
actually pick out one object even when there are 23 bowls or 179
bottles in the scene** so that **the policy learns to ground
referring expressions (rather than learning that "go to the bowl"
means "go to any bowl, the loss won't care")**.

## Motivation

PR #63's teleop driver emits `mission_text = f"go to the {label}"`
([`teleop_mission_picker.py:267`](../../../../source/strafer_lab/strafer_lab/tools/teleop_mission_picker.py#L267))
for every picker candidate. On `scene_high_quality_dgx_000_seed1`,
the picker offers **753 entries** drawn from the following label
histogram (top 10, from `picker_out_log.txt`):

```
263 shelf       17 food_box
179 bottle      12 lamp
 63 book_stack  12 hardware
 24 can         10 plant
 23 bowl         8 cup
 20 food_bag
 19 jar
 19 cabinet
 17 window
```

The naive template produces the same string ("go to the bowl") for
all 23 bowls, the same string ("go to the bottle") for all 179
bottles. Two failure modes follow:

1. **Operator UX.** The picker UI shows 23 entries labelled "bowl"
   that differ only by `instance_id` and `pos=(x,y,z)`. The operator
   picks one — say `[256] bowl id=6738208 pos=(+13.37, +15.93, +0.98)`
   — and the recorded `mission_text` is the same string the other 22
   bowls' rows would produce. There is no language record of which
   bowl was driven to.
2. **VLA training corruption.** The same `mission_text` is paired
   with 23 different trajectories. The model can either (a) learn
   that "the bowl" is a wildcard satisfied by any bowl (loss-zero on
   wrong-bowl predictions, no grounding pressure), or (b) memorize
   which trajectory came first in shuffle order. Both fail open at
   eval against a referring-expression test set.

A prior audit at this brief's earlier revision proposed adding
spatial qualifiers (room membership, bearing, neighbor anchor). That
gets us part of the way but is **insufficient at the scale shown
above** — many of the 23 bowls sit on a single shelf in a single
room with no nearby distinct anchor, and the spatial qualifier
degenerates ("the bowl in the kitchen, near the counter" picks out
the same 8 bowls). To disambiguate at that scale we need a richer
descriptor vocabulary: color, material, size, and conjunctive
relational language. The first three require an **Infinigen
extension**, which the prior audit had put out of scope. The operator
pushback ("would this involve Infinigen extensions? Maybe it
should.") is correct — the cost is bounded (see
[Operator FAQ](#operator-faq) below) and the disambiguation power
gained is decisive.

## Acceptance

Ship the work in two phases against one brief, one PR per phase:

### v1 — Spatial + conjunctive disambiguator (no Infinigen extension)

Uses only metadata already present in `scene_metadata.json` today
(plus the rooms[] block once `infinigen-scene-corpus` lands its
Blender-path extraction).

- [ ] **New module `strafer_lab.tools.mission_text_builder`** that
  takes a `MissionCandidate` (or one row of `objects[]`) plus the
  full `scene_metadata.json` and returns a disambiguating
  `mission_text`. Pure-Python (no `pxr`, no `bpy`); reuses
  [`spatial_description.classify_bearing`](../../../../source/strafer_lab/strafer_lab/tools/spatial_description.py#L71)
  + [`scene_labels.get_room_at_position`](../../../../source/strafer_lab/strafer_lab/tools/scene_labels.py#L148)
  + the existing region-bucket logic.
- [ ] **Disambiguator algorithm.** Given a target candidate `T`:
  1. Compute the **competitor set** = all other candidates with the
     same normalized label. If `len(competitors) == 0`, emit
     `"go to the {label}"` and stop.
  2. Apply qualifiers in order of cheapness, accumulating until
     **one** candidate is selected:
     - **Room scoping.** "in the {room_type}_{room_idx}" — drops
       competitors not in the same room. (Needs populated `rooms[]`;
       see prereq below.)
     - **Region scoping within room.** Cluster competitors by
       (xy)-DBSCAN with `eps ≈ 1.5 m`, label clusters by their
       relative compass position inside the room polygon
       ("northeast corner", "near the south wall"). Drop competitors
       not in `T`'s cluster.
     - **Vertical / horizontal extreme.** If `T` is the cluster's
       `argmin(z)` or `argmax(z)`, emit "the lowest / highest
       {label}". Same for argmin/argmax along the room's longer
       axis: "the leftmost / rightmost {label} on the shelf".
     - **Nearest-distinct-neighbor anchoring.** Find the closest
       object of a **different** label whose `(label, position)`
       uniquely picks out `T` from the competitor cluster. Emit
       "next to the {neighbor_label}". The "distinct" requirement
       prevents "the bowl next to the bowl" pathology.
     - **Relational anchoring.** If `T.relations` contains a
       `SupportedBy` / `StableAgainst` edge with a uniquely-named
       parent (e.g. a specific cabinet instance), prefer that:
       "on the {parent_label}_{parent_instance_id}".
  3. **Conjunction.** If no single qualifier disambiguates,
     compose two of the above with "and": "the {label} in the
     {room} near the {neighbor1} and to the right of the
     {neighbor2}". Up to three conjuncts; if still ambiguous,
     fall back to a positional tie-breaker
     "(approximately at xy = (13.4, 15.9))" so the row is at
     least uniquely keyed. This is the escape valve; every other
     mechanism is a quality improvement on top of it.
- [ ] **Picker integration.** `teleop_mission_picker.load_candidates`
  routes every produced `MissionCandidate.mission_text` through the
  new builder. The console display also shows the disambiguated
  string under each candidate so the operator can verify before
  picking. Defaults preserved: legacy `"go to the {label}"` is
  emitted when the builder returns no qualifier (singletons +
  failure fallback only).
- [ ] **Mission-queue integration.** `MissionGenerator` (bridge
  harness) calls the same builder so teleop and bridge produce
  matching strings on the same scene. Same import path; no
  duplicated logic.
- [ ] **Tests.**
  - Unit: each qualifier in isolation against a synthetic 5-object
    scene where the disambiguator is known to need exactly that
    qualifier (room-scope only, cluster only, extreme only,
    neighbor only, conjunctive).
  - Snapshot: run against `picker_out_log.txt`'s actual scene
    (`scene_high_quality_dgx_000_seed1`) and assert every produced
    `mission_text` is **globally unique** across the 753 candidates.
    The test fails if any string repeats — the central correctness
    bar.
  - Stability: assert determinism — same scene + same target
    produces the same string across runs (no randomization
    in the qualifier choice).
- [ ] **Documentation.** Cheatsheet picker section gains a one-line
  example of the new mission_text format.

### v2 — Infinigen extension for color / material / size

Adds physically-motivated material/color/size descriptors to every
spawned asset. Strictly additive: v1 still works without v2; v2
makes v1's "conjunction" step more expressive.

- [ ] **Per-asset descriptor capture hook.** A small mixin
  `strafer_lab.infinigen_extensions.DescriptorMixin` (lives in a
  new package directory under `source/strafer_lab/strafer_lab/`,
  vendored alongside Infinigen rather than upstreamed) that hooks
  `AssetFactory.finalize_assets` to record:
  - `dominant_color_hsv`: median HSV of the primary material
    sampled at generation time (`self.surface` for tableware; the
    materials assignment dict for furniture).
  - `dominant_material_kind`: one of `{wood, metal, ceramic, glass,
    plastic, fabric, plaster, marble}` derived from the
    [`material_assignments.py`](../../../../infinigen/infinigen/assets/composition/material_assignments.py)
    family the material came from (we already know which family
    was sampled — see
    [`material_assignments.py:266`](../../../../infinigen/infinigen/assets/composition/material_assignments.py#L266)
    for `cup`).
  - `size_bucket`: small / medium / large bucketed against the
    factory class's training-time scale range (e.g. for
    [`BowlFactory.scale = log_uniform(0.15, 0.4)`](../../../../infinigen/infinigen/assets/objects/tableware/bowl.py#L31),
    `< 0.22` is small, `> 0.32` is large).
  - `factory_subclass`: most-specific subclass name beyond the
    canonical label — e.g. `WhitePlywood` for a wood-family cabinet,
    `Marble` for a ceramic-family bowl. Sourced from the surface
    class name verbatim.
  Hook is registered globally via a one-liner monkey-patch in our
  prep_room_usds entry script — does NOT require forking Infinigen.
- [ ] **Plumbing into scene_metadata.json.** `extract_from_state`
  (and `extract_from_blend`, `extract_from_usd` insofar as they can
  recover the data) writes the new fields into each `objects[]`
  entry under a `descriptors` sub-dict:
  ```json
  {
    "descriptors": {
      "color_name": "red",
      "color_hsv": [0.02, 0.85, 0.55],
      "material": "ceramic",
      "material_subclass": "Marble",
      "size_bucket": "small"
    }
  }
  ```
  `color_name` is a coarse bucketing of `color_hsv` against an
  11-color basic-color vocabulary (`{red, orange, yellow, green,
  blue, purple, pink, brown, white, gray, black}`). Existing
  consumers ignore unknown fields, so this is backwards-compatible.
  The HSV bucketer + size bucketer live in
  `strafer_lab.tools.color_buckets` so they're reusable from
  validators.
- [ ] **Disambiguator promotion.** `mission_text_builder` gains two
  additional qualifiers ahead of "nearest-neighbor":
  - **Color.** "the {color_name} {label}" — drops competitors with
    a different `color_name`.
  - **Material.** "the {material} {label}" — drops competitors with
    a different `material`.
  Conjunctions combine them: "the small red ceramic bowl on the
  shelf in the kitchen".
- [ ] **Tests.**
  - Unit: HSV bucketer correctness against a frozen RGB test set.
  - Snapshot v2: same global-uniqueness test as v1, plus assert
    that the average string length **drops** between v1 and v2 (v2
    can use one color word where v1 needed three positional
    qualifiers) — quality + concision both improve.
- [ ] **Migration.** v2 is opt-in via a `prep_room_usds.py
  --with-descriptors` flag; existing scenes without descriptors
  still work (the builder skips qualifiers whose fields are
  missing). New corpus runs get the flag by default after a
  one-scene smoke confirms the descriptor capture doesn't slow
  generation by > 5%.
- [ ] **Documentation.** `extract_scene_metadata.py` docstring
  + `scene_labels.py:ObjectEntry` typed accessor + a new
  `docs/tasks/context/infinigen-extensions.md` context module
  documenting the descriptor schema and the mixin pattern so
  future extensions (e.g. per-object physics annotations, semantic
  function tags) plug in the same way.

### Prerequisites (shared by v1 and v2)

- **Populated `rooms[]`.** `extract_from_usd` can't recover room
  polygons; v1's room-scoping qualifier degrades to "skip" without
  them, leaving the disambiguator weaker on USD-only scenes. The
  brief acknowledges this — same caveat as
  [`infinigen-scene-corpus.md`](infinigen-scene-corpus.md)'s rooms=0
  fallback. v1 with rooms=0 still works; v1 with populated rooms is
  noticeably better.
- **Composes with `scene-metadata-in-usd`.** Once that brief lands
  the `customData` embedding, `descriptors` rides along inside the
  USD's `strafer_scene_metadata` dict for free — no extra plumbing.

## Operator FAQ

The three questions the operator raised in PR #63 review against an
earlier (in-scope/out-of-scope) draft of this brief.

### Q1. Would this involve Infinigen extensions? Maybe it should.

**Verdict: yes, in scope as v2.** The prior audit put color /
material descriptors out of scope on the assumption that extending
Infinigen would be expensive. After investigating the Infinigen
source ([`infinigen/core/placement/factory.py`](../../../../infinigen/infinigen/core/placement/factory.py),
[`infinigen/core/tags.py`](../../../../infinigen/infinigen/core/tags.py),
[`infinigen/assets/composition/material_assignments.py`](../../../../infinigen/infinigen/assets/composition/material_assignments.py),
[`infinigen/assets/colors.py`](../../../../infinigen/infinigen/assets/colors.py))
the cost is actually bounded:

- The extension surface is **a finalize-assets hook**, not a
  factory rewrite. Every `AssetFactory.finalize_assets(assets)`
  call has access to `self.surface`, `self.scale`, and the rest of
  the per-instance state that already determines color / material
  at generation time. Recording it costs one dict assignment per
  spawned asset.
- The information **already exists in the generator's memory**
  during a generation run — `BowlFactory.__init__` calls
  `weighted_sample(material_assignments.cup)` which picks one of
  ~9 material classes. We just need to log which one was picked.
  No forward-pass cost; no extra randomness; deterministic given
  the factory seed.
- The extension is **vendored, not forked**. The mixin lives under
  `source/strafer_lab/strafer_lab/infinigen_extensions/`. A
  one-liner in `prep_room_usds.py` registers it before the
  generation pass starts. Upstream Infinigen stays untouched, so
  pulling new Infinigen versions doesn't conflict.

**Cost vs benefit.** Cost: ~1 day of integration work, plus the
HSV → color-name bucketer (one afternoon). Benefit: the
disambiguator gets two new high-cardinality qualifiers (11 colors,
~8 materials) that combine multiplicatively with the spatial
qualifiers — turning a 23-way ambiguity into ~3-way for free in
the typical case, and into ~1-way once the conjunction picks up.

The cost/benefit math holds even if v2 never lands: v1 alone (no
Infinigen extension) closes the operator-UX gap (every picker entry
becomes distinguishable in the console output) and bounds the worst
case via the positional tie-breaker. v2 is a strict additive win on
top.

### Q2. Is there a standard format for extending Infinigen? Are there existing extensions with info I want?

**Yes.** Infinigen's extension surface is the `AssetFactory`
subclass pattern itself
([`infinigen/core/placement/factory.py:26`](../../../../infinigen/infinigen/core/placement/factory.py#L26)).
Every asset type in `infinigen/assets/objects/` subclasses
`AssetFactory` and overrides at most three methods:

- `__init__(self, factory_seed, coarse)` — pick parameters from
  the seed (this is where colors, scales, material choices get
  sampled).
- `create_asset(self, **params)` — emit the Blender object.
- `finalize_assets(self, assets)` — post-process (assign
  materials, apply wear, etc.). This is the hook v2 piggybacks on.

The **tag system**
([`infinigen/core/tags.py:32`](../../../../infinigen/infinigen/core/tags.py#L32))
is the canonical extension point for semantic annotation.
`Semantics` is an `EnumTag` that already enumerates room types,
furniture function categories (`Storage`, `Seating`, `Bathing`),
access modes (`AccessTop`, `AccessFront`, `AccessHand`), and object
roles (`Dishware`, `Cookware`, `Utensils`). The constraint solver
already attaches these tags to every `ObjectState`
([`infinigen/core/constraints/example_solver/state_def.py:48`](../../../../infinigen/infinigen/core/constraints/example_solver/state_def.py#L48))
during a generation pass.

**What survives to scene_metadata.json today.** Already captured by
`extract_from_state`
([`extract_scene_metadata.py:107`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py#L107)):

- `semantic_tags` — the `Semantics` enum values for each object.
- `relations` — `SupportedBy` / `StableAgainst` / `CoPlanar` /
  `Touching` edges
  ([`infinigen/core/constraints/constraint_language/relations.py:380-410`](../../../../infinigen/infinigen/core/constraints/constraint_language/relations.py#L380)).
- `materials` — Blender material slot names (e.g.
  `"shader_glass.001"`, `"VaseCeramic"`). These leak the surface
  class name but in a noisy form; v2's `material_subclass` is the
  cleaned version.
- Per-room `polygon`, `room_type`, `area_m2`, `story`, and a
  `room_graph` adjacency
  ([`extract_scene_metadata.py:128`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py#L128)).

**What does NOT survive.** Color is the conspicuous gap. Color
choices live in `infinigen/assets/colors.py` (e.g.
[`metal_bw_natural_hsv`](../../../../infinigen/infinigen/assets/colors.py#L369),
[`fabric_hsv`](../../../../infinigen/infinigen/assets/colors.py#L254))
and get baked into the material shader graph but are never read
back. Material class is similarly opaque — we know one of
`(MetalBasic, BrushedMetal, ...)` was chosen but the choice is
discarded after `finalize_assets` runs.

**Existing community extensions?** None that we found that
specifically carry color / material descriptors out to a sidecar
file. The closest precedent is
[`saved_mesh.json`](../../../../infinigen/infinigen/core/util/exporting.py#L339)
which the export pipeline writes per frame — it records material
slot names per object but not color values, and the format is
geared at render-time mesh accounting rather than scene-level
semantic export. Our extension fills that gap.

**Pattern we adopt.** Vendored mixin + `finalize_assets` hook +
one-liner monkey-patch at prep time. Concretely:

```python
# source/strafer_lab/strafer_lab/infinigen_extensions/descriptors.py
def patch_descriptors(asset_factory_module):
    orig = asset_factory_module.AssetFactory.finalize_assets
    def wrapped(self, assets):
        result = orig(self, assets)
        # record self.surface, self.scale, etc. into a dict keyed
        # by the spawned object name(s); flushed by extractor.
        ...
        return result
    asset_factory_module.AssetFactory.finalize_assets = wrapped
```

The hook is opt-in (only `prep_room_usds.py --with-descriptors`
patches it), so default Infinigen behavior is unchanged.

### Q3. The naive template "go to the {label} {spatial} in the {room}_{idx} near the {neighbor}" isn't powerful enough.

**Agreed — that's exactly the failure mode this brief addresses.**
At 23-bowls-in-a-kitchen scale, room scope drops nothing (all bowls
in the kitchen), and "near the {neighbor}" devolves into "near the
shelf" for ~18 of the bowls since they share the same shelf.

This brief's disambiguator uses **conjunctive constraints**, not
single qualifiers, as the default authoring mode. See the
algorithm in `Acceptance v1` above — qualifiers are accumulated
until the competitor set is reduced to 1, not picked one-at-a-time.
That alone closes most cases without v2. The picker_out_log.txt
worked examples below show what each strategy buys us, and which
cases still need v2's color / material axes.

## Approach

### Infinigen extension audit (file:line cites)

| Question | Where the answer lives | Decision |
|---|---|---|
| Where does each asset type plug in? | [`infinigen/core/placement/factory.py:26 — class AssetFactory`](../../../../infinigen/infinigen/core/placement/factory.py#L26) | Subclass it; override `__init__` / `create_asset` / `finalize_assets`. Every concrete asset in `infinigen/assets/objects/` does exactly this. |
| What's the canonical metadata channel? | [`infinigen/core/tags.py:32 — class Semantics(EnumTag)`](../../../../infinigen/infinigen/core/tags.py#L32) | Tag enum already covers room types, furniture functions, access modes, object roles. We add no new enum values; we add a separate `descriptors` sub-dict so v2 doesn't pollute the semantic-tag stream. |
| Where do color / material choices happen? | [`infinigen/assets/objects/tableware/base.py:30 — TablewareFactory.__init__ → weighted_sample(material_assignments.cup)`](../../../../infinigen/infinigen/assets/objects/tableware/base.py#L30); [`infinigen/assets/composition/material_assignments.py:266 — cup = decorative_hard + [...]`](../../../../infinigen/infinigen/assets/composition/material_assignments.py#L266); [`infinigen/assets/colors.py:369 — metal_bw_natural_hsv`](../../../../infinigen/infinigen/assets/colors.py#L369) | Both are picked from `factory_seed` deterministically. Recording them is one extra line in `finalize_assets`. |
| What gets serialized today? | [`source/strafer_lab/scripts/extract_scene_metadata.py:107 — extract_from_state`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py#L107) writes `semantic_tags`, `relations`, `materials`, `room_idx`, `position_3d`, `bbox_3d_*` per object. | Add a `descriptors` sub-dict alongside. Backwards-compatible: existing consumers ignore unknown fields. |
| Do existing community extensions cover this? | Searched `infinigen/`, `infinigen_examples/`, `infinigen_gpl/` for color/material capture — only [`infinigen/core/util/exporting.py:339 — saved_mesh.json`](../../../../infinigen/infinigen/core/util/exporting.py#L339) writes material slot names, no color values. Per-frame, not per-scene. | We're filling a real gap. The mixin pattern is novel for this repo but well-supported by Infinigen's extension surface. |
| Is forking required? | No — the mixin monkey-patches `AssetFactory.finalize_assets` from within `prep_room_usds.py` before Infinigen's generation pass starts. | Stay vendored. Upstream Infinigen can move without merge conflicts. |

### Disambiguation strategy beyond the naive template

The strategy is a **least-effort qualifier waterfall with
conjunctive composition** — not a single template. Quasi-pseudocode:

```
def disambiguate(target, all_candidates, scene_metadata):
    competitors = [c for c in all_candidates
                   if c.label == target.label and c is not target]
    if not competitors:
        return f"go to the {target.label}"

    qualifiers = []
    # qualifiers ordered by (a) information density per word,
    # (b) availability — room-scope drops to no-op without rooms[].
    for fn in [room_scope, region_scope, extremum, color, material,
               size, nearest_distinct_neighbor, relational_anchor]:
        qualifier_text, competitors = fn(target, competitors, scene_metadata)
        if qualifier_text is None:
            continue
        qualifiers.append(qualifier_text)
        if not competitors:
            break

    if competitors:
        # exhausted; use positional escape valve so the row is at
        # least uniquely keyed.
        qualifiers.append(
            f"approximately at xy=({target.position_3d[0]:.1f}, "
            f"{target.position_3d[1]:.1f})"
        )

    return compose(target.label, qualifiers)
```

Compose is a small grammar:

- Adjective-position qualifiers (color, material, size, extremum)
  go **before** the label: "the red ceramic bowl".
- Prepositional qualifiers (room, region, neighbor, relational)
  go **after**: "the red ceramic bowl on the kitchen counter,
  closest to the window".
- "and" joins prepositional clauses past the first; commas join
  adjective clauses.
- Avoid stutter ("the bowl on the bowl shelf") via the
  `distinct-label` constraint in `nearest_distinct_neighbor`.

The waterfall is deterministic — same seed gives same string, so
the snapshot test in `Acceptance v1` can pin the output. Quality
isn't measured against a held-out set in this brief (that's a
follow-up); the bar is *uniqueness* (every produced string picks
out one object) plus *naturalness* (a human reading the string
could find the target in the scene without seeing the position).

### Worked examples against `picker_out_log.txt`

Three examples from the actual 753-entry log. For each: the naive
mission_text, the v1 enriched mission_text, and the v2 enriched
mission_text. Positions are unchanged; only the language varies.

**Example A — densest bowl cluster.** Lines [253-275] of
picker_out_log.txt show 23 bowls. The cluster at xy ≈ (13.4..14.7,
15.9..18.6) holds 8 bowls within roughly a 3×3 m patch
(`[256]`, `[257]`, `[258]`, `[260]`, `[268]` are within ~1 m;
`[255]` slightly farther). Take `[256] bowl id=6738208
pos=(+13.37, +15.93, +0.98)`:

| Version | mission_text |
|---|---|
| Naive | `go to the bowl` |
| v1 | `go to the bowl at mid-height in the kitchen (lowest of the cluster on the west shelf), next to the bottle id=1551991` |
| v2 | `go to the small white ceramic bowl on the west shelf next to the green bottle` |

v1 is verbose but uniquely keyed (the conjunction of "mid-height",
"west shelf cluster", "lowest of cluster", neighbor bottle picks
out one bowl). v2 cuts the same uniqueness down to ~12 words by
spending one color word and one material word — these are
multiplicative axes that the spatial qualifiers can't reach.

**Example B — bottle row on a shelf.** Lines [229-243] of
picker_out_log.txt include bottles at (~15.4, ~18.6) ranged across
z = 0.34 .. 2.06, suggesting 6+ bottles stacked vertically on the
same shelf. Take `[240] bottle id=9430454 pos=(+15.37, +18.54,
+1.37)`:

| Version | mission_text |
|---|---|
| Naive | `go to the bottle` |
| v1 | `go to the bottle in the kitchen (third from the bottom on the east-wall shelf), between the bottle id=9279597 and the bottle id=9519584` |
| v2 | `go to the medium blue glass bottle on the third shelf of the east-wall rack` |

Note v1 violates the "distinct-label neighbor" constraint
intentionally as a fallback when no distinct-label neighbor is
within reach — it falls back to numeric instance_id anchors, which
are unique but unfriendly. v2 sidesteps the failure: color +
material + size buckets give us six bottles' worth of
distinguishability per axis (11 × 8 × 3 = 264 nominal
combinations) so the disambiguator can stay inside the natural
"adjective adjective noun on the ordinal shelf" form.

**Example C — singleton bowl in a different room.** Take `[259]
bowl id=6738208 pos=(-0.04, +10.58, +1.28)` — sits at the negative
x edge of the scene, far from the dense cluster of Example A.

| Version | mission_text |
|---|---|
| Naive | `go to the bowl` |
| v1 | `go to the bowl in the bathroom_2` (room scope alone disambiguates — no other bowl in that room) |
| v2 | `go to the bowl in the bathroom_2` (same; v2 adds no value here, the room scope alone was enough) |

The waterfall stops at the first qualifier that empties the
competitor set, so v2's color/material aren't appended unnecessarily
— concision is preserved.

### Implementation phasing

- **v1 ships first**, depends on nothing beyond the existing
  `scene_metadata.json` schema + the in-flight
  [`infinigen-scene-corpus`](infinigen-scene-corpus.md) work to
  populate `rooms[]`. The disambiguator gracefully degrades to a
  positional fallback when rooms[] is empty, so v1 lands even
  against today's USD-only scenes.
- **v2 ships behind a flag.** `prep_room_usds.py --with-descriptors`
  activates the mixin. New corpus runs use the flag by default
  after a smoke confirms no perf regression. Existing scenes
  continue to work unchanged.
- **v2's release gate.** Snapshot test from v1 still passes
  (uniqueness preserved); average mission_text length drops vs v1
  on the same scene (concision improved); no new
  `make test-lab-pure` failures.

## Out of scope

- **Hand-authored mission_text overrides.** Per-scene operator-edited
  strings ("the bowl Grandma uses") are a different lane —
  `mission-generator`'s LLM-paraphrase pass already covers stylistic
  variation. This brief produces the *canonical* programmatic
  string; paraphrases ride on top.
- **Free-form LLM-emitted mission_text.** That's
  [`mission-generator`](mission-generator.md)'s job. This brief
  feeds it by providing a unique anchor string per target; the LLM
  then paraphrases and adds path-shape language.
- **Real-time disambiguation during a running mission.** The
  builder runs offline (at picker-load time or queue-generation
  time). Runtime re-disambiguation against a moved object would
  need a different design.
- **Forking Infinigen.** v2 stays vendored — see Q2 above. Forking
  doubles the maintenance burden and breaks the "pull upstream
  cleanly" property.
- **Visual disambiguators (texture pattern, decal text).** v2 caps
  at color / material / size. Texture-level descriptors ("the
  striped bowl") would need a render-side pass that's heavier
  than this brief's scope.
- **Cross-scene canonical IDs.** Each scene's disambiguation is
  scoped to its own `objects[]`; a "bowl_id=6738208" in seed1 is
  not the same physical object as the same factory class in seed2.
  Cross-scene identity is out of scope; it's not even well-defined
  in Infinigen's generation model.

## Triggered by

PR #63 testing feedback — operator confirmed that the
753-entry picker (after the dedup work landed in `teleop-ergonomics`)
still presents many indistinguishable rows for the heavy-cardinality
labels (`shelf`, `bottle`, `bowl`, `book_stack`). The prior audit
filed `mission-text-enrichment.md` with spatial qualifiers only and
explicitly punted on color / material as "would require Infinigen
extensions." The operator's review of that draft pushed back on
three fronts (this brief's [Operator FAQ](#operator-faq) section)
and asked for a brief that takes the Infinigen extension cost
honestly. This is that brief.
