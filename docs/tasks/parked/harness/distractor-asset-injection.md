# Inject distractor / filler assets into Infinigen scenes (corpus diversity + sparse-goal fill)

**Type:** new feature (scene-generation pipeline; new scene-content source)
**Owner:** DGX agent
**Priority:** P3 — gates no current acceptance bar (the harness Tier-1 ≥4/≥2-scene bar is met by native Infinigen generation); a quality/scale multiplier.
**Estimate:** L (new injector stage + asset-pool curation/labeling + navigability-aware placement + the per-variant re-process orchestration + provenance).
**Branch:** `task/distractor-asset-injection`

**Blocked on / trigger (filed-on-trigger):** un-park when **either** (a) a real corpus run produces a **goal-sparse scene** that yields `< N` missions — the `objects=0 → 0 missions` incident on a regenerated scene is the filing motivation and exactly this trigger; **or** (b) a VLA/VLM training measurement shows **object-diversity saturation** on the fixed base-scene set (too few distinct object classes/instances limiting grounding generalization). Until a *measured* sparsity/diversity gap exists, native `high_quality_dgx` density + the referring-expression disambiguation already in the generator is sufficient.

## Story

As a **harness operator scaling VLA/VLM training data**, I want **to scatter labeled distractor / filler assets from a curated pool into Infinigen scenes — placed on navigable floor and surfaces without breaking connectivity — and have them flow into `objects[]` + the detections column through the existing authoring path**, so that **goal-sparse scenes become mission-bearing and the corpus gains object-presence diversity without re-running ~30-min Infinigen generation per variant.**

## Context bundle

Read before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) — the binding artifact contract; this is a **new scene-content source** that rides §g/§h.2's "programmatic adapter" path with zero downstream code changes (teleop_capture / picker / writer never import the source).
- Isaac Sim 6.0 Replicator tutorials (the API this builds on): [object-based SDG](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_object_based_sdg.html) + [Infinigen SDG](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_infinigen_sdg.html).

Sibling briefs (cross-reference, do **not** fold):
- [`infinigen-scene-corpus`](../../active/harness/infinigen-scene-corpus.md) — a one-time corpus *operation* ("no new gen script"); this brief is **new code** (a pipeline stage) that scene-corpus can later invoke as a multiplier, not a fold-in.
- [`domain-randomization-audit`](../../active/trained-policy/domain-randomization-audit.md) — *dynamics-side* DR; it explicitly scopes out scene-content randomization. This is the **content-side / object-presence DR** sibling, in the harness lane.
- [`infinigen-label-taxonomy`](infinigen-label-taxonomy.md) — injected-asset labels feed the **same** UsdSemantics detection vocab; the pool's label set must be reconciled with the canonical-noun taxonomy.
- [`mission-text-enrichment`](mission-text-enrichment.md) — the referring-expression generator that owns **disambiguation**. This brief is **NOT** a disambiguation mechanism (see Out of scope).

## Context

### This is diversity + sparse-goal fill, NOT disambiguation

Two **disjoint regimes** — keep them separate:
- **Referring-expression generation** (the generator's target REG + [`mission-text-enrichment`](mission-text-enrichment.md)) owns *"many same-label targets → emit a unique handle"* — a pure offline string operation over fixed metadata that almost always finds a unique handle.
- **Injection** owns *"zero / too-few targets → create some."* Adding an asset to *disambiguate* is the wrong tool: it **mutates the scene** — changing detections GT, occupancy/connectivity, and the spawn-obstacle set for *every* other mission and frame — whereas saying "the small red bowl on the west shelf" about a bowl that already exists is strictly cheaper and safer. The one niche where injection helps grounding is the genuinely goal-sparse case (no target to ground at all).

### It rides the existing authoring path

Replicator's `add_labels(prim, labels=[label], instance_name="class")` is the **same** UsdSemantics call [`extract_scene_metadata.py`](../../../../source/strafer_lab/scripts/extract_scene_metadata.py) already uses, so injected assets enter `objects[]` + the detections column exactly like native objects — *if* they are authored correctly. The catch: stock `extract_from_usd` recovers labels by **parsing Infinigen factory prim names** (`<Factory>_<id>__spawn_asset_<n>_`); injected USDs don't follow that convention, so they would be **skipped or mis-labeled**. The injector must therefore author `objects[]` records + canonical labels **directly** (via `author_scene_metadata` / explicit per-asset label maps), **not** rely on Isaac's filename-regex `load_auto_labeled_assets` (whose `002_banana→banana` vocab does not match this project's canonical nouns or the picker block list).

### The showstopper: full per-variant re-process, enforced

Every injection rewrites the scene USD, and [`check_occupancy_freshness`](../../../../source/strafer_lab/strafer_lab/tools/build_mission_queue.py) **hard-raises `StaleOccupancyError`** on any `usd_mtime_ns`/`usd_size` change vs `occupancy.json` — this is enforced protection, not a warning. So each variant requires, in fixed order: **re-author `objects[]` → re-attach colliders (furniture/convexHull path) → regenerate `occupancy.npy` + `connectivity[]` (the expensive Kit-bound omap pass) → re-run spawn-points.** Corpus multiplication is `N_scenes × N_variants × (Kit-bound re-process)`, dominated by the omap pass. **v1 must scope to the goal-sparse rescue** (inject only into mission-starved scenes), and batch all of a scene's injections into one atomic re-process — never inject incrementally.

## Approach

1. **New `_run_inject_assets` stage** in [`prep_room_usds.py`](../../../../source/strafer_lab/scripts/prep_room_usds.py)'s `generate_scenes` chain, mirroring the existing `_run_postprocess` / `_run_extract_metadata` / `_run_validate_connectivity` helper pattern (`_run_subprocess`, `rc != 0 → append+continue`), gated by a new `--inject-assets` / `--asset-pool` flag (default OFF). Run it **after** export+postprocess (so the injector authors its own convexHull colliders rather than tripping postprocess's Infinigen-prim-name floor/structural regexes) and **before** `extract_scene_metadata` + `validate_scene_connectivity` + `generate_scenes_metadata`. Author prims under `/World/InjectedAssets/...` so they ride the runtime `/World/Room` remount; keep the stage idempotent (clear+re-scatter a known scope).
2. **Navigability-aware, deterministic placement** — *not* the tutorial's table working-area + gravity scatter. Sample candidate floor XY from `occupancy_to_free_space(load_occupancy(scene_dir).grid, robot_radius_m=ROBOT_RADIUS_M)` free cells, constrain to a target room via `point_in_polygon` over `rooms[].footprint_xy`, exclude `via_doorway_xy` neighborhoods (~0.7 m), and place deterministically from a scene-seed-derived RNG (author final transforms directly — gravity-settle fights the pipeline's stable `instance_id` + mission ordering; if settling is wanted, run it on a scratch physics scene and read back transforms *before* authoring). Surface distractors target existing `objects[]` tops via `bbox_3d_max[2]`.
3. **Author `objects[]` + labels directly** under `$ISAACLAB` (`add_labels` needs the Kit runtime): canonical lowercase label + `instance_id` (minted to not collide with Infinigen factory IDs) + non-origin `position_3d` + a contract-conformant `prim_path`. **Validate every label against `scene_classes.STRUCTURAL_CLASSES`** (a `wall`/`floor` label silently vanishes from detections + missions; a structural object given a furniture label becomes a spurious "go to the X" target). Opportunistically populate the reserved `objects[].descriptors` namespace (color / material / size) from the pool's own metadata — clean grounding upside Infinigen can't easily provide.
4. **Mandatory re-runs, batched per scene:** `extract_scene_metadata` (always), `validate_scene_connectivity` (whenever a collider intersects the robot z-slab), `generate_scenes_metadata` spawn-points (so the robot can't spawn inside a distractor).
5. **Navigability acceptance gate:** compare post-injection `connectivity[]` against the pre-injection graph; **reject/relocate any variant** that flips a previously-reachable edge to `blocked`, makes the scene `multi_room_incompatible`, or eliminates a spawn point. Reuses the existing occupancy seam + shared planner — no new planner.
6. **Provenance:** record the injection RNG seed + asset-pool manifest (which USDs, counts, pose ranges) into `scene_config.json` so each variant is reproducible + auditable.

## Acceptance

- [ ] `_run_inject_assets` stage in `prep_room_usds.generate_scenes` behind `--inject-assets`/`--asset-pool` (default OFF), ordered after postprocess and before metadata/connectivity/spawn-points.
- [ ] A curated, **manually-labeled** asset pool with canonical-noun labels reconciled against `STRUCTURAL_CLASSES` + the taxonomy; **license checked** for training-use redistribution.
- [ ] Navigability-aware placement (free-cell + footprint + doorway-exclusion), deterministic by scene seed; injected prims under `/World/InjectedAssets`.
- [ ] Injected objects appear in customData `objects[]` with `add_labels(...,"class")` + detections, and become mission targets (a previously `objects=0` scene yields ≥ N missions).
- [ ] The per-variant re-process runs atomically; the navigability gate rejects any connectivity-breaking / spawn-blocking variant; occupancy/connectivity are regenerated so the freshness guard passes without `--allow-stale-occupancy`.
- [ ] Provenance (seed + manifest) in `scene_config.json`; the brief ships to `completed/` with stamp + BOARD clear.

## Out of scope

- **Disambiguation.** Owned by the generator's referring-expression target phrasing + [`mission-text-enrichment`](mission-text-enrichment.md). This brief must not be built as a grounding crutch for ambiguous targets — it creates targets where none exist; it does not distinguish existing ones.
- **`load_auto_labeled_assets` filename-regex labeling** — its vocab doesn't match the project's canonical nouns; author labels explicitly.
- **Injecting `STRUCTURAL_CLASSES` as targets** — distractors are non-structural by construction.
- **Folding into `infinigen-scene-corpus`** (a no-new-code corpus operation) or conflating with `proc_room.py`'s training-time GPU primitive scatter (a separate concern).
- **Blanket clutter-multiplication** in v1 — scope v1 to the goal-sparse rescue; defer N-variant multiplication until the per-variant omap cost is measured acceptable.
