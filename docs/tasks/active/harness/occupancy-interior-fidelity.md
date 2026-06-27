# Occupancy regen — seed2 + seed6 grids are systematically over-occupied

**Type:** occupancy / collider-fidelity fix (occupancy regeneration)
**Owner:** DGX agent
**Priority:** P1 — **the bulk-capture corpus depends on it.** 2 of 5 scenes
(seed2 + seed6) have over-occupied occupancy grids (~57–58% blocked), so the
coverage driver captures 0 episodes there; only seed1/5/7 are usable today.
(Originally filed P2 for a narrower symptom — a few sealed room interiors; the
corpus sweep below shows the fault is whole-scene.)
**Estimate:** M (Kit experiments to confirm the shared root cause + the fix, a
re-process, and matrix validation across all 5 corpus scenes).
**Branch:** `task/occupancy-interior-fidelity`

**Status:** ACTIVE — revised 2026-06-27 on PR #114. Root cause settled; the fix is
being moved from the agnostic occupancy generator to the Infinigen-specific bake
seam after review (see "Resolution in progress"). Surfaced live by PR #113
(coverage-spawn-from-occupancy), whose STOP gate refuses to capture over a corrupt
grid. Filed off PR #96 (scene-connectivity-validation).
**Follow-up:** [`capture-debug-overhead-cam.md`](../../parked/harness/capture-debug-overhead-cam.md) — overhead debug cam for scripted/coverage capture (invisible-collider QA).

## Resolution in progress (2026-06-27)

**Root cause — a `convexHull` collider on perimeter trim, confirmed independently
on seed2 AND seed6 (settled).** Each degenerate scene carries a
`skirtingboard_support` mesh: thin wall-base moulding (only ~1 % of its bounding
box holds geometry — a perimeter ring) given a `convexHull` collision
approximation. The convex hull of a room-perimeter ring is the *filled* rectangle,
so the collider is a phantom floor-to-~25 cm slab spanning the whole footprint. The
omap reads physics colliders, so it faithfully rasterized that real collider across
the navigable floor (Kit `--omap-debug-dump`: the ~58 % blocked mass is
`occupied(1.0)`, not `unknown(0.5)`). The lead "z-band" hypothesis was *refuted* —
the band is correctly anchored and the floor meshes clear it by 5 cm on every
scene; the discriminator is geometric, not a z-number. seed1/5 have no such prim;
seed7's skirting is named `*_ceiling`, so it was already given the faithful
exact-mesh collider — which is why PR #96's wall / furniture / flood-seed
experiments all missed it.

**Fix (the Infinigen bake seam).** The collider approximation is authored in the
source-specific bake (`postprocess_scene_usd.py`), which routes each mesh to an
exact-mesh (`none`) or a convex-hull collider by a name allow-list. Skirting named
`*_support` fell through that allow-list to the convex-hull furniture default,
while `*_ceiling` skirting matched the ceiling arm by coincidence — the same
allow-list-leak class the door-variant arm already records. The fix adds a
case-insensitive trim arm (skirting / baseboard / moulding / cornice / casing /
trim) to the structural allow-list so perimeter trim gets the exact mesh at bake
time; the case-insensitivity is scoped to the trim arm so the case-sensitive
wall/door arms do not widen. A CPU mis-hull audit across the corpus confirmed the
trim arm covers every phantom-slab-risk prim (the only other large convex-hull prim
is a rug at ~17 % vertex-coverage, correctly left convex and below the occupancy
band). The agnostic occupancy generator carries **no** Infinigen-specific collider
regeneration; seed2 + seed6 are re-baked and their occupancy regenerated from the
corrected USDs. (PR #114's first pass corrected the collider inside the agnostic
occupancy generator — the wrong layer — and that code is removed in favour of this
bake-seam fix.)

---

The diagnosis-first record below is preserved as filed.

## Scope correction (2026-06-27) — whole-scene, not a few sealed rooms

| scene | blocked | mid-row | mid-col | health |
|---|---|---|---|---|
| seed1 | 9.7% | 24.9% | 2.2% | OK |
| **seed2** | **58.1%** | **87.7%** | **88.5%** | DEGENERATE |
| seed5 | 6.5% | 11.2% | 6.7% | OK |
| **seed6** | **56.9%** | **82.2%** | **78.8%** | DEGENERATE |
| seed7 | 9.5% | 43.5% | 3.0% | OK |

The degenerate scenes are ~57–58% blocked with contiguous ~80–88% mid-row/col
slabs — a whole-scene rasterization fault, not a couple of collider-healed
doorways. The leading hypothesis is the omap z-slab (seed2's floor sits ~4.5 cm
lower → a lower occupancy z-band that may catch floor or clutter), but it is
**unconfirmed**: `z_slice_m` does NOT cleanly separate the degenerate scenes from
the healthy ones (seed6 sits among the healthy on that axis), so the shared cause
must be **diagnosed across BOTH seed2 and seed6**, not assumed. The PR #96
sealed-room investigation below is valid prior art, but its "2 sealed rooms"
scope understates the seed2/seed6 case. Health bar: regenerate the degenerate
grids to inflated-free/floor-bbox ≥ ~70% (target ~99%, like seed1), with **no
regression** on the healthy three.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/path-planning-architecture.md`](../../context/path-planning-architecture.md) — the cached-occupancy seam this affects.
- [`docs/SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) §b-conn "Known limitation".

## The problem (PR #96 — root cause still OPEN)

`validate_scene_connectivity.py` generates occupancy from the USD's physics
colliders via the occupancy-map extension (`isaacsim.asset.gen.omap`) and
verifies room-to-room reachability with the shared grid planner. It correctly
opens doors (drops every door collider + hides the leaf). The door **matcher
is complete** — verified on seed1/seed2 it catches every door prim incl. the
`GlassPanelDoor` variant; this is **not** a door-matching gap.

The symptom: a few rooms' interiors read **occupied** in the omap even with the
door open, so they're unreachable in the graph. seed1 leaves 1 room sealed
(closet); seed2 leaves 2 (bathroom + closet, both at 0 % free, omap buffer value
1.0 across the footprint). By eye these rooms look navigable (the bathroom has a
full-height door + room for the chassis), so the omap is most likely
**falsely** marking them occupied — the connectivity graph then faithfully
reports that (wrong) occupancy.

**Two hypotheses were tested on PR #96 and BOTH ruled out:**

1. **Wall collider heals the cutout** — re-approximating the wall/exterior
   colliders from `meshSimplification` to `none` (exact triangle mesh) left the
   rooms still 0 % free / 21-of-36 reachable. So it is **not** the wall collider
   fidelity.
2. **Flood seeded from the exterior corner** — moving the omap `start_location`
   to the largest interior room's centroid gave the **identical** result. So it
   is **not** the flood-seed location either.
3. **Furniture convexHull over-fill** — re-approximating all furniture to
   `convexDecomposition` (faithful) dropped overall occupied 0.58→0.27 (so
   convexHull *was* over-approximating furniture elsewhere) but the bathroom +
   closet stayed **0 % free**, and overall connectivity *regressed* to 11/36
   (living_room + bedroom newly isolated — the lower occupancy shifted which
   free component each room's representative point lands in). So it is **not**
   the furniture collider approximation, and `convexDecomposition` is **not** a
   safe swap for the omap path.

So the cause is genuinely open — every collider/seed lever tried leaves the two
rooms at 0 % free, which is why they look navigable yet read fully occupied. The
remaining suspect is the omap's own per-cell classification for small enclosed
rooms; a non-omap navigable mask (PhysX overlap query per cell, or the
footprint-rasterize path) is the most promising direction. Remaining suspects to chase: the omap's
per-cell occupied/free classification heuristic for enclosed small rooms;
furniture `convexHull` colliders bridging the doorway threshold; or the omap's
ray/slice geometry. (Note: the doorway *collider intrusion* an operator can see
in the viewport — walls poking into doorways — is a **separate**, real
runtime-collision issue; the `none`-vs-`meshSimplification` experiment above
shows it is **not** what seals these rooms.)

A naive fix — **carving the door footprint free in the occupancy grid** — was
tried on PR #96 and **rejected**: it opens only a thin strip at the doorway
while the omap-occupied interior remains, so `room_representative_xy` (nearest
free cell to the room centroid) jumps to the stranded carve-pocket, which is
disconnected from the corridor → connectivity *regressed* (seed2 21→2). Carving
can't fix an occupied interior.

## Candidate fixes (pick via a Kit experiment)

Both "wall collider" and "flood seed" are already ruled out (see above), so the
remaining candidates target the omap's classification itself:

- **Pure-collision occupancy.** Get occupancy that means "a collider is present
  here", not "reachable from the omap seed". Check whether the omap has a
  no-flood / overlap mode, or whether the raw buffer's occupied-vs-unknown can
  be re-derived so an *open but unreached* interior is free, not occupied.
- **Reclaim room interiors.** Combine the omap (accurate for wall/furniture
  colliders) with the room footprints: a cell inside a room footprint with no
  actual collider is free, regardless of the omap's flood verdict.
- **Replace the omap for the interior mask.** If the omap's enclosed-region
  classification can't be tamed, build the navigable mask another way (the
  `--rasterize-fallback` footprint+object-AABB path, or a direct PhysX overlap
  query per cell) and keep the omap only where it's accurate.
- **Carve + robust representative point.** Keep the door-footprint carve but
  make `room_representative_xy` choose a point in a sufficiently large free
  component (so a room with no real navigable interior is *correctly* isolated,
  not falsely connected through a carve-pocket).

(Separately — **not** this brief — the doorway *collider intrusion* operators
see in the viewport is a runtime-collision fidelity issue owned by
`postprocess_scene_usd.py`'s structural approximation: `meshSimplification`
colliders don't tightly follow the door cutout. Switching structural prims to
`none` (exact triangle mesh) or `convexDecomposition` tightens them; that is its
own change with a runtime-perf tradeoff, and the `--structural-approximation`
CLI flag already exists to do it.)

## Acceptance (sketch)

- [ ] seed1/seed2 re-processed: the rooms the robot can physically enter are
      reachable; genuinely-too-narrow ones (a real <0.4 m closet door) stay
      excluded. Report the before/after matrices.
- [ ] No regression on the rooms already reachable today.
- [ ] The chosen mechanism documented in `path-planning-architecture.md` +
      `SCENE_PROVIDER_CONTRACT.md` (replace the §b-conn "Known limitation" note).

## Out of scope

- Door matching (complete) and the door visual/physics open (shipped on #96).
