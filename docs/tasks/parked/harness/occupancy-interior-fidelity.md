# Occupancy fidelity for rooms sealed by collider-healed doorways

**Type:** occupancy / collider-fidelity fix
**Owner:** DGX agent
**Priority:** P2 — the connectivity graph faithfully reports *sim* reachability
today, but a room the robot could physically enter in reality is marked
unreachable in sim, so missions to it (e.g. a bathroom) are dropped.
**Estimate:** M (~1–2 days — needs Kit experiments to pick the fix + a
re-process + matrix validation on seed1/seed2).
**Branch:** `task/occupancy-interior-fidelity`

**Blocked on / trigger:** Filed off PR #96 (scene-connectivity-validation).
Un-park when the dropped rooms matter for a mission set — i.e. the
mission-generator wants to target a room the occupancy currently seals
(seed1 closet; seed2 bathroom + closet).

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

So the cause is genuinely open. Remaining suspects to chase: the omap's
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
