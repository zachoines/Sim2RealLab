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

## The problem (root-caused on PR #96)

`validate_scene_connectivity.py` generates occupancy from the USD's physics
colliders via the occupancy-map extension (`isaacsim.asset.gen.omap`) and
verifies room-to-room reachability with the shared grid planner. It correctly
opens doors (drops every door collider + hides the leaf). The door **matcher
is complete** — verified on seed1/seed2 it catches every door prim incl. the
`GlassPanelDoor` variant; this is **not** a door-matching gap.

The gap is in the occupancy itself. Two interacting facts:

1. **The wall collider heals the doorway shut.** `postprocess_scene_usd.py`
   gives structural prims (walls) a `meshSimplification` collider "to preserve
   door/window cutouts" — but for some rooms the simplified collider closes the
   door-sized cutout. So even with the door collider dropped, there is no
   robot-passable free corridor through the wall at the doorway.
2. **The omap flood marks the sealed interior as *occupied*.** The occupancy-map
   buffer classifies cells as occupied / free / unknown; a region its flood
   can't reach (an interior room sealed by (1)) is marked **occupied** (1.0),
   not unknown — verified on seed2 (bathroom + closet interiors = 0 % free,
   buffer value 1.0 across the footprint).

Net: the whole interior of such a room reads occupied, so it is unreachable in
the graph. seed1 leaves 1 room sealed (closet); seed2 leaves 2 (bathroom +
closet). The connectivity graph is *correct w.r.t. the current sim colliders*
— but the sim colliders don't match reality (the rooms are enterable).

A naive fix — **carving the door footprint free in the occupancy grid** — was
tried on PR #96 and **rejected**: it opens only a thin strip at the doorway
while the omap-occupied interior remains, so `room_representative_xy` (nearest
free cell to the room centroid) jumps to the stranded carve-pocket, which is
disconnected from the corridor → connectivity *regressed* (seed2 21→2). Carving
can't fix an occupied interior; do not re-introduce it without (2) also solved.

## Candidate fixes (pick via a Kit experiment)

- **Pure-collision occupancy.** Get occupancy that means "a collider is present
  here", not "reachable from the omap seed". Check whether the omap has a
  no-flood / overlap mode, or whether the raw buffer's occupied-vs-unknown can
  be re-derived so an *open but unreached* interior is free, not occupied.
- **Reclaim room interiors.** Combine the omap (accurate for wall/furniture
  colliders) with the room footprints: a cell inside a room footprint with no
  actual collider is free, regardless of the omap's flood verdict.
- **Better wall collider.** Re-collider walls/doors with an approximation that
  preserves the door cutout (`convexDecomposition`, or `none`/full mesh for the
  door-frame jamb) so the omap sees the real opening. Owned jointly with
  `postprocess_scene_usd.py`'s structural-approximation choice.
- **Carve + robust representative point.** Keep the door-footprint carve but
  make `room_representative_xy` choose a point in a sufficiently large free
  component (so a room with no real navigable interior is *correctly* isolated,
  not falsely connected through a carve-pocket).

## Acceptance (sketch)

- [ ] seed1/seed2 re-processed: the rooms the robot can physically enter are
      reachable; genuinely-too-narrow ones (a real <0.4 m closet door) stay
      excluded. Report the before/after matrices.
- [ ] No regression on the rooms already reachable today.
- [ ] The chosen mechanism documented in `path-planning-architecture.md` +
      `SCENE_PROVIDER_CONTRACT.md` (replace the §b-conn "Known limitation" note).

## Out of scope

- Door matching (complete) and the door visual/physics open (shipped on #96).
