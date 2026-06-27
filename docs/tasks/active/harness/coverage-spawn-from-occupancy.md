# Coverage spawn from occupancy — derive the capture start pose from in-driver free space

**Type:** refactor + bugfix (capture spawn sourcing)
**Owner:** DGX agent (coverage capture driver)
**Priority:** P1 — unblocks the coverage **live bulk run** (seed2's over-occupied occupancy grid zeroes the run today) and removes the hardcoded `scenes_metadata.json` dependency. seed1 captures fine, so it is not P0, but the full corpus is blocked.
**Estimate:** S — the driver already loads occupancy + computes `free_space`; pick a spawn cell and drop the index read.
**Branch:** task/coverage-spawn-from-occupancy

## Story

As **the coverage capture driver**, I want **the robot's start pose derived from the occupancy free-space the driver already loads**, so that **capture no longer depends on `scenes_metadata.json` spawn points — which are absent from USD and, for some scenes, in a frame incompatible with the occupancy grid, which zeroes the run.**

## The problem

The driver seeds the robot reset from `scenes_metadata.json` `spawn_points_xy` (the combined Assets index) via `_resolve_active_spawn_points` (USD-first, index fallback — the USD key is empty today). Two failure modes:

1. `spawn_points_xy` is not in USD customData, so the hardcoded combined index is the only source — a fragile path dependency (operator comment on PR #112).
2. **Over-occupied occupancy grid (surfaced live in #112, pre-existing — not a rework regression):** seed2's `occupancy.npy` is degenerate, *not* mis-framed. Its raw grid is ~58% blocked (mid-row/col slabs ~88%/89% occupied — the z-slab is catching floor or clutter), so the robot-radius-inflated free space is only ~239 m² of the ~466 m² floor bbox (~51%). Only 2/98 of the index `spawn_points_xy` land on a free cell of this grid vs seed1's 93.9% — the floor-mesh sampler and the grid disagree because the grid is corrupted, not because the frames differ (a frame error would zero seed1 too). Every coverage leg's path planner then returns "no collision-free path" → **0 episodes captured**. seed1's grid is healthy (~9.7% blocked, ~99% free/floor-bbox) and captures 99/99; the failure is per-scene grid corruption, which deriving the spawn from free-space sidesteps for healthy scenes but cannot repair for seed2 (that needs its occupancy regenerated).

The driver **already** loads `occupancy` and computes `free_space` (`scene_connectivity.occupancy_to_free_space`) for the coverage plan. A start cell drawn from that same grid is in the occupancy frame by construction — no mismatch, no external dependency.

## Scope

- Derive the capture start pose from the occupancy free-space the driver already loads: pick a reachable free cell (reuse `scene_connectivity.room_representative_xy` for the first planned room, or a deterministic seeded free cell), convert grid→world with the **same origin/resolution the coverage plan uses**, and set `env_cfg.events.reset_robot.params["spawn_points_xy"]` from it.
- Remove the coverage driver's `_resolve_active_spawn_points` / `scenes_metadata.json` read (resolves PR #112 comment 3).
- Deterministic under `--seed` (reproducible spawn).

## Context bundle

- [`coverage-capture-driver`](../../completed/harness/coverage-capture-driver.md) — the driver this extends (shipped #111, reworked #112).
- seed2's over-occupied occupancy grid surfaced in #112 — a pre-existing data bug (corrupted grid content, **not** a frame error). Deriving the spawn from free-space removes the external dependency for all scenes, but seed2 specifically needs its occupancy regenerated: its inflated free space is ~51% of the floor bbox, below the health floor, so a spawn drawn from it sits on the negative of a corrupted grid.
- [`scene-provider-floor-sampler-cli`](../../parked/harness/scene-provider-floor-sampler-cli.md) — the spawn-point **generation** tooling; keep disjoint. This brief is capture-time derivation, not generation; the index/occupancy frame reconciliation at the source may belong there.
- PR #112 comment 3 — the trigger.

## Acceptance

- [ ] The coverage driver derives its start pose from the occupancy free-space it loads; no read of `scenes_metadata.json` (or any external spawn list) for the capture spawn.
- [ ] The derived spawn is in the occupancy frame (grid→world via the plan's origin/resolution) and is a reachable free cell (respects the inscribed-radius inflation already in `free_space`).
- [ ] **seed2 captures ≥ 1 episode** once its occupancy is healthy; seed1 stays green. seed2's current grid is degenerate (~51% free/floor-bbox) — a regenerate-occupancy blocker the spawn derivation must not mask (no widening the planner snap, no falling back to the external read).
- [ ] Deterministic under `--seed`.
- [ ] Tests cover the grid→world spawn derivation + the free-cell selection (assertable without Kit).

## Out of scope

- Fixing the scene generator's `spawn_points_xy` frame, or embedding spawn points into USD customData — a generation-side concern (`scene-provider-floor-sampler-cli` or a `scene-metadata-in-usd` extension).
- Multi-env spawn — v1 is single-env; parallel-env coverage is its own follow-up.
- The training env's spawn sampling — unchanged; this is capture-only.
