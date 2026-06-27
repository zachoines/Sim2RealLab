# Coverage spawn from occupancy — derive the capture start pose from in-driver free space

**Type:** refactor + bugfix (capture spawn sourcing)
**Owner:** DGX agent (coverage capture driver)
**Priority:** P1 — unblocks the coverage **live bulk run** (the seed2 occupancy/spawn frame mismatch zeroes the run today) and removes the hardcoded `scenes_metadata.json` dependency. seed1 captures fine, so it is not P0, but the full corpus is blocked.
**Estimate:** S — the driver already loads occupancy + computes `free_space`; pick a spawn cell and drop the index read.
**Branch:** task/coverage-spawn-from-occupancy

## Story

As **the coverage capture driver**, I want **the robot's start pose derived from the occupancy free-space the driver already loads**, so that **capture no longer depends on `scenes_metadata.json` spawn points — which are absent from USD and, for some scenes, in a frame incompatible with the occupancy grid, which zeroes the run.**

## The problem

The driver seeds the robot reset from `scenes_metadata.json` `spawn_points_xy` (the combined Assets index) via `_resolve_active_spawn_points` (USD-first, index fallback — the USD key is empty today). Two failure modes:

1. `spawn_points_xy` is not in USD customData, so the hardcoded combined index is the only source — a fragile path dependency (operator comment on PR #112).
2. **Frame mismatch (surfaced live in #112, pre-existing — not a rework regression):** for seed2, the index `spawn_points_xy` and `occupancy.npy` are in incompatible frames — spawn0 lands ~5 m from any free cell, so every coverage leg's path planner returns "no collision-free path" → **0 episodes captured**. seed1 aligns (99/99) and captures normally; the failure is per-scene.

The driver **already** loads `occupancy` and computes `free_space` (`scene_connectivity.occupancy_to_free_space`) for the coverage plan. A start cell drawn from that same grid is in the occupancy frame by construction — no mismatch, no external dependency.

## Scope

- Derive the capture start pose from the occupancy free-space the driver already loads: pick a reachable free cell (reuse `scene_connectivity.room_representative_xy` for the first planned room, or a deterministic seeded free cell), convert grid→world with the **same origin/resolution the coverage plan uses**, and set `env_cfg.events.reset_robot.params["spawn_points_xy"]` from it.
- Remove the coverage driver's `_resolve_active_spawn_points` / `scenes_metadata.json` read (resolves PR #112 comment 3).
- Deterministic under `--seed` (reproducible spawn).

## Context bundle

- [`coverage-capture-driver`](../../completed/harness/coverage-capture-driver.md) — the driver this extends (shipped #111, reworked #112).
- The seed2 occupancy/spawn frame mismatch surfaced in #112 — pre-existing data bug; this is the capture-time fix that sidesteps it without per-scene frame reconciliation.
- [`scene-provider-floor-sampler-cli`](../../parked/harness/scene-provider-floor-sampler-cli.md) — the spawn-point **generation** tooling; keep disjoint. This brief is capture-time derivation, not generation; the index/occupancy frame reconciliation at the source may belong there.
- PR #112 comment 3 — the trigger.

## Acceptance

- [ ] The coverage driver derives its start pose from the occupancy free-space it loads; no read of `scenes_metadata.json` (or any external spawn list) for the capture spawn.
- [ ] The derived spawn is in the occupancy frame (grid→world via the plan's origin/resolution) and is a reachable free cell (respects the inscribed-radius inflation already in `free_space`).
- [ ] **seed2 captures ≥ 1 episode** (the run that yields 0 today); seed1 stays green.
- [ ] Deterministic under `--seed`.
- [ ] Tests cover the grid→world spawn derivation + the free-cell selection (assertable without Kit).

## Out of scope

- Fixing the scene generator's `spawn_points_xy` frame, or embedding spawn points into USD customData — a generation-side concern (`scene-provider-floor-sampler-cli` or a `scene-metadata-in-usd` extension).
- Multi-env spawn — v1 is single-env; parallel-env coverage is its own follow-up.
- The training env's spawn sampling — unchanged; this is capture-only.
