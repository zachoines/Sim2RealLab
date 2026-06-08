# Parameterize generate_scenes_metadata.py's floor + room-struct detection

**Type:** small refactor (un-baking the last Infinigen-coupled scene-gen seam)
**Owner:** DGX agent
**Priority:** P3 — does not block any acceptance bar; a second source can hand-author its `scenes_metadata.json` entry today (per the contract). This brief upgrades that fallback to "auto-sampler works for non-Infinigen floor names too."
**Estimate:** S (~half day — mirrors the `--ceiling-light-prim-pattern` work already shipped in `scene-provider-contract`; two CLI flags + a compile helper + one pxr-free override test + a contract-doc note flip).
**Branch:** `task/scene-provider-floor-sampler-cli`

**Blocked on / trigger:** Filed-on-trigger. Un-park when **the first non-Infinigen scene source** whose floor / structural prims are **not** named `<room>_<i>_<j>_floor` wants to use `generate_scenes_metadata.py`'s auto floor-sampler instead of hand-authoring its `spawn_points_xy` + `floor_top_z` manifest entry. Until that source exists, the hand-author path documented in [`SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) §c/§h.1 is sufficient and this brief stays parked.

## Story

As a **harness operator bringing in a non-Infinigen USD source whose floor prims aren't named like Infinigen's** I want **`generate_scenes_metadata.py`'s floor-detection and obstacle heuristics to take CLI pattern overrides — the same way `postprocess_scene_usd.py` already takes `--floor-prim-pattern` / `--structural-prim-pattern` / `--ceiling-light-prim-pattern`** so that **the auto floor-sampler produces spawn points for my source without me hand-writing the manifest entry, and the scene-gen pipeline has no remaining Infinigen-name-coupled knob that lacks an escape hatch.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`docs/SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) — the contract this completes. §c documents this exact seam as the one remaining non-overridable Infinigen coupling; §f is the postprocess CLI surface this brief mirrors.
- [`completed/scene-provider-contract.md`](../../completed/scene-provider-contract.md) — the parent brief. Its `--ceiling-light-prim-pattern` parameterization (in `postprocess_scene_usd.py`) is the exact pattern to copy: default constant + `_compile_*` helper + CLI flag + one pxr-free test.

## Motivation

`scene-provider-contract` (shipped 2026-05-30, PR #66) made `postprocess_scene_usd.py` fully source-agnostic by parameterizing its three prim-name regexes behind CLI flags. It deliberately left one seam documented-but-not-fixed: `generate_scenes_metadata.py`'s floor-sampler has two hardcoded Infinigen-coupled regexes with no CLI override:

- [`generate_scenes_metadata.py:60`](../../../../source/strafer_lab/scripts/generate_scenes_metadata.py#L60) — `_FLOOR_NAME_RE = re.compile(r"^[a-z]+(?:_[a-z]+)*(?:_\d+)+_floor$")`. Used by `_find_floor_bboxes` to decide which prims are floors to sample spawn points on.
- [`generate_scenes_metadata.py:61`](../../../../source/strafer_lab/scripts/generate_scenes_metadata.py#L61) — `_ROOM_STRUCT_RE = re.compile(r"^[a-z]+(?:_[a-z]+)*(?:_\d+)+_(floor|ceiling|wall|exterior|staircase)$")`. Used by `_find_obstacle_bboxes` to **exclude** room structure from being treated as spawn obstacles (everything above `--obstacle-min-height` that is *not* room structure is an obstacle).

A source whose floor prims are named differently (e.g. `/World/Floor_0`, not `living_room_0_0_floor`) gets zero floor matches → zero auto-sampled spawn points → must hand-author the manifest entry. The numeric knobs (`--wall-margin`, `--obstacle-min-height`, `--robot-radius`) are already CLI-overridable; only the two name regexes aren't.

The contract doc ([`SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) §c) already documents the hand-author fallback, so a second source is **not blocked** — this brief just removes the fallback's friction.

## Acceptance

- [ ] **`--floor-prim-pattern` CLI flag** on `generate_scenes_metadata.py`, default = current `_FLOOR_NAME_RE` source. Mirrors the `postprocess_scene_usd.py --floor-prim-pattern` shape (note: the two scripts' defaults differ — postprocess matches the full prim *path* `^/World/..._floor$`, this one matches the prim *name*; preserve each script's existing convention and document the difference, don't unify them in this brief).
- [ ] **`--room-struct-pattern` CLI flag**, default = current `_ROOM_STRUCT_RE` source. Drives the obstacle-exclusion set in `_find_obstacle_bboxes`.
- [ ] **`_compile_floor_pattern` / `_compile_room_struct_pattern` helpers** (or one shared `_compile_pattern`) extracted so the regex compile + the matching predicate are unit-testable **without `pxr`** — same pxr-free-testability move `scene-provider-contract` used for the ceiling-light flag.
- [ ] **One override test** under `tests/harness/` (extend `test_scene_provider_contract.py` or a sibling) asserting that a synthetic prim-name set matching a custom `--floor-prim-pattern` is detected as floor, and a default-Infinigen name is not detected under the custom pattern. Pure-Python, no Isaac Sim. `make test-lab-pure` stays green.
- [ ] **Contract-doc note flip.** [`SCENE_PROVIDER_CONTRACT.md`](../../../SCENE_PROVIDER_CONTRACT.md) §c currently says the floor-detection regex + obstacle heuristics are "**not** CLI-overridable today (unlike the postprocess patterns in §f)." Flip that to document the new flags + add `generate_scenes_metadata.py`'s flags to the §f-style surface (or a sibling table). The §g adapter checklist step 4 ("hand-write the `scenes_metadata.json` entry … only the floor-detection step is Infinigen-coupled") updates to "pass `--floor-prim-pattern` / `--room-struct-pattern` matching your source."
- [ ] **Worked-example update.** §h.1's step 4 fallback ("If this USD's floor prims are NOT named … hand-write the entry instead") gains the auto-sampler-with-overrides alternative.
- [ ] Brief shipped to `completed/` per [`conventions.md`](../../context/conventions.md) inside the shipping PR (stamp with the work commit + PR link).

## Out of scope

- **Unifying the path-vs-name matching convention** between `postprocess_scene_usd.py` (matches full prim path) and `generate_scenes_metadata.py` (matches prim name). Preserve each script's existing behavior; document the asymmetry. A unification is a separate, riskier refactor.
- **Parameterizing the numeric spawn heuristics** (`--wall-margin`, `--obstacle-min-height`, `--robot-radius`) — already CLI-overridable; nothing to do.
- **A `SceneSourceAdapter` ABC / plugin system.** Same as the parent brief: the contract is artifact-based, not interface-based. This is two more CLI flags, not an abstraction layer.
- **`infinigen_label_parser.py` or `extract_scene_metadata.py` generalization.** Correctly Infinigen-specific; a second source writes its own metadata extractor (per the contract). This brief only touches the floor-sampler in `generate_scenes_metadata.py`.
- **Generating or ingesting an actual second source.** This brief makes the auto-sampler ready; producing a real non-Infinigen scene is the consuming work that fires the trigger.

## Triggered by

PR #66 (`scene-provider-contract`) review, 2026-05-30. The agent surfaced this seam in its report rather than filing during the PR (per the prompt's out-of-scope rule); the orchestrator filed it parked so it isn't lost. It is the direct sibling of the `--ceiling-light-prim-pattern` parameterization that shipped in #66 — same shape, same test pattern, deferred until a second source needs auto floor-sampling.
