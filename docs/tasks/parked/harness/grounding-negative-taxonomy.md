# Expand the grounding hard-negative taxonomy to trajectory_violation

**Type:** new feature (capture data quality)
**Owner:** DGX agent
**Priority:** P2 — materially improves the grounding training corpus for
the validator + VLA/VLM consumers; the schema already anticipates this
mode (`trajectory_violation` is in the `outcome_category` /
`hard_negative_category` enums) but no tooling produces it.
**Estimate:** M (~2–3 days; the alternative-path generation + dispatch +
labeling + tests).
**Branch:** task/grounding-negative-taxonomy

**Blocked on / trigger:** [`mission-generator`](../../active/harness/mission-generator.md)
ships path-shape missions (the `path-shape` / `mixed` modes that emit
path-constraint language + `planned_path`). Without path-shape missions
there is no constraint to violate. Pick up once mission-generator's
queue carries path-shape rows.

## Story

As a **consumer training a grounding-aware validator / VLA**, I want
**the harness to emit `trajectory_violation` hard negatives — episodes
that reach the correct goal but via a route that violates the mission's
path-shape language**, so that **the corpus covers the full
language-trajectory misalignment space (wrong place *and* wrong path),
not just goal-position swaps.**

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/path-planning-architecture.md`](../../context/path-planning-architecture.md)
  — the shared planner this brief plans alternative paths with. Use it;
  do not write a planner.
- [`harness-architecture`](../../active/harness/harness-architecture.md)
  — the [Hard-negative injection](../../active/harness/harness-architecture.md#hard-negative-injection---inject-bad-grounding)
  section (the existing modes) and the `outcome_category` /
  `hard_negative_category` enums (which already include
  `trajectory_violation`).
- [`mission-generator`](../../active/harness/mission-generator.md) — the
  producer of path-shape missions this brief consumes.
- [`source/strafer_lab/strafer_lab/tools/grounding_injection.py`](../../../../source/strafer_lab/strafer_lab/tools/grounding_injection.py)
  — the existing injection tool this brief extends.

## Context

### Where the taxonomy stands

The hard-negative injection (`grounding_injection.py`) produces **goal-
position** perturbations:

- `wrong_room` — goal swapped to a different room.
- `wrong_instance` — goal swapped to a same-label sibling in the room.
- `wrong_object` — goal swapped to a **different-label** object in the
  same room (the highest-value category-confusion negative). *Landing in
  the Tier 2 bridge PR (#88); listed here for taxonomy completeness.*

All three move the *destination*. The schema's fourth mode —
`trajectory_violation` — is a different axis: **right destination, wrong
route**. The `outcome_category` enum already reserves it; no tool emits
it. This brief fills that gap.

### What a `trajectory_violation` negative is

A path-shape mission carries route language ("go to the chair **by
hugging the south wall**", "reach the kitchen **via the dining room**").
mission-generator emits these with a `planned_path` that satisfies the
constraint. A `trajectory_violation` negative keeps the **recorded
`mission_text` unchanged** (still says "hugging the south wall") but
drives a **different, constraint-violating route to the same goal** —
typically the plain shortest path that ignores the path-shape language.
The language says one route, the trajectory takes another: a route-level
grounding hard negative.

This needs **no structured-constraint schema** (`mission-generator`
deliberately retired that — VLAs consume free text). The "constraint" is
implicit in the mission text; the violation is generated mechanically by
planning an alternative route, not by parsing the constraint.

### How it's generated (reuses the shared planner)

For a path-shape mission:
1. The honest path is the mission's `planned_path` (satisfies the
   constraint).
2. The violation path is a **different goal-reaching route** — plan the
   plain shortest path with
   [`path_planner.plan_path`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/path_planner/) on the scene's
   occupancy grid (built by the shared Infinigen grid adapter, per
   [`path-planning-architecture`](../../context/path-planning-architecture.md)).
   When the shortest path *already* satisfies the constraint (no
   meaningful alternative), the mission is skipped for this mode — record
   the skip so the per-scene `trajectory_violation` yield is auditable
   (same drop-audit discipline as the existing injection modes).
3. Dispatch drives the violation path; `mission_text` is unchanged;
   `injection_mode = injection_mode_actual = "trajectory_violation"`,
   `outcome_category = "trajectory_violation"`.

The bridge driver dispatches via the autonomy stack (which plans its
own route), so `trajectory_violation` is most naturally produced by the
**scripted** driver (Tier 3), which follows a supplied `planned_path`
directly. Confirm at pickup whether bridge can be coerced into a
specified route or whether this mode is scripted-only; if scripted-only,
state that in the brief and the spec.

### Downstream filtering (unchanged contract)

Consumers key off `injection_mode_actual` exactly as for the existing
modes. `trajectory_violation` episodes are honest-goal / wrong-route, so
a consumer that wants only place-grounding negatives filters them out by
mode; a consumer training route-grounding includes them.

## Acceptance criteria

- [ ] `"trajectory_violation"` added to `INJECTION_MODES` and the
      `--inject-bad-grounding` choices, gated to path-shape missions.
- [ ] Alternative-path generation uses
      `path_planner.plan_path` via the shared Infinigen occupancy-grid
      adapter — **no new planner**. If the adapter from
      `mission-generator` / `path-planning-architecture` isn't present
      yet, this brief's pickup is still gated on it; coordinate so the
      adapter is written once.
- [ ] Skip-when-no-meaningful-alternative is recorded auditable
      (per-scene `trajectory_violation` yield reported), mirroring the
      existing silent-drop audit semantics.
- [ ] `mission_text` is preserved unchanged on violation episodes;
      `injection_mode_actual` + `outcome_category` set to
      `trajectory_violation`; original (honest) path retained for audit.
- [ ] Driver applicability (scripted-only vs bridge-capable) resolved
      and documented in both this brief and
      `harness-architecture.md`'s Hard-negative injection section.
- [ ] Unit tests: a path-shape mission with a distinct shortest-path
      alternative yields a violation; a mission whose shortest path
      already satisfies the constraint is skipped + audited; determinism
      under a seeded rng.
- [ ] `harness-architecture.md` Hard-negative injection section updated
      to document the four-mode taxonomy.
- [ ] `make test-lab-pure` green. Brief shipped to `completed/` in the
      shipping PR.

## Out of scope

- `wrong_object` — lands in the Tier 2 bridge PR (#88).
- Re-introducing a structured `path_constraints[]` schema — retired by
  `mission-generator` for good reasons; the violation is mechanical, not
  schema-driven.
- The shared Infinigen occupancy-grid adapter itself — owned jointly by
  `mission-generator` + `path-planning-architecture`; this brief is a
  consumer of it.
