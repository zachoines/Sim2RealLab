# Nav2 sim→real promotion architecture: split velocity-coupled from behavioral overrides

**Type:** task / refactor + process
**Owner:** Jetson agent (`source/strafer_ros/strafer_navigation/`)
**Priority:** P2
**Estimate:** M (~1 day refactor + per-knob real-robot validation laps that
stretch across multiple sessions)
**Branch:** task/nav2-sim-real-promotion-architecture

**Status:** Superseded 2026-07-05 (Jetson) by
[`completed/nav2-envelope-retirement.md`](nav2-envelope-retirement.md).
Layers A + B shipped in
[#137](https://github.com/zachoines/Sim2RealLab/pull/137) — the
`_patch_params` three-section split, the promotion context module, and
the `TestPromotionSplitInvariants` + byte-identical `TestPatchByteIdentical`
pins. Layer C (the four real-robot promotion laps) is **superseded by an
operator policy decision (2026-07-05)**: the `envelope_factor` gate was a
workaround for a misdiagnosed problem (Jetson CPU starvation, since fixed),
so the whole gate was retired and the validated sim Nav2 config promoted to
the universal baseline. There is no per-knob gate left to lap; the four
knobs now ship as YAML defaults on both lanes. The new model —
config-parity-by-construction plus a sim-first → temporary-flag → A/B →
universalize-and-delete lifecycle — lives in
[`context/nav2-config-parity.md`](../context/nav2-config-parity.md).

**PR:** bundled into the retirement PR from `task/nav2-envelope-retirement`.

## Story

As a **roboticist who develops behaviors in sim and ships them to
real**, I want **Nav2 overrides clearly split into two buckets —
"genuinely velocity-coupled" (stays gated on the lifted sim velocity
envelope) and "behavioral defaults" (applies on every lane) — with a
documented promotion process and per-knob validation laps**, so that
**the sim-to-real gap stops growing by accident** and **every
currently-sim-only knob has a clear path to either graduating or
staying gated with a justification**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [completed/mppi-critic-tuning-for-sim-envelope.md](mppi-critic-tuning-for-sim-envelope.md)
  — the originating brief for the `envelope_factor > 1.0` gate
  pattern. Filed option A specifically *because* the MPPI rebalance
  is velocity-coupled.
- [completed/nav2-startup-unknown-donut-path-noise.md](nav2-startup-unknown-donut-path-noise.md)
  — the predecessor that pattern-matched onto the gate for a
  behavioral change (SmoothPath BT), not a velocity-coupled one.
- [`completed/nav2-commit-and-follow-path-stability.md`](nav2-commit-and-follow-path-stability.md)
  — the parent brief (shipped in PR #50) that already ungated
  `allow_unknown` and the smoothing BT to the universal default. This
  brief picks up where that one left off: formalize the split and
  migrate the remaining gated knobs.

## Context

Operator observation (2026-05-22, during PR #50 review): the
`envelope_factor > 1.0` gate has become a catch-all for "sim-only
changes" — applied by pattern matching on precedent rather than by
the original justification (changes that depend on the lifted
velocity envelope). The result is a quiet sim-to-real gap: most of
the Nav2 tuning shipped to date is invisible to real-robot bringup,
and real-robot starts with largely default Nav2 behavior. PR #50
addressed the most recent two offenders (`allow_unknown`, smoothing
BT) by promoting them to YAML defaults, but the historical knobs
remain gated and the underlying architectural ambiguity is still
there.

Current `_patch_params` structure (after PR #50 lands):

| Knob | Gate | Justified gate? |
|---|---|---|
| Constants injection (velocity caps, costmap resolution, footprint, scan ranges) | Always applied | n/a — physical constants from `strafer_shared` |
| `vx_std`, `vy_std`, `wz_std`, `prune_distance` ×= `envelope_factor` | Always applied | Yes — sampling window must track exploration envelope |
| `vy_std` un-scaled back to baseline | `envelope_factor > 1.0` | Yes — lifted envelope is for forward+rotation; lateral stays baseline |
| `PathAlignCritic.cost_weight` 8.0 → 9.0 | `envelope_factor > 1.0` | **Possibly** — tuned for high-vel integration noise on straights; could matter less but still rebalance at low-vel |
| `PreferForwardCritic.cost_weight` 6.0 → 10.0 | `envelope_factor > 1.0` | **Probably not** — biases against strafe/spin; useful at any velocity, just more critical when exploration is wider |
| `PathFollowCritic.offset_from_furthest` 5 → 20 | `envelope_factor > 1.0` | **Possibly** — high-speed rollouts win when look-ahead is far ahead; less load-bearing at low-vel |
| `gamma` 0.015 → 0.008 | `envelope_factor > 1.0` | **Possibly** — smoothness/lag trade-off; lower lets command track the high-vx optimum |
| Smoothing BT swap | Was gated; now universal (PR #50) | n/a — promoted |
| `allow_unknown` | Was gated; now YAML default | n/a — promoted |

Three knobs in the "Probably / Possibly" rows are candidates for
graduation. Three more (`vx_std`/`vy_std`/`wz_std` scaling,
`vy_std` un-scale) are genuinely velocity-coupled and stay gated by
construction.

## Approach

Three layers, all in this brief:

### A. Refactor `_patch_params` to surface the split

Split the function body into three labeled sections:

```python
def _patch_params(...):
    # ── Constants injection (always applied) ────────────────────────
    # velocities, costmap resolution, footprint, scan ranges
    ...

    # ── Velocity-envelope coupling (envelope_factor) ────────────────
    # MPPI sampling stds, prune_distance scaling, vy_std un-scale
    # — these MUST track the velocity cap; gated by construction
    ...

    # ── Behavioral overrides (under per-knob promotion) ─────────────
    # Anything else that's currently sim-only sits here, each with
    # an inline comment naming the graduation criterion and the
    # validation lap that needs to land before it moves out.
    if envelope_factor > 1.0:
        # PathAlignCritic 8.0 → 9.0 — promotion criterion: real-robot
        # cornering smoke (translate forward 1 m → rotate 90° →
        # translate forward 1 m) lands within tolerance with
        # PathAlign=9.0 instead of 8.0.
        ...
```

Net effect: the file documents *why* each section is structured the
way it is. A future tuner adding a new knob can pick the right
section without re-reading the predecessor briefs.

### B. Document the promotion process in a context module

New context module: `docs/tasks/context/nav2-knob-promotion.md`. Lays
out the per-knob graduation contract:

1. **Identify**: is the knob velocity-coupled (stays gated) or
   behavioral (graduates)?
2. **Sim observable**: what does the current sim behavior look like?
   Captured as a baseline snapshot (Foxglove screencap, `/plan`
   sample, mission log).
3. **Real-robot validation lap**: which mission, what observable,
   what would constitute a regression. Concrete + falsifiable.
4. **Promote OR document**: if the lap passes, move the knob to the
   universal YAML default (or remove the gate in `_patch_params`).
   If it fails, document the failure mode and either keep the gate
   with a refreshed justification or file a follow-up that addresses
   the regression.
5. **Cross-link**: BOARD.md row pointing at the snapshot, the lap,
   and the disposition.

### C. Run the validation laps for the existing gated knobs

Per-knob validation, ordered by smallest-blast-radius first:

1. **`PreferForwardCritic.cost_weight` 6.0 → 10.0**: probably the most
   portable. Run the cornering smoke + a `translate forward 1 m`
   on real with the override applied. If MPPI doesn't over-bias
   forward at the indoor velocity cap, promote.
2. **`PathFollowCritic.offset_from_furthest` 5 → 20**: look-ahead
   distance at MAP_RESOLUTION=0.05 is ~1 m. At the real-robot vel
   cap (~0.78 m/s), this is ~1.3 s of preview — still reasonable.
   Validate on cornering + a long-corridor mission.
3. **`gamma` 0.015 → 0.008**: smoothness/lag. Lower values
   could overshoot at indoor speeds where chassis inertia
   dominates command-following error. Validate on the same
   cornering smoke and a `rotate 90°` smoke.
4. **`PathAlignCritic.cost_weight` 8.0 → 9.0**: most coupled to
   velocity; weakest portability case. Validate by checking
   whether MPPI tolerates small path-deviation noise without
   spending lateral effort tracking it at the indoor cap. If
   it does, promote; if it doesn't, document and keep gated
   with the refreshed justification.

Each validation lap captures: starting baseline (current behavior),
override applied, observed delta, disposition (promote / keep
gated / refile). Lands as updates to this brief in the same PR as
the promotion (or as a follow-up brief if it surfaces a regression).

## Acceptance criteria

- [x] `_patch_params` is restructured into the three labeled
      sections (universal defaults / velocity-envelope coupling /
      behavioral overrides under promotion). Each currently-gated knob
      carries an inline comment naming its promotion criterion.
      *(Layer A — landed.)*
- [x] `docs/tasks/context/nav2-knob-promotion.md` exists and is linked
      from `docs/tasks/context/README.md`'s `## Current modules` index.
      The module describes the five-step contract above and gives one
      worked example (`PreferForwardCritic`). *(Layer B — landed.)*
- [ ] At least one of the four MPPI knobs is run through the
      validation lap and either promoted to the universal default
      OR documented as "stays gated, refreshed justification: ___".
      The PR description carries the validation observations
      (mission, observed metric, disposition).
      *STAGED (Layer C): ready-to-run lap cards are below; no
      robot/operator access this session, so this opens when the
      operator schedules robot time.*
- [ ] The brief at
      [`completed/nav2-commit-and-follow-path-stability.md`](nav2-commit-and-follow-path-stability.md)
      gets its outstanding "real-robot validation lap" bullet
      checked off — i.e., a real-robot run of the smoothing BT +
      `allow_unknown: true` (SmacPlanner2D) universal defaults happens
      and is recorded.
      *STAGED (Layer C): Lap 0 card below; operator-gated.*
- [x] Unit tests pin the universal vs. gated split. Landed as
      `test_nav_config.py::TestPromotionSplitInvariants` — universal
      knobs resolve factor-independently, behavioral overrides apply
      strictly when `envelope_factor > 1.0` (and stay at baseline at
      `1.0` / sub-unity), and the velocity-envelope knobs track the
      factor — plus `TestPatchByteIdentical`, a byte-identical golden
      pin proving the refactor emits identical params to the
      pre-refactor code at `envelope_factor == 1.0` and `2.0`.
      Supersedes the `TestConstantsInjection`-only pattern.
- [x] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
      *Swept: no user-facing surface references the refactored knobs
      (no CLI / topic / env / public-API change); `context/README.md`
      gains the new module's index entry.*

## Validation lap cards (Layer C — staged, operator-gated)

Ready-to-run cards, ordered smallest-blast-radius first (Approach C).
Each is one real-robot lap; run at the indoor cap (real-robot bringup,
`STRAFER_NAV_VEL_SCALE` unset → `envelope_factor = 1.0`, ~0.78 m/s).
Baseline snapshots come from the sim run of the same mission (Foxglove
overlay + `/cmd_vel` / `/plan` samples), reusing the bisection-snapshot
harness from
[`completed/mppi-critic-tuning-for-sim-envelope.md`](mppi-critic-tuning-for-sim-envelope.md)'s
Investigation pointers. Record the disposition inline on the card and
mirror it to `BOARD.md`. **None run this session — no robot/operator
access; all dispositions PENDING.**

To promote a knob after a passing lap: move its value into
`nav2_params.yaml`, delete its line from `_patch_params`'s behavioral
section, and move its entry from `_BEHAVIORAL_OVERRIDES` to
`_UNIVERSAL_KNOBS` in `test_nav_config.py`. Then regenerate the golden
fixtures (they intentionally change) and note that in the PR.

### Lap 0 — universal-default inheritance (closes acceptance #4)

- **Under test:** the already-universal smoothing / event-driven-replan
  BT + `allow_unknown: true` (SmacPlanner2D, `cost_travel_multiplier:
  2.0`) — verify the real robot inherits them without regression. Not a
  promotion; this closes the parent brief's open real-robot bullet.
- **Mission:** `translate forward 3 m` after some warmup driving on a
  partially-mapped room, then the cornering smoke
  (`forward 1 m → rotate 90° → forward 1 m`).
- **Baseline:** the parent brief's sim `/plan` overlay (plan commits,
  prefers known-free, replans only on invalidation/goal-change).
- **Observable:** `/plan` publishes once at start and again only on
  goal-update or path invalidation (not per-0.5 m); plan stays inside
  known-free cells; no unknown-band cut-through; far goals still plan.
- **Regression (falsifiable):** `/plan` re-publishes on a fixed cadence
  or visibly jitters between invalidation events; OR a far goal fails
  planning outright on a small unknown patch (the NavFn brittleness
  the SmacPlanner swap fixed); OR real-lidar blips flap the replan.
- **Disposition:** PENDING.

### Lap 1 — `PreferForwardCritic.cost_weight` 6.0 → 10.0 (behavioral)

- **Mission:** `translate forward 1 m` + the cornering smoke.
- **Baseline:** sim trace — vx-dominant `/cmd_vel`, `/optimal_trajectory`
  tracks forward along `/plan`.
- **Observable:** vx-dominant `/cmd_vel`, no reverse-along-path, arrival
  within `xy_goal_tolerance = 0.15`.
- **Regression:** MPPI over-biases forward and overshoots the goal; or
  (if the weight is too strong for the lower real cap) the chassis drives
  tangent to / backward along the path.
- **Disposition:** PENDING. Promote if it holds — most portable of the four.

### Lap 2 — `PathFollowCritic.offset_from_furthest` 5 → 20

- **Mission:** cornering smoke + a long-corridor `translate forward 3 m`.
- **Observable:** look-ahead ~1 m at `MAP_RESOLUTION = 0.05` (~1.3 s
  preview at 0.78 m/s); the path is tracked without corner-cutting.
- **Regression:** at the lower real cap the far look-ahead makes the
  robot cut corners or overshoot before the goal.
- **Disposition:** PENDING.

### Lap 3 — `gamma` 0.015 → 0.008

- **Mission:** cornering smoke + a `rotate 90°` smoke.
- **Observable:** commanded mean tracks the optimum without overshoot or
  oscillation.
- **Regression:** overshoot / oscillation at indoor speeds where chassis
  inertia dominates command-following error — the lower `gamma` removes
  the smoothing that was masking it.
- **Disposition:** PENDING.

### Lap 4 — `PathAlignCritic.cost_weight` 8.0 → 9.0 (most velocity-coupled)

- **Mission:** cornering smoke on a curved plan with small path-deviation
  noise.
- **Observable:** MPPI tolerates the small deviation without spending
  lateral `vy` to chase it at the indoor cap.
- **Regression:** the chassis strafes (`vy`) to track path noise → promotion
  fails; keep gated, refresh the inline justification with the observed
  strafe. Weakest portability case of the four.
- **Disposition:** PENDING.

## Investigation pointers

- `source/strafer_ros/strafer_navigation/launch/navigation.launch.py`
  — `_patch_params`, now split into the three labeled sections
  (universal defaults / velocity-envelope coupling / behavioral
  overrides under promotion). The behavioral-overrides section is where
  the four candidate knobs sit, each with its inline promotion criterion.
- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml`
  — universal defaults end up here after promotion.
- `source/strafer_ros/strafer_navigation/test/test_nav_config.py`
  — `TestPromotionSplitInvariants` (per-section invariants) and
  `TestPatchByteIdentical` (byte-identical golden pin) are where the
  split is enforced.
- Predecessor briefs' validation harness: the sim-velocity
  bisection scripts referenced from
  [`completed/mppi-critic-tuning-for-sim-envelope.md`](mppi-critic-tuning-for-sim-envelope.md)'s
  Investigation pointers. Re-runnable for the validation laps.

## Out of scope

- **A `STRAFER_NAV_PROFILE` env var** (`stable` / `dev` / etc.) for
  selecting param profiles. Considered and rejected for v1 — the
  velocity-coupled split already gives us the two-bucket distinction;
  a profile env var would conflate "experimental knobs" with
  "lane-specific physics" again. File a follow-up if multiple
  simultaneous active experiments make the universal default hard to
  reason about.
- **A separate `nav2_params_sim.yaml`**. Predecessor briefs
  (`mppi-critic-tuning-for-sim-envelope.md` option C) evaluated and
  rejected this; doubles the maintenance surface for every future
  param change.
- **Promoting `vx_std`/`vy_std`/`wz_std`/`prune_distance` scaling**
  off the velocity-envelope gate. These are genuinely
  velocity-coupled; they stay gated by construction.
- **The executor-side knobs** (`MissionRunnerConfig`, motion
  timeouts, rotate-then-translate thresholds). Different concern;
  the executor is Python-only and doesn't share this gating issue.
- **Switching planners or motion models** (SmacPlanner2D, DiffDrive
  motion model for non-cardinal goals). Filed elsewhere.
