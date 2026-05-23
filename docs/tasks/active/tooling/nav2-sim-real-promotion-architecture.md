# Nav2 sim→real promotion architecture: split velocity-coupled from behavioral overrides

**Type:** task / refactor + process
**Owner:** Jetson agent (`source/strafer_ros/strafer_navigation/`)
**Priority:** P2
**Estimate:** M (~1 day refactor + per-knob real-robot validation laps that
stretch across multiple sessions)
**Branch:** task/nav2-sim-real-promotion-architecture

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
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/mppi-critic-tuning-for-sim-envelope.md](../../completed/mppi-critic-tuning-for-sim-envelope.md)
  — the originating brief for the `envelope_factor > 1.0` gate
  pattern. Filed option A specifically *because* the MPPI rebalance
  is velocity-coupled.
- [completed/nav2-startup-unknown-donut-path-noise.md](../../completed/nav2-startup-unknown-donut-path-noise.md)
  — the predecessor that pattern-matched onto the gate for a
  behavioral change (SmoothPath BT), not a velocity-coupled one.
- [`active/reliability/nav2-commit-and-follow-path-stability.md`](../reliability/nav2-commit-and-follow-path-stability.md)
  — the parent brief that already ungated `allow_unknown` and the
  smoothing BT to the universal default. This brief picks up where
  that one left off: formalize the split and migrate the remaining
  gated knobs.

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
| `PathAlignCritic.cost_weight` 14 → 9 | `envelope_factor > 1.0` | **Possibly** — tuned for high-vel integration noise on straights; could matter less but still rebalance at low-vel |
| `PreferForwardCritic.cost_weight` 3 → 10 | `envelope_factor > 1.0` | **Probably not** — biases against strafe/spin; useful at any velocity, just more critical when exploration is wider |
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
        # PathAlignCritic 14 → 9 — promotion criterion: real-robot
        # cornering smoke (translate forward 1 m → rotate 90° →
        # translate forward 1 m) lands within tolerance with
        # PathAlign=9 instead of 14.
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

1. **`PreferForwardCritic.cost_weight` 3 → 10**: probably the most
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
4. **`PathAlignCritic.cost_weight` 14 → 9**: most coupled to
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

- [ ] `_patch_params` is restructured into the three labeled
      sections (constants / velocity-envelope coupling / behavioral
      overrides). Each currently-gated knob carries an inline comment
      naming its promotion criterion.
- [ ] `docs/tasks/context/nav2-knob-promotion.md` exists and is linked
      from `docs/tasks/context/README.md` (or the context-module
      index, wherever conventions point). The module describes the
      five-step contract above and gives one worked example.
- [ ] At least one of the four MPPI knobs is run through the
      validation lap and either promoted to the universal default
      OR documented as "stays gated, refreshed justification: ___".
      The PR description carries the validation observations
      (mission, observed metric, disposition).
- [ ] The brief at
      [`active/reliability/nav2-commit-and-follow-path-stability.md`](../reliability/nav2-commit-and-follow-path-stability.md)
      gets its outstanding "real-robot validation lap" bullet
      checked off — i.e., a real-robot run of the smoothing BT +
      `allow_unknown=False` defaults happens and is recorded.
- [ ] Unit tests pin the universal vs. gated split: a test asserts
      every knob in the velocity-envelope section runs only when
      `envelope_factor > 1.0`; a test asserts every knob in the
      universal section runs on every `envelope_factor`. (Today's
      `test_nav_config.py::TestConstantsInjection` is the right
      pattern — extend it as the refactor lands.)
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- `source/strafer_ros/strafer_navigation/launch/navigation.launch.py`
  — `_patch_params`. The current `if envelope_factor > 1.0:` block
  is what's being refactored; the constants-injection block above it
  is the model.
- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml`
  — universal defaults end up here after promotion.
- `source/strafer_ros/strafer_navigation/test/test_nav_config.py`
  — `TestConstantsInjection` is the right place to add per-section
  invariants.
- Predecessor briefs' validation harness: the sim-velocity
  bisection scripts referenced from
  [`completed/mppi-critic-tuning-for-sim-envelope.md`](../../completed/mppi-critic-tuning-for-sim-envelope.md)'s
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
