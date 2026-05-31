# Multi-room validator re-test — owns the per-leg / sub-goal deviation contract

**Type:** investigation / docs
**Owner:** Either (the deviation-contract wiring touches the
executor's leg-type plumbing in `strafer_autonomy`; the metric
re-run + recalibration is DGX-led against multi-room harness
output — cross-lane like
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md))
**Priority:** P2 (filed-on-trigger; not pickable until both
prerequisites below ship)
**Estimate:** M (~half-week; small leg-type-plumbing change +
multi-room metric re-run + bar recalibration + write-up)
**Branch:** task/clip-multi-room-validator-remeasure

This brief is **named in
[`MISSION_VALIDATION_ARCHITECTURE.md` §4.3](../../../MISSION_VALIDATION_ARCHITECTURE.md#43-prerequisite-briefs-filed-alongside-this-recommendation)**
("to be filed when
[`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
ships"). It is filed **parked, filed-on-trigger**: the path is the
state — un-park it with a single `git mv` into
`active/clip-validation/` in the PR that picks it up.

It implements the per-leg / sub-goal deviation work recorded in
[`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md)
(the frozen spine) and the
[`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
granularity hierarchy (§2.3). Read the spine first; this brief is
the multi-room consumer of the shared trunk's validator head and
must stay consistent with the spine's decision and consumer table.

## Story

As an **operator who has shipped multi-room autonomy and a
single-room v1 CLIP cascade, and now wants the validator to behave
correctly across cross-room missions**, I want **the per-leg /
sub-goal DEVIATION CONTRACT made authoritative — every plan
primitive's validator state defined, both navigation backends
accounted for, exploration carved out as not-deviation — and the
per-case ROC-AUC metrics re-run on multi-room data with the §4.1
bars recalibrated**, so that **the cascade stops labeling
legitimate cross-room autonomy (frontier exploration, "via the
dining room" staging hops) as off-course, and the §4.1 bars
reflect the harder multi-room distribution rather than the
single-room subset v1 was calibrated on**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/bridge-runtime-invariants.md`](../../context/bridge-runtime-invariants.md)
- [`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md)
  — the frozen-trunk spine. This brief's validator is the same
  case-1 + case-2 consumer on the shared trunk; the deviation
  contract here governs *when* that consumer is armed, not *what*
  trunk it runs on.

Parent design doc:
[`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md).
§2.3 (the case-1 / case-2 / case-3 granularity hierarchy), §4.1
(the falsifiable bars this brief recalibrates), and §4.3 (which
names this brief).

Sibling briefs:
- [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
  — the v1 cascade. It wires the **single-room subset** of the
  leg-type suppression flag and accepts the per-leg
  `(leg_type, sub_goal_room, sub_goal_target_phrase)` context; it
  explicitly **defers the full contract to this brief.** v1
  shipping is a hard prerequisite.
- [`autonomy-stack`](../../active/multi-room/autonomy-stack.md) —
  ships the multi-room runtime: the
  `scan → navigate(intermediate) → scan → navigate(final)`
  plan-leg structure, `explore_until_visible`, and the
  staging-hop `navigate_to_pose`. It defers plan-repair
  ("no automatic re-plan" when a transit step fails). Its shipping
  is the filed-on-trigger condition for this brief.

## Context

### What this brief OWNS — the per-leg / sub-goal deviation contract

The v1 brief deliberately leaves the **authoritative** leg-type
table here so the contract lives in one place. This is that table.
It is keyed on the plan primitive the executor is currently
running and replaces "validate against the mission's final target"
with "validate against the *current leg's sub-goal*":

| Leg primitive | Validator state | Validates against | Notes |
|---|---|---|---|
| `scan_for_target` | **OFF** | nothing | No locomotion — there is nothing to be off-course about. |
| `explore_until_visible` | **DISARMED** | nothing (hard CLIP) | Exploration is **not** deviation: the robot is *meant* to walk ranked frontiers and visit plausible-but-wrong rooms. Frontier-vs-target consistency stays a **soft LLM ranking signal owned by [`llm-guided-frontier-gain`](../../parked/multi-room/llm-guided-frontier-gain.md)** (and the [`frontier-cognitive-fsm`](../../parked/multi-room/frontier-cognitive-fsm.md) variant) — **never** a hard CLIP abort. |
| `navigate_to_pose` (staging hop) | **ARMED (semantic)** | the **INTERMEDIATE hop's room** | A staging hop from [`planner-far-target-staging`](../../active/multi-room/planner-far-target-staging.md) is committed to reaching an intermediate room (e.g., the dining room en route to the kitchen), not the final target's room. Validate case-1 against the **hop's** room. **No CLIP geodesic** — geometric drift is the backend's job (see Backend awareness). |
| `navigate_to_pose` (final target) | **FULLY ARMED** | case-1 (target room) + case-2 (target instance) | The only leg that gets the full cascade against the mission's final target. **No CLIP geodesic.** |
| `verify_arrival` | existing arrival check | the arrived target | Unchanged from the existing `_verify_arrival` path. |

The executor supplies `(leg_type, sub_goal_room,
sub_goal_target_phrase)` at the validator's `activate()` per leg —
the same tuple v1 wires; this brief makes the full table
authoritative and validates it against the multi-room plan-leg
structure
(`scan → navigate(intermediate) → scan → navigate(final)`).

### Backend awareness — geometric drift is NOT a CLIP job on either backend

CLIP is the **semantic** validator (right room / right instance);
**geometric off-path** is owned by the navigation backend, and the
two backends own it differently:

- **`strafer_direct`** (the RL reactive backend — no explicit
  planned path): there is no geodesic to deviate from. Geometric
  "stuck / no-progress" is a **locomotion watchdog**
  (pose-advance + yaw-rate over a window), filed at
  [`nav-stall-multilayer-watchdog`](../../parked/reliability/nav-stall-multilayer-watchdog.md).
  Not a perception job.
- **`hybrid`** (Nav2 owns the path): Nav2 already knows when the
  robot is off **its own** plan. Validate semantics at Nav2
  waypoints; **never recompute a geodesic.** A re-derived
  geodesic-to-final-target would flag a legitimate "via the dining
  room" staging path as off-course, exactly the false positive the
  whole contract exists to avoid.

The unifying rule: **no CLIP recipe computes a geometric deviation
on either backend.** The validator answers room/instance questions
only; the geometric question is answered better, and earlier, by
the layer that owns the path (or the absence of one).

### Re-run the per-case metrics on multi-room data (original scope)

Beyond owning the contract, this brief keeps its original §4.3
scope: re-run the v1 cascade's per-case ROC-AUC + PR-AUC + Brier +
McNemar + time-to-decision metrics on the **multi-room** harness
corpus (the now-default scene set), disaggregated per case and per
signal exactly as v1 reports them, and **recalibrate the §4.1
bars** against the harder multi-room distribution. v1's
single-room bars are an achievable floor before multi-room raises
the difficulty; this brief records the multi-room bars and the
delta from single-room as the legible cross-distribution signal.

The signal-emphasis framing carries over from v1: image-vs-text is
the cold-start-safe primary, image-vs-image place recognition and
Step B memory are warm boosters. Multi-room missions cross more
rooms and exercise more cold-start legs (the robot enters
not-yet-mapped rooms), so the warm/cold disaggregation and the
contrast-pool-stratified Brier matter *more* here, not less.

## Acceptance criteria

- [ ] **Deviation contract is authoritative and wired.** The
      validator's `activate()` honors the full leg-type table
      above across the multi-room plan-leg structure: `scan` →
      OFF, `explore_until_visible` → DISARMED, staging-hop
      `navigate_to_pose` → ARMED-semantic against the
      intermediate hop's room, final `navigate_to_pose` → FULLY
      ARMED case-1 + case-2, `verify_arrival` unchanged. No leg
      type computes a CLIP geodesic. Unit-tested under
      [`source/strafer_autonomy/tests/`](../../../../source/strafer_autonomy/tests/),
      including a test that an `explore_until_visible` leg never
      fires a hard CLIP abort and a test that a staging hop
      validates against the hop's room, not the final target's.
- [ ] **Exploration consistency stays a soft signal.** No hard
      CLIP abort fires on `explore_until_visible`; frontier-vs-target
      consistency is confirmed to remain the soft LLM ranking
      owned by
      [`llm-guided-frontier-gain`](../../parked/multi-room/llm-guided-frontier-gain.md),
      not duplicated as a CLIP tripwire here.
- [ ] **Backend awareness.** The semantic validator runs at Nav2
      waypoints under `hybrid` without recomputing a geodesic, and
      under `strafer_direct` geometric drift is left to the
      locomotion watchdog
      ([`nav-stall-multilayer-watchdog`](../../parked/reliability/nav-stall-multilayer-watchdog.md)),
      not to CLIP. A multi-room mission whose plan routes "via the
      dining room" is **not** labeled off-course by the validator.
- [ ] **Per-case metrics re-run on multi-room data.** The v1
      measurement script is re-run on the multi-room harness
      corpus (≥ 3 distinct multi-room Infinigen scenes, ≥ 30
      missions), producing per-case (case-1 / case-2), per-signal
      (image-vs-text primary, image-vs-image warm booster, OR-fused)
      ROC-AUC + PR-AUC + Brier (stratified by contrast-pool
      composition) + McNemar + time-to-decision, with the
      warm/cold `map_state` disaggregation.
- [ ] **§4.1 bars recalibrated.** The recalibrated multi-room
      per-case bars and the delta from the single-room v1 bars are
      recorded in the §4.4 addendum to
      [`MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md);
      the per-leg-type breakdown shows the validator is armed only
      on the leg types the contract arms.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in single-room missions — the contract must
      reduce to the v1 single-room behavior when a mission has a
      single committed navigate-to-target leg. Smoke a single-room
      mission and confirm the validator arms exactly as v1 did.

## Investigation pointers

- The per-leg context the validator already accepts:
  `(leg_type, sub_goal_room, sub_goal_target_phrase)` at
  `activate()`, wired by
  [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
  for the single-room subset.
- The multi-room plan-leg structure and `explore_until_visible` /
  staging-hop primitives:
  [`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
  (the `scan → navigate(intermediate) → scan → navigate(final)`
  plan; plan-repair is deferred there).
- The staging hop's intermediate room comes from
  [`planner-far-target-staging`](../../active/multi-room/planner-far-target-staging.md).
- The locomotion watchdog that owns `strafer_direct` geometric
  drift:
  [`nav-stall-multilayer-watchdog`](../../parked/reliability/nav-stall-multilayer-watchdog.md).
- The measurement script to re-run:
  `Scripts/eval_transit_monitor.py` (DGX-side), produced by v1.

## Out of scope

- **Shipping the v1 cascade.** Owned by
  [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md);
  this brief is the multi-room re-test that runs after it.
- **Choosing the backbone.** Owned by
  [`backbone-bakeoff`](backbone-bakeoff.md) (runs before v1 per the
  spine's roadmap). This brief validates *when* the validator is
  armed, on whatever trunk the bake-off chose.
- **Computing geometric deviation in CLIP.** Explicitly dropped:
  geometric off-path is the backend's job (the `strafer_direct`
  watchdog and the `hybrid` Nav2 plan-freshness tracker), never a
  CLIP geodesic on either backend.
- **Frontier-ranking policy.** The soft LLM frontier-vs-target
  ranking lives in
  [`llm-guided-frontier-gain`](../../parked/multi-room/llm-guided-frontier-gain.md)
  / [`frontier-cognitive-fsm`](../../parked/multi-room/frontier-cognitive-fsm.md);
  this brief only confirms it is **not** promoted to a hard CLIP
  abort.
- **Plan repair on transit failure.** Deferred by
  [`autonomy-stack`](../../active/multi-room/autonomy-stack.md);
  this brief does not add re-planning.
- **Case-3 (trajectory-shape) validation.** A planner-side
  prerequisite (§2.3 / §4.3); filed separately when a real mission
  requires it.

## Pickup gate

Filed-on-trigger. Pickable only when **both** of the following
have shipped:

1. [`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
   ships — the multi-room runtime exists, so the leg types this
   contract governs are real.
2. [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
   v1 ships — the single-room cascade and the per-leg context
   tuple exist for this brief to extend.

Un-park is one `git mv` into `active/clip-validation/` plus the
[`BOARD.md`](../../BOARD.md) row update, in the PR that picks it up.
