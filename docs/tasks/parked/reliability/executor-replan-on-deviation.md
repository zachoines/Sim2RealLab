# Re-plan on a confirmed mid-mission deviation instead of failing

**Type:** new feature
**Owner:** Either (the deviation contract + the compiler's bounded
re-plan are DGX-lane; the executor's cancel-and-dispatch handler is
Jetson-lane — file the heart of the work in whichever lane picks it
up, and a follow-up in the other per
[`ownership-boundaries.md`](../../context/ownership-boundaries.md))
**Priority:** P2
**Estimate:** M (~2–3 days; compiler re-plan entry point + a
structured deviation-reason contract + executor cancel/route handler
+ tests across both backends)
**Branch:** task/executor-replan-on-deviation

**Pickup gate:** Blocked-on-deps until **both**:

1. The per-leg deviation contract is defined —
   [`clip-multi-room-validator-remeasure`](../../parked/clip-validation/clip-multi-room-validator-remeasure.md)
   (which leg-types arm the tripwire, what a *confirmed* deviation
   is, and the structured reason the arbiter emits).
2. The validator is wired into the runtime —
   [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
   (the cheap tripwire + VLM arbiter cascade actually runs
   mid-mission and produces a categorical signal).

Without (1) there is no contract to react to; without (2) there is
no signal to react to. Un-park by `git mv` per
[`README.md`'s Directory layout](../../README.md#directory-layout).

## Story

As a **mission operator whose robot is confirmed mid-mission to be
heading to the wrong room or the wrong instance of the target**, I
want **the executor to cancel the current goal and route a
structured reason back to the compiler for a bounded re-plan from
the robot's current pose**, so that **a recoverable deviation
becomes a corrected mission rather than an immediate failure**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
  — this brief spans both lanes; the boundary list says which side
  owns each piece.
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/perception-backbone-architecture.md`](../../context/perception-backbone-architecture.md)
  — the frozen perception spine the validator's signals ride on.
- [`clip-multi-room-validator-remeasure`](../../parked/clip-validation/clip-multi-room-validator-remeasure.md)
  — owns the per-leg deviation contract and the structured reason
  this brief consumes. **This brief reacts to that contract; it
  does not define it.**
- [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
  — the CLIP tripwire + VLM arbiter that emits the categorical
  signal (`on_course` / `wrong_room` / `wrong_instance` / ...).
- [`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
  — the compiler's transit-or-explore decision this brief re-enters
  on `wrong_room`. It explicitly **defers** plan-repair: "If the
  transit step itself fails, the mission fails — no auto-replan"
  (Out of scope, "Plan repair on transit failure").
- [`executor-grounding-loss-mid-mission-recovery`](../../active/reliability/executor-grounding-loss-mid-mission-recovery.md)
  — the sibling active recovery brief whose bounded mini-scan +
  semantic-map fallback this brief reuses on the `wrong_instance`
  re-ground path.

## Context

The operator's idea is **"detect deviation → stop → re-plan."**
The *detect* half is owned upstream: the validator's tripwire +
arbiter cascade in
[`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
confirms a deviation and emits a categorical reason, and
[`clip-multi-room-validator-remeasure`](../../parked/clip-validation/clip-multi-room-validator-remeasure.md)
owns the per-leg contract that decides *when* a leg is even eligible
to be scored as off-course. This brief is the **"stop → re-plan"
half**, which **no brief currently owns**:
[`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
explicitly defers it — "If the transit step itself fails, the
mission fails — no auto-replan." Today a confirmed deviation can
only abort.

The signal already carries the shape a useful response needs.
`MISSION_VALIDATION_ARCHITECTURE.md` §2.3 frames deviation as a
granularity hierarchy (wrong room / wrong instance / wrong
trajectory), and the arbiter's job is to "emit a *useful* abort
signal — not just 'stop' but 'stop, re-plan with the correct
instance'." This brief is where that "re-plan" lands.

### Stop → route a structured reason to the compiler

On a **confirmed** deviation (from the validator's arbiter, not a
raw tripwire fire), cancel the current nav goal and route the
**structured reason** to the compiler for a *bounded* re-plan from
the **current pose** — distinct from a mission failure:

- **`wrong_room`** → the compiler re-runs its transit-or-explore
  decision from the **current pose**, with `current_room` now
  updated to wherever the robot actually is. This is the same
  `_compile_go_to_target` transit-or-explore branch
  [`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
  already ships — it is simply re-entered with corrected world
  state, **not** treated as `navigate_via_staging_target_lost`.
- **`wrong_instance`** → a bounded **re-ground** before resuming:
  re-run `scan_for_target` **excluding the reached alternate
  instance**, then resume the leg against the corrected target.
  This is the same bounded mini-scan + semantic-map fallback the
  sibling
  [`executor-grounding-loss-mid-mission-recovery`](../../active/reliability/executor-grounding-loss-mid-mission-recovery.md)
  brief already defines; this brief adds the "exclude the alternate
  we just reached" predicate and the budget shared with that
  recovery path.

A bounded re-plan budget (shared with the grounding-loss recovery
budget) prevents thrash: after N re-plans the mission still fails,
preserving the existing terminal error path.

### Sibling of the grounding-loss recovery brief

This is a **sibling** of
[`executor-grounding-loss-mid-mission-recovery`](../../active/reliability/executor-grounding-loss-mid-mission-recovery.md)
— same "bound a recovery before escalating" principle, same
mini-scan + semantic-map-fallback primitives, same per-mission
budget. The cleanest implementation likely **extends that recovery
framework** (a new recovery trigger keyed on a confirmed deviation
reason) rather than starting a fresh handler. The difference is the
trigger and the routing: grounding-loss recovers *in place* when
the target is lost at arrival; this brief recovers when the
*destination itself* was wrong and the correction is a re-plan, not
a local re-scan.

### Backend handling — route above Nav2, never conflate

The cancel-and-route handler must respect the active backend:

- **`strafer_direct`** — there is no Nav2 leg; the RL policy is
  driving toward the goal pose via the strafer_inference action
  server. Abort the RL goal, then route the structured reason to
  the compiler.
- **`hybrid` (`STRAFER_NAV_BACKEND=hybrid_nav2_strafer`)** — cancel
  the Nav2 goal **and** route the **semantic** goal change to the
  compiler. This is deliberately distinct from Nav2's own
  **geometric** replan: Nav2 already replans a path to a *fixed*
  goal pose around new obstacles. This brief changes *which goal*,
  which is a semantic decision the compiler owns. **Route above
  Nav2 — do not conflate the two.** A geometric Nav2 replan to the
  wrong goal does not fix a `wrong_room` deviation.

### Cross-lane split

- **DGX lane:** the structured deviation reason contract (consumed
  here, owned by the validator-remeasure brief) and the compiler's
  bounded re-plan entry point that re-runs transit-or-explore from
  the current pose.
- **Jetson lane:** the executor handler that cancels the active
  goal (RL or Nav2 per backend) and dispatches the re-plan request
  / re-ground.

Per [`ownership-boundaries.md`](../../context/ownership-boundaries.md),
file the heart of the work in the lane that picks it up and a
follow-up in the other; do not reach across the boundary.

## Acceptance criteria

- [ ] **Confirmed deviation does not auto-fail.** On a confirmed
      `wrong_room` / `wrong_instance` from the validator's arbiter
      (not a raw tripwire fire), the executor cancels the active
      goal and routes a structured reason to the compiler instead
      of returning a terminal failure.
- [ ] **`wrong_room` re-enters transit-or-explore.** The compiler
      re-runs `_compile_go_to_target`'s transit-or-explore decision
      from the current pose with `current_room` updated. Verified
      by a test: a mission whose first leg lands in the wrong room
      re-plans a transit/explore step to the correct room rather
      than failing.
- [ ] **`wrong_instance` re-grounds excluding the alternate.**
      `scan_for_target` is re-run with the reached alternate
      instance excluded, then the leg resumes against the corrected
      target. Verified by a test in a scene with ≥ 2 same-label
      instances.
- [ ] **Backend-aware cancel.** Under `strafer_direct` the RL goal
      is aborted; under `hybrid` the Nav2 goal is cancelled. In
      neither case is the semantic re-plan delegated to Nav2's
      geometric replanner. A test pins that a `wrong_room`
      deviation under `hybrid` produces a *new compiler goal*, not
      a Nav2 re-route to the old goal pose.
- [ ] **Bounded budget.** Re-plans share a per-mission budget with
      the grounding-loss recovery budget; exhausting it returns the
      existing terminal failure with the re-plan count recorded in
      the mission outputs / stage log.
- [ ] **Cancel honored throughout.** An operator cancel during the
      re-plan round-trip or the `wrong_instance` re-ground
      terminates with `mission_canceled`, matching the sibling
      recovery brief's cancel semantics.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in clean missions: a mission with no confirmed
      deviation runs the existing plan path unchanged (smoke an
      existing single-room and a cross-room mission through the
      bridge harness; plan structure and outcomes match the
      pre-change baseline).

## Investigation pointers

- The deferral this brief closes:
  [`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
  Out of scope, "Plan repair on transit failure" — "the mission
  fails — no auto-replan."
- The signal source + categorical reasons:
  [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md)
  (tripwire + arbiter cascade) and the §2.3 granularity hierarchy +
  the "stop, re-plan with the correct instance" framing in
  [`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md).
- The per-leg deviation contract this brief reacts to:
  [`clip-multi-room-validator-remeasure`](../../parked/clip-validation/clip-multi-room-validator-remeasure.md).
- The recovery framework to extend (mini-scan + semantic-map
  fallback + per-mission budget + cancel semantics):
  [`executor-grounding-loss-mid-mission-recovery`](../../active/reliability/executor-grounding-loss-mid-mission-recovery.md).
- The compiler re-entry point: `_compile_go_to_target`'s
  transit-or-explore branch in
  [`planner/plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py),
  described in
  [`autonomy-stack`](../../active/multi-room/autonomy-stack.md).
- The backend dispatch seam (cancel RL vs. Nav2 goal):
  `JetsonRosClient.navigate_to_pose` and the
  `STRAFER_NAV_BACKEND` selection, per
  [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md).

## Out of scope

- **Defining the per-leg deviation contract.** Owned by
  [`clip-multi-room-validator-remeasure`](../../parked/clip-validation/clip-multi-room-validator-remeasure.md);
  this brief consumes it. Do not re-encode the leg-type / arm-state
  table here.
- **The detect half of the loop.** The tripwire, the arbiter
  cascade, and their calibration belong to
  [`validator-evaluation`](../../active/clip-validation/validator-evaluation.md).
- **Nav2's geometric replanning.** Replanning a path to a *fixed*
  goal pose around obstacles is Nav2's own job and unchanged here.
  This brief only changes *which goal* the robot pursues.
- **`wrong_trajectory` / case-3 deviations.** Trajectory-shape
  constraints need a planner-side decomposition that does not exist
  yet (§4 / §2.3 of `MISSION_VALIDATION_ARCHITECTURE.md`); only
  `wrong_room` and `wrong_instance` are in scope.
- **Operator-in-the-loop re-planning.** Asking the operator to
  re-state an ambiguous target is a separate escalation path, not
  this autonomous re-plan.
