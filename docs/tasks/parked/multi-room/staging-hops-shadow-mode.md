# Shadow-mode `staging_hops`: log the LLM's advisory hops without consuming them

**Type:** new feature (filed-on-deps)
**Owner:** DGX agent (planner prompt + payload + logging path;
small cross-lane wire-up in `mission_runner.py` for the
agreement-rate metric on the executor side).
**Priority:** P3 — pick up after the Option C compiler work
(`autonomy-stack` + `planner-far-target-staging`) ships, so the
metric being measured (LLM hops vs. compiler plan) compares
against the room-aware compiler, not the pre-Option-C stub.
**Estimate:** S–M (~2–3 days; schema field + prompt + few-shot +
logging + analysis script + ≥ a week of shadow data on the
integration mission queue).
**Branch:** task/staging-hops-shadow-mode

**Pickup gate:** Blocked on
[`autonomy-stack`](../../active/multi-room/autonomy-stack.md)
shipping (need the room-aware compiler to compare against) AND
[`planner-far-target-staging`](../../active/multi-room/planner-far-target-staging.md)
shipping (need the far-target compiler helper). Un-park by
`git mv parked/multi-room/<this>.md
active/multi-room/<this>.md` in the PR that picks it up, per
[`README.md`'s Directory layout](../../README.md#directory-layout).

## Story

As an **operator running the C → B migration**, I want **the
planner LLM to emit an advisory `staging_hops` list on every
`MissionIntent` while the compiler logs but ignores it**, so
that **the project accumulates empirical data on whether the
LLM's spatial reasoning agrees with, disagrees with, or
defensibly beats the deterministic compiler — without committing
the runtime to LLM-emitted plans before that data exists**.
The data feeds two downstream decisions:

1. Whether to promote `staging_hops` to advisory (step 3 of the
   migration in
   [`STRAFER_AUTONOMY_NEXT.md` §1.10.2](../../../STRAFER_AUTONOMY_NEXT.md#1102-planner-architecture-decision-option-c)).
2. Whether to file
   [`planner-scene-graph-expansion`](planner-scene-graph-expansion.md)
   active — that brief un-parks specifically when this brief's
   shadow data shows the LLM is bottlenecked on scene state,
   not on prompt engineering.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`context/planner-request-schema.md`](../../context/planner-request-schema.md)
  — defines `staging_hops` as the reserved advisory slot on
  `MissionIntent`; this brief is the one that actually fills it.
- [`planner-architecture-alignment`](../../completed/planner-architecture-alignment.md)
  — recorded Option C; spells out the C → B migration path and
  names this brief as step 2.
- [`planner-scene-graph-expansion`](planner-scene-graph-expansion.md)
  — sibling parked brief that consumes this brief's output as
  its trigger signal.

## Context

### What ships

Three small, independent surfaces:

1. **`MissionIntent.staging_hops`** — already reserved as an
   optional field per
   [`context/planner-request-schema.md`](../../context/planner-request-schema.md).
   This brief flips it from "reserved" to "populated by the
   LLM." Shape: `tuple[str, ...] | None`, where each entry is
   either a room label (`"kitchen"`) or a named landmark from
   the semantic map.
2. **Prompt + few-shot update.** Teaches the LLM when and how
   to emit `staging_hops` for `go_to_target` / `go_to_targets`
   / `patrol` intents. The instruction is "emit the rooms /
   landmarks you'd pass through" — not "emit a step sequence."
   Few-shot examples cover cross-room missions (current room ≠
   target room) and far-target missions (target past the SLAM
   horizon).
3. **Shadow logging in the compiler.** The compiler reads
   `intent.staging_hops`, computes the agreement label against
   its own emitted plan, and writes a structured log record.
   The field is **never used** to change the emitted plan.

### Agreement-rate metric

The disagreement classification is what
[`planner-scene-graph-expansion`](planner-scene-graph-expansion.md)'s
trigger triage consumes. Per-mission log record:

```python
{
    "mission_id": str,
    "raw_command": str,
    "current_room": str | None,
    "target_label": str,
    "target_room_inferred_by_compiler": str | None,
    "llm_staging_hops": list[str] | None,
    "compiler_transit_step_room": str | None,
    "agreement_label": str,    # one of the values below
}
```

`agreement_label` values:

| Label | Condition |
|---|---|
| `no_hops` | LLM did not emit `staging_hops`. |
| `agree` | LLM's first hop matches the compiler's transit-step room. |
| `disagree_room` | LLM's first hop is a different room than the compiler's. |
| `disagree_intra_room` | LLM and compiler agree on the target room but LLM proposed a specific landmark, compiler proposed the `room_anchor`. |
| `disagree_disambig` | Operator command was ambiguous ("the chair next to the table"); LLM emitted a disambiguating landmark, compiler defaulted. |
| `disagree_hallucination` | LLM emitted a room or landmark that doesn't exist in `world_state.known_rooms` or the semantic map. |

The classifier is deterministic — it just compares the two
plans against `world_state`. No human-in-the-loop required.

### Out-of-band analysis

An offline script aggregates the log records into a weekly
agreement-rate report. The report is the data product this
brief delivers; the brief is "done" only when ≥ a week of
production missions are in it.

## Approach

In order:

1. **Schema flip.** `MissionIntent.staging_hops` already
   reserved as `tuple[str, ...] | None = None`. Confirm wire
   payload + dataclass agree.
2. **Prompt update** in
   [`planner/prompt_builder.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py).
   Add the instruction + 3–5 few-shot examples that emit
   `staging_hops` for cross-room / far-target missions. Verify
   existing single-room missions emit `staging_hops: None`.
3. **Compiler shadow logging** in
   [`planner/plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py).
   After the compiler emits its plan, compute the
   agreement-label record. Write to a structured log channel
   (the existing planner-service log path is fine; if a
   dedicated channel makes downstream analysis easier, add one
   in this brief).
4. **Offline analysis script.** A small Python script that
   reads the log file(s) and emits the weekly report.
5. **Run + report.** Run ≥ a week of integration missions
   through the bridge harness (or production missions, if
   real-robot bringup has begun). Attach the report to this
   brief's PR description; it is the trigger signal for
   [`planner-scene-graph-expansion`](planner-scene-graph-expansion.md)
   per that brief's "Trigger detail" section.

## Acceptance criteria

- [ ] **`staging_hops` populated by the LLM.** A canned set of
      cross-room and far-target prompts produces non-empty
      `staging_hops`; canned single-room prompts produce
      `None`. Tested in the planner unit tests.
- [ ] **Compiler does NOT consume `staging_hops`.** A grep of
      [`plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py)
      shows `intent.staging_hops` is read into the log record
      only — never branched on, never inserted into a
      `SkillCall`. Verified by a unit test that emits a
      `MissionIntent` with arbitrary `staging_hops` and checks
      the compiled plan matches the no-hops baseline.
- [ ] **Agreement-label classifier ships.** Deterministic
      function `classify_staging_hops_agreement(intent, plan,
      world_state) -> AgreementLabel`. Unit-tested for each of
      the six label cases.
- [ ] **Log record emitted per mission.** Structured record
      with the seven fields above, written to a log channel
      the analysis script can read.
- [ ] **Offline analysis script.** `Scripts/analyze_staging_hops_agreement.py`
      (or equivalent) reads the log channel and emits a
      report with per-label counts + per-label example missions
      (raw_command + target_label) for triage.
- [ ] **Weekly shadow report.** ≥ a week of missions logged;
      report attached to the brief's PR description. Report
      labels the dominant disagreement mode per
      [`planner-scene-graph-expansion`](planner-scene-graph-expansion.md)'s
      trigger triage so the downstream un-park decision is
      criterion-driven.
- [ ] **No regression in compiled plans.** A regression set of
      canned `(intent, world_state)` pairs produces the same
      emitted plans before and after this brief. The shadow
      logging path is non-load-bearing.
- [ ] If your work invalidates a fact in any referenced
      context module, package README, top-level `Readme.md`,
      or guide under `docs/`, update those in the same commit.
      See
      [`conventions.md`'s user-facing documentation
      maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- Reserved slot location:
  [`context/planner-request-schema.md`](../../context/planner-request-schema.md)
  describes `MissionIntent.staging_hops` shape. The field is
  inert until this brief lands.
- Current prompt construction:
  [`planner/prompt_builder.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py).
- Current intent parser:
  [`planner/intent_parser.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/intent_parser.py)
  — adding the new field requires extending the parser's
  validation logic.
- Compiler hook point:
  [`planner/plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py)'s
  `compile_plan` is the natural place to compute the
  agreement label after emitting the plan.
- The room-aware compiler logic this brief compares against
  ships in
  [`autonomy-stack`](../../active/multi-room/autonomy-stack.md);
  the far-target staging helper ships in
  [`planner-far-target-staging`](../../active/multi-room/planner-far-target-staging.md).

## Out of scope

- **Consuming `staging_hops` to change the plan.** That is
  step 3 of the migration (advisory) and step 4 (authoritative),
  filed separately once this brief's data justifies them.
- **Scene-graph expansion on `world_state`.** Tracked at
  [`planner-scene-graph-expansion`](planner-scene-graph-expansion.md);
  un-parks when this brief's data shows the LLM is bottlenecked
  on missing scene state.
- **Constrained-output decoder.** Prompt-side instruction is
  enough for the shadow phase; constrained decoding is a
  separate brief if free-form generation produces unparseable
  hops too often.
- **Prompt-regression test harness.** The shadow phase needs
  the agreement metric, not a regression harness. The harness
  is filed alongside step 3 when the compiler starts consuming
  `staging_hops`.
- **Production-mission shadow data on the real robot.** The
  ≥ a week of shadow data is allowed to come from
  bridge-harness integration missions; real-robot bringup is
  not a prerequisite.
