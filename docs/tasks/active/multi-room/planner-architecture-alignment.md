# Decide: keep the planner as an intent classifier, or promote it to a multi-step planner

**Type:** investigation / docs
**Owner:** DGX agent (writes the decision); operator approves
**Priority:** P1 (hard prerequisite for `autonomy-stack` and
`planner-far-target-staging`; both depend on this decision and
neither can land cleanly without it)
**Estimate:** S (~half a day; write-up + decision call + brief
edits cascading from it)
**Branch:** task/planner-architecture-alignment

## Story

As an **operator with two in-flight briefs both proposing to
extend the planner's responsibilities (multi-room transit in
`autonomy-stack`, far-target staging in
`planner-far-target-staging`)**, I want **an explicit decision
on whether the DGX planner stays a thin intent classifier (with
the deterministic plan-compiler growing the new logic) or is
promoted to a multi-step planner (with the LLM emitting full
skill sequences)**, so that **the two consuming briefs ship
against one architecture instead of accidentally creating two,
and the project's planner direction is documented before the
next quarter of work commits to it**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`autonomy-stack`](autonomy-stack.md) — wants the compiler to
  emit room-transit steps OR the LLM to emit them. Acceptance
  criteria branch on this decision.
- [`planner-far-target-staging`](planner-far-target-staging.md)
  — wants the compiler to emit staging hops OR the LLM to emit
  them. Acceptance criteria branch on this decision.

## Context

### Where the planner is today

Verified against the codebase on 2026-05-13:

- The LLM is a **strict intent classifier**. Prompt at
  [`planner/prompt_builder.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py)
  emits one of `{go_to_target, wait_by_target, go_to_targets,
  patrol, rotate, translate, describe, query, cancel, status}`
  with a few structured fields. No multi-step output.
- The **plan compiler** at
  [`planner/plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py)
  is deterministic. Every `go_to_target` becomes the same
  5-step `scan → project → align → navigate → verify` sequence
  via `_compile_single_target_steps`.
- The executor at
  [`executor/mission_runner.py`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
  runs the compiled steps in order, with cancel + retry
  handling per step.

This is a clean separation: the LLM does language → intent, the
compiler does intent → executable plan. The LLM never sees
robot state.

### What the two in-flight briefs each implicitly want

- `autonomy-stack` wants either the compiler to grow
  `_compile_far_target_staging` (intent classifier preserved),
  OR the LLM to emit multi-hop plans directly (intent
  classifier promoted).
- `planner-far-target-staging` wants the same shift, but
  describes it as "LLM emits multi-hop plans" by default.
- The two briefs both add a `world_state` block to the planner
  request, but only one of them is currently consumed.

Without a decision, the project risks shipping two flavors of
"the planner is now multi-step" — one driven by the compiler,
one driven by the LLM — that don't compose.

### The three options

**Option A — keep the intent classifier; expand the compiler.**

- The LLM continues to emit a single intent. The compiler
  inspects `world_state` + the inferred target room (from
  [`observation-derived-room-state`](observation-derived-room-state.md))
  and prepends transit / staging / exploration steps when
  needed.
- Pros: deterministic; testable in isolation; no prompt-
  engineering risk; no schema-validator changes; existing tests
  cover the LLM surface unchanged.
- Cons: the compiler grows in complexity; spatial reasoning
  the LLM could do (picking intermediate landmarks from the
  operator's natural-language description) is left on the
  table; future skills require compiler updates.

**Option B — promote the LLM to a multi-step planner.**

- The LLM emits a sequence of skill calls directly. The
  compiler shrinks to a validator (schema, argument types,
  step legality).
- Pros: spatial reasoning lives where the spatial knowledge is
  (the LLM's training corpus); new skills become a prompt /
  few-shot update, not a compiler change; aligns with the
  state-of-the-art VLA-as-planner direction (CogNav, FSR-VLN,
  osmAG-LLM).
- Cons: prompt regressions can corrupt unrelated mission
  shapes; harder to test (every plan is a roll of the dice on
  decoding); needs prompt-test harness work; LLM tokens per
  call go up.

**Option C — hybrid: LLM emits intent + optional hops.**

- The LLM still emits one `intent_type` and a `target_label`,
  but is allowed to also emit a list of `staging_hops` (room
  labels or intermediate poses) as an optional field. The
  compiler reads the hops if present, falls back to its own
  staging logic if absent.
- Pros: opt-in; old plans keep working; the LLM's spatial
  reasoning is harnessed where it helps without making the
  whole planner LLM-driven; the compiler stays the source of
  truth on step legality.
- Cons: two ways to express the same thing; the LLM might
  emit hops the compiler also emits, double-stepping the
  plan; need a "the compiler wins on conflict" rule.

### Recommendation: C now, B as the destination

**B is more expressive and has the higher ceiling.** The
LLM-as-planner direction (CogNav, FSR-VLN, osmAG-LLM, π0,
GR00T) is where the field is converging — once grounded
spatial inputs are stable, a multi-step LLM planner can chain
arbitrary skill sequences, choose intermediate landmarks from
the operator's natural-language description, and recover from
failures via re-plan. As LLMs improve, B-shaped systems scale
with them automatically.

**C is the right choice for this project right now**, and is
specifically the right choice *because it is upgrade-compatible
to B*. The migration path is bounded:

1. **Land C as specified.** Compiler is staging authority; LLM
   emits a single intent. No `staging_hops` field yet.
2. **Add an optional `staging_hops` field to the LLM output
   schema; compiler ignores it.** Shadow-mode rollout —
   measure how often the LLM emits hops that match what the
   compiler picks, where they disagree, and which is better.
3. **Promote `staging_hops` to advisory.** Compiler reads
   them, uses them when they pass legality checks, falls back
   to its own logic when they don't.
4. **Promote `staging_hops` to authoritative for compositional
   intents.** Compiler shrinks to a validator. At this point
   the system is functionally Option B for the cases that
   matter.

Each step is a separate brief with its own rollback. The
operator can stop the migration at any step if the
prompt-engineering cost outweighs the gains. No single step
asks the project to commit to LLM-as-planner before the
grounded inputs are proven.

**Why not start at B directly?** The literature
(CogNav, FSR-VLN) consistently demonstrates that LLM-as-planner
quality is dominated by *grounded scene-state quality*, not
prompt quality. Sim2RealLab is mid-flight on three grounded-
input shifts simultaneously (observation-derived room state,
frontier exploration, semantic-map room clustering). Stacking
a multi-step LLM planner concurrently means debugging four
moving systems with overlapping failure modes. There is also
no prompt-test harness today, no constrained-output decoder,
no plan-validator for non-trivial multi-step outputs — all of
which a B-first design would need up front. Starting at C buys
the project a year of compounding capability with a known
upgrade lane; starting at B asks the project to ship
infrastructure that the grounded inputs are not yet ready to
exploit.

**Why not stay at A forever?** A's compiler ceiling is set by
the compiler programmer's imagination. The strafer's long-term
direction (VLA-v2, retrieval-augmented validation, free-text
mission grammar) is incompatible with a frozen compiler.
Option A is not a destination; it's the starting point C
inherits from.

So: **C as the next ship, B as the destination, A as the
status quo we're upgrading from.** The brief edits to
`autonomy-stack` and `planner-far-target-staging` write their
acceptance criteria against C, with explicit notes that the
LLM-hop fields can be added later without breaking the
compiler interface.

This keeps the project's risk surface small (no new prompt
behavior at the same time as new compiler behavior + new
room-state inference + new exploration skill), and it leaves a
clean upgrade path. The state-of-the-art literature consistently
shows LLM-emitted plans paying off only after the *grounded*
spatial-state inputs are stable — which we are still building.
Shipping the LLM upgrade after the inputs stabilize is
defensible; before is premature optimization.

## Acceptance criteria

- [ ] **Decision recorded.** One of A / B / C is selected, with
      a one-paragraph rationale, and added to
      [`STRAFER_AUTONOMY_NEXT.md`](../../../STRAFER_AUTONOMY_NEXT.md)
      under §1.11 (Implementation plan) or a new sub-section
      near the multi-room section. The text names the briefs
      that depend on this decision.
- [ ] **`autonomy-stack` updated.** Its compiler-vs-LLM
      branching language is replaced with concrete acceptance
      criteria matching the chosen option. Acceptance bullet
      about "implementation contingent on planner-architecture-
      alignment" is removed; replaced with concrete steps.
- [ ] **`planner-far-target-staging` updated.** Same as above
      for its branching language. The brief's primary acceptance
      criterion ("the LLM emits multi-hop plans") is rewritten
      to match the chosen option.
- [ ] **One `world_state` schema, agreed.** The combined room +
      pose + costmap + last-grounding schema co-designed
      between this brief, `autonomy-stack`, and
      `planner-far-target-staging` is documented in this brief
      (or in a new `context/planner-request-schema.md` module)
      as the canonical reference.
- [ ] **No code changes.** This brief is docs-only. Code lands
      via the consuming briefs.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- Current planner prompt:
  [`planner/prompt_builder.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py).
- Current compiler:
  [`planner/plan_compiler.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py).
- Recent literature on LLM-as-multi-step-planner for navigation:
  - CogNav (ICCV 2025) — LLM emits state-aware transitions
    between exploration and identification.
  - FSR-VLN (arXiv:2509.13733) — fast/slow reasoning with
    hierarchical scene graph; LLM emits multi-step.
  - osmAG-LLM (arXiv:2507.12753) — LLM reasons over
    topometric map.
  All three demonstrate that grounded scene state + a
  multi-step LLM is the current frontier, but also that the
  scene-state quality dominates the planner's prompt-
  engineering quality.

## Out of scope

- **Implementing the chosen architecture.** Code lands in
  `autonomy-stack` and `planner-far-target-staging`.
- **Replacing the LLM backend.** Whether the planner LLM stays
  the current model or moves to a frontier model is unrelated
  to the classifier-vs-multi-step question. Filed separately if
  pursued.
- **VLA / VLM planner unification.** Whether the planner LLM
  should be the same model as the grounding VLM is a much
  larger architectural question; filed under
  `STRAFER_AUTONOMY_NEXT.md`'s long-horizon section if pursued.
