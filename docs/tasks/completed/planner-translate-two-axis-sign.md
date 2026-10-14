# Planner drops the sign of "right" in a two-axis translate

**Status:** Shipped 2026-10-13 in `38ce883` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/240

**Type:** bug (planner prompt)
**Owner:** DGX agent
**Priority:** P1 — a translate that names both axes and turns right is
compiled as a move to the left. The executor composes and dispatches it
faithfully, so the robot goes to the mirror-image goal. On the real robot a
language mission that mirrors a lateral command is a safety defect, and the
product path (`make submit-deploy` → planner → executor) has no dry-check:
the one that caught this is harness-only (`tools/plan_check.py` in the
confirmation set's deposit).
*(2026-10-13: the P1 for goal-b transfers to
[`planner-translate-direction-schema`](../active/reliability/planner-translate-direction-schema.md).
See Outcome.)*
**Estimate:** S (prompt examples + planner fixtures)
**Branch:** `task/planner-translate-two-axis-sign`

## Story

As the **executor dispatching a translate the planner compiled**, I want
**"move X meters forward and Y meters right" to compile to
`translation_xy = [X, -Y]`**, so that **a two-axis move to the right goes
right, as the single-axis "strafe right" example already does.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [measurements/goal-a-cli-confirmation-2026-10-03](../../measurements/goal-a-cli-confirmation-2026-10-03/README.md)
  — the observation, with the request and the returned plan in its deposit.
- [`planner-rotate-direction-prompt`](../active/reliability/planner-rotate-direction-prompt.md) — the
  same sign class for rotations. Its out-of-scope note says translate had no
  field reports of mis-direction; this is one.

## Context

- On 2026-10-03, the CLI confirmation set dry-ran each mission's command
  through the planner's `/plan` with the executor's request shape, before
  submitting it.
- For `"move 1.712 meters forward and 1.372 meters right"`, Qwen3-4B (greedy
  decoding, the prompt at `ed0d5af`) returned one `translate` step with
  `dx_m = 1.712`, `dy_m = +1.372`. That is 1.372 m to the left. The deposit
  (`goal-a-cli-confirmation-2026-10-03`) holds the check's output in
  `logs/runs/R1.plancheck.out` (plan `dy_m` 1.372, expected
  `[1.712, -1.372]`) and its failure marker `logs/runs/R1.plancheck_failed`.
- The four two-axis commands to the left in the same set (`… meters left`)
  compiled with the correct sign and magnitudes to 0.001 m.
- The system prompt
  ([`prompt_builder.py`](../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py))
  states "Right = negative left" and has one right example, single-axis
  (`"strafe right 2 meters"` → `[0.0, -2.0]`). It has no two-axis example
  on either side.
- The plan compiler and the executor pass the sign through unchanged
  (`plan_compiler._compile_translate`; `mission_runner._translate`), so the
  fix is in the prompt.
  *(Superseded 2026-10-11: the prompt change alone left 24 of 48 two-axis
  commands scored mis-sign. The compiler guard added for it was withdrawn
  before merge (2026-10-13), and the fix moved to the planner's schema. See
  Outcome.)*
- The finding is one observation; how often the prompt mis-signs is not
  measured.
  *(Measured 2026-10-11: 31 of 48 before the change. See Outcome.)*
- **Operating rule until this ships:** every real-robot submission is
  compiled through the planner first and dispatched only if the returned
  plan matches the intended motion.
  *(2026-10-11: not retired by this ship. See Outcome.)*

## Acceptance criteria

- [x] Before any prompt change, the mis-sign rate at the prompt of `ed0d5af`
      is measured against the live Qwen3-4B (greedy) over N two-axis
      phrasings: forward and backward × left and right, several magnitudes
      and wordings. The count is recorded, so the post-fix fixtures show the
      fix changed something.
      *Met 2026-10-11 ([`planner-translate-sign-2026-10-11`](../../measurements/planner-translate-sign-2026-10-11/README.md)):*
      - *Served from `main` at `9f7d207`, whose `prompt_builder.py` is the same blob as at
        `ed0d5af` (`ae5b1482`). The requests are in the executor's shape, and R1's locally rebuilt
        messages hash to the confirmation deposit's sha256.*
      - *31 of 48 balanced commands mis-signed, and 12 exact: forward and backward × left and
        right, six wordings, two magnitude pairs each. 27 of the 31 flip only the lateral
        component, and 2 more flip both.*
      - *R1 reproduces `[1.712, 1.372]`. The four left-hand commands of the confirmation set are
        exact.*
      - *Two passes, identical request by request. The acceptance rule was committed to the
        evidence repository before the first request.*
- [x] The translate section of the system prompt carries two-axis examples on
      both sides (e.g. "move 1.5 meters forward and 0.5 meters right" →
      `[1.5, -0.5]`; "… left" → `[1.5, 0.5]`), and one with a backward
      component.
      *Met 2026-10-11 in `38ce883`:*
      - *"move 1.5 meters forward and 0.5 meters right" → `[1.5, -0.5]`, "… left" →
        `[1.5, 0.5]`, and "move 0.8 meters backward and 0.3 meters right" → `[-0.8, -0.3]`.*
      - *`TestTranslateExamples` (`1031643`) checks each translate example's signs against its
        words, and that the two-axis examples cover both sides and a backward component. It fails
        on `main`'s prompt and on a copy with one example's sign flipped (the deposit's
        `tests/mutation_runs.txt`; the record's §7).*
      - *On their own the examples leave 24 of 48 scored mis-sign. A sign guard in the plan
        compiler was added for it, then withdrawn before merge; the fix moved to the planner's
        schema (Outcome).*
- [x] Planner fixtures, run against the live Qwen3-4B
      (`make serve-planner`), pin the sign and the 3-decimal magnitudes for at
      least: forward+right, forward+left, backward+right, and the exact
      command above.
      *Met 2026-10-11 in `1031643`:*
      - *`source/strafer_autonomy/tests/test_planner_live_translate.py`, marker `live_planner`.
        It runs when `STRAFER_LIVE_PLANNER_URL` names a planner, and skips otherwise. Once opted
        in, its cases error at setup unless that planner reports its model loaded.*
      - *Four cases at 1.712 m and 1.372 m, each component to 1e-3: forward+right (the exact
        command above), forward+left, backward+right and backward+left.*
      - *Against the branch's planner all four pass. Against `main`'s, forward+right,
        backward+right and backward+left fail, and forward+left passes.*
      - *2026-10-13: with the guard withdrawn, backward-left fails against the shipped planner,
        `(-1.712, -1.372)`, and is a strict `xfail` naming the schema brief; the other three
        pass (the record's §7).*
- [x] Existing planner tests still pass.
      *Met 2026-10-11:*
      - *`-m "not requires_ros"`: 692 tests on `main` and 705 on the branch (701 passed, the 4
        live fixtures skipped), with 0 failures and 0 errors.*
      - *2026-10-13, at `b3be9b0` and again on the shipped tree, `2841f74`: 698 tests, 694 passed
        and the 4 live fixtures skipped, 0 failures and 0 errors.*
      - *Both runs leave out `source/strafer_autonomy/tests/test_sim_in_the_loop_runtime_env.py`.
        It fails to collect in `.venv_vlm` on `main` as on the branch, with
        `ModuleNotFoundError: No module named 'isaaclab_tasks'` through `strafer_lab`. This work
        does not touch it.*
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
      *Met 2026-10-11:*
      - *`source/strafer_autonomy/README.md` gains a sentence on the translate sign under
        "Deterministic plan compilation" (the compiler passes the model's values through; the
        guard's rule was removed with the guard, 2026-10-13) and the `live_planner` opt-in under
        Testing. Its intent list gains `translate`, which it had left out, and so does the
        top-level `Readme.md`'s.*
      - *[`planner-rotate-direction-prompt`](../active/reliability/planner-rotate-direction-prompt.md)'s
        out-of-scope note on the translate sign carries a dated correction.*
      - *The two links into this brief from the confirmation record and from
        [`executor-policy-nav-budget`](executor-policy-nav-budget.md) point at `completed/`.*

## Outcome

**What shipped.** The prompt examples, their tests, the live fixtures and the record. The fix itself
moved to [`planner-translate-direction-schema`](../active/reliability/planner-translate-direction-schema.md),
which takes this brief's P1 for goal-b.
- `38ce883`: three two-axis translate examples in the system prompt.
- `1031643`: the prompt-example tests, and the live fixtures with their `live_planner` marker. The
  fixtures opt in through a variable of their own, not the executor's `PLANNER_URL`, so a host that
  sets the executor's does not run them.
- `b3be9b0`: the backward-left fixture becomes a strict `xfail` naming the schema brief. It fails
  against the shipped planner today, and turns red the day it passes, so the marker must come off
  with the fix.
- `2841f74`: the marker expects only an `AssertionError`, and the fixture fails rather than
  asserts on an unready planner, so an unreachable or unready planner errors instead of counting as
  the expected failure.

**Not shipped: the compiler guard.** It was written, narrowed, and withdrawn before merge.
- `e568e4a` added a sign guard to `plan_compiler._compile_translate` that read the command's
  direction words. `14477af` narrowed it to the two-axis template CLI-built commands use, after it
  was shown to flip a correct sign where a direction word names a landmark's side.
- `d40cdca` withdrew it on review: a regex over the operator's phrasing is brittle, and opaque to
  an operator who does not know the protected phrasing. `b3be9b0` dropped its tests. The compiler
  passes the model's values through, as on `main`.
- The measurements stand as measurements (the record's §5, §5b and §5c). With the examples and the
  first guard the sweep read 0 of 48 mis-signed, 44 exact and all five anchors exact. That zero was
  by construction, since each command names each non-zero axis with exactly one of the guard's
  words. The narrowed guard read 22 of 48 on the logged answers.

**Deviation: the fix is in neither the prompt nor the compiler.**
- **Superseded in Context:** "The plan compiler and the executor pass the sign through unchanged
  (…), so the fix is in the prompt." Both still pass the sign through. The fix is not in the
  prompt either: it is in the schema the planner answers in.
- **Why.** Measured on the live planner in
  [`planner-translate-sign-2026-10-11`](../../measurements/planner-translate-sign-2026-10-11/README.md),
  against a rule registered before the runs:
  - the examples alone left 24 of 48 two-axis commands scored mis-sign, against 31 before, and
    forward-left's exact count fell from 9 to 8;
  - the examples plus the fallback rule sentence left 22, with two of the five anchors flipped and
    forward-left's exact count at 8;
  - neither met the rule. The prompt states the convention in every arm, and the model did not
    apply it reliably in any of the three prompts measured (a backward-left example was not
    tried). `translation_xy` asks it to.
- **The shipped tree, live** (served from `d40cdca`, whose package source the shipped tree,
  `2841f74`, keeps; the record's §5c): the 52 sweep plans equal the examples-only arm's, request
  by request; 24 of 48 mis-signed, 20 exact, all five anchors exact. Held out: 2 of 12 mis-signed
  (H2 and H8, both backward and left), H3's dropped axis, 9 exact.
  The held-out vocabulary lies entirely inside the first guard's word lists, so that set is evidence
  about the prompt, not about the guard.

**Residual, now owned by the schema brief.**
- **The mis-signs:** 24 of 48 and 2 of 12 on the record's commands, mostly the lateral component.
- **A dropped axis.** "strafe S Y meters while moving backward X meters" comes back with the
  backward component at 0 under every prompt measured (4 of 48), and "head forward 2.25 m, then
  0.35 m to the right" with the lateral one at 0. `planner-translate-dropped-axis`, filed for it on
  2026-10-12, is folded into the schema brief as a section.
- **Coverage.** One model, greedy decoding, and the record's 64 commands. Other wordings are not
  measured.

**The operating rule** in Context stays. Every real-robot submission is compiled through the
planner first and dispatched only if the plan matches the intended motion. It stays until a record
of the schema change reads 0 mis-signs and 0 dropped axes on the 48 and the 12.

## Out of scope

- The executor's translate handling (rotate-then-translate, budgets): it is
  sign-clean and dispatched each goal the set submitted to within 1.1 mm.
- Rotation sign: [`planner-rotate-direction-prompt`](../active/reliability/planner-rotate-direction-prompt.md).
