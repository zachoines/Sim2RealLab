# Planner drops the sign of "right" in a two-axis translate

**Status:** Shipped 2026-10-11 in `e568e4a` (DGX).
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
  commands scored mis-sign, and the compiler now sets the sign. See Outcome.)*
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
      - *On their own the examples leave 24 of 48 scored mis-sign, so the shipped fix adds a sign
        guard in the plan compiler (Outcome).*
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
- [x] Existing planner tests still pass.
      *Met 2026-10-11:*
      - *`-m "not requires_ros"`: 692 tests on `main` and 705 on the branch (701 passed, the 4
        live fixtures skipped), with 0 failures and 0 errors.*
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
      - *`source/strafer_autonomy/README.md` gains the compiler's translate sign rule under
        "Deterministic plan compilation" and the `live_planner` opt-in under Testing. Its intent
        list gains `translate`, which it had left out, and so does the top-level `Readme.md`'s.*
      - *[`planner-rotate-direction-prompt`](../active/reliability/planner-rotate-direction-prompt.md)'s
        out-of-scope note on the translate sign carries a dated correction.*
      - *The two links into this brief from the confirmation record and from
        [`executor-policy-nav-budget`](executor-policy-nav-budget.md) point at `completed/`.*

## Outcome

**What shipped.**
- `38ce883`: three two-axis translate examples in the system prompt.
- `e568e4a`: a sign guard in `plan_compiler._compile_translate`.
  - It reads the command's own words: "forward", "forwards" or "ahead" against "backward",
    "backwards" or "back" for x, and "left" against "right" for y.
  - When the command names exactly one direction on an axis, a non-zero component takes that
    sign. Otherwise the LLM's value passes through, and a zero stays zero.
  - It changes signs only, never a magnitude or an axis, and only in `translate` intents.
- `1031643`: the prompt-example tests, the guard's unit tests, and the live fixtures with their
  `live_planner` marker. The fixtures opt in through a variable of their own, not the executor's
  `PLANNER_URL`, so a host that sets the executor's does not run them.

**Deviation: the fix is not in the prompt alone.**
- **Superseded in Context:** "The plan compiler and the executor pass the sign through unchanged
  (…), so the fix is in the prompt." The compiler now sets a translate component's sign when the
  command names exactly one direction on that axis. The executor still passes the sign through.
- **Why.** Measured on the live planner in
  [`planner-translate-sign-2026-10-11`](../../measurements/planner-translate-sign-2026-10-11/README.md),
  against a rule registered before the runs:
  - the examples alone left 24 of 48 two-axis commands scored mis-sign, against 31 before, and
    forward-left's exact count fell from 9 to 8;
  - the examples plus the fallback rule sentence left 22, with two of the five anchors flipped and
    forward-left's exact count at 8;
  - neither met the rule.
- **The examples with the guard** meet it: 0 of 48 mis-signed, 44 exact, all five anchors exact,
  and every class's exact count at or above before. This combination was registered, under the same
  rule, before its first request, and after the first two combinations' results.
  - The zero is by construction. Each of these commands names each non-zero axis with exactly one
    of the guard's words, so the guard rules out a compiled mis-sign there. The runs measure the
    exact counts, the anchors and the dropped axis.
  - The guard changed 26 of the 52 compiled values, every one by sign alone.
  - On 12 held-out commands it compiles 0 mis-signs and 11 exact. The prompt alone mis-signed 2 of
    them.

**Residual.**
- **A dropped axis.** It is not a sign error, and the guard cannot restore a zero.
  - "strafe S Y meters while moving backward X meters" comes back with the backward component at 0
    under every prompt measured, 4 of 48.
  - One held-out command, "head forward 2.25 m, then 0.35 m to the right", comes back as
    `[2.25, 0.0]`.
- **Words, not meaning.** A command that names both directions on an axis keeps the model's sign.
  A direction word outside the guard's lists keeps the model's sign. A "right" or "back" that is not
  a direction still sets the sign of a non-zero component.
- **Coverage.** One model, greedy decoding, and the record's 64 commands. Other wordings are not
  measured.

**The operating rule** in Context is not retired here. The residual above is what that dry-check
still catches on these commands. No brief is filed for the dropped axis in this change.

## Out of scope

- The executor's translate handling (rotate-then-translate, budgets): it is
  sign-clean and dispatched each goal the set submitted to within 1.1 mm.
- Rotation sign: [`planner-rotate-direction-prompt`](../active/reliability/planner-rotate-direction-prompt.md).
