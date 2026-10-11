# Planner drops the sign of "right" in a two-axis translate

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

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [measurements/goal-a-cli-confirmation-2026-10-03](../../../measurements/goal-a-cli-confirmation-2026-10-03/README.md)
  — the observation, with the request and the returned plan in its deposit.
- [`planner-rotate-direction-prompt`](planner-rotate-direction-prompt.md) — the
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
  ([`prompt_builder.py`](../../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py))
  states "Right = negative left" and has one right example, single-axis
  (`"strafe right 2 meters"` → `[0.0, -2.0]`). It has no two-axis example
  on either side.
- The plan compiler and the executor pass the sign through unchanged
  (`plan_compiler._compile_translate`; `mission_runner._translate`), so the
  fix is in the prompt.
- The finding is one observation; how often the prompt mis-signs is not
  measured.
- **Operating rule until this ships:** every real-robot submission is
  compiled through the planner first and dispatched only if the returned
  plan matches the intended motion.

## Acceptance criteria

- [ ] Before any prompt change, the mis-sign rate at the prompt of `ed0d5af`
      is measured against the live Qwen3-4B (greedy) over N two-axis
      phrasings: forward and backward × left and right, several magnitudes
      and wordings. The count is recorded, so the post-fix fixtures show the
      fix changed something.
- [ ] The translate section of the system prompt carries two-axis examples on
      both sides (e.g. "move 1.5 meters forward and 0.5 meters right" →
      `[1.5, -0.5]`; "… left" → `[1.5, 0.5]`), and one with a backward
      component.
- [ ] Planner fixtures, run against the live Qwen3-4B
      (`make serve-planner`), pin the sign and the 3-decimal magnitudes for at
      least: forward+right, forward+left, backward+right, and the exact
      command above.
- [ ] Existing planner tests still pass.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- The executor's translate handling (rotate-then-translate, budgets): it is
  sign-clean and dispatched each goal the set submitted to within 1.1 mm.
- Rotation sign: [`planner-rotate-direction-prompt`](planner-rotate-direction-prompt.md).
