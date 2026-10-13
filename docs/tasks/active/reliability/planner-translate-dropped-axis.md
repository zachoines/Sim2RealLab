# Planner drops one axis of a two-axis translate

**Type:** bug (planner)
**Owner:** DGX
**Priority:** P2. A command that names two axes can compile to a motion along one of them, which is a
different motion from the one asked for. It is not mirrored, so it is P2 rather than P1. The
compile-first dry-check for real-robot submissions catches it, and that check stays until this class
is measured or fixed.
**Estimate:** S–M (a sweep before and after, a prompt or compiler change, fixtures)
**Branch:** `task/planner-translate-dropped-axis`

## Story

As **the executor dispatching a translate the planner compiled**, I want **a command that names a
movement on both axes to compile to a translate on both axes**, so that **a "while" or "then" clause
is not silently left out of the motion.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [`planner-translate-sign-2026-10-11`](../../../measurements/planner-translate-sign-2026-10-11/README.md)
  §3–§6: the sweep, the outcomes, and the held-out set this class was found in.
- [`planner-translate-two-axis-sign`](../../completed/planner-translate-two-axis-sign.md): the
  sign fix that shipped, and its template-only compiler guard.

## Context

- **The four backward commands.** In the sign record's sweep, Qwen3-4B at greedy decoding returned
  `dx = 0` on all four "strafe S Y meters while moving backward X meters" commands. It did so in
  every arm: before the prompt change, with the two-axis examples, with the extra rule sentence,
  and with the first sign guard.
  - The four:
    - "strafe right 3.533 meters while moving backward 0.032 meters";
    - "strafe right 0.9 meters while moving backward 0.6 meters";
    - "strafe left 3.533 meters while moving backward 0.032 meters";
    - "strafe left 0.9 meters while moving backward 0.6 meters".
  - The forward forms of the same wording keep `dx`.
- **H3.** In the held-out set, "head forward 2.25 m, then 0.35 m to the right" compiled to
  `[2.25, 0.0]`: the lateral component was dropped.
- **The sign guard cannot help.** It changes signs only, and a zero stays zero.
- **A wrong sign on top.** The shipped guard reads only the two-axis template, which these commands
  are not in. So under the shipped code the two "strafe left … while moving backward …" commands
  also keep the prompt's wrong lateral sign: `[0.0, -3.533]` and `[0.0, -0.9]`, a pure strafe to the
  right (`narrowed_guard_eval.txt` in the record's deposit).
- **The counts.** 4 of the sweep's 48 balanced commands and 1 of its 12 held-out commands. The rate
  over other wordings is not measured.

## Acceptance criteria

- [ ] **Before the change,** the dropped-axis rate is measured against the live Qwen3-4B (greedy)
      at the current prompt. Use a sweep of two-axis commands whose second axis comes in a second
      clause ("while moving …", "then …", lateral first), with both directions on each axis and
      several magnitudes. Use the executor's request shape and the sign record's scoring tool. The
      command list and the scoring rule are committed before the first request.
- [ ] **One of these changes** makes each command either compile to both axes or be refused:
  - a prompt change; or
  - a compiler check that refuses a translate whose command names a distance on both axes but
    whose plan moves along one (a 422 the operator can rephrase, instead of a silent one-axis
    motion).

  The change is measured with the same sweep after it, and both rates are recorded.
- [ ] **Tests.** Live planner fixtures (opt-in with `STRAFER_LIVE_PLANNER_URL`, as
      `test_planner_live_translate.py`) pin the four commands above and H3's wording, and a unit
      test covers any compiler check.
- [ ] **Regression.** The existing planner tests pass.
- [ ] **The dry-check.** The compile-first dry-check for real-robot submissions is named as still
      needed, or as no longer needed for this class, with the measurement that shows it.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- The sign of a two-axis translate:
  [`planner-translate-two-axis-sign`](../../completed/planner-translate-two-axis-sign.md).
- Rotation: [`planner-rotate-direction-prompt`](planner-rotate-direction-prompt.md).
- The executor's translate handling. It dispatches the components it is given.
