# Honor `STRAFER_NAVIGATION_TIMEOUT_S` in `plan_compiler` skill steps

**Status:** Shipped 2026-05-09 in `1ae9ffc` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/20
**Follow-ups:** [`progress-aware-nav-timeouts.md`](progress-aware-nav-timeouts.md) — per-step distance/angle-derived budgets + Nav2 stall watchdog (replaces the env-knob backstop with work-proportional deadlines).

**Type:** task / bug
**Owner:** Either (planner module is DGX-lane; executor consumer is
Jetson-lane — see Out of scope below for the lane decision)
**Priority:** P2
**Estimate:** S (~hours; small change + tests)
**Branch:** task/plan-compiler-skill-timeouts

## Story

As a **mission operator running sim-in-the-loop missions**, I want
**translate / navigate_to_pose / etc. skill steps to honor the
`STRAFER_NAVIGATION_TIMEOUT_S` env knob (180 s in
`env_sim_in_the_loop.env`)**, so that **slow-RTF sim missions don't
time out at the planner-side hardcode of 60–90 s while the executor
believes it has 180 s of sim time available**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
  — "Sim-time-aware navigation timeout (Jetson side)" section.
- [completed/sim-velocity-attenuation.md](../completed/sim-velocity-attenuation.md)
  — the predecessor that surfaced this; its bisection runs all timed
  out at the planner-side 60 s cap before the executor's
  `STRAFER_NAVIGATION_TIMEOUT_S=180` could take effect.

## Context

[`source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py`](../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py)
hardcodes per-skill timeouts on the emitted `SkillCall` objects:

| Line | Skill | Hardcoded `timeout_s` |
|------|-------|-----------------------|
| 56   | `wait`                 | 10.0 |
| 73   | `rotate_in_place`      | 60.0 |
| 80   | `look_for_target`      | 2.0  |
| 87   | `navigate_to_pose`     | 90.0 |
| 123  | (varies)               | 5.0  |
| 135  | (varies)               | 5.0  |
| 158  | `translate`            | 60.0 |
| 180  | `rotate_by_degrees`    | 30.0 |
| 192  | (varies)               | 30.0 |
| 221  | (varies)               | 30.0 |
| 233  | (varies)               | 5.0  |

The executor falls back to `default_navigation_timeout_s` (sourced
from `STRAFER_NAVIGATION_TIMEOUT_S`, default 90 s, bumped to 180 s in
`env_sim_in_the_loop.env`) **only when `step.timeout_s` is None or
zero**. Because the compiler always sets a non-zero value, the env
knob is silently overridden for navigate / translate / rotate steps.

The original `sim-velocity-attenuation.md` brief flagged the
`navigate_to_pose=90.0` case at line 87 as an explicit follow-up.
This brief generalizes that to the rest of the navigation-flavored
skill emissions — the same diagnosis applies to `translate=60.0`
(line 158) and `rotate_by_degrees=30.0` (line 180), which were the
ones the velocity-attenuation bisection actually tripped over.

Concrete repro from 2026-04-29: with `STRAFER_NAV_VEL_SCALE=1.0` and
`STRAFER_NAVIGATION_TIMEOUT_S=180`, a `translate forward 3 meters`
mission timed out with `state=timeout`,
`message="Navigation timed out after 60s."`, `elapsed_s=125 wall` —
the planner emitted `timeout_s=60.0` (line 158), executor enforced
that as a sim-time deadline, mission failed before the env-knob
budget was exhausted.

## Approach

Two reasonable shapes:

1. **Drop the navigation hardcodes; let the executor decide.** Pass
   `timeout_s=None` (or omit) on `navigate_to_pose`, `translate`, and
   `rotate_by_degrees` SkillCall emissions. The executor already
   defaults these via `STRAFER_NAVIGATION_TIMEOUT_S` /
   `STRAFER_ROTATE_TIMEOUT_S`. Non-navigation skills (`wait`,
   `look_for_target`) keep their compiler-side defaults — those are
   action-specific budgets, not chassis-motion budgets.

2. **Make the compiler env-aware.** Read the env vars in
   `plan_compiler` and use them instead of literals. More invasive
   (planner gains a runtime config dependency) but explicit. The
   first option is preferred — it keeps `plan_compiler` pure and
   centralizes timeout policy in the executor.

## Acceptance criteria

- [ ] `navigate_to_pose`, `translate`, and `rotate_by_degrees`
      SkillCalls emitted by `plan_compiler` no longer carry
      compiler-side hardcoded `timeout_s`. The executor's existing
      `STRAFER_NAVIGATION_TIMEOUT_S` / `STRAFER_ROTATE_TIMEOUT_S`
      defaults take effect.
- [ ] Sim-in-the-loop translate/rotate/navigate missions honor the
      `env_sim_in_the_loop.env` 180 s budget. Reproduce with a
      `translate forward 3 meters` mission and confirm the executor
      doesn't time out before sim has had a fair budget at the
      current RTF.
- [ ] No regression on real-robot or short-budget skills: `wait`,
      `look_for_target`, and the other non-motion skills keep their
      compiler-side defaults.
- [ ] Unit tests cover: each affected skill emission no longer sets
      `timeout_s`, and the executor's fallback path picks up the
      env-knob value.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py:56-233`
  — all timeout hardcodes listed in the table above.
- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`
  — `_translate`, `_rotate_by_degrees`, `_navigate_to_pose` use
  `step.timeout_s or self._config.default_navigation_timeout_s`.
- `source/strafer_autonomy/strafer_autonomy/executor/main.py:18`
  — env-knob plumbing for `default_navigation_timeout_s`.
- The translate path was the trigger but the pattern is identical
  across all motion skills emitted by `plan_compiler`.

## Out of scope

- **Lane.** `plan_compiler.py` lives under
  `source/strafer_autonomy/strafer_autonomy/planner/`, which is
  DGX-lane per [`context/ownership-boundaries.md`](../context/ownership-boundaries.md).
  The change is small and self-contained; the executor side already
  works correctly. Pick this up DGX-side.
- **Re-tuning the executor's default timeouts.** The 90 s real /
  180 s sim values are existing, separately-justified knobs. This
  brief just makes the compiler stop overriding them.
- **Tightening MPPI / Nav2 to actually finish translate inside the
  envelope.** That's the
  [`completed/sim-velocity-attenuation.md`](../completed/sim-velocity-attenuation.md)
  predecessor's territory; the linear MPPI std-scaling shipped there
  is the v1 fix and a separate brief can deepen the tuning if
  end-to-end mission reliability still needs more headroom.
