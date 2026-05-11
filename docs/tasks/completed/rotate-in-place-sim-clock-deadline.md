# `rotate_in_place` enforces its deadline on wall-clock, not sim-clock

**Status:** Shipped 2026-05-10 in `4f63699` (Jetson).
**PR:** _pending — populated when `gh pr create` runs._

**Type:** task / bug
**Owner:** Jetson agent
**Priority:** P1 — bumped from P2 after `progress-aware-nav-timeouts`
(PR #21) tightened the per-rotation budgets passed in. The wall-clock
mismatch now fails any rotation skill in single-digit sim-seconds at
sub-unity RTF, where pre-PR-#21 it had ~30 s headroom that masked the
bug for many missions. See "Context" below.
**Estimate:** S (~hours; small change + test)
**Branch:** task/rotate-in-place-sim-clock-deadline

## Story

As a **mission operator running sim-in-the-loop missions where sub-unity
RTF is the norm**, I want **the executor's `rotate_in_place` deadline
to count sim time, not wall time**, so that **rotates get the same
sim-time budget the executor's navigate path already gets, instead of
timing out after a wall-second budget that maps to a small fraction of
a sim-second at low RTF**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
  — "Sim-time-aware navigation timeout (Jetson side)" section
  documents the convention this brief makes `rotate_in_place`
  comply with.
- [completed/sim-velocity-attenuation.md](../completed/sim-velocity-attenuation.md)
  — surfaced this during rotation-timeout debugging.
- [completed/progress-aware-nav-timeouts.md](../completed/progress-aware-nav-timeouts.md)
  — immediate predecessor. Switched `_rotate_by_degrees` and
  `_orient_to_direction` from a flat 30 s timeout to a per-step budget
  derived from `|yaw_delta_rad| / NAV_ANGULAR_VEL * safety_factor +
  setup_overhead_s`. The new budget is correct *as a sim-time interval*
  but `rotate_in_place` enforces it as wall time, so the mismatch is
  now strictly worse than before this brief was filed.

## Context

[`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:1149`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
sets `deadline = time.monotonic() + timeout` and the loop at line
1150 checks `time.monotonic() < deadline`. That's wall clock, not
sim clock.

The bridge invariants module's "Sim-time-aware navigation timeout"
section documents the project-wide convention:

> The executor's nav-timeout enforcement uses `node.get_clock().now()`,
> which respects `use_sim_time`. On the sim-in-the-loop bringup launch
> (`use_sim_time:=true`), 90 s of sim-time wait is 90 s of `/clock`
> advance — not 90 s of wall clock.

That convention was put in place in `f60456e` for `navigate_to_pose`,
extended to all motion skills in PR #21 (`progress-aware-nav-timeouts`).
`rotate_in_place` is the lone holdout — it uses Python's monotonic
wall clock for both deadline computation and per-iteration
`sleep(0.02)`, with no `use_sim_time` awareness.

**Why the bug got worse after PR #21.** Pre-PR-#21,
`_rotate_by_degrees` passed a flat 30 s timeout into `rotate_in_place`.
At RTF=0.1, that wall-second budget mapped to 3 sim-seconds — tight
for a 90° rotation but sometimes survivable. Post-PR-#21, the same
call computes `|yaw_delta_rad| / NAV_ANGULAR_VEL * safety_factor +
setup_overhead_s`, which gives much smaller budgets *intended as
sim-seconds*:

| Rotation | New computed budget | Wall enforcement at RTF=0.1 |
|---|---|---|
| 30°  | ~5.4 s | 0.54 sim-s ❌ |
| 90°  | ~6.5 s | 0.65 sim-s ❌ |
| 180° | ~8.0 s | 0.80 sim-s ❌ |
| 360° | ~13.0 s | 1.30 sim-s ❌ |

At any sub-unity RTF, the chassis cannot complete the rotation before
the wall deadline trips. The progress-aware budget formula and the
sim-clock deadline source are two halves of the same correctness
story — neither is complete without the other. The brief estimates
the same fix as before (mirror `f60456e`'s pattern), only the urgency
has changed.

Pre-PR-#21 operator observation in `sim-velocity-attenuation`:
rotates tripped 30 s wall budget at RTF 0.07–0.5, mapping to 2–15 s
sim. That observation no longer applies post-PR-#21 — the budgets
are smaller and the failures faster.

## Approach

Mirror the navigation-side fix:

1. Read deadline as a sim-time interval via `node.get_clock().now()`
   + a `Duration(seconds=timeout)` for the deadline check, so that
   `use_sim_time:=true` makes the deadline count `/clock` advances.
2. Per-iteration sleep should remain `time.sleep(0.02)` — that's
   the *responsiveness* of the loop, not its budget. Wall-clock is
   correct for sleeping (the OS doesn't have a sim-clock sleep).
3. Real-robot bringup leaves `use_sim_time=False`, so
   `node.get_clock().now()` returns wall clock natively — same
   semantics as today, no regression.

## Acceptance criteria

- [ ] `rotate_in_place` deadline enforcement uses
      `node.get_clock().now()` (sim-clock-aware) for the deadline
      comparison, matching the navigation path's convention.
- [ ] Sim repro: with `use_sim_time:=true` and the bridge running at
      sub-unity RTF (target RTF ≈ 0.1–0.5), a `rotate_by_degrees 90`
      mission with the post-PR-#21 computed budget (~6.5 s) completes
      inside its sim-time budget instead of tripping wall-clock at
      low RTF. Compute the expected sim-time budget via
      `MissionRunner._motion_timeout_s` rather than hardcoding 30 s
      — the value will drift with `nav_budget_safety_factor` /
      `nav_budget_setup_overhead_s` tuning. Record before / after in
      the PR description.
- [ ] Real-robot bringup is unaffected: `use_sim_time=False` →
      `node.get_clock().now()` returns wall clock → identical
      semantics to today.
- [ ] Unit test or stub exercising the deadline path under both
      `use_sim_time=True` and `=False` clock sources, asserting the
      deadline is computed from the right source.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit (the
      `bridge-runtime-invariants.md` "Sim-time-aware navigation
      timeout" section's claim that the convention is project-wide
      becomes literally true rather than aspirational; PR #21 already
      tightened that section but the `rotate_in_place` holdout is
      still implicit).

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:1109-1180`
  — the full `rotate_in_place` body, including the deadline
  computation at line 1149 and the loop check at line 1150. (Line
  numbers shifted from the original 720/721 because PR #21 added
  `_ProgressTracker` and `_wait_for_nav_result` above.)
- `f60456e` — the navigation-side commit that introduced the
  sim-clock convention; mirror its shape. PR #21
  (`progress-aware-nav-timeouts`) extended the same pattern to all
  motion budgets — the deadline source change here completes the
  picture.
- `STRAFER_ROTATE_TIMEOUT_S` env knob (default 60.0 in
  `env_sim_in_the_loop.env`) feeds
  `self._config.default_rotate_timeout_s`. Post-PR-#21, this knob's
  scope is narrower: `_rotate_by_degrees` and `_orient_to_direction`
  now compute their own budgets via `MissionRunner._motion_timeout_s`
  and pass them explicitly. The env knob still gates the inter-heading
  rotations inside `scan_for_target` ([mission_runner.py:733](../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L733)),
  which calls `rotate_in_place` without `timeout_s`. Once this brief
  lands, the env-knob value becomes a sim-time budget for that path
  too.

## Out of scope

- **Rotation direction-sign bug.** Different brief —
  [`planner-rotate-direction-prompt.md`](planner-rotate-direction-prompt.md).
  That's a planner-prompt issue; this brief is purely about the
  deadline source.
- **Plan-compiler-side timeout hardcodes.** Shipped — see
  [`completed/plan-compiler-skill-timeouts.md`](../completed/plan-compiler-skill-timeouts.md)
  (PR #20). That removed the compiler's hardcoded `timeout_s` so the
  executor's per-skill policy could take effect.
- **Per-step budget computation.** Shipped — see
  [`completed/progress-aware-nav-timeouts.md`](../completed/progress-aware-nav-timeouts.md)
  (PR #21). That added the distance/angle-derived budget formula and
  the Nav2 stall watchdog. The `_rotate_by_degrees` and
  `_orient_to_direction` paths get sim-time-correct *budget values*
  from PR #21; this brief gives them sim-time-correct *enforcement*
  inside `rotate_in_place`.
- **Convergence-tolerance / physics-bouncing.** If chassis bounce in
  sim makes the yaw oscillate inside `tolerance_rad=0.1`, this brief
  doesn't fix that. File separately if the bounce is the actual
  blocker once the sim-clock fix is in.
