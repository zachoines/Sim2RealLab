# `rotate_in_place` enforces its deadline on wall-clock, not sim-clock

**Type:** task / bug
**Owner:** Jetson agent
**Priority:** P2
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
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)
  — "Sim-time-aware navigation timeout (Jetson side)" section
  documents the convention this brief makes `rotate_in_place`
  comply with.
- [completed/sim-velocity-attenuation.md](completed/sim-velocity-attenuation.md)
  — surfaced this during rotation-timeout debugging.

## Context

[`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:720`](../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
sets `deadline = time.monotonic() + timeout` and the loop checks
`time.monotonic() < deadline`. That's wall clock, not sim clock.

The bridge invariants module's "Sim-time-aware navigation timeout"
section documents the project-wide convention:

> The executor's nav-timeout enforcement uses `node.get_clock().now()`,
> which respects `use_sim_time`. On the sim-in-the-loop bringup launch
> (`use_sim_time:=true`), 90 s of sim-time wait is 90 s of `/clock`
> advance — not 90 s of wall clock.

That convention was put in place in `f60456e` for the navigation
path (`navigate_to_pose`). `rotate_in_place` is the lone holdout —
it uses Python's monotonic wall clock for both deadline computation
and per-iteration `sleep(0.02)`, with no `use_sim_time` awareness.

Operator-observed in the velocity-attenuation predecessor: rotates
trip their 30 s wall budget before the chassis can settle inside
`tolerance_rad=0.1`. With sim RTF in the 0.07–0.5 range typical for
the Infinigen scenes, the wall-clock deadline maps to 2–15 s of sim
time — far less than the 30 s the operator (and the bridge invariant)
expects.

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
      sub-unity RTF, a `rotate_by_degrees 90` mission with the
      executor's default 30 s timeout completes inside its sim-time
      budget instead of tripping wall-clock at low RTF (record before
      / after in the PR description).
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
      becomes literally true rather than aspirational).

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:680-751`
  — the full `rotate_in_place` body, including the deadline
  computation at line 720 and the loop check at 721.
- `f60456e` — the navigation-side commit that introduced the
  sim-clock convention; mirror its shape.
- `STRAFER_ROTATE_TIMEOUT_S` env knob (default 60.0 in
  `env_sim_in_the_loop.env`) feeds
  `self._config.default_rotate_timeout_s`. Once this brief lands,
  60 s sim-time becomes the actual budget, which should be more
  than sufficient.

## Out of scope

- **Rotation direction-sign bug.** Different brief —
  [`planner-rotate-direction-prompt.md`](planner-rotate-direction-prompt.md).
  That's a planner-prompt issue; this brief is purely about the
  deadline source.
- **Plan-compiler-side timeout hardcodes.** Different brief —
  [`plan-compiler-skill-timeouts.md`](plan-compiler-skill-timeouts.md).
  That's about the *value* of the timeout the executor receives;
  this is about *how* the executor enforces it.
- **Convergence-tolerance / physics-bouncing.** If chassis bounce in
  sim makes the yaw oscillate inside `tolerance_rad=0.1`, this brief
  doesn't fix that. File separately if the bounce is the actual
  blocker once the sim-clock fix is in.
