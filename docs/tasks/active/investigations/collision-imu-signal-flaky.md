# `test_collision_imu_mean_differs_from_free` flakes under restitution-0 collision physics

**Type:** investigation (test reliability / physics signal)
**Owner:** DGX (lane: `source/strafer_lab/test_sim/sensors/`, strafer chassis physics cfg)
**Priority:** P3 — doesn't block features, but it makes `run_tests.py all` a non-deterministic gate.
**Estimate:** S–M (measure the signal margin, then a one-line scenario/assertion decision).
**Branch:** task/collision-imu-signal-flaky

## Story

As **an agent running the Kit suite**, I want
**`test_collision_imu_mean_differs_from_free` to pass deterministically**,
so that **`run_tests.py all` (and the CI gate the
`unify-test-targets-and-ci` brief wraps around it) is a reliable green,
not a coin flip.**

## Context (measured)

`test_sim/sensors/test_imu_collision.py::test_collision_imu_mean_differs_from_free`
asserts that the IMU linear-acceleration distribution **during a sustained
collision** is statistically distinguishable (Welch t-test at
`CONFIDENCE_LEVEL`) from the free-motion distribution.

It is **flaky in isolation** — the exact same `run_tests.py imu` invocation
passed one run and failed the next:

```
FAIL test_collision_imu_mean_differs_from_free
  AssertionError: IMU distribution not significantly different during
  collision. Free mean: 15.55 m/s², Collision mean: 15.xx
```

The collision mean (~15 m/s²) now sits barely above the free-motion gravity
baseline (~15.5–15.8 m/s²). The most likely cause is the deliberately
**gentler contact physics** introduced to kill the high-yaw mecanum-roller
chassis bounce (PGS solver + `enable_stabilization` + restitution-0 roller
USD, shipped in `roller-contact-high-omega-bounce.md` /
`teleop-perf-architecture.md`): with restitution clamped to 0 the collision
no longer produces an acceleration spike that clears the significance bar, so
the test rides the noise floor.

This was **masked** at `0/0` in `run_tests.py all` until the
`strafer-lab-test-tree-unification` PR corrected the stale `from test.*`
imports — the test now actually collects, and the pre-existing flakiness
became visible. The flake is **not** caused by that reorg (the only change
to the file there is the import line); it ships on `main` too.

## The decision (the investigation)

Measure the collision-vs-free acceleration margin under current physics, then
pick one:

1. **Strengthen the collision scenario** so the IMU signal reliably clears
   the significance bar — higher approach speed / harder contact geometry /
   more settle+sample steps / larger `NUM_ENVS`.
2. **Re-frame the assertion** onto a signal that is robust under
   restitution-0 — e.g. assert on contact force or the sustained-collision
   counter rather than the IMU acceleration mean, keeping the sim↔real
   intent (a collision is detectable) without depending on a bounce.
3. **Retire this specific assertion** if restitution-0 genuinely makes the
   IMU mean indistinguishable, keeping the other three `test_imu_collision`
   tests that do not depend on it.

## Acceptance

- [ ] `run_tests.py imu` passes 4/4 across ≥5 consecutive runs (no flake).
- [ ] The sim↔real "a collision is detectable" contract is still guarded by
      a deterministic assertion — coverage is re-framed, not silently
      deleted.
- [ ] If physics is touched to strengthen the signal, the high-yaw
      mecanum-roller bounce fix does not regress.

## Out of scope

- The test-tree layout — shipped in `strafer-lab-test-tree-unification.md`.
- The restitution-0 / PGS chassis-physics tuning itself; this brief adapts
  the *test* to that physics, it does not re-open it.
