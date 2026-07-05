# Retire `envelope_factor`; promote the sim Nav2 config to the universal baseline

**Type:** task / refactor + process
**Owner:** Jetson agent (`source/strafer_ros/strafer_navigation/`, `strafer_bringup`, `strafer_inference`, `docs/tasks/`)
**Priority:** P2
**Branch:** task/nav2-envelope-retirement

**Status:** Shipped 2026-07-05 (Jetson) — code + tests + docs complete and
green; **merge gated on the operator rig-gate mission below** (the parity
cap has never run on the robot).
**PR:** _(opened from `task/nav2-envelope-retirement`; rig-gate evidence
required in the body before merge)_

Supersedes [`nav2-sim-real-promotion-architecture.md`](nav2-sim-real-promotion-architecture.md)
Layer C. New model: [`context/nav2-config-parity.md`](../context/nav2-config-parity.md).

## Story

As a **roboticist who ships one robot behavior to both sim and real**, I
want **the Nav2 configuration to be identical on both lanes by
construction**, so that **sim validation actually validates the real
config** and the sim-to-real gap can't grow by accident.

## The policy (operator decision, 2026-07-05)

1. **The velocity envelope was a workaround for a misdiagnosed problem.**
   Sim MPPI underperformed → the velocity cap was lifted 2× in sim
   (`STRAFER_NAV_VEL_SCALE=1.0` vs real `0.5`) and critics were tuned to
   chase it (PR #15, which converged at only 63% of its own acceptance
   bar). The real cause was Jetson CPU starvation, since fixed in
   `nav2_params.yaml` (`bt_loop_duration: 50`, `batch_size: 1000`, 20 Hz
   headroom). The envelope concept is retired.
2. **True 1-to-1 parity with the current sim config as the shared
   baseline.** The operator ran a full e2e bridge mission on
   `STRAFER_NAV_BACKEND=nav2` with the current config today — it ran
   perfectly. That configuration (including the four formerly-gated MPPI
   knobs) becomes the universal default for BOTH lanes. We do not strip
   the real robot back to near-stock Nav2 defaults.
3. **Future divergence lifecycle** replaces the promotion-lap contract:
   test in sim → land behind a *temporary* flag on the robot → A/B on the
   robot → universalize the winner and DELETE the flag. Permanent
   lane-gated knobs are no longer a thing. Documented in
   [`context/nav2-config-parity.md`](../context/nav2-config-parity.md).

## What changed

- **Four MPPI knobs universalized into `nav2_params.yaml` as defaults**,
  verbatim from their sim-gated values: `PreferForwardCritic.cost_weight
  10.0`, `PathAlignCritic.cost_weight 9.0`,
  `PathFollowCritic.offset_from_furthest 20`, `gamma 0.008`.
- **Envelope machinery deleted** from `navigation.launch.py`: the
  `_resolved_nav_velocities` env-override function, the `envelope_factor`
  computation, and `_patch_params`' velocity-envelope-coupling (std /
  prune scaling, `vy_std` un-scale) and behavioral-override sections.
  `_patch_params` is now constants-injection only; caps come from
  `NAV_LINEAR_VEL` / `NAV_ANGULAR_VEL` / `NAV_REVERSE_VEL` — one number,
  both lanes.
- **`STRAFER_NAV_VEL_SCALE` fully retired**, not just from Nav2. The
  shared export was removed from `env_sim_in_the_loop.env`, and
  `strafer_inference`'s `_resolve_vel_caps` (which mirrored the same env
  var to clamp the trained policy) now takes the constants cap directly.
  Net parity effect: sim inference now clamps the policy at the indoor cap
  (~0.78 m/s) exactly as real bringup always has — real was never at the
  lifted cap, so this *raises* sim-to-real fidelity. (Operator-decided:
  the "full retirement, touch inference" option.)
- **Tests** (`test_nav_config.py`): promotion-split invariants,
  factor-2.0 golden, and the `STRAFER_NAV_VEL_SCALE` override suite
  deleted; a single golden fixture (`fixtures/patched_params.yaml`)
  regenerated from the shipped config with the byte-identical pin kept
  (its meaning shifts from "the refactor is pure motion" to "the shipped
  baseline is exactly the rig-validated one"); an assertion pins the four
  universalized values in YAML against a silent revert.
- **Docs**: `context/nav2-knob-promotion.md` → `context/nav2-config-parity.md`
  (rewritten around the new lifecycle; inbound links fixed);
  `nav2-sim-real-promotion-architecture` closed out (Layer C superseded);
  `BOARD.md` updated.

## ⛔ Rig gate (operator — before merge)

Re-run **the same e2e bridge nav2 mission that passed today**, now on this
branch (parity caps + universalized knobs). Expected: **well but slow**
(~2× slower legs — the cap halves). Watch for the OLD failure signature
(veer-off-path + pull-back), which would mean a knob interacts badly with
the lower cap. Record in the PR: mission outcome, rough leg timing vs
today's run, any anomaly.

- **Clean →** merge.
- **A knob misbehaves at the parity cap →** tune it *in sim on this
  branch* (the new lifecycle's sim-first step), record before/after,
  regenerate the golden fixture, ship the tuned value as the universal
  default. Do NOT reintroduce a gate.

Also watch: `STRAFER_NAVIGATION_TIMEOUT_S` stays frozen at 180 s, but its
rationale referenced the retired chassis-max sim speed. At the halved
parity cap, legs run ~2× slower, so a sim-time nav timeout during the rig
gate is a value the operator may need to raise toward 600 s (a value
change the rig gate would surface).

## One honest caution (for the next real bringup)

The four universalized values were tuned at the 2× envelope and have
**never run on the real robot**. The first real-robot bringup after this
merges should include the cheap cornering smoke (translate 1 m → rotate
90° → translate 1 m) with eyes on MPPI. Any regression enters the new
lifecycle (sim-first → flag → A/B); it does not resurrect the gate.

## Acceptance

- [x] `grep -rn "envelope_factor\|STRAFER_NAV_VEL_SCALE" source/` → zero
      hits.
- [x] `_patch_params` = constants injection only; the four knobs live in
      YAML; single-golden byte pin green.
- [x] Suites green on the Jetson: `strafer_navigation` 41 passed;
      `strafer_inference` 270 passed / 8 skipped; `strafer_perception`
      goal-projection 36 passed.
- [x] Both lanes byte-identical in Nav2 config by construction (no
      per-lane branch in `_patch_params`).
- [ ] Rig-gate evidence recorded in the PR body (operator).
