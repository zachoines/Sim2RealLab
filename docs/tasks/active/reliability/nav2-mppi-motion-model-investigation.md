# Investigate MPPI motion model: Omni vs DiffDrive for the Strafer chassis

## Why

The MPPI controller is configured with `motion_model: "Omni"` to
match the mecanum chassis's holonomic kinematics. During the lap
testing of
[nav2-scan-ground-filter-and-mppi-mecanum-tuning](nav2-scan-ground-filter-and-mppi-mecanum-tuning.md)
the controller exhibited a robust set of behaviors that suggest the
3-DoF (vx, vy, wz) sampling space is causing trouble even when MPPI
is selecting a correct optimal trajectory:

- A `PreferForwardCritic` bisection (0 / 5 / 10) showed the chassis
  needs `cost_weight = 10` (2× Nav2's diff-drive-tuned default) just
  to prevent MPPI from preferring tangent-to-path or reverse-along-path
  trajectories that the Omni sampler can produce.
- Even at `cost_weight = 10` a residual veer-off-and-pull-back
  pattern remains on straight paths.
- The most striking signal: with MPPI visualization enabled, the
  operator observed `/optimal_trajectory` correctly heading back to
  `/plan` while the chassis was rotating 90° off-path or drifting
  backward. The gap is between MPPI's commanded velocity and what
  the wheels actually execute — downstream of MPPI tuning entirely.

This points at one of:
1. The Omni sampling subspace introduces velocity transients (vy
   commands the chassis can't physically deliver smoothly on lossy
   mecanum) that produce the observed drift.
2. The Isaac Sim mecanum dynamics model has limitations the
   real-robot doesn't share (slip, wheel-controller artifacts).
3. A frame mismatch between MPPI's commanded `/cmd_vel` and the
   chassis's interpretation of it.

A DiffDrive motion model collapses the sampling space to (vx, wz),
removing the vy degree of freedom entirely. Mecanum chassis can be
driven as diff-drive — they lose lateral motion as a Nav2 controller
primitive, but the [executor-prefer-rotate-then-translate](../../completed/executor-prefer-rotate-then-translate.md)
work already decomposes diagonal translates at the executor layer,
so the higher-level cardinal-strafe path is preserved.

## What

### A. Bench-measure both motion models on the same lap

Run an identical mission (e.g. `strafer-autonomy-cli submit "go to
the couch"` from a fixed start pose) under:

- Current: `motion_model: "Omni"` (3-DoF sampling)
- Alternative: `motion_model: "DiffDrive"` (2-DoF sampling)

Capture per-run:
- Path-tracking error (max lateral deviation from `/plan`)
- vy command magnitude over time (Omni should use it; DiffDrive zeros it)
- Mission success / progress-checker timeout count
- Foxglove visual: does the chassis follow `/optimal_trajectory`?

### B. Decide motion model going forward

Three possible outcomes:

1. **Omni is correct, sim is the issue.** Tracking is fine on real
   hardware; sim's mecanum dynamics have a bug. → keep Omni, add a
   sim-specific note. Don't try to fix sim.
2. **DiffDrive tracks better in both sim and real.** The Omni
   sampling space was producing fundamentally bad commands. →
   switch to DiffDrive on both lanes; rely on the executor's
   rotate-then-translate decomposition for diagonal motion;
   document that cardinal-strafe is now an executor concern, not a
   controller concern.
3. **Mixed.** Some missions benefit from Omni's lateral capability
   (e.g. tight-corridor sidestepping), others suffer. → may need a
   per-mission motion-model selector or just pick the most common
   case.

### C. If DiffDrive wins, follow-on changes

- `motion_model: "DiffDrive"` in
  [nav2_params.yaml](../../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml).
- Restore `PreferForwardCritic` to Nav2 default 5.0 (the elevated 10
  was compensating for Omni vy/reverse exploration; DiffDrive
  doesn't need it).
- Audit `TwirlingCritic` (currently 10.0 — DiffDrive-tuned default).
- Revisit `vy_max` in the velocity smoother — non-zero on Omni for
  manual strafe commands; could keep for the cardinal-strafe
  executor path or zero out depending on how strafe is plumbed.

## Acceptance
- Two side-by-side lap recordings on the same mission, one per motion
  model. Path-tracking-error number + qualitative observation logged.
- A decision recorded in the brief's body (or in a follow-up commit)
  with a one-line rationale: "going with X because Y."
- If switching, the follow-on YAML changes ship in the same PR.

## Out of scope

- Chassis hardware / wheel controller changes.
- Tuning Isaac Sim's mecanum dynamics. If sim is the broken one,
  accept it and note in the brief.
- The MPPI cost-landscape tuning (PathAlign, CostCritic, etc.) —
  that's the shared Nav2 config the [nav2-config-parity](../../context/nav2-config-parity.md)
  module governs (the older sim/real promotion split was retired for
  parity in [`completed/nav2-envelope-retirement`](../../completed/nav2-envelope-retirement.md)).

## Risks
- DiffDrive on a mecanum chassis can't strafe at the controller layer.
  The executor's cardinal-strafe path is the fallback, but it bypasses
  obstacle-aware closed-loop control during the strafe leg. Edge case
  to validate.
