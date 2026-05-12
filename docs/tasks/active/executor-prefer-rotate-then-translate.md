# Prefer rotate-then-translate over diagonal strafe for non-cardinal motion

**Type:** task / enhancement
**Owner:** Jetson agent (`source/strafer_autonomy/strafer_autonomy/executor/`, with possibly minor changes under `source/strafer_ros/strafer_navigation/`)
**Priority:** P2
**Estimate:** M (~1–2 days; executor-level decomposition + MPPI revisit if needed)
**Branch:** task/executor-prefer-rotate-then-translate

## Story

As a **mission operator**, I want **the executor (or controller) to decompose a non-cardinal translation into a rotate-to-face followed by a forward translate**, so that **the mecanum chassis traverses to off-axis goals at smooth forward speeds rather than tracking a diagonal strafe path that MPPI struggles to execute cleanly**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [completed/mppi-critic-tuning-for-sim-envelope.md](../completed/mppi-critic-tuning-for-sim-envelope.md)
  — what's already been tuned in the MPPI critic landscape.

## Context

Operator observation (2026-05-11): the robot has trouble strafing at non-cardinal angles. Pure cardinal strafe (forward, backward, left, right relative to body frame) works acceptably, but oblique paths trigger sustained sideways oscillation that the MPPI critic tuning in `mppi-critic-tuning-for-sim-envelope.md` and the global-path smoothing in this PR only partially fix. The mecanum kinematics technically support arbitrary planar velocity vectors, but at the velocity envelope MPPI is exploring, integration noise and wheel-slip transients make any non-axis-aligned commanded direction noisy in practice.

The operator wants the *policy* to bias toward forward-only motion: at the start of every translation step, rotate the chassis to face the goal direction, then drive forward. Strafing is then preserved only for cardinal-relative moves (sideways strafing remains useful for short corrections without re-orienting). This trades one rotation per goal for the dynamic smoothness of forward-only translation — a worthwhile trade in practice for mecanum platforms that aren't strictly required to maintain heading mid-traverse.

The natural place to implement this is the executor's translation skill: when issuing a Nav2 `navigate_to_pose` goal, set the goal yaw to face the goal position from the robot's current pose, and (optionally) issue a pre-orientation `rotate_in_place` step if the heading delta exceeds some threshold.

## Approach

Two layers, evaluate both before committing:

### A. Executor-level decomposition (recommended starting point)

In the executor's `_translate` / `_navigate_via_staging` paths:
1. Compute the heading delta between the robot's current yaw and the bearing from robot pose → goal position.
2. If `|delta_yaw| > heading_threshold` (e.g., 0.3 rad ≈ 17°), prepend a `rotate_in_place` step that aligns the chassis with the goal bearing.
3. Set the Nav2 goal pose's yaw to the goal-bearing yaw (so the controller drives forward, not sideways, into the goal).
4. Special case: cardinal-relative translations (the operator's "translate left 1 m" / "strafe right 0.5 m") should bypass the decomposition since the operator explicitly asked for body-frame strafe.

Pros: localized change in `executor/`; minimal risk to MPPI tuning; preserves cardinal strafe.

Cons: extra wall-clock time per mission step (one rotation per leg).

### B. Controller-level policy via MPPI critic

Bump `PreferForwardCritic.cost_weight` further and/or introduce a custom critic that penalizes non-cardinal commanded `vy` more aggressively. Could implement as a YAML-only change if the existing critic surface is enough.

Pros: fully transparent to executor; cardinal strafe still works at low cost.

Cons: high-blast-radius change to MPPI; would invalidate the recent critic-tuning shipped baseline and require re-validation of the cornering + far-goal regression checks.

### C. Differential-drive mode at angles

Treat MPPI's motion model as `DiffDrive` for non-cardinal goals (instead of `Omni`). The controller would then naturally rotate-then-translate. Requires either a per-goal motion-model switch (not standard in Nav2) or wrapping multiple controllers.

Pros: cleanest controller-level expression.

Cons: not natively supported; significant engineering effort.

**Recommended sequence:** A first. If A leaves residual oscillation on oblique paths because MPPI still finds reasons to strafe in flight, layer B's targeted critic bump on top.

## Acceptance criteria

- [ ] Operator-level test: issue an `align then translate` mission whose goal is offset 45° from the robot's spawn heading, at 2 m range. Expected behavior: robot rotates to face the goal (visible in Foxglove), then drives forward smoothly. `/cmd_vel.vy` peak ≤ 0.3 × `/cmd_vel.vx` peak during the translation phase.
- [ ] Cardinal strafe still works: an operator-issued `strafe right 1 m` still strafes sideways without re-orienting (the body-frame command is preserved).
- [ ] No regression on the cornering smoke check from `mppi-critic-tuning-for-sim-envelope.md`: `translate forward 1 m → rotate 90° → translate forward 1 m` lands within `xy_goal_tolerance=0.15` / `yaw_goal_tolerance=0.20`.
- [ ] No regression on `nav2-far-goal-staging.md`'s reference far-goal mission.
- [ ] Unit tests cover the executor-level decision (heading-threshold gating, cardinal-translate bypass).
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance) for the surface list and trigger heuristics.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py` — translation step dispatch.
- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py` — `rotate_in_place`, `navigate_to_pose`.
- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml` — MPPI critics + `general_goal_checker.yaw_goal_tolerance`.
- `source/strafer_autonomy/strafer_autonomy/schemas/` — the skill schema for any new "translate via rotate-then-forward" intent (or extend the existing `translate_to_pose` skill).

## Out of scope

- **Re-tuning MPPI critics broadly.** The previous brief covered that. Only adjacent tuning that supports option (A) lands here; broader rebalances go in their own brief.
- **Changing the motion model away from `Omni`.** That's option (C) above — file separately if (A) + (B) prove insufficient.
- **The trained RL policy's behavior at angles.** Policy-side fixes are tracked under `policy-goal-noise-training.md` and `strafer-inference-package.md`.
