# Planner emits the wrong sign for "rotate left" / "rotate right"

**Type:** task / bug
**Owner:** DGX agent
**Priority:** P2
**Estimate:** S (~hours; prompt edit + planner unit-test fixtures)
**Branch:** task/planner-rotate-direction-prompt

## Story

As a **mission operator issuing rotate commands**, I want
**"rotate left N°" to actually rotate the robot CCW and "rotate right N°"
to rotate CW**, so that **direction commands behave the way an
operator expects without me having to memorize the sign convention
the LLM happens to have settled on**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/sim-velocity-attenuation.md](../../completed/sim-velocity-attenuation.md)
  — surfaced this and two adjacent rotation issues during validation.

## Context

Operator-observed: `"rotate left 90 degrees"` rotates the robot to
its right (CW). The downstream code path is sign-clean end-to-end:

- [`source/strafer_autonomy/strafer_autonomy/planner/intent_parser.py:109-113`](../../../../source/strafer_autonomy/strafer_autonomy/planner/intent_parser.py)
  — `orientation_mode` is passed through unchanged.
- [`source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py:174-183`](../../../../source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py)
  — `float(mode)` with no sign munging.
- [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:1453`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
  — `yaw_delta = math.radians(degrees)` — no sign flip.
- [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:718`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
  — `direction = +1 if yaw_delta_rad >= 0 else -1`,
  `angular.z = direction * speed`. Standard ROS convention
  (positive `angular.z` = CCW = robot's left).

So the convention is correct; the bug is the LLM emitting the wrong
sign. The system prompt at
[`source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py:48-50`](../../../../source/strafer_autonomy/strafer_autonomy/planner/prompt_builder.py)
declares *"positive = CCW"* but the few-shot examples only show **one**
rotate case (line 79-80, "turn 90 degrees left" → `"90"`). With no
symmetric "right" example and no explicit "left = CCW = positive"
restatement, Qwen3-4B is unreliable on the mapping.

## Approach

Strengthen the rotate section of the system prompt:

1. **Restate the convention in plain English next to the rule.** Add
   a sentence like *"Left turns are positive (CCW from above);
   right turns are negative (CW)."* alongside the existing
   "positive = CCW" line.
2. **Add a symmetric "rotate right" example** (e.g.
   `"turn 90 degrees right" → {"orientation_mode": "-90", ...}`).
3. **Add a "clockwise / counterclockwise" example** so the LLM
   learns the synonyms map to the correct sign.
4. *(Optional, lower-confidence)* — drop a unit test that exercises
   "left" vs "right" via the live planner end-to-end on a small
   batch and asserts sign matches expectation. Local Qwen3-4B
   inference is already stood up via `make serve-planner`.

## Acceptance criteria

- [ ] System prompt has at least three rotate examples covering
      "left", "right", and one of {"clockwise", "counterclockwise"}
      with the correct sign in each.
- [ ] Manual operator-side smoke: `strafer-autonomy-cli submit
      "rotate left 90 degrees"` produces a `rotate_by_degrees` step
      with `degrees=+90`; the same command with "right" produces
      `degrees=-90`. Robot in sim rotates the matching direction.
- [ ] Existing planner tests still pass; any new tests covering
      direction sign live alongside.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- Already-verified end-to-end sign convention (above). The fix is
  prompt-only; no code-path change needed.
- The planner runs on the DGX via `make serve-planner` against
  Qwen3-4B; the DGX lane edits live entirely under
  `source/strafer_autonomy/strafer_autonomy/planner/`.
- For the rotation **timeout** issue the same operator session
  surfaced, see
  [`rotate-in-place-sim-clock-deadline.md`](../../completed/rotate-in-place-sim-clock-deadline.md)
  and
  [`plan-compiler-skill-timeouts.md`](../../completed/plan-compiler-skill-timeouts.md)
  — orthogonal, different lanes.

## Out of scope

- **Rotate timeout / rotate physics tuning.** Different briefs (above).
- **Cardinal-direction rotates** (`face north`, etc.). Those route
  through `_orient_to_direction` and use `_CARDINAL_YAWS` — already
  unambiguous, not affected by this fix.
- **Translate sign convention.** Already covered with explicit
  forward/backward + left/right examples in the same prompt; no
  field reports of mis-direction there.
