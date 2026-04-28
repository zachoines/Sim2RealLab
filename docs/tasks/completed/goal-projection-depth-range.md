# Goal projection depth range — decouple from policy normalization cap

**Status:** Shipped 2026-04-27 in `35018b1` (Jetson).
**Follow-ups:** [`real-d555-depth-range-survey.md`](../real-d555-depth-range-survey.md)
informs whether the conservative 6 m default should rise stack-wide.

**Type:** task / bug
**Owner:** Jetson agent
**Priority:** P1
**Estimate:** S (~half day; single-file plumbing + env-var override + tests)

## Story

As a **mission operator running sim-in-the-loop and real-robot
missions**, I want **goal projection to accept depth values as far as
the camera can reliably see**, so that **the VLM can ground a target
across a room and the planner gets a valid 3D goal pose instead of
hitting `goal_projection_failed: out of range`**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)

## Context

`goal_projection_node._median_depth` filters its 5×5 patch to the
range `[_DEPTH_MIN_M=0.3, _DEPTH_MAX_M=6.0]`. The 6 m cap is annotated
as "Beyond this the stereo estimate is noisy" — a conservative
heuristic for the real D555's stereo, not a sensor limit. Two
deployments are routinely tripping it:

1. **Sim missions.** Isaac Sim renders perpendicular-to-image-plane
   depth perfectly out to `D555_RENDER_FAR_CLIP_M = 50 m`. Any VLM
   ground larger than ~6 m sim-frame distance — a target on a far
   wall, across a hallway, doorway across a room — is rejected even
   though the depth is more accurate than the real D555 would be.
2. **Real-robot.** Operator testing reports usable detections past
   6 m on the real D555. The cap was a guess, not measured against
   the actual deployment optics + room conditions.

Concretely, today's failing mission:

```json
{
  "error_code": "goal_projection_failed",
  "message": "Invalid depth at pixel (801, 306): zero, NaN, or out of range."
}
```

The error message lumps three rejection reasons together so the
operator can't tell from the response that "out of range" specifically
fired (vs. genuine zero / NaN — which would indicate a different
problem at the bridge level).

This task is **independent of the broader 6 m question**:

- `DEPTH_MAX = 6.0` in `strafer_shared.constants` drives policy
  observation normalization (`DEPTH_SCALE = 1/DEPTH_MAX`) and Nav2
  costmap raytracing. Changing those numbers shifts the deployed
  policy's input distribution and would require retraining. **Do not
  touch them in this task.**
- `_DEPTH_MAX_M` in goal projection is only used to reject out-of-band
  depth before pinhole-projecting a single pixel into camera frame.
  It has no policy or Nav2 implications. This is the only depth-range
  knob this task changes.

## Scope of impact

- **Sim missions**: VLM-grounded targets up to the new cap (recommend
  ~15 m initial) are projected successfully. Goal-projection failures
  drop from "every mission with a far target" to "only genuine
  bridge / VLM problems".
- **Real-robot deployment**: behavior unchanged at the default. Cap
  stays at 6 m unless explicitly overridden. A follow-up campaign can
  re-evaluate the real D555's reliable range and decide whether to
  raise the default permanently.
- **Trained policy**: untouched. This change does not feed depth into
  the policy; goal_projection is upstream of Nav2 only.

## Acceptance criteria

- [ ] `goal_projection_node.py` reads `STRAFER_PROJECTION_DEPTH_MAX_M`
      and `STRAFER_PROJECTION_DEPTH_MIN_M` env vars at startup. Unset
      → defaults preserved (6.0 / 0.3) so real-robot bringup is
      unchanged. Set to a positive float → that value is used.
      Non-numeric / non-positive → log warning, keep default.
- [ ] Warning + default-preservation behavior matches the existing
      pattern from `STRAFER_NAV_VEL_SCALE` in `navigation.launch.py`.
      Same log voice ("Ignoring non-numeric ..." / "overrides ...").
- [ ] `env_sim_in_the_loop.env` exports
      `STRAFER_PROJECTION_DEPTH_MAX_M=15.0` with a comment block
      pointing at this task and explaining why sim wants a wider
      range than the real-D555 conservative default.
- [ ] Error message split: distinguish "zero or NaN at pixel" (bridge
      problem — depth dropped, sky region) from "all 25 patch values
      out of [min, max] range" (sensor reach problem). Operator
      should be able to tell from the response which path fired.
- [ ] Unit tests in `source/strafer_ros/strafer_perception/test/`:
      - default range preserved when env unset
      - env override applied to a fresh node instantiation
      - non-numeric / non-positive overrides fall back to default
      - the new error-message split actually distinguishes the two
        rejection reasons
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py:36-40`
  — current constants (`_DEPTH_MAX_M`, `_DEPTH_MIN_M`).
- `source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py:233-260`
  — `_median_depth`. The range filter `(valid >= _DEPTH_MIN_M) & (valid <= _DEPTH_MAX_M)`
  is what currently rejects at-range values without distinguishing
  from zero/NaN.
- `source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py:121-128`
  — error message construction. Splitting requires returning a
  reason from `_median_depth` (or a small enum) instead of just
  `None`.
- `source/strafer_ros/strafer_navigation/launch/navigation.launch.py:50-86`
  — reference implementation for the env-var override pattern
  (`_resolved_nav_velocities`). Mirror its tone: defensive parsing,
  one info-log on override, warning on bad input, fallback to
  constants on failure.
- `source/strafer_ros/strafer_bringup/config/env_sim_in_the_loop.env`
  — the only file that should export the new sim-only override.

## Out of scope

- **Raising `DEPTH_MAX = 6.0` in `strafer_shared.constants`.**
  Bigger task. Affects policy observation normalization and Nav2
  raytracing; needs policy retraining to be in-distribution. File a
  separate task if/when the real-robot range survey says 6 m is
  leaving real performance on the table.
- **Bridge render product resolution mismatch.** Sim cam_info is
  reporting `width=1280, height=720` while the perception camera
  prim is configured at 640×360 (see Isaac Sim startup log:
  `Forcing fy to fx (671.3043... != 671.3043...)` — fx=671 px
  matches `1.93 mm × 1280 / 3.68 mm`, not the expected 335.65 px
  for 640-wide). DGX-side issue, separate task.
- **Real-robot D555 range survey.** Out of scope here; would inform
  whether the default 6 m should be raised stack-wide.
