# Real-D555 reliable-depth-range survey

**Type:** investigation
**Owner:** Jetson agent
**Priority:** P2
**Estimate:** S–M (a few hours of measurement + write-up; M only if it
escalates into a separate retraining task)

## Story

As a **mission operator running real-robot deployments**, I want **a
measured number for how far the real D555 reliably reports depth in
our deployment rooms**, so that **the conservative 6 m default in
`goal_projection_node._DEPTH_MAX_M` is either confirmed as the right
real-robot bound or raised to admit cross-room VLM targets we already
know operators report seeing past 6 m**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](context/bridge-runtime-invariants.md)
- [completed/goal-projection-depth-range.md](completed/goal-projection-depth-range.md)
  — the shipped task this one follows from; explains why the 6 m
  default was an unmeasured guess and what was deferred to here.

## Context

`goal_projection_node._DEPTH_MAX_M = 6.0` is annotated as "Beyond
this the stereo estimate is noisy" — a heuristic, not a measurement.
The shipped goal-projection task
([`completed/goal-projection-depth-range.md`](completed/goal-projection-depth-range.md))
made the cap env-overridable so sim missions can use the renderer's
honest 15 m depth, but explicitly **did not change the real-robot
default** because the 6 m number was never validated against the
actual deployment optics + room conditions.

Two anecdotes motivate measuring:

1. Operator testing reports usable D555 detections past 6 m. If the
   real noise floor is closer to 8–10 m, the default is leaving real
   performance on the table.
2. Conversely, the D555's stereo baseline + 1280×720 native rate may
   genuinely degrade past 6 m in textured indoor rooms — the
   datasheet's "ideal range" claims don't necessarily hold under
   realistic lighting / wall-texture conditions. Confirming 6 m is
   the right number is also a valid outcome.

Two distinct downstream changes depend on this number:

- **Cheap follow-up (in scope of *this* task's recommendation if the
  data warrants it):** raise the default in
  `goal_projection_node._DEPTH_MAX_M`. Goal-projection is downstream
  of Nav2 / policy and changing this only affects which VLM-grounded
  targets are accepted vs. rejected. No retraining, no costmap
  recalibration.
- **Expensive follow-up (OUT OF SCOPE here — file a separate task if
  warranted):** raise `strafer_shared.constants.DEPTH_MAX = 6.0`.
  That value drives `DEPTH_SCALE = 1/DEPTH_MAX` for the policy's
  depth observation normalization AND Nav2 costmap raytracing
  ranges. Changing it shifts the trained policy's input distribution
  and requires retraining + re-tuning the costmap obstacle layer.

This survey produces the data both decisions need; it only commits
to the cheap one directly.

## Method

Run all measurements in a representative deployment room (one of the
rooms current real-robot missions actually operate in, not a tuning
fixture).

Suggested protocol — adjust if a better tool surfaces:

1. **Static reach.** Park the robot facing a textured wall.
   Translate the wall (or robot) through `[1, 2, 3, 4, 5, 6, 7, 8,
   10, 12] m`. At each station, capture ~5 s of
   `/d555/depth/image_rect_raw` via `ros2 bag record`. Use
   `realsense-viewer` or a small helper script to extract per-frame:
     - Mean and stddev of depth in a central 50×50 patch.
     - Pixel-wise dropout fraction (zeros / NaN) in the same patch.
     - Spread vs. ground-truth distance (tape-measured).
2. **Lighting variants.** Repeat the 4 m, 6 m, 8 m, 10 m stations in
   (a) bright daylight, (b) overhead fluorescent, (c) low-light
   evening — D555 stereo degrades fast under low-texture / low-light
   regions.
3. **Room corner / non-flat target.** A wall is the easy case.
   Repeat at the longest range that still gave usable numbers in
   step 1, but pointed at a more realistic target (chair, doorway,
   bookshelf), so the measurement reflects the kind of geometry VLM
   grounding actually sees.
4. **Scratch in the perf doc.** Land the table + interpretation in
   `docs/PERF_INVESTIGATION_SIM_IN_THE_LOOP.md` (or a new
   `PERF_INVESTIGATION_REAL_D555.md` if the existing doc is too
   sim-specific — judgment call). Include the room's lighting
   conditions and date so a re-survey under different conditions is
   reproducible.

The numbers we care about are:

- **Range at which dropout fraction exceeds ~5 %** — past this the
  median-of-25 patch in `_median_depth` starts coin-flipping.
- **Range at which depth bias exceeds ~10 %** of ground-truth — past
  this the projected goal pose is off by more than the standoff
  margin.
- **Stddev profile** — sudden growth past a particular range is the
  classic stereo-baseline noise floor signature.

## Scope of impact

- **Real-robot deployments**: if the measurement supports it, raise
  `goal_projection_node._DEPTH_MAX_M` to the new measured number.
  VLM targets currently rejected past 6 m (across-room targets) get
  projected. No policy / Nav2 changes — this is the same boundary
  the shipped env-override decoupling cleared.
- **Sim deployments**: untouched. Sim already uses
  `STRAFER_PROJECTION_DEPTH_MAX_M=15.0` from
  `env_sim_in_the_loop.env`.
- **Trained policy**: untouched. If the survey suggests the
  stack-wide `DEPTH_MAX = 6.0` is also leaving performance on the
  table, file a SEPARATE task; do not change it from this brief.

## Acceptance criteria

- [ ] A measured table of (range, dropout fraction, mean depth bias,
      stddev) for the D555 in at least one real deployment room,
      across at least three lighting conditions.
- [ ] A recommendation, written down in the perf doc, with one of:
      (a) keep `_DEPTH_MAX_M = 6.0` — survey confirms it; (b) raise
      to `<new value>` — the data shows reliability past 6 m;
      (c) `<new value>` plus a separate task brief filed for the
      stack-wide `strafer_shared.constants.DEPTH_MAX` change (with
      retraining implications spelled out).
- [ ] If outcome (b): the default in
      `source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py`
      changes in this task. The existing test that asserts the
      default updates accordingly (or, better, the test asserts the
      default matches a named constant rather than a literal).
- [ ] If outcome (c): a new task brief in `docs/tasks/` for the
      stack-wide change, including the measurement that justified
      it, the policy-retraining + Nav2-recalibration scope, and an
      explicit owner pick.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py:55-58`
  — the default `_DEPTH_MAX_M` constant whose justification this
  survey is testing.
- `source/strafer_shared/strafer_shared/constants.py` — `DEPTH_MAX`,
  `DEPTH_SCALE`, and the comments around them. Do NOT modify here;
  read for context on what depends on the stack-wide value.
- `source/strafer_ros/strafer_navigation/launch/navigation.launch.py:43-56`
  — how `DEPTH_MAX` and `DEPTH_MIN` propagate into Nav2's
  raytracing config. Useful for documenting what a stack-wide change
  would actually touch (informs the optional follow-up brief).
- `realsense-viewer` (Intel, ships with `librealsense`) — built-in
  depth quality / stereo error visualization. Faster than rolling
  custom analysis from a bag.
- `rs-depth-quality-tool` — Intel's CLI stereo evaluation tool.
  Better numbers than eyeballing realsense-viewer for the table.
- D555 datasheet "ideal stereo range" — Intel's spec sheet quotes a
  reliability bound; useful as one data point to calibrate
  expectations against, not as the answer.
- [`completed/goal-projection-depth-range.md`](completed/goal-projection-depth-range.md)
  — the shipped task that motivated this one. The "Out of scope"
  section there is exactly the work this brief picks up.

## Out of scope

- **Changing `strafer_shared.constants.DEPTH_MAX`.** That cascades
  into policy-input normalization, the trained checkpoint, Nav2
  costmap raytracing, and the costmap obstacle layer's behavior.
  If the survey says we should change it, file a separate task with
  retraining included. Do not bundle.
- **Surveying a different sensor** (D435, D455, etc.). This survey
  is specifically the deployed D555.
- **Changing `_DEPTH_MIN_M`.** The min cutoff is a stereo-baseline
  hard limit, not a noise heuristic, so a measurement-driven raise
  isn't the question being asked here.
