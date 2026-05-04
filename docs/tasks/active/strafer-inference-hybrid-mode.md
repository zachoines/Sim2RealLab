# Hybrid execution mode: Nav2 global plan + RL local control (Jetson-side)

**Type:** task / feature
**Owner:** Jetson (extends `strafer_inference` from
[`strafer-inference-package.md`](strafer-inference-package.md))
**Priority:** P3 — blocks on **two** other briefs:
[`strafer-inference-package.md`](strafer-inference-package.md)
(produces the `strafer_inference` package this brief extends) and
[`strafer-lab-subgoal-env.md`](strafer-lab-subgoal-env.md) (produces
the trained `NOCAM_SUBGOAL` policy this brief loads). Lifts mission
quality on long-horizon / cross-room navigation but isn't blocking
any current mission shape.
**Estimate:** M (~3–4 days: hybrid backend in `strafer_inference` +
Nav2 `/plan` subscription + dispatch wiring + sim validation)
**Branch:** task/strafer-inference-hybrid-mode

## Story

As a **mission operator running cross-room or cross-obstacle
navigation in a known map**, I want **Nav2's global planner to
produce a path while the trained `NOCAM_SUBGOAL` RL policy handles
local control between path subgoals**, so that **deployed missions
get Nav2's global geometry awareness and the RL policy's smooth
continuous control — neither backend has to solve the entire problem
alone**.

The DEPTH `strafer_direct` mode (in
[`strafer-inference-package.md`](strafer-inference-package.md))
solves direct-pose-goal navigation with the policy's own depth-based
obstacle avoidance, but in environments where Nav2's costmap-aware
global plan is preferable (long known-map traversals, missions
through doorways, recovery routes), routing the policy through Nav2's
plan gives the operator the best of both backends.

This brief is intentionally **Jetson-only**. The DGX-side work that
makes hybrid mode possible — defining `PolicyVariant.NOCAM_SUBGOAL`,
building the subgoal-following training env, and producing a
deployable checkpoint — lives in
[`strafer-lab-subgoal-env.md`](strafer-lab-subgoal-env.md). That
brief must ship first.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [strafer-inference-package.md](strafer-inference-package.md) —
  the predecessor; this brief extends its `execution_backend`
  dispatch with a third mode and reuses its observation-pipeline
  infrastructure.
- [strafer-lab-subgoal-env.md](strafer-lab-subgoal-env.md) — the
  DGX-side prerequisite. Defines `PolicyVariant.NOCAM_SUBGOAL`,
  the `SubgoalCommand` term, the new training env, and produces
  the deployable checkpoint this brief loads.
- [policy-export-tooling.md](../completed/policy-export-tooling.md) — the
  export path the new variant flows through. No new export-side
  work needed; the variant uses `--variant NOCAM_SUBGOAL`.

## Context

### What's in scope here vs. delegated to the prerequisite brief

| Concern | This brief (Jetson) | [`strafer-lab-subgoal-env.md`](strafer-lab-subgoal-env.md) (DGX) |
|---|---|---|
| `PolicyVariant.NOCAM_SUBGOAL` definition | consumes | defines |
| Subgoal-following training env | consumes outputs | builds |
| `SubgoalCommand` term in `mdp/commands.py` | n/a | builds |
| Reward shaping for path-tracking | n/a | builds |
| Trained checkpoint | consumes | trains |
| Sim-internal path planner (training) | n/a | builds |
| Nav2 `/plan` subscription (deployment) | builds | n/a |
| Rolling-subgoal selection from Nav2 path | builds | n/a |
| `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` dispatch | builds | n/a |
| End-to-end sim validation under hybrid mode | runs | n/a |

The original version of this brief tried to own both lanes and
glossed over the training-env work in a single paragraph. That
hid ~80% of the actual effort behind the appearance of a smaller
deliverable. Splitting makes the dependency chain visible and lets
each lane execute independently.

### Subgoal selection algorithm

Pure-pursuit-style **lookahead-distance**: pick the point on the
published Nav2 path that is exactly `hybrid_lookahead_m`
(default 1.0 m) ahead of the robot's current position along the
path's arc length. Standard pattern; matches what the training-env
`SubgoalCommand` does internally so deployment-time observation
matches training-time observation.

Two alternatives considered and rejected for the MVP:

- **Lookahead-time:** pick the point the robot would reach in
  `hybrid_lookahead_s` at expected speed. More velocity-aware;
  more knobs. Defer until lookahead-distance proves insufficient.
- **Fixed path-index step:** pick the Nth point past the closest
  point on the path. Cheaper but doesn't adapt to path resolution.

### Failure handling

If Nav2 fails to plan (no path), or the policy stalls (no progress
along the path for `stall_timeout_s`), the hybrid backend reports
failure — same shape as Nav2's `/navigate_to_pose` action failure.
Operator-side retry policy is unchanged.

If `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` is set but the
strafer_inference action server isn't running (e.g. the trained
checkpoint is missing), the dispatch falls back to `nav2` per the
pattern set by
[`strafer-inference-package.md`](strafer-inference-package.md)
Phase 4.

## Approach

Three phases. All Jetson-side; the DGX-side prerequisites are
assumed shipped (see Context bundle).

### Phase 1 — Hybrid backend in `strafer_inference` (2 days)

In [`source/strafer_ros/strafer_inference/`](../../../source/strafer_ros/strafer_inference/)
(must exist — i.e.
[`strafer-inference-package.md`](strafer-inference-package.md) has
shipped):

- Add a `mode: "strafer_direct" | "hybrid"` runtime config flag
  (default `strafer_direct` to preserve existing behavior).
- When in hybrid mode:
  - Subscribe to Nav2's `/plan` topic (`nav_msgs/Path`). Nav2's
    planner publishes whenever a new global plan is computed.
    Track the latest path; replan triggers replace it.
  - On each inference tick (rate already derived in
    `strafer-inference-package` Phase 2):
    1. Find the closest point on the latest path to the current
       robot pose (TF `map → base_link`).
    2. Advance `hybrid_lookahead_m` (default 1.0 m) along path
       arc length from that closest point.
    3. That's the rolling subgoal pose, in `map` frame.
  - Build the obs dict using the *subgoal pose* as the referent
    for `goal_relative` / `goal_distance` /
    `goal_heading_to_goal`. Body-frame transform via TF, same as
    `strafer-inference-package` Phase 2.
  - Run inference with the loaded `PolicyVariant.NOCAM_SUBGOAL`
    policy (loaded via existing `load_policy()`).
  - All other elements of the inference contract (deterministic-
    output, L1-clamp, watchdogs, debug logging) are inherited
    from the strafer-inference-package brief.

- Watchdog gains a 6th source for hybrid mode: stale `/plan`
  (older than `path_timeout_s`, default 2.0 s — Nav2 publishes
  on replan, not continuously, so this timeout is longer than
  the depth one).

### Phase 2 — Backend dispatch update (½ day)

In [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py):

- Update `JetsonRosClient.navigate_to_pose` dispatch (currently
  recognizes `nav2` and `strafer_direct`, falls back to `nav2`
  on unknown values per
  [`strafer-inference-package.md`](strafer-inference-package.md)
  Phase 4) to recognize `hybrid_nav2_strafer` as a third value.
- For hybrid, the dispatch sends the goal to **both**:
  - Nav2's planner — to populate `/plan`. The action client
    targets a planner-only endpoint (configure Nav2 to expose a
    `compute_path_to_pose`-only action without engaging the
    controller server, OR use the `/compute_path_to_pose`
    service Nav2 already exposes).
  - The strafer_inference action server (in hybrid mode) — to
    consume the resulting `/plan` and execute local control.
- Nav2's controller server is **NOT** used in hybrid mode — only
  the planner.
- The hybrid action completes when strafer_inference reports
  success (final subgoal reached, within `xy_goal_tolerance`).

### Phase 3 — End-to-end sim validation (1 day)

- Cross-room sim mission: pick the reference mission from
  [`completed/nav2-far-goal-staging.md`](../completed/nav2-far-goal-staging.md)
  ("Navigate to the open wood door on other side of the room").
  Verify hybrid mode completes it with Nav2 publishing the path
  and the policy executing local control between subgoals.
- Capture run-table via
  [`tune_capture.py`](../../../source/strafer_ros/strafer_navigation/scripts/tune_capture.py).
  Expected: sustained median odom vx ≥ 1.0 m/s on straight
  segments (the metric the MPPI brief plateaued under at 0.632
  m/s).
- No-regression on `strafer_direct` (translate forward 3 m
  acceptance from the strafer-inference-package brief).
- No-regression on `nav2` (same translate mission with default
  backend).

Real-robot validation is out of scope here — file a separate
brief (`strafer-inference-hybrid-real-robot-validation.md`) once
sim validation passes. Real-robot hybrid mode introduces TF
freshness concerns (SLAM stalls), Nav2 replan latency, and other
real-world variables that warrant their own scope.

## Acceptance criteria

### Integration

- [ ] `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` routes:
      - Goal pose → Nav2's planner (via `compute_path_to_pose`
        action or service)
      - Resulting `/plan` → `strafer_inference` consumes it for
        rolling-subgoal selection
      - Inference output → `/cmd_vel`. Nav2's controller server
        is not invoked.
- [ ] Subgoal selection unit-tested: given a synthetic
      `nav_msgs/Path` and a robot pose, the picked subgoal sits
      at `hybrid_lookahead_m` ahead along path arc length within
      tolerance.
- [ ] Watchdog (6-source under hybrid mode): hybrid mode adds
      `/plan` staleness to the 5 sources from the
      strafer-inference-package brief. Unit-tested.
- [ ] No regression on `strafer_direct` or `nav2` modes — same
      mission tests pass with their respective backends.

### End-to-end

- [ ] Sim reference mission ("Navigate to the open wood door on
      other side of the room") completes under hybrid mode. PR
      description includes:
      - Nav2 path summary (length, # subgoals tracked, # replans
        if any).
      - `tune_capture.py` run-table covering the active translation
        portion. Sustained median odom vx ≥ 1.0 m/s on straight
        segments.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- [`source/strafer_ros/strafer_inference/`](../../../source/strafer_ros/strafer_inference/)
  — once
  [`strafer-inference-package.md`](strafer-inference-package.md)
  ships, this is the extension point. Phase 1 adds a hybrid mode
  flag and the Nav2 `/plan` subscription alongside the existing
  observation pipeline.
- [`source/strafer_ros/strafer_navigation/config/nav2_params.yaml`](../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml)
  — Nav2 planner config. The `compute_path_to_pose` action /
  service the hybrid dispatch will target is part of the Nav2
  bringup.
- [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
  — `JetsonRosClient.navigate_to_pose`; Phase 2 edits the
  dispatch path here.
- Pure-pursuit / lookahead-distance arc-length implementations:
  the `nav2_regulated_pure_pursuit_controller` source has an
  Apache 2.0 implementation that's license-compatible to
  reference / port.
- [`policy_interface.py`](../../../source/strafer_shared/strafer_shared/policy_interface.py)
  — `PolicyVariant.NOCAM_SUBGOAL` (defined by the prerequisite
  brief).

## Out of scope

### Sequencing notes

- This brief blocks on **two** prerequisites:
  - [`strafer-inference-package.md`](strafer-inference-package.md)
    must ship first (provides the `strafer_inference` package
    this brief extends).
  - [`strafer-lab-subgoal-env.md`](strafer-lab-subgoal-env.md)
    must ship first (defines `PolicyVariant.NOCAM_SUBGOAL`,
    builds the training env, and produces the deployable
    checkpoint).
- Both prerequisites can run in parallel — strafer-inference-package
  is Jetson-lane work, strafer-lab-subgoal-env is DGX-lane.
- This brief itself is small (~3–4 days) once both prerequisites
  are shipped.

### Not addressed here

- **Pure-RL execution (`strafer_direct`).** That's
  [`strafer-inference-package.md`](strafer-inference-package.md).
  Hybrid coexists with both pure modes; this brief doesn't
  change them.
- **The training environment.** That's
  [`strafer-lab-subgoal-env.md`](strafer-lab-subgoal-env.md).
- **The trained NOCAM_SUBGOAL checkpoint.** Produced by Phase 5
  of [`strafer-lab-subgoal-env.md`](strafer-lab-subgoal-env.md).
- **Replacing Nav2 entirely.** Nav2 stays as the default backend
  and as the global planner in hybrid mode.
- **Costmap-aware local control.** Hybrid here uses Nav2 for
  global planning *only*; local obstacle avoidance is the
  trained policy's responsibility (implicit in its training
  distribution — but NOCAM_SUBGOAL has no perception, so the
  policy relies on the path being valid). If a depth-aware
  subgoal-following variant is wanted later (Nav2 plans the
  global route, RL handles late-arriving obstacles via depth),
  file a `DEPTH_SUBGOAL` follow-up.
- **Real-robot hybrid validation.** File as
  `strafer-inference-hybrid-real-robot-validation.md` once sim
  validation passes.
- **Performance comparison vs. Nav2-MPPI on the same mission.**
  Evaluation activity, not a controller-design brief.
