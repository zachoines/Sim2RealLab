# Hybrid execution mode: Nav2 global plan + RL local control (Jetson-side)

**Type:** task / feature
**Owner:** Jetson (extends `strafer_inference` from
[`inference-package`](inference-package.md))
**Priority:** P3 — blocks on **two** other briefs:
[`inference-package`](inference-package.md)
(produces the `strafer_inference` package this brief extends) and
[`subgoal-env`](subgoal-env.md) (produces
the trained `NOCAM_SUBGOAL` policy this brief loads). Lifts mission
quality on long-horizon / cross-room navigation but isn't blocking
any current mission shape.
**Estimate:** M (~3–4 days: hybrid backend in `strafer_inference` +
Nav2 `/plan` subscription + dispatch wiring + sim validation)
**Branch:** task/strafer-inference-hybrid-mode

**Status:** Shipped 2026-06-28 (Jetson) across three PRs — PR A
[#119](https://github.com/zachoines/Sim2RealLab/pull/119) (generator + `/plan`
subscription), PR B [#122](https://github.com/zachoines/Sim2RealLab/pull/122)
(variant lift + subgoal observation), PR C
[#123](https://github.com/zachoines/Sim2RealLab/pull/123)
(`hybrid_nav2_strafer` dispatch + the two-node plan-freshness guard).

## Scope: shipped across three PRs

This brief is executed as three PRs off
`task/strafer-inference-hybrid-mode` (the multi-PR-per-brief pattern,
as with [`harness-architecture`](../active/harness/harness-architecture.md)):

- **PR A — deploy-side rolling-subgoal generator + Nav2 `/plan`
  subscription.** A variant-agnostic numpy port of the training-time
  arc-length cursor selection rule (`RollingSubgoalGenerator` in
  `strafer_inference/generator.py`), plus a dedicated
  `strafer_subgoal_generator` node that subscribes to `/plan`
  (`nav_msgs/Path`, `map` frame) and publishes the rolling subgoal as a
  `geometry_msgs/PoseStamped` on `/strafer/subgoal`. Gated by a
  hand-computed ≤10 cm subgoal-position parity test (with an optional
  torch cross-check against the training cursor). Resolved decisions:
  lookahead fixed at `SUBGOAL_LOOKAHEAD_M`; a dedicated `/strafer/subgoal`
  topic (never the goal topic, which would trip the inference node's
  mid-mission hidden-state reset); `plan_topic` parameter defaulting to
  `/plan`. PR A records the latest-plan receipt time for PR C's watchdog
  but adds no watchdog source.
- **PR B — inference-node variant lift + subgoal observation assembly.**
  Lifts the hardcoded `PolicyVariant.DEPTH`, assembles the subgoal
  observation (body-frame transform of the `/strafer/subgoal` pose), and
  drops the depth precondition for the no-camera variant.
- **PR C — `hybrid_nav2_strafer` dispatch + plan-freshness guard.** The
  `JetsonRosClient.navigate_to_pose` dispatch routes the goal to Nav2's
  global planner (continuous `ComputePathToPose` replanning to keep `/plan`
  fresh; controller server not engaged) and to the `strafer_inference`
  action server for local control, with per-mission fallback to `nav2`. The
  plan-freshness guard is realized in two nodes — the split decoupled
  `/plan` consumption from `/cmd_vel`: the generator suppresses
  `/strafer/subgoal` once `/plan` ages past `path_timeout_s`, and the
  inference node adds a `subgoal` watchdog source that zero-twists
  `/cmd_vel` when `/strafer/subgoal` goes stale.

All three PRs have shipped; operator-driven sim validation rides as a
follow-up in [`strafer-hybrid-sim-validation`](trained-policy/strafer-hybrid-sim-validation.md).

## Story

As a **mission operator running cross-room or cross-obstacle
navigation in a known map**, I want **Nav2's global planner to
produce a path while the trained `NOCAM_SUBGOAL` RL policy handles
local control between path subgoals**, so that **deployed missions
get Nav2's global geometry awareness and the RL policy's smooth
continuous control — neither backend has to solve the entire problem
alone**.

### Architecture choice: this is one of four corners

This brief's "Nav2-global + RL-local" split is one of four possible
corners of the four-architecture matrix:

|  | Local control by Nav2 | Local control by RL |
|---|---|---|
| **Global planning by Nav2** | shipped today (Nav2-only) | **THIS BRIEF** (`hybrid_nav2_strafer`) |
| **Global planning by RL** | [`rl-global-nav2-local`](../parked/trained-policy/rl-global-nav2-local.md) (parked) | `strafer_direct` ([inference-package](inference-package.md), DEPTH MVP — shipping) |

The current direction (this brief + DEPTH MVP) is well-grounded: RL
is good at smooth continuous control under noise, Nav2 is good at
costmap-aware path planning on known maps. But the inverse corner
(RL as the *global* planner emitting waypoints + Nav2's controller
following them) might be a better fit for VLM-grounded missions
where the planning decision involves *intent* (which way around the
chair, whether to back up and re-approach a doorway, etc.) rather
than just geometry. Filed as a parked alternative in
[`rl-global-nav2-local`](../parked/trained-policy/rl-global-nav2-local.md) — pick up only
if/when this brief's deployment surfaces a "RL is doing local
control but the issue is global plan quality" failure mode. Don't
implement preemptively.

The DEPTH `strafer_direct` mode (in
[`inference-package`](inference-package.md))
solves direct-pose-goal navigation with the policy's own depth-based
obstacle avoidance, but in environments where Nav2's costmap-aware
global plan is preferable (long known-map traversals, missions
through doorways, recovery routes), routing the policy through Nav2's
plan gives the operator the best of both backends.

This brief is intentionally **Jetson-only**. The DGX-side work that
makes hybrid mode possible — defining `PolicyVariant.NOCAM_SUBGOAL`,
building the subgoal-following training env, and producing a
deployable checkpoint — lives in
[`subgoal-env`](subgoal-env.md). That
brief must ship first.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [strafer-inference-package.md](inference-package.md) —
  the predecessor; this brief extends its `execution_backend`
  dispatch with a third mode and reuses its observation-pipeline
  infrastructure.
- [strafer-lab-subgoal-env.md](subgoal-env.md) — the
  DGX-side prerequisite. Defines `PolicyVariant.NOCAM_SUBGOAL`,
  the `SubgoalCommand` term, the new training env, and produces
  the deployable checkpoint this brief loads.
- [policy-export-tooling.md](policy-export-tooling.md) — the
  export path the new variant flows through. No new export-side
  work needed; the variant uses `--variant NOCAM_SUBGOAL`.

## Context

### What's in scope here vs. delegated to the prerequisite brief

| Concern | This brief (Jetson) | [`subgoal-env`](subgoal-env.md) (DGX) |
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

**Lookahead parity (load-bearing — the policy's whole path view is
this one distance).** The NOCAM_SUBGOAL policy observes only the
resulting subgoal pose, never the Nav2 path's waypoints, so the
single train↔deploy parity surface is the lookahead distance, not
the path resolution. Two facts the training env (subgoal-env) fixed
that this backend must honor:

- The lookahead distance is the shared constant
  `strafer_shared.constants.SUBGOAL_LOOKAHEAD_M` (1.0 m). Set
  `hybrid_lookahead_m` *from that constant* rather than re-hardcoding
  1.0, so the two lanes cannot drift.
- The training env's robust tier randomizes its lookahead over a band
  (0.7–1.3 m) so the policy tracks a subgoal at any distance in that
  range. A NOCAM_SUBGOAL checkpoint trained on the robust tier is
  therefore robust to a deployed lookahead anywhere in the band — it
  does not require this backend to reproduce 1.0 m exactly.

**Open decision for this brief to make** (record the choice + reason
in its PR): either (a) fix `hybrid_lookahead_m` at
`SUBGOAL_LOOKAHEAD_M` and treat the robust-tier band purely as
training slack, or (b) let the backend advertise its actual lookahead
so the `strafer-hybrid-sim-validation` rosbag parity check can assert
it lands inside the band the checkpoint was trained against. (a) is
simpler; (b) makes the parity machine-checkable. Pick one when this
brief is picked up.

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
[`inference-package`](inference-package.md)
Phase 4.

## Approach

Three phases. All Jetson-side; the DGX-side prerequisites are
assumed shipped (see Context bundle).

### Phase 1 — Hybrid backend in `strafer_inference` (2 days)

In `source/strafer_ros/strafer_inference/`
(must exist — i.e.
[`inference-package.md`](inference-package.md) has
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
  [`inference-package`](inference-package.md)
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

### Phase 3 — End-to-end sim validation (extracted)

The cross-room sim mission, the per-tick / mission-start latency
benchmarks, the rig parity bounds, and the no-regression checks
against `strafer_direct` and `nav2` live in
[`strafer-hybrid-sim-validation`](trained-policy/strafer-hybrid-sim-validation.md)
(parked alongside this brief). Mirrors the
[`inference-package`](inference-package.md) →
[`strafer-direct-sim-validation`](../active/trained-policy/strafer-direct-sim-validation.md)
extraction precedent so this brief's runtime PR can ship with
unit-testable acceptance closed and the operator-driven validation
rides as a follow-up.

Real-robot hybrid validation is filed as a separate brief
(`strafer-inference-hybrid-real-robot-validation.md`) once the sim
validation brief ships. Real-robot hybrid introduces TF freshness
concerns (SLAM stalls), Nav2 replan latency under load, and
moved-obstacle distribution shift that warrant their own scope.

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

### Operator-driven sim validation (extracted)

The reference-mission, rosbag parity, latency, and no-regression
acceptance items live in
[`strafer-hybrid-sim-validation`](trained-policy/strafer-hybrid-sim-validation.md).
That brief gates on this one shipping + a trained checkpoint +
the sim-in-the-loop rig; none of those items are unit-testable, so
they ride as a follow-up rather than blocking this brief's PR.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_ros/strafer_inference/` — once
  [`inference-package.md`](inference-package.md)
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
  - [`inference-package`](inference-package.md)
    must ship first (provides the `strafer_inference` package
    this brief extends).
  - [`subgoal-env`](subgoal-env.md)
    must ship first (defines `PolicyVariant.NOCAM_SUBGOAL`,
    builds the training env, and produces the deployable
    checkpoint).
- Both prerequisites can run in parallel — strafer-inference-package
  is Jetson-lane work, strafer-lab-subgoal-env is DGX-lane.
- This brief itself is small (~3–4 days) once both prerequisites
  are shipped.

### Not addressed here

- **Pure-RL execution (`strafer_direct`).** That's
  [`inference-package`](inference-package.md).
  Hybrid coexists with both pure modes; this brief doesn't
  change them.
- **The training environment.** That's
  [`subgoal-env`](subgoal-env.md).
- **The trained NOCAM_SUBGOAL checkpoint.** Produced by Phase 5
  of [`subgoal-env`](subgoal-env.md).
- **Replacing Nav2 entirely.** Nav2 stays as the default backend
  and as the global planner in hybrid mode.
- **Costmap-aware local control.** Hybrid here uses Nav2 for
  global planning *only*; local obstacle avoidance is the
  trained policy's responsibility (implicit in its training
  distribution — but NOCAM_SUBGOAL has no perception, so the
  policy relies on the path being valid). The depth-aware
  subgoal-following variant (Nav2 plans the global route, RL
  handles late-arriving obstacles via depth) is filed as
  [`depth-subgoal-env`](../parked/trained-policy/depth-subgoal-env.md) (DGX, training)
  and
  [`depth-subgoal-hybrid-runtime`](../parked/trained-policy/depth-subgoal-hybrid-runtime.md)
  (Jetson, runtime extension). Both parked alongside this brief;
  un-park when this one + their other prerequisites ship.
- **Operator-driven sim validation.** Lives in
  [`strafer-hybrid-sim-validation`](trained-policy/strafer-hybrid-sim-validation.md)
  (parked alongside this brief); un-parks when this one ships.
  Carries the rosbag parity, latency benchmarks, and the
  cross-room reference mission.
- **Real-robot hybrid validation.** File as
  `strafer-inference-hybrid-real-robot-validation.md` once
  [`strafer-hybrid-sim-validation`](trained-policy/strafer-hybrid-sim-validation.md)
  passes.
- **Performance comparison vs. Nav2-MPPI on the same mission.**
  Evaluation activity, not a controller-design brief.
