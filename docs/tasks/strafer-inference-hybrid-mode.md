# Hybrid execution mode: Nav2 global plan + RL local control

**Type:** task / feature
**Owner:** Either — cross-lane work. The DGX lane defines a new
`PolicyVariant` and trains it; the Jetson lane wires the hybrid
backend into `strafer_inference`. See **Out of scope** for the
recommended sequencing.
**Priority:** P3 — depends on
[`strafer-inference-package.md`](strafer-inference-package.md)
shipping the `strafer_direct` mode first AND on a subgoal-following
policy being trained. Not blocking any current mission shape; lifts
quality on long-horizon / cross-room missions.
**Estimate:** L (~1–2 weeks total: ½ week new `PolicyVariant` +
training-env subgoal command + smoke training run; ~½ week training
to convergence; ~½ week Jetson integration + validation)
**Branch:** task/strafer-inference-hybrid-mode

## Story

As a **mission operator running cross-room or cross-obstacle
navigation in a known map**, I want **Nav2's global planner to
produce a path while the trained RL policy handles local control
between path subgoals**, so that **deployed missions get Nav2's
global geometry awareness and the RL policy's local execution
quality — neither backend has to solve the entire problem alone**.

The pure-RL `strafer_direct` mode (in
[`strafer-inference-package.md`](strafer-inference-package.md))
solves direct-pose-goal navigation but doesn't have global path
context, so it can't reliably navigate around walls or through
doorways without seeing them in training. Hybrid mode bridges that
gap: Nav2's GridBased planner finds the geometric route, the RL
policy follows it section-by-section.

## Context bundle

Read these before starting:
- [context/repo-topology.md](context/repo-topology.md)
- [context/ownership-boundaries.md](context/ownership-boundaries.md)
- [strafer-inference-package.md](strafer-inference-package.md) — the
  predecessor; this brief extends its `execution_backend` dispatch
  with a third mode and adds a new `PolicyVariant` to the shared
  contract.
- [policy-export-tooling.md](policy-export-tooling.md) — the export
  path the new variant flows through. No new export-side work
  needed; the variant uses the existing
  `--variant NOCAM_SUBGOAL` flag.
- [policy-goal-noise-training.md](policy-goal-noise-training.md) —
  the same noise-resilience pattern likely applies to subgoals;
  evaluate after the baseline subgoal policy converges.

## Context

### Why hybrid needs a different policy

`strafer_direct` consumes a **final goal pose** — the RL agent's
job is to converge on it. The observation contract uses
`goal_relative` / `goal_distance` / `goal_heading_to_goal` for
*the final destination*.

Hybrid consumes a **next subgoal pose** — a rolling target along
the Nav2 path, typically ~1 m ahead. The RL agent's job is to
*track* that subgoal as it advances, not converge on it. The same
observation fields point at a different referent.

This isn't a deployment-time configuration change; it's a different
*training distribution*. A goal-directed policy trained on
full-mission-length goals isn't useful for subgoal-following — it'll
under-shoot or over-shoot subgoals because its reward landscape
optimized for long-distance convergence. We need a new `PolicyVariant`
trained on a different command term.

### Two pieces of work

This is genuinely cross-lane:

| Lane | Work |
|------|------|
| **DGX (`strafer_lab`)** | Add `PolicyVariant.NOCAM_SUBGOAL` to `strafer_shared`. Add subgoal-emitting command term to `strafer_lab/tasks/navigation/mdp/commands.py`. Train the variant to convergence. Export via [`policy-export-tooling.md`](policy-export-tooling.md). |
| **Jetson (`strafer_ros`)** | Add `hybrid_nav2_strafer` execution backend to `strafer_inference`. Subscribe to Nav2's `/plan` topic. Pick rolling subgoal from the path. Feed through `assemble_observation` with the new variant. Wire dispatch in `JetsonRosClient.navigate_to_pose`. |

DGX work blocks Jetson integration (Jetson can't validate without a
trained checkpoint). Jetson can land structural plumbing in parallel
against a hand-built dummy artifact for the new variant, then bring
up end-to-end once the trained checkpoint arrives.

### Key design decisions to lock down before implementation

1. **Subgoal selection algorithm.** Three reasonable shapes:
   - **Lookahead-distance** (recommended starting point): pick the
     point on the published Nav2 path that is exactly
     `hybrid_lookahead_m` (default 1.0 m) ahead of the robot's
     current position along the path arc length. Standard
     pure-pursuit-style selection.
   - **Lookahead-time:** pick the point the robot would reach in
     `hybrid_lookahead_s` at expected speed. More velocity-aware;
     more knobs.
   - **Fixed path-index step:** pick the Nth point past the closest
     point on the path. Cheaper but doesn't adapt to path resolution.
   Start with lookahead-distance.
2. **Subgoal observation contract.** `PolicyVariant.NOCAM_SUBGOAL`
   should reuse the existing `goal_relative` / `goal_distance` /
   `goal_heading_to_goal` field shapes — same dims, same scales —
   but the *referent* is the rolling subgoal, not the mission goal.
   This keeps the network architecture identical to NOCAM and reuses
   `assemble_observation` mechanics; only the input semantics change.
3. **Mission-completion signal.** Hybrid returns mission-success
   when Nav2 reports the global path has been completed (i.e. the
   final subgoal is the actual mission goal and the policy converges
   on it). The action server's success/failure semantics match
   Nav2's, so `JetsonRosClient` doesn't need new completion logic.
4. **Failure handling.** If Nav2 fails to plan (no path), or the
   policy stalls (no progress along the path for `stall_timeout_s`),
   the hybrid backend reports failure — same shape as Nav2's
   `/navigate_to_pose` action failure. Operator-side retry policy is
   unchanged.

## Approach

### Phase 1 (DGX) — `PolicyVariant.NOCAM_SUBGOAL` + subgoal command term (~3 days)

In [`source/strafer_shared/strafer_shared/policy_interface.py`](../../source/strafer_shared/strafer_shared/policy_interface.py):

- Add `_NOCAM_SUBGOAL_FIELDS` mirroring `_NOCAM_FIELDS` exactly. The
  field shapes / scales are identical; the documented semantics
  change (the goal-related fields now refer to a subgoal pose).
- Add `PolicyVariant.NOCAM_SUBGOAL = _NOCAM_SUBGOAL_FIELDS`.
- Update the docstring to call out the contract: same network
  architecture as NOCAM; different training distribution; consumed
  by the hybrid backend.

In [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py):

- Add `SubgoalCommand` class (or extend `GoalCommand` with a subgoal
  mode). The command term:
  - Samples a long-horizon goal as today.
  - Emits a *rolling subgoal* `lookahead_m` ahead along the
    straight-line vector from the robot to the long-horizon goal.
    Initially the simplest implementation: subgoal = robot_pos +
    `lookahead_m` * unit_vector(goal - robot_pos), capped at the
    long-horizon goal pose itself when within `lookahead_m` of it.
  - The "follow a curved Nav2 path" part is *deployment-time only*;
    training-time can use straight-line interpolation since the
    policy only needs to learn "track the rolling target."
- Add `SubgoalCommandCfg` with knobs for `lookahead_m` (default 1.0)
  and the inherited goal-sampling knobs.
- Unit test: instantiate, advance simulated robot position toward
  the long-horizon goal, assert the published subgoal pose advances
  monotonically and converges on the long-horizon goal as the robot
  approaches.

### Phase 2 (DGX) — Train `NOCAM_SUBGOAL` to convergence (~3–5 days wall)

- Wire a new training task variant in
  [`Scripts/train_strafer_navigation.py`](../../Scripts/train_strafer_navigation.py)
  that uses the `SubgoalCommand` term and the
  `PolicyVariant.NOCAM_SUBGOAL` observation group.
- Train to convergence. Target metrics: subgoal-tracking error
  (median <  20 cm at convergence), episode reward stable.
- Export via [`policy-export-tooling.md`](policy-export-tooling.md)
  with `--variant NOCAM_SUBGOAL`.

### Phase 3 (Jetson) — Hybrid backend in `strafer_inference` (~2 days)

In [`source/strafer_ros/strafer_inference/`](../../source/strafer_ros/strafer_inference/)
(once
[`strafer-inference-package.md`](strafer-inference-package.md)
ships):

- Add a hybrid backend behind a runtime mode flag (e.g.
  `mode: "strafer_direct" | "hybrid"`, defaulting to
  `strafer_direct` to preserve current behavior).
- When in hybrid mode:
  - Subscribe to Nav2's `/plan` topic
    (`nav_msgs/Path`). The planner publishes whenever a new global
    plan is computed.
  - On each inference tick, find the point on the latest path that is
    `hybrid_lookahead_m` ahead of the current robot pose along the
    path arc length. That's the subgoal.
  - Build the obs dict using the *subgoal pose* as the referent for
    `goal_relative` / `goal_distance` / `goal_heading_to_goal`.
  - Run inference with the loaded `PolicyVariant.NOCAM_SUBGOAL`
    policy.
  - All other elements of the inference contract
    (deterministic-output, L1-clamp, watchdogs, debug logging) are
    inherited from the strafer-inference brief.

In [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py):

- Update `JetsonRosClient.navigate_to_pose` dispatch (currently
  recognizes `nav2` and `strafer_direct`, falls back to `nav2` on
  unknown values per the strafer-inference brief's Phase 4) to
  recognize `hybrid_nav2_strafer` as a third value.
- For hybrid, the dispatch sends the goal to *both* Nav2's planner
  (to populate `/plan`) and the strafer_inference action server (to
  consume the resulting `/plan`). Nav2's controller server is *not*
  used in hybrid mode — only the planner.

### Phase 4 — End-to-end validation (~2 days)

- **Sim cross-room mission**: pick the reference mission from
  [`completed/nav2-far-goal-staging.md`](completed/nav2-far-goal-staging.md)
  ("Navigate to the open wood door on other side of the room").
  Verify hybrid mode completes it with Nav2 publishing the path and
  the policy executing local control between subgoals. Capture run-
  table via [`tune_capture.py`](../../source/strafer_ros/strafer_navigation/scripts/tune_capture.py).
- **No-regression on `strafer_direct`** (translate forward 3 m
  acceptance from the strafer-inference brief).
- **No-regression on `nav2`** (same translate mission with default
  backend).
- **Real-robot mission** (after sim validation passes): same shape,
  small indoor environment.

## Acceptance criteria

### Contract (`strafer_shared` + `strafer_lab`)

- [ ] `PolicyVariant.NOCAM_SUBGOAL` defined in
      [`policy_interface.py`](../../source/strafer_shared/strafer_shared/policy_interface.py)
      with field semantics documented as "subgoal pose, not final
      goal pose."
- [ ] `obs_dim` matches `PolicyVariant.NOCAM` (same architecture;
      different training distribution). Anchored by a unit test.
- [ ] `SubgoalCommand` (or `GoalCommand` subgoal mode) in
      [`commands.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py)
      with unit-tested rolling-subgoal-along-vector behavior.

### Training (DGX)

- [ ] Trained checkpoint at convergence for `NOCAM_SUBGOAL`. PR
      description includes:
      - Episode reward + subgoal-tracking error (median + p95) at
        convergence.
      - Comparison plot vs. `NOCAM` baseline on the *same physical
        mission* (long-horizon traverse): which variant tracks the
        path more faithfully when given the same global plan?
- [ ] Exported via [`policy-export-tooling.md`](policy-export-tooling.md)
      `Scripts/export_policy.py --variant NOCAM_SUBGOAL`. Round-trip
      determinism asserted as in that brief.

### Integration (Jetson)

- [ ] `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` routes:
      - Goal pose → Nav2's planner (via the same action / topic Nav2
        normally consumes).
      - Resulting `/plan` → `strafer_inference` consumes it for
        rolling-subgoal selection.
      - Inference output → `/cmd_vel`. Nav2's controller server is
        not invoked.
- [ ] Subgoal selection is unit-tested: given a synthetic `nav_msgs/Path`
      and a robot pose, the picked subgoal sits at
      `hybrid_lookahead_m` ahead along path arc length within
      tolerance.
- [ ] Watchdog: hybrid mode also requires `/plan` to be fresh
      (older than `path_timeout_s` triggers zero twist + warning).
      Adds to the 4-source watchdog from the strafer-inference brief.
- [ ] No regression on `strafer_direct` or `nav2` modes — same
      mission tests pass with their respective backends.

### End-to-end

- [ ] Sim reference mission ("Navigate to the open wood door on
      other side of the room") completes under hybrid mode. PR
      description includes:
      - Nav2 path summary (length, # subgoals, # staging legs if
        any).
      - `tune_capture.py` run-table covering the active translation
        portion. Sustained median odom vx ≥ 1.0 m/s on straight
        segments (the inference brief's acceptance metric).
- [ ] Real-robot mission of equivalent shape — completes
      successfully with `xy_goal_tolerance` and `yaw_goal_tolerance`
      from
      [`nav2_params.yaml`](../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml)
      satisfied.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- [`source/strafer_shared/strafer_shared/policy_interface.py:42-79`](../../source/strafer_shared/strafer_shared/policy_interface.py)
  — `PolicyVariant` enum + `ObsField` definitions; the pattern for
  adding a new variant.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py)
  — `GoalCommand` (line 23), `GoalCommandCfg` (line 363),
  `GoalCommandProcRoom` (line 413). Subgoal command extends from
  these.
- [`source/strafer_ros/strafer_navigation/config/nav2_params.yaml`](../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml)
  — Nav2 planner config; confirm the topic Nav2 publishes the path on
  (`/plan` is the convention; verify the QoS for the strafer_inference
  subscription).
- [`source/strafer_ros/strafer_inference/`](../../source/strafer_ros/strafer_inference/)
  — once
  [`strafer-inference-package.md`](strafer-inference-package.md)
  ships, this is the extension point.
- [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`](../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py)
  — `JetsonRosClient.navigate_to_pose`; Phase 3 edits the dispatch
  path here.
- Pure-pursuit / lookahead-distance subgoal selection: classical
  reference patterns in the rover-navigation literature; the
  ros2/Nav2 `nav2_regulated_pure_pursuit_controller` source has a
  reasonable implementation of arc-length lookahead that can be
  ported (Apache 2.0 — license-compatible).

## Out of scope

### Sequencing within this brief

- Phase 1+2 (DGX) **must** complete before Phase 3 (Jetson)
  end-to-end validation, but Phase 3's *plumbing* can land in
  parallel against a hand-built dummy `NOCAM_SUBGOAL` artifact
  (any deterministic 19-dim → 3-dim mapping). End-to-end acceptance
  gates on the trained checkpoint.
- If the lanes split this into two PRs (DGX-side variant + training,
  Jetson-side hybrid backend), make the Jetson PR depend on the DGX
  PR and the strafer-inference brief both being merged. Same `task/`
  branch convention applies if a single owner picks up both lanes.

### Not addressed here

- **Pure-RL execution (`strafer_direct`).** That's
  [`strafer-inference-package.md`](strafer-inference-package.md).
  Hybrid coexists with both pure modes; this brief doesn't change
  them.
- **Replacing Nav2 entirely.** Nav2 stays as the default backend and
  as the global planner for hybrid mode.
- **Costmap-aware local control.** Hybrid here uses Nav2 for global
  planning *only*; local obstacle avoidance is the trained policy's
  responsibility (implicit in its training distribution: depth-aware
  variants may catch obstacles, NOCAM-subgoal won't). Adding explicit
  costmap input to the policy observation is a further variant —
  file separately if needed.
- **Performance comparison vs. Nav2-MPPI on the same mission.** That's
  an evaluation activity, not a controller-design brief. The
  comparison plot in the acceptance criteria is for the variant-vs-
  variant question (NOCAM vs NOCAM_SUBGOAL on a long-horizon mission),
  not the backend-vs-backend question.
- **DEPTH-variant subgoal training.** First subgoal target is NOCAM-
  shaped; if the depth-aware variant is wanted later, file as a
  follow-up brief once both
  [`strafer-inference-package.md`](strafer-inference-package.md)'s
  DEPTH brief and this brief have shipped.
- **Goal-position noise on subgoals.** The
  [`policy-goal-noise-training.md`](policy-goal-noise-training.md)
  pattern likely applies (subgoals from Nav2's path have planner-
  resolution noise of ~MAP_RESOLUTION = 5 cm, smaller than VLM
  noise but non-zero). Evaluate after the baseline subgoal
  checkpoint converges; file a follow-up if a noise pass is needed.
