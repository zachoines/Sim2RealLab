# Build the strafer_lab subgoal-following training env for `NOCAM_SUBGOAL` policy

**Type:** task / new training environment
**Owner:** DGX (`strafer_lab` lane — env config + path planner +
training run)
**Priority:** P2 — blocks
[`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md)
end-to-end validation but not the strafer_direct MVP path.
**Estimate:** L (~1.5–2 weeks: path generator + subgoal command term
+ reward shaping + termination events + initial training run).
**Branch:** task/strafer-lab-subgoal-env

## Story

As a **`strafer_lab` operator who wants to train an RL policy that
follows a Nav2-style global path**, I want **a training environment
that procedurally generates collision-free paths through sampled
rooms, emits a rolling subgoal as the agent's command target, and
shapes reward around tracking that subgoal**, so that **a trained
`PolicyVariant.NOCAM_SUBGOAL` checkpoint exists for the
`hybrid_nav2_strafer` Jetson backend to consume**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
  — this brief adds a new RL env variant; compose it over the axes and
  add a golden-hash gate, don't write a new subclass.
- [strafer-inference-hybrid-mode.md](../../parked/trained-policy/hybrid-mode.md)
  — the consumer-side brief; this brief produces the trainable env
  and the deployable checkpoint that brief loads.
- [strafer-inference-package.md](../../completed/inference-package.md) —
  the DEPTH-MVP sibling. Useful as a reference for how the
  observation contract, decimation, and reward conventions are
  structured in the existing ProcRoom-Depth env.
- [policy-export-tooling.md](../../completed/policy-export-tooling.md) — the
  export path the new variant will flow through. No new export-side
  work needed; the variant uses `--variant NOCAM_SUBGOAL`.

## Context

### NOCAM_SUBGOAL safety: it trusts Nav2's costmap absolutely

This brief's MVP target is `PolicyVariant.NOCAM_SUBGOAL` — same 19
proprioceptive dims as NOCAM, no perception. The deployment story
is: Nav2 plans a path through the costmap (which is up-to-date and
obstacle-aware), the policy follows the path.

**This is only safe when the costmap is trustworthy.** Failure
modes the NOCAM_SUBGOAL policy cannot detect or recover from:

- **Stale costmap.** RTAB-Map or the costmap layer hasn't seen a
  recent obstacle (moved chair, person, dropped object). The path
  goes through it. Policy follows; collision.
- **TF lag between costmap and policy.** Costmap planned at t=0,
  obstacle appeared at t=1, policy is at t=1.5. The path it sees
  is from before the obstacle existed.
- **SLAM jump.** Loop-closure snaps the `map` frame. The path was
  in old-map coordinates; in new-map coordinates it now goes
  through a wall.

The `strafer_direct` DEPTH MVP exists precisely because NOCAM-direct
is unsafe. NOCAM_SUBGOAL is safer than NOCAM-direct (Nav2 handles
global obstacle awareness for it) but strictly less safe than
DEPTH_SUBGOAL (where the policy itself can see late-arriving
obstacles).

Two implications:

1. **Document the trust boundary in the variant docstring.** The
   `PolicyVariant.NOCAM_SUBGOAL` definition added in Phase 2 must
   call out: this variant trusts Nav2's costmap absolutely; the
   deployment lane must include a costmap freshness watchdog and
   must not use it in dynamic-obstacle scenarios.
2. **Consider DEPTH_SUBGOAL as the actual MVP target.** This brief
   currently parks DEPTH_SUBGOAL "as a follow-up brief once both
   this brief and inference-package's DEPTH path have shipped."
   That ordering makes sense for code reuse but defers the
   architectural call. Worth re-evaluating at planning time
   whether DEPTH_SUBGOAL (Nav2 plans the route, RL handles
   late-arriving obstacles with depth) is the safer MVP and
   NOCAM_SUBGOAL is the speed-optimization follow-up. See
   `## Out of scope` for the parked DEPTH_SUBGOAL placeholder.

### Why `hybrid_nav2_strafer` needs a different policy

The `strafer_direct` mode (in
[`inference-package`](../../completed/inference-package.md))
consumes a **final goal pose** — the RL agent's job is to converge
on it, with depth perception for obstacle avoidance.

Hybrid mode consumes a **rolling subgoal** along Nav2's global path
— typically ~1 m ahead, advancing as the agent makes progress. The
agent's job is to *track* that subgoal, not converge on it. The
training distribution is fundamentally different:

| | `strafer_direct` (DEPTH) | `hybrid_nav2_strafer` (NOCAM_SUBGOAL) |
|---|---|---|
| Command target | Final goal pose | Rolling subgoal pose |
| Episode shape | Long horizon to far goal | Short horizon to near subgoal |
| Reward landscape | Goal-reaching + obstacle-avoidance | Path-tracking (cross-track + along-track) |
| Termination | At goal or timeout | Path complete, off-path divergence, timeout |
| Required perception | Yes (DEPTH) | No — Nav2 already planned around obstacles |

A goal-directed DEPTH policy isn't usable as a subgoal-tracker
because its reward landscape optimized for "reach the far point,"
not "stay on the path the planner gave you." So even though we
already have a trained DEPTH checkpoint, hybrid mode needs a
purpose-built training env.

### What already exists

[`source/strafer_lab/strafer_lab/tasks/navigation/`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/)
ships:

- `strafer_env_cfg.py` — the env config builder; the `_apply_runtime_defaults`
  helper sets `_DEFAULT_NAV_SIM_DT = 1/120`, `_DEFAULT_NAV_DECIMATION = 4`
  (the 30 Hz policy step rate the inference brief derives).
- `mdp/commands.py` — `GoalCommand` (line 23) + `GoalCommandProcRoom`
  (line 413). Existing command terms emit a single fixed goal pose;
  this brief adds a new term that emits a rolling subgoal along a
  generated path.
- `mdp/observations.py` — `goal_relative`, `goal_distance`,
  `goal_heading_to_goal`, `last_action`, `body_velocity_xy` — the
  NOCAM observation fields. These all reference *whatever the
  active command term emits* via `command_manager.get_command(...)`,
  so swapping the command term is enough; the observation terms
  re-bind automatically.
- `mdp/rewards.py` — existing reward functions (goal-distance
  shaping, action smoothing, etc.) some of which transfer; others
  need new path-aware variants for cross-track / along-track
  shaping.
- `mdp/terminations.py` — existing terminations (goal-reached,
  collision, timeout). Needs a new "off-path divergence"
  termination.
- `agents/rsl_rl_ppo_cfg.py` + `agents/__init__.py` — runner config
  + registered task IDs. New IDs get registered alongside the
  existing ProcRoom variants (e.g.
  `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0`).

### What's missing

1. **A path planner** that runs in sim against the current scene
   geometry. Sim already has ground-truth obstacle positions
   (Infinigen room boundaries + furniture from the scene-gen
   pipeline), so we don't need a costmap — just A* / RRT through
   the actual collision geometry. Must produce a `nav_msgs/Path`-
   shape (sequence of waypoints in env-local frame) so the
   inference-time hybrid backend can consume the same shape from
   Nav2's `/plan` topic.

2. **A `SubgoalCommand` term** in `mdp/commands.py`. Emits a
   rolling subgoal pose `lookahead_m` ahead along the path's arc
   length, advancing as the robot makes progress. Resamples a new
   path when the robot reaches the end (or termination triggers).

3. **A `PolicyVariant.NOCAM_SUBGOAL`** in `strafer_shared.policy_interface`.
   Same field shapes as NOCAM (19 dims) — the network architecture
   doesn't change. Only the *referent* of the goal-related fields
   changes (subgoal pose vs. final goal pose). Documented as such
   so future maintainers don't conclude the two variants are
   interchangeable at deployment time.

4. **Reward shaping** for path-tracking. Cross-track error
   (distance from current pose to nearest path point), along-track
   progress (arc-length advance per step), action smoothness.
   Likely need to re-tune coefficients; the existing
   goal-distance-shaping coefficients won't transfer cleanly.

5. **A new termination event** for off-path divergence. If the
   robot strays more than `max_off_path_m` (e.g. 0.5 m) from the
   path, terminate the episode with a penalty.

6. **Registered task IDs** in
   [`navigation/__init__.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/__init__.py).
   Following the existing pattern:
   `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0`,
   `Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-v0`,
   plus `-Play-v0` variants for visualization.

7. **A trained checkpoint** at convergence on the new task. The
   PPO setup likely transfers (same network architecture as NOCAM)
   but reward coefficients and PPO hyperparameters may need tuning
   for the new shaping. Output: a checkpoint that
   [`policy-export-tooling.md`](../../completed/policy-export-tooling.md) can
   convert to a deployable artifact.

## Approach

Five phases, sequenced. Phases 1–4 are dev work; Phase 5 is the
training run that gates downstream consumers.

### Phase 1 — Path planner (3–4 days)

Build a sim-internal path planner under
`source/strafer_lab/strafer_lab/tasks/navigation/path_planner/`
(new module). API:

```python
def plan_path(
    start_xy: np.ndarray,        # 2D, env-local
    goal_xy: np.ndarray,
    obstacle_geometry: Sequence[Polygon],  # from scene metadata
    *,
    method: str = "astar",       # or "rrt" / "rrt_star"
    discretization_m: float = 0.05,
) -> np.ndarray:                  # (N, 2) path points
    ...
```

#### Planner choice: minimize distributional gap with deployment

The deployed hybrid backend consumes paths from Nav2's actual
planner. The training-time planner produces a *distribution* of
paths against which the policy learns to track. A custom A* / RRT
written for this brief will differ from Nav2's GridBased or
SmacHybrid planner in heuristic choice, tie-breaking, smoothing,
and exact discretization rule. The differences are subtle but the
policy learns to track the training planner's quirks; at
deployment Nav2's quirks differ and the policy may oscillate,
shortcut, or hesitate on path segments where the two planners
disagree.

Two options to mitigate, in order of preference:

**Option A (recommended) — Use Nav2's planner offline at training
time.** `nav2_simple_commander` ships Python bindings to
NavFn / GridBased planners. Import directly from the
`env_isaaclab3` Python environment:
```python
from nav2_simple_commander.line_iterator import LineIterator
# plus relevant planner bindings
```
Pre-bake paths per scene at episode reset (the same scene is
used for many episodes, so the cost amortizes). The policy
trains against the exact paths Nav2 will produce at deployment.
Verify the Python bindings work on the DGX before committing —
Nav2 is ROS-stack-coupled and may need apt-installed deps.

**Option B (fallback) — Custom A* + path noise.** If Option A
proves infeasible, ship the custom A* but add per-tick path
*perturbation* during training: shift each waypoint by
`N(0, MAP_RESOLUTION/2)`. This trains the policy to be robust
to any planner-quirk-level disagreement up to ~2.5 cm. The
acceptance criterion's "≤ MAP_RESOLUTION * 2 median deviation"
sanity check becomes the noise envelope, not a parity bar.

The original framing of this brief assumed Option B implicitly. The
choice matters because: at deployment, the dominant failure mode of
"training planner ≠ Nav2 planner" is the kind of bug that surfaces
only after a checkpoint is shipped and a real-robot mission goes
wrong — the cheapest place to catch it is at the env-design
decision, not at the post-deployment debugging stage.

- Output shape matches `nav_msgs/Path` waypoint semantics so the
  inference-time hybrid backend can consume identical structure.

Tests in `source/strafer_lab/tests/test_path_planner.py`:
- Chosen planner (Option A or B) finds a path through a known
  obstacle config (collision-free + bounded length).
- For Option A: a parity test that the offline Nav2 call produces
  the same path the inference-side `/plan` topic publishes for the
  same scene + endpoints (within ROS-side numerical noise).
- For Option B: a noise-envelope test that perturbation magnitude
  is bounded and doesn't put waypoints in obstacles.
- Pathological cases (no path exists, start == goal, start in
  obstacle) raise meaningful exceptions.

### Phase 2 — `SubgoalCommand` term + `PolicyVariant.NOCAM_SUBGOAL` (2–3 days)

In [`source/strafer_shared/strafer_shared/policy_interface.py`](../../../../source/strafer_shared/strafer_shared/policy_interface.py):

- Add `_NOCAM_SUBGOAL_FIELDS` mirroring `_NOCAM_FIELDS` exactly
  (same shapes, same scales). The network architecture is
  identical to NOCAM; only the input semantics change.
- Add `PolicyVariant.NOCAM_SUBGOAL = _NOCAM_SUBGOAL_FIELDS`.
- Update the docstring to call out the contract: same shape as
  NOCAM, but the goal-related fields refer to a *rolling subgoal
  pose*, not a final goal pose. Consumers must use the matching
  command term at training and inference time; mixing variants is
  silent garbage.

In [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py):

- Add `SubgoalCommand` class. State:
  - The full sampled path (`np.ndarray (N, 2)`)
  - The current along-path index / arc-length cursor
  - The lookahead distance (`lookahead_m`, default 1.0)
- Per-tick update: compute the robot's closest point on the path,
  advance the cursor to `closest + lookahead_m` along arc length,
  emit the subgoal pose at the cursor.
- Resample triggers: end of path, off-path divergence
  termination, or scheduled resampling time.
- `SubgoalCommandCfg` exposes: `lookahead_m`, `path_method`
  (passed to the planner), `resampling_time_range`,
  `max_off_path_m`, knobs inherited from `GoalCommandCfg`.

Tests in `source/strafer_lab/tests/test_subgoal_command.py`:
- Synthetic robot trajectory along a known path: subgoal advances
  monotonically along arc length, stays at constant lookahead
  distance.
- Robot slightly off-path: subgoal still resolves to a valid path
  point at lookahead distance.
- Robot at end of path: subgoal converges on the final point and
  resampling triggers.

### Phase 3 — Reward shaping + termination (2–3 days)

In [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/rewards.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/rewards.py):

- Add `path_along_track_progress` — positive reward proportional
  to arc-length advance per step.
- Add `path_cross_track_error` — negative reward proportional to
  distance from nearest path point.
- Reuse existing `action_rate` / `action_smoothness` rewards
  unchanged.
- Optionally add `subgoal_reach_bonus` — small terminal reward
  when the final subgoal is reached (path complete).

In [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/terminations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/terminations.py):

- Add `path_complete` — terminate when the robot reaches the end
  of the current path within `goal_reach_threshold`.
- Add `off_path_divergence` — terminate if cross-track error
  exceeds `max_off_path_m` (penalty applied via reward).
- Reuse existing `time_out`, `collision` terminations.

The reward coefficients are NOT prescribed here — the operator
tunes them during the training run (Phase 5). This phase ships
the reward functions; coefficients are a hyperparameter sweep,
not durable code.

### Phase 4 — Env config + task registration (1 day)

In [`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py):

- Add `StraferNavEnvCfg_Real_ProcRoom_Subgoal_NoCam` (and
  `_Robust_` variant + `_PLAY` variants) — composes the existing
  ProcRoom scene + NOCAM observations + new SubgoalCommand +
  new rewards + new terminations.

In [`source/strafer_lab/strafer_lab/tasks/navigation/__init__.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/__init__.py):

- Register new task IDs:
  - `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0`
  - `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0`
  - `Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-v0`
  - `Isaac-Strafer-Nav-RLNoCam-Subgoal-Robust-Play-v0`

Smoke test:
```
python source/strafer_lab/scripts/test_strafer_env.py --task Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0
```
runs without errors, displays the path + subgoal markers in the
Kit viewport, robot follows the path at the configured policy
rate.

### Phase 5 — Training run + checkpoint (3–5 days wall, depending on convergence)

- Train against `Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-v0`
  with PPO (existing `agents/rsl_rl_ppo_cfg.py`, possibly with
  re-tuned learning rate / clip range).
- Target convergence metrics (record in PR description):
  - Episode reward at convergence
  - Median cross-track error at convergence (target: ≤ 10 cm)
  - Median time-to-subgoal (target: subgoal stays at ~lookahead_m
    distance with no oscillation)
- Save the converged checkpoint as
  `logs/rsl_rl/strafer_navigation/subgoal_baseline/model_<step>.pt`.
- Run a `play_strafer_navigation.py` rollout to confirm sane
  behavior — robot tracks the path smoothly, doesn't oscillate,
  doesn't shortcut through obstacles when the path goes around them.

If `policy-goal-noise-training.md` has shipped by the time this
brief gets to Phase 5, follow up with a noise-resilience pass on
top of the converged baseline. If not, file a
`policy-subgoal-noise-training.md` brief at ship time —
subgoal noise from Nav2 path resolution (~5 cm at MAP_RESOLUTION)
is smaller than VLM-grounded goal noise but non-zero, and the
deployed policy should be trained against it.

## Acceptance criteria

### Path planner

- [ ] `path_planner.plan_path(...)` produces collision-free paths
      on a known scene config (unit-tested with a synthetic
      obstacle layout).
- [ ] A* method matches Nav2's GridBased convention closely enough
      that paths produced by the two methods on equivalent scene
      geometries differ by ≤ `MAP_RESOLUTION * 2` median deviation
      along arc length (sanity check on parity, not byte
      identical).

### Command + variant

- [ ] `PolicyVariant.NOCAM_SUBGOAL` defined in
      `strafer_shared.policy_interface` with
      `obs_dim == PolicyVariant.NOCAM.obs_dim` and a docstring
      that explicitly contrasts the goal-field semantics.
- [ ] `SubgoalCommand` unit-tested for: monotonic arc-length
      advance under a synthetic robot trajectory; stable
      lookahead distance under small lateral noise; resampling
      trigger at end of path.

### Reward + termination

- [ ] `path_along_track_progress` and `path_cross_track_error`
      reward functions unit-tested against synthetic robot poses
      relative to a known path.
- [ ] `off_path_divergence` termination unit-tested at the
      `max_off_path_m` threshold.

### Env + registration

- [ ] All four task IDs registered (`Real`/`Robust` × non-play /
      play). `gym.make(<id>)` succeeds for each.
- [ ] `python source/strafer_lab/scripts/test_strafer_env.py --task Isaac-Strafer-Nav-RLNoCam-Subgoal-Real-Play-v0`
      runs to completion without errors, with path + subgoal
      markers visible in the viewport.

### Training run

- [ ] Converged checkpoint at
      `logs/rsl_rl/strafer_navigation/subgoal_baseline/model_<step>.pt`.
      PR description records: episode reward, cross-track error
      median + p95, training-time, training command + git commit.
- [ ] Play-script rollout confirms qualitative behavior: smooth
      path tracking, no oscillation, no shortcuts. Operator
      records a video / sequence of screenshots in the PR.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- [`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
  — env config patterns; `_apply_runtime_defaults` for the rate
  constants Phase 4 will reuse.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py)
  — `GoalCommand` (line 23) and `GoalCommandProcRoom` (line 413)
  for command-term patterns. `SubgoalCommand` extends the same
  base.
- [`source/strafer_lab/strafer_lab/tasks/navigation/__init__.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/__init__.py)
  — task registration pattern at lines 99–106 (existing ProcRoom
  variants).
- [`source/strafer_shared/strafer_shared/policy_interface.py`](../../../../source/strafer_shared/strafer_shared/policy_interface.py)
  — `PolicyVariant` enum; pattern for adding new variants.
- [`source/strafer_lab/scripts/train_strafer_navigation.py`](../../../../source/strafer_lab/scripts/train_strafer_navigation.py)
  — training entry point; should pick up the new task ID without
  changes if registration follows the existing pattern.
- [`source/strafer_lab/scripts/play_strafer_navigation.py`](../../../../source/strafer_lab/scripts/play_strafer_navigation.py)
  — visualization entry point for Phase 5's qualitative check.
- Reference implementations of A*/RRT in robotics-friendly form:
  `nav2_smac_planner` source (Apache 2.0, license-compatible),
  `python-rrt` packages on PyPI. Pick whichever has the cleaner
  API for this brief's path-planner module.

## Out of scope

- **The Jetson-side hybrid runtime.** That's
  [`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md).
  This brief produces the trainable env and the deployable
  checkpoint; the hybrid brief consumes them.
- **DEPTH variant of subgoal-following.** Filed (parked) as
  [`depth-subgoal-env`](../../parked/trained-policy/depth-subgoal-env.md)
  and the runtime side as
  [`depth-subgoal-hybrid-runtime`](../../parked/trained-policy/depth-subgoal-hybrid-runtime.md).
  Un-park triggers are spelled out in those briefs — primarily
  "this brief shipped AND NOCAM_SUBGOAL deployment evidence shows
  costmap-staleness / dynamic-obstacle failures depth would solve."
  Don't pre-empt; the NOCAM_SUBGOAL path may be sufficient for the
  deployment shapes that actually arise.
- **Goal-position noise on subgoals.** The
  [`goal-noise-training`](goal-noise-training.md)
  pattern likely applies (subgoals from Nav2's path have
  planner-resolution noise of ~5 cm, smaller than VLM noise but
  non-zero). Evaluate after the baseline subgoal checkpoint
  converges; file a `policy-subgoal-noise-training.md` follow-up
  if a noise pass is needed.
- **Multi-objective rewards (e.g. energy efficiency, comfort).**
  This brief targets a clean baseline subgoal-tracker. Additional
  reward terms can be layered in subsequent training passes.
- **Hyperparameter sweep over reward coefficients.** Reward
  *functions* are durable code (this brief); reward *coefficients*
  are tuning artifacts (Phase 5 training run). Significant
  coefficient changes that survive should land as separate
  training-prep briefs.
- **Replacing the existing GoalCommand-based ProcRoom envs.**
  The existing ProcRoom-NoCam / ProcRoom-Depth envs stay
  registered and continue to support the strafer_direct path.
  This brief adds new envs alongside, doesn't replace.
