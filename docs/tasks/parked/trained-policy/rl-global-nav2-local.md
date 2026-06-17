# Alternative architecture: RL as global waypoint planner + Nav2 local control

**Type:** investigation / research
**Owner:** Either (~DGX-lane training-env work + Jetson-lane runtime
work)
**Priority:** P3 — alternative to the
[`hybrid-mode`](hybrid-mode.md) direction. Not in critical path of any
shipping milestone. Filed-on-trigger: pick up only if the first
end-to-end deployment of `strafer_direct` (DEPTH MVP) or
`hybrid_nav2_strafer` (NOCAM_SUBGOAL) reveals that the current
"RL = local controller" division of labor isn't serving the deployment
needs — specifically when VLM-grounded missions need a global planner
that knows about *intent* (not just costmap geometry).
**Estimate:** L (~2 weeks: training-env design + new policy variant +
inference-side runtime + sim validation. Larger than its sibling hybrid
brief because the training env is a different shape, not just a
different command term.)
**Branch:** task/rl-global-nav2-local

## Story

As a **mission operator running VLM-grounded missions in unfamiliar
rooms (or rooms whose costmap is outdated)**, I want **the RL policy
to emit a sequence of intermediate waypoints that respect both goal
intent and local geometry, while Nav2's local controller executes
smooth per-waypoint motion**, so that **the policy's strength
(long-horizon geometric reasoning under noisy goals + depth) is
exposed at the planner layer where it does the most good, instead of
being limited to the local-controller layer where Nav2's MPPI already
performs adequately on short segments**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [inference-package.md](../../completed/inference-package.md)
  — the `strafer_direct` baseline this brief is the inverse of (RL =
  local control, Nav2 = no planner).
- [hybrid-mode.md](hybrid-mode.md) — the *Nav2-global + RL-local*
  hybrid this brief is the inverse of.
- [subgoal-env.md](../../completed/subgoal-env.md) — the
  closest training-env precedent. The variant proposed here
  (`PolicyVariant.DEPTH_WAYPOINT`) is a sibling of NOCAM_SUBGOAL with
  a different output target.

## Context

### The four-corner architecture matrix

|  | Local control by Nav2 | Local control by RL |
|---|---|---|
| **Global planning by Nav2** | shipped today | `hybrid_nav2_strafer` (parked, this epic) |
| **Global planning by RL** | **THIS BRIEF** | `strafer_direct` (DEPTH MVP — shipping) |

Three of the four corners have briefs. The fourth corner (this brief)
is unexplored. Worth understanding what it would unlock before
committing to one of the other three as the long-term direction.

### Why RL is good at the global planner role

The DEPTH MVP architecture treats the trained policy as a *local
controller*: given a goal pose + current depth, emit a body-frame
velocity command. This is the role classical MPC / MPPI is also good
at — the trained policy beats MPPI mostly because MPPI's critic
landscape plateaus on speed in cluttered scenes, not because
local-control is RL's natural home.

The role RL is *uniquely* good at is **long-horizon decision under
uncertainty**: deciding whether to go around an obstacle clockwise or
counterclockwise, deciding whether to attempt a doorway approach
head-on or at angle, deciding when to back up and re-approach. These
are *planning* decisions, not control decisions. Nav2's GridBased /
SmacHybrid planners do the same job but with a hand-coded heuristic;
they have no notion of:

- *Object intent* — "the goal is the chair" vs. "the goal is the
  general region near the chair." The RL planner can be trained
  against goal-position noise (per
  [`goal-noise-training`](../../active/trained-policy/goal-noise-training.md))
  so it implicitly handles "approximately reach here."
- *Locomotion constraints* — the chassis's actual reach-pattern
  (mecanum holonomic, smooth at certain angles, slow at others). A
  trained planner can emit waypoints the chassis reaches efficiently;
  a Nav2 planner emits geometrically-shortest waypoints the chassis
  may execute awkwardly.
- *Depth-derived discovered geometry* — Nav2's costmap is built from
  RTAB-Map + LiDAR (none on the Strafer) or D555 depth (slow update).
  The RL policy sees depth directly each tick; for unmapped
  obstacles, it reacts faster than the costmap can incorporate them.

### Peer comparison

- **FH-DRL** [arxiv:2407.18892](https://arxiv.org/html/2407.18892) —
  TD3 supplies waypoints to Nav2's controller. Exactly this corner.
  Reports better performance than Nav2-only on unknown environments.
- **Boston Dynamics Atlas** — high-level RL policy emits goal *poses*
  for a low-level controller. Same shape.
- **π0 / RT-2 / OpenVLA** hierarchical structure — VLA produces
  *trajectory tokens*, low-level controller executes. Same shape,
  different scope (VLA includes language).
- **Habitat-Sim2Real PointGoal baseline** — policy emits a discrete
  action (forward / turn-left / turn-right / stop), classical
  controller executes one step. Discrete sibling of this corner.

### What changes vs. the current direction

| Concern | Current direction (`strafer_direct` / `hybrid_nav2_strafer`) | This corner |
|---|---|---|
| Policy output | 3-D body velocity `(vx, vy, omega)` | N-D waypoint sequence `(x_1, y_1, ..., x_N, y_N)` in body frame |
| Inference rate | 30 Hz (matches sim decimation) | ~2 Hz (waypoint plans are slow-changing) |
| Local controller | RL itself | Nav2's MPPI or DWB |
| Training env | "Reach this goal" | "Emit waypoints that reach this goal" — needs reward shaping on path quality, not just goal-reach |
| Observation | Same as current (DEPTH + proprioception + goal) | Same shape; different reward chain |
| Action clamping | L1 + per-axis | Per-waypoint validity (within body-frame reach) |
| Watchdog | Inference brief's 5-source | Inference brief's 5-source + Nav2 controller's stall watchdog |

### Why this is parked, not active

Three reasons:

1. **The DEPTH MVP needs to ship first.** This brief is an
   architectural alternative; evaluating it before
   `strafer_direct` ships gives no data.
2. **N-dim waypoint regression is harder than 3-dim velocity
   regression for PPO.** Standard practice is to use a
   sequence-decoder (transformer) on top of the recurrent stack,
   which is a different model architecture from the current GRU
   stack. Bigger change than NOCAM_SUBGOAL.
3. **The trigger condition isn't observed yet.** We don't yet know
   that the local-control corner is *insufficient* for VLM-grounded
   missions. If `strafer_direct` + `goal-noise-training` is enough,
   this corner is wasted work.

The trigger to un-park: first VLM-grounded mission failure where the
root cause is "RL is doing local control but the issue is global
plan quality."

## Approach

Sketch only — flesh out at un-park time.

### Phase 1 — Training-env design

New training env `Isaac-Strafer-Nav-RLDepth-Waypoint-Real-v0`.
Differences from existing ProcRoom-Depth:

- Action space: N waypoint poses (N=8 likely; tune later) in body
  frame at policy time.
- Reward shaping: each waypoint contributes a small forward-progress
  reward; reaching the final waypoint contributes the large
  goal-reached bonus. Waypoint *validity* (not in obstacle, reachable
  from previous waypoint within chassis dynamics) is a soft penalty.
- The env *executes* the waypoints with a Nav2-equivalent local
  controller during training, so the policy learns plans that the
  classical controller can actually follow.

### Phase 2 — `PolicyVariant.DEPTH_WAYPOINT`

Add `_DEPTH_WAYPOINT_FIELDS` in `policy_interface.py`. Same input
shape as DEPTH; different output shape (N * 2 floats instead of 3).
The export pipeline handles this trivially — `action_dim` becomes a
config value.

### Phase 3 — Runtime: `strafer_inference` extension

New `STRAFER_NAV_BACKEND=strafer_waypoint_then_nav2_local`. Inference
node:
1. Subscribes same obs.
2. Each ~2 Hz tick, runs the policy → N waypoints.
3. Publishes a `nav_msgs/Path` to a Nav2 controller-only action
   server (`/follow_path`).
4. Watchdogs both layers.

### Phase 4 — Sim validation

Same cross-room reference mission as
[`hybrid-mode`](hybrid-mode.md). Compare:

- `strafer_direct` baseline
- `hybrid_nav2_strafer` baseline (if shipped)
- This corner

Metrics: time-to-goal, smoothness, collision rate, behavior on
deliberately-noisy goals.

## Acceptance criteria

*Filed-on-trigger; acceptance crystallizes at un-park.* Skeleton:

- [ ] Training env registered + smoke-tested.
- [ ] `PolicyVariant.DEPTH_WAYPOINT` defined; export pipeline handles
      the larger action_dim cleanly.
- [ ] Runtime backend `strafer_waypoint_then_nav2_local` ships in
      `strafer_inference`.
- [ ] Sim validation reports a quantitative comparison against
      `strafer_direct` and (if available) `hybrid_nav2_strafer` on a
      common reference mission.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit.

## Investigation pointers

- FH-DRL paper [arxiv:2407.18892](https://arxiv.org/html/2407.18892)
  — the closest peer architecture; their reward shaping + waypoint
  output design is directly transferable.
- Boston Dynamics Atlas hybrid stack — long-horizon high-level policy
  emitting pose targets; instructive for what "high-level" means at
  scale.
- [Nav2 controller-only action API](https://docs.nav2.org/) —
  `/follow_path` action server is the inference-side consumer.
- [`subgoal-env.md`](../../completed/subgoal-env.md) —
  closest in-repo training-env precedent. The differences are in the
  reward shaping (path-tracking vs. waypoint-emission) and in the
  output dim.

## Out of scope

- **Replacing `strafer_direct` or `hybrid_nav2_strafer`.** This is
  an additional corner, not a replacement. The four corners coexist.
- **Sequence transformer architecture in `strafer_lab`.** That's a
  separate experimental brief (parked under `experimental/`) if it
  ever matters. The MVP could use multi-head MLP regression for the
  N waypoints.
- **VLM-direct waypoint emission.** That's the VLA-v2 direction
  ([`parked/experimental/vla-v2-architecture.md`](../experimental/vla-v2-architecture.md));
  this brief is RL-in-the-loop, not language-in-the-loop.
- **Training infrastructure for sequence outputs.** rsl_rl supports
  arbitrary action_dim; no new infra needed at training time. Worth
  verifying during un-park.
