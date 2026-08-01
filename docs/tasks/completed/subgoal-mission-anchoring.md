# Anchor one path per mission so cross-track error can develop

**Status:** Shipped 2026-07-31 in `79d4646` (Jetson). Filed and shipped in
the same PR — the work came from the 2026-07-31 obs-parity decision session, not
from the queue, so there was never a window in which an active row would have
been true.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/174

**Type:** bug (train↔deploy semantics)
**Owner:** Jetson
**Priority:** P0 — it explains the both-policy subgoal-tracking failure and
needs no retrain.
**Estimate:** M
**Branch:** `task/subgoal-mission-anchoring`

## Story

As a **subgoal-following policy deployed behind Nav2**, I need **the path I am
tracking to stay where it was planned**, so that **when I drift off it I am told
so — the corrective lateral signal training taught me to consume.**

## Context bundle

- [context/conventions.md](../context/conventions.md)
- [context/path-planning-architecture.md](../context/path-planning-architecture.md)
- [context/recurrent-policy-contract.md](../context/recurrent-policy-contract.md)
- [context/nav2-config-parity.md](../context/nav2-config-parity.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context — the symptom, then the diagnosis

Both the v1 and v2 depth-subgoal policies failed to track their subgoal on the
rig, at 55–95° of heading error. The 2026-07-31 obs-parity session ruled the
observation channel clean (no signed left/right asymmetry anywhere in the obs
vector; the subgoal triplet is exact to 1.7e-8 rad), which left the tracking
failure unexplained by anything the policy could see — and then measured why.

**Measured, all 8 prior missions:** every published `/plan` starts under the
robot. Median `|plan[0] − robot|` 0.032–0.044 m with **no growth over 5.95 m of
travel** (mission `mB` first→last quartile 0.0398 → 0.0397 m); cross-track
median 0.029–0.044 m, never more than one `MAP_RESOLUTION` cell. On the
preserved parity bag the deploy generator re-installed a path every **2.29**
subgoal ticks (511 installs / 1172 ticks) and the "monotonic" cursor never
exceeded **7.8 cm** of arc over a 5.95 m mission (mean 1.30 cm).

Two mechanisms compound:

1. `ComputePathToPose` is requested with `use_start=False`, so every plan is
   computed from the robot's *current* pose.
2. Nav2's `planner_server` republishes `/plan` at **~12 Hz** independently of
   the node's own 0.5 s replan cadence — so "a plan arrived" carries no
   information at all, and the node installed every one of them.

The result is a path continuously re-rooted under the robot. The robot is never
allowed to BE off its path, so no corrective lateral signal can ever accumulate,
and the monotonic cursor is a no-op.

**Training does the opposite** (`SubgoalCommand._resample_command` →
`PathCursor.set_paths` at goal resample; `_update_command` only advances the
cursor; `resampling_time_range=(1e6, 1e6)`, i.e. once per episode). One path per
goal, anchored in the world frame.

## What shipped

**`generator.py`** gains the pure pieces anchoring needs, all rclpy-free:
`path_arc_lengths`, `arc_length_projection` (non-mutating closest-point
projection usable on a *candidate* path), `RollingSubgoalGenerator.project`,
`set_path(..., initial_cursor=)`, and `evaluate_admission` — the ruled predicate
with stable reason codes. `update()` now shares the one projection
implementation instead of carrying its own copy.

**`subgoal_generator_node.py`** installs the semantics:

- **`subgoal_anchoring: mission`** (shipped default). The first plan of a
  mission is anchored; later plans are planner liveness and are discarded unless
  admitted.
- **Admission classes** — `no_anchor`, `goal_changed` (including a mission
  boundary whose new goal happens to sit inside the goal-provenance tolerance),
  `anchor_in_collision` (the *remaining* anchored path crosses cells at or above
  the costmap's inscribed threshold), `cross_track_exceeded`
  (> `admission_cross_track_m`, default 0.5 m — just under the global costmap's
  0.55 m `inflation_radius` and ~15× the cross-track measured under the old
  behaviour). Precedence is fixed and unit-tested.
- **Plans are keyed on content + stamp, not arrival**, so the ~12 Hz republish
  is deduped and counted separately.
- **On admission the cursor is seeded by projection, never rewound**, so
  progress-toward-goal survives the replacement.
- **`subgoal_anchoring: rolling`** is the named legacy fallback, so the rig
  re-validation can A/B the two semantics without editing code.
- A periodic `anchor status:` line reports admitted / held / by-reason counts,
  cursor, and cross-track, so the semantics are legible from a normal mission
  log.

**The RC-6 guards keep their exact semantics.** Plan *liveness*
(`_last_plan_rx_t`, the input to `_plan_fresh`) is refreshed by any valid plan
for the active goal whether or not it is admitted, and a rejected plan still
clears the refusal streak and re-arms the starvation hold. A refusing planner
therefore starves the guards exactly as before. All 15 RC-6 tests and the #171
selector tests pass untouched.

## Acceptance criteria

- [x] One path anchored per mission; a fresh plan replaces it only under a
      ruled, unit-tested admission predicate.
- [x] The cursor is monotonic over an anchored path, and remaining-arc never
      jumps backwards across an admitted replacement.
- [x] Cross-track develops on an anchored path (> 0.3 m in the node-level drive
      test) and is provably pinned at 0 under `rolling` — the A/B is a test, not
      a claim.
- [x] Admission on plan content/stamp, not arrival; republishes counted
      separately.
- [x] RC-6 guards (`fallback_planner_id`, `starvation_hold_s`) unchanged in
      semantics, their tests unmodified.
- [x] No obs-assembly change anywhere.
- [x] Config-gated with the old behaviour as `subgoal_anchoring: rolling`.
- [x] User-facing docs updated in the same commit
      (`source/strafer_ros/README.md`).

## Out of scope

- Any change to the inference node's obs assembly.
- The #171 start-cell planner selector (topic-isolated from this node;
  `test_nav_config.py` actively forbids the generator gaining a selector topic).
- The behavioural re-validation itself — filed as
  [`subgoal-anchoring-rig-revalidation`](../active/trained-policy/subgoal-anchoring-rig-revalidation.md).
- The v2 signed leftward bias, which the same session routed to the training
  lane and which this change cannot and does not address.

## Known follow-up surfaced by this work

`max_path_points` becomes newly load-bearing: under `rolling` the installed path
was always a fresh short plan, so head-first truncation was unreachable in
practice. Anchoring one path per mission makes a long path persist. The shipped
default is `0` (unbounded), which is correct — but a non-zero value would now
silently truncate a mission's route. Called out here rather than changed.
