# Corner clearance for the rolling-subgoal path — planner-side, not inflation

**Type:** design / investigation (training-env + planner)
**Owner:** DGX (`strafer_lab` lane — ProcRoom `commands.py` waypoint generation;
the shared `path_planner` grid-A\* core stays untouched)
**Priority:** P3 — filed-on-trigger. Not in the critical path; the recurrent
arm is the first-line fix for corner-cutting.
**Estimate:** M (waypoint post-processing + clamping proof + unit tests; no
training-env distribution change if the lead candidate holds).
**Branch:** task/subgoal-corridor-clearance

**Trigger:** the recurrent (GRU) arm shipped in
[`nocam-subgoal-recurrent-runner`](../../completed/nocam-subgoal-recurrent-runner.md)
**still shows corner-cutting** at convergence — i.e. non-trivial
`sustained_collision` vs `path_complete` termination fractions and/or elevated
corner cross-track error on the two-arm play gate. If recurrence alone resolves
the cutting, this brief stays parked.

## Story

As a **`strafer_lab` operator whose `NOCAM_SUBGOAL` policy clips obstacle corners
even with a full anti-cutting penalty stack live**, I want **the training path
the policy tracks to carry corner clearance where the corridor has room for it**,
so that **a well-tracked corner is no longer a graze — without sealing the
procedural rooms' narrow doorways or biasing the generated scene distribution.**

## Why the obvious fix (a bigger inflation margin) is off the table

The original recurrence brief paired a corridor **tracking margin** (widen A\*
obstacle inflation from robot-radius to robot-radius + 0.10–0.15 m). That change
hit the doorway-passability STOP gate and is infeasible at this geometry. The
evidence, so a future picker does not re-propose it:

- ProcRoom's only *structured* guaranteed opening is a doorway of **min width
  0.8 m** (`mdp/proc_room.py`: `door_width = rand()*0.4 + 0.8`). Inflation is
  quantized to whole 0.1 m cells: `INFLATION_CELLS = ceil((ROBOT_HALF_WIDTH +
  margin) / GRID_RES)`, today `ceil(0.28/0.1) = 3` cells (0.3 m radius).

  | margin (m) | inflation cells | radius (m) | free gap through 0.8 m door |
  |---|---|---|---|
  | 0.00 (today) | 3 | 0.30 | 0.20 m |
  | ≤ 0.019 | 3 | 0.30 | 0.20 m (**no-op** — grid identical to today) |
  | 0.02 – 0.11 | 4 | 0.40 | **0.00 m — sealed** |
  | 0.15 | 5 | 0.50 | **−0.20 m — sealed** |

- The margin is **binary**: either no change (≤ 0.019 m) or +0.10 m radius
  (≥ 0.02 m), which seals the 0.8 m door. Even an idealized continuous planner
  leaves only `0.8 − 2·(0.28 + 0.10) = 0.04 m` center-slack at margin 0.10 — a
  **physical** incompatibility, not a discretization artifact.
- The doorway is a dead-end notch (the exterior is marked occupied), so the
  run-time A\* actually threads **interior furniture gaps**, which have **no
  guaranteed minimum width** — solvability is enforced post-hoc by a retry loop
  that strips objects. A 4-cell inflation shrinks those gaps by 0.2 m diameter,
  raising the (runtime-only, un-unit-testable) `path_fallback` rate and biasing
  the retry loop toward sparser rooms.
- Reframe: the 0.8 m door minus a 0.56 m robot leaves **±0.12 m** of total slack.
  That number **is the tracking-accuracy spec** the blind policy must meet.
  Corridors cannot widen; tracking must sharpen — which is the recurrent arm's
  job. There is no margin to add.

## Approach

### Lead candidate — medial-axis waypoint biasing (planner-side, no inflation change)

Keep inflation at the robot radius (3 cells) so doorways and solvability are
untouched. In `SubgoalCommand`'s waypoint post-processing (`perturb_waypoints`
in `mdp/commands.py` — ProcRoom infra, not the shared planner), bias each A\*
waypoint toward the **local free-space medial axis** by up to a target margin,
**clamped so it never leaves the inflated-free set**. Result: real clearance in
open areas, degrading **gracefully to zero in tight doorways** (where the medial
axis coincides with the path and there is no room to move). This delivers the
*intent* — corner clearance — with no doorway sealing and no fallback regression.

**Why this is more than "margin where available" — train/deploy path alignment.**
The training A\* runs on a *binary* free/occupied grid, so its shortest paths hug
the inside of corners at exactly the inflation radius. Nav2's planner — the
**deployment** path source for `hybrid_nav2_strafer` — runs on a costmap with
inflation-layer cost **decay**, which bows paths toward the corridor center.
Today's training paths are therefore systematically **more wall-hugging** than
the paths the deployed policy will be asked to track. Medial-axis biasing is thus
a **train/deploy path-distribution alignment** fix, not just cosmetic clearance —
which is what elevates it from nice-to-have to the correct follow-up if cutting
persists under the GRU arm.

Tests: assert biased waypoints stay inside the inflated-free set (clamp proof) on
a representative generated room; assert clearance increases in open corridors and
is ~0 through a min-width doorway; non-fallback path generation is unaffected.

### Rejected — widen the ProcRoom doorways (Option 2)

Widening `door_width` to ~[1.1, 1.5] m so a 4-cell inflation still leaves a gap is
**rejected on principle, not just scope.** Deployment doorways (the Infinigen
single-room scene, real homes) are ~0.8 m. Training on 1.1–1.5 m doors trains on
**unrepresentative geometry** and manufactures a sim-real gap to make a metric
look better. It also changes the env distribution for both training arms and does
not help the interior furniture-gap passages. Do not re-propose.

### Also here — decouple the off-path bound

`_SUBGOAL_MAX_OFF_PATH_M = INFLATION_CELLS * GRID_RES` (`strafer_env_cfg.py`)
couples the off-path termination/penalty corridor to the inflation cell count. It
was left untouched in the recurrence PR because, with no inflation change, it is
numerically identical to today (0.3 m) and decoupling would be churn. If any
clearance work here touches inflation semantics, pin the bound to the robot
radius (numerically 0.3 as today) and rewrite the comment so a future inflation
edit cannot silently widen the tracking tolerance. Verify no other consumer reads
`INFLATION_CELLS` expecting robot-radius semantics.

### Escape valve — accept recurrence + the live penalty stack (Option 4)

If the GRU arm's cutting is already within tolerance, do nothing: recurrence
(anticipation) plus the live penalty stack (cross_track −2, off_path −50,
collision −10/−5, obstacle_proximity −1) is the fix. This brief exists only for
the case where that is insufficient.

## Context bundle

- [`nocam-subgoal-recurrent-runner`](../../completed/nocam-subgoal-recurrent-runner.md)
  — the recurrence arm this brief is the parked sibling of; its two-arm play-gate
  metrics are this brief's trigger evaluation.
- [`subgoal-env`](../../completed/subgoal-env.md) — the grid-A\* planner +
  `SubgoalCommand` + `perturb_waypoints` this brief modifies.
- [`context/path-planning-architecture.md`](../../context/path-planning-architecture.md)
  — the shared grid-A\* spine (do **not** touch its core; the change is ProcRoom
  waypoint post-processing only).
- [`rl-global-nav2-local`](rl-global-nav2-local.md) — the Nav2/costmap deployment
  context behind the path-alignment argument above.
