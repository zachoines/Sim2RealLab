# Tame Nav2's wavy global path through the camera-blind-spot donut at mission start

**Status:** Shipped 2026-05-11 in `973e0a8` (Jetson). Multi-layer fix:
SLAM `Grid/RayTracing: "true"` fills depth-projection gaps that caused
the striated static map; a one-shot bringup-time `donut_warmup` node
in `strafer_bringup` (separate process, not in the navigate-to-pose BT,
because BT.CPP decorator state doesn't survive Nav2's halt/reset
between goals) rotates the chassis 360° once per launch to populate
the donut around `base_link`; the custom navigate-to-pose BT swaps
Nav2's stock `<RateController hz="1.0">` for `<DistanceController
distance="0.5">` so replanning is motion-gated, and inserts
`<SmoothPath smoother_id="simple_smoother">` between
`ComputePathToPose` and `FollowPath` to crush residual NavFn jaggies.
Sub-unity-RTF safety in the warmup loop uses a sim-time stall
detector rather than an absolute wall-clock cap. Validation runs
pending on live sim. Follow-ups filed in the same PR:
[`nav-deadline-sim-time-audit`](../active/reliability/nav-deadline-sim-time-audit.md),
[`executor-prefer-rotate-then-translate`](../active/reliability/executor-prefer-rotate-then-translate.md),
[`rtabmap-cold-start-determinism`](../active/reliability/rtabmap-cold-start-determinism.md),
[`windows-workstation-bringup`](../active/tooling/windows-workstation-bringup.md),
[`isaac-sim-rt-2-default-renderer`](../active/sim-performance/isaac-sim-rt-2-default-renderer.md).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/25

**Type:** task / bug
**Owner:** Jetson (Nav2 + costmap config + BT live in `strafer_navigation`)
**Priority:** P2
**Estimate:** M (~1–2 days; needs a reproducible path-shape capture + iteration on planner/costmap/BT)
**Branch:** task/nav2-startup-unknown-donut-path-noise

## Story

As a **mission operator running sim-in-the-loop missions on a freshly
mapped scene**, I want **the very first navigation goal after
`make clean-map` to follow a smooth path out of the robot's starting
pose**, so that **MPPI doesn't waste the first several seconds
oscillating laterally through an unknown costmap donut, with
`/strafer/odom.linear.x` averaging ~0 m/s while `/cmd_vel` thrashes
between forward / strafe / rotate commands**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../context/bridge-runtime-invariants.md)
- [mppi-critic-tuning-for-sim-envelope.md](../completed/mppi-critic-tuning-for-sim-envelope.md)
  — the sibling brief whose B-scenario validation surfaced this. The
  MPPI critic-tuning work makes the controller more robust to path
  noise, but the planner-side root cause documented here is upstream
  and out of scope there.

## Context

**Symptom.** On the first `translate forward 3 m` mission of a
session — `make kill && make clean-map && make launch-sim`, then
`strafer-autonomy-cli submit "translate forward 3 m"` — the robot
exhibits multi-second oscillation at mission start: short forward
nudges interleaved with sideways strafing and small rotations, total
displacement well under the goal distance, mission times out at
60 s (or 180 s after `plan-compiler-skill-timeouts.md` lands). The
operator can see in Foxglove that the global path itself has visible
kinks near the start pose and near the goal — not a smooth straight
line.

**B-pass 1 capture data (2026-05-01, gamma=0.008 sweep):**

| topic | axis | peak inst | 1 s sustained max | 1 s sustained median |
|---|---|---|---|---|
| /cmd_vel | vx | 0.889 | 0.441 | 0.343 |
| /cmd_vel | vy | **1.111** | 0.398 | 0.239 |
| /cmd_vel | wz | 0.669 | 0.286 | 0.158 |
| /strafer/odom | vx | 0.395 | 0.309 | **0.001** |
| /strafer/odom | vy | 0.387 | 0.047 | 0.001 |
| /strafer/odom | wz | 0.455 | 0.249 | 0.002 |

Two facts the table makes hard to ignore on a translate-forward-only
mission:
1. **Commanded `vy` peak (1.111) > `vx` peak (0.889).** MPPI is being
   told to strafe sideways harder than it's being told to go forward.
   That's only sensible if the global path is *asking for* lateral
   motion — i.e. it has lateral kinks.
2. **Observed `vx` sustained median ≈ 0.001 m/s.** Despite the
   non-trivial commanded magnitudes above, net displacement averages
   to zero. The wheels are getting conflicting cmd_vel updates fast
   enough that real motion cancels itself out.

**Root cause (operator hypothesis, supported by config inspection).**
On a fresh-mapped session the only known-free cells in the global
costmap are the ones the D555 has already observed since launch. The
camera has a blind-spot donut around the robot's footprint (no cells
inside ~0.4–0.5 m radius are observed at the robot's own pose). Those
cells are *unknown*, not *free*.

`strafer_navigation/config/nav2_params.yaml`:
- `global_costmap.track_unknown_space: true` — unknown cells are
  costmap-distinct from free, with default cost 255.
- `planner_server.GridBased.allow_unknown: true` — the NavfnPlanner
  is *allowed* to traverse unknown cells.
- No `nav2_smooth_path_action_bt_node` in the default
  `navigate_to_pose_w_replanning_and_recovery.xml` BT (the plugin
  *is* listed in `bt_navigator.plugin_lib_names` line 31, so it's
  *available* — just not called by the active BT).

So the planner produces a grid-discrete shortest path from robot
center, through a band of unknown cells, to the first free cells the
camera observed. Cost in the unknown band is uniform, which makes the
shortest-path heuristic indifferent across many neighboring cell
sequences — the result is a path with grid jaggies through the
donut. PathAlignCritic at `cost_weight=9.0` (sim) then *demands*
MPPI track those jaggies, which produces the sideways-and-rotate
oscillation that overwhelms the forward bias from PreferForwardCritic
even at `cost_weight=10.0`.

The MPPI critic-tuning brief
([`mppi-critic-tuning-for-sim-envelope.md`](../completed/mppi-critic-tuning-for-sim-envelope.md))
can attenuate the symptom — making MPPI less reactive to small path
kinks — but cannot fix the input. As long as the planner emits a
wavy path, the controller is stuck choosing between "follow the wave"
(oscillation) and "ignore the path" (drives off-path → PathAlign
penalty wins anyway).

## Approach

Four candidate fixes, ranked by separation-of-concerns and
implementation cost:

### A. Add a `nav2_smooth_path_action_bt_node` step to the BT (recommended starting point)

The plugin is already linked in. Use the
`navigate_to_pose_w_replanning_and_recovery.xml` BT as a starting
point and inject a `<SmoothPath>` between the planner and the
follower. `simple_smoother` with `max_its: 1000`,
`tolerance: 1e-10` (already configured under `smoother_server`)
should crush the grid jaggies into a clean curve.

Risk: the smoother itself may over-smooth corners on far-goal
missions (the `nav2-far-goal-staging.md` reference). Acceptance
must include a no-regression run of that mission.

### B. Pre-mission "look around" recovery

Before the first navigation goal of a session, prepend a
`rotate_by_degrees 360` to fill in the camera blind-spot donut. Two
shapes:

- **Plan-compiler level** — `plan_compiler` detects "first nav goal
  since launch" and prepends a rotate. Lives in DGX lane and adds a
  cross-host coupling.
- **Behavior-tree level** — a one-shot `Spin 6.28` recovery before
  any navigation goal. Cleaner; lives in the BT XML.

Risk: adds 2–4 s of sim time per session before the first mission.
Doesn't help if the *second* mission is far enough that the donut
between the new pose and the goal is also unknown.

### C. Tune unknown-cell cost in the global costmap

Currently unknown cells have implicit cost. We could:
- Set `track_unknown_space: false` — treats unknown as free. Risky
  for real-robot (might plan through walls). Sim-only override
  lives the same way the v2 critic gating does in `_patch_params`,
  but real-robot regression is the concern; better caught by
  acceptance tests.
- Add an `inflation_layer` whose `cost_scaling_factor` is tuned
  high enough that the unknown band has a strong gradient toward
  observed-free cells, so the planner picks a clean path along the
  free/unknown boundary instead of straight-shotting through the
  noisy interior.

### D. Switch the global planner

From `nav2_navfn_planner/NavfnPlanner` (Dijkstra-style on grid) to
`nav2_smac_planner/SmacPlannerHybrid` or `SmacPlanner2D`. SMAC
produces smoother paths natively. Higher CPU; needs config reauthor.
Largest blast radius.

**Recommended sequence:** A first (smallest change, highest
expected payoff). If A leaves residual oscillation, layer C on top.
B as a fallback if A+C aren't sufficient. D only if all of the
above fail.

## Acceptance criteria

- [ ] On a `make kill && make clean-map && make launch-sim` cold
      start, a `translate forward 3 m` mission completes inside the
      configured timeout with `/strafer/odom.linear.x` 1 s sustained
      median ≥ 0.5 m/s (≥ 50× the B-pass 1 baseline of 0.001 m/s
      from the table above).
- [ ] On the same mission, `/cmd_vel.vy` peak inst ≤ `/cmd_vel.vx`
      peak inst — i.e. the lateral-overrides-forward inversion is
      gone. Capture via the `tune_capture.py` harness and paste the
      table into the PR description.
- [ ] No regression on the `completed/nav2-far-goal-staging.md`
      reference mission ("Navigate to the open wood door on other
      side of the room"): mission still completes end-to-end with
      ≥ 2 staging legs and the executor still reports ≥ 2
      intermediate Nav2 goals.
- [ ] No regression on cornering: the `mppi-critic-tuning-for-sim-envelope.md`
      90° heading-change check still lands within
      `xy_goal_tolerance: 0.15` and `yaw_goal_tolerance: 0.20`.
- [ ] Real-robot bringup is unaffected. If the chosen approach
      requires a sim-only override (e.g. `track_unknown_space: false`
      under option C), gate it on `envelope_factor > 1.0` mirroring
      the existing pattern in `navigation.launch.py:_patch_params`,
      and assert the gate via a unit test alongside.
- [ ] Unit tests cover whichever option ships, with at least one
      assertion per knob touched (BT XML byte-comparison, costmap
      param presence, etc.).
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml`:
  - lines 197–211 — `planner_server` config (`allow_unknown`, plugin)
  - lines 213–225 — `smoother_server` config (already present, just
    not wired into the active BT)
  - lines 295–334 — `global_costmap` (`track_unknown_space`,
    `inflation_layer`, `static_layer` from `/rtabmap/map`)
  - lines 19–76 — `bt_navigator` plugin list (smoother BT node is
    available at line 31)
- `nav2_bt_navigator` ships several reference BTs under
  `/opt/ros/humble/share/nav2_bt_navigator/behavior_trees/`. The
  default `navigate_to_pose_w_replanning_and_recovery.xml` is the
  starting point; a custom variant with smoother gets dropped under
  `strafer_navigation/config/` and selected via `default_nav_to_pose_bt_xml`.
- `mppi-critic-tuning-for-sim-envelope.md`'s
  `tune_capture.py` harness at
  [`source/strafer_ros/strafer_navigation/scripts/tune_capture.py`](../../../source/strafer_ros/strafer_navigation/scripts/tune_capture.py)
  is the validation tool. Capture at 30 s `--duration` covers the
  mission's active phase.
- The "kink in the path" claim is verifiable in Foxglove by
  subscribing to `/plan` (the global path Nav2 publishes) and
  visualizing it as a Path display alongside the costmap.

## Out of scope

- **MPPI critic re-tuning.** That's
  [`mppi-critic-tuning-for-sim-envelope.md`](../completed/mppi-critic-tuning-for-sim-envelope.md).
  This brief assumes the controller is doing its job given a sane
  path; the fix here is to make the path sane.
- **`make clean-map` policy.** Whether the operator should be
  starting cold-mapped is a workflow question, not a controller
  question. Briefs touching that should land separately.
- **Real-robot path-noise.** The donut symptom is amplified in sim
  by the cold-start `make clean-map` workflow. On the real robot,
  RTAB-Map state typically persists across sessions and the donut
  is filled in from prior runs. If real-robot regressions surface,
  file a third brief; don't piggy-back on this one.
- **Plan-compiler timeout hardcodes.** That's
  [`plan-compiler-skill-timeouts.md`](plan-compiler-skill-timeouts.md);
  separate path. If validation here reveals the 60 s `translate`
  timeout is the proximate failure (vs. genuine controller wedge),
  that brief lands first.
