# Recover from mid-mission grounding loss instead of terminating

**Type:** task / enhancement
**Owner:** Jetson agent (`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`)
**Priority:** P2
**Estimate:** M (~1–2 days; new recovery sub-skill + cancel-aware
fallback policy + per-stage retry budget + tests)
**Branch:** task/executor-grounding-loss-mid-mission-recovery

## Story

As a **mission operator running `go_to_target` against a target that
was visible at scan-time but is no longer framed at staging-arrival-time
(viewpoint changed, robot drifted off the approach vector, target
occluded by an intermediate obstacle)**, I want **the executor to try
a bounded recovery (small scan, semantic-map fallback) before failing
the mission with `navigate_via_staging_target_lost`**, so that **a
mission doesn't abort the first time the camera momentarily loses the
object that's still standing right there**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/nav2-far-goal-staging.md](../../completed/nav2-far-goal-staging.md)
  — the executor staging loop this brief extends.
- [completed/align-after-scan-grounding.md](../../completed/align-after-scan-grounding.md)
  — predecessor that put the bearing math the recovery path will reuse.

## Context

`_navigate_via_staging`
([`mission_runner.py:1066-1365`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py))
runs a per-stage loop: dispatch a Nav2 goal to the next leg, capture
on arrival, re-ground the original target label, re-project, repeat.
When re-grounding returns `found=False`, the executor terminates the
mission immediately:

```python
# mission_runner.py:1285-1300
if not grounding.found or grounding.bbox_2d is None:
    stage_entry["regrounding_status"] = "target_not_found"
    stage_log.append(stage_entry)
    return self._failed_result(
        step,
        message=(
            f"Staging leg {stage_idx + 1}: target '{label}' not "
            "re-grounded after arrival."
        ),
        error_code="navigate_via_staging_target_lost",
        ...
```

This is brittle. The reasons the VLM might miss the target at
re-ground time but not at original scan time include:

1. **Heading drift between projection and arrival.** The standoff
   pose's yaw faces *the projected goal* but Nav2's
   `general_goal_checker.yaw_goal_tolerance: 0.20` rad allows arrival
   ~11° off-axis. Combined with mecanum slip (no slip detection
   today — see [imu-yaw-drift-no-magnetometer.md](../../parked/reliability/imu-yaw-drift-no-magnetometer.md)),
   the object can sit just outside the camera's 87° horizontal FOV.
2. **Partial occlusion** by an intermediate obstacle that wasn't on
   the path at scan time (a chair pushed in, a door that closed).
3. **VLM false negative.** Qwen2.5-VL drops detections under heavy
   motion blur or against backgrounds with similar texture; one bad
   inference shouldn't kill a mission.
4. **Lighting change** between scan and arrival (room lights toggled,
   shadow moved). The CLIP-stored semantic-map entry from
   `_store_scan_observation` still locates the *region*, even if
   instantaneous grounding fails.

The current "fail immediately" policy gives up information the system
already has: the goal pose from the prior leg is still on the
costmap, the semantic-map node at the goal location still embeds the
correct label, and a small recovery scan is much cheaper than aborting
the mission.

State-of-the-art VLA peer work has converged on detect-then-recover
patterns: OneTwoVLA explicitly "detects execution errors in real time,
reasons about corrective strategies, and performs agile recovery
actions"
([one-two-vla.github.io](https://one-two-vla.github.io/)); π0.5 and
GR00T N1.5 incorporate recovery via their planning layer
([pi.website/download/pi05.pdf](https://www.pi.website/download/pi05.pdf),
[arxiv.org/pdf/2503.14734](https://arxiv.org/pdf/2503.14734)).
The Strafer stack isn't a VLA, but the principle transfers: a
single failed inference is not authoritative; bound a recovery before
escalating.

## Approach

### A. Bounded local recovery before terminating (recommended)

In `_navigate_via_staging`, on re-grounding `found=False`, run a
recovery sequence with explicit per-stage budget:

1. **Mini-scan** — invoke `_scan_for_target` with
   `max_scan_steps=3, scan_arc_deg=120` (i.e., sweep ±60° from the
   current heading). Cheaper than the default 6-step / 360° scan
   because we have a strong prior on the bearing.
2. **Semantic-map fallback** — if the mini-scan also fails, query
   `self._semantic_map.query_by_label(label, max_age_s=300)` (the
   same primitive `_try_query_before_scan` uses) and check whether
   any cached node sits within `goal_radius_m` of the current leg's
   goal pose. If so, accept the cached pose as the goal — the robot
   is at the right *place*, the VLM just isn't framing the object
   this instant. Mark the mission outcome
   `outputs["recovery"] = "semantic_map_fallback"` so the operator can
   audit.
3. **Escalate** — if both fail, return
   `navigate_via_staging_target_lost` with the recovery attempts
   recorded in `stage_log`. Budget: max 2 recovery attempts per
   mission across stages (env knob
   `STRAFER_NAV_GROUNDING_RECOVERY_BUDGET`, default 2). The budget
   prevents thrash on a genuinely-gone target.

### B. Re-plan with a widened prompt

Re-issue grounding with a backed-off prompt (e.g.
`"any object matching '{label}'"` instead of the planner's specific
phrase). Cheap if the planner is on-host. Lower confidence than (A2)
because we have no proof the new prompt is more reliable. Could layer
on top of (A1).

### C. Spawn a frontier-exploration leg

Drive to the nearest unexplored frontier cell within
`goal_radius_m` and re-scan. This is a multi-room-epic primitive
([`frontier-exploration-primitive.md`](../../completed/frontier-exploration-primitive.md)),
not a single-room recovery. Out of scope here; a target that's
genuinely *not visible from anywhere on the current costmap* is a
multi-room mission planning problem.

**Recommended sequence:** A1 + A2 with shared budget. Skip B and C
for now — A1+A2 already covers the dominant failure modes
(transient VLM miss, FOV drift, instantaneous occlusion).

## Acceptance criteria

- [ ] `_navigate_via_staging` re-grounding-failure path no longer
      terminates immediately on first `found=False`. It invokes the
      A1 mini-scan first.
- [ ] Mini-scan is configured at the call site with
      `max_scan_steps=3, scan_arc_deg=120` (the operator can override
      via env: `STRAFER_NAV_RECOVERY_SCAN_STEPS`,
      `STRAFER_NAV_RECOVERY_SCAN_ARC_DEG`).
- [ ] On A1 failure, `_navigate_via_staging` queries
      `self._semantic_map.query_by_label(label, max_age_s=300)`. If a
      cached node within `goal_radius_m` of the current leg's goal
      exists, the leg is treated as succeeded with
      `outputs["recovery"] = "semantic_map_fallback"`. Otherwise the
      escalation path runs.
- [ ] Per-mission recovery budget: at most 2 recovery attempts across
      all stages, controlled by
      `STRAFER_NAV_GROUNDING_RECOVERY_BUDGET` (default 2). Exhausting
      it returns `navigate_via_staging_target_lost` with
      `outputs["recovery_attempted"] = N`.
- [ ] `cancel_event` honored throughout recovery — a cancel during
      the mini-scan or semantic-map query terminates with
      `error_code="mission_canceled"`, same semantics as
      `_scan_for_target`'s in-loop cancel check
      ([`mission_runner.py:721`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)).
- [ ] Unit tests cover: (a) mini-scan succeeds, leg resumes;
      (b) mini-scan fails + semantic-map fallback succeeds, leg
      resumes with `recovery="semantic_map_fallback"`;
      (c) both fail, leg fails with `navigate_via_staging_target_lost`
      and `recovery_attempted=1`;
      (d) budget exhausted, second-stage failure short-circuits without
      attempting recovery.
- [ ] Integration smoke (sim): a `go_to_target` mission against a
      target deliberately partially-occluded mid-mission (place a
      box in the path at runtime) completes via semantic-map fallback
      rather than aborting.
- [ ] No regression on
      [`nav2-far-goal-staging`](../../completed/nav2-far-goal-staging.md)
      reference mission: clean missions with no recovery trigger are
      bit-identical.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:1066-1365`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py) —
  `_navigate_via_staging` body. Recovery branch hooks in at line 1285.
- [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:620-774`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py) —
  `_scan_for_target`. Can be called recursively with overridden args
  (`max_scan_steps`, `scan_arc_deg`); the cancel-event check at line
  721 must work from within a recovery context.
- [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:2035-2070`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py) —
  `_try_query_before_scan`. Reuse the semantic-map query primitive
  rather than re-implementing.
- Peer work on detect-then-recover for foundation models:
  [OneTwoVLA](https://one-two-vla.github.io/),
  [π0.5 paper](https://www.pi.website/download/pi05.pdf),
  [GR00T N1 whitepaper](https://arxiv.org/pdf/2503.14734).
- Related parked: [`verify-arrival-occlusion-robustness.md`](verify-arrival-occlusion-robustness.md)
  — same family of failures at the *arrival* checkpoint.

## Out of scope

- **Cross-room recovery.** If the target is not visible from anywhere
  on the current costmap, the right escalation is a frontier explore,
  which belongs in the multi-room epic. This brief is single-room.
- **Re-tuning the VLM grounding confidence threshold
  (`min_grounding_confidence`).** Adjacent concern; lives separately.
- **CLIP arrival-verification under occlusion.** Different code path
  (`_verify_arrival`); separate brief filed at
  [`verify-arrival-occlusion-robustness.md`](verify-arrival-occlusion-robustness.md).
- **Replacing the `navigate_via_staging_target_lost` error code.** The
  code stays; this brief reduces *how often* it fires.
