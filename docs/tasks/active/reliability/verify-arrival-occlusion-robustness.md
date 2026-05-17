# `verify_arrival` false-negatives under partial occlusion

**Type:** task / bug
**Owner:** Jetson agent (`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`)
**Priority:** P2
**Estimate:** S–M (~1 day; tighten the verification semantics + add a
re-look-from-tilt fallback + tests)
**Branch:** task/verify-arrival-occlusion-robustness

## Story

As a **mission operator running `go_to_target` against a target
partially occluded at arrival (chair behind a box, door framed by
clutter, person partially behind a column)**, I want **`_verify_arrival`
to either confirm the arrival or fail with a recoverable error code
(`arrival_occluded`), not silently mis-classify a successful arrival
as `arrival_verification_failed`**, so that **end-to-end mission success
rate doesn't drop by the false-negative rate of CLIP under
real-world clutter**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/mid-mission-validation-investigation.md](../../completed/mid-mission-validation-investigation.md)
  — the design doc that audited CLIP usage and chose the cascade path.
- [completed/learned-mid-mission-validator.md](../../completed/learned-mid-mission-validator.md)
  — the retired alternative; explains why this brief stays narrow
  rather than proposing a new validator backbone.
- The active `clip-validation/validator-evaluation` brief is the home
  for backbone-bakeoff / threshold-tuning work; this brief is the
  *surgical fix* that lands before the bakeoff completes.

## Context

`_verify_arrival`
([`mission_runner.py:1424-1533`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py))
embeds the arrival camera frame with CLIP and queries the semantic
map. If fewer than `majority` (default 3) of the top-K (default 5)
results sit within `goal_radius_m` (default 3.0 m) of the projected
goal pose, the mission fails:

```python
# mission_runner.py:1519-1533
return self._failed_result(
    step,
    message=(
        f"Arrival mismatch: only {near_goal_count}/{top_k} top matches "
        f"near goal. Top-3 matches at: {', '.join(top_regions)}."
    ),
    error_code="arrival_verification_failed",
    ...
)
```

Three structural problems under partial occlusion:

1. **CLIP embedding drift under occlusion.** CLIP is trained on
   web-crawled images that *do* contain occluded objects, but the
   embedding's nearest-neighbors in our stored semantic map are
   captures of the same object *not occluded* (scan-time
   `_store_scan_observation`). The cosine similarity drops; the
   top-K shifts toward other regions whose embeddings happen to
   include the occluding clutter.

2. **Single-frame decision.** The verification is one camera grab at
   one pose. The robot is already at the goal — a 30° tilt or a 0.3 m
   side-step would reframe the target without re-running the entire
   mission. Today: no fallback. One bad frame = mission failure.

3. **`arrival_verification_failed` is terminal.** Today the operator
   gets back a `failed` mission with no actionable next step. The
   semantic map *already* contains the right node; the embedding
   distance just happened to be marginal this instant. The error
   code's failure shape doesn't tell the operator "the robot is
   plausibly at the goal but the camera frame is ambiguous" — it
   reads identical to "wrong place."

The
[`mid-mission-validation-investigation.md`](../../completed/mid-mission-validation-investigation.md)
shipped design doc anticipated this — its Section 3 alternatives
survey lists "smarter VLM scheduling" and "cascade improvements" as
peer directions. Cascade-arbiter and retrieval-augmented improvements
land under
[`clip-validation/validator-evaluation`](../clip-validation/validator-evaluation.md);
they take longer. This brief is the *short-term* fix that reduces
operator-visible false negatives without waiting for the bakeoff.

Peer SOTA for occlusion-robust arrival verification:
- π0.5 uses high-level semantic prediction + co-trained
  retrieval-augmented inference to verify task progress under partial
  observability ([pi.website/download/pi05.pdf](https://www.pi.website/download/pi05.pdf)).
- OpenVLA-style backbones improve on CLIP for object grounding under
  partial views ([openvla.github.io](https://openvla.github.io/)).
- Production teams (Mobile-ALOHA, GR00T deployments) commonly use
  multi-frame voting or a small re-position-then-verify policy when
  single-frame verification is ambiguous.

All three concur: bound the verification to a small recovery rather
than gate the mission on one frame.

## Approach

### A. Multi-frame voting + re-position fallback (recommended)

1. **Multi-frame capture.** Replace the single-frame embed with
   `N_FRAMES` (default 3) captures at `t=0`, `t=0.3 s`, `t=0.6 s`
   sim-time. Embed each, mean-pool the embeddings, query once.
   Catches motion blur / transient lighting; adds ~1 s sim-time.
2. **Re-position fallback on initial verification failure.**
   If the multi-frame query still fails the majority check, issue
   a small recovery sequence:
   - Tilt by `±15°` yaw (left or right depending on which side has
     more known-free costmap cells inside `0.5 m`),
   - Re-capture multi-frame, re-embed, re-query.
3. **Distinguish three terminal states:**
   - `verified=True` (majority passes either pre- or post-tilt).
   - `error_code="arrival_occluded"` — multi-frame post-tilt
     verification fails, **but** `query_by_label(label,
     max_age_s=300)` returns a cached node within `goal_radius_m`
     of the current pose. The robot is plausibly at the goal; the
     instantaneous view is ambiguous. Surface as a *soft failure*
     the operator can choose to accept.
   - `error_code="arrival_verification_failed"` — none of the above.
     The robot is genuinely at the wrong place.

### B. Lower the `majority` threshold to 2 / raise `goal_radius_m`

Smallest possible fix: relax the existing thresholds so single-frame
CLIP wins more often. Loses precision (now more *false positives*
where the robot accepts a wrong place); rejected as the primary
fix.

### C. Defer to `validator-evaluation` bakeoff

Wait until the bakeoff picks a better backbone or a cascade-arbiter
ships. Rejected: too slow. The bakeoff is P1 but it's
backbone-bakeoff scope, not a few-day verify-arrival fix.

**Recommended:** A.

## Acceptance criteria

- [ ] `_verify_arrival` captures `STRAFER_VERIFY_ARRIVAL_N_FRAMES`
      (default 3) frames at `STRAFER_VERIFY_ARRIVAL_FRAME_INTERVAL_S`
      (default 0.3 s sim-time) and mean-pools the CLIP embeddings
      before querying the semantic map.
- [ ] On initial failure (`near_goal_count < majority`), the executor
      runs a re-position fallback: yaw `±15°` toward the side with
      more known-free costmap cells within 0.5 m, then re-captures and
      re-queries.
- [ ] If post-tilt verification also fails *but*
      `query_by_label(label, max_age_s=300)` returns a node within
      `goal_radius_m` of current pose, return
      `error_code="arrival_occluded"`,
      `outputs={"verified": False, "soft_failure": True,
      "cached_node_distance_m": <float>}`.
- [ ] Otherwise return the existing
      `error_code="arrival_verification_failed"` — semantics unchanged
      for the hard-failure case.
- [ ] Cancel-event honored at every capture point (mirroring
      [`_scan_for_target`'s in-loop check](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)).
- [ ] Unit tests cover: (a) multi-frame passes initial verification,
      (b) initial fails + tilt-recovery passes, (c) both fail + cached
      node nearby → `arrival_occluded`, (d) both fail + no cached
      node → `arrival_verification_failed`, (e) cancel during
      multi-frame capture.
- [ ] Logging: each branch logs `{"verify_path": "single|tilt",
      "near_goal_count": N, "top_k": K, "soft_failure": bool}` so
      operator audit can reconstruct which path fired.
- [ ] No regression on
      [`align-after-scan-grounding`](../../completed/align-after-scan-grounding.md)
      reference mission. The verify path is the same on the happy
      case.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:1424-1533`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py) —
  `_verify_arrival` body. Multi-frame capture replaces the single
  `capture_scene_observation` call at line 1441; tilt-recovery is a
  new branch after line 1525.
- [`source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:2035-2070`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py) —
  `_try_query_before_scan` shows the cached-node-query pattern to
  mirror for the soft-failure check.
- [`docs/tasks/active/clip-validation/validator-evaluation.md`](../clip-validation/validator-evaluation.md) —
  the broader bakeoff; this brief is the surgical landing-strip while
  the bakeoff runs.
- The shipped design doc:
  [`docs/MISSION_VALIDATION_ARCHITECTURE.md`](../../../MISSION_VALIDATION_ARCHITECTURE.md)
  Section 2 (limitations) and Section 4 (recommendation).
- Peer references on occlusion-robust verification: π0.5
  ([pi.website/download/pi05.pdf](https://www.pi.website/download/pi05.pdf)),
  OpenVLA
  ([openvla.github.io](https://openvla.github.io/)).

## Out of scope

- **Replacing CLIP with a better backbone.** That's
  [`clip-validation/backbone-bakeoff`](../../parked/clip-validation/backbone-bakeoff.md)
  /
  [`validator-evaluation`](../clip-validation/validator-evaluation.md).
- **Cascade-arbiter (CLIP → Qwen2.5-VL judge → planner).**
  [`clip-validation/cotrained-retrieval-augmented`](../../parked/clip-validation/cotrained-retrieval-augmented.md).
- **End-to-end VLA replacement** of `_verify_arrival`. Tracked
  under [`experimental/vla-v2-architecture`](../../parked/experimental/vla-v2-architecture.md)
  (if filed) or its successor.
- **Mid-mission validation while in flight.** That's `TransitMonitor`
  in the design doc; this brief is only the arrival checkpoint.
- **Auto-accepting `arrival_occluded`.** The new error code surfaces
  the soft failure; the operator (or higher-level planner) decides
  whether to accept it. Auto-accept policy belongs in the planner
  layer, not the executor.
