# Retrain the depth-subgoal policy on the deploy depth field

**Type:** task (training run plus the gate that accepts it)
**Owner:** DGX
**Priority:** P1 — the v2 artifact keys on whether depth is far and featureless
at the per-pixel scale, and every depth inference on record is bridge depth, so
until a policy is trained on the field the bridge delivers the sim-bridge gate
cannot separate navigation from that habit.
**Estimate:** L (one GPU day for the run, plus a rig sitting for the gate)
**Branch:** `task/depth-subgoal-v3-retrain`

## Story

As the **depth-subgoal deployment**, I want **a policy trained on the depth
field the deploy path actually produces**, so that **the sim-bridge rig gate
measures where the obstacles are rather than which renderer drew them.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
- [measurements/deploy-resolution-depth-2026-09-19](../../../measurements/deploy-resolution-depth-2026-09-19/README.md)
  — the training depth path this brief trains on: the policy camera renders the
  deploy resolution and the observation term applies the deploy node's block
  median, byte-for-byte.
- [measurements/depth-noise-coverage-2026-09-18](../../../measurements/depth-noise-coverage-2026-09-18/README.md)
  — the per-band texture of the deploy field, the σ_d coverage curve, and the
  wall-clock cost of the render at 96 environments.
- [measurements/noise-texture-parity-2026-09-17](../../../measurements/noise-texture-parity-2026-09-17/README.md)
  — establishes that no trained artifact is an acceptance gate for a training
  distribution, and retires the v2 rig-class criterion.
- [measurements/goal-a-rig-gate-2026-08-17](../../../measurements/goal-a-rig-gate-2026-08-17/README.md)
  — the gate protocol this brief re-runs, and the six confounds it recorded.
- [measurements/depth-convention-fix-2026-09-13](../../../measurements/depth-convention-fix-2026-09-13/README.md)
  — the near-field convention every depth artifact before it was trained against.
- [measurements/depth-subgoal-v3-retrain-2026-09-21](../../../measurements/depth-subgoal-v3-retrain-2026-09-21/README.md)
  — the v3 run, the exported artifact the gate runs, its play smoke, and the
  descriptive tables.

## Context

**What changed under the retrain.** Three things landed after v2 was trained,
and each invalidates it as an artifact rather than merely ageing it:

1. The near-field convention was reconciled (`#219`): the far-clamp share of the
   near-field class went 0.4990 → 0.0000 and the p95 |Δ| against the deploy
   field 0.96667 → 0.00480.
2. The training depth path became the deploy depth path
   (`deploy-resolution-depth-2026-09-19`): the camera renders 640×360 and the
   observation term reduces with the deploy node's 8×8 block median. The clean
   training field is now the deploy field by construction rather than by
   resemblance.
3. The robust tier draws its subpixel disparity noise log-uniformly over
   (0.002, 0.16) per environment at reset, because the deploy field's texture
   rises with depth in a shape no single σ_d matches — the per-band matching
   σ_d spans about 400×.

**What this brief does not claim to fix.** σ_d's calibration against the real
sensor is undetermined: the within-block correlation ρ of real D555 depth has
never been measured, and no real-sensor depth inference exists on record. The
band is a coverage decision, not a calibration. A real-D555 capture
(`real-d555-depth-texture-capture`) would settle it; this retrain does not wait
on it, because the gate it is accepted against is sim-bridge, where the field is
the one training now renders.

**Cost.** The render at the deploy resolution costs 1.128× per iteration at 96
environments, measured on the shipped path: 101.584 s against the 80×45
baseline's 90.039 s, the PPO update dominating. That figure is established at 96
environments only — the two arms scale differently above it, and nothing between
96 and 192 was measured.

**What re-enabling the proximity penalty would cost is unpriced.**
`depth_obstacle_proximity` ships inert, and Isaac Lab skips zero-weight terms
before calling them, so it contributes nothing to the figure above. It reads the
policy field through the same reduction, so its referent is right, but its cost
at a live weight has never been measured.

**The command debug markers are out of the contract v3 was trained on.** v3 was
trained with `goal_command.debug_vis=True` and no visualizer registered, so the
markers never left the world origin, a grid corner outside every room at 96
environments. Since `debug-marker-leak-2026-09-21` the command terms draw no scene
geometry and `debug_vis` is off, so the composition contract differs from v3's training
contract by exactly `commands.goal_command.debug_vis` (`faf86756…` → `c98d18ba…`) and by
nothing in the observation semantics.

**How v3 responds to positioned markers is measured**
(`debug-marker-checks-2026-09-23`): with them in every environment its steering is biased 3.7°
to the left, and no outcome metric moves beyond its standard error. The shift is v3's response
to a novel object at its subgoal, not a defect. The 0.04 / 0.5° indifference band written into
the reading before the runs was too tight to certify indifference — a true null falls outside
the offset band about a fifth of the time — and the observed shift, 9.3 standard errors past it,
is real regardless. The retrain-on-markers question is closed by construction: since #224 no
camera can render a command marker.

## Run

| item | value |
|---|---|
| task | `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0` |
| environments | 96 |
| seed | 42 |
| iterations | 1000 |
| expected wall clock | ≈28 h at the measured 101.6 s/iteration |
| tier | the shipped ROBUST tier, with the band as above |

## Acceptance criteria

- [x] The run completes 1000 iterations at 96 environments, seed 42, through
      `tools/kit_boot_watchdog.sh`, with every attempt line kept and no NaN in
      the losses. Per-iteration collection and learning times are recorded and
      compared against the measured 101.6 s.
      Met 2026-09-21: one leg, one boot attempt, 95.25 s/iteration mean
      (35.66 collection, 59.59 learning), 26.48 h, no NaN
      ([record](../../../measurements/depth-subgoal-v3-retrain-2026-09-21/README.md) §1; the failed first launch's attempt line is in its
      `provenance.md`).
- [x] The checkpoint exports (`obs_dim` 3619, `action_dim` 3, `is_recurrent`)
      to ONNX and TorchScript, and a one-episode play with the exported
      artifact reaches an action. The export names the enriched robust play env
      explicitly — the variant default is the non-enriched one.
      Met 2026-09-21: `strafer_depth_subgoal_v3_999` from `model_999.pt`,
      `env_id` the enriched robust play env; 60 steps on that env, batch 1,
      60 finite distinct actions ([record](../../../measurements/depth-subgoal-v3-retrain-2026-09-21/README.md) §3).
- [x] **Gate.** The sim-bridge rig-gate protocol of `goal-a-rig-gate-2026-08-17`
      is re-run with the new artifact over the direct cable: six scored missions
      on `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, seed 42, fresh
      SLAM key, start heading 130° held to within ~2.8° by closed-loop in-place
      rotation, goals drawn from the plannability probe (`ComputePathToPose`)
      rather than by eye, with the stack-contract counters and the map→odom
      drift statistics restricted to the scored windows. Tolerance is 0.30 m
      (`GOAL_ARRIVAL_RADIUS_M`, also the node's `goal_reached_distance_m`); v2's
      closest approach across the six was 1.79 m.
      Met 2026-09-25: **4 of 6** reached within 0.30 m (G1, R1, L1, L2; R2 and R3 aborted at
      60 s sim at 0.350 and 0.508 m after advancing 2.25 and 2.38 m), which the pre-registered
      table reads as PASS. Median while-moving `v_par` +0.253 m/s against v2's 0.026–0.083;
      start heading within 0.6–2.2° of 130° on all nine v3 runs; no `map→odom` correction
      ≥ 0.30 m in the six scored windows, although R3 ran its whole window on a `map→odom`
      displaced 0.95 m / 76° by a step in the transit before it ([record](../../../measurements/goal-a-rig-gate-v3-2026-09-25/README.md)).
      The record also carries the reaches' terminal behaviour and a stable-frame distance for
      every run: in stable frames the six read three certain reaches plus G1 at the line, and the
      fixed leg 2 of 3. Goal-a's "via the autonomy CLI" clause stays open until
      [`executor-policy-nav-budget`](../reliability/executor-policy-nav-budget.md) lands and a
      CLI-submitted confirmation set runs; this gate drove the node's action server directly, as
      the 2026-08-17 set did.
- [x] **Gate, second leg.** The 2026-08-19 addendum's fixed-goal leg is run as
      well: the one goal (−2.00, 2.25) from a fixed start pose and heading,
      repeated three times. The six-mission set measures coverage because each
      mission uses a different goal; this leg is what measures reliability. v2
      scored 0/3 with all three runs ending within 4 cm of each other, so the
      comparison is against a sharp baseline.
      Met 2026-09-25: **3 of 3** reached (final 0.297, 0.292, 0.284 m; 10.7, 6.8, 6.8 s sim), so v3
      is reliable at that goal; v2 on the same goal in the same session ended 2.10 m away
      ([record](../../../measurements/goal-a-rig-gate-v3-2026-09-25/README.md)).
- [x] The six confounds the 2026-08-17 set recorded are each answered before the
      runs, not after, quoting that record's own list: (1) the collision
      admission rule duty-cycled — its freshness guards compare
      `time.monotonic()` against timeouts sized in sim units, leaving it
      unavailable for 49.5 % of the interval at RTF 0.106; (2) start pose varied
      while only start heading was fixed, landing 0.20–3.53 m from nominal;
      (3) a 1.20 m minimum-start-distance floor added to the harness mid-set;
      (4) the furniture-standoff goal moved outward; (5) the duplicate-content
      regime changed mid-run, 50 % → 0 % for the final 499 s sim, so M1/M2 and
      M3–M6 were not taken in the same render regime; (6) no secondary artifact
      arm was run. Confound 6 is answered by construction here — running v2 and
      v3 on the same set is exactly the arm that record names as the obvious
      first one, and it separates a policy-specific under-advance from a
      lane-wide one.
      Met 2026-09-25: each was answered in the pre-registration deposited before the first
      mission — (1) fixed by #229 and in the images; (2) a fixed nominal start via a scripted
      `cmd_vel` transit and heading hold, enforced at 0.15 m / 3.0°; (3) the 1.20 m floor in force
      from the first mission; (4) every goal fixed in advance; (5) one bridge launch at script
      defaults, with the render-side regime measured per mission; (6) v2 on two of the six goals
      — the "same set" above was narrowed to G1 and L1 when the thresholds were written, to
      bound rig time; two goals still separate an artifact-specific under-advance from a
      lane-wide one ([record](../../../measurements/goal-a-rig-gate-v3-2026-09-25/README.md)).
- [x] Acceptance thresholds are fixed and written down before the first mission
      runs. The 2026-08-17 set fixed ≥4/6 pass, 1–3/6 partial, 0/6 fail; a v3
      threshold is stated in the same form before any mission runs. The v2 rig-class texture criterion is
      **not** reused: it asks whether a depth field reads as a featureless far
      surface, its answer is pre-determined for any per-pixel sensor model, and
      the response it scores is the behaviour this retrain exists to remove.
      Met 2026-09-25: the thresholds, in the 2026-08-17 form, with the reading of PARTIAL's
      qualifiers and the definition of a reach, were committed to the evidence deposit at
      03:30 UTC, before the first goal at 03:34 UTC, and not edited afterwards
      ([record](../../../measurements/goal-a-rig-gate-v3-2026-09-25/README.md)).
- [x] Descriptive tables accompany the verdict and are not thresholds: v2 and v3
      off-goal distributions on the bridge capture side by side, the
      uniform-field reference curve, and the per-band texture statistic of the
      new training depth against the capture.
      Met 2026-09-21 ([record](../../../measurements/depth-subgoal-v3-retrain-2026-09-21/README.md) §2, §4, §5).
- [x] What is **not** claimed is stated in the record: real-sensor behaviour, ρ,
      and any calibration of σ_d against the real D555.
      Met 2026-09-21 ([record](../../../measurements/depth-subgoal-v3-retrain-2026-09-21/README.md) §7).
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression in the workflows the touched code supports.

## Out of scope

- Any change to σ_d values, the band, the hole/drop/hold terms, or the near-field
  convention. The training distribution is the one
  `deploy-resolution-depth-2026-09-19` shipped; a retrain that also retunes it
  measures two things at once.
- Raw-resolution noise injection (σ_d synthesised at 640×360 and reduced). That
  form depends on ρ and stays gated on the real-D555 capture.
- Environment counts above 96. The ≈1.1× cost is established at 96 only.
- Dropping the policy camera's unconsumed 640×360 RGB channel. It is a render
  cost the viewport needs; sizing it is its own measurement.
- The real-sensor half of the depth question, which
  `real-d555-depth-texture-capture` carries.
