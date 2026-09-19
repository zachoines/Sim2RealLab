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

**Cost.** The render at the deploy resolution costs ≈1.1× per iteration at 96
environments (90.0 → 103.0 s; collection 1.39×, the PPO update dominating).
That figure is established at 96 environments only — the two arms scale
differently above it, and nothing between 96 and 192 was measured.

## Run

| item | value |
|---|---|
| task | `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0` |
| environments | 96 |
| seed | 42 |
| iterations | 1000 |
| expected wall clock | ≈29 h at ≈103 s/iteration |
| tier | the shipped ROBUST tier, with the band as above |

## Acceptance criteria

- [ ] The run completes 1000 iterations at 96 environments, seed 42, through
      `tools/kit_boot_watchdog.sh`, with every attempt line kept and no NaN in
      the losses. Per-iteration collection and learning times are recorded and
      compared against the 103 s expectation.
- [ ] The checkpoint exports (`obs_dim` 3619, `action_dim` 3, `is_recurrent`)
      to ONNX and TorchScript, and a one-episode play with the exported
      artifact reaches an action. The export names the enriched robust play env
      explicitly — the variant default is the non-enriched one.
- [ ] **Gate.** The sim-bridge rig-gate protocol of `goal-a-rig-gate-2026-08-17`
      is re-run with the new artifact over the direct cable: six scored missions
      on `Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, seed 42, fresh
      SLAM key, start heading fixed, goals drawn from the plannability probe
      rather than by eye, with the stack-contract counters and the map→odom
      drift statistics restricted to the scored windows.
- [ ] **Gate, second leg.** The 2026-08-19 addendum's fixed-goal leg is run as
      well: one goal, one start pose and heading, repeated three times. The
      six-mission set measures coverage because each mission uses a different
      goal; this leg is what measures reliability.
- [ ] The six confounds the 2026-08-17 set recorded are each answered before the
      runs, not after: the duty-cycled collision-admission rule, the unfixed
      start pose, the minimum-start-distance floor added mid-set, the moved
      furniture-standoff goal, the duplicate-content rejection, and the
      RTF-versus-monotonic timeout comparison.
- [ ] Acceptance thresholds are fixed and written down before the first mission
      runs, as the 2026-08-17 set did. The v2 rig-class texture criterion is
      **not** reused: it asks whether a depth field reads as a featureless far
      surface, its answer is pre-determined for any per-pixel sensor model, and
      the response it scores is the behaviour this retrain exists to remove.
- [ ] Descriptive tables accompany the verdict and are not thresholds: v2 and v3
      off-goal distributions on the bridge capture side by side, the
      uniform-field reference curve, and the per-band texture statistic of the
      new training depth against the capture.
- [ ] What is **not** claimed is stated in the record: real-sensor behaviour, ρ,
      and any calibration of σ_d against the real D555.
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
