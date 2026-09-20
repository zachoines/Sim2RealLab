# Render the policy camera at the deploy resolution and reduce in the observation term

**Status:** Shipped 2026-09-19 in `3da7b50` (DGX).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/222

**Type:** task (training-path change plus the measurement that sizes it)
**Owner:** DGX
**Priority:** P1 — the next rig gate is sim-bridge, so the artifact is gated on
the deploy depth field; training on a separately rendered one leaves the gate
measuring a field the policy never saw.
**Estimate:** M (a small term change plus a measurement sitting)
**Branch:** `task/deploy-resolution-depth`

## Story

As the **depth-subgoal retrain**, I want **the clean training depth field to be
the deploy depth field by construction**, so that **the sim-bridge gate measures
where the obstacles are rather than which renderer drew them.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
- [measurements/deploy-resolution-depth-2026-09-19](../../../measurements/deploy-resolution-depth-2026-09-19/README.md)
  — this brief's record.
- [measurements/depth-noise-coverage-2026-09-18](../../../measurements/depth-noise-coverage-2026-09-18/README.md)
  — the cost re-cost and the scratch implementation this takes the reduction from.
- [measurements/noise-texture-parity-2026-09-17](../../../measurements/noise-texture-parity-2026-09-17/README.md)

## Context

The policy camera rendered 80×45 directly; the deploy node renders 640×360 and
reduces with an 8×8 block median. The two fields differ, and the 2026-08-01
rejection of the higher-resolution render in `depth-camera-vfov-parity` rested on
a throughput premise that #221 measured dead (≈1.1× at 96 environments, not the
"1–8 parallel envs" the docstring claimed).

## Acceptance criteria

- [x] `make_d555_camera_cfg` renders the deploy resolution; `depth_image`
      reduces with the deploy node's block median, gated on the rendered shape
      so a field already on the policy grid passes through. No cfg field selects
      the old render.
- [x] A contract test pins byte-equality against `obs_pipeline.downsample_depth`
      on deploy-resolution fields carrying +inf, NaN, −inf and a block
      straddling the near clip — 120 trials over six edge classes, all identical
      — plus the pass-through shape. Mutation-proven: substituting
      `torch.median` fails it on 3414/3600 pixels.
- [x] The camera resolution and the exact 8× block ratio are pinned by
      assertions on the composed variants as well as the factory, because no
      golden can see either seam: the serializer takes only `num_envs` and
      `env_spacing` off the scene, and the composition rebuilds the camera
      rather than inheriting it.
- [x] `disparity_noise_px_range=(0.002, 0.16)` on the ROBUST tier and nowhere
      else; the realistic tier's goldens hold.
- [x] Golden movement attributed by field name: exactly
      `disparity_noise_px_range` on the eight robust depth contracts, with the
      realistic contracts, the six camera-less contracts, the depth-observation
      golden and **both layout goldens** unmoved.
- [x] The reduction drift is measured directly, from co-located cameras at one
      tick over 30 distinct poses, reported per band. A first attempt was
      confounded by the mount-jitter event, which points the shipped camera prim
      and not a scratch one; its figures are withdrawn and the confound, measured
      by a third unpointed camera, is 67× the drift. Corrected: **1.69e-06**
      whole-frame, below the capture's own residual in four of five bands, which
      **confirms** the 2026-08-01 estimate rather than overturning it. Reported
      descriptively — the pre-registered threshold was a whole-frame figure
      applied per band and is not coherent as one.
- [x] A 20-iteration training smoke at 96 environments, seed 42, runs clean:
      101.584 s/iteration (1.128× the #221 baseline), no NaN, boot relaunch
      recorded. Export gives `obs_dim` 3619, `action_dim` 3, `is_recurrent`, and
      the exported artifact drives the play env for 60 steps.
- [x] The change's five consumers are fixed: the sim-in-the-loop capture and
      the teleop capture both reduce the policy depth through the same operator,
      `rgb_policy` folds into `rgb_full` at the one place stacks are normalised,
      the depth-noise suite's wall mask moves to the policy grid, and the
      perception-camera contract stops asserting the two cameras differ in size
      — that contract belonged to no suite and had never executed, so it moves to
      the pure tree. The depth-sensed proximity penalty reduces too, restoring
      the referent it had before the render changed.
- [x] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [x] No regression: pure suite 1355 passed / 1 skipped, composition contracts
      269, full Kit suite 499 with one failure — the filed P3 flake
      `collision-imu-signal-flaky`, which the deposit shows failing on
      2026-09-13 and 2026-09-14, before this branch.

## Out of scope

- The retrain itself, which [`depth-subgoal-v3-retrain`](../../active/trained-policy/depth-subgoal-v3-retrain.md)
  carries.
- Any change to σ_d values, the hole/drop/hold terms, or the #219 near-field
  conventions.
- Raw-resolution noise injection, which depends on ρ and stays gated on the
  real-D555 capture.
- Environment counts above 96.
- Dropping the policy camera's unconsumed 640×360 RGB channel.
- The `test_sim/` raw-pytest false-green recorded in the record's §8. It
  predates this work and `run_tests.py` is immune; the mechanism is located and
  the one-kwarg fix proven on the failing path, filed as
  [`kit-suite-exit-code-false-green`](../../active/tooling/kit-suite-exit-code-false-green.md).
- The policy camera's unconsumed colour channel, filed as
  [`policy-camera-unconsumed-rgb`](../../active/sim-performance/policy-camera-unconsumed-rgb.md).
