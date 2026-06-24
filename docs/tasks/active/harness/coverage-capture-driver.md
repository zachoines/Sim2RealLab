# Bulk-capture coverage driver — diverse-perspective dataset as the Tier-3 default

**Type:** new feature (Tier-3 capture driver + bulk-run dataset-schema readiness)
**Owner:** DGX agent (`source/strafer_lab/scripts/` capture path + the LeRobot writer)
**Priority:** P0 — the single **capture-blocking** decision for a flexible multi-model corpus. A goal-reaching / teleop demo set under-samples the views the trained models need, and that signal is **irrecoverable post-hoc** (re-capturing thousands of episodes is the cost to avoid).
**Estimate:** L (wire the `scripted × coverage` driver + the traversal policy + the bulk-run schema additions + the held-out seed config).
**Branch:** task/coverage-capture-driver

## Story

As **the harness producing a flexible multi-model training corpus**, I want **the bulk capture to be a diverse-perspective coverage sweep (not goal-reaching demos)** so that **the trained grounder, the spatial-encoder VPR/region heads, the backbone-bakeoff gate, and the eval harness all get the same-place / different-heading and off-axis / occluded views they require — captured once.**

## Why this is capture-blocking (from the 2026-06-23 data-requirements gap analysis)

A goal-reaching / teleop demo drives straight to the target and only ever sees objects from the approach corridor. That **silently fails** the consumers whose training signal is viewpoint diversity:
- **learned-spatial-encoder VPR head** — positive pairs are *by definition* "same place from a different heading," mined from repeated traversals. Diverse perspectives are not a benefit here, they are the training signal.
- **backbone-bakeoff** and **room-state-eval-harness** — need full-room coverage + held-out-seed splits.
- **multi-room grounder / vlm-grounding-finetune** — under-sampled off-axis / occluded views degrade grounding accuracy.

The `('scripted','coverage')` cell is **unwired today** — `capture.py` `VALID_COMBINATIONS` marks it `pending` → `NotImplementedError`. A `coverage` SENSOR_PRESET exists but no driver. This is **build-and-wire**, not a defaulting flip. Capture is Tier 3, not yet started, so the requirement folds in from the start (greenfield).

## Context bundle

- This brief is the **P0 pivot** of the 2026-06-23 multi-model data-requirements analysis — the demand→supply matrix of the downstream training epics vs. what the harness LeRobot v3 capture provides.
- [`harness-architecture.md`](harness-architecture.md) — Tier-3 owner; the coverage driver IS the Tier-3 scripted driver. Coordinate the `scripts/` paths in the same PR.
- [`../trained-policy/domain-randomization-audit.md`](../trained-policy/domain-randomization-audit.md) — owns the camera-DR taxonomy + the realized-mount-logging rule this brief implements as a column.
- [`../multi-room/learned-spatial-encoder.md`](../multi-room/learned-spatial-encoder.md), [`mission-generator.md`](mission-generator.md) — downstream consumers.

## Scope

1. **Wire the `('scripted','coverage')` capture cell** + the coverage traversal policy: visit every room ≥2×, re-approach landmarks from varied headings with a per-visit random rotation offset; full-room coverage, not straight-to-goal.
2. **Make it the bulk-run default** capture mode (goal-reaching/teleop demos are no longer the default for bulk data).
3. **Whole-scene held-out seed selection** baked into the run config + recorded (VPR home-to-home generalization is a capture-time decision).
4. **Detections ON by default**; bulk runs on the **bridge/scripted** driver, **never teleop** (no annotator).
5. **Bulk-run dataset-schema additions** (decided 2026-06-23):
   - **Realized camera-mount column** — log `env._d555_mount_quat` per episode. The ±2° D555 mount jitter (`randomize_d555_mount_offset`) stays **ON**; logging the realized quat is cheap insurance that keeps the realized camera derivable and future-proofs the corpus if the jitter magnitude is later raised. Rule + rationale owned by `domain-randomization-audit.md`.
   - **`scene_metadata` sidecar at `finalize()`** — embed the scene_metadata dict into the dataset (`meta/scenes/<id>/scene_metadata.json`). Today only `scene_metadata_hash` travels and the sidecar is doc-only; without the dict, every pose-derived label (region GT, GT room, narration, sibling phrases, memory-map case-1) breaks offline.

## Acceptance

- [ ] `('scripted','coverage')` is `wired` in `capture.py` and dispatches to a coverage driver.
- [ ] A coverage metric demonstrates each room is sampled ≥2× with heading diversity (revisit + off-axis pairs present).
- [ ] Coverage is the documented bulk-run default; teleop/goal-reaching is explicitly the non-bulk path.
- [ ] Per-episode realized-mount quat **and** the `scene_metadata` sidecar are present in the dataset and round-trip.
- [ ] Held-out whole-scene seed split is configurable + recorded in the run config.
- [ ] Tests green (`make test-lab-pure` + the capture path).
- [ ] Brief shipped to `completed/` in the PR per conventions when the driver lands (the bulk *run* itself is operator/GPU, tracked separately).

## Out of scope

- The grounder **found=false** labeling rule → [`grounding-found-false-negatives.md`](grounding-found-false-negatives.md) (rides on this driver's frames).
- **Adversarial / open-plan** scenes — deferred to an eval-only mini-capture (room-state-eval-harness), NOT the bulk run.
- The actual bulk capture run (operator / GPU).
- Post-process derivatives (captioner corpus, memory store, narration) — derivable later from this corpus, not capture changes.
