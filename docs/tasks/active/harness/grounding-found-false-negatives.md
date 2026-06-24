# Grounder found=false negatives — target-absent frame labeling

**Type:** new feature (dataset labeling rule)
**Owner:** DGX agent (LeRobot writer / detections labeling)
**Priority:** P1 — the VLM grounder needs target-absent (negative) frames or its found-accuracy degrades; **unowned today**. Rides on the coverage-capture corpus, so it can be designed in parallel and applied once those frames exist.
**Estimate:** S–M (a labeling rule + emit path over captured frames; no new capture).
**Branch:** task/grounding-found-false-negatives

## Story

As **the VLM grounder training corpus**, I want **frames where the queried target is ABSENT labeled as `found=false`**, so that **the grounder learns to answer "not here" instead of hallucinating a box — which a goal-reaching corpus (target always present) never teaches.**

## The gap (from the 2026-06-23 data-requirements gap analysis, G2)

The detections column enumerates only objects that are **present** in frame. The grounder also needs negatives: frames where the queried `target_label` is **absent**. This is **unowned**:
- [`grounding-negative-taxonomy.md`](grounding-negative-taxonomy.md) covers only `trajectory_violation` (right destination, wrong route) — a **different axis**, and is itself gated on path-shape missions.
- The `wrong_object` / `wrong_room` / `wrong_instance` injectors swap the goal to a **different-but-still-present** object — none yields a target-absent frame.

So the found-axis negative has no owner. This brief is it.

## Scope

- A labeling rule: over the coverage corpus, sample frames where the episode's `target_label` has **zero valid detections**, and emit them as `found=false` grounder-training rows.
- Define the positive/negative balance + the sampling policy (avoid trivially-easy negatives; prefer same-room-but-occluded and adjacent-room cases).

## Context bundle

- `HARNESS_DATA_REQUIREMENTS_GAP_ANALYSIS.md` (workspace root) — gap **G2**.
- [`coverage-capture-driver.md`](coverage-capture-driver.md) — provides the target-absent frames (the coverage sweep naturally generates them); this brief rides on that capture.
- `vlm-grounding-finetune.md` — the consumer (the grounder LoRA).
- [`grounding-negative-taxonomy.md`](grounding-negative-taxonomy.md) — the **different** (trajectory) negative axis; keep them disjoint.

## Acceptance

- [ ] A documented rule emits `found=false` rows from coverage frames where `target_label` has zero valid detections.
- [ ] Positive/negative balance + sampling policy documented and tunable.
- [ ] `vlm-grounding-finetune` can consume the negatives (vocab + format aligned with the detections column).
- [ ] Tests cover the zero-valid-detection selection + the emit path.

## Out of scope

- `trajectory_violation` negatives (owned by `grounding-negative-taxonomy`).
- The coverage capture itself (owned by `coverage-capture-driver`).
- The grounder model/training (owned by `vlm-grounding-finetune`).
