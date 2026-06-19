# Detections-overlay viz tool + synthetic-data hero video

**Type:** tooling + docs
**Owner:** DGX
**Priority:** P3 — quality-of-life / docs; not gating any acceptance bar.
**Estimate:** S
**Branch:** `task/detections-overlay-hero-video`
**Sequenced after:** [`scene-metadata-in-usd`](scene-metadata-in-usd.md) (PR #90) — the captured
artifact relies on that brief's `UsdSemantics` authoring for non-empty
detections. The overlay tool itself is capture-agnostic (it only reads the
recorded columns), so it is reviewable independently.

## Story

As someone evaluating the project I want **an at-a-glance view of the
harness's detection ground truth** so the state of synthetic-data
generation is obvious without reading code or inspecting parquet.

## Motivation

Per-frame detections are stored as **padded parquet columns**, not painted
into the camera video (see [`harness-architecture`](harness-architecture.md)'s
Detections section) — which is correct for training data, but means there
is no canonical way to *see* them. A small reusable overlay tool turns any
capture into an annotated video, and the README gains a hero clip showing
the harness boxing Infinigen furniture (and correctly skipping structure).

## Acceptance

- [ ] `strafer_lab.tools.detections_overlay` — a pure `draw_detection_boxes`
  helper plus `overlay_detections_video(dataset → MP4)`, reusable on **any**
  strafer capture (reads the recorded `observation.detections.*` columns,
  no Isaac Sim). Unit-tested in `tests/harness/test_detections_overlay.py`.
- [ ] A 360°-spin annotated MP4 + thumbnail under `docs/artifacts/`, captured
  on `scene_high_quality_dgx_000_seed2` (30-class furniture vocab; structural
  classes correctly absent).
- [ ] Top-level [`Readme.md`](../../../Readme.md) hero embed via the existing
  `<a href=…mp4><img src=…thumbnail.png>` pattern.
- [ ] Shipped per [`conventions.md`'s closed-brief lifecycle](../../context/conventions.md#closed-task-brief-lifecycle).

## Out of scope

- A reproducible **committed spin-capture script** — the artifact was captured
  via a pure-yaw sweep of the existing bridge harness (a throwaway built by
  copying `bridge_harness_smoke.py`); the durable, shipped code is the overlay
  tool. The detection *content* is reproducible once `scene-metadata-in-usd`
  lands: re-author a scene (`extract_scene_metadata --from-usd`) → capture →
  overlay.
- Any change to the dataset detections schema or the harness capture path.

## Triggered by

PR #90 review — an annotated still proved compelling; a 360-spin hero makes the
synthetic-data-gen state legible at the top of the README.
