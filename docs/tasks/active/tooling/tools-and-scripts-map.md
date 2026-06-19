# Tools and scripts map (context module)

**Type:** docs (context module)
**Owner:** DGX
**Priority:** P3 — discoverability / drift-prevention; not gating.
**Estimate:** S
**Branch:** `task/tools-and-scripts-map`
**Sequenced after:** [`scene-metadata-in-usd`](scene-metadata-in-usd.md) (PR #90)
and [`detections-overlay-hero-video`](detections-overlay-hero-video.md) (PR #92)
— the map lists the tools those add (`scene_metadata_reader`, `scene_classes`,
`detections_overlay`); merge them first so the map matches `main`.

## Story

As a **fresh agent** I want **one place that lists the project's tools and
scripts with their purpose** so I **don't re-create an existing tool, and I
know which modules a behavior change must revisit.**

## Motivation

Tools and scripts are scattered across five packages. The per-package README
inventories are authoritative but you have to know to look in each one, and
one-off tools accumulate quietly (several were added in a single recent
session). A single index context module makes them discoverable and turns the
implicit "update consumers when an API changes" expectation into an explicit,
written rule — the forcing function that makes agents revisit a changed tool's
callers.

## Acceptance

- [ ] `docs/tasks/context/tools-and-scripts-map.md` — an **index**: pointers to
  the per-package README inventories (the source of truth for signatures/args)
  + grouped one-line purposes for the `strafer_lab` tools and scripts + the
  cross-package pointers + the maintenance rule. It deliberately carries **no
  signatures** (those drift; they live in the READMEs).
- [ ] Conforms to the context-module rules in
  [`context/README.md`](../../context/README.md) — pointers not exposition; no
  history / in-flight / time-bound content.
- [ ] Shipped per [`conventions.md`'s closed-brief lifecycle](../../context/conventions.md#closed-task-brief-lifecycle).

## Out of scope

- Moving the per-package README inventories into this doc — they stay the
  source of truth; this is an index that points at them.
- Any file moves / subsystem regrouping — that's
  [`tools-package-reorg`](../../parked/tooling/tools-package-reorg.md) and
  [`script-tool-subsystem-grouping`](script-tool-subsystem-grouping.md). This
  brief is documentation only.

## Triggered by

PR #90 / #92 review discussion — several one-off tools created in one session
surfaced the need for an authoritative, discoverable tools reference that also
prompts agents to revisit tools whose API changed.
