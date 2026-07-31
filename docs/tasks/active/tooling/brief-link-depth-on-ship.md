# Shipping a brief breaks its relative links

**Type:** bug / tooling (docs hygiene)
**Owner:** Either (a scripted rewrite plus a per-link sanity pass; no hardware)
**Priority:** P2 (no blocked work, but 578 dead links is a discoverability tax on
every agent that opens a brief, and it regrows on every ship)
**Estimate:** M (~1 day: the fixer, the manual residue, the guard, one sweep)
**Branch:** `task/brief-link-depth-on-ship`

## Context bundle

- [context/conventions.md](../../context/conventions.md)
- [`docs/tasks/README.md`](../../README.md) — "Shipping a brief: order of
  operations", the step that causes this
- Prior art: [`brief-cross-reference-sweep`](../../completed/brief-cross-reference-sweep.md)
  swept the same *symptom* after the epic-structure reorg, one-time and with no
  guard

## The problem

Shipping a brief moves it from `docs/tasks/active/<epic>/<brief>.md` to
`docs/tasks/completed/<brief>.md` — from depth 3 to depth 2. Every relative link
in the brief keeps the `../` count it was authored with, so each one now
resolves one directory too high:

| Written in `active/<epic>/` | Resolves to | Should be |
|---|---|---|
| `../../context/conventions.md` | `docs/context/…` | `../context/conventions.md` |
| `../../../../source/strafer_ros/…` | above the repo root | `../../../source/…` |
| `mission-generator.md` (sibling in the same epic) | `completed/mission-generator.md` | wherever that brief lives now |

`README.md`'s shipping sequence says `git mv` and says to stamp and update the
board; it does not say to rewrite the link depth, so every brief that ships with
a context bundle acquires broken links, and nobody notices because nothing
checks.

Measured on `main` at `0bb65be`, over `docs/tasks/`:

```
578 broken relative links, in 70 of the tree's files

  338  strip one leading '../' resolves       (the ship-move depth shift)
  223  unique basename elsewhere in the repo  (the target itself moved)
   17  neither                                (needs a human)
```

97% are mechanically resolvable. The 223 are the same defect one hop out: a
brief links a sibling, the sibling later ships or un-parks, and the link is
never repointed.

## Why a one-time sweep is not enough

[`brief-cross-reference-sweep`](../../completed/brief-cross-reference-sweep.md)
(PR #31) already fixed this class once, for the legacy flat-slug names the
epic-structure reorg orphaned. It shipped no guard, and the count is back to 578
because the *structural* cause — the ship move changing depth — was never
addressed. A sweep without a check buys a few months.

## The design

Three pieces, in order:

1. **A resolver, not a `../`-stripper.** For each broken link: if stripping one
   leading `../` resolves, take it; else if the link's basename has exactly one
   match in `git ls-files`, repoint at that; else leave it and report it. The
   basename pass is what covers the 223, and it keeps working when a target
   moves again.
2. **A checker** that fails on any unresolvable relative link under
   `docs/tasks/`, wired into `make test-dgx` / `make test-jetson` the way
   `env-check` is. This is the piece that makes the fix stick; without it the
   brief is just PR #31 again.
3. **The convention amendment.** Add the link-depth step to
   [`README.md`](../../README.md)'s shipping sequence, next to the stamp and the
   board update, so the manual path is correct even when someone edits by hand.

Whether the fixer runs as a one-shot script or stays in the tree is the
implementer's call; the checker must stay.

## Acceptance

- [ ] `docs/tasks/` has zero unresolvable relative links, and the count is
      reported in the PR (before/after).
- [ ] The 17-odd links that resolve to nothing are each dispositioned
      explicitly — repointed, or deleted with a one-line note on why the target
      is gone. Not silently dropped.
- [ ] A checker fails on a newly-introduced broken link, mutation-tested by
      introducing one, and runs inside both host umbrellas.
- [ ] `README.md`'s "Shipping a brief: order of operations" names the link-depth
      step.
- [ ] The rewrite touches links only. No brief prose, stamp, or board row
      changes ride along — a 500-link diff must stay reviewable by inspection.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Out of scope

- Broken links outside `docs/tasks/` (package READMEs, top-level guides). Same
  class of rot, different surface; file separately if the checker is worth
  pointing at them too.
- Anchor fragments (`#section-name`). The checker resolves paths only —
  validating anchors needs a Markdown parse and is a different job.
- Absolute URLs. GitHub PR links in briefs are historical record and may
  legitimately 404.
- Changing the flat `completed/` layout to preserve depth. That would fix the
  cause but breaks the "completed is browsed by date/search" property
  `README.md` commits to; the checker is the cheaper contract.

## Triggered by

PR [#172](https://github.com/zachoines/Sim2RealLab/pull/172) review: the brief's
own context-bundle links broke when it moved to `completed/`, which prompted
measuring the rest of the tree.
